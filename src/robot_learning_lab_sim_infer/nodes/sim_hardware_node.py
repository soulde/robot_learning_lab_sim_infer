"""MuJoCo + Viser + DDS simulated hardware node.

The node owns exactly three application worker threads: physics, rendering,
and DDS I/O. DDS callbacks never access MuJoCo objects. ``MjData`` is protected
by a lock and only the physics worker mutates it.
"""

from __future__ import annotations

import threading
import time
import math
from typing import Any, Callable, Protocol

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..config_api import validate_sim_profile
from ..configs.hydra_sim import register_configs
from ..configs import RuntimeConfig
from ..runtime.rc import RCValues
from ..dds.transport import SimHardwareDDS
from ..profiles.xml_hardware import XmlHardwareProfile


class DDSIO(Protocol):
    def take_motor_command(self) -> Any | None: ...
    def publish_robot_state(self, sample: Any) -> None: ...
    def publish_rc_command(self, sample: Any) -> None: ...
    def close(self) -> None: ...


class Renderer(Protocol):
    def update(self, model: Any, data: Any) -> None: ...
    def close(self) -> None: ...


def _resolve_simulation_dt(model: Any, configured_dt: float | None) -> float:
    model_option = getattr(model, "opt", None)
    model_dt = getattr(model_option, "timestep", None)
    if configured_dt is None:
        resolved = float(model_dt) if model_dt is not None else 0.002
    else:
        resolved = float(configured_dt)
        if model_option is not None:
            model_option.timestep = resolved
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("simulation_dt must be a finite positive duration")
    return resolved


class SimHardwareNode:
    """Coordinate MuJoCo simulation, Viser rendering, and DDS hardware I/O."""

    def __init__(
        self,
        profile: Any,
        runtime: RuntimeConfig,
        dds: DDSIO,
        renderer: Renderer | Callable[
            [Any, Callable[[RCValues], None], Callable[[], None]], Renderer
        ],
        *,
        data_factory: Callable[[Any], Any] | None = None,
        physics_step: Callable[[Any, Any], None] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        validate_sim_profile(profile)
        for member in ("make_rc_command", "apply_safe_command"):
            if not callable(getattr(profile, member, None)):
                raise TypeError(f"Sim profile is missing required member: {member}")
        self.profile = profile
        self.runtime = runtime
        self.dds = dds
        self._renderer_source = renderer
        self.renderer: Renderer | None = None
        self._data_factory = data_factory or self._make_data
        self._physics_step = physics_step or self._step
        self._clock_ns = clock_ns
        self._monotonic = monotonic
        self._data_lock = threading.Lock()
        self._reset_lock = threading.Lock()
        self._reset_requested = False
        self._input_lock = threading.Lock()
        self._rc_values = RCValues()
        self._rc_button_mask = 0
        self._rc_button_until = float("-inf")
        self._command_lock = threading.Lock()
        self._motor_command: Any | None = None
        self._motor_command_received_at = float("-inf")
        self._stop = threading.Event()
        self._failure: BaseException | None = None
        self.model: Any | None = None
        self.data: Any | None = None
        self._simulation_dt: float | None = None
        self._threads: list[threading.Thread] = []

    @property
    def simulation_dt(self) -> float:
        if self._simulation_dt is not None:
            return self._simulation_dt
        return _resolve_simulation_dt(None, self.runtime.simulation_dt)

    @staticmethod
    def _make_data(model: Any) -> Any:
        import mujoco

        return mujoco.MjData(model)

    @staticmethod
    def _step(model: Any, data: Any) -> None:
        import mujoco

        mujoco.mj_step(model, data)

    def set_rc_values(self, values: RCValues) -> None:
        """Thread-safe input hook for Viser controls or a gamepad adapter."""
        with self._input_lock:
            if values.button_mask:
                self._rc_button_mask |= int(values.button_mask)
                self._rc_button_until = self._monotonic() + 0.2
            self._rc_values = RCValues(
                enabled=values.enabled, mode=values.mode, vx=values.vx, vy=values.vy,
                yaw_rate=values.yaw_rate,
            )

    def request_reset(self) -> None:
        """Queue a reset for the simulation worker; callers never touch MjData."""
        with self._reset_lock:
            self._reset_requested = True

    def _launch(self, name: str, target: Callable[[], None]) -> None:
        thread = threading.Thread(
            name=f"sim-hardware-{name}",
            target=self._guarded(target),
            daemon=False,
        )
        self._threads.append(thread)
        thread.start()

    def _guarded(self, target: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
            except BaseException as exc:
                self._failure = exc
                self._stop.set()

        return run

    def run(self, stop_event: threading.Event | None = None) -> None:
        """Initialize resources, start the three workers, then join on shutdown."""
        try:
            self.model = self.profile.build_model()
            self._simulation_dt = _resolve_simulation_dt(
                self.model, self.runtime.simulation_dt
            )
            self.data = self._data_factory(self.model)
            self.profile.reset(self.model, self.data)
            if callable(self._renderer_source) and not hasattr(self._renderer_source, "update"):
                self.renderer = self._renderer_source(
                    self.model, self.set_rc_values, self.request_reset
                )
            else:
                self.renderer = self._renderer_source
            self._launch("simulation", self._simulation_loop)
            self._launch("render", self._render_loop)
            self._launch("dds", self._dds_loop)
            while not self._stop.is_set():
                if stop_event is not None and stop_event.is_set():
                    break
                self._stop.wait(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            for thread in self._threads:
                thread.join()
            try:
                if self.renderer is not None:
                    self.renderer.close()
            finally:
                self.dds.close()
        if self._failure is not None:
            raise RuntimeError("Sim hardware worker failed") from self._failure

    def _simulation_loop(self) -> None:
        assert self.model is not None and self.data is not None
        period = self.simulation_dt
        deadline = self._monotonic()
        timeout = self.runtime.dds.motor_command_timeout_s
        while not self._stop.is_set():
            now = self._monotonic()
            with self._data_lock:
                with self._reset_lock:
                    reset_requested = self._reset_requested
                    self._reset_requested = False
                if reset_requested:
                    self.profile.reset(self.model, self.data)
                    with self._command_lock:
                        self._motor_command = None
                        self._motor_command_received_at = float("-inf")
                with self._command_lock:
                    command = self._motor_command
                    fresh = now - self._motor_command_received_at <= timeout
                if command is not None and fresh:
                    mode_apply = getattr(self.profile, "apply_motor_command_for_mode", None)
                    if callable(mode_apply):
                        with self._input_lock:
                            mode = int(self._rc_values.mode)
                        mode_apply(self.model, self.data, command, mode)
                    else:
                        self.profile.apply_motor_command(self.model, self.data, command)
                else:
                    self.profile.apply_safe_command(self.model, self.data)
                self._physics_step(self.model, self.data)
            deadline += period
            delay = deadline - self._monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                deadline = self._monotonic()

    def _render_loop(self) -> None:
        assert self.model is not None and self.data is not None
        assert self.renderer is not None
        period = 1.0 / max(1.0, self.runtime.viser.render_hz)
        while not self._stop.is_set():
            with self._data_lock:
                self.renderer.update(self.model, self.data)
            self._stop.wait(period)

    def _dds_loop(self) -> None:
        assert self.model is not None and self.data is not None
        period = 1.0 / max(1.0, self.runtime.dds.publish_hz)
        while not self._stop.is_set():
            command = self.dds.take_motor_command()
            if command is not None:
                with self._command_lock:
                    self._motor_command = command
                    self._motor_command_received_at = self._monotonic()
            timestamp_ns = self._clock_ns()
            with self._data_lock:
                robot_state = self.profile.extract_robot_state(
                    self.model, self.data, timestamp_ns
                )
            with self._input_lock:
                button_mask = self._rc_button_mask if self._monotonic() <= self._rc_button_until else 0
                if not button_mask:
                    self._rc_button_mask = 0
                base_values = self._rc_values
                rc_values = RCValues(
                    enabled=base_values.enabled, mode=base_values.mode, vx=base_values.vx,
                    vy=base_values.vy, yaw_rate=base_values.yaw_rate, button_mask=button_mask,
                )
            rc_command = self.profile.make_rc_command(rc_values, timestamp_ns)
            self.dds.publish_robot_state(robot_state)
            self.dds.publish_rc_command(rc_command)
            self._stop.wait(period)


class _NullRenderer:
    def update(self, model: Any, data: Any) -> None:
        return None

    def close(self) -> None:
        return None


def run_from_config(cfg: DictConfig) -> None:
    """Run the XML-backed node from a composed Hydra configuration."""
    runtime = OmegaConf.to_object(cfg.runtime)
    if not isinstance(runtime, RuntimeConfig):
        raise TypeError("Hydra runtime config must compose to RuntimeConfig")
    xml_path = to_absolute_path(str(cfg.xml))
    profile = XmlHardwareProfile(xml_path)
    simulation_dt = _resolve_simulation_dt(profile.build_model(), runtime.simulation_dt)
    dds = SimHardwareDDS(runtime.dds)

    if runtime.viser.enabled:
        from ..visualization.viser_mujoco import ViserMujocoRenderer

        renderer_factory = lambda model, callback, reset_callback: ViserMujocoRenderer(
            model, runtime.viser, runtime.rc, callback, reset_callback
        )
        viser_url = f"http://{runtime.viser.host}:{runtime.viser.port}"
    else:
        renderer_factory = lambda model, callback, reset_callback: _NullRenderer()
        viser_url = "disabled"
    node = SimHardwareNode(profile, runtime, dds, renderer_factory)
    print(
        f"sim_hardware_node xml={xml_path} domain={runtime.dds.domain_id} "
        f"simulation_dt={simulation_dt:g}s "
        f"sensor_publish_hz={runtime.dds.publish_hz:g} "
        f"render_hz={runtime.viser.render_hz:g} viser={viser_url} "
        f"joints={','.join(profile.joint_names)} threads=simulation,render,dds",
        flush=True,
    )
    node.run()


@hydra.main(version_base=None, config_path=None, config_name="sim_hardware")
def _hydra_main(cfg: DictConfig) -> None:
    run_from_config(cfg)


def main() -> None:
    """Console entry point; all launch options use Hydra overrides."""
    register_configs()
    _hydra_main()


if __name__ == "__main__":
    main()
