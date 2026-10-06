"""Viser UI and MuJoCo scene adapter for the sim hardware process."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from ..configs import RCConfig, ViserConfig
from ..runtime.rc import RCMapper, RCValues


class ViserMujocoRenderer:
    """Render live MuJoCo data and expose a mock remote controller."""

    def __init__(self, model: Any, config: ViserConfig, rc_config: RCConfig,
                 on_rc_update: Callable[[RCValues], None],
                 on_reset: Callable[[], None]):
        try:
            import viser
            from mjviser import ViserMujocoScene
        except ImportError as exc:
            raise RuntimeError("Viser MuJoCo scene requires the 'sim-hardware' extras") from exc
        self._server = viser.ViserServer(host=config.host, port=config.port)
        self._scene = ViserMujocoScene(self._server, model, num_envs=1)
        create_gui = getattr(self._scene, "create_visualization_gui", None)
        if create_gui is not None:
            create_gui()
        self._on_rc_update = on_rc_update
        self._keyboard = RCMapper(rc_config)
        self._keyboard_lock = threading.Lock()
        self._syncing_gui = False
        self._values = {
            "enabled": True, "mode": 0, "vx": 0.0, "vy": 0.0,
            "yaw_rate": 0.0, "body_height": sum(rc_config.body_height_range) / 2.0,
        }
        gui = self._server.gui
        with gui.add_folder("Mock remote controller"):
            axes = {}
            for axis in ("vx", "vy", "yaw_rate"):
                low, high = rc_config.axis_ranges.get(
                    axis, [-abs(float(rc_config.scales.get(axis, 1.0))),
                           abs(float(rc_config.scales.get(axis, 1.0)))])
                axes[axis] = gui.add_slider(
                    axis, min=float(low), max=float(high),
                    step=max((float(high) - float(low)) / 100.0, 0.001), initial_value=0.0)
            hlow, hhigh = rc_config.body_height_range
            height = gui.add_slider("Body height", min=float(hlow), max=float(hhigh),
                                    step=max((float(hhigh) - float(hlow)) / 100.0, 0.001),
                                    initial_value=self._values["body_height"])
            mode_options = tuple(rc_config.mode_keys)
            mode_widget = gui.add_dropdown("Mode", options=mode_options,
                                           initial_value=mode_options[0]) if mode_options else None
            gui.add_markdown(
                "Keyboard: I/K vx, J/L vy, U/O yaw; each tap changes one configured step. "
                "Z clears velocity commands. Keyboard commands are always enabled. "
                "R resets simulation and selects damping mode 0."
            )

        def publish(values: RCValues) -> None:
            self._values.update(enabled=values.enabled, mode=values.mode, vx=values.vx,
                                vy=values.vy, yaw_rate=values.yaw_rate,
                                body_height=values.body_height)
            self._on_rc_update(values)

        def update(_event: Any = None) -> None:
            if self._syncing_gui:
                return
            mode = self._values["mode"]
            if mode_widget is not None:
                mode = int(rc_config.mode_keys[mode_widget.value])
            values = RCValues(enabled=True, mode=mode,
                              vx=float(axes["vx"].value), vy=float(axes["vy"].value),
                              yaw_rate=float(axes["yaw_rate"].value),
                              body_height=float(height.value))
            with self._keyboard_lock:
                self._keyboard.set_values(values)
            publish(values)

        for handle in (*axes.values(), height):
            handle.on_update(update)
        if mode_widget is not None:
            mode_widget.on_update(update)

        def set_gui(values: RCValues) -> None:
            self._syncing_gui = True
            try:
                for axis in ("vx", "vy", "yaw_rate"):
                    axes[axis].value = getattr(values, axis)
                height.value = values.body_height
                if mode_widget is not None and values.mode in rc_config.mode_keys.values():
                    mode_widget.value = next(k for k, v in rc_config.mode_keys.items()
                                             if v == values.mode)
            finally:
                self._syncing_gui = False

        self._keyboard_gui_sync = set_gui
        update()
        bindings = {key.upper() for key in rc_config.key_axes}
        bindings.add(rc_config.clear_key.upper())
        bindings.update(key.upper() for key in rc_config.mode_keys)
        if len(bindings) != len(rc_config.key_axes) + 1 + len(rc_config.mode_keys):
            raise ValueError("RC keyboard and mode bindings must use unique keys")
        for key in sorted(bindings):
            if len(key) != 1:
                raise ValueError(f"Viser keyboard shortcut {key!r} is not a supported single key")
            command = gui.add_command(
                f"Adjust {key}", description=f"Apply keyboard command {key}.", hotkey=key)

            def apply_key(_event: Any, key=key) -> None:
                with self._keyboard_lock:
                    values = self._keyboard.key_event(key)
                    current = RCValues(enabled=True, mode=values.mode, vx=values.vx,
                                       vy=values.vy, yaw_rate=values.yaw_rate,
                                       body_height=float(self._values["body_height"]))
                    self._keyboard.set_values(current)
                    self._keyboard_gui_sync(current)
                publish(current)

            command.on_trigger(apply_key)

        reset_command = gui.add_command(
            "Reset simulation and select damping",
            description="Reset MuJoCo and clear RC velocity commands into damping mode 0.",
            hotkey="R",
        )

        def reset_simulation(_event: Any) -> None:
            with self._keyboard_lock:
                self._keyboard.reset()
                current = RCValues(
                    enabled=True,
                    mode=0,
                    vx=0.0,
                    vy=0.0,
                    yaw_rate=0.0,
                    body_height=float(self._values["body_height"]),
                )
                self._keyboard.set_values(current)
                self._keyboard_gui_sync(current)
            publish(current)
            on_reset()

        reset_command.on_trigger(reset_simulation)

    @property
    def url(self) -> str:
        return f"http://{self._server.get_host()}:{self._server.get_port()}"

    def update(self, model: Any, data: Any) -> None:
        del model
        self._scene.update_from_mjdata(data)

    def close(self) -> None:
        self._server.stop()
