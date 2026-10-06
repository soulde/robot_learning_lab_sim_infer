"""Hydra-configured DDS policy-switch placeholder and damping controller."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..configs import DDSConfig, StateMachineConfig
from ..dds.transport import PolicyDDS
from ..dds.types import MotorCommand, MotorControlMode
from ..runtime.state_machine import RuntimeStateMachine


@dataclass
class PolicySlotConfig:
    index: int
    name: str
    policy_file: str | None
    kp: float
    kd: float


@dataclass
class DummyControlConfig:
    rate_hz: float = 50.0
    log_hz: float = 2.0
    damping_kd: float = 0.5
    policies: list[PolicySlotConfig] = field(default_factory=list)


class PolicyWorker:
    """One future inference slot; current observe/infer are explicit stubs."""

    def __init__(self, config: PolicySlotConfig):
        self.config = config

    def observe(self, robot_state: Any, rc_command: Any, *, log: bool) -> np.ndarray:
        del rc_command
        if log:
            print(f"policy[{self.config.index}] observe placeholder", flush=True)
        return np.zeros(len(robot_state.joint_names), dtype=np.float32)

    def infer(self, observation: np.ndarray, action_dim: int, *, log: bool) -> np.ndarray:
        del observation
        if log:
            print(f"policy[{self.config.index}] infer placeholder", flush=True)
        return np.zeros(action_dim, dtype=np.float32)


class DummyControlNode:
    """Print current observations and publish stub or damping motor commands."""

    def __init__(self, dds: Any, config: DummyControlConfig, *, clock_ns=time.time_ns):
        if config.rate_hz <= 0.0:
            raise ValueError("dummy control rate_hz must be positive")
        if config.damping_kd < 0.0:
            raise ValueError("damping_kd must be non-negative")
        indices = [policy.index for policy in config.policies]
        if len(indices) != len(set(indices)) or any(index < 1 or index > 9 for index in indices):
            raise ValueError("policy indices must be unique integers from 1 through 9")
        for policy in config.policies:
            if policy.kp < 0.0 or policy.kd < 0.0:
                raise ValueError(f"policy {policy.index} kp and kd must be non-negative")

        self.dds = dds
        self.config = config
        self._clock_ns = clock_ns
        self._workers = {policy.index: PolicyWorker(policy) for policy in config.policies}
        states = {"damping": None}
        states.update({f"policy_{index}": str(index) for index in indices})
        mode_to_state = {0: "damping", **{i: f"policy_{i}" for i in indices}}
        events = {mode: f"mode_{mode}" for mode in mode_to_state}
        self._mode_events = events
        transitions = {
            state: {events[mode]: target for mode, target in mode_to_state.items()}
            for state in states
        }
        self._state_machine = RuntimeStateMachine(
            StateMachineConfig(
                initial_state="damping",
                transitions=transitions,
                state_policies=states,
                mode_events=events,
            )
        )

    def _update_mode(self, mode: int) -> None:
        event = self._mode_events.get(int(mode))
        if event is None:
            return
        previous = self._state_machine.decision.state
        decision = self._state_machine.update(event)
        if decision.state != previous:
            print(f"policy switch {previous} -> {decision.state}", flush=True)

    @staticmethod
    def _format(values: Any) -> list[float]:
        return [round(float(value), 3) for value in values]

    def step(self, robot_state: Any, rc_command: Any, *, log: bool = True) -> MotorCommand:
        names = list(robot_state.joint_names)
        if not names or len(names) != len(robot_state.joint_position):
            raise ValueError("RobotState must contain matching non-empty joint names and positions")
        if len(names) > 64 or len(names) != len(set(names)):
            raise ValueError("RobotState must contain at most 64 uniquely named joints")

        self._update_mode(int(rc_command.mode))
        decision = self._state_machine.decision
        if decision.state == "damping":
            action = np.zeros(len(names), dtype=np.float64)
            target_position = action
            kp = [0.0] * len(names)
            kd = [float(self.config.damping_kd)] * len(names)
            mode = "damping"
        else:
            policy_index = int(decision.policy)
            worker = self._workers[policy_index]
            observation = worker.observe(robot_state, rc_command, log=log)
            action = worker.infer(observation, len(names), log=log)
            # The zero-action placeholder holds the measured joint pose until
            # a real policy/action-to-joint mapping is connected.
            target_position = np.asarray(robot_state.joint_position, dtype=np.float64) + action
            kp = [float(worker.config.kp)] * len(names)
            kd = [float(worker.config.kd)] * len(names)
            mode = f"policy_{policy_index}"

        command = MotorCommand(
            timestamp_ns=int(self._clock_ns()),
            control_mode=MotorControlMode.MIT,
            joint_names=names,
            position=np.asarray(target_position, dtype=np.float64).tolist(),
            velocity=[0.0] * len(names),
            kp=kp,
            kd=kd,
            torque=[0.0] * len(names),
        )
        if log:
            print(
                f"obs_ns={robot_state.timestamp_ns} joints={len(names)} "
                f"base_z={float(robot_state.base_position[2]):.3f} "
                f"obs={{base_p:{self._format(robot_state.base_position)}, "
                f"base_q:{self._format(robot_state.base_orientation_xyzw)}, "
                f"base_v:{self._format(robot_state.base_linear_velocity)}, "
                f"base_w:{self._format(robot_state.base_angular_velocity)}, "
                f"q:{self._format(robot_state.joint_position)}, "
                f"dq:{self._format(robot_state.joint_velocity)}, "
                f"tau:{self._format(robot_state.joint_effort)}, "
                f"imu_q:{self._format(robot_state.imu_orientation_xyzw)}, "
                f"imu_gyro:{self._format(robot_state.imu_angular_velocity)}, "
                f"imu_accel:{self._format(robot_state.imu_linear_acceleration)}, "
                f"contact:{self._format(robot_state.contact_force_xyz)}}} "
                f"rc_ns={rc_command.timestamp_ns} enabled={bool(rc_command.enabled)} "
                f"mode={int(rc_command.mode)} vx={float(rc_command.vx):.2f} "
                f"vy={float(rc_command.vy):.2f} yaw={float(rc_command.yaw_rate):.2f} "
                f"controller={mode} action={self._format(action)} "
                f"kp={kp[0]:g} kd={kd[0]:g}",
                flush=True,
            )
        self.dds.publish_motor_command(command)
        return command

    def run(self, stop_event: threading.Event | None = None) -> None:
        state = rc = None
        try:
            period = 1.0 / self.config.rate_hz
            log_period = 1.0 / self.config.log_hz if self.config.log_hz > 0.0 else float("inf")
            next_log = time.monotonic()
            deadline = time.monotonic()
            while stop_event is None or not stop_event.is_set():
                latest_state = self.dds.take_robot_state()
                latest_rc = self.dds.take_rc_command()
                if latest_state is not None:
                    state = latest_state
                if latest_rc is not None:
                    rc = latest_rc
                if state is not None and rc is not None:
                    now = time.monotonic()
                    log = now >= next_log
                    if log:
                        next_log = now + log_period
                    self.step(state, rc, log=log)
                deadline += period
                delay = deadline - time.monotonic()
                if delay > 0.0:
                    if stop_event is None:
                        time.sleep(delay)
                    else:
                        stop_event.wait(delay)
                else:
                    deadline = time.monotonic()
        except KeyboardInterrupt:
            pass
        finally:
            self.dds.close()


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="dummy_control")
    def _hydra_main(cfg: DictConfig) -> None:
        values = OmegaConf.to_container(cfg, resolve=True)
        dds_config = DDSConfig(**values["dds"])
        control_values = dict(values["control"])
        policies = [PolicySlotConfig(**item) for item in control_values.pop("policies")]
        control_config = DummyControlConfig(policies=policies, **control_values)
        dds = PolicyDDS(dds_config)
        print(
            f"dummy_control_node domain={dds_config.domain_id} "
            f"sensor_hz={dds_config.publish_hz:g} control_hz={control_config.rate_hz:g} "
            f"policies={[policy.index for policy in policies]} damping_kd={control_config.damping_kd:g}",
            flush=True,
        )
        DummyControlNode(dds, control_config).run()

    _hydra_main()


if __name__ == "__main__":
    main()
