"""Independent state-machine and inference process."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import replace
from typing import Any, Protocol

import numpy as np

from ..config_api import load_external_config, validate_profile
from ..configs import RuntimeConfig
from ..dds.transport import PolicyDDS
from ..runtime.state_machine import RuntimeStateMachine


class PolicyIO(Protocol):
    def take_robot_state(self) -> Any | None: ...
    def take_rc_command(self) -> Any | None: ...
    def publish_motor_command(self, sample: Any) -> None: ...
    def close(self) -> None: ...


class PolicyNode:
    """Run configured state transitions and inference from DDS samples only."""

    def __init__(
        self,
        profile: Any,
        runtime: RuntimeConfig,
        dds: PolicyIO,
        *,
        clock_ns=time.time_ns,
        monotonic=time.monotonic,
    ):
        validate_profile(profile, kind="policy")
        self.profile = profile
        self.runtime = runtime
        self.dds = dds
        self._clock_ns = clock_ns
        self._monotonic = monotonic
        self._state_machine = RuntimeStateMachine(runtime.state_machine)

    def run(self, stop_event: threading.Event | None = None) -> None:
        try:
            self.profile.reset()
            robot_state = rc_command = None
            state_received = rc_received = float("-inf")
            previous_rc = None
            deadline = self._monotonic()
            period = float(self.profile.policy_dt)
            while stop_event is None or not stop_event.is_set():
                now = self._monotonic()
                new_state = self.dds.take_robot_state()
                new_rc = self.dds.take_rc_command()
                if new_state is not None:
                    robot_state, state_received = new_state, now
                if new_rc is not None:
                    rc_command, rc_received = new_rc, now

                if rc_command is not None and now - rc_received <= self.runtime.dds.rc_timeout_s:
                    event = self._rc_event(previous_rc, rc_command)
                    if event is not None:
                        self._state_machine.update(event)
                    previous_rc = rc_command

                if self._inputs_fresh(now, robot_state, rc_command, state_received, rc_received):
                    decision = self._state_machine.decision
                    if bool(rc_command.enabled) and decision.policy is not None:
                        self._publish_policy_command(decision.policy, robot_state, rc_command)

                deadline += period
                delay = deadline - self._monotonic()
                if delay > 0:
                    if stop_event is not None:
                        stop_event.wait(delay)
                    else:
                        time.sleep(delay)
                else:
                    deadline = self._monotonic()
        except KeyboardInterrupt:
            pass
        finally:
            self.dds.close()

    def _inputs_fresh(self, now, robot_state, rc_command, state_received, rc_received) -> bool:
        if robot_state is None or rc_command is None:
            return False
        return (
            now - state_received <= self.runtime.dds.state_timeout_s
            and now - rc_received <= self.runtime.dds.rc_timeout_s
        )

    def _rc_event(self, previous: Any | None, current: Any) -> str | None:
        cfg = self.runtime.state_machine
        if previous is None or bool(previous.enabled) != bool(current.enabled):
            return "enable" if current.enabled else "disable"
        if int(previous.mode) != int(current.mode):
            return cfg.mode_events.get(int(current.mode))
        changed = int(previous.button_mask) ^ int(current.button_mask)
        for bit, event in cfg.button_events.items():
            if changed & int(bit) and int(current.button_mask) & int(bit):
                return event
        return None

    def _publish_policy_command(self, policy_name: str, robot_state: Any, rc_command: Any) -> None:
        names = tuple(robot_state.joint_names)
        expected = tuple(self.profile.joint_names)
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError(
                "RobotState joint names do not match the policy profile; "
                f"expected={expected}, received={names}"
            )
        observation = np.asarray(
            self.profile.observe(policy_name, robot_state, rc_command), dtype=np.float32
        )
        if observation.ndim != 1 or not np.isfinite(observation).all():
            raise ValueError(f"Policy '{policy_name}' produced an invalid observation")
        action = np.asarray(self.profile.infer(policy_name, observation), dtype=np.float32)
        if action.ndim != 1 or not np.isfinite(action).all():
            raise ValueError(f"Policy '{policy_name}' produced an invalid action")
        command = self.profile.make_motor_command(policy_name, action, self._clock_ns())
        self.dds.publish_motor_command(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDS policy and inference node")
    parser.add_argument("--config", required=True, help="Python module:ConfigClass or module:instance")
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--cyclonedds-uri")
    args = parser.parse_args()
    external = load_external_config(args.config)
    profile = getattr(external, "policy_profile", external)
    runtime = getattr(external, "runtime", RuntimeConfig())
    if not isinstance(runtime, RuntimeConfig):
        raise TypeError("External config runtime must be a RuntimeConfig instance")
    dds_cfg = runtime.dds
    if args.domain_id is not None:
        dds_cfg = replace(dds_cfg, domain_id=args.domain_id)
    if args.cyclonedds_uri is not None:
        dds_cfg = replace(dds_cfg, cyclonedds_uri=args.cyclonedds_uri)
    runtime = replace(runtime, dds=dds_cfg)
    dds = PolicyDDS(runtime.dds)
    print(
        f"policy_node profile={profile.name} domain={runtime.dds.domain_id}",
        flush=True,
    )
    PolicyNode(profile, runtime, dds).run()


if __name__ == "__main__":
    main()
