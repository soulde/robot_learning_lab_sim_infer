"""Contracts implemented by externally supplied, process-specific profiles."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class SimHardwareProfile(Protocol):
    """Robot-specific operations that require ownership of MuJoCo state."""

    name: str
    joint_names: Sequence[str]

    def build_model(self) -> Any: ...

    def reset(self, model: Any, data: Any) -> None: ...

    def extract_robot_state(self, model: Any, data: Any, timestamp_ns: int) -> Any: ...

    def make_rc_command(self, rc_values: Any, timestamp_ns: int) -> Any: ...

    def apply_motor_command(self, model: Any, data: Any, command: Any) -> None: ...

    def apply_safe_command(self, model: Any, data: Any) -> None: ...


class PolicyProfile(Protocol):
    """Policy-side operations; methods receive wire samples, never MuJoCo data."""

    name: str
    joint_names: Sequence[str]
    policy_dt: float

    def reset(self) -> None: ...

    def observe(self, policy_name: str, robot_state: Any, rc_command: Any) -> np.ndarray: ...

    def infer(self, policy_name: str, observation: np.ndarray) -> np.ndarray: ...

    def make_motor_command(
        self, policy_name: str, action: np.ndarray, timestamp_ns: int
    ) -> Any: ...
