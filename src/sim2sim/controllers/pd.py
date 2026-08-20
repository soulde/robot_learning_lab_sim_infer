"""PD position controller."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim2sim.controllers.base import MotorController


class PDController(MotorController):
    """Standard PD position controller.

    Computes torque = Kp * (target_pos - current_pos) - Kd * current_vel.

    This is the default controller for most robot joints in simulation.
    """

    def __init__(
        self,
        kp: np.ndarray | list[float] | float = 30.0,
        kd: np.ndarray | list[float] | float = 0.8,
        torque_limit: np.ndarray | list[float] | None = None,
    ):
        """Initialize PD controller.

        Args:
            kp: Position gain (proportional). Can be scalar or per-joint.
            kd: Velocity gain (derivative). Can be scalar or per-joint.
            torque_limit: Maximum torque per joint. None for unlimited.
        """
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._torque_limit = (
            np.asarray(torque_limit, dtype=np.float64) if torque_limit is not None else None
        )

    def reset(self) -> None:
        """Reset controller state (no state to reset for PD)."""
        pass

    def compute(
        self,
        target: np.ndarray,
        q_pos: np.ndarray,
        q_vel: np.ndarray,
        dt: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute PD torques.

        Args:
            target: Desired joint positions.
            q_pos: Current joint positions.
            q_vel: Current joint velocities.
            dt: Time step (unused for static PD).
            **kwargs: Additional inputs (unused).

        Returns:
            Joint torques.
        """
        target = np.asarray(target, dtype=np.float64)
        q_pos = np.asarray(q_pos, dtype=np.float64)
        q_vel = np.asarray(q_vel, dtype=np.float64)

        position_error = target - q_pos
        torque = self._kp * position_error - self._kd * q_vel

        if self._torque_limit is not None:
            torque = np.clip(torque, -self._torque_limit, self._torque_limit)

        return torque.astype(np.float32)

    @property
    def name(self) -> str:
        """Controller name."""
        return "pd"

    @property
    def kp(self) -> np.ndarray:
        """Position gain."""
        return self._kp.copy()

    @property
    def kd(self) -> np.ndarray:
        """Velocity gain."""
        return self._kd.copy()
