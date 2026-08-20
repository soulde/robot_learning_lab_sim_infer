"""Abstract motor controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MotorController(ABC):
    """Abstract base class for motor controllers.

    A motor controller converts desired targets (e.g., joint positions)
    into motor torques applied to the simulator.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state."""
        ...

    @abstractmethod
    def compute(
        self,
        target: np.ndarray,
        q_pos: np.ndarray,
        q_vel: np.ndarray,
        dt: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute motor torques from target and current state.

        Args:
            target: Desired target (position, velocity, or torque depending on mode).
            q_pos: Current joint positions.
            q_vel: Current joint velocities.
            dt: Time step.
            **kwargs: Additional inputs (e.g., gravity, friction coefficients).

        Returns:
            Motor torques to apply.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller name."""
        ...
