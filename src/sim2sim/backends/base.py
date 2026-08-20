"""Abstract simulator backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sim2sim.config import DeployConfig


class SimulatorBackend(ABC):
    """Abstract base class for simulator backends.

    A backend provides environment simulation for policy validation.
    It loads a robot model, applies actions, and returns observations.
    """

    @abstractmethod
    def load(self, model_path: str, cfg: DeployConfig, **kwargs: Any) -> None:
        """Load a robot model and initialize the simulation.

        Args:
            model_path: Path to the robot model file (MJCF, URDF, etc.).
            cfg: Deploy configuration with actuator and observation/action specs.
            **kwargs: Backend-specific options.
        """
        ...

    @abstractmethod
    def reset(self) -> dict[str, np.ndarray]:
        """Reset the simulation to initial state.

        Returns:
            Dictionary mapping observation term names to raw values.
        """
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Execute one simulation step with the given action.

        Args:
            action: Action vector (already scaled/offset by the wrapper).

        Returns:
            Dictionary mapping observation term names to raw values.
        """
        ...

    @abstractmethod
    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observations without stepping.

        Returns:
            Dictionary mapping observation term names to raw values.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up simulation resources."""
        ...

    @property
    @abstractmethod
    def n_joints(self) -> int:
        """Number of actuated joints."""
        ...

    @property
    @abstractmethod
    def joint_names(self) -> list[str]:
        """Ordered list of actuated joint names."""
        ...

    @property
    @abstractmethod
    def dt(self) -> float:
        """Simulation time step."""
        ...
