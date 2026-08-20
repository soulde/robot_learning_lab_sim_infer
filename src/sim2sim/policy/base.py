"""Abstract policy loader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np


class PolicyLoader(ABC):
    """Abstract base class for loading and running policies.

    A policy loader reads a checkpoint file and provides an inference
    function that maps observations to actions.
    """

    @abstractmethod
    def load(self, checkpoint_path: str | Path, **kwargs: Any) -> None:
        """Load a policy checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint.
            **kwargs: Loader-specific options.
        """
        ...

    @abstractmethod
    def get_inference_policy(self, device: str | None = None) -> Callable[[np.ndarray], np.ndarray]:
        """Return the inference function.

        Args:
            device: Device to run inference on (if applicable).

        Returns:
            Callable that maps observation array to action array.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...
