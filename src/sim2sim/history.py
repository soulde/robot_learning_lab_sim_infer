"""Observation history buffer for encoder-based policies."""

from __future__ import annotations

import numpy as np


class HistoryBuffer:
    """Ring buffer for stacking observation history frames.

    Used by policies that require multiple past observations as input
    (e.g., encoder-based architectures with history_length > 0).
    """

    def __init__(self, obs_dim: int, history_length: int, dtype: np.dtype = np.float32):
        """Initialize the history buffer.

        Args:
            obs_dim: Dimension of a single observation frame.
            history_length: Number of history frames to stack (excluding current).
            dtype: Data type for the buffer.
        """
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.dtype = dtype
        self.total_dim = obs_dim * (history_length + 1)
        self._buffer = np.zeros((history_length + 1, obs_dim), dtype=dtype)
        self._initialized = False

    def reset(self, initial_obs: np.ndarray | None = None) -> np.ndarray:
        """Reset the buffer, optionally with an initial observation.

        Args:
            initial_obs: First observation to fill the buffer with.

        Returns:
            Stacked observation history.
        """
        if initial_obs is not None:
            self._buffer[:] = initial_obs.reshape(1, -1)
        else:
            self._buffer[:] = 0.0
        self._initialized = True
        return self.get()

    def append(self, obs: np.ndarray) -> np.ndarray:
        """Append a new observation and return stacked history.

        Args:
            obs: New observation frame.

        Returns:
            Stacked observation history.
        """
        obs = obs.reshape(1, -1)
        if not self._initialized:
            self._buffer[:] = obs
            self._initialized = True
        else:
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = obs
        return self.get()

    def get(self) -> np.ndarray:
        """Return the current stacked observation.

        Returns:
            Flattened stacked observation of shape (total_dim,).
        """
        return self._buffer.reshape(-1).astype(self.dtype)

    @property
    def shape(self) -> tuple[int]:
        """Shape of the stacked observation."""
        return (self.total_dim,)
