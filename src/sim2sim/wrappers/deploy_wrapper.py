"""Deploy configuration-based observation and action wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim2sim.config import DeployConfig, ObsTerm, ActionTerm
from sim2sim.history import HistoryBuffer


class DeployWrapper:
    """Wrapper that applies deploy.json transformations to observations and actions.

    Handles:
    - Observation scaling: obs_out = scale * raw_obs + offset
    - Action scaling: target = scale * action + offset
    - Clipping for both observations and actions
    - History frame stacking for encoder-based policies
    """

    def __init__(self, cfg: DeployConfig, obs_dim: int | None = None):
        """Initialize the deploy wrapper.

        Args:
            cfg: Deploy configuration.
            obs_dim: Total observation dimension (for history buffer).
        """
        self.cfg = cfg
        self._history_buffer: HistoryBuffer | None = None

        if cfg.history_length > 0:
            if obs_dim is None:
                obs_dim = self._compute_obs_dim()
            self._history_buffer = HistoryBuffer(obs_dim, cfg.history_length)

        self._last_action: np.ndarray | None = None

    def _compute_obs_dim(self) -> int:
        """Compute total observation dimension from concat groups."""
        total = 0
        for group_name, group in self.cfg.observations.concat.items():
            for key in group.keys:
                if key in self.cfg.observations.terms:
                    term = self.cfg.observations.terms[key]
                    total += int(np.prod(term.dim))
        return total

    def reset(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        """Process observations after environment reset.

        Args:
            raw_obs: Raw observation dictionary from simulator.

        Returns:
            Processed observation vector.
        """
        self._last_action = None

        obs = self._process_observations(raw_obs)

        if self._history_buffer is not None:
            self._history_buffer.reset(obs)

        return obs

    def step(self, raw_obs: dict[str, np.ndarray], action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process one step: transform action, step environment, transform observation.

        Args:
            raw_obs: Raw observation dictionary from simulator.
            action: Raw policy action.

        Returns:
            Tuple of (processed_obs, processed_action).
        """
        processed_action = self._process_action(action)
        self._last_action = action.copy()

        obs = self._process_observations(raw_obs)

        if self._history_buffer is not None:
            obs = self._history_buffer.append(obs)

        return obs, processed_action

    def _process_observations(self, raw_obs: dict[str, np.ndarray]) -> np.ndarray:
        """Process raw observations according to deploy.json spec.

        Args:
            raw_obs: Raw observation dictionary from simulator.

        Returns:
            Processed observation vector.
        """
        parts = []

        for group_name, group in self.cfg.observations.concat.items():
            for key in group.keys:
                if key not in self.cfg.observations.terms:
                    continue

                term = self.cfg.observations.terms[key]
                raw_value = self._extract_obs_value(raw_obs, term, key)

                processed = self._transform_obs(raw_value, term)
                parts.append(processed)

        if not parts:
            return np.array([], dtype=np.float32)

        return np.concatenate(parts).astype(np.float32)

    def _extract_obs_value(
        self, raw_obs: dict[str, np.ndarray], term: ObsTerm, key: str
    ) -> np.ndarray:
        """Extract raw observation value from simulator output.

        Handles different observation types by mapping simulator data
        to the expected term format.
        """
        obs_type = term.type

        if obs_type == "velocity_command":
            return raw_obs.get("velocity_command", raw_obs.get("commands", np.zeros(term.dim)))

        elif obs_type == "joint_position":
            if "joint_pos_rel" in raw_obs:
                return raw_obs["joint_pos_rel"]
            return raw_obs.get("joint_position", raw_obs.get("joint_pos", np.zeros(term.dim)))

        elif obs_type == "joint_velocity":
            if "joint_vel_rel" in raw_obs:
                return raw_obs["joint_vel_rel"]
            return raw_obs.get("joint_velocity", raw_obs.get("joint_vel", np.zeros(term.dim)))

        elif obs_type == "joint_torque":
            return raw_obs.get("joint_torque", raw_obs.get("joint_torques", np.zeros(term.dim)))

        elif obs_type == "projected_gravity":
            return raw_obs.get("projected_gravity", np.array([0.0, 0.0, -9.81]))

        elif obs_type == "base_ang_vel":
            return raw_obs.get("base_angular_velocity", raw_obs.get("ang_vel", np.zeros(term.dim)))

        elif obs_type == "last_action":
            if self._last_action is not None:
                return self._last_action
            return np.zeros(term.dim)

        elif obs_type == "array":
            return raw_obs.get(key, np.zeros(term.dim))

        else:
            return raw_obs.get(key, np.zeros(term.dim))

    def _transform_obs(self, raw_value: np.ndarray, term: ObsTerm) -> np.ndarray:
        """Apply scale, offset, and clip to an observation term.

        Formula: obs_out = scale * raw_value + offset
        """
        raw_value = np.asarray(raw_value, dtype=np.float32)
        scale = np.asarray(term.scale, dtype=np.float32)
        offset = np.asarray(term.offset, dtype=np.float32)

        if scale.size == 1:
            scale = np.full_like(raw_value, scale.item())
        if offset.size == 1:
            offset = np.full_like(raw_value, offset.item())

        result = scale * raw_value + offset

        if term.clip:
            clip = np.asarray(term.clip, dtype=np.float32)
            if clip.ndim == 1 and clip.size == 2:
                result = np.clip(result, clip[0], clip[1])
            elif clip.ndim == 2:
                for i, (lo, hi) in enumerate(clip):
                    if i < len(result):
                        result[i] = np.clip(result[i], lo, hi)

        return result

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process raw policy action according to deploy.json spec.

        Formula: target = scale * action + offset

        Args:
            action: Raw action from policy.

        Returns:
            Processed action for simulator.
        """
        parts = []

        for group_name, group in self.cfg.actions.concat.items():
            for key in group.keys:
                if key not in self.cfg.actions.terms:
                    continue

                term = self.cfg.actions.terms[key]
                action_dim = int(np.prod(term.dim))

                action_slice = action[:action_dim]
                action = action[action_dim:]

                processed = self._transform_action(action_slice, term)
                parts.append(processed)

        if not parts:
            return action

        return np.concatenate(parts).astype(np.float32)

    def _transform_action(self, raw_action: np.ndarray, term: ActionTerm) -> np.ndarray:
        """Apply scale, offset, and clip to an action term.

        Formula: target = scale * raw_action + offset
        """
        raw_action = np.asarray(raw_action, dtype=np.float32)
        scale = np.asarray(term.scale, dtype=np.float32)
        offset = np.asarray(term.offset, dtype=np.float32)

        if scale.size == 1:
            scale = np.full_like(raw_action, scale.item())
        if offset.size == 1:
            offset = np.full_like(raw_action, offset.item())

        result = scale * raw_action + offset

        if term.clip:
            clip = np.asarray(term.clip, dtype=np.float32)
            if clip.ndim == 1 and clip.size == 2:
                result = np.clip(result, clip[0], clip[1])
            elif clip.ndim == 2:
                for i, (lo, hi) in enumerate(clip):
                    if i < len(result):
                        result[i] = np.clip(result[i], lo, hi)

        return result

    def get_action_dim(self) -> int:
        """Get total action dimension from config."""
        total = 0
        for group_name, group in self.cfg.actions.concat.items():
            for key in group.keys:
                if key in self.cfg.actions.terms:
                    term = self.cfg.actions.terms[key]
                    total += int(np.prod(term.dim))
        return total

    def get_obs_dim(self) -> int:
        """Get total observation dimension from config."""
        dim = self._compute_obs_dim()
        if self._history_buffer is not None:
            return self._history_buffer.total_dim
        return dim
