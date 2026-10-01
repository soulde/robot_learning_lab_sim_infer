"""External robot profile contract for the generic sim2sim runner."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PolicyConfig:
    checkpoint: str | Path
    obs_dim: int
    action_dim: int
    control_overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeState:
    command: np.ndarray
    action: np.ndarray
    values: dict[str, Any] = field(default_factory=dict)


class Sim2SimProfile(Protocol):
    name: str
    policies: Mapping[str, PolicyConfig]
    policy_dt: float
    decimation: int

    def build_model(self): ...
    def reset(self, model, data, state: RuntimeState) -> None: ...
    def observe(self, model, data, state: RuntimeState) -> np.ndarray: ...
    def apply_action(self, model, data, action: np.ndarray, state: RuntimeState) -> None: ...

    def handle_key(self, key: str, model, data, state: RuntimeState) -> bool: ...

    def overlay(self, model, data, state: RuntimeState) -> tuple[str, str]: ...


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("external_sim2sim_profile", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load profile module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_profile(path: str | Path) -> Sim2SimProfile:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
    module = _load_module(profile_path)
    factory = getattr(module, "create_profile", None)
    if factory is None:
        raise AttributeError(f"Profile {profile_path} must define create_profile()")
    profile = factory()
    required = ("name", "policy_dt", "decimation", "build_model", "reset", "observe", "apply_action")
    missing = [name for name in required if not hasattr(profile, name)]
    if missing:
        raise TypeError(f"Profile {profile_path} is missing required members: {', '.join(missing)}")
    policies = getattr(profile, "policies", None)
    if not isinstance(policies, Mapping) or "primary" not in policies:
        raise ValueError(f"Profile {profile_path} must define policies with a required 'primary' entry")
    unknown = set(policies) - {"primary", "secondary"}
    if unknown:
        raise ValueError(f"Profile {profile_path} has unknown policy entries: {', '.join(sorted(unknown))}")
    normalized: dict[str, PolicyConfig] = {}
    for name, config in policies.items():
        if not isinstance(config, PolicyConfig):
            raise TypeError(f"Profile policy '{name}' must be a PolicyConfig")
        for dim_name in ("obs_dim", "action_dim"):
            value = getattr(config, dim_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Profile policy '{name}' {dim_name} must be a positive integer")
        checkpoint = Path(config.checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = profile_path.parent / checkpoint
        overrides = config.control_overrides
        if not isinstance(overrides, Mapping):
            raise TypeError(f"Profile policy '{name}' control_overrides must be a mapping")
        normalized[name] = replace(config, checkpoint=checkpoint.resolve())
    profile.policies = normalized
    defaults = getattr(profile, "control_defaults", {})
    if not isinstance(defaults, Mapping):
        raise TypeError("Profile control_defaults must be a mapping")
    return profile


def resolve_control_parameters(profile: Sim2SimProfile, policy_name: str) -> dict[str, Any]:
    """Merge defaults, primary overrides, and the selected policy's overrides."""
    if policy_name not in profile.policies:
        raise KeyError(f"Unknown policy '{policy_name}'")
    resolved = dict(getattr(profile, "control_defaults", {}))
    resolved.update(profile.policies["primary"].control_overrides)
    if policy_name != "primary":
        resolved.update(profile.policies[policy_name].control_overrides)
    return resolved


def dispatch_key(profile: Sim2SimProfile, key: str, model, data, state: RuntimeState) -> None:
    """Dispatch ordinary profile keys, enabling policy keys only for dual-policy profiles."""
    handler = getattr(profile, "handle_key", None)
    if handler is not None:
        handler(key, model, data, state)
    if "secondary" in profile.policies:
        policy_handler = getattr(profile, "handle_policy_key", None)
        if policy_handler is not None:
            policy_handler(key, model, data, state)
