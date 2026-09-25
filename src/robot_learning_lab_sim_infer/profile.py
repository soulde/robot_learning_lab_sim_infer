"""External robot profile contract for the generic sim2sim runner."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np


@dataclass
class RuntimeState:
    command: np.ndarray
    action: np.ndarray
    values: dict[str, Any] = field(default_factory=dict)


class Sim2SimProfile(Protocol):
    name: str
    obs_dim: int
    action_dim: int
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
    required = ("name", "obs_dim", "action_dim", "policy_dt", "decimation", "build_model", "reset", "observe", "apply_action")
    missing = [name for name in required if not hasattr(profile, name)]
    if missing:
        raise TypeError(f"Profile {profile_path} is missing required members: {', '.join(missing)}")
    return profile
