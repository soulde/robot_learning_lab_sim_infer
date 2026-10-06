"""Load and compose external node configuration without merging ownership."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .configs import RuntimeConfig


@dataclass(frozen=True)
class NodeConfig:
    runtime: RuntimeConfig
    external: Any


def load_external_config(spec: str) -> Any:
    """Load ``module:attribute`` and instantiate class factories without args."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Config spec must use the form 'python.module:ConfigClass'")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    return value() if isinstance(value, type) else value


def compose_config(external: Any, runtime: RuntimeConfig | None = None) -> NodeConfig:
    return NodeConfig(runtime=runtime or RuntimeConfig(), external=external)


def validate_sim_profile(profile: Any) -> None:
    """Validate the simulator-owned robot contract with useful errors."""
    required = (
        "name", "joint_names", "build_model", "reset",
        "extract_robot_state", "make_rc_command", "apply_motor_command",
        "apply_safe_command",
    )
    missing = [name for name in required if not hasattr(profile, name)]
    if missing:
        raise TypeError(f"Sim profile is missing required members: {', '.join(missing)}")
    names = tuple(profile.joint_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Sim profile joint_names must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("Sim profile joint_names contains duplicates")
