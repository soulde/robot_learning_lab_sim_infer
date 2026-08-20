"""Deploy configuration dataclasses and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConcatGroup:
    """Concatenation group for organizing obs/action terms."""

    dim: int
    keys: list[str]


@dataclass
class ActionTerm:
    """Action space term definition."""

    dim: list[int]
    type: str  # joint_position | joint_velocity | joint_torque
    scale: list[float]
    offset: list[float] = field(default_factory=lambda: [0.0])
    clip: list[list[float]] = field(default_factory=list)
    joints: list[str] = field(default_factory=list)


@dataclass
class ObsTerm:
    """Observation space term definition."""

    dim: list[int]
    type: str  # velocity_command | joint_position | joint_velocity | etc.
    scale: list[float]
    offset: list[float] = field(default_factory=lambda: [0.0])
    clip: list[list[float]] = field(default_factory=list)
    joints: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionSpace:
    """Action space with concat groups and terms."""

    concat: dict[str, ConcatGroup] = field(default_factory=dict)
    terms: dict[str, ActionTerm] = field(default_factory=dict)


@dataclass
class ObsSpace:
    """Observation space with concat groups and terms."""

    concat: dict[str, ConcatGroup] = field(default_factory=dict)
    terms: dict[str, ObsTerm] = field(default_factory=dict)


@dataclass
class ActuatorConfig:
    """PD controller gains."""

    stiffness: list[float]
    damping: list[float]


@dataclass
class DeployConfig:
    """Full deploy.json configuration."""

    format_version: int
    step_dt: float
    actuator: ActuatorConfig
    actions: ActionSpace
    observations: ObsSpace
    history_length: int = 0


def _parse_concat(data: dict[str, Any]) -> dict[str, ConcatGroup]:
    """Parse concat groups from dict."""
    result = {}
    for name, group in data.items():
        result[name] = ConcatGroup(dim=group["dim"], keys=group["keys"])
    return result


def _parse_action_terms(data: dict[str, Any]) -> dict[str, ActionTerm]:
    """Parse action terms from dict."""
    result = {}
    for name, term in data.items():
        scale = term["scale"]
        if isinstance(scale, (int, float)):
            scale = [scale]
        offset = term.get("offset", [0.0])
        if isinstance(offset, (int, float)):
            offset = [offset]
        clip = term.get("clip", [])
        if clip and isinstance(clip[0], (int, float)):
            clip = [clip]
        result[name] = ActionTerm(
            dim=term["dim"],
            type=term["type"],
            scale=scale,
            offset=offset,
            clip=clip,
            joints=term.get("joints", []),
        )
    return result


def _parse_obs_terms(data: dict[str, Any]) -> dict[str, ObsTerm]:
    """Parse observation terms from dict."""
    result = {}
    for name, term in data.items():
        scale = term["scale"]
        if isinstance(scale, (int, float)):
            scale = [scale]
        offset = term.get("offset", [0.0])
        if isinstance(offset, (int, float)):
            offset = [offset]
        clip = term.get("clip", [])
        if clip and isinstance(clip[0], (int, float)):
            clip = [clip]
        result[name] = ObsTerm(
            dim=term["dim"],
            type=term["type"],
            scale=scale,
            offset=offset,
            clip=clip,
            joints=term.get("joints", []),
            params=term.get("params", {}),
        )
    return result


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load and validate a deploy.json configuration file.

    Args:
        path: Path to the deploy.json file.

    Returns:
        Parsed DeployConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the configuration is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Deploy config not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if data.get("format_version") != 1:
        raise ValueError(f"Unsupported format_version: {data.get('format_version')}")

    if "actuator" not in data:
        raise ValueError("Missing required field: actuator")

    actuator_data = data["actuator"]
    if len(actuator_data["stiffness"]) != len(actuator_data["damping"]):
        raise ValueError("actuator.stiffness and actuator.damping length mismatch")

    actuator = ActuatorConfig(
        stiffness=actuator_data["stiffness"],
        damping=actuator_data["damping"],
    )

    actions_data = data.get("actions", {})
    actions = ActionSpace(
        concat=_parse_concat(actions_data.get("concat", {})),
        terms=_parse_action_terms(
            {k: v for k, v in actions_data.items() if k != "concat"}
        ),
    )

    obs_data = data.get("observations", {})
    observations = ObsSpace(
        concat=_parse_concat(obs_data.get("concat", {})),
        terms=_parse_obs_terms(
            {k: v for k, v in obs_data.items() if k != "concat"}
        ),
    )

    return DeployConfig(
        format_version=data["format_version"],
        step_dt=data["step_dt"],
        actuator=actuator,
        actions=actions,
        observations=observations,
        history_length=data.get("history_length", 0),
    )


def save_deploy_config(cfg: DeployConfig, path: str | Path) -> None:
    """Save a DeployConfig to a JSON file.

    Args:
        cfg: DeployConfig to save.
        path: Output file path.
    """
    data = {
        "format_version": cfg.format_version,
        "step_dt": cfg.step_dt,
        "actuator": {
            "stiffness": cfg.actuator.stiffness,
            "damping": cfg.actuator.damping,
        },
        "actions": {
            "concat": {k: {"dim": v.dim, "keys": v.keys} for k, v in cfg.actions.concat.items()},
        },
        "observations": {
            "concat": {k: {"dim": v.dim, "keys": v.keys} for k, v in cfg.observations.concat.items()},
        },
    }

    for name, term in cfg.actions.terms.items():
        data["actions"][name] = {
            "dim": term.dim,
            "type": term.type,
            "scale": term.scale,
            "offset": term.offset,
            "clip": term.clip,
            "joints": term.joints,
        }

    for name, term in cfg.observations.terms.items():
        data["observations"][name] = {
            "dim": term.dim,
            "type": term.type,
            "scale": term.scale,
            "offset": term.offset,
            "clip": term.clip,
            "joints": term.joints,
            "params": term.params,
        }

    if cfg.history_length > 0:
        data["history_length"] = cfg.history_length

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
