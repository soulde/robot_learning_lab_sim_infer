"""Hydra entry configuration for the XML-driven simulated hardware node."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore

from .runtime import RuntimeConfig


def _default_demo_xml() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "assets"
        / "dr02_motor"
        / "dr02_torque.xml"
    )


@dataclass
class SimHardwareConfig:
    xml: str = field(default_factory=_default_demo_xml)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def register_configs() -> None:
    ConfigStore.instance().store(name="sim_hardware", node=SimHardwareConfig)


__all__ = ["SimHardwareConfig", "register_configs"]
