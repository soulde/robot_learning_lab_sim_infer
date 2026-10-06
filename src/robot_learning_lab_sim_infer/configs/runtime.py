"""Generic node defaults; robot and task details belong in external configs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AxisBinding:
    axis: str
    direction: float


@dataclass
class DDSConfig:
    domain_id: int = 0
    cyclonedds_uri: str | None = None
    state_topic: str = "robot/state"
    rc_topic: str = "robot/rc_command"
    motor_command_topic: str = "robot/motor_command"
    state_timeout_s: float = 0.25
    rc_timeout_s: float = 0.5
    motor_command_timeout_s: float = 0.1
    publish_hz: float = 100.0


@dataclass
class RCConfig:
    """Generic keyboard-to-RC mapping; no policy or FSM semantics live here."""

    key_axes: dict[str, AxisBinding] = field(
        default_factory=lambda: {
            "I": AxisBinding("vx", 1.0),
            "K": AxisBinding("vx", -1.0),
            "J": AxisBinding("vy", 1.0),
            "L": AxisBinding("vy", -1.0),
            "U": AxisBinding("yaw_rate", 1.0),
            "O": AxisBinding("yaw_rate", -1.0),
        }
    )
    scales: dict[str, float] = field(
        default_factory=lambda: {"vx": 1.0, "vy": 1.0, "yaw_rate": 1.0}
    )
    key_steps: dict[str, float] = field(
        default_factory=lambda: {"vx": 0.1, "vy": 0.1, "yaw_rate": 0.1}
    )
    axis_ranges: dict[str, list[float]] = field(
        default_factory=lambda: {
            "vx": [-1.0, 1.0],
            "vy": [-0.6, 0.6],
            "yaw_rate": [-1.5, 1.5],
        }
    )
    body_height_range: list[float] = field(default_factory=lambda: [0.2, 0.5])
    clear_key: str = "Z"
    mode_keys: dict[str, int] = field(
        default_factory=lambda: {str(mode): mode for mode in range(10)}
    )


@dataclass
class StateMachineConfig:
    """Policy-node transition defaults. The sim node must not consume this."""

    initial_state: str = "idle"
    transitions: dict[str, dict[str, str]] = field(default_factory=dict)
    state_policies: dict[str, str | None] = field(
        default_factory=lambda: {"idle": None}
    )
    mode_events: dict[int, str] = field(default_factory=dict)
    button_events: dict[int, str] = field(default_factory=dict)


@dataclass
class ViserConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    enabled: bool = True
    render_hz: float = 30.0


@dataclass
class RuntimeConfig:
    simulation_dt: float = 0.002
    dds: DDSConfig = field(default_factory=DDSConfig)
    rc: RCConfig = field(default_factory=RCConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    viser: ViserConfig = field(default_factory=ViserConfig)
