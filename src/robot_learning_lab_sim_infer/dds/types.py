"""Cyclone DDS Python types matching :mod:`dds/idl/robot_v1.idl`.

The apt-provided ``idlc`` on Ubuntu ships the C generator. Cyclone DDS Python's
``py`` backend is tied to the Cyclone version and Python ABI used to build its
wheel, so these small wire structs are declared directly with the official
Cyclone ``IdlStruct`` API. Keep field order and bounds in sync with the IDL.
"""

from dataclasses import dataclass
from enum import auto

from cyclonedds.idl import IdlEnum, IdlStruct
from cyclonedds.idl import types

_TYPE_NS = "robot_learning_lab_sim_infer.msg.v1"


class MotorControlMode(IdlEnum, typename=f"{_TYPE_NS}.MotorControlMode"):
    POSITION = auto()
    VELOCITY = auto()
    TORQUE = auto()
    MIT = auto()


@dataclass
class RobotState(IdlStruct, typename=f"{_TYPE_NS}.RobotState"):
    timestamp_ns: types.uint64
    joint_names: types.sequence[types.bounded_str[64], 64]
    joint_position: types.sequence[types.float64, 64]
    joint_velocity: types.sequence[types.float64, 64]
    joint_effort: types.sequence[types.float64, 64]
    base_position: types.array[types.float64, 3]
    base_orientation_xyzw: types.array[types.float64, 4]
    base_linear_velocity: types.array[types.float64, 3]
    base_angular_velocity: types.array[types.float64, 3]
    imu_orientation_xyzw: types.array[types.float64, 4]
    imu_angular_velocity: types.array[types.float64, 3]
    imu_linear_acceleration: types.array[types.float64, 3]
    contact_force_xyz: types.sequence[types.float64, 256]


@dataclass
class RCCommand(IdlStruct, typename=f"{_TYPE_NS}.RCCommand"):
    timestamp_ns: types.uint64
    enabled: bool
    mode: types.uint16
    vx: types.float32
    vy: types.float32
    yaw_rate: types.float32
    body_height: types.float32
    button_mask: types.uint32


@dataclass
class MotorCommand(IdlStruct, typename=f"{_TYPE_NS}.MotorCommand"):
    timestamp_ns: types.uint64
    control_mode: MotorControlMode
    joint_names: types.sequence[types.bounded_str[64], 64]
    position: types.sequence[types.float64, 64]
    velocity: types.sequence[types.float64, 64]
    kp: types.sequence[types.float64, 64]
    kd: types.sequence[types.float64, 64]
    torque: types.sequence[types.float64, 64]


__all__ = ["MotorControlMode", "RobotState", "RCCommand", "MotorCommand"]
