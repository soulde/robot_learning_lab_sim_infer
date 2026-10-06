"""Generic MuJoCo hardware profile for XMLs with direct joint motor actuators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..dds.types import MotorControlMode, RCCommand, RobotState


class XmlHardwareProfile:
    """Build a simulated hardware interface from one MJCF file.

    The initial generic adapter expects one direct motor actuator per scalar
    hinge/slide joint. Position and velocity requests use the command's PD
    gains; torque requests pass through as joint effort.
    """

    def __init__(self, xml_path: str | Path):
        self.xml_path = Path(xml_path).expanduser().resolve()
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo XML does not exist: {self.xml_path}")
        self.name = self.xml_path.stem
        self._model: Any | None = None
        self._joint_ids: dict[str, int] = {}
        self._joint_actuators: dict[str, int] = {}
        self._base_body_id = 1

    def build_model(self):
        if self._model is None:
            import mujoco

            self._model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self._index_model(self._model, mujoco)
        return self._model

    @property
    def joint_names(self) -> tuple[str, ...]:
        self.build_model()
        return tuple(self._joint_actuators)

    def _index_model(self, model: Any, mujoco: Any) -> None:
        scalar_types = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
        transmission = mujoco.mjtTrn.mjTRN_JOINT
        no_bias = mujoco.mjtBias.mjBIAS_NONE
        for actuator_id in range(model.nu):
            if model.actuator_trntype[actuator_id] != transmission:
                continue
            if model.actuator_biastype[actuator_id] != no_bias:
                continue
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            if joint_id < 0 or model.jnt_type[joint_id] not in scalar_types:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                raise ValueError("All actuated scalar joints in an XML profile must be named")
            if name in self._joint_actuators:
                raise ValueError(f"XML has multiple direct motor actuators for joint {name!r}")
            self._joint_ids[name] = joint_id
            self._joint_actuators[name] = actuator_id
        if not self._joint_ids:
            raise ValueError(
                "MuJoCo XML must define at least one named hinge/slide joint driven by a direct motor"
            )
        if len(self._joint_ids) > 64:
            raise ValueError("This DDS schema supports at most 64 actuated joints")
        free_joint = next(
            (jid for jid in range(model.njnt)
             if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE),
            None,
        )
        if free_joint is not None:
            self._base_body_id = int(model.jnt_bodyid[free_joint])

    def reset(self, model: Any, data: Any) -> None:
        import mujoco

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    def extract_robot_state(self, model: Any, data: Any, timestamp_ns: int) -> RobotState:
        import mujoco

        positions, velocities, efforts = [], [], []
        for name, joint_id in self._joint_ids.items():
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            positions.append(float(data.qpos[qpos_adr]))
            velocities.append(float(data.qvel[dof_adr]))
            efforts.append(float(data.qfrc_actuator[dof_adr]))

        body_id = self._base_body_id
        base_position = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
        base_quat_wxyz = np.asarray(data.xquat[body_id], dtype=np.float64).copy()
        spatial_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial_velocity, 0
        )
        imu_quat = self._sensor(model, data, mujoco.mjtSensor.mjSENS_FRAMEQUAT)
        imu_gyro = self._sensor(model, data, mujoco.mjtSensor.mjSENS_GYRO)
        imu_accel = self._sensor(model, data, mujoco.mjtSensor.mjSENS_ACCELEROMETER)
        quat_wxyz = base_quat_wxyz if imu_quat is None else imu_quat
        angular_velocity = spatial_velocity[:3] if imu_gyro is None else imu_gyro
        linear_acceleration = np.zeros(3) if imu_accel is None else imu_accel
        contacts = np.asarray(data.cfrc_ext[1:, 3:6], dtype=np.float64).reshape(-1)

        return RobotState(
            timestamp_ns=int(timestamp_ns),
            joint_names=list(self._joint_ids),
            joint_position=positions,
            joint_velocity=velocities,
            joint_effort=efforts,
            base_position=base_position.tolist(),
            base_orientation_xyzw=quat_wxyz[[1, 2, 3, 0]].tolist(),
            base_linear_velocity=spatial_velocity[3:].tolist(),
            base_angular_velocity=spatial_velocity[:3].tolist(),
            imu_orientation_xyzw=quat_wxyz[[1, 2, 3, 0]].tolist(),
            imu_angular_velocity=angular_velocity.tolist(),
            imu_linear_acceleration=linear_acceleration.tolist(),
            contact_force_xyz=contacts[:256].tolist(),
        )

    def _sensor(self, model: Any, data: Any, sensor_type: Any) -> np.ndarray | None:
        import mujoco

        for sensor_id in range(model.nsensor):
            if model.sensor_type[sensor_id] == sensor_type:
                address = int(model.sensor_adr[sensor_id])
                size = int(model.sensor_dim[sensor_id])
                return np.asarray(data.sensordata[address : address + size], dtype=np.float64).copy()
        return None

    @staticmethod
    def make_rc_command(rc_values: Any, timestamp_ns: int) -> RCCommand:
        return RCCommand(
            timestamp_ns=int(timestamp_ns),
            enabled=bool(rc_values.enabled),
            mode=int(rc_values.mode),
            vx=float(rc_values.vx),
            vy=float(rc_values.vy),
            yaw_rate=float(rc_values.yaw_rate),
            body_height=float(rc_values.body_height),
            button_mask=0,
        )

    def apply_motor_command(self, model: Any, data: Any, command: Any) -> None:
        names = tuple(command.joint_names)
        if len(names) != len(set(names)):
            raise ValueError("MotorCommand joint_names contains duplicates")
        unknown = set(names) - set(self._joint_actuators)
        if unknown:
            raise ValueError(f"MotorCommand names joints not driven by this XML: {sorted(unknown)}")
        mode = command.control_mode
        for i, name in enumerate(names):
            joint_id = self._joint_ids[name]
            actuator_id = self._joint_actuators[name]
            qpos = float(data.qpos[int(model.jnt_qposadr[joint_id])])
            qvel = float(data.qvel[int(model.jnt_dofadr[joint_id])])
            feedforward = float(command.torque[i]) if i < len(command.torque) else 0.0
            target_velocity = float(command.velocity[i]) if i < len(command.velocity) else 0.0
            if mode == MotorControlMode.TORQUE:
                effort = feedforward
            elif mode == MotorControlMode.VELOCITY:
                kd = float(command.kd[i]) if i < len(command.kd) else 0.0
                effort = kd * (target_velocity - qvel) + feedforward
            elif mode in (MotorControlMode.POSITION, MotorControlMode.MIT):
                target_position = float(command.position[i]) if i < len(command.position) else 0.0
                kp = float(command.kp[i]) if i < len(command.kp) else 0.0
                kd = float(command.kd[i]) if i < len(command.kd) else 0.0
                effort = kp * (target_position - qpos) + kd * (target_velocity - qvel) + feedforward
            else:
                raise ValueError(f"Unsupported motor control mode: {mode!r}")

            gear = float(model.actuator_gear[actuator_id, 0])
            if gear == 0.0:
                raise ValueError(f"XML motor for joint {name!r} has zero gear")
            control = effort / gear
            if model.actuator_ctrllimited[actuator_id]:
                low, high = model.actuator_ctrlrange[actuator_id]
                control = float(np.clip(control, low, high))
            data.ctrl[actuator_id] = control

    @staticmethod
    def apply_safe_command(model: Any, data: Any) -> None:
        data.ctrl[:] = 0.0
