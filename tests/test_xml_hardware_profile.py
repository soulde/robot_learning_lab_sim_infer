import pytest

pytest.importorskip("mujoco")
pytest.importorskip("cyclonedds")

from robot_learning_lab_sim_infer.dds.types import (
    MotorCommand,
    MotorControlMode,
    RCCommand,
)
from robot_learning_lab_sim_infer.profiles.xml_hardware import XmlHardwareProfile
from robot_learning_lab_sim_infer.runtime.rc import RCValues


MODEL_XML = """
<mujoco model="xml_profile_test">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="base" pos="0 0 .6">
      <freejoint name="root"/>
      <geom type="box" size=".1 .1 .1" mass="2"/>
      <site name="imu"/>
      <body name="leg" pos="0 0 -.15">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".04" mass=".5"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="hip_motor" joint="hip" gear="1" ctrlrange="-20 20" ctrllimited="true"/></actuator>
  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_accel" site="imu"/>
  </sensor>
</mujoco>
"""


def test_xml_profile_extracts_sensor_state_and_maps_named_motor_command(tmp_path):
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(MODEL_XML)
    profile = XmlHardwareProfile(xml_path)
    model = profile.build_model()
    import mujoco

    data = mujoco.MjData(model)
    profile.reset(model, data)
    command = MotorCommand(
        timestamp_ns=42,
        control_mode=MotorControlMode.POSITION,
        joint_names=["hip"],
        position=[0.5],
        velocity=[0.0],
        kp=[10.0],
        kd=[1.0],
        torque=[0.0],
    )
    profile.apply_motor_command(model, data, command)
    assert data.ctrl[0] == pytest.approx(5.0)

    state = profile.extract_robot_state(model, data, 123)
    assert state.timestamp_ns == 123
    assert list(state.joint_names) == ["hip"]
    assert len(state.imu_orientation_xyzw) == 4
    assert len(state.imu_angular_velocity) == 3
    assert len(state.imu_linear_acceleration) == 3
    rc = profile.make_rc_command(RCValues(enabled=True, vx=0.5), 124)
    assert isinstance(rc, RCCommand)
    assert rc.enabled and rc.vx == pytest.approx(0.5)
