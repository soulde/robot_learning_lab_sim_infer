from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize_config_module

from robot_learning_lab_sim_infer.dds.types import MotorControlMode
from robot_learning_lab_sim_infer.nodes.dummy_control_node import (
    DummyControlConfig,
    DummyControlNode,
    PolicySlotConfig,
)


class FakeDDS:
    def __init__(self):
        self.commands = []
        self.closed = False

    def take_robot_state(self):
        return None

    def take_rc_command(self):
        return None

    def publish_motor_command(self, command):
        self.commands.append(command)

    def close(self):
        self.closed = True


def make_state():
    return SimpleNamespace(
        timestamp_ns=1000,
        joint_names=["hip", "knee"],
        joint_position=[0.3, -0.4],
        joint_velocity=[0.1, -0.2],
        joint_effort=[1.0, 2.0],
        base_position=[1.0, 2.0, 0.9],
        base_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        base_linear_velocity=[0.1, 0.2, 0.3],
        base_angular_velocity=[0.0, 0.0, 0.4],
        imu_orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        imu_angular_velocity=[0.0, 0.0, 0.4],
        imu_linear_acceleration=[1.0, 2.0, 3.0],
        contact_force_xyz=[0.0, 0.0, 4.0],
    )


def make_rc(mode):
    return SimpleNamespace(
        timestamp_ns=1100, enabled=True, mode=mode, vx=0.2, vy=-0.1,
        yaw_rate=0.3, body_height=0.35,
    )


def test_selected_policy_logs_switch_observe_infer_and_uses_its_gains(capsys):
    dds = FakeDDS()
    node = DummyControlNode(
        dds,
        DummyControlConfig(
            policies=[PolicySlotConfig(1, "walk", "walk.pt", 12.0, 0.8)],
            damping_kd=0.6,
        ),
        clock_ns=lambda: 1234,
    )

    command = node.step(make_state(), make_rc(1))

    assert command is dds.commands[0]
    assert command.timestamp_ns == 1234
    assert command.control_mode == MotorControlMode.MIT
    assert command.position == [0.3, -0.4]
    assert command.velocity == [0.0, 0.0]
    assert command.kp == [12.0, 12.0]
    assert command.kd == [0.8, 0.8]
    assert command.torque == [0.0, 0.0]
    output = capsys.readouterr().out
    assert "switch damping -> policy_1" in output
    assert "policy[1] observe placeholder" in output
    assert "policy[1] infer placeholder" in output
    assert "q:[0.3, -0.4]" in output and "imu_accel:[1.0, 2.0, 3.0]" in output


def test_unconfigured_policy_number_has_no_effect(capsys):
    dds = FakeDDS()
    node = DummyControlNode(
        dds,
        DummyControlConfig(policies=[PolicySlotConfig(1, "walk", "", 12.0, 0.8)]),
    )

    node.step(make_state(), make_rc(9))

    assert len(dds.commands) == 1
    assert dds.commands[-1].kp == [0.0, 0.0]
    assert "switch" not in capsys.readouterr().out


def test_unconfigured_number_does_not_leave_current_policy(capsys):
    node = DummyControlNode(
        FakeDDS(),
        DummyControlConfig(policies=[PolicySlotConfig(1, "walk", "", 12.0, 0.8)]),
    )
    node.step(make_state(), make_rc(1))
    capsys.readouterr()

    command = node.step(make_state(), make_rc(9))

    assert command.kp == [12.0, 12.0]
    assert "switch" not in capsys.readouterr().out


def test_zero_selects_configured_damping_command(capsys):
    dds = FakeDDS()
    node = DummyControlNode(
        dds,
        DummyControlConfig(
            policies=[PolicySlotConfig(1, "walk", "", 12.0, 0.8)], damping_kd=0.45
        ),
    )
    node.step(make_state(), make_rc(1))
    capsys.readouterr()

    command = node.step(make_state(), make_rc(0))

    assert command.control_mode == MotorControlMode.MIT
    assert command.position == [0.0, 0.0]
    assert command.kp == [0.0, 0.0]
    assert command.kd == [0.45, 0.45]
    assert command.torque == [0.0, 0.0]
    assert "switch policy_1 -> damping" in capsys.readouterr().out


def test_policy_config_is_hydra_yaml_and_accepts_overrides():
    with initialize_config_module(
        version_base=None,
        config_module="robot_learning_lab_sim_infer.configs",
    ):
        cfg = compose(
            config_name="dummy_control",
            overrides=[
                "dds.domain_id=31",
                "control.rate_hz=50",
                "control.policies=[{index:1,name:walk,policy_file:null,kp:24,kd:0.8}]",
                "control.damping_kd=0.7",
            ],
        )
    assert cfg.dds.domain_id == 31
    assert cfg.control.rate_hz == 50
    assert cfg.control.policies[0].kp == 24
    assert cfg.control.damping_kd == 0.7
    assert cfg.control.policies[0].policy_file is None
