from hydra import compose, initialize

from robot_learning_lab_sim_infer.profiles.xml_hardware import XmlHardwareProfile
from robot_learning_lab_sim_infer.configs.hydra_sim import SimHardwareConfig
from robot_learning_lab_sim_infer.configs.hydra_sim import register_configs


def test_hydra_sim_config_exposes_xml_and_runtime_overrides():
    register_configs()
    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name="sim_hardware",
            overrides=[
                "xml=/tmp/robot.xml",
                "runtime.simulation_dt=0.001",
                "runtime.dds.publish_hz=60",
                "runtime.viser.render_hz=40",
                "runtime.rc.axis_ranges.vx=[-0.7,1.1]",
                "runtime.rc.key_steps.vx=0.25",
            ],
        )
    assert cfg.xml == "/tmp/robot.xml"
    assert cfg.runtime.simulation_dt == 0.001
    assert cfg.runtime.dds.publish_hz == 60
    assert cfg.runtime.viser.render_hz == 40
    assert list(cfg.runtime.rc.axis_ranges.vx) == [-0.7, 1.1]
    assert cfg.runtime.rc.key_steps.vx == 0.25


def test_default_hydra_xml_is_dr02_motor_model():
    profile = XmlHardwareProfile(SimHardwareConfig().xml)
    model = profile.build_model()
    import mujoco

    data = mujoco.MjData(model)
    profile.reset(model, data)
    assert model.nu == len(profile.joint_names) == 29
    assert model.nsensor == 3
    assert "left_hip_y_joint" in profile.joint_names
    assert profile.extract_robot_state(model, data, 0).base_position[2] == 0.904
