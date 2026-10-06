from types import SimpleNamespace

import pytest

from robot_learning_lab_sim_infer.config_api import compose_config, validate_profile
from robot_learning_lab_sim_infer.configs import RCConfig, RuntimeConfig, StateMachineConfig
from robot_learning_lab_sim_infer.runtime.rc import RCMapper
from robot_learning_lab_sim_infer.runtime.state_machine import RuntimeStateMachine


def test_compose_config_keeps_package_and_external_ownership_separate():
    external = object()
    runtime = RuntimeConfig()
    result = compose_config(external, runtime)
    assert result.runtime is runtime
    assert result.external is external


def test_validate_policy_profile_reports_missing_members():
    with pytest.raises(TypeError, match="observe"):
        validate_profile(SimpleNamespace(), kind="policy")


def test_state_machine_transitions_and_reset():
    machine = RuntimeStateMachine(
        StateMachineConfig(
            initial_state="idle",
            transitions={"idle": {"enable": "walk"}, "walk": {"disable": "idle"}},
            state_policies={"idle": None, "walk": "velocity"},
        )
    )
    assert machine.decision.policy is None
    assert machine.update("enable").policy == "velocity"
    assert machine.update("disable").state == "idle"
    assert machine.update("unknown").state == "idle"


def test_rc_mapper_steps_axis_commands_z_clears_and_keyboard_stays_enabled():
    mapper = RCMapper(RCConfig())
    first = mapper.key_event("I", True)
    assert first.enabled and first.vx == 0.1
    assert mapper.key_event("I", True).vx == 0.2
    assert mapper.key_event("K", True).vx == 0.1
    values = mapper.key_event("Z", True)
    assert values.vx == values.vy == values.yaw_rate == 0.0
    assert values.enabled
    assert mapper.key_event("Z").enabled


def test_rc_mapper_limits_configured_axis_values():
    mapper = RCMapper(
        RCConfig(
            key_axes={"X": ("vx", 1.0), "Y": ("vx", 1.0)},
            key_steps={"vx": 0.75},
            axis_ranges={"vx": [-0.8, 1.2]},
        )
    )
    assert mapper.key_event("X", True).vx == 0.75
    assert mapper.key_event("Y", True).vx == 1.2


def test_rc_mapper_uses_configured_physical_ranges():
    mapper = RCMapper(
        RCConfig(
            key_axes={"W": ("vx", 1.0), "S": ("vx", -1.0)},
            key_steps={"vx": 2.0},
            axis_ranges={"vx": [-0.8, 1.2]},
        )
    )
    assert mapper.key_event("W", True).vx == 1.2
    assert mapper.key_event("S", True).vx == -0.8


def test_rc_mapper_keyboard_clear_preserves_enable_and_gui_values():
    mapper = RCMapper(RCConfig())
    mapper.set_values(SimpleNamespace(enabled=True, mode=2, vx=0.4, vy=-0.2,
                                      yaw_rate=0.1, body_height=0.33))
    cleared = mapper.key_event("Z")
    assert (cleared.vx, cleared.vy, cleared.yaw_rate) == (0.0, 0.0, 0.0)
    assert cleared.enabled and cleared.mode == 2


def test_rc_mapper_numeric_hotkeys_publish_policy_selection_and_damping_modes():
    mapper = RCMapper(RCConfig())
    assert mapper.key_event("1").mode == 1
    assert mapper.key_event("9").mode == 9
    assert mapper.key_event("0").mode == 0


def test_rc_mapper_reset_clears_motion_and_selects_damping_mode():
    mapper = RCMapper(RCConfig())
    mapper.key_event("I")
    mapper.key_event("1")

    values = mapper.reset()

    assert values.enabled
    assert values.mode == 0
    assert (values.vx, values.vy, values.yaw_rate) == (0.0, 0.0, 0.0)


def test_state_machine_rejects_undefined_transition_target():
    with pytest.raises(ValueError, match="unknown states"):
        RuntimeStateMachine(
            StateMachineConfig(
                initial_state="idle",
                transitions={"idle": {"go": "active"}},
                state_policies={"idle": None},
            )
        )
