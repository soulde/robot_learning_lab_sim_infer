from pathlib import Path

import pytest

from robot_learning_lab_sim_infer.profile import PolicyConfig, load_profile, resolve_control_parameters


PROFILE_MEMBERS = """
name = 'fixture'
obs_dim = 3
action_dim = 2
policy_dt = 0.02
decimation = 1
control_defaults = {'kp_scale': 1.0, 'kd_scale': 0.8, 'limit': 4}

def build_model(): return None
def reset(model, data, state): pass
def observe(model, data, state): pass
def apply_action(model, data, action, state): pass
"""


def write_profile(tmp_path: Path, policies: str, *, legacy_callbacks=True) -> Path:
    path = tmp_path / "sim2sim_profile.py"
    members = PROFILE_MEMBERS
    if not legacy_callbacks:
        members = members.replace("def observe(model, data, state): pass\n", "")
        members = members.replace("def apply_action(model, data, action, state): pass\n", "")
    path.write_text(
        "from robot_learning_lab_sim_infer.profile import PolicyConfig\n"
        + members
        + "\ndef create_profile():\n"
        + "    from types import SimpleNamespace\n"
        + f"    policies = {policies}\n"
        + "    return SimpleNamespace(**{key: value for key, value in globals().items() if key in "
        + "('name', 'obs_dim', 'action_dim', 'policy_dt', 'decimation', 'control_defaults', "
        + "'build_model', 'reset', 'observe', 'apply_action')}, policies=policies)\n"
    )
    return path


def test_load_profile_requires_primary_policy(tmp_path):
    path = write_profile(tmp_path, "{}")
    with pytest.raises(ValueError, match="primary"):
        load_profile(path)


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_load_profile_rejects_invalid_policy_dimensions(tmp_path, dimension):
    path = write_profile(
        tmp_path,
        f"{{'primary': PolicyConfig('actor.pt', {dimension!r}, 2)}}",
    )
    with pytest.raises(ValueError, match="obs_dim"):
        load_profile(path)


def test_load_profile_resolves_relative_checkpoints_from_profile(tmp_path):
    path = write_profile(tmp_path, "{'primary': PolicyConfig('weights/actor.pt', 3, 2)}")
    profile = load_profile(path)
    assert profile.policies["primary"].checkpoint == (tmp_path / "weights/actor.pt").resolve()


def test_secondary_control_overrides_inherit_primary_values(tmp_path):
    path = write_profile(
        tmp_path,
        "{'primary': PolicyConfig('stand.pt', 3, 2, {'kp_scale': 0.5, 'kd_scale': 0.4}), "
        "'secondary': PolicyConfig('track.pt', 5, 2, {'kp_scale': 1.2})}",
    )
    profile = load_profile(path)
    assert resolve_control_parameters(profile, "primary") == {
        "kp_scale": 0.5,
        "kd_scale": 0.4,
        "limit": 4,
    }
    assert resolve_control_parameters(profile, "secondary") == {
        "kp_scale": 1.2,
        "kd_scale": 0.4,
        "limit": 4,
    }


def test_single_policy_uses_profile_defaults(tmp_path):
    path = write_profile(tmp_path, "{'primary': PolicyConfig('actor.pt', 3, 2)}")
    profile = load_profile(path)
    assert resolve_control_parameters(profile, "primary") == {
        "kp_scale": 1.0,
        "kd_scale": 0.8,
        "limit": 4,
    }


def test_dual_profile_does_not_require_single_policy_callbacks(tmp_path):
    path = write_profile(
        tmp_path,
        "{'primary': PolicyConfig('stand.pt', 3, 2), "
        "'secondary': PolicyConfig('track.pt', 5, 2)}",
        legacy_callbacks=False,
    )
    profile = load_profile(path)
    assert set(profile.policies) == {"primary", "secondary"}
