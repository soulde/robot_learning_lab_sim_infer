from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import mujoco
import numpy as np
import pytest
import torch

from robot_learning_lab_sim_infer import runner as runner_module
from robot_learning_lab_sim_infer.profile import (
    PolicyConfig,
    RuntimeState,
    dispatch_key,
)
from robot_learning_lab_sim_infer.runner import run

XML = """<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""


class Actor(torch.nn.Module):
    def __init__(self, action_dim, conditional_nan=False):
        super().__init__()
        self.action_dim = action_dim
        self.conditional_nan = conditional_nan

    def forward(self, observation):
        action = observation[:, :1].expand(-1, self.action_dim)
        if self.conditional_nan:
            action = torch.where(observation[:, :1] > 0, action * float("nan"), action)
        return action


def save_actor(path: Path, obs_dim: int, action_dim: int, *, conditional_nan=False):
    actor = torch.jit.trace(
        Actor(action_dim, conditional_nan), torch.zeros(1, obs_dim), check_trace=False
    )
    actor.save(str(path))
    return path


class SingleProfile:
    name = "single"
    policy_dt = 0.0
    decimation = 1
    policies: ClassVar[dict] = {}

    def __init__(self, checkpoint):
        self.policies = {"primary": PolicyConfig(checkpoint, 3, 2)}
        self.actions = []
        self.reset_count = 0

    def build_model(self):
        return mujoco.MjModel.from_xml_string(XML)

    def reset(self, model, data, state):
        self.reset_count += 1
        mujoco.mj_resetData(model, data)

    def observe(self, model, data, state):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def apply_action(self, model, data, action, state):
        self.actions.append(action.copy())


class DualProfile(SingleProfile):
    name = "dual"

    def __init__(self, primary, secondary, active):
        super().__init__(primary)
        self.policies["secondary"] = PolicyConfig(secondary, 5, 3)
        self.active = iter(active)
        self.observed = []
        self.applied = []
        self.transitions = 0
        self.multi_keys = []

    def active_policy(self, model, data, state):
        return next(self.active)

    def observe_for_policy(self, policy_name, model, data, state):
        dim = self.policies[policy_name].obs_dim
        self.observed.append((policy_name, dim))
        return np.ones(dim, dtype=np.float32)

    def apply_policy_action(self, policy_name, model, data, action, state):
        self.applied.append((policy_name, action.copy(), state.action.shape))

    def apply_transition_control(self, model, data, state):
        self.transitions += 1

    def handle_policy_key(self, key, model, data, state):
        self.multi_keys.append(key)


class WeightedDualProfile(SingleProfile):
    name = "weighted-dual"

    def __init__(self, primary, secondary, weights):
        super().__init__(primary)
        self.policies["secondary"] = PolicyConfig(secondary, 5, 3)
        self.weights = iter(weights)
        self.observed = []
        self.blends = []
        self.transitions = 0

    def policy_weights(self, model, data, state):
        return next(self.weights)

    def observe_for_policy(self, policy_name, model, data, state):
        self.observed.append(policy_name)
        return np.ones(self.policies[policy_name].obs_dim, dtype=np.float32)

    def apply_weighted_policy_actions(self, model, data, actions, weights, state):
        self.blends.append(({name: action.copy() for name, action in actions.items()}, dict(weights)))

    def apply_transition_control(self, model, data, state):
        self.transitions += 1


def test_primary_only_run_keeps_legacy_profile_callbacks(tmp_path):
    checkpoint = save_actor(tmp_path / "primary.pt", 3, 2)
    profile = SingleProfile(checkpoint)
    run(profile, headless=True, steps=2)
    assert len(profile.actions) == 2
    assert profile.actions[0].shape == (2,)


def test_dual_policy_routes_distinct_actor_shapes_and_transition(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    secondary = save_actor(tmp_path / "secondary.pt", 5, 3)
    profile = DualProfile(primary, secondary, ["primary", "secondary", None])
    run(profile, headless=True, steps=3)
    assert profile.observed == [("primary", 3), ("secondary", 5)]
    assert [item[0] for item in profile.applied] == ["primary", "secondary"]
    assert profile.applied[0][1].shape == (2,)
    assert profile.applied[1][1].shape == (3,)
    assert profile.applied[1][2] == (3,)
    assert profile.transitions == 1


def test_weighted_dual_policy_infers_both_actors_and_routes_blend_weights(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    secondary = save_actor(tmp_path / "secondary.pt", 5, 3)
    profile = WeightedDualProfile(
        primary,
        secondary,
        [
            {"primary": 1.0, "secondary": 0.0},
            {"primary": 0.25, "secondary": 0.75},
        ],
    )

    run(profile, headless=True, steps=2)

    assert profile.observed == ["primary", "secondary", "primary", "secondary"]
    assert [set(actions) for actions, _ in profile.blends] == [
        {"primary", "secondary"},
        {"primary", "secondary"},
    ]
    assert [weights for _, weights in profile.blends] == [
        {"primary": 1.0, "secondary": 0.0},
        {"primary": 0.25, "secondary": 0.75},
    ]


def test_weighted_dual_policy_rejects_weights_that_do_not_sum_to_one(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    secondary = save_actor(tmp_path / "secondary.pt", 5, 3)
    profile = WeightedDualProfile(primary, secondary, [{"primary": 0.2, "secondary": 0.2}])

    with pytest.raises(ValueError, match="weights must sum to 1"):
        run(profile, headless=True, steps=1)
    assert profile.blends == []


@pytest.mark.parametrize("weights", [{"primary": 1.0}])
def test_weighted_dual_policy_requires_weights_for_every_actor(tmp_path, weights):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    secondary = save_actor(tmp_path / "secondary.pt", 5, 3)
    profile = WeightedDualProfile(primary, secondary, [weights])

    with pytest.raises(ValueError, match="Missing policy weights"):
        run(profile, headless=True, steps=1)

    assert profile.observed == []
    assert profile.blends == []


def test_weighted_dual_policy_allows_empty_weights_for_transition_control(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    secondary = save_actor(tmp_path / "secondary.pt", 5, 3)
    profile = WeightedDualProfile(primary, secondary, [{}])

    run(profile, headless=True, steps=1)

    assert profile.observed == []
    assert profile.blends == []
    assert profile.transitions == 1


def test_unknown_active_policy_is_rejected_before_action(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    secondary = save_actor(tmp_path / "secondary.pt", 5, 3)
    profile = DualProfile(primary, secondary, ["critic"])
    with pytest.raises(ValueError, match="critic.*step 0"):
        run(profile, headless=True, steps=1)
    assert profile.applied == []


def test_invalid_runtime_observation_is_rejected(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    profile = SingleProfile(primary)
    profile.observe = lambda *args: np.array([np.nan, 0, 0])
    with pytest.raises(ValueError, match="primary.*step 0"):
        run(profile, headless=True, steps=1)
    assert profile.actions == []


def test_non_finite_runtime_action_is_rejected_before_apply(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2, conditional_nan=True)
    profile = SingleProfile(primary)
    with pytest.raises(ValueError, match="non-finite action.*primary.*step 0"):
        run(profile, headless=True, steps=1)
    assert profile.actions == []


def test_policy_key_handler_is_only_called_with_secondary():
    calls = []
    profile = SimpleNamespace(
        policies={"primary": object()},
        handle_key=lambda key, model, data, state: calls.append(("single", key)),
        handle_policy_key=lambda key, model, data, state: calls.append(("multi", key)),
    )
    state = RuntimeState(np.zeros(3), np.zeros(1))
    dispatch_key(profile, "T", None, None, state)
    assert calls == [("single", "T")]

    profile.policies["secondary"] = object()
    dispatch_key(profile, "P", None, None, state)
    assert calls == [("single", "T"), ("single", "P"), ("multi", "P")]


def test_secondary_requires_multi_policy_callbacks_before_scene_creation(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    secondary = save_actor(tmp_path / "secondary.pt", 5, 3)
    profile = SimpleNamespace(
        name="invalid-dual",
        policy_dt=0.0,
        decimation=1,
        policies={
            "primary": PolicyConfig(primary, 3, 2),
            "secondary": PolicyConfig(secondary, 5, 3),
        },
        build_model=lambda: pytest.fail("contract validation must precede model construction"),
    )
    with pytest.raises(TypeError, match="active_policy"):
        run(profile, headless=True, steps=1)


def test_runner_continues_when_torch_interop_pool_is_already_initialized(tmp_path, monkeypatch):
    checkpoint = save_actor(tmp_path / "primary.pt", 3, 2)
    profile = SingleProfile(checkpoint)
    monkeypatch.setattr(runner_module, "_TORCH_THREADS_CONFIGURED", False)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)

    def reject_late_configuration(count):
        raise RuntimeError("cannot set interop threads after parallel work")

    monkeypatch.setattr(torch, "set_num_interop_threads", reject_late_configuration)
    run(profile, headless=True, steps=1)
    assert len(profile.actions) == 1
