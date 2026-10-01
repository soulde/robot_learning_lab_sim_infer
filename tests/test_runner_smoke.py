from pathlib import Path

import mujoco
import numpy as np
import torch

from robot_learning_lab_sim_infer.profile import PolicyConfig
from robot_learning_lab_sim_infer.runner import run


XML = "<mujoco><worldbody><body pos='0 0 1'><freejoint/><geom type='sphere' size='0.1'/></body></worldbody></mujoco>"


class FixtureActor(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, observation):
        return observation[:, :1].expand(-1, self.action_dim)


def make_actor(path: Path, obs_dim: int, action_dim: int):
    actor = torch.jit.trace(FixtureActor(action_dim), torch.zeros(1, obs_dim), check_trace=False)
    actor.save(str(path))
    return path


class SmokeProfile:
    name = "smoke"
    policy_dt = 0.0
    decimation = 2

    def __init__(self, primary, secondary=None):
        self.policies = {"primary": PolicyConfig(primary, 3, 2)}
        if secondary is not None:
            self.policies["secondary"] = PolicyConfig(secondary, 5, 3)
        self.actions = []
        self.transitions = 0
        self.selections = iter(["primary", "secondary"])

    def build_model(self):
        return mujoco.MjModel.from_xml_string(XML)

    def reset(self, model, data, state):
        mujoco.mj_resetData(model, data)

    def observe(self, model, data, state):
        return np.array([1, 0, 0], dtype=np.float32)

    def apply_action(self, model, data, action, state):
        self.actions.append(action.copy())

    def active_policy(self, model, data, state):
        return next(self.selections)

    def observe_for_policy(self, policy_name, model, data, state):
        return np.ones(self.policies[policy_name].obs_dim, dtype=np.float32)

    def apply_policy_action(self, policy_name, model, data, action, state):
        self.actions.append(action.copy())

    def apply_transition_control(self, model, data, state):
        self.transitions += 1


def test_headless_finite_step_smoke_for_single_and_dual_policy(tmp_path):
    primary = make_actor(tmp_path / "primary.pt", 3, 2)
    secondary = make_actor(tmp_path / "secondary.pt", 5, 3)

    single = SmokeProfile(primary)
    run(single, headless=True, steps=2)
    assert len(single.actions) == 4
    assert all(action.shape == (2,) and np.isfinite(action).all() for action in single.actions)

    dual = SmokeProfile(primary, secondary)
    run(dual, headless=True, steps=2)
    assert [action.shape for action in dual.actions] == [(2,), (2,), (3,), (3,)]
    assert dual.transitions == 0
