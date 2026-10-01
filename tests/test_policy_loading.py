import sys
from types import SimpleNamespace

import pytest
import torch

from robot_learning_lab_sim_infer import main
from robot_learning_lab_sim_infer.profile import PolicyConfig
from robot_learning_lab_sim_infer.runner import load_torchscript_policy, run


class Actor(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, observation):
        return observation[:, :1].expand(-1, self.action_dim)


class NonFiniteActor(torch.nn.Module):
    def forward(self, observation):
        return observation[:, :1] * torch.tensor(float("inf"))


def save_actor(path, obs_dim, action_dim, actor=None):
    actor = actor or Actor(action_dim)
    scripted = torch.jit.trace(actor, torch.zeros(1, obs_dim))
    scripted.save(str(path))
    return path


def test_torchscript_actors_validate_independent_dimensions(tmp_path):
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    secondary = save_actor(tmp_path / "secondary.pt", 5, 4)
    assert tuple(load_torchscript_policy(primary, 3, 2)(torch.zeros(1, 3)).shape) == (1, 2)
    assert tuple(load_torchscript_policy(secondary, 5, 4)(torch.zeros(1, 5)).shape) == (1, 4)


def test_torchscript_shape_error_names_checkpoint(tmp_path):
    path = save_actor(tmp_path / "wrong.pt", 3, 2)
    with pytest.raises(ValueError, match="wrong.pt"):
        load_torchscript_policy(path, 3, 3)


def test_malformed_torchscript_error_names_checkpoint(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_text("not a torchscript archive")
    with pytest.raises(RuntimeError, match="broken.pt"):
        load_torchscript_policy(path, 3, 2)


def test_non_finite_dummy_output_is_rejected(tmp_path):
    path = save_actor(tmp_path / "nonfinite.pt", 3, 1, NonFiniteActor())
    with pytest.raises(ValueError, match="non-finite"):
        load_torchscript_policy(path, 3, 1)


@pytest.mark.parametrize("option", ["--checkpoint", "--second-checkpoint"])
def test_cli_rejects_checkpoint_options(monkeypatch, option):
    monkeypatch.setattr(sys, "argv", ["sim-infer", "--profile", "unused.py", option, "actor.pt"])
    monkeypatch.setattr(main, "load_profile", lambda path: SimpleNamespace(name="fixture", policies={"primary": PolicyConfig("actor.pt", 3, 2)}))
    monkeypatch.setattr(main, "run", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit):
        main.main()


def test_run_loads_all_policies_before_building_scene(tmp_path, monkeypatch):
    import types

    monkeypatch.setitem(sys.modules, "mujoco", types.ModuleType("mujoco"))
    primary = save_actor(tmp_path / "primary.pt", 3, 2)
    profile = SimpleNamespace(
        policies={
            "primary": PolicyConfig(primary, 3, 2),
            "secondary": PolicyConfig(tmp_path / "missing.pt", 5, 2),
        },
        active_policy=lambda model, data, state: "primary",
        observe_for_policy=lambda name, model, data, state: torch.zeros(
            profile.policies[name].obs_dim
        ).numpy(),
        apply_policy_action=lambda name, model, data, action, state: None,
        apply_transition_control=lambda model, data, state: None,
        build_model=lambda: pytest.fail("scene construction must follow actor validation"),
    )
    with pytest.raises(RuntimeError, match="missing.pt"):
        run(profile, headless=True)
