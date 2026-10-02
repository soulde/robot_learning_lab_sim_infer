"""Generic real-time TorchScript MuJoCo runner with profile-driven UI."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from .profile import RuntimeState, Sim2SimProfile, dispatch_key

_TORCH_THREADS_CONFIGURED = False


def _validate_multi_policy_profile(profile: Sim2SimProfile) -> None:
    if callable(getattr(profile, "policy_weights", None)):
        required = ("observe_for_policy", "apply_weighted_policy_actions", "apply_transition_control")
        missing = [name for name in required if not callable(getattr(profile, name, None))]
        if missing:
            raise TypeError(
                "A weighted multi-policy profile must implement: " + ", ".join(missing)
            )
        return
    required = (
        "active_policy",
        "observe_for_policy",
        "apply_policy_action",
        "apply_transition_control",
    )
    missing = [name for name in required if not callable(getattr(profile, name, None))]
    if missing:
        raise TypeError(
            "A profile with a secondary policy must implement the multi-policy contract: "
            + ", ".join(missing)
        )


def _infer_policy_action(profile, policy, policy_name, model, data, state, step):
    config = profile.policies[policy_name]
    try:
        observation = np.asarray(profile.observe_for_policy(policy_name, model, data, state), dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Invalid observation for policy '{policy_name}' at step {step}") from exc
    if observation.shape != (config.obs_dim,) or not np.isfinite(observation).all():
        raise ValueError(f"Invalid observation for policy '{policy_name}' at step {step}: shape={observation.shape}")
    try:
        with torch.no_grad():
            output = policy(torch.from_numpy(observation).unsqueeze(0))
        action = output.detach().cpu().numpy().ravel()
    except Exception as exc:
        raise RuntimeError(f"Inference failed for policy '{policy_name}' at step {step}") from exc
    if action.shape != (config.action_dim,):
        raise ValueError(f"Invalid action for policy '{policy_name}' at step {step}: shape={action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"Invalid non-finite action for policy '{policy_name}' at step {step}")
    return action


def _validated_policy_weights(profile, model, data, state, step):
    weights = profile.policy_weights(model, data, state)
    if not isinstance(weights, Mapping):
        raise TypeError(f"policy_weights must return a mapping at step {step}")
    weights = dict(weights)
    # An empty mapping explicitly hands control to apply_transition_control.
    if not weights:
        return weights
    missing = set(profile.policies) - set(weights)
    if missing:
        raise ValueError(f"Missing policy weights at step {step}: {', '.join(sorted(missing))}")
    unknown = set(weights) - set(profile.policies)
    if unknown:
        raise ValueError(f"Unknown policy weights at step {step}: {', '.join(sorted(unknown))}")
    values = np.asarray(list(weights.values()), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"Policy weights must be finite and non-negative at step {step}")
    if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"Policy weights must sum to 1 at step {step}; got {values.sum():.8g}")
    return weights


def load_torchscript_policy(path: str | Path, obs_dim: int, action_dim: int, *, policy_name: str = "policy"):
    label = f"policy '{policy_name}' checkpoint {path}"
    try:
        policy = torch.jit.load(str(path), map_location="cpu")
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Expected a TorchScript actor for {label}") from exc
    policy.eval()
    with torch.no_grad():
        output = policy(torch.zeros(1, obs_dim, dtype=torch.float32))
    if tuple(output.shape) != (1, action_dim):
        raise ValueError(f"Output shape mismatch for {label}: expected (1, {action_dim}), got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError(f"non-finite output during validation for {label}")
    return policy


def run(
    profile: Sim2SimProfile,
    *,
    headless: bool = False,
    steps: int = 0,
    render_hz: float = 25.0,
) -> None:
    """Run policy inference in real time and optionally render a profile UI."""
    import mujoco

    global _TORCH_THREADS_CONFIGURED
    if not _TORCH_THREADS_CONFIGURED:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch cannot change this after its inter-op pool has started.
            pass
        _TORCH_THREADS_CONFIGURED = True
    multi_policy = "secondary" in profile.policies
    weighted_multi_policy = multi_policy and callable(getattr(profile, "policy_weights", None))
    if multi_policy:
        _validate_multi_policy_profile(profile)
    policies = {
        name: load_torchscript_policy(
            config.checkpoint,
            config.obs_dim,
            config.action_dim,
            policy_name=name,
        )
        for name, config in profile.policies.items()
    }
    primary_config = profile.policies["primary"]
    model = profile.build_model()
    data = mujoco.MjData(model)
    state = RuntimeState(command=np.zeros(3), action=np.zeros(primary_config.action_dim))
    profile.reset(model, data, state)

    glfw = window = camera = perturb = option = scene = context = viewport = None
    if not headless:
        from mujoco.glfw import glfw as mujoco_glfw

        glfw = mujoco_glfw
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        window = glfw.create_window(1600, 1000, f"{profile.name} sim2sim", None, None)
        if window is None:
            glfw.terminate()
            raise RuntimeError("Could not create MuJoCo viewer window")
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.0, 0.0, 0.8]
        camera.distance = 6.0
        camera.elevation = -15.0
        perturb = mujoco.MjvPerturb()
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=2000)
        context = mujoco.MjrContext(model, 300)
        viewport = mujoco.MjrRect(0, 0, 1600, 1000)
        mouse_button = {"button": None, "pressed": False, "last": (0.0, 0.0)}

        def key_callback(win, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key == glfw.KEY_R:
                profile.reset(model, data, state)
                return
            dispatch_key(profile, chr(key).upper(), model, data, state)

        def mouse_callback(win, button, action, mods):
            mouse_button["button"] = button
            mouse_button["pressed"] = action == glfw.PRESS
            mouse_button["last"] = glfw.get_cursor_pos(win)

        def cursor_callback(win, xpos, ypos):
            previous = mouse_button["last"]
            mouse_button["last"] = (xpos, ypos)
            if not mouse_button["pressed"]:
                return
            width, height = glfw.get_window_size(win)
            dx, dy = xpos - previous[0], ypos - previous[1]
            if mouse_button["button"] == glfw.MOUSE_BUTTON_RIGHT:
                camera.distance = max(0.2, camera.distance * (1.0 + dy / max(height, 1)))
            elif mouse_button["button"] == glfw.MOUSE_BUTTON_LEFT:
                mujoco.mjv_moveCamera(
                    model,
                    mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                    dx / max(width, 1),
                    dy / max(height, 1),
                    camera,
                )

        glfw.set_key_callback(window, key_callback)
        glfw.set_mouse_button_callback(window, mouse_callback)
        glfw.set_cursor_pos_callback(window, cursor_callback)

    policy_step = 0
    wall_start = time.perf_counter()
    next_render_deadline = wall_start
    last_fps_time = wall_start
    fps_policy = fps_physics = fps_render = 0
    render_period = 1.0 / render_hz if render_hz > 0 else float("inf")
    try:
        while headless or not glfw.window_should_close(window):
            if steps > 0 and policy_step >= steps:
                break
            if weighted_multi_policy:
                policy_weights = _validated_policy_weights(profile, model, data, state, policy_step)
                policy_actions = {
                    name: _infer_policy_action(profile, policies[name], name, model, data, state, policy_step)
                    for name in policy_weights
                }
                active_policy = "weighted" if policy_weights else None
            elif multi_policy:
                active_policy = profile.active_policy(model, data, state)
                if active_policy is not None and active_policy not in policies:
                    raise ValueError(
                        f"Invalid active policy '{active_policy}' at step {policy_step}; "
                        f"expected one of {tuple(policies)} or None"
                    )
            else:
                active_policy = "primary"

            if active_policy is None or weighted_multi_policy:
                action = None
            else:
                config = profile.policies[active_policy]
                observation_fn = (
                    profile.observe_for_policy if multi_policy else profile.observe
                )
                try:
                    observation = np.asarray(
                        observation_fn(model, data, state)
                        if not multi_policy
                        else observation_fn(active_policy, model, data, state),
                        dtype=np.float32,
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Invalid observation for policy '{active_policy}' at step {policy_step}"
                    ) from exc
                if observation.shape != (config.obs_dim,) or not np.isfinite(observation).all():
                    raise ValueError(
                        f"Invalid observation for policy '{active_policy}' at step {policy_step}: "
                        f"shape={observation.shape}"
                    )
                try:
                    with torch.no_grad():
                        output = policies[active_policy](torch.from_numpy(observation).unsqueeze(0))
                    action = output.detach().cpu().numpy().ravel()
                except Exception as exc:
                    raise RuntimeError(
                        f"Inference failed for policy '{active_policy}' at step {policy_step}"
                    ) from exc
                if action.shape != (config.action_dim,):
                    raise ValueError(
                        f"Invalid action for policy '{active_policy}' at step {policy_step}: "
                        f"shape={action.shape}"
                    )
                if not np.isfinite(action).all():
                    raise ValueError(
                        f"Invalid non-finite action for policy '{active_policy}' at step {policy_step}"
                    )
                state.action = action

            for _ in range(profile.decimation):
                if weighted_multi_policy and policy_weights:
                    profile.apply_weighted_policy_actions(
                        model, data, policy_actions, policy_weights, state
                    )
                elif active_policy is None:
                    profile.apply_transition_control(model, data, state)
                elif multi_policy:
                    profile.apply_policy_action(active_policy, model, data, state.action, state)
                else:
                    profile.apply_action(model, data, state.action, state)
                mujoco.mj_step(model, data)
                fps_physics += 1
            policy_step += 1
            fps_policy += 1

            now = time.perf_counter()
            if window is not None and now >= next_render_deadline:
                target_fn = getattr(profile, "camera_target", None)
                target = np.asarray(target_fn(model, data) if target_fn else [data.qpos[0], data.qpos[1], data.qpos[2] + 0.5])
                camera.lookat[:] += 0.2 * (target - camera.lookat[:])
                mujoco.mjv_updateScene(model, data, option, perturb, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                width, height = glfw.get_window_size(window)
                viewport.width, viewport.height = width, height
                mujoco.mjr_render(viewport, scene, context)
                overlay_fn = getattr(profile, "overlay", None)
                if overlay_fn is not None:
                    command_text, controls_text = overlay_fn(model, data, state)
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        viewport,
                        command_text,
                        controls_text,
                        context,
                    )
                glfw.swap_buffers(window)
                fps_render += 1
                while next_render_deadline <= now:
                    next_render_deadline += render_period
            if glfw is not None:
                glfw.poll_events()

            delay = wall_start + policy_step * profile.policy_dt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            now = time.perf_counter()
            if now - last_fps_time >= 1.0:
                elapsed = now - last_fps_time
                print(
                    f"[fps] physics={fps_physics / elapsed:.0f}Hz "
                    f"policy={fps_policy / elapsed:.0f}Hz render={fps_render / elapsed:.0f}Hz",
                    flush=True,
                )
                fps_policy = fps_physics = fps_render = 0
                last_fps_time = now
    finally:
        if window is not None:
            glfw.destroy_window(window)
            glfw.terminate()
