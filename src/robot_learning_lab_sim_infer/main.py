"""Sim2sim CLI entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot_learning_lab_sim_infer.backends.factory import create_backend, list_backends
from robot_learning_lab_sim_infer.config import load_deploy_config
from robot_learning_lab_sim_infer.controllers import PDController, MITCheetahController
from robot_learning_lab_sim_infer.wrappers.deploy_wrapper import DeployWrapper


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sim2sim policy validation using deploy.json configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic torque control with PD controller
  robot_learning_lab_sim_infer -d deploy.json -c model.pt --mjcf robot.xml

  # MIT Cheetah controller
  robot_learning_lab_sim_infer -d deploy.json -c model.pt --mjcf robot.xml --controller mit_cheetah

  # Custom PD gains
  robot_learning_lab_sim_infer -d deploy.json -c model.pt --mjcf robot.xml --kp 50 --kd 2.0

  # Render with human view
  robot_learning_lab_sim_infer -d deploy.json -c model.pt --mjcf robot.xml --render
        """,
    )

    parser.add_argument(
        "--deploy", "-d",
        required=True,
        help="Path to deploy.json configuration file.",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to model checkpoint file.",
    )
    parser.add_argument(
        "--backend", "-b",
        default="mujoco",
        help=f"Simulator backend. Available: {', '.join(list_backends())}",
    )
    parser.add_argument(
        "--mjcf",
        help="Path to MJCF model file (required for MuJoCo backend).",
    )
    parser.add_argument(
        "--policy-loader",
        default="rll_rl",
        choices=["rll_rl"],
        help="Policy checkpoint loader (default: rll_rl).",
    )
    parser.add_argument(
        "--runner-class",
        help="Runner class name for policy loading (auto-detected if not set).",
    )
    parser.add_argument(
        "--controller",
        default="pd",
        choices=["pd", "mit_cheetah"],
        help="Motor controller type (default: pd).",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=None,
        help="Override PD/Kp gain (per-joint or scalar).",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=None,
        help="Override PD/Kd gain (per-joint or scalar).",
    )
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=None,
        help="Torque limit per joint (Nm).",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (default: cpu).",
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Enable human rendering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--gravity",
        nargs=3,
        type=float,
        metavar=("GX", "GY", "GZ"),
        help="Gravity vector override (e.g., --gravity 0 0 -9.81).",
    )
    parser.add_argument(
        "--virtual-spring-k",
        type=float,
        default=None,
        help="Virtual spring stiffness for MIT Cheetah controller.",
    )
    parser.add_argument(
        "--virtual-damper-d",
        type=float,
        default=None,
        help="Virtual damper coefficient for MIT Cheetah controller.",
    )

    return parser.parse_args()


def run_episode(
    backend: Any,
    wrapper: DeployWrapper,
    policy_fn: Any,
    max_steps: int,
    verbose: bool = True,
) -> tuple[float, int]:
    """Run a single episode.

    Args:
        backend: Simulator backend.
        wrapper: Deploy wrapper for obs/action processing.
        policy_fn: Policy inference function.
        max_steps: Maximum number of steps.
        verbose: Print progress.

    Returns:
        Tuple of (total_reward, steps).
    """
    raw_obs = backend.reset()
    obs = wrapper.reset(raw_obs)

    total_reward = 0.0
    for step in range(max_steps):
        action = policy_fn(obs)
        raw_obs = backend.step(action)
        obs, processed_action = wrapper.step(raw_obs, action)

        reward = 0.0
        total_reward += reward

        if verbose and step % 100 == 0:
            print(f"  Step {step}/{max_steps}, reward: {total_reward:.2f}")

    return total_reward, step + 1


def main() -> None:
    """Main entry point."""
    args = parse_args()

    np.random.seed(args.seed)

    print(f"Loading deploy config from: {args.deploy}")
    cfg = load_deploy_config(args.deploy)
    print(f"  format_version: {cfg.format_version}")
    print(f"  step_dt: {cfg.step_dt}")
    print(f"  history_length: {cfg.history_length}")
    print(f"  actuator joints: {len(cfg.actuator.stiffness)}")

    print(f"\nLoading policy from: {args.checkpoint}")
    from robot_learning_lab_sim_infer.policy.rll_rl import RllRlPolicyLoader
    loader = RllRlPolicyLoader()
    loader.load(
        args.checkpoint,
        runner_class=args.runner_class,
        device=args.device,
    )
    policy_fn = loader.get_inference_policy(device=args.device)
    print("  Policy loaded successfully")

    print(f"\nInitializing {args.backend} backend...")
    backend = create_backend(args.backend, render_mode="human" if args.render else None)

    if args.backend == "mujoco":
        if not args.mjcf:
            print("Error: --mjcf is required for MuJoCo backend")
            sys.exit(1)

        controller_kwargs = {}
        if args.kp is not None:
            controller_kwargs["kp"] = args.kp
        if args.kd is not None:
            controller_kwargs["kd"] = args.kd
        if args.torque_limit is not None:
            controller_kwargs["torque_limit"] = args.torque_limit
        if args.virtual_spring_k is not None:
            controller_kwargs["virtual_spring_k"] = args.virtual_spring_k
        if args.virtual_damper_d is not None:
            controller_kwargs["virtual_damper_d"] = args.virtual_damper_d

        backend.load(
            args.mjcf,
            cfg,
            gravity=args.gravity,
            controller=args.controller,
            controller_kwargs=controller_kwargs,
        )
    else:
        raise NotImplementedError(f"Backend '{args.backend}' not yet implemented")

    print(f"  Loaded model with {backend.n_joints} joints")
    print(f"  Controller: {backend.controller.name}")

    wrapper = DeployWrapper(cfg, obs_dim=backend.n_joints * 2)

    print(f"\nRunning {args.episodes} episodes...")
    total_reward = 0.0
    total_steps = 0
    start_time = time.time()

    for episode in range(args.episodes):
        reward, steps = run_episode(backend, wrapper, policy_fn, args.max_steps)
        total_reward += reward
        total_steps += steps
        print(f"  Episode {episode + 1}: reward={reward:.2f}, steps={steps}")

    elapsed = time.time() - start_time
    print(f"\nResults:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Mean reward: {total_reward / args.episodes:.2f}")
    print(f"  Total steps: {total_steps}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  FPS: {total_steps / elapsed:.1f}")

    backend.close()
    loader.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
