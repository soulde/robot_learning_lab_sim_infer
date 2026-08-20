"""Go2 sim2sim validation example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_learning_lab_sim_infer import load_deploy_config, build_combined_mjcf
from robot_learning_lab_sim_infer.backends import create_backend
from robot_learning_lab_sim_infer.controllers import PDController, MITCheetahController
from robot_learning_lab_sim_infer.wrappers import DeployWrapper


def main() -> None:
    """Run Go2 validation with PD controller."""
    base_dir = Path(__file__).parent

    deploy_path = base_dir / "deploy.json"
    robot_path = base_dir / "go2.xml"

    print("Loading deploy config...")
    cfg = load_deploy_config(deploy_path)
    print(f"  step_dt: {cfg.step_dt}")
    print(f"  joints: {len(cfg.actuator.stiffness)}")

    print("\nBuilding combined scene...")
    scene_path = build_combined_mjcf(
        robot_path=robot_path,
        output_path=base_dir / "go2_scene.xml",
    )
    print(f"  Output: {scene_path}")

    print("\nCreating MuJoCo backend...")
    backend = create_backend("mujoco")

    controller = PDController(
        kp=np.array(cfg.actuator.stiffness),
        kd=np.array(cfg.actuator.damping),
    )

    backend.load(
        scene_path,
        cfg,
        controller=controller,
    )
    print(f"  Joints: {backend.n_joints}")
    print(f"  Controller: {backend.controller.name}")

    wrapper = DeployWrapper(cfg, obs_dim=backend.n_joints * 2)

    print("\nRunning simulation...")
    raw_obs = backend.reset()
    obs = wrapper.reset(raw_obs)

    for step in range(100):
        action = np.zeros(12)
        raw_obs = backend.step(action)
        obs, processed_action = wrapper.step(raw_obs, action)

        if step % 20 == 0:
            joint_pos = raw_obs["joint_position"]
            base_pos = raw_obs["base_position"]
            print(f"  Step {step}: base_z={base_pos[2]:.3f}, joints={joint_pos[:3]}")

    backend.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
