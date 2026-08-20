"""MuJoCo simulator backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_learning_lab_sim_infer.backends.base import SimulatorBackend
from robot_learning_lab_sim_infer.config import DeployConfig
from robot_learning_lab_sim_infer.controllers.base import MotorController
from robot_learning_lab_sim_infer.controllers.pd import PDController
from robot_learning_lab_sim_infer.controllers.mit import MITCheetahController


_CONTROLLER_REGISTRY: dict[str, type[MotorController]] = {
    "pd": PDController,
    "mit_cheetah": MITCheetahController,
}


def register_controller(name: str, cls: type[MotorController]) -> None:
    """Register a custom motor controller."""
    _CONTROLLER_REGISTRY[name] = cls


class MuJoCoBackend(SimulatorBackend):
    """MuJoCo-based simulator backend for policy validation.

    Defaults to torque control interface. Supports PD and MIT Cheetah controllers.
    """

    def __init__(self, render_mode: str | None = None):
        """Initialize the MuJoCo backend.

        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None for headless).
        """
        self.render_mode = render_mode
        self._model = None
        self._data = None
        self._renderer = None
        self._cfg: DeployConfig | None = None
        self._joint_ids: np.ndarray | None = None
        self._action_type: str = "joint_torque"
        self._default_qpos: np.ndarray | None = None
        self._controller: MotorController | None = None

    def load(
        self,
        model_path: str,
        cfg: DeployConfig,
        **kwargs: Any,
    ) -> None:
        """Load a MuJoCo model.

        Args:
            model_path: Path to MJCF XML file.
            cfg: Deploy configuration.
            **kwargs: Additional options:
                - gravity: Gravity vector override.
                - controller: Controller type ('pd', 'mit_cheetah', or MotorController instance).
                - controller_kwargs: Dict of kwargs passed to controller constructor.
        """
        import mujoco

        self._cfg = cfg
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {model_path}")

        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)

        gravity = kwargs.get("gravity")
        if gravity is not None:
            self._model.opt.gravity[:] = gravity

        joint_names = []
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is not None:
                joint_names.append(name)

        self._joint_names = joint_names

        self._joint_ids = np.array([
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
            if mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        ], dtype=np.int32)

        self._default_qpos = self._data.qpos[7:].copy()

        self._action_type = self._determine_actuator_type(cfg)

        controller_spec = kwargs.get("controller", "pd")
        controller_kwargs = kwargs.get("controller_kwargs", {})
        self._controller = self._build_controller(
            controller_spec, cfg, controller_kwargs
        )

        if self.render_mode == "human":
            self._renderer = mujoco.Renderer(self._model)

    def _determine_actuator_type(self, cfg: DeployConfig) -> str:
        """Determine the primary action type from config.

        Defaults to joint_torque for sim2real torque interface.
        """
        for group in cfg.actions.concat.values():
            for key in group.keys:
                if key in cfg.actions.terms:
                    term = cfg.actions.terms[key]
                    if term.type in ("joint_position", "joint_torque"):
                        return term.type
        return "joint_torque"

    def _build_controller(
        self,
        spec: str | MotorController,
        cfg: DeployConfig,
        extra_kwargs: dict,
    ) -> MotorController:
        """Build a motor controller from specification."""
        if isinstance(spec, MotorController):
            return spec

        if spec not in _CONTROLLER_REGISTRY:
            available = ", ".join(sorted(_CONTROLLER_REGISTRY.keys()))
            raise ValueError(f"Unknown controller '{spec}'. Available: {available}")

        cls = _CONTROLLER_REGISTRY[spec]

        n_joints = len(self._joint_ids)
        kwargs = dict(extra_kwargs)

        if "kp" not in kwargs and cfg.actuator.stiffness:
            kwargs["kp"] = np.array(cfg.actuator.stiffness[:n_joints])
        if "kd" not in kwargs and cfg.actuator.damping:
            kwargs["kd"] = np.array(cfg.actuator.damping[:n_joints])

        if spec == "mit_cheetah" and "default_joint_pos" not in kwargs:
            if self._default_qpos is not None:
                kwargs["default_joint_pos"] = self._default_qpos[:n_joints]

        return cls(**kwargs)

    def reset(self) -> dict[str, np.ndarray]:
        """Reset MuJoCo simulation to initial state.

        Returns:
            Dictionary of raw observation values.
        """
        import mujoco

        mujoco.mj_resetData(self._model, self._data)

        if self._default_qpos is not None:
            self._data.qpos[7:7 + len(self._default_qpos)] = self._default_qpos

        mujoco.mj_forward(self._model, self._data)

        if self._controller is not None:
            self._controller.reset()

        return self.get_observation()

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Execute one MuJoCo simulation step.

        Args:
            action: Action vector.
                - For 'joint_torque' mode: direct torque commands.
                - For 'joint_position' mode: position targets (PD controlled internally).

        Returns:
            Dictionary of raw observation values.
        """
        import mujoco

        n_joints = len(self._joint_ids)
        action = np.asarray(action, dtype=np.float64)

        if self._action_type == "joint_torque":
            if self._controller is not None:
                obs = self.get_observation()
                torque = self._controller.compute(
                    target=action,
                    q_pos=obs["joint_position"],
                    q_vel=obs["joint_velocity"],
                    dt=self._model.opt.timestep,
                    gravity_torque=self._compute_gravity_torque(),
                )
                self._data.ctrl[:n_joints] = torque
            else:
                self._data.ctrl[:n_joints] = action[:n_joints]

        elif self._action_type == "joint_position":
            target = action[:n_joints]
            obs = self.get_observation()
            torque = self._controller.compute(
                target=target,
                q_pos=obs["joint_position"],
                q_vel=obs["joint_velocity"],
                dt=self._model.opt.timestep,
                gravity_torque=self._compute_gravity_torque(),
            )
            self._data.ctrl[:n_joints] = torque

        else:
            raise ValueError(f"Unsupported action type: {self._action_type}")

        mujoco.mj_step(self._model, self._data)

        if self.render_mode == "human" and self._renderer is not None:
            self._renderer.update_scene(self._data)
            self._renderer.render()

        return self.get_observation()

    def _compute_gravity_torque(self) -> np.ndarray | None:
        """Compute gravity compensation torques using MuJoCo's built-in computation."""
        import mujoco

        n_joints = len(self._joint_ids)
        qfrc_gravity = np.zeros(self._model.nv)
        mujoco.mj_rne(self._model, self._data, 1, qfrc_gravity)
        return qfrc_gravity[7:7 + n_joints].copy()

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observations from MuJoCo.

        Returns:
            Dictionary mapping observation term names to raw values.
        """
        n_joints = len(self._joint_ids)
        obs = {}

        obs["joint_position"] = self._data.qpos[7:7 + n_joints].copy()
        obs["joint_velocity"] = self._data.qvel[7:7 + n_joints].copy()

        obs["base_position"] = self._data.qpos[:3].copy()
        obs["base_quaternion"] = self._data.qpos[3:7].copy()

        obs["base_linear_velocity"] = self._data.qvel[:3].copy()
        obs["base_angular_velocity"] = self._data.qvel[3:6].copy()

        obs["projected_gravity"] = self._projected_gravity()

        obs["joint_torque"] = self._data.qfrc_actuator[7:7 + n_joints].copy()

        obs["contact_forces"] = self._get_contact_forces()

        return obs

    def _projected_gravity(self) -> np.ndarray:
        """Compute projected gravity in body frame."""
        import mujoco

        gravity_world = np.array([0.0, 0.0, -9.81])
        body_quat = self._data.qpos[3:7]

        body_rot = np.zeros(9)
        mujoco.mju_quat2Mat(body_rot, body_quat)

        body_rot_inv = np.zeros(9)
        mujoco.mju_mat2Transpose(body_rot_inv, body_rot)

        projected = np.zeros(3)
        mujoco.mju_rotVecMat(projected, gravity_world, body_rot_inv)
        return projected

    def _get_contact_forces(self) -> np.ndarray:
        """Get contact forces on the robot body."""
        n_contacts = self._data.ncon
        forces = np.zeros(6)

        for i in range(min(n_contacts, 10)):
            contact = self._data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self._model, self._data, i, force)
            forces += force[:6]

        return forces

    def close(self) -> None:
        """Clean up MuJoCo resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._model = None
        self._data = None

    @property
    def n_joints(self) -> int:
        """Number of actuated joints."""
        return len(self._joint_ids) if self._joint_ids is not None else 0

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of actuated joint names."""
        return getattr(self, "_joint_names", [])

    @property
    def dt(self) -> float:
        """Simulation time step."""
        if self._model is not None:
            return self._model.opt.timestep
        return 0.002

    @property
    def controller(self) -> MotorController | None:
        """Current motor controller."""
        return self._controller

    def set_controller(self, controller: MotorController) -> None:
        """Set a custom motor controller.

        Args:
            controller: Motor controller instance.
        """
        self._controller = controller
