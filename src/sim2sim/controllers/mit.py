"""MIT Cheetah-style controller with feedforward and friction compensation."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim2sim.controllers.base import MotorController


class MITCheetahController(MotorController):
    """MIT Cheetah-style torque controller.

    Implements the control strategy from MIT Cheetah with:
    - PD position control
    - Velocity feedforward
    - Gravity compensation
    - Friction compensation
    - Optional virtual forces (e.g., virtual spring-damper for balance)

    Reference:
        Di Carlo, J., et al. "Dynamic Locomotion in the MIT Cheetah 3 Through
        Convex Model-Based Optimization." IROS 2018.
    """

    def __init__(
        self,
        kp: np.ndarray | list[float] | float = 40.0,
        kd: np.ndarray | list[float] | float = 1.0,
        ff_weight: float = 0.0,
        friction_coeff: float = 0.0,
        torque_limit: np.ndarray | list[float] | None = None,
        gravity_compensation: bool = True,
        virtual_spring_k: np.ndarray | list[float] | float | None = None,
        virtual_damper_d: np.ndarray | list[float] | float | None = None,
        default_joint_pos: np.ndarray | list[float] | None = None,
    ):
        """Initialize MIT Cheetah controller.

        Args:
            kp: Position gain (proportional).
            kd: Velocity gain (derivative).
            ff_weight: Feedforward torque weight (0 = no feedforward).
            friction_coeff: Coulomb friction coefficient per joint.
            torque_limit: Maximum torque per joint.
            gravity_compensation: Whether to include gravity compensation.
            virtual_spring_k: Virtual spring stiffness for balance (relative to default pos).
            virtual_damper_d: Virtual damper coefficient for balance.
            default_joint_pos: Default joint positions for virtual spring target.
        """
        self._kp = np.asarray(kp, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._ff_weight = ff_weight
        self._friction_coeff = friction_coeff
        self._torque_limit = (
            np.asarray(torque_limit, dtype=np.float64) if torque_limit is not None else None
        )
        self._gravity_compensation = gravity_compensation
        self._virtual_spring_k = (
            np.asarray(virtual_spring_k, dtype=np.float64) if virtual_spring_k is not None else None
        )
        self._virtual_damper_d = (
            np.asarray(virtual_damper_d, dtype=np.float64) if virtual_damper_d is not None else None
        )
        self._default_joint_pos = (
            np.asarray(default_joint_pos, dtype=np.float64) if default_joint_pos is not None else None
        )
        self._prev_error: np.ndarray | None = None

    def reset(self) -> None:
        """Reset controller state."""
        self._prev_error = None

    def compute(
        self,
        target: np.ndarray,
        q_pos: np.ndarray,
        q_vel: np.ndarray,
        dt: float,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute MIT Cheetah-style torques.

        Args:
            target: Desired joint positions.
            q_pos: Current joint positions.
            q_vel: Current joint velocities.
            dt: Time step.
            **kwargs: Additional inputs:
                - gravity_torque: Pre-computed gravity compensation torques.
                - friction_vel: Velocity for friction model.
                - contact_forces: External contact forces.

        Returns:
            Joint torques.
        """
        target = np.asarray(target, dtype=np.float64)
        q_pos = np.asarray(q_pos, dtype=np.float64)
        q_vel = np.asarray(q_vel, dtype=np.float64)

        n_joints = len(q_pos)

        kp = self._kp
        kd = self._kd
        if kp.size == 1:
            kp = np.full(n_joints, kp.item())
        if kd.size == 1:
            kd = np.full(n_joints, kd.item())

        position_error = target - q_pos

        torque = kp * position_error - kd * q_vel

        if self._ff_weight > 0 and self._prev_error is not None and dt > 0:
            error_dot = (position_error - self._prev_error) / dt
            torque += self._ff_weight * error_dot

        self._prev_error = position_error.copy()

        if self._gravity_compensation:
            gravity_torque = kwargs.get("gravity_torque")
            if gravity_torque is not None:
                torque += np.asarray(gravity_torque, dtype=np.float64)

        if self._friction_coeff > 0:
            friction = self._compute_friction(q_vel, n_joints)
            torque -= friction

        if self._virtual_spring_k is not None and self._default_joint_pos is not None:
            virtual_torque = self._compute_virtual_model(q_pos, q_vel, n_joints)
            torque += virtual_torque

        if self._torque_limit is not None:
            torque = np.clip(torque, -self._torque_limit, self._torque_limit)

        return torque.astype(np.float32)

    def _compute_friction(self, q_vel: np.ndarray, n_joints: int) -> np.ndarray:
        """Compute Coulomb + viscous friction.

        Args:
            q_vel: Joint velocities.
            n_joints: Number of joints.

        Returns:
            Friction torques.
        """
        coeff = self._friction_coeff
        if isinstance(coeff, (int, float)):
            coeff_arr = np.full(n_joints, coeff)
        else:
            coeff_arr = np.asarray(coeff, dtype=np.float64)

        viscous = 0.05 * q_vel

        coulomb = coeff_arr * np.sign(q_vel)

        return coulomb + viscous

    def _compute_virtual_model(
        self, q_pos: np.ndarray, q_vel: np.ndarray, n_joints: int
    ) -> np.ndarray:
        """Compute virtual spring-damper forces.

        Args:
            q_pos: Current joint positions.
            q_vel: Current joint velocities.
            n_joints: Number of joints.

        Returns:
            Virtual torques.
        """
        k = self._virtual_spring_k
        d = self._virtual_damper_d

        if k.size == 1:
            k = np.full(n_joints, k.item())
        if d is not None and d.size == 1:
            d = np.full(n_joints, d.item())

        pos_error = self._default_joint_pos[:n_joints] - q_pos
        torque = k[:n_joints] * pos_error

        if d is not None:
            torque -= d[:n_joints] * q_vel

        return torque

    @property
    def name(self) -> str:
        """Controller name."""
        return "mit_cheetah"

    @property
    def kp(self) -> np.ndarray:
        """Position gain."""
        return self._kp.copy()

    @property
    def kd(self) -> np.ndarray:
        """Velocity gain."""
        return self._kd.copy()

    def set_default_joint_pos(self, pos: np.ndarray) -> None:
        """Set default joint positions for virtual spring target.

        Args:
            pos: Default joint positions.
        """
        self._default_joint_pos = np.asarray(pos, dtype=np.float64)
