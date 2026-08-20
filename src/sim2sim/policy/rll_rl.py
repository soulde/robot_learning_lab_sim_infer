"""rll_rl checkpoint policy loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from sim2sim.policy.base import PolicyLoader


class RllRlPolicyLoader(PolicyLoader):
    """Load and run policies from rll_rl checkpoints.

    Supports PPO, AMP, FastSAC, and TDMPC2 runners.
    """

    def __init__(self) -> None:
        self._runner = None
        self._inference_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def load(self, checkpoint_path: str | Path, **kwargs: Any) -> None:
        """Load an rll_rl checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            **kwargs: Options:
                - runner_class: Runner class name ('PPORunner', 'AMPRunner', etc.).
                  If not provided, auto-detects from checkpoint.
                - runner_cfg: Runner configuration object.
                - env_wrapper: Environment wrapper instance (optional, for initialization).
                - device: Device for inference ('cpu', 'cuda', etc.).
        """
        import torch

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=kwargs.get("device", "cpu"), weights_only=False
        )

        runner_class_name = kwargs.get("runner_class")
        if runner_class_name is None:
            runner_class_name = self._detect_runner_class(checkpoint)

        runner_cls = self._get_runner_class(run_class_name)

        runner_cfg = kwargs.get("runner_cfg")
        env_wrapper = kwargs.get("env_wrapper")

        if runner_cfg is not None and env_wrapper is not None:
            self._runner = runner_cls(env_wrapper, runner_cfg, log_dir=".")
        else:
            self._runner = self._create_runner_from_checkpoint(
                runner_cls, checkpoint, kwargs
            )

        self._runner.policy.load_state_dict(checkpoint["policy"])
        self._runner.policy.eval()

        device = kwargs.get("device", "cpu")
        self._runner.policy.to(device)
        self._device = device

    def _detect_runner_class(self, checkpoint: dict) -> str:
        """Auto-detect runner class from checkpoint contents."""
        cfg = checkpoint.get("cfg", {})
        algorithm_cfg = cfg.get("algorithm", cfg.get("runner", {}).get("algorithm", {}))
        class_name = algorithm_cfg.get("class_name", "")

        if "AMP" in class_name.upper():
            return "AMPRunner"
        elif "FastSAC" in class_name:
            return "FastSACRunner"
        elif "TDMPC2" in class_name:
            return "TDMPC2Runner"
        else:
            return "PPORunner"

    def _get_runner_class(self, class_name: str) -> type:
        """Get runner class by name."""
        try:
            from rll_rl.ppo import PPORunner
            from rll_rl.amp import AMPRunner
            from rll_rl.fastsac import FastSACRunner
            from rll_rl.tdmpc2 import TDMPC2Runner
        except ImportError:
            raise ImportError(
                "rll_rl is required for policy loading. "
                "Install it with: pip install rll_rl"
            )

        runners = {
            "PPORunner": PPORunner,
            "AMPRunner": AMPRunner,
            "FastSACRunner": FastSACRunner,
            "TDMPC2Runner": TDMPC2Runner,
        }

        if class_name not in runners:
            raise ValueError(f"Unknown runner class: {class_name}")

        return runners[class_name]

    def _create_runner_from_checkpoint(
        self, runner_cls: type, checkpoint: dict, kwargs: dict
    ) -> Any:
        """Create a runner instance from checkpoint config."""
        cfg = checkpoint.get("cfg", {})

        device = kwargs.get("device", "cpu")

        class MockEnv:
            def __init__(self, obs_dim: int, action_dim: int, device: str):
                self.num_envs = 1
                self.obs_dim = obs_dim
                self.action_dim = action_dim
                self.device = device
                self.unwrapped = self

            def reset(self):
                return torch.zeros(1, self.obs_dim, device=self.device)

            def step(self, action):
                return torch.zeros(1, self.obs_dim, device=self.device), \
                       torch.zeros(1, device=self.device), \
                       torch.zeros(1, dtype=torch.bool, device=self.device), \
                       {}

        import torch

        policy_state = checkpoint.get("policy", {})
        obs_dim = 0
        action_dim = 0
        for key, value in policy_state.items():
            if "actor" in key and "weight" in key:
                if len(value.shape) == 2:
                    action_dim = value.shape[0]
            if "critic" in key and "weight" in key:
                if len(value.shape) == 2:
                    obs_dim = value.shape[1]

        mock_env = MockEnv(
            obs_dim=max(obs_dim, 1),
            action_dim=max(action_dim, 1),
            device=device,
        )

        try:
            runner = runner_cls(mock_env, cfg, log_dir=".")
        except Exception:
            class MinimalRunner:
                def __init__(self, policy, device):
                    self.policy = policy
                    self.device = device

            policy = self._create_policy_from_state_dict(policy_state, device)
            return MinimalRunner(policy, device)

    def _create_policy_from_state_dict(self, state_dict: dict, device: str) -> Any:
        """Create a minimal policy object from state dict."""
        import torch

        class MinimalPolicy(torch.nn.Module):
            def __init__(self, state_dict: dict):
                super().__init__()
                self._state_dict_keys = list(state_dict.keys())
                self._load_from_state_dict(state_dict, True, False)

            def forward(self, x):
                return x

            def act_inference(self, obs):
                return obs

        return MinimalPolicy(state_dict).to(device)

    def get_inference_policy(self, device: str | None = None) -> Callable[[np.ndarray], np.ndarray]:
        """Return the inference function.

        Args:
            device: Device for inference.

        Returns:
            Callable mapping observation to action.
        """
        import torch

        if self._runner is None:
            raise RuntimeError("No policy loaded. Call load() first.")

        target_device = torch.device(device or self._device)

        def policy_fn(obs: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=target_device
                ).unsqueeze(0)

                if hasattr(self._runner, "get_inference_policy"):
                    if self._inference_fn is None:
                        self._inference_fn = self._runner.get_inference_policy(
                            device=str(target_device)
                        )
                    action = self._inference_fn(obs_tensor)
                else:
                    action = self._runner.policy.act_inference(obs_tensor)

                return action.squeeze(0).cpu().numpy()

        return policy_fn

    def close(self) -> None:
        """Clean up resources."""
        self._runner = None
        self._inference_fn = None
