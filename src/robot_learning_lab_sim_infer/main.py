"""Command line entry point for external-profile sim2sim."""

from __future__ import annotations

import argparse

from .profile import load_profile
from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic TorchScript MuJoCo sim2sim runner")
    parser.add_argument("--profile", required=True, help="Python file defining create_profile()")
    parser.add_argument("--checkpoint", required=True, help="TorchScript actor policy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="Policy steps; 0 runs until viewer closes")
    parser.add_argument("--render-hz", type=float, default=25.0)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    print(f"profile={profile.name} obs_dim={profile.obs_dim} action_dim={profile.action_dim}")
    run(profile, args.checkpoint, headless=args.headless, steps=args.steps, render_hz=args.render_hz)


if __name__ == "__main__":
    main()
