#!/usr/bin/env python3
"""Convert a Chocolate tracking NPZ into the versioned runtime binary format.

The converter keeps deployment independent of NumPy/NPZ decoding. The source
archive remains external; the generated file contains only the 50 Hz target
joint and torso trajectories used by the native Chocolate plugin.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"RLLCHOC1"
EXPECTED_FPS = 50.0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source Chocolate/LAFAN NPZ")
    parser.add_argument("output", type=Path, help="output .rllchoc file")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "cpp/plugins/chocolate/joints.json",
        help="joint order manifest (defaults to the checked in Chocolate source manifest)",
    )
    parser.add_argument("--expected-fps", type=float, default=EXPECTED_FPS)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as stream:
        tracking_names = json.load(stream)["orders"]["sim2sim_tracking"]
    with np.load(args.input, allow_pickle=False) as archive:
        fps = float(archive["fps"])
        names = [str(item) for item in archive["joint_names"].tolist()]
        bodies = [str(item) for item in archive["body_names"].tolist()]
        joint_pos = np.asarray(archive["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(archive["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(archive["body_pos_w"], dtype=np.float32)
        body_quat_wxyz = np.asarray(archive["body_quat_w"], dtype=np.float32)

    if not np.isfinite(fps) or not np.isclose(fps, args.expected_fps, atol=1e-6):
        raise ValueError(f"motion fps {fps:g} must match expected policy rate {args.expected_fps:g} Hz")
    if len(set(names)) != len(names) or set(names) != set(tracking_names):
        raise ValueError("NPZ joint_names do not exactly match the Chocolate tracking joint manifest")
    if "torso_link" not in bodies:
        raise ValueError("NPZ body_names is missing the torso_link tracking anchor")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(names) or joint_vel.shape != joint_pos.shape:
        raise ValueError("NPZ joint_pos/joint_vel must have matching [frames, 23] shapes")
    if body_pos.shape != (joint_pos.shape[0], len(bodies), 3) or body_quat_wxyz.shape != (joint_pos.shape[0], len(bodies), 4):
        raise ValueError("NPZ torso body position/quaternion shapes do not match frame/body metadata")

    order = [names.index(name) for name in tracking_names]
    torso = bodies.index("torso_link")
    position = np.asarray(joint_pos[:, order], dtype="<f4")
    velocity = np.asarray(joint_vel[:, order], dtype="<f4")
    torso_position = np.asarray(body_pos[:, torso], dtype="<f4")
    # Isaac-style source quaternion is wxyz; the native contract stores xyzw.
    torso_quaternion_xyzw = np.asarray(body_quat_wxyz[:, torso, [1, 2, 3, 0]], dtype="<f4")
    payload = np.concatenate((position, velocity, torso_position, torso_quaternion_xyzw), axis=1)
    if not np.isfinite(payload).all():
        raise ValueError("motion payload contains NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(struct.pack("<8sIId", MAGIC, joint_pos.shape[0], len(tracking_names), fps))
        payload.tofile(stream)
    print(f"wrote {joint_pos.shape[0]} frames at {fps:g} Hz to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
