import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "tools" / "convert_chocolate_motion.py"
MANIFEST = ROOT / "cpp" / "plugins" / "chocolate" / "joints.json"


def make_npz(path: Path, *, fps=50.0, with_torso=True):
    tracking = json.loads(MANIFEST.read_text())["orders"]["sim2sim_tracking"]
    names = list(reversed(tracking))
    bodies = ["torso_link"] if with_torso else ["pelvis"]
    joint_pos = np.arange(2 * len(names), dtype=np.float32).reshape(2, len(names))
    joint_vel = joint_pos + 100.0
    body_pos = np.zeros((2, len(bodies), 3), dtype=np.float32)
    body_quat = np.zeros((2, len(bodies), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0  # source archives use WXYZ
    np.savez(path, fps=np.asarray(fps, dtype=np.float32), joint_names=np.asarray(names),
             body_names=np.asarray(bodies), joint_pos=joint_pos, joint_vel=joint_vel,
             body_pos_w=body_pos, body_quat_w=body_quat)
    return names, joint_pos


def test_converter_reorders_tracking_joints_and_converts_quaternion(tmp_path):
    source = tmp_path / "motion.npz"
    output = tmp_path / "motion.rllchoc"
    names, source_positions = make_npz(source)
    result = subprocess.run([sys.executable, str(CONVERTER), str(source), str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    blob = output.read_bytes()
    magic, frames, joints, fps = struct.unpack_from("<8sIId", blob)
    assert (magic, frames, joints, fps) == (b"RLLCHOC1", 2, 23, 50.0)
    values = np.frombuffer(blob, dtype="<f4", offset=24).reshape(frames, 53)
    tracking = json.loads(MANIFEST.read_text())["orders"]["sim2sim_tracking"]
    expected = [source_positions[0, names.index(name)] for name in tracking]
    np.testing.assert_array_equal(values[0, :23], expected)
    np.testing.assert_array_equal(values[0, 46:49], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(values[0, 49:53], [0.0, 0.0, 0.0, 1.0])


def test_converter_rejects_wrong_rate_and_missing_torso(tmp_path):
    source = tmp_path / "motion.npz"
    output = tmp_path / "motion.rllchoc"
    make_npz(source, fps=60.0)
    result = subprocess.run([sys.executable, str(CONVERTER), str(source), str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "must match expected policy rate" in result.stderr
    make_npz(source, with_torso=False)
    result = subprocess.run([sys.executable, str(CONVERTER), str(source), str(output)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "torso_link" in result.stderr
