#!/usr/bin/env python3
"""Exercise Python sim hardware ↔ native LibTorch/Cyclone policy over DDS."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=root / "build-cpp" / "rll-policy")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--domain-id", type=int, default=87)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"native policy binary not found: {binary}")
    plugin = root / "build-cpp" / "libtest_robot_plugin.so"
    checkpoint = root / "build-cpp" / "fixtures" / "velocity.pt"
    if not plugin.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("build the test_robot_plugin and inference_fixtures targets first")
    xml = root / "src" / "robot_learning_lab_sim_infer" / "assets" / "dr02_motor" / "dr02_torque.xml"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    mode_damping, mode_walk = "select_damping", "select_walk"
    config_text = "\n".join([
        "dds:", f"  domain_id: {args.domain_id}", "  state_topic: robot/state",
        "  rc_topic: robot/rc_command", "  motor_command_topic: robot/motor_command",
        "  sensor_reliability: best_effort", "  command_reliability: reliable", "timing:",
        "  policy_hz: 50.0", "  state_timeout_s: 0.5", "  rc_timeout_s: 0.5",
        "  transition_duration_s: 0.1", "device: cpu", "plugin:",
        f"  path: {json.dumps(str(plugin))}", "  config: test-plugin.yaml", "damping:", "  kd: 0.5",
        "policies:", "  - index: 1", "    name: velocity",
        f"    type: velocity", f"    checkpoint: {json.dumps(str(checkpoint))}", "    observation_dim: 2", "    action_dim: 2",
        "state_machine:", "  initial_state: damping", "  states:", "    damping:", "      policy: null",
        f"      transitions: {{ {mode_damping}: damping, {mode_walk}: velocity }}", "    velocity:",
        "      policy: velocity", f"      transitions: {{ {mode_damping}: damping, {mode_walk}: velocity }}",
        f"  mode_events: {{ 0: {mode_damping}, 1: {mode_walk} }}", "  button_events: {}",
    ]) + "\n"
    helper = root / "cpp" / "tests" / "fixtures" / "run_headless_sim.py"
    probe = root / "cpp" / "tests" / "fixtures" / "probe_motor_command.py"
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    with tempfile.TemporaryDirectory(prefix="rll-native-interop-") as scratch:
        config_path = Path(scratch) / "policy.yaml"
        config_path.write_text(config_text, encoding="utf-8")
        try:
            probe_proc = subprocess.Popen([str(args.python), str(probe), "--domain-id", str(args.domain_id)], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            processes.append(("probe", probe_proc))
            sim_proc = subprocess.Popen([str(args.python), str(helper), "--domain-id", str(args.domain_id), "--xml", str(xml)],
                                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            processes.append(("sim", sim_proc))
            time.sleep(1.0)
            policy_proc = subprocess.Popen([str(binary), "--config", str(config_path)], env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            processes.append(("policy", policy_proc))
            try:
                output, _ = probe_proc.communicate(timeout=20.0)
            except subprocess.TimeoutExpired:
                details = []
                for name, proc in processes:
                    proc.terminate()
                    try:
                        details.append(f"{name}: {proc.communicate(timeout=3)[0]}")
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        details.append(f"{name}: {proc.communicate()[0]}")
                raise RuntimeError("native sim interop timed out\n" + "\n".join(details))
            print(output, end="")
            if probe_proc.returncode != 0 or "PROBE_RECEIVED" not in output:
                details = [f"probe: {output}"]
                for name, proc in processes[1:]:
                    if proc.poll() is None:
                        proc.terminate()
                    try:
                        details.append(f"{name}: {proc.communicate(timeout=3)[0]}")
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        details.append(f"{name}: {proc.communicate()[0]}")
                raise RuntimeError("motor command probe failed\n" + "\n".join(details))
            policy_proc.terminate()
            policy_output, _ = policy_proc.communicate(timeout=3)
            if policy_proc.returncode not in (0, -15):
                raise RuntimeError(f"native policy process failed\n{policy_output}")
            print(policy_output, end="")
            return 0
        finally:
            for _, proc in reversed(processes):
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
