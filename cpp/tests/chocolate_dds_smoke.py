#!/usr/bin/env python3
"""Run both real Chocolate actors against the Chocolate MJCF sim over DDS."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from robot_learning_lab_sim_infer.configs import DDSConfig
from robot_learning_lab_sim_infer.dds.transport import SimHardwareDDS


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=root / "build-cpp/rll-policy")
    parser.add_argument("--plugin", type=Path, default=root / "build-cpp/libchocolate_robot_plugin.so")
    parser.add_argument("--velocity-checkpoint", type=Path, required=True)
    parser.add_argument("--tracking-checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True, help="Chocolate MJCF used by rll-sim-hardware")
    parser.add_argument("--domain-id", type=int, default=91)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--switch-after-s", type=float, default=6.0)
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    dds = SimHardwareDDS(DDSConfig(domain_id=args.domain_id))
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    with tempfile.TemporaryDirectory(prefix="rll-chocolate-smoke-") as scratch:
        temp = Path(scratch)
        plugin_yaml = temp / "plugin.yaml"
        plugin_yaml.write_text(f"motion_file: {args.motion.resolve()}\nexpected_fps: 50.0\n", encoding="utf-8")
        config = temp / "policy.yaml"
        config.write_text(f"""dds:
  domain_id: {args.domain_id}
  cyclonedds_uri: null
  state_topic: robot/state
  rc_topic: robot/rc_command
  motor_command_topic: robot/motor_command
  sensor_reliability: best_effort
  command_reliability: reliable
timing:
  policy_hz: 50.0
  state_timeout_s: 0.5
  rc_timeout_s: 0.5
  transition_duration_s: 0.2
device: cpu
plugin:
  path: {args.plugin.resolve()}
  config: {plugin_yaml}
damping:
  kd: 0.5
policies:
  - index: 1
    name: velocity
    checkpoint: {args.velocity_checkpoint.resolve()}
    observation_dim: 78
    action_dim: 23
  - index: 2
    name: tracking
    checkpoint: {args.tracking_checkpoint.resolve()}
    observation_dim: 124
    action_dim: 23
state_machine:
  initial_state: damping
  states:
    damping:
      policy: null
      transitions: {{select_damping: damping, select_velocity: velocity, select_tracking: tracking}}
    velocity:
      policy: velocity
      transitions: {{select_damping: damping, select_velocity: velocity, select_tracking: tracking}}
    tracking:
      policy: tracking
      transitions: {{select_damping: damping, select_velocity: velocity, select_tracking: tracking}}
  mode_events: {{0: select_damping, 1: select_velocity, 2: select_tracking}}
  button_events: {{}}
""", encoding="utf-8")
        helper = root / "cpp/tests/fixtures/run_headless_sim.py"
        try:
            sim = subprocess.Popen([
                str(Path(sys.executable)), str(helper), "--domain-id", str(args.domain_id),
                "--xml", str(args.xml.resolve()), "--mode", "1", "--switch-after-s", str(args.switch_after_s),
                "--second-mode", "2",
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            processes.append(("sim", sim))
            time.sleep(1.0)
            policy = subprocess.Popen([str(args.binary.resolve()), "--config", str(config)], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            processes.append(("policy", policy))
            started = time.monotonic()
            phase = 1
            phase2_at = args.switch_after_s - 1.0
            seen = {1: 0, 2: 0}
            while time.monotonic() - started < args.timeout:
                elapsed = time.monotonic() - started
                if elapsed >= phase2_at:
                    phase = 2
                while True:
                    command = dds.take_motor_command()
                    if command is None:
                        break
                    if len(command.joint_names) == 23 and len(command.position) == 23 and any(command.kp):
                        seen[phase] += 1
                for name, process in processes:
                    if process.poll() is not None:
                        output = process.stdout.read() if process.stdout else ""
                        raise RuntimeError(f"Chocolate {name} node exited {process.returncode}\n{output}")
                if elapsed > args.switch_after_s + 2.5 and seen[1] > 10 and seen[2] > 10:
                    break
                time.sleep(0.01)
            else:
                raise TimeoutError(f"Chocolate sim/DDS smoke timed out; commands={seen}")
            if seen[1] <= 10 or seen[2] <= 10:
                raise RuntimeError(f"did not receive active commands from both policy modes: {seen}")
            print(f"CHOCOLATE_DDS_SMOKE velocity_commands={seen[1]} tracking_commands={seen[2]}")
        finally:
            for _, process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            dds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
