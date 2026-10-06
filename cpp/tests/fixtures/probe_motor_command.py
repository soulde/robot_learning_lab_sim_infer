#!/usr/bin/env python3
"""Fail unless a nonzero named MotorCommand arrives on the configured DDS domain."""

from __future__ import annotations

import argparse
import time

from cyclonedds.core import Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.qos import Policy
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from robot_learning_lab_sim_infer.dds.types import MotorCommand


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    participant = DomainParticipant(domain_id=args.domain_id)
    topic = Topic(participant, "robot/motor_command", MotorCommand)
    reader = DataReader(participant, topic, qos=Qos(
        Policy.History.KeepLast(1), Policy.Reliability.Reliable(max_blocking_time=duration(milliseconds=100))))
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        sample = reader.take_next()
        if sample is None:
            time.sleep(0.01)
            continue
        if sample.joint_names and sample.position and any(abs(value) > 1.0e-6 for value in sample.position):
            print(f"PROBE_RECEIVED joints={sample.joint_names} position={sample.position}", flush=True)
            return 0
    raise TimeoutError("no nonzero named MotorCommand arrived before timeout")


if __name__ == "__main__":
    raise SystemExit(main())
