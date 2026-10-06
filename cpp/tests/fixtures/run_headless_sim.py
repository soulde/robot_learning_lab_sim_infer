#!/usr/bin/env python3
"""Run the actual sim hardware node with enabled test RC input."""

from __future__ import annotations

import argparse
import threading
import time

from robot_learning_lab_sim_infer.configs import DDSConfig, RuntimeConfig
from robot_learning_lab_sim_infer.dds.transport import SimHardwareDDS
from robot_learning_lab_sim_infer.nodes.sim_hardware_node import SimHardwareNode
from robot_learning_lab_sim_infer.profiles.xml_hardware import XmlHardwareProfile
from robot_learning_lab_sim_infer.runtime.rc import RCValues


class NullRenderer:
    def update(self, model, data) -> None:
        del model, data

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--switch-after-s", type=float, default=0.0)
    parser.add_argument("--second-mode", type=int, default=2)
    args = parser.parse_args()
    runtime = RuntimeConfig()
    runtime.dds = DDSConfig(domain_id=args.domain_id, publish_hz=100.0)
    node = SimHardwareNode(XmlHardwareProfile(args.xml), runtime, SimHardwareDDS(runtime.dds), NullRenderer())
    values = RCValues(enabled=True, mode=args.mode, vx=0.2)
    node.set_rc_values(values)
    if args.switch_after_s > 0:
        def switch_mode():
            time.sleep(args.switch_after_s)
            node.set_rc_values(RCValues(enabled=True, mode=args.second_mode, vx=0.2))
        threading.Thread(target=switch_mode, name="test-rc-mode-switch", daemon=True).start()
    node.run()


if __name__ == "__main__":
    main()
