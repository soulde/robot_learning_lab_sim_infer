#!/usr/bin/env python3
"""Prepare a non-destructive Chocolate MJCF copy for generic sim2sim nodes.

The training profile normally adds its ground plane and resets to the trained
standing pose in Python. This writes those details plus the velocity/tracking
actuator effort profiles as ordinary MJCF so the XML-only sim hardware node can
reproduce that setup. Meshes stay external and the source XML is never modified.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import mujoco


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_xml", type=Path)
    parser.add_argument("output_xml", type=Path)
    parser.add_argument(
        "--training-root", type=Path, default=Path.home() / "chocolate_training",
        help="Chocolate training checkout containing source/chocolate_asset",
    )
    args = parser.parse_args()
    training_root = args.training_root.expanduser().resolve()
    sys.path.insert(0, str(training_root / "source"))
    from chocolate_asset.profile import (
        DEFAULT_JOINT_POS,
        ISAACLAB_EFFORT_LIMIT,
        INITIAL_BASE_HEIGHT,
        JOINT_NAMES,
        SIM2SIM_PRIMARY_EFFORT_LIMIT,
    )

    source = args.source_xml.expanduser().resolve()
    mesh_dir = (training_root / "source/chocolate_asset/description/meshes").resolve()
    xml = source.read_text(encoding="utf-8")
    if "chocolate_training_ground" in xml or "rll_stand" in xml or "rll_effort_limit_mode_" in xml:
        raise ValueError("source already contains generated Chocolate sim2sim elements")
    compiler = re.search(r"<compiler\b[^>]*/>", xml)
    if compiler is None:
        raise ValueError("MJCF must have a self-closing compiler element")
    tag = compiler.group(0)
    if "meshdir=" in tag:
        tag = re.sub(r'meshdir="[^"]*"', f'meshdir="{mesh_dir}"', tag)
    else:
        tag = tag[:-2] + f' meshdir="{mesh_dir}"/>'
    xml = xml[:compiler.start()] + tag + xml[compiler.end():]

    motor_pattern = re.compile(r"<motor\b(?P<attrs>[^>]*)/>")
    seen_motors: set[str] = set()

    def limit_motor(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        joint_match = re.search(r'\bjoint="([^"]+)"', attrs)
        if joint_match is None:
            raise ValueError("every Chocolate motor must name its joint")
        name = joint_match.group(1)
        if name not in SIM2SIM_PRIMARY_EFFORT_LIMIT:
            raise ValueError(f"no sim2sim torque limit configured for motor joint {name!r}")
        if name in seen_motors:
            raise ValueError(f"MJCF has multiple direct motor actuators for {name!r}")
        seen_motors.add(name)
        if re.search(r'\b(?:ctrlrange|ctrllimited)=', attrs):
            raise ValueError(f"source motor {name!r} already defines a control limit")
        gear_match = re.search(r'\bgear="([^"]+)"', attrs)
        gear = float(gear_match.group(1).split()[0]) if gear_match else 1.0
        if gear == 0.0:
            raise ValueError(f"source motor {name!r} has zero gear")
        limit = SIM2SIM_PRIMARY_EFFORT_LIMIT[name] / abs(gear)
        return f'{match.group(0)[:-2]} ctrllimited="true" ctrlrange="{-limit:.12g} {limit:.12g}"/>'

    xml = motor_pattern.sub(limit_motor, xml)
    if seen_motors != set(JOINT_NAMES):
        missing = sorted(set(JOINT_NAMES) - seen_motors)
        raise ValueError(f"Chocolate MJCF motor set does not match the joint manifest; missing={missing}")

    model = mujoco.MjModel.from_xml_string(xml)
    if set(JOINT_NAMES) != {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    }:
        raise ValueError("Chocolate MJCF actuated scalar-joint names do not match the source manifest")
    actuator_joint_names = [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(model.actuator_trnid[actuator_id, 0]),
        )
        for actuator_id in range(model.nu)
    ]
    if len(actuator_joint_names) != len(JOINT_NAMES) or set(actuator_joint_names) != set(JOINT_NAMES):
        raise ValueError("Chocolate effort profiles require one direct actuator per manifest joint")
    qpos = model.qpos0.copy()
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_id < 0 or model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Chocolate MJCF must have the root free joint")
    root_qpos = int(model.jnt_qposadr[root_id])
    qpos[root_qpos:root_qpos + 7] = [0.0, 0.0, INITIAL_BASE_HEIGHT + 0.025, 1.0, 0.0, 0.0, 0.0]
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[int(model.jnt_qposadr[joint_id])] = DEFAULT_JOINT_POS[name]
    qpos_text = " ".join(f"{float(value):.12g}" for value in qpos)

    worldbody = re.search(r"<worldbody\s*>", xml)
    if worldbody is None:
        raise ValueError("MJCF is missing worldbody")
    ground = '\n    <geom name="chocolate_training_ground" type="plane" size="30 30 0.1" pos="0 0 0" contype="1" conaffinity="2"/>'
    xml = xml[:worldbody.end()] + ground + xml[worldbody.end():]
    closing = xml.rfind("</mujoco>")
    if closing < 0:
        raise ValueError("MJCF is missing </mujoco>")
    keyframe = f'\n  <keyframe><key name="rll_stand" qpos="{qpos_text}"/></keyframe>\n'
    velocity_effort = " ".join(
        f"{SIM2SIM_PRIMARY_EFFORT_LIMIT[name]:.12g}" for name in actuator_joint_names
    )
    tracking_effort = " ".join(
        f"{ISAACLAB_EFFORT_LIMIT[name]:.12g}" for name in actuator_joint_names
    )
    custom = (
        '\n  <custom>\n'
        f'    <numeric name="rll_effort_limit_mode_1" data="{velocity_effort}"/>\n'
        f'    <numeric name="rll_effort_limit_mode_2" data="{tracking_effort}"/>\n'
        '  </custom>\n'
    )
    xml = xml[:closing] + custom + keyframe + xml[closing:]
    # Validate the generated MJCF and keyframe before writing it.
    generated = mujoco.MjModel.from_xml_string(xml)
    if generated.nkey != 1 or generated.nq != model.nq or generated.nnumeric != 2:
        raise ValueError("generated MJCF keyframe/effort profiles did not preserve the model")
    args.output_xml.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_xml.expanduser().resolve().write_text(xml, encoding="utf-8")
    print(f"prepared XML with ground plane and trained standing keyframe: {args.output_xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
