"""Scene injection utilities for MuJoCo."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

StrOrPath = Union[str, Path]


SCENE_TEMPLATE = """\
<mujoco model="scene">
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
  <option timestep="{timestep}" gravity="{gravity}" integrator="implicitfast"/>
  <default>
    <joint armature="0.01" damping="0.1"/>
    <geom condim="3" margin="0.001"/>
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.7 0.8" rgb2="0.2 0.2 0.3" width="512" height="3072"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6" width="512" height="512"/>
    <material name="matplane" texture="texplane" texrepeat="10 10" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="matplane" friction="1 0 0"/>
    <camera name="track" pos="1.5 0 0.8" xyaxes="0 1 0 -1 0 0.5"/>
    <body name="anchor" pos="0 0 0">
      <site name="anchor" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
    </body>
    {robot_body}
  </worldbody>
  {robot_assets}
</mujoco>
"""


def extract_robot_body(mjcf_path: StrOrPath) -> tuple[str, str]:
    """Extract robot body and assets from an MJCF file.

    Args:
        mjcf_path: Path to the robot MJCF file.

    Returns:
        Tuple of (body_xml, assets_xml).
    """
    mjcf_path = Path(mjcf_path)
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    body_xml = ""
    for worldbody in root.iter("worldbody"):
        body_inner = ""
        for child in worldbody:
            body_inner += ET.tostring(child, encoding="unicode")
        body_xml = body_inner
        break

    assets_xml = ""
    for asset in root.iter("asset"):
        assets_xml = ET.tostring(asset, encoding="unicode")
        break

    return body_xml, assets_xml


def inject_robot_into_scene(
    scene_path: StrOrPath,
    robot_path: StrOrPath,
    output_path: Union[StrOrPath, None] = None,
    timestep: float = 0.002,
    gravity: str = "0 0 -9.81",
) -> Path:
    """Inject a robot MJCF into a scene template.

    Args:
        scene_path: Path to the scene template MJCF.
        robot_path: Path to the robot MJCF file.
        output_path: Output path for the combined MJCF. If None, creates temp file.
        timestep: Simulation timestep.
        gravity: Gravity vector string.

    Returns:
        Path to the combined MJCF file.
    """
    scene_path = Path(scene_path)
    robot_path = Path(robot_path)

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene template not found: {scene_path}")
    if not robot_path.exists():
        raise FileNotFoundError(f"Robot MJCF not found: {robot_path}")

    robot_body, robot_assets = extract_robot_body(robot_path)
    meshdir = str(robot_path.parent)

    combined_xml = SCENE_TEMPLATE.format(
        meshdir=meshdir,
        timestep=timestep,
        gravity=gravity,
        robot_body=robot_body,
        robot_assets=robot_assets,
    )

    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".xml"))
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined_xml)

    return output_path


def build_combined_mjcf(
    robot_path: StrOrPath,
    output_path: Union[StrOrPath, None] = None,
    scene_template: Union[StrOrPath, None] = None,
    timestep: float = 0.002,
    gravity: str = "0 0 -9.81",
) -> Path:
    """Build a combined MJCF from robot and optional scene.

    If robot MJCF already contains <worldbody>, it is used directly.
    Otherwise, robot is injected into the default scene.

    Args:
        robot_path: Path to the robot MJCF file.
        output_path: Output path. If None, uses temp file or same directory.
        scene_template: Path to scene template. If None, uses built-in default.
        timestep: Simulation timestep.
        gravity: Gravity vector string.

    Returns:
        Path to the final MJCF file.
    """
    robot_path = Path(robot_path)
    if not robot_path.exists():
        raise FileNotFoundError(f"Robot MJCF not found: {robot_path}")

    tree = ET.parse(robot_path)
    root = tree.getroot()

    has_worldbody = root.find("worldbody") is not None
    if has_worldbody:
        if output_path is None:
            output_path = robot_path.parent / f"{robot_path.stem}_scene.xml"
        output_path = Path(output_path)
        output_path.write_text(ET.tostring(root, encoding="unicode"))
        return output_path

    if scene_template is None:
        scene_template = Path(__file__).parent.parent.parent / "assets" / "scene.xml"
        if not scene_template.exists():
            scene_template = _create_default_scene()

    return inject_robot_into_scene(
        scene_path=scene_template,
        robot_path=robot_path,
        output_path=output_path,
        timestep=timestep,
        gravity=gravity,
    )


def _create_default_scene() -> Path:
    """Create a default scene template file."""
    scene_content = SCENE_TEMPLATE.format(
        meshdir=".",
        timestep=0.002,
        gravity="0 0 -9.81",
        robot_body="",
        robot_assets="",
    )
    path = Path(tempfile.mktemp(suffix=".xml"))
    path.write_text(scene_content)
    return path
