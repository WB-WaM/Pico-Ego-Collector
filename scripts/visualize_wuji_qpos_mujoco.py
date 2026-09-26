#!/usr/bin/env python3
"""Visualize retargeted Wuji Hand qpos in a standalone MuJoCo viewer.

The current wuji_sdk package ships URDF assets but not MJCF assets. MuJoCo cannot
load those URDFs directly in this environment because their mesh paths are
package-relative. This script therefore uses the same URDF with Pinocchio for
forward kinematics, then draws the resulting hand skeleton in MuJoCo.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

if any(arg == "--render-output" or arg.startswith("--render-output=") for arg in sys.argv[1:]):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("PICO_EGO_TMP", "/tmp/pico_ego_collector"))
WUJI_ROOT = PROJECT_ROOT / "third-party" / "wuji-retargeting"
if WUJI_ROOT.exists() and str(WUJI_ROOT) not in sys.path:
    sys.path.insert(0, str(WUJI_ROOT))

MODEL_XML = """
<mujoco model="wuji_qpos_viewer">
  <visual>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.01" zfar="10"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".22 .23 .25" rgb2=".16 .17 .18" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="2 2" reflectance=".05"/>
  </asset>
  <worldbody>
    <light pos="0 -0.5 1.2" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" pos="0 0 -0.12" material="grid"/>
  </worldbody>
</mujoco>
"""

RGBA_LEFT = np.array([0.12, 0.55, 1.0, 1.0], dtype=np.float64)
RGBA_RIGHT = np.array([1.0, 0.48, 0.16, 1.0], dtype=np.float64)
RGBA_INACTIVE = np.array([0.45, 0.46, 0.48, 0.35], dtype=np.float64)
RGBA_JOINT = np.array([0.94, 0.96, 0.98, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class HandModel:
    side: str
    urdf_path: Path
    robot: object
    link_names: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class HandMotion:
    side: str
    qpos: np.ndarray
    active: np.ndarray
    model: HandModel
    offset: np.ndarray


@dataclass(frozen=True)
class DirectModel:
    model: mujoco.MjModel
    path: Path
    qpos_perm: np.ndarray
    kind: str


def discover_motion(session_id: str | None, input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path
    if not session_id:
        raise SystemExit("Use --session or --input.")
    candidates = [
        TMP_ROOT / session_id / "wuji_retargeted" / f"{session_id}_wuji_hand.pkl",
        PROJECT_ROOT / "data" / "wuji_retargeted" / f"{session_id}_wuji_hand.pkl",
        PROJECT_ROOT / "data" / "retargeted" / f"{session_id}_wuji_hand.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    checked = "\n  ".join(str(path) for path in candidates)
    raise SystemExit(f"Wuji hand qpos pkl not found. Checked:\n  {checked}")


def default_render_path(input_path: Path, session_id: str | None) -> Path:
    if session_id:
        return TMP_ROOT / session_id / "previews" / f"{session_id}_wuji_hand_mujoco.mp4"
    return input_path.with_name(input_path.stem + "_mujoco.mp4")


def find_sdk_urdf(side: str, hand_model: str) -> Path | None:
    try:
        import wuji_sdk
    except ImportError:
        return None
    sdk_root = Path(wuji_sdk.__file__).resolve().parent
    path = sdk_root / "_retargeting" / "wuji-description" / hand_model / "body" / "urdf" / f"{side}.urdf"
    return path if path.exists() else None


def resolve_urdf(data: dict, side: str, hand_model: str) -> Path:
    saved_paths = data.get("urdf_paths") or {}
    saved = saved_paths.get(side)
    if saved and Path(saved).exists():
        return Path(saved)
    candidates = [
        WUJI_ROOT / "wuji_retargeting" / "wuji-description" / hand_model / "body" / "urdf" / f"{side}.urdf",
        find_sdk_urdf(side, hand_model),
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path.resolve()
    checked = "\n  ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"No Wuji {hand_model} URDF found for {side}. Checked:\n  {checked}")


def parse_urdf_links_and_edges(path: Path) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    root = ET.parse(path).getroot()
    links = {link.attrib["name"] for link in root.findall("link") if "name" in link.attrib}
    edges: list[tuple[str, str]] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib.get("link")
        child_name = child.attrib.get("link")
        if parent_name in links and child_name in links:
            edges.append((parent_name, child_name))

    ordered_links: list[str] = []
    for a, b in edges:
        if a not in ordered_links:
            ordered_links.append(a)
        if b not in ordered_links:
            ordered_links.append(b)
    for link in sorted(links):
        if link not in ordered_links:
            ordered_links.append(link)
    return tuple(ordered_links), tuple(edges)


def build_hand_model(data: dict, side: str) -> HandModel:
    from wuji_retargeting.robot import RobotWrapper

    hand_model = str(data.get("hand_model") or "hand")
    urdf_path = resolve_urdf(data, side, hand_model)
    robot = RobotWrapper(str(urdf_path), hand_side=side)
    link_names, edges = parse_urdf_links_and_edges(urdf_path)
    joint_names = tuple(str(name) for name in robot.dof_joint_names)
    return HandModel(side=side, urdf_path=urdf_path, robot=robot, link_names=link_names, edges=edges, joint_names=joint_names)


def primitive_box_for_link(link_name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    name = link_name.lower()
    if "palm" in name:
        return (0.055, 0.025, 0.075), (0.0, 0.0, 0.025)
    if "tip" in name:
        return (0.012, 0.012, 0.018), (0.0, 0.0, -0.006)
    if "link1" in name:
        return (0.012, 0.012, 0.018), (0.0, 0.0, 0.006)
    if "link2" in name:
        return (0.014, 0.014, 0.038), (0.0, 0.0, 0.019)
    if "link3" in name:
        return (0.013, 0.013, 0.030), (0.0, 0.0, 0.015)
    if "link4" in name:
        return (0.012, 0.012, 0.020), (0.0, 0.0, 0.010)
    return (0.012, 0.012, 0.018), (0.0, 0.0, 0.0)


def write_primitive_urdf(urdf_path: Path, side: str) -> Path:
    """Write a MuJoCo-loadable URDF with primitive visuals instead of missing meshes."""
    cache_key = hashlib.sha1(str(urdf_path.resolve()).encode("utf-8")).hexdigest()[:10]
    out_dir = TMP_ROOT / "_assets" / "wuji_urdf_viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{side}_{cache_key}_primitive.urdf"
    if out_path.exists() and out_path.stat().st_mtime >= urdf_path.stat().st_mtime:
        return out_path

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    mujoco_node = root.find("mujoco")
    if mujoco_node is not None:
        for compiler in mujoco_node.findall("compiler"):
            compiler.attrib.pop("meshdir", None)

    material_name = f"{side}_primitive_material"
    for material in list(root.findall("material")):
        root.remove(material)
    material = ET.Element("material", {"name": material_name})
    color = "0.12 0.55 1.0 1" if side == "left" else "1.0 0.48 0.16 1"
    ET.SubElement(material, "color", {"rgba": color})
    root.insert(0, material)

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for collision in list(link.findall("collision")):
            link.remove(collision)

        size, origin = primitive_box_for_link(link_name)
        visual = ET.SubElement(link, "visual")
        ET.SubElement(
            visual,
            "origin",
            {
                "xyz": f"{origin[0]:.6g} {origin[1]:.6g} {origin[2]:.6g}",
                "rpy": "0 0 0",
            },
        )
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "box", {"size": f"{size[0]:.6g} {size[1]:.6g} {size[2]:.6g}"})
        ET.SubElement(visual, "material", {"name": material_name})

        collision = ET.SubElement(link, "collision")
        ET.SubElement(
            collision,
            "origin",
            {
                "xyz": f"{origin[0]:.6g} {origin[1]:.6g} {origin[2]:.6g}",
                "rpy": "0 0 0",
            },
        )
        collision_geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(collision_geometry, "box", {"size": f"{size[0]:.6g} {size[1]:.6g} {size[2]:.6g}"})

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def load_motion(path: Path, sides: Iterable[str]) -> tuple[float, np.ndarray, list[HandMotion], dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    fps = float(data.get("fps", 30.0) or 30.0)
    times = np.asarray(data.get("times", []), dtype=np.float64)
    motions: list[HandMotion] = []
    sides = tuple(sides)
    both = set(sides) == {"left", "right"}

    for side in sides:
        qpos = np.asarray(data[f"{side}_qpos"], dtype=np.float64)
        if qpos.ndim != 2:
            raise ValueError(f"Expected {side}_qpos shape (T, D), got {qpos.shape}")
        active = np.asarray(data.get(f"{side}_active", np.ones(len(qpos), dtype=bool)), dtype=bool)
        model = build_hand_model(data, side)
        if qpos.shape[1] != len(model.joint_names):
            raise ValueError(
                f"{side}_qpos dim {qpos.shape[1]} does not match URDF joint count {len(model.joint_names)}"
            )
        offset = np.array([-0.08, 0.0, 0.0], dtype=np.float64) if both and side == "left" else np.zeros(3)
        if both and side == "right":
            offset = np.array([0.08, 0.0, 0.0], dtype=np.float64)
        motions.append(HandMotion(side=side, qpos=qpos, active=active[: len(qpos)], model=model, offset=offset))

    frame_count = min(len(motion.qpos) for motion in motions)
    motions = [
        HandMotion(m.side, m.qpos[:frame_count], m.active[:frame_count], m.model, m.offset)
        for m in motions
    ]
    if len(times) >= frame_count:
        times = times[:frame_count]
    else:
        times = np.arange(frame_count, dtype=np.float64) / fps
    return fps, times, motions, data


def sample_indices(
    frame_count: int,
    source_fps: float,
    preview_fps: float,
    start_sec: float,
    end_sec: float | None,
    stride: int,
    max_frames: int | None,
) -> np.ndarray:
    duration = frame_count / source_fps
    start = max(0.0, start_sec)
    end = min(duration, end_sec if end_sec is not None else duration)
    if end <= start:
        raise ValueError(f"Invalid time range: start={start:.3f}, end={end:.3f}")
    if preview_fps > 0:
        times = np.arange(start, end, 1.0 / preview_fps, dtype=np.float64)
        indices = np.rint(times * source_fps).astype(np.int64)
    else:
        indices = np.arange(int(start * source_fps), int(end * source_fps) + 1, dtype=np.int64)
    indices = np.clip(indices, 0, frame_count - 1)
    indices = indices[:: max(1, stride)]
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def valid_pos(pos: np.ndarray) -> bool:
    return pos.shape == (3,) and bool(np.all(np.isfinite(pos)))


def add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= len(scene.geoms) or not valid_pos(pos):
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        pos.astype(np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        rgba,
    )
    scene.ngeom += 1


def add_capsule(scene: mujoco.MjvScene, p0: np.ndarray, p1: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= len(scene.geoms) or not valid_pos(p0) or not valid_pos(p1):
        return
    if float(np.linalg.norm(p1 - p0)) < 1e-6:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        p0.astype(np.float64),
        p1.astype(np.float64),
    )
    scene.ngeom += 1


def hand_points(motion: HandMotion, frame_idx: int) -> dict[str, np.ndarray]:
    qpos = np.nan_to_num(motion.qpos[frame_idx], nan=0.0)
    motion.model.robot.compute_forward_kinematics(qpos)
    points: dict[str, np.ndarray] = {}
    for link_name in motion.model.link_names:
        try:
            link_id = motion.model.robot.get_link_index(link_name)
            pose = motion.model.robot.get_link_pose(link_id)
        except RuntimeError:
            continue
        points[link_name] = np.asarray(pose[:3, 3], dtype=np.float64) + motion.offset
    return points


def draw_hand(scene: mujoco.MjvScene, motion: HandMotion, frame_idx: int) -> None:
    points = hand_points(motion, frame_idx)
    rgba = RGBA_LEFT if motion.side == "left" else RGBA_RIGHT
    if frame_idx >= len(motion.active) or not bool(motion.active[frame_idx]):
        rgba = RGBA_INACTIVE

    for parent, child in motion.model.edges:
        if parent in points and child in points:
            add_capsule(scene, points[parent], points[child], 0.0035, rgba)
    for point in points.values():
        add_sphere(scene, point, 0.006, RGBA_JOINT if rgba[3] > 0.5 else RGBA_INACTIVE)


def update_scene(scene: mujoco.MjvScene, motions: list[HandMotion], frame_idx: int, keep_existing: bool = False) -> None:
    if not keep_existing:
        scene.ngeom = 0
    for motion in motions:
        draw_hand(scene, motion, frame_idx)


def camera_from_points(motions: list[HandMotion], indices: np.ndarray, max_samples: int = 80) -> tuple[np.ndarray, float]:
    all_points: list[np.ndarray] = []
    if len(indices) == 0:
        return np.zeros(3, dtype=np.float64), 0.45
    step = max(1, len(indices) // max_samples)
    for idx in indices[::step]:
        for motion in motions:
            points = list(hand_points(motion, int(idx)).values())
            if points:
                all_points.append(np.vstack(points))
    if not all_points:
        return np.zeros(3, dtype=np.float64), 0.45
    cloud = np.vstack(all_points)
    center = np.nanmean(cloud, axis=0)
    span = float(np.nanmax(np.linalg.norm(cloud - center, axis=1)))
    return center, max(0.25, span * 3.0)


def configure_camera(camera: mujoco.MjvCamera, center: np.ndarray, distance: float) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = distance
    camera.azimuth = 145.0
    camera.elevation = -25.0


def play(
    motions: list[HandMotion],
    indices: np.ndarray,
    source_fps: float,
    playback_fps: float,
    rate_limit: bool,
) -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    center, distance = camera_from_points(motions, indices)
    dt = 1.0 / (playback_fps if playback_fps > 0 else source_fps)

    print("Opening MuJoCo viewer. Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = center
        viewer.cam.distance = distance
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25
        i = 0
        while viewer.is_running():
            frame_idx = int(indices[i % len(indices)])
            loop_start = time.time()
            mujoco.mj_forward(model, data)
            with viewer.lock():
                update_scene(viewer.user_scn, motions, frame_idx)
            viewer.set_texts(
                (
                    None,
                    None,
                    "Wuji qpos",
                    f"frame {frame_idx}/{len(motions[0].qpos) - 1}  "
                    f"hands={','.join(m.side for m in motions)}",
                )
            )
            viewer.sync()
            i += 1
            if rate_limit:
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, dt - elapsed))


def render_video(
    motions: list[HandMotion],
    indices: np.ndarray,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    center, distance = camera_from_points(motions, indices)
    configure_camera(camera, center, distance)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for frame_idx in tqdm(indices, desc="Rendering Wuji hand MuJoCo"):
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            update_scene(renderer.scene, motions, int(frame_idx), keep_existing=True)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    return {
        "output": str(output_path),
        "frames": int(len(indices)),
        "fps": fps,
        "width": width,
        "height": height,
    }


def load_urdf_model(motion: HandMotion) -> tuple[mujoco.MjModel, Path]:
    model_path = write_primitive_urdf(motion.model.urdf_path, motion.side)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != motion.qpos.shape[1]:
        raise ValueError(
            f"URDF model nq {model.nq} does not match {motion.side}_qpos dim {motion.qpos.shape[1]}"
        )
    return model, model_path


def resolve_mjcf(motion: HandMotion) -> Path | None:
    parts = motion.model.urdf_path.parts
    hand_model = "hand2" if "hand2" in parts else "hand"
    candidates = [
        WUJI_ROOT
        / "wuji_retargeting"
        / "wuji-description"
        / hand_model
        / "body"
        / "mjcf"
        / f"{motion.side}.xml",
        WUJI_ROOT
        / "wuji_retargeting"
        / "wuji-description"
        / hand_model
        / "body-with-soft"
        / "mjcf"
        / f"{motion.side}.xml",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def absolutize_mesh_files(root: ET.Element, mjcf_path: Path) -> None:
    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    mesh_base = (mjcf_path.parent / meshdir).resolve()
    for mesh in root.findall("./asset/mesh"):
        file_name = mesh.attrib.get("file")
        if not file_name:
            continue
        file_path = Path(file_name)
        if not file_path.is_absolute():
            file_path = (mesh_base / file_path).resolve()
        mesh.set("file", str(file_path))


def write_combined_mjcf(motions: list[HandMotion]) -> Path:
    mjcf_paths = [resolve_mjcf(motion) for motion in motions]
    if any(path is None for path in mjcf_paths):
        missing = [motion.side for motion, path in zip(motions, mjcf_paths) if path is None]
        raise FileNotFoundError(f"Official Wuji MJCF missing for: {missing}")

    key_parts = []
    for path in mjcf_paths:
        assert path is not None
        key_parts.append(f"{path.resolve()}:{path.stat().st_mtime_ns}")
    cache_key = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:10]
    out_dir = TMP_ROOT / "_assets" / "wuji_mjcf_viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"both_{cache_key}.xml"
    if out_path.exists():
        return out_path

    combined = ET.Element("mujoco", {"model": "wuji_hands_both"})
    ET.SubElement(combined, "compiler", {"angle": "radian"})
    asset = ET.SubElement(combined, "asset")
    worldbody = ET.SubElement(combined, "worldbody")

    offsets = {
        "left": "-0.08 0 0",
        "right": "0.08 0 0",
    }
    copied_option = False
    copied_default = False
    for motion, path in zip(motions, mjcf_paths):
        assert path is not None
        src_root = ET.parse(path).getroot()
        absolutize_mesh_files(src_root, path)

        if not copied_option:
            option = src_root.find("option")
            if option is not None:
                combined.insert(1, copy.deepcopy(option))
                copied_option = True
        if not copied_default:
            default = src_root.find("default")
            if default is not None:
                combined.insert(2 if copied_option else 1, copy.deepcopy(default))
                copied_default = True

        src_asset = src_root.find("asset")
        if src_asset is not None:
            for child in list(src_asset):
                asset.append(copy.deepcopy(child))

        wrapper = ET.SubElement(
            worldbody,
            "body",
            {"name": f"{motion.side}_viewer_root", "pos": offsets.get(motion.side, "0 0 0")},
        )
        src_worldbody = src_root.find("worldbody")
        if src_worldbody is not None:
            for child in list(src_worldbody):
                if child.tag == "body":
                    wrapper.append(copy.deepcopy(child))

    ET.ElementTree(combined).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def qpos_perm_by_joint_name(model: mujoco.MjModel, source_joint_names: tuple[str, ...]) -> np.ndarray:
    if model.nq != len(source_joint_names):
        raise ValueError(f"MuJoCo model nq {model.nq} does not match qpos dim {len(source_joint_names)}")
    src_index = {name: i for i, name in enumerate(source_joint_names)}
    perm = np.arange(model.nq, dtype=np.int64)
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qadr = int(model.jnt_qposadr[joint_id])
        if name not in src_index:
            raise ValueError(f"MuJoCo joint {name!r} not found in qpos joint order")
        perm[qadr] = src_index[name]
    return perm


def source_joint_names(motions: list[HandMotion]) -> tuple[str, ...]:
    names: list[str] = []
    for motion in motions:
        names.extend(motion.model.joint_names)
    return tuple(names)


def frame_qpos(motions: list[HandMotion], frame_idx: int) -> np.ndarray:
    return np.concatenate(
        [np.nan_to_num(motion.qpos[frame_idx], nan=0.0) for motion in motions],
        axis=0,
    )


def load_direct_model(motions: list[HandMotion], prefer: str) -> DirectModel:
    if len(motions) > 1 and prefer == "urdf":
        raise ValueError("Primitive URDF render supports one hand at a time. Use --render-mode mjcf for both hands.")

    if prefer in {"mjcf", "auto"}:
        if len(motions) == 1:
            mjcf_path = resolve_mjcf(motions[0])
            if mjcf_path is not None:
                model = mujoco.MjModel.from_xml_path(str(mjcf_path))
                perm = qpos_perm_by_joint_name(model, source_joint_names(motions))
                return DirectModel(model=model, path=mjcf_path, qpos_perm=perm, kind="mjcf")
        else:
            mjcf_path = write_combined_mjcf(motions)
            model = mujoco.MjModel.from_xml_path(str(mjcf_path))
            perm = qpos_perm_by_joint_name(model, source_joint_names(motions))
            return DirectModel(model=model, path=mjcf_path, qpos_perm=perm, kind="mjcf")
        if prefer == "mjcf":
            raise FileNotFoundError(
                "Official Wuji MJCF not found. Run: "
                "git -C third-party/wuji-retargeting submodule update --init --recursive wuji_retargeting/wuji-description"
            )

    model, model_path = load_urdf_model(motions[0])
    perm = qpos_perm_by_joint_name(model, source_joint_names(motions))
    return DirectModel(model=model, path=model_path, qpos_perm=perm, kind="primitive_urdf")


def configure_urdf_camera(camera: mujoco.MjvCamera, side: str) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.045]
    camera.distance = 0.32
    camera.azimuth = 145.0 if side == "left" else -35.0
    camera.elevation = -25.0


def play_direct(
    motions: list[HandMotion],
    indices: np.ndarray,
    source_fps: float,
    playback_fps: float,
    rate_limit: bool,
    prefer: str,
) -> None:
    direct = load_direct_model(motions, prefer)
    model = direct.model
    data = mujoco.MjData(model)
    dt = 1.0 / (playback_fps if playback_fps > 0 else source_fps)

    print(f"Opening MuJoCo viewer with Wuji {direct.kind}: {direct.path}")
    print("Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.045]
        viewer.cam.distance = 0.32
        viewer.cam.azimuth = 145 if motions[0].side == "left" else -35
        viewer.cam.elevation = -25
        i = 0
        while viewer.is_running():
            frame_idx = int(indices[i % len(indices)])
            loop_start = time.time()
            data.qpos[:] = frame_qpos(motions, frame_idx)[direct.qpos_perm]
            mujoco.mj_forward(model, data)
            viewer.set_texts(
                (
                    None,
                    None,
                    "Wuji qpos",
                    f"{direct.kind} {','.join(m.side for m in motions)} frame {frame_idx}/{len(motions[0].qpos) - 1}",
                )
            )
            viewer.sync()
            i += 1
            if rate_limit:
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, dt - elapsed))


def render_direct_video(
    motions: list[HandMotion],
    indices: np.ndarray,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    prefer: str,
) -> dict[str, object]:
    direct = load_direct_model(motions, prefer)
    model = direct.model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    configure_urdf_camera(camera, motions[0].side)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        for frame_idx in tqdm(indices, desc=f"Rendering Wuji {direct.kind} MuJoCo"):
            data.qpos[:] = frame_qpos(motions, int(frame_idx))[direct.qpos_perm]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    return {
        "output": str(output_path),
        "model": str(direct.path),
        "model_kind": direct.kind,
        "frames": int(len(indices)),
        "fps": fps,
        "width": width,
        "height": height,
    }


def print_summary(path: Path, fps: float, times: np.ndarray, motions: list[HandMotion], data: dict) -> None:
    duration = len(times) / fps if fps > 0 else 0.0
    print(f"motion={path}")
    print(f"source_tracking={data.get('source_tracking', '')}")
    print(f"frames={len(times)} fps={fps:.6f} duration_sec={duration:.3f}")
    for motion in motions:
        active_pct = 100.0 * float(np.mean(motion.active[: len(times)])) if len(times) else 0.0
        print(f"{motion.side}_qpos_shape={motion.qpos.shape} active_pct={active_pct:.1f}")
        print(f"{motion.side}_urdf={motion.model.urdf_path}")
        print(f"{motion.side}_joint_order={list(motion.model.joint_names)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--fps", type=float, default=30.0, help="Playback/render sampling FPS. Use 0 for raw frame stepping.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--rate-limit", action="store_true", default=True)
    parser.add_argument("--no-rate-limit", dest="rate_limit", action="store_false")
    parser.add_argument("--render-output", type=Path, default=None)
    parser.add_argument(
        "--render-mode",
        choices=["auto", "mjcf", "urdf", "skeleton"],
        default="auto",
        help="auto uses official MJCF when available; skeleton is an explicit fallback/debug mode.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_path = discover_motion(args.session, args.input)
    sides = ("left", "right") if args.hand == "both" else (args.hand,)
    source_fps, times, motions, data = load_motion(motion_path, sides)
    print_summary(motion_path, source_fps, times, motions, data)
    if args.dry_run:
        return

    frame_count = min(len(motion.qpos) for motion in motions)
    indices = sample_indices(
        frame_count,
        source_fps,
        args.fps,
        args.start_sec,
        args.end_sec,
        args.stride,
        args.max_frames,
    )
    if len(indices) == 0:
        raise SystemExit("No frames selected.")

    auto_has_mjcf = args.render_mode == "auto" and all(resolve_mjcf(motion) is not None for motion in motions)
    use_direct_model = args.render_mode in {"mjcf", "urdf"} or auto_has_mjcf
    if args.render_mode == "urdf" and len(motions) != 1:
        raise SystemExit("Primitive URDF render supports one hand at a time. Use --render-mode mjcf for both.")
    direct_prefer = "mjcf" if args.render_mode in {"auto", "mjcf"} else "urdf"

    if args.render_output is not None:
        output_path = args.render_output
        if str(output_path) == "auto":
            output_path = default_render_path(motion_path, args.session)
        if use_direct_model:
            result = render_direct_video(
                motions,
                indices,
                output_path,
                args.fps if args.fps > 0 else source_fps,
                args.width,
                args.height,
                direct_prefer,
            )
        else:
            result = render_video(motions, indices, output_path, args.fps if args.fps > 0 else source_fps, args.width, args.height)
        print(result)
        return

    if use_direct_model:
        play_direct(motions, indices, source_fps, args.fps, args.rate_limit, direct_prefer)
    else:
        play(motions, indices, source_fps, args.fps, args.rate_limit)


if __name__ == "__main__":
    main()
