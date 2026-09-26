#!/usr/bin/env python3
"""Visualize Pico body and hand tracking skeletons in MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np


MODEL_XML = """
<mujoco model="pico_tracking_viewer">
  <compiler angle="degree"/>
  <option timestep="0.011111"/>
  <visual>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="20"/>
  </visual>
  <worldbody>
    <light name="key" pos="0 -2 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" pos="0 0 -1.75" size="3 3 0.01"
          rgba="0.16 0.17 0.18 1"/>
    <geom name="origin_x" type="capsule" fromto="0 0 -1.749 0.5 0 -1.749"
          size="0.005" rgba="0.8 0.2 0.2 1"/>
    <geom name="origin_y" type="capsule" fromto="0 0 -1.748 0 0.5 -1.748"
          size="0.005" rgba="0.2 0.8 0.2 1"/>
  </worldbody>
</mujoco>
"""


# Approximate Pico body order inferred from the collected 24-joint samples.
# Keep this list local and easy to edit once the capture-side SDK enum is fixed.
BODY_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (1, 4),
    (4, 7),
    (7, 10),
    (2, 5),
    (5, 8),
    (8, 11),
    (9, 13),
    (13, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (9, 14),
    (14, 17),
    (17, 19),
    (19, 21),
    (21, 23),
    (13, 14),
]


# The hand data has 26 ordered joint poses. The names are not stored in the
# JSONL, so this is a conventional 26-joint palm/finger topology for display.
HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (1, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (1, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
]


RGBA_BODY = np.array([0.15, 0.55, 1.0, 1.0], dtype=np.float32)
RGBA_LEFT = np.array([1.0, 0.70, 0.20, 1.0], dtype=np.float32)
RGBA_RIGHT = np.array([0.25, 0.95, 0.55, 1.0], dtype=np.float32)
RGBA_HEAD = np.array([1.0, 0.25, 0.25, 1.0], dtype=np.float32)
RGBA_INACTIVE = np.array([0.55, 0.55, 0.55, 0.35], dtype=np.float32)


@dataclass
class TrackingFrame:
    timestamp_ns: int
    body: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    head: np.ndarray
    left_active: bool
    right_active: bool


def transform_position(pos: np.ndarray, mode: str) -> np.ndarray:
    """Map Pico tracking coordinates into the MuJoCo display frame."""
    if mode == "raw":
        return pos
    if mode == "pico_y_up_to_mujoco_z_up":
        return np.array([pos[0], -pos[2], pos[1]], dtype=float)
    if mode == "pico_y_up_to_mujoco_z_up_no_forward_flip":
        return np.array([pos[0], pos[2], pos[1]], dtype=float)
    raise ValueError(f"Unknown coordinate transform: {mode}")


def transform_positions(points: np.ndarray, mode: str) -> np.ndarray:
    if len(points) == 0 or mode == "raw":
        return points
    return np.vstack([transform_position(point, mode) for point in points])


def parse_pose(value: object, dims: int = 7) -> np.ndarray:
    if not isinstance(value, str):
        return np.full((dims,), np.nan, dtype=float)
    out: list[float] = []
    for part in value.split(","):
        try:
            out.append(float(part))
        except ValueError:
            out.append(math.nan)
    if len(out) < dims:
        out.extend([math.nan] * (dims - len(out)))
    return np.asarray(out[:dims], dtype=float)


def parse_joint_positions(joints: object) -> np.ndarray:
    if not isinstance(joints, list):
        return np.zeros((0, 3), dtype=float)
    poses = [parse_pose(joint.get("p") if isinstance(joint, dict) else None)[:3] for joint in joints]
    return np.vstack(poses) if poses else np.zeros((0, 3), dtype=float)


def load_tracking(path: Path, coord_transform: str) -> list[TrackingFrame]:
    frames: list[TrackingFrame] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "notice" in record or "cameraIntrinsics" in record:
                continue

            hand = record.get("Hand", {}) or {}
            left = hand.get("leftHand", {}) or {}
            right = hand.get("rightHand", {}) or {}
            body = record.get("Body", {}) or {}
            head = record.get("Head", {}) or {}
            body_points = parse_joint_positions(body.get("joints"))
            left_points = parse_joint_positions(left.get("HandJointLocations"))
            right_points = parse_joint_positions(right.get("HandJointLocations"))
            head_point = parse_pose(head.get("pose"))[:3]
            frames.append(
                TrackingFrame(
                    timestamp_ns=int(record.get("timeStampNs", 0) or 0),
                    body=transform_positions(body_points, coord_transform),
                    left_hand=transform_positions(left_points, coord_transform),
                    right_hand=transform_positions(right_points, coord_transform),
                    head=transform_position(head_point, coord_transform),
                    left_active=bool(left.get("isActive", 0)),
                    right_active=bool(right.get("isActive", 0)),
                )
            )
    return frames


def discover_tracking(raw_dir: Path, session_id: str | None) -> Path:
    if session_id:
        matches = sorted((raw_dir / session_id).glob("trackingData_*.txt"))
    else:
        matches = sorted(raw_dir.glob("*/trackingData_*.txt"))
    if not matches and raw_dir == Path("raw"):
        legacy = PROJECT_ROOT / "data" / "raw"
        if session_id:
            matches = sorted((legacy / session_id).glob("trackingData_*.txt"))
        else:
            matches = sorted(legacy.glob("*/trackingData_*.txt"))
    if not matches:
        where = raw_dir / session_id if session_id else raw_dir
        raise SystemExit(f"No trackingData_*.txt found under {where}")
    return matches[0]


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


def add_capsule(
    scene: mujoco.MjvScene,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
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


def add_skeleton(
    scene: mujoco.MjvScene,
    points: np.ndarray,
    edges: Iterable[tuple[int, int]],
    rgba: np.ndarray,
    joint_radius: float,
    bone_radius: float,
) -> None:
    for i, j in edges:
        if i < len(points) and j < len(points):
            add_capsule(scene, points[i], points[j], bone_radius, rgba)
    for point in points:
        add_sphere(scene, point, joint_radius, rgba)


def update_scene(
    scene: mujoco.MjvScene,
    frame: TrackingFrame,
    show_body: bool,
    show_hands: bool,
) -> None:
    scene.ngeom = 0
    if show_body:
        add_skeleton(scene, frame.body, BODY_EDGES, RGBA_BODY, 0.025, 0.012)
        add_sphere(scene, frame.head, 0.04, RGBA_HEAD)
    if show_hands:
        left_rgba = RGBA_LEFT if frame.left_active else RGBA_INACTIVE
        right_rgba = RGBA_RIGHT if frame.right_active else RGBA_INACTIVE
        add_skeleton(scene, frame.left_hand, HAND_EDGES, left_rgba, 0.012, 0.005)
        add_skeleton(scene, frame.right_hand, HAND_EDGES, right_rgba, 0.012, 0.005)


def center_camera(frames: list[TrackingFrame], viewer: mujoco.viewer.Handle) -> None:
    points: list[np.ndarray] = []
    for frame in frames[:: max(1, len(frames) // 200)]:
        for arr in (frame.body, frame.left_hand, frame.right_hand):
            if len(arr):
                points.append(arr[np.all(np.isfinite(arr), axis=1)])
    if not points:
        return
    cloud = np.vstack(points)
    center = np.nanmean(cloud, axis=0)
    span = float(np.nanmax(np.linalg.norm(cloud - center, axis=1)))
    viewer.cam.lookat[:] = center
    viewer.cam.distance = max(2.0, span * 2.4)
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20


def print_summary(path: Path, frames: list[TrackingFrame]) -> None:
    if not frames:
        print(f"{path}: no tracking frames")
        return
    duration = (frames[-1].timestamp_ns - frames[0].timestamp_ns) / 1e9
    body_counts = sorted({len(frame.body) for frame in frames})
    left_counts = sorted({len(frame.left_hand) for frame in frames})
    right_counts = sorted({len(frame.right_hand) for frame in frames})
    left_active = 100 * sum(frame.left_active for frame in frames) / len(frames)
    right_active = 100 * sum(frame.right_active for frame in frames) / len(frames)
    print(f"tracking={path}")
    print(f"frames={len(frames)} duration_sec={duration:.3f}")
    print(f"body_joint_counts={body_counts}")
    print(f"left_hand_joint_counts={left_counts} active_pct={left_active:.1f}")
    print(f"right_hand_joint_counts={right_counts} active_pct={right_active:.1f}")


def play(
    frames: list[TrackingFrame],
    fps: float,
    stride: int,
    start: int,
    max_frames: int | None,
    show_body: bool,
    show_hands: bool,
) -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    selected = frames[start::stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise SystemExit("No frames selected for visualization.")

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=True
    ) as viewer:
        center_camera(selected, viewer)
        period = 1.0 / fps if fps > 0 else 0.0
        frame_i = 0
        while viewer.is_running():
            frame = selected[frame_i]
            with viewer.lock():
                update_scene(viewer.user_scn, frame, show_body, show_hands)
            viewer.set_texts(
                (
                    None,
                    None,
                    "Pico tracking",
                    f"frame {frame_i + 1}/{len(selected)}  "
                    f"t={(frame.timestamp_ns - selected[0].timestamp_ns) / 1e9:.3f}s",
                )
            )
            viewer.sync()
            frame_i = (frame_i + 1) % len(selected)
            if period:
                time.sleep(period)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--tracking", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--coord-transform",
        choices=[
            "pico_y_up_to_mujoco_z_up",
            "pico_y_up_to_mujoco_z_up_no_forward_flip",
            "raw",
        ],
        default="pico_y_up_to_mujoco_z_up",
        help=(
            "Coordinate mapping for visualization. Pico tracking is Y-up in the "
            "current samples; MuJoCo viewer uses Z-up."
        ),
    )
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--hands-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracking_path = args.tracking or discover_tracking(args.raw_dir, args.session)
    frames = load_tracking(tracking_path, args.coord_transform)
    print_summary(tracking_path, frames)
    print(f"coord_transform={args.coord_transform}")
    if args.dry_run:
        return
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    play(
        frames=frames,
        fps=args.fps,
        stride=args.stride,
        start=args.start,
        max_frames=args.max_frames,
        show_body=not args.hands_only,
        show_hands=not args.body_only,
    )


if __name__ == "__main__":
    main()
