#!/usr/bin/env python3
"""Retarget Pico XRoboToolkit body tracking frames to Unitree G1 with GMR."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("PICO_EGO_TMP", "/tmp/pico_ego_collector"))
GMR_ROOT = PROJECT_ROOT / "third-party" / "GMR"
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting import RobotMotionViewer  # noqa: E402


BODY_JOINT_NAMES = [
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
]

GMR_XROBOT_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
GMR_XROBOT_ROTATION_QUAT = R.from_matrix(GMR_XROBOT_ROTATION)
MAX_HAND_FILL_GAP_SEC = 0.25
# unitree_g1 is 29-DoF: 12 leg + 3 waist + 14 arm.  The arm DoFs are
# therefore the last 14 DoFs; the saved qpos has a 7-value free root before
# them.
G1_ARM_QPOS_SLICE = slice(7 + 15, 7 + 29)


def parse_pose_string(value: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [float(part) for part in value.split(",")[:7]]
    if len(parts) != 7:
        raise ValueError(f"Expected 7 pose values, got {len(parts)}")
    px, py, pz, qx, qy, qz, qw = parts
    pos = np.array([px, py, pz], dtype=float)
    quat_wxyz = np.array([qw, qx, qy, qz], dtype=float)
    return pos, quat_wxyz


def transform_xrobot_pose(pos: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match GMR XRobotStreamer.coordinate_transform_unity_data."""
    transformed_pos = pos @ GMR_XROBOT_ROTATION.T
    transformed_quat = (
        GMR_XROBOT_ROTATION_QUAT * R.from_quat(quat_wxyz, scalar_first=True)
    ).as_quat(scalar_first=True)
    return transformed_pos, transformed_quat


def frame_to_gmr_xrobot(frame: dict[str, Any]) -> dict[str, list[np.ndarray]]:
    joints = frame.get("Body", {}).get("joints", [])
    human_frame: dict[str, list[np.ndarray]] = {}
    for i, joint_name in enumerate(BODY_JOINT_NAMES):
        if i >= len(joints):
            continue
        pose = joints[i].get("p") if isinstance(joints[i], dict) else None
        if not isinstance(pose, str):
            continue
        pos, quat_wxyz = parse_pose_string(pose)
        pos, quat_wxyz = transform_xrobot_pose(pos, quat_wxyz)
        human_frame[joint_name] = [pos, quat_wxyz]
    return human_frame


def fill_short_g1_hand_gaps(
    human_frames: list[dict[str, list[np.ndarray]]],
    raw_frames: list[dict[str, Any]],
    times: np.ndarray,
    max_gap_sec: float = MAX_HAND_FILL_GAP_SEC,
) -> None:
    """Hold the previous valid G1 hand target across short tracked gaps."""
    for joint_name, hand_key in (("Left_Hand", "leftHand"), ("Right_Hand", "rightHand")):
        active = np.asarray(
            [
                bool(int(((frame.get("Hand", {}) or {}).get(hand_key, {}) or {}).get("isActive", 0) or 0))
                for frame in raw_frames
            ],
            dtype=bool,
        )
        valid = np.asarray(
            [
                joint_name in frame
                and np.isfinite(frame[joint_name][0]).all()
                and np.isfinite(frame[joint_name][1]).all()
                for frame in human_frames
            ],
            dtype=bool,
        )
        frame = 0
        while frame < len(human_frames):
            if valid[frame]:
                frame += 1
                continue
            run_start = frame
            while frame < len(human_frames) and not valid[frame]:
                frame += 1
            run_end = frame - 1
            left = run_start - 1
            right = run_end + 1
            if (
                left >= 0
                and right < len(human_frames)
                and active[left]
                and active[right]
                and float(times[run_end] - times[run_start]) <= max_gap_sec
                and valid[left]
            ):
                human_frames[run_start : run_end + 1] = [
                    {**item, joint_name: [value.copy() for value in human_frames[left][joint_name]]}
                    for item in human_frames[run_start : run_end + 1]
                ]


def load_tracking_recording(path: Path) -> tuple[list[dict[str, Any]], int | None]:
    frames: list[dict[str, Any]] = []
    camera_t0_ns: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "Body" not in record:
                timestamp = int(record.get("timeStampNs", 0) or 0)
                if timestamp > 0 and camera_t0_ns is None:
                    camera_t0_ns = timestamp
                continue
            frames.append(record)
    return frames, camera_t0_ns


def frame_times_from_camera_clock(
    frames: list[dict[str, Any]],
    camera_t0_ns: int | None,
) -> tuple[np.ndarray, int, str]:
    if camera_t0_ns is None:
        raise ValueError(
            "Tracking metadata has no camera first-frame timeStampNs; "
            "cannot create a frozen-schema G1 motion."
        )
    timestamps = np.asarray([int(frame.get("timeStampNs", 0) or 0) for frame in frames], dtype=np.int64)
    if len(timestamps) == 0 or not np.all(timestamps > 0):
        raise ValueError("Every G1 source tracking frame must have a positive timeStampNs.")
    times = (timestamps.astype(np.float64) - float(camera_t0_ns)) / 1e9
    if np.any(np.diff(times) <= 0):
        raise ValueError("Tracking timestamps must be strictly increasing for G1 retarget output.")
    return times, camera_t0_ns, "camera_first_frame_timestamp_ns"


def infer_fps(frames: list[dict[str, Any]]) -> float:
    timestamps = np.array([frame.get("timeStampNs", 0) for frame in frames], dtype=float)
    timestamps = timestamps[timestamps > 0]
    if len(timestamps) < 2:
        return 30.0
    dt = np.diff(timestamps) / 1e9
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def discover_tracking(raw_dir: Path, session_id: str | None) -> Path:
    if session_id:
        matches = sorted((raw_dir / session_id).glob("*trackingData_*.txt"))
    else:
        matches = sorted(raw_dir.glob("*/*trackingData_*.txt"))
    if not matches and raw_dir == Path("raw"):
        legacy = PROJECT_ROOT / "data" / "raw"
        if session_id:
            matches = sorted((legacy / session_id).glob("*trackingData_*.txt"))
        else:
            matches = sorted(legacy.glob("*/*trackingData_*.txt"))
    if not matches:
        where = raw_dir / session_id if session_id else raw_dir
        raise SystemExit(f"No trackingData_*.txt found under {where}")
    return matches[0]


def retarget_frames(
    human_frames: list[dict[str, list[np.ndarray]]],
    robot: str,
    actual_human_height: float | None,
    offset_to_ground: bool,
) -> tuple[np.ndarray, GMR]:
    retargeter = GMR(
        src_human="xrobot",
        tgt_robot=robot,
        actual_human_height=actual_human_height,
        solver="daqp",
        damping=1.0,
        verbose=False,
        use_velocity_limit=False,
    )
    qpos_list = []
    for human_frame in tqdm(human_frames, desc="Retargeting"):
        qpos = retargeter.retarget(human_frame, offset_to_ground=offset_to_ground)
        qpos_list.append(qpos.copy())
    return np.asarray(qpos_list), retargeter


def postprocess_g1_qpos(
    qpos: np.ndarray,
    times: np.ndarray,
    smooth_window: int = 3,
    max_velocity: float = 8.0,
    spike_threshold: float = 0.08,
) -> np.ndarray:
    """Remove isolated arm/wrist spikes without smoothing the whole motion.

    A tracking glitch normally appears as a one-frame excursion followed by
    an immediate return.  We replace only those excursions with the local
    temporal median, then apply a short centered moving average.  A velocity
    clamp protects against remaining discontinuities.  Root pose and legs
    are intentionally left untouched.
    """
    if qpos.ndim != 2 or len(qpos) < 2 or smooth_window <= 1:
        return qpos
    if smooth_window % 2 == 0:
        smooth_window += 1

    result = qpos.astype(np.float64, copy=True)
    arm = result[:, G1_ARM_QPOS_SLICE]
    if arm.shape[1] == 0:
        return qpos

    # Despike with a 3-frame median, but preserve deliberate movement.
    if len(arm) >= 3:
        median = np.empty_like(arm)
        median[0] = arm[0]
        median[-1] = arm[-1]
        median[1:-1] = np.median(
            np.stack((arm[:-2], arm[1:-1], arm[2:])), axis=0
        )
        # Require the neighboring frames to agree.  This distinguishes an
        # isolated tracking glitch from a real, fast motion transition.
        neighbors_agree = np.abs(arm[:-2] - arm[2:]) <= spike_threshold
        isolated = np.zeros_like(arm, dtype=bool)
        isolated[1:-1] = (
            (np.abs(arm[1:-1] - median[1:-1]) > spike_threshold)
            & neighbors_agree
        )
        result[:, G1_ARM_QPOS_SLICE][isolated] = median[isolated]
        arm = result[:, G1_ARM_QPOS_SLICE]

    # Limit physically implausible frame-to-frame jumps using real timestamps.
    dt = np.maximum(np.diff(times), 1e-4)
    for i in range(1, len(arm)):
        limit = max_velocity * dt[i - 1]
        arm[i] = arm[i - 1] + np.clip(arm[i] - arm[i - 1], -limit, limit)

    # Short centered average; edge padding avoids startup/end discontinuities.
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    pad = smooth_window // 2
    padded = np.pad(arm, ((pad, pad), (0, 0)), mode="edge")
    for j in range(arm.shape[1]):
        arm[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    result[:, G1_ARM_QPOS_SLICE] = arm
    return result.astype(qpos.dtype, copy=False)


def save_gmr_motion(
    output_path: Path,
    qpos: np.ndarray,
    times: np.ndarray,
    fps: float,
    tracking_path: Path,
    robot: str,
    time_origin_ns: int,
    time_source: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    motion_data = {
        "fps": fps,
        "times": np.asarray(times, dtype=np.float64),
        "time_origin_ns": time_origin_ns,
        "time_source": time_source,
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],  # GMR pkl convention: xyzw
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
        "source_tracking": str(tracking_path),
        "source_human_format": "xrobot",
        "target_robot": robot,
    }
    with output_path.open("wb") as f:
        pickle.dump(motion_data, f)


def visualize(qpos: np.ndarray, fps: float, robot: str, rate_limit: bool) -> None:
    viewer = RobotMotionViewer(robot_type=robot, motion_fps=fps, camera_follow=True)
    try:
        for i in range(len(qpos)):
            viewer.step(
                root_pos=qpos[i, :3],
                root_rot=qpos[i, 3:7],
                dof_pos=qpos[i, 7:],
                rate_limit=rate_limit,
                follow_camera=True,
            )
    finally:
        viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--tracking", type=Path, default=None)
    parser.add_argument("--robot", type=str, default="unitree_g1", choices=["unitree_g1"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--actual-human-height", type=float, default=None)
    parser.add_argument("--no-offset-to-ground", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--rate-limit", action="store_true")
    parser.add_argument(
        "--g1-smooth-window",
        type=int,
        default=3,
        help="Centered smoothing window for G1 arm/wrist DoFs; 1 disables it.",
    )
    parser.add_argument(
        "--g1-max-arm-velocity",
        type=float,
        default=8.0,
        help="Maximum arm joint velocity in rad/s during post-processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    tracking_path = args.tracking or discover_tracking(args.raw_dir, args.session)
    session_id = args.session or tracking_path.stem.split("trackingData_", 1)[-1]
    output_path = args.output or TMP_ROOT / session_id / "retargeted" / f"{session_id}_{args.robot}.pkl"

    raw_frames, camera_t0_ns = load_tracking_recording(tracking_path)
    source_fps = args.fps or infer_fps(raw_frames)
    all_times, time_origin_ns, time_source = frame_times_from_camera_clock(
        raw_frames, camera_t0_ns
    )
    selected_indices = np.arange(len(raw_frames), dtype=np.int64)[args.start :: args.stride]
    if args.max_frames is not None:
        selected_indices = selected_indices[: args.max_frames]
    if len(selected_indices) == 0:
        raise SystemExit("No frames selected.")
    selected = [raw_frames[int(index)] for index in selected_indices]
    selected_times = all_times[selected_indices]

    output_fps = source_fps / args.stride
    human_frames = [frame_to_gmr_xrobot(frame) for frame in selected]
    fill_short_g1_hand_gaps(human_frames, selected, selected_times)

    required = {"Pelvis", "Spine3", "Left_Foot", "Right_Foot", "Left_Wrist", "Right_Wrist"}
    missing = sorted(required - set(human_frames[0]))
    if missing:
        raise SystemExit(f"Missing required GMR xrobot bodies in first frame: {missing}")

    qpos, _ = retarget_frames(
        human_frames,
        robot=args.robot,
        actual_human_height=args.actual_human_height,
        offset_to_ground=not args.no_offset_to_ground,
    )
    qpos = postprocess_g1_qpos(
        qpos,
        selected_times,
        smooth_window=args.g1_smooth_window,
        max_velocity=args.g1_max_arm_velocity,
    )
    save_gmr_motion(
        output_path,
        qpos,
        selected_times,
        output_fps,
        tracking_path,
        args.robot,
        time_origin_ns,
        time_source,
    )

    print(f"tracking={tracking_path}")
    print(f"frames={len(qpos)} fps={output_fps:.3f}")
    print(f"time_source={time_source} time_range_sec=[{selected_times[0]:.6f}, {selected_times[-1]:.6f}]")
    print(f"qpos_shape={qpos.shape}")
    print(f"saved={output_path}")

    if args.visualize:
        visualize(qpos, output_fps, args.robot, args.rate_limit)


if __name__ == "__main__":
    main()
