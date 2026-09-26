#!/usr/bin/env python3
"""Retarget Pico XRoboToolkit hand tracking frames to Wuji Hand qpos."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("PICO_EGO_TMP", "/tmp/pico_ego_collector"))
WUJI_ROOT = PROJECT_ROOT / "third-party" / "wuji-retargeting"
if WUJI_ROOT.exists() and str(WUJI_ROOT) not in sys.path:
    sys.path.insert(0, str(WUJI_ROOT))

DEFAULT_CONFIG = WUJI_ROOT / "example" / "config" / "adaptive_analytical_avp.yaml"

PICO_HAND_JOINT_NAMES = [
    "Palm",
    "Wrist",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexMetacarpal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleMetacarpal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingMetacarpal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleMetacarpal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
]

MEDIAPIPE_HAND_JOINT_NAMES = [
    "Wrist",
    "ThumbCMC",
    "ThumbMCP",
    "ThumbIP",
    "ThumbTip",
    "IndexMCP",
    "IndexPIP",
    "IndexDIP",
    "IndexTip",
    "MiddleMCP",
    "MiddlePIP",
    "MiddleDIP",
    "MiddleTip",
    "RingMCP",
    "RingPIP",
    "RingDIP",
    "RingTip",
    "PinkyMCP",
    "PinkyPIP",
    "PinkyDIP",
    "PinkyTip",
]

# XRoboToolkit uses OpenXR ordering: Palm=0, Wrist=1, followed by the fingers.
# https://registry.khronos.org/OpenXR/specs/1.0/man/html/XrHandJointEXT.html
# MediaPipe has Wrist + thumb CMC/MCP/IP/tip + four fingers MCP/PIP/DIP/tip.
# Drop Pico Palm and non-thumb metacarpals.
PICO26_TO_MEDIAPIPE21 = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    12,
    13,
    14,
    15,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
)

COORD_TRANSFORMS = {
    "raw": np.eye(3, dtype=float),
    "xrobot_y_up_to_z_up": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    ),
}

DISTAL_JOINT_INDICES = np.array([3, 7, 11, 15, 19], dtype=np.int64)


def parse_position(value: str) -> np.ndarray:
    parts = [float(part) for part in value.split(",")[:3]]
    if len(parts) != 3:
        raise ValueError(f"Expected at least 3 pose values, got {len(parts)}")
    return np.asarray(parts, dtype=np.float64)


def load_tracking_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "Hand" not in record:
                continue
            frames.append(record)
    return frames


def infer_fps(frames: list[dict[str, Any]]) -> float:
    timestamps = np.asarray([frame.get("timeStampNs", 0) for frame in frames], dtype=float)
    timestamps = timestamps[timestamps > 0]
    if len(timestamps) < 2:
        return 30.0
    dt = np.diff(timestamps) / 1e9
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def frame_times(frames: list[dict[str, Any]], fps: float) -> np.ndarray:
    timestamps = np.asarray([frame.get("timeStampNs", 0) for frame in frames], dtype=float)
    if len(timestamps) > 0 and np.all(timestamps > 0):
        return (timestamps - timestamps[0]) / 1e9
    return np.arange(len(frames), dtype=float) / fps


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


def extract_pico_hand_points(
    frame: dict[str, Any],
    hand_side: str,
    use_inactive_points: bool,
) -> tuple[bool, np.ndarray | None]:
    hand_key = f"{hand_side}Hand"
    hand = frame.get("Hand", {}).get(hand_key, {})
    active = bool(int(hand.get("isActive", 0) or 0))
    if not active and not use_inactive_points:
        return False, None

    joints = hand.get("HandJointLocations", [])
    if not isinstance(joints, list) or len(joints) < len(PICO_HAND_JOINT_NAMES):
        return active, None

    points = np.full((len(PICO_HAND_JOINT_NAMES), 3), np.nan, dtype=np.float64)
    for i, joint in enumerate(joints[: len(PICO_HAND_JOINT_NAMES)]):
        pose = joint.get("p") if isinstance(joint, dict) else None
        if not isinstance(pose, str):
            return active, None
        points[i] = parse_position(pose)

    if not np.isfinite(points).all():
        return active, None
    return active, points


def pico26_to_mediapipe21(points: np.ndarray, coord_transform: str) -> np.ndarray:
    if points.shape != (26, 3):
        raise ValueError(f"Expected Pico hand points with shape (26, 3), got {points.shape}")
    mediapipe_points = points[list(PICO26_TO_MEDIAPIPE21)].astype(np.float64, copy=True)
    return mediapipe_points @ COORD_TRANSFORMS[coord_transform].T


def find_module_dir(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def infer_hand_model(config: dict[str, Any], fallback: str) -> str:
    urdf_path = str(config.get("optimizer", {}).get("urdf_path", ""))
    if "hand2" in urdf_path:
        return "hand2"
    return fallback


def patch_side_specific_link_names(config: dict[str, Any], hand_side: str) -> None:
    link_naming = config.get("optimizer", {}).get("link_naming")
    if not isinstance(link_naming, dict):
        return
    prefix = link_naming.get("prefix")
    if hand_side == "left" and prefix == "r_":
        link_naming["prefix"] = "l_"
    elif hand_side == "right" and prefix == "l_":
        link_naming["prefix"] = "r_"


def resolve_wuji_urdf(
    config_path: Path,
    config: dict[str, Any],
    hand_side: str,
    hand_model: str,
    urdf_override: Path | None,
) -> Path:
    if urdf_override is not None:
        path = urdf_override
        if path.is_dir():
            path = path / f"{hand_side}.urdf"
        if not path.exists():
            raise FileNotFoundError(f"URDF override not found: {path}")
        return path.resolve()

    opt_config = config.setdefault("optimizer", {})
    configured = opt_config.get("urdf_path")
    model = infer_hand_model(config, hand_model)
    candidates: list[Path] = []

    if configured:
        configured_path = Path(str(configured))
        if not configured_path.is_absolute():
            configured_path = (config_path.parent / configured_path).resolve()
        candidates.append(configured_path.with_name(f"{hand_side}.urdf"))
        candidates.append(configured_path)
    else:
        candidates.append(
            WUJI_ROOT
            / "wuji_retargeting"
            / "wuji-description"
            / model
            / "body"
            / "urdf"
            / f"{hand_side}.urdf"
        )

    sdk_dir = find_module_dir("wuji_sdk")
    if sdk_dir is not None:
        candidates.append(
            sdk_dir
            / "_retargeting"
            / "wuji-description"
            / model
            / "body"
            / "urdf"
            / f"{hand_side}.urdf"
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No Wuji {model} URDF found for {hand_side}. Checked:\n  {checked}")


def build_retargeter(
    config_path: Path,
    hand_side: str,
    hand_model: str,
    urdf_override: Path | None,
):
    from wuji_retargeting import Retargeter

    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["__yaml_dir"] = str(config_path.parent)
    config.setdefault("optimizer", {})
    config["optimizer"]["urdf_path"] = str(
        resolve_wuji_urdf(config_path, config, hand_side, hand_model, urdf_override)
    )
    patch_side_specific_link_names(config, hand_side)
    return Retargeter.from_config(config, hand_side), Path(config["optimizer"]["urdf_path"])


def load_pico_recommendations(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    recommendations = config.get("pico_ego_collector", {})
    return recommendations if isinstance(recommendations, dict) else {}


def inactive_qpos(policy: str, num_joints: int, last_qpos: np.ndarray | None) -> np.ndarray:
    if policy == "fill-last" and last_qpos is not None:
        return last_qpos.copy()
    if policy == "zero":
        return np.zeros(num_joints, dtype=np.float64)
    return np.full(num_joints, np.nan, dtype=np.float64)


def inactive_mediapipe(policy: str, last_points: np.ndarray | None) -> np.ndarray:
    if policy == "fill-last" and last_points is not None:
        return last_points.copy()
    if policy == "zero":
        return np.zeros((21, 3), dtype=np.float64)
    return np.full((21, 3), np.nan, dtype=np.float64)


def contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return ranges


def smooth_qpos(qpos: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return qpos
    if window % 2 == 0:
        window += 1

    smoothed = qpos.copy()
    finite = np.isfinite(qpos).all(axis=1)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2

    for start, end in contiguous_true_ranges(finite):
        segment = qpos[start:end]
        if len(segment) < 2:
            continue
        padded = np.pad(segment, ((pad, pad), (0, 0)), mode="edge")
        for joint_i in range(segment.shape[1]):
            smoothed[start:end, joint_i] = np.convolve(padded[:, joint_i], kernel, mode="valid")
    return smoothed


def limit_qpos_frame_step(qpos: np.ndarray, max_step: float | None) -> np.ndarray:
    if max_step is None or max_step <= 0:
        return qpos

    limited = qpos.copy()
    last: np.ndarray | None = None
    for i in range(len(limited)):
        if not np.isfinite(limited[i]).all():
            last = None
            continue
        if last is not None:
            delta = np.clip(limited[i] - last, -max_step, max_step)
            limited[i] = last + delta
        last = limited[i].copy()
    return limited


def postprocess_qpos(
    qpos: np.ndarray,
    distal_min_angle: float | None,
    smooth_window: int,
    max_frame_step: float | None,
) -> np.ndarray:
    processed = qpos.astype(np.float64, copy=True)
    processed = limit_qpos_frame_step(processed, max_frame_step)
    processed = smooth_qpos(processed, smooth_window)
    if distal_min_angle is not None:
        finite = np.isfinite(processed[:, DISTAL_JOINT_INDICES]).all(axis=1)
        processed[np.ix_(finite, DISTAL_JOINT_INDICES)] = np.maximum(
            processed[np.ix_(finite, DISTAL_JOINT_INDICES)],
            distal_min_angle,
        )
    return processed


def retarget_hand_sequence(
    frames: list[dict[str, Any]],
    hand_side: str,
    retargeter,
    coord_transform: str,
    inactive_policy: str,
    use_inactive_points: bool,
    apply_filter: bool,
) -> dict[str, np.ndarray]:
    num_joints = int(getattr(retargeter.optimizer, "num_joints", 20))
    qpos = np.full((len(frames), num_joints), np.nan, dtype=np.float64)
    mediapipe_points = np.full((len(frames), 21, 3), np.nan, dtype=np.float64)
    active = np.zeros(len(frames), dtype=bool)
    valid_points = np.zeros(len(frames), dtype=bool)

    last_qpos: np.ndarray | None = None
    last_points: np.ndarray | None = None

    for i, frame in enumerate(tqdm(frames, desc=f"Retargeting {hand_side} hand")):
        is_active, pico_points = extract_pico_hand_points(frame, hand_side, use_inactive_points)
        active[i] = is_active

        if pico_points is None:
            mediapipe_points[i] = inactive_mediapipe(inactive_policy, last_points)
            qpos[i] = inactive_qpos(inactive_policy, num_joints, last_qpos)
            continue

        mp_points = pico26_to_mediapipe21(pico_points, coord_transform)
        mediapipe_points[i] = mp_points
        valid_points[i] = True

        next_qpos = retargeter.retarget(mp_points, apply_filter=apply_filter).copy()
        qpos[i] = next_qpos
        last_points = mp_points
        last_qpos = next_qpos

    return {
        "qpos": qpos,
        "mediapipe_points": mediapipe_points,
        "active": active,
        "valid_points": valid_points,
    }


def zero_or_points(points: np.ndarray) -> np.ndarray:
    if np.isfinite(points).all():
        return points.astype(np.float32, copy=False)
    return np.zeros((21, 3), dtype=np.float32)


def build_mediapipe_replay(
    times: np.ndarray,
    left_points: np.ndarray | None,
    right_points: np.ndarray | None,
) -> list[dict[str, Any]]:
    replay = []
    for i, timestamp in enumerate(times):
        if left_points is None:
            left = np.zeros((21, 3), dtype=np.float32)
        else:
            left = zero_or_points(left_points[i])
        if right_points is None:
            right = np.zeros((21, 3), dtype=np.float32)
        else:
            right = zero_or_points(right_points[i])
        replay.append(
            {
                "t": float(timestamp),
                "left_fingers": left,
                "right_fingers": right,
            }
        )
    return replay


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--tracking", type=Path, default=None)
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hand-model", choices=["hand", "hand2"], default="hand")
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--replay-output", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--coord-transform", choices=sorted(COORD_TRANSFORMS), default=None)
    parser.add_argument("--inactive-policy", choices=["fill-last", "nan", "zero"], default="fill-last")
    parser.add_argument("--use-inactive-points", action="store_true")
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--filter", dest="force_filter", action="store_true", help="Force Wuji low-pass filtering.")
    filter_group.add_argument("--no-filter", dest="no_filter", action="store_true", help="Disable Wuji low-pass filtering.")
    parser.add_argument(
        "--distal-min-angle",
        type=float,
        default=None,
        help="Clamp Wuji finger*_joint4 qpos to this minimum angle in radians after retargeting.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Optional odd moving-average window, in frames, applied to qpos after retargeting.",
    )
    parser.add_argument(
        "--max-frame-step",
        type=float,
        default=None,
        help="Optional per-joint max qpos change per tracking frame, in radians, before smoothing.",
    )
    parser.add_argument("--no-replay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    tracking_path = args.tracking or discover_tracking(args.raw_dir, args.session)
    session_id = args.session or tracking_path.stem.split("trackingData_", 1)[-1]
    output_path = args.output or TMP_ROOT / session_id / "wuji_retargeted" / f"{session_id}_wuji_hand.pkl"
    replay_output_path = (
        args.replay_output or TMP_ROOT / session_id / "wuji_retargeted" / f"{session_id}_mediapipe_replay.pkl"
    )

    raw_frames = load_tracking_frames(tracking_path)
    selected = raw_frames[args.start :: args.stride]
    if args.max_frames is not None:
        selected = selected[: args.max_frames]
    if not selected:
        raise SystemExit("No frames selected.")

    recommendations = load_pico_recommendations(args.config)
    coord_transform = args.coord_transform or recommendations.get("recommended_coord_transform", "raw")
    if coord_transform not in COORD_TRANSFORMS:
        raise SystemExit(f"Unknown recommended coord transform in {args.config}: {coord_transform}")
    if args.force_filter:
        apply_filter = True
    elif args.no_filter:
        apply_filter = False
    elif "recommended_apply_filter" in recommendations:
        apply_filter = bool(recommendations["recommended_apply_filter"])
    else:
        apply_filter = True

    postprocess_recommendations = recommendations.get("qpos_postprocess", {})
    if not isinstance(postprocess_recommendations, dict):
        postprocess_recommendations = {}
    distal_min_angle = (
        args.distal_min_angle
        if args.distal_min_angle is not None
        else postprocess_recommendations.get("distal_min_angle")
    )
    smooth_window = (
        args.smooth_window
        if args.smooth_window is not None
        else int(postprocess_recommendations.get("smooth_window", 1))
    )
    max_frame_step = (
        args.max_frame_step
        if args.max_frame_step is not None
        else postprocess_recommendations.get("max_frame_step")
    )
    if smooth_window < 1:
        raise SystemExit("--smooth-window must be >= 1")
    if distal_min_angle is not None:
        distal_min_angle = float(distal_min_angle)
    if max_frame_step is not None:
        max_frame_step = float(max_frame_step)

    source_fps = args.fps or infer_fps(raw_frames)
    output_fps = source_fps / args.stride
    times = frame_times(selected, output_fps)
    hands = ["left", "right"] if args.hand == "both" else [args.hand]

    results: dict[str, dict[str, np.ndarray]] = {}
    urdf_paths: dict[str, str] = {}
    for hand_side in hands:
        retargeter, urdf_path = build_retargeter(
            args.config,
            hand_side,
            args.hand_model,
            args.urdf,
        )
        urdf_paths[hand_side] = str(urdf_path)
        results[hand_side] = retarget_hand_sequence(
            selected,
            hand_side,
            retargeter,
            coord_transform,
            args.inactive_policy,
            args.use_inactive_points,
            apply_filter=apply_filter,
        )
        results[hand_side]["qpos"] = postprocess_qpos(
            results[hand_side]["qpos"],
            distal_min_angle,
            smooth_window,
            max_frame_step,
        )

    payload: dict[str, Any] = {
        "fps": output_fps,
        "times": times,
        "source_tracking": str(tracking_path),
        "config": str(args.config),
        "urdf_paths": urdf_paths,
        "hand_model": args.hand_model,
        "coord_transform": coord_transform,
        "apply_filter": apply_filter,
        "qpos_postprocess": {
            "distal_min_angle": distal_min_angle,
            "smooth_window": smooth_window,
            "max_frame_step": max_frame_step,
            "distal_joint_indices": DISTAL_JOINT_INDICES.tolist(),
        },
        "inactive_policy": args.inactive_policy,
        "pico_hand_joint_names": PICO_HAND_JOINT_NAMES,
        "mediapipe_hand_joint_names": MEDIAPIPE_HAND_JOINT_NAMES,
        "pico26_to_mediapipe21": list(PICO26_TO_MEDIAPIPE21),
    }
    for hand_side, result in results.items():
        payload[f"{hand_side}_qpos"] = result["qpos"]
        payload[f"{hand_side}_active"] = result["active"]
        payload[f"{hand_side}_valid_points"] = result["valid_points"]
        payload[f"{hand_side}_mediapipe_points"] = result["mediapipe_points"]

    save_pickle(output_path, payload)

    if not args.no_replay:
        replay = build_mediapipe_replay(
            times,
            results.get("left", {}).get("mediapipe_points"),
            results.get("right", {}).get("mediapipe_points"),
        )
        save_pickle(replay_output_path, replay)

    print(f"tracking={tracking_path}")
    print(f"frames={len(selected)} fps={output_fps:.3f}")
    print(f"hand={args.hand} coord_transform={coord_transform} apply_filter={apply_filter}")
    print(
        "qpos_postprocess="
        f"distal_min_angle={distal_min_angle} "
        f"smooth_window={smooth_window} "
        f"max_frame_step={max_frame_step}"
    )
    for hand_side in hands:
        result = results[hand_side]
        print(
            f"{hand_side}_qpos_shape={result['qpos'].shape} "
            f"active_frames={int(result['active'].sum())} "
            f"valid_frames={int(result['valid_points'].sum())}"
        )
        print(f"{hand_side}_urdf={urdf_paths[hand_side]}")
    print(f"saved={output_path}")
    if not args.no_replay:
        print(f"replay_saved={replay_output_path}")


if __name__ == "__main__":
    main()
