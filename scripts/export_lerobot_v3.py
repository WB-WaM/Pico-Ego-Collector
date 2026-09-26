#!/usr/bin/env python3
"""Export annotated Pico sessions to the frozen Pico Ego LeRobot v3 schema v2.

Frozen schema v2 (see docs/lerobot_v3_release_spec.md):

- Video/tracking are aligned to the CAMERA clock: the first tracking line
  (metadata) carries the first-frame image ``timeStampNs`` and is used as t=0,
  so image time and pose time share one origin.
- Video and most low-dimensional fields use nearest-neighbor timestamp sampling;
  missing hand poses remain marked invalid for downstream handling.
- Pico poses are stored once in the native ``world`` frame (quat xyzw).
  Pelvis-/wrist-relative canonical poses are deterministic derived data and are
  intentionally left to downstream preprocessing.
- No aggregate ``observation.state`` is stored; downstream chooses and converts
  the component fields appropriate for its policy.
- The release is pure observation data; downstream training defines ``action``.
- Missing/non-finite poses are checked per frame and reported in an exported
  ``observation.pico_joint_valid`` mask. Missing hand poses use zero sentinels
  (not valid rotations); the mask remains zero. Export requires confirmation.
- Camera intrinsics/extrinsics and alignment metadata are written to a sidecar
  ``meta/pico_ego_release.json`` (LeRobot ``info.json`` cannot hold them).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import math
import os
import pickle
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def emit_progress(**payload: Any) -> None:
    """Emit one machine-readable progress line to stderr.

    stdout stays reserved for the final summary JSON; the annotator server
    parses these ``@PROGRESS {...}`` lines to drive the web progress bar.
    """
    try:
        sys.stderr.write("@PROGRESS " + json.dumps(payload, default=str) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def install_video_encoder_worker() -> None:
    """Make LeRobot use the local encoder with configurable FFmpeg options."""
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset
        try:
            from scripts.lerobot_export_video import encode_video_worker
        except ImportError:
            from lerobot_export_video import encode_video_worker

        lerobot_dataset._encode_video_worker = encode_video_worker
    except ImportError:
        # Dry-run and validation can run without the lerobot environment.
        pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("PICO_EGO_TMP", "/tmp/pico_ego_collector"))
SCHEMA_VERSION = "pico_ego_v3_schema_v2"
SCHEMA_STATUS = "frozen"

BODY_JOINT_COUNT = 24
HAND_JOINT_COUNT = 26
HANDS_JOINT_COUNT = HAND_JOINT_COUNT * 2  # left + right
POSE_DIM = 7
G1_QPOS_DIM = 36  # root(3) + quat(4) + 29 dof

HEAD_POSE_DIM = POSE_DIM
PICO_BODY_POSE_DIM = BODY_JOINT_COUNT * POSE_DIM          # 168
PICO_HANDS26_POSE_DIM = HANDS_JOINT_COUNT * POSE_DIM      # 364
JOINT_VALID_DIM = 1 + BODY_JOINT_COUNT + HANDS_JOINT_COUNT  # head + body + hands = 77
JOINT_VALID_ORDER = "head(1) + body(24) + left_hand(26) + right_hand(26) = 77"
JOINT_VALID_NAMES = (
    ["head"]
    + [f"body[{i}]" for i in range(BODY_JOINT_COUNT)]
    + [f"left_hand[{i}]" for i in range(HAND_JOINT_COUNT)]
    + [f"right_hand[{i}]" for i in range(HAND_JOINT_COUNT)]
)
MAX_HAND_FILL_GAP_SEC = 0.25

# This is a pure-observation dataset (no `action`); downstream defines its own
# action (e.g. next-frame observation.g1_qpos, or a delta) at training time.

# The Pico camera records a side-by-side stereo frame; by default we split it
# into two eyes. --mono keeps the raw combined frame under IMAGE_KEY_MONO.
IMAGE_KEY_MONO = "observation.images.pico_ego"
IMAGE_KEY_STEREO_LEFT = "observation.images.stereo_left"
IMAGE_KEY_STEREO_RIGHT = "observation.images.stereo_right"

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_pose(value: object, dims: int = POSE_DIM) -> np.ndarray:
    if not isinstance(value, str):
        return np.full((dims,), np.nan, dtype=np.float32)
    out: list[float] = []
    for part in value.split(","):
        try:
            out.append(float(part))
        except ValueError:
            out.append(math.nan)
    if len(out) < dims:
        out.extend([math.nan] * (dims - len(out)))
    return np.asarray(out[:dims], dtype=np.float32)


def parse_joint_poses(joints: object, count: int) -> np.ndarray:
    poses = np.full((count, POSE_DIM), np.nan, dtype=np.float32)
    if isinstance(joints, list):
        for i, joint in enumerate(joints[:count]):
            poses[i] = parse_pose(joint.get("p") if isinstance(joint, dict) else None)
    return poses


def parse_float_list(text: object) -> list[float]:
    if not isinstance(text, str):
        return []
    out: list[float] = []
    for tok in text.replace("[", " ").replace("]", " ").replace("|", " ").replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def parse_camera_metadata(record: dict[str, Any]) -> dict[str, Any]:
    intr = parse_float_list(record.get("cameraIntrinsics"))
    extr_raw = str(record.get("cameraExtrinsics", "") or "")
    extrinsics = []
    for block in extr_raw.split("|"):
        vals = parse_float_list(block)
        if len(vals) == 16:
            extrinsics.append(np.asarray(vals, dtype=np.float64).reshape(4, 4).tolist())
    return {
        "camera_intrinsics_raw": str(record.get("cameraIntrinsics", "") or ""),
        "camera_intrinsics": intr,  # sample layout: [cx, cy, fx, fy]
        "camera_extrinsics_raw": extr_raw,
        "camera_extrinsics": extrinsics,  # list of 4x4
        "first_frame_timestamp_ns": int(record.get("timeStampNs", 0) or 0),
        "notice": str(record.get("notice", "") or ""),
    }


# --------------------------------------------------------------------------- #
# Pose cleanup
# --------------------------------------------------------------------------- #
def hemisphere_continuity(quats: np.ndarray) -> np.ndarray:
    """Flip sign so consecutive frames stay on the same hemisphere. quats: (T, J, 4)."""
    if len(quats) < 2:
        return quats
    dots = np.sum(quats[1:] * quats[:-1], axis=-1)  # (T-1, J)
    signs = np.where(dots < 0.0, -1.0, 1.0).astype(np.float32)
    cum = np.cumprod(signs, axis=0)
    factors = np.concatenate([np.ones((1, quats.shape[1]), dtype=np.float32), cum], axis=0)
    return quats * factors[..., None]


def fill_and_normalize(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """poses: (T, J, 7). Returns (clean poses, per-joint validity mask (T, J))."""
    valid = np.isfinite(poses).all(axis=-1)
    pos = np.nan_to_num(poses[..., :3], nan=0.0, posinf=0.0, neginf=0.0)
    quat = np.nan_to_num(poses[..., 3:7], nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    bad = ~np.isfinite(norm) | (norm < 1e-6)
    quat = np.where(bad, IDENTITY_QUAT, quat / np.maximum(norm, 1e-8))
    quat = hemisphere_continuity(quat.astype(np.float32))
    clean = np.concatenate([pos.astype(np.float32), quat.astype(np.float32)], axis=-1)
    return clean, valid


def fill_short_hand_gaps(
    poses: np.ndarray,
    valid: np.ndarray,
    active: np.ndarray,
    times: np.ndarray,
    max_gap_sec: float = MAX_HAND_FILL_GAP_SEC,
) -> np.ndarray:
    """Hold the previous real hand pose only across short tracked gaps.

    A gap is eligible only when the hand was active immediately before and
    after it. This keeps normal standing periods (active=0) invalid. The
    validity mask is intentionally unchanged, so downstream can identify all
    synthesized values.
    """
    out = poses.astype(np.float32, copy=True)
    frame_count = len(times)
    for side in range(2):
        start_joint = side * HAND_JOINT_COUNT
        side_active = np.asarray(active[:, side] > 0.5, dtype=bool)
        for joint in range(HAND_JOINT_COUNT):
            index = start_joint + joint
            invalid = ~valid[:, index]
            frame = 0
            while frame < frame_count:
                if not invalid[frame]:
                    frame += 1
                    continue
                run_start = frame
                while frame < frame_count and invalid[frame]:
                    frame += 1
                run_end = frame - 1
                left = run_start - 1
                right = run_end + 1
                if (
                    left >= 0
                    and right < frame_count
                    and side_active[left]
                    and side_active[right]
                    and float(times[run_end] - times[run_start]) <= max_gap_sec
                    and valid[left, index]
                ):
                    out[run_start : run_end + 1, index] = out[left, index]
    return out


# --------------------------------------------------------------------------- #
# Rotation helpers for time sampling
# --------------------------------------------------------------------------- #
def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    out = quat.astype(np.float32, copy=True)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    ok = np.isfinite(norm) & (norm > 1e-8)
    return np.where(ok, out / np.maximum(norm, 1e-8), out)


# --------------------------------------------------------------------------- #
# Tracking / G1 loading
# --------------------------------------------------------------------------- #
def discover_session_files(raw_dir: Path, session_id: str) -> tuple[Path, Path]:
    session_dir = raw_dir / session_id
    if not session_dir.exists() and raw_dir == Path("raw"):
        legacy = PROJECT_ROOT / "data" / "raw" / session_id
        if legacy.exists():
            session_dir = legacy
    tracking_files = sorted(session_dir.glob("*trackingData_*.txt"))
    video_files = sorted(session_dir.glob("*CameraRecord_*.mp4"))
    if not tracking_files:
        raise FileNotFoundError(f"No trackingData_*.txt under {session_dir}")
    if not video_files:
        raise FileNotFoundError(f"No CameraRecord_*.mp4 under {session_dir}")
    return tracking_files[0], video_files[0]


def estimate_rate_hz(times: np.ndarray) -> float | None:
    if len(times) < 2:
        return None
    dt = np.diff(times.astype(np.float64))
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else None


def load_tracking_arrays(path: Path) -> dict[str, Any]:
    times: list[float] = []
    head_rows, body_rows, hand_rows, active_rows = [], [], [], []
    camera_meta: dict[str, Any] = {}
    camera_t0_ns: int | None = None  # camera first-frame clock, from metadata line

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "notice" in record or "cameraIntrinsics" in record:
                camera_meta = parse_camera_metadata(record)
                ts0 = camera_meta.get("first_frame_timestamp_ns", 0)
                if ts0 and camera_t0_ns is None:
                    camera_t0_ns = int(ts0)
                continue
            ts = int(record.get("timeStampNs", 0) or 0)
            if ts <= 0:
                continue
            if camera_t0_ns is None:  # no metadata clock -> fall back to first data frame
                camera_t0_ns = ts
            times.append((ts - camera_t0_ns) / 1e9)

            hand = record.get("Hand", {}) or {}
            left = hand.get("leftHand", {}) or {}
            right = hand.get("rightHand", {}) or {}
            body = record.get("Body", {}) or {}
            head_rows.append(parse_pose((record.get("Head", {}) or {}).get("pose")))
            body_rows.append(parse_joint_poses(body.get("joints"), BODY_JOINT_COUNT))
            left_pose = parse_joint_poses(left.get("HandJointLocations"), HAND_JOINT_COUNT)
            right_pose = parse_joint_poses(right.get("HandJointLocations"), HAND_JOINT_COUNT)
            hand_rows.append(np.concatenate([left_pose, right_pose], axis=0))
            active_rows.append(
                [float(int(left.get("isActive", 0) or 0)), float(int(right.get("isActive", 0) or 0))]
            )

    times_arr = np.asarray(times, dtype=np.float32)
    head = np.asarray(head_rows, dtype=np.float32).reshape(-1, 1, POSE_DIM)
    body = np.asarray(body_rows, dtype=np.float32).reshape(-1, BODY_JOINT_COUNT, POSE_DIM)
    hands = np.asarray(hand_rows, dtype=np.float32).reshape(-1, HANDS_JOINT_COUNT, POSE_DIM)

    head_f, head_valid = fill_and_normalize(head)
    body_f, body_valid = fill_and_normalize(body)
    hands_f, hands_valid = fill_and_normalize(hands)
    # A missing raw hand joint is an all-zero sentinel, including quaternion.
    # Never let normalization turn missing hand data into an observed rotation.
    hands_f[~hands_valid] = 0.0
    hands_f = fill_short_hand_gaps(hands_f, hands_valid, np.asarray(active_rows), times_arr)

    joint_valid = np.concatenate([head_valid, body_valid, hands_valid], axis=1).astype(np.float32)

    return {
        "times": times_arr,
        "head_pose": head_f.reshape(len(times_arr), HEAD_POSE_DIM),
        "body_pose": body_f.reshape(len(times_arr), PICO_BODY_POSE_DIM),
        "hands26_pose": hands_f.reshape(len(times_arr), PICO_HANDS26_POSE_DIM),
        "hand_active": np.asarray(active_rows, dtype=np.float32).reshape(-1, 2),
        "joint_valid": joint_valid,
        "camera_meta": camera_meta,
        "camera_t0_ns": camera_t0_ns,
        "first_data_offset_sec": float(times_arr[0]) if len(times_arr) else 0.0,
    }


def load_g1(path: Path, expected_time_origin_ns: int | None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"G1 retarget pkl not found: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    root = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    dof = np.asarray(data["dof_pos"], dtype=np.float32)
    qpos = np.concatenate([root, root_rot, dof], axis=1).astype(np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != G1_QPOS_DIM or len(qpos) == 0:
        raise ValueError(f"G1 qpos must have non-empty shape (T, {G1_QPOS_DIM}): {path}")
    if not np.isfinite(qpos).all():
        raise ValueError(f"G1 motion contains non-finite qpos values: {path}")

    stored_times = data.get("times")
    if stored_times is None:
        raise ValueError(
            f"G1 pkl has no per-frame times: {path}. Regenerate it with retarget_pico_to_g1.py."
        )
    times = np.asarray(stored_times, dtype=np.float64)
    if times.shape != (len(qpos),):
        raise ValueError(f"G1 times shape {times.shape} != qpos frames {(len(qpos),)}: {path}")
    if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError(f"G1 times must be finite and strictly increasing: {path}")

    time_source = data.get("time_source")
    time_origin_ns = data.get("time_origin_ns")
    if time_source != "camera_first_frame_timestamp_ns" or time_origin_ns is None:
        raise ValueError(
            f"G1 pkl must use camera_first_frame_timestamp_ns with time_origin_ns: {path}"
        )
    if expected_time_origin_ns is None:
        raise ValueError("Tracking metadata has no camera first-frame timeStampNs.")
    if int(time_origin_ns) != int(expected_time_origin_ns):
        raise ValueError(
            f"G1 time origin {time_origin_ns} != tracking camera origin {expected_time_origin_ns}: {path}"
        )
    return {
        "times": times,
        "qpos": qpos,
        "time_source": time_source,
        "time_origin_ns": int(time_origin_ns),
    }


# --------------------------------------------------------------------------- #
# Time sampling
# --------------------------------------------------------------------------- #
def nearest_time_indices(times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Return the nearest source index for each target time; ties choose left."""
    source = np.asarray(times, dtype=np.float64)
    targets = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if len(source) == 0:
        raise ValueError("Cannot sample an empty time series.")
    right = np.clip(np.searchsorted(source, targets, side="left"), 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_right = np.abs(source[right] - targets) < np.abs(targets - source[left])
    return np.where(choose_right, right, left).astype(np.int64)


def sample_nearest_by_time(
    times: np.ndarray,
    values: np.ndarray,
    t: float,
    normalize_slice: slice | None = None,
) -> np.ndarray:
    if len(times) == 0 or len(values) == 0:
        raise ValueError("Cannot sample empty time series.")
    if len(times) != len(values):
        raise ValueError(f"times length {len(times)} != values length {len(values)}")
    index = int(nearest_time_indices(times, np.asarray([t]))[0])
    out = values[index].astype(np.float32, copy=True)

    if normalize_slice is not None:
        quat = out[normalize_slice]
        norm = float(np.linalg.norm(quat))
        if norm > 1e-8 and np.isfinite(norm):
            out[normalize_slice] = quat / norm
    return out


def sample_pose_block(times: np.ndarray, values: np.ndarray, t: float, pose_count: int) -> np.ndarray:
    out = sample_nearest_by_time(times, values, t).reshape(pose_count, POSE_DIM)
    out[:, 3:7] = normalize_quat_xyzw(out[:, 3:7])
    return out.reshape(pose_count * POSE_DIM).astype(np.float32)


# --------------------------------------------------------------------------- #
# Sequential video reader (no per-frame keyframe seeks)
# --------------------------------------------------------------------------- #
def nearest_uniform_frame_index(t: float, fps: float) -> int:
    position = max(0.0, t * fps)
    left = math.floor(position)
    right = math.ceil(position)
    return left if (position - left) <= (right - position) else right


class SequentialVideoReader:
    def __init__(self, path: Path, fps: float, seek_threshold: int = 30) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.fps = fps if fps > 0 else 60.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.seek_threshold = seek_threshold
        self.pos = -1  # index of last decoded frame
        self._last_bgr: np.ndarray | None = None

    def frame_at_time(self, t: float) -> np.ndarray:
        target = nearest_uniform_frame_index(t, self.fps)
        if self.frame_count > 0:
            target = min(target, self.frame_count - 1)
        # Seek only on backward moves or large forward gaps (episode boundaries).
        if target < self.pos or (target - self.pos) > self.seek_threshold:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.pos = target - 1
        while self.pos < target:
            ok, frame = self.cap.read()
            if not ok:
                if self._last_bgr is not None:
                    break  # clamp to last good frame near EOF
                raise RuntimeError(f"Could not read video frame at t={t:.3f}s (index {target})")
            self.pos += 1
            self._last_bgr = frame
        if self._last_bgr is None:
            raise RuntimeError(f"No decodable frame at t={t:.3f}s")
        return cv2.cvtColor(self._last_bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self.cap.release()


def video_info(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, frames


# --------------------------------------------------------------------------- #
# Annotations / features / payload
# --------------------------------------------------------------------------- #
def load_annotations(path: Path, session_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("session_id") != session_id:
        raise ValueError(f"Annotation session mismatch: {payload.get('session_id')} != {session_id}")
    episodes = []
    for episode in payload.get("episodes", []):
        if episode.get("status", "keep") != "keep":
            continue
        start = float(episode.get("start", 0.0))
        end = float(episode.get("end", 0.0))
        prompt = str(episode.get("prompt", "")).strip()
        if end > start and prompt:
            episodes.append({"start": start, "end": end, "prompt": prompt})
    if not episodes:
        raise ValueError("No kept episodes with non-empty prompts found.")
    return sorted(episodes, key=lambda item: item["start"])


def episode_frame_count(episode: dict[str, Any], target_fps: int) -> int:
    return max(1, int(math.floor((episode["end"] - episode["start"]) * target_fps)))


def joint_valid_name(index: int) -> str:
    if index == 0:
        return "head"
    if index <= BODY_JOINT_COUNT:
        return f"body[{index - 1}]"
    left_end = 1 + BODY_JOINT_COUNT + HAND_JOINT_COUNT
    if index < left_end:
        return f"left_hand[{index - 1 - BODY_JOINT_COUNT}]"
    return f"right_hand[{index - left_end}]"


def episode_joint_validity_report(
    episode: dict[str, Any],
    episode_index: int,
    tracking: dict[str, Any],
    target_fps: int,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Check every exported pose sample using the same 77-D validity mask.

    A sample is valid only when all 77 poses in the nearest source tracking
    frame are finite. The check uses the exact same nearest-neighbor policy as
    the released low-dimensional fields.
    """
    source_times = np.asarray(tracking["times"], dtype=np.float64)
    source_valid = np.asarray(tracking["joint_valid"], dtype=np.float64)
    if len(source_times) == 0 or source_valid.shape != (len(source_times), JOINT_VALID_DIM):
        raise ValueError(
            "Invalid internal joint-valid array: "
            f"times={source_times.shape}, valid={source_valid.shape}"
        )

    frame_count = episode_frame_count(episode, target_fps)
    invalid_frame_count = 0
    invalid_joint_counts = np.zeros(JOINT_VALID_DIM, dtype=np.int64)
    hand_missing_frames = np.zeros(2, dtype=np.int64)
    hand_empty_frames = np.zeros(2, dtype=np.int64)
    first_invalid_time: float | None = None

    for offset in range(0, frame_count, chunk_size):
        count = min(chunk_size, frame_count - offset)
        frame_indices = np.arange(offset, offset + count, dtype=np.float64)
        target_times = float(episode["start"]) + frame_indices / target_fps

        source_indices = nearest_time_indices(source_times, target_times)
        invalid = source_valid[source_indices] < 0.5
        invalid_rows = np.any(invalid, axis=1)
        invalid_frame_count += int(np.count_nonzero(invalid_rows))
        invalid_joint_counts += invalid.sum(axis=0, dtype=np.int64)
        hand_invalid = invalid[:, 1 + BODY_JOINT_COUNT:].reshape(count, 2, HAND_JOINT_COUNT)
        hand_missing_frames += hand_invalid.any(axis=2).sum(axis=0)
        hand_empty_frames += hand_invalid.all(axis=2).sum(axis=0)
        if first_invalid_time is None and np.any(invalid_rows):
            first_invalid_time = float(target_times[int(np.flatnonzero(invalid_rows)[0])])

    invalid_joints = [
        {
            "index": int(index),
            "name": joint_valid_name(int(index)),
            "invalid_frames": int(invalid_joint_counts[index]),
        }
        for index in np.flatnonzero(invalid_joint_counts)
    ]
    return {
        "annotation_index": episode_index,
        "start_sec": float(episode["start"]),
        "end_sec": float(episode["end"]),
        "task": episode["prompt"],
        "sampled_frames": frame_count,
        "all_joint_poses_finite": invalid_frame_count == 0,
        "invalid_frames": invalid_frame_count,
        "first_invalid_time_sec": first_invalid_time,
        "invalid_joints": invalid_joints,
        "hand_missing_frames": hand_missing_frames.tolist(),
        "hand_empty_frames": hand_empty_frames.tolist(),
    }


def filter_episodes_by_joint_validity(
    episodes: list[dict[str, Any]],
    tracking: dict[str, Any],
    target_fps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        report = episode_joint_validity_report(episode, index, tracking, target_fps)
        kept.append(episode)
        if report["all_joint_poses_finite"]:
            continue
        invalid.append(report)
        names = ", ".join(item["name"] for item in report["invalid_joints"][:8])
        if len(report["invalid_joints"]) > 8:
            names += ", ..."
        sys.stderr.write(
            "WARNING: keeping annotation episode with invalid poses "
            f"{index} [{episode['start']:.3f}, {episode['end']:.3f}] "
            f"({report['invalid_frames']}/{report['sampled_frames']} sampled frames; "
            f"invalid joints: {names}). Downstream repair is required.\n"
        )
    emit_progress(
        stage="quality_checked",
        episodes_checked=len(episodes),
        episodes_kept=len(kept),
        episodes_with_invalid_poses=len(invalid),
        episodes_dropped=0,
    )
    return kept, invalid


def hand_export_warning(
    invalid_episodes: list[dict[str, Any]], total_frames: int
) -> dict[str, Any] | None:
    """Describe missing *source* observations on the exact export timeline."""
    affected = [ep for ep in invalid_episodes if any(ep["hand_missing_frames"])]
    if not affected:
        return None
    missing = np.sum([ep["hand_missing_frames"] for ep in affected], axis=0).tolist()
    empty = np.sum([ep["hand_empty_frames"] for ep in affected], axis=0).tolist()
    details = "; ".join(
        f"{name}: missing observations in {missing[side]}/{total_frames} frames, "
        f"including {empty[side]} frames with no data for the entire hand"
        for side, name in enumerate(("Left hand", "Right hand")) if missing[side]
    )
    return {
        "code": "missing_hand_data",
        "episodes_affected": len(affected),
        "total_frames": total_frames,
        "missing_frames": missing,
        "empty_frames": empty,
        "message": (
            f"Warning: {len(affected)} episodes have missing hand observations.\n{details}.\n"
            "Continuing will fill missing hand poses with zeros; eligible short tracking gaps "
            "keep the last valid pose. Missing observations retain validity=0; existing hand data is preserved."
        ),
    }


def image_features(height: int, width: int, stereo: bool, mono_key: str) -> dict[str, dict[str, Any]]:
    img_names = ["height", "width", "channel"]
    if stereo:
        half = width // 2
        return {
            IMAGE_KEY_STEREO_LEFT: {"dtype": "video", "shape": (height, half, 3), "names": img_names},
            IMAGE_KEY_STEREO_RIGHT: {"dtype": "video", "shape": (height, half, 3), "names": img_names},
        }
    return {mono_key: {"dtype": "video", "shape": (height, width, 3), "names": img_names}}


def build_features(height: int, width: int, stereo: bool, mono_key: str) -> dict[str, dict[str, Any]]:
    f32 = lambda shape, names=None: {  # noqa: E731
        "dtype": "float32",
        "shape": shape,
        **({"names": names} if names else {}),
    }
    return {
        **image_features(height, width, stereo, mono_key),
        "observation.pico_head_pose": f32((HEAD_POSE_DIM,)),
        "observation.pico_body_pose": f32((PICO_BODY_POSE_DIM,)),
        "observation.pico_hands26_pose": f32((PICO_HANDS26_POSE_DIM,)),
        "observation.g1_qpos": f32((G1_QPOS_DIM,)),
        "observation.pico_hand_active": f32((2,), ["left", "right"]),
        "observation.pico_joint_valid": f32((JOINT_VALID_DIM,), JOINT_VALID_NAMES),
    }


def split_stereo(frame: np.ndarray, stereo: bool, mono_key: str) -> dict[str, np.ndarray]:
    """Map a decoded RGB frame to the image feature payload."""
    if not stereo:
        return {mono_key: frame}
    half = frame.shape[1] // 2
    return {
        IMAGE_KEY_STEREO_LEFT: np.ascontiguousarray(frame[:, :half]),
        IMAGE_KEY_STEREO_RIGHT: np.ascontiguousarray(frame[:, half : half * 2]),
    }


def sample_observation(t: float, tracking: dict[str, Any], g1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    head = sample_pose_block(tracking["times"], tracking["head_pose"], t, 1)
    body = sample_pose_block(tracking["times"], tracking["body_pose"], t, BODY_JOINT_COUNT)
    hands = sample_pose_block(tracking["times"], tracking["hands26_pose"], t, HANDS_JOINT_COUNT)
    active = sample_nearest_by_time(tracking["times"], tracking["hand_active"], t)
    joint_valid = sample_nearest_by_time(tracking["times"], tracking["joint_valid"], t)
    g1_qpos = sample_nearest_by_time(g1["times"], g1["qpos"], t, normalize_slice=slice(3, 7))
    return {
        "observation.pico_head_pose": head,
        "observation.pico_body_pose": body,
        "observation.pico_hands26_pose": hands,
        "observation.g1_qpos": g1_qpos,
        "observation.pico_hand_active": active.astype(np.float32),
        "observation.pico_joint_valid": joint_valid.astype(np.float32),
    }


def prepare_export_frame(
    t: float,
    rgb_frame: np.ndarray,
    stereo: bool,
    mono_key: str,
    tracking: dict[str, Any],
    g1: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """Build one export frame without touching the shared LeRobot writer.

    This function is deliberately separate from ``dataset.add_frame``. The
    arrays in ``tracking`` and ``g1`` are read-only during export, so this
    preparation work can safely run in worker threads while the main thread
    preserves the original frame order for the LeRobot dataset writer.
    """
    return {
        **split_stereo(rgb_frame, stereo, mono_key),
        **sample_observation(t, tracking, g1),
        "task": task,
    }


def default_intermediate_dir(session_id: str, name: str, legacy_name: str) -> Path:
    path = TMP_ROOT / session_id / name
    if path.exists():
        return path
    legacy = PROJECT_ROOT / "data" / legacy_name
    return legacy if legacy.exists() else path


def default_g1_dir(session_id: str) -> Path:
    return default_intermediate_dir(session_id, "retargeted", "retargeted")


def write_sidecar_metadata(output_dir: Path, summary: dict[str, Any]) -> Path:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sidecar = meta_dir / "pico_ego_release.json"
    sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return sidecar


def remove_empty_image_dirs(dataset_root: Path) -> int:
    """Remove LeRobot's empty image-encoding directories after video finalization.

    ``Path.rmdir`` only removes empty directories, so unexpected image files are
    never discarded. Walking deepest-first also lets the top-level ``images``
    directory disappear when the entire temporary tree is empty.
    """
    images_root = dataset_root / "images"
    if not images_root.is_dir() or images_root.is_symlink():
        return 0

    directories = [
        path
        for path in images_root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    ]
    directories.append(images_root)
    removed = 0
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


def assert_safe_output_dir(output_dir: Path) -> None:
    """Refuse to treat a dangerous path as a dataset output (we rmtree it)."""
    resolved = output_dir.resolve()
    unsafe = {Path("/").resolve(), Path.home().resolve(), PROJECT_ROOT.resolve(), Path.cwd().resolve()}
    if resolved in unsafe or resolved.parent == resolved:
        raise SystemExit(f"Refusing to use unsafe output dir: {resolved}")
    # Must live under the project's lerobot output area, or an explicit abs path
    # that is clearly a leaf (not an ancestor of the project).
    if PROJECT_ROOT.resolve() in resolved.parents:
        return
    if resolved in PROJECT_ROOT.resolve().parents:
        raise SystemExit(f"Refusing output dir that contains the project root: {resolved}")


def stereo_camera_block(
    camera_meta: dict[str, Any], width: int, height: int, stereo: bool
) -> dict[str, Any] | None:
    """Describe the stereo split geometry HONESTLY.

    The device reports a single ``cameraIntrinsics`` = [cx, cy, fx, fy] whose
    principal point sits at the FULL side-by-side texture centre (cx≈W/2,
    cy≈H/2) -- i.e. it describes the whole WxH texture, not one eye. We do NOT
    fabricate confident per-eye intrinsics; instead we record the raw values,
    the mechanical per-eye split, a clearly-labelled per-eye derivation, and the
    measured baseline, and flag every unverified assumption for the capture rig.
    """
    if not stereo:
        return None
    intr = camera_meta.get("camera_intrinsics") or []
    extr = camera_meta.get("camera_extrinsics") or []
    cx, cy, fx, fy = (intr + [None] * 4)[:4]
    half = width // 2

    baseline = None
    if len(extr) >= 2:
        t0 = np.asarray(extr[0], dtype=np.float64)[:3, 3]
        t1 = np.asarray(extr[1], dtype=np.float64)[:3, 3]
        vec = (t1 - t0)
        axis = int(np.argmax(np.abs(vec)))
        baseline = {
            "vector_extrinsic_frame": vec.tolist(),
            "norm_mm": float(np.linalg.norm(vec) * 1000.0),
            "dominant_axis": "xyz"[axis],
            "note": "eyes are separated along the extrinsics' local "
            f"{'xyz'[axis]} axis; norm ≈ IPD.",
        }

    derived_per_eye = None
    if None not in (cx, cy, fx, fy):
        # IF (unverified) the reported intrinsics apply per-eye rendered into
        # each half, the local principal point is (cx - x0, cy). With cx≈W/2 this
        # lands at the seam edge of each half -> almost certainly means the
        # intrinsics are full-frame, not per-eye. Emit both, flagged.
        derived_per_eye = {
            IMAGE_KEY_STEREO_LEFT: {"fx": fx, "fy": fy, "cx_local": cx - 0, "cy": cy},
            IMAGE_KEY_STEREO_RIGHT: {"fx": fx, "fy": fy, "cx_local": cx - half, "cy": cy},
            "warning": "UNVERIFIED. Derived by shifting cx by the split offset; "
            "only valid if cameraIntrinsics are per-eye. cx≈W/2 suggests they are "
            "full-frame instead — confirm against the Pico capture SDK.",
        }

    return {
        "raw_intrinsics_cx_cy_fx_fy": intr,
        "intrinsics_frame": (
            f"Describes the FULL {width}x{height} side-by-side texture "
            f"(cx≈{width/2:.1f}=W/2, cy≈{height/2:.1f}=H/2), NOT a single eye. "
            "Do not attach as-is to a split half without adjusting the principal point."
        ),
        "split": {
            IMAGE_KEY_STEREO_LEFT: {"cols": [0, half], "size": [half, height]},
            IMAGE_KEY_STEREO_RIGHT: {"cols": [half, half * 2], "size": [half, height]},
        },
        "extrinsics": {
            IMAGE_KEY_STEREO_LEFT: extr[0] if len(extr) >= 1 else None,
            IMAGE_KEY_STEREO_RIGHT: extr[1] if len(extr) >= 2 else None,
        },
        "baseline": baseline,
        "derived_per_eye_intrinsics": derived_per_eye,
        "eye_assignment": {
            "assumption": "video-left-half = extrinsics[0] = left eye",
            "status": "UNVERIFIED — confirm left/right order, extrinsic direction, "
            "and per-eye intrinsics against the capture rig before publishing.",
        },
        "note": "true per-eye camera pose = head_pose ∘ extrinsic (HMD pose, not camera).",
    }


def release_metadata(
    session_id: str,
    tracking: dict[str, Any],
    g1: dict[str, Any],
    video_fps: float,
    target_fps: int,
    episodes: list[dict[str, Any]],
    invalid_episodes: list[dict[str, Any]],
    stereo: bool,
    video_codec: str,
    video_width: int,
    video_height: int,
    allow_missing_hands: bool = False,
) -> dict[str, Any]:
    camera_meta = tracking.get("camera_meta", {}) or {}
    stereo_extrinsics = stereo_camera_block(camera_meta, video_width, video_height, stereo)
    return {
        "session_id": session_id,
        "schema_version": SCHEMA_VERSION,
        "schema_status": SCHEMA_STATUS,
        "fps": target_fps,
        "stereo": stereo,
        "video_codec": video_codec,
        "coordinate_convention": {
            "raw_channels": "Pico native frame (Y-up), quaternion order xyzw",
            "g1_qpos": "GMR/Unitree G1 frame (Z-up): root_xyz(3) + quat_xyzw(4) + 29 dof",
        },
        "alignment": {
            "reference_clock": "camera first-frame timeStampNs (metadata line)",
            "camera_t0_ns": tracking.get("camera_t0_ns"),
            "tracking_first_data_offset_sec": tracking.get("first_data_offset_sec"),
            "video_fps": video_fps,
            "tracking_fps_estimate": estimate_rate_hz(tracking["times"]),
            "g1_time_source": g1["time_source"],
            "g1_fps_estimate": estimate_rate_hz(g1["times"]),
        },
        "sampling_policy": {
            "method": "nearest_neighbor_by_timestamp",
            "applies_to": ["video", "pico_head", "pico_body", "pico_hand_active", "g1_qpos"],
            "tie_break": "earlier source frame",
            "interpolation": "previous-value hold for short hand gaps only",
            "hand_gap_fill_max_sec": MAX_HAND_FILL_GAP_SEC,
        },
        "camera": camera_meta,
        "stereo_camera": stereo_extrinsics,
        "head_pose_note": (
            "observation.pico_head_pose is the HMD pose, NOT the camera pose. "
            "True per-eye camera pose = head_pose composed with the matching "
            "stereo_camera.extrinsics 4x4."
        ),
        "episodes": [
            {"episode_index": i, "start_sec": ep["start"], "end_sec": ep["end"], "task": ep["prompt"]}
            for i, ep in enumerate(episodes)
        ],
        "episode_quality_policy": {
            "criterion": "At every exported timestamp, all 77 poses in the nearest source "
            "tracking frame must contain finite xyz+quat values.",
            "internal_joint_order": JOINT_VALID_ORDER,
            "on_failure": "Warn, keep the episode, and export pico_joint_valid for downstream repair.",
            "invalid_pose_storage": "Short eligible hand gaps use the previous valid pose; "
            "other missing hand poses use all-zero xyz+quaternion sentinels (not rotations). "
            "Head/body placeholders use zero position + identity quaternion. "
            "The mask remains zero for synthesized values; finite source hands are preserved.",
            "missing_hands_confirmed": allow_missing_hands,
            "hand_active_note": "pico_hand_active is independent of finite-value validation "
            "and remains an exported per-frame feature.",
        },
        "quality_control": {
            "episodes_checked": len(episodes),
            "episodes_kept": len(episodes),
            "episodes_dropped": 0,
            "episodes_with_invalid_poses": len(invalid_episodes),
            "invalid_episodes": invalid_episodes,
        },
        "action": "none: pure-observation dataset. Downstream builds its own action (e.g. next-frame observation.g1_qpos, or a delta).",
        "notes": "Hands are raw Pico26 (not retargeted). Retarget hands per downstream robot.",
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    tracking_path, video_path = discover_session_files(args.raw_dir, args.session)
    requested_annotations = load_annotations(args.annotations, args.session)
    width, height, video_fps, video_frames = video_info(video_path)
    if video_fps <= 0:
        video_fps = float(args.fps)
    target_fps = int(args.fps)
    stereo = not args.mono
    features = build_features(height, width, stereo, args.image_key)

    tracking = load_tracking_arrays(tracking_path)
    g1_dir = args.g1_dir or default_g1_dir(args.session)
    g1 = load_g1(
        g1_dir / f"{args.session}_unitree_g1.pkl",
        tracking.get("camera_t0_ns"),
    )
    annotations, invalid_episodes = filter_episodes_by_joint_validity(
        requested_annotations, tracking, target_fps
    )
    episode_lengths = [episode_frame_count(ep, target_fps) for ep in annotations]
    hand_warning = hand_export_warning(invalid_episodes, sum(episode_lengths))

    summary: dict[str, Any] = {
        "session_id": args.session,
        "schema_version": SCHEMA_VERSION,
        "schema_status": SCHEMA_STATUS,
        "tracking_path": str(tracking_path),
        "video_path": str(video_path),
        "video_width": width,
        "video_height": height,
        "video_fps": video_fps,
        "video_frames": video_frames,
        "target_fps": target_fps,
        "repo_id": args.repo_id,
        "output_dir": str(args.output_dir),
        "episodes_requested": len(requested_annotations),
        "episodes": len(annotations),
        "episodes_dropped_invalid_joints": 0,
        "episodes_with_invalid_joints": len(invalid_episodes),
        "invalid_episodes": invalid_episodes,
        "frames": int(sum(episode_lengths)),
        "sampling": "nearest_neighbor_for_all_modalities_by_timestamp",
        "action": "none (pure observation; downstream defines action from observation.g1_qpos)",
        "features": list(features.keys()),
        "warnings": [hand_warning] if hand_warning else [],
        "requires_hand_confirmation": hand_warning is not None,
        "missing_hands_confirmed": bool(getattr(args, "allow_missing_hands", False)),
    }
    summary["tracking_frames"] = int(len(tracking["times"]))
    summary["tracking_duration_sec"] = float(tracking["times"][-1]) if len(tracking["times"]) else 0.0
    summary["tracking_fps_estimate"] = estimate_rate_hz(tracking["times"])
    summary["g1_fps_estimate"] = estimate_rate_hz(g1["times"])
    summary["g1_time_source"] = g1["time_source"]
    summary["camera_t0_ns"] = tracking.get("camera_t0_ns")
    summary["av_offset_sec"] = tracking.get("first_data_offset_sec")
    summary["no_valid_episodes"] = not annotations

    if args.dry_run:
        # Hard-validate that every episode is covered by video, tracking and G1,
        # with a tight one-frame tolerance (not the old loose +1s slack).
        tol = 1.0 / target_fps
        video_dur = (video_frames / video_fps) if video_fps else None
        tracking_dur = float(tracking["times"][-1]) if len(tracking["times"]) else 0.0
        g1_dur = float(g1["times"][-1]) if len(g1["times"]) else 0.0
        out_of_range = []
        for i, ep in enumerate(annotations):
            reasons = []
            if ep["start"] < -tol:
                reasons.append("start<0")
            if video_dur is not None and ep["end"] > video_dur + tol:
                reasons.append(f"end>{video_dur:.3f}s video")
            if ep["end"] > tracking_dur + tol:
                reasons.append(f"end>{tracking_dur:.3f}s tracking")
            if ep["end"] > g1_dur + tol:
                reasons.append(f"end>{g1_dur:.3f}s g1")
            if reasons:
                out_of_range.append({"episode": i, "start": ep["start"], "end": ep["end"], "reasons": reasons})
        summary["video_duration_sec"] = video_dur
        summary["max_episode_end_sec"] = max((ep["end"] for ep in annotations), default=None)
        summary["episode_out_of_range"] = bool(out_of_range)
        summary["out_of_range_detail"] = out_of_range
        return summary

    if not annotations:
        raise SystemExit("No annotated episodes with non-empty prompts found.")

    if hand_warning and not getattr(args, "allow_missing_hands", False):
        raise SystemExit(
            hand_warning["message"] + "\nConfirm in the web interface, or pass --allow-missing-hands on the command line."
        )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "LeRobot is not installed in this environment. "
            "Run it under the `lerobot` env (see README) before exporting."
        ) from exc

    # Refuse dangerous output paths (we rmtree the target on --overwrite) and
    # build into a staging dir so a failed export never destroys a good one.
    assert_safe_output_dir(args.output_dir)
    final_dir = args.output_dir
    if final_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"Output dir already exists: {final_dir}. "
            "Pass --overwrite to replace it (or redo the session first)."
        )
    staging = final_dir.parent / (final_dir.name + ".exporting")
    if staging.exists():
        shutil.rmtree(staging)

    create_kwargs: dict[str, Any] = {
        "repo_id": args.repo_id,
        "fps": target_fps,
        "features": features,
        "root": staging,
        "robot_type": args.robot_type,
        "use_videos": True,
        "batch_encoding_size": args.batch_encoding_size,
        "image_writer_processes": args.image_writer_processes,
        "image_writer_threads": args.image_writer_threads,
    }
    create_params = inspect.signature(LeRobotDataset.create).parameters
    if "video_files_size_in_mb" in create_params:
        create_kwargs["video_files_size_in_mb"] = args.video_files_size_mb
    if "data_files_size_in_mb" in create_params:
        create_kwargs["data_files_size_in_mb"] = args.data_files_size_mb
    if "vcodec" in create_params:
        create_kwargs["vcodec"] = args.video_codec
    if "encoder_threads" in create_params:
        create_kwargs["encoder_threads"] = args.encoder_threads
    os.environ["PICO_EXPORT_VIDEO_PRESET"] = args.video_preset
    os.environ["PICO_EXPORT_VIDEO_GOP"] = str(args.video_gop)
    os.environ["PICO_EXPORT_VIDEO_CRF"] = str(args.video_crf)
    install_video_encoder_worker()
    dataset = LeRobotDataset.create(**create_kwargs)

    target_total = int(sum(episode_lengths))
    if args.max_frames is not None:
        target_total = min(target_total, args.max_frames)
    emit_progress(stage="start", written=0, total=target_total, episodes=len(annotations))

    reader = SequentialVideoReader(video_path, video_fps)
    total_written = 0
    preprocess_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.preprocess_workers,
        thread_name_prefix="pico-export-preprocess",
    )
    # Keep only a small window of prepared frames in memory. The oldest future
    # is always written first, so parallel preparation cannot reorder frames.
    max_pending = max(1, args.preprocess_workers * 2)
    try:
        for ep_index, (episode, length) in enumerate(zip(annotations, episode_lengths)):
            pending: deque[concurrent.futures.Future[dict[str, Any]]] = deque()
            for i in range(length):
                if args.max_frames is not None and total_written >= args.max_frames:
                    break
                t = episode["start"] + i / target_fps
                pending.append(
                    preprocess_pool.submit(
                        prepare_export_frame,
                        t,
                        reader.frame_at_time(t),
                        stereo,
                        args.image_key,
                        tracking,
                        g1,
                        episode["prompt"],
                    )
                )
                if len(pending) >= max_pending:
                    dataset.add_frame(pending.popleft().result())
                    total_written += 1
                    if total_written % 30 == 0:
                        emit_progress(
                            stage="encode",
                            written=total_written,
                            total=target_total,
                            episode=ep_index,
                        )
            while pending:
                dataset.add_frame(pending.popleft().result())
                total_written += 1
                if total_written % 30 == 0:
                    emit_progress(
                        stage="encode",
                        written=total_written,
                        total=target_total,
                        episode=ep_index,
                    )
            dataset.save_episode()
            emit_progress(
                stage="episode_done",
                written=total_written,
                total=target_total,
                episode=ep_index,
                episodes=len(annotations),
            )
            if args.max_frames is not None and total_written >= args.max_frames:
                break
    finally:
        # Wait for any running preparation task and cancel tasks that have not
        # started. This prevents worker exceptions from being hidden while
        # keeping the existing staging/finalize behavior intact.
        preprocess_pool.shutdown(wait=True, cancel_futures=True)
        reader.release()
        dataset.finalize()

    # Sidecar (incl. episode start/end/prompt so the slicing stays reproducible
    # even after the tmp annotation dir is cleaned) goes into staging, then we
    # atomically swap staging into place.
    write_sidecar_metadata(
        staging,
        release_metadata(
            args.session, tracking, g1, video_fps, target_fps, annotations, invalid_episodes, stereo,
            args.video_codec, width, height,
            allow_missing_hands=bool(getattr(args, "allow_missing_hands", False)),
        ),
    )
    empty_image_dirs_removed = remove_empty_image_dirs(staging)
    if empty_image_dirs_removed:
        sys.stderr.write(
            f"Cleaned {empty_image_dirs_removed} empty intermediate image directories.\n"
        )
    emit_progress(
        stage="cleanup",
        written=total_written,
        total=target_total,
        empty_image_dirs_removed=empty_image_dirs_removed,
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    staging.rename(final_dir)
    emit_progress(stage="done", written=total_written, total=target_total)

    summary["written_frames"] = total_written
    summary["stereo"] = stereo
    summary["video_codec"] = args.video_codec
    summary["preprocess_workers"] = args.preprocess_workers
    summary["image_writer_threads"] = args.image_writer_threads
    summary["image_writer_processes"] = args.image_writer_processes
    summary["video_preset"] = args.video_preset
    summary["encoder_threads"] = args.encoder_threads
    summary["features"] = list(features.keys())
    summary["sidecar_meta"] = str(final_dir / "meta" / "pico_ego_release.json")
    summary["empty_image_dirs_removed"] = empty_image_dirs_removed
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--g1-dir", type=Path, default=None)
    parser.add_argument("--wuji-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-missing-hands", action="store_true",
        help="Confirm export with missing hand observations (zero sentinels, validity remains 0).",
    )
    parser.add_argument("--image-key", default=IMAGE_KEY_MONO, help="Feature key when --mono is set.")
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Store the full combined frame instead of splitting the stereo pair into left/right.",
    )
    parser.add_argument("--robot-type", default="unitree_g1")
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=int(os.environ.get("PICO_EXPORT_PREPROCESS_WORKERS", "4")),
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=int(os.environ.get("PICO_EXPORT_IMAGE_WRITER_THREADS", "8")),
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=int(os.environ.get("PICO_EXPORT_IMAGE_WRITER_PROCESSES", "0")),
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=int(os.environ.get("PICO_EXPORT_ENCODER_THREADS", "8")),
    )
    parser.add_argument(
        "--video-preset",
        default=os.environ.get("PICO_EXPORT_VIDEO_PRESET", "veryfast"),
        help="FFmpeg preset for software H.264/HEVC (default: veryfast).",
    )
    parser.add_argument(
        "--video-gop",
        type=int,
        default=int(os.environ.get("PICO_EXPORT_VIDEO_GOP", "2")),
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=int(os.environ.get("PICO_EXPORT_VIDEO_CRF", "30")),
    )
    parser.add_argument("--video-files-size-mb", type=int, default=512)
    parser.add_argument("--data-files-size-mb", type=int, default=256)
    parser.add_argument(
        "--video-codec",
        default="h264",
        choices=["h264", "hevc", "libsvtav1", "auto"],
        help="h264 = most compatible decode everywhere (default); libsvtav1 = smaller AV1 but needs a dav1d decoder.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dir before export.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.preprocess_workers <= 0:
        raise SystemExit("--preprocess-workers must be positive")
    if args.image_writer_threads <= 0:
        raise SystemExit("--image-writer-threads must be positive")
    if args.image_writer_processes < 0:
        raise SystemExit("--image-writer-processes must be non-negative")
    if args.encoder_threads <= 0:
        raise SystemExit("--encoder-threads must be positive")
    if args.video_gop <= 0:
        raise SystemExit("--video-gop must be positive")
    if not 0 <= args.video_crf <= 51:
        raise SystemExit("--video-crf must be between 0 and 51")
    if args.batch_encoding_size <= 0:
        raise SystemExit("--batch-encoding-size must be positive")
    summary = export_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    # "Check" (dry-run) must FAIL loudly when episodes fall outside coverage, so
    # the web UI blocks a real export that would silently clamp/duplicate frames.
    if args.dry_run and (summary.get("episode_out_of_range") or summary.get("no_valid_episodes")):
        sys.exit(1)


if __name__ == "__main__":
    main()
