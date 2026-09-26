#!/usr/bin/env python3
"""Analyze Pico egocentric tracking JSONL files and paired videos."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pico_ego_collector/matplotlib")
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACKING_GLOB = "trackingData_*.txt"
VIDEO_GLOB = "CameraRecord_*.mp4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrackingParseResult:
    frames: pd.DataFrame
    metadata: dict[str, Any]
    total_lines: int
    frame_lines: int
    parse_errors: int


def parse_float_list(value: Any, expected: int | None = None) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return []

    floats: list[float] = []
    for part in parts:
        try:
            floats.append(float(part))
        except (TypeError, ValueError):
            floats.append(math.nan)

    if expected is not None and len(floats) < expected:
        floats.extend([math.nan] * (expected - len(floats)))
    return floats[:expected] if expected is not None else floats


def parse_literal_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)):
        return []
    out: list[float] = []
    for item in parsed:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(math.nan)
    return out


def parse_joint_pose(joint: dict[str, Any] | None) -> list[float]:
    if not joint:
        return [math.nan] * 7
    return parse_float_list(joint.get("p"), expected=7)


def parse_tracking(path: Path) -> TrackingParseResult:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    total_lines = 0
    frame_lines = 0
    parse_errors = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines = line_no
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            if "notice" in record or "cameraIntrinsics" in record:
                metadata.update(
                    {
                        "notice": record.get("notice", metadata.get("notice", "")),
                        "first_image_timestamp_ns": record.get("timeStampNs"),
                        "camera_intrinsics": parse_literal_list(
                            record.get("cameraIntrinsics")
                        ),
                        "camera_extrinsics_raw": record.get("cameraExtrinsics", ""),
                    }
                )
                continue

            frame_lines += 1
            head_pose = parse_float_list(record.get("Head", {}).get("pose"), expected=7)
            hand = record.get("Hand", {}) or {}
            left = hand.get("leftHand", {}) or {}
            right = hand.get("rightHand", {}) or {}
            body = record.get("Body", {}) or {}

            left_joints = left.get("HandJointLocations") or []
            right_joints = right.get("HandJointLocations") or []
            body_joints = body.get("joints") or []
            left_root = parse_joint_pose(left_joints[0] if left_joints else None)
            right_root = parse_joint_pose(right_joints[0] if right_joints else None)
            body_root = parse_joint_pose(body_joints[0] if body_joints else None)

            rows.append(
                {
                    "line_no": line_no,
                    "timeStampNs": pd.to_numeric(
                        record.get("timeStampNs"), errors="coerce"
                    ),
                    "predictTime": pd.to_numeric(
                        record.get("predictTime"), errors="coerce"
                    ),
                    "input": record.get("Input"),
                    "app_focus": bool(record.get("appState", {}).get("focus", False)),
                    "head_status": record.get("Head", {}).get("status"),
                    "head_x": head_pose[0],
                    "head_y": head_pose[1],
                    "head_z": head_pose[2],
                    "head_qx": head_pose[3],
                    "head_qy": head_pose[4],
                    "head_qz": head_pose[5],
                    "head_qw": head_pose[6],
                    "left_active": int(left.get("isActive", 0) or 0),
                    "right_active": int(right.get("isActive", 0) or 0),
                    "left_joint_count": len(left_joints),
                    "right_joint_count": len(right_joints),
                    "body_joint_count": len(body_joints),
                    "left_root_x": left_root[0],
                    "left_root_y": left_root[1],
                    "left_root_z": left_root[2],
                    "right_root_x": right_root[0],
                    "right_root_y": right_root[1],
                    "right_root_z": right_root[2],
                    "body_root_x": body_root[0],
                    "body_root_y": body_root[1],
                    "body_root_z": body_root[2],
                }
            )

    frames = pd.DataFrame(rows)
    if not frames.empty:
        frames["timeStampNs"] = pd.to_numeric(frames["timeStampNs"], errors="coerce")
        frames["t_sec"] = (frames["timeStampNs"] - frames["timeStampNs"].iloc[0]) / 1e9
        frames["frame_index"] = np.arange(len(frames))
    return TrackingParseResult(frames, metadata, total_lines, frame_lines, parse_errors)


def path_length_m(frames: pd.DataFrame, columns: list[str]) -> float:
    valid = frames[columns].dropna()
    if len(valid) < 2:
        return math.nan
    diff = np.diff(valid.to_numpy(dtype=float), axis=0)
    return float(np.linalg.norm(diff, axis=1).sum())


def summarize_tracking(result: TrackingParseResult) -> dict[str, Any]:
    frames = result.frames
    summary: dict[str, Any] = {
        "tracking_total_lines": result.total_lines,
        "tracking_frame_lines": result.frame_lines,
        "tracking_parse_errors": result.parse_errors,
    }
    if frames.empty:
        return summary

    ts = frames["timeStampNs"].to_numpy(dtype=float)
    dt_ms = np.diff(ts) / 1e6
    duration_sec = float((ts[-1] - ts[0]) / 1e9) if len(ts) > 1 else 0.0
    median_dt = float(np.nanmedian(dt_ms)) if len(dt_ms) else math.nan
    gap_threshold_ms = max(50.0, median_dt * 3.0) if not math.isnan(median_dt) else 50.0
    gap_count = int(np.sum(dt_ms > gap_threshold_ms)) if len(dt_ms) else 0

    summary.update(
        {
            "tracking_frames": int(len(frames)),
            "tracking_duration_sec": duration_sec,
            "tracking_rate_hz": float((len(frames) - 1) / duration_sec)
            if duration_sec > 0 and len(frames) > 1
            else math.nan,
            "timestamp_dt_median_ms": median_dt,
            "timestamp_dt_p95_ms": float(np.nanpercentile(dt_ms, 95))
            if len(dt_ms)
            else math.nan,
            "timestamp_dt_max_ms": float(np.nanmax(dt_ms)) if len(dt_ms) else math.nan,
            "timestamp_nonmonotonic_count": int(np.sum(dt_ms <= 0)) if len(dt_ms) else 0,
            "timestamp_gap_threshold_ms": gap_threshold_ms,
            "timestamp_gap_count": gap_count,
            "head_path_length_m": path_length_m(
                frames, ["head_x", "head_y", "head_z"]
            ),
            "body_root_path_length_m": path_length_m(
                frames, ["body_root_x", "body_root_y", "body_root_z"]
            ),
            "left_active_pct": float(frames["left_active"].mean() * 100.0),
            "right_active_pct": float(frames["right_active"].mean() * 100.0),
            "left_joint_count_median": float(frames["left_joint_count"].median()),
            "right_joint_count_median": float(frames["right_joint_count"].median()),
            "body_joint_count_median": float(frames["body_joint_count"].median()),
            "app_focus_pct": float(frames["app_focus"].mean() * 100.0),
        }
    )
    return summary


def video_metadata(path: Path, thumbnail_path: Path | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "video_exists": path.exists(),
        "video_bytes": path.stat().st_size if path.exists() else math.nan,
    }
    if not path.exists():
        return summary

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        summary["video_opened"] = False
        return summary

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = float(frame_count / fps) if fps > 0 else math.nan
    summary.update(
        {
            "video_opened": True,
            "video_width": width,
            "video_height": height,
            "video_fps": fps,
            "video_frames": frame_count,
            "video_duration_sec": duration,
        }
    )

    if thumbnail_path is not None:
        ok, frame = cap.read()
        if ok:
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(thumbnail_path), frame)
            summary["thumbnail"] = str(thumbnail_path)
    cap.release()
    return summary


def plot_session(session_id: str, frames: pd.DataFrame, output_path: Path) -> None:
    if frames.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ts = frames["timeStampNs"].to_numpy(dtype=float)
    dt_ms = np.diff(ts) / 1e6
    t_mid = frames["t_sec"].iloc[1:].to_numpy(dtype=float) if len(frames) > 1 else []

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(f"{session_id} tracking overview")

    if len(dt_ms):
        axes[0].plot(t_mid, dt_ms, linewidth=1.0)
        axes[0].axhline(np.nanmedian(dt_ms), color="tab:red", linestyle="--", lw=1)
    axes[0].set_ylabel("dt (ms)")
    axes[0].set_xlabel("time (s)")
    axes[0].grid(True, alpha=0.3)

    for col, label in [("head_x", "x"), ("head_y", "y"), ("head_z", "z")]:
        axes[1].plot(frames["t_sec"], frames[col], label=label, linewidth=1.0)
    axes[1].set_ylabel("head position (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames["t_sec"], frames["left_active"], label="left hand", lw=1.0)
    axes[2].plot(frames["t_sec"], frames["right_active"], label="right hand", lw=1.0)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_ylabel("active")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def discover_sessions(raw_dir: Path) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    search_dirs = [raw_dir]
    legacy = PROJECT_ROOT / "data" / "raw"
    if raw_dir == Path("raw") and legacy.exists():
        search_dirs.append(legacy)
    seen: set[str] = set()
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        for session_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
            if session_dir.name in seen:
                continue
            seen.add(session_dir.name)
            tracking_files = sorted(session_dir.glob(TRACKING_GLOB))
            video_files = sorted(session_dir.glob(VIDEO_GLOB))
            sessions.append(
                {
                    "session_id": session_dir.name,
                    "session_dir": session_dir,
                    "tracking_path": tracking_files[0] if tracking_files else None,
                    "video_path": video_files[0] if video_files else None,
                }
            )
    return sessions


def quality_notes(row: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if row.get("tracking_parse_errors", 0):
        notes.append(f"{row['tracking_parse_errors']} tracking JSON parse errors")
    if row.get("timestamp_nonmonotonic_count", 0):
        notes.append(f"{row['timestamp_nonmonotonic_count']} non-monotonic timestamps")
    if row.get("timestamp_gap_count", 0):
        notes.append(f"{row['timestamp_gap_count']} timestamp gaps")
    if row.get("video_opened") is False:
        notes.append("video could not be opened by OpenCV")
    delta = row.get("duration_delta_sec")
    if isinstance(delta, (int, float)) and not math.isnan(delta) and abs(delta) > 1.0:
        notes.append(f"tracking/video duration delta {delta:.2f}s")
    if isinstance(row.get("left_active_pct"), (int, float)) and row["left_active_pct"] < 5:
        notes.append("left hand mostly inactive")
    if isinstance(row.get("right_active_pct"), (int, float)) and row["right_active_pct"] < 5:
        notes.append("right hand mostly inactive")
    return notes or ["ok"]


def write_markdown_report(summary: pd.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Pico Ego Collector Data Quality Report",
        "",
        f"Generated sessions: {len(summary)}",
        "",
    ]
    if summary.empty:
        lines.append("No sessions found.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    cols = [
        "session_id",
        "tracking_frames",
        "tracking_duration_sec",
        "tracking_rate_hz",
        "video_duration_sec",
        "duration_delta_sec",
        "left_active_pct",
        "right_active_pct",
        "timestamp_gap_count",
    ]
    printable = summary[cols].copy()
    for col in printable.select_dtypes(include=[float]).columns:
        printable[col] = printable[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    lines.append(printable.to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for _, row_series in summary.iterrows():
        row = row_series.to_dict()
        notes = "; ".join(quality_notes(row))
        lines.append(f"- `{row['session_id']}`: {notes}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for _, row_series in summary.iterrows():
        sid = row_series["session_id"]
        lines.append(f"- `{sid}`: `reports/figures/{sid}_tracking_overview.png`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(raw_dir: Path, processed_dir: Path, reports_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    figures_dir = reports_dir / "figures"
    for session in discover_sessions(raw_dir):
        session_id = session["session_id"]
        tracking_path: Path | None = session["tracking_path"]
        video_path: Path | None = session["video_path"]

        row: dict[str, Any] = {
            "session_id": session_id,
            "session_dir": str(session["session_dir"]),
            "tracking_path": str(tracking_path or ""),
            "video_path": str(video_path or ""),
        }

        if tracking_path is not None:
            parsed = parse_tracking(tracking_path)
            row.update(summarize_tracking(parsed))
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_csv = processed_dir / f"{session_id}_frames.csv"
            parsed.frames.to_csv(processed_csv, index=False)
            row["processed_frames_csv"] = str(processed_csv)
            plot_session(
                session_id,
                parsed.frames,
                figures_dir / f"{session_id}_tracking_overview.png",
            )
            intrinsics = parsed.metadata.get("camera_intrinsics", [])
            row["camera_intrinsics"] = json.dumps(intrinsics)
        else:
            row["tracking_missing"] = True

        if video_path is not None:
            row.update(
                video_metadata(video_path, figures_dir / f"{session_id}_thumbnail.jpg")
            )
        else:
            row["video_missing"] = True

        if "tracking_duration_sec" in row and "video_duration_sec" in row:
            row["duration_delta_sec"] = (
                float(row["tracking_duration_sec"]) - float(row["video_duration_sec"])
            )
        else:
            row["duration_delta_sec"] = math.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports_dir / "dataset_summary.csv", index=False)
    summary.to_json(
        reports_dir / "dataset_summary.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )
    write_markdown_report(summary, reports_dir / "quality_report.md")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("/tmp/pico_ego_collector/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("/tmp/pico_ego_collector/reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args.raw_dir, args.processed_dir, args.reports_dir)
    print(f"sessions={len(summary)}")
    print(f"summary={args.reports_dir / 'dataset_summary.csv'}")
    print(f"report={args.reports_dir / 'quality_report.md'}")


if __name__ == "__main__":
    main()
