#!/usr/bin/env python3
"""Serve the local Pico episode annotator."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import mimetypes
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import cv2
import numpy as np

from sync_pico_download import normalize_device_group, normalize_processing_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Background job registry (async export / generate with progress polling)
# --------------------------------------------------------------------------- #
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_JOB_SEQ = itertools.count(1)
_PROGRESS_PREFIX = "@PROGRESS"
_JOB_STORE_ROOT = Path(os.environ.get("PICO_EGO_TMP", "/tmp/pico_ego_collector")) / "jobs"
_JOB_LOG_LIMIT = 24000

# Client hung up mid-response (browser cancelled a preview poll / video range
# request). Nothing left to send — swallow instead of crashing the handler.
_CLIENT_DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


def _job_update(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(fields)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            _persist_job(job)


def _persist_job(job: dict) -> None:
    """Persist compact task state so a browser refresh keeps the task list."""
    try:
        _JOB_STORE_ROOT.mkdir(parents=True, exist_ok=True)
        snapshot = dict(job)
        snapshot["stdout"] = str(snapshot.get("stdout", ""))[-_JOB_LOG_LIMIT:]
        snapshot["stderr"] = str(snapshot.get("stderr", ""))[-_JOB_LOG_LIMIT:]
        path = _JOB_STORE_ROOT / f"{job['job_id']}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # A task must continue even if the optional state persistence fails.
        pass


def load_persisted_jobs() -> None:
    """Load prior task records and mark interrupted running jobs explicitly."""
    try:
        _JOB_STORE_ROOT.mkdir(parents=True, exist_ok=True)
        records = []
        for path in _JOB_STORE_ROOT.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("status") == "running":
                    job.update(
                        status="interrupted",
                        ok=False,
                        returncode=None,
                        stderr=(str(job.get("stderr", "")) + "\nServer restarted; background job interrupted.\n").strip(),
                    )
                records.append(job)
            except (OSError, ValueError, TypeError):
                continue
        with _JOBS_LOCK:
            for job in records:
                _JOBS[job["job_id"]] = job
    except Exception:
        pass


def list_jobs(limit: int = 100) -> list[dict]:
    with _JOBS_LOCK:
        jobs = [dict(job) for job in _JOBS.values()]
    jobs.sort(key=lambda job: str(job.get("updated_at", job.get("created_at", ""))), reverse=True)
    return jobs[:limit]


def get_job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None


def _stream_into_job(job_id: str, cmd: list[str], cwd: Path, out_chunks: list[str], err_chunks: list[str]) -> int:
    """Run one command, streaming stdout/stderr into shared buffers with LIVE
    per-line job updates so the web terminal refreshes in real time.
    Returns the return code (-1 if the process failed to start)."""
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1
        )
    except Exception as exc:  # noqa: BLE001
        err_chunks.append(f"failed to start: {exc}\n")
        _job_update(job_id, stderr="".join(err_chunks))
        return -1

    def drain_stdout() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            out_chunks.append(line)
            _job_update(job_id, stdout="".join(out_chunks))

    def drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            if line.startswith(_PROGRESS_PREFIX):
                try:
                    _job_update(job_id, progress=json.loads(line[len(_PROGRESS_PREFIX):].strip()))
                    continue
                except Exception:  # noqa: BLE001
                    pass
            err_chunks.append(line)
            _job_update(job_id, stderr="".join(err_chunks))

    t_out = threading.Thread(target=drain_stdout, daemon=True)
    t_err = threading.Thread(target=drain_stderr, daemon=True)
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()
    proc.wait()
    return proc.returncode


def _finish_job(job_id: str, ok: bool, rc: int, out_chunks: list[str], err_chunks: list[str], on_success, **extra) -> None:
    _job_update(
        job_id,
        status="done" if ok else "error",
        ok=ok,
        returncode=rc,
        stdout="".join(out_chunks),
        stderr="".join(err_chunks),
        **extra,
    )
    if ok and on_success is not None:
        try:
            on_success()
        except Exception as exc:  # noqa: BLE001
            _job_update(job_id, cleanup_error=str(exc))


def _run_job(job_id: str, cmd: list[str], cwd: Path, on_success) -> None:
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    rc = _stream_into_job(job_id, cmd, cwd, out_chunks, err_chunks)
    _finish_job(job_id, rc == 0, rc, out_chunks, err_chunks, on_success)


def _run_steps_job(job_id: str, steps: list[tuple[str, list[str]]], cwd: Path, on_success) -> None:
    """Run several commands sequentially inside ONE job, streaming all output."""
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    for stage, cmd in steps:
        _job_update(job_id, progress={"stage": stage}, stage=stage)
        err_chunks.append(f"\n=== {stage} ===\n")
        _job_update(job_id, stderr="".join(err_chunks))
        rc = _stream_into_job(job_id, cmd, cwd, out_chunks, err_chunks)
        if rc != 0:
            _finish_job(job_id, False, rc, out_chunks, err_chunks, None, stage=stage)
            return
    _job_update(job_id, progress={"stage": "done"})
    _finish_job(job_id, True, 0, out_chunks, err_chunks, on_success, stage="done")


def _new_job(kind: str, session_id: str, command: object, extra: dict | None) -> str:
    job_id = f"{kind}-{session_id}-{time.time_ns()}-{next(_JOB_SEQ)}"
    now = datetime.now(timezone.utc).isoformat()
    job = {
        "job_id": job_id,
        "kind": kind,
        "session_id": session_id,
        "status": "running",
        "ok": None,
        "returncode": None,
        "progress": {},
        "stdout": "",
        "stderr": "",
        "command": command,
        "created_at": now,
        "updated_at": now,
    }
    if extra:
        job.update(extra)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _persist_job(job)
    return job_id


def start_job(kind: str, session_id: str, cmd: list[str], cwd: Path, on_success=None, extra: dict | None = None) -> dict:
    job_id = _new_job(kind, session_id, cmd, extra)
    threading.Thread(target=_run_job, args=(job_id, cmd, cwd, on_success), daemon=True).start()
    return get_job(job_id)


def start_steps_job(
    kind: str, session_id: str, steps: list[tuple[str, list[str]]], cwd: Path, on_success=None, extra: dict | None = None
) -> dict:
    job_id = _new_job(kind, session_id, [cmd for _, cmd in steps], extra)
    threading.Thread(target=_run_steps_job, args=(job_id, steps, cwd, on_success), daemon=True).start()
    return get_job(job_id)
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
MEDIAPIPE_HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_pose(value: object, dims: int = 7) -> np.ndarray:
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


def parse_joint_positions(joints: object) -> np.ndarray:
    if not isinstance(joints, list):
        return np.zeros((0, 3), dtype=np.float32)
    poses = [parse_pose(joint.get("p") if isinstance(joint, dict) else None)[:3] for joint in joints]
    return np.vstack(poses).astype(np.float32) if poses else np.zeros((0, 3), dtype=np.float32)


def transform_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    out = np.empty_like(points, dtype=np.float32)
    out[:, 0] = points[:, 0]
    out[:, 1] = -points[:, 2]
    out[:, 2] = points[:, 1]
    return out


def transform_point_sequence(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return arr
    transformed = transform_points(arr.reshape(-1, 3))
    return transformed.reshape(arr.shape)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def array_to_json(array: np.ndarray, digits: int = 5) -> list:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return [None if not np.isfinite(v) else round(float(v), digits) for v in arr]
    return [array_to_json(row, digits) for row in arr]


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def default_conda_bin() -> Path:
    env_value = os.environ.get("PICO_CONDA_BIN") or os.environ.get("CONDA_EXE")
    if env_value:
        return Path(env_value)
    found = shutil.which("conda")
    return Path(found) if found else Path("conda")


def default_tmp_root() -> Path:
    return Path(os.environ.get("PICO_EGO_TMP", "/tmp/pico_ego_collector"))


class AnnotatorState:
    def __init__(
        self,
        root: Path,
        export_python: Path | None = None,
        export_conda_env: str | None = None,
        conda_bin: Path | None = None,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
        tmp_root: Path | None = None,
    ) -> None:
        self.root = root
        self.raw_dir = self.resolve_project_path(raw_dir) if raw_dir else self.default_raw_dir()
        self.legacy_raw_dir = root / "data" / "raw"
        self.output_dir = self.resolve_project_path(output_dir) if output_dir else root / "lerobot_v3"
        self.tmp_root = Path(tmp_root) if tmp_root else default_tmp_root()
        self.web_dir = root / "web"
        self.export_python = Path(export_python) if export_python else None
        self.export_conda_env = export_conda_env
        self.conda_bin = Path(conda_bin) if conda_bin else default_conda_bin()

    def resolve_project_path(self, path: Path | None) -> Path:
        resolved = Path(path)
        return resolved if resolved.is_absolute() else self.root / resolved

    def default_raw_dir(self) -> Path:
        raw = self.root / "raw"
        if raw.exists():
            return raw
        legacy = self.root / "data" / "raw"
        return legacy if legacy.exists() else raw

    def work_dir(self, session_id: str) -> Path:
        return self.tmp_root / session_id

    def cleanup_work_dir(self, session_id: str) -> None:
        work_dir = self.work_dir(session_id)
        if work_dir.exists():
            shutil.rmtree(work_dir)

    def remove_path(self, path: Path) -> str | None:
        if not path.exists():
            return None
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return str(path)

    def session_generated_paths(self, session_id: str) -> list[Path]:
        """Every per-session GENERATED artifact.

        Scoped to this session only: never touches raw/ and never touches the
        shared cross-session aggregates (dataset_summary.*, quality_report.md),
        which belong to all sessions and are regenerated by analyze_dataset.py.
        """
        paths: list[Path] = [
            self.work_dir(session_id),                         # tmp/<session>/: annotations, preview_cache,
                                                               #   retargeted, wuji_retargeted, previews
            self.output_dir / session_id,                      # lerobot_v3/<session>/ (incl. sidecar meta)
            self.tmp_root / "processed" / f"{session_id}_frames.csv",
            self.tmp_root / "reports" / "figures" / f"{session_id}_tracking_overview.png",
            self.tmp_root / "reports" / "figures" / f"{session_id}_thumbnail.jpg",
        ]
        # Legacy per-session intermediate locations that session_paths() still falls back to.
        for sub in ("retargeted", "wuji_retargeted"):
            legacy_dir = self.root / "data" / sub
            if legacy_dir.exists():
                paths.extend(sorted(legacy_dir.glob(f"{session_id}_*")))
        for name in (f"{session_id}_tracking_overview.png", f"{session_id}_thumbnail.jpg"):
            paths.append(self.root / "reports" / "figures" / name)
        return paths

    def redo_session(self, session_id: str) -> dict:
        if not re.fullmatch(r"\d{8}_\d{6}", session_id):
            raise ValueError("session_id must look like YYYYMMDD_HHMMSS")
        paths = self.session_paths(session_id)
        session_dir = Path(paths["session_dir"])
        if not session_dir.exists():
            raise FileNotFoundError(f"Raw session not found: {session_dir}")

        removed = [
            removed
            for path in self.session_generated_paths(session_id)
            if (removed := self.remove_path(path))
        ]
        return {
            "ok": True,
            "session_id": session_id,
            "raw_kept": str(session_dir),
            "removed": removed,
            "sessions": self.discover_sessions(),
        }

    def delete_session(self, session_id: str) -> dict:
        """Delete one imported session and all generated artifacts."""
        if not re.fullmatch(r"\d{8}_\d{6}", session_id):
            raise ValueError("session_id must look like YYYYMMDD_HHMMSS")
        paths = self.session_paths(session_id)
        session_dir = Path(paths["session_dir"])
        if not session_dir.exists():
            raise FileNotFoundError(f"Raw session not found: {session_dir}")

        removed = []
        for path in self.session_generated_paths(session_id):
            if path.exists():
                removed_path = self.remove_path(path)
                if removed_path:
                    removed.append(removed_path)
        removed_path = self.remove_path(session_dir)
        if removed_path:
            removed.append(removed_path)
        return {"ok": True, "session_id": session_id, "removed": removed, "sessions": self.discover_sessions()}

    def export_config(self) -> dict:
        command = self.export_command([str(self.root / "scripts" / "export_lerobot_v3.py")])
        if self.export_python:
            mode = "python"
            label = str(self.export_python)
        elif self.export_conda_env:
            mode = "conda"
            label = f"conda:{self.export_conda_env}"
        else:
            mode = "server"
            label = sys.executable
        return {
            "mode": mode,
            "label": label,
            "server_python": sys.executable,
            "conda_bin": str(self.conda_bin) if self.conda_bin else None,
            "command_prefix": command[:-1],
        }

    def manifest_path(self) -> Path:
        # Each request must own its result, including after a failed import.
        return self.tmp_root / "imports" / f"{uuid4().hex}.csv"

    def processing_status_path(self) -> Path:
        return self.root / "pico_processing_status.csv"

    @staticmethod
    def group_from_file_name(file_name: str) -> str:
        match = re.match(r"^(group\d+)_", file_name)
        return match.group(1) if match else "ungrouped"

    def load_processing_status(self) -> dict[str, dict[str, str]]:
        path = self.processing_status_path()
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8", newline="") as f:
            return {
                row.get("file_name", ""): row | {
                    "status": normalize_processing_status(row.get("status", "unprocessed")),
                    "device_group": normalize_device_group(row.get("device_group", "ungrouped")),
                }
                for row in csv.DictReader(f)
                if row.get("file_name", "")
            }

    def prune_processing_status(self) -> None:
        rows = self.load_processing_status()
        local_video_names = {
            path.name for path in self.raw_dir.glob("*/*CameraRecord_*.mp4")
        }
        pruned = {name: row for name, row in rows.items() if name in local_video_names}
        needs_upgrade = any(
            "tracking_file" not in row
            or (
                self.group_from_file_name(name) != "ungrouped"
                and row.get("device_group") != self.group_from_file_name(name)
            )
            for name, row in pruned.items()
        )
        if pruned != rows or needs_upgrade:
            for name, row in pruned.items():
                row.setdefault("tracking_file", f"trackingData_{Path(name).stem.replace('CameraRecord_', '')}.txt")
                row["device_group"] = self.group_from_file_name(name) if self.group_from_file_name(name) != "ungrouped" else row.get("device_group", "ungrouped")
            self.save_processing_status(pruned)

    def save_processing_status(self, rows: dict[str, dict[str, str]]) -> None:
        path = self.processing_status_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file_name", "tracking_file", "device_group", "status"],
            )
            writer.writeheader()
            for file_name in sorted(rows):
                row = rows[file_name]
                writer.writerow({
                    "file_name": file_name,
                    "tracking_file": row.get("tracking_file", ""),
                    "device_group": normalize_device_group(row.get("device_group", "")),
                    "status": normalize_processing_status(row.get("status", "unprocessed")),
                })

    def ensure_processing_status(self, session_id: str, status: str = "unprocessed") -> str:
        paths = self.session_paths(session_id)
        video_name = paths["video"].name if paths["video"] else f"CameraRecord_{session_id}.mp4"
        tracking_name = paths["tracking"].name if paths["tracking"] else f"trackingData_{session_id}.txt"
        rows = self.load_processing_status()
        current = rows.get(video_name, {}).get("status")
        if current not in {"unprocessed", "processed"}:
            rows[video_name] = {
                "file_name": video_name,
                "tracking_file": tracking_name,
                "device_group": self.group_from_file_name(video_name),
                "status": status,
            }
            self.save_processing_status(rows)
            return status
        return current

    def mark_session_processed(self, session_id: str) -> None:
        paths = self.session_paths(session_id)
        video_name = paths["video"].name if paths["video"] else f"CameraRecord_{session_id}.mp4"
        rows = self.load_processing_status()
        row = rows.get(video_name, {})
        row.update({
            "file_name": video_name,
            "tracking_file": row.get("tracking_file") or (paths["tracking"].name if paths["tracking"] else f"trackingData_{session_id}.txt"),
            "status": "processed",
        })
        rows[video_name] = row
        self.save_processing_status(rows)

    def has_completed_export(self, session_id: str) -> bool:
        for sidecar in self.output_dir.glob("*/meta/pico_ego_release.json"):
            try:
                if json.loads(sidecar.read_text(encoding="utf-8")).get("session_id") == session_id:
                    return True
            except (OSError, json.JSONDecodeError):
                continue
        return False

    def session_processing_warnings(self, session_id: str, row: dict | None = None) -> list[str]:
        warnings: list[str] = []
        row = row or {}
        paths = self.session_paths(session_id)
        session_dir = Path(paths["session_dir"])
        work_dir = self.work_dir(session_id)
        output_dir = self.output_dir / session_id

        raw_existed = str(row.get("raw_existed_before", "")).lower() == "true"
        tracking_existed = str(row.get("tracking_existed_before", "")).lower() == "true"
        video_existed = str(row.get("video_existed_before", "")).lower() == "true"
        if raw_existed or tracking_existed or video_existed:
            warnings.append("raw already exists")
        elif (
            row
            and row.get("tracking_status") == "skipped"
            and row.get("video_status") == "skipped"
        ):
            warnings.append("raw unchanged")

        if self.annotations_path(session_id).exists():
            warnings.append("has annotations")
        if (self.tmp_root / "processed" / f"{session_id}_frames.csv").exists():
            warnings.append("has processed csv")
        if (work_dir / "preview_cache").exists():
            warnings.append("has preview cache")
        if paths["g1"] and paths["g1"].exists():
            warnings.append("has G1 retarget")
        if paths["wuji"] and paths["wuji"].exists():
            warnings.append("has Wuji retarget")
        if paths["g1_preview_video"] and paths["g1_preview_video"].exists():
            warnings.append("has G1 MuJoCo preview")
        if output_dir.exists() and any(output_dir.iterdir()):
            warnings.append("has LeRobot v3 output")
        if session_dir.exists() and not (paths["tracking"] and paths["video"]):
            warnings.append("raw is incomplete")

        return warnings

    def enrich_sync_rows(self, rows: list[dict]) -> list[dict]:
        enriched = []
        for row in rows:
            item = dict(row)
            session_id = str(item.get("session_id", "")).strip()
            warnings = self.session_processing_warnings(session_id, item) if session_id else []
            item["warnings"] = warnings
            item["warning_count"] = len(warnings)
            enriched.append(item)
        return enriched

    def run_sync(self, payload: dict) -> dict:
        source = str(payload.get("source", "") or "").strip()
        date_filter = str(payload.get("date", "") or "").strip()
        start_date = str(payload.get("start_date", "") or "").strip()
        end_date = str(payload.get("end_date", "") or "").strip()
        manifest = self.manifest_path()
        cmd = [
            sys.executable,
            str(self.root / "scripts" / "sync_pico_download.py"),
            "--raw-dir",
            str(self.raw_dir),
            "--manifest",
            str(manifest),
        ]
        if source:
            cmd.extend(["--source", source])
        if date_filter:
            cmd.extend(["--date", date_filter])
        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])

        proc = subprocess.run(
            cmd,
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        rows = []
        returncode = proc.returncode
        stderr = proc.stderr
        if returncode == 0 and not manifest.is_file():
            returncode = 1
            stderr += "\nImport process did not create a manifest for this request; import results cannot be confirmed."
        if returncode == 0:
            with manifest.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        rows = self.enrich_sync_rows(rows)
        imported = [row["session_id"] for row in rows if row.get("paired") == "True"]
        sessions = self.discover_sessions()
        pending = [sid for sid in imported if any(item.get("session_id") == sid for item in sessions)]
        warning_rows = [row for row in rows if row.get("warnings")]
        return {
            "ok": returncode == 0,
            "returncode": returncode,
            "stdout": proc.stdout,
            "stderr": stderr,
            "command": cmd,
            "manifest": str(manifest),
            "rows": rows,
            "imported_session_ids": imported,
            "pending_session_ids": pending,
            "warning_count": len(warning_rows),
            "sessions": sessions,
        }

    def export_command(self, args: list[str]) -> list[str]:
        if self.export_python:
            return [str(self.export_python), *args]
        if self.export_conda_env:
            return [
                str(self.conda_bin),
                "run",
                "-n",
                self.export_conda_env,
                "python",
                *args,
            ]
        return [sys.executable, *args]

    def session_paths(self, session_id: str) -> dict[str, Path | None]:
        session_dir = self.raw_dir / session_id
        if not session_dir.exists() and self.legacy_raw_dir.exists():
            legacy_session_dir = self.legacy_raw_dir / session_id
            if legacy_session_dir.exists():
                session_dir = legacy_session_dir
        tracking = next(iter(sorted(session_dir.glob("*trackingData_*.txt"))), None)
        video = next(iter(sorted(session_dir.glob("*CameraRecord_*.mp4"))), None)
        work_dir = self.work_dir(session_id)
        g1 = work_dir / "retargeted" / f"{session_id}_unitree_g1.pkl"
        wuji = work_dir / "wuji_retargeted" / f"{session_id}_wuji_hand.pkl"
        legacy_g1 = self.root / "data" / "retargeted" / f"{session_id}_unitree_g1.pkl"
        legacy_wuji = self.root / "data" / "wuji_retargeted" / f"{session_id}_wuji_hand.pkl"
        return {
            "session_dir": session_dir,
            "tracking": tracking,
            "video": video,
            "g1": g1 if g1.exists() or not legacy_g1.exists() else legacy_g1,
            "wuji": wuji if wuji.exists() or not legacy_wuji.exists() else legacy_wuji,
            "g1_preview_video": work_dir / "previews" / f"{session_id}_g1_mujoco.mp4",
        }

    def discover_sessions(self) -> list[dict]:
        self.prune_processing_status()
        summary_rows = {}
        for summary_path in [
            self.tmp_root / "reports" / "dataset_summary.csv",
            self.root / "reports" / "dataset_summary.csv",
        ]:
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        summary_rows[row.get("session_id", "")] = row
                break

        sessions = []
        raw_dirs = []
        if self.raw_dir.exists():
            raw_dirs.append(self.raw_dir)
        if self.legacy_raw_dir.exists() and self.legacy_raw_dir != self.raw_dir:
            raw_dirs.append(self.legacy_raw_dir)
        if not raw_dirs:
            return sessions
        seen = set()
        for base_dir in raw_dirs:
            for session_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
                if session_dir.name in seen:
                    continue
                seen.add(session_dir.name)
                sid = session_dir.name
                paths = self.session_paths(sid)
                # Hide directories left by an interrupted MTP copy when only
                # a trackingData_*.txt.tmp or similar partial file exists.
                # Only complete local pairs are eligible for video
                # annotation.  A tracking-only directory can be left by an
                # interrupted import and must not enter the session picker.
                if not paths["tracking"] or not paths["video"]:
                    continue
                status = self.ensure_processing_status(sid)
                # Backfill the table for exports created before the status
                # feature existed.  Completed sessions remain selectable so
                # users can review their saved annotations.
                if status != "processed" and self.has_completed_export(sid):
                    self.mark_session_processed(sid)
                    status = "processed"
                summary = summary_rows.get(sid, {})
                status_row = self.load_processing_status().get(paths["video"].name, {})
                annotation_path = self.annotations_path(sid)
                annotation_count = 0
                annotation_updated_at = None
                if annotation_path.exists():
                    try:
                        annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                        annotation_count = len(annotation_payload.get("episodes", []))
                        annotation_updated_at = annotation_payload.get("updated_at")
                    except (OSError, json.JSONDecodeError, TypeError):
                        # A malformed annotation file should not hide the raw
                        # session from the picker; the detail endpoint will
                        # report the actual load error if selected.
                        annotation_count = 0
                sessions.append(
                    {
                        "session_id": sid,
                        "device_id": status_row.get("device_id", ""),
                        "device_group": status_row.get("device_group", "ungrouped"),
                        "processing_status": status,
                        "has_annotations": annotation_count > 0,
                        "annotation_count": annotation_count,
                        "annotation_updated_at": annotation_updated_at,
                        "has_tracking": bool(paths["tracking"] and paths["tracking"].exists()),
                        "has_video": bool(paths["video"] and paths["video"].exists()),
                        "has_g1": bool(paths["g1"] and paths["g1"].exists()),
                        "has_wuji": bool(paths["wuji"] and paths["wuji"].exists()),
                        "has_g1_preview_video": bool(
                            paths["g1_preview_video"] and paths["g1_preview_video"].exists()
                        ),
                        "tracking_duration_sec": parse_float(summary.get("tracking_duration_sec")),
                        "video_duration_sec": parse_float(summary.get("video_duration_sec")),
                        "left_active_pct": parse_float(summary.get("left_active_pct")),
                        "right_active_pct": parse_float(summary.get("right_active_pct")),
                    }
                )
        group_order = {"group1": 1, "group2": 2, "group3": 3}
        sessions.sort(
            key=lambda item: (
                group_order.get(str(item.get("device_group", "")), 99),
                str(item.get("session_id", "")),
            )
        )
        return sessions

    def annotations_path(self, session_id: str) -> Path:
        return self.work_dir(session_id) / "annotations" / f"{session_id}_episodes.json"

    def load_annotations(self, session_id: str) -> dict:
        path = self.annotations_path(session_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "version": 1,
            "session_id": session_id,
            "episodes": [],
            "updated_at": None,
        }

    def save_annotations(self, session_id: str, payload: dict) -> dict:
        episodes = payload.get("episodes", [])
        if not isinstance(episodes, list):
            raise ValueError("episodes must be a list")

        cleaned = []
        for i, episode in enumerate(episodes):
            start = float(episode.get("start", 0.0))
            end = float(episode.get("end", 0.0))
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            cleaned.append(
                {
                    "episode_id": int(episode.get("episode_id", i)),
                    "start": round(start, 4),
                    "end": round(end, 4),
                    "prompt": str(episode.get("prompt", "")).strip(),
                    "status": str(episode.get("status", "keep") or "keep"),
                    "notes": str(episode.get("notes", "") or ""),
                }
            )
        cleaned.sort(key=lambda item: (item["start"], item["end"]))
        for i, episode in enumerate(cleaned):
            episode["episode_id"] = i

        out = {
            "version": 1,
            "session_id": session_id,
            "source": {
                "tracking": str(self.session_paths(session_id)["tracking"] or ""),
                "video": str(self.session_paths(session_id)["video"] or ""),
            },
            "episodes": cleaned,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.annotations_path(session_id).parent.mkdir(parents=True, exist_ok=True)
        self.annotations_path(session_id).write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return out

    def build_preview(self, session_id: str, sample_hz: float) -> dict:
        sample_hz = max(1.0, min(float(sample_hz), 30.0))
        paths = self.session_paths(session_id)
        tracking = paths["tracking"]
        if tracking is None or not tracking.exists():
            raise FileNotFoundError(f"No tracking file for session {session_id}")

        # v5: use the G1 file's stored timestamps instead of reconstructing
        # them from fps.  This prevents long recordings from drifting against
        # the Pico hand timeline.  Bumping the version invalidates old caches.
        cache = self.work_dir(session_id) / "preview_cache" / f"{session_id}_preview_v5_{sample_hz:g}hz.json"
        inputs = [
            path
            for path in [tracking, paths["g1"], paths["wuji"], paths["g1_preview_video"]]
            if path and path.exists()
        ]
        newest_input = max(path.stat().st_mtime for path in inputs)
        if cache.exists() and cache.stat().st_mtime >= newest_input:
            return json.loads(cache.read_text(encoding="utf-8"))

        g1 = load_g1(paths["g1"])
        wuji = load_wuji(paths["wuji"])
        frames = []
        bounds_points = []
        next_sample_t = 0.0
        first_ts = None
        first_data_ts = None
        # Wuji timelines start at the first tracking DATA frame, while G1 uses
        # the camera-first-frame clock stored in the G1 pkl.  Keep both clock
        # conventions explicit when looking up the two streams.
        track_offset = 0.0
        last_t = 0.0
        stride = 1.0 / sample_hz

        with tracking.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "notice" in record or "cameraIntrinsics" in record:
                    # Use the camera first-frame timestamp as t=0 so the preview
                    # shares the export's clock (video/pose/G1 all aligned).
                    meta_ts = int(record.get("timeStampNs", 0) or 0)
                    if meta_ts > 0 and first_ts is None:
                        first_ts = meta_ts
                    continue
                ts = int(record.get("timeStampNs", 0) or 0)
                if ts <= 0:
                    continue
                if first_ts is None:
                    first_ts = ts
                if first_data_ts is None:
                    first_data_ts = ts
                    track_offset = (first_data_ts - first_ts) / 1e9
                t = (ts - first_ts) / 1e9
                last_t = t
                if t + 1e-9 < next_sample_t:
                    continue
                next_sample_t += stride

                hand = record.get("Hand", {}) or {}
                left = hand.get("leftHand", {}) or {}
                right = hand.get("rightHand", {}) or {}
                body = record.get("Body", {}) or {}
                body_points = transform_points(parse_joint_positions(body.get("joints")))
                left_points = transform_points(parse_joint_positions(left.get("HandJointLocations")))
                right_points = transform_points(parse_joint_positions(right.get("HandJointLocations")))
                if len(body_points):
                    bounds_points.append(body_points)
                frame = {
                    "t": round(float(t), 4),
                    "body": array_to_json(body_points),
                    "left_hand": array_to_json(left_points),
                    "right_hand": array_to_json(right_points),
                    "left_active": bool(left.get("isActive", 0)),
                    "right_active": bool(right.get("isActive", 0)),
                }
                # G1 stores camera-clock timestamps; Wuji stores timestamps
                # relative to the first tracking data frame.
                if g1 is not None:
                    idx = nearest_index(g1["times"], t)
                    qpos = g1["qpos"][idx]
                    frame["g1_qpos"] = array_to_json(qpos, digits=4)
                    frame["g1_root"] = array_to_json(g1["root"][idx], digits=4)
                if wuji is not None:
                    track_t = t - track_offset
                    idx = nearest_index(wuji["times"], track_t)
                    frame["wuji_left_points"] = array_to_json(wuji["left_points"][idx])
                    frame["wuji_right_points"] = array_to_json(wuji["right_points"][idx])
                    frame["wuji_left_qpos"] = array_to_json(wuji["left_qpos"][idx], digits=4)
                    frame["wuji_right_qpos"] = array_to_json(wuji["right_qpos"][idx], digits=4)
                    frame["wuji_left_active"] = bool(wuji["left_active"][idx])
                    frame["wuji_right_active"] = bool(wuji["right_active"][idx])
                frames.append(frame)

        bounds = compute_bounds(bounds_points)
        payload = {
            "session_id": session_id,
            "sample_hz": sample_hz,
            "duration": round(float(last_t), 4),
            "video_url": f"/media/video/{session_id}",
            "g1_preview_video_url": f"/media/g1_preview/{session_id}"
            if paths["g1_preview_video"] and paths["g1_preview_video"].exists()
            else None,
            "body_edges": BODY_EDGES,
            "hand_edges": HAND_EDGES,
            "mediapipe_hand_edges": MEDIAPIPE_HAND_EDGES,
            "bounds": bounds,
            "frames": frames,
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return payload


def parse_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def compute_bounds(chunks: list[np.ndarray]) -> dict:
    if not chunks:
        return {"min": [-1, -1, -1], "max": [1, 1, 1]}
    pts = np.vstack(chunks)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return {"min": [-1, -1, -1], "max": [1, 1, 1]}
    mins = np.nanpercentile(pts, 2, axis=0)
    maxs = np.nanpercentile(pts, 98, axis=0)
    pad = np.maximum((maxs - mins) * 0.1, 0.2)
    return {
        "min": array_to_json(mins - pad),
        "max": array_to_json(maxs + pad),
    }


def nearest_index(times: np.ndarray, t: float) -> int:
    if len(times) == 0:
        return 0
    idx = int(np.searchsorted(times, t))
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx if abs(times[idx] - t) < abs(times[idx - 1] - t) else idx - 1


def load_g1(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open("rb") as f:
        data = pickle.load(f)
    fps = float(data.get("fps", 30.0) or 30.0)
    root = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    dof = np.asarray(data["dof_pos"], dtype=np.float32)
    qpos = np.concatenate([root, root_rot, dof], axis=1)
    stored_times = np.asarray(data.get("times", []), dtype=np.float32)
    if stored_times.shape == (len(qpos),) and np.isfinite(stored_times).all():
        times = stored_times
        clock = "camera"
    else:
        # Legacy G1 files did not persist timestamps and are necessarily less
        # accurate; retain the old fallback for compatibility.
        times = np.arange(len(qpos), dtype=np.float32) / fps
        clock = "tracking"
    return {
        "times": times,
        "root": root,
        "qpos": qpos,
        "clock": clock,
    }


def load_wuji(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open("rb") as f:
        data = pickle.load(f)
    times = np.asarray(data.get("times"), dtype=np.float32)
    if times.size == 0:
        fps = float(data.get("fps", 30.0) or 30.0)
        times = np.arange(len(data["right_qpos"]), dtype=np.float32) / fps
    left_points = transform_point_sequence(np.asarray(data.get("left_mediapipe_points"), dtype=np.float32))
    right_points = transform_point_sequence(np.asarray(data.get("right_mediapipe_points"), dtype=np.float32))
    left_qpos = np.asarray(data.get("left_qpos"), dtype=np.float32)
    right_qpos = np.asarray(data.get("right_qpos"), dtype=np.float32)
    left_active = np.asarray(data.get("left_active", np.ones(len(left_qpos), dtype=bool)), dtype=bool)
    right_active = np.asarray(data.get("right_active", np.ones(len(right_qpos), dtype=bool)), dtype=bool)
    return {
        "times": times,
        "left_points": left_points,
        "right_points": right_points,
        "left_qpos": left_qpos,
        "right_qpos": right_qpos,
        "left_active": left_active,
        "right_active": right_active,
    }


class AnnotatorHandler(BaseHTTPRequestHandler):
    state: AnnotatorState

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        try:
            self._do_GET()
        except _CLIENT_DISCONNECT:
            pass
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_HEAD(self) -> None:
        try:
            self._do_GET(head_only=True)
        except _CLIENT_DISCONNECT:
            pass
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            self._do_POST()
        except _CLIENT_DISCONNECT:
            pass
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _do_GET(self, head_only: bool = False) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)

        if path in {"/", "/annotator.html"}:
            return self.send_file(self.state.web_dir / "annotator.html", head_only=head_only)
        if path == "/api/sessions":
            return self.send_json({"sessions": self.state.discover_sessions()})
        if path == "/api/export_config":
            return self.send_json(self.state.export_config())
        if path == "/api/jobs":
            return self.send_json({"jobs": list_jobs()})

        match = re.fullmatch(r"/api/job/([^/]+)", path)
        if match:
            job = get_job(match.group(1))
            if job is None:
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Job not found")
            return self.send_json(job)

        match = re.fullmatch(r"/api/session/([^/]+)/preview", path)
        if match:
            sample_hz = float(query.get("sample_hz", ["5"])[0])
            return self.send_json(self.state.build_preview(match.group(1), sample_hz))

        match = re.fullmatch(r"/api/annotations/([^/]+)", path)
        if match:
            return self.send_json(self.state.load_annotations(match.group(1)))

        match = re.fullmatch(r"/media/video/([^/]+)", path)
        if match:
            video = self.state.session_paths(match.group(1))["video"]
            if video is None:
                return self.send_error_json(HTTPStatus.NOT_FOUND, "Video not found")
            return self.send_file(video, head_only=head_only)

        match = re.fullmatch(r"/media/g1_preview/([^/]+)", path)
        if match:
            preview = self.state.session_paths(match.group(1))["g1_preview_video"]
            if preview is None or not preview.exists():
                return self.send_error_json(HTTPStatus.NOT_FOUND, "G1 preview video not found")
            return self.send_file(preview, head_only=head_only)

        return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def _do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        match = re.fullmatch(r"/api/annotations/([^/]+)", path)
        if match:
            payload = read_json_body(self)
            return self.send_json(self.state.save_annotations(match.group(1), payload))

        if path == "/api/sync_pico":
            payload = read_json_body(self)
            return self.send_json(self.state.run_sync(payload))

        if path == "/api/redo_session":
            payload = read_json_body(self)
            session_id = str(payload.get("session_id", "")).strip()
            if not session_id:
                raise ValueError("session_id is required")
            return self.send_json(self.state.redo_session(session_id))

        if path == "/api/delete_session":
            payload = read_json_body(self)
            session_id = str(payload.get("session_id", "")).strip()
            return self.send_json(self.state.delete_session(session_id))

        if path == "/api/export_lerobot":
            payload = read_json_body(self)
            session_id = str(payload.get("session_id", "")).strip()
            if not re.fullmatch(r"\d{8}_\d{6}", session_id):
                raise ValueError("session_id must look like YYYYMMDD_HHMMSS")
            annotation_path = self.state.annotations_path(session_id)
            paths = self.state.session_paths(session_id)
            # Constrain the export target to live under the output root — the
            # exporter rmtree's this path on --overwrite, so never let the web
            # point it at an arbitrary directory.
            requested = self.state.resolve_project_path(
                Path(payload.get("output_dir") or self.state.output_dir / session_id)
            )
            out_root = self.state.output_dir.resolve()
            if out_root not in requested.resolve().parents:
                raise ValueError(f"output_dir must be a subdirectory of {out_root}")
            output_dir = requested
            repo_id = str(payload.get("repo_id") or f"pico_ego/{session_id}")
            fps = 60
            dry_run = bool(payload.get("dry_run", False))
            cmd = self.state.export_command(
                [
                    str(self.state.root / "scripts" / "export_lerobot_v3.py"),
                    "--session",
                    session_id,
                    "--annotations",
                    str(annotation_path),
                    "--output-dir",
                    str(output_dir),
                    "--raw-dir",
                    str(Path(paths["session_dir"]).parent),
                    "--g1-dir",
                    str(Path(paths["g1"]).parent),
                    "--repo-id",
                    repo_id,
                    "--fps",
                    str(fps),
                ]
            )
            if dry_run:
                cmd.append("--dry-run")
            else:
                # Re-exporting a session: replace any previous output in place.
                cmd.append("--overwrite")
            if payload.get("allow_missing_hands") is True:
                cmd.append("--allow-missing-hands")
            def mark_processed() -> None:
                self.mark_session_processed(session_id)

            # Keep the per-session work directory after export. It contains
            # annotations and retarget artifacts that may still be edited or
            # audited while another session is being processed. Explicit
            # "Redo"/"Delete data" operations remain responsible for cleanup.
            on_success = None if dry_run else mark_processed
            job = start_job(
                "export",
                session_id,
                cmd,
                self.state.root,
                on_success=on_success,
                extra={
                    "dry_run": dry_run,
                    "export_config": self.state.export_config(),
                    "output_dir": str(output_dir),
                    "repo_id": repo_id,
                    "tuning": {
                        "video_preset": os.environ.get("PICO_EXPORT_VIDEO_PRESET", "veryfast"),
                        "encoder_threads": int(os.environ.get("PICO_EXPORT_ENCODER_THREADS", "8")),
                        "image_writer_processes": int(os.environ.get("PICO_EXPORT_IMAGE_WRITER_PROCESSES", "0")),
                        "image_writer_threads": int(os.environ.get("PICO_EXPORT_IMAGE_WRITER_THREADS", "8")),
                    },
                },
            )
            return self.send_json(job, status=HTTPStatus.ACCEPTED)

        if path == "/api/render_g1_preview":
            payload = read_json_body(self)
            session_id = str(payload.get("session_id", "")).strip()
            if not re.fullmatch(r"\d{8}_\d{6}", session_id):
                raise ValueError("session_id must look like YYYYMMDD_HHMMSS")
            paths = self.state.session_paths(session_id)
            tracking = paths["tracking"]
            if tracking is None or not tracking.exists():
                raise ValueError(f"Tracking file not found for session {session_id}")

            root = self.state.root
            work = self.state.work_dir(session_id)
            expected_g1 = work / "retargeted" / f"{session_id}_unitree_g1.pkl"
            expected_wuji = work / "wuji_retargeted" / f"{session_id}_wuji_hand.pkl"
            expected_wuji_replay = work / "wuji_retargeted" / f"{session_id}_mediapipe_replay.pkl"
            output = paths["g1_preview_video"]
            wuji_best_config = root / "configs" / "wuji_retarget_pico_best.yaml"

            # Always regenerate (never reuse stale pkl/mp4 after retuning).
            g1_cmd = [
                sys.executable, str(root / "scripts" / "retarget_pico_to_g1.py"),
                "--session", session_id, "--tracking", str(tracking), "--output", str(expected_g1),
            ]
            wuji_cmd = [
                sys.executable, str(root / "scripts" / "retarget_pico_hand_to_wuji.py"),
                "--session", session_id, "--tracking", str(tracking),
                "--output", str(expected_wuji), "--replay-output", str(expected_wuji_replay),
            ]
            if wuji_best_config.exists():
                wuji_cmd.extend(["--config", str(wuji_best_config)])
            render_cmd = [
                sys.executable, str(root / "scripts" / "render_g1_mujoco_preview.py"),
                "--session", session_id, "--input", str(expected_g1), "--output", str(output),
                "--preview-fps", str(int(payload.get("preview_fps") or 15)),
                "--camera-azimuth", str(float(payload.get("camera_azimuth") or 345.0)),
                "--width", str(int(payload.get("width") or 640)),
                "--height", str(int(payload.get("height") or 360)),
            ]
            # Wuji hand retarget is preview-only (hands stay raw Pico26 in the
            # release), so it is skipped by default. Pass include_wuji=true to run it.
            steps = [("g1_retarget", g1_cmd)]
            if bool(payload.get("include_wuji", False)):
                steps.append(("wuji_retarget", wuji_cmd))
            steps.append(("render", render_cmd))
            job = start_steps_job(
                "generate", session_id, steps, root,
                extra={"preview_url": f"/media/g1_preview/{session_id}", "included_wuji": bool(payload.get("include_wuji", False))},
            )
            return self.send_json(job, status=HTTPStatus.ACCEPTED)

        return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except _CLIENT_DISCONNECT:
            pass  # client went away before we finished responding

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def send_file(self, path: Path | None, head_only: bool = False) -> None:
        if path is None or not path.exists() or not path.is_file():
            return self.send_error_json(HTTPStatus.NOT_FOUND, "File not found")

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = size - 1
        status = HTTPStatus.OK

        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT

        length = end - start + 1
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            if path.suffix.lower() in {".html", ".css", ".js"}:
                self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head_only:
                return

            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except _CLIENT_DISCONNECT:
            pass  # browser cancelled the (often video range) request mid-stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory that stores Pico raw sessions.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for final LeRobot v3 outputs.")
    parser.add_argument("--tmp-root", type=Path, default=None, help="Temporary processing root.")
    parser.add_argument(
        "--export-python",
        type=Path,
        default=os.environ.get("PICO_LEROBOT_PYTHON"),
        help="Python executable used for full LeRobot export.",
    )
    parser.add_argument(
        "--export-conda-env",
        default=os.environ.get("PICO_LEROBOT_CONDA_ENV"),
        help="Conda environment name used for full LeRobot export.",
    )
    parser.add_argument(
        "--conda-bin",
        type=Path,
        default=os.environ.get("PICO_CONDA_BIN"),
        help="Path to conda when --export-conda-env is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_python and args.export_conda_env:
        raise SystemExit("Use either --export-python or --export-conda-env, not both.")
    load_persisted_jobs()
    AnnotatorHandler.state = AnnotatorState(
        args.root.resolve(),
        export_python=args.export_python,
        export_conda_env=args.export_conda_env,
        conda_bin=args.conda_bin,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        tmp_root=args.tmp_root,
    )
    server = ThreadingHTTPServer((args.host, args.port), AnnotatorHandler)
    print(f"Annotator running at http://{args.host}:{args.port}")
    print(f"Export runtime: {AnnotatorHandler.state.export_config()['label']}")
    print(f"Raw dir: {AnnotatorHandler.state.raw_dir}")
    print(f"Output dir: {AnnotatorHandler.state.output_dir}")
    print(f"Temp root: {AnnotatorHandler.state.tmp_root}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
