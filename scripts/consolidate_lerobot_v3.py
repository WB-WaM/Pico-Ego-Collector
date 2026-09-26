#!/usr/bin/env python3
"""Consolidate per-session Pico Ego datasets into one publishable LeRobot v3 release.

Each session is exported by ``export_lerobot_v3.py`` as a standalone
LeRobotDataset under ``lerobot_v3/<session_id>/`` (repo_id ``pico_ego/<sid>``).
This script merges several of them into a single dataset -- the thing you
actually publish to the Hub -- reusing LeRobot's official file-level
``aggregate_datasets`` (no video re-encode; tasks de-duplicated; stats
recomputed) and additionally merging our custom sidecar
``meta/pico_ego_release.json`` (camera intrinsics/extrinsics and A/V alignment
are per-capture, so they are kept grouped per session with globally-remapped
episode indices). Strict preflight/post-write validation protects the frozen
schema, and ``meta/task_partitions.json`` records a deterministic task index.

Pass ``--task-output-root releases/pico_ego_by_task`` to additionally build one
independent LeRobot v3 dataset per normalized task. This optional release-time
step re-encodes only video files that contain episodes from multiple tasks.

Run under the ``lerobot`` env (needs the lerobot package). Example:

    python scripts/consolidate_lerobot_v3.py \\
        --sessions-root lerobot_v3 \\
        --output-dir releases/pico_ego \\
        --repo-id pico_ego --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCHEMA_VERSION = "pico_ego_v3_schema_v1"
TASK_MANIFEST_NAME = "task_partitions.json"

# Keys in a per-session sidecar that are shared across the whole release and are
# asserted to be identical (they describe the schema, not the capture).
_SHARED_SIDECAR_KEYS = (
    "schema_version",
    "schema_status",
    "coordinate_convention",
    "sampling_policy",
    "head_pose_note",
    "episode_quality_policy",
    "action",
    "notes",
)
# Keys kept per-session (they describe the individual capture rig / clock).
_PER_SESSION_SIDECAR_KEYS = (
    "camera",
    "stereo_camera",
    "alignment",
    "stereo",
    "video_codec",
    "quality_control",
)


def emit_progress(**payload: Any) -> None:
    """Emit one ``@PROGRESS {...}`` line to stderr (parsed by the web server)."""
    try:
        sys.stderr.write("@PROGRESS " + json.dumps(payload, default=str) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def assert_safe_output_dir(output_dir: Path) -> None:
    """Refuse to treat a dangerous path as output (we rmtree it on --overwrite)."""
    resolved = output_dir.resolve()
    unsafe = {Path("/").resolve(), Path.home().resolve(), PROJECT_ROOT.resolve(), Path.cwd().resolve()}
    if resolved in unsafe or resolved.parent == resolved:
        raise SystemExit(f"Refusing to use unsafe output dir: {resolved}")
    if resolved in PROJECT_ROOT.resolve().parents:
        raise SystemExit(f"Refusing output dir that contains the project root: {resolved}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_task(task: object) -> str:
    """Normalize harmless Unicode/whitespace differences without rewriting semantics."""
    return " ".join(unicodedata.normalize("NFKC", str(task or "")).split())


def _task_slug_base(task: str) -> str:
    ascii_task = (
        unicodedata.normalize("NFKD", task).encode("ascii", "ignore").decode("ascii").lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_task).strip("-")[:64].rstrip("-")
    if slug:
        return slug
    digest = hashlib.sha1(task.encode("utf-8")).hexdigest()[:10]
    return f"task-{digest}"


def assign_task_slugs(tasks: list[str]) -> dict[str, str]:
    """Return deterministic, collision-safe slugs keyed by normalized task text."""
    result: dict[str, str] = {}
    used: dict[str, str] = {}
    for task in sorted(set(tasks), key=lambda value: (value.casefold(), value)):
        base = _task_slug_base(task)
        slug = base
        if slug in used and used[slug] != task:
            slug = f"{base[:53].rstrip('-')}-{hashlib.sha1(task.encode('utf-8')).hexdigest()[:10]}"
        used[slug] = task
        result[task] = slug
    return result


def _episode_columns(video_keys: list[str]) -> list[str]:
    columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for key in video_keys:
        columns.extend(
            [
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
                f"videos/{key}/to_timestamp",
            ]
        )
    return columns


def load_episode_rows(dataset_root: Path, video_keys: list[str]) -> list[dict[str, Any]]:
    """Read compact episode metadata without materializing the large stats columns."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for consolidation preflight. Run under the `lerobot` env."
        ) from exc

    episode_dir = dataset_root / "meta" / "episodes"
    paths = sorted(episode_dir.rglob("*.parquet")) if episode_dir.exists() else []
    if not paths:
        return []
    requested = _episode_columns(video_keys)
    rows: list[dict[str, Any]] = []
    for path in paths:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [name for name in requested if name not in schema_names]
        if missing:
            raise ValueError(f"Episode metadata {path} is missing columns: {missing}")
        rows.extend(pq.read_table(path, columns=requested).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _referenced_dataset_files(
    root: Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    video_keys: list[str],
) -> tuple[set[Path], set[Path]]:
    data_template = info.get("data_path")
    video_template = info.get("video_path")
    data_files: set[Path] = set()
    video_files: set[Path] = set()
    if isinstance(data_template, str):
        for row in episode_rows:
            data_files.add(
                root
                / data_template.format(
                    chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"]),
                )
            )
    if video_keys and isinstance(video_template, str):
        for row in episode_rows:
            for key in video_keys:
                video_files.add(
                    root
                    / video_template.format(
                        video_key=key,
                        chunk_index=int(row[f"videos/{key}/chunk_index"]),
                        file_index=int(row[f"videos/{key}/file_index"]),
                    )
                )
    return data_files, video_files


def discover_sessions(sessions_root: Path, only: list[str] | None) -> list[dict[str, Any]]:
    """Find per-session dataset dirs (each has meta/info.json). Returns them sorted
    by session id so the merged episode order is deterministic."""
    if not sessions_root.exists():
        raise SystemExit(f"sessions root does not exist: {sessions_root}")
    found: list[dict[str, Any]] = []
    for child in sorted(sessions_root.iterdir()):
        if not child.is_dir():
            continue
        info_path = child / "meta" / "info.json"
        if not info_path.exists():
            continue  # not a dataset dir (e.g. a stray staging/report dir)
        if only and child.name not in only:
            continue
        info = _load_json(info_path)
        sidecar_path = child / "meta" / "pico_ego_release.json"
        sidecar = _load_json(sidecar_path) if sidecar_path.exists() else {}
        feature_schema = info.get("features") or {}
        video_keys = sorted(
            key for key, spec in feature_schema.items() if spec.get("dtype") == "video"
        )
        try:
            episode_rows = load_episode_rows(child, video_keys)
        except (OSError, ValueError) as exc:
            episode_rows = []
            episode_metadata_error = str(exc)
        else:
            episode_metadata_error = None
        found.append(
            {
                "session_id": child.name,
                "root": child,
                "repo_id": info.get("repo_id") or f"pico_ego/{child.name}",
                "info": info,
                "sidecar": sidecar,
                "sidecar_path": sidecar_path,
                "total_episodes": int(info.get("total_episodes", 0)),
                "total_frames": int(info.get("total_frames", 0)),
                "total_tasks": int(info.get("total_tasks", 0)),
                "robot_type": info.get("robot_type"),
                "fps": info.get("fps"),
                "features": sorted(feature_schema.keys()),
                "feature_schema": feature_schema,
                "video_keys": video_keys,
                "episode_rows": episode_rows,
                "episode_metadata_error": episode_metadata_error,
                "schema_version": sidecar.get("schema_version"),
                "schema_status": sidecar.get("schema_status"),
            }
        )
    if only:
        missing = sorted(set(only) - {s["session_id"] for s in found})
        if missing:
            raise SystemExit(f"Requested sessions not found under {sessions_root}: {missing}")
    return found


def _session_validation_reasons(session: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    info = session["info"]
    sidecar = session["sidecar"]
    rows = session["episode_rows"]

    if session["episode_metadata_error"]:
        reasons.append(f"episode metadata unreadable: {session['episode_metadata_error']}")
    if not session["sidecar_path"].exists():
        reasons.append("missing meta/pico_ego_release.json")
    if session["schema_version"] != EXPECTED_SCHEMA_VERSION or session["schema_status"] != "frozen":
        reasons.append(
            f"expected frozen schema {EXPECTED_SCHEMA_VERSION}, got "
            f"{session['schema_version']} / {session['schema_status']}"
        )
    for key in _SHARED_SIDECAR_KEYS:
        if key not in sidecar:
            reasons.append(f"sidecar missing shared field {key}")
    if sidecar.get("session_id") != session["session_id"]:
        reasons.append(
            f"sidecar session_id {sidecar.get('session_id')!r} != directory {session['session_id']!r}"
        )
    if sidecar.get("fps") != session["fps"]:
        reasons.append(f"sidecar fps {sidecar.get('fps')} != info fps {session['fps']}")
    if session["fps"] is None or float(session["fps"]) <= 0:
        reasons.append(f"invalid fps {session['fps']}")
    if not session["feature_schema"]:
        reasons.append("info.json has no features")
    video_feature_codecs = {
        spec.get("info", {}).get("video.codec")
        for key, spec in session["feature_schema"].items()
        if key in session["video_keys"] and spec.get("info", {}).get("video.codec")
    }
    sidecar_codec = sidecar.get("video_codec")
    codec_to_metadata = {"h264": "h264", "hevc": "hevc", "libsvtav1": "av1"}
    expected_metadata_codec = codec_to_metadata.get(sidecar_codec)
    if len(video_feature_codecs) > 1:
        reasons.append(f"video features use mixed codecs: {sorted(video_feature_codecs)}")
    elif expected_metadata_codec and video_feature_codecs != {expected_metadata_codec}:
        reasons.append(
            f"sidecar video_codec {sidecar_codec!r} != feature codec(s) "
            f"{sorted(video_feature_codecs)}"
        )
    stereo_keys = {
        "observation.images.stereo_left",
        "observation.images.stereo_right",
    }
    if bool(sidecar.get("stereo")) != stereo_keys.issubset(session["video_keys"]):
        reasons.append(
            f"sidecar stereo={sidecar.get('stereo')!r} does not match video feature keys"
        )
    if session["total_episodes"] <= 0:
        reasons.append(f"invalid total_episodes {session['total_episodes']}")
    if session["total_frames"] <= 0:
        reasons.append(f"invalid total_frames {session['total_frames']}")

    if len(rows) != session["total_episodes"]:
        reasons.append(
            f"episode metadata rows {len(rows)} != info total_episodes {session['total_episodes']}"
        )
    indices = [int(row["episode_index"]) for row in rows]
    if indices != list(range(len(rows))):
        reasons.append(f"episode indices are not contiguous from zero: {indices[:12]}")
    lengths = [int(row["length"]) for row in rows]
    if any(length <= 0 for length in lengths):
        reasons.append("one or more episodes have non-positive length")
    if sum(lengths) != session["total_frames"]:
        reasons.append(
            f"episode lengths sum {sum(lengths)} != info total_frames {session['total_frames']}"
        )

    expected_from = 0
    normalized_tasks: set[str] = set()
    for row in rows:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        data_from = int(row["dataset_from_index"])
        data_to = int(row["dataset_to_index"])
        if data_from != expected_from or data_to - data_from != length:
            reasons.append(
                f"episode {episode_index} data range [{data_from}, {data_to}) "
                f"does not match expected start {expected_from} and length {length}"
            )
        expected_from = data_to

        tasks = [normalize_task(task) for task in (row.get("tasks") or [])]
        if len(tasks) != 1 or not tasks[0]:
            reasons.append(
                f"episode {episode_index} must have exactly one non-empty task, got {tasks}"
            )
        else:
            normalized_tasks.add(tasks[0])

        for key in session["video_keys"]:
            start = float(row[f"videos/{key}/from_timestamp"])
            end = float(row[f"videos/{key}/to_timestamp"])
            expected_duration = length / float(session["fps"] or 1)
            if start < 0 or end <= start or abs((end - start) - expected_duration) > 1e-5:
                reasons.append(
                    f"episode {episode_index} {key} duration {end - start:.9f}s "
                    f"!= {length}/{session['fps']}={expected_duration:.9f}s"
                )

    if session["total_tasks"] != len(normalized_tasks):
        reasons.append(
            f"info total_tasks {session['total_tasks']} != used normalized tasks {len(normalized_tasks)}"
        )

    sidecar_episodes = sidecar.get("episodes") or []
    if len(sidecar_episodes) != session["total_episodes"]:
        reasons.append(
            f"sidecar episodes {len(sidecar_episodes)} != info total_episodes {session['total_episodes']}"
        )
    for index, (row, episode) in enumerate(zip(rows, sidecar_episodes, strict=False)):
        if int(episode.get("episode_index", -1)) != index:
            reasons.append(f"sidecar episode {index} has index {episode.get('episode_index')}")
        row_tasks = [normalize_task(task) for task in (row.get("tasks") or [])]
        sidecar_task = normalize_task(episode.get("task"))
        if len(row_tasks) == 1 and row_tasks[0] != sidecar_task:
            reasons.append(
                f"episode {index} task mismatch: metadata={row_tasks[0]!r}, sidecar={sidecar_task!r}"
            )

    data_files, video_files = _referenced_dataset_files(
        session["root"], info, rows, session["video_keys"]
    )
    for path in sorted(data_files | video_files):
        if not path.is_file() or path.stat().st_size <= 0:
            reasons.append(f"referenced file missing or empty: {path.relative_to(session['root'])}")
    if session["video_keys"] and not isinstance(info.get("video_path"), str):
        reasons.append("video features exist but info.json has no video_path")
    if not isinstance(info.get("data_path"), str):
        reasons.append("info.json has no data_path")
    return reasons


def check_compatibility(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every per-session and cross-session preflight failure."""
    if not sessions:
        return [{"session_id": None, "reasons": ["no sessions found"], "reference": None}]

    ref = sessions[0]
    problems: list[dict[str, Any]] = []
    for session in sessions:
        reasons = _session_validation_reasons(session)
        if session is not ref:
            if session["fps"] != ref["fps"]:
                reasons.append(f"fps {session['fps']} != reference {ref['fps']}")
            if session["robot_type"] != ref["robot_type"]:
                reasons.append(
                    f"robot_type {session['robot_type']} != reference {ref['robot_type']}"
                )
            if _canonical_json(session["feature_schema"]) != _canonical_json(ref["feature_schema"]):
                only_ref = sorted(set(ref["features"]) - set(session["features"]))
                only_session = sorted(set(session["features"]) - set(ref["features"]))
                reasons.append(
                    "full feature schema differs from reference "
                    f"(missing={only_ref}, extra={only_session}, or dtype/shape/info mismatch)"
                )
            for key in _SHARED_SIDECAR_KEYS:
                if _canonical_json(session["sidecar"].get(key)) != _canonical_json(ref["sidecar"].get(key)):
                    reasons.append(f"shared sidecar field {key} differs from reference")
            for key in ("stereo", "video_codec"):
                if session["sidecar"].get(key) != ref["sidecar"].get(key):
                    reasons.append(
                        f"sidecar {key} {session['sidecar'].get(key)!r} "
                        f"!= reference {ref['sidecar'].get(key)!r}"
                    )
        if reasons:
            problems.append(
                {
                    "session_id": session["session_id"],
                    "reasons": reasons,
                    "reference": ref["session_id"],
                }
            )
    return problems


def build_task_inventory(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    global_offset = 0
    for session in sessions:
        for row in session["episode_rows"]:
            tasks = [normalize_task(task) for task in (row.get("tasks") or [])]
            if len(tasks) != 1 or not tasks[0]:
                continue
            display_task = tasks[0]
            key = display_task
            group = grouped.setdefault(
                key,
                {
                    "task": display_task,
                    "source_task_texts": set(),
                    "episodes": [],
                    "total_frames": 0,
                    "session_ids": set(),
                },
            )
            group["source_task_texts"].add(display_task)
            local_index = int(row["episode_index"])
            length = int(row["length"])
            group["episodes"].append(
                {
                    "global_episode_index": global_offset + local_index,
                    "session_id": session["session_id"],
                    "session_episode_index": local_index,
                    "frames": length,
                }
            )
            group["total_frames"] += length
            group["session_ids"].add(session["session_id"])
        global_offset += session["total_episodes"]

    display_tasks = [group["task"] for group in grouped.values()]
    slugs = assign_task_slugs(display_tasks)
    inventory = []
    for group in grouped.values():
        episodes = sorted(group["episodes"], key=lambda item: item["global_episode_index"])
        inventory.append(
            {
                "task": group["task"],
                "slug": slugs[group["task"]],
                "source_task_texts": sorted(group["source_task_texts"]),
                "total_episodes": len(episodes),
                "total_frames": int(group["total_frames"]),
                "session_ids": sorted(group["session_ids"]),
                "episodes": episodes,
            }
        )
    return sorted(inventory, key=lambda item: item["slug"])


def task_manifest(repo_id: str, task_inventory: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "repo_id": repo_id,
        "grouping": "exact task text after Unicode NFKC + whitespace normalization",
        "total_tasks": len(task_inventory),
        "total_episodes": sum(item["total_episodes"] for item in task_inventory),
        "total_frames": sum(item["total_frames"] for item in task_inventory),
        "tasks": task_inventory,
    }


def merge_sidecars(sessions: list[dict[str, Any]], aggr_repo_id: str) -> dict[str, Any]:
    """Combine per-session sidecars into one release-level sidecar.

    Shared schema fields are taken from the first session after strict preflight
    has established that every session carries the same values.
    Per-session capture fields (camera, alignment, stereo split) stay grouped
    under ``sessions`` with each episode's index remapped to the global order."""
    merged: dict[str, Any] = {
        "repo_id": aggr_repo_id,
        "num_sessions": len(sessions),
        "total_episodes": sum(s["total_episodes"] for s in sessions),
        "total_frames": sum(s["total_frames"] for s in sessions),
        "sessions": [],
    }
    stereo_flags: set[bool] = set()
    codecs: set[str] = set()
    first_sidecar: dict[str, Any] | None = None
    ep_offset = 0
    for s in sessions:
        sidecar = s["sidecar"]
        if first_sidecar is None and sidecar:
            first_sidecar = sidecar
        if "stereo" in sidecar:
            stereo_flags.add(bool(sidecar["stereo"]))
        if sidecar.get("video_codec"):
            codecs.add(sidecar["video_codec"])
        local_eps = sidecar.get("episodes", [])
        remapped = [
            {
                "global_episode_index": ep_offset + i,
                "session_episode_index": ep.get("episode_index", i),
                "start_sec": ep.get("start_sec"),
                "end_sec": ep.get("end_sec"),
                "task": ep.get("task"),
            }
            for i, ep in enumerate(local_eps)
        ]
        # If the sidecar is missing (older export), still record the session with
        # its declared episode count so global offsets stay correct.
        if not remapped and s["total_episodes"]:
            remapped = [
                {"global_episode_index": ep_offset + i, "session_episode_index": i,
                 "start_sec": None, "end_sec": None, "task": None}
                for i in range(s["total_episodes"])
            ]
        entry = {
            "session_id": s["session_id"],
            "episode_index_offset": ep_offset,
            "num_episodes": len(remapped),
            "episodes": remapped,
        }
        for key in _PER_SESSION_SIDECAR_KEYS:
            if key in sidecar:
                entry[key] = sidecar[key]
        merged["sessions"].append(entry)
        ep_offset += len(remapped)

    # Shared schema fields lifted to the top level.
    base = first_sidecar or {}
    for key in _SHARED_SIDECAR_KEYS:
        if key in base:
            merged[key] = base[key]
    merged["fps"] = base.get("fps")
    # Surface release-wide consistency of the capture options.
    merged["stereo"] = (next(iter(stereo_flags)) if len(stereo_flags) == 1 else sorted(stereo_flags))
    merged["video_codec"] = (next(iter(codecs)) if len(codecs) == 1 else sorted(codecs))
    merged["stereo_uniform"] = len(stereo_flags) <= 1
    merged["video_codec_uniform"] = len(codecs) <= 1
    return merged


def remove_empty_image_dirs(dataset_root: Path) -> int:
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


def validate_dataset_tree(
    root: Path,
    expected_episodes: int,
    expected_frames: int,
    expected_features: dict[str, Any],
    require_task_manifest: bool = True,
) -> dict[str, Any]:
    """Validate the staged release before it can replace an existing good one."""
    errors: list[str] = []
    info_path = root / "meta" / "info.json"
    sidecar_path = root / "meta" / "pico_ego_release.json"
    manifest_path = root / "meta" / TASK_MANIFEST_NAME
    if not info_path.is_file():
        return {"ok": False, "errors": ["missing meta/info.json"]}

    info = _load_json(info_path)
    features = info.get("features") or {}
    video_keys = sorted(key for key, spec in features.items() if spec.get("dtype") == "video")
    try:
        rows = load_episode_rows(root, video_keys)
    except (OSError, ValueError) as exc:
        rows = []
        errors.append(f"episode metadata unreadable: {exc}")

    if int(info.get("total_episodes", -1)) != expected_episodes:
        errors.append(
            f"info total_episodes {info.get('total_episodes')} != expected {expected_episodes}"
        )
    if int(info.get("total_frames", -1)) != expected_frames:
        errors.append(f"info total_frames {info.get('total_frames')} != expected {expected_frames}")
    if _canonical_json(features) != _canonical_json(expected_features):
        errors.append("output feature schema differs from the validated input schema")
    if len(rows) != expected_episodes:
        errors.append(f"episode metadata rows {len(rows)} != expected {expected_episodes}")
    episode_indices = [int(row["episode_index"]) for row in rows]
    if episode_indices != list(range(expected_episodes)):
        errors.append("output episode indices are not contiguous from zero")
    lengths = [int(row["length"]) for row in rows]
    if sum(lengths) != expected_frames:
        errors.append(f"output episode lengths sum {sum(lengths)} != expected {expected_frames}")

    data_files, video_files = _referenced_dataset_files(root, info, rows, video_keys)
    for path in sorted(data_files | video_files):
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"referenced file missing or empty: {path.relative_to(root)}")

    video_frame_counts: dict[str, int] = defaultdict(int)
    video_frame_counts_known: dict[str, bool] = {key: True for key in video_keys}
    try:
        import av

        for path in sorted(video_files):
            relative = path.relative_to(root)
            key = relative.parts[1] if len(relative.parts) > 1 else ""
            if key not in features or not path.is_file():
                continue
            with av.open(str(path)) as container:
                if not container.streams.video:
                    errors.append(f"no video stream in {relative}")
                    continue
                stream = container.streams.video[0]
                feature_info = features[key].get("info") or {}
                expected_codec = feature_info.get("video.codec")
                actual_codec = stream.codec_context.name
                if expected_codec and actual_codec != expected_codec:
                    errors.append(
                        f"{relative} codec {actual_codec} != metadata {expected_codec}"
                    )
                expected_width = int(feature_info.get("video.width", features[key]["shape"][1]))
                expected_height = int(feature_info.get("video.height", features[key]["shape"][0]))
                if stream.codec_context.width != expected_width or stream.codec_context.height != expected_height:
                    errors.append(
                        f"{relative} size {stream.codec_context.width}x{stream.codec_context.height} "
                        f"!= metadata {expected_width}x{expected_height}"
                    )
                actual_fps = float(stream.average_rate) if stream.average_rate else None
                if actual_fps is not None and abs(actual_fps - float(info.get("fps", 0))) > 1e-6:
                    errors.append(f"{relative} fps {actual_fps} != metadata {info.get('fps')}")
                if stream.frames > 0:
                    video_frame_counts[key] += int(stream.frames)
                else:
                    video_frame_counts_known[key] = False
    except Exception as exc:  # PyAV exposes backend-specific exception classes.
        errors.append(f"video container validation failed: {exc}")

    for key in video_keys:
        if video_frame_counts_known[key] and video_frame_counts[key] != expected_frames:
            errors.append(
                f"{key} video frames {video_frame_counts[key]} != expected {expected_frames}"
            )

    data_rows = 0
    episode_frame_counts: dict[int, int] = defaultdict(int)
    next_global_index = 0
    total_tasks = int(info.get("total_tasks", 0))
    try:
        import numpy as np
        import pyarrow.parquet as pq

        for path in sorted(data_files):
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                columns=["index", "episode_index", "frame_index", "task_index"],
                batch_size=65536,
            ):
                indexes = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
                episode_indexes = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
                frame_indexes = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64)
                task_indexes = batch.column(3).to_numpy(zero_copy_only=False).astype(np.int64)
                expected = np.arange(next_global_index, next_global_index + len(indexes), dtype=np.int64)
                if not np.array_equal(indexes, expected):
                    errors.append(f"global index discontinuity in {path.relative_to(root)}")
                    next_global_index = int(indexes[-1]) + 1 if len(indexes) else next_global_index
                else:
                    next_global_index += len(indexes)
                if np.any(task_indexes < 0) or np.any(task_indexes >= total_tasks):
                    errors.append(f"task_index out of range in {path.relative_to(root)}")
                for episode_index in np.unique(episode_indexes):
                    mask = episode_indexes == episode_index
                    start = episode_frame_counts[int(episode_index)]
                    expected_frames_for_batch = np.arange(
                        start, start + int(mask.sum()), dtype=np.int64
                    )
                    if not np.array_equal(frame_indexes[mask], expected_frames_for_batch):
                        errors.append(
                            f"frame_index discontinuity for episode {int(episode_index)}"
                        )
                    episode_frame_counts[int(episode_index)] += int(mask.sum())
                data_rows += len(indexes)
    except (ImportError, OSError, ValueError) as exc:
        errors.append(f"data parquet validation failed: {exc}")

    if data_rows != expected_frames:
        errors.append(f"data parquet rows {data_rows} != expected {expected_frames}")
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_frame_counts.get(episode_index, 0) != int(row["length"]):
            errors.append(
                f"episode {episode_index} data rows {episode_frame_counts.get(episode_index, 0)} "
                f"!= metadata length {row['length']}"
            )

    if not sidecar_path.is_file():
        errors.append("missing meta/pico_ego_release.json")
    else:
        sidecar = _load_json(sidecar_path)
        if int(sidecar.get("total_episodes", expected_episodes)) != expected_episodes:
            errors.append("sidecar total_episodes does not match output")
        if int(sidecar.get("total_frames", expected_frames)) != expected_frames:
            errors.append("sidecar total_frames does not match output")
    if require_task_manifest:
        if not manifest_path.is_file():
            errors.append(f"missing meta/{TASK_MANIFEST_NAME}")
        else:
            manifest = _load_json(manifest_path)
            if int(manifest.get("total_episodes", -1)) != expected_episodes:
                errors.append("task manifest total_episodes does not match output")
            if int(manifest.get("total_frames", -1)) != expected_frames:
                errors.append("task manifest total_frames does not match output")
            if int(manifest.get("total_tasks", -1)) != total_tasks:
                errors.append("task manifest total_tasks does not match output")

    return {
        "ok": not errors,
        "errors": errors,
        "episodes": len(rows),
        "frames": data_rows,
        "tasks": total_tasks,
        "data_files": len(data_files),
        "video_files": len(video_files),
        "empty_image_dirs_removed": remove_empty_image_dirs(root),
    }


def atomic_replace_directory(staging: Path, final_dir: Path) -> None:
    """Swap a validated staging tree into place and restore the old tree on failure."""
    backup = final_dir.with_name(final_dir.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    had_existing = final_dir.exists()
    if had_existing:
        final_dir.rename(backup)
    try:
        staging.rename(final_dir)
    except Exception:
        if had_existing and backup.exists() and not final_dir.exists():
            backup.rename(final_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def partition_sidecar(
    merged_sidecar: dict[str, Any],
    group: dict[str, Any],
    partition_repo_id: str,
) -> dict[str, Any]:
    selected = [int(item["global_episode_index"]) for item in group["episodes"]]
    remap = {old: new for new, old in enumerate(selected)}
    partition = deepcopy(merged_sidecar)
    partition_sessions = []
    for session in merged_sidecar.get("sessions", []):
        episodes = []
        for episode in session.get("episodes", []):
            old_index = int(episode["global_episode_index"])
            if old_index not in remap:
                continue
            item = deepcopy(episode)
            item["source_global_episode_index"] = old_index
            item["global_episode_index"] = remap[old_index]
            episodes.append(item)
        if not episodes:
            continue
        entry = deepcopy(session)
        entry["episode_index_offset"] = episodes[0]["global_episode_index"]
        entry["num_episodes"] = len(episodes)
        entry["episodes"] = episodes
        if "quality_control" in entry:
            entry["source_quality_control"] = entry["quality_control"]
        entry["quality_control"] = {
            "episodes_checked": len(episodes),
            "episodes_kept": len(episodes),
            "episodes_dropped": 0,
            "dropped_episodes": [],
            "scope": "task partition after the source session quality gate",
        }
        partition_sessions.append(entry)

    partition.update(
        {
            "repo_id": partition_repo_id,
            "parent_repo_id": merged_sidecar.get("repo_id"),
            "num_sessions": len(partition_sessions),
            "total_episodes": int(group["total_episodes"]),
            "total_frames": int(group["total_frames"]),
            "sessions": partition_sessions,
            "task_partition": {
                "task": group["task"],
                "slug": group["slug"],
                "source_global_episode_indices": selected,
            },
        }
    )
    return partition


def create_task_partitions(
    source_root: Path,
    source_repo_id: str,
    source_sidecar: dict[str, Any],
    task_inventory: list[dict[str, Any]],
    output_root: Path,
    video_codec: str,
) -> list[dict[str, Any]]:
    """Build independent LeRobot v3 datasets for each one-task episode group.

    LeRobot 0.4.4's public ``split_dataset`` hard-codes AV1 when a shared video
    file must be filtered. The project pins 0.4.4, so we use the same official
    implementation helpers while explicitly preserving our release codec.
    """
    try:
        from lerobot.datasets.dataset_tools import (
            _copy_and_reindex_data,
            _copy_and_reindex_episodes_metadata,
            _copy_and_reindex_videos,
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError as exc:
        raise SystemExit(
            "LeRobot 0.4.4 dataset partition helpers are unavailable. "
            "Recreate the pinned `lerobot` environment from requirements_lerobot.txt."
        ) from exc

    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_root)
    source_dataset = SimpleNamespace(root=source_root, meta=source_meta)
    output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for position, group in enumerate(task_inventory, start=1):
        slug = group["slug"]
        partition_repo_id = f"{source_repo_id}-{slug}"
        partition_root = output_root / slug
        episode_indices = sorted(
            int(item["global_episode_index"]) for item in group["episodes"]
        )
        episode_mapping = {
            old_index: new_index for new_index, old_index in enumerate(episode_indices)
        }
        emit_progress(
            stage="partition_task",
            task=group["task"],
            task_index=position,
            tasks=len(task_inventory),
            episodes=len(episode_indices),
        )

        destination_meta = LeRobotDatasetMetadata.create(
            repo_id=partition_repo_id,
            fps=source_meta.fps,
            features=source_meta.features,
            robot_type=source_meta.robot_type,
            root=partition_root,
            use_videos=len(source_meta.video_keys) > 0,
            chunks_size=source_meta.chunks_size,
            data_files_size_in_mb=source_meta.data_files_size_in_mb,
            video_files_size_in_mb=source_meta.video_files_size_in_mb,
        )
        video_metadata = None
        if source_meta.video_keys:
            video_metadata = _copy_and_reindex_videos(
                source_dataset,
                destination_meta,
                episode_mapping,
                vcodec=video_codec,
                pix_fmt="yuv420p",
            )
        data_metadata = _copy_and_reindex_data(
            source_dataset, destination_meta, episode_mapping
        )
        _copy_and_reindex_episodes_metadata(
            source_dataset,
            destination_meta,
            episode_mapping,
            data_metadata,
            video_metadata,
        )

        partition_meta_dir = partition_root / "meta"
        partition_meta_dir.mkdir(parents=True, exist_ok=True)
        sidecar = partition_sidecar(source_sidecar, group, partition_repo_id)
        (partition_meta_dir / "pico_ego_release.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        partition_group = deepcopy(group)
        for new_index, episode in enumerate(partition_group["episodes"]):
            old_index = int(episode["global_episode_index"])
            episode["source_global_episode_index"] = old_index
            episode["global_episode_index"] = new_index
        single_task_manifest = task_manifest(partition_repo_id, [partition_group])
        (partition_meta_dir / TASK_MANIFEST_NAME).write_text(
            json.dumps(single_task_manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        validation = validate_dataset_tree(
            partition_root,
            expected_episodes=int(group["total_episodes"]),
            expected_frames=int(group["total_frames"]),
            expected_features=source_meta.features,
        )
        if not validation["ok"]:
            raise RuntimeError(
                f"Task partition {slug} failed validation: "
                + json.dumps(validation["errors"], ensure_ascii=False)
            )
        results.append(
            {
                "task": group["task"],
                "slug": slug,
                "repo_id": partition_repo_id,
                "output_dir": str(partition_root),
                "episodes": int(group["total_episodes"]),
                "frames": int(group["total_frames"]),
                "validation": validation,
            }
        )

    index_partitions = []
    for item in results:
        index_item = deepcopy(item)
        index_item.pop("output_dir", None)
        index_item["path"] = item["slug"]
        index_partitions.append(index_item)
    (output_root / "index.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "parent_repo_id": source_repo_id,
                "video_codec": video_codec,
                "partitions": index_partitions,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return results


def consolidate(args: argparse.Namespace) -> dict[str, Any]:
    only = [s.strip() for s in args.sessions.split(",") if s.strip()] if args.sessions else None
    sessions = discover_sessions(args.sessions_root, only)
    problems = check_compatibility(sessions)
    task_inventory = build_task_inventory(sessions)
    task_summary = [
        {
            "task": item["task"],
            "slug": item["slug"],
            "episodes": item["total_episodes"],
            "frames": item["total_frames"],
            "sessions": len(item["session_ids"]),
        }
        for item in task_inventory
    ]

    summary: dict[str, Any] = {
        "sessions_root": str(args.sessions_root),
        "output_dir": str(args.output_dir),
        "repo_id": args.repo_id,
        "num_sessions": len(sessions),
        "session_ids": [s["session_id"] for s in sessions],
        "total_episodes": sum(s["total_episodes"] for s in sessions),
        "total_frames": sum(s["total_frames"] for s in sessions),
        "incompatible": bool(problems),
        "incompatible_detail": problems,
        "strict_preflight": True,
        "task_groups": task_summary,
        "task_output_root": str(args.task_output_root) if args.task_output_root else None,
    }

    if args.dry_run:
        return summary

    if problems:
        raise SystemExit(f"Cannot consolidate: incompatible sessions: {json.dumps(problems)}")
    if len(sessions) < 1:
        raise SystemExit("Nothing to consolidate (no sessions found).")

    try:
        from lerobot.datasets.aggregate import aggregate_datasets
    except ImportError as exc:
        raise SystemExit(
            "LeRobot is not installed here. Run this under the `lerobot` env (see README)."
        ) from exc

    assert_safe_output_dir(args.output_dir)
    final_dir = args.output_dir
    sessions_root_resolved = args.sessions_root.resolve()
    if sessions_root_resolved in final_dir.resolve().parents:
        raise SystemExit(
            "--output-dir must be outside --sessions-root; otherwise the next run would "
            "rediscover the release as another input session. Use releases/<name>."
        )
    if final_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output dir already exists: {final_dir}. Pass --overwrite to replace it.")
    staging = final_dir.parent / (final_dir.name + ".consolidating")
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)

    task_final = args.task_output_root
    task_staging: Path | None = None
    if task_final is not None:
        assert_safe_output_dir(task_final)
        if _paths_overlap(final_dir, task_final):
            raise SystemExit("--task-output-root must not equal, contain, or be inside --output-dir")
        if sessions_root_resolved in task_final.resolve().parents:
            raise SystemExit("--task-output-root must be outside --sessions-root")
        if task_final.exists() and not args.overwrite:
            raise SystemExit(
                f"Task output root already exists: {task_final}. Pass --overwrite to replace it."
            )
        task_staging = task_final.parent / (task_final.name + ".partitioning")
        task_staging.parent.mkdir(parents=True, exist_ok=True)
        if task_staging.exists():
            shutil.rmtree(task_staging)

    emit_progress(stage="start", sessions=len(sessions), total=len(sessions))
    aggregate_datasets(
        repo_ids=[s["repo_id"] for s in sessions],
        aggr_repo_id=args.repo_id,
        roots=[s["root"] for s in sessions],
        aggr_root=staging,
        data_files_size_in_mb=args.data_files_size_mb,
        video_files_size_in_mb=args.video_files_size_mb,
        chunk_size=args.chunk_size,
    )
    emit_progress(stage="aggregated", sessions=len(sessions), total=len(sessions))

    # Merge project metadata and write an explicit task-to-episode inventory.
    merged_sidecar = merge_sidecars(sessions, args.repo_id)
    meta_dir = staging / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "pico_ego_release.json").write_text(
        json.dumps(merged_sidecar, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    manifest = task_manifest(args.repo_id, task_inventory)
    (meta_dir / TASK_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    emit_progress(stage="validate", sessions=len(sessions), total=len(sessions))
    validation = validate_dataset_tree(
        staging,
        expected_episodes=summary["total_episodes"],
        expected_frames=summary["total_frames"],
        expected_features=sessions[0]["feature_schema"],
    )
    if not validation["ok"]:
        raise RuntimeError(
            "Consolidated staging dataset failed validation: "
            + json.dumps(validation["errors"], ensure_ascii=False)
        )

    task_partitions: list[dict[str, Any]] = []
    if task_staging is not None:
        codec = merged_sidecar.get("video_codec") or "h264"
        if codec == "auto":
            codec = "h264"
        task_partitions = create_task_partitions(
            source_root=staging,
            source_repo_id=args.repo_id,
            source_sidecar=merged_sidecar,
            task_inventory=task_inventory,
            output_root=task_staging,
            video_codec=codec,
        )
        for item in task_partitions:
            item["output_dir"] = str(task_final / item["slug"])
        emit_progress(stage="partitions_done", tasks=len(task_partitions), total=len(task_inventory))

    atomic_replace_directory(staging, final_dir)
    if task_staging is not None and task_final is not None:
        atomic_replace_directory(task_staging, task_final)
    emit_progress(stage="done", sessions=len(sessions), total=len(sessions))

    # Read back the aggregated info.json for the authoritative totals.
    info = _load_json(final_dir / "meta" / "info.json")
    summary["written_episodes"] = int(info.get("total_episodes", 0))
    summary["written_frames"] = int(info.get("total_frames", 0))
    summary["sidecar_meta"] = str(final_dir / "meta" / "pico_ego_release.json")
    summary["task_manifest"] = str(final_dir / "meta" / TASK_MANIFEST_NAME)
    summary["stereo"] = merged_sidecar["stereo"]
    summary["video_codec"] = merged_sidecar["video_codec"]
    summary["validation"] = validation
    summary["task_partitions"] = task_partitions
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions-root", type=Path, default=Path("lerobot_v3"),
                        help="Directory holding per-session dataset dirs (default: lerobot_v3).")
    parser.add_argument("--sessions", default=None,
                        help="Comma-separated session ids to include (default: all found).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the consolidated release dataset.")
    parser.add_argument("--repo-id", default="pico_ego",
                        help="repo_id for the merged dataset (default: pico_ego).")
    parser.add_argument("--video-files-size-mb", type=int, default=512)
    parser.add_argument("--data-files-size-mb", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Maximum files per LeRobot chunk (default: 1000).")
    parser.add_argument(
        "--task-output-root",
        type=Path,
        default=None,
        help=(
            "Optionally create one independent LeRobot v3 dataset per normalized task "
            "under this directory. Mixed-task video files are re-encoded."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dir.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run strict compatibility/file/task checks and report the planned task groups; write nothing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video_files_size_mb <= 0 or args.data_files_size_mb <= 0 or args.chunk_size <= 0:
        raise SystemExit("file size limits and --chunk-size must be positive")
    summary = consolidate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if args.dry_run and summary.get("incompatible"):
        sys.exit(1)


if __name__ == "__main__":
    main()
