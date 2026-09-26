#!/usr/bin/env python3
"""Sync paired Pico tracking/video files from Download into the project."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


TRACKING_RE = re.compile(r"^trackingData_(?P<session>\d{8}_\d{6})\.txt$")
VIDEO_RE = re.compile(r"^CameraRecord_(?P<session>\d{8}_\d{6})\.mp4$")
SESSION_DATE_RE = re.compile(r"^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_\d{6}$")
STATUS_UNPROCESSED = "unprocessed"
STATUS_PROCESSED = "processed"
STATUS_FIELDS = ["file_name", "tracking_file", "device_group", "status"]
GROUP_PREFIX_RE = re.compile(r"^(group\d+)_")


def normalize_processing_status(value: str) -> str:
    """Accept legacy Chinese CSV values without resetting completed sessions."""
    return {
        "\u672a\u5904\u7406": STATUS_UNPROCESSED,
        "\u5df2\u5904\u7406": STATUS_PROCESSED,
    }.get(value, value)


def normalize_device_group(value: str) -> str:
    """Read the legacy label used for recordings without a device prefix."""
    return "ungrouped" if value == "\u672a\u5206\u7ec4" else value


# Optional installation-specific routing; device identifiers stay out of Git.
DEVICE_GROUPS_FILE = Path(os.environ.get(
    "PICO_DEVICE_GROUPS_FILE",
    Path(__file__).resolve().parents[1] / "configs" / "pico_devices.local.json",
))


@dataclass(frozen=True)
class SourcePair:
    session_id: str
    tracking: Path | None
    video: Path | None


def find_pico_download(
    run_user: Path = Path("/run/user"),
    local_roots: tuple[Path, ...] = (Path("/media"), Path("/mnt")),
) -> Path | None:
    """Probe mounts separately so one stale MTP mount cannot abort discovery."""
    errors: list[str] = []
    mtp_mounts: list[Path] = []

    def children(directory: Path) -> list[Path]:
        try:
            return sorted(directory.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            return []
        except OSError as exc:
            errors.append(f"{directory}: {exc}")
            return []

    def readable(directory: Path) -> bool:
        try:
            # is_dir alone can succeed on a disconnected GVFS mount. Actually
            # ask the device to list this directory, without reading file data.
            with os.scandir(directory) as entries:
                next(entries, None)
            return True
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError as exc:
            errors.append(f"{directory}: {exc}")
            return False

    for user_root in children(run_user):
        for mount in children(user_root / "gvfs"):
            if not mount.name.startswith("mtp:host="):
                continue
            mtp_mounts.append(mount)
            # The first spelling is the Chinese-localized Android storage path.
            for relative in ("\u5185\u90e8\u5171\u4eab\u5b58\u50a8\u7a7a\u95f4/Download", "Internal shared storage/Download", "Download"):
                candidate = mount / relative
                if readable(candidate):
                    return candidate

    for base in local_roots:
        # os.walk skips individual inaccessible branches, unlike Path.glob on
        # Python 3.10, which can propagate an MTP I/O error mid-iteration.
        for directory, _, _ in os.walk(base, onerror=lambda exc: errors.append(str(exc))):
            candidate = Path(directory)
            if candidate.name == "Download" and readable(candidate):
                return candidate
    if mtp_mounts or errors:
        detail = errors[0] if errors else str(mtp_mounts[0])
        raise OSError(
            "Cannot read Pico Download: the MTP mount is unavailable or internal storage is not accessible. "
            "Wake and unlock the Pico, allow USB file transfer, reconnect USB, and retry."
            f"\nMount details: {detail}"
        )
    return None


def discover_pairs(source_dir: Path) -> list[SourcePair]:
    tracking: dict[str, Path] = {}
    videos: dict[str, Path] = {}

    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        track_match = TRACKING_RE.match(path.name)
        if track_match:
            tracking[track_match.group("session")] = path
            continue
        video_match = VIDEO_RE.match(path.name)
        if video_match:
            videos[video_match.group("session")] = path

    session_ids = sorted(set(tracking) | set(videos))
    return [
        SourcePair(session_id=sid, tracking=tracking.get(sid), video=videos.get(sid))
        for sid in session_ids
    ]


def infer_device_id(source_dir: Path) -> str:
    """Extract the stable Pico MTP host/device identifier when available."""
    match = re.search(r"host=([^/]+)", str(source_dir))
    if match:
        return match.group(1)
    return source_dir.name or "unknown_device"


def infer_device_group(source_dir: Path) -> str:
    if not DEVICE_GROUPS_FILE.exists():
        # A fresh checkout accepts any Pico or an already copied Download folder.
        return "group0"
    groups = json.loads(DEVICE_GROUPS_FILE.read_text(encoding="utf-8"))
    if not isinstance(groups, dict) or any(
        not isinstance(group, str) or re.fullmatch(r"group\d+", group) is None
        for group in groups.values()
    ):
        raise ValueError(f"Invalid device mapping in {DEVICE_GROUPS_FILE}; expected device IDs mapped to group0, group1, ...")
    device_id = infer_device_id(source_dir)
    group = groups.get(device_id)
    if group is None:
        raise ValueError(
            f"Unknown Pico device '{device_id}'. Add it to {DEVICE_GROUPS_FILE} "
            "before importing; refusing to create an ungrouped session."
        )
    return group


def group_from_file_name(file_name: str) -> str:
    match = GROUP_PREFIX_RE.match(file_name)
    return match.group(1) if match else "ungrouped"


def session_date(session_id: str) -> date | None:
    match = SESSION_DATE_RE.match(session_id)
    if not match:
        return None
    return date(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
    )


def parse_date_value(value: str | None) -> date | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d").date()
    raise argparse.ArgumentTypeError(
        f"Invalid date '{value}'. Use YYYYMMDD, for example 20260712."
    )


def filter_pairs(
    pairs: list[SourcePair],
    only_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[SourcePair]:
    if only_date is not None:
        start_date = only_date
        end_date = only_date
    if start_date is None and end_date is None:
        return pairs
    out = []
    for pair in pairs:
        day = session_date(pair.session_id)
        if day is None:
            continue
        if start_date is not None and day < start_date:
            continue
        if end_date is not None and day > end_date:
            continue
        out.append(pair)
    return out


def copy_if_needed(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size and filecmp.cmp(src, dst, shallow=False):
        return "skipped"

    # Use a unique temporary path so two accidental import requests cannot
    # remove or replace each other's in-progress copy.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    os.close(fd)
    Path(tmp_name).unlink(missing_ok=True)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp)
        if tmp.stat().st_size != src.stat().st_size:
            raise IOError(
                f"Incomplete copy for {src.name}: {tmp.stat().st_size} / {src.stat().st_size} bytes"
            )
        tmp.replace(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return "copied"


def delete_source_file(path: Path | None) -> str:
    """Delete a Pico source file only after its local copy was verified."""
    if path is None:
        return "missing"
    try:
        path.unlink()
        return "deleted"
    except FileNotFoundError:
        return "already_missing"
    except OSError as exc:
        return f"failed:{exc}"


def file_size(path: Path | None) -> int | None:
    return path.stat().st_size if path and path.exists() else None


def file_mtime_iso(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def sync_pairs(
    source_dir: Path,
    raw_dir: Path,
    delete_source: bool = True,
    only_date: date | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    device_id = infer_device_id(source_dir)
    device_group = infer_device_group(source_dir)
    local_prefix = f"{device_group}_" if device_group != "group0" else ""
    for pair in filter_pairs(discover_pairs(source_dir), only_date, start_date, end_date):
        session_dir = raw_dir / pair.session_id
        tracking_dst = (
            session_dir / f"{local_prefix}{pair.tracking.name}" if pair.tracking is not None else None
        )
        video_dst = session_dir / f"{local_prefix}{pair.video.name}" if pair.video is not None else None
        raw_existed_before = session_dir.exists()
        tracking_existed_before = bool(tracking_dst and tracking_dst.exists())
        video_existed_before = bool(video_dst and video_dst.exists())

        tracking_status = "missing"
        video_status = "missing"
        if pair.tracking is not None and tracking_dst is not None:
            tracking_status = copy_if_needed(pair.tracking, tracking_dst)
        if pair.video is not None and video_dst is not None:
            video_status = copy_if_needed(pair.video, video_dst)

        source_tracking_status = "kept"
        source_video_status = "kept"
        # Do not remove an unpaired source.  Both local files must be safely
        # present before deleting either Pico file.
        if delete_source and pair.tracking is not None and pair.video is not None:
            if tracking_status in {"copied", "skipped"} and video_status in {"copied", "skipped"}:
                source_tracking_status = delete_source_file(pair.tracking)
                source_video_status = delete_source_file(pair.video)

        rows.append(
            {
                "session_id": pair.session_id,
                "device_id": device_id,
                "device_group": device_group,
                "video_file": video_dst.name if video_dst else "",
                "tracking_file": tracking_dst.name if tracking_dst else "",
                "paired": pair.tracking is not None and pair.video is not None,
                "tracking_status": tracking_status,
                "video_status": video_status,
                "source_tracking_status": source_tracking_status,
                "source_video_status": source_video_status,
                "raw_existed_before": raw_existed_before,
                "tracking_existed_before": tracking_existed_before,
                "video_existed_before": video_existed_before,
                "tracking_source": str(pair.tracking or ""),
                "video_source": str(pair.video or ""),
                "tracking_local": str(tracking_dst or ""),
                "video_local": str(video_dst or ""),
                "tracking_bytes": file_size(pair.tracking),
                "video_bytes": file_size(pair.video),
                "tracking_mtime": file_mtime_iso(pair.tracking),
                "video_mtime": file_mtime_iso(pair.video),
            }
        )
    return rows


def write_manifest(rows: list[dict[str, object]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "session_id",
        "device_id",
        "device_group",
        "video_file",
        "tracking_file",
        "paired",
        "tracking_status",
        "video_status",
        "source_tracking_status",
        "source_video_status",
        "raw_existed_before",
        "tracking_existed_before",
        "video_existed_before",
        "tracking_source",
        "video_source",
        "tracking_local",
        "video_local",
        "tracking_bytes",
        "video_bytes",
        "tracking_mtime",
        "video_mtime",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_processing_status(rows: list[dict[str, object]], status_path: Path) -> None:
    """Register imported videos without changing completed statuses."""
    existing: dict[str, dict[str, str]] = {}
    if status_path.exists():
        with status_path.open("r", encoding="utf-8", newline="") as f:
            existing = {
                row.get("file_name", ""): row
                for row in csv.DictReader(f)
                if row.get("file_name", "")
            }
    # The status table describes locally retained raw videos, not historical
    # Pico files that have already been removed or manually deleted.
    local_video_names = {
        path.name for path in status_path.parent.joinpath("raw").glob("*/*CameraRecord_*.mp4")
    }
    existing = {name: row for name, row in existing.items() if name in local_video_names}
    for row in rows:
        if row.get("paired"):
            file_name = str(row["video_file"])
            current = existing.get(file_name, {})
            current.update({
                "file_name": file_name,
                "tracking_file": str(row["tracking_file"]),
                "device_group": group_from_file_name(file_name) if group_from_file_name(file_name) != "ungrouped" else str(row["device_group"]),
            })
            current.setdefault("status", STATUS_UNPROCESSED)
            existing[file_name] = current
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for file_name in sorted(existing):
            row = existing[file_name]
            status = normalize_processing_status(row.get("status", STATUS_UNPROCESSED))
            if status not in {STATUS_UNPROCESSED, STATUS_PROCESSED}:
                status = STATUS_UNPROCESSED
            writer.writerow({
                "file_name": file_name,
                "tracking_file": row.get("tracking_file", ""),
                "device_group": normalize_device_group(row.get("device_group", "")),
                "status": status,
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Pico Download directory. If omitted, the script auto-detects MTP.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--manifest", type=Path, default=Path("/tmp/pico_ego_collector/manifest.csv"))
    parser.add_argument(
        "--status-table",
        type=Path,
        default=None,
        help="Reserved for the web status table; defaults to <raw-dir>/../pico_processing_status.csv.",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep files on Pico instead of deleting them after verified copy.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Import only one session date parsed from session id. Format: YYYYMMDD.",
    )
    parser.add_argument("--start-date", default=None, help="Inclusive start date filter. Format: YYYYMMDD.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date filter. Format: YYYYMMDD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        only_date = parse_date_value(args.date)
        start_date = parse_date_value(args.start_date)
        end_date = parse_date_value(args.end_date)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if start_date and end_date and start_date > end_date:
        raise SystemExit("--start-date must be <= --end-date")

    try:
        source_dir = args.source or find_pico_download()
        if source_dir is None:
            raise SystemExit("Pico Download not found. Connect and unlock the Pico, allow USB file transfer, or specify the Download path.")
        rows = sync_pairs(
            source_dir,
            args.raw_dir,
            delete_source=not args.keep_source,
            only_date=only_date,
            start_date=start_date,
            end_date=end_date,
        )
    except OSError as exc:
        raise SystemExit(
            f"Import incomplete: {exc}\nCheck the Pico file-transfer connection and local directory. Copied files are kept; retry after fixing the issue."
        ) from exc
    write_manifest(rows, args.manifest)
    status_path = args.status_table or args.raw_dir.resolve().parent / "pico_processing_status.csv"
    update_processing_status(rows, status_path)
    paired = sum(1 for row in rows if row["paired"])
    print(f"source={source_dir}")
    print(f"device_id={infer_device_id(source_dir)}")
    print(f"device_group={infer_device_group(source_dir)}")
    if only_date:
        print(f"date={only_date.strftime('%Y%m%d')}")
    if start_date or end_date:
        start_text = start_date.strftime("%Y%m%d") if start_date else ""
        end_text = end_date.strftime("%Y%m%d") if end_date else ""
        print(f"date_range={start_text}..{end_text}")
    print(f"sessions={len(rows)} paired={paired}")
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
