#!/usr/bin/env python3
"""Render a Unitree G1 MuJoCo preview video from GMR retargeted qpos."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("PICO_EGO_TMP", "/tmp/pico_ego_collector"))
DEFAULT_XML = PROJECT_ROOT / "third-party" / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"


def discover_motion(session_id: str | None, input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path
    if not session_id:
        raise SystemExit("Use --session or --input.")
    path = TMP_ROOT / session_id / "retargeted" / f"{session_id}_unitree_g1.pkl"
    if not path.exists():
        path = PROJECT_ROOT / "data" / "retargeted" / f"{session_id}_unitree_g1.pkl"
    if not path.exists():
        raise SystemExit(f"G1 retarget pkl not found: {path}")
    return path


def load_motion(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    fps = float(data.get("fps", 30.0) or 30.0)
    root = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float64)
    dof = np.asarray(data["dof_pos"], dtype=np.float64)
    stored_times = np.asarray(data.get("times", []), dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (T, 3), got {root.shape}")
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"Expected root_rot shape (T, 4), got {root_rot_xyzw.shape}")
    if dof.ndim != 2:
        raise ValueError(f"Expected dof_pos shape (T, D), got {dof.shape}")
    n = min(len(root), len(root_rot_xyzw), len(dof))
    root = root[:n]
    root_rot_wxyz = root_rot_xyzw[:n][:, [3, 0, 1, 2]]
    norm = np.linalg.norm(root_rot_wxyz, axis=1, keepdims=True)
    root_rot_wxyz = root_rot_wxyz / np.maximum(norm, 1e-8)
    qpos = np.concatenate([root, root_rot_wxyz, dof[:n]], axis=1)
    if len(stored_times) >= n:
        times = stored_times[:n] - stored_times[0]
    else:
        times = np.arange(n, dtype=np.float64) / max(fps, 1e-8)
    return fps, qpos, times


def sample_indices(
    source_times: np.ndarray,
    preview_fps: float,
    start_sec: float,
    end_sec: float | None,
) -> np.ndarray:
    if len(source_times) == 0:
        raise ValueError("Cannot sample an empty motion.")
    duration = float(source_times[-1])
    start = max(0.0, start_sec)
    end = min(duration, end_sec if end_sec is not None else duration)
    if end <= start:
        raise ValueError(f"Invalid time range: start={start:.3f}, end={end:.3f}")
    target_times = np.arange(start, end + 1e-9, 1.0 / preview_fps, dtype=np.float64)
    idx = np.searchsorted(source_times, target_times, side="left")
    idx = np.clip(idx, 0, len(source_times) - 1)
    previous = np.maximum(idx - 1, 0)
    choose_previous = np.abs(source_times[previous] - target_times) < np.abs(
        source_times[idx] - target_times
    )
    idx[choose_previous] = previous[choose_previous]
    return idx


# View rotated clockwise by 120° from the previous 145° view.
DEFAULT_CAMERA_AZIMUTH = 265.0


def configure_camera(camera: mujoco.MjvCamera, root: np.ndarray, azimuth: float) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [root[0], root[1], root[2] + 0.2]
    camera.distance = 3.0
    camera.azimuth = azimuth % 360.0
    camera.elevation = -12.0


def render_preview(
    motion_path: Path,
    output_path: Path,
    xml_path: Path,
    width: int,
    height: int,
    preview_fps: float,
    camera_azimuth: float,
    start_sec: float,
    end_sec: float | None,
) -> dict[str, object]:
    source_fps, qpos, source_times = load_motion(motion_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion qpos dim {qpos.shape[1]} does not match MuJoCo model nq {model.nq}")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    indices = sample_indices(source_times, preview_fps, start_sec, end_sec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=preview_fps,
        codec="libx264",
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    try:
        for idx in tqdm(indices, desc="Rendering G1 MuJoCo"):
            data.qpos[:] = qpos[idx]
            mujoco.mj_forward(model, data)
            configure_camera(camera, qpos[idx, :3], camera_azimuth)
            renderer.update_scene(data, camera=camera, scene_option=option)
            rgb = renderer.render()
            writer.append_data(rgb)
    finally:
        writer.close()
        renderer.close()

    return {
        "motion": str(motion_path),
        "xml": str(xml_path),
        "output": str(output_path),
        "source_fps": source_fps,
        "source_duration": float(source_times[-1]),
        "preview_fps": preview_fps,
        "camera_azimuth": camera_azimuth,
        "frames": int(len(indices)),
        "width": width,
        "height": height,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--preview-fps", type=float, default=15.0)
    parser.add_argument("--camera-azimuth", type=float, default=DEFAULT_CAMERA_AZIMUTH)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_path = discover_motion(args.session, args.input)
    session_id = args.session or motion_path.stem.replace("_unitree_g1", "")
    output_path = args.output or TMP_ROOT / session_id / "previews" / f"{session_id}_g1_mujoco.mp4"
    result = render_preview(
        motion_path=motion_path,
        output_path=output_path,
        xml_path=args.xml,
        width=args.width,
        height=args.height,
        preview_fps=args.preview_fps,
        camera_azimuth=args.camera_azimuth,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
