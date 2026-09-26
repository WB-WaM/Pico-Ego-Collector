"""LeRobot video worker used by the Pico exporter.

This is a separate importable module because LeRobot uses ProcessPoolExecutor
for multi-camera episode encoding and worker functions must be pickleable.
"""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
from pathlib import Path


def encode_video_worker(
    video_key: str,
    episode_index: int,
    root: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    encoder_threads: int | None = None,
) -> Path:
    """Encode one episode video with configurable FFmpeg options."""
    from PIL import Image
    import av
    from lerobot.datasets.video_utils import _get_codec_options, resolve_vcodec

    root = Path(root)
    temp_path = Path(tempfile.mkdtemp(dir=root)) / f"{video_key}_{episode_index:03d}.mp4"
    image_template = root / "images" / video_key / f"episode-{episode_index:06d}" / "frame-*.png"
    input_list = sorted(
        glob.glob(str(image_template)),
        key=lambda item: int(Path(item).stem.split("-")[-1]),
    )
    if not input_list:
        raise FileNotFoundError(f"No images found for {video_key} episode {episode_index}")

    with Image.open(input_list[0]) as first_image:
        width, height = first_image.size
    vcodec = resolve_vcodec(vcodec)
    gop = int(os.environ.get("PICO_EXPORT_VIDEO_GOP", "2"))
    crf = int(os.environ.get("PICO_EXPORT_VIDEO_CRF", "30"))
    preset = os.environ.get("PICO_EXPORT_VIDEO_PRESET", "veryfast").strip()
    options = _get_codec_options(vcodec, g=gop, crf=crf, preset=None)
    if preset and vcodec in {"h264", "hevc"}:
        options["preset"] = preset
    if vcodec == "libsvtav1":
        options["preset"] = preset if preset.isdigit() else "12"
    if encoder_threads is not None:
        if vcodec == "libsvtav1":
            options["svtav1-params"] = f"lp={encoder_threads}"
        else:
            options["threads"] = str(encoder_threads)

    with av.open(str(temp_path), "w") as output:
        stream = output.add_stream(vcodec, fps, options=options)
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        for image_path in input_list:
            with Image.open(image_path) as image:
                packet = stream.encode(av.VideoFrame.from_image(image.convert("RGB")))
                if packet:
                    output.mux(packet)
        packet = stream.encode()
        if packet:
            output.mux(packet)

    shutil.rmtree(image_template.parent)
    return temp_path
