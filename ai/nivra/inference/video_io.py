"""MP4 writing / probing through ffmpeg + ffprobe subprocesses."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    pass


def write_mp4(frames: np.ndarray, path, fps: int = 8, upscale: int = 4) -> Path:
    """frames: uint8 [T,H,W,3]. H.264 / yuv420p. upscale = integer nearest-neighbour scale applied to the file only."""
    if shutil.which("ffmpeg") is None:
        raise FFmpegError("ffmpeg not found on PATH")
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    T, H, W, _ = frames.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-"]
    if upscale > 1:
        cmd += ["-vf", f"scale={W * upscale}:{H * upscale}:flags=neighbor"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "veryfast", "-movflags", "+faststart", str(path)]
    p = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise FFmpegError(p.stderr.decode(errors="replace")[:500])
    return path


def probe_mp4(path) -> dict:
    """Returns {'frames': counted frames, 'width', 'height', 'fps', 'codec'} using ffprobe -count_frames."""
    cmd = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
           "stream=nb_read_frames,width,height,r_frame_rate,codec_name,pix_fmt", "-of", "json", str(path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise FFmpegError(p.stderr.decode(errors="replace")[:500])
    s = json.loads(p.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"frames": int(s["nb_read_frames"]), "width": s["width"], "height": s["height"],
            "fps": float(num) / float(den), "codec": s["codec_name"], "pix_fmt": s["pix_fmt"]}


def decode_mp4(path) -> np.ndarray:
    """Decode an MP4 back to uint8 [T,H,W,3] (used by evaluation of written files)."""
    info = probe_mp4(path)
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise FFmpegError(p.stderr.decode(errors="replace")[:500])
    arr = np.frombuffer(p.stdout, dtype=np.uint8)
    return arr.reshape(info["frames"], info["height"], info["width"], 3)
