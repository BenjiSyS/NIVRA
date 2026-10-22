"""ffprobe / ffmpeg helpers. Frames only ever travel through pipes, one clip window at a time."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MediaError(RuntimeError):
    pass


def need_tools():
    for t in ("ffmpeg", "ffprobe"):
        if shutil.which(t) is None:
            raise MediaError(f"{t} not found on PATH")


@dataclass
class VideoInfo:
    codec: str
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int | None


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def _fraction(s: str) -> float:
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else 0.0
        return float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path) -> VideoInfo:
    """ffprobe validation. Raises MediaError for unreadable / video-less / zero-length files."""
    r = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)], 60)
    if r.returncode != 0:
        raise MediaError(f"ffprobe failed: {r.stderr.decode(errors='replace').strip()[:200]}")
    try:
        d = json.loads(r.stdout or b"{}")
    except json.JSONDecodeError as e:
        raise MediaError(f"ffprobe output unreadable: {e}")
    vs = [s for s in d.get("streams", []) if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) != 1]
    if not vs:
        raise MediaError("no video stream")
    s = vs[0]
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    fps = _fraction(s.get("avg_frame_rate", "0")) or _fraction(s.get("r_frame_rate", "0"))
    dur = 0.0
    for v in (s.get("duration"), d.get("format", {}).get("duration")):
        try:
            dur = float(v)
            break
        except (TypeError, ValueError):
            continue
    if w <= 0 or h <= 0:
        raise MediaError("invalid resolution")
    if fps <= 0:
        raise MediaError("invalid frame rate")
    if dur <= 0:
        raise MediaError("invalid duration")
    nb = s.get("nb_frames")
    return VideoInfo(s.get("codec_name", "?"), w, h, fps, dur, int(nb) if nb and str(nb).isdigit() else None)


_PTS = re.compile(r"pts_time:\s*([0-9.]+)")


def detect_cuts(path: Path, threshold: float) -> list[float]:
    """Scene-change timestamps (seconds, start of each new scene) from ffmpeg's `scene` score.

    The whole file is decoded once at reduced resolution (cheap, memory-bounded). A non-zero exit means the
    stream is corrupt -> MediaError.
    """
    vf = f"scale=160:-2:flags=fast_bilinear,select='gt(scene,{threshold})',showinfo"
    r = _run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-i", str(path), "-an", "-sn", "-vf", vf, "-f", "null", "-"])
    err = r.stderr.decode(errors="replace")
    if r.returncode != 0:
        raise MediaError(f"decode failed during scene detection: {err.strip().splitlines()[-1][:200] if err.strip() else r.returncode}")
    return sorted({float(m.group(1)) for m in _PTS.finditer(err)})


def scenes_from_cuts(duration: float, cuts: list[float], min_len: float = 0.0) -> list[tuple[float, float]]:
    pts = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    return [(a, b) for a, b in zip(pts, pts[1:]) if b - a > 1e-6 and b - a >= min_len]


def read_clip(path: Path, start: float, n_frames: int, fps: float, width: int, height: int) -> np.ndarray | None:
    """Decode `n_frames` RGB frames at `fps` starting at `start`, aspect-preserving resize (cover) + center crop.

    Returns uint8 [n,H,W,3], or None if the stream ended early. Reads straight into one preallocated array.
    """
    vf = (f"fps={fps:.6f},scale={width}:{height}:force_original_aspect_ratio=increase:flags=bicubic,"
          f"crop={width}:{height}")
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{start:.4f}", "-i", str(path), "-an", "-sn",
           "-vf", vf, "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    out = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    frame_bytes = height * width * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    got = 0
    try:
        view = memoryview(out.reshape(-1))
        total = n_frames * frame_bytes
        pos = 0
        while pos < total:
            n = p.stdout.readinto(view[pos:total])
            if not n:
                break
            pos += n
        got = pos // frame_bytes
    finally:
        p.stdout.close()
        if got < n_frames:
            p.kill()
        err = p.stderr.read()
        p.stderr.close()
        p.wait()
    if got < n_frames:
        return None
    if err.strip():
        raise MediaError(f"decode warnings while reading clip: {err.decode(errors='replace').strip()[:200]}")
    return out


def read_gray_window(path: Path, start: float, seconds: float, fps: float, src_width: int, src_height: int,
                     width: int = 96) -> np.ndarray | None:
    """Decode `seconds` of video from `start` at constant `fps` (the SOURCE frame rate: nothing is resampled away)
    as small gray frames uint8 [n,h,w] -- used for animation hold detection. None if fewer than 2 frames decode."""
    n = max(2, int(round(seconds * fps)))
    h = max(2, int(round(width * src_height / src_width / 2.0)) * 2)
    vf = f"fps={fps:.6f},scale={width}:{h}:flags=area,format=gray"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{start:.4f}", "-i", str(path), "-an", "-sn",
           "-vf", vf, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        raise MediaError(f"gray decode failed: {r.stderr.decode(errors='replace').strip()[:200]}")
    cnt = len(r.stdout) // (width * h)
    if cnt < 2:
        return None
    return np.frombuffer(r.stdout[: cnt * width * h], dtype=np.uint8).reshape(cnt, h, width)
