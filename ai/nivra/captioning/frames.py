"""Frame sampling with ffmpeg."""
from __future__ import annotations

import json
from pathlib import Path

from . import CaptionError
from .subtitles import run_tool, short_err


def probe_duration(video: Path) -> float:
    r = run_tool(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=duration", "-of", "json", str(video)],
                 "inspect videos")
    if r.returncode != 0:
        raise CaptionError(f"ffprobe could not read the file: {short_err(r.stderr)}")
    data = json.loads(r.stdout or "{}")
    cands = [data.get("format", {}).get("duration")] + [s.get("duration") for s in data.get("streams", [])]
    for c in cands:
        try:
            d = float(c)
            if d > 0:
                return d
        except (TypeError, ValueError):
            pass
    raise CaptionError("could not determine the video duration")


def frame_times(duration: float, n: int) -> list[float]:
    """n evenly spaced timestamps (centres of n equal slices of the middle 90%)."""
    if n < 1:
        raise ValueError("frames must be >= 1")
    lo, span = 0.05 * duration, 0.90 * duration
    return [lo + span * (i + 0.5) / n for i in range(n)]


def extract_frames(video: Path, n: int, out_dir: Path, max_side: int = 768) -> list[Path]:
    return extract_frames_timed(video, n, out_dir, max_side)[0]


def extract_frames_timed(video: Path, n: int, out_dir: Path, max_side: int = 768) -> tuple[list[Path], list[float]]:
    """(frame files, their timestamps in seconds)."""
    times = frame_times(probe_duration(video), n)
    paths = []
    for i, t in enumerate(times):
        out = out_dir / f"f{i:02d}.jpg"
        vf = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
        r = run_tool(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                      "-vf", vf, "-q:v", "2", str(out)], "extract frames")
        if r.returncode != 0 or not out.exists():
            raise CaptionError(f"ffmpeg could not extract a frame at {t:.1f}s: {short_err(r.stderr)}")
        paths.append(out)
    return paths, times
