"""Long videos: draft -> refine (parallel) -> stitch (docs/architecture/LONG_VIDEO.md).

Pure logic with callbacks; the executor handlers (nivra.executor) validate params, load the bundle and map errors.
Nothing here holds more than one segment of frames in memory: the draft and the stitched video are streamed through
ffmpeg pipes, a refined segment is decoded straight into its writer.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .hd import HDPipeline, padded
from .sampler import Stopped
from .video_io import FFmpegError, Mp4Writer, RawReader, probe_fast


class LongVideoError(ValueError):
    """Bad input data (maps to INPUT_INVALID in the executor)."""


# ===================================================================== plan (must equal Go segmentPlan)
def total_frames(seconds: float, fps: int) -> int:
    """F = round(seconds * fps) + 1, rounding half away from zero like Go's math.Round."""
    return int(math.floor(round(seconds * fps, 6) + 0.5)) + 1


def segment_plan(frames: int, segment_frames: int) -> list:
    """[[start, end], ...]; neighbours share their boundary frame; N = ceil((F-1)/(S-1))."""
    if segment_frames < 2:
        raise ValueError("segment_frames must be >= 2")
    step = segment_frames - 1
    out = [[s, min(s + step, frames - 1)] for s in range(0, frames - 1, step)]
    return out or [[0, max(0, frames - 1)]]


def _noop(*_a, **_k):
    return None


# ===================================================================== ai.draft
def run_draft(pipe: HDPipeline, *, prompt: str, seed: int, seconds: float, fps: int, segment_frames: int,
              width: int, height: int, out_mp4, plan_path, style: Optional[str] = None, steps: Optional[int] = None,
              cfg_scale: Optional[float] = None, sampler: Optional[str] = None,
              on_progress: Optional[Callable] = None, should_stop: Optional[Callable] = None) -> dict:
    """Generate the whole video at draft size with the continuation chain, streaming frames into `out_mp4`
    (all F frames) and writing the plan JSON. on_progress(stage, fraction_overall, segments_done, segments_total)."""
    vc = pipe.vae.cfg
    tf = vc.temporal_factor
    if not vc.valid_frames(segment_frames) or segment_frames < 1 + tf:
        raise ValueError(f"segment_frames must be 1 + {tf}k with k >= 1, got {segment_frames}")
    F = total_frames(seconds, fps)
    plan = segment_plan(F, segment_frames)
    N = len(plan)
    progress = on_progress or _noop
    kw = dict(width=width, height=height, fps=fps, sampler=sampler)
    if steps:
        kw["steps"] = steps
    if cfg_scale is not None:
        kw["cfg_scale"] = cfg_scale
    pipe.resolve_style(style)                      # ValueError before any heavy work
    weights = {"text": (0.0, 0.02), "sample": (0.02, 0.78), "decode": (0.80, 0.18), "write": (0.98, 0.02)}
    writer = Mp4Writer(out_mp4, width, height, fps, crf=16)
    t0 = time.time()
    state = None
    written = 0
    meta0: dict = {}
    try:
        for i, (a, b) in enumerate(plan):
            def cb(stage, k, n, _i=i):
                st, span = weights.get(stage, (0.0, 0.0))
                progress("draft", (_i + st + span * (k / max(1, n))) / N, _i, N)

            need = b - a + (1 if i == 0 else 0)            # frames this segment adds to the video
            if i == 0:
                res = pipe.generate(prompt, seed=seed, frames=1 + tf * math.ceil((need - 1) / tf), style=style,
                                    on_progress=cb, should_stop=should_stop, **kw)
                meta0 = res.meta
            else:
                overlap = vc.frames_for_latent(2)
                res = pipe.continue_video(state, frames=overlap + tf * math.ceil(need / tf), on_progress=cb,
                                          should_stop=should_stop, **kw)
            if res.frames.shape[0] < need:
                raise RuntimeError(f"segment {i} produced {res.frames.shape[0]} frames, expected {need}")
            writer.write(res.frames[:need])
            written += need
            state = res.state
            progress("draft", (i + 1) / N, i + 1, N)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    assert written == F, (written, F)
    plan_doc = {"frames": F, "fps": fps, "segment_frames": segment_frames,
                "segments": [{"index": i, "start": a, "end": b} for i, (a, b) in enumerate(plan)]}
    Path(plan_path).write_text(json.dumps(plan_doc, indent=1))
    return {"frames": F, "segments": N, "fps": fps, "width": width, "height": height,
            "seconds": F / fps, "seconds_total": time.time() - t0,
            "model_version_id": meta0.get("model_version_id"),
            "style": meta0.get("style"), "peak_rss_mb": meta0.get("peak_rss_mb")}


# ===================================================================== ai.refine_segment
def load_draft_frames(draft_path, start: int, end: int, width: int, height: int, tf: int, spatial: int) -> tuple:
    """Draft frames start..end upscaled (bicubic) to width x height, edge-padded to the VAE spatial multiple and
    repeated at the end up to the 1 + tf*k rule. Returns (uint8 [T,Hp,Wp,3], n = end-start+1)."""
    if start < 0 or end < start:
        raise LongVideoError(f"bad segment range {start}..{end}")
    try:
        info = probe_fast(draft_path)
    except FFmpegError as e:
        raise LongVideoError(f"draft_video is not readable: {e}")
    if end >= info["frames"]:
        raise LongVideoError(f"draft_video has {info['frames']} frames, segment needs frames up to {end}")
    n = end - start + 1
    T = 1 + tf * math.ceil((n - 1) / tf)
    Hp, Wp = padded(height, spatial), padded(width, spatial)
    arr = np.empty((T, Hp, Wp, 3), dtype=np.uint8)
    got = 0
    try:
        with RawReader(draft_path, info["width"], info["height"], info["fps"], start=start, count=n,
                       size=(width, height)) as r:
            tmp = np.empty((height, width, 3), dtype=np.uint8)
            while got < n:
                f = r.read(tmp)
                if f is None:
                    break
                arr[got, :height, :width] = tmp
                got += 1
    except FFmpegError as e:
        raise LongVideoError(f"draft_video is not readable: {e}")
    if got != n:
        raise LongVideoError(f"draft_video yielded {got} of {n} frames for {start}..{end}")
    if T > n:
        arr[n:, :height, :width] = arr[n - 1, :height, :width]
    if Hp > height:
        arr[:, height:, :width] = arr[:, height - 1:height, :width]
    if Wp > width:
        arr[:, :, width:] = arr[:, :, width - 1:width]
    return arr, n


def run_refine(pipe: HDPipeline, *, draft_path, prompt: str, seed: int, width: int, height: int, fps: int,
               start: int, end: int, strength: float, out_mp4, style: Optional[str] = None,
               steps: Optional[int] = None, cfg_scale: Optional[float] = None, sampler: Optional[str] = None,
               on_progress: Optional[Callable] = None, should_stop: Optional[Callable] = None) -> dict:
    vc = pipe.vae.cfg
    pipe.resolve_style(style)
    t0 = time.time()
    frames, n = load_draft_frames(draft_path, start, end, width, height, vc.temporal_factor, vc.spatial_factor)
    seconds_load = time.time() - t0
    writer = Mp4Writer(out_mp4, width, height, fps, crf=12)
    left = [n]

    def sink(chunk: np.ndarray):
        if left[0] <= 0:
            return
        take = chunk[: left[0]]
        writer.write(take)
        left[0] -= take.shape[0]

    try:
        meta = pipe.refine(prompt, frames, width, height, seed=seed, strength=strength, steps=steps,
                           cfg_scale=cfg_scale, sampler=sampler, style=style, on_progress=on_progress,
                           should_stop=should_stop, sink=sink)
        if left[0] != 0:
            raise RuntimeError(f"decoded {n - left[0]} frames, expected {n}")
        writer.close()
    except BaseException:
        writer.abort()
        raise
    meta.update(frames=n, seconds_load=seconds_load, seconds_total=time.time() - t0)
    return meta


# ===================================================================== ai.stitch
def effective_crossfade(crossfade: int, n_a: int, n_b: int) -> int:
    """Seam window actually used between segments of n_a and n_b frames (both windows must fit in the segments)."""
    return max(0, min(int(crossfade), (n_a - 1) // 2, (n_b - 1) // 2))


def _blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return np.rint(a.astype(np.float32) * (1.0 - w) + b.astype(np.float32) * w).astype(np.uint8)


def run_stitch(segments: list, out_mp4, fps: int, crossfade_frames: int = 4,
               should_stop: Optional[Callable] = None, on_progress: Optional[Callable] = None) -> dict:
    """Concatenate segments in order, dropping every later segment's first (shared boundary) frame, with a
    crossfade over `crossfade_frames` output frames around each seam; total = sum(n_i) - (N-1) frames.

    The seam sits on the shared boundary frame B (last frame of segment i == first of i+1). With c frames in the
    window, pre = c//2 frames before B, B, and c-pre-1 frames after it are blends of both segments; where one
    segment has no frame at that time (before its start / after its end) its boundary frame is held. Weights ramp
    linearly (j+1)/(c+1) towards the next segment. Streams one frame at a time: memory is O(window)."""
    if not segments:
        raise LongVideoError("no segments")
    progress = on_progress or _noop
    stop = should_stop or (lambda: False)
    infos = []
    for p in segments:
        try:
            infos.append(probe_fast(p))
        except FFmpegError as e:
            raise LongVideoError(f"segment {p} is not readable: {e}")
    w0, h0 = infos[0]["width"], infos[0]["height"]
    for p, inf in zip(segments, infos):
        if (inf["width"], inf["height"]) != (w0, h0):
            raise LongVideoError(f"segment {p} is {inf['width']}x{inf['height']}, expected {w0}x{h0}")
        if inf["frames"] < 1:
            raise LongVideoError(f"segment {p} has no frames")
    ns = [inf["frames"] for inf in infos]
    N = len(segments)
    expected = sum(ns) - (N - 1)
    cs = [effective_crossfade(crossfade_frames, ns[i], ns[i + 1]) for i in range(N - 1)]
    writer = Mp4Writer(out_mp4, w0, h0, fps, crf=14)
    t0 = time.time()
    tail: list = []                      # frames of the previous segment held back for the seam window
    try:
        for i, path in enumerate(segments):
            n = ns[i]
            c_in = cs[i - 1] if i > 0 else 0
            c_out = cs[i] if i < N - 1 else 0
            post = c_in - c_in // 2 - 1 if c_in > 0 else 0
            hold = c_out // 2 + 1 if c_out > 0 else 0
            with RawReader(path, w0, h0, infos[i]["fps"]) as r:
                idx = 0
                if i > 0:
                    first = r.read().copy()
                    idx = 1
                    if c_in > 0:
                        head = [first]
                        for _ in range(post):
                            head.append(r.read().copy())
                        idx += post
                        pre = c_in // 2
                        last_a = tail[pre]
                        for j in range(c_in):
                            a = tail[j] if j <= pre else last_a
                            b = head[0] if j <= pre else head[j - pre]
                            writer.write(_blend(a, b, (j + 1) / (c_in + 1)))
                        tail = []
                stop_at = n - hold
                buf: list = []
                while idx < n:
                    if stop():
                        raise Stopped()
                    f = r.read()
                    if f is None:
                        raise LongVideoError(f"segment {path}: stream ended after {idx} of {n} frames")
                    if idx < stop_at:
                        writer.write(f)
                    else:
                        buf.append(f.copy())
                    idx += 1
                tail = buf
            progress("stitch", (i + 1) / N, i + 1, N)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    if writer.frames != expected:
        raise RuntimeError(f"stitched {writer.frames} frames, expected {expected}")
    return {"frames": writer.frames, "seconds": writer.frames / fps, "segments": N, "width": w0, "height": h0,
            "fps": fps, "crossfade_frames": crossfade_frames, "seconds_total": time.time() - t0}
