"""Orchestration: find videos, decide what to write, call the (lazy) models, write files atomically."""
from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import VIDEO_EXTS, CaptionError, DependencyError
from .describe import merge_caption, resolve_style
from .frames import extract_frames
from .subtitles import extract_embedded, probe_streams, run_tool, short_err, split_subtitle_streams


@dataclass
class Options:
    style: str = "auto"
    subtitles: str = "auto"
    frames: int = 6
    device: str = "auto"
    overwrite: bool = False
    model: str | None = None
    limit: int | None = None  # caption at most this many videos that still need captions


@dataclass
class Result:
    video: Path
    status: str  # ok | skipped | failed
    caption_path: Path | None = None
    subtitles_path: Path | None = None
    subtitle_source: str = ""  # embedded | speech | ""
    subtitle_lines: int = 0
    notes: list[str] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0


def natural_key(rel: Path) -> list:
    """Natural sort of a relative path: clip2 < clip10, compared folder by folder."""
    key: list = []
    for part in rel.parts:
        key.append([(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", part) if t])
    return key


def find_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    vids = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(vids, key=lambda p: natural_key(p.relative_to(path)))


def caption_path_for(video: Path) -> Path:
    return video.with_suffix(".txt")


def subtitles_path_for(video: Path) -> Path:
    return video.with_suffix(".subtitles.txt")


def write_text_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text if text.endswith("\n") else text + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def extract_audio_wav(video: Path, out: Path) -> bool:
    r = run_tool(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                  str(out)], "extract audio")
    if r.returncode != 0 or not out.exists():
        raise CaptionError(f"ffmpeg could not extract the audio: {short_err(r.stderr)}")
    return True


class Lazy:
    """Create an expensive object on first use (so skipped / subtitle-less runs never load a model)."""

    def __init__(self, factory: Callable[[], object]):
        self._factory, self._obj = factory, None

    def get(self):
        if self._obj is None:
            self._obj = self._factory()
        return self._obj


def get_subtitles(video: Path, mode: str, whisper: Lazy, notes: list[str]) -> tuple[str, list[str]]:
    """Return (source, lines). source '' = nothing found."""
    if mode == "off":
        return "", []
    streams = probe_streams(video)
    if mode in ("auto", "embedded"):
        text_streams, bitmap = split_subtitle_streams(streams)
        for s in bitmap:
            notes.append(f"subtitle stream {s.index} is bitmap-based ({s.codec}) - unsupported (no OCR)")
        for s in text_streams:
            text = extract_embedded(video, s)
            if text:
                return "embedded", text.split("\n")
            notes.append(f"subtitle stream {s.index} ({s.codec}) has no text")
        if mode == "embedded":
            return "", []
    if not any(s.kind == "audio" for s in streams):
        return "", []
    with tempfile.TemporaryDirectory(prefix="nivra-aud-") as td:
        wav = Path(td) / "a.wav"
        extract_audio_wav(video, wav)
        lines = whisper.get().transcribe(wav)  # type: ignore[attr-defined]
    return ("speech", lines) if lines else ("", [])


def describe_video(video: Path, opts: Options, florence: Lazy) -> str:
    with tempfile.TemporaryDirectory(prefix="nivra-frames-") as td:
        frames = extract_frames(video, opts.frames, Path(td))
        detailed, early, late, shorts = florence.get().caption_frames(frames)  # type: ignore[attr-defined]
    style = resolve_style(opts.style, shorts + [detailed])
    cap = merge_caption(style, detailed, early, late)
    if not cap:
        raise CaptionError("the model returned an empty description")
    return cap


def process_video(video: Path, opts: Options, florence: Lazy, whisper: Lazy) -> Result:
    t0 = time.time()
    res = Result(video, "ok")
    cap_p, sub_p = caption_path_for(video), subtitles_path_for(video)
    if cap_p.exists() and not opts.overwrite:
        res.status, res.caption_path = "skipped", cap_p
        res.notes.append(f"{cap_p.name} already exists (use --overwrite to replace it)")
        return res
    try:
        text = describe_video(video, opts, florence)
        src, lines = get_subtitles(video, opts.subtitles, whisper, res.notes)
        if lines:  # subtitles first: the .txt is the "done" marker, so it is always written last
            write_text_atomic(sub_p, "\n".join(lines))
            res.subtitles_path, res.subtitle_source, res.subtitle_lines = sub_p, src, len(lines)
        write_text_atomic(cap_p, text)
        res.caption_path = cap_p
    except DependencyError:
        raise
    except CaptionError as e:
        res.status, res.error = "failed", str(e)
    except Exception as e:  # keep going with the next video
        res.status, res.error = "failed", f"{type(e).__name__}: {e}"
    res.seconds = time.time() - t0
    return res


def format_result(r: Result, idx: int, total: int, root: Path | None = None) -> list[str]:
    try:
        name = str(r.video.relative_to(root)) if root else r.video.name
    except ValueError:
        name = str(r.video)
    pre = f"[{idx}/{total}] {name} ... "
    if r.status == "ok":
        head = pre + f"done ({r.seconds:.0f} s)"
        if r.subtitles_path:
            head += f" + subtitles ({r.subtitle_source}, {r.subtitle_lines} lines)"
    elif r.status == "skipped":
        head = pre + "skipped (caption exists)"
    else:
        head = pre + f"FAILED: {r.error}"
    return [head] + [f"    note: {n}" for n in r.notes if r.status != "skipped"]


def write_failed_list(folder: Path, failed: list[Result]) -> Path:
    p = folder / "captions-failed.txt"
    write_text_atomic(p, "\n".join(f"{r.video}\t{r.error}" for r in failed))
    return p


def _needs_caption(v: Path, opts: Options) -> bool:
    return opts.overwrite or not caption_path_for(v).exists()


def run(path: Path, opts: Options, *, florence: Lazy | None = None, whisper: Lazy | None = None,
        out: Callable[[str], None] = print) -> int:
    from .models import Florence, Whisper  # light: heavy imports happen inside the constructors
    if not path.exists():
        out(f"error: {path} does not exist")
        return 2
    if opts.frames < 1 or (opts.limit is not None and opts.limit < 1):
        out("error: --frames and --limit must be at least 1")
        return 2
    florence = florence or Lazy(lambda: Florence(opts.model, opts.device))
    whisper = whisper or Lazy(lambda: Whisper(opts.device))
    videos = find_videos(path)
    if not videos:
        out(f"no videos ({', '.join(VIDEO_EXTS)}) found in {path}")
        return 0
    root = path if path.is_dir() else None
    t0 = time.time()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failed: list[Result] = []
    sub_count = attempted = 0
    interrupted = stopped_early = False
    next_idx = 0  # index (0-based) of the first video not handled
    try:
        for i, v in enumerate(videos):
            if opts.limit is not None and attempted >= opts.limit and _needs_caption(v, opts):
                stopped_early = True
                break
            r = process_video(v, opts, florence, whisper)
            next_idx = i + 1
            if r.status != "skipped":
                attempted += 1
            counts[r.status] += 1
            sub_count += 1 if r.subtitles_path else 0
            if r.status == "failed":
                failed.append(r)
            for line in format_result(r, i + 1, len(videos), root):
                out(line)
    except KeyboardInterrupt:
        interrupted = True
        out("interrupted - stopping (finished files are kept; the current video was abandoned, no partial files)")
    except DependencyError as e:
        out(f"error: {e}")
        return 1
    remaining = sum(1 for v in videos[next_idx:] if _needs_caption(v, opts))
    if failed:
        folder = path if path.is_dir() else path.parent
        out(f"failed list: {write_failed_list(folder, failed)}")
    more = f"; {remaining} more need captions - run the same command again" if (stopped_early or interrupted) and remaining else ""
    out(f"summary: {counts['ok']} done ({sub_count} with subtitles), {counts['skipped']} skipped (existing), "
        f"{counts['failed']} failed, total {time.time() - t0:.0f} s{more}")
    if interrupted:
        return 130
    return 1 if counts["failed"] else 0
