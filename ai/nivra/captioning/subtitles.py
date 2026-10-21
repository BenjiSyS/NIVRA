"""Subtitle handling: probe streams, extract embedded text subtitles, clean to plain text."""
from __future__ import annotations

import html
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import CaptionError

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "microdvd", "subviewer", "sami", "realtext"}
BITMAP_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub", "dvb_teletext", "hdmv_text_subtitle"}


@dataclass
class Stream:
    index: int  # absolute stream index in the container
    kind: str   # subtitle | audio | video
    codec: str
    lang: str = ""


def short_err(stderr: str) -> str:
    """Last non-empty line of a tool's stderr, without the '[demuxer @ 0x...]' prefix."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    msg = re.sub(r"^\[[^\]]*\]\s*", "", lines[-1]) if lines else "unknown error"
    return msg[:200]


def run_tool(args: list[str], what: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise CaptionError(f"{args[0]} not found on PATH (needed to {what}); install FFmpeg and try again")


def probe_streams(video: Path) -> list[Stream]:
    r = run_tool(["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type,codec_name:stream_tags=language",
                  "-of", "json", str(video)], "inspect videos")
    if r.returncode != 0:
        raise CaptionError(f"ffprobe could not read the file: {short_err(r.stderr)}")
    data = json.loads(r.stdout or "{}")
    return [Stream(int(s["index"]), s.get("codec_type", ""), s.get("codec_name", ""),
                   (s.get("tags") or {}).get("language", "")) for s in data.get("streams", [])]


def split_subtitle_streams(streams: list[Stream]) -> tuple[list[Stream], list[Stream]]:
    text, bitmap = [], []
    for s in streams:
        if s.kind != "subtitle":
            continue
        (bitmap if s.codec in BITMAP_CODECS else text).append(s)
    return text, bitmap


# ---------------------------------------------------------------- cleaning

_TIME = re.compile(r"^\s*(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->\s*(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3}")
_ASS_TAG = re.compile(r"\{[^}]*\}")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_ASS_DRAWING = re.compile(r"^[mnlbspc\d\s.\-]+$")


def _clean_cue_text(s: str) -> str:
    s = s.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    s = _ASS_TAG.sub("", s)
    s = _HTML_TAG.sub("", s)
    s = html.unescape(s)
    s = s.replace(chr(0x200b), "").replace(chr(0xfeff), "")
    return s


def _cue_lines(s: str) -> list[str]:
    out = []
    for ln in _clean_cue_text(s).split("\n"):
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln and not (len(ln) > 6 and _ASS_DRAWING.match(ln) and re.search(r"\b[mlb]\b", ln)):
            out.append(ln)
    return out


def _parse_ass(raw: str) -> list[str]:
    cues, fmt = [], None
    for ln in raw.splitlines():
        if ln.startswith("Format:") and fmt is None:
            fmt = [f.strip().lower() for f in ln[7:].split(",")]
        elif ln.startswith("Dialogue:"):
            n = (fmt.index("text") + 1) if fmt and "text" in fmt else 10
            parts = ln[9:].split(",", n - 1)
            if len(parts) >= n:
                cues.append(parts[n - 1])
    return cues


def parse_cues(raw: str) -> list[str]:
    """Raw SRT / WebVTT / ASS text -> one string (possibly multi-line) per cue."""
    raw = raw.lstrip(chr(0xfeff))
    if re.search(r"^\[Script Info\]|^Dialogue:", raw, re.M):
        return _parse_ass(raw)
    cues: list[list[str]] = []
    cur: list[str] | None = None
    for ln in raw.splitlines():
        if _TIME.match(ln):
            cur = []
            cues.append(cur)
        elif not ln.strip():
            cur = None
        elif cur is not None:
            cur.append(ln)
    return ["\n".join(c) for c in cues]


def subtitles_to_text(raw: str) -> str:
    """Strip timing, tags and consecutive duplicate lines; one line per cue (cue lines joined by a space)."""
    lines: list[str] = []
    for cue in parse_cues(raw):
        parts = _cue_lines(cue)
        if not parts:
            continue
        line = " ".join(parts)
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_embedded(video: Path, stream: Stream) -> str:
    """Extract one text subtitle stream via ffmpeg and return cleaned plain text ('' if empty)."""
    with tempfile.TemporaryDirectory(prefix="nivra-sub-") as td:
        out = Path(td) / "s.srt"
        r = run_tool(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", f"0:{stream.index}", "-c:s", "srt", str(out)],
                     "extract subtitles")
        if r.returncode != 0 or not out.exists():
            raise CaptionError(f"ffmpeg could not extract subtitle stream {stream.index}: {short_err(r.stderr)}")
        return subtitles_to_text(out.read_text(encoding="utf-8", errors="replace"))
