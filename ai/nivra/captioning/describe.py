"""Deterministic caption assembly: boilerplate stripping, style detection, merge. No model code here."""
from __future__ import annotations

import re

from . import STYLES

MAX_WORDS = 77

# leading Florence boilerplate, applied repeatedly
_LEAD = [
    r"in (?:this|the) (?:image|picture|photo|photograph|video|scene|frame|illustration)[,]?\s*(?:there (?:is|are)|we can see|you can see|i can see)?",
    r"(?:this|the) (?:image|picture|photo|photograph|video frame|frame|illustration) (?:shows|is of|depicts|features|displays|appears to (?:be|show|depict)|is a|contains|captures|presents)",
    r"there (?:is|are)",
    r"we can see",
    r"you can see",
]
_LEAD_RE = re.compile(r"^\s*(?:" + "|".join(_LEAD) + r")[\s,:]*", re.I)


def _sentence_case(s: str) -> str:
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


def clean_caption(text: str) -> str:
    """Strip Florence boilerplate and say 'video'/'scene' instead of 'image'."""
    t = re.sub(r"\s+", " ", text or "").strip()
    sents = re.split(r"(?<=[.!?])\s+", t)
    out = []
    for s in sents:
        prev = None
        while prev != s:
            prev = s
            s = _LEAD_RE.sub("", s, count=1)
        s = _sentence_case(s)
        if s and not re.fullmatch(r"[.!?\s]*", s):
            out.append(s)
    t = " ".join(out)
    t = re.sub(r"\b(?:in|on|of|from|to|at|by) the (?:image|picture|photo|photograph|frame)\b",
               lambda m: m.group(0).rsplit(" ", 1)[0] + " scene", t, flags=re.I)
    t = re.sub(r"\b(?:this|that) (?:image|picture|photo|photograph|frame)\b", "this scene", t, flags=re.I)
    t = re.sub(r"\ban (?:image|picture|photograph|photo)\b", "a video", t, flags=re.I)
    t = re.sub(r"\ba (?:image|picture|photograph|photo)\b", "a video", t, flags=re.I)
    t = re.sub(r"\b(the )?(?:image|picture)\b",
               lambda m: ("The " if m.group(1) and m.group(1)[0] == "T" else "the " if m.group(1) else "") + "video",
               t, flags=re.I)
    t = re.sub(r"\bthe the\b", "the", t, flags=re.I)
    return t.strip()


def _words(s: str) -> list[str]:
    return s.split()


def truncate_words(text: str, limit: int) -> str:
    """Keep whole sentences while they fit; if even the first does not fit, hard-cut at a word."""
    if limit <= 0:
        return ""
    if len(_words(text)) <= limit:
        return text.strip()
    kept, n = [], 0
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        w = len(_words(s))
        if n + w > limit:
            break
        kept.append(s)
        n += w
    if kept:
        return " ".join(kept)
    cut = " ".join(_words(text)[:limit]).rstrip(",;:- ")
    return cut + ("" if cut.endswith((".", "!", "?")) else ".")


# ---------------------------------------------------------------- style

# (canonical style, keyword regex); evaluated per caption, voted across captions.
_STYLE_KEYWORDS: list[tuple[str, str]] = [
    ("pixel art", r"pixel(?:ated)?[ -]?art|pixelated|8-?bit|16-?bit|pixel graphics"),
    ("claymation", r"claymation|clay\b|clay-?like|plasticine|stop[- ]motion"),
    ("cel-shaded 3d", r"cel[- ]?shad|toon[- ]shad"),
    ("anime", r"anime|manga"),
    ("3d animation", r"3d|three-dimensional|cgi|computer[- ]generated|rendered|\brender\b|pixar|low[- ]poly|voxel"),
    ("cartoon", r"cartoon|comic|caricature"),
    ("painterly", r"oil painting|watercolou?r|painting|painterly|brush ?strokes?|painted"),
    ("realistic", r"realistic|photograph|photo-?realistic|\bphoto\b|live[- ]action"),
]
# generic "animated" is only a fallback: it cannot tell 2d from 3d
_GENERIC_2D = r"animated|animation|illustration|illustrated|hand-?drawn|drawing|drawn"


def detect_style(captions: list[str]) -> str | None:
    """Keyword vote across frame captions. None when nothing is detected (never guesses 'anime')."""
    scores: dict[str, int] = {}
    for c in captions:
        low = c.lower()
        for style, rx in _STYLE_KEYWORDS:
            if re.search(rx, low):
                scores[style] = scores.get(style, 0) + 1
    if scores:
        order = [s for s, _ in _STYLE_KEYWORDS]
        best = max(scores.values())
        return next(s for s in order if scores.get(s) == best)
    if any(re.search(_GENERIC_2D, c.lower()) for c in captions):
        return "2d animation"
    return None


def resolve_style(requested: str, captions: list[str]) -> str | None:
    if requested == "none":
        return None
    if requested != "auto":
        if requested not in STYLES:
            raise ValueError(f"unknown style {requested!r}")
        return requested
    return detect_style(captions)


# ---------------------------------------------------------------- merge

def _tokset(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _differs(a: str, b: str) -> bool:
    A, B = _tokset(a), _tokset(b)
    if not A or not B:
        return False
    return len(A & B) / len(A | B) < 0.7


def _brief(s: str, n: int) -> str:
    s = s.strip().rstrip(".!? ")
    w = _words(s)
    s = " ".join(w[:n])
    s = re.sub(r"^(?:a|an|the)\s+", "", s, flags=re.I)
    return s[:1].lower() + s[1:]


def change_note(early: str | None, late: str | None, words_each: int = 10) -> str:
    if not early or not late or not _differs(early, late):
        return ""
    return f"The scene changes from {_brief(early, words_each)} to {_brief(late, words_each)}."


def merge_caption(style: str | None, detailed: str, early: str | None = None, late: str | None = None,
                  max_words: int = MAX_WORDS) -> str:
    """"<style> style. <detailed> <change note>" kept under max_words words."""
    prefix = f"{style} style." if style else ""
    prefix = _sentence_case(prefix)
    d = clean_caption(detailed)
    e = clean_caption(early) if early else None
    l = clean_caption(late) if late else None
    note = change_note(e, l)
    budget = max_words - len(_words(prefix))
    if note and budget - len(_words(note)) < 20:  # keep the description dominant
        note = change_note(e, l, words_each=5)
        if budget - len(_words(note)) < 20:
            note = ""
    d = truncate_words(d, budget - len(_words(note)))
    return " ".join(p for p in (prefix, d, note) if p).strip()
