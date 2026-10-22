"""Structured video captions (Qwen3-VL): prompt, tolerant JSON parsing, short + dense assembly. No model code here.

The model sees several frames of one clip in time order and answers with a JSON object. We turn it into

    line 1   short caption: "<style> style. <what happens>", at most SHORT_WORDS words
    line 2   dense paragraph: subjects, action and motion over time, camera, setting, art style, lighting

and an optional JSON record with the individual fields. Speech and on-screen text never enter the visual caption.
"""
from __future__ import annotations

import json
import re

from . import STYLES, CaptionError
from .describe import detect_style, resolve_style, truncate_words

SHORT_WORDS = 25
DENSE_WORDS = 200
FIELDS = ("subjects", "motion", "camera", "setting", "art_style", "lighting")
SCHEMA = "nivra.caption/1"


def build_prompt(n_frames: int, times: list[float] | None = None) -> str:
    when = ""
    if times:
        when = " at about " + ", ".join(f"{t:.1f}s" for t in times)
    return (
        f"These {n_frames} images are frames of ONE short animated video clip, in time order{when}. "
        "Describe the whole clip, not a single frame. Describe only what is visible: ignore speech, subtitles, "
        "logos and any written text, and never quote text. Answer with one JSON object and nothing else, with keys:\n"
        f'"style": one of {json.dumps(STYLES)} (the medium of the clip),\n'
        f'"short": one sentence of at most {SHORT_WORDS - 2} words saying who does what,\n'
        '"subjects": the characters or objects and their appearance,\n'
        '"motion": what moves and how, in order over time (direction, speed, changes between the frames),\n'
        '"camera": camera angle, shot type and camera movement,\n'
        '"setting": the place and background,\n'
        '"art_style": the art style and colours,\n'
        '"lighting": the lighting and mood.'
    )


_QUOTED = re.compile(r"[\"“”][^\"“”]{3,}[\"“”]")
_SPEECH = re.compile(r"\b(subtitles?|dialogue|dialog|speech|speaks?|speaking|says|said|saying|shouts?|whispers?|"
                     r"voice[- ]?over|narrat\w+|text reads|caption reads|written text|on-screen text)\b", re.I)


def strip_speech(text: str) -> str:
    """Drop quoted text and any sentence that talks about speech / subtitles (the visual caption must not teach the model to render text)."""
    t = _QUOTED.sub("", text or "")
    sents = re.split(r"(?<=[.!?])\s+", t.strip())
    return " ".join(s for s in sents if s and not _SPEECH.search(s)).strip()


def _clean(s) -> str:
    if isinstance(s, (list, tuple)):
        s = ". ".join(str(x) for x in s)
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return strip_speech(s)


def _sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    s = s[:1].upper() + s[1:]
    return s if s.endswith((".", "!", "?")) else s + "."


def parse_output(raw: str) -> dict:
    """The JSON object in the model's answer (code fences / chatter around it tolerated). {} when there is none."""
    if not raw:
        return {}
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I)
    start = txt.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(txt)):
            ch = txt[i]
            if in_str:
                esc = (not esc and ch == "\\")
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(txt[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = txt.find("{", start + 1)
    return {}


def assemble(raw: str, requested_style: str = "auto") -> dict:
    """Model answer -> {"style", "short", "dense", "fields"}. Raises CaptionError if nothing usable came back."""
    obj = parse_output(raw)
    fields = {k: _clean(obj.get(k)) for k in FIELDS}
    short = _clean(obj.get("short"))
    if not obj:  # the model ignored the format: use its text as the dense description
        plain = _clean(re.sub(r"[{}\[\]\"]", " ", raw or ""))
        if not plain:
            raise CaptionError("the model returned an empty description")
        fields["subjects"] = plain
        short = truncate_words(plain, SHORT_WORDS)
    if not short:
        first = next((v for v in (fields["subjects"], fields["motion"]) if v), "")
        if not first:
            raise CaptionError("the model returned an empty description")
        short = truncate_words(first, SHORT_WORDS)
    model_style = str(obj.get("style") or "").strip().lower()
    texts = [short, *fields.values()]
    if requested_style == "auto":
        style = model_style if model_style in STYLES else detect_style(texts)
    else:
        style = resolve_style(requested_style, texts)
    prefix = f"{style[:1].upper() + style[1:]} style." if style else ""
    short = re.sub(r"^" + re.escape(style or "\0") + r" style[.,:]?\s*", "", short, flags=re.I) if style else short
    budget = SHORT_WORDS - len(prefix.split())
    short_line = " ".join(p for p in (prefix, truncate_words(_sentence(short), budget)) if p)
    dense = " ".join(_sentence(fields[k]) for k in FIELDS if fields[k])
    dense = truncate_words(dense, DENSE_WORDS) if dense else short_line
    return {"style": style, "short": short_line, "dense": dense, "fields": fields}


def txt_of(cap: dict) -> str:
    """Contents of <name>.txt: short caption, newline, dense paragraph."""
    return cap["short"] + "\n" + cap["dense"]


def record_of(cap: dict, *, captioner: str, model: str, times: list[float]) -> dict:
    return {"schema": SCHEMA, "captioner": captioner, "model": model, "style": cap["style"], "short": cap["short"],
            "dense": cap["dense"], "fields": cap["fields"], "frame_times": [round(t, 3) for t in times]}
