"""Caption generation (EN + SV) from a scene spec."""
from __future__ import annotations

from . import lexicon as L
from .scene import SceneSpec

STYLES = ("shape", "object")


def _group_phrase(color, shape, motion, n, lang, style):
    generic = (style == "object") or shape is None
    if lang == "en":
        count = L.EN_COUNT[n]
        noun = (L.EN_OBJECT if generic else L.EN_SHAPE[shape])[0 if n == 1 else 1]
        motion_s = (L.EN_MOTION_SG if n == 1 else L.EN_MOTION_PL)[motion]
        return f"{count} {L.EN_COLOR[color]} {noun} {motion_s}"
    # Swedish agreement
    if generic:  # ett-word: ett rött föremål / två röda föremål
        count = L.SV_COUNT_ETT_GENDER[n]
        col = L.SV_COLOR[color][1 if n == 1 else 2]
        noun = L.SV_OBJECT[0 if n == 1 else 1]
    else:        # en-words
        count = L.SV_COUNT_EN_GENDER[n]
        col = L.SV_COLOR[color][0 if n == 1 else 2]
        noun = L.SV_SHAPE[shape][0 if n == 1 else 1]
    motion_s = (L.SV_MOTION_SG if n == 1 else L.SV_MOTION_PL)[motion]
    return f"{count} {col} {noun} {motion_s}"


def caption(spec: SceneSpec, lang: str = "en", style: str = "shape") -> str:
    """Caption for a scene. style='object' replaces shape nouns by the generic noun
    (only used when it stays unambiguous enough; the evaluator treats shape as unspecified)."""
    groups = spec.groups()
    if style == "object":
        # merge groups that become identical once the shape is dropped
        merged: dict = {}
        for (color, shape, motion), n in groups:
            k = (color, None, motion)
            merged[k] = merged.get(k, 0) + n
        groups = list(merged.items())
    parts = []
    for (color, shape, motion), n in groups:
        # counts above 3 are split to stay inside the closed vocabulary
        while n > 0:
            m = min(n, 3)
            parts.append(_group_phrase(color, shape, motion, m, lang, style))
            n -= m
    conj = L.CONJ[lang]
    if len(parts) == 1:
        return parts[0]
    return " ".join([", ".join(parts[:-1]), conj, parts[-1]]) if len(parts) > 1 else parts[0]
