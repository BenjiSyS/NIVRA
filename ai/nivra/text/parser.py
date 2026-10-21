"""Deterministic prompt -> scene-spec parser (EN + SV) used by evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data import lexicon as L
from ..data.scene import Entity, SceneSpec
from .tokenizer import normalize

_COUNT = {}
for _n, _w in L.EN_COUNT.items():
    _COUNT[_w] = _n
for _d in (L.SV_COUNT_EN_GENDER, L.SV_COUNT_ETT_GENDER):
    for _n, _w in _d.items():
        _COUNT[_w] = _n
_COLOR = {}
for _c in L.COLORS:
    _COLOR[L.EN_COLOR[_c]] = _c
    for _w in L.SV_COLOR[_c]:
        _COLOR[_w] = _c
_SHAPE = {}
for _s in L.SHAPES:
    for _w in L.EN_SHAPE[_s] + L.SV_SHAPE[_s]:
        _SHAPE[_w] = _s
_OBJECT = set(L.EN_OBJECT + L.SV_OBJECT)
_STILL = {"still", "stilla"}
_SV_WORDS = {"och", "en", "ett", "två", "tre", "rör", "sig", "från", "vänster", "höger", "till", "uppåt", "nedåt", "står", "stilla", "föremål"}


@dataclass
class ParseResult:
    spec: SceneSpec
    lang: str
    unknown: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors


def detect_lang(words) -> str:
    sv = sum(1 for w in words if w in _SV_WORDS or w in {x for t in L.SV_COLOR.values() for x in t}
             or w in {x for p in L.SV_SHAPE.values() for x in p})
    return "sv" if sv else "en"


def _motion(words: list) -> Optional[str]:
    ws = set(words)
    if ws & _STILL:
        return "stationary"
    for a, b, r in (("left", "right", "l2r"), ("vänster", "höger", "l2r")):
        if a in words and b in words:
            ia, ib = words.index(a), words.index(b)
            return r if ia < ib else "r2l"
    if "up" in ws or "uppåt" in ws:
        return "up"
    if "down" in ws or "nedåt" in ws:
        return "down"
    return None


def parse_prompt(text: str) -> ParseResult:
    words = normalize(text)
    lang = detect_lang(words)
    entities: list = []
    unknown, errors = [], []
    # clause segmentation at count/article words
    clauses: list = []
    for w in words:
        if w in _COUNT:
            clauses.append([w])
        elif clauses:
            clauses[-1].append(w)
        else:
            unknown.append(w)
    if not clauses:
        errors.append("no entity clause found")
    for cl in clauses:
        n = _COUNT[cl[0]]
        color = shape = None
        generic = False
        rest = []
        for w in cl[1:]:
            if w in _COLOR and color is None:
                color = _COLOR[w]
            elif w in _SHAPE and shape is None:
                shape = _SHAPE[w]
            elif w in _OBJECT:
                generic = True
            else:
                rest.append(w)
        motion = _motion(rest)
        known_rest = {"moves", "move", "from", "to", "left", "right", "up", "down", "stays", "stay", "still", "stilla",
                      "och", "and", "från", "till", "rör", "sig", "vänster", "höger", "uppåt", "nedåt", "står"}
        unknown.extend(w for w in rest if w not in known_rest)
        if color is None:
            errors.append(f"clause '{' '.join(cl)}': no color")
            continue
        if shape is None and not generic:
            errors.append(f"clause '{' '.join(cl)}': no shape")
            continue
        if motion is None:
            errors.append(f"clause '{' '.join(cl)}': no motion")
            continue
        for _ in range(n):
            entities.append(Entity(shape=shape, color=color, motion=motion))
    return ParseResult(SceneSpec(entities), lang, unknown, errors)
