"""Closed bilingual (English / Swedish) lexicon for the synthetic domain."""
from __future__ import annotations

SHAPES = ("circle", "square", "triangle")
COLORS = ("red", "green", "blue", "yellow", "white", "purple")
MOTIONS = ("l2r", "r2l", "up", "down", "stationary")
LANGS = ("en", "sv")

# RGB palette (0-255)
COLOR_RGB = {
    "red": (230, 40, 40),
    "green": (40, 200, 60),
    "blue": (50, 80, 240),
    "yellow": (240, 220, 40),
    "white": (240, 240, 240),
    "purple": (160, 60, 200),
}

EN_COUNT = {1: "a", 2: "two", 3: "three"}
SV_COUNT_EN_GENDER = {1: "en", 2: "två", 3: "tre"}   # common gender (en-words)
SV_COUNT_ETT_GENDER = {1: "ett", 2: "två", 3: "tre"}  # neuter (ett-words)

EN_SHAPE = {"circle": ("circle", "circles"), "square": ("square", "squares"),
            "triangle": ("triangle", "triangles")}
SV_SHAPE = {"circle": ("cirkel", "cirklar"), "square": ("kvadrat", "kvadrater"),
            "triangle": ("triangel", "trianglar")}  # all en-words
# Generic noun (shape unspecified). Swedish "foeremaal" is an ett-word.
EN_OBJECT = ("object", "objects")
SV_OBJECT = ("föremål", "föremål")  # ett-word, plural identical

EN_COLOR = {c: c for c in COLORS}
# (common/en, neuter/ett, plural/definite-agreement)
SV_COLOR = {
    "red": ("röd", "rött", "röda"),
    "green": ("grön", "grönt", "gröna"),
    "blue": ("blå", "blått", "blå"),
    "yellow": ("gul", "gult", "gula"),
    "white": ("vit", "vitt", "vita"),
    "purple": ("lila", "lila", "lila"),
}

EN_MOTION_SG = {"l2r": "moves from left to right", "r2l": "moves from right to left",
                "up": "moves up", "down": "moves down", "stationary": "stays still"}
EN_MOTION_PL = {"l2r": "move from left to right", "r2l": "move from right to left",
                "up": "move up", "down": "move down", "stationary": "stay still"}
SV_MOTION_SG = {"l2r": "rör sig från vänster till höger", "r2l": "rör sig från höger till vänster",
                "up": "rör sig uppåt", "down": "rör sig nedåt", "stationary": "står still"}
SV_MOTION_PL = {"l2r": "rör sig från vänster till höger", "r2l": "rör sig från höger till vänster",
                "up": "rör sig uppåt", "down": "rör sig nedåt", "stationary": "står stilla"}
# Neuter singular predicate adjective for 'still' (not used by templates; ett-word clauses use 'still' too)

CONJ = {"en": "and", "sv": "och"}


def all_caption_words() -> list[str]:
    """Every word that can appear in a generated caption (sorted, unique, lowercase)."""
    words: set[str] = set()

    def add(s: str) -> None:
        words.update(s.lower().split())

    for d in (EN_COUNT, SV_COUNT_EN_GENDER, SV_COUNT_ETT_GENDER):
        for v in d.values():
            add(v)
    for d in (EN_SHAPE, SV_SHAPE):
        for a, b in d.values():
            add(a); add(b)
    for a, b in (EN_OBJECT, SV_OBJECT):
        add(a); add(b)
    for v in EN_COLOR.values():
        add(v)
    for tup in SV_COLOR.values():
        for v in tup:
            add(v)
    for d in (EN_MOTION_SG, EN_MOTION_PL, SV_MOTION_SG, SV_MOTION_PL):
        for v in d.values():
            add(v)
    for v in CONJ.values():
        add(v)
    return sorted(words)
