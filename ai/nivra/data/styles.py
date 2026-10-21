"""Style vocabulary for Nivra's stylized (3D animation / anime / cartoon / 2D animation) specialization.

One place for the canonical style tags, their aliases and the three helpers every other module uses:

* ``normalize_style(text)``  -> canonical tag or None
* ``detect_leading_style(caption)`` -> (style | None, rest of the caption) for captions like
  ``"anime style. A girl runs across a rooftop."``
* ``apply_style(prompt, style)`` -> ``"<style> style. <prompt>"`` unless the prompt already names a style

"realistic" is a valid tag but is never a default and is not part of the stylized family.
"none" is not a style: it is the explicit "do not touch the prompt" switch (``apply_style(p, "none") == p``).
"""
from __future__ import annotations

import re

NONE_STYLE = "none"
UNSPECIFIED = "unspecified"          # ingest value for clips whose style is unknown

# canonical tag -> aliases (the canonical tag itself is always an alias)
STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "anime": ("anime", "manga-style", "manga style", "japanese animation"),
    "3d animation": ("3d animation", "3d", "3d render", "cgi", "pixar-style", "pixar style", "blender"),
    "cartoon": ("cartoon", "western cartoon"),
    "2d animation": ("2d animation", "2d", "hand-drawn", "hand drawn"),
    "cel-shaded 3d": ("cel-shaded 3d", "cel shaded 3d", "cel-shaded", "cel shaded", "cel shading"),
    "pixel art": ("pixel art", "pixel-art", "pixelart"),
    "claymation": ("claymation", "stop-motion", "stop motion", "clay"),
    "painterly": ("painterly",),
    "realistic": ("realistic", "photorealistic", "photo-realistic"),
}
STYLES: tuple[str, ...] = tuple(STYLE_ALIASES)                       # realistic last
STYLIZED_STYLES: tuple[str, ...] = tuple(s for s in STYLES if s != "realistic")


def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", " ", s.strip().lower())


_LOOKUP: dict[str, str] = {}
for _canon, _aliases in STYLE_ALIASES.items():
    for _a in _aliases:
        _LOOKUP[_norm(_a)] = _canon
        _LOOKUP[_norm(_a).replace("-", " ")] = _canon
# "<tag> style" ("anime style") is also accepted wherever a tag is
for _a, _canon in list(_LOOKUP.items()):
    _LOOKUP.setdefault(_a + " style", _canon)


def _alias_regex(alias: str) -> str:
    parts = re.split(r"[\s-]+", alias.strip().lower())
    return r"[\s-]+".join(re.escape(p) for p in parts)


_ALL_ALIASES = sorted({a for al in STYLE_ALIASES.values() for a in al}, key=len, reverse=True)
_ALIAS_ALT = "|".join(_alias_regex(a) for a in _ALL_ALIASES)
_MENTION = re.compile(rf"(?<![A-Za-z0-9])(?:{_ALIAS_ALT})(?![A-Za-z0-9])", re.IGNORECASE)
_LEADING = re.compile(rf"^\s*(?P<alias>{_ALIAS_ALT})(?P<style>[\s-]+style)?\s*(?P<sep>[.:;|,])\s*(?P<rest>.*)$",
                      re.IGNORECASE | re.DOTALL)


def normalize_style(text) -> str | None:
    """Canonical style for a tag or alias (case-insensitive), else None. 'none'/'' are not styles."""
    if not isinstance(text, str):
        return None
    return _LOOKUP.get(_norm(text))


def mentions_style(text: str) -> bool:
    """True when `text` contains any style alias as a whole word (case-insensitive)."""
    return bool(_MENTION.search(text or ""))


def detect_leading_style(caption: str) -> tuple[str | None, str]:
    """('anime style. A girl runs.') -> ('anime', 'A girl runs.'); no leading style phrase -> (None, caption).
    The phrase is an alias, optionally followed by 'style', then one of . : ; | (a comma only after 'style')."""
    m = _LEADING.match(caption or "")
    if not m or not m.group("rest").strip():
        return None, caption
    if m.group("sep") == "," and not (m.group("style") or m.group("alias").lower().endswith("style")):
        return None, caption
    style = normalize_style(re.sub(r"[\s-]+", " ", m.group("alias")))
    if style is None:
        return None, caption
    return style, m.group("rest").strip()


def apply_style(prompt: str, style) -> str:
    """'<style> style. <prompt>', unchanged when style is None/'none'/'unspecified' or the prompt already
    mentions a style alias. An unknown style raises ValueError."""
    if style is None:
        return prompt
    if isinstance(style, str) and _norm(style) in (NONE_STYLE, UNSPECIFIED, ""):
        return prompt
    canon = normalize_style(style)
    if canon is None:
        raise ValueError(f"unknown style {style!r}; known: {', '.join(STYLES)}, or 'none'")
    if mentions_style(prompt):
        return prompt
    return f"{canon} style. {prompt}"


def parse_style_arg(style) -> str:
    """Validate a user-supplied style ('anime', an alias, 'none') -> canonical tag or 'none'. Raises ValueError."""
    if not isinstance(style, str):
        raise ValueError("style must be a string")
    if _norm(style) == NONE_STYLE:
        return NONE_STYLE
    canon = normalize_style(style)
    if canon is None:
        raise ValueError(f"unknown style {style!r}; use one of: {', '.join(STYLES)}, or 'none'")
    return canon


def parse_data_style(value) -> str:
    """Style value from data files (captions.jsonl / PROVENANCE.json / options): canonical tag or 'unspecified'
    for none/unspecified. Raises ValueError for anything else."""
    if isinstance(value, str) and _norm(value) in (NONE_STYLE, UNSPECIFIED):
        return UNSPECIFIED
    canon = normalize_style(value)
    if canon is None:
        raise ValueError(f"unknown style {value!r}; use one of: {', '.join(STYLES)}")
    return canon


def validate_style_list(styles, where: str = "styles") -> list[str]:
    """Config check: a list of distinct canonical tags."""
    if not isinstance(styles, (list, tuple)) or not styles:
        raise ValueError(f"{where} must be a non-empty list")
    out = []
    for s in styles:
        if s not in STYLE_ALIASES:
            raise ValueError(f"{where}: {s!r} is not a canonical style; choose from {list(STYLES)}")
        if s in out:
            raise ValueError(f"{where}: duplicate style {s!r}")
        out.append(s)
    return out


def resolve_style(style, supported=(), default=None) -> str | None:
    """Style a model should use for a request: None -> `default` (may be None); 'none' -> 'none' (no tag);
    otherwise the canonical tag. Raises ValueError for unknown styles and, when `supported` is non-empty, for
    known styles the model was not configured for."""
    if style is None:
        return default
    canon = parse_style_arg(style)
    if canon != NONE_STYLE and supported and canon not in supported:
        raise ValueError(f"style {canon!r} is not supported by this model; supported: {list(supported)}")
    return canon
