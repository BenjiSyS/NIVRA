"""Text-encoder registry: `get_text_encoder("nivra:<ckpt dir>")` / `get_text_encoder("external:<dir>")`."""
from __future__ import annotations

from .interface import TextEncoder


def get_text_encoder(spec: str, device: str = "cpu", **kw) -> TextEncoder:
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"bad text encoder spec {spec!r}; use 'nivra:<checkpoint dir>' or 'external:<dir>'")
    kind, _, target = spec.partition(":")
    if not target:
        raise ValueError(f"text encoder spec {spec!r} has no path")
    if kind == "nivra":
        from .nivra_text import NivraTextEncoder
        return NivraTextEncoder.from_checkpoint(target, device=device)
    if kind == "external":
        from .external import load_external
        return load_external(target, device=device, **kw)
    raise ValueError(f"unknown text encoder kind {kind!r}; expected 'nivra' or 'external'")
