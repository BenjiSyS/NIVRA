"""The text-encoder interface every Nivra generator conditions on (design contract section 3)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class TextEncoder(Protocol):
    id: str          # e.g. "nivra-text-base@1a2b3c4d5e6f" or "external:umt5-xl@<sha12>"
    dim: int         # embedding width D_text
    max_len: int

    def encode(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        """Return (emb [B, L, D] float32, mask [B, L] bool; True = real token)."""
        ...
