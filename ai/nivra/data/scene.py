"""Scene specification (ground truth) shared by generator, parser and evaluator."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from . import lexicon as L


@dataclass
class Entity:
    shape: Optional[str]      # None = unspecified (generic 'object')
    color: str
    motion: str
    # initial position (fractions of the frame, x right, y down) - only set for rendered scenes
    x0: float = 0.5
    y0: float = 0.5
    radius: float = 7.0       # pixels at 64x64 reference size

    def key(self):
        return (self.color, self.shape, self.motion)

    def to_dict(self):
        return asdict(self)


@dataclass
class SceneSpec:
    entities: list = field(default_factory=list)

    def ordered(self) -> "SceneSpec":
        """Canonical order: by initial x then y (ties by color, shape)."""
        es = sorted(self.entities, key=lambda e: (round(e.x0, 4), round(e.y0, 4), e.color, e.shape or ""))
        return SceneSpec(es)

    def groups(self):
        """List of (Entity-key, count) in first-appearance order of the canonical ordering."""
        out: dict = {}
        for e in self.ordered().entities:
            out[e.key()] = out.get(e.key(), 0) + 1
        return list(out.items())

    def counts_by_color(self):
        c: dict = {}
        for e in self.entities:
            c[e.color] = c.get(e.color, 0) + 1
        return c

    def to_dict(self):
        return {"entities": [e.to_dict() for e in self.ordered().entities],
                "counts_by_color": self.counts_by_color(),
                "n": len(self.entities)}

    @staticmethod
    def from_dict(d):
        return SceneSpec([Entity(**e) for e in d["entities"]])

    def semantic_key(self):
        """Language/position independent identity (multiset of (color, shape, motion))."""
        return tuple(sorted((e.color, e.shape or "*", e.motion) for e in self.entities))
