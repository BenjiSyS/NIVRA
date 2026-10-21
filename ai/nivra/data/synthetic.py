"""Deterministic procedural video generator (original, legal training data).

A sample is fully determined by (GENERATOR_VERSION, seed, index, SynthConfig).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import lexicon as L
from .captions import caption
from .scene import Entity, SceneSpec

GENERATOR_VERSION = "nivra-synth/1"
SPEED = 0.30 / 16.0   # frame-fraction per frame (constant for every clip -> continuation is well defined)
REF_SIZE = 64


@dataclass(frozen=True)
class SynthConfig:
    frames: int = 16
    size: int = 64
    max_objects: int = 3
    p_objects: tuple = (0.4, 0.35, 0.25)   # P(1), P(2), P(3)
    p_clone: float = 0.3                   # P(second object shares color+shape with the first)

    def to_dict(self):
        d = asdict(self)
        d["p_objects"] = list(self.p_objects)
        return d


@dataclass
class Sample:
    index: int
    video: np.ndarray        # uint8 [T,H,W,3]
    spec: SceneSpec
    captions: dict           # {"en": {"shape":..., "object":...}, "sv": {...}}

    def caption(self, lang="en", style="shape"):
        return self.captions[lang][style]


def positions(e: Entity, t: np.ndarray):
    """Centre position (frame fractions) of entity e at (possibly fractional / > T) frame times t."""
    x = np.full(t.shape, e.x0, dtype=np.float64)
    y = np.full(t.shape, e.y0, dtype=np.float64)
    if e.motion == "l2r":
        x = e.x0 + SPEED * t
    elif e.motion == "r2l":
        x = e.x0 - SPEED * t
    elif e.motion == "up":
        y = e.y0 - SPEED * t
    elif e.motion == "down":
        y = e.y0 + SPEED * t
    return x, y


_START_RANGE = {"l2r": ("x", 0.15, 0.50), "r2l": ("x", 0.50, 0.85),
                "up": ("y", 0.50, 0.85), "down": ("y", 0.15, 0.50)}


def sample_spec(rng: np.random.Generator, cfg: SynthConfig) -> SceneSpec:
    probs = np.array(cfg.p_objects[:cfg.max_objects], dtype=np.float64)
    n = int(rng.choice(np.arange(1, cfg.max_objects + 1), p=probs / probs.sum()))
    ents: list = []
    for i in range(n):
        if i == 1 and rng.random() < cfg.p_clone:
            shape, color = ents[0].shape, ents[0].color
            motion = ents[0].motion if rng.random() < 0.5 else str(rng.choice(L.MOTIONS))
        else:
            shape = str(rng.choice(L.SHAPES))
            color = str(rng.choice(L.COLORS))
            motion = str(rng.choice(L.MOTIONS))
        ents.append(Entity(shape=shape, color=color, motion=motion, radius=float(rng.uniform(6.0, 8.0))))
    t = np.arange(cfg.frames, dtype=np.float64)
    margin = 4.0
    for attempt in range(5000):
        if attempt > 300:
            margin = 2.0
        for e in ents:
            if e.motion == "stationary":
                e.x0, e.y0 = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))
            else:
                axis, lo, hi = _START_RANGE[e.motion]
                s = float(rng.uniform(lo, hi))
                p = float(rng.uniform(0.2, 0.8))
                e.x0, e.y0 = (s, p) if axis == "x" else (p, s)
        paths = [positions(e, t) for e in ents]
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(((paths[i][0] - paths[j][0]) * REF_SIZE) ** 2 + ((paths[i][1] - paths[j][1]) * REF_SIZE) ** 2)
                if d.min() < ents[i].radius + ents[j].radius + margin:
                    ok = False
        if ok:
            break
    return SceneSpec(ents).ordered()


# triangle (apex up): outward unit normals of the two slanted edges
_NR = np.array([1.55, -0.95]) / np.sqrt(1.55 * 1.55 + 0.95 * 0.95)
_NL = np.array([-1.55, -0.95]) / np.sqrt(1.55 * 1.55 + 0.95 * 0.95)


def _coverage(shape: str, px, py, r, X, Y):
    """Anti-aliased coverage [T,H,W] for one entity. px,py in pixels [T]."""
    dx = X[None] - px[:, None, None]
    dy = Y[None] - py[:, None, None]
    if shape == "circle":
        sdf = np.sqrt(dx * dx + dy * dy) - r
    elif shape == "square":
        sdf = np.maximum(np.abs(dx), np.abs(dy)) - 0.88 * r
    else:
        d_bottom = dy - 0.55 * r
        d_r = _NR[0] * dx + _NR[1] * (dy + r)
        d_l = _NL[0] * dx + _NL[1] * (dy + r)
        sdf = np.maximum(np.maximum(d_r, d_l), d_bottom)
    return np.clip(0.5 - sdf, 0.0, 1.0)


def render(spec: SceneSpec, frames: int = 16, size: int = 64, t0: int = 0, bg=(12, 12, 16)) -> np.ndarray:
    """Render frames t0..t0+frames-1 of the (infinite) trajectories as uint8 [T,H,W,3]."""
    scale = size / REF_SIZE
    t = np.arange(t0, t0 + frames, dtype=np.float64)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    img = np.empty((frames, size, size, 3), dtype=np.float64)
    img[:] = np.array(bg, dtype=np.float64)
    for e in spec.entities:
        px, py = positions(e, t)
        cov = _coverage(e.shape or "circle", px * size, py * size, e.radius * scale, xs, ys)[..., None]
        col = np.array(L.COLOR_RGB[e.color], dtype=np.float64)
        img = img * (1 - cov) + col * cov
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def captions_for(spec: SceneSpec) -> dict:
    return {lang: {style: caption(spec, lang, style) for style in ("shape", "object")} for lang in L.LANGS}


def make_sample(seed: int, index: int, cfg: SynthConfig = SynthConfig()) -> Sample:
    rng = np.random.default_rng([int(seed), int(index), 0x4E495652])
    spec = sample_spec(rng, cfg)
    bgv = rng.integers(8, 20, size=3)
    video = render(spec, cfg.frames, cfg.size, 0, bg=tuple(int(v) for v in bgv))
    return Sample(index=int(index), video=video, spec=spec, captions=captions_for(spec))


def fingerprint(seed: int, start: int, count: int, cfg: SynthConfig = SynthConfig()) -> str:
    blob = json.dumps({"generator_version": GENERATOR_VERSION, "config": cfg.to_dict(), "seed": int(seed),
                       "start": int(start), "count": int(count)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def manifest(seed: int, start: int, count: int, cfg: SynthConfig = SynthConfig()) -> dict:
    return {"generator_version": GENERATOR_VERSION, "config": cfg.to_dict(), "seed": int(seed),
            "start": int(start), "count": int(count), "fingerprint": fingerprint(seed, start, count, cfg)}


def content_hash(seed: int, start: int, count: int, cfg: SynthConfig = SynthConfig()) -> str:
    """Hash of the actual rendered pixels + specs (shard manifests / determinism tests)."""
    h = hashlib.sha256()
    for i in range(start, start + count):
        s = make_sample(seed, i, cfg)
        h.update(s.video.tobytes())
        h.update(json.dumps(s.spec.to_dict(), sort_keys=True).encode())
    return h.hexdigest()


class SynthDataset:
    """Index-addressable view over the generator with an optional in-memory cache."""

    def __init__(self, seed: int, start: int, count: int, cfg: SynthConfig = SynthConfig(), cache: bool = True):
        self.seed, self.start, self.count, self.cfg = seed, start, count, cfg
        self._cache: Optional[dict] = {} if cache else None

    def __len__(self):
        return self.count

    def sample(self, i: int) -> Sample:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        s = make_sample(self.seed, self.start + i, self.cfg)
        if self._cache is not None:
            self._cache[i] = s
        return s

    def manifest(self):
        return manifest(self.seed, self.start, self.count, self.cfg)

    @property
    def fingerprint(self):
        return fingerprint(self.seed, self.start, self.count, self.cfg)
