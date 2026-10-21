"""Classical blob detector for the synthetic domain: colour (RGB-normalised chromaticity),
shape (fill ratio / bbox), per-frame centroids and tracks -> motion direction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data import lexicon as L

_PAL = {c: np.array(v, dtype=np.float64) / max(v) for c, v in L.COLOR_RGB.items()}
_PAL_NAMES = list(_PAL)
_PAL_MAT = np.stack([_PAL[c] for c in _PAL_NAMES])      # [6,3]

FG_MIN_MAX_CHANNEL = 90      # pixel counts as foreground if its brightest channel exceeds this
COLOR_MAX_DIST = 0.28        # max chromaticity distance to a palette colour
MIN_AREA = 14                # pixels (at 64x64)
MOVE_THRESHOLD_PX = 4.0      # total displacement (at 64 px) above which an entity counts as moving
LINK_MAX_JUMP = 14.0


@dataclass
class Blob:
    color: str
    shape: str
    cx: float
    cy: float
    area: int
    bbox: tuple


@dataclass
class Track:
    color: str
    shapes: list
    frames: list          # frame indices
    cx: list
    cy: list
    areas: list

    @property
    def shape(self):
        return max(set(self.shapes), key=self.shapes.count)


def classify_pixels(frame: np.ndarray):
    """frame uint8 [H,W,3] -> label map int [H,W] (-1 background, else palette index)."""
    f = frame.astype(np.float64)
    mx = f.max(axis=2)
    fg = mx > FG_MIN_MAX_CHANNEL
    chroma = f / np.maximum(mx, 1.0)[..., None]
    d = np.linalg.norm(chroma[..., None, :] - _PAL_MAT[None, None], axis=-1)   # [H,W,6]
    idx = d.argmin(-1)
    ok = fg & (d.min(-1) < COLOR_MAX_DIST)
    return np.where(ok, idx, -1), fg


def _components(mask: np.ndarray):
    """4-connected components of a boolean mask -> list of (ys, xs) arrays."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        py, px = [], []
        while stack:
            y, x = stack.pop()
            py.append(y)
            px.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((np.array(py), np.array(px)))
    return out


def classify_shape(ys, xs) -> str:
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    fill = len(ys) / float(h * w)
    # circle ~0.785, square ~1.0, triangle ~0.5 (decision boundaries midway)
    if fill < 0.64:
        return "triangle"
    if fill < 0.89:
        return "circle"
    return "square"


def detect_frame(frame: np.ndarray, min_area: Optional[int] = None) -> list:
    H = frame.shape[0]
    min_area = int(MIN_AREA * (H / 64.0) ** 2) if min_area is None else min_area
    labels, _ = classify_pixels(frame)
    blobs = []
    for ci, cname in enumerate(_PAL_NAMES):
        for ys, xs in _components(labels == ci):
            if len(ys) < min_area:
                continue
            blobs.append(Blob(cname, classify_shape(ys, xs), float(xs.mean() + 0.5) / H * 64.0,
                              float(ys.mean() + 0.5) / H * 64.0, int(len(ys) * (64.0 / H) ** 2),
                              (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    return blobs


def build_tracks(per_frame: list) -> list:
    """Greedy nearest-neighbour linking of same-colour blobs between consecutive frames."""
    tracks: list = []
    active: list = []   # tracks seen in previous frame
    for t, blobs in enumerate(per_frame):
        unused = list(range(len(blobs)))
        new_active = []
        for tr in active:
            best, bd = None, LINK_MAX_JUMP
            for j in unused:
                b = blobs[j]
                if b.color != tr.color:
                    continue
                d = np.hypot(b.cx - tr.cx[-1], b.cy - tr.cy[-1])
                if d < bd:
                    best, bd = j, d
            if best is not None:
                b = blobs[best]
                unused.remove(best)
                tr.frames.append(t); tr.cx.append(b.cx); tr.cy.append(b.cy); tr.areas.append(b.area); tr.shapes.append(b.shape)
                new_active.append(tr)
        for j in unused:
            b = blobs[j]
            tr = Track(b.color, [b.shape], [t], [b.cx], [b.cy], [b.area])
            tracks.append(tr)
            new_active.append(tr)
        active = new_active
    return tracks


def track_motion(tr: Track, n_frames: int) -> str:
    k = max(1, min(3, len(tr.cx) // 3))
    dx = float(np.mean(tr.cx[-k:]) - np.mean(tr.cx[:k]))
    dy = float(np.mean(tr.cy[-k:]) - np.mean(tr.cy[:k]))
    if np.hypot(dx, dy) < MOVE_THRESHOLD_PX:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "l2r" if dx > 0 else "r2l"
    return "down" if dy > 0 else "up"


@dataclass
class Detection:
    n_frames: int
    per_frame: list
    tracks: list            # significant tracks only
    motions: list           # motion label per significant track
    max_blobs_per_frame: int = 0


def detect_video(video: np.ndarray, min_track_frac: float = 0.5) -> Detection:
    per_frame = [detect_frame(f) for f in video]
    tracks = build_tracks(per_frame)
    T = len(video)
    sig = [t for t in tracks if len(t.frames) >= max(2, int(min_track_frac * T))]
    return Detection(T, per_frame, sig, [track_motion(t, T) for t in sig], max((len(b) for b in per_frame), default=0))
