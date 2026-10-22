"""Quality statistics / filters and perceptual-hash deduplication for decoded clips."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

N_HASH_FRAMES = 8
HOLD_EPS = 0.05          # mean |luma diff| (0-255) at or below this = the frame was held (codec noise only)


@dataclass
class QualityLimits:
    min_brightness: float = 12.0      # mean luma (0-255) below this -> "black"
    max_brightness: float = 243.0     # above this -> "too_bright"
    min_motion: float = 0.35          # mean |frame diff| (0-255 luma, on a 1/4-res grid) below this -> "static"
    min_contrast: float = 4.0         # luma std over the clip below this -> "flat" (blank / single colour)
    # Held-frame aware "static" test (stylized profile; 0 = off, the plain `min_motion` test is used):
    min_motion_active: float = 0.0    # mean |diff| over the frame pairs that actually change, below this -> "static"
    min_active_fraction: float = 0.0  # fraction of frame pairs that change at all, below this -> "static"

    @staticmethod
    def for_profile(profile: str) -> "QualityLimits":
        """'live' = the original thresholds. 'stylized': large flat colour areas are normal in anime / cel
        shading (reject only truly single-colour clips) and animation is drawn on 2s / 3s (frames are held),
        so a clip is static only if almost none of its frame pairs change or the changes are negligible."""
        if profile == "live":
            return QualityLimits()
        if profile == "stylized":
            return QualityLimits(min_contrast=1.0, min_motion=0.0, min_motion_active=0.3, min_active_fraction=0.15)
        raise ValueError(f"unknown ingest profile {profile!r}; use 'stylized' or 'live'")


def _luma(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def quality_stats(clip: np.ndarray) -> dict:
    """Stats on a uint8 [T,H,W,3] clip, computed on a 4x-subsampled grid, a few frames at a time (bounded RAM)."""
    T = clip.shape[0]
    prev, diffs, means, stds = None, [], [], []
    for i in range(0, T, 4 if T > 32 else 1):
        y = _luma(clip[i, ::4, ::4])
        means.append(float(y.mean()))
        stds.append(float(y.std()))
        if prev is not None:
            diffs.append(float(np.abs(y - prev).mean()))
        prev = y
    active = [d for d in diffs if d > HOLD_EPS]
    return {"brightness": round(float(np.mean(means)), 3), "contrast": round(float(np.mean(stds)), 3),
            "motion": round(float(np.mean(diffs)) if diffs else 0.0, 4),
            "motion_max": round(float(np.max(diffs)) if diffs else 0.0, 4),
            "motion_active": round(float(np.mean(active)) if active else 0.0, 4),
            "active_fraction": round(len(active) / len(diffs), 4) if diffs else 0.0}


def quality_reject(stats: dict, lim: QualityLimits) -> str | None:
    if stats["brightness"] < lim.min_brightness:
        return "black"
    if stats["brightness"] > lim.max_brightness:
        return "too_bright"
    if stats["contrast"] < lim.min_contrast:
        return "flat"
    if lim.min_motion_active > 0 or lim.min_active_fraction > 0:
        if (stats.get("motion_active", 0.0) < lim.min_motion_active
                or stats.get("active_fraction", 0.0) < lim.min_active_fraction):
            return "static"
    if stats["motion"] < lim.min_motion:
        return "static"
    return None


def dhash_frame(frame: np.ndarray) -> int:
    """64-bit difference hash: gray -> 8x9 area average -> sign of horizontal gradient."""
    y = torch.from_numpy(_luma(frame[None])[0])[None, None]
    small = F.adaptive_avg_pool2d(y, (8, 9))[0, 0]
    bits = (small[:, 1:] > small[:, :-1]).flatten().tolist()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def clip_phash(clip: np.ndarray) -> list[int]:
    T = clip.shape[0]
    idx = np.linspace(0, T - 1, N_HASH_FRAMES).round().astype(int)
    return [dhash_frame(clip[i]) for i in idx]


def phash_hex(h: list[int]) -> str:
    return "".join(f"{x:016x}" for x in h)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class DedupIndex:
    """Near-duplicate detector: mean per-frame Hamming distance (bits of 64) <= threshold."""

    def __init__(self, threshold: float = 4.0):
        self.threshold = threshold
        self._rows: list[np.ndarray] = []
        self._arr: np.ndarray | None = None

    def _matrix(self) -> np.ndarray:
        if self._arr is None or len(self._arr) != len(self._rows):
            self._arr = np.stack(self._rows) if self._rows else np.zeros((0, N_HASH_FRAMES), dtype=np.uint64)
        return self._arr

    def nearest(self, h: list[int]) -> tuple[int, float]:
        """(index, mean distance) of the closest stored hash, or (-1, inf)."""
        m = self._matrix()
        if not len(m):
            return -1, float("inf")
        x = np.bitwise_xor(m, np.array(h, dtype=np.uint64)[None, :])
        dist = _POP[x.view(np.uint8)].reshape(len(m), -1).sum(1) / N_HASH_FRAMES
        i = int(np.argmin(dist))
        return i, float(dist[i])

    def is_duplicate(self, h: list[int]) -> bool:
        return self.nearest(h)[1] <= self.threshold

    def add(self, h: list[int]):
        self._rows.append(np.array(h, dtype=np.uint64))


# ===================================================================== animation frame holds ("on 2s / 3s")
HOLD_ABS = 0.15      # mean |luma diff| (0-255) at or below this is never a new drawing (codec noise)
HOLD_REL = 0.08      # ... nor is anything below this fraction of the clip's 90th-percentile frame difference
MAX_HOLD = 6


def gray_diffs(gray: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames of a uint8/float gray clip [T,h,w] -> [T-1]."""
    g = gray.astype(np.float32)
    if g.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    return np.abs(g[1:] - g[:-1]).mean(axis=(1, 2))


def estimate_holds(diffs: np.ndarray, src_fps: float, max_hold: int = MAX_HOLD) -> dict:
    """Effective drawing rate of a clip from its consecutive-frame differences.

    A frame pair is a CHANGE when its difference exceeds max(HOLD_ABS, HOLD_REL * p90(diffs)); the gaps between
    consecutive changes are how many frames each drawing is shown. `hold` is the most common gap (1 = on ones,
    2 = on twos, 3 = on threes, ...), `effective_fps` = src_fps / mean gap (a 2,3,2,3 pattern gives ~9.6 fps at
    24), `hold_purity` the share of gaps equal to `hold`, `hold_fraction` the share of frame pairs that are
    repeats. `hold` / `effective_fps` are None when the clip has fewer than two changes (static or too short to
    tell). The first and last drawing are cut by the clip window, so only gaps BETWEEN changes are used."""
    d = np.asarray(diffs, dtype=np.float64)
    out = {"hold": None, "effective_fps": None, "hold_purity": None, "hold_fraction": None, "changes": 0}
    if d.size < 3 or src_fps <= 0:
        return out
    thr = max(HOLD_ABS, HOLD_REL * float(np.percentile(d, 90)))
    changed = np.flatnonzero(d > thr)
    out["changes"] = int(changed.size)
    out["hold_fraction"] = round(float(1.0 - changed.size / d.size), 4)
    if changed.size < 2:
        return out
    gaps = np.diff(changed)
    gaps = gaps[gaps <= max_hold * 2] if (gaps <= max_hold * 2).any() else gaps     # a pause is not a drawing rate
    counts = np.bincount(np.minimum(gaps, max_hold).astype(int), minlength=max_hold + 1)
    hold = int(np.argmax(counts[1:]) + 1)
    out["hold"] = hold
    out["hold_purity"] = round(float(counts[hold] / gaps.size), 4)
    out["effective_fps"] = round(float(src_fps / max(1.0, float(gaps.mean()))), 3)
    return out
