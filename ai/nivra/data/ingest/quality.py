"""Quality statistics / filters and perceptual-hash deduplication for decoded clips."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

N_HASH_FRAMES = 8


@dataclass
class QualityLimits:
    min_brightness: float = 12.0      # mean luma (0-255) below this -> "black"
    max_brightness: float = 243.0     # above this -> "too_bright"
    min_motion: float = 0.35          # mean |frame diff| (0-255 luma, on a 1/4-res grid) below this -> "static"
    min_contrast: float = 4.0         # luma std over the clip below this -> "flat" (blank / single colour)


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
    return {"brightness": round(float(np.mean(means)), 3), "contrast": round(float(np.mean(stds)), 3),
            "motion": round(float(np.mean(diffs)) if diffs else 0.0, 4),
            "motion_max": round(float(np.max(diffs)) if diffs else 0.0, 4)}


def quality_reject(stats: dict, lim: QualityLimits) -> str | None:
    if stats["brightness"] < lim.min_brightness:
        return "black"
    if stats["brightness"] > lim.max_brightness:
        return "too_bright"
    if stats["contrast"] < lim.min_contrast:
        return "flat"
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
