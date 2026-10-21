"""Numerical bounds for HD (fp16) pseudo-gradient updates (runs on the Raspberry Pi controller).

An honest worker moves every weight by at most about ``lr`` per optimizer step (Adam-style updates are bounded
by a small constant times lr; the 8-bit optimizer additionally clamps its ratio), so after H inner steps

    ||delta||_inf  <= c * H * lr                    per tensor, per element
    ||delta||_2    <= c * H * lr * sqrt(numel)      per tensor

with c = ``bound_c`` (default 3). Because the bounds are absolute they need neither the base weights nor the other
workers' updates, which makes a single scaled / sign-flipped / random update detectable on its own. They do NOT
prevent a bounded-norm backdoor (documented in HD_TRAINING.md).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

DEFAULT_BOUND_C = 3.0
DEFAULT_MAX_REL_NORM = 0.5
ZERO_UPDATE_NORM = 1e-8
FP16_SLACK = 1.0 + 2.0 ** -9           # the update was rounded to fp16 (relative error <= 2^-11) after the optimizer


def elementwise_bound(c: float, inner_steps: int, lr: float) -> float:
    return float(c) * int(inner_steps) * float(lr)


class BoundViolation(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def check_tensor(name: str, d32: torch.Tensor, bound: float) -> float:
    """Validate one fp32 view of a delta tensor; returns its squared L2 norm (float). Raises BoundViolation."""
    if d32.numel() == 0:
        return 0.0
    if not bool(torch.isfinite(d32).all()):
        raise BoundViolation("non_finite", f"{name}: update contains NaN or Inf")
    linf = float(d32.abs().max())
    lim = bound * FP16_SLACK
    if linf > lim:
        raise BoundViolation("bound_exceeded", f"{name}: ||delta||_inf {linf:.4g} > c*H*lr = {bound:.4g}")
    l2sq = float(d32.double().pow(2).sum())
    l2lim = bound * FP16_SLACK * math.sqrt(d32.numel())
    if math.sqrt(l2sq) > l2lim:
        raise BoundViolation("bound_exceeded", f"{name}: ||delta||_2 {math.sqrt(l2sq):.4g} > c*H*lr*sqrt(numel) = {l2lim:.4g}")
    return l2sq


def median(xs: list) -> float:
    return float(torch.tensor(xs, dtype=torch.float64).median())


def outlier_cut(norms: list, ratio: float) -> Optional[float]:
    """Norms above ratio x median are outliers; needs >= 3 samples to mean anything (None otherwise)."""
    if len(norms) < 3:
        return None
    med = median(norms)
    return ratio * med if med > 0 else None


def clip_cap(norms: list, clip_factor: float) -> float:
    """Updates larger than clip_factor x the median accepted norm are scaled down to it (inf if no clipping)."""
    if not norms:
        return math.inf
    med = median(norms)
    return clip_factor * med if med > 0 else math.inf
