"""3D rotary position embedding (time, height, width) for variable-size latent grids."""
from __future__ import annotations

import torch


def rope_axes(head_dim: int) -> tuple:
    """Split an (even) head dim into even (t, h, w) parts: h = w = 2*(hd//6), t gets the rest."""
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    a = 2 * (head_dim // 6)
    t = head_dim - 2 * a
    if a <= 0 or t <= 0:
        raise ValueError(f"head_dim {head_dim} too small for 3D RoPE")
    return t, a, a


def _axis_angles(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    freqs = theta ** (-torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim)
    return pos.double()[:, None] * freqs[None, :]


def rope_3d(T: int, H: int, W: int, head_dim: int, theta: float = 10000.0, t_offset: int = 0,
            device=None) -> tuple:
    """cos, sin of shape [T*H*W, head_dim/2] (float32) for tokens in t-major order (t, h, w)."""
    dt, dh, dw = rope_axes(head_dim)
    t = torch.arange(T, device=device) + t_offset
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)
    gt, gh, gw = torch.meshgrid(t, h, w, indexing="ij")
    ang = torch.cat([_axis_angles(gt.reshape(-1), dt, theta), _axis_angles(gh.reshape(-1), dh, theta),
                     _axis_angles(gw.reshape(-1), dw, theta)], dim=1)
    return ang.cos().float(), ang.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, N, hd]; rotate-half form, computed in float32, returned in x.dtype."""
    half = x.shape[-1] // 2
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    c, s = cos.to(x.device), sin.to(x.device)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)
