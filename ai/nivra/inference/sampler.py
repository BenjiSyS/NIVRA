"""Euler ODE sampler for rectified flow with classifier-free guidance and temporal inpainting."""
from __future__ import annotations

from typing import Callable, Optional

import torch

from ..text.tokenizer import default_tokenizer


class Stopped(Exception):
    pass


@torch.no_grad()
def sample_latents(model, prompts_ids: torch.Tensor, shape: tuple, steps: int = 30, cfg_scale: float = 3.0,
                   seed: int = 0, known: Optional[torch.Tensor] = None, known_start: int = 0,
                   noise: Optional[torch.Tensor] = None, should_stop: Optional[Callable] = None,
                   device: str = "cpu", on_step: Optional[Callable] = None) -> torch.Tensor:
    """Sample scaled latents.

    prompts_ids: [B,L] token ids. shape: (B,C,T,H,W).
    known: optional [B,C,K,H,W] clean (scaled) latents for latent frames known_start..known_start+K-1.
      Temporal conditioning is inpainting-style: at each step the known frames are replaced by their
      noised version on the straight flow path (1-t)*known + t*noise, with the same noise as the
      initial sample, and are set exactly to `known` at the end.
    """
    B = shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    if noise is None:
        noise = torch.randn(shape, generator=gen).to(device)
    x = noise.clone()
    tok = default_tokenizer()
    empty = torch.tensor(tok.encode(""), device=device)[None].expand(B, -1)
    ids = prompts_ids.to(device)
    use_cfg = cfg_scale != 1.0
    text_c = model.encode_text(ids)
    text_u = model.encode_text(empty) if use_cfg else None
    ts = torch.linspace(1.0, 0.0, steps + 1)
    K = known.shape[2] if known is not None else 0
    sl = slice(known_start, known_start + K)
    for i in range(steps):
        if should_stop is not None and should_stop():
            raise Stopped()
        t, t_next = float(ts[i]), float(ts[i + 1])
        if K:
            x[:, :, sl] = (1 - t) * known + t * noise[:, :, sl]
        tt = torch.full((B,), t, device=device)
        v_c = model(x, tt, text=text_c)
        if use_cfg:
            v_u = model(x, tt, text=text_u)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c
        x = x + (t_next - t) * v
        if on_step:
            on_step(i + 1, steps)
    if K:
        x[:, :, sl] = known
    return x
