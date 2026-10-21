"""NivraSmall: rectified-flow transformer over VAE latents.

Factorized spatial / temporal self-attention, cross-attention to text tokens,
adaLN-Zero timestep(+pooled text) conditioning, patch size 2 and learned factorized
3D positional embeddings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..text.encoder import TextEncoder

ARCH_VERSION = "nivra-small/1"


@dataclass
class GenConfig:
    arch_version: str = ARCH_VERSION
    latent_channels: int = 4
    latent_t: int = 8
    latent_h: int = 16
    latent_w: int = 16
    patch: int = 2
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 3
    text_layers: int = 2
    text_heads: int = 4
    max_text_len: int = 32

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        keys = GenConfig.__dataclass_fields__.keys()
        return GenConfig(**{k: v for k, v in d.items() if k in keys})


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx=None, mask=None):
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        M = ctx.shape[1]
        h = self.heads
        q = self.q(x).view(B, N, h, D // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, M, 2, h, D // h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class NivraBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        ln = lambda: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n_s, self.n_t, self.n_c, self.n_m = ln(), ln(), ln(), ln()
        self.spatial = MHA(dim, heads)
        self.temporal = MHA(dim, heads)
        self.cross = MHA(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(approximate="tanh"), nn.Linear(dim * mlp_ratio, dim))
        self.ada = nn.Linear(dim, 12 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, text, text_mask):
        # x: [B,T,S,D] (S = h*w tokens per latent frame); c: [B,D]; text: [B,L,D]
        B, T, S, D = x.shape
        m = self.ada(F.silu(c)).view(B, 12, 1, 1, D).unbind(1)
        # spatial attention within each latent frame
        h = modulate(self.n_s(x), m[0], m[1]).reshape(B * T, S, D)
        x = x + m[2] * self.spatial(h).view(B, T, S, D)
        # temporal attention across frames at each spatial location
        h = modulate(self.n_t(x), m[3], m[4]).permute(0, 2, 1, 3).reshape(B * S, T, D)
        x = x + m[5] * self.temporal(h).view(B, S, T, D).permute(0, 2, 1, 3)
        # cross attention to text
        h = modulate(self.n_c(x), m[6], m[7]).reshape(B, T * S, D)
        attn_mask = text_mask[:, None, None, :]
        x = x + m[8] * self.cross(h, text, attn_mask).view(B, T, S, D)
        # MLP
        h = modulate(self.n_m(x), m[9], m[10])
        return x + m[11] * self.mlp(h)


class NivraSmall(nn.Module):
    def __init__(self, cfg: GenConfig, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        d, p, C = cfg.dim, cfg.patch, cfg.latent_channels
        self.gt, self.gh, self.gw = cfg.latent_t, cfg.latent_h // p, cfg.latent_w // p
        self.text = TextEncoder(vocab_size, d, cfg.text_layers, cfg.text_heads, cfg.max_text_len, pad_id)
        self.patch_in = nn.Linear(C * p * p, d)
        self.pos_t = nn.Parameter(torch.randn(self.gt, d) * 0.02)
        self.pos_h = nn.Parameter(torch.randn(self.gh, d) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(self.gw, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.text_pool = nn.Linear(d, d)
        self.blocks = nn.ModuleList(NivraBlock(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.final_ada = nn.Linear(d, 2 * d)
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final = nn.Linear(d, C * p * p)
        for m in (self.final_ada, self.final):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.d = d

    def encode_text(self, ids):
        return self.text(ids)

    def forward(self, x, t, ids=None, text=None):
        """x: [B,C,T,H,W] noisy latents; t: [B] in [0,1]; ids: [B,L] token ids. Returns velocity [B,C,T,H,W]."""
        B, C, T, H, W = x.shape
        p, d = self.cfg.patch, self.d
        if text is None:
            text = self.text(ids)
        txt, tmask = text
        gh, gw = H // p, W // p
        tok = x.view(B, C, T, gh, p, gw, p).permute(0, 2, 3, 5, 1, 4, 6).reshape(B, T, gh * gw, C * p * p)
        h = self.patch_in(tok)
        pos = (self.pos_t[:T, None, None, :] + self.pos_h[None, :gh, None, :] + self.pos_w[None, None, :gw, :]).reshape(T, gh * gw, d)
        h = h + pos[None]
        denom = tmask.sum(1, keepdim=True).clamp_min(1)
        pooled = (txt * tmask[..., None]).sum(1) / denom
        c = self.t_mlp(timestep_embedding(t, d)) + self.text_pool(pooled)
        for blk in self.blocks:
            h = blk(h, c, txt, tmask)
        shift, scale = self.final_ada(F.silu(c)).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift[:, None, None], scale[:, None, None])
        out = self.final(h)  # [B,T,S,C*p*p]
        out = out.view(B, T, gh, gw, C, p, p).permute(0, 4, 1, 2, 5, 3, 6).reshape(B, C, T, H, W)
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def flow_matching_loss(model, x0, ids, gen: torch.Generator, t=None, noise=None):
    """Rectified flow: x_t=(1-t)x0+t*noise, target v = noise - x0, logit-normal t."""
    B = x0.shape[0]
    if noise is None:
        noise = torch.randn(x0.shape, generator=gen).to(device=x0.device, dtype=x0.dtype)
    if t is None:
        t = torch.sigmoid(torch.randn(B, generator=gen)).to(device=x0.device, dtype=x0.dtype)
    tt = t.view(B, 1, 1, 1, 1)
    xt = (1 - tt) * x0 + tt * noise
    v = model(xt, t, ids)
    target = noise - x0
    return F.mse_loss(v.float(), target.float())
