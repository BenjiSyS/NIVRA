"""Nivra video tokenizer: 3D-conv VAE with continuous latents.

3x16x64x64 video -> C x8x16x16 latent (spatial x4, temporal x2).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

VAE_ARCH_VERSION = "nivra-vae/1"


@dataclass
class VAEConfig:
    arch_version: str = VAE_ARCH_VERSION
    in_channels: int = 3
    latent_channels: int = 4
    c1: int = 32
    c2: int = 48
    n_res: int = 1
    kl_weight: float = 1e-4
    motion_weight: float = 1.0
    spatial_factor: int = 4
    temporal_factor: int = 2

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        keys = VAEConfig.__dataclass_fields__.keys()
        return VAEConfig(**{k: v for k, v in d.items() if k in keys})

    def latent_shape(self, frames=16, size=64):
        return (self.latent_channels, frames // self.temporal_factor,
                size // self.spatial_factor, size // self.spatial_factor)


def _gn(c):
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class ResBlock3d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1 = _gn(c), nn.Conv3d(c, c, 3, padding=1)
        self.n2, self.c2 = _gn(c), nn.Conv3d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class VideoVAE(nn.Module):
    def __init__(self, cfg: VAEConfig = VAEConfig()):
        super().__init__()
        self.cfg = cfg
        C = cfg.latent_channels
        self.enc_in = nn.Conv3d(cfg.in_channels, cfg.c1, (3, 4, 4), stride=(1, 2, 2), padding=1)
        self.enc_down = nn.Conv3d(cfg.c1, cfg.c2, 4, stride=2, padding=1)
        self.enc_res = nn.Sequential(*[ResBlock3d(cfg.c2) for _ in range(cfg.n_res)])
        self.enc_norm = _gn(cfg.c2)
        self.enc_out = nn.Conv3d(cfg.c2, 2 * C, 3, padding=1)
        self.dec_in = nn.Conv3d(C, cfg.c2, 3, padding=1)
        self.dec_res = nn.Sequential(*[ResBlock3d(cfg.c2) for _ in range(cfg.n_res)])
        self.dec_norm = _gn(cfg.c2)
        self.dec_up1 = nn.ConvTranspose3d(cfg.c2, cfg.c1, 4, stride=2, padding=1)
        self.dec_norm2 = _gn(cfg.c1)
        self.dec_up2 = nn.ConvTranspose3d(cfg.c1, cfg.in_channels, (3, 4, 4), stride=(1, 2, 2), padding=1)
        self.register_buffer("scaling_factor", torch.tensor(1.0))

    # x: [B,3,T,H,W] in [-1,1]
    def encode_dist(self, x):
        h = F.silu(self.enc_in(x))
        h = F.silu(self.enc_down(h))
        h = self.enc_res(h)
        h = self.enc_out(F.silu(self.enc_norm(h)))
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar.clamp(-20.0, 10.0)

    def encode(self, x, sample=False):
        mean, logvar = self.encode_dist(x)
        if sample:
            return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return mean

    def decode_raw(self, z):
        h = self.dec_in(z)
        h = self.dec_res(h)
        h = F.silu(self.dec_norm(h))
        h = F.silu(self.dec_norm2(self.dec_up1(h)))
        return self.dec_up2(h)

    def forward(self, x, sample=True):
        mean, logvar = self.encode_dist(x)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean
        return self.decode_raw(z), mean, logvar

    # --- scaled-latent API used by the generator ---
    def to_scaled(self, z):
        return z * self.scaling_factor

    def from_scaled(self, z):
        return z / self.scaling_factor

    @torch.no_grad()
    def decode(self, z_scaled, tile=0, overlap=2):
        """Decode scaled latents [B,C,T,H,W] -> video [B,3,2T,4H,4W] in [-1,1].
        tile>0: decode in spatial/temporal tiles of `tile` latent cells with `overlap` blended cells."""
        z = self.from_scaled(z_scaled)
        if not tile:
            return self.decode_raw(z)
        return _tiled_decode(self, z, tile, overlap)


def _starts(n, tile, step):
    if n <= tile:
        return [0]
    return sorted(set(list(range(0, n - tile, step)) + [n - tile]))


def _feather(n, ov, at_start, at_end):
    w = torch.ones(n)
    if ov > 0:
        ramp = (torch.arange(ov, dtype=torch.float32) + 1) / (ov + 1)
        if at_start:
            w[:ov] = ramp
        if at_end:
            w[n - ov:] = ramp.flip(0)
    return w


def _tiled_decode(vae: VideoVAE, z, tile, overlap):
    B, C, T, H, W = z.shape
    tf, sf = vae.cfg.temporal_factor, vae.cfg.spatial_factor
    out = torch.zeros(B, vae.cfg.in_channels, T * tf, H * sf, W * sf)
    wsum = torch.zeros(1, 1, T * tf, H * sf, W * sf)
    step = max(1, tile - overlap)
    for t0 in _starts(T, tile, step):
        for h0 in _starts(H, tile, step):
            for w0 in _starts(W, tile, step):
                t1, h1, w1 = min(T, t0 + tile), min(H, h0 + tile), min(W, w0 + tile)
                piece = vae.decode_raw(z[:, :, t0:t1, h0:h1, w0:w1])
                wt = _feather((t1 - t0) * tf, overlap * tf, t0 > 0, t1 < T)
                wh = _feather((h1 - h0) * sf, overlap * sf, h0 > 0, h1 < H)
                ww = _feather((w1 - w0) * sf, overlap * sf, w0 > 0, w1 < W)
                w3 = wt[:, None, None] * wh[None, :, None] * ww[None, None, :]
                ts, hs, ws = slice(t0 * tf, t1 * tf), slice(h0 * sf, h1 * sf), slice(w0 * sf, w1 * sf)
                out[:, :, ts, hs, ws] += piece * w3
                wsum[:, :, ts, hs, ws] += w3
    return out / wsum.clamp_min(1e-6)


def vae_loss(vae: VideoVAE, x, sample=True):
    """x in [-1,1], [B,3,T,H,W]. Returns total loss tensor and metric dict."""
    cfg = vae.cfg
    recon, mean, logvar = vae(x, sample=sample)
    l1 = (recon - x).abs().mean()
    mse = F.mse_loss(recon, x)
    dx, dr = x[:, :, 1:] - x[:, :, :-1], recon[:, :, 1:] - recon[:, :, :-1]
    motion = (dr - dx).abs().mean()
    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
    total = l1 + mse + cfg.motion_weight * motion + cfg.kl_weight * kl
    return total, {"l1": l1.item(), "mse": mse.item(), "motion_l1": motion.item(), "kl": kl.item()}


@torch.no_grad()
def reconstruction_metrics(vae: VideoVAE, videos_u8, batch=8, tile=0):
    """videos_u8: uint8 tensor [N,T,H,W,3]. PSNR (dB, 0..255 scale), motion error (L1 of frame differences)."""
    vae.eval()
    se, motion, n = 0.0, 0.0, 0
    enc_t = dec_t = 0.0
    for i in range(0, videos_u8.shape[0], batch):
        x = videos_u8[i:i + batch].permute(0, 4, 1, 2, 3).float() / 127.5 - 1.0
        t0 = time.time()
        z = vae.to_scaled(vae.encode(x))
        t1 = time.time()
        r = vae.decode(z, tile=tile).clamp(-1, 1)
        t2 = time.time()
        enc_t += t1 - t0
        dec_t += t2 - t1
        xq = ((x + 1) * 127.5).round()
        rq = ((r + 1) * 127.5).round()
        se += ((xq - rq) ** 2).mean(dim=(1, 2, 3, 4)).sum().item()
        dx, dr = x[:, :, 1:] - x[:, :, :-1], r[:, :, 1:] - r[:, :, :-1]
        motion += (dr - dx).abs().mean(dim=(1, 2, 3, 4)).sum().item() * 127.5
        n += x.shape[0]
    mse = se / n
    psnr = 10 * torch.log10(torch.tensor(255.0 ** 2 / max(mse, 1e-9))).item()
    return {"psnr_db": psnr, "motion_l1_255": motion / n, "mse_255": mse,
            "encode_s_per_clip": enc_t / n, "decode_s_per_clip": dec_t / n, "clips": n}
