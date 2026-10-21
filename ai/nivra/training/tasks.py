"""Training tasks: VAE reconstruction and latent flow-matching generator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import save_file, load_file

from ..compression.vae import VideoVAE, vae_loss
from ..data import lexicon as L
from ..data.synthetic import SynthConfig, SynthDataset, make_sample, fingerprint
from ..models.nivra_small import NivraSmall, flow_matching_loss
from ..text.tokenizer import default_tokenizer
from ..config import synth_config

DATA_SEED = 1234
HELDOUT_START = 1_000_000
CAPTION_VARIANTS = [("en", "shape"), ("en", "object"), ("sv", "shape"), ("sv", "object")]


def videos_to_tensor(videos_u8: np.ndarray) -> torch.Tensor:
    """uint8 [B,T,H,W,3] -> float [B,3,T,H,W] in [-1,1]."""
    return torch.from_numpy(videos_u8).permute(0, 4, 1, 2, 3).float() / 127.5 - 1.0


# ---------------------------------------------------------------- VAE
class VAETask:
    def __init__(self, vae: VideoVAE, scfg: SynthConfig, n_clips: int, start: int = 0, seed: int = DATA_SEED):
        self.model = vae
        self.ds = SynthDataset(seed, start, n_clips, scfg, cache=True)
        self.n = n_clips
        self.fingerprint = self.ds.fingerprint

    def get_batch(self, idx, gen):
        return videos_to_tensor(np.stack([self.ds.sample(int(i)).video for i in idx]))

    def loss(self, batch, gen):
        dev = next(self.model.parameters()).device
        return vae_loss(self.model, batch.to(dev), sample=True)


@torch.no_grad()
def compute_scaling_factor(vae: VideoVAE, scfg: SynthConfig, n_clips: int = 128, seed: int = DATA_SEED, start: int = 0) -> float:
    """scaling = 1/std of posterior-mean latents over data (so scaled latents have ~unit std)."""
    vae.eval()
    zs = []
    for i in range(0, n_clips, 16):
        vids = np.stack([make_sample(seed, start + j, scfg).video for j in range(i, min(i + 16, n_clips))])
        zs.append(vae.encode(videos_to_tensor(vids)))
    z = torch.cat(zs)
    return float(1.0 / z.std().clamp_min(1e-6))


# ---------------------------------------------------------------- latents
def encode_dataset(vae: VideoVAE, scfg: SynthConfig, seed: int, start: int, count: int, batch: int = 16,
                   device: str = "cpu", progress=None):
    """Returns (latents [N,C,T,H,W] scaled float32, ids [N,4,L] long) for clips [start,start+count)."""
    tok = default_tokenizer()
    vae.eval()
    lat, ids = [], []
    with torch.no_grad():
        for i in range(0, count, batch):
            n = min(batch, count - i)
            samples = [make_sample(seed, start + i + j, scfg) for j in range(n)]
            vids = videos_to_tensor(np.stack([s.video for s in samples])).to(device)
            lat.append(vae.to_scaled(vae.encode(vids)).cpu())
            ids.append(torch.tensor([[tok.encode(s.captions[lg][st]) for lg, st in CAPTION_VARIANTS] for s in samples]))
            if progress:
                progress(i + n, count)
    return torch.cat(lat), torch.cat(ids)


def latent_cache_key(vae_sha: str, scfg: SynthConfig, seed: int, start: int, count: int) -> str:
    return hashlib.sha256(json.dumps([vae_sha, scfg.to_dict(), seed, start, count]).encode()).hexdigest()[:20]


def get_latents(vae, vae_sha, scfg, seed, start, count, cache_dir: Optional[Path], device="cpu", progress=None):
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"latents_{latent_cache_key(vae_sha, scfg, seed, start, count)}.safetensors"
        if p.exists():
            d = load_file(str(p))
            return d["latents"], d["ids"]
    lat, ids = encode_dataset(vae, scfg, seed, start, count, device=device, progress=progress)
    if cache_dir is not None:
        save_file({"latents": lat.contiguous(), "ids": ids.contiguous()}, str(p))
    return lat, ids


class GenTask:
    def __init__(self, model: NivraSmall, latents: torch.Tensor, ids: torch.Tensor, scfg: SynthConfig,
                 vae_sha: str, seed: int = DATA_SEED, start: int = 0, caption_dropout: float = 0.1,
                 p_object_style: float = 0.15):
        self.model = model
        self.latents, self.ids = latents, ids
        self.n = latents.shape[0]
        self.caption_dropout, self.p_object_style = caption_dropout, p_object_style
        self.empty_ids = torch.tensor(default_tokenizer().encode(""))
        self.fingerprint = hashlib.sha256((fingerprint(seed, start, self.n, scfg) + vae_sha).encode()).hexdigest()

    def get_batch(self, idx, gen):
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        B = len(idx_t)
        u = torch.rand(B, 3, generator=gen)
        is_en = u[:, 0] < 0.5
        style = (u[:, 1] < self.p_object_style).long()
        # variants: 0 en/shape, 1 en/object, 2 sv/shape, 3 sv/object
        variant = torch.where(is_en, style, 2 + style)
        ids = self.ids[idx_t, variant]
        drop = u[:, 2] < self.caption_dropout
        ids = torch.where(drop[:, None], self.empty_ids[None].expand_as(ids), ids)
        return self.latents[idx_t], ids

    def loss(self, batch, gen):
        dev = next(self.model.parameters()).device
        z, ids = batch
        loss = flow_matching_loss(self.model, z.to(dev), ids.to(dev), gen)
        return loss, {}


@torch.no_grad()
def heldout_flow_loss(model: NivraSmall, latents: torch.Tensor, ids_cond: torch.Tensor, seed: int = 4242,
                      t_levels=(0.1, 0.3, 0.5, 0.7, 0.9), batch: int = 16) -> float:
    """Fixed-noise, fixed-t flow-matching loss on held-out latents (comparable across models)."""
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    noise_all = torch.randn(latents.shape, generator=gen)
    tot, cnt = 0.0, 0
    for ti, t in enumerate(t_levels):
        for i in range(0, latents.shape[0], batch):
            z = latents[i:i + batch].to(dev)
            nz = noise_all[i:i + batch].to(dev)
            tv = torch.full((z.shape[0],), t, device=dev)
            loss = flow_matching_loss(model, z, ids_cond[i:i + batch].to(dev), gen, t=tv, noise=nz)
            tot += loss.item() * z.shape[0]
            cnt += z.shape[0]
    model.train()
    return tot / cnt
