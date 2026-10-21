"""Worker side of DiLoCo: H inner AdamW steps from a base version -> pseudo-gradient update."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..config import build_gen, load_config, synth_config
from ..data.synthetic import fingerprint
from ..inference.bundle import BundleError
from ..inference.sampler import Stopped
from ..models.nivra_small import count_params, flow_matching_loss
from ..modelio import load_vae
from ..text.tokenizer import default_tokenizer
from ..training.checkpoint import sha256_file
from ..training.tasks import GenTask, encode_dataset, heldout_flow_loss
from ..training.trainer import EpochSampler
from . import version as ver
from .update import write_update, l2_norm


def load_base(base_weights, base_manifest, vae_path):
    try:
        m = ver.read_manifest(base_manifest)
        w = ver.load_weights(m, base_weights)
    except (ver.VersionError, OSError) as e:
        raise BundleError(str(e)) from e
    if m["vocab_hash"] != default_tokenizer().vocab_hash:
        raise BundleError("tokenizer vocab hash mismatch")
    if m["vae_sha256"] and sha256_file(vae_path) != m["vae_sha256"]:
        raise BundleError("VAE weights do not match manifest vae_sha256")
    cfg = load_config(m["config"])
    model = build_gen(cfg)
    try:
        model.load_state_dict(w, strict=True)
    except RuntimeError as e:
        raise BundleError(f"base weights incompatible with architecture: {e}") from e
    return m, cfg, model, w


def local_train(base_weights, base_manifest, vae_path, shard: dict, update_id: str, out_dir, *,
                inner_steps: int = 30, batch_size: int = 8, lr: float = 3e-4, weight_decay: float = 0.0,
                grad_clip: float = 1.0, seed: int = 0, eval_clips: int = 16, device: str = "cpu",
                should_stop: Optional[Callable] = None, on_metric: Optional[Callable] = None) -> dict:
    t_start = time.time()
    m, cfg, model, base = load_base(base_weights, base_manifest, vae_path)
    model.to(device)
    vae = load_vae(vae_path).to(device)
    scfg = synth_config(cfg)
    n = int(shard["count"])
    n_train = n - eval_clips
    if n_train < batch_size:
        raise ValueError("shard too small for batch_size + eval_clips")
    lat, ids = encode_dataset(vae, scfg, shard["seed"], shard["start"], n, device=device)
    del vae
    task = GenTask(model, lat[:n_train], ids[:n_train], scfg, m["vae_sha256"] or "", seed=shard["seed"], start=shard["start"])
    loss_before = heldout_flow_loss(model, lat[n_train:], ids[n_train:, 0]) if eval_clips else None
    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.99))
    sampler = EpochSampler(n_train, batch_size, seed)
    gen = torch.Generator().manual_seed(seed + 7)
    warm = max(1, min(5, inner_steps // 4))
    losses = []
    model.train()
    for step in range(inner_steps):
        if should_stop is not None and should_stop():
            raise Stopped()
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warm)
        batch = task.get_batch(sampler.next(), gen)
        opt.zero_grad(set_to_none=True)
        loss, _ = task.loss(batch, gen)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if torch.isfinite(gn):
            opt.step()
        losses.append(float(loss.item()))
        if on_metric:
            on_metric("loss", losses[-1], step + 1)
    loss_after = heldout_flow_loss(model, lat[n_train:], ids[n_train:, 0]) if eval_clips else None
    local = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    delta = {k: base[k].float() - local[k].float() for k in base}
    k5 = min(5, len(losses))
    meta = {"update_id": update_id, "base_version_id": m["version_id"], "base_weights_sha256": m["weights_sha256"],
            "architecture_version": m["architecture_version"], "vocab_hash": m["vocab_hash"],
            "inner_steps": inner_steps, "samples": inner_steps * batch_size,
            "loss_first": float(np.mean(losses[:k5])), "loss_last": float(np.mean(losses[-k5:])),
            "loss_before": loss_before, "loss_after": loss_after, "delta_l2": l2_norm(delta),
            "shard": {"seed": shard["seed"], "start": shard["start"], "count": n,
                      "fingerprint": fingerprint(shard["seed"], shard["start"], n, scfg)},
            "hyper": {"lr": lr, "batch_size": batch_size, "weight_decay": weight_decay, "grad_clip": grad_clip, "seed": seed},
            "wall_seconds": time.time() - t_start, "device": device}
    return write_update(out_dir, delta, meta)
