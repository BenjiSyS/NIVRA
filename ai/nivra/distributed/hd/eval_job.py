"""Held-out evaluation of an HD DiT on fixed (latent, t, noise) triples (Pi gate and ``ai.hd_eval``).

The triples are a pure function of (triples_seed, the val shards' content/order, n, buckets), so the base and the
candidate version are compared on identical inputs. Rectified-flow loss, text-conditioned, no caption dropout.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

import torch
from safetensors.torch import load_file

from ...models.dit import rf_loss
from ...models.hd_config import build_dit, load_hd_config
from .latent_shard import LatentShard
from .stream_io import StReader

EVAL_SCHEMA = "nivra.hd_eval/1"


def make_triples(shards: list, seed: int, n: int, buckets: int) -> list:
    """[(shard_index, clip_index, t, noise_seed)] of length n; clips drawn without replacement while possible."""
    pairs = [(si, i) for si, s in enumerate(shards) for i in range(len(s))]
    if not pairs:
        raise ValueError("no validation clips")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    buckets = max(1, int(buckets))
    out = []
    for j in range(n):
        si, ci = pairs[j % len(pairs)]
        t = ((j % buckets) + 0.5) / buckets
        out.append((si, ci, t, seed * 1000 + j))
    return out


def load_dit(manifest: dict, weights_path) -> torch.nn.Module:
    """fp32 DiT from a manifest config + an fp32/fp16 weights file."""
    cfg = load_hd_config(manifest["config"])
    model = build_dit(cfg)
    model.load_state_dict(load_file(str(weights_path)), strict=True)    # copy_ upcasts fp16 -> fp32
    return model.eval().requires_grad_(False)


@torch.no_grad()
def heldout_loss(model, shard_paths: list, seed: int, n: int = 8, buckets: int = 4,
                 should_stop: Optional[Callable] = None) -> dict:
    shards = [LatentShard(p) for p in sorted(str(x) for x in shard_paths)]
    triples = make_triples(shards, seed, n, buckets)
    losses, per_bucket = [], {}
    for j, (si, ci, t, ns) in enumerate(triples):
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped")
        z, e, m = shards[si].get(ci)
        noise = torch.randn(z.shape, generator=torch.Generator().manual_seed(ns))
        loss = float(rf_loss(model, z[None], e, m, torch.tensor([t]), noise[None], None))
        losses.append(loss)
        per_bucket.setdefault(round(t, 4), []).append(loss)
    return {"loss": sum(losses) / len(losses), "n": len(losses),
            "per_t": {str(k): sum(v) / len(v) for k, v in sorted(per_bucket.items())}}


def heldout_before_after(base_manifest: dict, base_weights, new_weights, shard_paths: list, seed: int, n: int = 8,
                         buckets: int = 4) -> tuple:
    b = heldout_loss(load_dit(base_manifest, base_weights), shard_paths, seed, n, buckets)["loss"]
    a = heldout_loss(load_dit(base_manifest, new_weights), shard_paths, seed, n, buckets)["loss"]
    return b, a


def hd_eval_job(weights, manifest_path, val_paths: list, out_dir, *, version_id: str, triples_seed: int, n: int,
                timestep_buckets: int, should_stop=None) -> dict:
    from .. import version as ver
    m = ver.read_manifest(manifest_path)
    if m.get("family") != "hd":
        raise ValueError("manifest is not an HD manifest")
    with StReader(weights) as r:
        want = {k: tuple(v) for k, v in m["tensor_shapes"].items()}
        if r.shapes() != want:
            raise ValueError("weights do not match the manifest tensor shapes")
    res = heldout_loss(load_dit(m, weights), val_paths, triples_seed, n, timestep_buckets, should_stop)
    rep = {"schema": EVAL_SCHEMA, "version_id": version_id, "loss": res["loss"], "n": res["n"], "per_t": res["per_t"]}
    p = Path(out_dir) / "eval_report.json"
    p.write_text(json.dumps(rep, indent=1, sort_keys=True))
    return {"loss": res["loss"], "n": res["n"], "report": str(p)}
