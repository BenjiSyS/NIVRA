"""The ``latent_shard`` artifact: encoded clips + precomputed text embeddings, bound to the models that made them.

One safetensors file (deterministic bytes: safetensors orders tensors itself and ``nivra`` is the only metadata key):

    latents    float16 [N, C, T, H, W]   SCALED latents ((z - latent_shift) * latent_scale), posterior mean
    text_emb   float16 [N, L, D]         text-encoder output of ``rows[i]["text"]`` (zero padded)
    text_mask  uint8   [N, L]            1 = real token
    null_emb   float16 [1, Ln, D]        encoder output of the empty prompt (the CFG null; the DiT itself uses its
    null_mask  uint8   [1, Ln]           learned null context for caption dropout, this one is for tooling)

    metadata["nivra"] = JSON {format "nivra-latents/1", parent_shard_sha256, vae_version, text_encoder_id,
        tokenizer_sha256, stage{h,w,frames,fps}, latent_shift, latent_scale, split, dataset_id?, target_dataset_id?,
        shard_idx?, rows:[{id, caption, text, style, lang, split}]}

Workers training the DiT never load the VAE or the text encoder: ``latents`` and ``text_emb`` are all they read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file

FORMAT = "nivra-latents/1"
BIND_KEYS = ("parent_shard_sha256", "vae_version", "text_encoder_id", "stage")


class LatentShardError(Exception):
    pass


def write_latent_shard(path, latents: torch.Tensor, text_emb: torch.Tensor, text_mask: torch.Tensor,
                       null_emb: torch.Tensor, null_mask: torch.Tensor, meta: dict) -> str:
    """Write a latent shard and return the file's sha256. ``meta`` must carry the BIND_KEYS and ``rows``."""
    missing = [k for k in BIND_KEYS + ("rows",) if k not in meta]
    if missing:
        raise LatentShardError(f"latent shard metadata lacks {missing}")
    n = latents.shape[0]
    if len(meta["rows"]) != n or text_emb.shape[0] != n or text_mask.shape[:1] != (n,):
        raise LatentShardError("latents / text / rows disagree on the clip count")
    body = dict(meta, format=FORMAT)
    arrays = {
        "latents": np.ascontiguousarray(latents.detach().cpu().to(torch.float16).numpy()),
        "text_emb": np.ascontiguousarray(text_emb.detach().cpu().to(torch.float16).numpy()),
        "text_mask": np.ascontiguousarray(text_mask.detach().cpu().to(torch.uint8).numpy()),
        "null_emb": np.ascontiguousarray(null_emb.detach().cpu().to(torch.float16).numpy()),
        "null_mask": np.ascontiguousarray(null_mask.detach().cpu().to(torch.uint8).numpy()),
    }
    np_save_file(arrays, str(path), metadata={"nivra": json.dumps(body, sort_keys=True, ensure_ascii=False)})
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class LatentShard:
    """Read access to one latent shard (lazy: tensors are sliced out of the file per clip)."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            with safe_open(str(self.path), framework="pt") as f:
                md = f.metadata() or {}
                keys = set(f.keys())
                self.shape = tuple(f.get_slice("latents").get_shape()) if "latents" in keys else None
                self.text_shape = tuple(f.get_slice("text_emb").get_shape()) if "text_emb" in keys else None
        except Exception as e:  # noqa: BLE001
            raise LatentShardError(f"{self.path.name}: unreadable safetensors: {e}") from e
        for k in ("latents", "text_emb", "text_mask", "null_emb", "null_mask"):
            if k not in keys:
                raise LatentShardError(f"{self.path.name}: not a latent shard (no {k!r})")
        try:
            self.meta = json.loads(md["nivra"])
        except Exception as e:  # noqa: BLE001
            raise LatentShardError(f"{self.path.name}: missing nivra metadata") from e
        if self.meta.get("format") != FORMAT:
            raise LatentShardError(f"{self.path.name}: unsupported latent shard format {self.meta.get('format')!r}")
        if len(self.shape) != 5 or self.shape[0] != len(self.meta.get("rows", ())):
            raise LatentShardError(f"{self.path.name}: latents shape {self.shape} disagrees with rows")

    def __len__(self):
        return self.shape[0]

    @property
    def rows(self) -> list:
        return self.meta["rows"]

    def clip_shape(self) -> tuple:
        return self.shape[1:]

    def get(self, i: int) -> tuple:
        """(latent float32 [C,T,H,W], text_emb float32 [1,L,D], text_mask bool [1,L])."""
        with safe_open(str(self.path), framework="pt") as f:
            z = f.get_slice("latents")[i:i + 1][0].float()
            e = f.get_slice("text_emb")[i:i + 1].float()
            m = f.get_slice("text_mask")[i:i + 1].bool()
        return z, e, m

    def null(self) -> tuple:
        with safe_open(str(self.path), framework="pt") as f:
            return f.get_tensor("null_emb").float(), f.get_tensor("null_mask").bool()

    def load_all(self) -> dict:
        with safe_open(str(self.path), framework="pt") as f:
            return {"latents": f.get_tensor("latents"), "text_emb": f.get_tensor("text_emb"),
                    "text_mask": f.get_tensor("text_mask")}


def check_bound_to(shard: LatentShard, *, stage: Optional[dict] = None, text_encoder_id: Optional[str] = None,
                   vae_version: Optional[str] = None) -> Optional[str]:
    """None if the shard is bound to the expected stage / encoder / VAE, else a message."""
    from .validate import norm_stage
    m = shard.meta
    if stage is not None and norm_stage(m.get("stage")) != norm_stage(stage):
        return f"{shard.path.name}: shard stage {m.get('stage')} != run stage {stage}"
    if text_encoder_id is not None and m.get("text_encoder_id") != text_encoder_id:
        return f"{shard.path.name}: text encoder {m.get('text_encoder_id')} != {text_encoder_id}"
    if vae_version is not None and m.get("vae_version") != vae_version:
        return f"{shard.path.name}: vae {m.get('vae_version')} != {vae_version}"
    return None
