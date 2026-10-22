"""``ai.hd_encode``: raw clip shard -> latent shard (scaled VAE posterior means + precomputed text embeddings).

Deterministic for a given (shard, VAE, text encoder): the posterior mean is used, never a sample, so an audit
re-encode on another device agrees to float tolerance. The raw shard carries its own per-clip rows (id, caption,
style, lang) in its safetensors metadata (written by ``nivra data-ingest``); the clips' text is the caption with the
style tag, with a deterministic ``style_dropout`` fraction left bare (the DiT then also works without a tag).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import fields
from pathlib import Path
from typing import Callable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ...compression.hc_vae import HCVAE, HCVAEConfig, HCVAE_VERSIONS
from ...text.bpe import BPETokenizer
from ...text.nivra_text import NivraTextConfig, NivraTextEncoder, NivraTextModel
from ...training.hd_trainer import fit_video, training_caption
from .latent_shard import write_latent_shard
from .stream_io import sha256_file


class EncodeInputError(Exception):
    pass


# ---------------------------------------------------------------------------- models from loose files
def load_vae_files(weights, manifest, device="cpu") -> HCVAE:
    """VAE from vae.safetensors + its manifest: a bundle manifest (model_version.json) or hc_vae.json."""
    try:
        m = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if m.get("family") == "hd":
            from ...models.hd_config import load_hd_config, vae_config
            vc = vae_config(load_hd_config(m["config"]))
        elif m.get("version") in HCVAE_VERSIONS:
            vc = HCVAEConfig.from_dict(m["config"])
        else:
            with safe_open(str(weights), framework="pt") as f:
                vc = HCVAEConfig.from_dict(json.loads((f.metadata() or {})["config"]))
        vae = HCVAE(vc)
        vae.load_state_dict(load_file(str(weights)), strict=True)
    except (KeyError, ValueError, OSError, RuntimeError) as e:
        raise EncodeInputError(f"cannot load the VAE: {e}") from e
    return vae.to(device).eval().requires_grad_(False)


_HEADS = {64: 4, 256: 4, 768: 12}
_MAXLEN = {64: 64, 256: 128, 768: 256}


def load_text_encoder_files(weights, tokenizer, device="cpu") -> NivraTextEncoder:
    """NivraText encoder from model.safetensors + tokenizer.json. The config comes from the weights' metadata
    (checkpoints written by this version); older ones are inferred from the tensor shapes."""
    try:
        tok = BPETokenizer.load(tokenizer)
        sd = load_file(str(weights))
        with safe_open(str(weights), framework="pt") as f:
            md = (f.metadata() or {}).get("nivra_text_config")
        if md:
            cfg = NivraTextConfig(**{k: v for k, v in json.loads(md).items() if k in {x.name for x in fields(NivraTextConfig)}})
        else:
            v, d = sd["tok.weight"].shape
            nl = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
            cfg = NivraTextConfig("custom", v, d, nl, _HEADS.get(d, max(1, d // 64)), sd["blocks.0.f1.weight"].shape[0],
                                  _MAXLEN.get(d, 64))
        if cfg.vocab_size != tok.vocab_size:
            raise EncodeInputError("tokenizer vocab size differs from the text encoder")
        model = NivraTextModel(cfg)
        model.load_state_dict(sd)
    except EncodeInputError:
        raise
    except (KeyError, ValueError, OSError, RuntimeError) as e:
        raise EncodeInputError(f"cannot load the text encoder: {e}") from e
    return NivraTextEncoder(model, tok, device)


# ---------------------------------------------------------------------------- encoding
@torch.no_grad()
def encode_videos(vae: HCVAE, videos: torch.Tensor, stage: dict, device="cpu", batch: int = 2,
                  on_progress: Optional[Callable] = None, t_chunk: int = 2) -> torch.Tensor:
    """uint8 [N,T,H,W,3] -> scaled float32 latents [N,C,T',H',W'] at the stage geometry."""
    h, w, frames = int(stage["h"]), int(stage["w"]), int(stage["frames"])
    vae.cfg.latent_shape(frames, h, w)                       # raises on a stage the VAE cannot take
    out = []
    for i in range(0, videos.shape[0], batch):
        xs = []
        for v in videos[i:i + batch]:
            x = v.permute(3, 0, 1, 2).float().div(127.5).sub(1.0)
            xs.append(fit_video(x, frames, h, w))
        z = vae.encode_tiled(torch.stack(xs).to(device), t_chunk=t_chunk)
        out.append(vae.to_scaled(z.float()).cpu())
        if on_progress:
            on_progress(min(videos.shape[0], i + batch), videos.shape[0])
    return torch.cat(out)


@torch.no_grad()
def encode_texts(enc, texts: list, batch: int = 32, cache=None) -> tuple:
    """(emb float32 [N,L,D], mask bool [N,L]) zero padded to the longest text. ``cache``: an
    ``nivra.text.cache.EmbeddingCache`` (offline embedding cache, mainly for the big licensed encoders)."""
    if cache is not None:
        from ...text.cache import encode_cached
        return encode_cached(enc, texts, cache, batch)
    parts = [enc.encode(texts[i:i + batch]) for i in range(0, len(texts), batch)]
    L = max(e.shape[1] for e, _ in parts)
    D = parts[0][0].shape[2]
    emb = torch.zeros(len(texts), L, D)
    mask = torch.zeros(len(texts), L, dtype=torch.bool)
    k = 0
    for e, m in parts:
        emb[k:k + e.shape[0], :e.shape[1]] = e.cpu()
        mask[k:k + e.shape[0], :m.shape[1]] = m.cpu()
        k += e.shape[0]
    return emb, mask


def read_raw_shard(path) -> tuple:
    """(videos uint8 [N,T,H,W,3], rows, split) from a raw clip shard written by data-ingest."""
    try:
        with safe_open(str(path), framework="pt") as f:
            md = json.loads((f.metadata() or {}).get("nivra", "{}"))
            videos = f.get_tensor("videos")
    except Exception as e:  # noqa: BLE001
        raise EncodeInputError(f"cannot read raw shard: {e}") from e
    rows = md.get("rows")
    if not isinstance(rows, list) or len(rows) != videos.shape[0]:
        raise EncodeInputError("raw shard has no embedded per-clip rows (captions); re-run data-ingest with this version")
    return videos, rows, md.get("split", "train")


def encode_shard(raw_path, out_path, *, vae: HCVAE, enc, stage: dict, parent_sha: str, vae_version: str,
                 text_encoder_id: str, style_dropout: float = 0.1, seed: int = 0, device="cpu", extra_meta=None,
                 on_progress=None, rows_override: Optional[list] = None, videos_override=None, split_override=None,
                 text_cache=None) -> dict:
    """Encode one raw shard to a latent shard file; returns the summary."""
    if videos_override is not None:
        videos, rows, split = videos_override, rows_override, split_override or "train"
    else:
        videos, rows, split = read_raw_shard(raw_path)
    texts = [training_caption(r, style_dropout, seed, 0) for r in rows]
    lat = encode_videos(vae, videos, stage, device, on_progress=on_progress)
    emb, mask = encode_texts(enc, texts, cache=text_cache)
    nemb, nmask = encode_texts(enc, [""], cache=text_cache)
    if not (bool(torch.isfinite(emb.half()).all()) and bool(torch.isfinite(nemb.half()).all())):
        raise EncodeInputError("text embeddings are not finite in fp16 (the text encoder's activations are too large)")
    lat16 = lat.to(torch.float16)
    if not bool(torch.isfinite(lat16).all()):
        raise EncodeInputError("encoded latents are not finite in fp16 (VAE latent scale looks wrong)")
    meta = {"parent_shard_sha256": parent_sha, "vae_version": vae_version, "text_encoder_id": text_encoder_id,
            "text_encoder_name": getattr(enc, "id", None), "tokenizer_sha256": getattr(getattr(enc, "tokenizer", None), "sha256", None),
            "stage": {"h": int(stage["h"]), "w": int(stage["w"]), "frames": int(stage["frames"]), "fps": int(stage["fps"])},
            "latent_shift": vae.cfg.latent_shift, "latent_scale": vae.cfg.latent_scale, "split": split,
            "rows": [{"id": r.get("id"), "caption": r.get("caption", ""), "text": t, "style": r.get("style"),
                      "lang": r.get("lang"), "split": split,
                      **({"effective_fps": r["effective_fps"]} if r.get("effective_fps") else {})} for r, t in zip(rows, texts)], **(extra_meta or {})}
    sha = write_latent_shard(out_path, lat16, emb, mask, nemb, nmask, meta)
    std = float(lat.std()) if lat.numel() > 1 else 0.0
    return {"clips": int(lat.shape[0]), "sha256": sha, "latent_mean": float(lat.mean()),
            "latent_std": std if math.isfinite(std) else 0.0}


def load_external_encoder(spec: str, device="cpu", backend_factory=None):
    """(encoder, id) for ``spec`` = "external:<name>" (looked up in the local text-encoder directory, see
    nivra.text.fetch.encoder_dir) or "external:<path>". Never downloads."""
    from ...text.external import ExternalEncoderError, load_external
    from ...text.fetch import encoder_dir
    kind, _, target = str(spec).partition(":")
    if kind != "external" or not target:
        raise EncodeInputError(f"bad text_encoder_spec {spec!r}; use external:<name>")
    d = Path(target) if (os.sep in target or "/" in target or Path(target).is_dir()) else encoder_dir(target)
    try:
        enc = load_external(d, device=device, backend_factory=backend_factory)
    except ExternalEncoderError as e:
        raise EncodeInputError(f"cannot load the external text encoder: {e}") from e
    return enc, enc.id


def hd_encode_job(inputs: dict, params: dict, out_dir, *, device="cpu", on_progress=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = params
    raw_sha = sha256_file(inputs["raw_shard"])
    if raw_sha != str(p["shard_sha256"]).lower():
        raise EncodeInputError("raw shard sha256 differs from params.shard_sha256")
    vae = load_vae_files(inputs["vae_weights"], inputs["vae_manifest"], device)
    spec = p.get("text_encoder_spec")
    if spec:                                      # a licensed encoder installed on THIS machine (text-encoder-fetch)
        enc, te_id = load_external_encoder(spec, device)
    else:
        if not inputs.get("text_encoder_weights") or not inputs.get("tokenizer"):
            raise EncodeInputError("text_encoder_weights and tokenizer inputs are required (or params.text_encoder_spec)")
        enc = load_text_encoder_files(inputs["text_encoder_weights"], inputs["tokenizer"], device)
        te_id = sha256_file(inputs["text_encoder_weights"])
    cache = None
    cdir = os.environ.get("NIVRA_TEXT_CACHE")
    if cdir and spec:
        from ...text.cache import EmbeddingCache
        cache = EmbeddingCache(cdir, enc.id, enc.max_len)
    stage = p["stage"]
    extra = {"dataset_id": p.get("dataset_id"), "target_dataset_id": p.get("target_dataset_id"), "shard_idx": p.get("shard_idx")}
    s = encode_shard(inputs["raw_shard"], out / "latent_shard.safetensors", vae=vae, enc=enc, stage=stage,
                     parent_sha=raw_sha, vae_version=sha256_file(inputs["vae_weights"]),
                     text_encoder_id=te_id, style_dropout=p.get("style_dropout", 0.1),
                     seed=p.get("seed", 0), device=device, extra_meta=extra, on_progress=on_progress, text_cache=cache)
    s["path"] = str(out / "latent_shard.safetensors")
    return s
