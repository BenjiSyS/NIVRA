"""Reading / writing model weights (safetensors + JSON, never pickle)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

from .compression.vae import VideoVAE, VAEConfig
from .training.checkpoint import sha256_file


class ModelFileError(Exception):
    pass


def save_vae(path, vae: VideoVAE, extra: dict | None = None) -> str:
    meta = {"kind": "nivra-vae", "config": json.dumps(vae.cfg.to_dict(), sort_keys=True),
            "extra": json.dumps(extra or {}, sort_keys=True)}
    tensors = {k: v.detach().cpu().contiguous() for k, v in vae.state_dict().items()}
    save_file(tensors, str(path), metadata=meta)
    return sha256_file(path)


def load_vae(path) -> VideoVAE:
    path = Path(path)
    try:
        if path.stat().st_size > (1 << 30):
            raise ModelFileError("VAE file larger than 1 GiB")
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        cfg = VAEConfig.from_dict(json.loads(meta["config"]))
        vae = VideoVAE(cfg)
        vae.load_state_dict(load_file(str(path)))
    except Exception as e:  # noqa: BLE001
        raise ModelFileError(f"cannot load VAE {path}: {e}") from e
    vae.eval()
    return vae


def vae_metadata(path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    return {"config": json.loads(meta.get("config", "{}")), "extra": json.loads(meta.get("extra", "{}"))}


def save_tensors(path, tensors: dict) -> str:
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(path))
    return sha256_file(path)


def load_tensors(path) -> dict:
    try:
        return load_file(str(path))
    except Exception as e:  # noqa: BLE001
        raise ModelFileError(f"cannot read tensors {path}: {e}") from e


def load_gen_weights(model: torch.nn.Module, path) -> None:
    sd = load_tensors(path)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise ModelFileError(f"weights incompatible with architecture: {e}") from e
