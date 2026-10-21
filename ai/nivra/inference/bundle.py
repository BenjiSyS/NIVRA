"""Model bundle: generator weights (+version manifest) and the frozen VAE, verified on load."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .. import modelio
from ..config import build_gen, build_vae, config_hash, load_config, synth_config
from ..distributed import version as ver
from ..models.nivra_small import ARCH_VERSION, NivraSmall
from ..compression.vae import VideoVAE
from ..text.tokenizer import default_tokenizer, TOKENIZER_VERSION
from ..training.checkpoint import sha256_file


class BundleError(Exception):
    """Weights/manifest missing, corrupt or incompatible (exec exit code 5)."""


@dataclass
class Bundle:
    model: NivraSmall
    vae: VideoVAE
    manifest: dict
    cfg: dict
    vae_sha256: str

    @property
    def version_id(self) -> str:
        return self.manifest["version_id"]


def load_bundle_files(weights, manifest, vae_weights, device: str = "cpu") -> Bundle:
    try:
        m = ver.read_manifest(manifest)
        w = ver.load_weights(m, weights)
    except (ver.VersionError, modelio.ModelFileError, OSError) as e:
        raise BundleError(str(e)) from e
    tok = default_tokenizer()
    if m["architecture_version"] != ARCH_VERSION:
        raise BundleError(f"incompatible architecture {m['architecture_version']}")
    if m["tokenizer_version"] != TOKENIZER_VERSION or m["vocab_hash"] != tok.vocab_hash:
        raise BundleError("incompatible tokenizer / vocab hash")
    try:
        vsha = sha256_file(vae_weights)
    except OSError as e:
        raise BundleError(f"cannot read VAE: {e}") from e
    if m["vae_sha256"] and vsha != m["vae_sha256"]:
        raise BundleError("VAE weights do not match the VAE this model version was trained against")
    cfg = load_config(m["config"])
    try:
        model = build_gen(cfg)
        model.load_state_dict(w, strict=True)
        vae = modelio.load_vae(vae_weights)
    except (RuntimeError, modelio.ModelFileError) as e:
        raise BundleError(f"weights do not match architecture: {e}") from e
    model.to(device).eval()
    vae.to(device).eval()
    for p in list(model.parameters()) + list(vae.parameters()):
        p.requires_grad_(False)
    return Bundle(model, vae, m, cfg, vsha)


def load_bundle(directory, device: str = "cpu") -> Bundle:
    d = Path(directory)
    return load_bundle_files(d / ver.WEIGHTS_FILE, d / ver.MANIFEST_FILE, d / "vae.safetensors", device)


def init_random_bundle(cfg: dict, vae_path, out_dir, seed: int = 0) -> dict:
    """Randomly initialised (untrained) generator packaged like a trained one (baseline for benchmarks)."""
    from ..training.pipeline import export_weights
    torch.manual_seed(seed)
    model = build_gen(cfg)
    return export_weights(cfg, {k: v.clone() for k, v in model.state_dict().items()}, vae_path, out_dir)
