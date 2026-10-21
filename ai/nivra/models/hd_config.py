"""Config loading for the Nivra HD family (HC-VAE + DiT). Separate from nivra.config, which only accepts
the nivra-small architecture; both live in ai/configs/*.json."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..compression.hc_vae import HCVAE, HCVAEConfig
from .dit import DiTConfig, NivraDiT

HD_ARCH_VERSION = "nivra-hd/1"
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
TOP_KEYS = {"name", "arch_version", "vae", "model", "inference", "train"}


class HDConfigError(ValueError):
    pass


def load_hd_config(name_or_obj) -> dict:
    if isinstance(name_or_obj, dict):
        cfg = copy.deepcopy(name_or_obj)
    else:
        p = Path(str(name_or_obj))
        if not p.exists():
            p = CONFIG_DIR / (str(name_or_obj) + ".json")
        if not p.exists():
            raise HDConfigError(f"unknown HD config {name_or_obj!r}")
        cfg = json.loads(p.read_text(encoding="utf-8"))
    unknown = set(cfg) - TOP_KEYS
    if unknown:
        raise HDConfigError(f"unknown HD config keys {sorted(unknown)}")
    if cfg.get("arch_version") != HD_ARCH_VERSION:
        raise HDConfigError(f"not an HD config (arch_version {cfg.get('arch_version')!r})")
    for k in ("vae", "model", "inference", "train"):
        cfg.setdefault(k, {})
    cfg.setdefault("name", "custom")
    vae_config(cfg)
    dit_config(cfg)
    return cfg


def vae_config(cfg: dict) -> HCVAEConfig:
    return HCVAEConfig.from_dict(cfg["vae"])


def dit_config(cfg: dict) -> DiTConfig:
    d = dict(cfg["model"])
    d["latent_channels"] = vae_config(cfg).latent_channels
    return DiTConfig.from_dict(d)


def build_hcvae(cfg: dict) -> HCVAE:
    return HCVAE(vae_config(cfg))


def build_dit(cfg: dict) -> NivraDiT:
    return NivraDiT(dit_config(cfg))


def hd_config_hash(cfg: dict) -> str:
    keep = {"arch_version": cfg["arch_version"], "vae": vae_config(cfg).to_dict(), "model": dit_config(cfg).to_dict()}
    for k in ("latent_shift", "latent_scale"):   # calibration values are not architecture
        keep["vae"].pop(k, None)
    return hashlib.sha256(json.dumps(keep, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
