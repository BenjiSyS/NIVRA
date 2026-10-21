"""Config loading for the Nivra HD family (HC-VAE + DiT). Separate from nivra.config, which only accepts
the nivra-small architecture; both live in ai/configs/*.json."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..compression.hc_vae import HCVAE, HCVAEConfig
from ..data.styles import normalize_style, validate_style_list
from .dit import DiTConfig, NivraDiT

HD_ARCH_VERSION = "nivra-hd/1"
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
TOP_KEYS = {"name", "arch_version", "vae", "model", "inference", "train", "domain", "styles", "default_style"}
DOMAINS = ("general", "stylized")
VAE_TRAIN_KEYS = ("edge_weight",)      # loss options that live in the "vae" section but are not architecture


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
    cfg.setdefault("domain", "general")
    cfg.setdefault("styles", [])
    cfg.setdefault("default_style", None)
    _validate_domain(cfg)
    vae_config(cfg)
    vae_edge_weight(cfg)
    dit_config(cfg)
    sd = cfg["train"].get("style_dropout", 0.0)
    if isinstance(sd, bool) or not isinstance(sd, (int, float)) or not 0.0 <= sd <= 1.0:
        raise HDConfigError(f"train.style_dropout must be a number in [0, 1], got {sd!r}")
    return cfg


def _validate_domain(cfg: dict) -> None:
    if cfg["domain"] not in DOMAINS:
        raise HDConfigError(f"domain must be one of {list(DOMAINS)}, got {cfg['domain']!r}")
    try:
        cfg["styles"] = validate_style_list(cfg["styles"]) if cfg["styles"] else []
    except ValueError as e:
        raise HDConfigError(str(e)) from e
    if cfg["domain"] == "stylized" and not cfg["styles"]:
        raise HDConfigError("a stylized config needs a non-empty 'styles' list")
    ds = cfg["default_style"]
    if ds is not None:
        canon = normalize_style(ds) if isinstance(ds, str) else None
        if canon is None:
            raise HDConfigError(f"default_style {ds!r} is not a known style")
        if cfg["styles"] and canon not in cfg["styles"]:
            raise HDConfigError(f"default_style {canon!r} is not in styles {cfg['styles']}")
        cfg["default_style"] = canon


def vae_edge_weight(cfg: dict) -> float:
    """Weight of the HC-VAE edge/gradient loss (0 = off, the original loss)."""
    w = cfg["vae"].get("edge_weight", 0.0)
    if isinstance(w, bool) or not isinstance(w, (int, float)) or not w >= 0.0 or w != w or w == float("inf"):
        raise HDConfigError(f"vae.edge_weight must be a finite number >= 0, got {w!r}")
    return float(w)

def vae_config(cfg: dict) -> HCVAEConfig:
    return HCVAEConfig.from_dict({k: v for k, v in cfg["vae"].items() if k not in VAE_TRAIN_KEYS})


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
