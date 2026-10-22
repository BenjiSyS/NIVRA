"""Config loading for the Nivra HD family (HC-VAE + DiT). Separate from nivra.config, which only accepts
the nivra-small architecture; both live in ai/configs/*.json."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..compression.hc_vae import HCVAE, HCVAE_VERSION, HCVAE_VERSION_2, HCVAEConfig
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
    train_options(cfg)
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


STYLIZED_EDGE_WEIGHT = 0.3     # default when a stylized config does not set vae.edge_weight (thin line art)


def vae_edge_weight(cfg: dict) -> float:
    """Weight of the HC-VAE edge/gradient loss. Absent -> 0.3 for stylized configs (line art is what blur destroys
    first), 0 (the original loss) for general ones; an explicit value always wins."""
    w = cfg["vae"].get("edge_weight", STYLIZED_EDGE_WEIGHT if cfg.get("domain") == "stylized" else 0.0)
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
    # calibration values are not architecture: the hash must not change when the latent statistics are filled in
    # (which is also what promotes a per-channel VAE config from hc-vae/1 to /2)
    if keep["vae"].pop("latent_norm", "scalar") == "channel" and keep["vae"]["version"] == HCVAE_VERSION_2:
        keep["vae"]["version"] = HCVAE_VERSION
    for k in ("latent_shift", "latent_scale"):
        keep["vae"].pop(k, None)
    return hashlib.sha256(json.dumps(keep, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ===================================================================== training options (train section)
def _num(d: dict, key: str, default, lo=None, hi=None, integer: bool = False, where: str = "train"):
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise HDConfigError(f"{where}.{key} must be a finite number, got {v!r}")
    if integer and int(v) != v:
        raise HDConfigError(f"{where}.{key} must be an integer, got {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise HDConfigError(f"{where}.{key} must be in [{lo}, {hi}], got {v!r}")
    return int(v) if integer else float(v)


def _sub(tc: dict, key: str, allowed: set) -> dict:
    d = tc.get(key, {})
    if not isinstance(d, dict):
        raise HDConfigError(f"train.{key} must be an object")
    unknown = set(d) - allowed
    if unknown:
        raise HDConfigError(f"unknown train.{key} keys {sorted(unknown)}")
    return d


def validate_stage(st: dict, where: str) -> None:
    if not isinstance(st, dict):
        raise HDConfigError(f"{where} must be an object")
    for k in ("height", "width", "frames", "steps"):
        if k in st:
            _num(st, k, 1, lo=1, integer=True, where=where)
    if "image_fraction" in st:
        _num(st, "image_fraction", 0.0, 0.0, 1.0, where=where)
    if "buckets" in st:
        bs = st["buckets"]
        if not isinstance(bs, list) or not bs:
            raise HDConfigError(f"{where}.buckets must be a non-empty list")
        for i, b in enumerate(bs):
            for k in ("height", "width", "frames"):
                if k not in b:
                    raise HDConfigError(f"{where}.buckets[{i}] needs {k}")
                _num(b, k, None, lo=1, integer=True, where=f"{where}.buckets[{i}]")
            _num(b, "weight", 1.0, 0.0, None, where=f"{where}.buckets[{i}]")


def train_options(cfg: dict) -> dict:
    """Validated, defaulted view of the new (all optional, all off-by-default-for-old-configs) training options.
    Keys absent from an old config give exactly the old behaviour."""
    tc = cfg["train"]
    depth = int(cfg["model"].get("depth", 24))
    frame_cond = bool(cfg["model"].get("frame_cond", False))
    o: dict = {}
    # ---- VAE
    o["vae_gan_weight"] = _num(tc, "vae_gan_weight", 0.0, 0.0)
    o["vae_gan_start"] = _num(tc, "vae_gan_start", 0, 0, integer=True)
    o["vae_gan_lr"] = _num(tc, "vae_gan_lr", float(tc.get("vae_lr", tc.get("lr", 1e-4))), 0.0)
    o["vae_gan_base"] = _num(tc, "vae_gan_base", 32, 1, integer=True)
    o["vae_gan_layers"] = _num(tc, "vae_gan_layers", 3, 1, 5, integer=True)
    ln = tc.get("vae_latent_norm", "scalar")
    if ln not in ("scalar", "channel"):
        raise HDConfigError(f"train.vae_latent_norm must be 'scalar' or 'channel', got {ln!r}")
    o["vae_latent_norm"] = ln
    vs = tc.get("vae_stages")
    if vs is not None:
        if not isinstance(vs, list) or not vs:
            raise HDConfigError("train.vae_stages must be a non-empty list of stages")
        for i, st in enumerate(vs):
            validate_stage(st, f"train.vae_stages[{i}]")
    o["vae_stages"] = vs
    for i, st in enumerate(tc.get("stages", [])):
        validate_stage(st, f"train.stages[{i}]")
    # ---- EMA
    o["ema_decay"] = _num(tc, "ema_decay", 0.0, 0.0, 0.99999999)
    ew = tc.get("ema_warmup", True)
    if not isinstance(ew, bool):
        raise HDConfigError("train.ema_warmup must be a boolean")
    o["ema_warmup"] = ew
    # ---- SRA / REPA
    sra = _sub(tc, "sra", {"weight", "student_layer", "teacher_layer", "t_ratio", "until_step", "hidden"})
    o["sra"] = {"weight": _num(sra, "weight", 0.0, 0.0, where="train.sra"),
                "student_layer": _num(sra, "student_layer", max(0, depth // 4), 0, integer=True, where="train.sra"),
                "teacher_layer": _num(sra, "teacher_layer", max(1, (2 * depth) // 3), 0, integer=True, where="train.sra"),
                "t_ratio": _num(sra, "t_ratio", 0.5, 0.0, 1.0, where="train.sra"),
                "until_step": _num(sra, "until_step", 0, 0, integer=True, where="train.sra"),
                "hidden": _num(sra, "hidden", 0, 0, integer=True, where="train.sra")}
    if o["sra"]["weight"] > 0:
        if not o["sra"]["student_layer"] < o["sra"]["teacher_layer"] < depth:
            raise HDConfigError(f"train.sra needs student_layer < teacher_layer < depth ({depth}), got "
                                f"{o['sra']['student_layer']} / {o['sra']['teacher_layer']}")
    repa = _sub(tc, "repa", {"enabled", "teacher", "weight", "layer"})
    if repa.get("enabled"):
        raise HDConfigError("train.repa.enabled: REPA needs precomputed features of an external teacher (DINOv2, a "
                            "licensed opt-in add-on) and is not implemented; use train.sra (no external model)")
    o["repa"] = {"enabled": False}
    # ---- diffusion forcing / frame conditioning
    df = _sub(tc, "diffusion_forcing", {"fraction", "cond_fraction"})
    o["diffusion_forcing"] = {"fraction": _num(df, "fraction", 0.0, 0.0, 1.0, where="train.diffusion_forcing"),
                              "cond_fraction": _num(df, "cond_fraction", 0.0, 0.0, 1.0, where="train.diffusion_forcing")}
    if o["diffusion_forcing"]["cond_fraction"] > 0 and not frame_cond:
        raise HDConfigError("train.diffusion_forcing.cond_fraction > 0 needs model.frame_cond = true")
    o["fps_dropout"] = _num(tc, "fps_dropout", 0.1, 0.0, 1.0)
    # ---- fp16 hardening
    for k in ("fp32_attn",):
        v = tc.get(k, False)
        if not isinstance(v, bool):
            raise HDConfigError(f"train.{k} must be a boolean")
        o[k] = v
    ls = _sub(tc, "loss_scale", {"init", "growth_interval", "growth", "backoff"})
    o["loss_scale"] = {"init": _num(ls, "init", 65536.0, 1.0, where="train.loss_scale"),
                       "growth_interval": _num(ls, "growth_interval", 2000, 1, integer=True, where="train.loss_scale"),
                       "growth": _num(ls, "growth", 2.0, 1.0, where="train.loss_scale"),
                       "backoff": _num(ls, "backoff", 0.5, 0.0, 1.0, where="train.loss_scale")}
    return o
