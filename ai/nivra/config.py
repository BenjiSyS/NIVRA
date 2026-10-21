"""Config loading (JSON only), device/thread helpers, hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import torch

from .compression.vae import VAEConfig, VideoVAE
from .data.synthetic import SynthConfig
from .models.nivra_small import GenConfig, NivraSmall, ARCH_VERSION
from .text.tokenizer import default_tokenizer

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
TOP_KEYS = {"name", "arch_version", "data", "vae", "model", "train_vae", "train_gen"}


class ConfigError(ValueError):
    pass


def load_config(name_or_obj) -> dict:
    if isinstance(name_or_obj, dict):
        cfg = copy.deepcopy(name_or_obj)
    else:
        p = Path(str(name_or_obj))
        if not p.exists():
            p = CONFIG_DIR / (str(name_or_obj) + ".json")
        if not p.exists():
            raise ConfigError(f"unknown config {name_or_obj!r}")
        cfg = json.loads(p.read_text(encoding="utf-8"))
    unknown = set(cfg) - TOP_KEYS
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    if cfg.get("arch_version", ARCH_VERSION) != ARCH_VERSION:
        raise ConfigError(f"unsupported arch_version {cfg.get('arch_version')}")
    cfg.setdefault("name", "custom")
    cfg.setdefault("arch_version", ARCH_VERSION)
    cfg.setdefault("data", {})
    cfg.setdefault("vae", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train_vae", {})
    cfg.setdefault("train_gen", {})
    return cfg


def synth_config(cfg: dict) -> SynthConfig:
    d = cfg["data"]
    return SynthConfig(frames=int(d.get("frames", 16)), size=int(d.get("size", 64)))


def vae_config(cfg: dict) -> VAEConfig:
    return VAEConfig.from_dict(cfg["vae"])


def gen_config(cfg: dict) -> GenConfig:
    sc, vc = synth_config(cfg), vae_config(cfg)
    c, t, h, w = vc.latent_shape(sc.frames, sc.size)
    d = dict(cfg["model"])
    d.update(latent_channels=c, latent_t=t, latent_h=h, latent_w=w)
    d.setdefault("max_text_len", default_tokenizer().max_len)
    return GenConfig.from_dict(d)


def build_vae(cfg: dict) -> VideoVAE:
    return VideoVAE(vae_config(cfg))


def build_gen(cfg: dict) -> NivraSmall:
    tok = default_tokenizer()
    return NivraSmall(gen_config(cfg), len(tok), tok.pad_id)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: dict) -> str:
    keep = {k: cfg[k] for k in ("arch_version", "data", "vae", "model")}
    return hashlib.sha256(canonical_json(keep).encode()).hexdigest()


def get_threads() -> int:
    try:
        return max(1, int(os.environ.get("NIVRA_THREADS", "4")))
    except ValueError:
        return 4


def setup_runtime() -> str:
    torch.set_num_threads(get_threads())
    return get_device()


def get_device() -> str:
    dev = os.environ.get("NIVRA_DEVICE", "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0],
            "torch": torch.__version__, "threads": torch.get_num_threads(), "device": get_device(),
            "cpu_count": os.cpu_count()}
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (Windows: PeakWorkingSetSize)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / (1024 * 1024)
            return -1.0
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if sys.platform != "darwin" else r / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return -1.0
