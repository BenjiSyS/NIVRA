"""High-level training drivers: VAE stage, generator stage, export of a model version."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .. import modelio
from ..config import (build_gen, build_vae, config_hash, synth_config, gen_config, setup_runtime, peak_rss_mb,
                      hardware_info)
from ..compression.vae import reconstruction_metrics
from ..data.synthetic import make_sample
from ..distributed import version as ver
from ..models.nivra_small import count_params, ARCH_VERSION
from ..text.tokenizer import default_tokenizer, TOKENIZER_VERSION
from . import checkpoint as ck
from .tasks import (VAETask, GenTask, DATA_SEED, HELDOUT_START, compute_scaling_factor, get_latents,
                    heldout_flow_loss, videos_to_tensor)
from .trainer import Trainer, TrainerConfig


def _tcfg(section: dict, steps: int, device: str, seed: int, **over) -> TrainerConfig:
    d = dict(total_steps=steps, batch_size=section.get("batch_size", 8), lr=section.get("lr", 1e-3),
             warmup=section.get("warmup", 50), weight_decay=section.get("weight_decay", 0.0),
             grad_clip=section.get("grad_clip", 1.0), ema_decay=section.get("ema_decay", 0.0), seed=seed, device=device)
    d.update(over)
    return TrainerConfig(**d)


def vae_compat(cfg: dict) -> dict:
    return {"kind": "vae", "arch_version": cfg["vae"].get("arch_version", "nivra-vae/1"), "config_hash": config_hash(cfg)}


def gen_compat(cfg: dict, vae_sha: str) -> dict:
    tok = default_tokenizer()
    return {"kind": "gen", "arch_version": ARCH_VERSION, "config_hash": config_hash(cfg),
            "tokenizer_version": TOKENIZER_VERSION, "vocab_hash": tok.vocab_hash, "vae_sha256": vae_sha}


def train_vae(cfg: dict, run_dir, steps: int, seed: int = 0, resume: bool = True, log=print, **over) -> dict:
    device = setup_runtime()
    torch.manual_seed(seed)
    run_dir = Path(run_dir)
    scfg = synth_config(cfg)
    sec = cfg["train_vae"]
    vae = build_vae(cfg)
    task = VAETask(vae, scfg, int(sec.get("train_clips", 4000)))
    tr = Trainer(task, _tcfg(sec, steps, device, seed, **over), run_dir, vae_compat(cfg), cfg,
                 on_log=lambda r: log(f"[vae] step {r['step']} loss {r['loss']:.4f} l1 {r['l1']:.4f} motion {r['motion_l1']:.4f} "
                                      f"kl {r['kl']:.3f} lr {r['lr']:.2e} {r['sec_per_step']:.2f}s/step"))
    if resume and tr.resume():
        log(f"[vae] resumed at step {tr.step}")
    res = tr.run()
    if res["status"] != "done":
        return res
    res.update(finalize_vae(cfg, vae, run_dir, log))
    res["params"] = count_params(vae)
    res["peak_rss_mb"] = peak_rss_mb()
    return res


def finalize_vae(cfg: dict, vae, run_dir, log=print) -> dict:
    run_dir = Path(run_dir)
    scfg = synth_config(cfg)
    vae.eval()
    sf = compute_scaling_factor(vae, scfg, 128)
    vae.scaling_factor.fill_(sf)
    exp = run_dir / "export"
    exp.mkdir(exist_ok=True)
    held = np.stack([make_sample(DATA_SEED, HELDOUT_START + i, scfg).video for i in range(64)])
    m = reconstruction_metrics(vae, torch.from_numpy(held))
    t0 = time.time()
    tm = reconstruction_metrics(vae, torch.from_numpy(held[:8]), tile=4)
    m["tiled_decode_psnr_db"] = tm["psnr_db"]
    m["scaling_factor"] = sf
    m["latent_shape"] = list(cfg_latent_shape(cfg))
    m["params"] = count_params(vae)
    sha = modelio.save_vae(exp / "vae.safetensors", vae, {"metrics": m, "config_hash": config_hash(cfg)})
    m["vae_sha256"] = sha
    (run_dir / "vae_metrics.json").write_text(json.dumps(m, indent=1))
    log(f"[vae] heldout PSNR {m['psnr_db']:.2f} dB motionL1(255) {m['motion_l1_255']:.2f} scale {sf:.4f}")
    return {"vae_metrics": m, "vae_path": str(exp / "vae.safetensors")}


def cfg_latent_shape(cfg: dict):
    from ..config import vae_config
    sc = synth_config(cfg)
    return vae_config(cfg).latent_shape(sc.frames, sc.size)


def train_generator(cfg: dict, vae_path, run_dir, steps: int, seed: int = 0, resume: bool = True, log=print,
                    heldout_clips: int = 128, **over) -> dict:
    device = setup_runtime()
    torch.manual_seed(seed)
    run_dir = Path(run_dir)
    scfg = synth_config(cfg)
    sec = cfg["train_gen"]
    vae = modelio.load_vae(vae_path)
    vae_sha = ck.sha256_file(vae_path)
    n = int(sec.get("train_clips", 8000))
    t0 = time.time()
    cache = run_dir.parent / "latent_cache"
    lat, ids = get_latents(vae, vae_sha, scfg, DATA_SEED, 0, n, cache,
                           progress=lambda i, N: (log(f"[gen] encoding latents {i}/{N}") if i % 1024 == 0 else None))
    hlat, hids = get_latents(vae, vae_sha, scfg, DATA_SEED, HELDOUT_START, heldout_clips, cache)
    log(f"[gen] latents ready {tuple(lat.shape)} in {time.time() - t0:.0f}s; scaled std {lat.std():.3f}")
    model = build_gen(cfg)
    log(f"[gen] params {count_params(model):,}")
    task = GenTask(model, lat, ids, scfg, vae_sha, caption_dropout=float(sec.get("caption_dropout", 0.1)))

    def on_log(r):
        log(f"[gen] step {r['step']} loss {r['loss']:.4f} gn {r['grad_norm']:.2f} lr {r['lr']:.2e} {r['sec_per_step']:.2f}s/step")

    tr = Trainer(task, _tcfg(sec, steps, device, seed, **over), run_dir, gen_compat(cfg, vae_sha), cfg, on_log=on_log)
    if resume and tr.resume():
        log(f"[gen] resumed at step {tr.step}")
    res = tr.run()
    res["params"] = count_params(model)
    res["peak_rss_mb"] = peak_rss_mb()
    if res["status"] != "done":
        return res
    # evaluate raw and EMA on held-out latents (English shape captions, fixed noise)
    hl = heldout_flow_loss(model, hlat, hids[:, 0])
    out = {"heldout_loss_raw": hl}
    if tr.ema is not None:
        import copy
        em = copy.deepcopy(model)
        em.load_state_dict(tr.ema.shadow)
        out["heldout_loss_ema"] = heldout_flow_loss(em, hlat, hids[:, 0])
    res.update(out)
    log(f"[gen] heldout loss {out}")
    res["export"] = export_version(cfg, tr, vae_path, run_dir / "export", use_ema=tr.ema is not None)
    (run_dir / "final_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def export_version(cfg: dict, tr: Trainer, vae_path, out_dir, use_ema: bool = True) -> dict:
    weights = {k: v.clone() for k, v in (tr.ema.shadow if (use_ema and tr.ema) else tr.model.state_dict()).items()}
    return export_weights(cfg, weights, vae_path, out_dir)


def export_weights(cfg: dict, weights: dict, vae_path, out_dir) -> dict:
    tok = default_tokenizer()
    vae_sha = ck.sha256_file(vae_path)
    out_dir = Path(out_dir)
    m = ver.write_version(out_dir, weights, config=cfg, config_name=cfg["name"], arch_version=ARCH_VERSION,
                          tokenizer_version=TOKENIZER_VERSION, vocab_hash=tok.vocab_hash, vae_sha256=vae_sha, rnd=0)
    import shutil
    shutil.copyfile(vae_path, out_dir / "vae.safetensors")
    tok.save(out_dir / "tokenizer.json")
    return m
