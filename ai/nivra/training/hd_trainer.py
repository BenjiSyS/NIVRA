"""Progressive-resolution trainer for the Nivra HC-VAE and the Nivra DiT.

* Buckets / stages from the config (e.g. 256p/2 s -> 480p/4 s -> 720p/8 s); the stage is a function of the
  global step, so resuming lands in the right stage.
* Data: datasets written by ``nivra.data.ingest`` (``manifest.jsonl`` rows with ``shard``/``index``/
  ``caption``/``lang``/``split``; shards are safetensors with ``videos`` uint8 [N,T,H,W,3] or
  ``latents`` float16 [N,C,T,H,W]). Clips are read one at a time through safetensors slices.
* fp16 autocast + GradScaler on CUDA (untested on GPU), activation checkpointing, gradient accumulation,
  plain AdamW or a pure-PyTorch AdamW with block-quantised 8-bit moment states.
* Checkpoints: safetensors + JSON through ``nivra.training.checkpoint`` (atomic, sha256, no pickle).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from ..compression.hc_vae import HCVAE, hcvae_loss, load_hcvae, save_hcvae
from ..models.dit import NivraDiT, rf_loss, resolution_shift, sample_t
from ..models.hd_config import HD_ARCH_VERSION, build_dit, build_hcvae, hd_config_hash, load_hd_config
from .checkpoint import (apply_retention, capture_rng, load_checkpoint, module_tensors, newest_valid,
                         optimizer_from_flat, optimizer_to_flat, restore_rng, save_checkpoint)


# ===================================================================== 8-bit AdamW (pure PyTorch)
def _q_signed(x: torch.Tensor, block: int):
    """Blockwise absmax quantisation with square-root companding -> (int8 codes, float32 scales)."""
    n = x.numel()
    pad = (-n) % block
    xb = F.pad(x.reshape(-1).float(), (0, pad)).view(-1, block)
    s = xb.abs().amax(1).clamp_min(1e-30)
    q = torch.round(torch.sign(xb) * (xb.abs() / s[:, None]).sqrt() * 127).to(torch.int8)
    return q.view(-1), s


def _dq_signed(q: torch.Tensor, s: torch.Tensor, n: int, block: int, shape):
    qb = q.view(-1, block).float() / 127
    return (torch.sign(qb) * qb.pow(2) * s[:, None]).view(-1)[:n].view(shape)


def _q_pos(x: torch.Tensor, block: int):
    """Non-negative values (second moment): store v^(1/4) uniformly in uint8 per block."""
    n = x.numel()
    pad = (-n) % block
    xb = F.pad(x.reshape(-1).float(), (0, pad)).view(-1, block).clamp_min(0)
    s = xb.amax(1).clamp_min(1e-30)
    q = torch.round((xb / s[:, None]).pow(0.25) * 255).to(torch.uint8)
    return q.view(-1), s


def _dq_pos(q: torch.Tensor, s: torch.Tensor, n: int, block: int, shape):
    return ((q.view(-1, block).float() / 255).pow(4) * s[:, None]).view(-1)[:n].view(shape)


class AdamW8bit(torch.optim.Optimizer):
    """AdamW with blockwise-quantised moments (int8 m with sqrt companding, uint8 v^(1/4)): ~2 bytes of state
    per parameter instead of 8. Small tensors (< min_8bit_size) keep float32 states. The per-element update
    ratio is bounded to |m/(sqrt(v)+eps)| <= 5 as a guard against quantisation artefacts (exact Adam is bounded
    by ~(1-b1)/sqrt(1-b2) ~ 3.2 anyway)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, block=256,
                 min_8bit_size=4096):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block=block,
                                      min_8bit_size=min_8bit_size))

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for st in self.state.values():           # the base class casts states to the param dtype; undo that
            if "m_q" in st:
                st["m_q"] = st["m_q"].to(torch.int8)
                st["v_q"] = st["v_q"].to(torch.uint8)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g["betas"]
            blk = g["block"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                st = self.state[p]
                small = p.numel() < g["min_8bit_size"]
                if not st:
                    st["step"] = torch.zeros((), dtype=torch.float32)
                    if small:
                        st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    else:
                        z = torch.zeros_like(p, dtype=torch.float32)
                        st["m_q"], st["m_s"] = _q_signed(z, blk)
                        st["v_q"], st["v_s"] = _q_pos(z, blk)
                st["step"] += 1
                t = float(st["step"])
                if small:
                    m, v = st["exp_avg"], st["exp_avg_sq"]
                else:
                    m = _dq_signed(st["m_q"], st["m_s"], p.numel(), blk, p.shape)
                    v = _dq_pos(st["v_q"], st["v_s"], p.numel(), blk, p.shape)
                m.mul_(b1).add_(grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                mh = m / (1 - b1 ** t)
                vh = v / (1 - b2 ** t)
                upd = (mh / (vh.sqrt() + g["eps"])).clamp_(-5.0, 5.0)
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd.to(p.dtype), alpha=-g["lr"])
                if not small:
                    st["m_q"], st["m_s"] = _q_signed(m, blk)
                    st["v_q"], st["v_s"] = _q_pos(v, blk)
        return loss


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    if name == "adamw8":
        return AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    raise ValueError(f"unknown optimizer {name!r}")


# ===================================================================== data
class ClipShards:
    """Reader for ingest datasets (manifest.jsonl + safetensors shards with `videos` or `latents`)."""

    def __init__(self, root, split: Optional[str] = "train"):
        self.root = Path(root)
        mf = self.root / "manifest.jsonl"
        if not mf.is_file():
            raise FileNotFoundError(f"{mf} not found (expected a dataset written by nivra data ingest)")
        rows = [json.loads(l) for l in mf.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.rows = [r for r in rows if split is None or r.get("split", "train") == split]
        if not self.rows:
            raise ValueError(f"no clips for split {split!r} in {self.root}")
        self._kind: dict = {}

    def __len__(self):
        return len(self.rows)

    def kind(self, shard: str) -> str:
        if shard not in self._kind:
            with safe_open(str(self.root / shard), framework="pt") as f:
                keys = set(f.keys())
            self._kind[shard] = "latents" if "latents" in keys else "videos"
            if self._kind[shard] == "videos" and "videos" not in keys:
                raise ValueError(f"shard {shard} has neither 'videos' nor 'latents'")
        return self._kind[shard]

    def get(self, i: int) -> tuple:
        """Returns (kind, tensor, row): videos -> float32 [3,T,H,W] in [-1,1]; latents -> float32 [C,T,H,W]."""
        r = self.rows[i]
        k = self.kind(r["shard"])
        with safe_open(str(self.root / r["shard"]), framework="pt") as f:
            x = f.get_slice(k)[r["index"]: r["index"] + 1][0]
        if k == "videos":
            x = x.permute(3, 0, 1, 2).float().div(127.5).sub(1.0)
        else:
            x = x.float()
        return k, x, r


def fit_video(x: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    """[3,T,H,W] -> [3,frames,height,width]: first `frames` frames (last frame repeated if short), resize to
    cover, centre crop."""
    T = x.shape[1]
    if T < frames:
        x = torch.cat([x, x[:, -1:].expand(-1, frames - T, -1, -1)], 1)
    x = x[:, :frames]
    H, W = x.shape[-2:]
    s = max(height / H, width / W)
    nh, nw = max(height, round(H * s)), max(width, round(W * s))
    if (nh, nw) != (H, W):
        x = F.interpolate(x.transpose(0, 1), size=(nh, nw), mode="bilinear", align_corners=False,
                          antialias=s < 1).transpose(0, 1)
    top, left = (nh - height) // 2, (nw - width) // 2
    return x[:, :, top:top + height, left:left + width].clamp(-1, 1)


def fit_latent(z: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    C, T, H, W = z.shape
    if T < t or H < h or W < w:
        raise ValueError(f"latent clip {tuple(z.shape)} smaller than bucket {(t, h, w)}")
    top, left = (H - h) // 2, (W - w) // 2
    return z[:, :t, top:top + h, left:left + w]


# ===================================================================== trainer
def _stage_at(stages: list, step: int) -> tuple:
    acc = 0
    for i, s in enumerate(stages):
        acc += int(s["steps"])
        if step < acc:
            return i, s
    return len(stages) - 1, stages[-1]


def train_hd(cfg, data_root, run_dir, target: str = "dit", text_encoder=None, vae: Optional[HCVAE] = None,
             vae_dir=None, total_steps: Optional[int] = None, device="cpu", seed: int = 0, resume: bool = True,
             log: Callable = print, stop_file=None, latent_norm: Optional[tuple] = None,
             model: Optional[nn.Module] = None) -> dict:
    """Train the HC-VAE (target='vae') or the DiT (target='dit'). Returns a summary dict."""
    if target not in ("vae", "dit"):
        raise ValueError("target must be 'vae' or 'dit'")
    cfg = load_hd_config(cfg)
    tc = cfg["train"]
    stages = tc["stages"]
    total = int(total_steps if total_steps is not None else sum(int(s["steps"]) for s in stages))
    dev = torch.device(device)
    use_amp = dev.type == "cuda" and bool(tc.get("fp16", True))
    run_dir = Path(run_dir)
    data = ClipShards(data_root, "train")

    torch.manual_seed(seed)
    random.seed(seed)
    if target == "vae":
        net = model if model is not None else build_hcvae(cfg)
        lr = float(tc.get("vae_lr", tc.get("lr", 1e-4)))
    else:
        if text_encoder is None:
            raise ValueError("DiT training needs a text encoder")
        net = model if model is not None else build_dit(cfg)
        if text_encoder.dim != net.cfg.text_dim:
            raise ValueError(f"text encoder dim {text_encoder.dim} != model text_dim {net.cfg.text_dim}")
        net.set_memory(grad_checkpoint=bool(tc.get("grad_checkpoint", True)))
        if vae is None and vae_dir is not None:
            vae = load_hcvae(vae_dir)
        if vae is not None:
            vae = vae.to(dev).eval().requires_grad_(False)
        lr = float(tc.get("lr", 1e-4))
    net.to(dev).train()
    opt = make_optimizer(tc.get("optimizer", "adamw"), net.parameters(), lr, float(tc.get("weight_decay", 0.0)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = max(1, int(tc.get("grad_accum", 1)))
    bs = max(1, int(tc.get("batch_size", 1)))
    warmup = int(tc.get("warmup", 0))
    clip = float(tc.get("grad_clip", 0.0))
    compat = {"arch_version": HD_ARCH_VERSION, "config_hash": hd_config_hash(cfg), "target": target,
              "text_encoder": getattr(text_encoder, "id", None) if target == "dit" else None,
              "text_dim": net.cfg.text_dim if target == "dit" else None}

    step, epoch, pos, history = 0, 0, 0, []
    gen = torch.Generator().manual_seed(seed + 17)       # timesteps / noise / caption dropout
    if resume and (ck := newest_valid(run_dir)) is not None:
        man, groups, st = load_checkpoint(ck, expect_compat=compat)
        net.load_state_dict(groups["model"])
        optimizer_from_flat(opt, groups.get("optimizer", {}), st["opt_groups"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        restore_rng(groups["rng"], st["rng"])
        gen.set_state(groups["rng"]["trainer_gen"].to(torch.uint8))
        step, epoch, pos, history = st["step"], st["epoch"], st["pos"], st.get("history", [])
        log(f"resumed {target} from {ck.name} at step {step}")

    def order(ep):
        o = list(range(len(data)))
        random.Random(seed * 1000003 + ep).shuffle(o)
        return o

    def next_batch(stage):
        nonlocal epoch, pos
        xs, caps = [], []
        o = order(epoch)
        while len(xs) < bs:
            if pos >= len(o):
                epoch, pos = epoch + 1, 0
                o = order(epoch)
            kind, x, row = data.get(o[pos])
            pos += 1
            if kind == "videos":
                x = fit_video(x, int(stage["frames"]), int(stage["height"]), int(stage["width"]))
            elif target == "vae":
                raise ValueError("VAE training needs `videos` shards, got latents")
            else:
                vc = (vae.cfg if vae is not None else None)
                sf = vc.spatial_factor if vc else 32
                tf = vc.temporal_factor if vc else 8
                x = fit_latent(x, 1 + (int(stage["frames"]) - 1) // tf, int(stage["height"]) // sf,
                               int(stage["width"]) // sf)
            xs.append(x)
            caps.append(row.get("caption", ""))
        return kind, torch.stack(xs).to(dev), caps

    def save(protected=False):
        ot, og = optimizer_to_flat(opt)
        rt, rj = capture_rng()
        rt["trainer_gen"] = gen.get_state().clone()
        p = save_checkpoint(run_dir, step, {"model": module_tensors(net), "optimizer": ot, "rng": rt},
                            {"step": step, "epoch": epoch, "pos": pos, "opt_groups": og, "rng": rj,
                             "scaler": scaler.state_dict() if use_amp else {}, "history": history[-1000:]},
                            compat, {"name": cfg["name"], "train": tc, "target": target}, epoch=epoch,
                            protected=protected)
        apply_retention(run_dir, int(tc.get("keep_last", 3)))
        return p

    save_every = int(tc.get("save_every", 1000))
    status = "done"
    last_ckpt = None
    while step < total:
        if stop_file is not None and Path(stop_file).exists():
            status = "stopped"
            break
        si, stage = _stage_at(stages, step)
        cur_lr = lr * min(1.0, (step + 1) / max(1, warmup)) if warmup else lr
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        logs = {}
        for _ in range(accum):
            kind, x, caps = next_batch(stage)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                if target == "vae":
                    recon, m, lv = net(x, generator=None)
                    out = hcvae_loss(x, recon, m, lv, float(tc.get("vae_kl_weight", 1e-6)),
                                     float(tc.get("vae_perc_weight", 0.5)), float(tc.get("vae_tdiff_weight", 0.5)))
                    loss = out["loss"]
                    logs = {k: float(v.detach()) for k, v in out.items()}
                else:
                    if kind == "videos":
                        if vae is None:
                            raise ValueError("DiT training on `videos` shards needs an HC-VAE (vae or vae_dir)")
                        with torch.no_grad():
                            z0 = vae.to_scaled(vae.encode(x).float())
                    else:
                        sh, sc = latent_norm if latent_norm is not None else (
                            (vae.cfg.latent_shift, vae.cfg.latent_scale) if vae is not None else (0.0, 1.0))
                        z0 = (x - sh) * sc
                    emb, mask = text_encoder.encode(caps)
                    B = z0.shape[0]
                    n_tok = z0.shape[2] * z0.shape[3] * z0.shape[4]
                    t = sample_t(B, resolution_shift(n_tok, int(tc.get("shift_base_tokens", 4096))), generator=gen,
                                 device=dev)
                    noise = torch.randn(z0.shape, generator=gen).to(dev)
                    drop = torch.rand(B, generator=gen) < float(tc.get("caption_dropout", 0.1))
                    loss = rf_loss(net, z0, emb.to(dev), mask.to(dev), t, noise, drop.to(dev))
                    logs = {"loss": float(loss.detach())}
            scaler.scale(loss / accum).backward()
        if clip > 0:
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), clip))
            logs["grad_norm"] = gn
        if not all(math.isfinite(v) for v in logs.values()) and not use_amp:
            raise FloatingPointError(f"non-finite loss/grad at step {step}: {logs}")
        scaler.step(opt)
        scaler.update()
        step += 1
        history.append({"step": step, "stage": stage["name"], "lr": cur_lr, **logs})
        if step % max(1, save_every) == 0 or step == total:
            last_ckpt = save()
        log(f"[{target}] step {step}/{total} stage {stage['name']} " +
            " ".join(f"{k}={v:.4f}" for k, v in logs.items()))
    if status == "stopped":
        last_ckpt = save()
    result = {"status": status, "step": step, "history": history, "ckpt": str(last_ckpt) if last_ckpt else None,
              "params": sum(p.numel() for p in net.parameters())}
    if target == "vae" and status == "done":
        # calibrate the latent scale on a few training clips and export the VAE (safetensors + JSON)
        net.eval()
        with torch.no_grad():
            _, stage = _stage_at(stages, max(0, step - 1))
            zs = [net.encode(next_batch(stage)[1]).float().cpu() for _ in range(2)]
        result["latent_norm"] = net.calibrate(torch.cat(zs))
        result["vae_dir"] = str(run_dir / "hc_vae")
        save_hcvae(net, run_dir / "hc_vae")
    return result
