"""Progressive-resolution trainer for the Nivra HC-VAE and the Nivra DiT.

* Buckets / stages from the config (e.g. 256p/2 s -> 480p/4 s -> 720p/8 s); the stage is a function of the
  global step, so resuming lands in the right stage.
* Data: datasets written by ``nivra.data.ingest`` (``manifest.jsonl`` rows with ``shard``/``index``/
  ``caption``/``lang``/``split``; shards are safetensors with ``videos`` uint8 [N,T,H,W,3] or
  ``latents`` float16 [N,C,T,H,W]). Clips are read one at a time through safetensors slices.
* fp16 autocast + dynamic GradScaler on CUDA (untested on GPU; overflow steps are skipped and logged),
  activation checkpointing, gradient accumulation, plain AdamW or a pure-PyTorch AdamW with block-quantised
  8-bit moment states.
* 2026 additions, all config-gated (see ``hd_config.train_options``; absent keys = the old behaviour exactly):
  staged VAE schedule (image AE -> low-res clips -> target) with images mixed into video batches, hinge patch-GAN
  with an adaptive weight, per-channel latent calibration; DiT image-first / joint image+video / bucket stages,
  diffusion forcing and first/last latent-frame conditioning, SRA, EMA (the exported weights when on), fps
  conditioning from the manifest's ``effective_fps``.
* Checkpoints: safetensors + JSON through ``nivra.training.checkpoint`` (atomic, sha256, no pickle).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from ..compression.hc_vae import (HCVAE, PatchDiscriminator, adaptive_gan_weight, hcvae_loss, hinge_d_loss, hinge_g_loss,
                                  load_hcvae, save_hcvae)
from ..data.shards import style_counts
from ..data.styles import UNSPECIFIED, apply_style
from ..models.dit import NivraDiT, make_xt, resolution_shift, rf_loss_ex, sample_t, sample_t_frames
from ..models.hd_config import (HD_ARCH_VERSION, build_dit, build_hcvae, hd_config_hash, load_hd_config,
                                train_options, vae_edge_weight)
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


# ===================================================================== style captions
def style_dropped(seed: int, epoch: int, clip_id: str, p: float) -> bool:
    """Deterministic style-tag dropout for one sample: a pure function of (seed, epoch, clip id), so it needs no
    RNG state in checkpoints (exact resume) and does not disturb the trainer's timestep / noise stream."""
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    h = hashlib.sha256(f"style-dropout/{seed}/{epoch}/{clip_id}".encode()).digest()
    return int.from_bytes(h[:8], "little") / 2 ** 64 < p


def training_caption(row: dict, style_dropout: float = 0.0, seed: int = 0, epoch: int = 0) -> str:
    """Caption text the DiT is trained on: ``apply_style(caption, row style)`` when the clip's style is known,
    except for a `style_dropout` fraction of samples that keep the bare caption (so the model also works
    without a style tag). Unknown / 'unspecified' styles never get a tag."""
    cap = row.get("caption", "")
    style = row.get("style")
    if not style or style == UNSPECIFIED:
        return cap
    if style_dropped(seed, epoch, str(row.get("id", row.get("index", ""))), style_dropout):
        return cap
    try:
        return apply_style(cap, style)
    except ValueError:               # a style tag this version does not know: train on the plain caption
        return cap


# ===================================================================== trainer
def _stage_at(stages: list, step: int) -> tuple:
    acc = 0
    for i, s in enumerate(stages):
        acc += int(s["steps"])
        if step < acc:
            return i, s
    return len(stages) - 1, stages[-1]


def _hash_unit(salt: str, *parts) -> float:
    h = hashlib.sha256(("/".join([salt, *map(str, parts)])).encode()).digest()
    return int.from_bytes(h[:8], "little") / 2 ** 64


def fps_dropped(seed: int, epoch: int, clip_id: str, p: float) -> bool:
    """Deterministic fps-conditioning dropout (pure function of seed / epoch / clip: no RNG state to checkpoint)."""
    return p >= 1.0 or (p > 0.0 and _hash_unit("fps-dropout", seed, epoch, clip_id) < p)


def clip_fps_value(row: dict) -> float:
    """Effective (drawing-rate) fps of a manifest row, NaN when unknown."""
    v = row.get("effective_fps")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) and v > 0 else float("nan")


def pick_bucket(stage: dict, gen: torch.Generator, force_video: bool = False) -> tuple:
    """(height, width, frames, is_image) for one micro-batch of a stage.

    A stage is a single bucket {height,width,frames}, or a list `buckets` of them (weighted choice). With
    `image_fraction` p a micro-batch is, with probability p, an IMAGE batch (T=1 stills at the bucket's resolution);
    a stage whose frames == 1 is always an image stage. Random draws are made only when the stage asks for
    them, so the draw sequence of a stage without these keys is exactly the old one."""
    bks = stage.get("buckets")
    if bks:
        if len(bks) > 1:
            w = torch.tensor([float(b.get("weight", 1.0)) for b in bks])
            i = int(torch.multinomial(w / w.sum(), 1, generator=gen))
        else:
            i = 0
        b = bks[i]
    else:
        b = stage
    h, w_, f = int(b["height"]), int(b["width"]), int(b["frames"])
    pf = float(stage.get("image_fraction", 0.0))
    if not force_video and f > 1 and pf > 0.0 and float(torch.rand(1, generator=gen)) < pf:
        return h, w_, 1, True
    return h, w_, f, (f == 1 and not force_video)


class SRAProjector(nn.Module):
    """Small MLP that maps early-block features of the online DiT into the space of the (deeper-block) teacher
    features. Training-only: it is never exported into a bundle."""

    def __init__(self, dim: int, hidden: int = 0):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, h):
        return self.net(h)


def sra_loss(student: torch.Tensor, teacher: torch.Tensor, proj: nn.Module,
             token_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Self-representation alignment: 1 - cosine(proj(student tokens), stop-grad(teacher tokens)), averaged over
    tokens (optionally weighted, e.g. to skip clean conditioning frames)."""
    s = F.normalize(proj(student.float()), dim=-1)
    t = F.normalize(teacher.detach().float(), dim=-1)
    d = 1.0 - (s * t).sum(-1)                                  # [B,N]
    if token_weight is None:
        return d.mean()
    return (d * token_weight).sum() / token_weight.sum().clamp_min(1.0)


class RepaHook:
    """Interface for REPA (representation alignment with an EXTERNAL teacher such as DINOv2) -- deliberately NOT
    implemented. DINOv2 would be a licensed opt-in add-on (never downloaded by Nivra, so bundles stay "original
    weights"); a future implementation provides `features(row) -> tensor [N_tokens, D_teacher]` precomputed once per
    clip into the latent shards, and the trainer would add `weight * (1 - cos(proj(h_layer), features))` exactly like
    ``sra_loss``. `train.repa.enabled = true` is rejected by the config loader until then."""

    def features(self, row: dict, shape: tuple) -> torch.Tensor:                # pragma: no cover - interface only
        raise NotImplementedError("REPA needs an external teacher; use train.sra")


def ema_decay_at(decay: float, step: int, warmup: bool = True) -> float:
    """EMA decay at optimizer step `step` (0-based count of finished updates): ramps (1+n)/(10+n) up to `decay`,
    so a short run is not dominated by the random init."""
    if decay <= 0:
        return 0.0
    return min(decay, (1.0 + step) / (10.0 + step)) if warmup else decay


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.lerp_(p.detach().to(pe.dtype), 1.0 - decay)


MIN_LOSS_SCALE = 1.0


def _finite_or_none(v):
    return v if isinstance(v, (int, float, str)) and (not isinstance(v, float) or math.isfinite(v)) else None


def train_hd(cfg, data_root, run_dir, target: str = "dit", text_encoder=None, vae: Optional[HCVAE] = None,
             vae_dir=None, total_steps: Optional[int] = None, device="cpu", seed: int = 0, resume: bool = True,
             log: Callable = print, stop_file=None, latent_norm: Optional[tuple] = None,
             model: Optional[nn.Module] = None) -> dict:
    """Train the HC-VAE (target='vae') or the DiT (target='dit'). Returns a summary dict.

    Everything new is config-gated through ``train`` (see hd_config.train_options); a config without those keys
    trains exactly as before. DiT: logit-normal timesteps with the resolution shift, optional image / bucket
    stages, diffusion forcing (per-frame timesteps on a fraction of batches), first/last latent-frame
    conditioning, SRA, EMA (exported by default when enabled), fp32 islands + dynamic loss scaling with logged
    overflow skips. VAE: staged image -> low-res clips -> target fine-tune with images mixed into video batches,
    hinge patch-GAN after ``vae_gan_start`` with an adaptive weight, edge loss, per-channel latent calibration."""
    if target not in ("vae", "dit"):
        raise ValueError("target must be 'vae' or 'dit'")
    cfg = load_hd_config(cfg)
    tc = cfg["train"]
    topt = train_options(cfg)
    stages = tc["stages"]
    if target == "vae" and topt["vae_stages"]:
        stages = topt["vae_stages"]
    total = int(total_steps if total_steps is not None else sum(int(s["steps"]) for s in stages))
    dev = torch.device(device)
    use_amp = dev.type == "cuda" and bool(tc.get("fp16", True))
    ls_cfg = topt["loss_scale"]
    scaling = use_amp or bool(tc.get("force_loss_scaling", False))      # the CPU switch exists for tests / debugging
    run_dir = Path(run_dir)
    data = ClipShards(data_root, "train")
    style_dropout = float(tc.get("style_dropout", 0.0))
    edge_weight = vae_edge_weight(cfg)
    styles_seen = style_counts(data.rows)
    log(f"[{target}] clips {len(data)} styles {json.dumps(styles_seen)} style_dropout {style_dropout} "
        f"edge_weight {edge_weight}")

    torch.manual_seed(seed)
    random.seed(seed)
    extras: dict = {}                  # name -> (module, optimizer): trained alongside the main net, own checkpoint groups
    ema_model = None
    sra_on = False
    if target == "vae":
        net = model if model is not None else build_hcvae(cfg)
        lr = float(tc.get("vae_lr", tc.get("lr", 1e-4)))
        if topt["vae_gan_weight"] > 0:
            disc = PatchDiscriminator(net.cfg.in_channels, topt["vae_gan_base"], topt["vae_gan_layers"]).to(dev).train()
            extras["disc"] = (disc, torch.optim.AdamW(disc.parameters(), lr=topt["vae_gan_lr"], betas=(0.5, 0.9),
                                                      weight_decay=0.0))
    else:
        if text_encoder is None:
            raise ValueError("DiT training needs a text encoder")
        net = model if model is not None else build_dit(cfg)
        if text_encoder.dim != net.cfg.text_dim:
            raise ValueError(f"text encoder dim {text_encoder.dim} != model text_dim {net.cfg.text_dim}")
        net.set_memory(grad_checkpoint=bool(tc.get("grad_checkpoint", True)), attn_fp32=topt["fp32_attn"] or None)
        if vae is None and vae_dir is not None:
            vae = load_hcvae(vae_dir)
        if vae is not None:
            vae = vae.to(dev).eval().requires_grad_(False)
        lr = float(tc.get("lr", 1e-4))
        if topt["ema_decay"] > 0:
            ema_model = copy.deepcopy(net).to(dev).eval().requires_grad_(False)
        sra_on = topt["sra"]["weight"] > 0
        if sra_on:
            if max(topt["sra"]["student_layer"], topt["sra"]["teacher_layer"]) >= len(net.blocks):
                raise ValueError("train.sra layers exceed the model depth")
            proj = SRAProjector(net.cfg.dim, topt["sra"]["hidden"]).to(dev).train()
            extras["sra"] = (proj, torch.optim.AdamW(proj.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0))
            if ema_model is None:
                log("[dit] SRA without EMA: the teacher is the online model (no gradient); set train.ema_decay for a "
                    "slower, more stable EMA teacher")
    net.to(dev).train()
    opt = make_optimizer(tc.get("optimizer", "adamw"), net.parameters(), lr, float(tc.get("weight_decay", 0.0)))
    scaler = torch.amp.GradScaler("cuda" if dev.type == "cuda" else "cpu", enabled=scaling,
                                  init_scale=ls_cfg["init"], growth_factor=ls_cfg["growth"],
                                  backoff_factor=ls_cfg["backoff"], growth_interval=ls_cfg["growth_interval"])
    accum = max(1, int(tc.get("grad_accum", 1)))
    bs = max(1, int(tc.get("batch_size", 1)))
    warmup = int(tc.get("warmup", 0))
    clip = float(tc.get("grad_clip", 0.0))
    compat = {"arch_version": HD_ARCH_VERSION, "config_hash": hd_config_hash(cfg), "target": target,
              "text_encoder": getattr(text_encoder, "id", None) if target == "dit" else None,
              "text_dim": net.cfg.text_dim if target == "dit" else None}
    df, fps_model = topt["diffusion_forcing"], target == "dit" and bool(net.cfg.fps_cond)

    step, epoch, pos, history = 0, 0, 0, []
    skips = 0
    gen = torch.Generator().manual_seed(seed + 17)       # timesteps / noise / caption dropout / bucket choice
    if resume and (ck := newest_valid(run_dir)) is not None:
        man, groups, st = load_checkpoint(ck, expect_compat=compat)
        net.load_state_dict(groups["model"])
        optimizer_from_flat(opt, groups.get("optimizer", {}), st["opt_groups"])
        if ema_model is not None:
            if "ema" in groups:
                ema_model.load_state_dict(groups["ema"])
            else:                                         # EMA switched on for a run that did not have it
                ema_model.load_state_dict(groups["model"])
        for name, (mod, o) in extras.items():
            if name in groups:
                mod.load_state_dict(groups[name])
                optimizer_from_flat(o, groups.get("opt_" + name, {}), st["opt_groups_" + name])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        restore_rng(groups["rng"], st["rng"])
        gen.set_state(groups["rng"]["trainer_gen"].to(torch.uint8))
        step, epoch, pos, history = st["step"], st["epoch"], st["pos"], st.get("history", [])
        skips = int(st.get("skipped_steps", 0))
        log(f"resumed {target} from {ck.name} at step {step}")

    def order(ep):
        o = list(range(len(data)))
        random.Random(seed * 1000003 + ep).shuffle(o)
        return o

    def next_batch(stage, force_video: bool = False):
        nonlocal epoch, pos
        h_, w_, f_, image = pick_bucket(stage, gen, force_video)
        xs, caps, fpss = [], [], []
        o = order(epoch)
        guard = 0
        while len(xs) < bs:
            if pos >= len(o):
                epoch, pos = epoch + 1, 0
                o = order(epoch)
            i = o[pos]
            kind, x, row = data.get(i)
            pos += 1
            if kind == "videos":
                if image:                                  # a still: a random frame of the clip
                    r = int(torch.randint(0, x.shape[1], (1,), generator=gen))
                    x = x[:, r:r + 1]
                x = fit_video(x, f_, h_, w_)
            elif target == "vae":
                raise ValueError("VAE training needs `videos` shards, got latents")
            else:
                vc = (vae.cfg if vae is not None else None)
                sf = vc.spatial_factor if vc else 32
                tf = vc.temporal_factor if vc else 8
                need_t = 1 + (f_ - 1) // tf
                if x.shape[1] < need_t and x.shape[1] == 1:    # a still-image latent cannot serve a video bucket
                    guard += 1
                    if guard > 4 * len(o):
                        raise ValueError("no latent clip is long enough for this stage (only stills in the dataset)")
                    continue
                x = fit_latent(x, need_t, h_ // sf, w_ // sf)
            xs.append(x)
            caps.append(training_caption(row, style_dropout, seed, epoch))
            v = clip_fps_value(row)
            if fps_dropped(seed, epoch, str(row.get("id", row.get("index", ""))), topt["fps_dropout"]):
                v = float("nan")
            fpss.append(v)
        return kind, torch.stack(xs).to(dev), caps, torch.tensor(fpss), image

    def save(protected=False):
        ot, og = optimizer_to_flat(opt)
        rt, rj = capture_rng()
        rt["trainer_gen"] = gen.get_state().clone()
        groups = {"model": module_tensors(net), "optimizer": ot, "rng": rt}
        sstate = {"step": step, "epoch": epoch, "pos": pos, "opt_groups": og, "rng": rj,
                  "scaler": scaler.state_dict() if scaling else {}, "history": history[-1000:],
                  "skipped_steps": skips}
        if ema_model is not None:
            groups["ema"] = module_tensors(ema_model)
        for name, (mod, o) in extras.items():
            groups[name] = module_tensors(mod)
            groups["opt_" + name], sstate["opt_groups_" + name] = optimizer_to_flat(o)
        p = save_checkpoint(run_dir, step, groups, sstate, compat,
                            {"name": cfg["name"], "train": tc, "target": target}, epoch=epoch, protected=protected)
        apply_retention(run_dir, int(tc.get("keep_last", 3)))
        return p

    def cond_mask_for(B: int, Tg: int):
        """First/last latent-frame conditioning pattern for one batch: (first 1), (first 2), (last 1), (both)."""
        mode = int(torch.randint(0, 4, (1,), generator=gen))
        m = torch.zeros(B, Tg, dtype=torch.bool)
        nf, nl = [(1, 0), (2, 0), (0, 1), (1, 1)][mode]
        if nf + nl >= Tg:                                  # keep at least one noisy frame
            nf, nl = min(nf, Tg - 1), 0
        if nf:
            m[:, :nf] = True
        if nl:
            m[:, -nl:] = True
        return m

    save_every = int(tc.get("save_every", 1000))
    status = "done"
    last_ckpt = None
    consecutive_skips = 0
    gan_active = False
    while step < total:
        if stop_file is not None and Path(stop_file).exists():
            status = "stopped"
            break
        si, stage = _stage_at(stages, step)
        cur_lr = lr * min(1.0, (step + 1) / max(1, warmup)) if warmup else lr
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        for name, (_, o) in extras.items():
            o.zero_grad(set_to_none=True)
            if name == "sra":                       # the projector follows the main schedule; the discriminator has its own lr
                for g in o.param_groups:
                    g["lr"] = cur_lr
        logs = {}
        scale_now = float(scaler.get_scale()) if scaling else 1.0
        for _ in range(accum):
            kind, x, caps, fpsv, is_image = next_batch(stage)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                if target == "vae":
                    recon, m, lv = net(x, generator=None)
                    out = hcvae_loss(x, recon, m, lv, float(tc.get("vae_kl_weight", 1e-6)),
                                     float(tc.get("vae_perc_weight", 0.5)), float(tc.get("vae_tdiff_weight", 0.5)), edge_weight)
                    loss = out["loss"]
                    logs = {k: float(v.detach()) for k, v in out.items()}
                    logs["image_batch"] = float(is_image)
                    gan_active = "disc" in extras and step >= topt["vae_gan_start"]
                    if gan_active:
                        disc, _dopt = extras["disc"]
                        disc.requires_grad_(False)
                        g_loss = hinge_g_loss(disc(recon))
                        disc.requires_grad_(True)
                        aw = adaptive_gan_weight(loss, g_loss, net.last_layer_weight(), scale_now)
                        loss = loss + topt["vae_gan_weight"] * aw * g_loss
                        logs.update(gan_g=float(g_loss.detach()), gan_w=float(aw), loss=float(loss.detach()))
                else:
                    if kind == "videos":
                        if vae is None:
                            raise ValueError("DiT training on `videos` shards needs an HC-VAE (vae or vae_dir)")
                        with torch.no_grad():
                            z0 = vae.to_scaled(vae.encode(x).float())
                    else:
                        if latent_norm is None and vae is not None:
                            z0 = vae.to_scaled(x)                  # scalar or per-channel calibration of the VAE
                        else:
                            sh, sc = latent_norm if latent_norm is not None else (0.0, 1.0)
                            z0 = (x - sh) * sc
                    emb, mask = text_encoder.encode(caps)
                    B = z0.shape[0]
                    Tl = z0.shape[2]
                    pt = net.cfg.patch[0]
                    Tg = Tl // pt
                    n_tok = Tl * z0.shape[3] * z0.shape[4]
                    shift = resolution_shift(n_tok, int(tc.get("shift_base_tokens", 4096)))
                    use_df = df["fraction"] > 0 and float(torch.rand(1, generator=gen)) < df["fraction"]
                    use_cond = (df["cond_fraction"] > 0 and Tg > 1
                                and float(torch.rand(1, generator=gen)) < df["cond_fraction"])
                    t = sample_t(B, shift, generator=gen, device=dev)
                    noise = torch.randn(z0.shape, generator=gen).to(dev)
                    drop = torch.rand(B, generator=gen) < float(tc.get("caption_dropout", 0.1))
                    if use_df:                                   # independent noise level per latent frame
                        t = sample_t_frames(B, Tg, shift, generator=gen, device=dev)
                    cmask = cond_mask_for(B, Tg).to(dev) if use_cond else None
                    if cmask is not None and t.dim() == 1:
                        t = t[:, None].expand(B, Tg).contiguous()
                    fps_arg = fpsv.to(dev) if fps_model else None
                    do_sra = (sra_on and (topt["sra"]["until_step"] == 0 or step < topt["sra"]["until_step"]))
                    feats = (topt["sra"]["student_layer"],) if do_sra else None
                    lr_loss, fd = rf_loss_ex(net, z0, emb.to(dev), mask.to(dev), t, noise, drop.to(dev), cmask, fps_arg,
                                             feats)
                    loss = lr_loss
                    logs = {"loss": float(lr_loss.detach())}
                    if do_sra:
                        t_teach = t * topt["sra"]["t_ratio"]
                        with torch.no_grad():
                            teacher = ema_model if ema_model is not None else net
                            xt_t = make_xt(z0.float(), noise.float(), t_teach, cmask)
                            gm = (cmask[:, ::pt] if (cmask is not None and pt > 1) else cmask)
                            tg = t_teach if (t_teach.dim() == 1 or pt == 1) else t_teach[:, ::pt]
                            _, td = teacher(xt_t, tg, emb.to(dev), mask.to(dev), drop.to(dev), cond_mask=gm, fps=fps_arg,
                                            feats=(topt["sra"]["teacher_layer"],))
                        hs, ht = fd[topt["sra"]["student_layer"]], td[topt["sra"]["teacher_layer"]]
                        tw = None
                        if cmask is not None:                    # clean conditioning frames carry no noise to align
                            hw = z0.shape[3] // net.cfg.patch[1] * (z0.shape[4] // net.cfg.patch[2])
                            tw = (~(cmask[:, ::pt] if pt > 1 else cmask)).float()[:, :, None].expand(B, Tg, hw).reshape(B, -1)
                        sl = sra_loss(hs, ht, extras["sra"][0], tw)
                        loss = loss + topt["sra"]["weight"] * sl
                        logs.update(sra=float(sl.detach()), loss=float(loss.detach()), rf=float(lr_loss.detach()))
                    logs["tokens"] = float(n_tok)
                    if use_df or use_cond:
                        logs["df"] = float(use_df)
                        logs["cond"] = float(use_cond)
            scaler.scale(loss / accum).backward()
            if target == "vae" and gan_active:
                disc, dopt = extras["disc"]
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    d_loss = hinge_d_loss(disc(x), disc(recon.detach()))
                scaler.scale(d_loss / accum).backward()
                logs["gan_d"] = float(d_loss.detach())
        if clip > 0:
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), clip))
            logs["grad_norm"] = gn
        if not all(math.isfinite(v) for v in logs.values()) and not scaling:
            raise FloatingPointError(f"non-finite loss/grad at step {step}: {logs}")
        # an overflow = any non-finite gradient (the scaler skips exactly those steps); detected from the gradients
        # themselves so it still works once the scale has been driven to its floor
        overflow = scaling and not all(bool(torch.isfinite(p.grad).all()) for m in [net] + [e[0] for e in extras.values()]
                                       for p in m.parameters() if p.grad is not None)
        scale_before = float(scaler.get_scale()) if scaling else 1.0
        scaler.step(opt)
        for name, (mod, o) in extras.items():
            if any(p.grad is not None for p in mod.parameters()):      # e.g. no D before gan_start, no SRA after until_step
                scaler.step(o)
        scaler.update()
        if scaling and float(scaler.get_scale()) < MIN_LOSS_SCALE:     # a scale below 1 only shrinks gradients
            scaler.update(MIN_LOSS_SCALE)
        skipped = scaling and (overflow or float(scaler.get_scale()) < scale_before)
        if skipped:
            skips += 1
            consecutive_skips += 1
            log(f"[{target}] step {step + 1}: fp16 overflow (non-finite gradients) - optimizer step skipped, "
                f"loss scale {scale_before:g} -> {float(scaler.get_scale()):g} ({skips} skipped so far)")
            if consecutive_skips >= 200:
                raise FloatingPointError(f"{consecutive_skips} consecutive skipped steps at step {step}: the loss is "
                                         "overflowing even at the minimum loss scale (see train.fp32_attn)")
        else:
            consecutive_skips = 0
            if ema_model is not None:
                ema_update(ema_model, net, ema_decay_at(topt["ema_decay"], step, topt["ema_warmup"]))
        if scaling:
            logs["loss_scale"] = float(scaler.get_scale())
            logs["skipped"] = float(skipped)
        step += 1
        history.append({"step": step, "stage": stage["name"], "lr": cur_lr,
                        **{k: _finite_or_none(v) for k, v in logs.items()}})
        if step % max(1, save_every) == 0 or step == total:
            last_ckpt = save()
        log(f"[{target}] step {step}/{total} stage {stage['name']} " +
            " ".join(f"{k}={v:.4f}" for k, v in logs.items()))
    if status == "stopped":
        last_ckpt = save()
    result = {"status": status, "step": step, "history": history, "ckpt": str(last_ckpt) if last_ckpt else None,
              "params": sum(p.numel() for p in net.parameters()), "styles": styles_seen, "skipped_steps": skips}
    if ema_model is not None:
        result["export_state"] = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
        result["export_kind"] = "ema"
    elif target == "dit":
        result["export_kind"] = "raw"
    if target == "vae" and status == "done":
        # calibrate the latent scale on a few training clips and export the VAE (safetensors + JSON)
        net.eval()
        per = topt["vae_latent_norm"] == "channel"
        with torch.no_grad():
            _, stage = _stage_at(stages, max(0, step - 1))
            zs = [net.encode(next_batch(stage, force_video=True)[1]).float().cpu() for _ in range(4 if per else 2)]
        result["latent_norm"] = net.calibrate(torch.cat(zs), per_channel=per)
        result["vae_dir"] = str(run_dir / "hc_vae")
        save_hcvae(net, run_dir / "hc_vae")
    return result
