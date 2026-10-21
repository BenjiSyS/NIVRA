"""Nivra HD compute / memory budget.

Computes (estimates) FLOPs, parameter memory, peak activation memory and wall-clock time per
config x resolution x frames x steps x GPU profile, and MEASURES real per-block forward time on the local
CPU at reduced token counts to sanity-check the FLOP model. Writes a Markdown report that keeps measured and
estimated numbers apart.

    python ai/scripts/hd_budget.py --out docs/research/NIVRA_720P_MEASUREMENTS.md
    python ai/scripts/hd_budget.py --quick --out <tmp>.md      # fast smoke run (tests)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from nivra.compression.hc_vae import HCVAE, shape_trace, tile_ranges  # noqa: E402
from nivra.inference.hd import frames_for, padded  # noqa: E402
from nivra.models.dit import DiTBlock, NivraDiT, count_params_meta, quantize_int8_, resolution_shift  # noqa: E402
from nivra.models.hd_config import dit_config, load_hd_config, vae_config  # noqa: E402
from nivra.models.rope import rope_3d  # noqa: E402

GB = 1024 ** 3

# --------------------------------------------------------------------------------------------- assumptions
# Effective (achieved, not peak) fp16 throughput in TFLOPS as (low, high). These are ASSUMPTIONS, not
# measurements: nothing in this repository has been run on these GPUs.
# RTX 2060 desktop (TU106, 240 tensor cores @ ~1.68 GHz): peak fp16 tensor ~51.6 TFLOPS with fp16 accumulate,
# ~25.8 TFLOPS with fp32 accumulate (what PyTorch GEMMs use). Large GEMMs typically reach 40-60 % of that;
# SDPA on sm_75 uses the memory-efficient kernel (FlashAttention-2 needs sm_80+), assumed 20-35 %;
# cuDNN fp16 conv3d assumed 15-30 %. Laptop RTX 2060 parts clock lower (subtract ~15-25 %).
GPU_PROFILES = [
    {"name": "RTX 2060 6 GB (Turing)", "vram_gb": 6, "lin": (10.0, 15.0), "attn": (5.0, 9.0), "conv": (4.0, 8.0),
     "pcie_gbs": (10.0, 12.0)},
    {"name": "8 GB Ampere/Ada class (e.g. RTX 3060 Ti / 4060)", "vram_gb": 8, "lin": (18.0, 30.0),
     "attn": (12.0, 22.0), "conv": (8.0, 15.0), "pcie_gbs": (10.0, 24.0)},
    {"name": "12 GB (e.g. RTX 3060 12 GB)", "vram_gb": 12, "lin": (14.0, 22.0), "attn": (9.0, 16.0),
     "conv": (6.0, 11.0), "pcie_gbs": (10.0, 24.0)},
    {"name": "16 GB (e.g. RTX 4060 Ti 16 GB / 4070 Ti S)", "vram_gb": 16, "lin": (25.0, 50.0),
     "attn": (18.0, 40.0), "conv": (10.0, 25.0), "pcie_gbs": (10.0, 24.0)},
    {"name": "24 GB (e.g. RTX 3090 / 4090)", "vram_gb": 24, "lin": (35.0, 100.0), "attn": (25.0, 80.0),
     "conv": (15.0, 45.0), "pcie_gbs": (20.0, 24.0)},
]
CUDA_OVERHEAD_GB = 0.7          # CUDA context + cuDNN/cuBLAS workspaces + allocator fragmentation (assumed)
TEXT_MIN = (1.0, 3.0)           # text-encoder load + encode + offload, minutes (assumed, design doc section 5)
LOAD_MIN = (0.3, 1.5)           # DiT + HC-VAE weights from disk to GPU, minutes (assumed)
TEXT_TOKENS = 128

RESOLUTIONS = [("256p", 448, 256, 2), ("480p", 832, 480, 4), ("720p", 1280, 720, 8)]
STEPS = [20, 30, 40, 50]


# --------------------------------------------------------------------------------------------- FLOP model
def dit_forward_flops(cfg, n_tokens: int, text_tokens: int = TEXT_TOKENS) -> dict:
    """Analytic FLOPs of one DiT forward (multiply-add = 2 FLOPs)."""
    d, h, L, N = cfg.dim, cfg.mlp_hidden, cfg.depth, n_tokens
    lin_tok = 2 * N * (3 * d * d + d * d + d * d + d * d + 2 * d * h)      # qkv, proj, cross q, cross proj, mlp
    lin_txt = 2 * text_tokens * 2 * d * d                                    # cross k, v
    attn = 4 * N * N * d + 4 * N * text_tokens * d                           # QK^T + AV (self, cross)
    io = 2 * N * cfg.latent_channels * d * 2 + 2 * text_tokens * cfg.text_dim * d
    return {"linear": L * (lin_tok + lin_txt) + io, "attention": L * attn}


def activation_bytes_inference(cfg, n_tokens: int, mlp_chunk: int, dtype_bytes: int = 2) -> int:
    """Peak transient memory of one block at inference (estimate): residual stream, modulated input, qkv,
    rope'd q/k, attention output and projection in fp16, rope float32 temporaries (q or k at a time), one
    MLP chunk (hidden + output), memory-efficient attention keeps O(N) extra state."""
    N, d = n_tokens, cfg.dim
    c = min(N, mlp_chunk) if mlp_chunk > 0 else N
    fp16 = N * d * dtype_bytes * (1 + 1 + 3 + 2 + 1 + 1)
    rope32 = N * d * 4 * 3
    mlp = c * (cfg.mlp_hidden + d) * dtype_bytes * 2
    sampler = 5 * N * cfg.latent_channels * 4                     # x, noise, v, v_uncond, prev_v (float32)
    return fp16 + rope32 + mlp + sampler


def weight_bytes(params: dict, cfg, mode: str) -> int:
    if mode == "fp16":
        return 2 * params["total"]
    if mode == "int8":                                        # block linear weights int8 + fp16 rest + scales
        lin_w = params["per_block_linear"] * cfg.depth
        return lin_w + 2 * (params["total"] - lin_w) + 2 * cfg.depth * 8 * cfg.dim
    raise ValueError(mode)


def training_bytes(params: dict, cfg, n_tokens: int, optimizer: str) -> int:
    """fp32 master weights + grads + optimizer states + checkpointed activations (estimate)."""
    P = params["total"]
    opt = 8 * P if optimizer == "adamw" else 2 * P + P // 64
    stored = cfg.depth * n_tokens * cfg.dim * 2                  # block inputs kept by activation checkpointing
    one_block = activation_bytes_inference(cfg, n_tokens, 0) * 3  # recompute + backward of one block
    return 4 * P + 4 * P + opt + stored + one_block


def vae_decode_plan(vcfg, Hl: int, Wl: int, t_chunk: int, tile: int, overlap: int, margin) -> dict:
    """Tiling overhead (compute multiplier) and peak activation estimate from a meta-device trace."""
    with torch.device("meta"):
        vae = HCVAE(vcfg)
    m = vae.decoder.radius if margin is None else margin
    rh, rw = tile_ranges(Hl, tile, overlap), tile_ranges(Wl, tile, overlap)
    area = 0
    biggest_tile = (0, 0)
    for a, b in rh:
        for c, d in rw:
            hh = min(Hl, b + m) - max(0, a - m)
            ww = min(Wl, d + m) - max(0, c - m)
            area += hh * ww
            biggest_tile = max(biggest_tile, (hh, ww), key=lambda s: s[0] * s[1])
    largest = [0]

    def hook(_m, _i, out):
        if torch.is_tensor(out):
            largest[0] = max(largest[0], out.numel())
    with torch.no_grad(), torch.device("meta"):
        z = torch.empty(1, vcfg.latent_channels, t_chunk, *biggest_tile)
        vae.decoder.set_streaming(True)
        vae.decoder(z)                                    # first chunk (1 + 8(t-1) frames)
        hs = [mod.register_forward_hook(hook) for mod in vae.decoder.modules()]
        vae.decoder(z)                                    # steady-state chunk (8t frames) is the peak
        vae.decoder.set_streaming(False)
    for h in hs:
        h.remove()
    dec_params = sum(p.numel() for p in vae.decoder.parameters())
    peak = 4 * largest[0] * 2 + 2 * dec_params                # ~4 live tensors of the largest size, fp16
    return {"overhead": area / (Hl * Wl), "tiles": len(rh) * len(rw), "tile_cells": biggest_tile, "margin": m,
            "radius": vae.decoder.radius, "t_chunk": t_chunk, "tile": tile, "overlap": overlap,
            "largest_activation_gb": largest[0] * 2 / GB, "peak_gb": peak / GB}


# --------------------------------------------------------------------------------------------- measurement
def _time(fn, repeat: int) -> float:
    fn()                                                      # warm-up
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def measure_block(cfg, n_tokens: int, repeat: int, int8: bool = False) -> dict:
    torch.manual_seed(0)
    blk = DiTBlock(cfg).eval()
    if int8:
        holder = torch.nn.Module()
        holder.blocks = torch.nn.ModuleList([blk])
        quantize_int8_(holder)
    side = max(1, round(math.sqrt(n_tokens / 4)))
    T, H, W = 4, side, max(1, n_tokens // (4 * side))
    N = T * H * W
    x = torch.randn(1, N, cfg.dim)
    mod = torch.randn(1, 6, cfg.dim) * 0.1
    ctx = torch.randn(1, TEXT_TOKENS, cfg.dim)
    mask = torch.ones(1, TEXT_TOKENS, dtype=torch.bool)
    cos, sin = rope_3d(T, H, W, cfg.head_dim)
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            blk(x, mod, ctx, mask, cos, sin)
        sec = _time(lambda: blk(x, mod, ctx, mask, cos, sin), repeat)
    model = dit_forward_flops(cfg, N)
    lin = (model["linear"] - 2 * N * cfg.latent_channels * cfg.dim * 2 - 2 * TEXT_TOKENS * cfg.text_dim * cfg.dim) / cfg.depth
    att = model["attention"] / cfg.depth
    # FlopCounterMode sees the aten matmuls on CPU but not the fused CPU SDPA kernel, so the counted number
    # checks the linear part of the model exactly; attention FLOPs come from the model (verified on meta
    # tensors in tests/test_hd_budget.py).
    return {"tokens": N, "seconds": sec, "flops_counted_linear": fc.get_total_flops(), "flops_model": lin + att,
            "flops_model_linear": lin, "flops_model_attention": att, "gflops": (lin + att) / sec / 1e9,
            "int8": int8}


def measure_vae_decode(vcfg, repeat: int) -> dict:
    torch.manual_seed(0)
    vae = HCVAE(vcfg).eval()
    z = torch.randn(1, vcfg.latent_channels, 2, 2, 2)          # 9 frames of 64x64
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            vae.decode(z)
        sec = _time(lambda: vae.decode(z), repeat)
    return {"latent": list(z.shape), "frames": 9, "pixels": "64x64", "seconds": sec,
            "flops": fc.get_total_flops(), "gflops": fc.get_total_flops() / sec / 1e9}


def fit_rates(rows: list) -> tuple:
    """Least squares t = a*F_lin + b*F_attn -> effective linear / attention GFLOPS (CPU)."""
    import numpy as np
    A = np.array([[r["flops_model_linear"], r["flops_model_attention"]] for r in rows])
    y = np.array([r["seconds"] for r in rows])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([a, b])
    rel = float(np.max(np.abs(pred - y) / y))
    return (1 / a / 1e9 if a > 0 else float("nan")), (1 / b / 1e9 if b > 0 else float("nan")), rel


# --------------------------------------------------------------------------------------------- estimate
def plan_mode(params, cfg, vram_gb, act_gb):
    """Cheapest-quality-loss mode that fits: fp16 resident > int8 resident > fp16 offload > int8 offload."""
    budget = 0.95 * vram_gb - CUDA_OVERHEAD_GB        # 5 % safety margin on top of the CUDA overhead
    per_block16 = 2 * params["per_block"] / GB
    for mode, offload in (("fp16", False), ("int8", False), ("fp16", True), ("int8", True)):
        w = weight_bytes(params, cfg, mode) / GB
        resident = (2 * params["non_block"] / GB + 2 * per_block16 * (0.5 if mode == "int8" else 1.0)) if offload else w
        if resident + act_gb <= budget:
            return mode, offload, resident
    return None, None, None


def estimate(cfg_name: str, quick: bool) -> dict:
    cfg = load_hd_config(cfg_name)
    dc, vc = dit_config(cfg), vae_config(cfg)
    inf = cfg["inference"]
    params = count_params_meta(dc)
    rows = []
    for rname, W, H, sec in RESOLUTIONS:
        frames = frames_for(sec, 30, vc.temporal_factor)
        Hp, Wp = padded(H, vc.spatial_factor), padded(W, vc.spatial_factor)
        C, Tl, Hl, Wl = vc.latent_shape(frames, Hp, Wp)
        N = Tl * Hl * Wl
        fl = dit_forward_flops(dc, N)
        act_gb = activation_bytes_inference(dc, N, int(inf.get("mlp_chunk", 0))) / GB
        if quick and rname != "720p":
            vplan, vtr = None, None
        else:
            vplan = vae_decode_plan(vc, Hl, Wl, int(inf.get("vae_t_chunk", 1)), int(inf.get("vae_tile", 0)),
                                    int(inf.get("vae_overlap", 2)), inf.get("vae_margin"))
            vtr = shape_trace(vc, frames, Hp, Wp)
        rows.append({"res": rname, "width": W, "height": H, "padded": [Hp, Wp], "frames": frames,
                     "latent": [C, Tl, Hl, Wl], "tokens": N, "shift": resolution_shift(N, int(inf.get("shift_base_tokens", 4096))),
                     "flops_linear": fl["linear"], "flops_attention": fl["attention"],
                     "act_gb": act_gb, "vae_plan": vplan, "vae_trace": vtr,
                     "train_gb_adamw8": training_bytes(params, dc, N, "adamw8") / GB,
                     "train_gb_adamw": training_bytes(params, dc, N, "adamw") / GB})
    return {"name": cfg_name, "params": params, "dit": dc, "vae": vc, "rows": rows}


def gpu_times(est: dict, row: dict, prof: dict, steps: int) -> dict:
    params, dc = est["params"], est["dit"]
    mode, offload, resident = plan_mode(params, dc, prof["vram_gb"], row["act_gb"])
    if mode is None:
        return {"fits": False}
    out = {"fits": True, "mode": mode + (" + block offload" if offload else ""), "resident_gb": resident}
    lo_hi = []
    for k in (1, 0):        # optimistic uses the high throughput, pessimistic the low one
        lin = row["flops_linear"] / (prof["lin"][k] * 1e12)
        att = row["flops_attention"] / (prof["attn"][k] * 1e12)
        if mode == "int8":
            lin *= 1.15     # on-the-fly dequantisation overhead (assumed)
        fwd = lin + att
        if offload:
            xfer = weight_bytes(params, dc, mode) / (prof["pcie_gbs"][k] * 1e9)
            fwd = max(fwd, xfer) + 0.1 * min(fwd, xfer)   # prefetch overlaps copies with compute (assumed 90 %)
        step = 2 * fwd                                     # CFG: conditional + null-text pass
        vae = 0.0
        if row["vae_trace"] is not None:
            vae = row["vae_trace"]["decode_flops"] * row["vae_plan"]["overhead"] / (prof["conv"][k] * 1e12)
            vae += row["vae_trace"]["encode_flops"] * 9 / row["frames"] / (prof["conv"][k] * 1e12)  # tail re-encode
        total_min = steps * step / 60 + vae / 60 + TEXT_MIN[1 - k] + LOAD_MIN[1 - k]
        lo_hi.append({"step_s": step, "sampling_min": steps * step / 60, "vae_min": vae / 60, "total_min": total_min})
    out["optimistic"], out["pessimistic"] = lo_hi
    return out


# --------------------------------------------------------------------------------------------- report
def fmt_gb(x):
    return f"{x:.2f}"


def report(ests, meas, args) -> str:
    L = []
    w = L.append
    w("# Nivra 720p budget: measurements and estimates")
    w("")
    w(f"Generated by `ai/scripts/hd_budget.py` on {time.strftime('%Y-%m-%d')}"
      f"{' (quick mode)' if args.quick else ''}. Two kinds of numbers appear here and are never mixed:")
    w("")
    w("* **MEASURED**: wall-clock times actually observed on this development machine's CPU (no GPU here).")
    w("* **ESTIMATED**: arithmetic from the FLOP/memory model plus *assumed* GPU throughputs. No GPU number in "
      "this document is a measurement; nothing has been run on an RTX 2060 or any other GPU.")
    w("")
    w("## 1. Architecture facts (exact)")
    w("")
    w("| Model | d | layers | heads | head dim | parameters (exact, meta device) | fp16 weights | int8 weight-only |")
    w("|---|--:|--:|--:|--:|--:|--:|--:|")
    for e in ests:
        dc, p = e["dit"], e["params"]
        w(f"| {e['name']} | {dc.dim} | {dc.depth} | {dc.heads} | {dc.head_dim} | {p['total']:,} | "
          f"{fmt_gb(weight_bytes(p, dc, 'fp16') / GB)} GiB | {fmt_gb(weight_bytes(p, dc, 'int8') / GB)} GiB |")
    w("")
    w("Per block: " + "; ".join(f"{e['name']} {e['params']['per_block']:,}" for e in ests) + " parameters "
      "(self-attention 4d^2, cross-attention 4d^2, MLP 2*4d^2, plus a 6xd adaLN table). adaLN-Zero is the "
      "shared form (one timestep MLP + per-block learned table), which is why the exact counts are below the "
      "design doc's rough estimates (Medium 0.93 B vs 1.0 B, Large 2.78 B vs 3.5 B); per-block adaLN "
      "projections would add 6d^2 per block.")
    e0 = ests[0]
    r720 = [r for r in e0["rows"] if r["res"] == "720p"][0]
    vt = r720["vae_trace"]
    w("")
    w(f"HC-VAE ({e0['name']} config): {vt['params']:,} parameters (encoder {vt['encoder_params']:,}, "
      f"decoder {vt['decoder_params']:,}); 1280x736x241 -> latent {r720['latent']} = {r720['tokens']:,} tokens "
      f"(shape trace on the meta device). Decode {vt['decode_flops'] / 1e12:.0f} TFLOP, encode "
      f"{vt['encode_flops'] / 1e12:.0f} TFLOP for the full clip (counted by FlopCounterMode on meta tensors).")
    vp = r720["vae_plan"]
    w(f"Configured decode tiling: {vp['t_chunk']} latent frame(s) per chunk, tile {vp['tile']} / overlap "
      f"{vp['overlap']} / margin {vp['margin']} latent cells (decoder receptive radius {vp['radius']} cells; "
      f"margin >= radius is exact): {vp['tiles']} tiles of up to {vp['tile_cells']} cells, compute overhead "
      f"x{vp['overhead']:.2f}, largest activation {vp['largest_activation_gb']:.2f} GiB, estimated decode peak "
      f"{vp['peak_gb']:.2f} GiB (fp16, estimate).")
    w("")
    w("## 2. MEASURED on this CPU")
    w("")
    hw = meas["hardware"]
    w(f"Machine: {hw['processor']} ({hw['platform']}), torch {hw['torch']}, {hw['threads']} threads, float32, "
      f"median of {meas['repeat']} runs after one warm-up. Real DiT blocks with the production widths, "
      "random weights, 128 text tokens.")
    w("")
    w("| Block | tokens | time (s) | linear FLOPs counted (aten) | linear FLOPs model | attention FLOPs model | "
      "achieved GFLOPS (model total / time) |")
    w("|---|--:|--:|--:|--:|--:|--:|")
    for name, rows in meas["blocks"].items():
        for r in rows:
            w(f"| {name}{' int8' if r['int8'] else ''} | {r['tokens']:,} | {r['seconds']:.3f} | "
              f"{r['flops_counted_linear'] / 1e9:.2f} G | {r['flops_model_linear'] / 1e9:.2f} G | "
              f"{r['flops_model_attention'] / 1e9:.2f} G | {r['gflops']:.1f} |")
    w("")
    for name, fit in meas["fits"].items():
        w(f"* {name}: least-squares fit t = F_lin/R_lin + F_attn/R_attn over the fp32 rows gives R_lin = "
          f"{fit[0]:.1f} GFLOPS, R_attn = {fit[1]:.1f} GFLOPS, max relative residual {fit[2] * 100:.1f} %. "
          "A small residual means time scales the way the FLOP model says (linear + quadratic terms).")
    v = meas["vae"]
    w(f"* HC-VAE decoder (production config, {v['latent']} latent -> {v['frames']} frames of {v['pixels']}): "
      f"{v['seconds']:.3f} s, {v['flops'] / 1e9:.1f} GFLOP, {v['gflops']:.1f} GFLOPS.")
    med = [e for e in ests if e["name"] == "nivra_medium"]
    if med and meas["fits"]:
        r = [x for x in med[0]["rows"] if x["res"] == "720p"][0]
        fit = list(meas["fits"].values())[0]
        if fit[0] > 0 and fit[1] > 0:
            cpu_step = 2 * (r["flops_linear"] / (fit[0] * 1e9) + r["flops_attention"] / (fit[1] * 1e9))
            w(f"* Extrapolated from these CPU rates, one Medium 720p CFG step would take ~{cpu_step / 3600:.1f} h "
              "on this CPU: HD sampling is GPU-only.")
    w("")
    w("## 3. ESTIMATED: per-step FLOPs and memory")
    w("")
    w("| Model | res | frames | latent (C,T,H,W) | tokens | shift | linear TFLOP/fwd | attention TFLOP/fwd | "
      "CFG step TFLOP | inference activations (GiB) | train fp32+AdamW8 (GiB) | train fp32+AdamW (GiB) |")
    w("|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for e in ests:
        for r in e["rows"]:
            w(f"| {e['name']} | {r['res']} | {r['frames']} | {tuple(r['latent'])} | {r['tokens']:,} | {r['shift']:.2f} | "
              f"{r['flops_linear'] / 1e12:.1f} | {r['flops_attention'] / 1e12:.1f} | "
              f"{2 * (r['flops_linear'] + r['flops_attention']) / 1e12:.0f} | {r['act_gb']:.2f} | "
              f"{r['train_gb_adamw8']:.1f} | {r['train_gb_adamw']:.1f} |")
    w("")
    w(f"Memory model: weights resident + one block's transient activations + {CUDA_OVERHEAD_GB} GiB CUDA "
      "context/workspace/fragmentation must fit in 95 % of VRAM; the mode chosen per GPU is the first that fits "
      "of fp16, int8, fp16 + block offload, int8 + block offload. The HC-VAE decodes after the DiT has left the GPU. Training columns: fp32 "
      "master weights + gradients + optimizer states + checkpointed activations, batch 1 (estimate); compare "
      "them with each profile's VRAM. None of the Medium/Large training rows fits a 6 GB card.")
    w("")
    w("## 4. ESTIMATED: wall-clock per GPU profile")
    w("")
    w("Assumed effective fp16 throughput (TFLOPS, low-high; not measured): " + "; ".join(
        f"{p['name']}: linear {p['lin'][0]:g}-{p['lin'][1]:g}, attention {p['attn'][0]:g}-{p['attn'][1]:g}, "
        f"conv3d {p['conv'][0]:g}-{p['conv'][1]:g}" for p in GPU_PROFILES) + ". "
      f"Fixed costs: text encoder {TEXT_MIN[0]:g}-{TEXT_MIN[1]:g} min, weight loading {LOAD_MIN[0]:g}-{LOAD_MIN[1]:g} min.")
    w("")
    w("| Model | res/frames | GPU | mode | steps | sec/step (opt-pess) | sampling min | VAE min | total min (opt-pess) |")
    w("|---|---|---|---|--:|--:|--:|--:|--:|")
    verdict = {}
    for e in ests:
        for r in e["rows"]:
            if r["vae_trace"] is None:
                continue
            for prof in GPU_PROFILES:
                for s in STEPS:
                    g = gpu_times(e, r, prof, s)
                    if not g["fits"]:
                        if s == STEPS[0]:
                            w(f"| {e['name']} | {r['res']}/{r['frames']} | {prof['name']} | does not fit | | | | | |")
                        continue
                    o, p = g["optimistic"], g["pessimistic"]
                    if r["res"] == "720p" and prof["vram_gb"] == 6:
                        verdict[(e["name"], s)] = (g["mode"], o["total_min"], p["total_min"], o["step_s"], p["step_s"])
                    if r["res"] != "720p" and s not in (30,):
                        continue
                    w(f"| {e['name']} | {r['res']}/{r['frames']} | {prof['name']} | {g['mode']} | {s} | "
                      f"{o['step_s']:.1f}-{p['step_s']:.1f} | {o['sampling_min']:.1f}-{p['sampling_min']:.1f} | "
                      f"{o['vae_min']:.1f}-{p['vae_min']:.1f} | {o['total_min']:.0f}-{p['total_min']:.0f} |")
    w("")
    w("## 5. Verdict: 720p / 8 s on an RTX 2060 6 GB in 30-40 minutes (ESTIMATE)")
    w("")
    for name in [e["name"] for e in ests]:
        rows = sorted((s, v) for (n, s), v in verdict.items() if n == name)
        if not rows:
            continue
        mode = rows[0][1][0]
        line = ", ".join(f"{s} steps: {v[1]:.0f}-{v[2]:.0f} min" for s, v in rows)
        ok_opt = [s for s, v in rows if v[1] <= 40]
        ok_pess = [s for s, v in rows if v[2] <= 40]
        w(f"* **{name}** ({mode}): {line}.")
        if name == "nivra_medium":
            s0, v0 = rows[0]
            fixed_opt = v0[1] - s0 * v0[3] / 60          # text + load + VAE (minutes)
            fixed_pess = v0[2] - s0 * v0[4] / 60
            budget_steps_opt = int((40 - fixed_opt) / (v0[3] / 60))
            budget_steps_pess = int((40 - fixed_pess) / (v0[4] / 60))
            w(f"  Within 40 minutes the sampler can afford about {max(0, budget_steps_pess)} CFG steps under the "
              f"pessimistic assumptions and about {budget_steps_opt} under the optimistic ones "
              f"(steps that fit in the table: optimistic {ok_opt or 'none'}, pessimistic {ok_pess or 'none'}).")
            w(f"  Plain answer: **plausible but not demonstrated.** With fp16 weights resident (no int8 needed: "
              f"{fmt_gb(weight_bytes(e0['params'], e0['dit'], 'fp16') / GB)} GiB weights + "
              f"{r720['act_gb']:.2f} GiB activations fit in 6 GB), the 2nd-order `ab2` sampler at about 20-30 "
              "CFG steps lands in or near 30-40 minutes if Turing sustains the assumed throughput; 40+ steps "
              "only fits under the optimistic assumptions. Attention over 28.5 k tokens is "
              f"{r720['flops_attention'] / (r720['flops_attention'] + r720['flops_linear']) * 100:.0f} % of the "
              "FLOPs and runs on the memory-efficient SDPA kernel, so the attention throughput assumption "
              "dominates the uncertainty. int8 weights are not needed for memory and would cost time "
              "(dequantisation); they are the fallback for 4-5 GB cards.")
        else:
            w("  Large does not fit in 6 GB with fp16 weights; it needs int8 weights (fits) or block offload, "
              "and at ~3x Medium's FLOPs per step it is not a 30-40 minute model on a 2060.")
    w("")
    w("Must be replaced by real measurements on a 2060 (run `HDPipeline(..., device='cuda')` and record "
      "`seconds_sampling`, `seconds_decode`, `cuda_max_allocated_mb` from the result metadata).")
    w("")
    w("## 6. Untested / caveats")
    w("")
    w("* All CUDA paths (fp16 autocast and `.half()` inference, GradScaler, pinned-memory/stream prefetch "
      "offload, cuDNN conv3d, SDPA kernel choice on sm_75) are implemented but untested on GPU.")
    w("* Decode tiling with a margin smaller than the decoder's receptive radius is approximate (blended "
      "seams); the configured margin is in section 1. Exactness at margin = radius is covered by tests.")
    w("* Activation and peak-memory figures are model estimates, not allocator measurements.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT.parent / "docs" / "research" / "NIVRA_720P_MEASUREMENTS.md"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny token counts, Medium only (smoke test)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, min(4, args.threads)))
    names = ["nivra_medium"] if args.quick else ["nivra_medium", "nivra_large"]
    ests = [estimate(n, args.quick) for n in names]
    token_sets = {"nivra_medium": [128, 256] if args.quick else [512, 1024, 2048, 4096],
                  "nivra_large": [512, 1024, 2048, 3072]}
    meas = {"hardware": {"processor": platform.processor(), "platform": platform.platform(),
                         "torch": torch.__version__, "threads": torch.get_num_threads(), "cpus": os.cpu_count()},
            "repeat": 1 if args.quick else args.repeat, "blocks": {}, "fits": {}}
    for e in ests:
        rows = [measure_block(e["dit"], n, meas["repeat"]) for n in token_sets[e["name"]]]
        meas["fits"][e["name"]] = fit_rates(rows)
        rows.append(measure_block(e["dit"], token_sets[e["name"]][-1], meas["repeat"], int8=True))
        meas["blocks"][e["name"]] = rows
    meas["vae"] = measure_vae_decode(ests[0]["vae"], meas["repeat"])
    text = report(ests, meas, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"measured": meas, "estimates": [
            {"name": e["name"], "params": e["params"], "rows": e["rows"]} for e in ests]}, indent=1, default=str))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
