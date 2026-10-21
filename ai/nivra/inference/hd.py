"""Nivra HD (720p) text-to-video pipeline: prompt -> text encoder -> rectified-flow sampler with CFG
(Euler or a 2nd-order Adams-Bashforth multistep, "UniPC-style") -> HC-VAE chunked/tiled causal decode ->
MP4 (existing video_io), with progress callbacks, stop-file support and GenerationState continuation.

Memory plan on a 6 GB GPU (CUDA path untested on GPU here): the text encoder runs first and is moved to
CPU; the DiT (fp16, optionally int8 weights or block offload) samples; the DiT leaves the GPU; the HC-VAE
decodes in fp16 one temporal chunk x spatial tile at a time; decoded frames are converted to uint8 on CPU.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from ..compression.hc_vae import HCVAE
from ..config import hardware_info, peak_rss_mb
from ..models.dit import NivraDiT, quantize_int8_, resolution_shift, shift_t
from ..models.hd_config import HD_ARCH_VERSION, build_dit, build_hcvae, load_hd_config
from ..text.interface import TextEncoder
from .generate import GenerationState, StateError
from .sampler import Stopped
from .video_io import write_mp4

ProgressFn = Callable[[str, int, int], None]


# ===================================================================== dev/test text encoder
class StubTextEncoder:
    """Deterministic hashed-word encoder that satisfies the TextEncoder protocol. For tests and dry runs
    only: it carries no learned language knowledge."""

    def __init__(self, dim: int = 16, max_len: int = 16):
        self.dim, self.max_len = int(dim), int(max_len)
        self.id = f"stub-hash/{self.dim}x{self.max_len}"

    def _vec(self, word: str) -> torch.Tensor:
        seed = int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")
        return torch.randn(self.dim, generator=torch.Generator().manual_seed(seed))

    def encode(self, prompts: list) -> tuple:
        toks = [p.lower().split()[: self.max_len] for p in prompts]
        L = max(1, max(len(t) for t in toks))
        emb = torch.zeros(len(prompts), L, self.dim)
        mask = torch.zeros(len(prompts), L, dtype=torch.bool)
        for b, words in enumerate(toks):
            for i, w in enumerate(words):
                emb[b, i] = self._vec(w)
                mask[b, i] = True
        return emb, mask


# ===================================================================== geometry
def frames_for(seconds: float, fps: int, temporal_factor: int = 8) -> int:
    """Smallest 1 + tf*k frame count covering `seconds` at `fps` (8 s @ 30 fps -> 241)."""
    n = max(0, int(math.ceil(round(seconds * fps, 6) / temporal_factor)))
    return 1 + temporal_factor * n


def padded(n: int, multiple: int) -> int:
    return int(math.ceil(n / multiple) * multiple)


# ===================================================================== sampler
@torch.no_grad()
def sample_flow(model: NivraDiT, shape: tuple, text: Optional[torch.Tensor], text_mask: Optional[torch.Tensor],
                steps: int = 30, cfg_scale: float = 5.0, seed: int = 0, sampler: str = "ab2", shift: float = 1.0,
                known: Optional[torch.Tensor] = None, known_start: int = 0, t_offset: int = 0,
                should_stop: Optional[Callable] = None, on_step: Optional[Callable] = None,
                device="cpu") -> torch.Tensor:
    """Integrate the rectified-flow ODE from t=1 (noise) to t=0 on a shifted schedule.

    sampler: "euler" or "ab2" (variable-step 2nd-order Adams-Bashforth on the velocity: one model
    evaluation per step like Euler, second-order accurate; a UniPC-style multistep predictor).
    CFG runs the conditional and the null-text pass sequentially (lower peak memory than batching).
    known: optional [B,C,K,h,w] clean scaled latents imposed (noised along the flow path) at every step."""
    if sampler not in ("euler", "ab2"):
        raise ValueError(f"unknown sampler {sampler!r}")
    B = shape[0]
    gen = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(shape, generator=gen).to(device)
    x = noise.clone()
    ts = shift_t(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64), shift).tolist()
    K = known.shape[2] if known is not None else 0
    sl = slice(known_start, known_start + K)
    if K:
        known = known.to(device, torch.float32)
    use_cfg = cfg_scale != 1.0
    prev_v = prev_h = None
    for i in range(steps):
        if should_stop is not None and should_stop():
            raise Stopped()
        t, tn = ts[i], ts[i + 1]
        if K:
            x[:, :, sl] = (1 - t) * known + t * noise[:, :, sl]
        tt = torch.full((B,), t, device=device)
        v = model(x, tt, text, text_mask, t_offset=t_offset)
        if use_cfg:
            v_u = model(x, tt, None, None, t_offset=t_offset)
            v = v_u + cfg_scale * (v - v_u)
        h = tn - t
        if sampler == "ab2" and prev_v is not None:
            r = h / prev_h
            x = x + h * ((1 + 0.5 * r) * v - 0.5 * r * prev_v)
        else:
            x = x + h * v
        prev_v, prev_h = v, h
        if on_step:
            on_step(i + 1, steps)
    if K:
        x[:, :, sl] = known
    return x


# ===================================================================== pipeline
@dataclass
class HDResult:
    frames: np.ndarray                  # uint8 [T,H,W,3] of this segment (new frames only for continuations)
    latents: torch.Tensor               # scaled latents [C,t,h,w] of the segment
    state: GenerationState
    meta: dict = field(default_factory=dict)
    mp4: Optional[Path] = None


def _to_u8(video: torch.Tensor) -> np.ndarray:
    v = ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return v.permute(0, 2, 3, 4, 1).cpu().numpy()


class HDPipeline:
    def __init__(self, dit: NivraDiT, vae: HCVAE, text_encoder: TextEncoder, device="cpu", fp16: bool = True,
                 int8: bool = False, offload: bool = False, attn_chunk: int = 0, mlp_chunk: int = 0,
                 vae_t_chunk: int = 1, vae_tile: int = 0, vae_overlap: int = 2, vae_margin: Optional[int] = None,
                 shift_base_tokens: int = 4096, model_version_id: str = "unversioned", defaults: Optional[dict] = None):
        if not isinstance(text_encoder, TextEncoder):
            raise TypeError("text_encoder must implement nivra.text.interface.TextEncoder")
        if text_encoder.dim != dit.cfg.text_dim:
            raise ValueError(f"text encoder {text_encoder.id} has dim {text_encoder.dim}, "
                             f"generator expects {dit.cfg.text_dim}")
        if vae.cfg.latent_channels != dit.cfg.latent_channels:
            raise ValueError("HC-VAE latent channels do not match the generator")
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.half = self.cuda and fp16
        self.dit, self.vae, self.text_encoder = dit.eval(), vae.eval(), text_encoder
        self.offload = offload
        self.vae_opts = dict(t_chunk=vae_t_chunk, tile=vae_tile, overlap=vae_overlap, margin=vae_margin)
        self.shift_base_tokens = shift_base_tokens
        self.model_version_id = model_version_id
        self.defaults = dict(defaults or {})
        dit.set_memory(grad_checkpoint=False, attn_chunk=attn_chunk, mlp_chunk=mlp_chunk)
        if int8:
            quantize_int8_(dit)
        if self.half:
            dit.half()                 # untested on GPU
        if offload:
            dit.enable_offload(self.device)

    @classmethod
    def from_config(cls, cfg, text_encoder: TextEncoder, dit: Optional[NivraDiT] = None,
                    vae: Optional[HCVAE] = None, device="cpu", **overrides) -> "HDPipeline":
        cfg = load_hd_config(cfg)
        inf = dict(cfg["inference"])
        inf.update(overrides)
        dit = dit if dit is not None else build_dit(cfg)
        vae = vae if vae is not None else build_hcvae(cfg)
        keys = ("fp16", "int8", "offload", "attn_chunk", "mlp_chunk", "vae_t_chunk", "vae_tile", "vae_overlap",
                "vae_margin", "shift_base_tokens", "model_version_id")
        kw = {k: inf[k] for k in keys if k in inf}
        defaults = {k: inf[k] for k in ("width", "height", "seconds", "fps", "steps", "cfg_scale", "sampler") if k in inf}
        return cls(dit, vae, text_encoder, device=device, defaults=defaults, **kw)

    # ------------------------------------------------------------ helpers
    def _offload_text_encoder(self):
        enc = self.text_encoder
        if isinstance(enc, nn.Module):
            enc.to("cpu")
        elif hasattr(enc, "offload"):
            enc.offload()

    def _dit_to(self, dev):
        if not self.offload:
            self.dit.to(dev)

    def _encode_text(self, prompt: str):
        emb, mask = self.text_encoder.encode([prompt])
        self._offload_text_encoder()
        if emb.dim() != 3 or emb.shape[-1] != self.dit.cfg.text_dim or mask.shape != emb.shape[:2]:
            raise ValueError("text encoder returned tensors that violate the TextEncoder contract")
        return emb.float(), mask.bool()

    def _decode(self, zs: torch.Tensor, H: int, W: int, tail_frames: int, progress, stop) -> tuple:
        vae = self.vae.to(self.device, torch.float16 if self.half else torch.float32)
        z = vae.from_scaled(zs.float()).to(self.device, torch.float16 if self.half else torch.float32)
        T = z.shape[2]
        t_chunk = max(1, int(self.vae_opts["t_chunk"]))
        total = math.ceil(T / t_chunk)
        out, tail = [], []
        with torch.no_grad():
            for k, chunk in enumerate(self.vae.decode_stream(z, **self.vae_opts)):
                if stop():
                    raise Stopped()
                out.append(_to_u8(chunk[..., :H, :W])[0])
                tail.append(chunk.float().cpu())
                tail_t = sum(c.shape[2] for c in tail)
                while tail and tail_t - tail[0].shape[2] >= tail_frames:
                    tail_t -= tail.pop(0).shape[2]
                progress("decode", k + 1, total)
        tail_v = torch.cat(tail, 2)[:, :, -tail_frames:] if tail_frames else None
        if self.cuda:
            self.vae.to("cpu")
            torch.cuda.empty_cache()
        return np.concatenate(out, 0), tail_v

    def _reencode_tail(self, tail_v: torch.Tensor) -> torch.Tensor:
        """Encode the last 1+8(K-1) decoded frames so continuation latents carry the causal-VAE
        first-frame semantics (latent 0 = one frame)."""
        dt = torch.float16 if self.half else torch.float32
        vae = self.vae.to(self.device, dt)
        with torch.no_grad():
            z = vae.encode_tiled(tail_v.to(self.device, dt), t_chunk=max(1, int(self.vae_opts["t_chunk"])),
                                 tile=self.vae_opts["tile"], overlap=self.vae_opts["overlap"],
                                 margin=self.vae_opts["margin"])
        zs = vae.to_scaled(z.float()).cpu()
        if self.cuda:
            self.vae.to("cpu")
        return zs[0]

    # ------------------------------------------------------------ public API
    @torch.no_grad()
    def generate(self, prompt: str, seed: int = 0, width: Optional[int] = None, height: Optional[int] = None,
                 seconds: Optional[float] = None, fps: Optional[int] = None, frames: Optional[int] = None,
                 steps: Optional[int] = None, cfg_scale: Optional[float] = None, sampler: Optional[str] = None,
                 out_path=None, on_progress: Optional[ProgressFn] = None, should_stop: Optional[Callable] = None,
                 stop_file=None, known: Optional[torch.Tensor] = None, t_offset: int = 0,
                 continuation_frames: int = 2, segment: int = 1) -> HDResult:
        d = self.defaults
        width = int(width or d.get("width", 1280))
        height = int(height or d.get("height", 720))
        fps = int(fps or d.get("fps", 30))
        steps = int(steps or d.get("steps", 30))
        cfg_scale = float(d.get("cfg_scale", 5.0) if cfg_scale is None else cfg_scale)
        sampler = sampler or d.get("sampler", "ab2")
        vc = self.vae.cfg
        if frames is None:
            frames = frames_for(float(seconds if seconds is not None else d.get("seconds", 8)), fps, vc.temporal_factor)
        if not vc.valid_frames(frames):
            raise ValueError(f"frames must be 1 + {vc.temporal_factor}k, got {frames}")
        Hp, Wp = padded(height, vc.spatial_factor), padded(width, vc.spatial_factor)
        C, Tl, Hl, Wl = vc.latent_shape(frames, Hp, Wp)
        n_tokens = Tl * Hl * Wl
        progress = on_progress or (lambda *_: None)

        def stop():
            return bool((stop_file is not None and Path(stop_file).exists()) or (should_stop is not None and should_stop()))

        timings = {}
        t0 = time.time()
        progress("text", 0, 1)
        emb, mask = self._encode_text(prompt)
        progress("text", 1, 1)
        timings["seconds_text"] = time.time() - t0

        t0 = time.time()
        self._dit_to(self.device)
        shift = resolution_shift(n_tokens, self.shift_base_tokens)
        if known is not None and known.dim() == 4:
            known = known.unsqueeze(0)
        zs = sample_flow(self.dit, (1, C, Tl, Hl, Wl), emb.to(self.device), mask.to(self.device), steps, cfg_scale,
                         seed, sampler, shift, known=known, t_offset=t_offset, should_stop=stop,
                         on_step=lambda i, n: progress("sample", i, n), device=self.device)
        if self.cuda:
            self._dit_to("cpu")
            torch.cuda.empty_cache()
        timings["seconds_sampling"] = time.time() - t0

        t0 = time.time()
        K = max(1, min(int(continuation_frames), Tl))
        tail_frames = vc.frames_for_latent(K)
        video, tail_v = self._decode(zs.cpu(), height, width, tail_frames, progress, stop)
        recent = self._reencode_tail(tail_v)
        timings["seconds_decode"] = time.time() - t0

        state = GenerationState(prompt=prompt, seed=seed, model_version_id=self.model_version_id,
                                arch_version=HD_ARCH_VERSION, vocab_hash=self.text_encoder.id, text_ids=[],
                                recent_latents=recent.contiguous(), temporal_position=t_offset + Tl,
                                segments=segment, steps=steps, cfg_scale=cfg_scale,
                                text_embedding=emb[0][mask[0]].contiguous(), frames_so_far=None)
        meta = {"mode": "text_to_video" if known is None else "continuation", "prompt": prompt, "seed": seed,
                "steps": steps, "cfg_scale": cfg_scale, "sampler": sampler, "shift": shift, "width": width,
                "height": height, "padded": [Hp, Wp], "frames": int(video.shape[0]), "fps": fps,
                "latent_shape": [C, Tl, Hl, Wl], "tokens": n_tokens, "text_encoder": self.text_encoder.id,
                "model_version_id": self.model_version_id, "arch_version": HD_ARCH_VERSION,
                "fp16": self.half, "int8": bool(self.dit.quantized), "offload": self.offload,
                "vae": dict(self.vae_opts), "hardware": hardware_info(), "peak_rss_mb": peak_rss_mb(), **timings}
        if self.cuda:
            meta["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        mp4 = None
        if out_path is not None:
            progress("write", 0, 1)
            mp4 = write_mp4(video, out_path, fps=fps, upscale=1)
            progress("write", 1, 1)
        return HDResult(video, zs[0].cpu(), state, meta, mp4)

    @torch.no_grad()
    def continue_video(self, state: GenerationState, prompt: Optional[str] = None, seed: Optional[int] = None,
                       known_latent_frames: int = 2, **kw) -> HDResult:
        """Next segment conditioned on the previous segment's last latent frames (inpainting-style, like the
        small model). Returned frames exclude the 1+8(K-1) frames that overlap the previous segment."""
        if state.arch_version != HD_ARCH_VERSION:
            raise StateError(f"state architecture {state.arch_version} is not {HD_ARCH_VERSION}")
        if state.model_version_id != self.model_version_id:
            raise StateError(f"state was made by model {state.model_version_id}, pipeline is {self.model_version_id}")
        if state.vocab_hash != self.text_encoder.id:
            raise StateError(f"state used text encoder {state.vocab_hash}, pipeline has {self.text_encoder.id}")
        K = min(int(known_latent_frames), state.recent_latents.shape[1])
        known = state.recent_latents[:, -K:]
        if kw.get("width") is None and kw.get("height") is None:     # keep the previous segment's (padded) frame size
            sf = self.vae.cfg.spatial_factor
            kw["height"], kw["width"] = state.recent_latents.shape[2] * sf, state.recent_latents.shape[3] * sf
        kw.setdefault("steps", state.steps)
        kw.setdefault("cfg_scale", state.cfg_scale)
        res = self.generate(state.prompt if prompt is None else prompt,
                            seed=state.seed + state.segments if seed is None else seed, known=known,
                            t_offset=state.temporal_position - K, continuation_frames=known_latent_frames,
                            segment=state.segments + 1, **{k: v for k, v in kw.items() if k != "out_path"})
        overlap = self.vae.cfg.frames_for_latent(K)
        res.frames = res.frames[overlap:]
        res.meta.update(known_latent_frames=K, dropped_overlap_frames=overlap, frames=int(res.frames.shape[0]))
        if kw.get("out_path") is not None:
            res.mp4 = write_mp4(res.frames, kw["out_path"], fps=res.meta["fps"], upscale=1)
        return res
