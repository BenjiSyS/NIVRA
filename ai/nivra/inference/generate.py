"""Text-to-video, video continuation and image-to-video on top of the latent sampler."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from safetensors.torch import save_file, load_file

from ..config import hardware_info, peak_rss_mb
from ..text.tokenizer import default_tokenizer
from .bundle import Bundle
from .sampler import sample_latents
from .video_io import write_mp4

STATE_VERSION = "nivra.genstate/1"


class StateError(Exception):
    pass


@dataclass
class GenerationState:
    """Everything needed to continue a video: text conditioning, the most recent latent frames,
    the temporal position, seed and model version."""
    prompt: str
    seed: int
    model_version_id: str
    arch_version: str
    vocab_hash: str
    text_ids: list
    recent_latents: torch.Tensor          # scaled latents [C,K,H,W]
    temporal_position: int                # number of latent frames generated so far (video frames = 2x)
    segments: int = 1
    steps: int = 30
    cfg_scale: float = 3.0
    text_embedding: Optional[torch.Tensor] = None   # [L,D] encoder output for text_ids (informational / cache)
    frames_so_far: Optional[np.ndarray] = None      # uint8 [T,H,W,3] decoded frames of the whole video
    version: str = STATE_VERSION

    def save(self, directory) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        tensors = {"recent_latents": self.recent_latents.contiguous()}
        if self.text_embedding is not None:
            tensors["text_embedding"] = self.text_embedding.contiguous()
        if self.frames_so_far is not None:
            tensors["frames_so_far"] = torch.from_numpy(np.ascontiguousarray(self.frames_so_far))
        save_file(tensors, str(d / "state.safetensors"))
        meta = {"version": self.version, "prompt": self.prompt, "seed": self.seed,
                "model_version_id": self.model_version_id, "arch_version": self.arch_version,
                "vocab_hash": self.vocab_hash, "text_ids": self.text_ids, "temporal_position": self.temporal_position,
                "segments": self.segments, "steps": self.steps, "cfg_scale": self.cfg_scale}
        (d / "state.json").write_text(json.dumps(meta, indent=1))
        return d

    @staticmethod
    def load(directory, expect: Optional[Bundle] = None) -> "GenerationState":
        d = Path(directory)
        try:
            meta = json.loads((d / "state.json").read_text())
            t = load_file(str(d / "state.safetensors"))
        except Exception as e:  # noqa: BLE001
            raise StateError(f"cannot read generation state: {e}") from e
        if meta.get("version") != STATE_VERSION:
            raise StateError(f"unsupported state version {meta.get('version')}")
        st = GenerationState(prompt=meta["prompt"], seed=meta["seed"], model_version_id=meta["model_version_id"],
                             arch_version=meta["arch_version"], vocab_hash=meta["vocab_hash"],
                             text_ids=meta["text_ids"], recent_latents=t["recent_latents"],
                             temporal_position=meta["temporal_position"], segments=meta["segments"],
                             steps=meta["steps"], cfg_scale=meta["cfg_scale"],
                             text_embedding=t.get("text_embedding"),
                             frames_so_far=t["frames_so_far"].numpy() if "frames_so_far" in t else None)
        if expect is not None:
            if st.arch_version != expect.manifest["architecture_version"] or st.vocab_hash != expect.manifest["vocab_hash"]:
                raise StateError("state was produced by an incompatible architecture / tokenizer")
        return st


@dataclass
class GenResult:
    frames: np.ndarray                 # uint8 [T,H,W,3] (for continuation: the full video so far)
    new_frames: np.ndarray             # frames added by this call
    latents: torch.Tensor              # scaled latents of the last generated segment [C,T,H,W]
    state: GenerationState
    meta: dict = field(default_factory=dict)


def to_u8(video: torch.Tensor) -> np.ndarray:
    """[B,3,T,H,W] in [-1,1] -> uint8 [B,T,H,W,3]."""
    v = ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return v.permute(0, 2, 3, 4, 1).cpu().numpy()


def _latent_shape(b: Bundle, B=1):
    c = b.model.cfg
    return (B, c.latent_channels, c.latent_t, c.latent_h, c.latent_w)


def _timing_meta(b: Bundle, t_sample, t_decode, **extra) -> dict:
    m = {"model_version_id": b.version_id, "vae_sha256": b.vae_sha256, "seconds_sampling": t_sample,
         "seconds_decode": t_decode, "hardware": hardware_info(), "peak_rss_mb": peak_rss_mb()}
    m.update(extra)
    return m


@torch.no_grad()
def generate(b: Bundle, prompt: str, seed: int = 0, steps: int = 30, cfg_scale: float = 3.0, tile: int = 0,
             should_stop: Optional[Callable] = None, device: str = "cpu") -> GenResult:
    tok = default_tokenizer()
    ids = torch.tensor([tok.encode(prompt)])
    t0 = time.time()
    z = sample_latents(b.model, ids, _latent_shape(b), steps, cfg_scale, seed, should_stop=should_stop, device=device)
    t1 = time.time()
    frames = to_u8(b.vae.decode(z, tile=tile))[0]
    t2 = time.time()
    K = 2
    emb = b.model.encode_text(ids.to(device))[0][0].cpu()
    state = GenerationState(prompt=prompt, seed=seed, model_version_id=b.version_id,
                            arch_version=b.manifest["architecture_version"], vocab_hash=tok.vocab_hash,
                            text_ids=ids[0].tolist(), recent_latents=z[0, :, -K:].cpu().clone(),
                            temporal_position=z.shape[2], steps=steps, cfg_scale=cfg_scale, text_embedding=emb,
                            frames_so_far=frames)
    meta = _timing_meta(b, t1 - t0, t2 - t1, prompt=prompt, seed=seed, steps=steps, cfg_scale=cfg_scale,
                        unknown_words=tok.unknown_words(prompt), mode="text_to_video", tile=tile)
    return GenResult(frames, frames, z[0].cpu(), state, meta)


@torch.no_grad()
def continue_video(b: Bundle, state: GenerationState, prompt: Optional[str] = None, seed: Optional[int] = None,
                   steps: Optional[int] = None, cfg_scale: Optional[float] = None, known_latent_frames: int = 2,
                   tile: int = 0, should_stop: Optional[Callable] = None, device: str = "cpu") -> GenResult:
    """Generate the next segment conditioned on the previous segment's last latent frames.

    The first `known_latent_frames` latent frames of the new segment are the previous segment's last ones;
    they are re-imposed (noised along the flow path) at every Euler step, so the model has to produce a
    temporally consistent continuation (inpainting-style conditioning), not an unrelated clip."""
    if state.model_version_id != b.version_id:
        raise StateError(f"state was made by model {state.model_version_id}, bundle is {b.version_id}")
    tok = default_tokenizer()
    prompt = state.prompt if prompt is None else prompt
    seed = state.seed + state.segments if seed is None else seed
    steps = steps or state.steps
    cfg_scale = state.cfg_scale if cfg_scale is None else cfg_scale
    K = min(known_latent_frames, state.recent_latents.shape[1])
    known = state.recent_latents[:, -K:].unsqueeze(0).to(device)
    ids = torch.tensor([tok.encode(prompt)])
    t0 = time.time()
    z = sample_latents(b.model, ids, _latent_shape(b), steps, cfg_scale, seed, known=known, known_start=0,
                       should_stop=should_stop, device=device)
    t1 = time.time()
    seg = to_u8(b.vae.decode(z, tile=tile))[0]
    t2 = time.time()
    new_frames = seg[2 * K:]                       # drop frames that overlap the previous segment
    prev = state.frames_so_far if state.frames_so_far is not None else np.zeros((0,) + seg.shape[1:], np.uint8)
    frames = np.concatenate([prev, new_frames], axis=0)
    new_state = GenerationState(prompt=prompt, seed=seed, model_version_id=b.version_id,
                                arch_version=state.arch_version, vocab_hash=state.vocab_hash,
                                text_ids=ids[0].tolist(), recent_latents=z[0, :, -K:].cpu().clone(),
                                temporal_position=state.temporal_position + z.shape[2] - K,
                                segments=state.segments + 1, steps=steps, cfg_scale=cfg_scale,
                                text_embedding=b.model.encode_text(ids.to(device))[0][0].cpu(), frames_so_far=frames)
    meta = _timing_meta(b, t1 - t0, t2 - t1, prompt=prompt, seed=seed, steps=steps, cfg_scale=cfg_scale,
                        mode="continuation", known_latent_frames=K, segment=new_state.segments,
                        unknown_words=tok.unknown_words(prompt), tile=tile)
    return GenResult(frames, new_frames, z[0].cpu(), new_state, meta)


@torch.no_grad()
def image_to_video(b: Bundle, image_u8: np.ndarray, prompt: str, seed: int = 0, steps: int = 30,
                   cfg_scale: float = 3.0, tile: int = 0, device: str = "cpu") -> GenResult:
    """First-frame conditioning with the same mechanism: the image (repeated over the 2 frames of the first
    latent slot) is encoded and imposed as latent frame 0."""
    tok = default_tokenizer()
    img = torch.from_numpy(image_u8).float() / 127.5 - 1.0                      # [H,W,3]
    clip = img.permute(2, 0, 1)[:, None].repeat(1, 2, 1, 1)[None].to(device)    # [1,3,2,H,W]
    known = b.vae.to_scaled(b.vae.encode(clip))                                  # [1,C,1,h,w]
    ids = torch.tensor([tok.encode(prompt)])
    t0 = time.time()
    z = sample_latents(b.model, ids, _latent_shape(b), steps, cfg_scale, seed, known=known, known_start=0, device=device)
    t1 = time.time()
    frames = to_u8(b.vae.decode(z, tile=tile))[0]
    t2 = time.time()
    state = GenerationState(prompt=prompt, seed=seed, model_version_id=b.version_id,
                            arch_version=b.manifest["architecture_version"], vocab_hash=tok.vocab_hash,
                            text_ids=ids[0].tolist(), recent_latents=z[0, :, -2:].cpu().clone(),
                            temporal_position=z.shape[2], steps=steps, cfg_scale=cfg_scale, frames_so_far=frames)
    meta = _timing_meta(b, t1 - t0, t2 - t1, prompt=prompt, seed=seed, steps=steps, cfg_scale=cfg_scale,
                        mode="image_to_video", unknown_words=tok.unknown_words(prompt), tile=tile)
    return GenResult(frames, frames, z[0].cpu(), state, meta)


def save_result(res: GenResult, out_dir, name: str, fps: int = 8, upscale: int = 4, extra_meta: Optional[dict] = None,
                save_state: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = write_mp4(res.frames, out_dir / f"{name}.mp4", fps=fps, upscale=upscale)
    meta = dict(res.meta)
    meta.update(frames=int(res.frames.shape[0]), fps=fps, upscale=upscale, native_size=list(res.frames.shape[1:3]),
                mp4=mp4.name, temporal_position=res.state.temporal_position, segments=res.state.segments)
    if extra_meta:
        meta.update(extra_meta)
    if save_state:
        res.state.save(out_dir / f"{name}.state")
        meta["state_dir"] = f"{name}.state"
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta
