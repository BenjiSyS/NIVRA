"""Nivra HC-VAE: original high-compression causal 3D video VAE (design contract section 2).

* 32x spatial / 8x temporal compression, 128 latent channels (configurable).
* Strictly causal in time: every temporal convolution only sees the current and past frames; the
  first frame is encoded on its own, so clips have 1 + 8k frames and map to 1 + k latent frames.
* Encoder: residual causal 3D blocks + strided convolutions. Decoder: residual causal 3D blocks +
  depth-to-time and depth-to-space (pixel-shuffle) upsampling.
* Normalisation is a per-position channel RMSNorm (no spatial or temporal statistics), which makes
  chunked causal decoding *exactly* equal to a full decode and spatial tiling with a context margin
  equal to the receptive radius *exactly* equal to an untiled decode.
* Streaming: every CausalConv3d keeps a cache of its last input frames, every time-upsampler remembers
  whether it already emitted the "first frame"; per-tile stream states are swapped in and out so a
  long 720p video is decoded chunk by chunk (time) and tile by tile (space) with linear-blend overlap.

No pretrained networks anywhere (the perceptual term is an original gradient/contrast proxy).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HCVAE_VERSION = "nivra-hc-vae/1"
# v2 = the config uses a v2-only feature (per-channel latent normalisation: latent_shift / latent_scale as
# [C] vectors). v1 configs, weights and bundles load unchanged; v1 readers reject v2 files explicitly.
HCVAE_VERSION_2 = "nivra-hc-vae/2"
HCVAE_VERSIONS = (HCVAE_VERSION, HCVAE_VERSION_2)
TEMPORAL_DOWN_FOR_FACTOR = {8: (1, 2, 3), 4: (2, 3)}   # `temporal_factor` option -> temporal_down (32x spatial)


@dataclass
class HCVAEConfig:
    in_channels: int = 3
    latent_channels: int = 128
    # channels per resolution level; len = number of spatial downsamplings + 1 (6 levels -> 32x)
    channels: tuple = (64, 128, 256, 256, 512, 512)
    # which level transitions (0-based, transition i goes from level i to i+1) also halve time
    temporal_down: tuple = (1, 2, 3)
    n_res: int = 2              # residual blocks per level (encoder)
    n_res_dec: int = 2          # residual blocks per level (decoder)
    logvar_min: float = -30.0
    logvar_max: float = 20.0
    # z_scaled = (z - shift) * scale  (calibrated after training). A float ("scalar" mode, v1) or a list of
    # latent_channels floats ("channel" mode, v2: every channel gets its own mean / 1/std from encode statistics).
    latent_shift: object = 0.0
    latent_scale: object = 1.0
    latent_norm: str = "scalar"   # "scalar" | "channel"
    version: str = HCVAE_VERSION

    @property
    def spatial_factor(self) -> int:
        return 2 ** (len(self.channels) - 1)

    @property
    def temporal_factor(self) -> int:
        return 2 ** len(self.temporal_down)

    @property
    def per_channel(self) -> bool:
        return self.latent_norm == "channel"

    def validate(self) -> None:
        if self.version not in HCVAE_VERSIONS:
            raise ValueError(f"unsupported HC-VAE version {self.version!r}")
        if len(self.channels) < 2:
            raise ValueError("need at least two levels")
        n_tr = len(self.channels) - 1
        if any(not 0 <= i < n_tr for i in self.temporal_down) or len(set(self.temporal_down)) != len(self.temporal_down):
            raise ValueError(f"temporal_down must be distinct transition indices in [0,{n_tr})")
        if self.latent_norm not in ("scalar", "channel"):
            raise ValueError(f"latent_norm must be 'scalar' or 'channel', got {self.latent_norm!r}")
        vec = [isinstance(self.latent_shift, (list, tuple)), isinstance(self.latent_scale, (list, tuple))]
        if self.per_channel != all(vec) or (not self.per_channel and any(vec)):
            raise ValueError("latent_norm 'channel' needs latent_shift and latent_scale lists of latent_channels "
                             "floats; 'scalar' needs plain numbers")
        for name in ("latent_shift", "latent_scale"):
            v = getattr(self, name)
            vals = list(v) if isinstance(v, (list, tuple)) else [v]
            if isinstance(v, (list, tuple)) and len(vals) != self.latent_channels:
                raise ValueError(f"{name} has {len(vals)} entries, latent_channels is {self.latent_channels}")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vals):
                raise ValueError(f"{name} must be finite numbers")
        if self.per_channel and self.version == HCVAE_VERSION:
            raise ValueError("per-channel latent normalisation needs HC-VAE version nivra-hc-vae/2")
        if any(x <= 0 for x in (self.latent_scale if isinstance(self.latent_scale, (list, tuple)) else [self.latent_scale])):
            raise ValueError("latent_scale must be > 0")

    @staticmethod
    def from_dict(d: dict) -> "HCVAEConfig":
        known = set(HCVAEConfig.__dataclass_fields__) | {"temporal_factor"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown HC-VAE config keys {sorted(unknown)}")
        d = dict(d)
        if "temporal_factor" in d:
            tf = d.pop("temporal_factor")
            if isinstance(tf, bool) or tf not in TEMPORAL_DOWN_FOR_FACTOR:
                raise ValueError(f"temporal_factor must be 4 or 8, got {tf!r}")
            td = TEMPORAL_DOWN_FOR_FACTOR[tf]
            if "temporal_down" in d and tuple(d["temporal_down"]) != td:
                raise ValueError(f"temporal_factor {tf} contradicts temporal_down {list(d['temporal_down'])} "
                                 f"(expected {list(td)})")
            if len(d.get("channels", HCVAEConfig.channels)) != 6:
                raise ValueError("temporal_factor needs the standard 32x spatial layout (6 channel levels); "
                                 "give temporal_down explicitly for other layouts")
            d["temporal_down"] = td
        for k in ("channels", "temporal_down"):
            if k in d:
                d[k] = tuple(d[k])
        for k in ("latent_shift", "latent_scale"):
            if isinstance(d.get(k), (list, tuple)):
                d[k] = [float(x) for x in d[k]]
        if d.get("latent_norm") == "channel" and "version" not in d:
            d["version"] = HCVAE_VERSION_2
        c = HCVAEConfig(**d)
        c.validate()
        return c

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        d["temporal_down"] = list(self.temporal_down)
        if self.latent_norm == "scalar":
            d.pop("latent_norm")            # v1 configs serialise exactly as before (stable hd_config_sha)
        else:
            d["latent_shift"], d["latent_scale"] = list(self.latent_shift), list(self.latent_scale)
        return d

    # ---------------------------------------------------------------- geometry
    def valid_frames(self, frames: int) -> bool:
        return frames >= 1 and (frames - 1) % self.temporal_factor == 0

    def latent_shape(self, frames: int, height: int, width: int) -> tuple:
        s, tf = self.spatial_factor, self.temporal_factor
        if not self.valid_frames(frames):
            raise ValueError(f"frames must be 1 + {tf}k, got {frames}")
        if height % s or width % s:
            raise ValueError(f"height/width must be multiples of {s}, got {height}x{width}")
        return (self.latent_channels, 1 + (frames - 1) // tf, height // s, width // s)

    def frames_for_latent(self, latent_t: int) -> int:
        return 1 + (latent_t - 1) * self.temporal_factor


# ===================================================================== building blocks
class ChannelRMSNorm(nn.Module):
    """RMS normalisation over channels at every (t, h, w) position independently."""

    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(1, keepdim=True) + self.eps)
        return (y * self.weight.float().view(1, -1, 1, 1, 1)).to(x.dtype)


class CausalConv3d(nn.Module):
    """3D conv that is causal in time (left replicate-padding of the first frame) with a streaming cache.

    In streaming mode the last (kt - stride_t) frames of the (padded) input are kept and prepended to the
    next chunk, so processing a video chunk by chunk gives exactly the same output as one full pass."""

    def __init__(self, cin: int, cout: int, k=(3, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        kt, kh, kw = k
        self.kt, self.st = kt, stride[0]
        self.conv = nn.Conv3d(cin, cout, k, stride=stride, padding=(0, kh // 2, kw // 2))
        self.stream = False
        self.cache: Optional[torch.Tensor] = None

    @property
    def stateful(self) -> bool:
        return not (self.kt == 1 and self.st == 1)

    def forward(self, x):
        if not self.stateful:
            return self.conv(x)
        if self.stream and self.cache is not None:
            x = torch.cat([self.cache.to(x.dtype), x], 2)
        elif self.kt > 1:
            x = torch.cat([x[:, :, :1].expand(-1, -1, self.kt - 1, -1, -1), x], 2)
        if self.stream:
            if (x.shape[2] - self.kt) % self.st:
                raise ValueError("streaming chunk length is not aligned to the temporal stride "
                                 "(encoder chunks must be 1+f*a frames first, then multiples of f, f = the temporal factor)")
            keep = self.kt - self.st
            # clone: a slice is a view that keeps the whole (cache + chunk) input alive, i.e. up to
            # (kt-1+chunk)/(kt-1) times the memory the cache needs, for every layer of every tile
            self.cache = x[:, :, x.shape[2] - keep:].detach().clone() if keep > 0 else x[:, :, :0].detach().clone()
        return self.conv(x)

    def get_state(self):
        return self.cache

    def set_state(self, s):
        self.cache = s


class ResBlock3d(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.n1 = ChannelRMSNorm(cin)
        self.c1 = CausalConv3d(cin, cout)
        self.n2 = ChannelRMSNorm(cout)
        self.c2 = CausalConv3d(cout, cout)
        self.skip = CausalConv3d(cin, cout, (1, 1, 1)) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class Downsample(nn.Module):
    """Strided spatial conv (2x) per frame, optionally followed by a strided causal temporal conv (2x)."""

    def __init__(self, cin: int, cout: int, temporal: bool):
        super().__init__()
        self.space = CausalConv3d(cin, cout, (1, 3, 3), (1, 2, 2))
        self.time = CausalConv3d(cout, cout, (3, 1, 1), (2, 1, 1)) if temporal else None

    def forward(self, x):
        x = self.space(x)
        return self.time(x) if self.time is not None else x


class TimeUp(nn.Module):
    """Depth-to-time 2x upsampling. The very first latent frame stands for one video frame only, so its
    first output frame is dropped (1 + k -> 1 + 2k). Streaming-aware."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = CausalConv3d(c, 2 * c, (3, 1, 1))
        self.stream = False
        self.first_done = False

    def forward(self, x):
        B, C, T, H, W = x.shape
        y = self.conv(x).view(B, 2, C, T, H, W).permute(0, 2, 3, 1, 4, 5).reshape(B, C, 2 * T, H, W)
        drop = not (self.stream and self.first_done)
        if self.stream:
            self.first_done = True
        return y[:, :, 1:] if drop else y

    def get_state(self):
        return self.first_done

    def set_state(self, s):
        self.first_done = bool(s)


class SpaceUp(nn.Module):
    """Depth-to-space (pixel-shuffle) 2x spatial upsampling, per frame."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = CausalConv3d(cin, 4 * cout, (1, 3, 3))
        self.cout = cout

    def forward(self, x):
        y = self.conv(x)
        B, _, T, H, W = y.shape
        y = y.view(B, self.cout, 2, 2, T, H, W).permute(0, 1, 4, 5, 2, 6, 3)
        return y.reshape(B, self.cout, T, 2 * H, 2 * W)


class _Streamable(nn.Module):
    def _stateful(self) -> list:
        return [m for m in self.modules()
                if isinstance(m, TimeUp) or (isinstance(m, CausalConv3d) and m.stateful)]

    def set_streaming(self, on: bool) -> None:
        for m in self._stateful():
            m.stream = on
            m.set_state(None if isinstance(m, CausalConv3d) else False)

    def get_stream_state(self) -> list:
        return [m.get_state() for m in self._stateful()]

    def set_stream_state(self, state: Optional[list]) -> None:
        mods = self._stateful()
        if state is None:
            state = [None if isinstance(m, CausalConv3d) else False for m in mods]
        for m, s in zip(mods, state):
            m.set_state(s)


class HCEncoder(_Streamable):
    def __init__(self, cfg: HCVAEConfig):
        super().__init__()
        ch = cfg.channels
        self.conv_in = CausalConv3d(cfg.in_channels, ch[0])
        radius_px = 1.0
        self.levels = nn.ModuleList()
        for i, c in enumerate(ch):
            blocks = nn.ModuleList([ResBlock3d(c, c) for _ in range(cfg.n_res)])
            radius_px += 2 * cfg.n_res * 2 ** i
            down = None
            if i < len(ch) - 1:
                down = Downsample(c, ch[i + 1], temporal=i in cfg.temporal_down)
                radius_px += 2 ** i
            self.levels.append(nn.ModuleDict({"blocks": blocks, **({"down": down} if down is not None else {})}))
        self.norm_out = ChannelRMSNorm(ch[-1])
        self.conv_out = CausalConv3d(ch[-1], 2 * cfg.latent_channels)
        radius_px += 2 ** (len(ch) - 1)
        # receptive radius in latent cells (used as default tiling margin for exact tiles)
        self.radius = math.ceil(radius_px / 2 ** (len(ch) - 1))

    def forward(self, x):
        h = self.conv_in(x)
        for lvl in self.levels:
            for b in lvl["blocks"]:
                h = b(h)
            if "down" in lvl:
                h = lvl["down"](h)
        return self.conv_out(F.silu(self.norm_out(h)))


class HCDecoder(_Streamable):
    def __init__(self, cfg: HCVAEConfig):
        super().__init__()
        ch = list(reversed(cfg.channels))
        n_tr = len(ch) - 1
        self.conv_in = CausalConv3d(cfg.latent_channels, ch[0])
        radius = 1.0                      # in latent cells
        self.levels = nn.ModuleList()
        for i, c in enumerate(ch):
            scale = 2 ** i                # pixels of this level per latent cell
            blocks = nn.ModuleList([ResBlock3d(c, c) for _ in range(cfg.n_res_dec)])
            radius += 2 * cfg.n_res_dec / scale
            mods = {"blocks": blocks}
            if i < n_tr:
                # decoder transition i mirrors encoder transition (n_tr - 1 - i)
                if (n_tr - 1 - i) in cfg.temporal_down:
                    mods["tup"] = TimeUp(c)
                mods["sup"] = SpaceUp(c, ch[i + 1])
                radius += 1 / scale
            self.levels.append(nn.ModuleDict(mods))
        self.norm_out = ChannelRMSNorm(ch[-1])
        self.conv_out = CausalConv3d(ch[-1], cfg.in_channels)
        radius += 1 / 2 ** n_tr
        self.radius = math.ceil(radius)

    def forward(self, z):
        h = self.conv_in(z)
        for lvl in self.levels:
            for b in lvl["blocks"]:
                h = b(h)
            if "tup" in lvl:
                h = lvl["tup"](h)
            if "sup" in lvl:
                h = lvl["sup"](h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ===================================================================== tiling helpers
def tile_ranges(n: int, tile: int, overlap: int) -> list:
    """Covering ranges [a, b) of length `tile` with at least `overlap` overlap; one range if tile >= n."""
    if tile <= 0 or tile >= n:
        return [(0, n)]
    if not 0 <= overlap < tile:
        raise ValueError("need 0 <= overlap < tile")
    stride = tile - overlap
    out, a = [], 0
    while True:
        if a + tile >= n:
            out.append((n - tile, n))
            break
        out.append((a, a + tile))
        a += stride
    return sorted(set(out))


def _ramp(n: int, ramp: int, left: bool, right: bool, device, dtype) -> torch.Tensor:
    w = torch.ones(n, device=device, dtype=dtype)
    r = min(ramp, n)
    if r > 0:
        up = (torch.arange(r, device=device, dtype=dtype) + 0.5) / r
        if left:
            w[:r] = torch.minimum(w[:r], up)
        if right:
            w[n - r:] = torch.minimum(w[n - r:], up.flip(0))
    return w


def available_memory_bytes(device=None) -> Optional[int]:
    """Free memory of `device` (CUDA free memory, else available host RAM); None when it cannot be determined."""
    try:
        if device is not None and torch.device(device).type == "cuda":
            return int(torch.cuda.mem_get_info(torch.device(device))[0])
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(min(ms.ullAvailPhys, ms.ullAvailPageFile))
            return None
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError, RuntimeError):
        return None


class StreamStateTooLarge(ValueError):
    """Tiled streaming would keep more per-tile causal-cache state than fits in memory."""


# ===================================================================== the VAE
class HCVAE(nn.Module):
    def __init__(self, cfg: HCVAEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.encoder = HCEncoder(cfg)
        self.decoder = HCDecoder(cfg)

    # ------------------------------------------------------------ full passes
    def encode_dist(self, x: torch.Tensor) -> tuple:
        """x: [B,3,T,H,W] in [-1,1] with T = 1 + 8k. Returns (mean, logvar) [B,C,1+k,H/32,W/32]."""
        self._check_input(x)
        m, lv = self.encoder(x).chunk(2, dim=1)
        return m, lv.clamp(self.cfg.logvar_min, self.cfg.logvar_max)

    def encode(self, x: torch.Tensor, sample: bool = False, generator=None) -> torch.Tensor:
        m, lv = self.encode_dist(x)
        if not sample:
            return m
        eps = torch.randn(m.shape, generator=generator, device=m.device, dtype=m.dtype)
        return m + torch.exp(0.5 * lv) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, generator=None) -> tuple:
        m, lv = self.encode_dist(x)
        z = m + torch.exp(0.5 * lv) * torch.randn(m.shape, generator=generator, device=m.device, dtype=m.dtype)
        return self.decode(z), m, lv

    def _check_input(self, x):
        _, c, t, h, w = x.shape
        self.cfg.latent_shape(t, h, w)   # raises on bad geometry
        if c != self.cfg.in_channels:
            raise ValueError(f"expected {self.cfg.in_channels} channels, got {c}")

    # ------------------------------------------------------------ scaling
    def _norm(self, z: torch.Tensor):
        """(shift, scale) broadcastable against z [B,C,...]: Python floats in scalar mode, [1,C,1,..] tensors in
        per-channel mode (cast to z's dtype / device)."""
        sh, sc = self.cfg.latent_shift, self.cfg.latent_scale
        if self.cfg.per_channel:
            shape = (1, -1) + (1,) * (z.dim() - 2)
            sh = torch.tensor(sh, dtype=z.dtype, device=z.device).view(shape)
            sc = torch.tensor(sc, dtype=z.dtype, device=z.device).view(shape)
        return sh, sc

    def to_scaled(self, z):
        sh, sc = self._norm(z)
        return (z - sh) * sc

    def from_scaled(self, zs):
        sh, sc = self._norm(zs)
        return zs / sc + sh

    @torch.no_grad()
    def calibrate(self, latents: torch.Tensor, per_channel: Optional[bool] = None) -> dict:
        """Set latent shift/scale from unscaled latents [N,C,T,H,W] so scaled latents have mean 0, std 1: one
        scalar pair, or (per_channel=True, or the config's latent_norm == 'channel') one pair per latent channel
        computed from the encode statistics of every channel (Wan-style [C] mean / std)."""
        z = latents.float()
        per = self.cfg.per_channel if per_channel is None else bool(per_channel)
        if per:
            if z.dim() < 3 or z.shape[1] != self.cfg.latent_channels:
                raise ValueError(f"per-channel calibration needs latents [N,{self.cfg.latent_channels},...], "
                                 f"got {tuple(z.shape)}")
            dims = [0] + list(range(2, z.dim()))
            mean = z.mean(dim=dims)
            std = z.std(dim=dims) if z.numel() // z.shape[1] > 1 else torch.ones_like(mean)
            self.cfg.latent_norm = "channel"
            if self.cfg.version == HCVAE_VERSION:
                self.cfg.version = HCVAE_VERSION_2
            self.cfg.latent_shift = [float(x) for x in mean]
            self.cfg.latent_scale = [float(1.0 / max(float(x), 1e-6)) for x in std]
        else:
            self.cfg.latent_norm = "scalar"
            self.cfg.latent_shift = float(z.mean())
            self.cfg.latent_scale = float(1.0 / max(z.std().item(), 1e-6))
        self.cfg.validate()
        return {"latent_shift": self.cfg.latent_shift, "latent_scale": self.cfg.latent_scale,
                "latent_norm": self.cfg.latent_norm}

    def last_layer_weight(self) -> torch.nn.Parameter:
        """Weight of the decoder's output convolution (the layer the adaptive GAN weight is computed at)."""
        return self.decoder.conv_out.conv.weight

    # ------------------------------------------------------------ chunked / tiled passes
    def stream_state_bytes_per_tile(self, which: str, win_h: int, win_w: int, dtype=torch.float32, batch: int = 1) -> int:
        """Bytes of causal-cache state ONE tile keeps between chunks, for a tile window of win_h x win_w latent
        cells (`which`: 'decoder' | 'encoder'). Dry run on the meta device: no memory is allocated."""
        with torch.device("meta"):
            net = HCDecoder(self.cfg) if which == "decoder" else HCEncoder(self.cfg)
            if which == "decoder":
                x = torch.empty(batch, self.cfg.latent_channels, 1, win_h, win_w, dtype=dtype)
            else:
                s = self.cfg.spatial_factor
                x = torch.empty(batch, self.cfg.in_channels, 1, win_h * s, win_w * s, dtype=dtype)
            net.to(dtype).set_streaming(True)
            net(x)
            return sum(c.numel() * c.element_size() for c in net.get_stream_state() if isinstance(c, torch.Tensor))

    def _check_state_budget(self, x: torch.Tensor, n_tiles: int, win: tuple, which: str,
                           max_state_bytes: Optional[int]) -> None:
        """Raise StreamStateTooLarge up front instead of letting the allocator die mid-decode (the old failure
        mode was a native crash, exit 139 / access violation, once the per-tile caches exhausted memory)."""
        per_tile = self.stream_state_bytes_per_tile(which, win[0], win[1], x.dtype, x.shape[0])
        need = per_tile * n_tiles
        budget = max_state_bytes
        if budget is None:
            free = available_memory_bytes(x.device)
            budget = int(free * 0.6) if free else 8 << 30
        if need > budget:
            raise StreamStateTooLarge(
                f"tiled streaming needs ~{need / 2 ** 30:.1f} GiB of per-tile cache state ({n_tiles} tiles x "
                f"{per_tile / 2 ** 20:.0f} MiB) but only {budget / 2 ** 30:.1f} GiB is allowed; use a larger tile "
                f"(fewer tiles), a smaller output, or pass max_state_bytes")

    def _stream_tiled(self, net: _Streamable, x: torch.Tensor, chunks: list, in_scale: int, out_scale: int,
                      tile: int, overlap: int, margin: Optional[int],
                      max_state_bytes: Optional[int] = None) -> Iterator[torch.Tensor]:
        """Run `net` causally chunk by chunk (time) and tile by tile (space, in latent-cell units)."""
        B, _, _, H, W = x.shape
        Hg, Wg = H // in_scale, W // in_scale
        m = net.radius if margin is None else margin
        th, tw = tile_ranges(Hg, tile, overlap), tile_ranges(Wg, tile, overlap)
        tiles = [(a, b, c, d) for (a, b) in th for (c, d) in tw]
        if len(tiles) > 1:
            win = (max(min(Hg, b + m) - max(0, a - m) for a, b in th), max(min(Wg, d + m) - max(0, c - m) for c, d in tw))
            self._check_state_budget(x, len(tiles), win, "decoder" if net is self.decoder else "encoder", max_state_bytes)
        states = [None] * len(tiles)
        net.set_streaming(True)
        try:
            for sl in chunks:
                xc = x[:, :, sl]
                out = wsum = None
                for k, (ha, hb, wa, wb) in enumerate(tiles):
                    ha0, hb0 = max(0, ha - m), min(Hg, hb + m)
                    wa0, wb0 = max(0, wa - m), min(Wg, wb + m)
                    net.set_stream_state(states[k])
                    y = net(xc[..., ha0 * in_scale:hb0 * in_scale, wa0 * in_scale:wb0 * in_scale])
                    states[k] = net.get_stream_state()
                    y = y[..., (ha - ha0) * out_scale:(hb - ha0) * out_scale, (wa - wa0) * out_scale:(wb - wa0) * out_scale]
                    if len(tiles) == 1:
                        out, wsum = y, None
                        break
                    if out is None:
                        out = y.new_zeros(y.shape[:3] + (Hg * out_scale, Wg * out_scale), dtype=torch.float32)
                        wsum = y.new_zeros((Hg * out_scale, Wg * out_scale), dtype=torch.float32)
                    ramp = overlap * out_scale
                    wh = _ramp(y.shape[-2], ramp, ha > 0, hb < Hg, y.device, torch.float32)
                    ww = _ramp(y.shape[-1], ramp, wa > 0, wb < Wg, y.device, torch.float32)
                    w2 = wh[:, None] * ww[None, :]
                    out[..., ha * out_scale:hb * out_scale, wa * out_scale:wb * out_scale] += y.float() * w2
                    wsum[ha * out_scale:hb * out_scale, wa * out_scale:wb * out_scale] += w2
                yield out if wsum is None else (out / wsum).to(y.dtype)
        finally:
            net.set_streaming(False)

    def decode_stream(self, z: torch.Tensor, t_chunk: int = 2, tile: int = 0, overlap: int = 2,
                      margin: Optional[int] = None, max_state_bytes: Optional[int] = None) -> Iterator[torch.Tensor]:
        """Yield decoded frame chunks [B,3,t,H,W]: first chunk 1 + 8(t_chunk-1) frames, then 8*t_chunk frames.
        tile/overlap/margin are in latent cells (tile=0: no spatial tiling). margin=None uses the decoder's
        receptive radius, which makes tiles exactly equal to an untiled decode."""
        T = z.shape[2]
        t_chunk = max(1, int(t_chunk))
        chunks = [slice(i, min(T, i + t_chunk)) for i in range(0, T, t_chunk)]
        yield from self._stream_tiled(self.decoder, z, chunks, 1, self.cfg.spatial_factor, tile, overlap, margin,
                                      max_state_bytes)

    def decode_tiled(self, z: torch.Tensor, t_chunk: int = 2, tile: int = 0, overlap: int = 2,
                     margin: Optional[int] = None, max_state_bytes: Optional[int] = None) -> torch.Tensor:
        return torch.cat(list(self.decode_stream(z, t_chunk, tile, overlap, margin, max_state_bytes)), 2)

    def encode_tiled(self, x: torch.Tensor, t_chunk: int = 2, tile: int = 0, overlap: int = 2,
                     margin: Optional[int] = None, max_state_bytes: Optional[int] = None) -> torch.Tensor:
        """Chunked causal (and optionally tiled) encode; returns the posterior mean."""
        self._check_input(x)
        T, tf = x.shape[2], self.cfg.temporal_factor
        t_chunk = max(1, int(t_chunk))
        first = 1 + tf * (t_chunk - 1)
        chunks = [slice(0, min(T, first))]
        i = first
        while i < T:
            chunks.append(slice(i, min(T, i + tf * t_chunk)))
            i += tf * t_chunk
        outs = list(self._stream_tiled(self.encoder, x, chunks, self.cfg.spatial_factor, 1, tile, overlap, margin,
                                       max_state_bytes))
        h = torch.cat(outs, 2)
        return h[:, : self.cfg.latent_channels]

    # ------------------------------------------------------------ metadata
    def manifest(self) -> dict:
        c = self.cfg
        return {"version": c.version, "config": c.to_dict(), "spatial_factor": c.spatial_factor,
                "temporal_factor": c.temporal_factor, "latent_channels": c.latent_channels, "causal": True,
                "frame_rule": f"1+{c.temporal_factor}k", "pad_multiple": c.spatial_factor,
                "latent_shift": c.latent_shift, "latent_scale": c.latent_scale, "latent_norm": c.latent_norm,
                "params": sum(p.numel() for p in self.parameters()),
                "decoder_radius_latent": self.decoder.radius, "encoder_radius_latent": self.encoder.radius}


# ===================================================================== losses
_SOBEL = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) / 4.0


def _frames2d(x):
    B, C, T, H, W = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


def perceptual_proxy(recon: torch.Tensor, target: torch.Tensor, scales: tuple = (1, 2, 4)) -> torch.Tensor:
    """Original LPIPS-free structural proxy: multi-scale L1 on Sobel gradients and on local contrast
    (3x3 local standard deviation), computed per frame. No pretrained network."""
    a, b = _frames2d(recon.float()), _frames2d(target.float())
    C = a.shape[1]
    kx = _SOBEL.to(a.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ky = kx.transpose(2, 3)
    loss = a.new_zeros(())
    for s in scales:
        if s > 1:
            if min(a.shape[-2:]) < 3 * s:
                break
            a_s, b_s = F.avg_pool2d(a, s), F.avg_pool2d(b, s)
        else:
            a_s, b_s = a, b
        for k in (kx, ky):
            loss = loss + (F.conv2d(a_s, k, padding=1, groups=C) - F.conv2d(b_s, k, padding=1, groups=C)).abs().mean()

        def lstd(v):
            mu = F.avg_pool2d(v, 3, 1, 1, count_include_pad=False)
            return (F.avg_pool2d(v * v, 3, 1, 1, count_include_pad=False) - mu * mu).clamp_min(0).add(1e-6).sqrt()
        loss = loss + (lstd(a_s) - lstd(b_s)).abs().mean()
    return loss


def edge_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """L1 between the spatial finite differences (horizontal and vertical) of x and recon, [B,C,T,H,W].
    Penalises blurred outlines: crisp line art / cel edges. Zero iff both gradient fields agree."""
    x, recon = x.float(), recon.float()
    dx = ((recon[..., :, 1:] - recon[..., :, :-1]) - (x[..., :, 1:] - x[..., :, :-1])).abs().mean()
    dy = ((recon[..., 1:, :] - recon[..., :-1, :]) - (x[..., 1:, :] - x[..., :-1, :])).abs().mean()
    return 0.5 * (dx + dy)


def hcvae_loss(x: torch.Tensor, recon: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor,
               kl_weight: float = 1e-6, perc_weight: float = 0.5, tdiff_weight: float = 0.5,
               edge_weight: float = 0.0) -> dict:
    rec = (recon.float() - x.float()).abs().mean()
    kl = 0.5 * (mean.float().pow(2) + logvar.float().exp() - 1.0 - logvar.float()).mean()
    if x.shape[2] > 1:
        td = ((recon[:, :, 1:] - recon[:, :, :-1]).float() - (x[:, :, 1:] - x[:, :, :-1]).float()).abs().mean()
    else:
        td = rec.new_zeros(())
    perc = perceptual_proxy(recon, x)
    total = rec + kl_weight * kl + perc_weight * perc + tdiff_weight * td
    out = {"l1": rec.detach(), "kl": kl.detach(), "perc": perc.detach(), "tdiff": td.detach()}
    if edge_weight:
        ed = edge_loss(x, recon)
        total = total + edge_weight * ed
        out["edge"] = ed.detach()
    return {"loss": total, **out}


# ===================================================================== adversarial (patch-GAN) loss
class PatchDiscriminator(nn.Module):
    """Small 3D patch discriminator (a few strided (3,4,4) convolutions + GroupNorm(1), LeakyReLU) returning a
    grid of real/fake logits [B,1,T,h',w']. Images are T=1 clips (the temporal kernel just sees zero padding), so
    one discriminator serves the joint image + video batches of the staged VAE recipe. Trained from scratch: no
    pretrained network, no licence concern."""

    def __init__(self, in_channels: int = 3, base: int = 32, n_layers: int = 3):
        super().__init__()
        if n_layers < 1 or base < 1:
            raise ValueError("n_layers and base must be >= 1")
        layers = [nn.Conv3d(in_channels, base, (3, 4, 4), (1, 2, 2), (1, 1, 1)), nn.LeakyReLU(0.2)]
        c = base
        for i in range(1, n_layers):
            c2 = min(base * 2 ** i, 8 * base)
            layers += [nn.Conv3d(c, c2, (3, 4, 4), (1, 2, 2), (1, 1, 1)), nn.GroupNorm(1, c2), nn.LeakyReLU(0.2)]
            c = c2
        layers += [nn.Conv3d(c, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - real_logits.float()).mean() + F.relu(1.0 + fake_logits.float()).mean())


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.float().mean()


def adaptive_gan_weight(rec_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.nn.Parameter,
                        loss_scale: float = 1.0, max_weight: float = 1e4) -> torch.Tensor:
    """VQGAN / LTX-style adaptive weight  ||dL_rec/dW|| / (||dL_gan/dW|| + 1e-4)  at the decoder's last layer,
    clamped to [0, max_weight] and detached. Both losses are multiplied by the same `loss_scale` (the AMP
    GradScaler scale) so fp16 gradients do not underflow; the ratio does not depend on it. Returns 0 when either
    gradient is not finite, so a bad step never injects NaN into the objective."""
    g_rec = torch.autograd.grad(rec_loss * loss_scale, last_layer, retain_graph=True, allow_unused=True)[0]
    g_gan = torch.autograd.grad(g_loss * loss_scale, last_layer, retain_graph=True, allow_unused=True)[0]
    if g_rec is None or g_gan is None:
        return rec_loss.new_zeros(()).detach()
    w = (g_rec.float().norm() / loss_scale) / (g_gan.float().norm() / loss_scale + 1e-4)
    if not bool(torch.isfinite(w)):
        return rec_loss.new_zeros(()).detach()
    return w.clamp(0.0, max_weight).detach()


# ===================================================================== save / load (safetensors + JSON)
def save_hcvae(vae: HCVAE, directory) -> dict:
    from safetensors.torch import save_file
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "hc_vae.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in vae.state_dict().items()}, str(p),
              metadata={"kind": "nivra-hcvae", "config": json.dumps(vae.cfg.to_dict(), sort_keys=True)})
    man = vae.manifest()
    man["weights_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (d / "hc_vae.json").write_text(json.dumps(man, indent=1))
    return man


def load_hcvae(directory, map_location="cpu") -> HCVAE:
    from safetensors.torch import load_file
    d = Path(directory)
    man = json.loads((d / "hc_vae.json").read_text())
    if man.get("version") not in HCVAE_VERSIONS:
        raise ValueError(f"unsupported HC-VAE version {man.get('version')!r}")
    p = d / "hc_vae.safetensors"
    if hashlib.sha256(p.read_bytes()).hexdigest() != man.get("weights_sha256"):
        raise ValueError("HC-VAE weights checksum mismatch")
    vae = HCVAE(HCVAEConfig.from_dict(man["config"]))
    vae.load_state_dict(load_file(str(p), device=str(map_location)))
    return vae


@torch.no_grad()
def shape_trace(cfg: HCVAEConfig, frames: int, height: int, width: int) -> dict:
    """Dry run on the meta device (no memory): latent and decoded shapes + FLOPs of encode and decode."""
    from torch.utils.flop_counter import FlopCounterMode
    with torch.device("meta"):
        vae = HCVAE(cfg)
        x = torch.empty(1, cfg.in_channels, frames, height, width)
    fc = FlopCounterMode(display=False)
    with fc:
        z = vae.encode(x)
    enc_flops = fc.get_total_flops()
    fc = FlopCounterMode(display=False)
    with fc:
        y = vae.decode(z)
    return {"latent": tuple(z.shape), "decoded": tuple(y.shape), "encode_flops": enc_flops,
            "decode_flops": fc.get_total_flops(), "params": sum(p.numel() for p in vae.parameters()),
            "encoder_params": sum(p.numel() for p in vae.encoder.parameters()),
            "decoder_params": sum(p.numel() for p in vae.decoder.parameters())}
