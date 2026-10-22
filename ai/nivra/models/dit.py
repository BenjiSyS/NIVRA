"""Nivra DiT: original rectified-flow video diffusion transformer (design contract section 4).

* Full spatio-temporal self-attention over latent tokens via ``F.scaled_dot_product_attention`` (on
  Turing it dispatches to the memory-efficient kernel; no FlashAttention-2 and no bf16 needed).
* 3D RoPE (variable T/H/W), QK RMSNorm (fp16 stability), cross-attention to text tokens through a
  learned projection from any encoder width, learned null context for caption dropout / CFG.
* adaLN-Zero conditioning in the "shared" form: one timestep MLP produces the 6 modulation vectors,
  every block adds its own learned (zero-initialised) table; gates start at zero so every block starts
  as the identity.
* Memory features: activation checkpointing, query-chunked attention, token-chunked MLP, sequential CPU
  offload of transformer blocks, int8 weight-only quantisation (pure PyTorch), fp16 autocast on CUDA.

* fp32 islands (fp16 on Turing has no bf16 fallback): the timestep / fps embedding MLPs, the adaLN modulation
  projection, every LayerNorm / RMSNorm statistic, the modulation arithmetic, optionally the attention logits
  (``set_memory(attn_fp32=True)``) and the loss are computed in float32 even under fp16 autocast / a half model.
  In float32 these paths are bit-identical to the original implementation.
* Per-frame timesteps (diffusion forcing): ``t`` may be ``[B]`` or ``[B, T_grid]``; with ``cond_mask`` (v2 models,
  ``frame_cond``) masked latent frames are clean conditioning frames: their timestep is forced to 0 and a learned
  flag embedding is added to their tokens. ``fps`` (v2 models, ``fps_cond``) conditions on the effective drawing rate.
* ``feats`` returns hidden states of chosen blocks for self-representation alignment (SRA) in the trainer.

GPU-specific paths (CUDA autocast, pinned-memory/stream prefetch offload) are untested on GPU: this
repository's development machine has no CUDA device.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .rope import apply_rope, rope_3d, rope_axes

DIT_VERSION = "nivra-dit/1"
# v2 = the model has the optional frame-conditioning flag embedding (`frame_cond`) and/or the effective-fps
# embedding (`fps_cond`). v1 configs / weights load and run unchanged.
DIT_VERSION_2 = "nivra-dit/2"
DIT_VERSIONS = (DIT_VERSION, DIT_VERSION_2)


@dataclass
class DiTConfig:
    dim: int = 1536
    depth: int = 24
    heads: int = 24
    mlp_ratio: float = 4.0
    latent_channels: int = 128
    patch: tuple = (1, 1, 1)
    text_dim: int = 768          # width of the text encoder this generator is trained with
    qk_norm: bool = True
    rope_theta: float = 10000.0
    freq_dim: int = 256
    frame_cond: bool = False     # v2: clean-frame conditioning (cond_mask) with a learned flag embedding
    fps_cond: bool = False       # v2: condition on the clip's effective (drawing-rate) fps
    version: str = DIT_VERSION

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def mlp_hidden(self) -> int:
        return int(round(self.dim * self.mlp_ratio))

    def validate(self) -> None:
        if self.version not in DIT_VERSIONS:
            raise ValueError(f"unsupported DiT version {self.version!r}")
        for k in ("frame_cond", "fps_cond"):
            if not isinstance(getattr(self, k), bool):
                raise ValueError(f"{k} must be a boolean")
        if (self.frame_cond or self.fps_cond) and self.version == DIT_VERSION:
            raise ValueError("frame_cond / fps_cond need DiT version nivra-dit/2")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        rope_axes(self.head_dim)

    @staticmethod
    def from_dict(d: dict) -> "DiTConfig":
        unknown = set(d) - set(DiTConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown DiT config keys {sorted(unknown)}")
        d = dict(d)
        if "patch" in d:
            d["patch"] = tuple(d["patch"])
        if (d.get("frame_cond") or d.get("fps_cond")) and "version" not in d:
            d["version"] = DIT_VERSION_2
        c = DiTConfig(**d)
        c.validate()
        return c

    def to_dict(self) -> dict:
        d = asdict(self)
        d["patch"] = list(self.patch)
        for k in ("frame_cond", "fps_cond"):
            if not d[k]:
                d.pop(k)                 # v1 configs serialise exactly as before (stable hd_config_sha)
        return d


# ===================================================================== small modules
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class LayerNorm32(nn.LayerNorm):
    """LayerNorm whose statistics and affine are computed in float32 and returned in the input dtype (a half
    model or an fp16 autocast region). Identical to nn.LayerNorm for float32 inputs; same state-dict keys."""

    def forward(self, x):
        if x.dtype == torch.float32:
            return super().forward(x)
        w = None if self.weight is None else self.weight.float()
        b = None if self.bias is None else self.bias.float()
        return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)


def linear32(m: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """nn.Linear applied in float32 regardless of autocast / module dtype (weights are upcast on the fly)."""
    with torch.autocast(device_type=x.device.type if x.device.type in ("cuda", "cpu") else "cpu", enabled=False):
        return F.linear(x.float(), m.weight.float(), None if m.bias is None else m.bias.float())


def mlp32(seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """Linear -> SiLU -> Linear (the timestep / fps embedders) entirely in float32."""
    y = x
    for m in seq:
        y = linear32(m, y) if isinstance(m, nn.Linear) else m(y)
    return y


class QuantLinear(nn.Module):
    """int8 weight-only linear (per-output-channel absmax scales); dequantises to the input dtype on the fly."""

    def __init__(self, qweight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.in_features, self.out_features = qweight.shape[1], qweight.shape[0]

    @staticmethod
    def from_linear(lin: nn.Linear, scale_dtype=torch.float32) -> "QuantLinear":
        w = lin.weight.detach().float()
        scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        b = lin.bias.detach().clone() if lin.bias is not None else None
        return QuantLinear(q, scale.to(scale_dtype), b)

    @property
    def weight(self):  # dequantised view (for dtype queries / debugging)
        return self.qweight.float() * self.scale.float()[:, None]

    def forward(self, x):
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)[:, None]
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


def quantize_int8_(model: nn.Module, prefix: str = "blocks") -> int:
    """Replace every nn.Linear under `model.<prefix>` with QuantLinear in place. Returns count. Inference only."""
    root = getattr(model, prefix)
    n = 0
    for name, mod in list(root.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, QuantLinear.from_linear(child))
                n += 1
    model.quantized = True
    return n


def _per_frame(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """v [B,Tg,D] -> broadcastable against the token view x.view(B,Tg,N/Tg,D)."""
    return v[:, :, None, :]


def modulate(x, shift, scale):
    """adaLN: x * (1 + scale) + shift. shift/scale are [B,D] or per latent frame [B,Tg,D] (tokens are t-major).
    Arithmetic in float32 whenever x is a half type (result cast back); float32 inputs take the original
    expression."""
    dt = x.dtype
    if dt != torch.float32:
        x, shift, scale = x.float(), shift.float(), scale.float()
    if shift.dim() == 2:
        out = x * (1 + scale[:, None]) + shift[:, None]
    else:
        B, N, D = x.shape
        Tg = shift.shape[1]
        out = (x.reshape(B, Tg, N // Tg, D) * (1 + _per_frame(scale, x)) + _per_frame(shift, x)).reshape(B, N, D)
    return out if dt == torch.float32 else out.to(dt)


def gated_add(x, gate, y):
    """x + gate * y with gate [B,D] or per latent frame [B,Tg,D]; the product is formed in float32 for half x
    and the residual stream keeps x's dtype."""
    half = x.dtype != torch.float32
    g, yy = (gate.float(), y.float()) if half else (gate, y)
    if g.dim() == 2:
        d = g[:, None] * yy
    else:
        B, N, D = yy.shape
        d = (yy.reshape(B, g.shape[1], N // g.shape[1], D) * _per_frame(g, yy)).reshape(B, N, D)
    return x + d.to(x.dtype) if half else x + d


def sdpa_chunked(q, k, v, mask=None, chunk: int = 0, fp32: bool = False):
    """SDPA over query chunks (keys/values whole) to bound the attention working set. fp32=True runs the logits /
    softmax / value product in float32 for half inputs (result cast back): no fp16 overflow in the logits."""
    dt = q.dtype
    if fp32 and dt != torch.float32:
        q, k, v = q.float(), k.float(), v.float()
    N = q.shape[-2]
    if chunk <= 0 or N <= chunk:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        o = torch.cat([F.scaled_dot_product_attention(q[..., i:i + chunk, :], k, v, attn_mask=mask)
                       for i in range(0, N, chunk)], dim=-2)
    return o.to(dt) if o.dtype != dt else o


def scalar_embedding(v: torch.Tensor, dim: int, max_period: float = 1000.0) -> torch.Tensor:
    """Sinusoidal embedding of a raw scalar (e.g. fps in frames per second)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=v.device) / half)
    args = v.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ===================================================================== attention + block
class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, qk_norm: bool):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.fp32 = False

    def forward(self, x, cos, sin, chunk: int = 0):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q = apply_rope(self.qn(q), cos, sin)
        k = apply_rope(self.kn(k), cos, sin)
        o = sdpa_chunked(q, k, v, None, chunk, self.fp32)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class CrossAttention(nn.Module):
    def __init__(self, d: int, heads: int, qk_norm: bool):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.qn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.hd) if qk_norm else nn.Identity()
        self.fp32 = False

    def forward(self, x, ctx, mask, chunk: int = 0):
        B, N, D = x.shape
        L = ctx.shape[1]
        q = self.qn(self.q(x).view(B, N, self.h, self.hd).transpose(1, 2))
        k, v = self.kv(ctx).view(B, L, 2, self.h, self.hd).permute(2, 0, 3, 1, 4)
        k = self.kn(k)
        o = sdpa_chunked(q, k, v, mask[:, None, None, :], chunk, self.fp32)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.dim
        self.norm1 = LayerNorm32(d, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(d, cfg.heads, cfg.qk_norm)
        self.norm2 = LayerNorm32(d, eps=1e-6)
        self.cross = CrossAttention(d, cfg.heads, cfg.qk_norm)
        self.norm3 = LayerNorm32(d, elementwise_affine=False, eps=1e-6)
        self.fc1 = nn.Linear(d, cfg.mlp_hidden)
        self.fc2 = nn.Linear(cfg.mlp_hidden, d)
        self.mod_table = nn.Parameter(torch.zeros(6, d))
        self.attn_chunk = 0
        self.mlp_chunk = 0

    def _mlp(self, x, shift, scale):
        return self.fc2(F.gelu(self.fc1(modulate(self.norm3(x), shift, scale)), approximate="tanh"))

    def forward(self, x, mod, ctx, mask, cos, sin):
        # mod: [B,6,D] (one timestep per sample) or [B,Tg,6,D] (one per latent frame); float32
        table = self.mod_table.float()
        m = mod + (table[None] if mod.dim() == 3 else table[None, None])
        sh1, sc1, g1, sh2, sc2, g2 = m.unbind(-2)
        x = gated_add(x, g1, self.attn(modulate(self.norm1(x), sh1, sc1), cos, sin, self.attn_chunk))
        x = x + self.cross(self.norm2(x), ctx, mask, self.attn_chunk)
        c, N = self.mlp_chunk, x.shape[1]
        if c <= 0 or N <= c or g2.dim() == 3:       # per-frame modulation is not chunked
            return gated_add(x, g2, self._mlp(x, sh2, sc2))
        if not torch.is_grad_enabled():           # inference: update in place, chunk by chunk
            for i in range(0, N, c):
                xs = x[:, i:i + c]
                xs.copy_(gated_add(xs, g2, self._mlp(xs, sh2, sc2)))
            return x
        return gated_add(x, g2, torch.cat([self._mlp(x[:, i:i + c], sh2, sc2) for i in range(0, N, c)], 1))


# ===================================================================== offload
class BlockOffloader:
    """Sequential CPU offload: block weights stay in (pinned) CPU memory; each block is copied to the
    compute device right before it runs (next block prefetched on a side CUDA stream) and released
    afterwards. Inference only. CUDA path untested on GPU."""

    def __init__(self, blocks: nn.ModuleList, device):
        self.blocks = blocks
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.cpu = []
        for b in blocks:
            ts = {}
            for n, t in list(b.named_parameters()) + list(b.named_buffers()):
                c = t.data.cpu()
                if self.cuda:
                    c = c.pin_memory()
                t.data = c
                ts[n] = c
            self.cpu.append(ts)
        self.stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.events = {}

    def _tensors(self, i):
        b = self.blocks[i]
        return list(b.named_parameters()) + list(b.named_buffers())

    def _copy(self, i):
        for n, t in self._tensors(i):
            t.data = self.cpu[i][n].to(self.device, non_blocking=True)

    def prefetch(self, i):
        if i >= len(self.blocks) or i in self.events:
            return
        if self.cuda:
            with torch.cuda.stream(self.stream):
                self._copy(i)
                ev = torch.cuda.Event()
                ev.record(self.stream)
            self.events[i] = ev
        else:
            self._copy(i)
            self.events[i] = None

    def before(self, i):
        self.prefetch(i)
        ev = self.events.pop(i)
        if self.cuda:
            cur = torch.cuda.current_stream(self.device)
            cur.wait_event(ev)
            for _, t in self._tensors(i):
                t.data.record_stream(cur)
        self.prefetch(i + 1)

    def after(self, i):
        for n, t in self._tensors(i):
            t.data = self.cpu[i][n]


# ===================================================================== the model
class NivraDiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d, C = cfg.dim, cfg.latent_channels
        pt, ph, pw = cfg.patch
        self.x_embed = nn.Linear(C * pt * ph * pw, d)
        self.t_embed = nn.Sequential(nn.Linear(cfg.freq_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.t_mod = nn.Linear(d, 6 * d)
        self.text_norm = LayerNorm32(cfg.text_dim, elementwise_affine=False, eps=1e-6)
        self.text_proj = nn.Sequential(nn.Linear(cfg.text_dim, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.null_ctx = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.blocks = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.depth)])
        self.final_norm = LayerNorm32(d, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Linear(d, 2 * d)
        self.final_table = nn.Parameter(torch.zeros(2, d))
        self.out = nn.Linear(d, C * pt * ph * pw)
        if cfg.frame_cond:
            self.cond_embed = nn.Parameter(torch.zeros(d))        # flag added to clean conditioning frames
        if cfg.fps_cond:
            self.fps_embed = nn.Sequential(nn.Linear(cfg.freq_dim, d), nn.SiLU(), nn.Linear(d, d))
            self.fps_null = nn.Parameter(torch.zeros(d))          # used when the fps is unknown / dropped
        self.grad_checkpoint = False
        self.amp = True               # fp16 autocast when running on CUDA (ignored on CPU)
        self.quantized = False
        self._offload: Optional[BlockOffloader] = None
        self._init_weights()

    def _init_weights(self):
        if self.out.weight.is_meta:
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)
        for z in (self.t_mod, self.final_mod, self.out):     # adaLN-Zero + zero output
            nn.init.zeros_(z.weight)
            nn.init.zeros_(z.bias)
        if self.cfg.fps_cond:                                # a fresh fps embedder starts as a no-op
            nn.init.normal_(self.fps_embed[0].weight, std=0.02)
            nn.init.zeros_(self.fps_embed[2].weight)
        for b in self.blocks:
            nn.init.zeros_(b.cross.proj.weight)
            nn.init.zeros_(b.cross.proj.bias)

    # ------------------------------------------------------------ knobs
    def set_memory(self, grad_checkpoint: Optional[bool] = None, attn_chunk: Optional[int] = None,
                   mlp_chunk: Optional[int] = None, attn_fp32: Optional[bool] = None) -> None:
        if grad_checkpoint is not None:
            self.grad_checkpoint = bool(grad_checkpoint)
        for b in self.blocks:
            if attn_chunk is not None:
                b.attn_chunk = int(attn_chunk)
            if mlp_chunk is not None:
                b.mlp_chunk = int(mlp_chunk)
            if attn_fp32 is not None:
                b.attn.fp32 = b.cross.fp32 = bool(attn_fp32)

    def enable_offload(self, device) -> None:
        """Keep blocks on CPU and stream them to `device` one at a time; everything else moves to `device`."""
        for name, child in self.named_children():
            if name != "blocks":
                child.to(device)
        for p in self._parameters.values():            # direct parameters (null_ctx, final_table, cond_embed, ...)
            if p is not None:
                p.data = p.data.to(device)
        self._offload = BlockOffloader(self.blocks, device)

    def disable_offload(self) -> None:
        self._offload = None

    @property
    def compute_dtype(self):
        return self.x_embed.weight.dtype

    def num_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        for m in self.modules():
            if isinstance(m, QuantLinear):
                n += m.qweight.numel() + (m.bias.numel() if m.bias is not None else 0)
        return n

    # ------------------------------------------------------------ helpers
    def grid(self, latent_shape) -> tuple:
        T, H, W = latent_shape[-3:]
        pt, ph, pw = self.cfg.patch
        if T % pt or H % ph or W % pw:
            raise ValueError(f"latent grid {T}x{H}x{W} not divisible by patch {self.cfg.patch}")
        return T // pt, H // ph, W // pw

    def _patchify(self, x):
        B, C, T, H, W = x.shape
        pt, ph, pw = self.cfg.patch
        x = x.view(B, C, T // pt, pt, H // ph, ph, W // pw, pw).permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(B, (T // pt) * (H // ph) * (W // pw), C * pt * ph * pw)

    def _unpatchify(self, y, shape):
        B, C, T, H, W = shape
        pt, ph, pw = self.cfg.patch
        y = y.view(B, T // pt, H // ph, W // pw, C, pt, ph, pw).permute(0, 4, 1, 5, 2, 6, 3, 7)
        return y.reshape(B, C, T, H, W)

    def project_text(self, emb: Optional[torch.Tensor], mask: Optional[torch.Tensor], B: int,
                     drop: Optional[torch.Tensor] = None) -> tuple:
        """Project encoder tokens [B,L,D_text] to model width; rows with drop=True (or no valid token, or
        emb=None) get the learned null context (one valid token)."""
        dev, dt = self.null_ctx.device, self.compute_dtype
        if emb is None:
            ctx = self.null_ctx.to(dt).expand(B, 1, -1)
            return ctx, torch.ones(B, 1, dtype=torch.bool, device=dev)
        if emb.shape[-1] != self.cfg.text_dim:
            raise ValueError(f"text encoder width {emb.shape[-1]} != model text_dim {self.cfg.text_dim}")
        emb = emb.to(dev)
        mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=dev) if mask is None else mask.to(dev).bool()
        if emb.shape[1] == 0:
            return self.project_text(None, None, B)
        ctx = self.text_proj(self.text_norm(emb.to(dt)))
        use_null = ~mask.any(1)
        if drop is not None:
            use_null = use_null | drop.to(dev).bool()
        if bool(use_null.any()):
            null_row = torch.zeros_like(ctx)
            null_row[:, :1] = self.null_ctx.to(ctx.dtype)
            null_mask = torch.zeros_like(mask)
            null_mask[:, 0] = True
            ctx = torch.where(use_null[:, None, None], null_row, ctx)
            mask = torch.where(use_null[:, None], null_mask, mask)
        return ctx, mask

    # ------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, t: torch.Tensor, text: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None, drop_text: Optional[torch.Tensor] = None,
                t_offset: int = 0, cond_mask: Optional[torch.Tensor] = None, fps: Optional[torch.Tensor] = None,
                feats: Optional[tuple] = None):
        """x: [B,C,T,H,W] scaled noisy latents; t: [B] or per grid-frame [B,T/pt] in [0,1] (1 = pure noise).
        cond_mask [B,T/pt] bool (frame_cond models): those frames of x are CLEAN conditioning frames (t is forced to
        0 and the flag embedding is added). fps [B] float (fps_cond models; NaN = unknown). feats: block indices whose
        hidden states [B,N,D] (float32) are returned as a second value ``{index: tensor}``. Returns the velocity
        (float32), or ``(velocity, feats)``."""
        use_amp = self.amp and x.is_cuda and self.compute_dtype == torch.float32
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            v, fd = self._forward(x, t, text, text_mask, drop_text, t_offset, cond_mask, fps, feats)
        return (v.float(), fd) if feats is not None else v.float()

    def _forward(self, x, t, text, text_mask, drop_text, t_offset, cond_mask=None, fps=None, feats=None):
        B = x.shape[0]
        shape = x.shape
        dev = self.null_ctx.device
        Tg, Hg, Wg = self.grid(shape)
        h = self.x_embed(self._patchify(x.to(dev, self.compute_dtype)))
        t = t.to(dev)
        if cond_mask is not None:
            if not self.cfg.frame_cond:
                raise ValueError("this model has no frame_cond embedding (nivra-dit/1); cond_mask is not supported")
            cm = cond_mask.to(dev).bool()
            if cm.shape != (B, Tg):
                raise ValueError(f"cond_mask must be [B,{Tg}], got {tuple(cm.shape)}")
            t = torch.where(cm, torch.zeros((), dtype=t.dtype, device=dev), t if t.dim() == 2 else t[:, None].expand(B, Tg))
            h = (h.reshape(B, Tg, Hg * Wg, -1) + cm[:, :, None, None] * self.cond_embed.to(h.dtype)).reshape(B, Tg * Hg * Wg, -1)
        if t.dim() == 2 and t.shape != (B, Tg):
            raise ValueError(f"per-frame t must be [B,{Tg}], got {tuple(t.shape)}")
        per_frame = t.dim() == 2
        te = timestep_embedding(t.reshape(-1), self.cfg.freq_dim)
        temb = mlp32(self.t_embed, te)
        if per_frame:
            temb = temb.view(B, Tg, -1)
        if self.cfg.fps_cond:
            temb = temb + (self._fps_vec(fps, B, dev)[:, None] if per_frame else self._fps_vec(fps, B, dev))
        elif fps is not None:
            raise ValueError("this model has no fps_cond embedding (nivra-dit/1); fps is not supported")
        st = F.silu(temb)
        mod = linear32(self.t_mod, st).view(*st.shape[:-1], 6, self.cfg.dim)
        ctx, cmask = self.project_text(text, text_mask, B, drop_text)
        cos, sin = rope_3d(Tg, Hg, Wg, self.cfg.head_dim, self.cfg.rope_theta, t_offset, device=dev)
        off = self._offload
        if off is not None and torch.is_grad_enabled():
            raise RuntimeError("block offload is inference-only (use torch.no_grad())")
        want = set(feats or ())
        fd = {}
        for i, blk in enumerate(self.blocks):
            if off is not None:
                off.before(i)
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                h = checkpoint(blk, h, mod, ctx, cmask, cos, sin, use_reentrant=False)
            else:
                h = blk(h, mod, ctx, cmask, cos, sin)
            if off is not None:
                off.after(i)
            if i in want:
                fd[i] = h.float()
        fm = linear32(self.final_mod, st).view(*st.shape[:-1], 2, self.cfg.dim) + self.final_table.float()
        h = self.out(modulate(self.final_norm(h), fm.select(-2, 0), fm.select(-2, 1)))
        return self._unpatchify(h, shape), fd

    def _fps_vec(self, fps, B, dev):
        null = self.fps_null.float().expand(B, -1)
        if fps is None:
            return null
        f = fps.to(dev).float().reshape(-1)
        if f.shape[0] != B:
            raise ValueError("fps must have one value per batch element")
        ok = torch.isfinite(f)
        e = mlp32(self.fps_embed, scalar_embedding(torch.where(ok, f, torch.zeros_like(f)), self.cfg.freq_dim))
        return torch.where(ok[:, None], e, null)


# ===================================================================== rectified flow
def resolution_shift(n_tokens: int, base_tokens: int = 4096, max_shift: float = 6.0) -> float:
    """Timestep shift for larger token counts (more tokens -> spend more time at high noise)."""
    return float(min(max_shift, max(1.0, math.sqrt(n_tokens / base_tokens))))


def shift_t(t: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * t / (1 + (shift - 1) * t)


def sample_t(B: int, shift: float = 1.0, mean: float = 0.0, std: float = 1.0, generator=None, device="cpu"):
    """Logit-normal timesteps in (0,1), then resolution shift."""
    u = torch.randn(B, generator=generator) * std + mean
    return shift_t(torch.sigmoid(u), shift).to(device)


def sample_t_frames(B: int, T: int, shift: float = 1.0, mean: float = 0.0, std: float = 1.0, generator=None,
                    device="cpu") -> torch.Tensor:
    """Independent logit-normal (+ shift) timesteps per latent frame: [B,T] (diffusion forcing)."""
    return sample_t(B * T, shift, mean, std, generator, device).view(B, T)


def make_xt(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, cond_mask: Optional[torch.Tensor] = None):
    """Noisy input x_t = (1-t) x0 + t eps for t [B] or per-frame [B,T]; frames in cond_mask [B,T] stay clean."""
    if t.dim() == 1:
        tt = t.view(-1, 1, 1, 1, 1).to(x0.dtype)
    else:
        tt = t.view(t.shape[0], 1, t.shape[1], 1, 1).to(x0.dtype)
    xt = (1 - tt) * x0 + tt * noise
    if cond_mask is not None:
        cm = cond_mask.to(x0.device).bool()
        xt = torch.where(cm.view(cm.shape[0], 1, cm.shape[1], 1, 1), x0, xt)
    return xt


def rf_loss_ex(model: NivraDiT, x0: torch.Tensor, text, text_mask, t: torch.Tensor, noise: torch.Tensor,
               drop_text: Optional[torch.Tensor] = None, cond_mask: Optional[torch.Tensor] = None,
               fps: Optional[torch.Tensor] = None, feats: Optional[tuple] = None) -> tuple:
    """Rectified flow: x_t = (1-t) x0 + t eps, target velocity v = eps - x0 (MSE, computed in float32).

    t is [B] (one timestep per clip) or [B,T] (one per latent frame: diffusion forcing; patch-time grouped models
    need T = latent T). cond_mask [B,T] bool marks clean conditioning frames: their input is x0 itself (t = 0), the
    model gets the mask, and they are excluded from the loss. Returns (loss, feats dict or None)."""
    x0 = x0.float()
    noise = noise.float()
    xt = make_xt(x0, noise, t, cond_mask)
    pt = model.cfg.patch[0]
    gm = None
    if cond_mask is not None:
        cm = cond_mask.to(x0.device).bool()
        gm = cm[:, ::pt] if pt > 1 else cm
    tg = t if t.dim() == 1 or pt == 1 else t[:, ::pt]
    out = model(xt, tg, text, text_mask, drop_text, cond_mask=gm, fps=fps, feats=feats)
    v, fd = out if feats is not None else (out, None)
    target = (noise - x0).float()
    if cond_mask is None:
        return F.mse_loss(v.float(), target), fd
    w = (~cm).view(cm.shape[0], 1, cm.shape[1], 1, 1).to(x0.dtype)
    sq = (v.float() - target).pow(2) * w
    denom = (w.expand_as(sq)).sum().clamp_min(1.0)
    return sq.sum() / denom, fd


def rf_loss(model: NivraDiT, x0: torch.Tensor, text, text_mask, t: torch.Tensor, noise: torch.Tensor,
            drop_text: Optional[torch.Tensor] = None, cond_mask: Optional[torch.Tensor] = None,
            fps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Rectified-flow MSE (see rf_loss_ex); the original [B]-timestep call signature is unchanged."""
    return rf_loss_ex(model, x0, text, text_mask, t, noise, drop_text, cond_mask, fps)[0]


def count_params_meta(cfg: DiTConfig) -> dict:
    """Exact parameter counts without allocating memory."""
    with torch.device("meta"):
        m = NivraDiT(cfg)
    blocks = sum(p.numel() for p in m.blocks.parameters())
    total = sum(p.numel() for p in m.parameters())
    per_block = sum(p.numel() for p in m.blocks[0].parameters())
    lin_block = sum(mod.weight.numel() + mod.bias.numel() for mod in m.blocks[0].modules() if isinstance(mod, nn.Linear))
    return {"total": total, "blocks": blocks, "per_block": per_block, "per_block_linear": lin_block,
            "non_block": total - blocks}
