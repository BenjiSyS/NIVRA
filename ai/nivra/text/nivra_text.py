"""Nivra Text: original open-vocabulary transformer text encoder (design contract section 3).

* byte-level BPE (bpe.py) -> pre-norm transformer encoder with RoPE positions and a final LayerNorm
* masked-language-model pretraining (`pretrain_mlm`), CPU-capable at tiny scale; CUDA fp16 path is UNTESTED
* checkpoint = directory {config.json, model.safetensors, tokenizer.json}, bound together by tokenizer sha256
* `NivraTextEncoder.from_checkpoint(dir)` implements the TextEncoder interface
* `info_nce` / `ContrastiveHead` are the caption<->video alignment hook for later joint training
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .bpe import BOS, EOS, MASK, PAD, BPETokenizer, expand_text_paths

CKPT_VERSION = 1


@dataclass
class NivraTextConfig:
    name: str = "tiny"
    vocab_size: int = 512
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_len: int = 64
    dropout: float = 0.0
    rope_base: float = 10000.0

    @classmethod
    def preset(cls, size: str, vocab_size: int) -> "NivraTextConfig":
        if size == "tiny":     # tests / CPU smoke runs
            return cls("tiny", vocab_size, 64, 2, 4, 256, 64)
        if size == "small":    # ~ 10-25M, laptop-trainable in principle
            return cls("small", vocab_size, 256, 6, 4, 1024, 128)
        if size == "base":     # ~110M with a 32k vocab
            return cls("base", vocab_size, 768, 12, 12, 3072, 256)
        raise ValueError(f"unknown size {size!r} (tiny|small|base)")


def _rope_cache(seq: int, head_dim: int, base: float, device, dtype=torch.float32):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    return f.cos().to(dtype), f.sin().to(dtype)


def _apply_rope(x, cos, sin):          # x [B,H,L,Dh]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class _Block(nn.Module):
    def __init__(self, c: NivraTextConfig):
        super().__init__()
        self.h = c.n_heads
        self.n1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.o = nn.Linear(c.d_model, c.d_model)
        self.n2 = nn.LayerNorm(c.d_model)
        self.f1 = nn.Linear(c.d_model, c.d_ff)
        self.f2 = nn.Linear(c.d_ff, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, attn_mask, cos, sin):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.f2(F.gelu(self.f1(self.n2(x)))))


class NivraTextModel(nn.Module):
    def __init__(self, c: NivraTextConfig):
        super().__init__()
        assert c.d_model % c.n_heads == 0 and (c.d_model // c.n_heads) % 2 == 0
        self.cfg = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model, padding_idx=PAD)
        self.blocks = nn.ModuleList(_Block(c) for _ in range(c.n_layers))
        self.final_norm = nn.LayerNorm(c.d_model)
        self.mlm_dense = nn.Linear(c.d_model, c.d_model)
        self.mlm_norm = nn.LayerNorm(c.d_model)
        self.mlm_bias = nn.Parameter(torch.zeros(c.vocab_size))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, keep):
        """ids [B,L] long, keep [B,L] bool -> hidden [B,L,D] after final layer norm."""
        B, L = ids.shape
        x = self.tok(ids)
        cos, sin = _rope_cache(L, self.cfg.d_model // self.cfg.n_heads, self.cfg.rope_base, ids.device, x.dtype)
        attn = keep[:, None, None, :]                       # [B,1,1,L] bool: True = may attend
        for b in self.blocks:
            x = b(x, attn, cos, sin)
        return self.final_norm(x)

    def mlm_logits(self, hidden):
        h = self.mlm_norm(F.gelu(self.mlm_dense(hidden)))
        return h @ self.tok.weight.t() + self.mlm_bias       # tied embeddings


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def count_params_for(cfg: NivraTextConfig) -> int:
    """Parameter count without allocating real weights (meta device)."""
    with torch.device("meta"):
        return count_params(NivraTextModel(cfg))


# ---------------------------------------------------------------------------- encoder (interface)
class NivraTextEncoder:
    def __init__(self, model: NivraTextModel, tokenizer: BPETokenizer, device="cpu", ckpt_id: str | None = None):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dim = model.cfg.d_model
        self.max_len = model.cfg.max_len
        self.id = ckpt_id or f"nivra-text-{model.cfg.name}@{tokenizer.sha256[:12]}"

    def tokenize(self, prompts: list[str]):
        seqs = []
        for p in prompts:
            ids = self.tokenizer.encode(p)[: self.max_len - 2]
            seqs.append([BOS] + ids + [EOS])
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return ids, ids != PAD

    @torch.no_grad()
    def encode(self, prompts: list[str]):
        if not prompts:
            raise ValueError("encode() needs at least one prompt")
        ids, mask = self.tokenize(prompts)
        ids, mask = ids.to(self.device), mask.to(self.device)
        emb = self.model(ids, mask).float()
        return (emb * mask[..., None]).contiguous(), mask

    @classmethod
    def from_checkpoint(cls, directory, device="cpu") -> "NivraTextEncoder":
        d = Path(directory)
        meta = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if meta.get("ckpt_version") != CKPT_VERSION:
            raise ValueError(f"{d}: unsupported checkpoint version {meta.get('ckpt_version')!r}")
        tok = BPETokenizer.load(d / "tokenizer.json")
        if tok.sha256 != meta["tokenizer_sha256"]:
            raise ValueError(f"{d}: tokenizer.json does not match the tokenizer this model was trained with")
        cfg = NivraTextConfig(**meta["model"])
        if cfg.vocab_size != tok.vocab_size:
            raise ValueError(f"{d}: config vocab_size {cfg.vocab_size} != tokenizer {tok.vocab_size}")
        model = NivraTextModel(cfg)
        model.load_state_dict(load_file(str(d / "model.safetensors")))
        return cls(model, tok, device, ckpt_id=meta.get("id"))


def save_checkpoint(directory, model: NivraTextModel, tokenizer: BPETokenizer, extra: dict | None = None) -> dict:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    tokenizer.save(d / "tokenizer.json")
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(d / "model.safetensors"), metadata={"nivra_text_config": json.dumps(asdict(model.cfg), sort_keys=True)})
    wsha = hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()
    meta = {"ckpt_version": CKPT_VERSION, "id": f"nivra-text-{model.cfg.name}@{tokenizer.sha256[:12]}",
            "model": asdict(model.cfg), "tokenizer_sha256": tokenizer.sha256, "weights_sha256": wsha,
            "params": count_params(model), "origin": "nivra-original", **(extra or {})}
    (d / "config.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------- MLM pretraining
def build_token_stream(paths, tokenizer: BPETokenizer, max_tokens: int | None = None) -> torch.Tensor:
    """All lines of the corpus -> one flat int tensor, each line wrapped in <bos> ... <eos>."""
    ids: list[int] = []
    for p in expand_text_paths(paths):
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ids.extend(tokenizer.encode(line, add_bos=True, add_eos=True))
                if max_tokens and len(ids) >= max_tokens:
                    return torch.tensor(ids[:max_tokens], dtype=torch.long)
    return torch.tensor(ids, dtype=torch.long)


def mask_tokens(ids: torch.Tensor, vocab_size: int, gen: torch.Generator, p: float = 0.15):
    """BERT-style: p of real tokens selected; 80% -> <mask>, 10% -> random byte/merge token, 10% unchanged."""
    real = ids >= 4
    sel = (torch.rand(ids.shape, generator=gen) < p) & real
    if not sel.any():                                   # guarantee at least one target per batch
        flat = real.nonzero()
        i = flat[torch.randint(len(flat), (1,), generator=gen)][0]
        sel[i[0], i[1]] = True
    labels = torch.where(sel, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, generator=gen)
    out = ids.clone()
    out[sel & (r < 0.8)] = MASK
    rnd = sel & (r >= 0.8) & (r < 0.9)
    out[rnd] = torch.randint(4, vocab_size, (int(rnd.sum()),), generator=gen)
    return out, labels


def pretrain_mlm(corpus, tokenizer: BPETokenizer, out_dir, size="tiny", steps=200, batch=16, seq_len=None,
                 lr=1e-3, warmup=20, seed=0, device="cpu", fp16=False, save_every=0, log=print,
                 cfg: NivraTextConfig | None = None, max_tokens: int | None = None) -> dict:
    """Masked-LM pretraining. Returns {"history": [(step, loss)], "meta": ckpt meta, ...}.

    fp16=True (CUDA only) uses autocast + GradScaler. That path is UNTESTED on this machine.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    cfg = cfg or NivraTextConfig.preset(size, tokenizer.vocab_size)
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError("config vocab_size must equal tokenizer vocab size")
    seq_len = min(seq_len or cfg.max_len, cfg.max_len)
    stream = build_token_stream(corpus, tokenizer, max_tokens)
    if len(stream) < seq_len + 1:
        raise ValueError(f"corpus too small: {len(stream)} tokens < seq_len {seq_len}")
    model = NivraTextModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    use_amp = bool(fp16) and torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_chunks = len(stream) // seq_len
    chunks = stream[: n_chunks * seq_len].view(n_chunks, seq_len)
    hist, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        cur = lr * min(1.0, step / max(1, warmup)) * (0.5 * (1 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = cur
        idx = torch.randint(n_chunks, (batch,), generator=gen)
        ids = chunks[idx]
        inp, labels = mask_tokens(ids, cfg.vocab_size, gen)
        inp, labels = inp.to(device), labels.to(device)
        keep = torch.ones_like(inp, dtype=torch.bool)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=use_amp):
            h = model(inp, keep)
            sel = labels != -100
            logits = model.mlm_logits(h[sel])
            loss = F.cross_entropy(logits.float(), labels[sel])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        lv = loss.detach().item()
        hist.append((step, lv))
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            log(f"[mlm] step {step}/{steps} loss {lv:.4f} lr {cur:.2e}")
        if save_every and step % save_every == 0 and step != steps:
            save_checkpoint(out_dir, model, tokenizer, {"mlm_steps_done": step})
    meta = save_checkpoint(out_dir, model, tokenizer, {"mlm_steps_done": steps, "mlm_final_loss": hist[-1][1],
                                                       "corpus_tokens": int(len(stream)), "seed": seed})
    return {"history": hist, "meta": meta, "seconds": time.time() - t0, "tokens": int(len(stream)),
            "random_baseline_loss": math.log(cfg.vocab_size)}


# ---------------------------------------------------------------------------- contrastive alignment hook
def masked_mean_pool(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(emb.dtype)
    return (emb * m).sum(1) / m.sum(1).clamp_min(1.0)


def info_nce(text_pooled: torch.Tensor, video_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between [B,D] text and [B,D] video embeddings (matching rows are positives)."""
    if text_pooled.shape != video_emb.shape:
        raise ValueError(f"shape mismatch {tuple(text_pooled.shape)} vs {tuple(video_emb.shape)}")
    t = F.normalize(text_pooled.float(), dim=-1)
    v = F.normalize(video_emb.float(), dim=-1)
    logits = t @ v.t() / temperature
    y = torch.arange(len(t), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y))


class ContrastiveHead(nn.Module):
    """Projects pooled text (D_text) and video (D_video) embeddings into a shared space; use with `info_nce`."""

    def __init__(self, d_text: int, d_video: int, d_shared: int = 256):
        super().__init__()
        self.t = nn.Linear(d_text, d_shared)
        self.v = nn.Linear(d_video, d_shared)

    def loss(self, text_emb, text_mask, video_emb, temperature: float = 0.07):
        return info_nce(self.t(masked_mean_pool(text_emb, text_mask)), self.v(video_emb), temperature)
