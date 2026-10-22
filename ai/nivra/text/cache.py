"""Offline text-embedding cache for encode jobs: each distinct (encoder id, max_len, text) is embedded once.

Entries are fp16 (what latent shards store anyway) and hold only the real tokens, so a cache hit and a fresh encode
give bit-identical shards. Safe to share between processes (atomic writes) and to delete at any time.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class EmbeddingCache:
    def __init__(self, directory, encoder_id: str, max_len: int = 0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.encoder_id, self.max_len = encoder_id, int(max_len)
        self.hits = self.misses = 0

    def path(self, text: str) -> Path:
        k = hashlib.sha256(f"{self.encoder_id}\0{self.max_len}\0{text}".encode("utf-8")).hexdigest()
        return self.dir / k[:2] / f"{k}.safetensors"

    def get(self, text: str):
        p = self.path(text)
        if not p.is_file():
            return None
        try:
            return load_file(str(p))["emb"]
        except Exception:                       # damaged / half-written entry: recompute
            return None

    def put(self, text: str, emb: torch.Tensor) -> None:
        p = self.path(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        os.close(fd)
        try:
            save_file({"emb": emb.to(torch.float16).contiguous()}, tmp)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


@torch.no_grad()
def encode_cached(enc, texts: list, cache: EmbeddingCache, batch: int = 32) -> tuple:
    """Same result as encode_job.encode_texts(enc, texts) (float32 [N,L,D], bool [N,L], zero padded) but every text
    that was embedded before comes from the cache."""
    rows: list = [None] * len(texts)
    todo: dict = {}
    for i, t in enumerate(texts):
        e = cache.get(t)
        if e is not None:
            rows[i] = e
            cache.hits += 1
        else:
            todo.setdefault(t, []).append(i)
    uniq = list(todo)
    for s in range(0, len(uniq), batch):
        chunk = uniq[s:s + batch]
        emb, mask = enc.encode(chunk)
        for t, e, m in zip(chunk, emb, mask):
            n = int(m.sum())
            row = e[:n].to(torch.float16).cpu()
            cache.put(t, row)
            cache.misses += 1
            for i in todo[t]:
                rows[i] = row
    L = max(r.shape[0] for r in rows)
    D = rows[0].shape[1]
    out = torch.zeros(len(texts), L, D)
    mask = torch.zeros(len(texts), L, dtype=torch.bool)
    for i, r in enumerate(rows):
        out[i, :r.shape[0]] = r.float()
        mask[i, :r.shape[0]] = True
    return out, mask
