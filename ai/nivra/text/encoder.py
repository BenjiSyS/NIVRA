"""Small transformer text encoder, trained from scratch jointly with the generator."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_padding_mask=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        mask = None
        if key_padding_mask is not None:  # True = keep
            mask = key_padding_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class TextBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=3):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        return x + self.mlp(self.n2(x))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim=128, layers=2, heads=4, max_len=32, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(TextBlock(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids):
        mask = ids != self.pad_id                     # [B,L] True = real token
        x = self.tok(ids) + self.pos[:, : ids.shape[1]]
        for b in self.blocks:
            x = b(x, mask)
        return self.norm(x), mask
