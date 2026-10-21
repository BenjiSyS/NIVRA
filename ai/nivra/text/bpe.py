"""Byte-level BPE tokenizer, implemented from scratch (no `tokenizers` dependency).

* Base vocabulary: 4 specials (<pad>=0 <bos>=1 <eos>=2 <mask>=3) + the 256 byte values (ids 4..259), then
  learned merges (ids 260+). Every UTF-8 string is representable: there is no unknown token.
* Text is split into pre-tokens by a lossless regex (letters / digits / symbols, each with an optional
  leading space, plus whitespace runs); merges never cross pre-token boundaries.
* Training is deterministic: ties are broken by the smaller pair of ids.
* Serialisation: JSON {version, kind, specials, merges, vocab_size, sha256}; sha256 covers the canonical
  content and is verified on load.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

FORMAT_VERSION = 1
KIND = "nivra-bytebpe"
SPECIALS = ["<pad>", "<bos>", "<eos>", "<mask>"]
PAD, BOS, EOS, MASK = 0, 1, 2, 3
N_SPECIAL = len(SPECIALS)
BYTE_BASE = N_SPECIAL          # byte b has id BYTE_BASE + b
FIRST_MERGE_ID = N_SPECIAL + 256

_PRETOK = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def pretokenize(text: str) -> list[str]:
    parts = _PRETOK.findall(text)
    if "".join(parts) != text:          # defensive: the pattern is lossless, but never silently drop text
        return [text]
    return parts


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab_size = FIRST_MERGE_ID + len(self.merges)
        self._bytes: list[bytes] = [b""] * N_SPECIAL + [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._cache: dict[str, tuple[int, ...]] = {}
        self.pad_id, self.bos_id, self.eos_id, self.mask_id = PAD, BOS, EOS, MASK

    def __len__(self):
        return self.vocab_size

    # ---------------------------------------------------------------- encode / decode
    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        ids = [BYTE_BASE + b for b in piece.encode("utf-8")]
        ranks = self.ranks
        while len(ids) > 1:
            best, best_rank = None, None
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            pair = (ids[best], ids[best + 1])
            new_id = FIRST_MERGE_ID + best_rank
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        res = tuple(ids)
        if len(self._cache) < 500_000:
            self._cache[piece] = res
        return res

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        out: list[int] = [BOS] if add_bos else []
        for piece in pretokenize(text):
            out.extend(self._encode_piece(piece))
        if add_eos:
            out.append(EOS)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        data = b"".join(self._bytes[i] for i in ids if i >= N_SPECIAL)
        return data.decode("utf-8", errors="replace")

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        return b"".join(self._bytes[i] for i in ids if i >= N_SPECIAL)

    # ---------------------------------------------------------------- training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, min_frequency: int = 2) -> "BPETokenizer":
        if vocab_size < FIRST_MERGE_ID:
            raise ValueError(f"vocab_size must be >= {FIRST_MERGE_ID} (specials + 256 bytes)")
        freq: Counter = Counter()
        for t in texts:
            freq.update(pretokenize(t))
        words: list[list[int]] = []
        wfreq: list[int] = []
        for piece, f in sorted(freq.items()):            # sorted: deterministic order
            words.append([BYTE_BASE + b for b in piece.encode("utf-8")])
            wfreq.append(f)
        pair_count: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_count[p] += wfreq[wi]
                where[p].add(wi)
        merges: list[tuple[int, int]] = []
        while FIRST_MERGE_ID + len(merges) < vocab_size and pair_count:
            pair, cnt = max(pair_count.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))
            if cnt < min_frequency:
                break
            new_id = FIRST_MERGE_ID + len(merges)
            merges.append(pair)
            for wi in list(where.pop(pair, ())):
                w = words[wi]
                f = wfreq[wi]
                for p in zip(w, w[1:]):                  # remove old contributions of this word
                    pair_count[p] -= f
                    if pair_count[p] <= 0:
                        pair_count.pop(p, None)
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):
                    pair_count[p] += f
                    where[p].add(wi)
            pair_count.pop(pair, None)
        return cls(merges)

    @classmethod
    def train_files(cls, paths: list[str | Path], vocab_size: int, min_frequency: int = 2) -> "BPETokenizer":
        def lines():
            for p in expand_text_paths(paths):
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        yield line
        return cls.train(lines(), vocab_size, min_frequency)

    # ---------------------------------------------------------------- serialisation
    def _content(self) -> dict:
        return {"version": FORMAT_VERSION, "kind": KIND, "specials": SPECIALS,
                "merges": [list(m) for m in self.merges], "vocab_size": self.vocab_size}

    @property
    def sha256(self) -> str:
        blob = json.dumps(self._content(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def save(self, path: str | Path) -> str:
        d = self._content()
        d["sha256"] = self.sha256
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, sort_keys=True), encoding="utf-8")
        return d["sha256"]

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("kind") != KIND or d.get("version") != FORMAT_VERSION:
            raise ValueError(f"{path}: not a {KIND} v{FORMAT_VERSION} file")
        tok = cls([tuple(m) for m in d["merges"]])
        if tok.sha256 != d.get("sha256"):
            raise ValueError(f"{path}: sha256 mismatch (file corrupted or edited)")
        if tok.vocab_size != d.get("vocab_size"):
            raise ValueError(f"{path}: vocab_size mismatch")
        return tok


def expand_text_paths(paths) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in (".txt", ".md", ".text")))
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError("no text files found")
    return out
