"""Original word-level tokenizer over the closed bilingual vocabulary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..data.lexicon import all_caption_words

TOKENIZER_VERSION = "nivra-wordtok/1"
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)


def normalize(text: str) -> list:
    """Lowercase, strip punctuation (anything not alphanumeric), split on whitespace."""
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalnum() else " ")
    return "".join(out).split()


class WordTokenizer:
    def __init__(self, words=None, max_len: int = 32):
        words = list(words) if words is not None else all_caption_words()
        self.itos = list(SPECIALS) + [w for w in sorted(set(words)) if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_len = max_len
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = (self.stoi[s] for s in SPECIALS)

    def __len__(self):
        return len(self.itos)

    @property
    def vocab_hash(self) -> str:
        blob = json.dumps({"version": TOKENIZER_VERSION, "itos": self.itos}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def encode(self, text: str, pad: bool = True):
        ids = [self.bos_id] + [self.stoi.get(w, self.unk_id) for w in normalize(text)][: self.max_len - 2] + [self.eos_id]
        if pad:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return ids

    def unknown_words(self, text: str) -> list:
        return [w for w in normalize(text) if w not in self.stoi]

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            w = self.itos[int(i)]
            if w in (PAD, BOS):
                continue
            if w == EOS:
                break
            words.append(w)
        return " ".join(words)

    def to_json(self) -> dict:
        return {"version": TOKENIZER_VERSION, "max_len": self.max_len, "itos": self.itos, "vocab_hash": self.vocab_hash}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, d: dict) -> "WordTokenizer":
        if d.get("version") != TOKENIZER_VERSION:
            raise ValueError(f"incompatible tokenizer version {d.get('version')}")
        tok = cls.__new__(cls)
        tok.itos = list(d["itos"]); tok.stoi = {w: i for i, w in enumerate(tok.itos)}
        tok.max_len = int(d["max_len"])
        tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id = (tok.stoi[s] for s in SPECIALS)
        if tok.vocab_hash != d.get("vocab_hash"):
            raise ValueError("vocab hash mismatch")
        return tok

    @classmethod
    def load(cls, path) -> "WordTokenizer":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


_DEFAULT = None


def default_tokenizer() -> WordTokenizer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = WordTokenizer()
    return _DEFAULT
