"""Reader for datasets written by `nivra.data.ingest`: yields (video uint8 tensor, caption, metadata)."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from safetensors import safe_open


def read_manifest(root: Path) -> list[dict]:
    return [json.loads(l) for l in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def style_counts(rows) -> dict[str, int]:
    """Clips per style tag; rows from datasets written before the style field existed count as 'unspecified'."""
    out: dict[str, int] = {}
    for r in rows:
        k = r.get("style") or "unspecified"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def verify_shards(root, check_fingerprint: bool = True) -> list[str]:
    """Return a list of problems (empty = dataset is intact): shard checksums, sizes, manifest fingerprint."""
    root = Path(root)
    problems: list[str] = []
    try:
        ds = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [f"dataset.json unreadable: {e}"]
    for s in ds.get("shards", []):
        p = root / s["file"]
        if not p.is_file():
            problems.append(f"missing shard {s['file']}")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != s["sha256"]:
            problems.append(f"checksum mismatch {s['file']}")
    if check_fingerprint:
        try:
            fp = hashlib.sha256((root / "manifest.jsonl").read_bytes()).hexdigest()
            if fp != ds.get("fingerprint"):
                problems.append("manifest fingerprint mismatch")
        except OSError as e:
            problems.append(f"manifest unreadable: {e}")
    return problems


class ShardDataset:
    """Random-access / iterable view over one split ('train', 'val' or None = all). Memory-bounded: clips are
    read one at a time with safetensors slices, so a shard is never fully resident."""

    def __init__(self, root, split: str | None = "train", verify: bool = False):
        self.root = Path(root)
        if verify:
            bad = verify_shards(self.root)
            if bad:
                raise ValueError(f"dataset {self.root} failed verification: {bad}")
        self.info = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        rows = read_manifest(self.root)
        self.rows = [r for r in rows if split is None or r["split"] == split]
        self.fingerprint = self.info["fingerprint"]
        self.styles = style_counts(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        with safe_open(str(self.root / r["shard"]), framework="pt") as f:
            video = f.get_slice("videos")[r["index"]: r["index"] + 1][0].contiguous()
        return video, r["caption"], {k: v for k, v in r.items() if k != "caption"}

    def __iter__(self):
        for i in range(len(self.rows)):
            yield self[i]

    def shuffled(self, seed: int):
        order = list(range(len(self.rows)))
        random.Random(seed).shuffle(order)
        for i in order:
            yield self[i]

    @staticmethod
    def to_float(video: torch.Tensor) -> torch.Tensor:
        """uint8 [T,H,W,3] -> float32 [3,T,H,W] in [-1, 1]."""
        return video.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
