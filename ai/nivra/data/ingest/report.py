"""`data-inspect`: summary of a dataset directory plus integrity validation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..shards import read_manifest, verify_shards
import json


def inspect_dataset(root, samples: int = 5) -> dict:
    root = Path(root)
    ds = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    rows = read_manifest(root)
    problems = verify_shards(root)
    return {
        "dir": str(root), "fingerprint": ds["fingerprint"], "clips": len(rows),
        "per_bucket": dict(Counter(r["bucket"] for r in rows)), "per_split": dict(Counter(r["split"] for r in rows)),
        "per_lang": dict(Counter(r["lang"] for r in rows)),
        "per_license": dict(Counter(r["provenance"]["license"] for r in rows)),
        "shards": len(ds.get("shards", [])), "bytes": sum(s["bytes"] for s in ds.get("shards", [])),
        "rejected": ds.get("rejected", {}),
        "sample_captions": [r["caption"] for r in rows[:: max(1, len(rows) // samples)][:samples]],
        "ok": not problems, "problems": problems}
