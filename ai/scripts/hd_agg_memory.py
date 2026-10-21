"""Measure the peak RSS of the streaming HD aggregate on a synthetic model with a real config's tensor shapes.

    python ai/scripts/hd_agg_memory.py --config nivra_medium --updates 3 --work D:/tmp/aggmem

Writes a random fp32 base, N random fp16 deltas inside the elementwise bound and their manifests (tensor by
tensor, so generating them needs little RAM either), then runs ``hd_aggregate`` in a fresh subprocess and prints
the child's peak working set (Windows PeakWorkingSetSize / Linux ru_maxrss). Needs ~ (4 + 2N + 4 + 2 + 4) bytes
per parameter of disk. psutil is not required.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHILD = """
import json, sys
sys.path.insert(0, {root!r})
from nivra.config import peak_rss_mb
from nivra.distributed.hd.aggregate import hd_aggregate
w = Path = None
import pathlib
w = pathlib.Path({work!r})
ups = [{{"path": str(w / f"u{{i}}.safetensors"), "manifest_path": str(w / f"u{{i}}.json"), "update_id": f"u{{i}}"}} for i in range({n})]
s = hd_aggregate(w / "model_version.json", w / "base.safetensors", ups, w / "out", inner_steps={steps}, lr={lr}, version_id="v1")
print(json.dumps({{"peak_rss_mb": peak_rss_mb(), "seconds": s["seconds"], "aggregated": s["aggregated_count"],
                  "rejected": s["rejected"], "method": s["aggregation_method"]}}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nivra_medium")
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    from nivra.distributed import version as ver
    from nivra.distributed.hd.stream_io import StWriter, sha256_file
    from nivra.models.hd_config import HD_ARCH_VERSION, build_dit, hd_config_hash, load_hd_config
    cfg = load_hd_config(a.config)
    with torch.device("meta"):
        shapes = {k: tuple(v.shape) for k, v in build_dit(cfg).state_dict().items()}
    total = sum(torch.Size(s).numel() for s in shapes.values())
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    steps, lr = 100, 1e-4
    bound = 3 * steps * lr
    entries = list(shapes.items())
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    with StWriter(work / "base.safetensors", entries, torch.float32) as w:
        for k, s in entries:
            w.write(k, torch.randn(s, generator=g) * 0.02)
        bsha = w.close()
    man = {"schema": ver.SCHEMA, "family": "hd", "version_id": "v0", "round": 0, "parent_version_id": None,
           "architecture_version": HD_ARCH_VERSION, "config": cfg, "vocab_hash": "x", "tensor_shapes": {k: list(s) for k, s in entries},
           "weights_sha256": bsha, "applied_update_ids": [], "hd_config_sha": hd_config_hash(cfg), "outer_state_sha256": None}
    (work / "model_version.json").write_text(json.dumps(man))
    for i in range(a.updates):
        with StWriter(work / f"u{i}.safetensors", entries, torch.float16) as w:
            for k, s in entries:
                w.write(k, (torch.rand(s, generator=g) * 2 - 1) * bound * 0.2)
            usha = w.close()
        (work / f"u{i}.json").write_text(json.dumps({
            "schema": "nivra.update/1", "family": "hd", "update_id": f"u{i}", "update_sha256": usha, "base_version_id": "v0",
            "architecture_version": HD_ARCH_VERSION, "vocab_hash": "x", "delta_dtype": "float16", "inner_steps": steps, "lr": lr,
            "run_id": "r", "round": 0, "round_batch": 0, "attempt_id": f"a{i}", "target": "dit",
            "hd_config_sha": man["hd_config_sha"], "stage": {"h": 256, "w": 448, "frames": 17, "fps": 8}, "shards": [],
            "samples_seen": steps * 8}))
    print(f"params {total / 1e6:.0f}M generated in {time.time() - t0:.0f}s", flush=True)
    code = CHILD.format(root=str(ROOT), work=str(work), n=a.updates, steps=steps, lr=lr)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-2000:])
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
