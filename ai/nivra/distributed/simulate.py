"""Local N-worker DiLoCo simulation: separate OS processes via the exec protocol, disjoint data shards.

python -m nivra.distributed.simulate --base runs/small/export --out runs/diloco --workers 2 --rounds 3 --inner-steps 40
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _job(path: Path, job_type, params, inputs, out_dir, name):
    job = {"protocol": "nivra.exec/1", "job_type": job_type, "job_id": name, "attempt_id": name, "params": params,
           "inputs": inputs, "output_dir": str(out_dir), "stop_file": str(path.parent / "STOP"),
           "limits": {"max_seconds": 7200}}
    path.write_text(json.dumps(job))
    return path


def _spawn(job_path: Path, threads: int):
    env = dict(os.environ, NIVRA_THREADS=str(threads), PYTHONUTF8="1")
    return subprocess.Popen([sys.executable, "-m", "nivra.cli", "exec", "--job", str(job_path)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _result(p):
    out, err = p.communicate()
    events = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    if p.returncode != 0:
        raise RuntimeError(f"job failed rc={p.returncode}: {events[-1:] } {err[-2000:]}")
    return events[-1]["summary"]


def heldout(version_dir: Path, vae: Path, n: int = 64):
    from safetensors.torch import load_file
    from ..config import build_gen, load_config, synth_config
    from ..modelio import load_vae
    from ..training.tasks import encode_dataset, heldout_flow_loss
    from . import version as ver
    m = ver.read_manifest(version_dir / ver.MANIFEST_FILE)
    cfg = load_config(m["config"])
    lat, ids = encode_dataset(load_vae(vae), synth_config(cfg), 424242, 3_000_000, n)
    model = build_gen(cfg)
    model.load_state_dict(load_file(str(version_dir / ver.WEIGHTS_FILE)))
    return heldout_flow_loss(model, lat, ids[:, 0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="version directory (model_version.json, weights, vae)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--inner-steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shard-count", type=int, default=336)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--heldout-clips", type=int, default=64)
    a = ap.parse_args(argv)
    base = Path(a.base)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vae = base / "vae.safetensors"
    cur = base
    outer = None
    rows = [{"round": 0, "version_dir": str(cur), "heldout_loss": heldout(cur, vae, a.heldout_clips)}]
    print(json.dumps(rows[0]), flush=True)
    for r in range(1, a.rounds + 1):
        t0 = time.time()
        procs, dirs = [], []
        for w in range(a.workers):
            wd = out / f"round{r}" / f"worker{w}"
            wd.mkdir(parents=True, exist_ok=True)
            params = {"update_id": f"r{r}-w{w}", "shard": {"seed": 1000 + w, "start": (r - 1) * a.shard_count, "count": a.shard_count},
                      "inner_steps": a.inner_steps, "batch_size": a.batch_size, "lr": a.lr, "seed": 17 * r + w, "eval_clips": 16}
            inputs = {"base_weights": str(cur / "weights.safetensors"), "base_manifest": str(cur / "model_version.json"),
                      "vae_weights": str(vae)}
            jp = _job(wd / "job.json", "ai.local_train", params, inputs, wd / "out", f"r{r}w{w}")
            procs.append(_spawn(jp, a.threads))
            dirs.append(wd / "out")
        sums = [_result(p) for p in procs]
        t_workers = time.time() - t0
        ad = out / f"round{r}" / "agg"
        ad.mkdir(parents=True, exist_ok=True)
        inputs = {"base_weights": str(cur / "weights.safetensors"), "base_manifest": str(cur / "model_version.json"),
                  "updates": [{"path": str(d / "update.safetensors"), "manifest_path": str(d / "update.json")} for d in dirs]}
        if outer:
            inputs["outer_state"] = str(outer)
        jp = _job(ad / "job.json", "ai.aggregate", {}, inputs, ad / "out", f"r{r}agg")
        agg = _result(_spawn(jp, a.threads))
        cur = ad / "out"
        outer = cur / "outer_state.safetensors"
        (cur / "vae.safetensors").write_bytes(vae.read_bytes()) if not (cur / "vae.safetensors").exists() else None
        row = {"round": r, "version_id": agg["new_version_id"], "heldout_loss": heldout(cur, vae, a.heldout_clips),
               "workers": [{k: s[k] for k in ("update_id", "loss_first", "loss_last", "loss_before", "loss_after", "delta_l2")} for s in sums],
               "aggregation": agg["aggregation_method"], "rejected": agg["rejected"], "worker_wall_s": t_workers}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "diloco_results.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
