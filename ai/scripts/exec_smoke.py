"""End-to-end smoke run of every executor job type through `python -m nivra.cli exec --job job.json`.

python ai/scripts/exec_smoke.py --ckpt runs/small/gen/ckpt_00006000 --vae runs/small/vae/export/vae.safetensors --out runs/smoke
Writes <out>/smoke_report.json (exit code, artifact kinds, wall seconds, summary per job).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent


def run(out: Path, name, job_type, params=None, inputs=None):
    wd = out / name
    wd.mkdir(parents=True, exist_ok=True)
    job = {"protocol": "nivra.exec/1", "job_type": job_type, "job_id": name, "attempt_id": name + "-1",
           "params": params or {}, "inputs": inputs or {}, "output_dir": str(wd / "out"), "stop_file": str(wd / "STOP"),
           "limits": {"max_seconds": 3600, "max_ram_mb": 4096, "max_vram_mb": 0}}
    (wd / "job.json").write_text(json.dumps(job, indent=1))
    env = dict(os.environ, PYTHONPATH=str(AI_DIR), PYTHONUTF8="1")
    env.setdefault("NIVRA_THREADS", "2")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "nivra.cli", "exec", "--job", str(wd / "job.json")], env=env,
                       capture_output=True, text=True)
    wall = time.time() - t0
    ev = [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]
    arts = {e["kind"]: e["path"] for e in ev if e["event"] == "artifact"}
    term = ev[-1] if ev else {}
    rec = {"name": name, "job_type": job_type, "exit_code": p.returncode, "terminal": term.get("event"),
           "wall_s": round(wall, 1), "artifact_kinds": sorted(arts), "summary": term.get("summary") or term}
    print(json.dumps({k: rec[k] for k in ("name", "exit_code", "wall_s", "artifact_kinds")}), flush=True)
    return rec, arts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="generator checkpoint dir (for ai.init_model from_checkpoint)")
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    recs = []

    def step(*args, **kw):
        r, arts = run(out, *args, **kw)
        recs.append(r)
        if r["exit_code"] != 0:
            raise SystemExit(f"{r['name']} failed: {r['summary']}")
        return arts

    v0 = step("init_model", "ai.init_model", {"from_checkpoint": str(Path(a.ckpt).resolve()), "vae_checkpoint": str(Path(a.vae).resolve())})
    base = {"base_weights": v0["model_weights"], "base_manifest": v0["model_version"], "vae_weights": v0["vae_weights"]}
    step("dataset_shard", "ai.dataset_shard", {"seed": 31, "start": 0, "count": 16})
    u1 = step("local_train_a", "ai.local_train", {"update_id": "smoke-a", "shard": {"seed": 31, "start": 0, "count": 40},
                                                  "inner_steps": 4, "batch_size": 4, "eval_clips": 8}, base)
    u2 = step("local_train_b", "ai.local_train", {"update_id": "smoke-b", "shard": {"seed": 32, "start": 0, "count": 40},
                                                  "inner_steps": 4, "batch_size": 4, "eval_clips": 8}, base)
    step("validate_update", "ai.validate_update", {}, {"update": u1["update_weights"], "update_manifest": u1["update"],
                                                       "base_manifest": v0["model_version"], "base_weights": v0["model_weights"]})
    agg = step("aggregate", "ai.aggregate", {"eval_heldout": True},
               {"base_weights": v0["model_weights"], "base_manifest": v0["model_version"], "vae_weights": v0["vae_weights"],
                "updates": [{"path": u["update_weights"], "manifest_path": u["update"]} for u in (u1, u2)]})
    gi = {"weights": v0["model_weights"], "manifest": v0["model_version"], "vae_weights": v0["vae_weights"]}
    step("generate", "ai.generate", {"prompt": "a red circle moves from left to right", "seed": 1, "steps": 20, "name": "smoke"}, gi)
    step("evaluate_detector", "ai.evaluate", {"mode": "detector", "num_clips": 16})
    step("evaluate_reconstruction", "ai.evaluate", {"mode": "reconstruction", "num_clips": 8}, {"manifest": v0["model_version"], "vae_weights": v0["vae_weights"]})
    step("evaluate_adherence", "ai.evaluate", {"mode": "adherence", "num_prompts": 2, "steps": 10}, gi)
    (out / "smoke_report.json").write_text(json.dumps({"aggregated_version_artifacts": sorted(agg), "jobs": recs}, indent=1))
    print("all", len(recs), "jobs exit 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
