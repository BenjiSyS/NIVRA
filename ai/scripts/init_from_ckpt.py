"""Create a model version from a generator checkpoint through the real executor (ai.init_model from_checkpoint).

python ai/scripts/init_from_ckpt.py --ckpt runs/small/gen/ckpt_00000600 --vae runs/small/vae/export/vae.safetensors --out runs/diloco_base600 [--no-ema]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ema", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    job = {"protocol": "nivra.exec/1", "job_type": "ai.init_model", "job_id": "init", "attempt_id": "init-1",
           "params": {"from_checkpoint": str(Path(a.ckpt).resolve()), "vae_checkpoint": str(Path(a.vae).resolve()),
                      "use_ema": not a.no_ema},
           "inputs": {}, "output_dir": str(out / "version"), "stop_file": str(out / "STOP"), "limits": {"max_seconds": 600}}
    (out / "job.json").write_text(json.dumps(job, indent=1))
    env = dict(os.environ, PYTHONPATH=str(AI_DIR), PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "nivra.cli", "exec", "--job", str(out / "job.json")], env=env,
                       capture_output=True, text=True)
    print(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-1000:])
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
