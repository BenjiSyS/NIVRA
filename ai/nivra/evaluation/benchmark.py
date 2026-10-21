"""Benchmark drivers: detector validation on ground truth, prompt-adherence on generated videos."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..config import CONFIG_DIR, synth_config
from ..data.synthetic import make_sample, SynthConfig
from ..inference.generate import generate
from ..text.parser import parse_prompt
from .adherence import score_video, aggregate

DEFAULT_PROMPTS = CONFIG_DIR / "benchmark_prompts.json"


def load_prompts(path=None) -> list:
    d = json.loads(Path(path or DEFAULT_PROMPTS).read_text(encoding="utf-8"))
    return d["prompts"]


def detector_validation(n_clips: int = 200, seed: int = 99, start: int = 5_000_000, scfg: SynthConfig = SynthConfig(),
                        should_stop: Optional[Callable] = None) -> dict:
    """Run the detector + scorer on ground-truth synthetic clips against their true specs."""
    results = []
    for i in range(n_clips):
        if should_stop and should_stop():
            break
        s = make_sample(seed, start + i, scfg)
        results.append(score_video(s.video, s.spec))
    out = aggregate(results)
    out["clips"] = len(results)
    out["note"] = "scores of the detector on real (ground-truth) synthetic clips; 1.0 = perfect"
    return out


def run_benchmark(bundle, prompts: list, seeds_per_prompt: int = 1, steps: int = 20, cfg_scale: float = 3.0,
                  seed0: int = 777, save_dir=None, should_stop: Optional[Callable] = None, progress=None) -> dict:
    per = []
    t0 = time.time()
    save_dir = Path(save_dir) if save_dir else None
    for pi, p in enumerate(prompts):
        parsed = parse_prompt(p["prompt"])
        for s in range(seeds_per_prompt):
            if should_stop and should_stop():
                break
            seed = seed0 + 1000 * s + pi
            res = generate(bundle, p["prompt"], seed=seed, steps=steps, cfg_scale=cfg_scale, should_stop=should_stop)
            sc = score_video(res.frames, parsed.spec)
            rec = {"id": p["id"], "category": p["category"], "lang": parsed.lang, "prompt": p["prompt"], "seed": seed, **sc}
            per.append(rec)
            if save_dir is not None:
                from ..inference.generate import save_result
                save_result(res, save_dir, f"{p['id']}_s{s}", extra_meta={"benchmark": rec})
            if progress:
                progress(len(per), len(prompts) * seeds_per_prompt)
    report = {"model_version_id": bundle.version_id, "steps": steps, "cfg_scale": cfg_scale,
              "seeds_per_prompt": seeds_per_prompt, "samples": len(per), "seconds": time.time() - t0,
              "overall": aggregate(per), "per_sample": per}
    cats = sorted({r["category"] for r in per})
    report["by_category"] = {c: aggregate([r for r in per if r["category"] == c]) for c in cats}
    report["by_lang"] = {l: aggregate([r for r in per if r["lang"] == l]) for l in sorted({r["lang"] for r in per})}
    return report


def summarize(report: dict) -> dict:
    o = report["overall"]
    return {k: o[k] for k in ("videos", "all_correct_rate", "entity_recall", "counts_correct_rate",
                              "temporal_consistent_rate", "missing_per_video", "wrong_color_per_video",
                              "wrong_shape_per_video", "wrong_direction_per_video", "invented_per_video")}
