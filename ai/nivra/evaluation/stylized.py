"""Stylized-domain evaluation: a fixed English prompt set (3D animation / anime / cartoon / 2D animation / ...) run
through an HD pipeline. The report holds per-sample plumbing statistics (non-blank frames, brightness, motion, edge
density) so runs can be compared; it does NOT measure prompt adherence or visual quality (there is no automatic
detector for open-vocabulary stylized content)."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROMPTS_FILE = Path(__file__).with_name("stylized_prompts.json")


def load_stylized_prompts(path=None) -> list:
    from ..distributed.version import read_json_limited
    d = read_json_limited(path or PROMPTS_FILE)
    return d["prompts"]


def frame_stats(frames: np.ndarray) -> dict:
    """uint8 [T,H,W,3] -> brightness / motion (mean |diff| between consecutive frames) / edge density."""
    y = frames[..., 0] * 0.299 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114
    motion = float(np.abs(np.diff(y, axis=0)).mean()) if len(y) > 1 else 0.0
    edge = float((np.abs(np.diff(y, axis=2)).mean() + np.abs(np.diff(y, axis=1)).mean()) / 2)
    return {"brightness": round(float(y.mean()), 3), "motion": round(motion, 4), "edge_density": round(edge, 4)}


def run_hd_prompt_set(pipe, prompts: list, seed0: int = 777, steps: Optional[int] = None, cfg_scale: Optional[float] = None,
                      should_stop: Optional[Callable] = None, progress: Optional[Callable] = None,
                      use_style_field: bool = True) -> dict:
    per, t0 = [], time.time()
    for i, p in enumerate(prompts):
        if should_stop and should_stop():
            break
        kw = {}
        if steps is not None:
            kw["steps"] = steps
        if cfg_scale is not None:
            kw["cfg_scale"] = cfg_scale
        style = p.get("style") if use_style_field else None
        try:
            pipe.resolve_style(style)
        except ValueError:                       # a style this model does not list: use the model's default instead
            style = None
        res = pipe.generate(p["prompt"], seed=seed0 + i, style=style, should_stop=should_stop, **kw)
        per.append({"id": p["id"], "category": p.get("category"), "style": res.meta["style"], "prompt": p["prompt"],
                    "prompt_used": res.meta["prompt_used"], "seed": seed0 + i, "frames": int(res.frames.shape[0]),
                    "nonblank": bool(res.frames.std() > 0), **frame_stats(res.frames)})
        if progress:
            progress(len(per), len(prompts))
    n = max(1, len(per))
    return {"kind": "stylized_prompt_set", "model_version_id": pipe.model_version_id, "domain": pipe.domain,
            "samples": len(per), "seconds": time.time() - t0, "styles": dict(Counter(r["style"] for r in per)),
            "overall": {"videos": len(per), "nonblank_rate": sum(r["nonblank"] for r in per) / n,
                        "mean_brightness": sum(r["brightness"] for r in per) / n,
                        "mean_motion": sum(r["motion"] for r in per) / n,
                        "mean_edge_density": sum(r["edge_density"] for r in per) / n},
            "note": "plumbing statistics only; no prompt-adherence or quality claim", "per_sample": per}


def summarize_hd(report: dict) -> dict:
    return {"videos": report["overall"]["videos"], "nonblank_rate": report["overall"]["nonblank_rate"],
            "mean_motion": report["overall"]["mean_motion"], "mean_edge_density": report["overall"]["mean_edge_density"],
            "domain": report["domain"], "styles": report["styles"]}
