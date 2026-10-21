"""Prompt-adherence scoring: compare detected tracks with the parsed scene specification."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.scene import SceneSpec
from .detector import detect_video, Detection

AREA_CV_MAX = 0.35
COVERAGE_MIN = 0.8


def score_detection(det: Detection, spec: SceneSpec) -> dict:
    tracks = list(range(len(det.tracks)))
    ents = list(range(len(spec.entities)))
    match = {}           # entity idx -> (track idx, kind)

    def take(pred, kind):
        for e in list(ents):
            ent = spec.entities[e]
            for t in list(tracks):
                if pred(ent, det.tracks[t], det.motions[t]):
                    match[e] = (t, kind)
                    ents.remove(e)
                    tracks.remove(t)
                    break

    # pass 1: colour + shape + motion; pass 2: colour + shape; pass 3: colour only (shape wrong); pass 4: any colour, same motion
    take(lambda en, tr, m: tr.color == en.color and (en.shape is None or tr.shape == en.shape) and m == en.motion, "ok")
    take(lambda en, tr, m: tr.color == en.color and (en.shape is None or tr.shape == en.shape), "wrong_direction")
    take(lambda en, tr, m: tr.color == en.color and m == en.motion, "wrong_shape")
    take(lambda en, tr, m: tr.color == en.color, "wrong_shape_direction")
    take(lambda en, tr, m: m == en.motion and (en.shape is None or tr.shape == en.shape), "wrong_color")
    take(lambda en, tr, m: m == en.motion, "wrong_color")
    take(lambda en, tr, m: True, "wrong_color")

    n = len(spec.entities)
    res = {"entities": n, "detected_tracks": len(det.tracks), "ok": 0, "missing": len(ents), "invented": len(tracks),
           "wrong_color": 0, "wrong_shape": 0, "wrong_direction": 0, "fragments": 0}
    stability = []
    for e, (t, kind) in match.items():
        tr = det.tracks[t]
        if kind == "ok":
            res["ok"] += 1
        else:
            if kind == "wrong_color":
                res["wrong_color"] += 1
            if "shape" in kind:
                res["wrong_shape"] += 1
            if "direction" in kind:
                res["wrong_direction"] += 1
        cov = len(tr.frames) / det.n_frames
        areas = np.array(tr.areas, dtype=np.float64)
        cv = float(areas.std() / max(areas.mean(), 1e-6))
        stability.append({"coverage": cov, "area_cv": cv, "ok": bool(cov >= COVERAGE_MIN and cv <= AREA_CV_MAX)})
    for t in tracks:
        tr = det.tracks[t]
        cov = len(tr.frames) / det.n_frames
        areas = np.array(tr.areas, dtype=np.float64)
        stability.append({"coverage": cov, "area_cv": float(areas.std() / max(areas.mean(), 1e-6)),
                          "ok": bool(cov >= COVERAGE_MIN)})
    # colour-count agreement
    want = spec.counts_by_color()
    got: dict = {}
    for tr in det.tracks:
        got[tr.color] = got.get(tr.color, 0) + 1
    colors = set(want) | set(got)
    res["count_errors"] = sum(1 for c in colors if want.get(c, 0) != got.get(c, 0))
    res["counts_correct"] = res["count_errors"] == 0
    res["temporal_consistent"] = bool(stability) and all(s["ok"] for s in stability)
    res["mean_area_cv"] = float(np.mean([s["area_cv"] for s in stability])) if stability else None
    res["max_blobs_in_a_frame"] = det.max_blobs_per_frame
    res["all_correct"] = bool(res["ok"] == n and res["invented"] == 0 and res["counts_correct"])
    return res


def score_video(video: np.ndarray, spec: SceneSpec) -> dict:
    det = detect_video(video)
    r = score_detection(det, spec)
    return r


def aggregate(results: list) -> dict:
    """results: list of per-video dicts from score_video."""
    if not results:
        return {}
    n = len(results)
    ent = sum(r["entities"] for r in results)
    out = {"videos": n, "entities": ent,
           "all_correct_rate": sum(r["all_correct"] for r in results) / n,
           "entity_recall": sum(r["ok"] for r in results) / max(ent, 1),
           "counts_correct_rate": sum(r["counts_correct"] for r in results) / n,
           "temporal_consistent_rate": sum(r["temporal_consistent"] for r in results) / n}
    for k in ("missing", "wrong_color", "wrong_shape", "wrong_direction", "invented"):
        out[f"{k}_per_video"] = sum(r[k] for r in results) / n
        out[f"{k}_rate_of_entities"] = sum(r[k] for r in results) / max(ent, 1)
    out["videos_with_invented"] = sum(1 for r in results if r["invented"] > 0) / n
    return out
