"""Two-pass streaming DiLoCo outer step for HD models (runs on the Raspberry Pi controller).

Pass 1 (update by update, one tensor in memory): structure, sha256, exact size, echo fields, finiteness, per-tensor
bounds, norms -> accept / reject, median outlier test, median clipping factors.
Pass 2 (tensor by tensor): read base + momentum + one tensor of each accepted update (fp16, or int8 dequantised on the
fly -> fp32), combine (weighted mean / weighted trimmed mean of the clipped deltas; weights = samples_seen relative to
the median, capped at 2, times the staleness discount for one-version-old deltas), outer Nesterov SGD with an optional
outer-lr warm-up, and stream three files at once:
fp32 master, fp16 train copy, fp32 outer state. Peak RAM ~ largest tensor x (N + 4) x 4 B, never a whole model.
(The held-out Pi gate, if requested, loads the <= 300M model twice, after pass 2.)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import torch

from .. import version as ver
from . import bounds, quant
from .stream_io import StReader, StWriter, StreamIOError, sha256_file
from .validate import DEFAULT_MAX_BYTES, validate_hd_update

TRAIN_FILE = "train_weights.safetensors"


class AggregateError(Exception):
    pass


def _delta_entries(shapes: dict) -> list:
    return [(k, tuple(v)) for k, v in shapes.items()]


def expected_delta_bytes(shapes: dict) -> int:
    from .stream_io import expected_file_size
    return expected_file_size(_delta_entries(shapes), torch.float16)


def _load_base(base_manifest_path, base_weights_path, outer_state_path):
    try:
        bm = ver.read_manifest(base_manifest_path)
    except ver.VersionError as e:
        raise AggregateError(str(e)) from e
    if bm.get("family") != "hd":
        raise AggregateError("base manifest is not an HD manifest (family != hd)")
    shapes = {k: tuple(v) for k, v in bm["tensor_shapes"].items()}
    try:
        with StReader(base_weights_path) as r:
            if r.shapes() != shapes or any(c not in ("F32", "F16") for c in r.dtypes().values()):
                raise AggregateError("base weights do not match the base manifest (names/shapes/dtypes)")
    except StreamIOError as e:
        raise AggregateError(str(e)) from e
    if sha256_file(base_weights_path) != bm["weights_sha256"]:
        raise AggregateError("base weights sha256 does not match the base manifest")
    if outer_state_path:
        try:
            with StReader(outer_state_path) as r:
                if r.shapes() != shapes or set(r.dtypes().values()) != {"F32"}:
                    raise AggregateError("outer state does not match the base (names/shapes, fp32)")
        except StreamIOError as e:
            raise AggregateError(str(e)) from e
        if bm.get("outer_state_sha256") and sha256_file(outer_state_path) != bm["outer_state_sha256"]:
            raise AggregateError("outer state sha256 does not match the base manifest")
    return bm, shapes


MAX_WEIGHT = 2.0


def sample_weights(samples: list, stale: list, weight_by_samples: bool, discount: float) -> list:
    """Per-update weights: samples_seen / median(samples) capped at MAX_WEIGHT (a worker cannot buy influence by
    over-claiming), times ``discount`` for one-version-stale updates. Equal samples and no stale update -> all 1.0."""
    n = len(samples)
    if weight_by_samples and n:
        med = bounds.median([float(x) for x in samples])
        base = [min(MAX_WEIGHT, float(x) / med) if med > 0 else 1.0 for x in samples]
    else:
        base = [1.0] * n
    return [w * (discount if st else 1.0) for w, st in zip(base, stale)]


def combine_weighted(stack: torch.Tensor, weights: torch.Tensor, trim_k: int) -> torch.Tensor:
    """Weighted mean over dim 0 of ``stack`` [N, ...]; with trim_k > 0 the trim_k smallest and largest values of every
    coordinate are dropped first (weighted trimmed mean). All-equal weights reproduce ``torch.mean`` / the unweighted
    trimmed mean."""
    n = stack.shape[0]
    w = weights.to(stack.dtype).reshape(n, *([1] * (stack.dim() - 1)))
    if trim_k <= 0:
        return (stack * w).sum(dim=0) / w.sum()
    s, idx = torch.sort(stack, dim=0)
    ws = w.expand_as(stack).gather(0, idx)
    s, ws = s[trim_k:n - trim_k], ws[trim_k:n - trim_k]
    return (s * ws).sum(dim=0) / ws.sum(dim=0)


def outer_lr_at(outer_lr: float, round_done: int, warmup_rounds: int, start_frac: float) -> float:
    """Effective outer lr for the aggregation that follows ``round_done`` completed rounds: linear from
    outer_lr * start_frac (round 0) to outer_lr (round >= warmup_rounds). warmup_rounds = 0 disables it."""
    if warmup_rounds <= 0 or round_done >= warmup_rounds:
        return float(outer_lr)
    return float(outer_lr) * (start_frac + (1.0 - start_frac) * round_done / warmup_rounds)


def hd_aggregate(base_manifest_path, base_weights_path, updates: list, out_dir, *, outer_state_path=None,
                 outer_lr: float = 0.7, momentum: float = 0.9, nesterov: bool = True, aggregation: str = "auto",
                 trim_ratio: float = 0.2, max_norm_ratio: float = 5.0, max_rel_norm: float = bounds.DEFAULT_MAX_REL_NORM,
                 bound_c: float = bounds.DEFAULT_BOUND_C, inner_steps: int, lr: float, clip_factor: float = 2.0,
                 version_id: Optional[str] = None, max_bytes: int = DEFAULT_MAX_BYTES, eval_gate: str = "off",
                 eval_latents: Optional[list] = None, eval_triples_seed: int = 0, eval_n: int = 8,
                 eval_buckets: int = 4, should_stop=None, delta_dtype: str = "float16", weight_by_samples: bool = True,
                 max_staleness: int = 0, staleness_discount: float = 0.5, outer_lr_warmup_rounds: int = 0,
                 outer_lr_warmup_start: float = 0.3) -> dict:
    """updates: [{"path": delta.safetensors, "manifest_path": update.json, "update_id": optional str}].

    delta_dtype: "float16" | "int8" (the format every update must have). max_staleness 0 (default) accepts only
    updates computed against the current base; 1 also accepts updates against its parent version (with weight
    ``staleness_discount``). Higher values are refused: under heavy staleness outer momentum hurts."""
    t0 = time.time()
    if delta_dtype not in ("float16", "int8"):
        raise AggregateError(f"delta_dtype must be float16 or int8, not {delta_dtype!r}")
    if max_staleness not in (0, 1):
        raise AggregateError("max_staleness must be 0 (off) or 1")
    if not (0.0 < staleness_discount <= 1.0) or not (0.0 < outer_lr_warmup_start <= 1.0) or outer_lr_warmup_rounds < 0:
        raise AggregateError("staleness_discount and outer_lr_warmup_start must be in (0, 1], warmup rounds >= 0")
    bm, shapes = _load_base(base_manifest_path, base_weights_path, outer_state_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    applied = set(bm["applied_update_ids"])
    exp = {"base_version_id": bm["version_id"], "inner_steps": inner_steps, "lr": lr, "delta_dtype": delta_dtype}
    if bm.get("hd_config_sha"):
        exp["hd_config_sha"] = bm["hd_config_sha"]
    from .validate import base_l2
    b_l2 = base_l2(bm, base_weights_path)
    stale_ids = (bm["parent_version_id"],) if max_staleness >= 1 and bm.get("parent_version_id") else ()
    rejected, cand, seen_ids, seen_sha = [], [], set(), set()
    # ---------------------------------------------------------------- pass 1
    for u in updates:
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped")
        e = dict(exp)
        if u.get("update_id"):
            e["update_id"] = u["update_id"]
        v = validate_hd_update(u["path"], u["manifest_path"], bm, expected=e, bound_c=bound_c, max_rel_norm=max_rel_norm,
                               max_bytes=max_bytes, base_l2_value=b_l2, delta_dtype=delta_dtype,
                               allowed_stale_base_ids=stale_ids)
        uid = u.get("update_id") or v.update_id
        if not v.valid:
            code, msg = v.reasons[0]
            rejected.append({"update_id": uid, "reason": code, "detail": msg})
            continue
        sha = sha256_file(u["path"])
        if uid in applied or uid in seen_ids or sha in seen_sha:
            rejected.append({"update_id": uid, "reason": "duplicate_update", "detail": "already applied or repeated"})
            continue
        seen_ids.add(uid)
        seen_sha.add(sha)
        um = ver.read_json_limited(u["manifest_path"])         # validated above
        ss = um.get("samples_seen")
        if not isinstance(ss, int) or isinstance(ss, bool) or ss < 1:
            rejected.append({"update_id": uid, "reason": "manifest_mismatch", "detail": "samples_seen must be a positive integer"})
            continue
        cand.append({"id": uid, "path": u["path"], "l2": v.l2_norm, "samples": ss,
                     "stale": um.get("base_version_id") != bm["version_id"]})
    accepted = cand
    cut = bounds.outlier_cut([c["l2"] for c in cand], max_norm_ratio)
    if cut is not None:
        accepted = []
        for c in cand:
            if c["l2"] > cut:
                rejected.append({"update_id": c["id"], "reason": "norm_outlier",
                                 "detail": f"{c['l2']:.4g} > {max_norm_ratio} x median"})
            else:
                accepted.append(c)
    summary = {"rejected": rejected, "aggregated_count": len(accepted), "applied_update_ids": [a["id"] for a in accepted],
               "base_version_id": bm["version_id"],
               "delta_bytes": quant.expected_int8_bytes(shapes) if delta_dtype == "int8" else expected_delta_bytes(shapes),
               "delta_dtype": delta_dtype}
    if not accepted:
        summary.update(new_version_id=None, no_op=True, aggregation_method=None, clipped_update_ids=[])
        return summary
    cap = bounds.clip_cap([a["l2"] for a in accepted], clip_factor)
    for a in accepted:
        a["scale"] = cap / a["l2"] if a["l2"] > cap > 0 else 1.0
    n = len(accepted)
    wts = sample_weights([a["samples"] for a in accepted], [a["stale"] for a in accepted], weight_by_samples, staleness_discount)
    wt_t = torch.tensor(wts, dtype=torch.float32)
    stale_applied = [a["id"] for a in accepted if a["stale"]]
    lr_eff = outer_lr_at(outer_lr, int(bm["round"]), outer_lr_warmup_rounds, outer_lr_warmup_start)
    use_trim = aggregation == "trimmed_mean" or (aggregation == "auto" and n >= 3)
    use_trim = use_trim and n >= 3
    trim_k = 0
    if use_trim:
        trim_k = max(1, int(trim_ratio * n))
        while n - 2 * trim_k < 1:
            trim_k -= 1
    method = "trimmed_mean" if trim_k > 0 else "mean"
    # ---------------------------------------------------------------- pass 2
    names = list(shapes)
    entries = _delta_entries(shapes)
    wf, tf, of = out / ver.WEIGHTS_FILE, out / TRAIN_FILE, out / ver.OUTER_FILE
    readers = [StReader(a["path"]) for a in accepted]
    base_r = StReader(base_weights_path)
    outer_r = StReader(outer_state_path) if outer_state_path else None
    ww, wt, wo = StWriter(wf, entries, torch.float32), StWriter(tf, entries, torch.float16), StWriter(of, entries, torch.float32)
    tensor_l2, dl2 = {}, 0.0
    try:
        for name in names:
            stack = torch.stack([quant.read_delta_f32(r, name, delta_dtype) * a["scale"] for r, a in zip(readers, accepted)])
            d = combine_weighted(stack, wt_t, trim_k)
            del stack
            w = base_r.read_f32(name)
            mp = outer_r.read_f32(name) if outer_r is not None else torch.zeros_like(w)
            m = momentum * mp + d
            step = d + momentum * m if nesterov else m
            new = w - lr_eff * step
            h16 = new.to(torch.float16)
            if not (bool(torch.isfinite(new).all()) and bool(torch.isfinite(h16).all())):
                raise AggregateError(f"aggregated tensor {name} is not finite as fp16/fp32")
            ww.write(name, new)
            wt.write(name, h16)
            wo.write(name, m)
            tensor_l2[name] = float(new.double().pow(2).sum().sqrt())
            dl2 += float(d.double().pow(2).sum())
        wsha, tsha, osha = ww.close(), wt.close(), wo.close()
    except BaseException:
        for x in (ww, wt, wo):
            x.abort()
        raise
    finally:
        for r in readers + [base_r] + ([outer_r] if outer_r else []):
            r.close()
    rnd = int(bm["round"]) + 1
    all_ids = list(bm["applied_update_ids"]) + [a["id"] for a in accepted]
    man = dict(bm)
    man.update(version_id=version_id or ver.make_version_id(rnd, wsha), round=rnd, parent_version_id=bm["version_id"],
               weights_file=ver.WEIGHTS_FILE, weights_dtype="float32", weights_sha256=wsha,
               train_weights_file=TRAIN_FILE, train_weights_dtype="float16", train_weights_sha256=tsha,
               outer_state_file=ver.OUTER_FILE, outer_state_sha256=osha, applied_update_ids=all_ids,
               created_unix=int(time.time()), tensor_l2=tensor_l2,
               weights_l2=math.sqrt(sum(v * v for v in tensor_l2.values())), delta_bytes=summary["delta_bytes"])
    (out / ver.MANIFEST_FILE).write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    summary.update(new_version_id=man["version_id"], no_op=False, all_applied_update_ids=all_ids,
                   aggregation_method=method, delta_l2=math.sqrt(dl2), outer_lr=lr_eff, outer_lr_nominal=outer_lr, momentum=momentum,
                   update_weights={a["id"]: round(w, 6) for a, w in zip(accepted, wts)}, stale_update_ids=stale_applied,
                   clipped_update_ids=[a["id"] for a in accepted if a["scale"] < 1.0], weights_sha256=wsha,
                   train_weights_sha256=tsha, manifest_path=str(out / ver.MANIFEST_FILE))
    if eval_gate == "pi" and eval_latents:
        from .eval_job import heldout_before_after
        b, a = heldout_before_after(bm, base_weights_path, wf, list(eval_latents), eval_triples_seed, eval_n, eval_buckets)
        summary["heldout_loss_before"], summary["heldout_loss_after"] = b, a
    summary["seconds"] = round(time.time() - t0, 2)
    return summary
