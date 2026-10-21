"""DiLoCo-style outer step: validate, combine pseudo-gradients, outer Nesterov SGD, new version.

Depends only on torch / numpy / safetensors (runs on the Raspberry Pi controller).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

from . import version as ver
from .update import validate_update, l2_norm, DEFAULT_MAX_BYTES


def combine(deltas: list, method: str = "auto", trim_ratio: float = 0.2) -> tuple:
    """Combine a list of {name: tensor} deltas. Returns (combined, method_used)."""
    n = len(deltas)
    if n == 0:
        raise ValueError("no deltas")
    use_trim = (method == "trimmed_mean") or (method == "auto" and n >= 3)
    if use_trim and n < 3:
        use_trim = False
    out = {}
    k = 0
    if use_trim:
        k = max(1, int(trim_ratio * n))
        while n - 2 * k < 1:
            k -= 1
    for name in deltas[0]:
        stack = torch.stack([d[name].float() for d in deltas])
        if use_trim and k > 0:
            s, _ = torch.sort(stack, dim=0)           # coordinate-wise
            out[name] = s[k:n - k].mean(dim=0)
        else:
            out[name] = stack.mean(dim=0)
    return out, ("trimmed_mean" if use_trim and k > 0 else "mean")


def outer_step(base: dict, delta: dict, outer_state: Optional[dict], lr: float = 0.7, momentum: float = 0.9,
               nesterov: bool = True) -> tuple:
    """Outer SGD with (Nesterov) momentum on the pseudo-gradient.  new = base - lr * step."""
    new, buf = {}, {}
    for k, w in base.items():
        g = delta[k].to(w.dtype)
        m_prev = outer_state[k].to(w.dtype) if outer_state is not None and k in outer_state else torch.zeros_like(w)
        m = momentum * m_prev + g
        step = g + momentum * m if nesterov else m
        new[k] = w - lr * step
        buf[k] = m
    return new, buf


def aggregate(base_manifest_path, base_weights_path, updates: list, out_dir, *, outer_state_path=None,
              outer_lr: float = 0.7, momentum: float = 0.9, nesterov: bool = True, aggregation: str = "auto",
              trim_ratio: float = 0.2, max_norm_ratio: float = 5.0, max_bytes: int = DEFAULT_MAX_BYTES,
              version_id: Optional[str] = None) -> dict:
    """updates: list of {"path": update.safetensors, "manifest_path": update.json}.
    Returns a summary dict (new_version_id None and no_op True when nothing was accepted)."""
    base_m = ver.read_manifest(base_manifest_path)
    base = ver.load_weights(base_m, base_weights_path)
    outer_state = load_file(str(outer_state_path)) if outer_state_path else None
    applied = set(base_m["applied_update_ids"])
    rejected, cand = [], []
    seen_ids = set()
    # first pass: structural validation without the norm test (need all norms for the median)
    for u in updates:
        val, tensors = validate_update(u["path"], u["manifest_path"], base_m, max_norm_ratio=max_norm_ratio,
                                       max_bytes=max_bytes, base_weights=base, return_tensors=True)
        uid = val.update_id
        if uid is not None and (uid in applied or uid in seen_ids):
            rejected.append({"update_id": uid, "reason": "duplicate_update_id", "detail": "already applied or repeated"})
            continue
        if not val.valid:
            code, msg = val.reasons[0]
            rejected.append({"update_id": uid, "reason": code, "detail": msg})
            continue
        seen_ids.add(uid)
        cand.append((uid, tensors, val.l2_norm))
    # norm outlier test against the median of the structurally valid peers (needs >= 3 to be meaningful)
    accepted = cand
    if len(cand) >= 3:
        med = float(torch.tensor([c[2] for c in cand], dtype=torch.float64).median())
        accepted = []
        for uid, t, nrm in cand:
            if med > 0 and nrm > max_norm_ratio * med:
                rejected.append({"update_id": uid, "reason": "norm_outlier", "detail": f"{nrm:.4g} > {max_norm_ratio} x median {med:.4g}"})
            else:
                accepted.append((uid, t, nrm))
    summary = {"rejected": rejected, "aggregated_count": len(accepted), "applied_update_ids": [a[0] for a in accepted],
               "base_version_id": base_m["version_id"]}
    if not accepted:
        summary.update(new_version_id=None, no_op=True, all_applied_update_ids=sorted(applied), aggregation_method=None)
        return summary
    delta, used = combine([a[1] for a in accepted], aggregation, trim_ratio)
    new_w, buf = outer_step(base, delta, outer_state, outer_lr, momentum, nesterov)
    rnd = base_m["round"] + 1
    all_ids = list(base_m["applied_update_ids"]) + [a[0] for a in accepted]
    manifest = ver.write_version(out_dir, new_w, config=base_m["config"], config_name=base_m["config_name"],
                                 arch_version=base_m["architecture_version"], tokenizer_version=base_m["tokenizer_version"],
                                 vocab_hash=base_m["vocab_hash"], vae_sha256=base_m["vae_sha256"], rnd=rnd,
                                 parent_version_id=base_m["version_id"], applied_update_ids=all_ids, outer_state=buf,
                                 version_id=version_id)
    summary.update(new_version_id=manifest["version_id"], no_op=False, all_applied_update_ids=all_ids,
                   aggregation_method=used, delta_l2=l2_norm(delta), outer_lr=outer_lr, momentum=momentum,
                   manifest_path=str(Path(out_dir) / ver.MANIFEST_FILE))
    return summary
