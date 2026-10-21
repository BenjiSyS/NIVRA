"""Pseudo-gradient updates: writing, reading and validation.

Depends only on torch / numpy / safetensors (runs on the Raspberry Pi controller).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file, load_file

from .version import (sha256_file, read_json_limited, check_tensor_file, ManifestTooLarge, VersionError)

SCHEMA = "nivra.update/1"
UPDATE_FILE = "update.safetensors"
MANIFEST_FILE = "update.json"
DEFAULT_MAX_BYTES = 1 << 30


def write_update(out_dir, delta: dict, meta: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().float().contiguous() for k, v in delta.items()}, str(out / UPDATE_FILE))
    m = dict(meta)
    m.update(schema=SCHEMA, update_file=UPDATE_FILE, update_sha256=sha256_file(out / UPDATE_FILE),
             tensor_shapes={k: list(v.shape) for k, v in delta.items()})
    (out / MANIFEST_FILE).write_text(json.dumps(m, indent=1, sort_keys=True))
    return m


def l2_norm(tensors: dict) -> float:
    s = 0.0
    for v in tensors.values():
        s += float(v.double().pow(2).sum())
    return math.sqrt(s)


@dataclass
class Validation:
    valid: bool
    reasons: list = field(default_factory=list)      # list of (code, message)
    l2_norm: Optional[float] = None
    update_id: Optional[str] = None
    num_tensors: int = 0
    bytes: int = 0

    @property
    def codes(self):
        return [c for c, _ in self.reasons]

    def reason_strings(self):
        return [f"{c}: {m}" for c, m in self.reasons]


REQUIRED_KEYS = ("schema", "update_id", "base_version_id", "update_sha256", "inner_steps", "samples",
                 "architecture_version", "vocab_hash")


DEFAULT_MAX_REL_NORM = 0.5          # ||delta|| <= max_rel_norm * ||base||   (always enforced, no peers needed)
TENSOR_NORM_FACTOR = 4.0            # ||delta_t|| <= 4 * ||base_t|| + TENSOR_NORM_SLACK
TENSOR_NORM_SLACK = 1e-3
TENSOR_ELEM_SLACK = 0.05           # extra per-element allowance (x sqrt(numel)): zero-initialised tensors
                                   # (adaLN-zero, output heads) legitimately move a lot in the first rounds
ZERO_UPDATE_NORM = 1e-8


def tensor_norms(tensors: dict) -> dict:
    return {k: float(v.double().pow(2).sum().sqrt()) for k, v in tensors.items()}


def base_norms(base_manifest: dict, base_weights: Optional[dict]):
    """(per-tensor L2 dict, total L2) of the base weights, from the real weights or the manifest; None if unknown."""
    if base_weights is not None:
        tn = tensor_norms(base_weights)
        return tn, math.sqrt(sum(v * v for v in tn.values()))
    tn = base_manifest.get("tensor_l2")
    if isinstance(tn, dict) and tn:
        return tn, math.sqrt(sum(float(v) ** 2 for v in tn.values()))
    return None


def validate_update(update_path, manifest_path, base_manifest: dict, *, peer_norms=None, max_norm_ratio: float = 5.0,
                    max_bytes: int = DEFAULT_MAX_BYTES, max_norm: Optional[float] = None,
                    expected_base_version_id: Optional[str] = None, base_weights: Optional[dict] = None,
                    max_rel_norm: float = DEFAULT_MAX_REL_NORM, return_tensors: bool = False):
    """Structural + numerical validation. Returns Validation (and the tensors if requested and valid).

    Resource limits first: the manifest is size-checked before it is parsed and the safetensors header
    (names / shapes / dtypes / total bytes) is checked against the base before any tensor is loaded."""
    v = Validation(False)
    tensors = None

    def fail(code, msg):
        v.reasons.append((code, msg))
        return (v, None) if return_tensors else v

    try:
        m = read_json_limited(manifest_path)
    except ManifestTooLarge as e:
        return fail("manifest_too_large", str(e))
    except Exception as e:  # noqa: BLE001
        return fail("manifest_invalid", f"cannot read update manifest: {e}")
    if not isinstance(m, dict):
        return fail("manifest_invalid", "update manifest must be a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in m]
    if m.get("schema") != SCHEMA or missing:
        return fail("manifest_invalid", f"schema/keys: {m.get('schema')!r} missing {missing}")
    v.update_id = str(m["update_id"])
    p = Path(update_path)
    try:
        v.bytes = p.stat().st_size
    except OSError as e:
        return fail("manifest_invalid", f"update file unreadable: {e}")
    if v.bytes > max_bytes:
        return fail("size_limit", f"{v.bytes} bytes > {max_bytes}")
    if sha256_file(p) != m["update_sha256"]:
        return fail("checksum_mismatch", "update file sha256 differs from its manifest")
    exp_base = expected_base_version_id or base_manifest["version_id"]
    if m["base_version_id"] != exp_base:
        v.reasons.append(("stale_base_version", f"update base {m['base_version_id']} != current {exp_base}"))
    if m["architecture_version"] != base_manifest["architecture_version"] or m["vocab_hash"] != base_manifest["vocab_hash"]:
        v.reasons.append(("manifest_mismatch", "architecture / tokenizer differs from base"))
    if m.get("base_weights_sha256") and m["base_weights_sha256"] != base_manifest["weights_sha256"]:
        v.reasons.append(("stale_base_version", "base_weights_sha256 differs from current weights"))
    expected = {k: tuple(s) for k, s in base_manifest["tensor_shapes"].items()} if "tensor_shapes" in base_manifest else \
        ({k: tuple(t.shape) for k, t in base_weights.items()} if base_weights else None)
    try:
        check_tensor_file(p, expected, "F32", max_bytes)      # header only, before load_file
    except VersionError as e:
        shape_problem = "expected" in str(e) or "differ" in str(e)
        return fail("shape_mismatch" if shape_problem else "manifest_invalid", str(e))
    try:
        tensors = load_file(str(p))
    except Exception as e:  # noqa: BLE001
        return fail("manifest_invalid", f"update safetensors unreadable: {e}")
    v.num_tensors = len(tensors)
    if base_weights is not None and set(base_weights) == set(tensors):
        for k, t in tensors.items():
            if t.shape != base_weights[k].shape or t.dtype != base_weights[k].dtype:
                v.reasons.append(("shape_mismatch", f"{k} differs from the real base weights"))
                break
    finite = all(bool(torch.isfinite(t).all()) for t in tensors.values())
    if not finite:
        v.reasons.append(("non_finite", "update contains NaN or Inf"))
    else:
        tn = tensor_norms(tensors)
        v.l2_norm = math.sqrt(sum(x * x for x in tn.values()))
        if v.l2_norm < ZERO_UPDATE_NORM:
            v.reasons.append(("zero_update", f"norm {v.l2_norm:.3g} < {ZERO_UPDATE_NORM}"))
        bn = base_norms(base_manifest, base_weights)
        if bn is None:
            v.reasons.append(("base_norm_unavailable", "no base weights or manifest tensor_l2: cannot bound the update"))
        else:
            b_tn, b_total = bn
            if v.l2_norm > max_rel_norm * b_total:
                v.reasons.append(("norm_outlier", f"norm {v.l2_norm:.4g} > {max_rel_norm} x ||base|| {b_total:.4g}"))
            for k, x in tn.items():
                if k not in b_tn:
                    continue
                bound = TENSOR_NORM_FACTOR * float(b_tn[k]) + TENSOR_NORM_SLACK + TENSOR_ELEM_SLACK * math.sqrt(tensors[k].numel())
                if x > bound:
                    v.reasons.append(("tensor_norm_outlier", f"{k}: {x:.4g} > bound {bound:.4g} (4 x ||base_t|| {float(b_tn[k]):.4g} + slack)"))
                    break
        if peer_norms:
            med = float(torch.tensor(peer_norms, dtype=torch.float64).median())
            if med > 0 and v.l2_norm > max_norm_ratio * med:
                v.reasons.append(("norm_outlier", f"norm {v.l2_norm:.4g} > {max_norm_ratio} x median {med:.4g}"))
        if max_norm is not None and v.l2_norm > max_norm:
            v.reasons.append(("norm_outlier", f"norm {v.l2_norm:.4g} > max_norm {max_norm}"))
    v.valid = not v.reasons
    if return_tensors:
        return v, (tensors if v.valid else None)
    return v
