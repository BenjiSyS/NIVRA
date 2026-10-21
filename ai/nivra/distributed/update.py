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

from .version import sha256_file

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


def validate_update(update_path, manifest_path, base_manifest: dict, *, peer_norms=None, max_norm_ratio: float = 5.0,
                    max_bytes: int = DEFAULT_MAX_BYTES, max_norm: Optional[float] = None,
                    expected_base_version_id: Optional[str] = None, base_weights: Optional[dict] = None,
                    return_tensors: bool = False):
    """Structural + numerical validation. Returns Validation (and the tensors if requested and valid)."""
    v = Validation(False)
    tensors = None
    try:
        m = json.loads(Path(manifest_path).read_text())
    except Exception as e:  # noqa: BLE001
        v.reasons.append(("manifest_invalid", f"cannot read update manifest: {e}"))
        return (v, None) if return_tensors else v
    missing = [k for k in REQUIRED_KEYS if k not in m]
    if m.get("schema") != SCHEMA or missing:
        v.reasons.append(("manifest_invalid", f"schema/keys: {m.get('schema')!r} missing {missing}"))
        return (v, None) if return_tensors else v
    v.update_id = str(m["update_id"])
    p = Path(update_path)
    try:
        v.bytes = p.stat().st_size
    except OSError as e:
        v.reasons.append(("manifest_invalid", f"update file unreadable: {e}"))
        return (v, None) if return_tensors else v
    if v.bytes > max_bytes:
        v.reasons.append(("size_limit", f"{v.bytes} bytes > {max_bytes}"))
        return (v, None) if return_tensors else v
    if sha256_file(p) != m["update_sha256"]:
        v.reasons.append(("checksum_mismatch", "update file sha256 differs from its manifest"))
        return (v, None) if return_tensors else v
    exp_base = expected_base_version_id or base_manifest["version_id"]
    if m["base_version_id"] != exp_base:
        v.reasons.append(("stale_base_version", f"update base {m['base_version_id']} != current {exp_base}"))
    if m["architecture_version"] != base_manifest["architecture_version"] or m["vocab_hash"] != base_manifest["vocab_hash"]:
        v.reasons.append(("manifest_mismatch", "architecture / tokenizer differs from base"))
    if m.get("base_weights_sha256") and m["base_weights_sha256"] != base_manifest["weights_sha256"]:
        v.reasons.append(("stale_base_version", "base_weights_sha256 differs from current weights"))
    try:
        tensors = load_file(str(p))
    except Exception as e:  # noqa: BLE001
        v.reasons.append(("manifest_invalid", f"update safetensors unreadable: {e}"))
        return (v, None) if return_tensors else v
    v.num_tensors = len(tensors)
    expected = {k: tuple(s) for k, s in base_manifest["tensor_shapes"].items()} if "tensor_shapes" in base_manifest else \
        ({k: tuple(t.shape) for k, t in base_weights.items()} if base_weights else None)
    if expected is not None:
        if set(expected) != set(tensors):
            v.reasons.append(("shape_mismatch", "tensor names differ from base"))
        else:
            for k, t in tensors.items():
                if tuple(t.shape) != expected[k] or t.dtype != torch.float32:
                    v.reasons.append(("shape_mismatch", f"{k}: {tuple(t.shape)} {t.dtype} vs {expected[k]} float32"))
                    break
    if base_weights is not None and expected is not None and set(base_weights) == set(tensors):
        for k, t in tensors.items():
            if t.shape != base_weights[k].shape or t.dtype != base_weights[k].dtype:
                v.reasons.append(("shape_mismatch", f"{k} differs from the real base weights"))
                break
    finite = all(bool(torch.isfinite(t).all()) for t in tensors.values())
    if not finite:
        v.reasons.append(("non_finite", "update contains NaN or Inf"))
    else:
        v.l2_norm = l2_norm(tensors)
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
