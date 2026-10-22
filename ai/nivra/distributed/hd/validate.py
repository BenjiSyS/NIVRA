"""Validation of one HD (fp16) update, streaming tensor by tensor (runs on the Raspberry Pi controller).

Entry point for the controller: executor job ``ai.hd_validate_update`` (see HD_TRAINING.md). The same function
is the first pass of ``ai.hd_aggregate``.

Reason codes (``Validation.codes``): manifest_too_large, manifest_invalid, size_limit, size_mismatch,
checksum_mismatch, shape_mismatch, stale_base_version, manifest_mismatch, non_finite, bound_exceeded,
zero_update, norm_outlier, duplicate_update, int8_invalid.

Two delta formats: ``float16`` (default; one F16 tensor per base tensor) and ``int8`` (``quant.py``: I8 values + F32
per-row scales, exact size checked before any tensor is read, bounds applied to the dequantised values).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from .. import version as ver
from ..update import SCHEMA, Validation
from . import bounds, quant
from .stream_io import StReader, StreamIOError, check_exact_size, sha256_file

DELTA_DTYPE = "float16"
DELTA_DTYPES = ("float16", "int8")
DEFAULT_MAX_BYTES = 8 << 30
ECHO_FIELDS = ("run_id", "round", "round_batch", "base_version_id", "update_id", "attempt_id", "target",
               "hd_config_sha", "stage", "inner_steps", "lr", "shards", "samples_seen", "delta_dtype")
EXPECT_KEYS = ("run_id", "round", "round_batch", "base_version_id", "update_id", "attempt_id", "target",
               "hd_config_sha", "stage", "inner_steps", "lr", "shards", "delta_dtype", "batch_size", "grad_accum")


def norm_stage(s) -> Optional[tuple]:
    """{h,w,frames,fps} (or {height,width,...}) -> comparable tuple."""
    if not isinstance(s, dict):
        return None
    g = lambda *ks: next((s[k] for k in ks if k in s), None)       # noqa: E731
    return (g("h", "height"), g("w", "width"), g("frames"), g("fps"))


def norm_shards(sh) -> Optional[list]:
    if not isinstance(sh, list):
        return None
    try:
        return [(str(x["artifact_id"]), str(x["sha256"]).lower()) for x in sh]
    except (TypeError, KeyError):
        return None


def check_echo(m: dict, expected: dict) -> list:
    """Every echoed field must be present and equal the job parameter the controller issued. Returns [(code, msg)]."""
    out = []
    for k in ECHO_FIELDS:
        if k not in m:
            out.append(("manifest_mismatch", f"update manifest lacks {k}"))
    for k, want in expected.items():
        if want is None or k not in ECHO_FIELDS or k not in m:
            continue
        got = m[k]
        if k == "stage":
            ok = norm_stage(got) == norm_stage(want)
        elif k == "shards":
            ok = norm_shards(got) == norm_shards(want)
        elif k == "lr":
            ok = isinstance(got, (int, float)) and math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=0.0)
        elif k in ("round", "round_batch", "inner_steps"):
            ok = isinstance(got, int) and not isinstance(got, bool) and got == int(want)
        else:
            ok = got == want
        if not ok:
            out.append(("manifest_mismatch", f"{k}: update says {got!r}, job parameters say {want!r}"))
    bs, ga, steps = expected.get("batch_size"), expected.get("grad_accum"), expected.get("inner_steps")
    if "samples_seen" in m and bs and ga and steps:
        if m["samples_seen"] != int(steps) * int(bs) * int(ga):
            out.append(("manifest_mismatch", f"samples_seen {m['samples_seen']} != inner_steps x batch_size x grad_accum "
                                             f"= {int(steps) * int(bs) * int(ga)}"))
    return out


def base_l2(base_manifest: dict, base_weights_path=None) -> Optional[float]:
    """||base||_2 from the manifest (weights_l2 / tensor_l2) or by streaming the weights; None if unknown."""
    w = base_manifest.get("weights_l2")
    if isinstance(w, (int, float)) and w > 0:
        return float(w)
    tn = base_manifest.get("tensor_l2")
    if isinstance(tn, dict) and tn:
        return math.sqrt(sum(float(v) ** 2 for v in tn.values()))
    if base_weights_path is not None:
        s = 0.0
        with StReader(base_weights_path) as r:
            for k in r.names():
                s += float(r.read_f32(k).double().pow(2).sum())
        return math.sqrt(s)
    return None


def validate_hd_update(update_path, manifest_path, base_manifest: dict, *, expected: Optional[dict] = None,
                       bound_c: float = bounds.DEFAULT_BOUND_C, inner_steps: Optional[int] = None,
                       lr: Optional[float] = None, max_rel_norm: float = bounds.DEFAULT_MAX_REL_NORM,
                       max_bytes: int = DEFAULT_MAX_BYTES, peer_norms=None, max_norm_ratio: float = 5.0,
                       expected_base_version_id: Optional[str] = None, base_weights_path=None,
                       base_l2_value: Optional[float] = None, delta_dtype: Optional[str] = None,
                       allowed_stale_base_ids=()) -> Validation:
    """Structural + numerical validation of one update. ``expected`` carries the job parameters the manifest must
    echo (EXPECT_KEYS); inner_steps / lr default to expected's. ``delta_dtype`` (default: expected's, else float16)
    selects the format. ``allowed_stale_base_ids``: older base versions whose deltas are still accepted (staleness
    window); such an update is then checked against that base id instead of the current one.
    Never loads more than one tensor."""
    expected = dict(expected or {})
    want_dtype = delta_dtype or expected.get("delta_dtype") or DELTA_DTYPE
    if want_dtype not in DELTA_DTYPES:
        return Validation(False, [("manifest_mismatch", f"unsupported delta_dtype {want_dtype!r}")])
    expected["delta_dtype"] = want_dtype
    steps = int(inner_steps if inner_steps is not None else expected.get("inner_steps", 0) or 0)
    lr_ = float(lr if lr is not None else expected.get("lr", 0.0) or 0.0)
    v = Validation(False)

    def fail(code, msg):
        v.reasons.append((code, msg))
        return v

    try:
        m = ver.read_json_limited(manifest_path)
    except ver.ManifestTooLarge as e:
        return fail("manifest_too_large", str(e))
    except Exception as e:  # noqa: BLE001
        return fail("manifest_invalid", f"cannot read update manifest: {e}")
    if not isinstance(m, dict) or m.get("schema") != SCHEMA or m.get("family") != "hd" or "update_id" not in m \
            or "update_sha256" not in m:
        return fail("manifest_invalid", "not an HD update manifest (schema nivra.update/1, family hd)")
    v.update_id = str(m["update_id"])
    p = Path(update_path)
    try:
        v.bytes = p.stat().st_size
    except OSError as e:
        return fail("manifest_invalid", f"update file unreadable: {e}")
    if v.bytes > max_bytes:
        return fail("size_limit", f"{v.bytes} bytes > {max_bytes}")
    shapes = base_manifest.get("tensor_shapes")
    if not isinstance(shapes, dict) or not shapes:
        return fail("manifest_invalid", "base manifest has no tensor_shapes")
    if want_dtype == "int8":
        msg = quant.check_exact_size_int8(p, shapes)           # header only, before any tensor is read
    else:
        msg = check_exact_size(p, shapes, "F16")
    if msg:
        return fail("shape_mismatch" if "tensor" in msg else "size_mismatch", msg)
    if sha256_file(p) != str(m["update_sha256"]).lower():
        return fail("checksum_mismatch", "update file sha256 differs from its manifest")
    exp_base = expected_base_version_id or expected.get("base_version_id") or base_manifest["version_id"]
    stale_ok = m.get("base_version_id") in set(allowed_stale_base_ids) and m.get("base_version_id") != exp_base
    if stale_ok:
        exp_base = m["base_version_id"]
    if m.get("base_version_id") != exp_base:
        v.reasons.append(("stale_base_version", f"update base {m.get('base_version_id')} != current {exp_base}"))
    if m.get("architecture_version") != base_manifest.get("architecture_version") \
            or m.get("vocab_hash") != base_manifest.get("vocab_hash"):
        v.reasons.append(("manifest_mismatch", "architecture / text encoder differs from the base"))
    if m.get("delta_dtype") != want_dtype:
        v.reasons.append(("manifest_mismatch", f"delta_dtype {m.get('delta_dtype')!r} != {want_dtype!r}"))
    exp = {k: val for k, val in expected.items() if k in EXPECT_KEYS}
    exp.setdefault("base_version_id", exp_base)
    if stale_ok:
        exp["base_version_id"] = exp_base
    v.reasons.extend(check_echo(m, exp))
    if steps <= 0 or lr_ <= 0:
        v.reasons.append(("manifest_mismatch", "inner_steps and lr are required to bound the update"))
        return v
    bound = bounds.elementwise_bound(bound_c, steps, lr_)
    total = 0.0
    int8 = want_dtype == "int8"
    try:
        with StReader(p) as r:
            v.num_tensors = len(r.info) // (2 if int8 else 1)
            for name in (shapes if int8 else r.names()):
                try:
                    if int8:
                        d32 = quant.read_int8_delta(r, name)
                        total += bounds.check_tensor(name, d32, bound, quant.INT8_SLACK)
                    else:
                        total += bounds.check_tensor(name, r.read_f32(name), bound)
                except bounds.BoundViolation as e:
                    v.reasons.append((e.code, e.message))
                    return v
                except quant.QuantError as e:
                    v.reasons.append((e.code, e.message))
                    return v
    except StreamIOError as e:
        return fail("shape_mismatch", str(e))
    v.l2_norm = math.sqrt(total)
    if v.l2_norm < bounds.ZERO_UPDATE_NORM:
        v.reasons.append(("zero_update", f"norm {v.l2_norm:.3g} < {bounds.ZERO_UPDATE_NORM}"))
    b = base_l2_value if base_l2_value is not None else base_l2(base_manifest, base_weights_path)
    if b is not None and v.l2_norm > max_rel_norm * b:
        v.reasons.append(("norm_outlier", f"norm {v.l2_norm:.4g} > {max_rel_norm} x ||base|| {b:.4g}"))
    if peer_norms:
        cut = bounds.outlier_cut([float(x) for x in peer_norms], max_norm_ratio)
        if cut is not None and v.l2_norm > cut:
            v.reasons.append(("norm_outlier", f"norm {v.l2_norm:.4g} > {max_norm_ratio} x median of peers"))
    v.valid = not v.reasons
    return v
