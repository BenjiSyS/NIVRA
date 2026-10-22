"""int8 per-row-scaled pseudo-gradients with error feedback ("int8-rowscale/1").

File layout (safetensors), for every base tensor ``name`` of shape S (in the base's order)::

    name            I8   shape S            quantised values q, every q in [-127, 127] (-128 is never written)
    name#scale      F32  shape [rows(S)]    per-row scale s >= 0 (finite)

    rows(S) = 0 if numel(S) == 0 else (S[0] if ndim(S) >= 2 else 1)       (1-D / 0-d tensors share one scale)
    dequant(name)[r, :] = float32(q[r, :]) * s[r]                         (float32 multiply, the Go validator does the same)
    s[r] = max|x[r, :]| / 127 and q = round(x / s) (all-zero row: s = 0, q = 0)

Exact file size = 8 + header_len + sum(numel) + 4 * sum(rows). ``#scale`` cannot collide with a model tensor name
(``build_entries`` refuses a base that has such a name).

Error feedback: the worker keeps ``residual = (delta + previous residual) - dequant(quant(delta + previous residual))``
per tensor across rounds, so the quantisation error is re-sent in the next update instead of being lost
(Streaming DiLoCo, INTELLECT-1). The residual is only valid for the very next round of the same run, see ``ResidualStore``.

The golden fixture ``ai/tests/fixtures/hd_int8_golden.safetensors`` is shared with the Go validator test.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import torch

from .stream_io import StReader, StWriter, StreamIOError, _numel, sha256_file

FORMAT = "int8-rowscale/1"
SCALE_SUFFIX = "#scale"
INT8_SLACK = 1.0 + 2.0 ** -7           # error feedback may add up to half a quantisation step of the previous round


class QuantError(Exception):
    """A malformed int8 delta (code is a validation reason code)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def rows_of(shape) -> int:
    shape = tuple(int(d) for d in shape)
    if _numel(shape) == 0:
        return 0
    return shape[0] if len(shape) >= 2 else 1


def scale_name(name: str) -> str:
    return name + SCALE_SUFFIX


def build_entries(shapes: dict) -> list:
    """Header entries [(name, shape, dtype)] of an int8 delta for the base tensor shapes."""
    out = []
    for name, shape in shapes.items():
        if name.endswith(SCALE_SUFFIX):
            raise QuantError("shape_mismatch", f"tensor name {name!r} ends with the reserved {SCALE_SUFFIX!r}")
        out.append((name, tuple(shape), torch.int8))
        out.append((scale_name(name), (rows_of(shape),), torch.float32))
    return out


def expected_int8_bytes(shapes: dict) -> int:
    from .stream_io import expected_file_size
    return expected_file_size(build_entries(shapes), torch.int8)


def quantize_rows(x: torch.Tensor) -> tuple:
    """fp32 tensor -> (q int8 same shape, scale fp32 [rows]). Raises ValueError on NaN / Inf."""
    x = x.detach().float()
    if x.numel() == 0:
        return torch.zeros(x.shape, dtype=torch.int8), torch.zeros(0, dtype=torch.float32)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("cannot quantise a tensor with NaN or Inf")
    rows = rows_of(x.shape)
    m = x.reshape(rows, -1)
    scale = (m.abs().amax(dim=1) / 127.0).float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(m / safe[:, None]).clamp_(-127, 127)
    q = torch.where(scale[:, None] > 0, q, torch.zeros_like(q))
    return q.to(torch.int8).reshape(x.shape), scale


def dequantize_rows(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(q int8, scale fp32 [rows]) -> fp32 tensor of q's shape (float32 multiply)."""
    if q.numel() == 0:
        return torch.zeros(q.shape, dtype=torch.float32)
    rows = scale.numel()
    return (q.reshape(rows, -1).float() * scale.float()[:, None]).reshape(q.shape)


def check_int8_tensor(name: str, q: torch.Tensor, scale: torch.Tensor) -> None:
    """Strict content rules; raises QuantError."""
    if scale.numel() != rows_of(q.shape):
        raise QuantError("shape_mismatch", f"{name}: {scale.numel()} scales for {rows_of(q.shape)} rows")
    if q.numel() == 0:
        return
    if not bool(torch.isfinite(scale).all()):
        raise QuantError("non_finite", f"{name}: scale contains NaN or Inf")
    if bool((scale < 0).any()):
        raise QuantError("int8_invalid", f"{name}: negative scale")
    if bool((q == -128).any()):
        raise QuantError("int8_invalid", f"{name}: int8 value -128 is not allowed (range is -127..127)")
    zero_rows = scale == 0
    if bool(zero_rows.any()) and bool((q.reshape(scale.numel(), -1)[zero_rows] != 0).any()):
        raise QuantError("int8_invalid", f"{name}: non-zero values in a row whose scale is 0")


def read_int8_delta(r: StReader, name: str, strict: bool = True) -> torch.Tensor:
    """Dequantised fp32 delta of one tensor from an int8 update (one tensor in memory)."""
    q, s = r.read(name), r.read(scale_name(name))
    if strict:
        check_int8_tensor(name, q, s)
    return dequantize_rows(q, s)


def read_delta_f32(r: StReader, name: str, delta_dtype: str) -> torch.Tensor:
    """fp32 view of one tensor of an update of either format."""
    if delta_dtype == "int8":
        return read_int8_delta(r, name, strict=False)       # the validator already checked it
    return r.read_f32(name)


def check_exact_size_int8(path, shapes: dict) -> Optional[str]:
    """None if the file holds exactly the int8 layout for ``shapes``, else a message."""
    try:
        want = build_entries(shapes)
    except QuantError as e:
        return str(e)
    try:
        with StReader(path) as r:
            if set(r.info) != {e[0] for e in want}:
                return "tensor names differ from the base"
            total = 0
            for name, shape, dt in want:
                code, got_shape, a, b = r.info[name]
                code_want = "I8" if dt == torch.int8 else "F32"
                if code != code_want:
                    return f"tensor {name} has dtype {code}, expected {code_want}"
                if got_shape != tuple(shape):
                    return f"tensor {name} has shape {got_shape}, expected {tuple(shape)}"
                total += _numel(shape) * (1 if dt == torch.int8 else 4)
            if r.size != r.data_start + total:
                return f"file is {r.size} bytes, expected {r.data_start + total}"
    except StreamIOError as e:
        return str(e)
    return None


# ------------------------------------------------------------------------------------------------- error feedback
class ResidualStore:
    """Per-run quantisation residual on the worker's disk (``<state_dir>/hd_residual``).

    One residual per (run, target). It is valid only if it was produced by the update that the current base's manifest
    lists as applied in its parent step: ``meta.base_version_id == base.parent_version_id`` and
    ``meta.update_id in base.applied_update_ids`` (plus run id, config hash, tensor shapes, file sha256). Any other
    situation (new run, missed rounds, update rejected, version rolled back, edited file) resets it to zero.
    """

    def __init__(self, state_dir, run_id: str, target: str = "dit"):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{run_id}_{target}")[:96]
        d = Path(state_dir) / "hd_residual"
        d.mkdir(parents=True, exist_ok=True)
        self.run_id, self.target = run_id, target
        self.file, self.meta_file = d / f"{safe}.safetensors", d / f"{safe}.json"
        self.tmp = d / f"{safe}.safetensors.tmp"
        self.reset_reason: Optional[str] = None

    def clear(self) -> None:
        for p in (self.file, self.meta_file, self.tmp):
            try:
                p.unlink()
            except OSError:
                pass

    def open_valid(self, base_manifest: dict, shapes: dict, config_sha: Optional[str]) -> Optional[StReader]:
        """A reader over the residual if it may be used for a round starting from ``base_manifest``; else None
        (``reset_reason`` says why; stale files are deleted)."""
        if not self.meta_file.is_file() or not self.file.is_file():
            self.reset_reason = "no residual yet"
            self.clear()
            return None
        try:
            meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        why = None
        if not isinstance(meta, dict) or meta.get("run_id") != self.run_id or meta.get("target") != self.target:
            why = "residual belongs to another run"
        elif meta.get("hd_config_sha") != config_sha:
            why = "model config changed"
        elif not base_manifest.get("parent_version_id") or meta.get("base_version_id") != base_manifest.get("parent_version_id"):
            why = "version mismatch (the base is not the direct successor of the residual's base)"
        elif meta.get("update_id") not in set(base_manifest.get("applied_update_ids") or []):
            why = "the update the residual belongs to was not applied"
        elif meta.get("sha256") != sha256_file(self.file):
            why = "residual file is damaged"
        r = None
        if why is None:
            try:
                r = StReader(self.file)
                if r.shapes() != {k: tuple(v) for k, v in shapes.items()} or set(r.dtypes().values()) != {"F32"}:
                    why = "residual shapes differ from the model"
            except StreamIOError:
                why = "residual file unreadable"
        if why is not None:
            if r is not None:
                r.close()
            self.reset_reason = why
            self.clear()
            return None
        self.reset_reason = None
        return r

    def writer(self, shapes: dict) -> StWriter:
        return StWriter(self.tmp, [(k, tuple(v)) for k, v in shapes.items()], torch.float32)

    def commit(self, writer_sha256: str, *, base_version_id: str, update_id: str, config_sha: Optional[str]) -> None:
        """Publish the residual written by ``writer()`` (call only after the update files are safely on disk)."""
        os.replace(self.tmp, self.file)
        meta = {"run_id": self.run_id, "target": self.target, "hd_config_sha": config_sha, "base_version_id": base_version_id,
                "update_id": update_id, "sha256": writer_sha256, "format": FORMAT}
        tmp = self.meta_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.meta_file)

    def discard_tmp(self) -> None:
        try:
            self.tmp.unlink()
        except OSError:
            pass


def rel_error(err2: float, ref2: float) -> float:
    return math.sqrt(err2 / ref2) if ref2 > 0 else 0.0
