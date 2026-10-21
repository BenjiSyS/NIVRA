"""Global model version manifests (see docs/training/DISTRIBUTED.md section 0.1).

Depends only on torch / numpy / safetensors (runs on the Raspberry Pi controller).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Optional

from safetensors import safe_open
from safetensors.torch import save_file, load_file

SCHEMA = "nivra.model_version/1"
WEIGHTS_FILE = "weights.safetensors"
OUTER_FILE = "outer_state.safetensors"
MANIFEST_FILE = "model_version.json"


MAX_JSON_BYTES = 1 << 20            # every JSON input is size-checked before it is read
DEFAULT_MAX_TENSOR_BYTES = 1 << 30
_ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class VersionError(Exception):
    pass


class ManifestTooLarge(VersionError):
    pass


def read_json_limited(path, max_bytes: int = MAX_JSON_BYTES):
    """Parse a JSON file after checking its size (never reads a file of unknown size into memory)."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as e:
        raise VersionError(f"cannot read {p}: {e}") from e
    if size > max_bytes:
        raise ManifestTooLarge(f"{p.name}: {size} bytes exceeds the {max_bytes} byte limit for JSON inputs")
    try:
        with open(p, "rb") as f:
            return json.loads(f.read(max_bytes + 1).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise VersionError(f"cannot parse {p.name}: {e}") from e


def inspect_safetensors(path) -> dict:
    """Header only: {name: (shape tuple, dtype string like 'F32')}.  No tensor data is read."""
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            out[k] = (tuple(sl.get_shape()), sl.get_dtype())
    return out


def check_tensor_file(path, expected_shapes: Optional[dict] = None, dtype: Optional[str] = None,
                      max_bytes: int = DEFAULT_MAX_TENSOR_BYTES) -> dict:
    """Validate a safetensors file from its header BEFORE any tensor is loaded.  Raises VersionError."""
    try:
        size = Path(path).stat().st_size
    except OSError as e:
        raise VersionError(f"cannot stat {path}: {e}") from e
    if size > max_bytes:
        raise VersionError(f"{Path(path).name}: {size} bytes > limit {max_bytes}")
    try:
        hdr = inspect_safetensors(path)
    except Exception as e:  # noqa: BLE001
        raise VersionError(f"{Path(path).name}: unreadable safetensors header: {e}") from e
    total = 0
    for k, (shape, dt) in hdr.items():
        n = 1
        for d in shape:
            n *= d
        total += n * _ITEMSIZE.get(dt, 8)
        if dtype is not None and dt != dtype:
            raise VersionError(f"{Path(path).name}: tensor {k} has dtype {dt}, expected {dtype}")
    if total > max_bytes:
        raise VersionError(f"{Path(path).name}: declared tensor bytes {total} > limit {max_bytes}")
    if expected_shapes is not None:
        exp = {k: tuple(v) for k, v in expected_shapes.items()}
        if set(exp) != set(hdr):
            raise VersionError(f"{Path(path).name}: tensor names differ from the expected set")
        for k, (shape, _) in hdr.items():
            if shape != exp[k]:
                raise VersionError(f"{Path(path).name}: tensor {k} has shape {shape}, expected {exp[k]}")
    return hdr


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _l2(t) -> float:
    return float(t.detach().double().pow(2).sum().sqrt())


def make_version_id(rnd: int, weights_sha: str) -> str:
    return f"v{rnd:06d}-{weights_sha[:12]}"


def write_version(out_dir, weights: dict, *, config: dict, config_name: str, arch_version: str,
                  tokenizer_version: str, vocab_hash: str, vae_sha256: str, rnd: int = 0,
                  parent_version_id: Optional[str] = None, applied_update_ids=(),
                  outer_state: Optional[dict] = None, version_id: Optional[str] = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in weights.items()}, str(out / WEIGHTS_FILE))
    wsha = sha256_file(out / WEIGHTS_FILE)
    osha = None
    if outer_state is not None:
        save_file({k: v.detach().cpu().contiguous() for k, v in outer_state.items()}, str(out / OUTER_FILE))
        osha = sha256_file(out / OUTER_FILE)
    manifest = {
        "schema": SCHEMA,
        "version_id": version_id or make_version_id(rnd, wsha),
        "round": int(rnd),
        "parent_version_id": parent_version_id,
        "architecture_version": arch_version,
        "config_name": config_name,
        "config": config,
        "tokenizer_version": tokenizer_version,
        "vocab_hash": vocab_hash,
        "param_count": int(sum(v.numel() for v in weights.values())),
        "tensor_l2": {k: _l2(v) for k, v in weights.items()},
        "weights_l2": math.sqrt(sum(_l2(v) ** 2 for v in weights.values())),
        "tensor_shapes": {k: list(v.shape) for k, v in weights.items()},
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": wsha,
        "outer_state_file": OUTER_FILE if outer_state is not None else None,
        "outer_state_sha256": osha,
        "vae_sha256": vae_sha256,
        "applied_update_ids": list(applied_update_ids),
        "created_unix": int(time.time()),
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def read_manifest(path) -> dict:
    try:
        m = read_json_limited(path)
    except ManifestTooLarge:
        raise
    except Exception as e:  # noqa: BLE001
        raise VersionError(f"cannot read version manifest {path}: {e}") from e
    if not isinstance(m, dict):
        raise VersionError("version manifest must be a JSON object")
    if m.get("schema") != SCHEMA:
        raise VersionError(f"unsupported version manifest schema {m.get('schema')!r}")
    for k in ("version_id", "round", "weights_sha256", "applied_update_ids", "architecture_version", "vocab_hash"):
        if k not in m:
            raise VersionError(f"version manifest missing {k}")
    return m


def load_weights(manifest: dict, weights_path, max_bytes: int = DEFAULT_MAX_TENSOR_BYTES) -> dict:
    """Verify size, hash and header (names/shapes vs manifest) of the weights file, then return tensors."""
    check_tensor_file(weights_path, None, None, max_bytes)           # size + header sanity before hashing/loading
    if sha256_file(weights_path) != manifest["weights_sha256"]:
        raise VersionError("weights sha256 does not match version manifest")
    check_tensor_file(weights_path, manifest.get("tensor_shapes"), None, max_bytes)
    return load_file(str(weights_path))
