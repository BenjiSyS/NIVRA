"""Global model version manifests (see docs/training/DISTRIBUTED.md section 0.1).

Depends only on torch / numpy / safetensors (runs on the Raspberry Pi controller).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from safetensors.torch import save_file, load_file

SCHEMA = "nivra.model_version/1"
WEIGHTS_FILE = "weights.safetensors"
OUTER_FILE = "outer_state.safetensors"
MANIFEST_FILE = "model_version.json"


class VersionError(Exception):
    pass


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


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
        m = json.loads(Path(path).read_text())
    except Exception as e:  # noqa: BLE001
        raise VersionError(f"cannot read version manifest {path}: {e}") from e
    if m.get("schema") != SCHEMA:
        raise VersionError(f"unsupported version manifest schema {m.get('schema')!r}")
    for k in ("version_id", "round", "weights_sha256", "applied_update_ids", "architecture_version", "vocab_hash"):
        if k not in m:
            raise VersionError(f"version manifest missing {k}")
    return m


def load_weights(manifest: dict, weights_path) -> dict:
    """Verify the weights file against the manifest hash and return tensors."""
    if sha256_file(weights_path) != manifest["weights_sha256"]:
        raise VersionError("weights sha256 does not match version manifest")
    return load_file(str(weights_path))
