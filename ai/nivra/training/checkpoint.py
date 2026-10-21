"""Checkpointing without pickle: safetensors tensors + JSON manifest with sha256 of every file.

Layout of one checkpoint directory ``ckpt_<step:08d>``::

    manifest.json            structure, scalars, compat info, per-file sha256, manifest_digest
    <group>.safetensors      one file per tensor group (model, ema, optimizer, rng_torch, ...)
    state.json               python/numpy RNG states, scheduler + grad-scaler state, sampler position

Writes go to a temporary sibling directory and are published with an atomic rename.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import save_file, load_file

SCHEMA = "nivra.checkpoint/1"


class CheckpointError(Exception):
    """Corrupt / unreadable checkpoint."""


class IncompatibleCheckpoint(CheckpointError):
    """Checkpoint is intact but not compatible with the requested architecture/tokenizer."""


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _digest(manifest: dict) -> str:
    m = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- RNG / optimizer helpers
def capture_rng() -> tuple:
    """Returns (tensors, json_state)."""
    tensors = {"torch_cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        for i, s in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"torch_cuda_{i}"] = s.clone()
    ns = np.random.get_state()
    py = random.getstate()
    js = {"python": [py[0], list(py[1]), py[2]],
          "numpy": [ns[0], ns[1].tolist(), int(ns[2]), int(ns[3]), float(ns[4])]}
    return tensors, js


def restore_rng(tensors: dict, js: dict) -> None:
    torch.set_rng_state(tensors["torch_cpu"].to(torch.uint8))
    if torch.cuda.is_available():
        states = [tensors[k].to(torch.uint8) for k in sorted(k for k in tensors if k.startswith("torch_cuda_"))]
        if states:
            torch.cuda.set_rng_state_all(states)
    py = js["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = js["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def generator_state_tensor(gen: torch.Generator) -> torch.Tensor:
    return gen.get_state().clone()


def optimizer_to_flat(opt: torch.optim.Optimizer) -> tuple:
    sd = opt.state_dict()
    tensors = {}
    for i, st in sd["state"].items():
        for k, v in st.items():
            tensors[f"state.{i}.{k}"] = v.detach().cpu().contiguous() if torch.is_tensor(v) else torch.tensor(float(v))
    groups = [{k: v for k, v in g.items()} for g in sd["param_groups"]]
    return tensors, groups


def optimizer_from_flat(opt: torch.optim.Optimizer, tensors: dict, groups: list) -> None:
    state: dict = {}
    for key, v in tensors.items():
        _, i, k = key.split(".", 2)
        state.setdefault(int(i), {})[k] = v
    opt.load_state_dict({"state": state, "param_groups": groups})


def module_tensors(module: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}


# ---------------------------------------------------------------- save / load
def save_checkpoint(run_dir, step: int, groups: dict, state: dict, compat: dict, config: dict,
                    epoch: int = 0, protected: bool = False, extra: Optional[dict] = None) -> Path:
    """groups: {name: {tensor_name: Tensor}}. state: JSON-able dict (rng, scheduler, sampler, scaler...)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / f"ckpt_{step:08d}"
    tmp = run_dir / f".tmp_ckpt_{step:08d}_{os.getpid()}_{int(time.time() * 1000)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        files = {}
        for name, tensors in groups.items():
            p = tmp / f"{name}.safetensors"
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(p))
            files[p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        sp = tmp / "state.json"
        sp.write_text(json.dumps(state, sort_keys=True))
        files[sp.name] = {"sha256": sha256_file(sp), "bytes": sp.stat().st_size}
        manifest = {"schema": SCHEMA, "step": int(step), "epoch": int(epoch), "created_unix": time.time(),
                    "protected": bool(protected), "files": files, "compat": compat, "config": config,
                    "extra": extra or {}}
        manifest["manifest_digest"] = _digest(manifest)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        # fsync directory contents best-effort
        for f in tmp.iterdir():
            with open(f, "r+b") as fh:
                os.fsync(fh.fileno())
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)  # atomic publish
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def read_manifest(ckpt_dir) -> dict:
    p = Path(ckpt_dir) / "manifest.json"
    try:
        m = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        raise CheckpointError(f"cannot read manifest in {ckpt_dir}: {e}") from e
    if m.get("schema") != SCHEMA:
        raise CheckpointError(f"unknown checkpoint schema {m.get('schema')}")
    if _digest(m) != m.get("manifest_digest"):
        raise CheckpointError("manifest digest mismatch (manifest was modified or corrupted)")
    return m


def verify_checkpoint(ckpt_dir) -> dict:
    m = read_manifest(ckpt_dir)
    for name, info in m["files"].items():
        p = Path(ckpt_dir) / name
        if not p.exists():
            raise CheckpointError(f"missing file {name}")
        if p.stat().st_size != info["bytes"] or sha256_file(p) != info["sha256"]:
            raise CheckpointError(f"checksum mismatch for {name}")
    return m


def check_compat(manifest: dict, expect: Optional[dict]) -> None:
    if not expect:
        return
    have = manifest["compat"]
    for k, v in expect.items():
        if have.get(k) != v:
            raise IncompatibleCheckpoint(f"incompatible checkpoint: {k} is {have.get(k)!r}, expected {v!r}")


def load_checkpoint(ckpt_dir, expect_compat: Optional[dict] = None) -> tuple:
    """Verify checksums, check compatibility, return (manifest, groups, state)."""
    m = verify_checkpoint(ckpt_dir)
    check_compat(m, expect_compat)
    groups = {}
    for name in m["files"]:
        if name.endswith(".safetensors"):
            groups[name[: -len(".safetensors")]] = load_file(str(Path(ckpt_dir) / name))
    state = json.loads((Path(ckpt_dir) / "state.json").read_text())
    return m, groups, state


# ---------------------------------------------------------------- retention
def list_checkpoints(run_dir) -> list:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    out = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("ckpt_") and not p.name.startswith(".tmp"):
            try:
                out.append((int(p.name.split("_")[1]), p))
            except ValueError:
                pass
    return sorted(out)


def newest_valid(run_dir) -> Optional[Path]:
    for _, p in reversed(list_checkpoints(run_dir)):
        try:
            verify_checkpoint(p)
            return p
        except CheckpointError:
            continue
    return None


def protect(ckpt_dir) -> None:
    """Mark a checkpoint as protected (retention never deletes it)."""
    (Path(ckpt_dir) / "PROTECTED").write_text("protected\n")


def apply_retention(run_dir, keep_last: int = 3) -> list:
    """Delete old checkpoints. Never deletes the newest valid checkpoint, protected ones,
    or the ``keep_last`` newest checkpoints. Returns deleted paths."""
    cks = list_checkpoints(run_dir)
    keep = set()
    newest = newest_valid(run_dir)
    if newest is not None:
        keep.add(newest)
    for _, p in cks[-max(keep_last, 1):]:
        keep.add(p)
    deleted = []
    for _, p in cks:
        if p in keep or (p / "PROTECTED").exists():
            continue
        shutil.rmtree(p, ignore_errors=True)
        deleted.append(p)
    for tmp in Path(run_dir).glob(".tmp_ckpt_*"):
        # stale temp dirs from crashed writers (older than 10 minutes)
        if time.time() - tmp.stat().st_mtime > 600:
            shutil.rmtree(tmp, ignore_errors=True)
    return deleted
