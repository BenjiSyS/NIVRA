"""Executor for protocol nivra.exec/1 (docs/architecture/EXECUTOR_PROTOCOL.md).

Usage: python -m nivra.cli exec --job <job.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

PROTOCOL = "nivra.exec/1"
EXIT_OK, EXIT_INTERNAL, EXIT_SPEC, EXIT_STOPPED, EXIT_OOM, EXIT_INPUT = 0, 1, 2, 3, 4, 5
TOP_KEYS = {"protocol", "job_type", "job_id", "attempt_id", "params", "inputs", "output_dir", "stop_file", "limits"}
LIMIT_KEYS = {"max_seconds", "max_ram_mb", "max_vram_mb"}


class SpecError(Exception):
    code, exit = "INVALID_SPEC", EXIT_SPEC


class InputInvalid(Exception):
    code, exit = "INPUT_INVALID", EXIT_INPUT


class JobStopped(Exception):
    code, exit = "STOPPED", EXIT_STOPPED


# ---------------------------------------------------------------- events
class Emitter:
    def __init__(self, stream=None):
        self.out = stream or sys.stdout

    def _emit(self, obj):
        self.out.write(json.dumps(obj, ensure_ascii=True, separators=(",", ":")) + "\n")
        self.out.flush()

    def log(self, message, level="info"):
        self._emit({"event": "log", "level": level, "message": str(message)})

    def progress(self, stage, fraction, step=None, total=None, message=None):
        e = {"event": "progress", "stage": stage, "fraction": max(0.0, min(1.0, float(fraction)))}
        if step is not None:
            e["step"] = step
        if total is not None:
            e["total"] = total
        if message:
            e["message"] = message
        self._emit(e)

    def metric(self, name, value, step=None):
        e = {"event": "metric", "name": name, "value": float(value)}
        if step is not None:
            e["step"] = step
        self._emit(e)

    def artifact(self, kind, path):
        p = Path(path)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        self._emit({"event": "artifact", "kind": kind, "path": str(p), "sha256": h.hexdigest(), "bytes": p.stat().st_size})

    def result(self, summary):
        s = json.dumps(summary, ensure_ascii=True, default=str)
        if len(s) > 32 * 1024:
            summary = {"truncated": True, "keys": list(summary)[:50]}
        self._emit({"event": "result", "status": "ok", "summary": json.loads(json.dumps(summary, default=str))})

    def error(self, code, message):
        self._emit({"event": "error", "code": code, "message": str(message)[:4000]})


# ---------------------------------------------------------------- param validation
def _validate(params: Any, schema: dict, where: str = "params") -> dict:
    if not isinstance(params, dict):
        raise SpecError(f"{where} must be an object")
    unknown = set(params) - set(schema)
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}")
    out = {}
    for k, s in schema.items():
        if k not in params:
            if s.get("req"):
                raise SpecError(f"{where}.{k} is required")
            out[k] = s.get("default")
            continue
        v = params[k]
        t = s["type"]
        if t == "int":
            if isinstance(v, bool) or not isinstance(v, int):
                raise SpecError(f"{where}.{k} must be an integer")
        elif t == "float":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                raise SpecError(f"{where}.{k} must be a finite number")
            v = float(v)
        elif t == "str":
            if not isinstance(v, str):
                raise SpecError(f"{where}.{k} must be a string")
        elif t == "bool":
            if not isinstance(v, bool):
                raise SpecError(f"{where}.{k} must be a boolean")
        elif t == "str_or_obj":
            if not isinstance(v, (str, dict)):
                raise SpecError(f"{where}.{k} must be a string or object")
        elif t == "floatlist":
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) or len(v) > 4096:
                raise SpecError(f"{where}.{k} must be a list of numbers")
        elif t == "obj":
            v = _validate(v, s["schema"], f"{where}.{k}")
        elif t == "objlist":
            if not isinstance(v, list) or len(v) > s.get("maxlen", 256) or not all(isinstance(x, dict) for x in v):
                raise SpecError(f"{where}.{k} must be a list of objects")
            v = [_validate(x, s["schema"], f"{where}.{k}[{i}]") for i, x in enumerate(v)]
        if "min" in s and not (v >= s["min"]):
            raise SpecError(f"{where}.{k} must be >= {s['min']}")
        if "max" in s and not (v <= s["max"]):
            raise SpecError(f"{where}.{k} must be <= {s['max']}")
        if "exclusive_min" in s and not (v > s["exclusive_min"]):
            raise SpecError(f"{where}.{k} must be > {s['exclusive_min']}")
        if "choices" in s and v not in s["choices"]:
            raise SpecError(f"{where}.{k} must be one of {s['choices']}")
        if "maxlen" in s and t != "objlist" and len(v) > s["maxlen"]:
            raise SpecError(f"{where}.{k} longer than {s['maxlen']}")
        if t == "str" and s.get("minlen", 0) > len(v):
            raise SpecError(f"{where}.{k} too short")
        out[k] = v
    return out


def I(default=None, req=False, **kw):
    return {"type": "int", "default": default, "req": req, **kw}


def F(default=None, req=False, **kw):
    return {"type": "float", "default": default, "req": req, **kw}


def S(default=None, req=False, **kw):
    return {"type": "str", "default": default, "req": req, **kw}


def B(default=None, **kw):
    return {"type": "bool", "default": default, **kw}


STAGE_SPEC = {"type": "obj", "req": True, "schema": {"h": I(req=True, min=32, max=4096), "w": I(req=True, min=32, max=4096),
                                                "frames": I(req=True, min=1, max=1025), "fps": I(8, min=1, max=120)}}

SCHEMAS = {
    "ai.init_model": dict(
        params={"config": {"type": "str_or_obj", "default": None}, "seed": I(0, min=0), "from_checkpoint": S(None),
                "use_ema": B(True), "vae_checkpoint": S(None), "version_id": S(None, maxlen=128)},
        inputs={"from_checkpoint": False, "vae_checkpoint": False}),
    "ai.dataset_shard": dict(
        params={"seed": I(req=True, min=0), "start": I(req=True, min=0), "count": I(req=True, min=1, max=4096),
                "frames": I(16, min=2, max=32), "size": I(64, min=16, max=128)},
        inputs={}),
    "ai.local_train": dict(
        params={"update_id": S(req=True, minlen=1, maxlen=128),
                "shard": {"type": "obj", "req": True, "schema": {"seed": I(req=True, min=0), "start": I(req=True, min=0),
                                                                  "count": I(req=True, min=8, max=4096)}},
                "inner_steps": I(30, min=1, max=2000), "batch_size": I(8, min=1, max=64),
                "lr": F(3e-4, exclusive_min=0.0, max=1e-2), "weight_decay": F(0.0, min=0.0, max=0.2),
                "grad_clip": F(1.0, exclusive_min=0.0, max=100.0), "seed": I(0, min=0), "eval_clips": I(16, min=0, max=256)},
        inputs={"base_weights": True, "base_manifest": True, "vae_weights": True}),
    "ai.validate_update": dict(
        params={"max_norm_ratio": F(5.0, min=1.0), "max_bytes": I(1 << 30, min=1), "max_norm": F(None, exclusive_min=0.0), "max_rel_norm": F(0.5, exclusive_min=0.0),
                "peer_norms": {"type": "floatlist", "default": None}, "expected_base_version_id": S(None, maxlen=128)},
        inputs={"update": True, "update_manifest": True, "base_manifest": True, "base_weights": False}),
    "ai.aggregate": dict(
        params={"outer_lr": F(0.7, exclusive_min=0.0, max=10.0), "momentum": F(0.9, min=0.0, max=0.9999), "nesterov": B(True),
                "aggregation": S("auto", choices=["auto", "mean", "trimmed_mean"]), "trim_ratio": F(0.2, min=0.0, max=0.45),
                "max_norm_ratio": F(5.0, min=1.0), "max_bytes": I(1 << 30, min=1), "version_id": S(None, maxlen=128),
                "max_rel_norm": F(0.5, exclusive_min=0.0), "eval_heldout": B(False)},
        inputs={"base_weights": True, "base_manifest": True, "outer_state": False, "updates": True, "vae_weights": False}),
    "ai.evaluate": dict(
        params={"mode": S(req=True, choices=["adherence", "reconstruction", "detector"]), "num_prompts": I(16, min=1, max=200),
                "seeds_per_prompt": I(1, min=1, max=8), "steps": I(20, min=2, max=100), "cfg_scale": F(3.0, min=0.0, max=20.0),
                "num_clips": I(64, min=1, max=512), "eval_seed": I(777), "prompts_file": S(None)},
        inputs={"weights": False, "manifest": False, "vae_weights": False}),
    "ai.generate": dict(
        params={"prompt": S(req=True, minlen=1, maxlen=200), "seed": I(0, min=0), "steps": I(30, min=1, max=200),
                "cfg_scale": F(3.0, min=0.0, max=20.0), "fps": I(8, min=1, max=60), "upscale": I(4, min=1, max=8),
                "tile": I(0, min=0, max=16), "name": S("video", maxlen=64),
                # HD models only (rejected for small bundles); None = the bundle's own default
                "width": I(None, min=16, max=4096), "height": I(None, min=16, max=4096),
                "seconds": F(None, exclusive_min=0.0, max=60.0), "sampler": S(None, choices=["euler", "ab2"]),
                "fp16": B(None), "offload": S(None, choices=["none", "blocks"]),
                # style tag (canonical, alias or "none"); None = the HD manifest's default_style
                "style": S(None, maxlen=64)},
        inputs={"weights": True, "manifest": True, "vae_weights": True}),
    # long videos (docs/architecture/LONG_VIDEO.md)
    "ai.draft": dict(
        params={"prompt": S(req=True, minlen=1, maxlen=200), "seed": I(0, min=0), "style": S(None, maxlen=64),
                "sampler": S(None, choices=["euler", "ab2"]), "seconds": F(req=True, exclusive_min=0.0, max=10800.0),
                "fps": I(30, min=1, max=60), "draft_width": I(256, min=32, max=1024), "draft_height": I(144, min=32, max=1024),
                "segment_frames": I(241, min=9, max=4097), "steps": I(None, min=1, max=200),
                "cfg_scale": F(None, min=0.0, max=20.0), "name": S("draft", maxlen=64), "fp16": B(None),
                "offload": S(None, choices=["none", "blocks"])},
        inputs={"weights": True, "manifest": True, "vae_weights": True}),
    "ai.refine_segment": dict(
        params={"prompt": S(req=True, minlen=1, maxlen=200), "seed": I(0, min=0), "style": S(None, maxlen=64),
                "sampler": S(None, choices=["euler", "ab2"]), "width": I(req=True, min=32, max=2560),
                "height": I(req=True, min=32, max=1440), "fps": I(30, min=1, max=60), "index": I(0, min=0),
                "start": I(req=True, min=0), "end": I(req=True, min=0), "strength": F(0.55, exclusive_min=0.0, max=1.0),
                "steps": I(None, min=1, max=200), "cfg_scale": F(None, min=0.0, max=20.0),
                "name": S("segment", maxlen=64), "fp16": B(None), "offload": S(None, choices=["none", "blocks"])},
        inputs={"weights": True, "manifest": True, "vae_weights": True, "draft_video": True}),
    "ai.stitch": dict(
        params={"fps": I(30, min=1, max=60), "crossfade_frames": I(4, min=0, max=16), "count": I(req=True, min=1, max=4096),
                "name": S("video", maxlen=64)},
        inputs={}, input_regex=r"segment_\d{3,}"),
    # HD distributed training (docs/architecture/HD_TRAINING.md)
    "ai.hd_encode": dict(
        params={"dataset_id": S(None, maxlen=128), "target_dataset_id": S(None, maxlen=128), "shard_idx": I(0, min=0),
                "shard_sha256": S(req=True, minlen=64, maxlen=64), "latent": S("mean", choices=["mean"]),
                "stage": STAGE_SPEC,
                "style_dropout": F(0.1, min=0.0, max=1.0), "seed": I(0, min=0), "audit": B(False)},
        inputs={"raw_shard": True, "vae_weights": True, "vae_manifest": True, "text_encoder_weights": True,
                "tokenizer": True}),
    "ai.hd_train": dict(
        params={"update_id": S(req=True, minlen=1, maxlen=128), "run_id": S(None, maxlen=128), "round": I(0, min=0),
                "round_batch": I(0, min=0), "base_version_id": S(None, maxlen=128), "target": S("dit", choices=["dit", "vae"]),
                "hd_config": {"type": "str_or_obj", "default": None}, "hd_config_sha": S(None, maxlen=128),
                "stage": STAGE_SPEC,
                "inner_steps": I(req=True, min=1, max=100000), "batch_size": I(2, min=1, max=64),
                "grad_accum": I(1, min=1, max=256), "lr": F(req=True, exclusive_min=0.0, max=1e-2),
                "warmup_steps": I(20, min=0, max=100000), "weight_decay": F(0.0, min=0.0, max=0.2),
                "grad_clip": F(1.0, min=0.0, max=100.0), "cfg_dropout": F(0.1, min=0.0, max=1.0), "seed": I(0, min=0),
                "shards": {"type": "objlist", "default": None, "maxlen": 64,
                           "schema": {"artifact_id": S(req=True, maxlen=128), "sha256": S(req=True, minlen=64, maxlen=64)}},
                "delta_dtype": S("float16", choices=["float16"])},
        inputs={"base_train_weights": True, "base_manifest": True}, input_regex=r"data_shard_\d{3,}"),
    "ai.hd_eval": dict(
        params={"version_id": S(None, maxlen=128), "triples_seed": I(0), "n": I(8, min=1, max=256),
                "timestep_buckets": I(4, min=1, max=64)},
        inputs={"weights": True, "manifest": True}, input_regex=r"val_shard_\d{3,}"),
    "ai.hd_aggregate": dict(
        params={"outer_lr": F(0.7, exclusive_min=0.0, max=10.0), "momentum": F(0.9, min=0.0, max=0.9999), "nesterov": B(True),
                "version_id": S(None, maxlen=128), "bound_c": F(3.0, exclusive_min=0.0, max=100.0),
                "inner_steps": I(req=True, min=1, max=100000), "lr": F(req=True, exclusive_min=0.0, max=1e-2),
                "eval_gate": S("off", choices=["pi", "worker", "off"]), "eval_triples_seed": I(0),
                "aggregation": S("auto", choices=["auto", "mean", "trimmed_mean"]), "trim_ratio": F(0.2, min=0.0, max=0.45),
                "max_norm_ratio": F(5.0, min=1.0), "max_rel_norm": F(0.5, exclusive_min=0.0),
                "clip_factor": F(2.0, exclusive_min=0.0), "eval_n": I(8, min=1, max=64), "eval_buckets": I(4, min=1, max=64)},
        inputs={"base_weights": True, "base_manifest": True, "outer_state": False, "updates": True, "eval_latents": False}),
}


# ---------------------------------------------------------------- job context
class Ctx:
    def __init__(self, job: dict, workdir: Path, em: Emitter):
        self.job, self.workdir, self.em = job, workdir, em
        self.out = Path(job["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop_file = Path(job["stop_file"]) if job.get("stop_file") else None
        self.t0 = time.time()
        self.max_seconds = (job.get("limits") or {}).get("max_seconds")
        self.inputs: dict = {}

    def should_stop(self) -> bool:
        if self.stop_file is not None and self.stop_file.exists():
            return True
        return bool(self.max_seconds) and (time.time() - self.t0) > self.max_seconds

    def check_stop(self):
        if self.should_stop():
            raise JobStopped("stop file present or time limit reached")

    def path(self, key: str, required: bool = True) -> Optional[Path]:
        v = self.inputs.get(key)
        if v is None:
            if required:
                raise SpecError(f"inputs.{key} is required")
            return None
        p = Path(v)
        if not p.is_absolute():
            p = self.workdir / p
        if not p.exists():
            raise InputInvalid(f"inputs.{key}: {p} does not exist")
        return p


def _device():
    from .config import get_device
    return get_device()


# ---------------------------------------------------------------- handlers
def h_init_model(ctx: Ctx, p: dict) -> dict:
    from .config import load_config, build_gen, ConfigError
    from .inference.bundle import init_random_bundle
    from .training.pipeline import export_weights
    from .training import checkpoint as ck
    import torch
    fc = p["from_checkpoint"] or ctx.inputs.get("from_checkpoint")
    vc = p["vae_checkpoint"] or ctx.inputs.get("vae_checkpoint")
    if bool(fc) == (p["config"] is not None):
        raise SpecError("exactly one of params.config / from_checkpoint is required")
    vae_path = None
    if vc:
        vp = Path(vc)
        vp = vp if vp.is_absolute() else ctx.workdir / vp
        cands = [vp, vp / "vae.safetensors", vp / "export" / "vae.safetensors"]
        vae_path = next((c for c in cands if c.is_file()), None)
        if vae_path is None:
            raise InputInvalid(f"vae_checkpoint {vc}: no vae.safetensors found")
    if fc:
        d = Path(fc)
        d = d if d.is_absolute() else ctx.workdir / d
        try:
            m, groups, _ = ck.load_checkpoint(d)
        except ck.CheckpointError as e:
            raise InputInvalid(str(e))
        if m["compat"].get("kind") != "gen":
            raise InputInvalid("checkpoint is not a generator checkpoint")
        cfg = load_config(m["config"])
        w = groups["ema"] if (p["use_ema"] and "ema" in groups) else groups["model"]
        if vae_path is None:
            raise InputInvalid("from_checkpoint requires vae_checkpoint (the VAE the generator was trained against)")
        if ck.sha256_file(vae_path) != m["compat"]["vae_sha256"]:
            raise InputInvalid("vae_checkpoint does not match the VAE of the generator checkpoint")
        manifest = export_weights(cfg, w, vae_path, ctx.out)
    else:
        try:
            cfg = load_config(p["config"])
        except ConfigError as e:
            raise SpecError(str(e))
        torch.manual_seed(p["seed"])
        if vae_path is None:
            from .distributed import version as ver
            from .text.tokenizer import default_tokenizer, TOKENIZER_VERSION
            model = build_gen(cfg)
            tok = default_tokenizer()
            manifest = ver.write_version(ctx.out, model.state_dict(), config=cfg, config_name=cfg["name"],
                                         arch_version=cfg["arch_version"], tokenizer_version=TOKENIZER_VERSION,
                                         vocab_hash=tok.vocab_hash, vae_sha256=None, rnd=0)
        else:
            manifest = init_random_bundle(cfg, vae_path, ctx.out, p["seed"])
    if p["version_id"]:
        from .distributed import version as ver
        manifest["version_id"] = p["version_id"]
        (ctx.out / ver.MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    ctx.em.artifact("model_version", ctx.out / "model_version.json")
    ctx.em.artifact("model_weights", ctx.out / "weights.safetensors")
    if (ctx.out / "vae.safetensors").exists():
        ctx.em.artifact("vae_weights", ctx.out / "vae.safetensors")
    return {k: manifest[k] for k in ("version_id", "architecture_version", "tokenizer_version", "vocab_hash",
                                     "param_count", "weights_sha256")}


def h_dataset_shard(ctx: Ctx, p: dict) -> dict:
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from .data.synthetic import SynthConfig, make_sample, manifest as dmanifest
    scfg = SynthConfig(frames=p["frames"], size=p["size"])
    vids, caps, specs = [], [], []
    for i in range(p["count"]):
        if i % 16 == 0:
            ctx.check_stop()
            ctx.em.progress("generate", i / p["count"], i, p["count"])
        s = make_sample(p["seed"], p["start"] + i, scfg)
        vids.append(s.video)
        caps.append(s.captions)
        specs.append(s.spec.to_dict())
    arr = np.ascontiguousarray(np.stack(vids), dtype=np.uint8)
    # content_sha256 = sha256 of exactly the bytes of the `videos` tensor stored in shard.safetensors
    content_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    # deterministic file: no metadata, no timestamps; safetensors orders the header by (dtype, name)
    save_file({"videos": torch.from_numpy(arr)}, str(ctx.out / "shard.safetensors"))
    from .distributed.version import sha256_file
    file_sha = sha256_file(ctx.out / "shard.safetensors")
    m = dmanifest(p["seed"], p["start"], p["count"], scfg)
    m.update(content_sha256=content_sha, shard_file_sha256=file_sha, captions=caps, specs=specs)
    (ctx.out / "shard.json").write_text(json.dumps(m, ensure_ascii=False))
    ctx.em.artifact("dataset_shard", ctx.out / "shard.safetensors")
    ctx.em.artifact("dataset_manifest", ctx.out / "shard.json")
    return {"count": p["count"], "fingerprint": m["fingerprint"], "content_sha256": content_sha,
            "shard_file_sha256": file_sha}


def h_local_train(ctx: Ctx, p: dict) -> dict:
    from .distributed.local import local_train
    from .inference.bundle import BundleError
    try:
        meta = local_train(ctx.path("base_weights"), ctx.path("base_manifest"), ctx.path("vae_weights"), p["shard"],
                           p["update_id"], ctx.out, inner_steps=p["inner_steps"], batch_size=p["batch_size"], lr=p["lr"],
                           weight_decay=p["weight_decay"], grad_clip=p["grad_clip"], seed=p["seed"],
                           eval_clips=p["eval_clips"], device=_device(), should_stop=ctx.should_stop,
                           on_metric=lambda n, v, s: (ctx.em.metric(n, v, s),
                                                      ctx.em.progress("train", s / p["inner_steps"], s, p["inner_steps"])))
    except BundleError as e:
        raise InputInvalid(str(e))
    except Exception as e:  # noqa: BLE001
        from .inference.sampler import Stopped
        if isinstance(e, Stopped):
            raise JobStopped("stopped")
        raise
    for k in ("loss_before", "loss_after"):
        if meta.get(k) is not None:
            ctx.em.metric(k, meta[k])
    ctx.em.artifact("update", ctx.out / "update.json")
    ctx.em.artifact("update_weights", ctx.out / "update.safetensors")
    return {k: meta[k] for k in ("update_id", "base_version_id", "inner_steps", "samples", "loss_first", "loss_last",
                                 "loss_before", "loss_after", "delta_l2", "update_sha256")}


def h_validate_update(ctx: Ctx, p: dict) -> dict:
    from .distributed import version as ver
    from .distributed.update import validate_update
    try:
        bm = ver.read_manifest(ctx.path("base_manifest"))
    except ver.VersionError as e:
        raise InputInvalid(str(e))
    bw = None
    bwp = ctx.path("base_weights", required=False)
    if bwp is not None:
        try:
            bw = ver.load_weights(bm, bwp)
        except Exception as e:  # noqa: BLE001
            raise InputInvalid(str(e))
    v = validate_update(ctx.path("update"), ctx.path("update_manifest"), bm, peer_norms=p["peer_norms"],
                        max_norm_ratio=p["max_norm_ratio"], max_bytes=p["max_bytes"], max_norm=p["max_norm"],
                        expected_base_version_id=p["expected_base_version_id"], base_weights=bw,
                        max_rel_norm=p["max_rel_norm"])
    if "manifest_too_large" in v.codes:
        raise InputInvalid(v.reason_strings()[0])
    return {"valid": v.valid, "reasons": v.reason_strings(), "reason_codes": v.codes, "l2_norm": v.l2_norm,
            "update_id": v.update_id, "num_tensors": v.num_tensors, "bytes": v.bytes}


def h_aggregate(ctx: Ctx, p: dict) -> dict:
    from .distributed import version as ver
    from .distributed.aggregate import aggregate
    ups = ctx.inputs.get("updates")
    if not isinstance(ups, list) or not (1 <= len(ups) <= 64):
        raise SpecError("inputs.updates must be a list of 1..64 entries")
    entries = []
    for i, u in enumerate(ups):
        if not isinstance(u, dict) or set(u) - {"path", "manifest_path"} or "path" not in u or "manifest_path" not in u:
            raise SpecError(f"inputs.updates[{i}] must be {{path, manifest_path}}")
        pp, mp = Path(u["path"]), Path(u["manifest_path"])
        pp = pp if pp.is_absolute() else ctx.workdir / pp
        mp = mp if mp.is_absolute() else ctx.workdir / mp
        entries.append({"path": pp, "manifest_path": mp})
    try:
        s = aggregate(ctx.path("base_manifest"), ctx.path("base_weights"), entries, ctx.out,
                      outer_state_path=ctx.path("outer_state", required=False), outer_lr=p["outer_lr"],
                      momentum=p["momentum"], nesterov=p["nesterov"], aggregation=p["aggregation"],
                      trim_ratio=p["trim_ratio"], max_norm_ratio=p["max_norm_ratio"], max_bytes=p["max_bytes"],
                      version_id=p["version_id"], max_rel_norm=p["max_rel_norm"])
    except (ver.VersionError, OSError) as e:
        raise InputInvalid(str(e))
    if not s["no_op"]:
        if p["eval_heldout"] and ctx.path("vae_weights", required=False) is not None:
            from .distributed.evalhook import heldout_before_after
            b, a = heldout_before_after(ctx.path("base_manifest"), ctx.path("base_weights"), ctx.out / ver.WEIGHTS_FILE,
                                        ctx.path("vae_weights"))
            s["heldout_loss_before"], s["heldout_loss_after"] = b, a
        ctx.em.artifact("model_version", ctx.out / ver.MANIFEST_FILE)
        ctx.em.artifact("model_weights", ctx.out / ver.WEIGHTS_FILE)
        ctx.em.artifact("outer_state", ctx.out / ver.OUTER_FILE)
    s.pop("manifest_path", None)
    return s


def _bundle(ctx: Ctx):
    from .inference.bundle import load_bundle_files, BundleError
    try:
        return load_bundle_files(ctx.path("weights"), ctx.path("manifest"), ctx.path("vae_weights"), _device())
    except BundleError as e:
        raise InputInvalid(str(e))


HD_ONLY_PARAMS = ("width", "height", "seconds", "sampler", "fp16", "offload", "style")
HD_STAGE_WEIGHTS = {"text": (0.0, 0.02), "sample": (0.02, 0.78), "decode": (0.80, 0.18), "write": (0.98, 0.02)}


def _hd_manifest(ctx: Ctx):
    """The HD manifest dict if inputs.manifest is one, else None (small bundles / unreadable: old code path)."""
    from .inference.hd_bundle import read_manifest_if_hd
    try:
        return read_manifest_if_hd(ctx.path("manifest"))
    except InputInvalid:
        return None


def h_generate_hd(ctx: Ctx, p: dict, m: dict) -> dict:
    from .inference.bundle import BundleError
    from .inference.hd_bundle import hd_summary, load_hd_bundle_files, save_hd_result
    from .inference.sampler import Stopped
    given = ctx.job.get("params", {})
    if "upscale" in given and p["upscale"] != 1:
        raise SpecError("params.upscale must be 1 for HD models")
    from .data.styles import resolve_style
    try:                                   # fail fast (exit 2) before loading any weights
        resolve_style(p["style"], m.get("styles") or [], m.get("default_style"))
    except ValueError as e:
        raise SpecError(f"params.style: {e}")
    over = {}
    if p["fp16"] is not None:
        over["fp16"] = p["fp16"]
    if p["offload"] is not None:
        over["offload"] = p["offload"] == "blocks"
    try:
        pipe = load_hd_bundle_files(ctx.path("weights"), ctx.path("manifest"), ctx.path("vae_weights"), _device(), **over)
    except BundleError as e:
        raise InputInvalid(str(e))
    ctx.check_stop()

    def on_progress(stage, i, n):
        start, span = HD_STAGE_WEIGHTS.get(stage, (0.0, 0.0))
        ctx.em.progress(stage, start + span * (i / max(1, n)), i, n)

    kw = {k: p[k] for k in ("steps", "cfg_scale", "fps") if k in given}     # else: the bundle's own defaults
    for k in ("width", "height", "seconds", "sampler"):
        if p[k] is not None:
            kw[k] = p[k]
    mp4 = ctx.out / f"{p['name']}.mp4"
    try:
        res = pipe.generate(p["prompt"], seed=p["seed"], out_path=mp4, on_progress=on_progress,
                            should_stop=ctx.should_stop, style=p["style"], **kw)
    except Stopped:
        raise JobStopped("stopped during generation")
    meta = save_hd_result(res, ctx.out, p["name"])
    ctx.em.artifact("video", mp4)
    ctx.em.artifact("video_metadata", ctx.out / f"{p['name']}.json")
    return hd_summary(meta)


# ---------------------------------------------------------------- long videos
def _long_check(p: dict) -> None:
    for k in ("width", "height", "draft_width", "draft_height"):
        if p.get(k) is not None and p[k] % 2:
            raise SpecError(f"params.{k} must be even (yuv420p)")


def _long_pipe(ctx: Ctx, p: dict, job: str):
    """Fail-fast param checks (exit 2) before loading weights, then the HD pipeline (exit 5 on bad bundle)."""
    from .data.styles import resolve_style
    from .inference.bundle import BundleError
    from .inference.hd_bundle import load_hd_bundle_files
    m = _hd_manifest(ctx)
    if m is None:
        raise InputInvalid(f"{job} requires an HD bundle")
    try:
        resolve_style(p["style"], m.get("styles") or [], m.get("default_style"))
    except ValueError as e:
        raise SpecError(f"params.style: {e}")
    over = {}
    if p["fp16"] is not None:
        over["fp16"] = p["fp16"]
    if p["offload"] is not None:
        over["offload"] = p["offload"] == "blocks"
    try:
        pipe = load_hd_bundle_files(ctx.path("weights"), ctx.path("manifest"), ctx.path("vae_weights"), _device(), **over)
    except BundleError as e:
        raise InputInvalid(str(e))
    ctx.check_stop()
    return pipe


def _long_call(fn, *a, **kw):
    from .inference.longvideo import LongVideoError
    from .inference.sampler import Stopped
    from .inference.video_io import FFmpegError
    try:
        return fn(*a, **kw)
    except Stopped:
        raise JobStopped("stopped")
    except LongVideoError as e:
        raise InputInvalid(str(e))
    except FFmpegError as e:
        raise RuntimeError(f"ffmpeg: {e}")


def h_draft(ctx: Ctx, p: dict) -> dict:
    from .inference import longvideo as lv
    _long_check(p)
    pipe = _long_pipe(ctx, p, "ai.draft")
    try:
        vc = pipe.vae.cfg
        if not vc.valid_frames(p["segment_frames"]):
            raise SpecError(f"params.segment_frames must be 1 + {vc.temporal_factor}k")
    except AttributeError:
        pass
    mp4, plan = ctx.out / f"{p['name']}.mp4", ctx.out / f"{p['name']}_plan.json"
    kw = {k: p[k] for k in ("steps", "cfg_scale", "sampler") if p[k] is not None}
    summ = _long_call(lv.run_draft, pipe, prompt=p["prompt"], seed=p["seed"], seconds=p["seconds"], fps=p["fps"],
                      segment_frames=p["segment_frames"], width=p["draft_width"], height=p["draft_height"],
                      out_mp4=mp4, plan_path=plan, style=p["style"], should_stop=ctx.should_stop,
                      on_progress=lambda st, f, i, n: ctx.em.progress(st, f, i, n), **kw)
    ctx.em.artifact("draft_video", mp4)
    ctx.em.artifact("draft_plan", plan)
    return summ


REFINE_STAGE_WEIGHTS = {"text": (0.0, 0.02), "encode": (0.02, 0.10), "sample": (0.12, 0.68), "decode": (0.80, 0.18)}


def h_refine_segment(ctx: Ctx, p: dict) -> dict:
    from .inference import longvideo as lv
    if p["end"] < p["start"]:
        raise SpecError("params.end must be >= params.start")
    _long_check(p)
    draft = ctx.path("draft_video")
    pipe = _long_pipe(ctx, p, "ai.refine_segment")

    def on_progress(stage, i, n):
        start, span = REFINE_STAGE_WEIGHTS.get(stage, (0.0, 0.0))
        ctx.em.progress(stage, start + span * (i / max(1, n)), i, n)

    mp4 = ctx.out / f"{p['name']}.mp4"
    kw = {k: p[k] for k in ("steps", "cfg_scale", "sampler") if p[k] is not None}
    meta = _long_call(lv.run_refine, pipe, draft_path=draft, prompt=p["prompt"], seed=p["seed"], width=p["width"],
                      height=p["height"], fps=p["fps"], start=p["start"], end=p["end"], strength=p["strength"],
                      out_mp4=mp4, style=p["style"], should_stop=ctx.should_stop, on_progress=on_progress, **kw)
    ctx.em.progress("write", 1.0, 1, 1)
    ctx.em.artifact("segment_video", mp4)
    keys = ("frames", "width", "height", "strength", "steps", "seconds_load", "seconds_text", "seconds_encode",
            "seconds_sampling", "seconds_decode", "seconds_total", "peak_rss_mb", "cuda_max_allocated_mb",
            "model_version_id", "sampler", "style")
    s = {k: meta[k] for k in keys if k in meta}
    s["index"] = p["index"]
    return s


def h_stitch(ctx: Ctx, p: dict) -> dict:
    from .inference import longvideo as lv
    n = p["count"]
    want = [f"segment_{i:03d}" for i in range(n)]
    extra = set(ctx.inputs) - set(want)
    if extra:
        raise SpecError(f"inputs {sorted(extra)} not within count={n}")
    paths = []
    for k in want:
        if k not in ctx.inputs:
            raise SpecError(f"inputs.{k} is required")
        paths.append(ctx.path(k))
    mp4 = ctx.out / f"{p['name']}.mp4"
    summ = _long_call(lv.run_stitch, paths, mp4, p["fps"], p["crossfade_frames"], should_stop=ctx.should_stop,
                      on_progress=lambda st, f, i, t: ctx.em.progress(st, f, i, t))
    ctx.em.artifact("video", mp4)
    return summ


def h_generate(ctx: Ctx, p: dict) -> dict:
    from .inference.generate import generate, save_result
    from .inference.sampler import Stopped
    hd = _hd_manifest(ctx)
    given = ctx.job.get("params", {})
    if hd is not None:
        return h_generate_hd(ctx, p, hd)
    bad = [k for k in HD_ONLY_PARAMS if k in given]
    if bad:
        raise SpecError(f"params {bad} are only valid for HD models")
    b = _bundle(ctx)
    ctx.check_stop()
    try:
        res = generate(b, p["prompt"], seed=p["seed"], steps=p["steps"], cfg_scale=p["cfg_scale"], tile=p["tile"],
                       should_stop=ctx.should_stop, device=_device())
    except Stopped:
        raise JobStopped("stopped during sampling")
    meta = save_result(res, ctx.out, p["name"], fps=p["fps"], upscale=p["upscale"])
    ctx.em.artifact("video", ctx.out / f"{p['name']}.mp4")
    ctx.em.artifact("video_metadata", ctx.out / f"{p['name']}.json")
    return {"frames": meta["frames"], "fps": meta["fps"], "seconds_sampling": meta["seconds_sampling"],
            "seconds_decode": meta["seconds_decode"], "unknown_words": meta["unknown_words"]}


def _stylized_hd(ctx: Ctx) -> bool:
    try:
        m = _hd_manifest(ctx) if "manifest" in getattr(ctx, "inputs", {}) else None
    except InputInvalid:
        return False
    return m is not None and m.get("domain") == "stylized"


def _evaluate_stylized_hd(ctx: Ctx, p: dict) -> dict:
    """adherence mode on an HD bundle whose domain is stylized: the stylized prompt set (or prompts_file)."""
    from .evaluation import stylized as sty
    from .inference.bundle import BundleError
    from .inference.hd_bundle import load_hd_bundle_files
    from .inference.sampler import Stopped
    try:
        pipe = load_hd_bundle_files(ctx.path("weights"), ctx.path("manifest"), ctx.path("vae_weights"), _device())
    except BundleError as e:
        raise InputInvalid(str(e))
    prompts = sty.load_stylized_prompts(p["prompts_file"])[: p["num_prompts"]]
    try:
        rep = sty.run_hd_prompt_set(pipe, prompts, p["eval_seed"], should_stop=ctx.should_stop,
                                    progress=lambda i, n: ctx.em.progress("benchmark", i / n, i, n))
    except Stopped:
        raise JobStopped("stopped during evaluation")
    (ctx.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    ctx.em.artifact("evaluation_report", ctx.out / "report.json")
    return sty.summarize_hd(rep)


def h_evaluate(ctx: Ctx, p: dict) -> dict:
    from .evaluation import benchmark as bm
    mode = p["mode"]
    if mode == "detector":
        rep = bm.detector_validation(p["num_clips"], should_stop=ctx.should_stop)
        summ = {k: rep[k] for k in ("clips", "all_correct_rate", "entity_recall", "counts_correct_rate")}
    elif mode == "reconstruction":
        import numpy as np
        import torch
        from . import modelio
        from .compression.vae import reconstruction_metrics
        from .config import load_config, synth_config
        from .data.synthetic import make_sample
        from .distributed import version as ver
        from .training.tasks import DATA_SEED, HELDOUT_START
        try:
            cfg = load_config(ver.read_manifest(ctx.path("manifest"))["config"])
            vae = modelio.load_vae(ctx.path("vae_weights"))
        except (ver.VersionError, modelio.ModelFileError) as e:
            raise InputInvalid(str(e))
        scfg = synth_config(cfg)
        vids = np.stack([make_sample(DATA_SEED, HELDOUT_START + i, scfg).video for i in range(p["num_clips"])])
        rep = reconstruction_metrics(vae, torch.from_numpy(vids))
        summ = {k: rep[k] for k in ("psnr_db", "motion_l1_255", "clips")}
    elif _stylized_hd(ctx):
        return _evaluate_stylized_hd(ctx, p)
    else:
        b = _bundle(ctx)
        prompts = bm.load_prompts(p["prompts_file"])[: p["num_prompts"]]
        from .inference.sampler import Stopped
        try:
            rep = bm.run_benchmark(b, prompts, p["seeds_per_prompt"], p["steps"], p["cfg_scale"], p["eval_seed"],
                                   should_stop=ctx.should_stop,
                                   progress=lambda i, n: ctx.em.progress("benchmark", i / n, i, n))
        except Stopped:
            raise JobStopped("stopped during evaluation")
        summ = bm.summarize(rep)
    (ctx.out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    ctx.em.artifact("evaluation_report", ctx.out / "report.json")
    return summ


def _paths(ctx: Ctx, key: str, entries) -> list:
    out = []
    for e in entries:
        pp = Path(e)
        pp = pp if pp.is_absolute() else ctx.workdir / pp
        if not pp.exists():
            raise InputInvalid(f"inputs.{key}: {pp} does not exist")
        out.append(str(pp))
    return out


def _role_inputs(ctx: Ctx, prefix: str) -> dict:
    return {k: str(ctx.path(k)) for k in sorted(ctx.inputs) if k.startswith(prefix)}


def h_hd_encode(ctx: Ctx, p: dict) -> dict:
    from .distributed.hd.encode_job import EncodeInputError, hd_encode_job
    ins = {k: str(ctx.path(k)) for k in ("raw_shard", "vae_weights", "vae_manifest", "text_encoder_weights", "tokenizer")}
    try:
        s = hd_encode_job(ins, p, ctx.out, device=_device(),
                          on_progress=lambda i, n: (ctx.check_stop(), ctx.em.progress("encode", i / max(1, n), i, n)))
    except (EncodeInputError, ValueError) as e:
        raise InputInvalid(str(e))
    ctx.em.artifact("latent_shard", Path(s.pop("path")))
    return s


def h_hd_train(ctx: Ctx, p: dict) -> dict:
    from .distributed.hd.train_job import TrainInputError, hd_train_job
    from .inference.sampler import Stopped
    if p["target"] != "dit":
        raise SpecError("params.target 'vae' is not supported by this executor version (dit only)")
    ins = {k: str(ctx.path(k)) for k in ("base_train_weights", "base_manifest")}
    ins.update(_role_inputs(ctx, "data_shard_"))
    try:
        s = hd_train_job(ins, p, ctx.out, attempt_id=ctx.job.get("attempt_id") or "", device=_device(),
                         should_stop=ctx.should_stop,
                         on_metric=lambda n, v, st: (ctx.em.metric(n, v, st),
                                                     ctx.em.progress("train", st / p["inner_steps"], st, p["inner_steps"])))
    except TrainInputError as e:
        raise InputInvalid(str(e))
    except Stopped:
        raise JobStopped("stopped")
    ctx.em.artifact("update", ctx.out / "update.json")
    ctx.em.artifact("update_weights", ctx.out / "update.safetensors")
    s.pop("update_sha256", None)
    return s


def h_hd_eval(ctx: Ctx, p: dict) -> dict:
    from .distributed import version as ver
    from .distributed.hd.eval_job import hd_eval_job
    val = list(_role_inputs(ctx, "val_shard_").values())
    if not val:
        raise SpecError("at least one val_shard_NNN input is required")
    try:
        s = hd_eval_job(ctx.path("weights"), ctx.path("manifest"), val, ctx.out, version_id=p["version_id"] or "",
                        triples_seed=p["triples_seed"], n=p["n"], timestep_buckets=p["timestep_buckets"],
                        should_stop=ctx.should_stop)
    except InterruptedError:
        raise JobStopped("stopped")
    except (ver.VersionError, ValueError, OSError) as e:
        raise InputInvalid(str(e))
    ctx.em.artifact("eval_report", Path(s.pop("report")))
    return s


def h_hd_aggregate(ctx: Ctx, p: dict) -> dict:
    from .distributed import version as ver
    from .distributed.hd.aggregate import AggregateError, TRAIN_FILE, hd_aggregate
    ups = ctx.inputs.get("updates")
    if not isinstance(ups, list) or not (1 <= len(ups) <= 64):
        raise SpecError("inputs.updates must be a list of 1..64 entries")
    entries = []
    for i, u in enumerate(ups):
        if not isinstance(u, dict) or set(u) - {"path", "manifest_path", "update_id"} or "path" not in u \
                or "manifest_path" not in u:
            raise SpecError(f"inputs.updates[{i}] must be {{path, manifest_path, update_id?}}")
        pp, mp = _paths(ctx, "updates", [u["path"], u["manifest_path"]])
        entries.append({"path": pp, "manifest_path": mp, "update_id": u.get("update_id")})
    ev = ctx.inputs.get("eval_latents")
    if ev is not None and not (isinstance(ev, list) and all(isinstance(x, str) for x in ev)):
        raise SpecError("inputs.eval_latents must be a list of paths")
    try:
        s = hd_aggregate(ctx.path("base_manifest"), ctx.path("base_weights"), entries, ctx.out,
                         outer_state_path=ctx.path("outer_state", required=False), outer_lr=p["outer_lr"],
                         momentum=p["momentum"], nesterov=p["nesterov"], aggregation=p["aggregation"],
                         trim_ratio=p["trim_ratio"], max_norm_ratio=p["max_norm_ratio"], max_rel_norm=p["max_rel_norm"],
                         bound_c=p["bound_c"], inner_steps=p["inner_steps"], lr=p["lr"], clip_factor=p["clip_factor"],
                         version_id=p["version_id"], eval_gate=p["eval_gate"],
                         eval_latents=_paths(ctx, "eval_latents", ev) if ev else None,
                         eval_triples_seed=p["eval_triples_seed"], eval_n=p["eval_n"], eval_buckets=p["eval_buckets"],
                         should_stop=ctx.should_stop)
    except InterruptedError:
        raise JobStopped("stopped")
    except (AggregateError, ver.VersionError, OSError) as e:
        raise InputInvalid(str(e))
    if not s["no_op"]:
        ctx.em.artifact("model_manifest", ctx.out / ver.MANIFEST_FILE)
        ctx.em.artifact("model_weights", ctx.out / ver.WEIGHTS_FILE)
        ctx.em.artifact("model_train_weights", ctx.out / TRAIN_FILE)
        ctx.em.artifact("outer_state", ctx.out / ver.OUTER_FILE)
    s.pop("manifest_path", None)
    return s


HANDLERS = {"ai.init_model": h_init_model, "ai.dataset_shard": h_dataset_shard, "ai.local_train": h_local_train,
            "ai.validate_update": h_validate_update, "ai.aggregate": h_aggregate, "ai.evaluate": h_evaluate,
            "ai.generate": h_generate, "ai.draft": h_draft, "ai.refine_segment": h_refine_segment,
            "ai.stitch": h_stitch, "ai.hd_encode": h_hd_encode, "ai.hd_train": h_hd_train, "ai.hd_eval": h_hd_eval,
            "ai.hd_aggregate": h_hd_aggregate}


# ---------------------------------------------------------------- main
def _check_job(job: Any) -> None:
    if not isinstance(job, dict):
        raise SpecError("job must be a JSON object")
    if job.get("protocol") != PROTOCOL:
        raise SpecError(f"unsupported protocol {job.get('protocol')!r}, expected {PROTOCOL!r}")
    unknown = set(job) - TOP_KEYS
    if unknown:
        raise SpecError(f"unknown job keys {sorted(unknown)}")
    if job.get("job_type") not in SCHEMAS:
        raise SpecError(f"unsupported job_type {job.get('job_type')!r}")
    if not isinstance(job.get("output_dir"), str) or not job["output_dir"]:
        raise SpecError("output_dir is required")
    if not isinstance(job.get("params", {}), dict) or not isinstance(job.get("inputs", {}), dict):
        raise SpecError("params / inputs must be objects")
    lim = job.get("limits") or {}
    if not isinstance(lim, dict) or set(lim) - LIMIT_KEYS:
        raise SpecError("invalid limits")
    for k in ("job_id", "attempt_id", "stop_file"):
        if k in job and job[k] is not None and not isinstance(job[k], str):
            raise SpecError(f"{k} must be a string")


def run_job(job_path: Path, em: Emitter) -> int:
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        em.error("INVALID_SPEC", f"cannot read job file: {e}")
        return EXIT_SPEC
    try:
        _check_job(job)
        schema = SCHEMAS[job["job_type"]]
        params = _validate(job.get("params", {}), schema["params"])
        inputs = job.get("inputs", {})
        rx = schema.get("input_regex")
        bad = {k for k in inputs if k not in schema["inputs"] and not (rx and re.fullmatch(rx, k))}
        if bad:
            raise SpecError(f"unknown inputs keys {sorted(bad)}")
        for k, req in schema["inputs"].items():
            if req and k not in inputs:
                raise SpecError(f"inputs.{k} is required")
        ctx = Ctx(job, job_path.resolve().parent, em)
        ctx.inputs = inputs
        em.log(f"job {job.get('job_id')} type {job['job_type']} started")
        ctx.check_stop()
        from .config import setup_runtime
        setup_runtime()
        summary = HANDLERS[job["job_type"]](ctx, params)
        em.result(summary)
        return EXIT_OK
    except (SpecError, InputInvalid, JobStopped) as e:
        em.error(e.code, str(e))
        return e.exit
    except MemoryError as e:
        em.error("OOM", f"out of memory: {e}")
        return EXIT_OOM
    except Exception as e:  # noqa: BLE001
        if "out of memory" in str(e).lower():
            em.error("OOM", str(e))
            return EXIT_OOM
        em.error("INTERNAL", f"{type(e).__name__}: {e}")
        sys.stderr.write(traceback.format_exc())
        return EXIT_INTERNAL


def exec_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="nivra exec")
    ap.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    return run_job(Path(a.job), Emitter())
