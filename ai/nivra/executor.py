"""Executor for protocol nivra.exec/1 (docs/architecture/EXECUTOR_PROTOCOL.md).

Usage: python -m nivra.cli exec --job <job.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        if "min" in s and not (v >= s["min"]):
            raise SpecError(f"{where}.{k} must be >= {s['min']}")
        if "max" in s and not (v <= s["max"]):
            raise SpecError(f"{where}.{k} must be <= {s['max']}")
        if "exclusive_min" in s and not (v > s["exclusive_min"]):
            raise SpecError(f"{where}.{k} must be > {s['exclusive_min']}")
        if "choices" in s and v not in s["choices"]:
            raise SpecError(f"{where}.{k} must be one of {s['choices']}")
        if "maxlen" in s and len(v) > s["maxlen"]:
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
        params={"max_norm_ratio": F(5.0, min=1.0), "max_bytes": I(1 << 30, min=1), "max_norm": F(None, exclusive_min=0.0),
                "peer_norms": {"type": "floatlist", "default": None}, "expected_base_version_id": S(None, maxlen=128)},
        inputs={"update": True, "update_manifest": True, "base_manifest": True, "base_weights": False}),
    "ai.aggregate": dict(
        params={"outer_lr": F(0.7, exclusive_min=0.0, max=10.0), "momentum": F(0.9, min=0.0, max=0.9999), "nesterov": B(True),
                "aggregation": S("auto", choices=["auto", "mean", "trimmed_mean"]), "trim_ratio": F(0.2, min=0.0, max=0.45),
                "max_norm_ratio": F(5.0, min=1.0), "max_bytes": I(1 << 30, min=1), "version_id": S(None, maxlen=128),
                "eval_heldout": B(False)},
        inputs={"base_weights": True, "base_manifest": True, "outer_state": False, "updates": True, "vae_weights": False}),
    "ai.evaluate": dict(
        params={"mode": S(req=True, choices=["adherence", "reconstruction", "detector"]), "num_prompts": I(16, min=1, max=200),
                "seeds_per_prompt": I(1, min=1, max=8), "steps": I(20, min=2, max=100), "cfg_scale": F(3.0, min=0.0, max=20.0),
                "num_clips": I(64, min=1, max=512), "eval_seed": I(777), "prompts_file": S(None)},
        inputs={"weights": False, "manifest": False, "vae_weights": False}),
    "ai.generate": dict(
        params={"prompt": S(req=True, minlen=1, maxlen=200), "seed": I(0, min=0), "steps": I(30, min=1, max=200),
                "cfg_scale": F(3.0, min=0.0, max=20.0), "fps": I(8, min=1, max=60), "upscale": I(4, min=1, max=8),
                "tile": I(0, min=0, max=16), "name": S("video", maxlen=64)},
        inputs={"weights": True, "manifest": True, "vae_weights": True}),
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
    h = hashlib.sha256()
    for i in range(p["count"]):
        if i % 16 == 0:
            ctx.check_stop()
            ctx.em.progress("generate", i / p["count"], i, p["count"])
        s = make_sample(p["seed"], p["start"] + i, scfg)
        vids.append(s.video)
        caps.append(s.captions)
        specs.append(s.spec.to_dict())
        h.update(s.video.tobytes())
        h.update(json.dumps(s.spec.to_dict(), sort_keys=True).encode())
    save_file({"videos": torch.from_numpy(np.stack(vids))}, str(ctx.out / "shard.safetensors"))
    m = dmanifest(p["seed"], p["start"], p["count"], scfg)
    m.update(content_sha256=h.hexdigest(), captions=caps, specs=specs)
    (ctx.out / "shard.json").write_text(json.dumps(m, ensure_ascii=False))
    ctx.em.artifact("dataset_shard", ctx.out / "shard.safetensors")
    ctx.em.artifact("dataset_manifest", ctx.out / "shard.json")
    return {"count": p["count"], "fingerprint": m["fingerprint"], "content_sha256": m["content_sha256"]}


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
                        expected_base_version_id=p["expected_base_version_id"], base_weights=bw)
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
                      version_id=p["version_id"])
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


def h_generate(ctx: Ctx, p: dict) -> dict:
    from .inference.generate import generate, save_result
    from .inference.sampler import Stopped
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


HANDLERS = {"ai.init_model": h_init_model, "ai.dataset_shard": h_dataset_shard, "ai.local_train": h_local_train,
            "ai.validate_update": h_validate_update, "ai.aggregate": h_aggregate, "ai.evaluate": h_evaluate,
            "ai.generate": h_generate}


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
        bad = set(inputs) - set(schema["inputs"])
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
