"""Worker side of HD DiLoCo (``ai.hd_train``): H inner steps on precomputed latents -> fp16 pseudo-gradient.

* Workers never load the VAE or the text encoder: shards carry scaled latents and text embeddings.
* Fresh inner optimizer every round, linear warmup to ``lr`` then constant (so ||delta||_inf <= c*H*lr holds).
* Deterministic: data order, timesteps, noise and caption dropout are functions of ``seed``.
* delta = received_weights - trained_weights, both fp32 (the received file may be the fp16 train copy or, for
  version 0, the fp32 master), written tensor by tensor as fp16.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, Optional

import torch

from ...inference.sampler import Stopped
from ...models.dit import resolution_shift, rf_loss, sample_t
from ...models.hd_config import HD_ARCH_VERSION, hd_config_hash, load_hd_config
from ...training.hd_trainer import make_optimizer
from .. import version as ver
from ..update import MANIFEST_FILE, SCHEMA, UPDATE_FILE
from .latent_shard import LatentShard, LatentShardError, check_bound_to
from .stream_io import StReader, StWriter, StreamIOError, sha256_file
from .validate import norm_stage


class TrainInputError(Exception):
    """Bad inputs (maps to exit 5 in the executor)."""


def load_base_model(base_manifest_path, base_weights_path, expect_version_id=None, expect_config_sha=None):
    try:
        bm = ver.read_manifest(base_manifest_path)
    except ver.VersionError as e:
        raise TrainInputError(str(e)) from e
    if bm.get("family") != "hd" or bm.get("architecture_version") != HD_ARCH_VERSION:
        raise TrainInputError("base manifest is not an HD manifest")
    if expect_version_id and bm["version_id"] != expect_version_id:
        raise TrainInputError(f"base manifest version {bm['version_id']} != params.base_version_id {expect_version_id}")
    cfg = load_hd_config(bm["config"])
    have = bm.get("hd_config_sha") or bm.get("config_hash") or hd_config_hash(cfg)
    if expect_config_sha and have != expect_config_sha:
        raise TrainInputError("hd_config_sha does not match the base manifest")
    shapes = {k: tuple(v) for k, v in bm["tensor_shapes"].items()}
    try:
        with StReader(base_weights_path) as r:
            if r.shapes() != shapes:
                raise TrainInputError("base weights do not match the manifest (names/shapes)")
    except StreamIOError as e:
        raise TrainInputError(str(e)) from e
    sha = sha256_file(base_weights_path)
    if sha not in (bm.get("train_weights_sha256"), bm["weights_sha256"]):
        raise TrainInputError("base weights sha256 matches neither train_weights_sha256 nor weights_sha256")
    return bm, cfg, sha


def hd_train_job(inputs: dict, params: dict, out_dir, *, attempt_id: str = "", device: str = "cpu",
                 should_stop: Optional[Callable] = None, on_metric: Optional[Callable] = None) -> dict:
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = params
    wpath = inputs["base_train_weights"]
    bm, cfg, wsha = load_base_model(inputs["base_manifest"], wpath, p.get("base_version_id"), p.get("hd_config_sha"))
    from ...models.hd_config import build_dit
    from safetensors.torch import load_file
    model = build_dit(cfg)
    model.load_state_dict(load_file(str(wpath)), strict=True)
    tc = cfg["train"]
    # ---- data
    roles = sorted(k for k in inputs if k.startswith("data_shard_"))
    if not roles:
        raise TrainInputError("no data_shard_* inputs")
    want = p.get("shards") or []
    if want and len(want) != len(roles):
        raise TrainInputError(f"{len(roles)} data shards given but params.shards lists {len(want)}")
    shards, tensors = [], []
    for i, role in enumerate(roles):
        if want and sha256_file(inputs[role]) != str(want[i]["sha256"]).lower():
            raise TrainInputError(f"{role}: sha256 differs from params.shards[{i}]")
        try:
            sh = LatentShard(inputs[role])
        except LatentShardError as e:
            raise TrainInputError(str(e)) from e
        bad = check_bound_to(sh, stage=p.get("stage"))
        if bad:
            raise TrainInputError(bad)
        shards.append(sh)
        tensors.append(sh.load_all())
    clips = [(si, i) for si, sh in enumerate(shards) for i in range(len(sh))]
    if len(clips) < 1:
        raise TrainInputError("shards hold no clips")
    # ---- optimisation
    steps, bs, accum = int(p["inner_steps"]), int(p["batch_size"]), int(p.get("grad_accum", 1))
    lr, warm = float(p["lr"]), max(1, int(p.get("warmup_steps", 0) or 0))
    clip, drop_p = float(p.get("grad_clip", 0.0) or 0.0), float(p.get("cfg_dropout", 0.1))
    seed = int(p.get("seed", 0))
    dev = torch.device(device)
    use_amp = dev.type == "cuda" and bool(tc.get("fp16", True))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 17)
    order_rng = random.Random(seed)
    model.set_memory(grad_checkpoint=bool(tc.get("grad_checkpoint", True)))
    model.to(dev).train()
    wd = float(p.get("weight_decay", 0.0) or 0.0)
    opt = make_optimizer(tc.get("optimizer", "adamw"), model.parameters(), lr, wd)      # fresh every round
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    shift_base = int(tc.get("shift_base_tokens", 4096))
    order, pos = [], 0

    def next_batch():
        nonlocal order, pos
        zs, es, ms = [], [], []
        while len(zs) < bs:
            if pos >= len(order):
                order = list(range(len(clips)))
                order_rng.shuffle(order)
                pos = 0
            si, i = clips[order[pos]]
            pos += 1
            d = tensors[si]
            zs.append(d["latents"][i].float())
            es.append(d["text_emb"][i].float())
            ms.append(d["text_mask"][i].bool())
        L = max(e.shape[0] for e in es)
        emb = torch.zeros(bs, L, es[0].shape[1])
        mask = torch.zeros(bs, L, dtype=torch.bool)
        for k, (e, m) in enumerate(zip(es, ms)):
            emb[k, :e.shape[0]] = e
            mask[k, :m.shape[0]] = m
        return torch.stack(zs), emb, mask

    losses = []
    for step in range(steps):
        if should_stop is not None and should_stop():
            raise Stopped()
        cur = lr * min(1.0, (step + 1) / warm)
        for g in opt.param_groups:
            g["lr"] = cur
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(accum):
            z0, emb, mask = next_batch()
            n_tok = z0.shape[2] * z0.shape[3] * z0.shape[4]
            t = sample_t(bs, resolution_shift(n_tok, shift_base), generator=gen, device=dev)
            noise = torch.randn(z0.shape, generator=gen).to(dev)
            drop = (torch.rand(bs, generator=gen) < drop_p).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = rf_loss(model, z0.to(dev), emb.to(dev), mask.to(dev), t, noise, drop)
            scaler.scale(loss / accum).backward()
            acc += float(loss.detach()) / accum
        if clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not use_amp and acc != acc:
            raise FloatingPointError(f"non-finite loss at step {step}")
        scaler.step(opt)
        scaler.update()
        losses.append(acc)
        if on_metric:
            on_metric("loss", acc, step + 1)
    # ---- delta against exactly what was received
    trained = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    names = list(bm["tensor_shapes"])
    upath = out / UPDATE_FILE
    with StReader(wpath) as r, StWriter(upath, [(k, trained[k].shape) for k in names], torch.float16) as w:
        err2 = d2 = 0.0
        for k in names:
            d = r.read_f32(k) - trained[k].float()
            h = d.to(torch.float16)
            if not bool(torch.isfinite(h).all()):
                raise FloatingPointError(f"delta of {k} is not finite as fp16")
            err2 += float((h.double() - d.double()).pow(2).sum())
            d2 += float(d.double().pow(2).sum())
            w.write(k, d)
        usha = w.close()
    fp16_rel_err = (err2 / d2) ** 0.5 if d2 > 0 else 0.0
    k5 = max(1, min(5, len(losses)))
    stage = p.get("stage")
    man = {"schema": SCHEMA, "family": "hd", "update_file": UPDATE_FILE, "update_sha256": usha,
           "update_id": p["update_id"], "run_id": p.get("run_id"), "round": p.get("round"), "round_batch": p.get("round_batch"),
           "base_version_id": bm["version_id"], "attempt_id": attempt_id, "target": p.get("target", "dit"),
           "hd_config_sha": p.get("hd_config_sha") or bm.get("hd_config_sha") or bm.get("config_hash"),
           "stage": stage, "inner_steps": steps, "lr": lr, "shards": p.get("shards"),
           "samples_seen": steps * bs * accum, "delta_dtype": "float16", "base_weights_sha256": wsha,
           "architecture_version": bm["architecture_version"], "vocab_hash": bm["vocab_hash"],
           "tensor_shapes": {k: list(v) for k, v in bm["tensor_shapes"].items()},
           "loss_first": sum(losses[:k5]) / k5, "loss_last": sum(losses[-k5:]) / k5,
           "hyper": {"batch_size": bs, "grad_accum": accum, "warmup_steps": warm, "weight_decay": wd, "grad_clip": clip,
                     "cfg_dropout": drop_p, "seed": seed}}
    (out / MANIFEST_FILE).write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    cls = "cpu" if dev.type != "cuda" else f"cuda:{torch.cuda.get_device_name(dev)}"
    return {"update_id": p["update_id"], "base_version_id": bm["version_id"], "loss_first": man["loss_first"],
            "loss_last": man["loss_last"], "samples_seen": man["samples_seen"], "wall_sec": time.time() - t_start,
            "device_class": cls, "update_sha256": usha, "fp16_rel_err": fp16_rel_err}
