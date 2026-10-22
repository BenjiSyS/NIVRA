"""Worker side of HD DiLoCo (``ai.hd_train``): H inner steps on precomputed latents -> fp16 pseudo-gradient.

* Workers never load the VAE or the text encoder: shards carry scaled latents and text embeddings.
* Fresh inner optimizer every round, linear warmup to ``lr`` then constant (so ||delta||_inf <= c*H*lr holds).
* Deterministic: data order, timesteps, noise and caption dropout are functions of ``seed``.
* The round uses the same training recipe as ``hd_trainer.train_hd`` (the config's ``train`` options, through
  ``train_options``): diffusion forcing (per-latent-frame timesteps on a fraction of micro-batches), first/last latent
  frame conditioning (``frame_cond`` models), fps conditioning from each clip's ``effective_fps`` (with
  ``fps_dropout``), fp32 attention (``fp32_attn``), a dynamic GradScaler with skip-on-overflow (the step is skipped,
  the loss scale backs off, the round still counts ``inner_steps`` and the delta stays bounded) and SRA. Losses and
  samplers are the trainer's own functions (``rf_loss_ex``, ``sample_t_frames``, ``sra_loss``, ``SRAProjector``).
* SRA in a distributed round: there is no EMA on a worker (the outer step averages), so the teacher is a FROZEN copy
  of the weights received for this round; the projector is training-only and is re-initialised every round (it is not
  part of the delta). Costs one extra model copy in memory plus one no-grad forward per micro-batch.
* delta = received_weights - trained_weights, both fp32 (the received file may be the fp16 train copy or, for
  version 0, the fp32 master), written tensor by tensor as fp16 (default) or as int8 with per-row scales
  (``delta_dtype: "int8"``, see quant.py). int8 uses error feedback: the quantisation residual is kept on disk
  (``state_dir/hd_residual``) and added to the next round's delta; it is discarded whenever the base is not the direct
  successor of the version the residual was computed against, or when our last update was not applied.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Optional

import torch

from ...inference.sampler import Stopped
from ...models.dit import make_xt, resolution_shift, rf_loss_ex, sample_t, sample_t_frames
from ...models.hd_config import HD_ARCH_VERSION, hd_config_hash, load_hd_config, train_options
from ...training.hd_trainer import (MIN_LOSS_SCALE, SRAProjector, clip_fps_value, fps_dropped, make_optimizer,
                                    sra_loss)
from .. import version as ver
from . import quant
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
                 should_stop: Optional[Callable] = None, on_metric: Optional[Callable] = None,
                 state_dir=None) -> dict:
    """``state_dir``: persistent per-worker directory for the int8 error-feedback residual (None = no error feedback)."""
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = params
    delta_dtype = p.get("delta_dtype") or "float16"
    if delta_dtype not in ("float16", "int8"):
        raise TrainInputError(f"unsupported delta_dtype {delta_dtype!r}")
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
    topt = train_options(cfg)
    use_amp = dev.type == "cuda" and bool(tc.get("fp16", True))
    scaling = use_amp or bool(tc.get("force_loss_scaling", False))          # the CPU switch is for tests / debugging
    ls = topt["loss_scale"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 17)
    order_rng = random.Random(seed)
    model.set_memory(grad_checkpoint=bool(tc.get("grad_checkpoint", True)), attn_fp32=topt["fp32_attn"] or None)
    df, fps_model = topt["diffusion_forcing"], bool(model.cfg.fps_cond)
    sra, sra_on = topt["sra"], topt["sra"]["weight"] > 0
    teacher = proj = proj_opt = None
    if sra_on:
        if max(sra["student_layer"], sra["teacher_layer"]) >= len(model.blocks):
            raise TrainInputError("train.sra layers exceed the model depth")
        teacher = copy.deepcopy(model).to(dev).eval().requires_grad_(False)     # frozen: the weights received this round
    model.to(dev).train()
    wd = float(p.get("weight_decay", 0.0) or 0.0)
    opt = make_optimizer(tc.get("optimizer", "adamw"), model.parameters(), lr, wd)      # fresh every round
    if sra_on:
        proj = SRAProjector(model.cfg.dim, sra["hidden"]).to(dev).train()
        proj_opt = torch.optim.AdamW(proj.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda" if dev.type == "cuda" else "cpu", enabled=scaling, init_scale=ls["init"],
                                  growth_factor=ls["growth"], backoff_factor=ls["backoff"],
                                  growth_interval=ls["growth_interval"])
    shift_base = int(tc.get("shift_base_tokens", 4096))
    order, pos, epoch = [], 0, 0
    fps_p = topt["fps_dropout"]

    def next_batch():
        nonlocal order, pos, epoch
        zs, es, ms, fp = [], [], [], []
        while len(zs) < bs:
            if pos >= len(order):
                order = list(range(len(clips)))
                order_rng.shuffle(order)
                pos = 0
                epoch += 1
            si, i = clips[order[pos]]
            pos += 1
            d = tensors[si]
            zs.append(d["latents"][i].float())
            es.append(d["text_emb"][i].float())
            ms.append(d["text_mask"][i].bool())
            row = shards[si].rows[i]
            v = clip_fps_value(row) if fps_model else float("nan")
            if fps_dropped(seed, epoch, str(row.get("id", i)), fps_p):
                v = float("nan")
            fp.append(v)
        L = max(e.shape[0] for e in es)
        emb = torch.zeros(bs, L, es[0].shape[1])
        mask = torch.zeros(bs, L, dtype=torch.bool)
        for k, (e, m) in enumerate(zip(es, ms)):
            emb[k, :e.shape[0]] = e
            mask[k, :m.shape[0]] = m
        return torch.stack(zs), emb, mask, torch.tensor(fp)

    def cond_mask_for(B: int, Tg: int):
        """First/last latent-frame conditioning pattern (same four modes as the trainer)."""
        mode = int(torch.randint(0, 4, (1,), generator=gen))
        m = torch.zeros(B, Tg, dtype=torch.bool)
        nf, nl = [(1, 0), (2, 0), (0, 1), (1, 1)][mode]
        if nf + nl >= Tg:
            nf, nl = min(nf, Tg - 1), 0
        if nf:
            m[:, :nf] = True
        if nl:
            m[:, -nl:] = True
        return m

    losses, skips, consecutive, sra_vals = [], 0, 0, []
    for step in range(steps):
        if should_stop is not None and should_stop():
            raise Stopped()
        cur = lr * min(1.0, (step + 1) / warm)
        for g in opt.param_groups:
            g["lr"] = cur
        opt.zero_grad(set_to_none=True)
        if proj_opt is not None:
            proj_opt.zero_grad(set_to_none=True)
            for g in proj_opt.param_groups:
                g["lr"] = cur
        acc = 0.0
        for _ in range(accum):
            z0, emb, mask, fpsv = next_batch()
            B, Tl = z0.shape[0], z0.shape[2]
            pt = model.cfg.patch[0]
            Tg = Tl // pt
            n_tok = Tl * z0.shape[3] * z0.shape[4]
            shift = resolution_shift(n_tok, shift_base)
            use_df = df["fraction"] > 0 and float(torch.rand(1, generator=gen)) < df["fraction"]
            use_cond = (df["cond_fraction"] > 0 and Tg > 1 and float(torch.rand(1, generator=gen)) < df["cond_fraction"])
            t = sample_t(bs, shift, generator=gen, device=dev)
            noise = torch.randn(z0.shape, generator=gen).to(dev)
            drop = (torch.rand(bs, generator=gen) < drop_p).to(dev)
            if use_df:
                t = sample_t_frames(B, Tg, shift, generator=gen, device=dev)
            cmask = cond_mask_for(B, Tg).to(dev) if use_cond else None
            if cmask is not None and t.dim() == 1:
                t = t[:, None].expand(B, Tg).contiguous()
            fps_arg = fpsv.to(dev) if fps_model else None
            feats = (sra["student_layer"],) if sra_on else None
            emb_d, mask_d, z0d = emb.to(dev), mask.to(dev), z0.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                rf, fd = rf_loss_ex(model, z0d, emb_d, mask_d, t, noise, drop, cmask, fps_arg, feats)
                loss = rf
                if sra_on:
                    t_teach = t * sra["t_ratio"]
                    with torch.no_grad():
                        xt_t = make_xt(z0d.float(), noise.float(), t_teach, cmask)
                        gm = (cmask[:, ::pt] if (cmask is not None and pt > 1) else cmask)
                        tg = t_teach if (t_teach.dim() == 1 or pt == 1) else t_teach[:, ::pt]
                        _, td = teacher(xt_t, tg, emb_d, mask_d, drop, cond_mask=gm, fps=fps_arg,
                                        feats=(sra["teacher_layer"],))
                    tw = None
                    if cmask is not None:
                        hw = z0.shape[3] // model.cfg.patch[1] * (z0.shape[4] // model.cfg.patch[2])
                        tw = (~(cmask[:, ::pt] if pt > 1 else cmask)).float()[:, :, None].expand(B, Tg, hw).reshape(B, -1)
                    sl = sra_loss(fd[sra["student_layer"]], td[sra["teacher_layer"]], proj, tw)
                    loss = loss + sra["weight"] * sl
                    sra_vals.append(float(sl.detach()))
            scaler.scale(loss / accum).backward()
            acc += float(rf.detach()) / accum
        if clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not scaling and acc != acc:
            raise FloatingPointError(f"non-finite loss at step {step}")
        mods = [model] + ([proj] if proj is not None else [])
        overflow = scaling and not all(bool(torch.isfinite(q.grad).all()) for m_ in mods for q in m_.parameters()
                                       if q.grad is not None)
        scale_before = float(scaler.get_scale()) if scaling else 1.0
        scaler.step(opt)
        if proj_opt is not None and any(q.grad is not None for q in proj.parameters()):
            scaler.step(proj_opt)
        scaler.update()
        if scaling and float(scaler.get_scale()) < MIN_LOSS_SCALE:
            scaler.update(MIN_LOSS_SCALE)
        skipped = scaling and (overflow or float(scaler.get_scale()) < scale_before)
        if skipped:
            skips += 1
            consecutive += 1
            if consecutive >= 200:
                raise FloatingPointError(f"{consecutive} consecutive skipped steps: the loss overflows even at the "
                                         "minimum loss scale (see train.fp32_attn)")
        else:
            consecutive = 0
        losses.append(acc)
        if on_metric:
            on_metric("loss", acc, step + 1)
    # ---- delta against exactly what was received
    trained = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    names = list(bm["tensor_shapes"])
    upath = out / UPDATE_FILE
    ef = {"error_feedback": False, "residual_reset": None}
    if delta_dtype == "float16":
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
        rel_err = quant.rel_error(err2, d2)
    else:
        usha, rel_err, ef, commit = _write_int8_delta(wpath, trained, names, bm, upath, p, state_dir)
    k5 = max(1, min(5, len(losses)))
    stage = p.get("stage")
    man = {"schema": SCHEMA, "family": "hd", "update_file": UPDATE_FILE, "update_sha256": usha,
           "update_id": p["update_id"], "run_id": p.get("run_id"), "round": p.get("round"), "round_batch": p.get("round_batch"),
           "base_version_id": bm["version_id"], "attempt_id": attempt_id, "target": p.get("target", "dit"),
           "hd_config_sha": p.get("hd_config_sha") or bm.get("hd_config_sha") or bm.get("config_hash"),
           "stage": stage, "inner_steps": steps, "lr": lr, "shards": p.get("shards"),
           "samples_seen": steps * bs * accum, "delta_dtype": delta_dtype, "base_weights_sha256": wsha,
           "delta_format": quant.FORMAT if delta_dtype == "int8" else "fp16", "error_feedback": ef["error_feedback"],
           "residual_reset": ef["residual_reset"], "quant_rel_err": rel_err,
           "architecture_version": bm["architecture_version"], "vocab_hash": bm["vocab_hash"],
           "tensor_shapes": {k: list(v) for k, v in bm["tensor_shapes"].items()},
           "loss_first": sum(losses[:k5]) / k5, "loss_last": sum(losses[-k5:]) / k5,
           "hyper": {"batch_size": bs, "grad_accum": accum, "warmup_steps": warm, "weight_decay": wd, "grad_clip": clip,
                     "cfg_dropout": drop_p, "seed": seed},
           "recipe": {"sra": sra_on, "sra_teacher": "received_base_frozen" if sra_on else None,
                      "diffusion_forcing": df["fraction"], "frame_cond": df["cond_fraction"], "fps_cond": fps_model,
                      "fp32_attn": topt["fp32_attn"], "loss_scaling": scaling, "skipped_steps": skips}}
    (out / MANIFEST_FILE).write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    if delta_dtype == "int8" and commit is not None:
        commit()                                                # the update is on disk: now the residual may advance
    cls = "cpu" if dev.type != "cuda" else f"cuda:{torch.cuda.get_device_name(dev)}"
    return {"update_id": p["update_id"], "base_version_id": bm["version_id"], "loss_first": man["loss_first"],
            "loss_last": man["loss_last"], "samples_seen": man["samples_seen"], "wall_sec": time.time() - t_start,
            "device_class": cls, "update_sha256": usha, "fp16_rel_err": rel_err if delta_dtype == "float16" else 0.0,
            "delta_dtype": delta_dtype, "quant_rel_err": rel_err, "skipped_steps": skips,
            "sra_last": (sum(sra_vals[-accum:]) / len(sra_vals[-accum:])) if sra_vals else None, "error_feedback": ef["error_feedback"],
            "residual_reset": ef["residual_reset"]}


def _write_int8_delta(wpath, trained: dict, names: list, bm: dict, upath, p: dict, state_dir):
    """Quantise (delta + residual) tensor by tensor; returns (sha256, relative error, ef-info, commit-callback)."""
    shapes = {k: tuple(bm["tensor_shapes"][k]) for k in names}
    run_id, cfg_sha = p.get("run_id"), p.get("hd_config_sha") or bm.get("hd_config_sha") or bm.get("config_hash")
    ef = {"error_feedback": False, "residual_reset": "no state_dir or run_id: no error feedback"}
    store = res_r = res_w = None
    if state_dir and run_id:
        store = quant.ResidualStore(state_dir, str(run_id), str(p.get("target", "dit")))
        res_r = store.open_valid(bm, shapes, cfg_sha)
        ef = {"error_feedback": True, "residual_reset": store.reset_reason}
        res_w = store.writer(shapes)
    err2 = ref2 = 0.0
    try:
        with StReader(wpath) as r, StWriter(upath, quant.build_entries(shapes), torch.int8) as w:
            for k in names:
                d = r.read_f32(k) - trained[k].float()
                if res_r is not None:
                    d = d + res_r.read_f32(k)
                if not bool(torch.isfinite(d).all()):
                    raise FloatingPointError(f"delta of {k} is not finite")
                q, sc = quant.quantize_rows(d)
                deq = quant.dequantize_rows(q, sc)
                err = d - deq
                err2 += float(err.double().pow(2).sum())
                ref2 += float(d.double().pow(2).sum())
                w.write(k, q)
                w.write(quant.scale_name(k), sc)
                if res_w is not None:
                    res_w.write(k, err)
            usha = w.close()
        rsha = res_w.close() if res_w is not None else None
    except BaseException:
        if res_w is not None:
            res_w.abort()
            store.discard_tmp()
        raise
    finally:
        if res_r is not None:
            res_r.close()
    commit = None
    if store is not None:
        def commit():
            store.commit(rsha, base_version_id=bm["version_id"], update_id=p["update_id"], config_sha=cfg_sha)
    return usha, quant.rel_error(err2, ref2), ef, commit
