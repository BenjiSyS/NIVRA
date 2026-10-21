"""Generic resumable trainer (VAE stage and generator stage share it)."""
from __future__ import annotations

import json
import math
import os
import signal
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from . import checkpoint as ck


# ---------------------------------------------------------------- sampler / schedule / ema
class EpochSampler:
    """Deterministic epoch sampler; its position is part of every checkpoint."""

    def __init__(self, n: int, batch_size: int, seed: int):
        if n < batch_size:
            raise ValueError("dataset smaller than batch size")
        self.n, self.bs, self.seed = n, batch_size, seed
        self.epoch, self.pos = 0, 0
        self._perm = None
        self._perm_epoch = -1

    def _get_perm(self):
        if self._perm_epoch != self.epoch:
            self._perm = np.random.default_rng([self.seed, self.epoch, 0x53414D50]).permutation(self.n)
            self._perm_epoch = self.epoch
        return self._perm

    def next(self) -> np.ndarray:
        if self.pos + self.bs > self.n:
            self.epoch += 1
            self.pos = 0
        idx = self._get_perm()[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return idx

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed, "n": self.n, "bs": self.bs}

    def load_state_dict(self, d):
        if (d["seed"], d["n"], d["bs"]) != (self.seed, self.n, self.bs):
            raise ck.IncompatibleCheckpoint("sampler configuration differs from checkpoint")
        self.epoch, self.pos = d["epoch"], d["pos"]


class WarmupCosine:
    def __init__(self, base_lr, warmup, total, min_ratio=0.05):
        self.base_lr, self.warmup, self.total, self.min_ratio = base_lr, warmup, max(total, 1), min_ratio
        self.last_step = 0

    def lr_at(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * (step + 1) / max(self.warmup, 1)
        prog = min(1.0, (step - self.warmup) / max(self.total - self.warmup, 1))
        return self.base_lr * (self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))

    def state_dict(self):
        return {"type": "warmup_cosine", "base_lr": self.base_lr, "warmup": self.warmup, "total": self.total,
                "min_ratio": self.min_ratio, "last_step": self.last_step}

    def load_state_dict(self, d):
        self.last_step = d["last_step"]


class EMA:
    def __init__(self, module: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module, step: int):
        d = min(self.decay, (1 + step) / (10 + step))
        for k, v in module.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def state(self):
        return {k: v.cpu().contiguous() for k, v in self.shadow.items()}

    def load(self, tensors):
        for k in self.shadow:
            self.shadow[k].copy_(tensors[k])


# ---------------------------------------------------------------- config
@dataclass
class TrainerConfig:
    total_steps: int = 1000
    batch_size: int = 8
    lr: float = 1e-3
    warmup: int = 50
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.99)
    grad_clip: float = 1.0
    ema_decay: float = 0.0
    save_every_steps: int = 500
    save_every_minutes: float = 10.0
    keep_last: int = 3
    log_every: int = 10
    seed: int = 0
    protect_steps: tuple = ()
    device: str = "cpu"

    def to_dict(self):
        d = asdict(self)
        d["betas"] = list(self.betas)
        d["protect_steps"] = list(self.protect_steps)
        return d


class Trainer:
    """task must provide: model (nn.Module), n (int), fingerprint (str),
    get_batch(indices, gen) -> object, loss(batch, gen) -> (tensor, metrics dict)."""

    def __init__(self, task, tcfg: TrainerConfig, run_dir, compat: dict, config: dict, stop_file=None,
                 on_log: Optional[Callable] = None, should_stop: Optional[Callable] = None):
        self.task, self.cfg = task, tcfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.compat = dict(compat)
        self.compat["dataset_fingerprint"] = task.fingerprint
        self.config = config
        self.stop_file = Path(stop_file) if stop_file else self.run_dir / "STOP"
        self.on_log = on_log
        self.should_stop = should_stop
        self.model = task.model.to(tcfg.device)
        decay, no_decay = [], []
        for n_, p in self.model.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim < 2 else decay).append(p)
        groups = [{"params": decay, "weight_decay": tcfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        self.opt = torch.optim.AdamW(groups, lr=tcfg.lr, betas=tcfg.betas)
        self.sched = WarmupCosine(tcfg.lr, tcfg.warmup, tcfg.total_steps, tcfg.min_lr_ratio)
        self.ema = EMA(self.model, tcfg.ema_decay) if tcfg.ema_decay > 0 else None
        self.use_amp = tcfg.device.startswith("cuda")   # fp16 autocast only on CUDA (untested on GPU)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None
        self.sampler = EpochSampler(task.n, tcfg.batch_size, tcfg.seed)
        self.gen = torch.Generator().manual_seed(tcfg.seed + 1)
        self.step = 0
        self._stop_flag = False
        self._last_save = time.time()
        self.log_path = self.run_dir / "log.jsonl"

    # ---- checkpoint plumbing
    def save(self, protected: bool = False) -> Path:
        rng_t, rng_js = ck.capture_rng()
        rng_t["train_gen"] = ck.generator_state_tensor(self.gen)
        opt_t, opt_groups = ck.optimizer_to_flat(self.opt)
        groups = {"model": ck.module_tensors(self.model), "optimizer": opt_t, "rng": rng_t}
        if self.ema is not None:
            groups["ema"] = self.ema.state()
        state = {"rng": rng_js, "optimizer_groups": opt_groups, "scheduler": self.sched.state_dict(),
                 "scaler": self.scaler.state_dict() if self.scaler is not None else {},
                 "sampler": self.sampler.state_dict(), "step": self.step, "trainer": self.cfg.to_dict()}
        path = ck.save_checkpoint(self.run_dir, self.step, groups, state, self.compat, self.config,
                                  epoch=self.sampler.epoch, protected=protected)
        if protected or self.step in self.cfg.protect_steps:
            ck.protect(path)
        ck.apply_retention(self.run_dir, self.cfg.keep_last)
        self._last_save = time.time()
        return path

    def resume(self, ckpt_dir=None) -> bool:
        path = Path(ckpt_dir) if ckpt_dir else ck.newest_valid(self.run_dir)
        if path is None:
            return False
        m, groups, state = ck.load_checkpoint(path, expect_compat={k: v for k, v in self.compat.items()})
        self.model.load_state_dict(groups["model"])
        ck.optimizer_from_flat(self.opt, groups["optimizer"], state["optimizer_groups"])
        if self.ema is not None and "ema" in groups:
            self.ema.load(groups["ema"])
        ck.restore_rng({k: v for k, v in groups["rng"].items() if k != "train_gen"}, state["rng"])
        self.gen.set_state(groups["rng"]["train_gen"].to(torch.uint8))
        self.sched.load_state_dict(state["scheduler"])
        if self.scaler is not None and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.sampler.load_state_dict(state["sampler"])
        self.step = int(state["step"])
        return True

    # ---- loop
    def _log(self, rec: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if self.on_log:
            self.on_log(rec)

    def _handle_signal(self, *_):
        self._stop_flag = True

    def _stop_requested(self) -> bool:
        return self._stop_flag or self.stop_file.exists() or (self.should_stop is not None and self.should_stop())

    def train_step(self) -> dict:
        cfg = self.cfg
        idx = self.sampler.next()
        batch = self.task.get_batch(idx, self.gen)
        lr = self.sched.lr_at(self.step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, metrics = self.task.loss(batch, self.gen)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss, metrics = self.task.loss(batch, self.gen)
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        if not torch.isfinite(gn):
            # skip a non-finite step instead of corrupting weights
            self.opt.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.update()
            metrics = dict(metrics, skipped=1.0)
        else:
            if self.scaler is not None:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
        self.step += 1
        self.sched.last_step = self.step
        if self.ema is not None:
            self.ema.update(self.model, self.step)
        out = dict(metrics)
        out.update(loss=float(loss.item()), lr=lr, grad_norm=float(gn))
        return out

    def run(self, max_steps: Optional[int] = None, install_signals: bool = True) -> dict:
        """Train until total_steps (or max_steps more steps). Returns a status dict."""
        old = {}
        if install_signals:
            for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
                if hasattr(signal, name):
                    try:
                        old[name] = signal.signal(getattr(signal, name), self._handle_signal)
                    except ValueError:
                        pass  # not in main thread
        t_start = time.time()
        start_step = self.step
        acc, accn = {}, 0
        status = "done"
        try:
            while self.step < self.cfg.total_steps:
                if max_steps is not None and self.step - start_step >= max_steps:
                    status = "paused"
                    break
                if self._stop_requested():
                    self.save()
                    status = "stopped"
                    break
                t0 = time.time()
                m = self.train_step()
                dt = time.time() - t0
                for k, v in m.items():
                    acc[k] = acc.get(k, 0.0) + v
                accn += 1
                if self.step % self.cfg.log_every == 0 or self.step == self.cfg.total_steps:
                    rec = {k: v / accn for k, v in acc.items()}
                    rec.update(step=self.step, epoch=self.sampler.epoch, sec_per_step=dt,
                               elapsed=time.time() - t_start)
                    self._log(rec)
                    acc, accn = {}, 0
                due_steps = self.cfg.save_every_steps > 0 and self.step % self.cfg.save_every_steps == 0
                due_time = self.cfg.save_every_minutes > 0 and (time.time() - self._last_save) > self.cfg.save_every_minutes * 60
                if due_steps or due_time or self.step in self.cfg.protect_steps:
                    self.save()
            else:
                self.save()
        finally:
            for name, h in old.items():
                signal.signal(getattr(signal, name), h)
        return {"status": status, "step": self.step, "elapsed": time.time() - t_start,
                "steps_run": self.step - start_step}
