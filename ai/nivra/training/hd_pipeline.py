"""Single-machine HD training pipeline: captioned videos -> a bundle the Nivra app / controller can load.

    ingest -> tokenizer (byte BPE) -> text (NivraText MLM) -> vae (HC-VAE) -> latents (latent_shard format)
           -> dit (rectified-flow DiT) -> bundle (write_hd_bundle)

Every stage writes ``<run>/markers/<stage>.json`` when it finishes; the marker stores a fingerprint of the stage's
inputs (options + the output fingerprints of the stages it depends on). A stage is skipped when its marker matches
the current inputs and its output files still exist; changing an option or an upstream result re-runs it and
everything downstream. The two long training stages (vae, dit) additionally resume from their own step
checkpoints, so Ctrl+C and restart loses at most ``save_every`` steps.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

STAGES = ["ingest", "tokenizer", "text", "vae", "latents", "dit", "bundle"]
EFFECTIVE_FLOPS = {"cpu": 1.2e11, "cuda": 8.0e12}        # sustained, order-of-magnitude (RTX 2060 class for cuda)
LATENT_DS_FORMAT = "nivra-latents-dataset/1"


class PipelineError(Exception):
    pass


@dataclass
class PipelineOptions:
    config: str = "nivra_base"
    out: str = "runs/hd"
    videos: Optional[str] = None
    dataset: Optional[str] = None
    bucket: str = "256p2s"
    profile: str = "stylized"
    default_style: Optional[str] = None
    fps: Optional[float] = None
    max_clips: Optional[int] = None
    val_fraction: float = 0.05
    vocab_size: int = 4000
    text_steps: int = 1500
    vae_steps: int = 4000
    dit_steps: int = 8000
    seed: int = 0
    device: str = "auto"
    style_dropout: Optional[float] = None
    name: Optional[str] = None
    install_dir: Optional[str] = None
    bundle_out: Optional[str] = None
    from_stage: str = "ingest"
    to_stage: str = "bundle"
    force: bool = False
    save_every: Optional[int] = None


def _fp(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _sha_file(p) -> str:
    from ..distributed.hd.stream_io import sha256_file
    return sha256_file(p)


def resolve_device(d: str) -> str:
    if d != "auto":
        return d
    env = os.environ.get("NIVRA_DEVICE")
    if env:
        return env if not env.startswith("cuda") or torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def floor_mult(n: int, m: int) -> int:
    return max(m, (int(n) // m) * m)


def fit_stages(stages: list, height: int, width: int, frames: int) -> list:
    """Stages whose clip geometry fits inside (height, width, frames); never empty."""
    ok = [dict(s) for s in stages if s["height"] <= height and s["width"] <= width and s["frames"] <= frames]
    if ok:
        return ok
    f = 1 + 8 * ((frames - 1) // 8)
    return [{"name": "dataset", "height": floor_mult(height, 32), "width": floor_mult(width, 32), "frames": f, "steps": 1}]


def scale_stages(stages: list, total: int) -> list:
    """Same geometry, steps scaled to sum to exactly `total` (stages that would get 0 steps are dropped)."""
    total = max(1, int(total))
    base = sum(int(s["steps"]) for s in stages) or 1
    out = [dict(s, steps=max(1, round(int(s["steps"]) * total / base))) for s in stages]
    while len(out) > 1 and sum(s["steps"] for s in out) > total and out[-1]["steps"] <= sum(s["steps"] for s in out) - total:
        out.pop()
    out[-1]["steps"] = max(1, out[-1]["steps"] + total - sum(s["steps"] for s in out))
    return out


def text_config_for(dim: int, vocab: int):
    from ..text.nivra_text import NivraTextConfig
    for size, d in (("tiny", 64), ("small", 256), ("base", 768)):
        if d == dim:
            return NivraTextConfig.preset(size, vocab)
    heads = next((h for h in (8, 4, 2, 1) if dim % h == 0 and (dim // h) % 2 == 0), None)
    if heads is None:
        raise PipelineError(f"text_dim {dim} cannot be split into attention heads with an even head size")
    return NivraTextConfig("custom", vocab, dim, 2 if dim <= 32 else 4, heads, 4 * dim, 64)


class Pipeline:
    def __init__(self, opts: PipelineOptions, log: Callable = print):
        from ..models.hd_config import load_hd_config
        self.o, self.log = opts, log
        if opts.videos and opts.dataset:
            raise PipelineError("give exactly one of --videos (captioned folder) or --dataset (already ingested)")
        if not opts.videos and not opts.dataset and opts.from_stage == "ingest":
            raise PipelineError("give --videos (captioned folder) or --dataset (already ingested), or --from-stage later")
        for s in (opts.from_stage, opts.to_stage):
            if s not in STAGES:
                raise PipelineError(f"unknown stage {s!r}; stages: {', '.join(STAGES)}")
        if STAGES.index(opts.from_stage) > STAGES.index(opts.to_stage):
            raise PipelineError("--from-stage comes after --to-stage")
        self.cfg = load_hd_config(opts.config)
        if opts.default_style:
            from ..data.styles import normalize_style
            ds = normalize_style(opts.default_style)
            if ds is None:
                raise PipelineError(f"unknown style {opts.default_style!r}")
            self.cfg["default_style"] = ds
            if self.cfg["styles"] and ds not in self.cfg["styles"]:
                self.cfg["styles"].append(ds)
        self.root = Path(opts.out)
        self.markers = self.root / "markers"
        self.device = resolve_device(opts.device)
        self.results: dict = {}

    # ------------------------------------------------------------------ paths
    def p(self, *a) -> Path:
        return self.root.joinpath(*a)

    def dataset_dir(self) -> Path:
        return Path(self.o.dataset) if self.o.dataset else self.p("dataset")

    # ------------------------------------------------------------------ markers
    def marker(self, stage: str) -> Optional[dict]:
        f = self.markers / f"{stage}.json"
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_marker(self, stage: str, inputs_fp: str, outputs: dict, seconds: float) -> None:
        self.markers.mkdir(parents=True, exist_ok=True)
        m = {"stage": stage, "inputs_fp": inputs_fp, "outputs": outputs, "outputs_fp": _fp(outputs),
             "seconds": round(seconds, 1), "finished_unix": int(time.time())}
        tmp = self.markers / f"{stage}.json.tmp"
        tmp.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.markers / f"{stage}.json")

    def _outputs_exist(self, stage: str, outputs: dict) -> bool:
        return all(Path(p).exists() for p in outputs.get("files", []))

    # ------------------------------------------------------------------ stage inputs (fingerprints)
    def videos_listing(self) -> list:
        root = Path(self.o.videos)
        exts = {".mp4", ".mov", ".mkv", ".webm", ".txt", ".json", ".jsonl"}
        out = []
        for q in sorted(root.rglob("*")):
            if q.is_file() and q.suffix.lower() in exts:
                st = q.stat()
                out.append((q.relative_to(root).as_posix(), st.st_size, int(st.st_mtime)))
        return out

    def stage_inputs(self, stage: str, up: dict) -> dict:
        o = self.o
        if stage == "ingest":
            if o.dataset:
                try:
                    ds = json.loads((Path(o.dataset) / "dataset.json").read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    raise PipelineError(f"--dataset {o.dataset}: dataset.json unreadable ({e})") from e
                return {"dataset_fingerprint": ds.get("fingerprint")}
            return {"listing": self.videos_listing(), "bucket": o.bucket, "profile": o.profile, "fps": o.fps,
                    "max_clips": o.max_clips, "val_fraction": o.val_fraction, "default_style": self.cfg["default_style"]}
        if stage == "tokenizer":
            return {"data": up["ingest"], "vocab_size": o.vocab_size}
        if stage == "text":
            return {"tok": up["tokenizer"], "data": up["ingest"], "steps": o.text_steps, "seed": o.seed,
                    "dim": self.cfg["model"]["text_dim"]}
        if stage == "vae":
            return {"data": up["ingest"], "steps": o.vae_steps, "seed": o.seed, "cfg": self.cfg["vae"],
                    "train": self.cfg["train"], "device_kind": self.device.split(":")[0]}
        if stage == "latents":
            return {"data": up["ingest"], "vae": up["vae"], "text": up["text"], "cfg_train": self.cfg["train"]["stages"],
                    "style_dropout": self.style_dropout()}
        if stage == "dit":
            return {"latents": up["latents"], "steps": o.dit_steps, "seed": o.seed, "model": self.cfg["model"],
                    "train": self.cfg["train"], "device_kind": self.device.split(":")[0]}
        if stage == "bundle":
            return {"dit": up["dit"], "vae": up["vae"], "text": up["text"], "name": o.name, "style": self.cfg["default_style"]}
        raise AssertionError(stage)

    def style_dropout(self) -> float:
        return float(self.o.style_dropout if self.o.style_dropout is not None else self.cfg["train"].get("style_dropout", 0.1))

    # ------------------------------------------------------------------ plan
    def plan(self) -> list:
        """[(stage, action, detail)] with action in run | cached | skip; also a time estimate in seconds."""
        up, dirty, rows = {}, False, []
        i_from, i_to = STAGES.index(self.o.from_stage), STAGES.index(self.o.to_stage)
        for i, st in enumerate(STAGES):
            m = self.marker(st)
            if i < i_from:
                ok = m is not None and not dirty and self._outputs_exist(st, m["outputs"])
                if not ok and i <= i_to:
                    raise PipelineError(f"--from-stage {self.o.from_stage}: stage {st!r} has no finished result in {self.root}")
                up[st] = m["outputs_fp"] if m else None
                rows.append((st, "cached", f"finished earlier ({m['seconds']}s)" if m else ""))
                continue
            if i > i_to:
                rows.append((st, "skip", "after --to-stage"))
                continue
            try:
                ins_fp = _fp(self.stage_inputs(st, {k: v for k, v in up.items()}))
            except KeyError:
                ins_fp = None
            valid = (not self.o.force and not dirty and m is not None and m["inputs_fp"] == ins_fp
                     and self._outputs_exist(st, m["outputs"]))
            if valid:
                up[st] = m["outputs_fp"]
                rows.append((st, "cached", f"inputs unchanged ({m['seconds']}s last time)"))
            else:
                dirty = True
                up[st] = None
                rows.append((st, "run", ""))
        return rows

    def estimates(self) -> dict:
        """Rough seconds per stage from FLOP counts and the effective-throughput table (order of magnitude only)."""
        from ..compression.hc_vae import shape_trace
        from ..models.dit import count_params_meta
        from ..models.hd_config import dit_config, vae_config
        o, cfg = self.o, self.cfg
        dev = "cuda" if self.device.startswith("cuda") else "cpu"
        rate = EFFECTIVE_FLOPS[dev]
        tc = cfg["train"]
        vc, dc = vae_config(cfg), dit_config(cfg)
        stages = fit_stages(tc["stages"], *self.geometry())
        st = stages[-1]
        try:
            tr = shape_trace(vc, st["frames"], st["height"], st["width"])
            vae_step = 3.0 * (tr["encode_flops"] + tr["decode_flops"]) * int(tc.get("batch_size", 1)) * int(tc.get("grad_accum", 1))
            enc = tr["encode_flops"]
        except Exception:  # noqa: BLE001
            vae_step, enc = 0.0, 0.0
        n_tok = (1 + (st["frames"] - 1) // vc.temporal_factor) * (st["height"] // vc.spatial_factor) * (st["width"] // vc.spatial_factor)
        P = count_params_meta(dc)["total"]
        bs = int(tc.get("batch_size", 1)) * int(tc.get("grad_accum", 1))
        dit_step = 8.0 * P * n_tok * bs                       # fwd + bwd + activation recompute
        text_params = text_config_for(int(cfg["model"]["text_dim"]), o.vocab_size)
        from ..text.nivra_text import count_params_for
        text_step = 6.0 * count_params_for(text_params) * 16 * 64
        clips = self.clip_count()
        return {"vae": o.vae_steps * vae_step / rate, "dit": o.dit_steps * dit_step / rate,
                "text": o.text_steps * text_step / rate, "latents": clips * enc / rate,
                "tokenizer": 5.0 + clips * 0.01, "ingest": clips * 1.5 if o.videos else 0.0, "bundle": 20.0,
                "_params": P, "_tokens": n_tok, "_device": dev}

    def geometry(self) -> tuple:
        try:
            ds = json.loads((self.dataset_dir() / "dataset.json").read_text(encoding="utf-8"))
            return int(ds["height"]), int(ds["width"]), int(ds["frames"])
        except (OSError, ValueError, KeyError):
            from ..data.ingest import BUCKETS
            b = BUCKETS[self.o.bucket]
            return b.height, b.width, b.frames

    def clip_count(self) -> int:
        try:
            return int(json.loads((self.dataset_dir() / "dataset.json").read_text(encoding="utf-8"))["clips"])
        except (OSError, ValueError, KeyError):
            return int(self.o.max_clips or 100)

    def format_plan(self) -> str:
        rows = self.plan()
        est = self.estimates()
        lines = [f"Nivra train-pipeline: config {self.o.config}  device {self.device}  run dir {self.root}",
                 f"  DiT {est['_params'] / 1e6:.0f}M params, {est['_tokens']} tokens at the largest stage; "
                 f"times are rough estimates for {est['_device']} (~{EFFECTIVE_FLOPS[est['_device']] / 1e12:.2g} TFLOP/s sustained)",
                 f"  {'stage':<10}{'action':<8}{'steps':>8}  {'est. time':>10}  note"]
        steps = {"text": self.o.text_steps, "vae": self.o.vae_steps, "dit": self.o.dit_steps}
        total = 0.0
        for st, act, note in rows:
            t = est.get(st, 0.0) if act == "run" else 0.0
            total += t
            lines.append(f"  {st:<10}{act:<8}{steps.get(st, ''):>8}  {fmt_dur(t) if act == 'run' else '-':>10}  {note}")
        lines.append(f"  total estimated: {fmt_dur(total)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        i_from, i_to = STAGES.index(self.o.from_stage), STAGES.index(self.o.to_stage)
        self.log(self.format_plan())
        up, dirty = {}, False
        for i, st in enumerate(STAGES):
            if i > i_to:
                break
            m = self.marker(st)
            if i < i_from:
                up[st] = m["outputs_fp"]
                self.results[st] = m["outputs"]
                continue
            ins_fp = _fp(self.stage_inputs(st, up))
            if (not self.o.force and not dirty and m is not None and m["inputs_fp"] == ins_fp
                    and self._outputs_exist(st, m["outputs"])):
                self.log(f"[{st}] cached")
                up[st] = m["outputs_fp"]
                self.results[st] = m["outputs"]
                continue
            dirty = True
            self.log(f"[{st}] start")
            t0 = time.time()
            outputs = getattr(self, f"stage_{st}")()
            self._write_marker(st, ins_fp, outputs, time.time() - t0)
            up[st] = _fp(outputs)
            self.results[st] = outputs
            self.log(f"[{st}] done in {fmt_dur(time.time() - t0)}")
        return self.results

    # ------------------------------------------------------------------ stages
    def stage_ingest(self) -> dict:
        if self.o.dataset:
            ds_dir = Path(self.o.dataset)
        else:
            from ..data.ingest import IngestError, IngestOptions, ingest_folder
            ds_dir = self.p("dataset")
            io = IngestOptions(bucket=self.o.bucket, profile=self.o.profile, default_style=self.cfg["default_style"],
                               fps=self.o.fps, max_clips=self.o.max_clips, val_fraction=self.o.val_fraction, overwrite=True)
            try:
                ingest_folder(self.o.videos, ds_dir, io, log=self.log)
            except IngestError as e:
                raise PipelineError(f"ingest failed: {e}") from e
        ds = json.loads((ds_dir / "dataset.json").read_text(encoding="utf-8"))
        if int(ds.get("train_clips", 0)) < 1:
            raise PipelineError(f"{ds_dir}: the dataset has no train clips (videos too short / missing captions? "
                                f"see the 'rejected' counts in dataset.json: {ds.get('rejected')})")
        if int(ds.get("val_clips", 0)) < 1:
            self.log("WARNING: the dataset has no validation split (too few videos for --val-fraction "
                     f"{self.o.val_fraction}). Local training is unaffected, but network training with "
                     "eval_gate=pi needs held-out validation shards and will not be possible with this dataset; "
                     "add more videos or raise --val-fraction.")
        return {"files": [str(ds_dir / "dataset.json"), str(ds_dir / "manifest.jsonl")], "dataset_dir": str(ds_dir),
                "fingerprint": ds["fingerprint"], "clips": ds["clips"], "train_clips": ds["train_clips"],
                "val_clips": ds["val_clips"], "styles": ds.get("styles")}

    def manifest_rows(self) -> list:
        from ..data.shards import read_manifest
        return read_manifest(self.dataset_dir())

    def stage_tokenizer(self) -> dict:
        from ..text.bpe import BPETokenizer
        from ..training.hd_trainer import training_caption
        rows = self.manifest_rows()
        lines = sorted({training_caption(r, 0.0) for r in rows} | {r.get("caption", "") for r in rows} - {""})
        corpus = self.p("text", "corpus.txt")
        corpus.parent.mkdir(parents=True, exist_ok=True)
        reps = 1
        text = "\n".join(lines) + "\n"
        reps = int(min(500, max(1, math.ceil(30000 / max(1, len(text))))))
        corpus.write_text(text * reps, encoding="utf-8")
        tok = BPETokenizer.train_files([corpus], self.o.vocab_size, 2)
        sha = tok.save(self.p("text", "tokenizer.json"))
        self.log(f"[tokenizer] vocab {tok.vocab_size} from {len(lines)} distinct caption lines")
        return {"files": [str(self.p("text", "tokenizer.json")), str(corpus)], "tokenizer": str(self.p("text", "tokenizer.json")),
                "corpus": str(corpus), "sha256": sha, "vocab_size": tok.vocab_size}

    def stage_text(self) -> dict:
        from ..text.bpe import BPETokenizer
        from ..text.nivra_text import build_token_stream, pretrain_mlm
        tok = BPETokenizer.load(self.p("text", "tokenizer.json"))
        cfg = text_config_for(int(self.cfg["model"]["text_dim"]), tok.vocab_size)
        corpus = self.p("text", "corpus.txt")
        n_tok = len(build_token_stream([corpus], tok))
        seq = int(max(8, min(cfg.max_len, n_tok // 8)))
        out = self.p("text_encoder")
        res = pretrain_mlm([corpus], tok, out, steps=self.o.text_steps, batch=16, seq_len=seq, lr=1e-3,
                           seed=self.o.seed, device=self.device, cfg=cfg, log=self.log)
        h = res["history"]
        self.log(f"[text] MLM loss {h[0][1]:.3f} -> {h[-1][1]:.3f} (random baseline {res['random_baseline_loss']:.3f})")
        w = out / "model.safetensors"
        return {"files": [str(w), str(out / "tokenizer.json"), str(out / "config.json")], "dir": str(out),
                "sha256": _sha_file(w), "id": res["meta"]["id"], "params": res["meta"]["params"],
                "first_loss": h[0][1], "last_loss": h[-1][1]}

    def train_cfg(self, total: int, stages: Optional[list] = None) -> dict:
        cfg = copy.deepcopy(self.cfg)
        h, w, f = self.geometry()
        base = stages if stages is not None else cfg["train"]["stages"]
        cfg["train"]["stages"] = scale_stages(fit_stages(base, h, w, f), total)
        if self.o.save_every:
            cfg["train"]["save_every"] = self.o.save_every
        return cfg

    def stage_vae(self) -> dict:
        from ..compression.hc_vae import save_hcvae
        from ..training.hd_trainer import train_hd
        cfg = self.train_cfg(self.o.vae_steps)
        self.log(f"[vae] stages {[(s['name'], s['height'], s['width'], s['frames'], s['steps']) for s in cfg['train']['stages']]}")
        res = train_hd(cfg, self.dataset_dir(), self.p("vae_run"), target="vae", total_steps=self.o.vae_steps,
                       device=self.device, seed=self.o.seed, resume=True, log=self.log)
        if res["status"] != "done":
            raise PipelineError(f"VAE training ended with status {res['status']}")
        vdir = Path(res["vae_dir"])
        hist = res["history"]
        return {"files": [str(vdir / "hc_vae.safetensors"), str(vdir / "hc_vae.json")], "dir": str(vdir),
                "latent_norm": res["latent_norm"], "sha256": _sha_file(vdir / "hc_vae.safetensors"),
                "params": res["params"], "first_l1": hist[0].get("l1"), "last_l1": hist[-1].get("l1")}

    def load_vae(self):
        from ..compression.hc_vae import load_hcvae
        return load_hcvae(self.results["vae"]["dir"])

    def load_text(self):
        from ..text.nivra_text import NivraTextEncoder
        return NivraTextEncoder.from_checkpoint(self.results["text"]["dir"], device=self.device)

    def encode_stage_geometry(self) -> dict:
        h, w, f = self.geometry()
        stages = self.cfg["train"]["stages"]
        sh = [s for s in stages if s["height"] <= h and s["width"] <= w and s["frames"] <= f] or stages
        eh = floor_mult(min(h, max(s["height"] for s in sh)), 32)
        ew = floor_mult(min(w, max(s["width"] for s in sh)), 32)
        ef = min(f, max(s["frames"] for s in sh))
        ef = 1 + 8 * ((ef - 1) // 8)
        ds = json.loads((self.dataset_dir() / "dataset.json").read_text(encoding="utf-8"))
        return {"h": eh, "w": ew, "frames": ef, "fps": int(round(float(ds.get("fps", 8))))}

    def stage_latents(self) -> dict:
        from safetensors import safe_open
        from ..distributed.hd.encode_job import encode_shard
        ds_dir = self.dataset_dir()
        ds = json.loads((ds_dir / "dataset.json").read_text(encoding="utf-8"))
        rows = self.manifest_rows()
        vae, enc = self.load_vae().to(self.device).eval(), self.load_text()
        stage = self.encode_stage_geometry()
        vsha, tsha = self.results["vae"]["sha256"], self.results["text"]["sha256"]
        out = self.p("latents")
        if out.exists():
            for q in out.glob("lat-*.safetensors"):
                q.unlink()
        out.mkdir(parents=True, exist_ok=True)
        new_rows, files = [], []
        for sh in ds["shards"]:
            mine = sorted((r for r in rows if r["shard"] == sh["file"]), key=lambda r: r["index"])
            with safe_open(str(ds_dir / sh["file"]), framework="pt") as f:
                videos = f.get_tensor("videos")
            name = "lat-" + sh["file"].split("shard-", 1)[1]
            s = encode_shard(None, out / name, vae=vae, enc=enc, stage=stage, parent_sha=sh["sha256"], vae_version=vsha,
                             text_encoder_id=tsha, style_dropout=self.style_dropout(), seed=self.o.seed, device=self.device,
                             extra_meta={"shard_idx": len(files)}, videos_override=videos[: len(mine)], rows_override=mine,
                             split_override=sh["split"])
            self.log(f"[latents] {name}: {s['clips']} clips, latent mean {s['latent_mean']:.3f} std {s['latent_std']:.3f}")
            for r in mine:
                nr = {k: r[k] for k in ("id", "split", "caption", "style", "lang") if k in r}
                nr.update(shard=name, index=r["index"])
                new_rows.append(nr)
            files.append({"file": name, "split": sh["split"], "clips": s["clips"], "sha256": s["sha256"],
                          "bytes": (out / name).stat().st_size, "parent_shard_sha256": sh["sha256"]})
        mtext = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in new_rows)
        (out / "manifest.jsonl").write_bytes(mtext.encode("utf-8"))
        fp = hashlib.sha256(mtext.encode("utf-8")).hexdigest()
        info = {"format": LATENT_DS_FORMAT, "fingerprint": fp, "parent_fingerprint": ds["fingerprint"], "vae_version": vsha,
                "text_encoder_id": tsha, "stage": stage, "latent_shift": 0.0, "latent_scale": 1.0, "scaled": True,
                "clips": len(new_rows), "shards": files}
        (out / "dataset.json").write_text(json.dumps(info, indent=1, sort_keys=True), encoding="utf-8")
        return {"files": [str(out / "dataset.json"), str(out / "manifest.jsonl")] + [str(out / f["file"]) for f in files],
                "dir": str(out), "fingerprint": fp, "stage": stage, "clips": len(new_rows)}

    def stage_dit(self) -> dict:
        from ..models.hd_config import build_dit
        from ..training.hd_trainer import train_hd
        lat = self.results["latents"]
        stage = lat["stage"]
        cfg = self.train_cfg(self.o.dit_steps)
        cfg["train"]["stages"] = scale_stages(fit_stages(self.cfg["train"]["stages"], stage["h"], stage["w"], stage["frames"]),
                                              self.o.dit_steps)
        self.log(f"[dit] stages {[(s['name'], s['height'], s['width'], s['frames'], s['steps']) for s in cfg['train']['stages']]}")
        torch.manual_seed(self.o.seed)
        net = build_dit(cfg)
        res = train_hd(cfg, lat["dir"], self.p("dit_run"), target="dit", text_encoder=self.load_text(), vae=self.load_vae(),
                       total_steps=self.o.dit_steps, device=self.device, seed=self.o.seed, resume=True, log=self.log,
                       latent_norm=(0.0, 1.0), model=net)
        if res["status"] != "done":
            raise PipelineError(f"DiT training ended with status {res['status']}")
        from safetensors.torch import save_file
        net.cpu()
        w = self.p("dit_final", "dit.safetensors")
        w.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().contiguous() for k, v in net.state_dict().items()}, str(w))
        hist = res["history"]
        k = max(1, min(10, len(hist) // 5))
        return {"files": [str(w)], "weights": str(w), "sha256": _sha_file(w), "params": res["params"],
                "loss_first": sum(h["loss"] for h in hist[:k]) / k, "loss_last": sum(h["loss"] for h in hist[-k:]) / k,
                "steps": res["step"]}

    def bundle_dir(self) -> Path:
        o = self.o
        name = o.name or f"{self.cfg['name']}-{self.results['dit']['sha256'][:8]}"
        if o.bundle_out:
            return Path(o.bundle_out)
        if o.install_dir:
            return Path(o.install_dir) / "models" / name
        return self.p("bundle")

    def stage_bundle(self) -> dict:
        from safetensors.torch import load_file
        from ..inference.hd_bundle import load_hd_bundle, write_hd_bundle
        cfg = copy.deepcopy(self.cfg)
        ln = self.results["vae"]["latent_norm"]
        cfg["vae"]["latent_shift"], cfg["vae"]["latent_scale"] = ln["latent_shift"], ln["latent_scale"]
        out = self.bundle_dir()
        if out.exists():
            for q in ("model_version.json", "weights.safetensors", "vae.safetensors"):
                (out / q).unlink(missing_ok=True)
        m = write_hd_bundle(out, cfg, load_file(self.results["dit"]["weights"]),
                            load_file(str(Path(self.results["vae"]["dir"]) / "hc_vae.safetensors")),
                            text_encoder=f"nivra:{self.results['text']['dir']}", seed=self.o.seed,
                            version_id=None)
        load_hd_bundle(out, "cpu")                      # fail here, not in the app, if the bundle does not load
        self.log(f"[bundle] wrote {out}  (version {m['version_id']}, {m['param_count'] / 1e6:.1f}M params)")
        return {"files": [str(out / "model_version.json"), str(out / "weights.safetensors"), str(out / "vae.safetensors")],
                "dir": str(out), "version_id": m["version_id"], "param_count": m["param_count"],
                "weights_sha256": m["weights_sha256"]}


def fmt_dur(s: float) -> str:
    s = float(s)
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"
