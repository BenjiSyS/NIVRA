"""Nivra command line.

Human commands:  train-vae | train | generate | continue | evaluate | info
Executor:        exec --job job.json   (protocol nivra.exec/1, see docs/architecture/EXECUTOR_PROTOCOL.md)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKER_JOB_TYPES = ["ai.dataset_shard", "ai.local_train", "ai.evaluate", "ai.generate"]
CONTROLLER_JOB_TYPES = ["ai.init_model", "ai.validate_update", "ai.aggregate"]


def _cmd_train_vae(a):
    from .config import load_config
    from .training.pipeline import train_vae
    cfg = load_config(a.config)
    over = {}
    if a.save_every_steps is not None:
        over["save_every_steps"] = a.save_every_steps
    res = train_vae(cfg, a.run, a.steps, seed=a.seed, resume=not a.no_resume, **over)
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=str))
    return 0 if res["status"] == "done" else 3


def _cmd_train(a):
    from .config import load_config
    from .training.pipeline import train_generator
    cfg = load_config(a.config)
    over = {}
    if a.save_every_steps is not None:
        over["save_every_steps"] = a.save_every_steps
    if a.protect:
        over["protect_steps"] = tuple(int(s) for s in a.protect.split(","))
    res = train_generator(cfg, a.vae, a.run, a.steps, seed=a.seed, resume=not a.no_resume, **over)
    print(json.dumps({k: v for k, v in res.items() if k != "export"}, indent=1, default=str))
    return 0 if res["status"] == "done" else 3


def _cmd_generate_hd(a, dev):
    from .inference.bundle import BundleError
    from .inference.hd_bundle import hd_summary, load_hd_bundle, save_hd_result
    if a.upscale not in (None, 1):
        print("error: --upscale must be 1 (or omitted) for HD models", file=sys.stderr)
        return 2
    over = {}
    if a.fp16 is not None:
        over["fp16"] = a.fp16
    if a.offload is not None:
        over["offload"] = a.offload == "blocks"
    try:
        pipe = load_hd_bundle(a.bundle, dev, **over)
    except BundleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 5
    out = Path(a.out)
    kw = {k: v for k, v in (("steps", a.steps), ("cfg_scale", a.cfg), ("fps", a.fps), ("width", a.width),
                            ("height", a.height), ("seconds", a.seconds), ("sampler", a.sampler)) if v is not None}
    res = pipe.generate(a.prompt, seed=a.seed, out_path=out / f"{a.name}.mp4", **kw)
    meta = save_hd_result(res, out, a.name)
    s = hd_summary(meta)
    s["mp4"] = meta["mp4"]
    print(json.dumps(s))
    return 0


def _cmd_generate(a):
    from .config import setup_runtime
    from .inference.bundle import load_bundle
    from .inference.generate import generate, save_result
    dev = setup_runtime()
    from .inference.hd_bundle import read_manifest_if_hd
    if read_manifest_if_hd(Path(a.bundle) / "model_version.json") is not None:
        return _cmd_generate_hd(a, dev)
    hd_only = [f"--{k}" for k in ("width", "height", "seconds", "sampler", "offload") if getattr(a, k) is not None]
    if a.fp16 is not None:
        hd_only.append("--fp16/--no-fp16")
    if hd_only:
        print(f"error: {', '.join(hd_only)} are only valid for HD bundles", file=sys.stderr)
        return 2
    steps = 30 if a.steps is None else a.steps
    cfg = 3.0 if a.cfg is None else a.cfg
    fps = 8 if a.fps is None else a.fps
    upscale = 4 if a.upscale is None else a.upscale
    b = load_bundle(a.bundle, dev)
    res = generate(b, a.prompt, seed=a.seed, steps=steps, cfg_scale=cfg, tile=a.tile, device=dev)
    meta = save_result(res, a.out, a.name, fps=fps, upscale=upscale)
    print(json.dumps({k: meta[k] for k in ("mp4", "frames", "fps", "seconds_sampling", "seconds_decode", "peak_rss_mb")}))
    return 0


def _cmd_init_hd_bundle(a):
    import torch
    from .inference.bundle import BundleError
    from .inference.hd_bundle import write_hd_bundle
    try:
        m = write_hd_bundle(a.out, a.config, text_encoder=a.text_encoder, seed=a.seed,
                            dtype={"auto": None, "fp32": torch.float32, "fp16": torch.float16}[a.dtype])
    except (BundleError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"version_id": m["version_id"], "param_count": m["param_count"], "min_vram_mb": m["min_vram_mb"],
                      "weights_sha256": m["weights_sha256"], "vae_sha256": m["vae_sha256"],
                      "text_encoder": m["text_encoder"]["id"]}))
    return 0


def _cmd_continue(a):
    from .config import setup_runtime
    from .inference.bundle import load_bundle
    from .inference.generate import GenerationState, continue_video, save_result
    dev = setup_runtime()
    b = load_bundle(a.bundle, dev)
    st = GenerationState.load(a.state, expect=b)
    res = continue_video(b, st, prompt=a.prompt, seed=a.seed, steps=a.steps, cfg_scale=a.cfg,
                         known_latent_frames=a.known, tile=a.tile, device=dev)
    meta = save_result(res, a.out, a.name, fps=a.fps, upscale=a.upscale)
    print(json.dumps({k: meta[k] for k in ("mp4", "frames", "segments", "temporal_position", "seconds_sampling")}))
    return 0


def _cmd_evaluate(a):
    from .config import setup_runtime
    from .evaluation import benchmark as bm
    dev = setup_runtime()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "detector":
        rep = bm.detector_validation(a.clips)
    else:
        from .inference.bundle import load_bundle
        b = load_bundle(a.bundle, dev)
        prompts = bm.load_prompts(a.prompts)
        rep = bm.run_benchmark(b, prompts, a.seeds, a.steps, a.cfg, save_dir=a.save_videos)
    out.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(bm.summarize(rep) if a.mode != "detector" else rep, indent=1))
    return 0


def _cmd_init_random(a):
    from .config import load_config
    from .inference.bundle import init_random_bundle
    m = init_random_bundle(load_config(a.config), a.vae, a.out, a.seed)
    print(json.dumps({"version_id": m["version_id"], "param_count": m["param_count"]}))
    return 0


def _cmd_info(a):
    from .config import load_config, build_gen, build_vae, gen_config, vae_config, synth_config
    from .models.nivra_small import count_params
    from .text.tokenizer import default_tokenizer
    cfg = load_config(a.config)
    tok = default_tokenizer()
    info = {"config": cfg["name"], "vae_params": count_params(build_vae(cfg)), "generator_params": count_params(build_gen(cfg)),
            "latent_shape": list(vae_config(cfg).latent_shape(synth_config(cfg).frames, synth_config(cfg).size)),
            "vocab_size": len(tok), "vocab_hash": tok.vocab_hash}
    print(json.dumps(info, indent=1))
    return 0


def _cmd_text_train_tokenizer(a):
    import time
    from .text.bpe import BPETokenizer, expand_text_paths
    files = expand_text_paths(a.input)
    t0 = time.time()
    tok = BPETokenizer.train_files(files, a.vocab_size, a.min_frequency)
    dt = time.time() - t0
    sha = tok.save(a.out)
    nbytes = sum(f.stat().st_size for f in files)
    print(json.dumps({"out": a.out, "vocab_size": tok.vocab_size, "sha256": sha, "files": len(files),
                      "train_bytes": nbytes, "train_seconds": round(dt, 2)}))
    return 0


def _cmd_text_pretrain(a):
    from .config import setup_runtime
    from .text.bpe import BPETokenizer
    from .text.nivra_text import pretrain_mlm
    rt = setup_runtime()
    dev = rt if a.device == "auto" else a.device
    tok = BPETokenizer.load(a.tokenizer)
    res = pretrain_mlm(a.corpus, tok, a.out, size=a.size, steps=a.steps, batch=a.batch, seq_len=a.seq_len, lr=a.lr,
                       seed=a.seed, device=dev, fp16=a.fp16, save_every=a.save_every_steps)
    h = res["history"]
    print(json.dumps({"out": a.out, "id": res["meta"]["id"], "params": res["meta"]["params"], "steps": len(h),
                      "first_loss": h[0][1], "last_loss": h[-1][1], "random_baseline_loss": res["random_baseline_loss"],
                      "tokens": res["tokens"], "seconds": round(res["seconds"], 1)}))
    return 0


def _cmd_data_ingest(a):
    from .data.ingest import IngestError, IngestOptions, ingest_folder
    opts = IngestOptions(bucket=a.bucket, fps=a.fps, max_clips=a.max_clips,
                         scene_threshold=None if a.no_scenes else a.scene_threshold, shard_bytes=a.shard_mb * 1024 * 1024,
                         val_fraction=a.val_fraction, dedup_bits=a.dedup_bits, allow_upscale=a.allow_upscale,
                         default_lang=a.default_lang, overwrite=a.overwrite)
    try:
        s = ingest_folder(a.input, a.output, opts)
    except IngestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: s[k] for k in ("fingerprint", "bucket", "fps", "clips", "train_clips", "val_clips", "videos_seen", "rejected")}, indent=1))
    return 0


def _cmd_data_inspect(a):
    from .data.ingest import inspect_dataset
    rep = inspect_dataset(a.dataset)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if rep["ok"] else 1


def build_parser():
    p = argparse.ArgumentParser(prog="nivra")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("train-vae", help="train the video tokenizer (VAE)")
    s.add_argument("--config", default="nivra_small")
    s.add_argument("--run", required=True)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--save-every-steps", type=int)
    s.add_argument("--no-resume", action="store_true")
    s.set_defaults(fn=_cmd_train_vae)

    s = sub.add_parser("train", help="train the NivraSmall generator on frozen VAE latents")
    s.add_argument("--config", default="nivra_small")
    s.add_argument("--vae", required=True, help="path to vae.safetensors")
    s.add_argument("--run", required=True)
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--save-every-steps", type=int)
    s.add_argument("--protect", help="comma separated steps whose checkpoints are protected from retention")
    s.add_argument("--no-resume", action="store_true")
    s.set_defaults(fn=_cmd_train)

    s = sub.add_parser("generate", help="text-to-video from a model bundle")
    s.add_argument("--bundle", required=True)
    s.add_argument("--prompt", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--name", default="video")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, help="default 30 (small) / the HD bundle's own default")
    s.add_argument("--cfg", type=float, help="default 3.0 (small) / the HD bundle's own default")
    s.add_argument("--fps", type=int, help="default 8 (small) / the HD bundle's own default")
    s.add_argument("--upscale", type=int, help="small bundles only (default 4); must be 1 or omitted for HD")
    s.add_argument("--tile", type=int, default=0, help="small bundles only (HD tiling comes from the bundle config)")
    s.add_argument("--width", type=int, help="HD bundles only")
    s.add_argument("--height", type=int, help="HD bundles only")
    s.add_argument("--seconds", type=float, help="HD bundles only")
    s.add_argument("--sampler", choices=["euler", "ab2"], help="HD bundles only")
    s.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None, help="HD bundles only (CUDA)")
    s.add_argument("--offload", choices=["none", "blocks"], help="HD bundles only (CUDA)")
    s.set_defaults(fn=_cmd_generate)

    s = sub.add_parser("continue", help="continue a generated video from its saved generation state")
    s.add_argument("--bundle", required=True)
    s.add_argument("--state", required=True, help="a <name>.state directory written by generate/continue")
    s.add_argument("--out", required=True)
    s.add_argument("--name", default="video_continued")
    s.add_argument("--prompt")
    s.add_argument("--seed", type=int)
    s.add_argument("--steps", type=int)
    s.add_argument("--cfg", type=float)
    s.add_argument("--known", type=int, default=2, help="latent frames taken from the previous segment")
    s.add_argument("--fps", type=int, default=8)
    s.add_argument("--upscale", type=int, default=4)
    s.add_argument("--tile", type=int, default=0)
    s.set_defaults(fn=_cmd_continue)

    s = sub.add_parser("evaluate", help="prompt-adherence benchmark / detector validation")
    s.add_argument("--mode", choices=["adherence", "detector"], default="adherence")
    s.add_argument("--bundle")
    s.add_argument("--out", required=True)
    s.add_argument("--prompts")
    s.add_argument("--seeds", type=int, default=1)
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--cfg", type=float, default=3.0)
    s.add_argument("--clips", type=int, default=200)
    s.add_argument("--save-videos")
    s.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("init-random", help="package a randomly initialised (untrained) generator as a bundle")
    s.add_argument("--config", default="nivra_small")
    s.add_argument("--vae", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_init_random)

    s = sub.add_parser("init-hd-bundle", help="write a random-init HD bundle (plumbing tests, not a trained model)")
    s.add_argument("--config", default="nivra_hd_tiny", help="nivra_hd_tiny | nivra_medium | nivra_large | path.json")
    s.add_argument("--out", required=True)
    s.add_argument("--text-encoder", default="stub", help="stub | nivra:<checkpoint dir> | external:<dir>")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default="auto")
    s.set_defaults(fn=_cmd_init_hd_bundle)

    s = sub.add_parser("info", help="parameter counts / shapes of a config")
    s.add_argument("--config", default="nivra_small")
    s.set_defaults(fn=_cmd_info)

    s = sub.add_parser("text-train-tokenizer", help="train the Nivra byte-level BPE tokenizer on text files")
    s.add_argument("--input", nargs="+", required=True, help="text files or directories (.txt/.md)")
    s.add_argument("--out", required=True, help="tokenizer.json to write")
    s.add_argument("--vocab-size", type=int, default=32000)
    s.add_argument("--min-frequency", type=int, default=2)
    s.set_defaults(fn=_cmd_text_train_tokenizer)

    s = sub.add_parser("text-pretrain", help="masked-LM pretraining of the Nivra text encoder")
    s.add_argument("--corpus", nargs="+", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out", required=True, help="checkpoint directory")
    s.add_argument("--size", choices=["tiny", "small", "base"], default="tiny")
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--seq-len", type=int)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default="auto")
    s.add_argument("--fp16", action="store_true", help="CUDA autocast fp16 (untested path)")
    s.add_argument("--save-every-steps", type=int, default=0)
    s.set_defaults(fn=_cmd_text_pretrain)

    s = sub.add_parser("data-ingest", help="folder of videos + captions + PROVENANCE.json -> sharded dataset")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--bucket", choices=["256p2s", "480p4s", "720p8s"], default="256p2s")
    s.add_argument("--fps", type=float, help="default: 8 for 256p2s, 30 otherwise")
    s.add_argument("--max-clips", type=int)
    s.add_argument("--scene-threshold", type=float, default=0.3)
    s.add_argument("--no-scenes", action="store_true", help="skip scene detection; fixed-length windows only")
    s.add_argument("--shard-mb", type=int, default=128)
    s.add_argument("--val-fraction", type=float, default=0.05)
    s.add_argument("--dedup-bits", type=float, default=4.0, help="mean dHash bits (of 64) under which clips are duplicates")
    s.add_argument("--allow-upscale", action="store_true")
    s.add_argument("--default-lang", default="und")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=_cmd_data_ingest)

    s = sub.add_parser("data-inspect", help="summarise and verify an ingested dataset")
    s.add_argument("dataset")
    s.set_defaults(fn=_cmd_data_inspect)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "exec":
        from .executor import exec_main
        return exec_main(argv[1:])
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
