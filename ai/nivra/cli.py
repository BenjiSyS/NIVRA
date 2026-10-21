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


def _cmd_generate(a):
    from .config import setup_runtime
    from .inference.bundle import load_bundle
    from .inference.generate import generate, save_result
    dev = setup_runtime()
    b = load_bundle(a.bundle, dev)
    res = generate(b, a.prompt, seed=a.seed, steps=a.steps, cfg_scale=a.cfg, tile=a.tile, device=dev)
    meta = save_result(res, a.out, a.name, fps=a.fps, upscale=a.upscale)
    print(json.dumps({k: meta[k] for k in ("mp4", "frames", "fps", "seconds_sampling", "seconds_decode", "peak_rss_mb")}))
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
    s.add_argument("--steps", type=int, default=30)
    s.add_argument("--cfg", type=float, default=3.0)
    s.add_argument("--fps", type=int, default=8)
    s.add_argument("--upscale", type=int, default=4)
    s.add_argument("--tile", type=int, default=0)
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

    s = sub.add_parser("info", help="parameter counts / shapes of a config")
    s.add_argument("--config", default="nivra_small")
    s.set_defaults(fn=_cmd_info)
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
