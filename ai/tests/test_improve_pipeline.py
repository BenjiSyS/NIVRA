"""End-to-end `train-pipeline` runs with the 2026 improvements switched on (shipped nivra_hd_tiny, and a v2 tiny with
4x temporal factor + frame/fps conditioning), plus the staged-schedule helpers."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from nivra.cli import main as cli_main
from nivra.compression.hc_vae import HCVAE_VERSION_2
from nivra.distributed.hd.latent_shard import LatentShard
from nivra.inference.hd_bundle import load_hd_bundle
from nivra.models.hd_config import hd_config_hash, load_hd_config
from nivra.training.checkpoint import list_checkpoints, load_checkpoint
from nivra.training.hd_pipeline import Pipeline, PipelineOptions, fit_stages, scale_stages

from test_train_pipeline import make_dataset, opts


def with_fps(root: Path, fps_by_clip=(12.0, 8.0, 24.0, None)) -> Path:
    """Add per-clip `effective_fps` / `hold` (as the ingest hold detector writes them) to a dataset's manifest."""
    mf = root / "manifest.jsonl"
    rows = [json.loads(l) for l in mf.read_text().splitlines()]
    for i, r in enumerate(rows):
        f = fps_by_clip[i % len(fps_by_clip)]
        r["effective_fps"], r["hold"] = f, (None if f is None else int(round(24 / f)))
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    mf.write_bytes(text.encode())
    ds = json.loads((root / "dataset.json").read_text())
    ds["fingerprint"] = hashlib.sha256(text.encode()).hexdigest()
    (root / "dataset.json").write_text(json.dumps(ds))
    return root


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return with_fps(make_dataset(tmp_path_factory.mktemp("impds")))


def grp(run_dir):
    return load_checkpoint(list_checkpoints(run_dir)[-1][1])


# ===================================================================== shipped nivra_hd_tiny, 8x temporal
@pytest.fixture(scope="module")
def tiny(ds, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny")
    logs = []
    pipe = Pipeline(opts(tmp, ds, install_dir=str(tmp / "install"), name="imp-tiny", save_every=1), log=logs.append)
    return {"tmp": tmp, "res": pipe.run(), "logs": logs, "pipe": pipe}


def test_tiny_vae_stage_follows_the_staged_recipe_with_gan_and_channel_norm(tiny):
    run = tiny["tmp"] / "run" / "vae_run"
    man, groups, st = grp(run)
    names = [h["stage"] for h in st["history"]]
    assert names == ["ae_img"] * 1 + ["ae_clips_lowres"] * 2 + ["ae_target"] * 1       # 2/3/3 scaled to 4 steps
    # GAN start 4 (of 8) scaled to the 4-step run -> 2: first adversarial term at the third step
    assert ["gan_g" in h for h in st["history"]] == [False, False, True, True]
    assert "disc" in groups and "opt_disc" in groups
    vae = tiny["res"]["vae"]
    assert vae["latent_norm"]["latent_norm"] == "channel" and len(vae["latent_norm"]["latent_scale"]) == 8
    hc = json.loads((Path(vae["dir"]) / "hc_vae.json").read_text())
    assert hc["version"] == HCVAE_VERSION_2 and hc["latent_norm"] == "channel"


def test_tiny_latents_include_image_stills_and_carry_fps(tiny):
    lat = Path(tiny["res"]["latents"]["dir"])
    info = json.loads((lat / "dataset.json").read_text())
    assert info["still_shards"] and info["shards"] and not set(f["file"] for f in info["still_shards"]) & {f["file"] for f in info["shards"]}
    rows = [json.loads(l) for l in (lat / "manifest.jsonl").read_text().splitlines()]
    stills = [r for r in rows if r.get("still")]
    vids = [r for r in rows if not r.get("still")]
    assert len(stills) == len(vids) == 8 and all(r["lat_t"] == 1 for r in stills) and all(r["lat_t"] == 3 for r in vids)
    assert {r["effective_fps"] for r in vids} == {12.0, 8.0, 24.0, None} and all(r["id"].endswith(":still") for r in stills)
    sh = LatentShard(lat / info["still_shards"][0]["file"])
    assert sh.shape[2] == 1 and sh.meta["stage"]["frames"] == 1 and sh.meta["kind"] == "still"
    mainsh = LatentShard(lat / info["shards"][0]["file"])
    assert mainsh.shape[2] == 3
    z, _, _ = sh.get(0)
    assert z.shape[1] == 1 and torch.isfinite(z).all()


def test_tiny_bundle_exports_ema_weights_and_calibrated_vae(tiny):
    res = tiny["res"]
    assert res["dit"]["export_kind"] == "ema" and res["dit"]["skipped_steps"] == 0
    man, groups, _ = grp(tiny["tmp"] / "run" / "dit_run")
    exported = load_file(res["dit"]["weights"])
    assert set(exported) == set(groups["model"])
    for k in exported:
        torch.testing.assert_close(exported[k], groups["ema"][k])
    assert any(not torch.equal(groups["ema"][k], groups["model"][k]) for k in exported)           # EMA != raw weights
    assert {"sra", "opt_sra"} <= set(groups)                                                       # SRA ran, projector is checkpoint-only
    assert not any(k.startswith(("sra", "proj")) for k in exported)
    # the bundle carries the exact EMA tensors and the per-channel calibration
    b = Path(res["bundle"]["dir"])
    bw = load_file(str(b / "weights.safetensors"))
    for k in exported:
        torch.testing.assert_close(bw[k], exported[k])
    m = json.loads((b / "model_version.json").read_text())
    vcfg = m["config"]["vae"]
    assert vcfg["latent_norm"] == "channel" and len(vcfg["latent_shift"]) == 8 and vcfg["version"] == HCVAE_VERSION_2
    assert m["hd_config_sha"] == hd_config_hash(load_hd_config("nivra_hd_tiny"))                   # calibration is not architecture
    pipe = load_hd_bundle(b, "cpu")
    assert pipe.vae.cfg.per_channel
    r = pipe.generate("a red ball bounces", seed=1, steps=2)
    assert r.frames.shape[0] == 17 and np.isfinite(r.frames).all()
    r2 = pipe.continue_video(r.state, seed=2, steps=2)
    assert r2.meta["conditioning"] == "replace"                    # the shipped tiny has no frame_cond


def test_tiny_dit_stages_include_image_first_and_joint(tiny):
    man, groups, st = grp(tiny["tmp"] / "run" / "dit_run")
    names = [h["stage"] for h in st["history"]]
    assert names[0] == "s0_img" and set(names) == {"s0_img", "s1_32p", "s2_64p"}
    assert all(np.isfinite(h["loss"]) for h in st["history"]) and any("sra" in h for h in st["history"])


# ===================================================================== v2 tiny: 4x temporal, frame_cond, fps_cond
@pytest.fixture(scope="module")
def v2(ds, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v2")
    cfg = load_hd_config("nivra_hd_tiny")
    cfg["name"] = "tiny_v2"
    del cfg["vae"]["temporal_down"]
    cfg["vae"]["temporal_factor"] = 4
    cfg["model"].update(frame_cond=True, fps_cond=True)
    cfg["train"]["diffusion_forcing"] = {"fraction": 0.5, "cond_fraction": 0.5}
    cfg["train"]["fps_dropout"] = 0.25
    path = tmp / "tiny_v2.json"
    path.write_text(json.dumps(cfg))
    logs = []
    pipe = Pipeline(opts(tmp, ds, config=str(path), dit_steps=8, install_dir=str(tmp / "install"), name="imp-v2"), log=logs.append)
    return {"tmp": tmp, "res": pipe.run(), "logs": logs, "pipe": pipe}


def test_v2_pipeline_trains_with_4x_temporal_cond_and_fps(v2):
    res = v2["res"]
    lat = res["latents"]
    assert lat["stage"]["frames"] == 17
    rows = [json.loads(l) for l in (Path(lat["dir"]) / "manifest.jsonl").read_text().splitlines()]
    assert {r["lat_t"] for r in rows if not r.get("still")} == {5}                  # 1 + 16/4
    man, groups, st = grp(v2["tmp"] / "run" / "dit_run")
    assert all(np.isfinite(h["loss"]) for h in st["history"])
    assert any(h.get("cond") == 1.0 for h in st["history"]) and any(h.get("df") == 1.0 for h in st["history"])
    w = load_file(res["dit"]["weights"])
    assert "cond_embed" in w and "fps_null" in w and "fps_embed.0.weight" in w


def test_v2_bundle_loads_generates_and_conditions_with_masks(v2):
    b = Path(v2["res"]["bundle"]["dir"])
    m = json.loads((b / "model_version.json").read_text())
    assert m["config"]["model"]["frame_cond"] and m["config"]["vae"]["temporal_down"] == [2, 3]
    pipe = load_hd_bundle(b, "cpu")
    assert pipe.dit.cfg.version == "nivra-dit/2" and pipe.vae.cfg.temporal_down == (2, 3)
    assert pipe.vae.cfg.temporal_factor == 4 and pipe.dit.cfg.fps_cond
    r = pipe.generate("a red ball bounces", seed=1, steps=2, frames=9, effective_fps=12)
    assert r.frames.shape[0] == 9 and r.meta["latent_shape"][1] == 3
    r2 = pipe.continue_video(r.state, seed=2, steps=2, frames=9)
    assert r2.meta["conditioning"] == "mask"
    # long-video refine of a segment (anchored first / last latent frames, mask conditioning)
    out = []
    meta = pipe.refine("a red ball bounces", r.frames, 96, 64, seed=3, steps=2, strength=0.5, sink=out.append)
    assert meta["conditioning"] == "mask" and sum(c.shape[0] for c in out) == 9


def test_v2_longvideo_draft_reports_mask_conditioning(v2, tmp_path):
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    from nivra.inference import longvideo as lv
    pipe = load_hd_bundle(Path(v2["res"]["bundle"]["dir"]), "cpu")
    out = lv.run_draft(pipe, prompt="a red ball bounces", seed=1, seconds=1.0, fps=8, segment_frames=5, width=96, height=64,
                       out_mp4=tmp_path / "draft.mp4", plan_path=tmp_path / "plan.json", steps=2)
    assert out["segments"] >= 2 and out["conditioning"] == "mask"


# ===================================================================== staged helpers
def test_scale_and_fit_keep_image_and_bucket_keys():
    st = [{"name": "i", "height": 32, "width": 64, "frames": 1, "steps": 10},
          {"name": "j", "height": 64, "width": 64, "frames": 17, "steps": 30, "image_fraction": 0.2}]
    out = scale_stages(fit_stages(st, 64, 96, 17), 8)
    assert sum(s["steps"] for s in out) == 8 and out[1]["image_fraction"] == 0.2 and out[0]["frames"] == 1
    assert fit_stages(st, 16, 16, 5, tf=4)[0]["frames"] == 5                      # fallback stage uses 1 + tf*k
    assert fit_stages(st, 16, 16, 5)[0]["frames"] == 1
    tc = {"vae_gan_start": 20000, "sra": {"weight": 0.5, "until_step": 60000}}
    Pipeline.rescale_step_keys(tc, 0.001)
    assert tc["vae_gan_start"] == 20 and tc["sra"]["until_step"] == 60 and tc["sra"]["weight"] == 0.5
    plain = {"lr": 1}
    Pipeline.rescale_step_keys(plain, 0.5)
    assert plain == {"lr": 1}


def test_cli_plan_mentions_nothing_broken_for_shipped_configs(capsys, ds, tmp_path):
    for name in ("nivra_hd_tiny", "nivra_base"):
        code = cli_main(["train-pipeline", "--config", name, "--dataset", str(ds), "--out", str(tmp_path / name),
                         "--plan-only"])
        assert code == 0
    out = capsys.readouterr().out
    assert "nivra_base" in out and "DiT" in out
