"""The distributed worker round (ai.hd_train) follows the trainer's recipe: SRA with a frozen received-weights teacher,
diffusion forcing, frame/fps conditioning, fp32 attention, dynamic loss scaling with skip-on-overflow. The update and
echo format are unchanged, so the same validator accepts the result."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from nivra.distributed import version as ver
from nivra.distributed.hd import quant
from nivra.distributed.hd.latent_shard import write_latent_shard
from nivra.distributed.hd.train_job import hd_train_job
from nivra.distributed.hd.validate import validate_hd_update
from nivra.inference.hd_bundle import write_hd_bundle
from nivra.models.hd_config import load_hd_config

from test_hd_distributed import H, LAT, LR, STAGE, params


def v2_config(**train):
    cfg = copy.deepcopy(load_hd_config("nivra_hd_tiny"))
    cfg["name"] = "tiny_recipe"
    cfg["model"].update(frame_cond=True, fps_cond=True)
    cfg["train"].update(diffusion_forcing={"fraction": 0.5, "cond_fraction": 0.5}, fps_dropout=0.25, **train)
    return cfg


def shard(path, seed, fps=True, n=6):
    g = torch.Generator().manual_seed(seed)
    rows = [{"id": f"c{seed}-{i}", "caption": "x", "text": "x", "style": "anime", "lang": "en", "split": "train",
             **({"effective_fps": 8.0 if i % 2 else 12.0} if fps else {})} for i in range(n)]
    meta = {"parent_shard_sha256": "0" * 64, "vae_version": "v" * 64, "text_encoder_id": "t" * 64, "stage": STAGE, "rows": rows}
    return write_latent_shard(path, torch.randn(n, *LAT, generator=g), torch.randn(n, 5, 16, generator=g),
                              torch.ones(n, 5, dtype=torch.uint8), torch.zeros(1, 2, 16), torch.ones(1, 2, dtype=torch.uint8), meta)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("recipe")
    out = {"root": root}
    for key, cfg in (("v2", v2_config()),
                     ("scale", v2_config(force_loss_scaling=True, loss_scale={"init": 3.0e38, "backoff": 0.0625}))):
        b = root / key
        out[key] = {"bundle": b, "manifest": write_hd_bundle(b, cfg, text_encoder="stub", seed=2, dtype=torch.float32)}
    out["shard"] = (root / "s.safetensors", None)
    out["shard"] = (root / "s.safetensors", shard(root / "s.safetensors", 1))
    return out


def run(env, key, uid, out, **kw):
    e = env[key]
    p, sha = env["shard"]
    ins = {"base_train_weights": str(e["bundle"] / "weights.safetensors"), "base_manifest": str(e["bundle"] / "model_version.json"),
           "data_shard_000": str(p)}
    pr = params({"manifest": e["manifest"], "shards": [(p, sha)]}, uid, 0, 5, **kw)
    s = hd_train_job(ins, pr, out, attempt_id="a")
    return s, {"path": str(out / "update.safetensors"), "manifest_path": str(out / "update.json")}


def validate(env, key, u, uid):
    bm = ver.read_manifest(env[key]["bundle"] / "model_version.json")
    return validate_hd_update(u["path"], u["manifest_path"], bm, expected={
        "base_version_id": bm["version_id"], "inner_steps": H, "lr": LR, "update_id": uid, "stage": STAGE,
        "batch_size": 2, "grad_accum": 1, "delta_dtype": "float16"})


def test_full_recipe_round_is_a_valid_update(env, tmp_path):
    s, u = run(env, "v2", "r-0", tmp_path / "u")
    v = validate(env, "v2", u, "r-0")
    assert v.valid, v.reasons
    m = json.loads(Path(u["manifest_path"]).read_text())
    assert m["recipe"]["sra"] and m["recipe"]["sra_teacher"] == "received_base_frozen"
    assert m["recipe"]["diffusion_forcing"] == 0.5 and m["recipe"]["frame_cond"] == 0.5 and m["recipe"]["fps_cond"] is True
    assert m["samples_seen"] == H * 2 and m["delta_dtype"] == "float16"
    assert s["sra_last"] is not None and 0.0 <= s["sra_last"] <= 2.0 and math.isfinite(s["loss_last"])
    assert s["skipped_steps"] == 0


def test_recipe_round_is_deterministic_and_uses_features(env, tmp_path):
    a, ua = run(env, "v2", "r-0", tmp_path / "a")
    b, ub = run(env, "v2", "r-0", tmp_path / "b")
    assert a["update_sha256"] == b["update_sha256"]
    # switching the recipe off changes the result (the features are really used)
    cfg0 = v2_config()
    cfg0["train"].update(sra={"weight": 0.0}, diffusion_forcing={"fraction": 0.0, "cond_fraction": 0.0})
    b0 = tmp_path / "plain"
    man0 = write_hd_bundle(b0, cfg0, text_encoder="stub", seed=2, dtype=torch.float32)
    env2 = dict(env, plain={"bundle": b0, "manifest": man0})
    c, _ = run(env2, "plain", "r-0", tmp_path / "c")
    assert c["update_sha256"] != a["update_sha256"] and c["sra_last"] is None


def test_fps_conditioning_reads_row_fps(env, tmp_path):
    p = tmp_path / "nofps.safetensors"
    sha = shard(p, 1, fps=False)
    e = dict(env, shard=(p, sha))
    a, _ = run(e, "v2", "r-0", tmp_path / "a")
    b, _ = run(env, "v2", "r-0", tmp_path / "b")
    assert a["update_sha256"] != b["update_sha256"]               # rows with effective_fps vs without


def test_overflow_is_skipped_not_fatal(env, tmp_path):
    s, u = run(env, "scale", "r-1", tmp_path / "u")
    assert 0 < s["skipped_steps"] < H                             # the huge loss scale overflowed and backed off
    v = validate(env, "scale", u, "r-1")
    assert v.valid, v.reasons
    m = json.loads(Path(u["manifest_path"]).read_text())
    assert m["recipe"]["loss_scaling"] is True and m["recipe"]["skipped_steps"] == s["skipped_steps"]


def test_int8_with_recipe(env, tmp_path):
    s, u = run(env, "v2", "r-2", tmp_path / "u", delta_dtype="int8")
    bm = ver.read_manifest(env["v2"]["bundle"] / "model_version.json")
    v = validate_hd_update(u["path"], u["manifest_path"], bm, expected={
        "base_version_id": bm["version_id"], "inner_steps": H, "lr": LR, "update_id": "r-2", "stage": STAGE,
        "batch_size": 2, "grad_accum": 1, "delta_dtype": "int8"})
    assert v.valid, v.reasons
    assert Path(u["path"]).stat().st_size == quant.expected_int8_bytes(bm["tensor_shapes"])


def test_encode_job_loads_v2_per_channel_vae(tmp_path):
    from nivra.compression.hc_vae import HCVAE_VERSION_2, save_hcvae
    from nivra.distributed.hd.encode_job import encode_videos, load_vae_files
    from nivra.models.hd_config import build_hcvae
    cfg = load_hd_config("nivra_hd_tiny")
    vae = build_hcvae(cfg)
    vae.calibrate(torch.randn(4, vae.cfg.latent_channels, 2, 1, 2) * torch.linspace(0.5, 2, vae.cfg.latent_channels).view(1, -1, 1, 1, 1),
                  per_channel=True)
    save_hcvae(vae, tmp_path / "v")
    man = json.loads((tmp_path / "v" / "hc_vae.json").read_text())
    assert man["version"] == HCVAE_VERSION_2
    got = load_vae_files(tmp_path / "v" / "hc_vae.safetensors", tmp_path / "v" / "hc_vae.json")
    assert got.cfg.per_channel and got.cfg.latent_scale == vae.cfg.latent_scale
    vids = torch.randint(0, 255, (1, 9, 32, 64, 3), dtype=torch.uint8)
    a = encode_videos(got, vids, {"h": 32, "w": 64, "frames": 9})
    z = vae.eval().encode_tiled(vids[0].permute(3, 0, 1, 2).float().div(127.5).sub(1.0)[None], t_chunk=2)
    assert torch.allclose(a, vae.to_scaled(z.float()), atol=1e-5)


def test_cli_ingest_hold_flags_parse_and_reach_options():
    from nivra import cli
    p = cli.build_parser()
    a = p.parse_args(["data-ingest", "--input", "i", "--output", "o", "--no-detect-holds"])
    assert a.no_detect_holds and not a.drawing_rate_sampling
    a = p.parse_args(["data-ingest", "--input", "i", "--output", "o", "--drawing-rate-sampling"])
    assert a.drawing_rate_sampling and not a.no_detect_holds
    a = p.parse_args(["train-pipeline", "--out", "r", "--videos", "v", "--drawing-rate-sampling"])
    assert a.drawing_rate_sampling
