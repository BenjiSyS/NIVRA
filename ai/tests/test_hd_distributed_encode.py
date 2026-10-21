"""ai.hd_encode: raw shard -> latent shard bound to VAE / text encoder / parent shard; then trainable by ai.hd_train."""
import json
import math

import numpy as np
import pytest
import torch

from nivra.data.ingest.pipeline import ShardWriter
from nivra.distributed.hd.encode_job import EncodeInputError, hd_encode_job, load_text_encoder_files
from nivra.distributed.hd.latent_shard import LatentShard
from nivra.distributed.hd.stream_io import sha256_file
from nivra.distributed.hd.train_job import hd_train_job
from nivra.inference.hd_bundle import write_hd_bundle
from nivra.text.bpe import BPETokenizer
from nivra.text.nivra_text import NivraTextConfig, NivraTextModel, save_checkpoint

STAGE = {"h": 32, "w": 64, "frames": 9, "fps": 8}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("enc")
    bundle = root / "bundle"
    m = write_hd_bundle(bundle, "nivra_hd_tiny", text_encoder="stub", seed=2, dtype=torch.float32)
    tok = BPETokenizer.train(["a red ball bounces", "anime girl runs", "a blue cat sleeps"] * 3, 300, 2)
    torch.manual_seed(0)
    model = NivraTextModel(NivraTextConfig("t", tok.vocab_size, 16, 1, 2, 32, 16))      # text_dim 16 = nivra_hd_tiny
    te = root / "te"
    save_checkpoint(te, model, tok)
    rng = np.random.default_rng(0)
    w = ShardWriter(root, "train", (9, 40, 72, 3), 1 << 24)
    for i in range(5):
        clip = rng.integers(0, 255, (9, 40, 72, 3), dtype=np.uint8)
        w.add(clip, {"id": f"c{i}", "caption": f"a red ball {i}", "style": "anime" if i % 2 else "cartoon", "lang": "en"})
    w.flush()
    raw = root / w.files[0]["file"]
    return {"root": root, "bundle": bundle, "manifest": m, "te": te, "raw": raw, "raw_sha": w.files[0]["sha256"]}


def job_inputs(env):
    return {"raw_shard": str(env["raw"]), "vae_weights": str(env["bundle"] / "vae.safetensors"),
            "vae_manifest": str(env["bundle"] / "model_version.json"),
            "text_encoder_weights": str(env["te"] / "model.safetensors"), "tokenizer": str(env["te"] / "tokenizer.json")}


def test_encode_binds_and_summarises(env, tmp_path):
    p = {"dataset_id": "d1", "target_dataset_id": "d2", "shard_idx": 0, "shard_sha256": env["raw_sha"], "stage": STAGE,
         "latent": "mean", "style_dropout": 0.0}
    s = hd_encode_job(job_inputs(env), p, tmp_path / "o")
    assert s["clips"] == 5 and math.isfinite(s["latent_mean"]) and s["latent_std"] > 0
    assert s["sha256"] == sha256_file(s["path"])
    sh = LatentShard(s["path"])
    assert len(sh) == 5 and sh.clip_shape() == (8, 2, 1, 2)
    m = sh.meta
    assert m["parent_shard_sha256"] == env["raw_sha"]
    assert m["vae_version"] == sha256_file(env["bundle"] / "vae.safetensors")
    assert m["text_encoder_id"] == sha256_file(env["te"] / "model.safetensors")
    assert m["stage"] == STAGE and m["target_dataset_id"] == "d2"
    assert m["rows"][1]["text"].startswith("anime") or "anime" in m["rows"][1]["text"]      # style tag applied
    assert sh.null()[0].shape[-1] == 16
    # deterministic: a second encode (audit on another device) is byte-identical on the same machine
    s2 = hd_encode_job(job_inputs(env), p, tmp_path / "o2")
    assert s2["sha256"] == s["sha256"]
    # and the latent shard trains
    b = env["bundle"]
    tp = {"update_id": "u", "base_version_id": env["manifest"]["version_id"], "stage": STAGE, "inner_steps": 3, "batch_size": 2,
          "lr": 1e-3, "warmup_steps": 1, "seed": 1, "hd_config_sha": env["manifest"]["hd_config_sha"], "round": 0, "round_batch": 0,
          "run_id": "r", "target": "dit", "delta_dtype": "float16", "shards": [{"artifact_id": "a", "sha256": s["sha256"]}]}
    r = hd_train_job({"base_train_weights": str(b / "weights.safetensors"), "base_manifest": str(b / "model_version.json"),
                      "data_shard_000": s["path"]}, tp, tmp_path / "t")
    assert math.isfinite(r["loss_last"])


def test_style_dropout_is_deterministic_fraction(env, tmp_path):
    base = {"shard_sha256": env["raw_sha"], "stage": STAGE}
    a = hd_encode_job(job_inputs(env), dict(base, style_dropout=1.0), tmp_path / "a")
    rows = LatentShard(a["path"]).rows
    assert all(r["text"] == r["caption"] for r in rows)                      # everything dropped


def test_encode_rejects_wrong_sha_and_bad_stage(env, tmp_path):
    with pytest.raises(EncodeInputError, match="sha256"):
        hd_encode_job(job_inputs(env), {"shard_sha256": "0" * 64, "stage": STAGE}, tmp_path / "x")
    with pytest.raises(ValueError):
        hd_encode_job(job_inputs(env), {"shard_sha256": env["raw_sha"], "stage": dict(STAGE, frames=10)}, tmp_path / "y")


def test_text_encoder_config_inferred_for_old_checkpoints(env, tmp_path):
    from safetensors.torch import load_file, save_file
    sd = load_file(str(env["te"] / "model.safetensors"))
    old = tmp_path / "old.safetensors"
    save_file(sd, str(old))                                                   # no metadata -> shapes are inferred
    enc = load_text_encoder_files(old, env["te"] / "tokenizer.json")
    assert enc.dim == 16
