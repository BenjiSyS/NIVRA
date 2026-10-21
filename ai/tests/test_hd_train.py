"""Nivra HD trainer at tiny scale on CPU: shard reading, progressive stages, overfit, checkpoint resume,
8-bit optimizer states."""
import json

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from nivra.inference.hd import StubTextEncoder
from nivra.models.hd_config import load_hd_config
from nivra.training.checkpoint import list_checkpoints, load_checkpoint
from nivra.training.hd_trainer import AdamW8bit, ClipShards, fit_video, train_hd


def moving_square(T, H, W, seed):
    rng = np.random.default_rng(seed)
    v = np.full((T, H, W, 3), 30, np.uint8)
    col = rng.integers(80, 255, 3)
    y, x = rng.integers(0, H // 2), rng.integers(0, W // 2)
    for t in range(T):
        xx = (x + 2 * t) % (W - 12)
        v[t, y:y + 12, xx:xx + 12] = col
    return v


def write_dataset(root, n=2, T=17, H=64, W=64, key="videos", latents=None):
    root.mkdir(parents=True, exist_ok=True)
    arr = latents if latents is not None else np.stack([moving_square(T, H, W, i) for i in range(n)])
    save_file({key: arr}, str(root / "shard-train-00000.safetensors"))
    caps = ["a red square moves right", "a green square slides"]
    rows = [{"id": f"c{i}", "split": "train", "shard": "shard-train-00000.safetensors", "index": i,
             "caption": caps[i % 2], "lang": "en", "source": "synthetic-test", "license": "CC0"}
            for i in range(arr.shape[0])]
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "dataset.json").write_text(json.dumps({"shards": []}), encoding="utf-8")
    return root


def cfg_with(stages, **train):
    cfg = load_hd_config("nivra_hd_tiny")
    cfg["train"]["stages"] = stages
    cfg["train"].update(train)
    return cfg


def test_shards_and_fit(tmp_path):
    ds = ClipShards(write_dataset(tmp_path / "d"))
    kind, x, row = ds.get(1)
    assert kind == "videos" and x.shape == (3, 17, 64, 64) and -1 <= x.min() and x.max() <= 1
    assert row["license"] == "CC0"
    y = fit_video(x, 9, 32, 64)
    assert y.shape == (3, 9, 32, 64)
    assert fit_video(x[:, :5], 9, 64, 64).shape == (3, 9, 64, 64)


def test_vae_overfit_and_export(tmp_path):
    data = write_dataset(tmp_path / "d", n=1)
    cfg = cfg_with([{"name": "s", "height": 64, "width": 64, "frames": 9, "steps": 40}], batch_size=1,
                   vae_lr=3e-3, save_every=100)
    res = train_hd(cfg, data, tmp_path / "run", target="vae", log=lambda *_: None)
    losses = [h["loss"] for h in res["history"]]
    assert res["status"] == "done" and len(losses) == 40
    assert np.mean(losses[-5:]) < 0.6 * np.mean(losses[:5])
    assert (tmp_path / "run" / "hc_vae" / "hc_vae.json").is_file()
    assert res["latent_norm"]["latent_scale"] > 0


def test_dit_overfit_progressive_on_videos(tmp_path):
    data = write_dataset(tmp_path / "d", n=2)
    enc = StubTextEncoder(dim=16)
    cfg = cfg_with([{"name": "s1", "height": 32, "width": 64, "frames": 9, "steps": 5},
                    {"name": "s2", "height": 64, "width": 64, "frames": 17, "steps": 55}],
                   batch_size=2, lr=3e-3, caption_dropout=0.0, save_every=1000)
    from nivra.models.hd_config import build_hcvae
    torch.manual_seed(0)
    vae = build_hcvae(cfg)
    res = train_hd(cfg, data, tmp_path / "run", target="dit", text_encoder=enc, vae=vae, log=lambda *_: None)
    hist = res["history"]
    assert [h["stage"] for h in hist[:6]] == ["s1"] * 5 + ["s2"]
    late = [h["loss"] for h in hist if h["stage"] == "s2"]
    assert np.mean(late[-10:]) < 0.7 * np.mean(late[:10])


def test_dit_on_latent_shards_and_exact_resume(tmp_path):
    rng = np.random.default_rng(0)
    lat = rng.standard_normal((3, 8, 3, 2, 2)).astype(np.float16)
    data = write_dataset(tmp_path / "d", key="latents", latents=lat)
    enc = StubTextEncoder(dim=16)
    stages = [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 6}]
    for opt in ("adamw", "adamw8"):
        cfg = cfg_with(stages, batch_size=2, save_every=2, optimizer=opt, keep_last=5)
        full = train_hd(cfg, data, tmp_path / f"full_{opt}", text_encoder=enc, log=lambda *_: None)
        part = train_hd(cfg, data, tmp_path / f"part_{opt}", text_encoder=enc, total_steps=4, log=lambda *_: None)
        assert part["step"] == 4
        resumed = train_hd(cfg, data, tmp_path / f"part_{opt}", text_encoder=enc, log=lambda *_: None)
        assert resumed["step"] == 6
        a = load_checkpoint(list_checkpoints(tmp_path / f"full_{opt}")[-1][1])[1]["model"]
        b = load_checkpoint(list_checkpoints(tmp_path / f"part_{opt}")[-1][1])[1]["model"]
        for k in a:
            assert torch.allclose(a[k], b[k], atol=1e-6), (opt, k)
    # resuming with a different text encoder is refused
    from nivra.training.checkpoint import IncompatibleCheckpoint
    with pytest.raises(IncompatibleCheckpoint):
        train_hd(cfg, data, tmp_path / "part_adamw8", text_encoder=StubTextEncoder(dim=16, max_len=3),
                 total_steps=8, log=lambda *_: None)


def test_stop_file_saves_checkpoint(tmp_path):
    rng = np.random.default_rng(1)
    data = write_dataset(tmp_path / "d", key="latents", latents=rng.standard_normal((2, 8, 3, 2, 2)).astype(np.float16))
    stop = tmp_path / "STOP"
    stop.write_text("x")
    cfg = cfg_with([{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 5}])
    res = train_hd(cfg, data, tmp_path / "run", text_encoder=StubTextEncoder(16), stop_file=stop, log=lambda *_: None)
    assert res["status"] == "stopped" and res["ckpt"]


def test_adamw8bit_tracks_adamw():
    torch.manual_seed(0)
    w0 = torch.randn(64, 128)
    a, b = w0.clone().requires_grad_(), w0.clone().requires_grad_()
    oa = torch.optim.AdamW([a], lr=1e-2, betas=(0.9, 0.99), weight_decay=0.0)
    ob = AdamW8bit([b], lr=1e-2, betas=(0.9, 0.99), min_8bit_size=1)
    target = torch.randn(64, 128)
    for _ in range(50):
        for p, o in ((a, oa), (b, ob)):
            o.zero_grad()
            ((p - target) ** 2).mean().backward()
            o.step()
    st = ob.state[b]
    assert st["m_q"].dtype == torch.int8 and st["v_q"].dtype == torch.uint8
    da, db = (a - w0).detach(), (b - w0).detach()
    assert torch.nn.functional.cosine_similarity(da.flatten(), db.flatten(), dim=0) > 0.98
    sd = ob.state_dict()
    ob2 = AdamW8bit([b], lr=1e-2, min_8bit_size=1)
    ob2.load_state_dict(sd)
    assert ob2.state[b]["m_q"].dtype == torch.int8
