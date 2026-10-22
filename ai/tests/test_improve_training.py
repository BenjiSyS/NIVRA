"""2026 improvement pass, training side: VAE staged recipe + patch-GAN + per-channel calibration, DiT image-first /
bucket stages, diffusion forcing, frame conditioning, SRA, EMA, fp16 loss scaling with logged overflow skips, fps
conditioning, and exact backward compatibility of old configs (loss histories captured with the pre-change code)."""
import copy
import json
import math

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from nivra.compression.hc_vae import HCVAE
from nivra.inference.hd import StubTextEncoder
from nivra.models.dit import NivraDiT
from nivra.models.hd_config import build_dit, build_hcvae, load_hd_config
from nivra.training.checkpoint import list_checkpoints, load_checkpoint
from nivra.training.hd_trainer import (ClipShards, SRAProjector, clip_fps_value, ema_decay_at, ema_update, fps_dropped,
                                       pick_bucket, sra_loss, train_hd)

from test_hd_train import write_dataset
from test_improve_models import OLD_TINY, tiny_cfg

QUIET = dict(log=lambda *_: None)

# Loss histories of the PRE-CHANGE trainer (git HEAD ff517e7) for a config that uses only old keys, same data / seeds.
LEGACY = {
    "vae": [1.754073, 1.494156, 0.801989, 0.5503, 0.516153, 0.395106],
    "vae_norm": [0.112888, 2.917491],
    "dit_videos": [0.69921, 0.709664, 1.025439, 1.064186, 0.899412, 1.009678],
    "dit_latents": [1.979429, 2.040475, 1.875235, 1.830717, 2.022207, 2.097437],
}


def legacy_cfg():
    cfg = copy.deepcopy(OLD_TINY)
    cfg["train"] = {"stages": [{"name": "s1", "height": 32, "width": 64, "frames": 9, "steps": 3},
                               {"name": "s2", "height": 64, "width": 64, "frames": 17, "steps": 3}],
                    "batch_size": 2, "grad_accum": 1, "lr": 3e-3, "warmup": 2, "grad_clip": 1.0, "optimizer": "adamw",
                    "grad_checkpoint": True, "fp16": True, "caption_dropout": 0.1, "style_dropout": 0.1,
                    "shift_base_tokens": 16, "vae_lr": 2e-3, "save_every": 100}
    return cfg


@pytest.fixture(scope="module")
def vdata(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("vd") / "d", n=3, T=17, H=64, W=64)


@pytest.fixture(scope="module")
def ldata(tmp_path_factory):
    rng = np.random.default_rng(0)
    return write_dataset(tmp_path_factory.mktemp("ld") / "l", key="latents",
                         latents=rng.standard_normal((3, 8, 3, 2, 2)).astype(np.float16))


def hist(res, key="loss"):
    return [h[key] for h in res["history"]]


# ===================================================================== backward compatibility
def test_old_config_trains_exactly_as_before(vdata, ldata, tmp_path):
    cfg = legacy_cfg()
    r = train_hd(cfg, vdata, tmp_path / "v", target="vae", **QUIET)
    assert hist(r) == pytest.approx(LEGACY["vae"], abs=1e-4)
    assert [r["latent_norm"]["latent_shift"], r["latent_norm"]["latent_scale"]] == pytest.approx(LEGACY["vae_norm"], abs=1e-4)
    assert r["latent_norm"]["latent_norm"] == "scalar"
    assert not any(k in r["history"][0] for k in ("gan_g", "gan_d", "sra", "df"))
    torch.manual_seed(0)
    vae = build_hcvae(cfg)
    r = train_hd(cfg, vdata, tmp_path / "dv", target="dit", text_encoder=StubTextEncoder(16), vae=vae, **QUIET)
    assert hist(r) == pytest.approx(LEGACY["dit_videos"], abs=1e-4)
    assert "export_state" not in r and r["export_kind"] == "raw" and r["skipped_steps"] == 0
    cfg["train"]["stages"] = [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 6}]
    r = train_hd(cfg, ldata, tmp_path / "dl", target="dit", text_encoder=StubTextEncoder(16), **QUIET)
    assert hist(r) == pytest.approx(LEGACY["dit_latents"], abs=1e-4)


# ===================================================================== small helpers
def test_ema_schedule_and_update():
    assert ema_decay_at(0.0, 5) == 0.0
    assert ema_decay_at(0.9999, 0) == pytest.approx(0.1)
    assert ema_decay_at(0.9999, 10) == pytest.approx(11 / 20)
    assert ema_decay_at(0.9, 10 ** 6) == 0.9                                  # capped
    assert ema_decay_at(0.9, 0, warmup=False) == 0.9
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    old = [p.clone() for p in a.parameters()]
    ema_update(a, b, 0.75)
    for p, o, q in zip(a.parameters(), old, b.parameters()):
        torch.testing.assert_close(p, 0.75 * o + 0.25 * q)


def test_pick_bucket_draws_only_when_asked():
    g = torch.Generator().manual_seed(3)
    st0 = g.get_state().clone()
    plain = {"height": 32, "width": 64, "frames": 9}
    assert pick_bucket(plain, g) == (32, 64, 9, False)
    assert torch.equal(g.get_state(), st0)                                   # no draw: the old random stream is intact
    assert pick_bucket({**plain, "frames": 1}, g) == (32, 64, 1, True) and torch.equal(g.get_state(), st0)
    assert pick_bucket({**plain, "image_fraction": 1.0}, g) == (32, 64, 1, True)
    assert pick_bucket({**plain, "image_fraction": 1.0}, g, force_video=True)[3] is False
    frac = np.mean([pick_bucket({**plain, "image_fraction": 0.3}, g)[3] for _ in range(1500)])
    assert 0.25 < frac < 0.35
    bk = {"height": 32, "width": 64, "frames": 9, "buckets": [{"height": 32, "width": 64, "frames": 9, "weight": 1.0},
                                                               {"height": 64, "width": 64, "frames": 5, "weight": 3.0}]}
    picks = [pick_bucket(bk, g)[:3] for _ in range(1200)]
    assert set(picks) == {(32, 64, 9), (64, 64, 5)} and picks.count((64, 64, 5)) > 800


def test_fps_value_and_dropout_are_deterministic():
    assert clip_fps_value({"effective_fps": 12.0}) == 12.0
    for bad in ({}, {"effective_fps": None}, {"effective_fps": 0}, {"effective_fps": float("nan")}, {"effective_fps": "x"}):
        assert math.isnan(clip_fps_value(bad))
    assert fps_dropped(0, 0, "a", 0.0) is False and fps_dropped(0, 0, "a", 1.0) is True
    assert fps_dropped(1, 2, "clip", 0.5) == fps_dropped(1, 2, "clip", 0.5)
    rate = np.mean([fps_dropped(0, 0, f"c{i}", 0.3) for i in range(2000)])
    assert 0.25 < rate < 0.35


def test_sra_loss_semantics():
    ident = torch.nn.Identity()
    h = torch.randn(2, 5, 8)
    assert sra_loss(h, h * 3.0, ident).item() == pytest.approx(0.0, abs=1e-6)             # cosine: scale-free
    assert sra_loss(h, -h, ident).item() == pytest.approx(2.0, abs=1e-6)
    a, b = torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)
    a[..., 0], b[..., 1] = 1, 1
    assert sra_loss(a, b, ident).item() == pytest.approx(1.0)
    w = torch.tensor([[1.0, 0.0]])
    s = torch.stack([a[0, 0], b[0, 0]])[None]
    assert sra_loss(s, s[:, [0, 0]], ident, w).item() == pytest.approx(0.0, abs=1e-6)     # weighted: only token 0 counts
    proj = SRAProjector(8, hidden=0)
    student = h.clone().requires_grad_(True)
    teacher = torch.randn(2, 5, 8, requires_grad=True)
    sra_loss(student, teacher, proj).backward()
    assert student.grad is not None and teacher.grad is None                              # teacher is stop-grad
    assert all(p.grad is not None for p in proj.parameters())


# ===================================================================== VAE training
def vae_cfg(**train):
    cfg = load_hd_config("nivra_hd_tiny")
    cfg["train"].update({"batch_size": 2, "save_every": 100, "vae_gan_start": 3, "vae_gan_weight": 0.05, **train})
    return cfg


def test_vae_staged_recipe_images_clips_target_with_gan_after_start(vdata, tmp_path):
    cfg = vae_cfg()
    seen = []

    class Spy(HCVAE):
        def forward(self, x, generator=None):
            seen.append(tuple(x.shape))
            return super().forward(x, generator)

    net = Spy(build_hcvae(cfg).cfg)
    res = train_hd(cfg, vdata, tmp_path / "v", target="vae", model=net, **QUIET)
    assert res["status"] == "done" and res["step"] == 8
    names = [h["stage"] for h in res["history"]]
    assert names == ["ae_img"] * 2 + ["ae_clips_lowres"] * 3 + ["ae_target"] * 3
    assert all(s[2] == 1 for s, n in zip(seen, names) if n == "ae_img")                 # stage 1: images (T=1)
    assert all(s[2] in (1, 9) for s, n in zip(seen, names) if n == "ae_clips_lowres")  # low-res clips, images mixed in
    assert all(s[2] in (1, 17) for s, n in zip(seen, names) if n == "ae_target")
    assert any(s[2] == 1 for s, n in zip(seen, names) if n != "ae_img")                 # images mixed into video stages
    assert any(s[2] > 1 for s, n in zip(seen, names) if n != "ae_img")
    gan = [("gan_g" in h) for h in res["history"]]
    assert gan == [False, False, False, True, True, True, True, True]                  # starts at vae_gan_start = 3
    assert all(math.isfinite(h["gan_w"]) and h["gan_w"] >= 0 for h in res["history"] if "gan_w" in h)
    assert all(math.isfinite(h["loss"]) for h in res["history"])
    assert res["latent_norm"]["latent_norm"] == "channel" and len(res["latent_norm"]["latent_shift"]) == 8


def test_vae_discriminator_untouched_before_start_and_updated_after(vdata, tmp_path):
    cfg = vae_cfg(vae_gan_start=2)
    train_hd(cfg, vdata, tmp_path / "v", target="vae", total_steps=2, **QUIET)
    ck = list_checkpoints(tmp_path / "v")[-1][1]
    _, g2, _ = load_checkpoint(ck)
    torch.manual_seed(0)
    train_hd(cfg, vdata, tmp_path / "w", target="vae", total_steps=4, **QUIET)
    _, g4, _ = load_checkpoint(list_checkpoints(tmp_path / "w")[-1][1])
    assert "disc" in g2 and "opt_disc" in g2
    # before gan_start no discriminator step happened: its optimizer has no state
    assert not any(k.startswith("state.") for k in g2["opt_disc"])
    assert any(k.startswith("state.") for k in g4["opt_disc"])
    assert any(not torch.equal(g2["disc"][k], g4["disc"][k]) for k in g2["disc"])


def test_vae_gan_resume_is_exact(vdata, tmp_path):
    cfg = vae_cfg(vae_gan_start=2, save_every=2, keep_last=5)
    train_hd(cfg, vdata, tmp_path / "full", target="vae", total_steps=6, **QUIET)
    train_hd(cfg, vdata, tmp_path / "part", target="vae", total_steps=4, **QUIET)
    r = train_hd(cfg, vdata, tmp_path / "part", target="vae", total_steps=6, **QUIET)
    assert r["step"] == 6
    a = load_checkpoint(list_checkpoints(tmp_path / "full")[-1][1])[1]
    b = load_checkpoint(list_checkpoints(tmp_path / "part")[-1][1])[1]
    for grp in ("model", "disc"):
        for k in a[grp]:
            assert torch.allclose(a[grp][k], b[grp][k], atol=1e-5), (grp, k)


def test_vae_image_only_stage_and_stage_keys(vdata, tmp_path):
    cfg = vae_cfg(vae_gan_weight=0.0)
    cfg["train"]["vae_stages"] = [{"name": "img", "height": 32, "width": 64, "frames": 1, "steps": 3}]
    res = train_hd(cfg, vdata, tmp_path / "v", target="vae", **QUIET)
    assert [h["image_batch"] for h in res["history"]] == [1.0, 1.0, 1.0] and "gan_g" not in res["history"][0]
    # calibration after an image stage still uses video batches (force_video) and per-channel statistics
    assert len(res["latent_norm"]["latent_scale"]) == 8


def test_vae_scalar_norm_is_still_the_default_when_not_requested(vdata, tmp_path):
    cfg = legacy_cfg()
    res = train_hd(cfg, vdata, tmp_path / "v", target="vae", total_steps=2, **QUIET)
    assert isinstance(res["latent_norm"]["latent_shift"], float)
    man = json.loads((tmp_path / "v" / "hc_vae" / "hc_vae.json").read_text())
    assert man["version"] == "nivra-hc-vae/1" and "latent_norm" in man and "latent_norm" not in man["config"]


# ===================================================================== DiT training
def dit_cfg(stages=None, **train):
    cfg = load_hd_config(tiny_cfg(train={
        "stages": stages or [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 6}],
        "batch_size": 2, "lr": 3e-3, "caption_dropout": 0.0, "save_every": 100, "keep_last": 5, **train}))
    return cfg


def test_dit_sra_df_ema_train_and_export_ema(ldata, tmp_path):
    cfg = dit_cfg(ema_decay=0.9, sra={"weight": 0.5, "student_layer": 0, "teacher_layer": 1},
                  diffusion_forcing={"fraction": 1.0}, fp32_attn=True)
    net = build_dit(cfg)
    res = train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, **QUIET)
    h = res["history"]
    assert all("sra" in e and e["df"] == 1.0 and math.isfinite(e["loss"]) for e in h)
    assert all(e["loss"] == pytest.approx(e["rf"] + 0.5 * e["sra"], rel=1e-4) for e in h)
    assert 0.0 <= h[0]["sra"] <= 2.0
    assert res["export_kind"] == "ema" and set(res["export_state"]) == set(net.state_dict())
    assert any(not torch.allclose(res["export_state"][k], v.cpu()) for k, v in net.state_dict().items())
    assert all(net.blocks[0].attn.fp32 and b.cross.fp32 for b in net.blocks)
    # the projector is training-only: not part of the model, but checkpointed
    assert not any(k.startswith(("sra", "proj")) for k in net.state_dict())
    groups = load_checkpoint(list_checkpoints(tmp_path / "r")[-1][1])[1]
    assert {"model", "ema", "sra", "opt_sra", "optimizer"} <= set(groups)
    for k, v in res["export_state"].items():
        torch.testing.assert_close(groups["ema"][k], v)


def test_dit_sra_ema_resume_is_exact(ldata, tmp_path):
    cfg = dit_cfg(ema_decay=0.9, sra={"weight": 0.5, "student_layer": 0, "teacher_layer": 1},
                  diffusion_forcing={"fraction": 0.5}, save_every=2)
    enc = StubTextEncoder(16)
    train_hd(cfg, ldata, tmp_path / "full", text_encoder=enc, **QUIET)
    train_hd(cfg, ldata, tmp_path / "part", text_encoder=enc, total_steps=4, **QUIET)
    r = train_hd(cfg, ldata, tmp_path / "part", text_encoder=enc, **QUIET)
    assert r["step"] == 6
    a = load_checkpoint(list_checkpoints(tmp_path / "full")[-1][1])[1]
    b = load_checkpoint(list_checkpoints(tmp_path / "part")[-1][1])[1]
    for grp in ("model", "ema", "sra"):
        for k in a[grp]:
            assert torch.allclose(a[grp][k], b[grp][k], atol=1e-6), (grp, k)


def test_dit_ema_can_be_switched_on_when_resuming_an_old_run(ldata, tmp_path):
    enc = StubTextEncoder(16)
    old = dit_cfg()
    train_hd(old, ldata, tmp_path / "r", text_encoder=enc, total_steps=3, **QUIET)
    new = dit_cfg(ema_decay=0.9)
    r = train_hd(new, ldata, tmp_path / "r", text_encoder=enc, **QUIET)
    assert r["step"] == 6 and r["export_kind"] == "ema"


def test_sra_without_ema_uses_the_online_model_as_teacher(ldata, tmp_path):
    logs = []
    cfg = dit_cfg(sra={"weight": 1.0, "student_layer": 0, "teacher_layer": 1})
    res = train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), log=logs.append)
    assert "export_state" not in res and any("without EMA" in str(l) for l in logs)
    assert all("sra" in h for h in res["history"])


def test_sra_until_step_stops_the_alignment(ldata, tmp_path):
    cfg = dit_cfg(ema_decay=0.9, sra={"weight": 0.5, "student_layer": 0, "teacher_layer": 1, "until_step": 3})
    res = train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), **QUIET)
    assert ["sra" in h for h in res["history"]] == [True] * 3 + [False] * 3


class SpyDiT(NivraDiT):
    log = []

    def forward(self, x, t, text=None, text_mask=None, drop_text=None, t_offset=0, cond_mask=None, fps=None, feats=None):
        self.log.append({"t": t.clone(), "mask": None if cond_mask is None else cond_mask.clone(),
                         "fps": None if fps is None else fps.clone(), "T": x.shape[2]})
        return super().forward(x, t, text, text_mask, drop_text, t_offset, cond_mask, fps, feats)


def spy_net(cfg):
    SpyDiT.log = []
    torch.manual_seed(0)
    return SpyDiT(build_dit(cfg).cfg)


def test_diffusion_forcing_and_frame_cond_batches(ldata, tmp_path):
    cfg = load_hd_config(tiny_cfg(model={"frame_cond": True}, train={
        "stages": [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 8}], "batch_size": 2, "lr": 3e-3,
        "caption_dropout": 0.0, "save_every": 100, "diffusion_forcing": {"fraction": 0.5, "cond_fraction": 0.5}}))
    net = spy_net(cfg)
    res = train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, **QUIET)
    log = SpyDiT.log
    assert len(log) == 8 and all(math.isfinite(h["loss"]) for h in res["history"])
    per_frame = [c for c in log if c["t"].dim() == 2]
    plain = [c for c in log if c["t"].dim() == 1 and c["mask"] is None]
    cond = [c for c in log if c["mask"] is not None]
    assert per_frame and plain and cond
    assert all(c["t"].shape == (2, 3) for c in per_frame)
    distinct = [len(set(c["t"][0].tolist())) for c in per_frame]
    assert 3 in distinct                                       # diffusion forcing: independent levels per latent frame
    assert set(distinct) <= {1, 3}                             # (1 = a cond-only batch: one level broadcast to all frames)
    for c in cond:
        m = c["mask"]
        assert m.shape == (2, 3) and m.any() and not m.all()   # always at least one noisy frame
        assert m[:, 0].all() or m[:, -1].all()                 # first and/or last latent frame
    flags = [(h.get("df", 0.0), h.get("cond", 0.0)) for h in res["history"]]
    assert sum(f[1] for f in flags) == len(cond) and sum(f[0] for f in flags) >= 1       # logged flags match what the model saw


def test_cond_mask_patterns_cover_first_last_both(ldata, tmp_path):
    cfg = load_hd_config(tiny_cfg(model={"frame_cond": True}, train={
        "stages": [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 40}], "batch_size": 1, "lr": 1e-3,
        "caption_dropout": 0.0, "save_every": 1000, "diffusion_forcing": {"cond_fraction": 1.0}}))
    net = spy_net(cfg)
    train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, **QUIET)
    pats = {tuple(c["mask"][0].tolist()) for c in SpyDiT.log}
    assert pats == {(True, False, False), (True, True, False), (False, False, True), (True, False, True)}


def test_fps_conditioning_reads_manifest_and_drops_deterministically(tmp_path):
    rng = np.random.default_rng(0)
    root = write_dataset(tmp_path / "d", key="latents", latents=rng.standard_normal((4, 8, 3, 2, 2)).astype(np.float16))
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
    for i, r in enumerate(rows):
        r["effective_fps"] = [12.0, 8.0, 24.0, None][i]
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = load_hd_config(tiny_cfg(model={"fps_cond": True}, train={
        "stages": [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 4}], "batch_size": 4, "lr": 1e-3,
        "caption_dropout": 0.0, "save_every": 100, "fps_dropout": 0.0}))
    net = spy_net(cfg)
    train_hd(cfg, root, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, **QUIET)
    for c in SpyDiT.log:
        f = c["fps"]
        assert f.shape == (4,) and int(f.isnan().sum()) == 1                                           # the None row
        assert sorted(x for x in f.tolist() if x == x) == [8.0, 12.0, 24.0]
    cfg["train"]["fps_dropout"] = 1.0
    net = spy_net(cfg)
    train_hd(cfg, root, tmp_path / "r2", text_encoder=StubTextEncoder(16), model=net, **QUIET)
    assert all(bool(c["fps"].isnan().all()) for c in SpyDiT.log)


def test_image_first_and_joint_stages_on_latent_shards_with_stills(tmp_path):
    """Video latents (T=3) and still latents (T=1) in one dataset: image stages draw from both, video stages only
    from the video latents (a T=1 still cannot serve a video bucket)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "d"
    root.mkdir()
    save_file({"latents": rng.standard_normal((2, 8, 3, 2, 2)).astype(np.float16)}, str(root / "lat-0.safetensors"))
    save_file({"latents": rng.standard_normal((2, 8, 1, 2, 2)).astype(np.float16)}, str(root / "lat-still.safetensors"))
    rows = [{"id": f"v{i}", "split": "train", "shard": "lat-0.safetensors", "index": i, "caption": "a red ball", "lat_t": 3}
            for i in range(2)] + [{"id": f"s{i}", "split": "train", "shard": "lat-still.safetensors", "index": i,
                                   "caption": "a red ball", "lat_t": 1, "still": True} for i in range(2)]
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = load_hd_config(tiny_cfg(train={
        "stages": [{"name": "img", "height": 64, "width": 64, "frames": 1, "steps": 4},
                   {"name": "joint", "height": 64, "width": 64, "frames": 17, "steps": 8, "image_fraction": 0.5}],
        "batch_size": 4, "lr": 1e-3, "caption_dropout": 0.0, "save_every": 100}))
    net = spy_net(cfg)
    res = train_hd(cfg, root, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, **QUIET)
    assert all(math.isfinite(h["loss"]) for h in res["history"])
    Ts = [c["T"] for c in SpyDiT.log]
    assert Ts[:4] == [1, 1, 1, 1] and set(Ts[4:]) == {1, 3}           # image stage, then image + video batches
    only_stills = tmp_path / "s"
    only_stills.mkdir()
    save_file({"latents": rng.standard_normal((2, 8, 1, 2, 2)).astype(np.float16)}, str(only_stills / "x.safetensors"))
    (only_stills / "manifest.jsonl").write_text("".join(json.dumps({"id": f"s{i}", "split": "train", "shard": "x.safetensors",
                                                                   "index": i, "caption": "c", "still": True}) + "\n" for i in range(2)))
    with pytest.raises(ValueError, match="only stills"):
        train_hd(cfg, only_stills, tmp_path / "r2", text_encoder=StubTextEncoder(16), total_steps=6, **QUIET)


def test_image_stage_takes_first_latent_frame_of_video_latents(ldata, tmp_path):
    cfg = dit_cfg(stages=[{"name": "img", "height": 64, "width": 64, "frames": 1, "steps": 2}])
    net = spy_net(cfg)
    train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, **QUIET)
    assert [c["T"] for c in SpyDiT.log] == [1, 1]


def test_image_stage_from_video_shards_uses_random_frames_and_per_resolution_shift(vdata, tmp_path):
    """Image batches sampled from raw videos go through the VAE as T=1; the timestep shift is computed from the
    batch's own token count (tokens = T*H*W of the latent), the same rule inference applies."""
    from nivra.models.dit import resolution_shift
    cfg = dit_cfg(stages=[{"name": "img", "height": 32, "width": 64, "frames": 1, "steps": 2},
                          {"name": "vid", "height": 64, "width": 64, "frames": 17, "steps": 2}],
                  shift_base_tokens=4)
    net = spy_net(cfg)
    torch.manual_seed(0)
    vae = build_hcvae(cfg)
    train_hd(cfg, vdata, tmp_path / "r", text_encoder=StubTextEncoder(16), vae=vae, model=net, **QUIET)
    assert [c["T"] for c in SpyDiT.log] == [1, 1, 3, 3]
    toks = [h["tokens"] for h in train_hd(cfg, vdata, tmp_path / "r2", text_encoder=StubTextEncoder(16), vae=vae,
                                         model=spy_net(cfg), **QUIET)["history"]]
    assert toks == [2.0, 2.0, 12.0, 12.0]
    assert resolution_shift(2, 4) == 1.0 and resolution_shift(12, 4) == pytest.approx(math.sqrt(3))


# ===================================================================== fp16 loss scaling
class BlowupDiT(NivraDiT):
    """The first `blow` forward calls return a velocity multiplied by 1e36: finite loss, but backward gradients far
    above float32 max once multiplied by the loss scale -- a deterministic stand-in for an fp16 overflow."""
    blow = 1
    calls = 0
    mult = 1e36

    def forward(self, *a, **kw):
        out = super().forward(*a, **kw)
        type(self).calls += 1
        v = out[0] if isinstance(out, tuple) else out
        return v * type(self).mult if type(self).calls <= type(self).blow else v


def blowup_net(cfg, blow, mult=1e36):
    BlowupDiT.blow, BlowupDiT.calls, BlowupDiT.mult = blow, 0, mult
    torch.manual_seed(0)
    return BlowupDiT(build_dit(cfg).cfg)


def test_overflow_is_skipped_logged_and_does_not_touch_the_weights(ldata, tmp_path):
    logs = []
    cfg = dit_cfg(ema_decay=0.9, force_loss_scaling=True, stages=[{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 1}])
    net = blowup_net(cfg, 1)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    res = train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, log=logs.append)
    h = res["history"][0]
    assert res["skipped_steps"] == 1 and h["skipped"] == 1.0 and h["loss_scale"] == 32768.0       # 65536 * backoff 0.5
    assert any("overflow" in str(l) and "skipped" in str(l) and "65536" in str(l) for l in logs)
    for k, v in net.state_dict().items():
        assert torch.equal(v, before[k]), k                              # optimizer step skipped
    for k, v in res["export_state"].items():                              # and the EMA was not advanced either
        assert torch.equal(v, before[k]), k


def test_loss_scale_backs_off_until_finite_then_trains(ldata, tmp_path):
    cfg = dit_cfg(force_loss_scaling=True, loss_scale={"growth_interval": 1000}, ema_decay=0.5)
    net = blowup_net(cfg, 3)
    res = train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, **QUIET)
    sk = [h["skipped"] for h in res["history"]]
    assert sk == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0] and res["skipped_steps"] == 3
    assert [h["loss_scale"] for h in res["history"]][:4] == [32768.0, 16384.0, 8192.0, 8192.0]
    assert all(math.isfinite(h["loss"]) for h in res["history"])


def test_loss_scale_grows_after_finite_steps_and_survives_resume(ldata, tmp_path):
    cfg = dit_cfg(force_loss_scaling=True, loss_scale={"init": 1024.0, "growth_interval": 2}, save_every=2)
    enc = StubTextEncoder(16)
    train_hd(cfg, ldata, tmp_path / "r", text_encoder=enc, total_steps=4, **QUIET)
    st = load_checkpoint(list_checkpoints(tmp_path / "r")[-1][1])[2]
    assert st["scaler"]["scale"] == 4096.0 and st["skipped_steps"] == 0               # doubled every 2 finite steps
    r = train_hd(cfg, ldata, tmp_path / "r", text_encoder=enc, **QUIET)
    assert r["step"] == 6 and r["history"][4]["loss_scale"] == 4096.0


def test_history_and_checkpoints_are_json_safe_with_nonfinite_values(ldata, tmp_path):
    cfg = dit_cfg(force_loss_scaling=True, save_every=1, grad_clip=1.0)
    net = blowup_net(cfg, 1)
    res = train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, total_steps=2, **QUIET)
    assert res["history"][0]["grad_norm"] is None and res["history"][1]["grad_norm"] > 0       # inf -> null
    json.dumps(res["history"], allow_nan=False)
    st = load_checkpoint(list_checkpoints(tmp_path / "r")[-1][1])[2]
    json.dumps(st["history"], allow_nan=False)


def test_consecutive_overflow_limit_raises(ldata, tmp_path):
    cfg = dit_cfg(force_loss_scaling=True, stages=[{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 400}])
    net = blowup_net(cfg, 10 ** 9, float('nan'))          # the loss itself is NaN: no scale can fix it
    with pytest.raises(FloatingPointError, match="consecutive skipped steps"):
        train_hd(cfg, ldata, tmp_path / "r", text_encoder=StubTextEncoder(16), model=net, **QUIET)
