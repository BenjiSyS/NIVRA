"""2026 improvement pass, model side: HC-VAE (patch-GAN pieces, per-channel latent norm, temporal_factor), DiT
(per-frame timesteps, frame conditioning, fps conditioning, fp32 islands, SRA features) and backward compatibility
of old configs / bundles (golden values captured from the code before these changes)."""
import copy
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from nivra.compression.hc_vae import (HCVAE, HCVAE_VERSION, HCVAE_VERSION_2, HCVAEConfig, PatchDiscriminator,
                                      adaptive_gan_weight, hinge_d_loss, hinge_g_loss, load_hcvae, save_hcvae)
from nivra.inference.hd import HDPipeline, StubTextEncoder, sample_flow
from nivra.inference.hd_bundle import load_hd_bundle, write_hd_bundle
from nivra.models.dit import (DIT_VERSION_2, DiTConfig, LayerNorm32, NivraDiT, linear32, make_xt, modulate,
                              rf_loss, rf_loss_ex, sample_t_frames, sdpa_chunked)
from nivra.models.hd_config import (HDConfigError, build_dit, hd_config_hash, load_hd_config, train_options, vae_config,
                                    vae_edge_weight)

# The configuration the shipped nivra_hd_tiny had BEFORE the 2026 pass (architecture + inference; old train section).
OLD_TINY = {
    "name": "old_tiny", "arch_version": "nivra-hd/1", "domain": "stylized", "styles": ["anime", "cartoon"],
    "default_style": "anime",
    "vae": {"latent_channels": 8, "channels": [8, 8, 16, 16, 16, 16], "temporal_down": [1, 2, 3], "n_res": 1,
            "n_res_dec": 1, "edge_weight": 0.5},
    "model": {"dim": 32, "depth": 2, "heads": 2, "mlp_ratio": 2.0, "patch": [1, 1, 1], "text_dim": 16, "qk_norm": True},
    "inference": {"width": 96, "height": 64, "seconds": 0.53, "fps": 30, "steps": 4, "cfg_scale": 3.0, "sampler": "ab2",
                  "shift_base_tokens": 16, "fp16": True, "int8": False, "offload": False, "attn_chunk": 0,
                  "mlp_chunk": 0, "vae_t_chunk": 1, "vae_tile": 0, "vae_overlap": 1, "vae_margin": None},
    "train": {"stages": [{"name": "a", "height": 32, "width": 64, "frames": 9, "steps": 2}], "batch_size": 1}}

# Captured with the pre-change code (git HEAD ff517e7) on this machine: write_hd_bundle(OLD_TINY, seed=3), then
# generate("a red ball bounces", seed=5, steps=3) and continue_video(state, seed=7, steps=3).
GOLDEN = {
    "hash": "7ee73ebc77a595cfe6e9e376ac24dbb7d0c70b50e89694f1ce312532985afbe8",
    "weights_sha": "7578457e03cc14e94e6f8061efe7acef1266ea4cdc2f0a5fc129f6a788032f92",
    "vae_sha": "746fb29ac30d859d5195415fcd5ad229319ee93d490d1d601aafc0aeb67d03c1",
    "params": (45896, 187371),
    # the DiT path is bit-identical (latent sum below matches to 1e-12); the VAE decode conv rounds differently per
    # thread count (+-1 pixel LSB), so pixels are compared by statistics instead of a hash
    "latent_sum": 11.498518005013466,
    "latent_head": [1.778386, 0.42018, -1.750405, -1.613286, 2.109216, -0.151546],
    "frames_mean": 141.8177, "frame_means": [142.482, 143.209, 142.427, 142.713, 142.66],
    "cont_mean": 141.5586, "cont_shape": (8, 64, 96, 3),
}
OLD_HASHES = {   # hd_config_sha of the shipped medium / large configs before the pass (their architecture is unchanged)
    "nivra_medium": "4d49b440c0ac85212b2bbc1a8a76295a829d7fe86bbce9aa966ea0618e32ed24",
    "nivra_large": "a926c41ae652ddf1dbc8b21dfe11a91d0870303154aed6e1c99528ad8348dcfc",
}


def small_vae_cfg(**kw):
    d = {"latent_channels": 4, "channels": [4, 4, 8, 8, 8, 8], "n_res": 1, "n_res_dec": 1}
    d.update(kw)
    return HCVAEConfig.from_dict(d)


def tiny_cfg(model=None, vae=None, train=None, inference=None):
    cfg = copy.deepcopy(OLD_TINY)
    cfg["model"].update(model or {})
    cfg["vae"].update(vae or {})
    cfg["train"].update(train or {})
    cfg["inference"].update(inference or {})
    return cfg


# ===================================================================== backward compatibility
def test_old_tiny_bundle_loads_and_generates_identically(tmp_path):
    assert hd_config_hash(load_hd_config(OLD_TINY)) == GOLDEN["hash"]
    m = write_hd_bundle(tmp_path / "b", OLD_TINY, seed=3)
    assert (m["weights_sha256"], m["vae_sha256"], m["hd_config_sha"]) == (GOLDEN["weights_sha"], GOLDEN["vae_sha"], GOLDEN["hash"])
    assert (m["param_count"], m["vae_param_count"]) == GOLDEN["params"]
    pipe = load_hd_bundle(tmp_path / "b", "cpu")
    r = pipe.generate("a red ball bounces", seed=5, steps=3)
    assert float(r.latents.double().sum()) == pytest.approx(GOLDEN["latent_sum"], rel=1e-12)
    assert r.latents.flatten()[:6].tolist() == pytest.approx(GOLDEN["latent_head"], abs=1e-6)
    assert r.frames.shape == (17, 64, 96, 3) and float(r.frames.mean()) == pytest.approx(GOLDEN["frames_mean"], abs=0.01)
    assert r.frames.reshape(17, -1).mean(1)[:5].round(3).tolist() == pytest.approx(GOLDEN["frame_means"], abs=0.01)
    r2 = pipe.continue_video(r.state, seed=7, steps=3)
    assert r2.frames.shape == GOLDEN["cont_shape"] and float(r2.frames.mean()) == pytest.approx(GOLDEN["cont_mean"], abs=0.01)
    assert r2.meta["conditioning"] == "replace"          # old model: noise-path replacement, as before
    assert "latent_norm" not in m["config"]["vae"] and "frame_cond" not in m["config"]["model"]


def test_old_configs_keep_their_hash_and_dict_shape():
    for name, h in OLD_HASHES.items():
        assert hd_config_hash(load_hd_config(name)) == h
    assert "latent_norm" not in vae_config(load_hd_config(OLD_TINY)).to_dict()
    assert set(DiTConfig.from_dict({"dim": 32, "depth": 1, "heads": 2, "latent_channels": 4, "text_dim": 8}).to_dict()) \
        == {"dim", "depth", "heads", "mlp_ratio", "latent_channels", "patch", "text_dim", "qk_norm", "rope_theta",
            "freq_dim", "version"}


def test_shipped_configs_load_and_new_options_validate():
    for n in ("nivra_hd_tiny", "nivra_base", "nivra_medium", "nivra_large"):
        cfg = load_hd_config(n)
        o = train_options(cfg)
        assert o["ema_decay"] > 0 and o["sra"]["weight"] > 0 and o["vae_gan_weight"] > 0
        assert o["diffusion_forcing"]["fraction"] == 0.25 and o["vae_latent_norm"] == "channel" and o["fp32_attn"]
        assert vae_edge_weight(cfg) == 0.5 and o["vae_stages"]
    base = load_hd_config("nivra_base")
    assert vae_config(base).temporal_factor == 4 and base["model"]["frame_cond"] and base["model"]["fps_cond"]
    assert train_options(base)["diffusion_forcing"]["cond_fraction"] > 0
    assert load_hd_config("nivra_medium")["vae"]["temporal_down"] == [1, 2, 3]          # existing configs stay 8x


def test_edge_weight_defaults_on_for_stylized_only():
    st = load_hd_config(tiny_cfg())
    del st["vae"]["edge_weight"]
    assert vae_edge_weight(st) == pytest.approx(0.3)
    gen = copy.deepcopy(st)
    gen["domain"] = "general"
    assert vae_edge_weight(gen) == 0.0
    st["vae"]["edge_weight"] = 0.0
    assert vae_edge_weight(st) == 0.0                                                   # explicit wins


@pytest.mark.parametrize("bad", [
    {"sra": {"weight": 1.0, "student_layer": 5, "teacher_layer": 3}},
    {"sra": {"weight": 1.0, "student_layer": 0, "teacher_layer": 9}},
    {"sra": {"nonsense": 1}},
    {"diffusion_forcing": {"fraction": 1.5}},
    {"diffusion_forcing": {"cond_fraction": 0.5}},          # model has no frame_cond
    {"ema_decay": 1.0},
    {"vae_latent_norm": "per-pixel"},
    {"vae_gan_weight": -1},
    {"loss_scale": {"init": 0.5}},
    {"vae_stages": []},
    {"vae_stages": [{"name": "x", "height": 32, "width": 32, "frames": 1, "steps": 1, "image_fraction": 2}]},
    {"repa": {"enabled": True, "teacher": "dinov2"}},
])
def test_bad_train_options_rejected(bad):
    with pytest.raises(HDConfigError):
        load_hd_config(tiny_cfg(train=bad))


def test_repa_hook_is_documented_but_unimplemented():
    from nivra.training.hd_trainer import RepaHook
    with pytest.raises(NotImplementedError):
        RepaHook().features({}, (1,))
    with pytest.raises(HDConfigError, match="DINOv2"):
        load_hd_config(tiny_cfg(train={"repa": {"enabled": True}}))


# ===================================================================== VAE
def test_temporal_factor_option():
    c8 = small_vae_cfg()
    c4 = small_vae_cfg(temporal_factor=4)
    assert (c8.temporal_factor, c8.temporal_down) == (8, (1, 2, 3))
    assert (c4.temporal_factor, c4.temporal_down) == (4, (2, 3))
    assert small_vae_cfg(temporal_factor=8).temporal_down == (1, 2, 3)
    assert c4.valid_frames(9) and not c4.valid_frames(10) and c4.latent_shape(9, 32, 64) == (4, 3, 1, 2)
    assert c4.frames_for_latent(3) == 9
    with pytest.raises(ValueError, match="temporal_factor"):
        small_vae_cfg(temporal_factor=2)
    with pytest.raises(ValueError, match="contradicts"):
        small_vae_cfg(temporal_factor=4, temporal_down=[1, 2, 3])
    # temporal_factor is an alias for temporal_down: the stored dict (and so hd_config_sha) only has temporal_down
    assert "temporal_factor" not in c4.to_dict() and HCVAEConfig.from_dict(c4.to_dict()).temporal_factor == 4


def test_vae_4x_encode_decode_and_streaming_equal_full():
    torch.manual_seed(0)
    vae = HCVAE(small_vae_cfg(temporal_factor=4)).eval()
    x = torch.randn(1, 3, 9, 32, 64).clamp(-1, 1)
    with torch.no_grad():
        z = vae.encode(x)
        assert z.shape == (1, 4, 3, 1, 2)
        y = vae.decode(z)
        assert y.shape == x.shape
        torch.testing.assert_close(vae.decode_tiled(z, t_chunk=1), y, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(vae.encode_tiled(x, t_chunk=1), z, atol=1e-4, rtol=1e-4)
        img = vae.encode(x[:, :, :1])
        torch.testing.assert_close(img, z[:, :, :1], atol=1e-5, rtol=1e-5)      # a still == latent frame 0


def test_per_channel_latent_norm(tmp_path):
    torch.manual_seed(1)
    vae = HCVAE(small_vae_cfg(channels=[4, 4, 8, 8, 8, 8])).eval()
    assert vae.cfg.version == HCVAE_VERSION and vae.cfg.latent_norm == "scalar"
    with torch.no_grad():
        z = vae.encode(torch.rand(2, 3, 9, 32, 64) * 2 - 1)
    z = z * torch.tensor([0.1, 1.0, 5.0, 20.0]).view(1, 4, 1, 1, 1) + torch.tensor([3.0, -2.0, 0.0, 10.0]).view(1, 4, 1, 1, 1)
    info = vae.calibrate(z, per_channel=True)
    assert info["latent_norm"] == "channel" and len(info["latent_shift"]) == 4 and len(info["latent_scale"]) == 4
    assert vae.cfg.version == HCVAE_VERSION_2
    zs = vae.to_scaled(z)
    torch.testing.assert_close(zs.mean((0, 2, 3, 4)), torch.zeros(4), atol=1e-4, rtol=0)
    torch.testing.assert_close(zs.std((0, 2, 3, 4)), torch.ones(4), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(vae.from_scaled(zs), z, atol=1e-4, rtol=1e-5)
    assert vae.to_scaled(z[:1, :, :1]).shape == (1, 4, 1, 1, 2)                 # works on any T
    z16 = z.half()
    assert vae.to_scaled(z16).dtype == torch.float16
    # stored in the VAE metadata (hc_vae.json + safetensors metadata) and survives save / load
    man = save_hcvae(vae, tmp_path / "v")
    assert man["latent_norm"] == "channel" and man["config"]["latent_shift"] == vae.cfg.latent_shift
    back = load_hcvae(tmp_path / "v")
    assert back.cfg.latent_norm == "channel" and back.cfg.latent_scale == vae.cfg.latent_scale
    assert back.cfg.version == HCVAE_VERSION_2
    torch.testing.assert_close(back.to_scaled(z), zs)
    # scalar calibration is still available and still emits a v1-shaped dict
    v2 = HCVAE(small_vae_cfg())
    v2.calibrate(z)
    assert isinstance(v2.cfg.latent_shift, float) and "latent_norm" not in v2.cfg.to_dict()
    # validation
    with pytest.raises(ValueError, match="latent_channels"):
        HCVAEConfig.from_dict({**vae.cfg.to_dict(), "latent_shift": [0.0, 1.0]})
    with pytest.raises(ValueError, match="nivra-hc-vae/2"):
        HCVAEConfig.from_dict({**vae.cfg.to_dict(), "version": HCVAE_VERSION})
    with pytest.raises(ValueError, match="scale"):
        HCVAEConfig.from_dict({**vae.cfg.to_dict(), "latent_scale": [1.0, 1.0, 0.0, 1.0]})


def test_per_channel_calibration_does_not_change_config_hash():
    cfg = load_hd_config(tiny_cfg())
    h0 = hd_config_hash(cfg)
    v = HCVAE(vae_config(cfg))
    v.calibrate(torch.randn(2, 8, 2, 1, 2), per_channel=True)
    cfg2 = copy.deepcopy(cfg)
    cfg2["vae"] = {**v.cfg.to_dict(), "edge_weight": 0.5}
    assert hd_config_hash(load_hd_config(cfg2)) == h0


def test_patch_discriminator_and_hinge_losses():
    torch.manual_seed(0)
    d = PatchDiscriminator(3, base=8, n_layers=2)
    vid, img = torch.randn(2, 3, 5, 32, 64), torch.randn(2, 3, 1, 32, 64)
    assert d(vid).shape == (2, 1, 5, 8, 16) and d(img).shape == (2, 1, 1, 8, 16)
    real, fake = torch.full((4,), 2.0), torch.full((4,), -2.0)
    assert hinge_d_loss(real, fake).item() == pytest.approx(0.0)           # confident and right
    assert hinge_d_loss(fake, real).item() == pytest.approx(3.0)           # confident and wrong: 0.5*(3+3)
    assert hinge_g_loss(real).item() == pytest.approx(-2.0)
    assert sum(p.numel() for p in PatchDiscriminator(3).parameters()) < 1_000_000          # small
    with pytest.raises(ValueError):
        PatchDiscriminator(3, n_layers=0)


def test_adaptive_gan_weight_matches_gradient_norm_ratio_and_guards_nan():
    w = nn.Parameter(torch.tensor([1.0, 2.0]))
    rec = (w * torch.tensor([3.0, 4.0])).sum()          # |grad| = 5
    gan = (w * torch.tensor([0.0, 0.5])).sum()          # |grad| = 0.5
    aw = adaptive_gan_weight(rec, gan, w)
    assert aw.item() == pytest.approx(5.0 / (0.5 + 1e-4), rel=1e-4) and not aw.requires_grad
    assert adaptive_gan_weight(rec, gan, w, loss_scale=1024.0).item() == pytest.approx(aw.item(), rel=1e-4)
    assert adaptive_gan_weight(rec, gan * 0, w, max_weight=7.0).item() == pytest.approx(7.0)    # clamped
    bad = (w * torch.tensor([float("inf"), 1.0])).sum()
    assert adaptive_gan_weight(bad, gan, w).item() == 0.0                                          # non-finite -> off
    unrelated = nn.Parameter(torch.ones(1))
    assert adaptive_gan_weight(rec, gan, unrelated).item() == 0.0                                  # no path to layer


# ===================================================================== DiT
def make_dit(**kw):
    cfg = DiTConfig.from_dict({"dim": 32, "depth": 3, "heads": 2, "mlp_ratio": 2.0, "latent_channels": 4, "text_dim": 16,
                               **kw})
    torch.manual_seed(0)
    m = NivraDiT(cfg)
    with torch.no_grad():                               # leave the zero-init behaviour: make every path matter
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m.eval()


def inputs(B=2, T=4, H=2, W=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 4, T, H, W, generator=g)
    emb = torch.randn(B, 5, 16, generator=g)
    mask = torch.ones(B, 5, dtype=torch.bool)
    return x, emb, mask


def test_dit_versions_and_dict_shape():
    v1 = DiTConfig.from_dict({"dim": 32, "depth": 1, "heads": 2, "latent_channels": 4, "text_dim": 8})
    assert v1.version == "nivra-dit/1" and "frame_cond" not in v1.to_dict()
    v2 = DiTConfig.from_dict({"dim": 32, "depth": 1, "heads": 2, "latent_channels": 4, "text_dim": 8, "frame_cond": True})
    assert v2.version == DIT_VERSION_2 and v2.to_dict()["frame_cond"] is True and "fps_cond" not in v2.to_dict()
    assert DiTConfig.from_dict(v2.to_dict()) == v2
    with pytest.raises(ValueError, match="nivra-dit/2"):
        DiTConfig.from_dict({"dim": 32, "depth": 1, "heads": 2, "latent_channels": 4, "text_dim": 8, "fps_cond": True,
                             "version": "nivra-dit/1"})
    with pytest.raises(ValueError):
        DiTConfig.from_dict({"dim": 32, "depth": 1, "heads": 2, "latent_channels": 4, "text_dim": 8, "frame_cond": 1})
    h1 = hd_config_hash(load_hd_config(tiny_cfg()))
    h2 = hd_config_hash(load_hd_config(tiny_cfg(model={"frame_cond": True})))
    assert h1 != h2
    names = {k for k in NivraDiT(v2).state_dict()}
    assert "cond_embed" in names and not any(k.startswith("fps_embed") for k in names)
    assert not ({"cond_embed", "fps_null"} & set(NivraDiT(v1).state_dict()))


def test_constant_per_frame_t_equals_scalar_t_on_a_v1_model():
    m = make_dit()
    x, emb, mask = inputs()
    t = torch.tensor([0.3, 0.8])
    with torch.no_grad():
        a = m(x, t, emb, mask)
        b = m(x, t[:, None].expand(2, 4).contiguous(), emb, mask)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    t2 = t[:, None].expand(2, 4).clone()
    t2[:, 2] = 0.05                                     # one frame at another noise level changes the output
    with torch.no_grad():
        c = m(x, t2, emb, mask)
    assert not torch.allclose(a, c, atol=1e-4)
    with pytest.raises(ValueError, match="per-frame t"):
        m(x, torch.rand(2, 3), emb, mask)


def test_per_frame_modulation_matches_manual_loop():
    x = torch.randn(2, 6, 4)                              # tokens: 3 frames x 2
    sh, sc = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    out = modulate(x, sh, sc)
    for b in range(2):
        for f in range(3):
            for k in range(2):
                n = f * 2 + k
                torch.testing.assert_close(out[b, n], x[b, n] * (1 + sc[b, f]) + sh[b, f])
    torch.testing.assert_close(modulate(x, sh[:, 0], sc[:, 0]), x * (1 + sc[:, :1, :].squeeze(1)[:, None]) + sh[:, 0][:, None])


def test_frame_cond_forces_t_zero_and_adds_flag():
    m = make_dit(frame_cond=True)
    x, emb, mask = inputs()
    cm = torch.zeros(2, 4, dtype=torch.bool)
    cm[:, 0] = True
    cm[1, 3] = True
    t = torch.tensor([0.6, 0.9])
    t_pf = t[:, None].expand(2, 4).clone()
    t_zero = torch.where(cm, torch.zeros(()), t_pf)
    with torch.no_grad():
        m.cond_embed.zero_()
        a = m(x, t, emb, mask, cond_mask=cm)
        b = m(x, t_zero, emb, mask)                       # same thing written as explicit per-frame t
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        m.cond_embed.normal_()
        c = m(x, t, emb, mask, cond_mask=cm)
    assert not torch.allclose(a, c, atol=1e-4)            # the flag embedding reaches the output
    with pytest.raises(ValueError, match="frame_cond"):
        make_dit()(x, t, emb, mask, cond_mask=cm)
    with pytest.raises(ValueError, match="cond_mask"):
        m(x, t, emb, mask, cond_mask=cm[:, :3])


def test_fps_cond_null_and_effect():
    m = make_dit(fps_cond=True)
    x, emb, mask = inputs()
    t = torch.tensor([0.4, 0.4])
    with torch.no_grad():
        m.fps_embed[2].weight.zero_()
        m.fps_embed[2].bias.zero_()
        m.fps_null.zero_()
        base = m(x, t, emb, mask)
        torch.testing.assert_close(m(x, t, emb, mask, fps=torch.tensor([12.0, 8.0])), base)       # fresh: a no-op
        m.fps_embed[2].weight.normal_()
        m.fps_null.normal_()
        none = m(x, t, emb, mask)
        nan = m(x, t, emb, mask, fps=torch.tensor([float("nan")] * 2))
        torch.testing.assert_close(none, nan)                                                     # NaN == unknown
        f12 = m(x, t, emb, mask, fps=torch.tensor([12.0, 12.0]))
        f8 = m(x, t, emb, mask, fps=torch.tensor([8.0, 8.0]))
        mixed = m(x, t, emb, mask, fps=torch.tensor([12.0, float("nan")]))
    assert not torch.allclose(f12, f8, atol=1e-4) and not torch.allclose(f12, none, atol=1e-4)
    torch.testing.assert_close(mixed[0], f12[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mixed[1], none[1], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="fps_cond"):
        make_dit()(x, t, emb, mask, fps=torch.tensor([8.0, 8.0]))
    with pytest.raises(ValueError, match="one value"):
        m(x, t, emb, mask, fps=torch.tensor([8.0]))


def test_per_frame_t_with_fps_cond_and_frame_cond_together():
    m = make_dit(frame_cond=True, fps_cond=True)
    x, emb, mask = inputs()
    cm = torch.zeros(2, 4, dtype=torch.bool)
    cm[:, -1] = True
    t = torch.rand(2, 4)
    with torch.no_grad():
        v = m(x, t, emb, mask, cond_mask=cm, fps=torch.tensor([12.0, 8.0]))
    assert v.shape == x.shape and torch.isfinite(v).all()


def test_rf_loss_ex_masks_conditioning_frames_and_is_backward_compatible():
    m = make_dit(frame_cond=True)
    x0, emb, mask = inputs()
    noise = torch.randn_like(x0)
    t = torch.tensor([0.5, 0.7])
    plain = rf_loss(m, x0, emb, mask, t, noise)
    ref = torch.nn.functional.mse_loss(m(make_xt(x0, noise, t), t, emb, mask), noise - x0)
    torch.testing.assert_close(plain, ref)
    cm = torch.zeros(2, 4, dtype=torch.bool)
    cm[:, 0] = True
    cm[0, 3] = True
    loss, fd = rf_loss_ex(m, x0, emb, mask, t[:, None].expand(2, 4), noise, None, cm)
    assert fd is None
    xt = make_xt(x0, noise, t[:, None].expand(2, 4), cm)
    torch.testing.assert_close(xt[:, :, 0], x0[:, :, 0])                              # clean conditioning frame
    v = m(xt, t[:, None].expand(2, 4), emb, mask, cond_mask=cm)
    keep = (~cm).view(2, 1, 4, 1, 1).float()
    expect = (((v - (noise - x0)) ** 2) * keep).sum() / keep.expand_as(v).sum()
    torch.testing.assert_close(loss, expect)
    # a v1 model keeps the exact original single-timestep loss
    m1 = make_dit()
    torch.testing.assert_close(rf_loss(m1, x0, emb, mask, t, noise),
                               torch.nn.functional.mse_loss(m1(make_xt(x0, noise, t), t, emb, mask), noise - x0))


def test_feats_return_block_states_and_backprop_into_early_blocks():
    m = make_dit()
    m.train()
    x, emb, mask = inputs()
    t = torch.tensor([0.5, 0.5])
    v, fd = m(x, t, emb, mask, feats=(0, 2))
    N = 4 * 2 * 3
    assert set(fd) == {0, 2} and fd[0].shape == (2, N, 32) and fd[0].dtype == torch.float32
    fd[0].sum().backward()
    assert m.blocks[0].fc1.weight.grad is not None and m.blocks[2].fc1.weight.grad is None
    plain = m(x, t, emb, mask)
    assert isinstance(plain, torch.Tensor)


def test_sample_t_frames_shape_and_shift():
    g = torch.Generator().manual_seed(0)
    t = sample_t_frames(3, 5, shift=1.0, generator=g)
    assert t.shape == (3, 5) and ((t > 0) & (t < 1)).all() and len(set(t.flatten().tolist())) == 15
    big = sample_t_frames(2000, 4, shift=4.0, generator=torch.Generator().manual_seed(0))
    assert big.mean() > t.mean()


# ===================================================================== fp32 islands / fp16 hardening
def test_layernorm32_is_exact_in_fp32_and_stable_in_half():
    ln, ref = LayerNorm32(8, eps=1e-6), nn.LayerNorm(8, eps=1e-6)
    x = torch.randn(3, 5, 8)
    torch.testing.assert_close(ln(x), ref(x), atol=0, rtol=0)
    big = (torch.randn(2, 4, 8) * 400).half()                       # variance 160000 > fp16 max 65504
    with torch.no_grad():
        out = ln(big)
    assert out.dtype == torch.float16 and torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref(big.float()), atol=2e-2, rtol=2e-2)


def test_modulate_and_linear32_run_in_fp32_for_half_inputs():
    x = (torch.randn(2, 6, 4) * 300).half()
    sh, sc = torch.randn(2, 4), torch.randn(2, 4) * 200
    out = modulate(x, sh, sc)
    assert out.dtype == torch.float16
    ref = x.float() * (1 + sc[:, None]) + sh[:, None]
    big = ref.abs() > 60000
    torch.testing.assert_close(out.float()[~big], ref[~big].half().float(), atol=0, rtol=2e-3)
    lin = nn.Linear(4, 3).half()
    y = linear32(lin, torch.randn(2, 4).half())
    assert y.dtype == torch.float32
    f32 = nn.Linear(4, 3)
    z = torch.randn(2, 4)
    torch.testing.assert_close(linear32(f32, z), f32(z), atol=0, rtol=0)


def test_fp32_attention_logits_do_not_overflow():
    q = torch.full((1, 2, 6, 8), 30.0).half()
    k = torch.full((1, 2, 6, 8), 30.0).half()
    v = torch.randn(1, 2, 6, 8).half()
    try:
        plain = sdpa_chunked(q, k, v, None, 0, False)
    except RuntimeError:
        pytest.skip("this torch build has no fp16 CPU attention")
    safe = sdpa_chunked(q, k, v, None, 2, True)
    assert safe.dtype == torch.float16 and torch.isfinite(safe).all()
    ref = sdpa_chunked(q.float(), k.float(), v.float(), None, 0, False)
    torch.testing.assert_close(safe.float(), ref, atol=2e-2, rtol=2e-2)
    del plain


def test_dit_modulation_is_float32_under_reduced_precision_autocast():
    m = make_dit()
    x, emb, mask = inputs()
    seen = {}
    m.blocks[0].register_forward_pre_hook(lambda mod, a: seen.update(mod=a[1].dtype, x=a[0].dtype))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        v = m(x, torch.tensor([0.5, 0.5]), emb, mask)
    assert seen["mod"] == torch.float32 and seen["x"] == torch.bfloat16     # modulation fp32 while the stream is bf16
    assert v.dtype == torch.float32 and torch.isfinite(v).all()
    with torch.no_grad():
        ref = m(x, torch.tensor([0.5, 0.5]), emb, mask)
    assert (v - ref).abs().max() < 0.25


def test_half_model_forward_is_finite_and_close_to_fp32():
    m = make_dit(frame_cond=True, fps_cond=True)
    m.set_memory(attn_fp32=True)
    x, emb, mask = inputs()
    t = torch.tensor([0.5, 0.2])
    cm = torch.zeros(2, 4, dtype=torch.bool)
    cm[:, 0] = True
    with torch.no_grad():
        ref = m(x, t, emb, mask, cond_mask=cm, fps=torch.tensor([12.0, 8.0]))
        try:
            mh = copy.deepcopy(m).half()
            out = mh(x.half(), t, emb.half(), mask, cond_mask=cm, fps=torch.tensor([12.0, 8.0]))
        except RuntimeError as e:
            pytest.skip(f"fp16 CPU kernels unavailable: {e}")
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    assert (out - ref).abs().max() < 0.1


# ===================================================================== inference-side conditioning
class SpyDiT(NivraDiT):
    calls = None

    def forward(self, x, t, *a, **kw):
        type(self).calls.append({"mask": kw.get("cond_mask"), "x0": x[:, :, :2].clone(), "t": t.clone()})
        return super().forward(x, t, *a, **kw)


def spy_model(**kw):
    cfg = DiTConfig.from_dict({"dim": 32, "depth": 1, "heads": 2, "latent_channels": 4, "text_dim": 16, **kw})
    SpyDiT.calls = []
    torch.manual_seed(0)
    m = SpyDiT(cfg).eval()
    return m


def test_sample_flow_uses_clean_conditioning_frames_for_v2_and_replacement_for_v1():
    known = torch.randn(1, 4, 2, 2, 3)
    emb, mask = torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)
    m2 = spy_model(frame_cond=True)
    out2 = sample_flow(m2, (1, 4, 5, 2, 3), emb, mask, steps=3, cfg_scale=1.0, seed=1, known=known)
    torch.testing.assert_close(out2[:, :, :2], known)
    assert all(c["mask"] is not None and c["mask"][0, :2].all() and not c["mask"][0, 2:].any() for c in SpyDiT.calls)
    for c in SpyDiT.calls:                                      # the known frames the model sees are CLEAN, every step
        torch.testing.assert_close(c["x0"], known)
    m1 = spy_model()
    sample_flow(m1, (1, 4, 5, 2, 3), emb, mask, steps=3, cfg_scale=1.0, seed=1, known=known)
    assert all(c["mask"] is None for c in SpyDiT.calls)
    assert not torch.allclose(SpyDiT.calls[0]["x0"], known)                 # noised along the flow path (t = 1 -> noise)
    with pytest.raises(ValueError, match="frame_cond"):
        sample_flow(m1, (1, 4, 5, 2, 3), emb, mask, steps=2, known=known, clean_cond=True)


def test_sample_flow_anchors_and_cfg_pass_the_mask_to_both_branches():
    m = spy_model(frame_cond=True)
    emb, mask = torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)
    anchors = ([0, 4], torch.randn(1, 4, 2, 2, 3))
    out = sample_flow(m, (1, 4, 5, 2, 3), emb, mask, steps=2, cfg_scale=3.0, seed=2, init=torch.randn(1, 4, 5, 2, 3),
                      strength=0.6, anchors=anchors)
    torch.testing.assert_close(out[:, :, [0, 4]], anchors[1])
    assert len(SpyDiT.calls) >= 2 and all(c["mask"][0].tolist() == [True, False, False, False, True] for c in SpyDiT.calls)


def test_pipeline_conditioning_modes_and_fps(tmp_path):
    cfg = tiny_cfg(model={"frame_cond": True, "fps_cond": True})
    write_hd_bundle(tmp_path / "v2", cfg, seed=4)
    pipe = load_hd_bundle(tmp_path / "v2", "cpu")
    assert pipe.dit.cfg.version == DIT_VERSION_2
    r = pipe.generate("a red ball bounces", seed=1, steps=2)
    assert r.meta["conditioning"] is None
    r2 = pipe.continue_video(r.state, seed=2, steps=2)
    assert r2.meta["conditioning"] == "mask" and r2.frames.shape[0] > 0
    # explicit effective fps is accepted and changes the (non-degenerate) model's output
    a = pipe.generate("a red ball bounces", seed=1, steps=2, effective_fps=6)
    b = pipe.generate("a red ball bounces", seed=1, steps=2, effective_fps=24)
    assert not np.array_equal(a.frames, b.frames)
    # refine anchors first / last frame with the mask as well
    metas = []
    frames = r.frames
    meta = pipe.refine("a red ball bounces", frames, 96, 64, seed=3, steps=2, strength=0.5, sink=metas.append)
    assert meta["conditioning"] == "mask" and sum(f.shape[0] for f in metas) == frames.shape[0]


def test_temporal_factor_4_bundle_generates_and_continues(tmp_path):
    cfg = tiny_cfg()
    del cfg["vae"]["temporal_down"]
    cfg["vae"]["temporal_factor"] = 4
    cfg["inference"]["shift_base_tokens"] = 16
    write_hd_bundle(tmp_path / "t4", cfg, seed=2)
    pipe = load_hd_bundle(tmp_path / "t4", "cpu")
    assert pipe.vae.cfg.temporal_factor == 4
    r = pipe.generate("a blue cat", seed=1, steps=2, frames=9)
    assert r.frames.shape[0] == 9 and r.latents.shape[1] == 3
    r2 = pipe.continue_video(r.state, seed=2, steps=2, frames=9)
    assert r2.frames.shape[0] == 9 - 1 - 4          # K=2 known latent frames overlap 1 + 4 frames
    with pytest.raises(ValueError, match="1 \\+ 4k"):
        pipe.generate("x", frames=10, steps=1)
