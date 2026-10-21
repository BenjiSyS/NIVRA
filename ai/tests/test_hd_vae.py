"""Nivra HC-VAE: geometry, causality, chunked/tiled equivalence, training step, save/load."""
import pytest
import torch

from nivra.compression.hc_vae import (HCVAE, HCVAEConfig, hcvae_loss, load_hcvae, save_hcvae, shape_trace,
                                      tile_ranges)
from nivra.models.hd_config import load_hd_config, vae_config

TINY = dict(latent_channels=8, channels=(8, 8, 16, 16, 16, 16), n_res=1, n_res_dec=1)


@pytest.fixture(scope="module")
def vae():
    torch.manual_seed(0)
    return HCVAE(HCVAEConfig(**TINY)).eval()


def test_720p_shape_trace_on_meta():
    cfg = vae_config(load_hd_config("nivra_medium"))
    assert (cfg.spatial_factor, cfg.temporal_factor, cfg.latent_channels) == (32, 8, 128)
    tr = shape_trace(cfg, 241, 736, 1280)
    assert tr["latent"] == (1, 128, 31, 23, 40)          # 28,520 tokens
    assert tr["decoded"] == (1, 3, 241, 736, 1280)
    assert tr["decode_flops"] > 0 and tr["params"] > 0


def test_geometry_rules():
    c = HCVAEConfig()
    assert c.latent_shape(241, 736, 1280) == (128, 31, 23, 40)
    assert c.frames_for_latent(31) == 241
    with pytest.raises(ValueError):
        c.latent_shape(240, 736, 1280)
    with pytest.raises(ValueError):
        c.latent_shape(241, 720, 1280)
    with pytest.raises(ValueError):
        HCVAEConfig.from_dict({"bogus": 1})


def test_forward_backward_small():
    torch.manual_seed(1)
    v = HCVAE(HCVAEConfig(**TINY))
    x = torch.rand(2, 3, 9, 64, 64) * 2 - 1
    recon, m, lv = v(x)
    assert recon.shape == x.shape and m.shape == (2, 8, 2, 2, 2)
    out = hcvae_loss(x, recon, m, lv)
    out["loss"].backward()
    grads = [p.grad for p in v.parameters() if p.grad is not None]
    assert len(grads) == len(list(v.parameters()))
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(torch.isfinite(t) for t in out.values())


def test_causality_future_frames_do_not_change_past_latents(vae):
    torch.manual_seed(2)
    x = torch.rand(1, 3, 25, 64, 64) * 2 - 1
    with torch.no_grad():
        z = vae.encode(x)
        x2 = x.clone()
        x2[:, :, 9:] = torch.rand_like(x2[:, :, 9:])     # frames 9.. belong to latent frames 2..
        z2 = vae.encode(x2)
        assert torch.equal(z[:, :, :2], z2[:, :, :2])
        assert not torch.allclose(z[:, :, 2:], z2[:, :, 2:])
        # decoder causality: changing latent frame 2 leaves the first 9 decoded frames untouched
        y = vae.decode(z)
        z3 = z.clone()
        z3[:, :, 2:] += 1.0
        y3 = vae.decode(z3)
        assert torch.equal(y[:, :, :9], y3[:, :, :9])
        assert y.shape[2] == 25


def test_chunked_and_tiled_decode_equal_full(vae):
    torch.manual_seed(3)
    z = torch.randn(1, 8, 3, 3, 4)
    with torch.no_grad():
        full = vae.decode(z)
        for t_chunk, tile, ov in [(1, 0, 0), (2, 0, 0), (1, 2, 1), (2, 3, 1)]:
            y = vae.decode_tiled(z, t_chunk=t_chunk, tile=tile, overlap=ov)
            assert y.shape == full.shape == (1, 3, 17, 96, 128)
            assert (y - full).abs().max() < 1e-4, (t_chunk, tile, ov)
        # margins below the receptive radius are approximate; the error must shrink as the margin grows
        errs = [float((vae.decode_tiled(z, t_chunk=1, tile=2, overlap=1, margin=m) - full).abs().mean())
                for m in (0, 1, vae.decoder.radius)]
        assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-5


def test_chunked_and_tiled_encode_equal_full(vae):
    torch.manual_seed(4)
    x = torch.rand(1, 3, 17, 96, 128) * 2 - 1
    with torch.no_grad():
        full = vae.encode(x)
        for t_chunk, tile in [(1, 0), (2, 2), (1, 3)]:
            z = vae.encode_tiled(x, t_chunk=t_chunk, tile=tile, overlap=1)
            assert z.shape == full.shape
            assert (z - full).abs().max() < 1e-4


def test_tile_ranges_cover():
    for n, t, o in [(23, 24, 2), (40, 24, 2), (40, 7, 3), (5, 2, 1)]:
        r = tile_ranges(n, t, o)
        assert r[0][0] == 0 and r[-1][1] == n
        for (a, b), (c, d) in zip(r, r[1:]):
            assert c < b and b - c >= min(o, t)        # consecutive tiles overlap by >= overlap


def test_save_load_manifest(tmp_path):
    torch.manual_seed(5)
    v = HCVAE(HCVAEConfig(**TINY))
    v.cfg.latent_scale = 0.5
    man = save_hcvae(v, tmp_path / "v")
    assert man["causal"] and man["spatial_factor"] == 32 and man["temporal_factor"] == 8
    v2 = load_hcvae(tmp_path / "v")
    assert v2.cfg.latent_scale == 0.5
    for a, b in zip(v.state_dict().values(), v2.state_dict().values()):
        assert torch.equal(a, b)
    (tmp_path / "v" / "hc_vae.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_hcvae(tmp_path / "v")
