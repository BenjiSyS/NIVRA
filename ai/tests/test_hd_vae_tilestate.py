"""Regression: `vae_tile=8` crashed (exit 139 / access violation) in HCVAE.decode_stream at 2560x1440.

Root cause (measured): every spatial tile keeps its own causal-cache state between time chunks. At 2560x1440 with
tile=8 there are 8 x 13 = 104 tiles, each holding ~0.1 GiB (tiny VAE) to ~2 GiB (Medium) of cache, and each cache
was a *view* that also kept the whole padded chunk input alive. Memory ran out inside oneDNN, which does not
handle the failed allocation. Fixed by (1) cloning the cache to just the frames it needs and (2) estimating the
total tile state up front and raising StreamStateTooLarge instead of crashing.
"""
import pytest
import torch

from nivra.compression.hc_vae import CausalConv3d, HCVAE, StreamStateTooLarge, tile_ranges
from nivra.models.hd_config import build_hcvae, load_hd_config


@pytest.fixture(scope="module")
def vae():
    return build_hcvae(load_hd_config("nivra_hd_tiny")).eval()


def test_tile_count_at_2560x1440_is_the_cause():
    assert len(tile_ranges(45, 8, 2)) * len(tile_ranges(80, 8, 2)) == 104


def test_guard_raises_instead_of_crashing_at_2560x1440(vae):
    z = torch.randn(1, vae.cfg.latent_channels, 2, 1440 // 32, 2560 // 32)
    with pytest.raises(StreamStateTooLarge, match="tiles"):
        next(iter(vae.decode_stream(z, 1, 8, 2, max_state_bytes=1 << 30)))


def test_estimate_matches_measured_state(vae):
    z = torch.randn(1, vae.cfg.latent_channels, 2, 6, 6)
    est = vae.stream_state_bytes_per_tile("decoder", 6, 6, torch.float32)
    with torch.no_grad():
        vae.decoder.set_streaming(True)
        for i in range(2):                                        # 2nd chunk: caches are used and re-stored
            vae.decoder(z[:, :, i:i + 1])
        got = sum(c.numel() * c.element_size() for c in vae.decoder.get_stream_state() if isinstance(c, torch.Tensor))
        vae.decoder.set_streaming(False)
    assert est == got and est > 0


def test_cache_does_not_retain_the_chunk_input(vae):
    z = torch.randn(1, vae.cfg.latent_channels, 3, 4, 4)
    with torch.no_grad():
        vae.decoder.set_streaming(True)
        for i in range(3):
            vae.decoder(z[:, :, i:i + 1])
        caches = [m.cache for m in vae.decoder.modules() if isinstance(m, CausalConv3d) and m.cache is not None and m.cache.numel()]
        vae.decoder.set_streaming(False)
    assert caches
    for c in caches:                   # a view would keep (kt-1+chunk) frames of storage alive for kt-1 frames of cache
        assert c.untyped_storage().nbytes() == c.numel() * c.element_size()


def test_tiled_decode_still_equals_untiled(vae):
    z = torch.randn(1, vae.cfg.latent_channels, 3, 12, 12)
    with torch.no_grad():
        full = vae.decode(z)
        tiled = vae.decode_tiled(z, t_chunk=1, tile=6, overlap=2, max_state_bytes=2 << 30)
    assert tiled.shape == full.shape and (tiled - full).abs().max() < 1e-4


def test_small_budget_via_decode_tiled_and_encode(vae):
    x = torch.randn(1, 3, 9, 128, 128)
    with pytest.raises(StreamStateTooLarge):
        vae.encode_tiled(x, t_chunk=1, tile=1, overlap=0, margin=0, max_state_bytes=1)
