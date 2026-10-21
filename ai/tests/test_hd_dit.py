"""Nivra DiT: exact Medium/Large parameter counts (meta device), forward/backward, memory features."""
import pytest
import torch

from nivra.models.dit import (DiTConfig, NivraDiT, QuantLinear, count_params_meta, quantize_int8_, rf_loss,
                              resolution_shift, sample_t)
from nivra.models.hd_config import dit_config, load_hd_config
from nivra.models.rope import rope_axes

MEDIUM_PARAMS = 932_333_696
LARGE_PARAMS = 2_775_536_768


def tiny(**kw):
    torch.manual_seed(0)
    c = dict(dim=32, depth=2, heads=2, mlp_ratio=2.0, latent_channels=8, text_dim=16)
    c.update(kw)
    m = NivraDiT(DiTConfig(**c))
    with torch.no_grad():          # break the zero init so every path contributes to the output
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def inputs(B=2, T=3, H=2, W=3, L=5):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, 8, T, H, W, generator=g)
    t = torch.rand(B, generator=g)
    e = torch.randn(B, L, 16, generator=g)
    mk = torch.ones(B, L, dtype=torch.bool)
    mk[-1, 3:] = False
    return x, t, e, mk


def test_param_counts_meta():
    med = count_params_meta(dit_config(load_hd_config("nivra_medium")))
    lar = count_params_meta(dit_config(load_hd_config("nivra_large")))
    assert med["total"] == MEDIUM_PARAMS
    assert lar["total"] == LARGE_PARAMS
    assert rope_axes(64) == (24, 20, 20) and rope_axes(72) == (24, 24, 24)


def test_forward_shapes_variable_grid():
    m = tiny()
    for T, H, W in [(1, 1, 1), (3, 2, 3), (4, 5, 2)]:
        x, t, e, mk = inputs(T=T, H=H, W=W)
        assert m(x, t, e, mk).shape == x.shape


def test_patch_122():
    m = tiny(patch=(1, 2, 2))
    x, t, e, mk = inputs(T=2, H=4, W=6)
    assert m(x, t, e, mk).shape == x.shape


def test_zero_init_outputs_zero():
    m = NivraDiT(DiTConfig(dim=32, depth=2, heads=2, latent_channels=8, text_dim=16))
    x, t, e, mk = inputs()
    assert m(x, t, e, mk).abs().max() == 0


def test_backward_and_param_change():
    m = tiny()
    x, t, e, mk = inputs()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = rf_loss(m, x, e, mk, t, torch.randn_like(x), torch.tensor([True, False]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    opt.step()
    changed = [k for k, v in m.state_dict().items() if not torch.equal(v, before[k])]
    assert len(changed) == len(before)


def test_checkpointing_and_chunking_match():
    m = tiny()
    x, t, e, mk = inputs(T=3, H=3, W=3)
    y = m(x, t, e, mk)
    y.square().mean().backward()
    g_ref = [None if p.grad is None else p.grad.clone() for p in m.parameters()]   # null_ctx unused here
    m.zero_grad()
    m.set_memory(grad_checkpoint=True, attn_chunk=7, mlp_chunk=5)
    m.train()
    y2 = m(x, t, e, mk)
    y2.square().mean().backward()
    assert (y - y2).abs().max() < 1e-5
    for a, p in zip(g_ref, m.parameters()):
        assert (a is None) == (p.grad is None)
        if a is not None:
            assert (a - p.grad).abs().max() < 1e-5
    with torch.no_grad():                     # in-place chunked MLP path
        assert (m(x, t, e, mk) - y).abs().max() < 1e-5


def test_rope_relative_time_offset_invariance():
    m = tiny()
    x, t, e, mk = inputs()
    with torch.no_grad():
        assert (m(x, t, e, mk, t_offset=0) - m(x, t, e, mk, t_offset=7)).abs().max() < 1e-4


def test_caption_dropout_is_null_text_and_dim_check():
    m = tiny()
    x, t, e, mk = inputs()
    with torch.no_grad():
        u = m(x, t, None, None)
        d = m(x, t, e, mk, drop_text=torch.tensor([True, True]))
        c = m(x, t, e, mk)
    assert (u - d).abs().max() < 1e-6
    assert (u - c).abs().max() > 1e-4
    with pytest.raises(ValueError):
        m(x, t, torch.randn(2, 5, 12), mk)


def test_offload_matches_on_cpu():
    m = tiny().eval()
    x, t, e, mk = inputs()
    with torch.no_grad():
        ref = m(x, t, e, mk)
        m.enable_offload("cpu")
        out = m(x, t, e, mk)
    assert (ref - out).abs().max() < 1e-6
    with pytest.raises(RuntimeError):
        m(x, t, e, mk)                         # offload is inference-only


def test_int8_weight_only_quantisation():
    m = tiny().eval()
    x, t, e, mk = inputs()
    with torch.no_grad():
        ref = m(x, t, e, mk)
        n = quantize_int8_(m)
        q = m(x, t, e, mk)
    assert n == 2 * 7 and isinstance(m.blocks[0].fc1, QuantLinear)
    assert m.blocks[0].fc1.qweight.dtype == torch.int8
    assert (q - ref).norm() / ref.norm() < 0.03
    assert m.num_params() == sum(p.numel() for p in tiny().parameters())


def test_timestep_sampling_and_shift():
    assert resolution_shift(4096) == 1.0 and resolution_shift(28520) > 2.5
    t1 = sample_t(4000, 1.0, generator=torch.Generator().manual_seed(0))
    t3 = sample_t(4000, 3.0, generator=torch.Generator().manual_seed(0))
    assert 0 < t1.min() and t1.max() < 1 and abs(float(t1.mean()) - 0.5) < 0.03
    assert float(t3.mean()) > float(t1.mean()) + 0.1
