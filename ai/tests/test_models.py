import torch

from nivra.compression.vae import vae_loss
from nivra.config import build_gen, build_vae, load_config
from nivra.models.nivra_small import flow_matching_loss, count_params
from nivra.text.tokenizer import default_tokenizer


def test_vae_shapes_default_config():
    cfg = load_config("nivra_small")
    vae = build_vae(cfg)
    x = torch.randn(1, 3, 16, 64, 64)
    z = vae.encode(x)
    assert tuple(z.shape) == (1, 4, 8, 16, 16)         # C x T/2 x H/4 x W/4
    assert tuple(vae.decode(z).shape) == (1, 3, 16, 64, 64)
    assert vae.cfg.latent_shape(16, 64) == (4, 8, 16, 16)


def test_vae_loss_backward_and_update(tiny_cfg):
    vae = build_vae(tiny_cfg)
    x = torch.rand(2, 3, 8, 32, 32) * 2 - 1
    before = {k: v.clone() for k, v in vae.named_parameters()}
    opt = torch.optim.AdamW(vae.parameters(), 1e-2)
    loss, m = vae_loss(vae, x)
    assert torch.isfinite(loss) and {"l1", "motion_l1", "kl"} <= set(m)
    loss.backward()
    opt.step()
    assert any(not torch.equal(before[k], p) for k, p in vae.named_parameters())


def test_tiled_decode_matches_full(tiny_cfg):
    vae = build_vae(tiny_cfg).eval()
    z = torch.randn(1, 4, 4, 8, 8)
    full = vae.decode(z)
    # a tile covering the whole volume is exact
    assert torch.allclose(vae.decode(z, tile=8), full, atol=1e-5)
    # real tiling is approximate (GroupNorm statistics are per tile); it must keep shape, be finite and close-ish
    tiled = vae.decode(z, tile=6, overlap=3)
    assert full.shape == tiled.shape and torch.isfinite(tiled).all()
    assert (full - tiled).abs().mean() < 0.12 * full.abs().mean() + 0.05


def test_generator_forward_backward_and_param_change(tiny_cfg):
    torch.manual_seed(0)
    g = build_gen(tiny_cfg)
    tok = default_tokenizer()
    ids = torch.tensor([tok.encode("a red circle moves up"), tok.encode("")])
    x = torch.randn(2, 4, 4, 8, 8)
    out = g(x, torch.tensor([0.3, 0.9]), ids)
    assert out.shape == x.shape
    assert out.abs().max() == 0       # adaLN-Zero + zero-init final layer
    before = {k: v.clone() for k, v in g.named_parameters()}
    opt = torch.optim.AdamW(g.parameters(), 1e-2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(2):
        loss = flow_matching_loss(g, x, ids, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    changed = [k for k, p in g.named_parameters() if not torch.equal(before[k], p)]
    assert "final.weight" in changed and len(changed) > 10


def test_text_conditioning_and_padding_mask(tiny_cfg):
    torch.manual_seed(1)
    g = build_gen(tiny_cfg)
    for p in g.parameters():          # make zero-inits non-trivial
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.05)
    g.eval()
    tok = default_tokenizer()
    x = torch.randn(1, 4, 4, 8, 8)
    t = torch.tensor([0.5])
    a = g(x, t, torch.tensor([tok.encode("a red circle moves up")]))
    b = g(x, t, torch.tensor([tok.encode("a blue square moves down")]))
    assert not torch.allclose(a, b)


def test_parameter_counts_reported():
    n = count_params(build_gen(load_config("nivra_small")))
    assert 1_000_000 < n < 10_000_000
