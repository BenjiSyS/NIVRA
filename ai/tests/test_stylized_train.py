"""Stylized specialization: HD config fields + backward compat, HC-VAE edge loss, trainer caption styling and
deterministic style-tag dropout."""
import copy
import json

import numpy as np
import pytest
import torch

from nivra.compression.hc_vae import edge_loss, hcvae_loss
from nivra.inference.hd import StubTextEncoder
from nivra.models.hd_config import (HDConfigError, hd_config_hash, load_hd_config, vae_config, vae_edge_weight)
from nivra.training.hd_trainer import style_dropped, train_hd, training_caption

from test_hd_train import cfg_with, write_dataset

STYLE_LIST = ["anime", "3d animation", "cartoon", "2d animation", "cel-shaded 3d", "pixel art", "claymation",
              "painterly", "realistic"]


# ------------------------------------------------------------------ configs
@pytest.mark.parametrize("name", ["nivra_hd_tiny", "nivra_medium", "nivra_large"])
def test_shipped_configs_are_stylized(name):
    cfg = load_hd_config(name)
    assert cfg["domain"] == "stylized" and cfg["default_style"] == "anime"
    assert cfg["styles"] == STYLE_LIST and cfg["styles"][-1] == "realistic"
    assert cfg["train"]["style_dropout"] == 0.1 and vae_edge_weight(cfg) == 0.5
    vae_config(cfg)                                    # edge_weight is a loss option, not an architecture field


def test_old_config_without_new_fields_still_loads():
    cfg = load_hd_config("nivra_hd_tiny")
    old = copy.deepcopy(cfg)
    for k in ("domain", "styles", "default_style"):
        old.pop(k)
    old["vae"].pop("edge_weight")
    old["train"].pop("style_dropout")
    loaded = load_hd_config(old)
    assert loaded["domain"] == "general" and loaded["default_style"] is None and loaded["styles"] == []
    assert vae_edge_weight(loaded) == 0.0
    assert hd_config_hash(loaded) == hd_config_hash(cfg)          # loss options / domain are not architecture


def test_config_validation():
    base = load_hd_config("nivra_hd_tiny")

    def bad(**kw):
        c = copy.deepcopy(base)
        for k, v in kw.items():
            if k.startswith("vae_"):
                c["vae"][k[4:]] = v
            elif k.startswith("train_"):
                c["train"][k[6:]] = v
            else:
                c[k] = v
        return c

    with pytest.raises(HDConfigError, match="domain"):
        load_hd_config(bad(domain="photo"))
    with pytest.raises(HDConfigError, match="canonical"):
        load_hd_config(bad(styles=["anime", "manga-style"]))
    with pytest.raises(HDConfigError, match="not in styles"):
        load_hd_config(bad(styles=["anime", "cartoon"], default_style="pixel art"))
    with pytest.raises(HDConfigError, match="not a known style"):
        load_hd_config(bad(default_style="banana"))
    with pytest.raises(HDConfigError, match="non-empty"):
        load_hd_config(bad(styles=[]))                                # stylized domain needs styles
    with pytest.raises(HDConfigError, match="edge_weight"):
        load_hd_config(bad(vae_edge_weight=-1))
    with pytest.raises(HDConfigError, match="style_dropout"):
        load_hd_config(bad(train_style_dropout=1.5))
    ok = load_hd_config(bad(default_style="Manga-Style"))              # alias normalised to the canonical tag
    assert ok["default_style"] == "anime"
    with pytest.raises(HDConfigError, match="unknown HD config keys"):
        load_hd_config(bad(flavour="x"))


# ------------------------------------------------------------------ edge loss
def _vid(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, 5, 16, 16, generator=g).clamp(-1, 1)


def test_edge_loss_zero_for_identical_positive_otherwise():
    x = _vid()
    assert edge_loss(x, x.clone()).item() == 0.0
    assert edge_loss(x, _vid(1)).item() > 0
    blurred = torch.nn.functional.avg_pool3d(x, (1, 3, 3), 1, (0, 1, 1))
    assert edge_loss(x, blurred).item() > 0                           # blurred outlines are penalised
    # a constant offset leaves gradients untouched
    assert edge_loss(x, x + 0.3).item() == pytest.approx(0.0, abs=1e-6)


def test_hcvae_loss_edge_weight():
    x, r = _vid(0), _vid(1)
    mean, logvar = torch.zeros(1, 4, 2, 2, 2), torch.zeros(1, 4, 2, 2, 2)
    old = hcvae_loss(x, r, mean, logvar)
    zero = hcvae_loss(x, r, mean, logvar, edge_weight=0.0)
    assert torch.equal(old["loss"], zero["loss"]) and set(old) == set(zero) == {"loss", "l1", "kl", "perc", "tdiff"}
    w = hcvae_loss(x, r, mean, logvar, edge_weight=0.5)
    assert w["loss"].item() == pytest.approx(old["loss"].item() + 0.5 * edge_loss(x, r).item(), rel=1e-5)
    assert w["edge"].item() == pytest.approx(edge_loss(x, r).item(), rel=1e-6) and w["edge"].item() > 0
    same = hcvae_loss(x, x.clone(), mean, logvar, edge_weight=0.5)
    assert same["edge"].item() == 0.0
    r.requires_grad_(True)
    hcvae_loss(x, r, mean, logvar, edge_weight=0.5)["loss"].backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()


def test_vae_training_uses_edge_weight(tmp_path):
    data = write_dataset(tmp_path / "d", n=1)
    stage = [{"name": "s", "height": 64, "width": 64, "frames": 9, "steps": 2}]
    on = train_hd(cfg_with(stage, batch_size=1, save_every=100), data, tmp_path / "on", target="vae", log=lambda *_: None)
    c0 = cfg_with(stage, batch_size=1, save_every=100)
    c0["vae"]["edge_weight"] = 0
    off = train_hd(c0, data, tmp_path / "off", target="vae", log=lambda *_: None)
    assert "edge" in on["history"][0] and "edge" not in off["history"][0]


# ------------------------------------------------------------------ captions + style dropout
def test_training_caption_rules():
    row = {"id": "c1", "caption": "A girl runs.", "style": "anime"}
    assert training_caption(row, 0.0) == "anime style. A girl runs."
    assert training_caption(row, 1.0) == "A girl runs."
    assert training_caption({"id": "c2", "caption": "A girl runs.", "style": "unspecified"}, 0.0) == "A girl runs."
    assert training_caption({"id": "c3", "caption": "A girl runs."}, 0.0) == "A girl runs."          # pre-style manifests
    assert training_caption({"id": "c4", "caption": "A cartoon dog.", "style": "anime"}, 0.0) == "A cartoon dog."
    assert training_caption({"id": "c5", "caption": "A girl runs.", "style": "mystery"}, 0.0) == "A girl runs."


def test_style_dropout_deterministic_and_calibrated():
    ids = [f"clip-{i}" for i in range(4000)]
    a = [style_dropped(7, 0, i, 0.1) for i in ids]
    assert a == [style_dropped(7, 0, i, 0.1) for i in ids]
    assert 0.07 < np.mean(a) < 0.13
    assert a != [style_dropped(8, 0, i, 0.1) for i in ids]          # seed changes the pattern
    assert a != [style_dropped(7, 1, i, 0.1) for i in ids]          # so does the epoch
    assert not any(style_dropped(7, 0, i, 0.0) for i in ids) and all(style_dropped(7, 0, i, 1.0) for i in ids)


class RecordingEncoder(StubTextEncoder):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []

    def encode(self, prompts):
        self.seen.extend(prompts)
        return super().encode(prompts)


def _styled_latent_dataset(root, styles):
    rng = np.random.default_rng(0)
    lat = rng.standard_normal((len(styles), 8, 3, 2, 2)).astype(np.float16)
    write_dataset(root, key="latents", latents=lat)
    mf = root / "manifest.jsonl"
    rows = [json.loads(l) for l in mf.read_text(encoding="utf-8").splitlines()]
    for r, s in zip(rows, styles):
        r["style"] = s
    mf.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


STAGE = [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 6}]


def _captions(tmp_path, tag, styles, seed=0, **train):
    data = _styled_latent_dataset(tmp_path / f"d_{tag}", styles)
    enc = RecordingEncoder(dim=16)
    cfg = cfg_with(STAGE, batch_size=2, save_every=2, **train)
    res = train_hd(cfg, data, tmp_path / f"run_{tag}", text_encoder=enc, seed=seed, log=lambda *_: None)
    return enc.seen, res


def test_trainer_applies_style_to_captions(tmp_path):
    seen, res = _captions(tmp_path, "all", ["anime", "cartoon", "pixel art", "anime"], style_dropout=0.0)
    assert len(seen) == 12 and all(c.split(" style. ")[0] in ("anime", "cartoon", "pixel art") for c in seen)
    assert res["styles"] == {"anime": 2, "cartoon": 1, "pixel art": 1}
    seen, _ = _captions(tmp_path, "unknown", ["unspecified"] * 4, style_dropout=0.0)
    assert all(" style. " not in c for c in seen)
    seen, _ = _captions(tmp_path, "drop", ["anime"] * 4, style_dropout=1.0)
    assert all(" style. " not in c for c in seen)


def test_trainer_style_dropout_seeded_and_partial(tmp_path):
    styles = ["anime", "cartoon", "pixel art", "anime"] * 3
    stage = [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 40}]
    runs = []
    for tag, seed in (("x1", 3), ("x2", 3), ("y", 4)):
        data = _styled_latent_dataset(tmp_path / f"d_{tag}", styles)
        enc = RecordingEncoder(dim=16)
        train_hd(cfg_with(stage, batch_size=2, save_every=1000, style_dropout=0.5), data, tmp_path / f"r_{tag}",
                 text_encoder=enc, seed=seed, log=lambda *_: None)
        runs.append(enc.seen)
    tagged = [" style. " in c for c in runs[0]]
    assert runs[0] == runs[1], "same seed -> identical captions"
    assert 0.3 < np.mean(tagged) < 0.7, "dropout 0.5 keeps roughly half of the tags"
    assert runs[0] != runs[2], "a different seed gives a different dropout pattern"


def test_style_dropout_exact_across_resume(tmp_path):
    styles = ["anime", "cartoon", "pixel art", "anime"] * 3
    stage = [{"name": "s", "height": 64, "width": 64, "frames": 17, "steps": 8}]
    cfg = cfg_with(stage, batch_size=2, save_every=2, style_dropout=0.5, keep_last=5)
    full_enc = RecordingEncoder(dim=16)
    train_hd(cfg, _styled_latent_dataset(tmp_path / "d1", styles), tmp_path / "full", text_encoder=full_enc,
             log=lambda *_: None)
    d2 = _styled_latent_dataset(tmp_path / "d2", styles)
    e1, e2 = RecordingEncoder(dim=16), RecordingEncoder(dim=16)
    train_hd(cfg, d2, tmp_path / "part", text_encoder=e1, total_steps=4, log=lambda *_: None)
    train_hd(cfg, d2, tmp_path / "part", text_encoder=e2, log=lambda *_: None)
    assert e1.seen + e2.seen == full_enc.seen
