import hashlib
import json

import pytest
import torch

from nivra.text.bpe import BPETokenizer, FIRST_MERGE_ID, pretokenize
from nivra.text.interface import TextEncoder
from nivra.text.nivra_text import (ContrastiveHead, NivraTextConfig, NivraTextEncoder, count_params_for, info_nce,
                                   masked_mean_pool, mask_tokens, pretrain_mlm)
from nivra.text.registry import get_text_encoder

CORPUS = [
    "a red ball rolls across the green grass\n", "a blue cube spins slowly in the dark\n",
    "en röd boll rullar över det gröna gräset\n", "en liten båt seglar på sjön vid solnedgången\n",
    "the camera pans over a snowy mountain at dawn\n", "kameran panorerar över ett snöigt berg i gryningen\n",
] * 40
TRICKY = ["", " ", "hello world", "  leading and trailing  ", "line\nbreak\r\n\ttab", "Åsa går över ån på ön — ä ö å",
          "emoji 😀🎬👨‍👩‍👧 and 中文 and العربية", "snake_case_name 12345 3.14", "\u0000​﻿", "x" * 500]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=400)


def test_roundtrip_any_utf8(tok):
    for s in TRICKY:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, repr(s)
        assert tok.decode_bytes(ids) == s.encode("utf-8")
    assert pretokenize("a  b") == ["a", " ", " b"]


def test_untrained_tokenizer_is_pure_bytes():
    t = BPETokenizer()
    assert t.vocab_size == FIRST_MERGE_ID and t.decode(t.encode("åäö😀")) == "åäö😀"


def test_deterministic_training_and_save_load(tok, tmp_path):
    t2 = BPETokenizer.train(CORPUS, vocab_size=400)
    assert t2.merges == tok.merges and t2.sha256 == tok.sha256
    sha = tok.save(tmp_path / "tok.json")
    loaded = BPETokenizer.load(tmp_path / "tok.json")
    assert loaded.sha256 == sha and loaded.encode(TRICKY[5]) == tok.encode(TRICKY[5])
    d = json.loads((tmp_path / "tok.json").read_text())
    d["merges"][0][0] += 1
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="sha256"):
        BPETokenizer.load(tmp_path / "bad.json")


def test_merges_compress_and_swedish(tok):
    s = "en röd boll rullar över det gröna gräset"
    assert len(tok.encode(s)) < len(s.encode("utf-8")) * 0.6
    assert tok.vocab_size > FIRST_MERGE_ID


def test_bpe_unchanged_small_tokenizer_still_there():
    from nivra.text.tokenizer import default_tokenizer
    assert len(default_tokenizer()) > 0


@pytest.fixture(scope="module")
def ckpt(tok, tmp_path_factory):
    d = tmp_path_factory.mktemp("txt")
    (d / "c.txt").write_text("".join(CORPUS), encoding="utf-8")
    out = d / "ckpt"
    res = pretrain_mlm([d / "c.txt"], tok, out, size="tiny", steps=60, batch=8, seq_len=32, lr=3e-3, warmup=5,
                       log=lambda *_: None)
    return out, res


def test_mlm_loss_decreases_and_checkpoint(ckpt):
    out, res = ckpt
    h = [l for _, l in res["history"]]
    assert sum(h[-10:]) / 10 < sum(h[:5]) / 5 - 0.3
    meta = json.loads((out / "config.json").read_text())
    assert meta["tokenizer_sha256"] == BPETokenizer.load(out / "tokenizer.json").sha256
    assert meta["origin"] == "nivra-original"


def test_encoder_interface(ckpt):
    out, _ = ckpt
    enc = get_text_encoder(f"nivra:{out}")
    assert isinstance(enc, TextEncoder) and enc.id.startswith("nivra-text-tiny@")
    emb, mask = enc.encode(["en röd boll 😀", "a"])
    assert emb.dtype == torch.float32 and emb.shape[0] == 2 and emb.shape[2] == enc.dim == 64
    assert mask.dtype == torch.bool and mask.shape == emb.shape[:2]
    assert mask[0].sum() > mask[1].sum() == 3          # bos + 'a' + eos
    assert torch.all(emb[1][~mask[1]] == 0)
    long, lm = enc.encode(["word " * 500])
    assert long.shape[1] == enc.max_len
    e2, _ = enc.encode(["a"])                           # deterministic, padding-independent
    assert torch.allclose(e2[0], emb[1, : e2.shape[1]], atol=1e-5)


def test_checkpoint_tokenizer_mismatch_rejected(ckpt, tmp_path):
    import shutil
    out, _ = ckpt
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    other = BPETokenizer.train(CORPUS[:6], vocab_size=300)
    other.save(bad / "tokenizer.json")
    with pytest.raises(ValueError, match="tokenizer"):
        NivraTextEncoder.from_checkpoint(bad)


def test_base_config_is_about_110m():
    n = count_params_for(NivraTextConfig.preset("base", 32000))
    assert 100e6 < n < 125e6, n


def test_mask_tokens_never_masks_specials():
    ids = torch.tensor([[1, 50, 60, 70, 2, 0, 0, 0]])
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        inp, lab = mask_tokens(ids, 400, g, p=0.5)
        assert (lab[0, [0, 4, 5, 6, 7]] == -100).all() and (lab != -100).any()


def test_infonce_and_head():
    t = torch.randn(8, 16)
    assert info_nce(t, t.clone()) < info_nce(t, torch.randn(8, 16))
    with pytest.raises(ValueError):
        info_nce(t, torch.randn(8, 4))
    head = ContrastiveHead(64, 32, 16)
    emb, mask = torch.randn(4, 5, 64), torch.ones(4, 5, dtype=torch.bool)
    loss = head.loss(emb, mask, torch.randn(4, 32))
    loss.backward()
    assert torch.isfinite(loss) and masked_mean_pool(emb, mask).shape == (4, 64)


def test_registry_errors():
    for bad in ("nope", "foo:bar", "nivra:"):
        with pytest.raises(ValueError):
            get_text_encoder(bad)


# ------------------------------------------------------------------ external adapter
from nivra.text.external import ExternalEncoderError, LICENSE_FILE, load_external  # noqa: E402


class FakeBackend:
    dim = 24

    def __init__(self, d):
        self.d = d

    def encode(self, prompts, max_len):
        L = min(max(len(p.split()) for p in prompts) + 1, max_len)
        emb = torch.ones(len(prompts), L, self.dim)
        mask = torch.zeros(len(prompts), L, dtype=torch.bool)
        for i, p in enumerate(prompts):
            mask[i, : min(len(p.split()) + 1, L)] = True
        return emb, mask


def _make_ext(tmp_path, sha=None, lic=True):
    d = tmp_path / "ext"
    d.mkdir()
    w = d / "model.safetensors"
    w.write_bytes(b"not really weights")
    real = hashlib.sha256(w.read_bytes()).hexdigest()
    if lic:
        (d / LICENSE_FILE).write_text(json.dumps({"name": "mini-umt5", "license": "Apache-2.0",
                                                  "source_url": "https://example.invalid/mini", "sha256": sha or real}))
    return d, real


def test_external_loads_with_license(tmp_path):
    d, real = _make_ext(tmp_path)
    enc = load_external(d, backend_factory=FakeBackend)
    assert enc.id == f"external:mini-umt5@{real[:12]}"
    emb, mask = enc.encode(["a b c", "a"])
    assert emb.dtype == torch.float32 and emb.shape == (2, 4, 24) and mask.dtype == torch.bool
    m = enc.manifest_entry()
    assert m["license"] == "Apache-2.0" and m["origin"] == "external-third-party"
    assert isinstance(enc, TextEncoder)
    assert get_text_encoder(f"external:{d}", backend_factory=FakeBackend).id == enc.id


def test_external_requires_license_and_matching_hash(tmp_path):
    (tmp_path / "a").mkdir()
    d, _ = _make_ext(tmp_path / "a", lic=False)
    with pytest.raises(ExternalEncoderError, match="NIVRA_LICENSE"):
        load_external(d, backend_factory=FakeBackend)
    (tmp_path / "b").mkdir()
    d2, _ = _make_ext(tmp_path / "b", sha="0" * 64)
    with pytest.raises(ExternalEncoderError, match="does not match"):
        load_external(d2, backend_factory=FakeBackend)


def test_external_refuses_network_and_missing(tmp_path):
    for bad in ("https://huggingface.co/google/umt5-small", "\\\\server\\share\\enc", "//server/share", str(tmp_path / "nope")):
        with pytest.raises(ExternalEncoderError):
            load_external(bad, backend_factory=FakeBackend)


def test_external_clear_error_without_transformers(tmp_path):
    try:
        import transformers  # noqa: F401
        pytest.skip("transformers installed")
    except ImportError:
        pass
    d, _ = _make_ext(tmp_path)
    with pytest.raises(ExternalEncoderError, match="transformers"):
        load_external(d)
