"""umT5-base as a licensed external text encoder: fetch/convert (mocked hub), licence file, cache, encode job."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from nivra.text import cache as tcache
from nivra.text import fetch as tfetch
from nivra.text.external import LICENSE_FILE, ExternalEncoderError, hash_weights, load_external


def test_catalog_is_pinned_and_licensed():
    spec = tfetch.CATALOG["umt5-base"]
    assert spec["repo"] == "google/umt5-base" and spec["license"] == "Apache-2.0"
    assert len(spec["revision"]) == 40 and len(spec["weights"][1]) == 64 and spec["dim"] == 768
    with pytest.raises(tfetch.FetchError, match="unknown encoder"):
        tfetch.fetch("gpt-4", Path("nowhere"))


def test_encoder_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("NIVRA_TEXT_ENCODER_DIR", raising=False)
    monkeypatch.setenv("NIVRA_MODELS_DIR", str(tmp_path))
    assert tfetch.encoder_dir("umt5-base") == tmp_path / "text" / "umt5-base"
    monkeypatch.setenv("NIVRA_TEXT_ENCODER_DIR", str(tmp_path / "x"))
    assert tfetch.encoder_dir("umt5-base") == tmp_path / "x" / "umt5-base"


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    """A tiny UMT5 checkpoint in the official layout (pickle .bin + tokenizer files) behind a fake hf_hub_download."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.UMT5Config(vocab_size=64, d_model=24, d_kv=8, d_ff=32, num_layers=2, num_decoder_layers=2, num_heads=3,
                                  decoder_start_token_id=0)
    torch.manual_seed(0)
    model = transformers.UMT5ForConditionalGeneration(cfg)
    src = tmp_path / "hub"
    src.mkdir()
    torch.save(model.state_dict(), src / "pytorch_model.bin")
    cfg.save_pretrained(src)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
        (src / f).write_bytes(b"stub " + f.encode())
    sha = hashlib.sha256((src / "pytorch_model.bin").read_bytes()).hexdigest()
    spec = dict(tfetch.CATALOG["umt5-base"], weights=("pytorch_model.bin", sha, (src / "pytorch_model.bin").stat().st_size))
    monkeypatch.setitem(tfetch.CATALOG, "umt5-base", spec)
    calls = []

    def fake_download(repo, filename, revision=None, cache_dir=None, **kw):
        calls.append((repo, filename, revision))
        return str(src / filename)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return {"model": model, "cfg": cfg, "src": src, "calls": calls, "sha": sha, "transformers": transformers}


def test_fetch_converts_to_encoder_only_safetensors_with_licence(fake_hub, tmp_path):
    dest = tmp_path / "models" / "umt5-base"
    logs = []
    out = tfetch.fetch("umt5-base", dest, log=logs.append)
    assert out == dest and {c[0] for c in fake_hub["calls"]} == {"google/umt5-base"}
    assert {c[2] for c in fake_hub["calls"]} == {tfetch.CATALOG["umt5-base"]["revision"]}
    names = sorted(p.name for p in dest.iterdir())
    assert names == sorted(["NIVRA_LICENSE.json", "config.json", "model.safetensors", "special_tokens_map.json",
                            "spiece.model", "tokenizer.json", "tokenizer_config.json"])          # no pickle left behind
    lic = json.loads((dest / LICENSE_FILE).read_text())
    assert lic["name"] == "umt5-base" and lic["license"] == "Apache-2.0" and lic["source_url"] == "https://huggingface.co/google/umt5-base"
    assert lic["sha256"] == hash_weights(dest) and lic["source_file_sha256"] == fake_hub["sha"]
    # the converted weights are exactly the encoder of the original and load in UMT5EncoderModel with nothing missing
    tr = fake_hub["transformers"]
    enc, info = tr.UMT5EncoderModel.from_pretrained(str(dest), output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    ids = torch.tensor([[3, 4, 5, 1]])
    ref = fake_hub["model"].get_encoder().eval()(input_ids=ids).last_hidden_state
    assert torch.allclose(enc.eval()(input_ids=ids).last_hidden_state, ref, atol=1e-5)
    # the adapter accepts it and names it external:umt5-base@sha12 (backend injected: the stub tokenizer is not real)
    class B:
        dim = 24
        def encode(self, prompts, max_len): return torch.ones(len(prompts), 2, 24), torch.ones(len(prompts), 2, dtype=torch.bool)
    e = load_external(dest, backend_factory=lambda d: B())
    assert e.id == f"external:umt5-base@{lic['sha256'][:12]}" and e.manifest_entry()["license"] == "Apache-2.0"
    # idempotent: a second call does not download again
    n = len(fake_hub["calls"])
    tfetch.fetch("umt5-base", dest, log=logs.append)
    assert len(fake_hub["calls"]) == n and any("already installed" in l for l in logs)


def test_fetch_refuses_wrong_hash_and_leaves_nothing(fake_hub, tmp_path, monkeypatch):
    spec = dict(tfetch.CATALOG["umt5-base"], weights=("pytorch_model.bin", "0" * 64, 1))
    monkeypatch.setitem(tfetch.CATALOG, "umt5-base", spec)
    root = tmp_path / "models"
    with pytest.raises(tfetch.FetchError, match="differs from the pinned"):
        tfetch.fetch("umt5-base", root / "umt5-base", log=lambda *_: None)
    assert not (root / "umt5-base").exists() and list(root.iterdir()) == []


def test_extract_encoder_state_rules():
    sd = {"shared.weight": torch.ones(4, 2), "encoder.block.0.x": torch.zeros(2), "decoder.block.0.x": torch.zeros(2),
          "lm_head.weight": torch.zeros(4, 2), "encoder.final_layer_norm.weight": torch.ones(2)}
    out = tfetch.extract_encoder_state(sd)
    assert set(out) == {"shared.weight", "encoder.block.0.x", "encoder.final_layer_norm.weight", "encoder.embed_tokens.weight"}
    assert out["shared.weight"].data_ptr() != sd["shared.weight"].data_ptr()
    with pytest.raises(tfetch.FetchError):
        tfetch.extract_encoder_state({"decoder.x": torch.zeros(1)})


class FakeEnc:
    id = "external:fake@abc"
    max_len = 8
    dim = 6

    def __init__(self):
        self.seen = []

    def encode(self, prompts):
        self.seen.extend(prompts)
        L = max(len(p.split()) for p in prompts) + 1
        emb = torch.zeros(len(prompts), L, self.dim)
        mask = torch.zeros(len(prompts), L, dtype=torch.bool)
        for i, p in enumerate(prompts):
            n = len(p.split()) + 1
            emb[i, :n] = (torch.arange(n * self.dim, dtype=torch.float32).view(n, self.dim) * 1.37 + len(p)) / 7.0
            mask[i, :n] = True
        return emb, mask


def test_cache_matches_uncached_and_encodes_each_text_once(tmp_path):
    from nivra.distributed.hd.encode_job import encode_texts
    texts = ["a red ball", "b", "a red ball", "", "one two three four"]
    enc = FakeEnc()
    c = tcache.EmbeddingCache(tmp_path / "c", enc.id, enc.max_len)
    e1, m1 = encode_texts(enc, texts, cache=c)
    assert sorted(set(enc.seen)) == sorted(set(texts)) and len(enc.seen) == 4 and (c.hits, c.misses) == (0, 4)
    ref_e, ref_m = encode_texts(FakeEnc(), texts)
    assert torch.equal(m1, ref_m) and torch.equal(e1, ref_e.half().float())          # fp16 rounding is the shard's anyway
    enc2 = FakeEnc()
    c2 = tcache.EmbeddingCache(tmp_path / "c", enc.id, enc.max_len)
    e2, m2 = encode_texts(enc2, texts, cache=c2)
    assert enc2.seen == [] and c2.hits == 5 and torch.equal(e1, e2) and torch.equal(m1, m2)
    # a different encoder id or max_len never shares entries; a damaged entry is recomputed
    assert tcache.EmbeddingCache(tmp_path / "c", "external:other@1", 8).get("b") is None
    assert tcache.EmbeddingCache(tmp_path / "c", enc.id, 16).get("b") is None
    c.path("b").write_bytes(b"junk")
    enc3 = FakeEnc()
    e3, _ = encode_texts(enc3, ["b"], cache=tcache.EmbeddingCache(tmp_path / "c", enc.id, enc.max_len))
    assert enc3.seen == ["b"] and e3.shape[1] == 2


def test_encode_shard_works_with_an_external_encoder_and_cache(tmp_path):
    """No `.tokenizer`, text_dim != 16: the latent shard still binds (tokenizer_sha256 None) and trains the DiT."""
    from nivra.distributed.hd.encode_job import encode_shard, load_vae_files
    from nivra.distributed.hd.latent_shard import LatentShard
    from nivra.inference.hd_bundle import write_hd_bundle
    from nivra.models.hd_config import load_hd_config
    cfg = json.loads(json.dumps(load_hd_config("nivra_hd_tiny")))
    cfg["name"], cfg["model"]["text_dim"] = "tiny_ext", FakeEnc.dim
    b = tmp_path / "b"
    man = write_hd_bundle(b, cfg, text_encoder="stub", seed=3, dtype=torch.float32)
    vae = load_vae_files(b / "vae.safetensors", b / "model_version.json")
    vids = torch.randint(0, 255, (3, 9, 40, 72, 3), dtype=torch.uint8)
    rows = [{"id": f"c{i}", "caption": f"a red ball {i}", "style": "anime", "lang": "en", "effective_fps": 8.0} for i in range(3)]
    stage = {"h": 32, "w": 64, "frames": 9, "fps": 8}
    enc = FakeEnc()
    cache = tcache.EmbeddingCache(tmp_path / "cache", enc.id, enc.max_len)
    s = encode_shard(None, tmp_path / "l.safetensors", vae=vae, enc=enc, stage=stage, parent_sha="0" * 64, vae_version="v" * 64,
                     text_encoder_id=enc.id, style_dropout=0.0, videos_override=vids, rows_override=rows, text_cache=cache)
    sh = LatentShard(tmp_path / "l.safetensors")
    assert sh.meta["text_encoder_id"] == enc.id and sh.meta["tokenizer_sha256"] is None and sh.rows[0]["effective_fps"] == 8.0
    assert sh.null()[0].shape[-1] == 6 and s["clips"] == 3 and cache.misses == 4
    from nivra.distributed.hd.train_job import hd_train_job
    from nivra.distributed.hd.stream_io import sha256_file
    tp = {"update_id": "u", "base_version_id": man["version_id"], "stage": stage, "inner_steps": 3, "batch_size": 2, "lr": 1e-3,
          "warmup_steps": 1, "seed": 1, "hd_config_sha": man["hd_config_sha"], "round": 0, "round_batch": 0, "run_id": "r",
          "target": "dit", "delta_dtype": "float16", "shards": [{"artifact_id": "a", "sha256": sha256_file(tmp_path / "l.safetensors")}]}
    r = hd_train_job({"base_train_weights": str(b / "weights.safetensors"), "base_manifest": str(b / "model_version.json"),
                      "data_shard_000": str(tmp_path / "l.safetensors")}, tp, tmp_path / "t")
    assert r["loss_last"] == r["loss_last"]


def test_encode_job_external_spec_and_missing_inputs(tmp_path, monkeypatch):
    from nivra.distributed.hd import encode_job
    d = tmp_path / "models" / "umt5-base"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    (d / LICENSE_FILE).write_text(json.dumps({"name": "umt5-base", "license": "Apache-2.0", "source_url": "https://x", "sha256": sha}))
    monkeypatch.setenv("NIVRA_TEXT_ENCODER_DIR", str(tmp_path / "models"))

    class B:
        dim = 6
        def encode(self, prompts, max_len): return torch.ones(len(prompts), 2, 6), torch.ones(len(prompts), 2, dtype=torch.bool)
    enc, eid = encode_job.load_external_encoder("external:umt5-base", backend_factory=lambda p: B())
    assert eid == f"external:umt5-base@{sha[:12]}" and enc.dim == 6
    with pytest.raises(encode_job.EncodeInputError, match="bad text_encoder_spec"):
        encode_job.load_external_encoder("nivra:whatever")
    with pytest.raises(encode_job.EncodeInputError, match="cannot load"):
        encode_job.load_external_encoder("external:not-installed")


UMT5 = tfetch.encoder_dir("umt5-base")


@pytest.mark.skipif(not (UMT5 / LICENSE_FILE).is_file(), reason="umt5-base not installed (nivra text-encoder-fetch --name umt5-base)")
def test_real_umt5_base(tmp_path):
    enc = load_external(UMT5, max_len=32)
    assert enc.dim == 768 and enc.id.startswith("external:umt5-base@")
    emb, mask = enc.encode(["a red ball bounces", "en röd boll studsar över gräset"])
    assert emb.shape[0] == 2 and emb.shape[2] == 768 and bool(torch.isfinite(emb.half()).all()) and mask.sum(1).min() > 1
    assert not torch.equal(emb[0, :3], emb[1, :3])
