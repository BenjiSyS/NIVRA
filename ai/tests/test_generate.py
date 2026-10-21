import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from nivra.inference.bundle import load_bundle, BundleError
from nivra.inference.generate import (generate, continue_video, image_to_video, save_result, GenerationState,
                                      StateError)
from nivra.inference.sampler import sample_latents
from nivra.inference.video_io import probe_mp4, write_mp4, decode_mp4
from nivra.text.tokenizer import default_tokenizer


@pytest.fixture(scope="module")
def bundle(tiny_env):
    return load_bundle(tiny_env["bundle"])


def test_generate_mp4_frame_count_and_sidecar(bundle, tmp_path):
    res = generate(bundle, "a red circle moves up", seed=3, steps=3, cfg_scale=2.0)
    assert res.frames.shape == (8, 32, 32, 3) and res.frames.dtype == np.uint8
    meta = save_result(res, tmp_path, "clip", fps=8, upscale=4)
    info = probe_mp4(tmp_path / "clip.mp4")
    assert info["frames"] == 8 == meta["frames"]
    assert (info["width"], info["height"]) == (128, 128) and info["codec"] == "h264" and info["pix_fmt"] == "yuv420p"
    side = json.loads((tmp_path / "clip.json").read_text())
    for k in ("prompt", "seed", "model_version_id", "steps", "seconds_sampling", "seconds_decode", "hardware", "fps"):
        assert k in side
    assert side["prompt"] == "a red circle moves up" and side["seed"] == 3
    assert decode_mp4(tmp_path / "clip.mp4").shape == (8, 128, 128, 3)


def test_seed_determinism_and_tiled_path(bundle):
    a = generate(bundle, "a blue square moves down", seed=1, steps=2)
    b = generate(bundle, "a blue square moves down", seed=1, steps=2)
    c = generate(bundle, "a blue square moves down", seed=2, steps=2)
    assert np.array_equal(a.frames, b.frames) and not np.array_equal(a.frames, c.frames)
    t = generate(bundle, "a blue square moves down", seed=1, steps=2, tile=6)
    assert t.frames.shape == a.frames.shape


def test_unknown_words_are_reported(bundle):
    r = generate(bundle, "a flying banana", seed=0, steps=1, cfg_scale=1.0)
    assert r.meta["unknown_words"] == ["flying", "banana"]


def test_known_latent_frames_are_imposed(bundle):
    tok = default_tokenizer()
    ids = torch.tensor([tok.encode("a red circle moves up")])
    c = bundle.model.cfg
    shape = (1, c.latent_channels, c.latent_t, c.latent_h, c.latent_w)
    known = torch.randn(1, c.latent_channels, 2, c.latent_h, c.latent_w)
    z = sample_latents(bundle.model, ids, shape, steps=3, cfg_scale=2.0, seed=0, known=known)
    assert torch.equal(z[:, :, :2], known)
    z2 = sample_latents(bundle.model, ids, shape, steps=3, cfg_scale=2.0, seed=0)
    assert not torch.equal(z2[:, :, :2], known)


def test_state_roundtrip_and_continuation(bundle, tmp_path):
    first = generate(bundle, "a green square moves from left to right", seed=4, steps=2)
    meta = save_result(first, tmp_path, "seg1", fps=8, upscale=2)
    st = GenerationState.load(tmp_path / "seg1.state", expect=bundle)
    assert st.prompt == first.state.prompt and st.seed == 4 and st.model_version_id == bundle.version_id
    assert torch.equal(st.recent_latents, first.state.recent_latents)
    assert st.temporal_position == bundle.model.cfg.latent_t and np.array_equal(st.frames_so_far, first.frames)
    assert st.text_embedding is not None and st.text_ids == first.state.text_ids
    cont = continue_video(bundle, st, steps=2, known_latent_frames=2)
    assert cont.state.segments == 2 and cont.state.temporal_position == st.temporal_position + bundle.model.cfg.latent_t - 2
    assert cont.new_frames.shape[0] == 8 - 4 and cont.frames.shape[0] == 8 + 4
    assert np.array_equal(cont.frames[:8], first.frames)            # earlier frames are preserved
    save_result(cont, tmp_path, "seg2", fps=8, upscale=2)
    assert probe_mp4(tmp_path / "seg2.mp4")["frames"] == 12
    # continue again from the continuation state (chained)
    cont2 = continue_video(bundle, GenerationState.load(tmp_path / "seg2.state", expect=bundle), steps=2)
    assert cont2.frames.shape[0] == 16
    # incompatible model version is refused
    bad = GenerationState.load(tmp_path / "seg1.state")
    bad.model_version_id = "v999999-other"
    with pytest.raises(StateError):
        continue_video(bundle, bad, steps=1)
    with pytest.raises(StateError):
        s = tmp_path / "seg1.state" / "state.json"
        d = json.loads(s.read_text()); d["version"] = "nivra.genstate/99"; s.write_text(json.dumps(d))
        GenerationState.load(tmp_path / "seg1.state")


def test_image_to_video_keeps_first_frame_latent(bundle):
    img = np.zeros((32, 32, 3), np.uint8)
    img[10:20, 10:20] = (230, 40, 40)
    r = image_to_video(bundle, img, "a red square stays still", seed=0, steps=2)
    assert r.frames.shape == (8, 32, 32, 3)
    clip = torch.from_numpy(img).float().permute(2, 0, 1)[:, None].repeat(1, 2, 1, 1)[None] / 127.5 - 1
    known = bundle.vae.to_scaled(bundle.vae.encode(clip))
    assert torch.allclose(r.latents[:, :1], known[0], atol=1e-5)


def test_corrupted_or_mismatched_bundle_is_refused(tiny_env, tmp_path):
    d = tmp_path / "b"
    shutil.copytree(tiny_env["bundle"], d)
    w = d / "weights.safetensors"
    raw = bytearray(w.read_bytes()); raw[len(raw) // 2] ^= 1; w.write_bytes(bytes(raw))
    with pytest.raises(BundleError):
        load_bundle(d)
    d2 = tmp_path / "b2"
    shutil.copytree(tiny_env["bundle"], d2)
    raw = bytearray((d2 / "vae.safetensors").read_bytes()); raw[-3] ^= 1; (d2 / "vae.safetensors").write_bytes(bytes(raw))
    with pytest.raises(BundleError):
        load_bundle(d2)
    d3 = tmp_path / "b3"
    shutil.copytree(tiny_env["bundle"], d3)
    m = json.loads((d3 / "model_version.json").read_text()); m["vocab_hash"] = "0" * 64
    (d3 / "model_version.json").write_text(json.dumps(m))
    with pytest.raises(BundleError):
        load_bundle(d3)


def test_write_mp4_exact_frame_counts(tmp_path):
    for n in (1, 5, 16):
        fr = np.random.default_rng(0).integers(0, 255, (n, 16, 16, 3), dtype=np.uint8)
        write_mp4(fr, tmp_path / f"{n}.mp4", fps=8, upscale=1)
        assert probe_mp4(tmp_path / f"{n}.mp4")["frames"] == n
