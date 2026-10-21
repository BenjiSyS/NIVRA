"""Nivra HD pipeline at tiny scale: frame count through a real MP4 (ffprobe), stop file, continuation."""
import pytest
import torch

from nivra.inference.generate import GenerationState, StateError
from nivra.inference.hd import HDPipeline, StubTextEncoder, frames_for, sample_flow
from nivra.inference.sampler import Stopped
from nivra.inference.video_io import probe_mp4
from nivra.models.hd_config import build_dit, build_hcvae, load_hd_config
from nivra.text.interface import TextEncoder


@pytest.fixture(scope="module")
def pipe():
    torch.manual_seed(0)
    cfg = load_hd_config("nivra_hd_tiny")
    dit = build_dit(cfg)
    with torch.no_grad():
        for p in dit.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return HDPipeline.from_config(cfg, StubTextEncoder(dim=16), dit=dit, vae=build_hcvae(cfg),
                                  model_version_id="test-v1")


def test_frames_rule():
    assert frames_for(8, 30) == 241 and frames_for(4, 30) == 121 and frames_for(2, 30) == 65
    assert isinstance(StubTextEncoder(), TextEncoder)


def test_generate_writes_mp4_with_exact_frame_count(pipe, tmp_path):
    seen = []
    res = pipe.generate("a red ball rolls left", seed=3, width=96, height=40, frames=17, steps=3,
                        out_path=tmp_path / "a.mp4", on_progress=lambda s, i, n: seen.append((s, i, n)))
    assert res.frames.shape == (17, 40, 96, 3)
    info = probe_mp4(res.mp4)
    assert info["frames"] == 17 and info["width"] == 96 and info["height"] == 40 and info["fps"] == 30
    assert res.meta["padded"] == [64, 96] and res.meta["latent_shape"] == [8, 3, 2, 3]
    stages = {s for s, _, _ in seen}
    assert {"text", "sample", "decode", "write"} <= stages
    assert ("sample", 3, 3) in seen
    assert res.state.recent_latents.shape == (8, 2, 2, 3)


def test_seconds_parameter_and_determinism(pipe):
    a = pipe.generate("waves", seed=1, width=64, height=64, seconds=0.5, fps=16, steps=2)
    b = pipe.generate("waves", seed=1, width=64, height=64, seconds=0.5, fps=16, steps=2)
    assert a.frames.shape[0] == 9
    assert (a.frames == b.frames).all()


def test_stop_file(pipe, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("stop")
    with pytest.raises(Stopped):
        pipe.generate("x", width=64, height=64, frames=9, steps=2, stop_file=stop)


def test_cfg_and_samplers_run():
    torch.manual_seed(0)
    cfg = load_hd_config("nivra_hd_tiny")
    dit = build_dit(cfg).eval()
    with torch.no_grad():
        for p in dit.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    e, m = StubTextEncoder(16).encode(["a cat"])
    for s in ("euler", "ab2"):
        z = sample_flow(dit, (1, 8, 2, 2, 2), e, m, steps=4, cfg_scale=4.0, seed=0, sampler=s, shift=2.0)
        assert z.shape == (1, 8, 2, 2, 2) and torch.isfinite(z).all()
    with pytest.raises(ValueError):
        sample_flow(dit, (1, 8, 2, 2, 2), e, m, steps=1, sampler="nope")


def test_continuation_roundtrip(pipe, tmp_path):
    first = pipe.generate("a boat at sea", seed=2, width=64, height=64, frames=17, steps=2)
    first.state.save(tmp_path / "st")
    st = GenerationState.load(tmp_path / "st")
    nxt = pipe.continue_video(st, steps=2, frames=17, width=64, height=64, out_path=tmp_path / "b.mp4")
    assert nxt.frames.shape[0] == 17 - 9                     # 1+8(K-1) overlap frames dropped (K=2)
    assert probe_mp4(nxt.mp4)["frames"] == 8
    assert nxt.state.segments == 2 and nxt.state.temporal_position == 3 + 3 - 2
    assert nxt.meta["mode"] == "continuation"
    st.model_version_id = "other"
    with pytest.raises(StateError):
        pipe.continue_video(st, steps=1, frames=9, width=64, height=64)


def test_text_dim_mismatch_refused():
    cfg = load_hd_config("nivra_hd_tiny")
    with pytest.raises(ValueError):
        HDPipeline.from_config(cfg, StubTextEncoder(dim=12))
