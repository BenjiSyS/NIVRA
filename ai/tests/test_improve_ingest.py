"""Anime frame holds ("on 1s / 2s / 3s"): drawing-rate estimation from frame differences, ingest manifest fields
(`hold`, `effective_fps`), drawing-rate sampling."""
import json
import shutil

import numpy as np
import pytest

from nivra.data.ingest import IngestError, IngestOptions, ingest_folder
from nivra.data.ingest.quality import MAX_HOLD, estimate_holds, gray_diffs
from nivra.data.shards import read_manifest

from test_stylized_ingest import PROV, _encode, needs_ff, shape_clip


def diffs_for(gaps, lead=0, tail=0, level=3.0, noise=0.02, seed=0):
    """Frame-pair differences of a clip whose drawings last `gaps` frames each (a change every gap)."""
    rng = np.random.default_rng(seed)
    d = [0.0] * lead
    for g in gaps:
        d += [level * (0.7 + 0.6 * rng.random())] + [0.0] * (g - 1)
    d += [0.0] * tail
    return np.asarray(d) + rng.random(len(d)) * noise


# ===================================================================== estimator (no ffmpeg)
@pytest.mark.parametrize("gap,fps,eff", [(1, 24, 24.0), (2, 24, 12.0), (3, 24, 8.0), (4, 24, 6.0), (2, 30, 15.0)])
def test_uniform_holds(gap, fps, eff):
    r = estimate_holds(diffs_for([gap] * 10, lead=1, tail=1), fps)
    assert r["hold"] == gap and r["effective_fps"] == pytest.approx(eff, rel=1e-3)
    assert r["hold_purity"] == 1.0 and r["changes"] >= 9
    d = diffs_for([gap] * 10, lead=1, tail=1)
    assert r["hold_fraction"] == pytest.approx(float((d < 1.0).mean()), abs=1e-3)


def test_mixed_twos_and_threes_gives_a_fractional_rate():
    r = estimate_holds(diffs_for([2, 3] * 6), 24)
    assert r["hold"] in (2, 3) and r["effective_fps"] == pytest.approx(24 / 2.5, rel=0.03)
    assert 0.4 < r["hold_purity"] < 0.6


def test_codec_noise_is_not_a_drawing_change():
    r = estimate_holds(diffs_for([3] * 8, noise=0.1), 24)                     # held frames differ by <= 0.1 (codec noise)
    assert r["hold"] == 3
    d = diffs_for([3] * 8)
    d[d < 1] = 0.14                                                           # still below HOLD_ABS
    assert estimate_holds(d, 24)["hold"] == 3


def test_slow_motion_scaled_with_the_clips_own_peak():
    # a clip with one huge pose change and small moves: the relative threshold must not hide the small drawings
    d = diffs_for([2] * 10, level=0.5)
    d[7] = 40.0
    r = estimate_holds(d, 24)
    assert r["hold"] == 2


def test_static_or_too_short_clips_have_no_rate():
    for d in (np.zeros(16), np.zeros(0), np.array([5.0, 0.0]), np.array([0.0, 0.0, 5.0, 0.0, 0.0])):
        r = estimate_holds(d, 24)
        assert r["hold"] is None and r["effective_fps"] is None
    assert estimate_holds(np.zeros(16), 24)["hold_fraction"] == 1.0
    assert estimate_holds(diffs_for([2] * 5), 0.0)["hold"] is None


def test_long_pauses_do_not_set_the_drawing_rate():
    r = estimate_holds(diffs_for([2, 2, 2, 2, 2, 2, 60, 2, 2]), 24)           # a 60-frame pause between drawings
    assert r["hold"] == 2 and r["effective_fps"] == pytest.approx(12.0, rel=1e-3)
    assert MAX_HOLD >= 4


def test_gray_diffs():
    g = np.zeros((4, 8, 8), np.uint8)
    g[2:] = 10
    assert gray_diffs(g).tolist() == [0.0, 10.0, 0.0]
    assert gray_diffs(g[:1]).size == 0


# ===================================================================== ingest end to end (ffmpeg)
PROV_ROOT = dict(PROV)


def make_folder(tmp_path):
    d = tmp_path / "vids"
    d.mkdir()

    def clip(hold, bg, size=14, step=6):
        return np.concatenate([shape_clip(T=24, H=360, W=640, hold=hold, size=size, bg=bg, step=step)] * 3)   # 3 s @ 24 fps

    _encode(d / "twos.mp4", clip(2, (40, 60, 200)))
    _encode(d / "threes.mp4", clip(3, (200, 60, 40)))
    _encode(d / "ones.mp4", clip(1, (40, 160, 60), size=20, step=20))
    _encode(d / "busy.mp4", clip(1, (120, 40, 160), size=60))                # busy enough for the live profile too
    pattern = []                                                  # drawings of 2 and 3 frames alternating (10 fps)
    base = shape_clip(T=60, H=360, W=640, hold=1, size=14, bg=(90, 90, 160))
    t = 0
    k = 0
    while len(pattern) < 72:
        pattern += [base[(k * 3) % 60]] * (2 if k % 2 == 0 else 3)
        k += 1
    _encode(d / "mixed.mp4", np.stack(pattern[:72]))
    for n in ("twos", "threes", "ones", "mixed", "busy"):
        (d / f"{n}.txt").write_text(f"a square hops ({n})", encoding="utf-8")
    (d / "PROVENANCE.json").write_text(json.dumps(PROV_ROOT), encoding="utf-8")
    return d


def run(folder, out, **kw):
    kw.setdefault("val_fraction", 0.0)
    kw.setdefault("fps", 24)
    kw.setdefault("scene_threshold", None)
    kw.setdefault("dedup_bits", -1.0)
    return ingest_folder(folder, out, IngestOptions(**kw), log=lambda *_: None)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    return make_folder(tmp_path_factory.mktemp("holds"))


@needs_ff
def test_stylized_ingest_stores_hold_and_effective_fps(folder, tmp_path):
    s = run(folder, tmp_path / "o")
    rows = read_manifest(tmp_path / "o")
    by = {}
    for r in rows:
        by.setdefault(r["source_file"], []).append(r)
    assert s["options"]["detect_holds"] is True and s["ingest_version"] == 3
    assert {r["hold"] for r in by["twos.mp4"]} == {2} and all(r["effective_fps"] == pytest.approx(12.0, rel=0.05) for r in by["twos.mp4"])
    assert {r["hold"] for r in by["threes.mp4"]} == {3} and all(r["effective_fps"] == pytest.approx(8.0, rel=0.05) for r in by["threes.mp4"])
    assert {r["hold"] for r in by["ones.mp4"]} == {1} and all(r["effective_fps"] == pytest.approx(24.0, rel=0.05) for r in by["ones.mp4"])
    assert all(r["hold_purity"] is not None and 0.0 <= r["hold_fraction"] <= 1.0 for r in rows)
    assert all(r["fps"] == 24.0 for r in rows)                                         # sampling rate untouched by default
    mixed = by["mixed.mp4"]
    assert mixed and all(8.8 <= r["effective_fps"] <= 11.0 for r in mixed), [r["effective_fps"] for r in mixed]


@needs_ff
def test_live_profile_has_no_hold_fields_unless_asked(folder, tmp_path):
    s = run(folder, tmp_path / "live", profile="live", scene_threshold=None)
    rows = read_manifest(tmp_path / "live")
    assert s["options"]["detect_holds"] is False and rows and all("hold" not in r for r in rows)
    assert {r["source_file"] for r in rows} <= {"busy.mp4", "ones.mp4"}               # the held-frame clips fail the live filter
    run(folder, tmp_path / "live2", profile="live", detect_holds=True)
    live2 = read_manifest(tmp_path / "live2")
    assert live2 and all("hold" in r and "effective_fps" in r for r in live2) and {r["hold"] for r in live2} == {1}
    run(folder, tmp_path / "off", detect_holds=False)
    assert all("hold" not in r for r in read_manifest(tmp_path / "off"))


@needs_ff
def test_drawing_rate_sampling_rereads_the_clip_at_the_drawing_rate(folder, tmp_path):
    s = run(folder, tmp_path / "dr", drawing_rate_sampling=True)
    rows = read_manifest(tmp_path / "dr")
    threes = sorted((r for r in rows if r["source_file"] == "threes.mp4"), key=lambda r: r["start"])
    first = threes[0]
    assert first["fps"] == pytest.approx(8.0, rel=0.05) and first["effective_fps"] == pytest.approx(8.0, rel=0.05)
    assert first["end"] - first["start"] == pytest.approx(17 / first["fps"], abs=1e-3)       # spans more source time
    assert first["frames"] == 17
    ones = [r for r in rows if r["source_file"] == "ones.mp4"]
    assert ones and all(r["fps"] == 24.0 for r in ones)                                    # already on ones: unchanged
    assert s["options"]["drawing_rate_sampling"] is True
    # the re-read clip really has no repeated frames (every frame a new drawing)
    from safetensors.numpy import load_file
    vids = load_file(str(tmp_path / "dr" / first["shard"]))["videos"]
    clip = vids[first["index"]]
    d = gray_diffs(clip[..., :].mean(-1)[:, ::4, ::4])
    assert (d > 0.15).mean() > 0.9


@needs_ff
def test_drawing_rate_sampling_needs_hold_detection(folder, tmp_path):
    with pytest.raises(IngestError, match="hold detection"):
        run(folder, tmp_path / "x", drawing_rate_sampling=True, detect_holds=False)
    assert not (tmp_path / "x").exists()
