"""Long videos (docs/architecture/LONG_VIDEO.md): segment plan parity with Go, SDEdit sampler, streaming stitch,
and the real ai.draft -> ai.refine_segment -> ai.stitch chain through `nivra.cli exec` at tiny scale on CPU."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from nivra.inference import hd_bundle as hb
from nivra.inference import longvideo as lv
from nivra.inference.hd import sample_flow
from nivra.inference.sampler import Stopped
from nivra.inference.video_io import Mp4Writer, RawReader, probe_fast, probe_mp4

AI_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ plan (vectors shared with Go segmentPlan)
def test_segment_plan_vectors():
    assert lv.segment_plan(601, 241) == [[0, 240], [240, 480], [480, 600]]
    assert lv.segment_plan(241, 241) == [[0, 240]]
    p = lv.segment_plan(108001, 241)
    assert len(p) == 450 and p[0] == [0, 240] and p[-1] == [107760, 108000]
    p = lv.segment_plan(1000, 193)
    assert p[0][0] == 0 and p[-1][1] == 999
    assert all(a[1] == b[0] for a, b in zip(p, p[1:]))
    assert len(p) == -(-(1000 - 1) // 192)
    assert lv.segment_plan(1, 241) == [[0, 0]]
    assert lv.segment_plan(2, 9) == [[0, 1]]


def test_total_frames_rounds_half_up_like_go():
    assert lv.total_frames(3, 8) == 25
    assert lv.total_frames(2.5, 1) == 4          # math.Round(2.5) = 3, not Python's banker's 2
    assert lv.total_frames(3600, 30) == 108001
    assert lv.total_frames(0.4, 1) == 1


# ------------------------------------------------------------------ SDEdit sampler
class _ZeroNet:
    def __call__(self, x, t, text, mask, t_offset=0):
        return torch.zeros_like(x)


class _RandNet:
    def __init__(self):
        self.g = torch.Generator().manual_seed(0)

    def __call__(self, x, t, text, mask, t_offset=0):
        return torch.randn(x.shape, generator=self.g)


def test_sdedit_start_and_anchors():
    init = torch.randn(1, 2, 5, 2, 2)
    net = _ZeroNet()
    seen = []
    # zero velocity: the result is the noised init at the first scheduled t <= strength
    out = sample_flow(net, tuple(init.shape), None, None, steps=10, cfg_scale=1.0, seed=3, sampler="euler",
                      init=init, strength=0.5, on_step=lambda i, n: seen.append((i, n)))
    assert seen[-1][0] == seen[-1][1] and seen[-1][1] < 10      # only the remaining part of the schedule ran
    assert not torch.allclose(out, init) and (out - init).abs().mean() < 1.5
    # anchors are restored exactly at the end, even with a random velocity field
    out = sample_flow(_RandNet(), tuple(init.shape), None, None, steps=6, cfg_scale=1.0, seed=3, sampler="ab2",
                      init=init, strength=0.7, anchors=([0, 4], init[:, :, [0, 4]]))
    assert torch.equal(out[:, :, [0, 4]], init[:, :, [0, 4]])
    assert not torch.equal(out[:, :, 2], init[:, :, 2])
    with pytest.raises(ValueError):
        sample_flow(net, tuple(init.shape), None, None, steps=4, init=init, strength=0.0)
    # default path unchanged: same call without the new args equals the explicit defaults
    a = sample_flow(_ZeroNet(), (1, 2, 3, 2, 2), None, None, steps=4, cfg_scale=1.0, seed=1)
    b = sample_flow(_ZeroNet(), (1, 2, 3, 2, 2), None, None, steps=4, cfg_scale=1.0, seed=1, init=None, anchors=None)
    assert torch.equal(a, b)


# ------------------------------------------------------------------ stitch on synthetic segments
def _seg(path, n, level0, w=32, h=16, fps=8):
    """n constant-colour frames whose gray level is level0 + 4*k (k = frame index)."""
    wr = Mp4Writer(path, w, h, fps, crf=1)
    for k in range(n):
        wr.write(np.full((h, w, 3), level0 + 4 * k, dtype=np.uint8))
    wr.close()
    return path


def _levels(path):
    with RawReader(path, 32, 16, 8) as r:
        out = []
        while True:
            f = r.read()
            if f is None:
                return out
            out.append(float(f.mean()))


@pytest.mark.parametrize("ns,cf", [([9, 9, 9], 0), ([9, 9, 9], 4), ([9, 9, 9], 5), ([9, 2, 9], 4), ([9, 3, 9, 2], 6), ([1, 9], 4), ([9], 4)])
def test_stitch_counts_and_order(tmp_path, ns, cf):
    segs = [_seg(tmp_path / f"s{i}.mp4", n, 10 + 40 * i) for i, n in enumerate(ns)]
    out = tmp_path / "v.mp4"
    s = lv.run_stitch(segs, out, 8, cf)
    expect = sum(ns) - (len(ns) - 1)
    assert s["frames"] == expect and probe_mp4(out)["frames"] == expect
    lev = _levels(out)
    assert len(lev) == expect
    if cf == 0:                                   # plain concatenation minus the duplicate boundary frame
        ref = []
        for i, n in enumerate(ns):
            ref += [10 + 40 * i + 4 * k for k in range(n)][(1 if i else 0):]
        assert np.allclose(lev, ref, atol=3)


def test_stitch_crossfade_is_monotone_ramp(tmp_path):
    # segment 0 ends at level 10+4*8=42, segment 1 starts at 90: the seam ramps up without overshoot
    a, b = _seg(tmp_path / "a.mp4", 9, 10), _seg(tmp_path / "b.mp4", 9, 90)
    lv.run_stitch([a, b], tmp_path / "v.mp4", 8, 4)
    lev = _levels(tmp_path / "v.mp4")
    assert len(lev) == 17
    seam = lev[5:9]                               # window = 2 before B, B, 1 after
    assert all(x < y for x, y in zip(seam, seam[1:]))
    assert lev[4] < seam[0] + 3 and lev[9] > seam[-1] - 3
    # a hard cut has a jump of ~48 at the seam, the fade splits it into smaller steps
    steps = np.diff(lev)
    assert steps.max() < 40


def test_stitch_rejects_mismatch_and_garbage(tmp_path):
    a = _seg(tmp_path / "a.mp4", 9, 10)
    b = _seg(tmp_path / "b.mp4", 9, 10, w=64)
    with pytest.raises(lv.LongVideoError):
        lv.run_stitch([a, b], tmp_path / "v.mp4", 8)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(lv.LongVideoError):
        lv.run_stitch([a, bad], tmp_path / "v.mp4", 8)
    assert not (tmp_path / "v.mp4").exists()


def test_stitch_stop_removes_partial(tmp_path):
    segs = [_seg(tmp_path / f"s{i}.mp4", 9, 10) for i in range(3)]
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 5

    with pytest.raises(Stopped):
        lv.run_stitch(segs, tmp_path / "v.mp4", 8, 4, should_stop=stop)
    assert not (tmp_path / "v.mp4").exists()


def test_raw_reader_frame_accurate_seek(tmp_path):
    p = _seg(tmp_path / "s.mp4", 30, 10)
    with RawReader(p, 32, 16, 8, start=17, count=5) as r:
        got = []
        while (f := r.read()) is not None:
            got.append(float(f.mean()))
    assert np.allclose(got, [10 + 4 * k for k in range(17, 22)], atol=3)
    assert probe_fast(p)["frames"] == 30


# ------------------------------------------------------------------ real jobs
@pytest.fixture(scope="module")
def hd(tmp_path_factory):
    d = tmp_path_factory.mktemp("lv") / "bundle"
    hb.write_hd_bundle(d, "nivra_hd_tiny", seed=1)
    return d


def models(d):
    return {"weights": str(d / "weights.safetensors"), "manifest": str(d / "model_version.json"),
            "vae_weights": str(d / "vae.safetensors")}


def run(workdir, job_type, params, ins, name, stop_file=None):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    job = {"protocol": "nivra.exec/1", "job_type": job_type, "job_id": "j-" + name, "attempt_id": "a-" + name,
           "params": params, "inputs": ins, "output_dir": str(workdir / ("out_" + name)),
           "stop_file": str(stop_file or workdir / "STOP"), "limits": {"max_seconds": 600}}
    jp = workdir / f"{name}.json"
    jp.write_text(json.dumps(job))
    env = dict(os.environ, PYTHONPATH=str(AI_DIR), NIVRA_THREADS="2", PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "nivra.cli", "exec", "--job", str(jp)], cwd=str(workdir), env=env,
                       capture_output=True, text=True, timeout=900)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    return p.returncode, ev, Path(job["output_dir"]), p


def arts(ev):
    return {e["kind"]: e["path"] for e in ev if e["event"] == "artifact"}


def result(ev):
    return [e for e in ev if e["event"] == "result"][0]["summary"]


@pytest.fixture(scope="module")
def chain(hd, tmp_path_factory):
    w = tmp_path_factory.mktemp("chain")
    common = {"prompt": "a red ball", "fps": 8, "steps": 3, "style": "anime"}
    rc, ev, out, p = run(w, "ai.draft", {**common, "seed": 5, "seconds": 3, "draft_width": 64, "draft_height": 32,
                                         "segment_frames": 9, "name": "draft"}, models(hd), "draft")
    assert rc == 0, p.stdout + p.stderr
    a = arts(ev)
    plan = json.loads(Path(a["draft_plan"]).read_text())
    segs = []
    for s in plan["segments"]:
        rc, ev2, o2, p2 = run(w, "ai.refine_segment",
                              {**common, "seed": 5 + s["index"], "width": 96, "height": 64, "index": s["index"],
                               "start": s["start"], "end": s["end"], "name": f"segment_{s['index']:03d}"},
                              {**models(hd), "draft_video": a["draft_video"]}, f"seg{s['index']}")
        assert rc == 0, p2.stdout + p2.stderr
        segs.append((ev2, arts(ev2)["segment_video"]))
    ins = {f"segment_{i:03d}": path for i, (_, path) in enumerate(segs)}
    rc, ev3, o3, p3 = run(w, "ai.stitch", {"fps": 8, "count": len(segs), "crossfade_frames": 4, "name": "video"}, ins, "stitch")
    assert rc == 0, p3.stdout + p3.stderr
    return {"draft_ev": ev, "plan": plan, "draft": a, "segs": segs, "stitch_ev": ev3, "video": arts(ev3)["video"], "work": w}


def test_draft_artifacts_and_plan(chain):
    ev, plan, a = chain["draft_ev"], chain["plan"], chain["draft"]
    assert set(a) == {"draft_video", "draft_plan"}
    assert plan == {"frames": 25, "fps": 8, "segment_frames": 9,
                    "segments": [{"index": 0, "start": 0, "end": 8}, {"index": 1, "start": 8, "end": 16},
                                 {"index": 2, "start": 16, "end": 24}]}
    info = probe_mp4(a["draft_video"])
    assert (info["frames"], info["width"], info["height"], info["fps"]) == (25, 64, 32, 8.0)
    s = result(ev)
    assert s["frames"] == 25 and s["segments"] == 3 and "seconds_total" in s
    prog = [e["fraction"] for e in ev if e["event"] == "progress"]
    assert prog == sorted(prog) and prog[-1] == 1.0 and len(prog) > 3


def test_refine_outputs_and_boundary_continuity(chain):
    plan = chain["plan"]["segments"]
    vids = []
    for (ev, path), s in zip(chain["segs"], plan):
        info = probe_mp4(path)
        assert (info["frames"], info["width"], info["height"], info["fps"]) == (s["end"] - s["start"] + 1, 96, 64, 8.0)
        r = result(ev)
        assert r["index"] == plan.index(s) and r["frames"] == info["frames"]
        from nivra.inference.video_io import decode_mp4
        vids.append(decode_mp4(path).astype(np.float32))
    for a, b in zip(vids, vids[1:]):
        boundary = np.abs(a[-1] - b[0]).mean()
        unrelated = np.abs(a[0] - b[-1]).mean()
        within = np.abs(a[3] - a[4]).mean()
        # anchored to the same draft frame: neighbouring segments meet in (nearly) the same picture
        assert boundary < 40 and boundary <= max(unrelated, within) + 5, (boundary, unrelated, within)


def test_stitch_final_video(chain):
    info = probe_mp4(chain["video"])
    assert (info["frames"], info["width"], info["height"], info["fps"]) == (25, 96, 64, 8.0)
    s = result(chain["stitch_ev"])
    assert s["frames"] == 25 and s["seconds"] == 25 / 8


def test_refine_is_deterministic(chain, hd):
    w = chain["work"]
    s = chain["plan"]["segments"][1]
    params = {"prompt": "a red ball", "fps": 8, "steps": 3, "style": "anime", "seed": 6, "width": 96, "height": 64,
              "index": 1, "start": s["start"], "end": s["end"], "name": "again"}
    rc, ev, out, p = run(w, "ai.refine_segment", params, {**models(hd), "draft_video": chain["draft"]["draft_video"]}, "again")
    assert rc == 0, p.stderr
    from nivra.inference.video_io import decode_mp4
    assert np.array_equal(decode_mp4(arts(ev)["segment_video"]), decode_mp4(chain["segs"][1][1]))


def test_refine_odd_size_and_short_segment(chain, hd):
    w = chain["work"]
    # 2-frame segment (rounded up to 9 internally, trimmed), size not a multiple of 32
    params = {"prompt": "a red ball", "fps": 8, "steps": 2, "seed": 1, "width": 98, "height": 66, "index": 0,
              "start": 3, "end": 4, "name": "odd"}
    rc, ev, out, p = run(w, "ai.refine_segment", params, {**models(hd), "draft_video": chain["draft"]["draft_video"]}, "odd")
    assert rc == 0, p.stderr
    info = probe_mp4(arts(ev)["segment_video"])
    assert (info["frames"], info["width"], info["height"]) == (2, 98, 66)


# ------------------------------------------------------------------ stop / errors
def test_stop_file_present_exits_3(hd, tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("x")
    rc, ev, out, p = run(tmp_path, "ai.draft", {"prompt": "x", "seconds": 3, "fps": 8, "segment_frames": 9}, models(hd), "s",
                         stop_file=stop)
    assert rc == 3 and ev[-1]["code"] == "STOPPED"


def test_draft_stops_between_segments_and_removes_partial(hd, tmp_path):
    pipe = hb.load_hd_bundle(hd)
    n = [0]

    def stop():
        n[0] += 1
        return n[0] > 12          # a few sampling steps of the first segment, then mid-chain

    out = tmp_path / "d.mp4"
    with pytest.raises(Stopped):
        lv.run_draft(pipe, prompt="a", seed=1, seconds=3, fps=8, segment_frames=9, width=64, height=32,
                     out_mp4=out, plan_path=tmp_path / "p.json", steps=3, should_stop=stop)
    assert n[0] > 12 and not out.exists() and not (tmp_path / "p.json").exists()


@pytest.mark.parametrize("job,params,extra,code", [
    ("ai.draft", {"prompt": "x", "seconds": 3, "bogus": 1}, {}, 2),
    ("ai.draft", {"prompt": "x", "seconds": 3, "style": "no-such-style"}, {}, 2),
    ("ai.draft", {"prompt": "x", "seconds": 3, "draft_width": 63}, {}, 2),
    ("ai.draft", {"prompt": "x", "seconds": 3, "segment_frames": 10}, {}, 2),
    ("ai.draft", {"seconds": 3}, {}, 2),
    ("ai.draft", {"prompt": "x", "seconds": 20000}, {}, 2),
    ("ai.refine_segment", {"prompt": "x", "width": 96, "height": 64, "start": 5, "end": 2}, {"draft_video": "MISSING"}, 2),
    ("ai.refine_segment", {"prompt": "x", "width": 97, "height": 64, "start": 0, "end": 2}, {"draft_video": "MISSING"}, 2),
    ("ai.refine_segment", {"prompt": "x", "width": 96, "height": 64, "start": 0, "end": 2, "strength": 0}, {"draft_video": "MISSING"}, 2),
    ("ai.refine_segment", {"prompt": "x", "width": 96, "height": 64, "start": 0, "end": 2}, {"draft_video": "MISSING"}, 5),
    ("ai.refine_segment", {"prompt": "x", "width": 96, "height": 64, "start": 0, "end": 2}, {}, 2),
    ("ai.stitch", {"fps": 8, "count": 2}, {"segment_000": "A"}, 2),
    ("ai.stitch", {"fps": 8, "count": 1}, {"segment_000": "A", "segment_001": "A"}, 2),
    ("ai.stitch", {"fps": 8}, {"segment_000": "A"}, 2),
    ("ai.stitch", {"fps": 8, "count": 1}, {"segment_000": "MISSING"}, 5),
    ("ai.stitch", {"fps": 8, "count": 1}, {"weights": "A"}, 2),
])
def test_bad_inputs(hd, tmp_path, job, params, extra, code):
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"nope")
    ins = {} if job == "ai.stitch" else models(hd)
    ins = {**ins, **{k: (str(garbage) if v == "A" else str(tmp_path / "missing.mp4")) for k, v in extra.items()}}
    rc, ev, out, p = run(tmp_path, job, params, ins, "bad")
    assert rc == code, (rc, p.stdout, p.stderr)


def test_garbage_inputs_exit_5(hd, tmp_path):
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"nope")
    rc, ev, out, p = run(tmp_path, "ai.stitch", {"fps": 8, "count": 1}, {"segment_000": str(garbage)}, "g1")
    assert rc == 5 and ev[-1]["code"] == "INPUT_INVALID"
    rc, ev, out, p = run(tmp_path, "ai.refine_segment", {"prompt": "x", "width": 96, "height": 64, "start": 0, "end": 2, "steps": 2},
                         {**models(hd), "draft_video": str(garbage)}, "g2")
    assert rc == 5, p.stderr
    # segment range beyond the draft
    seg = _seg(tmp_path / "d.mp4", 9, 10, w=64, h=32)
    rc, ev, out, p = run(tmp_path, "ai.refine_segment", {"prompt": "x", "width": 96, "height": 64, "start": 5, "end": 20, "steps": 2},
                         {**models(hd), "draft_video": str(seg)}, "g3")
    assert rc == 5 and "frames" in ev[-1]["message"]


def test_workers_advertise_every_job_type_the_executor_handles():
    """Regression: ai.draft / ai.refine_segment / ai.stitch were missing from WORKER_JOB_TYPES, so no worker ever
    claimed the first job of a long video and it sat 'ready' forever (found by the real two-worker e2e run)."""
    from nivra import cli, executor
    worker, controller = set(cli.WORKER_JOB_TYPES), set(cli.CONTROLLER_JOB_TYPES)
    assert not worker & controller
    assert worker | controller == set(executor.HANDLERS)
    assert {"ai.draft", "ai.refine_segment", "ai.stitch"} <= worker
