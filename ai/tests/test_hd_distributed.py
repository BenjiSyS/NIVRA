"""HD distributed training: streaming IO, bounds, train/encode/eval/aggregate jobs (docs/architecture/HD_TRAINING.md)."""
import copy
import io
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from nivra.distributed import version as ver
from nivra.distributed.hd import bounds
from nivra.distributed.hd.aggregate import TRAIN_FILE, expected_delta_bytes, hd_aggregate
from nivra.distributed.hd.latent_shard import LatentShard, write_latent_shard
from nivra.distributed.hd.stream_io import (StReader, StreamIOError, StWriter, check_exact_size, expected_file_size,
                                            sha256_file, write_tensors)
from nivra.distributed.hd.train_job import hd_train_job
from nivra.distributed.hd.validate import validate_hd_update
from nivra.inference.hd_bundle import write_hd_bundle
from nivra.models.hd_config import load_hd_config

STAGE = {"h": 32, "w": 64, "frames": 9, "fps": 8}
LR, H = 1e-3, 50
LAT = (8, 2, 1, 2)          # nivra_hd_tiny latent at STAGE


def make_shard(path, n=6, seed=0, stage=STAGE, L=5, D=16, split="train"):
    g = torch.Generator().manual_seed(seed)
    rows = [{"id": f"c{seed}-{i}", "caption": f"clip {i}", "text": f"clip {i}", "style": "anime", "lang": "en", "split": split}
            for i in range(n)]
    meta = {"parent_shard_sha256": "0" * 64, "vae_version": "v" * 64, "text_encoder_id": "t" * 64, "stage": stage, "rows": rows}
    mask = torch.ones(n, L, dtype=torch.uint8)
    return write_latent_shard(path, torch.randn(n, *LAT, generator=g), torch.randn(n, L, D, generator=g), mask,
                              torch.zeros(1, 2, D), torch.ones(1, 2, dtype=torch.uint8), meta)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("hdd")
    bundle = root / "bundle0"
    m = write_hd_bundle(bundle, "nivra_hd_tiny", text_encoder="stub", seed=1, dtype=torch.float32)
    shards = []
    for i in range(3):
        p = root / f"s{i}.safetensors"
        sha = make_shard(p, seed=i)
        shards.append((p, sha))
    val = root / "val.safetensors"
    make_shard(val, n=4, seed=99, split="val")
    return {"root": root, "bundle": bundle, "manifest": m, "shards": shards, "val": val}


def params(env, uid, shard_idx, seed, **kw):
    p, sha = env["shards"][shard_idx]
    d = {"update_id": uid, "run_id": "run1", "round": 0, "round_batch": 0, "base_version_id": env["manifest"]["version_id"],
         "target": "dit", "hd_config_sha": env["manifest"]["hd_config_sha"], "stage": STAGE, "inner_steps": H,
         "batch_size": 2, "grad_accum": 1, "lr": LR, "warmup_steps": 5, "weight_decay": 0.0, "grad_clip": 1.0,
         "cfg_dropout": 0.1, "seed": seed, "shards": [{"artifact_id": f"a{shard_idx}", "sha256": sha}],
         "delta_dtype": "float16"}
    d.update(kw)
    return d


def run_train(env, uid, shard_idx, seed, out, **kw):
    b = env["bundle"]
    ins = {"base_train_weights": str(b / "weights.safetensors"), "base_manifest": str(b / "model_version.json"),
           "data_shard_000": str(env["shards"][shard_idx][0])}
    return hd_train_job(ins, params(env, uid, shard_idx, seed, **kw), out, attempt_id=f"att-{uid}")


@pytest.fixture(scope="module")
def updates(env):
    ups = []
    for i in range(3):
        out = env["root"] / f"u{i}"
        s = run_train(env, f"upd-{i}", i, 10 + i, out)
        ups.append({"path": str(out / "update.safetensors"), "manifest_path": str(out / "update.json"),
                    "update_id": f"upd-{i}", "summary": s})
    return ups


# ---------------------------------------------------------------- streaming IO
def test_stream_roundtrip_and_exact_size(tmp_path):
    t = {"a": torch.randn(3, 4), "b": torch.randn(5), "c": torch.tensor(2.5)}
    for dt in (torch.float32, torch.float16):
        p = tmp_path / f"x{dt}.safetensors"
        sha = write_tensors(p, t, dt)
        assert sha == sha256_file(p)
        assert p.stat().st_size == expected_file_size([(k, v.shape) for k, v in t.items()], dt)
        back = load_file(str(p))                                    # real safetensors reads our file
        for k in t:
            assert torch.allclose(back[k].float(), t[k].to(dt).float())
        with StReader(p) as r:
            assert r.shapes() == {k: tuple(v.shape) for k, v in t.items()}
            assert torch.equal(r.read("a"), t["a"].to(dt))
    code = "F16"
    shapes = {k: v.shape for k, v in t.items()}
    assert check_exact_size(tmp_path / f"x{torch.float16}.safetensors", shapes, code) is None
    assert "dtype" in check_exact_size(tmp_path / f"x{torch.float32}.safetensors", shapes, code)
    with open(tmp_path / f"x{torch.float16}.safetensors", "ab") as f:
        f.write(b"\0")
    assert check_exact_size(tmp_path / f"x{torch.float16}.safetensors", shapes, code)       # trailing byte


def test_reader_rejects_bad_files(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(b"\xff" * 16)
    with pytest.raises(StreamIOError):
        StReader(p)
    q = tmp_path / "ok.safetensors"
    write_tensors(q, {"a": torch.zeros(4)}, torch.float16)
    raw = q.read_bytes()
    q.write_bytes(raw[:-2])                                           # truncated data
    with pytest.raises(StreamIOError):
        StReader(q)


def test_writer_enforces_order(tmp_path):
    w = StWriter(tmp_path / "w.safetensors", [("a", (2,)), ("b", (2,))])
    with pytest.raises(StreamIOError):
        w.write("b", torch.zeros(2))
    w.abort()


# ---------------------------------------------------------------- the worker job
def test_honest_run_bounds_and_manifest(env, updates):
    bm = ver.read_manifest(env["bundle"] / "model_version.json")
    for u in updates:
        s = u["summary"]
        assert s["fp16_rel_err"] < 1e-3                               # fp16 round trip of the fp32 delta
        assert s["samples_seen"] == H * 2 and math.isfinite(s["loss_last"])
        m = json.loads(Path(u["manifest_path"]).read_text())
        for k in ("run_id", "round", "round_batch", "base_version_id", "update_id", "attempt_id", "target", "hd_config_sha",
                  "stage", "inner_steps", "lr", "shards", "samples_seen", "delta_dtype"):
            assert k in m
        assert m["attempt_id"] == f"att-{u['update_id']}" and m["schema"].startswith("nivra.update/")
        assert Path(u["path"]).stat().st_size == expected_delta_bytes(bm["tensor_shapes"])
        v = validate_hd_update(u["path"], u["manifest_path"], bm, expected={
            "base_version_id": bm["version_id"], "inner_steps": H, "lr": LR, "update_id": u["update_id"],
            "stage": STAGE, "batch_size": 2, "grad_accum": 1, "delta_dtype": "float16"})
        assert v.valid, v.reasons
        assert v.l2_norm > 0


def test_training_is_deterministic(env, updates):
    out = env["root"] / "u0b"
    run_train(env, "upd-0", 0, 10, out)
    assert sha256_file(out / "update.safetensors") == sha256_file(updates[0]["path"])


def test_bad_inputs_rejected(env):
    with pytest.raises(Exception, match="sha256"):
        run_train(env, "x", 0, 1, env["root"] / "bad1", shards=[{"artifact_id": "a", "sha256": "f" * 64}])
    with pytest.raises(Exception, match="stage"):
        run_train(env, "x", 0, 1, env["root"] / "bad2", stage={"h": 64, "w": 64, "frames": 9, "fps": 8})
    with pytest.raises(Exception, match="base_version_id"):
        run_train(env, "x", 0, 1, env["root"] / "bad3", base_version_id="other")


# ---------------------------------------------------------------- adversarial updates
def _tamper(env, u, name, fn, tmp_name):
    """Copy an update, rewrite its tensors with fn(tensor)->tensor, fix the sha in the manifest."""
    d = env["root"] / tmp_name
    d.mkdir(exist_ok=True)
    t = load_file(u["path"])
    t = {k: fn(k, v) for k, v in t.items()}
    p = d / "update.safetensors"
    write_tensors(p, t, torch.float16)
    m = json.loads(Path(u["manifest_path"]).read_text())
    m["update_sha256"] = sha256_file(p)
    mp = d / "update.json"
    mp.write_text(json.dumps(m))
    return {"path": str(p), "manifest_path": str(mp), "update_id": u["update_id"]}


def _expected(env, u):
    bm = ver.read_manifest(env["bundle"] / "model_version.json")
    return bm, {"base_version_id": bm["version_id"], "inner_steps": H, "lr": LR, "update_id": u["update_id"]}


def test_scaled_nan_zero_and_corrupt_rejected(env, updates):
    u = updates[0]
    bm, exp = _expected(env, u)
    big = _tamper(env, u, "scaled", lambda k, v: v * 1000, "t_scaled")
    r = validate_hd_update(big["path"], big["manifest_path"], bm, expected=exp)
    assert not r.valid and "bound_exceeded" in r.codes
    nan = _tamper(env, u, "nan", lambda k, v: torch.full_like(v, float("nan")) if k == next(iter(load_file(u["path"]))) else v, "t_nan")
    r = validate_hd_update(nan["path"], nan["manifest_path"], bm, expected=exp)
    assert "non_finite" in r.codes
    zero = _tamper(env, u, "zero", lambda k, v: torch.zeros_like(v), "t_zero")
    r = validate_hd_update(zero["path"], zero["manifest_path"], bm, expected=exp)
    assert "zero_update" in r.codes
    # sha mismatch (bit flip)
    d = env["root"] / "t_flip"
    d.mkdir(exist_ok=True)
    shutil.copy(u["path"], d / "update.safetensors")
    shutil.copy(u["manifest_path"], d / "update.json")
    raw = bytearray((d / "update.safetensors").read_bytes())
    raw[-1] ^= 1
    (d / "update.safetensors").write_bytes(bytes(raw))
    r = validate_hd_update(d / "update.safetensors", d / "update.json", bm, expected=exp)
    assert "checksum_mismatch" in r.codes
    # echo mismatch: wrong lr in the job params
    r = validate_hd_update(u["path"], u["manifest_path"], bm, expected=dict(exp, lr=2e-3))
    assert "manifest_mismatch" in r.codes
    # stale base
    r = validate_hd_update(u["path"], u["manifest_path"], bm, expected=dict(exp, base_version_id="v-other"))
    assert "stale_base_version" in r.codes


# ---------------------------------------------------------------- streaming aggregate == in-memory reference
def reference_aggregate(env, ups, outer_lr=0.7, momentum=0.9, clip_factor=2.0):
    base = load_file(str(env["bundle"] / "weights.safetensors"))
    ds = [load_file(u["path"]) for u in ups]
    norms = [math.sqrt(sum(float(v.double().pow(2).sum()) for v in d.values())) for d in ds]
    med = float(torch.tensor(norms, dtype=torch.float64).median())
    cap = clip_factor * med
    out_w, out_m = {}, {}
    n = len(ds)
    for k, w in base.items():
        parts = []
        for d, nr in zip(ds, norms):
            x = d[k].float()
            parts.append(x * (cap / nr) if nr > cap else x)
        st = torch.stack(parts)
        if n >= 3:                                   # auto -> trimmed mean, k = max(1, int(0.2 n)) = 1
            s, _ = torch.sort(st, dim=0)
            g = s[1:n - 1].mean(0)
        else:
            g = st.mean(0)
        m = momentum * torch.zeros_like(w) + g
        out_w[k] = w - outer_lr * (g + momentum * m)
        out_m[k] = m
    return out_w, out_m


def test_streaming_aggregate_matches_reference(env, updates, tmp_path):
    out = tmp_path / "agg"
    s = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", updates, out,
                     inner_steps=H, lr=LR, version_id="v-new")
    assert not s["no_op"] and s["aggregated_count"] == 3 and s["new_version_id"] == "v-new"
    assert s["aggregation_method"] == "trimmed_mean"
    ref_w, ref_m = reference_aggregate(env, updates)
    got_w = load_file(str(out / "weights.safetensors"))
    got_m = load_file(str(out / "outer_state.safetensors"))
    got_t = load_file(str(out / TRAIN_FILE))
    for k in ref_w:
        assert (got_w[k] - ref_w[k]).abs().max() < 1e-6
        assert (got_m[k] - ref_m[k]).abs().max() < 1e-6
        assert got_t[k].dtype == torch.float16 and got_w[k].dtype == torch.float32
        assert (got_t[k].float() - ref_w[k]).abs().max() <= 1e-3 * max(1.0, float(ref_w[k].abs().max()))
    man = json.loads((out / "model_version.json").read_text())
    assert man["parent_version_id"] == env["manifest"]["version_id"] and man["round"] == 1
    assert man["weights_sha256"] == sha256_file(out / "weights.safetensors")
    assert man["train_weights_sha256"] == sha256_file(out / TRAIN_FILE)
    assert man["applied_update_ids"] == ["upd-0", "upd-1", "upd-2"]
    assert man["family"] == "hd" and man["hd_config_sha"] == env["manifest"]["hd_config_sha"]
    # the result is itself a valid base: next round trains from the fp16 copy and aggregates against the master
    assert ver.read_manifest(out / "model_version.json")["version_id"] == "v-new"


def test_two_update_mean_and_second_round(env, updates, tmp_path):
    out = tmp_path / "agg2"
    s = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", updates[:2], out,
                     inner_steps=H, lr=LR)
    assert s["aggregation_method"] == "mean"
    # round 2: train from the fp16 train copy of the new version, aggregate with the outer state
    ins = {"base_train_weights": str(out / TRAIN_FILE), "base_manifest": str(out / "model_version.json"),
           "data_shard_000": str(env["shards"][2][0])}
    nv = json.loads((out / "model_version.json").read_text())["version_id"]
    o2 = tmp_path / "r2u"
    hd_train_job(ins, params(env, "r2-0", 2, 5, base_version_id=nv, round=1), o2, attempt_id="a")
    s2 = hd_aggregate(out / "model_version.json", out / "weights.safetensors",
                      [{"path": str(o2 / "update.safetensors"), "manifest_path": str(o2 / "update.json"), "update_id": "r2-0"}],
                      tmp_path / "agg3", outer_state_path=out / "outer_state.safetensors", inner_steps=H, lr=LR)
    assert not s2["no_op"] and s2["applied_update_ids"] == ["r2-0"]
    assert len(json.loads((tmp_path / "agg3" / "model_version.json").read_text())["applied_update_ids"]) == 3


def test_aggregate_rejects_replay_scaled_nan_duplicates(env, updates, tmp_path):
    u0, u1, u2 = updates
    scaled = _tamper(env, u1, "s", lambda k, v: v * 500, "a_scaled")
    nan = _tamper(env, u2, "n", lambda k, v: torch.full_like(v, float("nan")), "a_nan")
    dup = dict(u0, update_id="upd-copy")                       # same bytes under another id: replay of the content
    out = tmp_path / "aggr"
    s = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", [u0, scaled, nan, dup],
                     out, inner_steps=H, lr=LR)
    why = {r["update_id"]: r["reason"] for r in s["rejected"]}
    assert why["upd-1"] == "bound_exceeded" and why["upd-2"] == "non_finite"
    assert s["applied_update_ids"] == ["upd-0"]
    assert any(r["reason"] in ("duplicate_update", "manifest_mismatch") for r in s["rejected"] if r["update_id"] == "upd-copy")
    # replay of an already applied update id against the new version
    first = json.loads((out / "model_version.json").read_text())
    s2 = hd_aggregate(out / "model_version.json", out / "weights.safetensors", [u0], tmp_path / "aggr2",
                      inner_steps=H, lr=LR)
    assert s2["no_op"] and s2["rejected"][0]["reason"] in ("stale_base_version", "duplicate_update")
    assert first["applied_update_ids"] == ["upd-0"]


def test_all_rejected_is_noop(env, updates, tmp_path):
    bad = _tamper(env, updates[0], "z", lambda k, v: torch.zeros_like(v), "a_zero")
    s = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", [bad], tmp_path / "noop",
                     inner_steps=H, lr=LR)
    assert s["no_op"] and s["new_version_id"] is None and s["rejected"][0]["reason"] == "zero_update"


def _norm(u):
    return math.sqrt(sum(float(v.double().pow(2).sum()) for v in load_file(u["path"]).values()))


def test_outlier_median_and_clipping(env, updates, tmp_path):
    n = [_norm(u) for u in updates]
    med = sorted(n)[1]
    others = [updates[1], updates[2]]
    f = 3.0 * med / n[0]          # 3x the median norm: below the 5x outlier cut, above the 2x clip cap
    big = _tamper(env, updates[0], "c", lambda k, v: (v.float() * f).to(torch.float16), "a_x3")
    s = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", [big] + others,
                     tmp_path / "clip", inner_steps=H, lr=LR)
    assert "upd-0" in s["clipped_update_ids"] and s["aggregated_count"] == 3
    g = 9.0 * med / n[0]
    huge = _tamper(env, updates[0], "h", lambda k, v: (v.float() * g).to(torch.float16), "a_x9")
    s = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", [huge] + others,
                     tmp_path / "clip2", inner_steps=H, lr=LR)
    assert any(r["reason"] == "norm_outlier" and r["update_id"] == "upd-0" for r in s["rejected"]) or         any(r["reason"] == "bound_exceeded" for r in s["rejected"])


def test_pi_gate_reports_heldout(env, updates, tmp_path):
    s = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", updates, tmp_path / "pi",
                     inner_steps=H, lr=LR, eval_gate="pi", eval_latents=[str(env["val"])], eval_triples_seed=7)
    assert math.isfinite(s["heldout_loss_before"]) and math.isfinite(s["heldout_loss_after"])
    s2 = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", updates, tmp_path / "pi2",
                      inner_steps=H, lr=LR, eval_gate="pi", eval_latents=[str(env["val"])], eval_triples_seed=7)
    assert s["heldout_loss_before"] == s2["heldout_loss_before"]       # fixed triples


def test_bounds_module():
    b = bounds.elementwise_bound(3, 500, 1e-4)
    assert b == pytest.approx(0.15)
    assert bounds.outlier_cut([1, 1], 5) is None and bounds.outlier_cut([1, 1, 1], 5) == 5
    with pytest.raises(bounds.BoundViolation):
        bounds.check_tensor("t", torch.full((4,), 1.0), 0.1)
    assert bounds.check_tensor("t", torch.full((4,), 0.05), 0.1) == pytest.approx(4 * 0.05 ** 2)


# ---------------------------------------------------------------- latent shard + executor
def test_latent_shard_roundtrip(env):
    sh = LatentShard(env["shards"][0][0])
    assert len(sh) == 6 and sh.clip_shape() == LAT
    z, e, m = sh.get(2)
    assert z.shape == LAT and e.shape == (1, 5, 16) and m.dtype == torch.bool
    for k in ("parent_shard_sha256", "vae_version", "text_encoder_id", "stage"):
        assert k in sh.meta


def _exec(job, tmp_path, name):
    from nivra.executor import Emitter, run_job
    jp = tmp_path / f"{name}.json"
    jp.write_text(json.dumps(job))
    buf = io.StringIO()
    code = run_job(jp, Emitter(buf))
    ev = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
    return code, ev


def test_executor_hd_jobs(env, tmp_path):
    b = env["bundle"]
    p, sha = env["shards"][0]
    pr = params(env, "ex-0", 0, 3, inner_steps=6, warmup_steps=2)
    job = {"protocol": "nivra.exec/1", "job_type": "ai.hd_train", "job_id": "j1", "attempt_id": "attempt-xyz", "params": pr,
           "inputs": {"base_train_weights": str(b / "weights.safetensors"), "base_manifest": str(b / "model_version.json"),
                      "data_shard_000": str(p)}, "output_dir": str(tmp_path / "o1")}
    code, ev = _exec(job, tmp_path, "train")
    assert code == 0, ev
    kinds = {e["kind"] for e in ev if e["event"] == "artifact"}
    assert kinds == {"update", "update_weights"}
    res = next(e for e in ev if e["event"] == "result")["summary"]
    for k in ("loss_first", "loss_last", "samples_seen", "wall_sec", "device_class"):
        assert k in res
    man = json.loads((tmp_path / "o1" / "update.json").read_text())
    assert man["attempt_id"] == "attempt-xyz"
    # unknown role / missing shard input / bad param are exit 2 / 5
    bad = copy.deepcopy(job)
    bad["inputs"]["bogus"] = "x"
    assert _exec(bad, tmp_path, "bad1")[0] == 2
    bad = copy.deepcopy(job)
    bad["params"]["shards"] = [{"artifact_id": "a", "sha256": "0" * 64}]
    assert _exec(bad, tmp_path, "bad2")[0] == 5
    # aggregate
    agg = {"protocol": "nivra.exec/1", "job_type": "ai.hd_aggregate", "job_id": "j2", "params": {
        "outer_lr": 0.7, "momentum": 0.9, "nesterov": True, "version_id": "uuid-1", "bound_c": 3.0, "inner_steps": 6,
        "lr": LR, "eval_gate": "pi", "eval_triples_seed": 5},
        "inputs": {"base_weights": str(b / "weights.safetensors"), "base_manifest": str(b / "model_version.json"),
                   "updates": [{"path": str(tmp_path / "o1" / "update.safetensors"),
                                "manifest_path": str(tmp_path / "o1" / "update.json"), "update_id": "ex-0"}],
                   "eval_latents": [str(env["val"])]}, "output_dir": str(tmp_path / "o2")}
    code, ev = _exec(agg, tmp_path, "agg")
    assert code == 0, ev
    kinds = {e["kind"] for e in ev if e["event"] == "artifact"}
    assert kinds == {"model_manifest", "model_weights", "model_train_weights", "outer_state"}
    res = next(e for e in ev if e["event"] == "result")["summary"]
    assert res["new_version_id"] == "uuid-1" and "heldout_loss_before" in res and res["applied_update_ids"] == ["ex-0"]
    # eval on the new fp16 train copy
    ev_job = {"protocol": "nivra.exec/1", "job_type": "ai.hd_eval", "job_id": "j3",
              "params": {"version_id": "uuid-1", "triples_seed": 5, "n": 4, "timestep_buckets": 2},
              "inputs": {"weights": str(tmp_path / "o2" / TRAIN_FILE), "manifest": str(tmp_path / "o2" / "model_version.json"),
                         "val_shard_000": str(env["val"])}, "output_dir": str(tmp_path / "o3")}
    code, ev = _exec(ev_job, tmp_path, "eval")
    assert code == 0, ev
    rep = json.loads((tmp_path / "o3" / "eval_report.json").read_text())
    assert rep["schema"] == "nivra.hd_eval/1" and rep["version_id"] == "uuid-1" and rep["n"] == 4 and math.isfinite(rep["loss"])
    assert next(e for e in ev if e["event"] == "result")["summary"]["loss"] == rep["loss"]


def test_nivra_base_param_count():
    from nivra.models.dit import count_params_meta
    from nivra.models.hd_config import dit_config
    cfg = load_hd_config("nivra_base")
    n = count_params_meta(dit_config(cfg))["total"]
    assert 150e6 <= n <= 250e6
    assert cfg["default_style"] == "anime" and cfg["domain"] == "stylized"
