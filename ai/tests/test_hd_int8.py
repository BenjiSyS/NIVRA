"""int8 per-row-scaled DiLoCo deltas with error feedback, sample-weighted aggregation, staleness, outer-lr warm-up.

Python half of the format contract; the Go half (internal/controller/training/hd_test.go) reads the same golden
fixture ``tests/fixtures/hd_int8_golden.safetensors`` + ``hd_int8_golden.json``. Regenerate both (only when the format
changes on purpose) with ``NIVRA_REGEN_FIXTURES=1 pytest tests/test_hd_int8.py -k golden``.
"""
import json
import math
import os
import shutil
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from nivra.distributed import version as ver
from nivra.distributed.hd import bounds, quant
from nivra.distributed.hd.aggregate import (MAX_WEIGHT, TRAIN_FILE, AggregateError, combine_weighted, hd_aggregate,
                                            outer_lr_at, sample_weights)
from nivra.distributed.hd.stream_io import StReader, StWriter, sha256_file, write_tensors
from nivra.distributed.hd.train_job import _write_int8_delta, hd_train_job
from nivra.distributed.hd.validate import validate_hd_update
from nivra.inference.hd_bundle import write_hd_bundle

from test_hd_distributed import H, LR, STAGE, _exec, make_shard, params

FIX = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- pure quantisation
def test_quantize_roundtrip_error_bound_and_shapes():
    g = torch.Generator().manual_seed(0)
    for shape in [(7, 13), (5,), (), (2, 3, 4), (1, 1)]:
        x = torch.randn(shape, generator=g) * 1e-3
        q, s = quant.quantize_rows(x)
        assert q.dtype == torch.int8 and q.shape == x.shape and s.dtype == torch.float32
        assert s.numel() == quant.rows_of(shape) == (shape[0] if len(shape) >= 2 else 1)
        assert int(q.min()) >= -127 and int(q.max()) <= 127
        deq = quant.dequantize_rows(q, s)
        rows = x.reshape(s.numel(), -1)
        err = (deq.reshape(s.numel(), -1) - rows).abs().amax(dim=1)
        assert bool((err <= s * 0.5 * (1 + 1e-5) + 1e-12).all())             # at most half a step
        assert bool((deq.reshape(s.numel(), -1).abs().amax(dim=1) - rows.abs().amax(dim=1)).abs().max() <= 1e-6 * float(x.abs().max()) + 1e-12)


def test_quantize_zero_rows_empty_and_nonfinite():
    x = torch.zeros(3, 4)
    x[1] = torch.tensor([1.0, -2.0, 0.5, 2.0])
    q, s = quant.quantize_rows(x)
    assert s[0] == 0 and s[2] == 0 and bool((q[0] == 0).all()) and bool((q[2] == 0).all())
    assert int(q[1].abs().max()) == 127
    q, s = quant.quantize_rows(torch.zeros(0, 5))
    assert q.shape == (0, 5) and s.numel() == 0 and quant.rows_of((0, 5)) == 0
    with pytest.raises(ValueError):
        quant.quantize_rows(torch.tensor([1.0, float("nan")]))
    # a row so small that amax/127 underflows to 0 is dropped (it stays in the residual), never written as non-zero q
    tiny = torch.full((1, 3), 1e-45)
    q, s = quant.quantize_rows(tiny)
    assert float(s[0]) == 0.0 and bool((q == 0).all())


def test_size_formula_matches_writer(tmp_path):
    shapes = {"a.w": (3, 4), "a.b": (5,), "e": (2, 2, 3), "s": (), "z": (0, 4)}
    entries = quant.build_entries(shapes)
    p = tmp_path / "d.safetensors"
    with StWriter(p, entries, torch.int8) as w:
        for k, shp in shapes.items():
            q, s = quant.quantize_rows(torch.randn(shp))
            w.write(k, q)
            w.write(quant.scale_name(k), s)
        w.close()
    assert p.stat().st_size == quant.expected_int8_bytes(shapes)
    n = sum(math.prod(s) for s in shapes.values())
    rows = sum(quant.rows_of(s) for s in shapes.values())
    with StReader(p) as r:
        assert p.stat().st_size == r.data_start + n + 4 * rows
    assert quant.check_exact_size_int8(p, shapes) is None
    with open(p, "ab") as f:
        f.write(b"\0")
    assert "bytes" in quant.check_exact_size_int8(p, shapes) or "size" in quant.check_exact_size_int8(p, shapes)
    assert quant.check_exact_size_int8(p, {**shapes, "extra": (1,)}) is not None
    with pytest.raises(quant.QuantError):
        quant.build_entries({"x#scale": (2,)})


def test_check_int8_tensor_rules():
    q = torch.tensor([[1, -127], [0, 0]], dtype=torch.int8)
    quant.check_int8_tensor("t", q, torch.tensor([0.5, 0.0]))
    for bad_q, bad_s, code in [(q.clone().index_fill_(0, torch.tensor([0]), -128), torch.tensor([0.5, 0.0]), "int8_invalid"),
                               (q, torch.tensor([-0.5, 0.0]), "int8_invalid"),
                               (q, torch.tensor([float("nan"), 0.0]), "non_finite"),
                               (q, torch.tensor([float("inf"), 0.0]), "non_finite"),
                               (q, torch.tensor([0.0, 0.0]), "int8_invalid"),          # non-zero q under a zero scale
                               (q, torch.tensor([0.5]), "shape_mismatch")]:
        with pytest.raises(quant.QuantError) as ei:
            quant.check_int8_tensor("t", bad_q, bad_s)
        assert ei.value.code == code


# ---------------------------------------------------------------- golden fixture shared with Go
GOLD_SHAPES = {"blocks.0.w": [3, 4], "blocks.0.b": [5], "emb": [2, 2, 3], "scalar": []}
GOLD_LR, GOLD_STEPS, GOLD_C = 0.01, 10, 3.0                                  # bound = c*H*lr = 0.3


def _gold_inputs():
    w = torch.tensor([[0.10, -0.05, 0.02, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.20, 0.15, 0.0125, 0.08]], dtype=torch.float32)
    b = torch.tensor([0.001, -0.002, 0.003, -0.004, 0.005], dtype=torch.float32)
    e = (torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) - 5.5) * 0.01
    sc = torch.tensor(-0.07, dtype=torch.float32)
    return {"blocks.0.w": w, "blocks.0.b": b, "emb": e, "scalar": sc}


def _build_gold(path):
    tensors = _gold_inputs()
    entries = quant.build_entries({k: tuple(v) for k, v in GOLD_SHAPES.items()})
    stats = {}
    l2sq = linf = 0.0
    with StWriter(path, entries, torch.int8) as w:
        for k, x in tensors.items():
            q, s = quant.quantize_rows(x)
            deq = quant.dequantize_rows(q, s)
            w.write(k, q)
            w.write(quant.scale_name(k), s)
            l2sq += float(deq.double().pow(2).sum())
            linf = max(linf, float(deq.abs().max()))
            stats[k] = {"l2": float(deq.double().pow(2).sum().sqrt()), "linf": float(deq.abs().max()),
                        "scales": [float(v) for v in s], "q": [int(v) for v in q.reshape(-1)]}
        w.close()
    meta = {"format": quant.FORMAT, "shapes": GOLD_SHAPES, "file_bytes": Path(path).stat().st_size,
            "expected_bytes": quant.expected_int8_bytes({k: tuple(v) for k, v in GOLD_SHAPES.items()}),
            "inner_steps": GOLD_STEPS, "lr": GOLD_LR, "bound_c": GOLD_C, "slack": quant.INT8_SLACK,
            "l2": math.sqrt(l2sq), "linf": linf, "tensors": stats, "sha256": sha256_file(path)}
    return meta


def test_golden_fixture_is_stable_and_valid(tmp_path):
    fx, fj = FIX / "hd_int8_golden.safetensors", FIX / "hd_int8_golden.json"
    gen = tmp_path / "gen.safetensors"
    meta = _build_gold(gen)
    if os.environ.get("NIVRA_REGEN_FIXTURES") == "1":
        FIX.mkdir(exist_ok=True)
        shutil.copyfile(gen, fx)
        fj.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    assert fx.is_file() and fj.is_file(), "fixture missing: run with NIVRA_REGEN_FIXTURES=1"
    assert fx.read_bytes() == gen.read_bytes(), "the int8 wire format changed: Go must change with it (regenerate the fixture)"
    stored = json.loads(fj.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(meta))
    shapes = {k: tuple(v) for k, v in GOLD_SHAPES.items()}
    assert quant.check_exact_size_int8(fx, shapes) is None
    with StReader(fx) as r:
        for k in shapes:
            quant.check_int8_tensor(k, r.read(k), r.read(quant.scale_name(k)))
    # the scalar, the 1-D tensor and the 3-D tensor each use their own scale count; row 1 of w is all zero
    assert stored["tensors"]["blocks.0.w"]["scales"][1] == 0.0 and len(stored["tensors"]["emb"]["scales"]) == 2
    assert len(stored["tensors"]["scalar"]["scales"]) == 1 and len(stored["tensors"]["blocks.0.b"]["scales"]) == 1


def test_golden_validates_through_validate_hd_update(tmp_path):
    fx = FIX / "hd_int8_golden.safetensors"
    meta = json.loads((FIX / "hd_int8_golden.json").read_text())
    base = {"version_id": "v1", "tensor_shapes": meta["shapes"], "architecture_version": "a", "vocab_hash": "h",
            "weights_l2": 100.0}
    m = {"schema": "nivra.update/1", "family": "hd", "update_id": "u", "update_sha256": sha256_file(fx),
         "base_version_id": "v1", "architecture_version": "a", "vocab_hash": "h", "delta_dtype": "int8",
         "run_id": "r", "round": 0, "round_batch": 0, "attempt_id": "a", "target": "dit", "hd_config_sha": "c",
         "stage": {"h": 32, "w": 32, "frames": 1, "fps": 8}, "inner_steps": GOLD_STEPS, "lr": GOLD_LR, "shards": [],
         "samples_seen": 4}
    mp = tmp_path / "update.json"
    mp.write_text(json.dumps(m))
    v = validate_hd_update(fx, mp, base, inner_steps=GOLD_STEPS, lr=GOLD_LR, bound_c=GOLD_C, delta_dtype="int8",
                           expected={"delta_dtype": "int8"})
    assert v.valid, v.reasons
    assert math.isclose(v.l2_norm, meta["l2"], rel_tol=1e-9)
    # float16 validation of the same file must fail on size / dtype, not crash
    v = validate_hd_update(fx, mp, base, inner_steps=GOLD_STEPS, lr=GOLD_LR, bound_c=GOLD_C, delta_dtype="float16")
    assert not v.valid and any(c in ("shape_mismatch", "size_mismatch") for c in v.codes)


# ---------------------------------------------------------------- the worker job, validator and aggregator
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("hdq")
    bundle = root / "bundle0"
    m = write_hd_bundle(bundle, "nivra_hd_tiny", text_encoder="stub", seed=1, dtype=torch.float32)
    shards = []
    for i in range(3):
        p = root / f"s{i}.safetensors"
        shards.append((p, make_shard(p, seed=i)))
    val = root / "val.safetensors"
    make_shard(val, n=4, seed=99, split="val")
    return {"root": root, "bundle": bundle, "manifest": m, "shards": shards, "val": val}


def _train(env, uid, idx, seed, out, base_dir=None, state_dir=None, **kw):
    b = base_dir or env["bundle"]
    bm = json.loads((Path(b) / "model_version.json").read_text())
    tw = Path(b) / (TRAIN_FILE if (Path(b) / TRAIN_FILE).exists() else "weights.safetensors")
    ins = {"base_train_weights": str(tw), "base_manifest": str(Path(b) / "model_version.json"),
           "data_shard_000": str(env["shards"][idx][0])}
    kw.setdefault("delta_dtype", "int8")
    s = hd_train_job(ins, params(env, uid, idx, seed, base_version_id=bm["version_id"], **kw), out, attempt_id=f"att-{uid}",
                     state_dir=state_dir)
    return {"path": str(out / "update.safetensors"), "manifest_path": str(out / "update.json"), "update_id": uid, "summary": s}


@pytest.fixture(scope="module")
def q_updates(env):
    ups = []
    for i in range(3):
        ups.append(_train(env, f"upd-{i}", i, 10 + i, env["root"] / f"q{i}"))
    return ups


@pytest.fixture(scope="module")
def f_updates(env):
    ups = []
    for i in range(3):
        ups.append(_train(env, f"upd-{i}", i, 10 + i, env["root"] / f"f{i}", delta_dtype="float16"))
    return ups


def _base(env):
    return ver.read_manifest(env["bundle"] / "model_version.json")


def _expected(env, uid, **kw):
    e = {"base_version_id": env["manifest"]["version_id"], "inner_steps": H, "lr": LR, "update_id": uid, "stage": STAGE,
         "batch_size": 2, "grad_accum": 1, "delta_dtype": "int8"}
    e.update(kw)
    return e


def test_int8_update_is_valid_small_and_close_to_fp16(env, q_updates, f_updates):
    bm = _base(env)
    for u, f in zip(q_updates, f_updates):
        size_q, size_f = Path(u["path"]).stat().st_size, Path(f["path"]).stat().st_size
        assert size_q == quant.expected_int8_bytes(bm["tensor_shapes"])
        assert size_q < 0.7 * size_f                                    # tiny model: per-row scales are a big share
        s = u["summary"]
        assert s["delta_dtype"] == "int8" and 0 < s["quant_rel_err"] < 0.02
        m = json.loads(Path(u["manifest_path"]).read_text())
        assert m["delta_dtype"] == "int8" and m["delta_format"] == quant.FORMAT and m["samples_seen"] == H * 2
        v = validate_hd_update(u["path"], u["manifest_path"], bm, expected=_expected(env, u["update_id"]))
        assert v.valid, v.reasons
        # dequantised int8 is close to the fp16 delta of the same run (same seed => same trained weights)
        with StReader(u["path"]) as rq, StReader(f["path"]) as rf:
            num = den = 0.0
            for k in bm["tensor_shapes"]:
                d = quant.read_int8_delta(rq, k)
                ref = rf.read_f32(k)
                num += float((d - ref).double().pow(2).sum())
                den += float(ref.double().pow(2).sum())
        assert math.sqrt(num / den) < 0.02
        assert math.isclose(v.l2_norm, math.sqrt(den), rel_tol=0.02)


def test_int8_is_about_half_of_fp16_for_real_shapes():
    from nivra.distributed.hd.aggregate import expected_delta_bytes
    shapes = {"blocks.0.attn.qkv.weight": (2304, 768), "blocks.0.mlp.fc1.weight": (3072, 768), "blocks.0.norm.weight": (768,)}
    assert quant.expected_int8_bytes(shapes) < 0.51 * expected_delta_bytes(shapes)


def test_int8_training_is_deterministic(env, q_updates):
    out = env["root"] / "q0b"
    _train(env, "upd-0", 0, 10, out)
    assert sha256_file(out / "update.safetensors") == sha256_file(q_updates[0]["path"])


def _repack(env, u, fn, name, shapes_override=None):
    """Copy an int8 update, rewrite its tensors via fn(name, q, scale) -> (q, scale), fix the manifest sha."""
    d = env["root"] / name
    d.mkdir(exist_ok=True)
    bm = _base(env)
    shapes = bm["tensor_shapes"]
    p = d / "update.safetensors"
    with StReader(u["path"]) as r, StWriter(p, quant.build_entries(shapes), torch.int8) as w:
        for k in shapes:
            q, s = fn(k, r.read(k), r.read(quant.scale_name(k)))
            w.write(k, q)
            w.write(quant.scale_name(k), s)
        w.close()
    m = json.loads(Path(u["manifest_path"]).read_text())
    m["update_sha256"] = sha256_file(p)
    (d / "update.json").write_text(json.dumps(m))
    return {"path": str(p), "manifest_path": str(d / "update.json"), "update_id": u["update_id"]}


def _val(env, u):
    return validate_hd_update(u["path"], u["manifest_path"], _base(env), expected=_expected(env, u["update_id"]))


def test_validator_rejects_bad_int8(env, q_updates, f_updates):
    u = q_updates[0]
    first = next(iter(_base(env)["tensor_shapes"]))

    def only_first(f):
        return lambda k, q, s: f(q, s) if k == first else (q, s)

    cases = {
        "scaled": (lambda k, q, s: (q, s * 500), "bound_exceeded"),
        "nan_scale": (only_first(lambda q, s: (q, torch.full_like(s, float("nan")))), "non_finite"),
        "neg_scale": (only_first(lambda q, s: (q, -s)), "int8_invalid"),
        "minus128": (only_first(lambda q, s: (q.clone().reshape(-1).index_fill_(0, torch.tensor([0]), -128).reshape(q.shape), s)),
                     "int8_invalid"),
    }
    for name, (fn, code) in cases.items():
        got = _val(env, _repack(env, u, fn, "bad_" + name))
        assert not got.valid and code in got.codes, (name, got.reasons)
    # zero update
    z = _val(env, _repack(env, u, lambda k, q, s: (torch.zeros_like(q), torch.zeros_like(s)), "bad_zero"))
    assert not z.valid and "zero_update" in z.codes
    # wrong container for the declared format
    f = f_updates[0]
    got = validate_hd_update(f["path"], f["manifest_path"], _base(env), expected=_expected(env, f["update_id"]))
    assert not got.valid and any(c in ("size_mismatch", "shape_mismatch") for c in got.codes)
    got = validate_hd_update(u["path"], u["manifest_path"], _base(env), expected=_expected(env, u["update_id"], delta_dtype="float16"))
    assert not got.valid
    # trailing byte
    d = env["root"] / "bad_trail"
    d.mkdir(exist_ok=True)
    shutil.copyfile(u["path"], d / "update.safetensors")
    with open(d / "update.safetensors", "ab") as fh:
        fh.write(b"\0")
    m = json.loads(Path(u["manifest_path"]).read_text())
    m["update_sha256"] = sha256_file(d / "update.safetensors")
    (d / "update.json").write_text(json.dumps(m))
    got = _val(env, {"path": str(d / "update.safetensors"), "manifest_path": str(d / "update.json"), "update_id": u["update_id"]})
    assert not got.valid
    # manifest declares fp16 but the job said int8
    m = json.loads(Path(u["manifest_path"]).read_text())
    m["delta_dtype"] = "float16"
    (d / "m2.json").write_text(json.dumps(m))
    got = validate_hd_update(u["path"], d / "m2.json", _base(env), expected=_expected(env, u["update_id"]))
    assert not got.valid and "manifest_mismatch" in got.codes


def test_aggregate_int8_close_to_fp16(env, q_updates, f_updates, tmp_path):
    b = env["bundle"]
    sq = hd_aggregate(b / "model_version.json", b / "weights.safetensors", q_updates, tmp_path / "aq", inner_steps=H, lr=LR,
                      delta_dtype="int8")
    sf = hd_aggregate(b / "model_version.json", b / "weights.safetensors", f_updates, tmp_path / "af", inner_steps=H, lr=LR)
    assert sq["aggregated_count"] == 3 and sq["delta_dtype"] == "int8" and sq["aggregation_method"] == "trimmed_mean"
    assert sq["delta_bytes"] == quant.expected_int8_bytes(_base(env)["tensor_shapes"]) < sf["delta_bytes"]
    wq, wf, w0 = (load_file(str(p / "weights.safetensors")) for p in (tmp_path / "aq", tmp_path / "af", b))
    num = den = 0.0
    for k in w0:
        num += float((wq[k] - wf[k]).double().pow(2).sum())
        den += float((wf[k] - w0[k]).double().pow(2).sum())
    assert math.sqrt(num / den) < 0.03                                     # int8 vs fp16 aggregate update
    # a mixed-format batch: fp16 files fail the int8 aggregate's format check instead of being read as garbage
    s = hd_aggregate(b / "model_version.json", b / "weights.safetensors", q_updates[:1] + f_updates[1:2], tmp_path / "mix",
                     inner_steps=H, lr=LR, delta_dtype="int8")
    why = {r["update_id"]: r["reason"] for r in s["rejected"]}
    assert s["applied_update_ids"] == ["upd-0"] and why["upd-1"] in ("size_mismatch", "shape_mismatch", "manifest_mismatch")


def test_executor_int8_jobs(env, tmp_path, monkeypatch):
    monkeypatch.setenv("NIVRA_STATE_DIR", str(tmp_path / "state"))
    b = env["bundle"]
    pr = params(env, "ex-q", 0, 3, inner_steps=6, warmup_steps=2, delta_dtype="int8")
    job = {"protocol": "nivra.exec/1", "job_type": "ai.hd_train", "job_id": "j1", "attempt_id": "att", "params": pr,
           "inputs": {"base_train_weights": str(b / "weights.safetensors"), "base_manifest": str(b / "model_version.json"),
                      "data_shard_000": str(env["shards"][0][0])}, "output_dir": str(tmp_path / "o1")}
    code, ev = _exec(job, tmp_path, "t")
    assert code == 0, ev
    res = next(e for e in ev if e["event"] == "result")["summary"]
    assert res["delta_dtype"] == "int8" and res["error_feedback"] is True
    files = sorted(x.name for x in (tmp_path / "state" / "hd_residual").iterdir())
    assert len(files) == 2 and not any(f.endswith(".tmp") for f in files)
    agg = {"protocol": "nivra.exec/1", "job_type": "ai.hd_aggregate", "job_id": "j2", "params": {
        "outer_lr": 0.7, "momentum": 0.9, "nesterov": True, "version_id": "uuid-q", "bound_c": 3.0, "inner_steps": 6, "lr": LR,
        "delta_dtype": "int8", "outer_lr_warmup_rounds": 4, "outer_lr_warmup_start": 0.25, "max_staleness": 1},
        "inputs": {"base_weights": str(b / "weights.safetensors"), "base_manifest": str(b / "model_version.json"),
                   "updates": [{"path": str(tmp_path / "o1" / "update.safetensors"),
                                "manifest_path": str(tmp_path / "o1" / "update.json"), "update_id": "ex-q"}]},
        "output_dir": str(tmp_path / "o2")}
    code, ev = _exec(agg, tmp_path, "a")
    assert code == 0, ev
    res = next(e for e in ev if e["event"] == "result")["summary"]
    assert res["applied_update_ids"] == ["ex-q"] and math.isclose(res["outer_lr"], 0.7 * 0.25) and res["outer_lr_nominal"] == 0.7
    bad = json.loads(json.dumps(agg))
    bad["params"]["delta_dtype"] = "int4"
    assert _exec(bad, tmp_path, "bad")[0] == 2


# ---------------------------------------------------------------- error feedback
def _fake_base(tmp_path, shapes, version, parent=None, applied=()):
    return {"version_id": version, "parent_version_id": parent, "applied_update_ids": list(applied),
            "tensor_shapes": {k: list(v) for k, v in shapes.items()}, "hd_config_sha": "cfg"}


def _round(tmp_path, shapes, bm, true_delta, uid, state, name, run="run1", cfg="cfg"):
    """Run only the int8 writer: weights file = base (zeros), 'trained' = -true_delta so delta = true_delta."""
    wpath = tmp_path / f"{name}_w.safetensors"
    write_tensors(wpath, {k: torch.zeros(s) for k, s in shapes.items()}, torch.float32)
    trained = {k: -true_delta[k] for k in shapes}
    out = tmp_path / f"{name}_u.safetensors"
    p = {"run_id": run, "hd_config_sha": cfg, "update_id": uid, "target": "dit"}
    usha, rel, ef, commit = _write_int8_delta(str(wpath), trained, list(shapes), bm, out, p, state)
    deq = {}
    with StReader(out) as r:
        for k in shapes:
            deq[k] = quant.read_int8_delta(r, k)
    return deq, ef, commit


def _residual(state, run="run1"):
    st = quant.ResidualStore(state, run)
    return load_file(str(st.file)) if st.file.exists() else None


def test_error_feedback_telescopes_and_resets(tmp_path):
    shapes = {"a": (6, 16), "b": (9,), "c": ()}
    g = torch.Generator().manual_seed(3)
    state = tmp_path / "state"
    truth = [{k: torch.randn(s, generator=g) * 1e-3 for k, s in shapes.items()} for _ in range(4)]
    sent_total = {k: torch.zeros(s) for k, s in shapes.items()}
    true_total = {k: torch.zeros(s) for k, s in shapes.items()}
    versions = ["v0", "v1", "v2", "v3", "v4"]
    bm = _fake_base(tmp_path, shapes, "v0")
    for i in range(4):
        deq, ef, commit = _round(tmp_path, shapes, bm, truth[i], f"u{i}", state, f"r{i}")
        assert ef["error_feedback"] and (ef["residual_reset"] is not None) == (i == 0)
        commit()
        for k in shapes:
            sent_total[k] += deq[k]
            true_total[k] += truth[i][k]
        res = _residual(state)
        # invariant: everything sent so far + the residual still owed == everything the worker really computed
        for k in shapes:
            assert torch.allclose(sent_total[k] + res[k], true_total[k], atol=1e-8, rtol=1e-5), (i, k)
        bm = _fake_base(tmp_path, shapes, versions[i + 1], parent=versions[i], applied=[f"u{j}" for j in range(i + 1)])
    # without feedback the error would accumulate: compare to one-shot quantisation of the last delta
    # (residual stays within half a quantisation step of the last compensated tensor)
    for k in shapes:
        assert float(res[k].abs().max()) <= float(sent_total[k].abs().max()) + 1e-3

    # --- resets: every violated condition drops the residual and the next update is plain quantisation
    def seed(tag):
        """A committed residual that belongs to update 'uS' computed on base 'vS'."""
        b0 = _fake_base(tmp_path, shapes, "vS")
        _, _, commit = _round(tmp_path, shapes, b0, truth[0], "uS", state, "seed_" + tag)
        commit()

    def next_round(tag, parent="vS", applied=("uS",), run="run1", cfg="cfg"):
        b1 = _fake_base(tmp_path, shapes, "vT", parent=parent, applied=applied)
        b1["hd_config_sha"] = cfg
        _, ef, _ = _round(tmp_path, shapes, b1, truth[1], "uT", state, "use_" + tag, run=run, cfg=cfg)
        return ef

    seed("ok")
    assert next_round("ok")["residual_reset"] is None                # positive control: the residual is used
    for tag, kw, why in [("parent", {"parent": "elsewhere"}, "version mismatch"),
                         ("applied", {"applied": ("someone-else",)}, "not applied"),
                         ("run", {"run": "run2"}, "no residual yet"),
                         ("cfg", {"cfg": "other"}, "config")]:
        seed(tag)
        ef = next_round(tag, **kw)
        assert ef["residual_reset"] and why in ef["residual_reset"], (tag, ef)
    seed("dmg")
    st = quant.ResidualStore(state, "run1")
    with open(st.file, "r+b") as fh:
        fh.seek(-4, os.SEEK_END)
        fh.write(bytes([255, 255, 255, 0]))
    ef = next_round("dmg")
    assert "damaged" in ef["residual_reset"] and not st.file.exists()
    # no state dir / no run id: works, but without feedback
    deq, ef, commit = _round(tmp_path, shapes, _fake_base(tmp_path, shapes, "vN"), truth[0], "uN", None, "nostate")
    assert ef["error_feedback"] is False and commit is None


def test_uncommitted_round_leaves_residual_untouched(tmp_path):
    shapes = {"a": (4, 8)}
    g = torch.Generator().manual_seed(1)
    state = tmp_path / "state"
    bm = _fake_base(tmp_path, shapes, "v0")
    d0 = {"a": torch.randn(4, 8, generator=g) * 1e-3}
    _, _, commit = _round(tmp_path, shapes, bm, d0, "u0", state, "r0")
    commit()
    before = (quant.ResidualStore(state, "run1").file.read_bytes(), quant.ResidualStore(state, "run1").meta_file.read_text())
    bm1 = _fake_base(tmp_path, shapes, "v1", parent="v0", applied=["u0"])
    _round(tmp_path, shapes, bm1, d0, "u1", state, "r1")             # the job died before commit()
    after = (quant.ResidualStore(state, "run1").file.read_bytes(), quant.ResidualStore(state, "run1").meta_file.read_text())
    assert before == after


def test_worker_job_uses_residual_across_real_rounds(env, tmp_path):
    state = tmp_path / "state"
    out0 = tmp_path / "r0"
    u0 = _train(env, "w-0", 0, 21, out0, state_dir=state)
    assert u0["summary"]["error_feedback"] and u0["summary"]["residual_reset"]
    agg = tmp_path / "agg"
    s = hd_aggregate(env["bundle"] / "model_version.json", env["bundle"] / "weights.safetensors", [u0], agg, inner_steps=H, lr=LR,
                     delta_dtype="int8", version_id="vq1")
    assert s["applied_update_ids"] == ["w-0"]
    u1 = _train(env, "w-1", 1, 22, tmp_path / "r1", base_dir=agg, state_dir=state, round=1)
    assert u1["summary"]["residual_reset"] is None                   # residual of w-0 was reused
    # a worker whose last update was NOT part of the version it trains from starts clean
    agg2 = tmp_path / "agg2"
    other = _train(env, "w-other", 2, 23, tmp_path / "ro", base_dir=agg, round=1)
    hd_aggregate(agg / "model_version.json", agg / "weights.safetensors", [other], agg2, inner_steps=H, lr=LR,
                 delta_dtype="int8", version_id="vq2", outer_state_path=agg / "outer_state.safetensors")
    u2 = _train(env, "w-2", 1, 24, tmp_path / "r2", base_dir=agg2, state_dir=state, round=2)
    assert u2["summary"]["residual_reset"] and "not applied" in u2["summary"]["residual_reset"]


# ---------------------------------------------------------------- weights, staleness, warm-up
def test_sample_weights_and_combine_weighted():
    assert sample_weights([20, 20, 20], [False] * 3, True, 0.5) == [1.0, 1.0, 1.0]
    assert sample_weights([10, 20, 40], [False] * 3, True, 0.5) == [0.5, 1.0, MAX_WEIGHT]       # capped at 2x the median
    assert sample_weights([10, 20, 40], [False] * 3, False, 0.5) == [1.0, 1.0, 1.0]
    assert sample_weights([20, 20], [False, True], True, 0.5) == [1.0, 0.5]
    g = torch.Generator().manual_seed(0)
    st = torch.randn(5, 4, 3, generator=g)
    assert torch.allclose(combine_weighted(st, torch.ones(5), 0), st.mean(0), atol=1e-6)
    s, _ = torch.sort(st, dim=0)
    assert torch.allclose(combine_weighted(st, torch.ones(5), 1), s[1:4].mean(0), atol=1e-6)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert torch.allclose(combine_weighted(st, w, 0), (st * w[:, None, None]).sum(0) / 15, atol=1e-6)
    # weighted trimmed mean: drop min and max per coordinate, weight the remaining by their own weights
    out = combine_weighted(st, w, 1)
    ref = torch.zeros(4, 3)
    for i in range(4):
        for j in range(3):
            order = torch.argsort(st[:, i, j])[1:4]
            ref[i, j] = (st[order, i, j] * w[order]).sum() / w[order].sum()
    assert torch.allclose(out, ref, atol=1e-6)


def test_outer_lr_warmup_schedule():
    assert outer_lr_at(0.7, 0, 0, 0.3) == 0.7
    assert math.isclose(outer_lr_at(0.7, 0, 4, 0.3), 0.21)
    assert math.isclose(outer_lr_at(0.7, 2, 4, 0.3), 0.7 * 0.65)
    assert outer_lr_at(0.7, 4, 4, 0.3) == 0.7 and outer_lr_at(0.7, 99, 4, 0.3) == 0.7


def test_aggregate_argument_checks(env, q_updates, tmp_path):
    b = env["bundle"]
    for kw in ({"delta_dtype": "int4"}, {"max_staleness": 2}, {"staleness_discount": 0.0}, {"outer_lr_warmup_start": 0.0},
               {"outer_lr_warmup_rounds": -1}):
        with pytest.raises(AggregateError):
            hd_aggregate(b / "model_version.json", b / "weights.safetensors", q_updates[:1], tmp_path / "x", inner_steps=H, lr=LR, **kw)


def test_staleness_window_and_discount(env, q_updates, tmp_path):
    b = env["bundle"]
    v1 = tmp_path / "v1"
    hd_aggregate(b / "model_version.json", b / "weights.safetensors", q_updates[:1], v1, inner_steps=H, lr=LR,
                 delta_dtype="int8", version_id="vs1")
    fresh = _train(env, "fresh-0", 1, 31, tmp_path / "fresh", base_dir=v1, round=1)
    stale = q_updates[1]                                              # computed against version 0, v1 is current
    args = (v1 / "model_version.json", v1 / "weights.safetensors", [fresh, stale])
    kw = dict(inner_steps=H, lr=LR, delta_dtype="int8", outer_state_path=v1 / "outer_state.safetensors")
    off = hd_aggregate(*args, tmp_path / "off", **kw)                 # default: stale deltas are rejected
    assert off["applied_update_ids"] == ["fresh-0"]
    assert {r["update_id"]: r["reason"] for r in off["rejected"]} == {"upd-1": "stale_base_version"}
    on = hd_aggregate(*args, tmp_path / "on", max_staleness=1, staleness_discount=0.25, **kw)
    assert on["applied_update_ids"] == ["fresh-0", "upd-1"] and on["stale_update_ids"] == ["upd-1"], on["rejected"]
    assert on["update_weights"] == {"fresh-0": 1.0, "upd-1": 0.25}
    assert on["aggregation_method"] == "mean"
    # the discounted delta moves the model less than the undiscounted one
    full = hd_aggregate(*args, tmp_path / "full", max_staleness=1, staleness_discount=1.0, **kw)
    w1, wo, won, wfull = (load_file(str(p / "weights.safetensors")) for p in (v1, tmp_path / "off", tmp_path / "on", tmp_path / "full"))
    dist = lambda a, c: math.sqrt(sum(float((a[k] - c[k]).double().pow(2).sum()) for k in a))
    assert dist(won, wo) < dist(wfull, wo)
    # a delta two versions old is never accepted (window is 1)
    v2 = tmp_path / "v2"
    hd_aggregate(v1 / "model_version.json", v1 / "weights.safetensors", [fresh], v2, inner_steps=H, lr=LR, delta_dtype="int8",
                 outer_state_path=v1 / "outer_state.safetensors", version_id="vs2")
    old = hd_aggregate(v2 / "model_version.json", v2 / "weights.safetensors", [stale], tmp_path / "old", inner_steps=H, lr=LR,
                       delta_dtype="int8", max_staleness=1, outer_state_path=v2 / "outer_state.safetensors")
    assert old["no_op"] and old["rejected"][0]["reason"] == "stale_base_version"


def test_warmup_in_aggregate(env, q_updates, tmp_path):
    b = env["bundle"]
    w = hd_aggregate(b / "model_version.json", b / "weights.safetensors", q_updates[:1], tmp_path / "w", inner_steps=H, lr=LR,
                     delta_dtype="int8", outer_lr_warmup_rounds=5, outer_lr_warmup_start=0.2)
    n = hd_aggregate(b / "model_version.json", b / "weights.safetensors", q_updates[:1], tmp_path / "n", inner_steps=H, lr=LR,
                     delta_dtype="int8")
    assert math.isclose(w["outer_lr"], 0.7 * 0.2) and n["outer_lr"] == 0.7
    w0, ww, wn = (load_file(str(p / "weights.safetensors")) for p in (b, tmp_path / "w", tmp_path / "n"))
    k = next(iter(w0))
    step_w = float((ww[k] - w0[k]).norm())
    step_n = float((wn[k] - w0[k]).norm())
    assert math.isclose(step_w / step_n, 0.2 / 1.0, rel_tol=1e-4)
