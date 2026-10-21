import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from nivra.distributed import version as ver
from nivra.distributed.aggregate import aggregate, combine, outer_step
from nivra.distributed.update import write_update, validate_update


def _base(env):
    d = Path(env["bundle"])
    m = ver.read_manifest(d / ver.MANIFEST_FILE)
    return d, m, load_file(str(d / ver.WEIGHTS_FILE))


def make_update(root, name, m, uid, value, base_version_id=None, nan=False):
    """Hand-made pseudo-gradient: every coordinate equals `value`."""
    delta = {k: torch.full(tuple(s), float(value)) for k, s in m["tensor_shapes"].items()}
    if nan:
        first = next(iter(delta))
        delta[first].view(-1)[0] = float("nan")
    meta = {"update_id": uid, "base_version_id": base_version_id or m["version_id"],
            "base_weights_sha256": m["weights_sha256"], "architecture_version": m["architecture_version"],
            "vocab_hash": m["vocab_hash"], "inner_steps": 1, "samples": 1}
    write_update(Path(root) / name, delta, meta)
    d = Path(root) / name
    return {"path": d / "update.safetensors", "manifest_path": d / "update.json"}


def run_agg(env, root, ups, out, base_dir=None, base_m=None, outer=None, **kw):
    d = base_dir or Path(env["bundle"])
    return aggregate(d / ver.MANIFEST_FILE, d / ver.WEIGHTS_FILE, ups, out, outer_state_path=outer, **kw)


def test_two_updates_mean_and_nesterov_outer_step(tiny_env, tmp_path):
    d, m, base = _base(tiny_env)
    u1 = make_update(tmp_path, "u1", m, "a", 0.01)
    u2 = make_update(tmp_path, "u2", m, "b", 0.03)
    s = run_agg(tiny_env, tmp_path, [u1, u2], tmp_path / "v1")
    assert s["aggregation_method"] == "mean" and s["aggregated_count"] == 2 and not s["rejected"]
    assert s["applied_update_ids"] == ["a", "b"]
    new = load_file(str(tmp_path / "v1" / "weights.safetensors"))
    D = 0.02
    expected_shift = 0.7 * (D + 0.9 * D)           # outer lr 0.7, nesterov, zero initial momentum
    for k in base:
        assert torch.allclose(base[k] - new[k], torch.full_like(base[k], expected_shift), atol=1e-6), k
    m1 = ver.read_manifest(tmp_path / "v1" / "model_version.json")
    assert m1["round"] == 1 and m1["parent_version_id"] == m["version_id"] and m1["applied_update_ids"] == ["a", "b"]
    assert m1["weights_sha256"] == ver.sha256_file(tmp_path / "v1" / "weights.safetensors")
    # round 2 with persisted outer momentum
    v1 = tmp_path / "v1"
    u3 = make_update(tmp_path, "u3", m1, "c", 0.02)
    s2 = run_agg(tiny_env, tmp_path, [u3], tmp_path / "v2", base_dir=v1, outer=v1 / "outer_state.safetensors")
    new2 = load_file(str(tmp_path / "v2" / "weights.safetensors"))
    m_prev, g = D, 0.02
    mom = 0.9 * m_prev + g
    shift = 0.7 * (g + 0.9 * mom)
    for k in new:
        assert torch.allclose(new[k] - new2[k], torch.full_like(new[k], shift), atol=1e-6)
    m2 = ver.read_manifest(tmp_path / "v2" / "model_version.json")
    assert m2["applied_update_ids"] == ["a", "b", "c"] and m2["round"] == 2


def test_stale_update_rejected(tiny_env, tmp_path):
    d, m, _ = _base(tiny_env)
    stale = make_update(tmp_path, "s", m, "stale1", 0.1, base_version_id="v000099-old")
    s = run_agg(tiny_env, tmp_path, [stale], tmp_path / "o")
    assert s["no_op"] and s["new_version_id"] is None
    assert s["rejected"][0]["reason"] == "stale_base_version" and s["rejected"][0]["update_id"] == "stale1"
    assert not (tmp_path / "o" / "weights.safetensors").exists()


def test_duplicate_update_id_rejected_within_and_across_versions(tiny_env, tmp_path):
    d, m, _ = _base(tiny_env)
    a1 = make_update(tmp_path, "a1", m, "same", 0.01)
    a2 = make_update(tmp_path, "a2", m, "same", 0.02)
    s = run_agg(tiny_env, tmp_path, [a1, a2], tmp_path / "v1")
    assert s["applied_update_ids"] == ["same"] and s["rejected"][0]["reason"] == "duplicate_update_id"
    m1 = ver.read_manifest(tmp_path / "v1" / "model_version.json")
    again = make_update(tmp_path, "a3", m1, "same", 0.01)          # right base version, but id already applied
    s2 = run_agg(tiny_env, tmp_path, [again], tmp_path / "v2", base_dir=tmp_path / "v1")
    assert s2["no_op"] and s2["rejected"][0]["reason"] == "duplicate_update_id"


def test_nan_and_wrong_shape_rejected(tiny_env, tmp_path):
    d, m, _ = _base(tiny_env)
    bad = make_update(tmp_path, "nan", m, "nan1", 0.1, nan=True)
    good = make_update(tmp_path, "ok", m, "ok1", 0.01)
    s = run_agg(tiny_env, tmp_path, [bad, good], tmp_path / "v1")
    assert s["applied_update_ids"] == ["ok1"]
    assert {r["update_id"]: r["reason"] for r in s["rejected"]} == {"nan1": "non_finite"}
    new = load_file(str(tmp_path / "v1" / "weights.safetensors"))
    assert all(torch.isfinite(v).all() for v in new.values())
    # wrong shape
    delta = {k: torch.zeros(tuple(sh)) for k, sh in m["tensor_shapes"].items()}
    first = next(iter(delta))
    delta[first] = torch.zeros(3)
    meta = {"update_id": "shape1", "base_version_id": m["version_id"], "base_weights_sha256": m["weights_sha256"],
            "architecture_version": m["architecture_version"], "vocab_hash": m["vocab_hash"], "inner_steps": 1, "samples": 1}
    write_update(tmp_path / "sh", delta, meta)
    v = validate_update(tmp_path / "sh" / "update.safetensors", tmp_path / "sh" / "update.json", m)
    assert not v.valid and "shape_mismatch" in v.codes


def test_tampered_update_file_rejected(tiny_env, tmp_path):
    d, m, _ = _base(tiny_env)
    u = make_update(tmp_path, "t", m, "t1", 0.1)
    raw = bytearray(u["path"].read_bytes())
    raw[-5] ^= 0xFF
    u["path"].write_bytes(bytes(raw))
    v = validate_update(u["path"], u["manifest_path"], m)
    assert not v.valid and v.codes[0] == "checksum_mismatch"


def test_norm_outlier_and_trimmed_mean(tiny_env, tmp_path):
    d, m, base = _base(tiny_env)
    ups = [make_update(tmp_path, f"u{i}", m, f"id{i}", v) for i, v in enumerate((0.01, 0.02, 0.03))]
    ups.append(make_update(tmp_path, "huge", m, "huge", 500.0))
    s = run_agg(tiny_env, tmp_path, ups, tmp_path / "v1")
    assert {r["update_id"]: r["reason"] for r in s["rejected"]} == {"huge": "norm_outlier"}
    assert s["aggregation_method"] == "trimmed_mean" and s["aggregated_count"] == 3
    # trimmed mean of [1,2,100] per coordinate is 2 (drops min and max)
    out, used = combine([{"w": torch.tensor([1.0, 5.0])}, {"w": torch.tensor([2.0, 4.0])}, {"w": torch.tensor([100.0, -9.0])}])
    assert used == "trimmed_mean" and torch.allclose(out["w"], torch.tensor([2.0, 4.0]))
    out2, used2 = combine([{"w": torch.tensor([1.0])}, {"w": torch.tensor([3.0])}])
    assert used2 == "mean" and out2["w"].item() == 2.0


def test_outer_step_zero_momentum_is_plain_sgd():
    base = {"w": torch.ones(3)}
    new, buf = outer_step(base, {"w": torch.full((3,), 0.5)}, None, lr=1.0, momentum=0.0)
    assert torch.allclose(new["w"], torch.full((3,), 0.5))


def test_aggregator_dependency_hygiene():
    code = ("import sys; import nivra.distributed.aggregate, nivra.distributed.update, nivra.distributed.version;"
            "bad=[m for m in sys.modules if m.startswith('nivra.') and m.split('.')[1] not in ('distributed',)];"
            "print('BAD', bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    assert "BAD []" in out.stdout, out.stdout


def test_real_local_train_produces_valid_update_and_aggregates(tiny_env, tmp_path):
    from nivra.distributed.local import local_train
    d, m, base = _base(tiny_env)
    meta = local_train(d / ver.WEIGHTS_FILE, d / ver.MANIFEST_FILE, tiny_env["vae"], {"seed": 3, "start": 0, "count": 24},
                       "real1", tmp_path / "lt", inner_steps=4, batch_size=4, lr=1e-2, eval_clips=8, seed=1)
    assert meta["inner_steps"] == 4 and meta["samples"] == 16 and meta["base_version_id"] == m["version_id"]
    assert meta["loss_before"] is not None and meta["loss_after"] is not None and meta["delta_l2"] > 0
    ents = {"path": tmp_path / "lt" / "update.safetensors", "manifest_path": tmp_path / "lt" / "update.json"}
    v = validate_update(ents["path"], ents["manifest_path"], m, base_weights=base)
    assert v.valid, v.reasons
    delta = load_file(str(ents["path"]))
    assert any(t.abs().max() > 0 for t in delta.values())
    s = run_agg(tiny_env, tmp_path, [ents], tmp_path / "v1")
    assert s["applied_update_ids"] == ["real1"]


# ------------------------------------------------------------------ hardening (absolute bounds, resource limits)
def _raw_update(root, name, m, delta, uid, **meta_over):
    meta = {"update_id": uid, "base_version_id": m["version_id"], "base_weights_sha256": m["weights_sha256"],
            "architecture_version": m["architecture_version"], "vocab_hash": m["vocab_hash"], "inner_steps": 1, "samples": 1}
    meta.update(meta_over)
    write_update(Path(root) / name, delta, meta)
    d = Path(root) / name
    return {"path": d / "update.safetensors", "manifest_path": d / "update.json"}


def test_absolute_norm_bound_without_peers(tiny_env, tmp_path):
    d, m, base = _base(tiny_env)
    huge = make_update(tmp_path, "huge", m, "huge1", 1e3)             # single update, zero peers
    v = validate_update(huge["path"], huge["manifest_path"], m)
    assert not v.valid and "norm_outlier" in v.codes
    s = run_agg(tiny_env, tmp_path, [huge], tmp_path / "o")
    assert s["no_op"] and s["rejected"][0]["reason"] == "norm_outlier"
    ok = make_update(tmp_path, "ok", m, "ok1", 0.01)
    assert validate_update(ok["path"], ok["manifest_path"], m).valid
    assert not validate_update(ok["path"], ok["manifest_path"], m, max_rel_norm=1e-4).valid   # configurable


def test_single_tensor_blowup_rejected(tiny_env, tmp_path):
    d, m, base = _base(tiny_env)
    delta = {k: torch.zeros(tuple(sh)) for k, sh in m["tensor_shapes"].items()}
    first = next(iter(delta))
    delta[first].view(-1)[0] = 1e30
    u = _raw_update(tmp_path, "blow", m, delta, "blow1")
    v = validate_update(u["path"], u["manifest_path"], m)
    assert not v.valid and "tensor_norm_outlier" in v.codes
    assert "tensor_norm_outlier" in validate_update(u["path"], u["manifest_path"], m, base_weights=base).codes


def test_zero_update_rejected(tiny_env, tmp_path):
    d, m, _ = _base(tiny_env)
    z = _raw_update(tmp_path, "z", m, {k: torch.zeros(tuple(sh)) for k, sh in m["tensor_shapes"].items()}, "zero1")
    assert "zero_update" in validate_update(z["path"], z["manifest_path"], m).codes
    s = run_agg(tiny_env, tmp_path, [z], tmp_path / "o")
    assert s["no_op"] and s["rejected"][0]["reason"] == "zero_update"


def test_aggregate_clips_to_twice_median_norm(tiny_env, tmp_path):
    d, m, base = _base(tiny_env)
    ups = [make_update(tmp_path, f"c{i}", m, f"c{i}", v) for i, v in enumerate((0.01, 0.01, 0.04))]
    s = run_agg(tiny_env, tmp_path, ups, tmp_path / "v1", aggregation="mean", momentum=0.0, nesterov=False, outer_lr=1.0)
    assert s["clipped_update_ids"] == ["c2"]                              # 0.04 > 2 x median(0.01)
    new = load_file(str(tmp_path / "v1" / "weights.safetensors"))
    k = next(iter(base))
    assert torch.allclose(base[k] - new[k], torch.full_like(base[k], (0.01 + 0.01 + 0.02) / 3), atol=1e-6)


def test_oversized_manifest_rejected_before_parsing(tiny_env, tmp_path):
    d, m, _ = _base(tiny_env)
    u = make_update(tmp_path, "big", m, "big1", 0.01)
    u["manifest_path"].write_text('{"x": "' + "a" * (2 << 20) + '"}')
    v = validate_update(u["path"], u["manifest_path"], m)
    assert not v.valid and v.codes == ["manifest_too_large"]
    s = run_agg(tiny_env, tmp_path, [u], tmp_path / "o")
    assert s["no_op"] and s["rejected"][0]["reason"] == "manifest_too_large"
    big = tmp_path / "model_version.json"
    big.write_text('{"x": "' + "a" * (2 << 20) + '"}')
    with pytest.raises(ver.ManifestTooLarge):
        ver.read_manifest(big)


def test_header_checked_before_tensors_are_loaded(tiny_env, tmp_path, monkeypatch):
    import nivra.distributed.update as upd
    from safetensors.torch import save_file
    d, m, _ = _base(tiny_env)
    delta = {k: torch.zeros(tuple(sh)) for k, sh in m["tensor_shapes"].items()}
    delta[next(iter(delta))] = torch.ones(7)
    u = _raw_update(tmp_path, "hdr", m, delta, "hdr1")
    u2 = _raw_update(tmp_path, "h16", m, {k: torch.zeros(tuple(sh)) for k, sh in m["tensor_shapes"].items()}, "h16")
    save_file({k: torch.zeros(tuple(sh), dtype=torch.float16) for k, sh in m["tensor_shapes"].items()}, str(u2["path"]))
    mj = json.loads(u2["manifest_path"].read_text())
    mj["update_sha256"] = ver.sha256_file(u2["path"])
    u2["manifest_path"].write_text(json.dumps(mj))
    u3 = make_update(tmp_path, "bud", m, "bud1", 0.01)

    def boom(*a, **k):
        raise AssertionError("tensors were loaded before the header check")
    monkeypatch.setattr(upd, "load_file", boom)
    v = validate_update(u["path"], u["manifest_path"], m)
    assert not v.valid and v.codes == ["shape_mismatch"]
    v2 = validate_update(u2["path"], u2["manifest_path"], m)
    assert not v2.valid and "F32" in v2.reasons[0][1]
    assert "size_limit" in validate_update(u3["path"], u3["manifest_path"], m, max_bytes=1000).codes


def test_executor_oversized_manifest_exit_5(tiny_env, tmp_path):
    from test_exec_protocol import run_job, last
    d, m, _ = _base(tiny_env)
    u = make_update(tmp_path, "x", m, "x1", 0.01)
    big = tmp_path / "big.json"
    big.write_text('{"x": "' + "a" * (2 << 20) + '"}')
    code, ev, _, _ = run_job(tmp_path, "ai.validate_update", {}, {"update": str(u["path"]), "update_manifest": str(big),
                                                                   "base_manifest": str(d / "model_version.json")})
    assert code == 5 and last(ev)["code"] == "INPUT_INVALID"
    code, ev, _, _ = run_job(tmp_path, "ai.validate_update", {}, {"update": str(u["path"]), "update_manifest": str(u["manifest_path"]),
                                                                   "base_manifest": str(big)}, name="b2")
    assert code == 5
