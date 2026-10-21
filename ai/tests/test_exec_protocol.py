"""Executor protocol tests: every test runs `python -m nivra.cli exec --job job.json` as a real subprocess."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nivra.inference.video_io import probe_mp4

AI_DIR = Path(__file__).resolve().parent.parent


def run_job(workdir, job_type, params=None, inputs=None, extra=None, name="job", stop=False, raw=None):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    job = {"protocol": "nivra.exec/1", "job_type": job_type, "job_id": "j-" + name, "attempt_id": "a-" + name,
           "params": params or {}, "inputs": inputs or {}, "output_dir": str(workdir / ("out_" + name)),
           "stop_file": str(workdir / "STOP"), "limits": {"max_seconds": 600, "max_ram_mb": 4096, "max_vram_mb": 0}}
    job.update(extra or {})
    if stop:
        (workdir / "STOP").write_text("stop")
    jp = workdir / f"{name}.json"
    jp.write_text(raw if raw is not None else json.dumps(job))
    env = dict(os.environ, PYTHONPATH=str(AI_DIR), NIVRA_THREADS="2", PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "nivra.cli", "exec", "--job", str(jp)], cwd=str(workdir), env=env,
                       capture_output=True, text=True, timeout=600)
    events = []
    for line in p.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    if (workdir / "STOP").exists():
        (workdir / "STOP").unlink()
    return p.returncode, events, Path(job["output_dir"]), p


def last(events):
    return events[-1]


def check_terminal(events):
    terminals = [e for e in events if e["event"] in ("result", "error")]
    assert len(terminals) == 1 and events[-1] is terminals[0]


def artifacts(events, kind=None):
    a = [e for e in events if e["event"] == "artifact"]
    for e in a:
        data = Path(e["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == e["sha256"] and len(data) == e["bytes"]
    return [e for e in a if kind is None or e["kind"] == kind]


# ------------------------------------------------------------------ error paths
def test_bad_protocol_exit_2(tmp_path):
    code, ev, _, _ = run_job(tmp_path, "ai.dataset_shard", {"seed": 1, "start": 0, "count": 2}, extra={"protocol": "nivra.exec/2"})
    assert code == 2 and last(ev)["event"] == "error" and last(ev)["code"] == "INVALID_SPEC"
    check_terminal(ev)


def test_unparseable_job_exit_2(tmp_path):
    code, ev, _, _ = run_job(tmp_path, "ai.dataset_shard", raw="{not json")
    assert code == 2 and last(ev)["code"] == "INVALID_SPEC"


@pytest.mark.parametrize("params,extra", [
    ({"seed": 1, "start": 0, "count": 2, "bogus": 1}, None),             # unknown key
    ({"seed": 1, "start": 0, "count": 0}, None),                        # below min
    ({"seed": 1, "start": 0, "count": 10 ** 9}, None),                  # above max
    ({"seed": "1", "start": 0, "count": 2}, None),                      # wrong type
    ({"seed": True, "start": 0, "count": 2}, None),                     # bool is not int
    ({"start": 0, "count": 2}, None),                                   # missing required
    ({"seed": 1, "start": 0, "count": 2}, {"surprise": 1}),             # unknown top-level key
    ({"seed": 1, "start": 0, "count": 2}, {"job_type": "ai.nope"}),     # unknown job type
])
def test_strict_validation_exit_2(tmp_path, params, extra):
    code, ev, _, _ = run_job(tmp_path, "ai.dataset_shard", params, extra=extra)
    assert code == 2 and last(ev)["code"] == "INVALID_SPEC", ev
    check_terminal(ev)


def test_unknown_input_key_and_missing_required_input(tmp_path):
    code, ev, _, _ = run_job(tmp_path, "ai.dataset_shard", {"seed": 1, "start": 0, "count": 2}, inputs={"x": "y"})
    assert code == 2
    code, ev, _, _ = run_job(tmp_path, "ai.generate", {"prompt": "a red circle moves up"}, inputs={})
    assert code == 2 and "inputs" in last(ev)["message"]


def test_stop_file_exit_3(tmp_path):
    code, ev, _, _ = run_job(tmp_path, "ai.dataset_shard", {"seed": 1, "start": 0, "count": 4}, stop=True)
    assert code == 3 and last(ev)["event"] == "error" and last(ev)["code"] == "STOPPED"
    check_terminal(ev)


def test_missing_input_file_exit_5(tmp_path):
    code, ev, _, _ = run_job(tmp_path, "ai.generate", {"prompt": "a red circle moves up"},
                             inputs={"weights": str(tmp_path / "no.safetensors"), "manifest": str(tmp_path / "no.json"),
                                     "vae_weights": str(tmp_path / "no2")})
    assert code == 5 and last(ev)["code"] == "INPUT_INVALID"


# ------------------------------------------------------------------ happy paths
def test_dataset_shard(tmp_path):
    code, ev, out, _ = run_job(tmp_path, "ai.dataset_shard", {"seed": 5, "start": 10, "count": 6, "frames": 8, "size": 32})
    assert code == 0 and last(ev)["event"] == "result", ev
    check_terminal(ev)
    assert {a["kind"] for a in artifacts(ev)} == {"dataset_shard", "dataset_manifest"}
    m = json.loads((out / "shard.json").read_text())
    assert m["count"] == 6 and len(m["captions"]) == 6 and m["fingerprint"] == last(ev)["summary"]["fingerprint"]
    code2, ev2, _, _ = run_job(tmp_path, "ai.dataset_shard", {"seed": 5, "start": 10, "count": 6, "frames": 8, "size": 32}, name="again")
    assert last(ev2)["summary"]["content_sha256"] == last(ev)["summary"]["content_sha256"]    # deterministic


@pytest.fixture(scope="module")
def model_version(tiny_env, tmp_path_factory):
    """ai.init_model via the executor -> global version 0."""
    wd = tmp_path_factory.mktemp("init")
    code, ev, out, p = run_job(wd, "ai.init_model", {"config": "nivra_tiny", "seed": 0, "vae_checkpoint": str(tiny_env["vae"])})
    assert code == 0, (ev, p.stderr)
    return {"dir": out, "events": ev}


def test_init_model(model_version):
    ev = model_version["events"]
    check_terminal(ev)
    mv = artifacts(ev, "model_version")
    assert len(mv) == 1
    m = json.loads(Path(mv[0]["path"]).read_text())
    assert m["applied_update_ids"] == [] and m["round"] == 0 and m["architecture_version"] == "nivra-small/1"
    assert last(ev)["summary"]["version_id"] == m["version_id"] and m["param_count"] > 0
    assert len(m["vocab_hash"]) == 64 and m["tokenizer_version"]
    assert {a["kind"] for a in artifacts(ev)} == {"model_version", "model_weights", "vae_weights"}


def test_init_model_requires_exactly_one_source(tmp_path):
    code, ev, _, _ = run_job(tmp_path, "ai.init_model", {})
    assert code == 2
    code, ev, _, _ = run_job(tmp_path, "ai.init_model", {"config": "nivra_tiny", "from_checkpoint": "x"})
    assert code == 2
    code, ev, _, _ = run_job(tmp_path, "ai.init_model", {"config": "does_not_exist"})
    assert code == 2


def test_full_distributed_round_through_executor(tiny_env, model_version, tmp_path):
    d = model_version["dir"]
    base = {"base_weights": str(d / "weights.safetensors"), "base_manifest": str(d / "model_version.json"),
            "vae_weights": str(d / "vae.safetensors")}
    outs = []
    for i in range(2):
        code, ev, out, p = run_job(tmp_path, "ai.local_train",
                                   {"update_id": f"upd-{i}", "shard": {"seed": 40 + i, "start": 0, "count": 24},
                                    "inner_steps": 3, "batch_size": 4, "lr": 0.005, "eval_clips": 8, "seed": i},
                                   inputs=base, name=f"lt{i}")
        assert code == 0, (ev, p.stderr)
        check_terminal(ev)
        s = last(ev)["summary"]
        assert s["update_id"] == f"upd-{i}" and s["inner_steps"] == 3 and s["samples"] == 12
        assert s["base_version_id"] == json.loads((d / "model_version.json").read_text())["version_id"]
        assert {"loss_first", "loss_last"} <= set(s)
        assert len(artifacts(ev, "update")) == 1
        assert any(e["event"] == "metric" and e["name"] == "loss" for e in ev)
        outs.append(out)
    # validate
    code, ev, _, _ = run_job(tmp_path, "ai.validate_update", {"max_norm_ratio": 5.0},
                             inputs={"update": str(outs[0] / "update.safetensors"), "update_manifest": str(outs[0] / "update.json"),
                                     "base_manifest": base["base_manifest"], "base_weights": base["base_weights"]}, name="val")
    assert code == 0 and last(ev)["summary"]["valid"] is True, ev
    # aggregate
    ups = [{"path": str(o / "update.safetensors"), "manifest_path": str(o / "update.json")} for o in outs]
    code, ev, out, p = run_job(tmp_path, "ai.aggregate", {"eval_heldout": True},
                               inputs={"base_weights": base["base_weights"], "base_manifest": base["base_manifest"],
                                       "updates": ups, "vae_weights": base["vae_weights"]}, name="agg")
    assert code == 0, (ev, p.stderr)
    s = last(ev)["summary"]
    assert sorted(s["applied_update_ids"]) == ["upd-0", "upd-1"] and s["rejected"] == []
    assert s["new_version_id"] and "heldout_loss_before" in s and "heldout_loss_after" in s
    assert {a["kind"] for a in artifacts(ev)} == {"model_version", "model_weights", "outer_state"}
    # replay: feed the same updates again against the NEW version -> rejected (already applied), no-op
    code, ev, out2, _ = run_job(tmp_path, "ai.aggregate", {},
                                inputs={"base_weights": str(out / "weights.safetensors"), "base_manifest": str(out / "model_version.json"),
                                        "outer_state": str(out / "outer_state.safetensors"), "updates": ups}, name="agg2")
    s2 = last(ev)["summary"]
    assert code == 0 and s2["no_op"] and {r["reason"] for r in s2["rejected"]} == {"duplicate_update_id"}
    # stale: an update computed against the OLD version but with a fresh id
    m_old = json.loads((outs[0] / "update.json").read_text())
    m_old["update_id"] = "fresh-id"
    (outs[0] / "update_fresh.json").write_text(json.dumps(m_old))
    code, ev, _, _ = run_job(tmp_path, "ai.aggregate", {},
                             inputs={"base_weights": str(out / "weights.safetensors"), "base_manifest": str(out / "model_version.json"),
                                     "updates": [{"path": ups[0]["path"], "manifest_path": str(outs[0] / "update_fresh.json")}]},
                             name="agg2b")
    assert code == 0 and last(ev)["summary"]["rejected"][0]["reason"] == "stale_base_version"
    # corrupted base weights -> INPUT_INVALID (5)
    bad = tmp_path / "bad_weights.safetensors"
    shutil.copyfile(base["base_weights"], bad)
    raw = bytearray(bad.read_bytes()); raw[-4] ^= 1; bad.write_bytes(bytes(raw))
    code, ev, _, _ = run_job(tmp_path, "ai.aggregate", {}, inputs={"base_weights": str(bad), "base_manifest": base["base_manifest"],
                                                                  "updates": ups}, name="agg3")
    assert code == 5 and last(ev)["code"] == "INPUT_INVALID"


def test_generate_job_and_frame_count(model_version, tmp_path):
    d = model_version["dir"]
    code, ev, out, p = run_job(tmp_path, "ai.generate", {"prompt": "a red circle moves up", "steps": 3, "seed": 1, "name": "v", "upscale": 2},
                               inputs={"weights": str(d / "weights.safetensors"), "manifest": str(d / "model_version.json"),
                                       "vae_weights": str(d / "vae.safetensors")}, name="gen")
    assert code == 0, (ev, p.stderr)
    check_terminal(ev)
    assert probe_mp4(out / "v.mp4")["frames"] == last(ev)["summary"]["frames"] == 8
    assert {a["kind"] for a in artifacts(ev)} == {"video", "video_metadata"}


def test_stop_during_generate(model_version, tmp_path):
    d = model_version["dir"]
    code, ev, _, _ = run_job(tmp_path, "ai.generate", {"prompt": "a red circle moves up"}, stop=True,
                             inputs={"weights": str(d / "weights.safetensors"), "manifest": str(d / "model_version.json"),
                                     "vae_weights": str(d / "vae.safetensors")}, name="gen_stop")
    assert code == 3


def test_evaluate_modes(model_version, tmp_path):
    d = model_version["dir"]
    code, ev, out, _ = run_job(tmp_path, "ai.evaluate", {"mode": "detector", "num_clips": 6}, name="ev1")
    assert code == 0 and last(ev)["summary"]["all_correct_rate"] == 1.0
    ins = {"weights": str(d / "weights.safetensors"), "manifest": str(d / "model_version.json"), "vae_weights": str(d / "vae.safetensors")}
    code, ev, out, p = run_job(tmp_path, "ai.evaluate", {"mode": "reconstruction", "num_clips": 4}, inputs=ins, name="ev2")
    assert code == 0 and "psnr_db" in last(ev)["summary"], (ev, p.stderr)
    code, ev, out, p = run_job(tmp_path, "ai.evaluate", {"mode": "adherence", "num_prompts": 2, "steps": 2}, inputs=ins, name="ev3")
    assert code == 0 and "all_correct_rate" in last(ev)["summary"], (ev, p.stderr)
    assert json.loads((out / "report.json").read_text())["samples"] == 2
