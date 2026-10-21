"""Stylized HD bundles: manifest fields, loader attributes, ai.generate `style` param (real subprocess, exit codes),
end-to-end tiny job, stylized evaluation prompt set."""
import json
from pathlib import Path

import pytest

from nivra.data.styles import STYLES
from nivra.evaluation.stylized import load_stylized_prompts
from nivra.inference import hd_bundle as hb
from nivra.inference.bundle import BundleError
from nivra.inference.video_io import probe_mp4

from test_hd_bundle import inputs, run

FAST = {"steps": 2, "width": 64, "height": 32, "seconds": 0.27}


@pytest.fixture(scope="module")
def hd(tmp_path_factory):
    d = tmp_path_factory.mktemp("sty") / "bundle"
    m = hb.write_hd_bundle(d, "nivra_hd_tiny", seed=1)
    return {"dir": d, "manifest": m}


# ------------------------------------------------------------------ manifest + loader
def test_manifest_fields(hd):
    m = json.loads((hd["dir"] / "model_version.json").read_text())
    for k in ("domain", "styles", "default_style"):
        assert k in hb.HD_MANIFEST_FIELDS and k in m
    assert m["domain"] == "stylized" and m["default_style"] == "anime" and m["styles"] == list(STYLES)
    assert m["styles"][-1] == "realistic"
    assert m["config"]["domain"] == "stylized"


def test_loader_exposes_style_info(hd):
    pipe = hb.load_hd_bundle(hd["dir"])
    assert (pipe.domain, pipe.default_style, pipe.styles) == ("stylized", "anime", list(STYLES))
    r = pipe.generate("a red ball rolls", seed=1, frames=9, **{k: FAST[k] for k in ("steps", "width", "height")})
    assert r.meta["style"] == "anime" and r.meta["prompt_used"] == "anime style. a red ball rolls"
    assert r.meta["prompt"] == "a red ball rolls" and r.state.prompt == "anime style. a red ball rolls"
    r2 = pipe.generate("a red ball rolls", seed=1, frames=9, style="none", steps=2, width=64, height=32)
    assert r2.meta["style"] == "none" and r2.meta["prompt_used"] == "a red ball rolls"
    assert not (r.frames == r2.frames).all(), "the style tag reaches the text encoder"
    r3 = pipe.generate("a red ball rolls", seed=1, frames=9, style="Manga-Style", steps=2, width=64, height=32)
    assert (r3.frames == r.frames).all(), "alias == canonical == default here"
    with pytest.raises(ValueError, match="unknown style"):
        pipe.generate("x", style="banana", steps=1, frames=9, width=64, height=32)


def test_continuation_keeps_style_once(hd):
    pipe = hb.load_hd_bundle(hd["dir"])
    first = pipe.generate("a red ball rolls", seed=1, frames=9, style="cartoon", steps=2, width=64, height=32)
    nxt = pipe.continue_video(first.state, steps=2, frames=9)
    assert nxt.meta["prompt_used"] == "cartoon style. a red ball rolls"      # not "anime style. cartoon style. ..."


def test_old_manifest_without_style_fields_loads(hd, tmp_path):
    import shutil
    d = tmp_path / "old"
    shutil.copytree(hd["dir"], d)
    m = json.loads((d / "model_version.json").read_text())
    for k in ("domain", "styles", "default_style"):
        m.pop(k)
    for k in ("domain", "styles", "default_style"):
        m["config"].pop(k)
    (d / "model_version.json").write_text(json.dumps(m))
    pipe = hb.load_hd_bundle(d)
    assert (pipe.domain, pipe.styles, pipe.default_style) == ("general", [], None)
    r = pipe.generate("a red ball", steps=1, frames=9, width=64, height=32)
    assert r.meta["style"] is None and r.meta["prompt_used"] == "a red ball"
    assert pipe.generate("a red ball", style="pixel art", steps=1, frames=9, width=64, height=32).meta["style"] == "pixel art"


def test_tampered_style_fields_refused(hd, tmp_path):
    import shutil
    d = tmp_path / "bad"
    shutil.copytree(hd["dir"], d)
    m = json.loads((d / "model_version.json").read_text())
    m["default_style"] = "banana"
    (d / "model_version.json").write_text(json.dumps(m))
    with pytest.raises(BundleError, match="style"):
        hb.load_hd_bundle(d)


def test_general_config_bundle_has_defaults(tmp_path):
    from nivra.models.hd_config import load_hd_config
    cfg = load_hd_config("nivra_hd_tiny")
    cfg.update(domain="general", styles=[], default_style=None)
    m = hb.write_hd_bundle(tmp_path / "g", cfg, seed=2)
    assert (m["domain"], m["styles"], m["default_style"]) == ("general", [], None)


# ------------------------------------------------------------------ ai.generate via the executor
def test_exec_style_default_explicit_none(hd, tmp_path):
    code, ev, out, p = run(tmp_path, {"prompt": "a red ball rolls", "seed": 4, "name": "d", **FAST}, inputs(hd["dir"]), "d")
    assert code == 0, (ev, p.stderr)
    s = ev[-1]["summary"]
    assert s["style"] == "anime" and s["prompt_used"] == "anime style. a red ball rolls"
    code, ev, out, p = run(tmp_path, {"prompt": "a red ball rolls", "seed": 4, "name": "e", "style": "3D render", **FAST},
                           inputs(hd["dir"]), "e")
    assert code == 0 and ev[-1]["summary"]["style"] == "3d animation"
    assert ev[-1]["summary"]["prompt_used"] == "3d animation style. a red ball rolls"
    assert json.loads((out / "e.json").read_text())["prompt_used"] == "3d animation style. a red ball rolls"
    code, ev, out, p = run(tmp_path, {"prompt": "a red ball rolls", "seed": 4, "name": "n", "style": "none", **FAST},
                           inputs(hd["dir"]), "n")
    assert code == 0 and ev[-1]["summary"]["style"] == "none" and ev[-1]["summary"]["prompt_used"] == "a red ball rolls"
    code, ev, out, p = run(tmp_path, {"prompt": "a cartoon dog runs", "seed": 4, "name": "m", "style": "pixel art", **FAST},
                           inputs(hd["dir"]), "m")
    assert code == 0 and ev[-1]["summary"]["prompt_used"] == "a cartoon dog runs"      # prompt already names a style


def test_exec_unknown_style_exit_2(hd, tmp_path):
    code, ev, out, p = run(tmp_path, {"prompt": "x", "style": "banana", **FAST}, inputs(hd["dir"]), "u")
    assert code == 2 and ev[-1]["event"] == "error" and ev[-1]["code"] == "INVALID_SPEC"
    assert "banana" in ev[-1]["message"] and "style" in ev[-1]["message"]
    assert not list(out.glob("*.mp4"))


def test_exec_style_unsupported_by_model_exit_2(hd, tmp_path):
    import shutil
    d = tmp_path / "narrow"
    shutil.copytree(hd["dir"], d)
    m = json.loads((d / "model_version.json").read_text())
    m["styles"], m["default_style"] = ["anime", "cartoon"], "anime"
    (d / "model_version.json").write_text(json.dumps(m))
    code, ev, _, _ = run(tmp_path / "w", {"prompt": "x", "style": "pixel art", **FAST}, inputs(d), "ns")
    assert code == 2 and "not supported" in ev[-1]["message"]


def test_exec_small_bundle_rejects_style(tiny_env, tmp_path):
    b = tiny_env["bundle"]
    ins = {"weights": str(b / "weights.safetensors"), "manifest": str(b / "model_version.json"),
           "vae_weights": str(b / "vae.safetensors")}
    code, ev, _, _ = run(tmp_path, {"prompt": "a red circle", "style": "anime", "steps": 1}, ins, "sm")
    assert code == 2 and ev[-1]["code"] == "INVALID_SPEC" and "style" in ev[-1]["message"]


def test_exec_end_to_end_mp4_with_style_in_summary(hd, tmp_path):
    code, ev, out, p = run(tmp_path, {"prompt": "a girl with pink hair runs", "seed": 9, "steps": 3, "name": "clip",
                                      "style": "anime"}, inputs(hd["dir"]), "e2e")
    assert code == 0, (ev, p.stderr)
    s = ev[-1]["summary"]
    info = probe_mp4(out / "clip.mp4")
    assert info["frames"] == s["frames"] == 17 and info["width"] == 96 and info["height"] == 64
    assert s["style"] == "anime" and s["prompt_used"] == "anime style. a girl with pink hair runs" and s["family"] == "hd"


# ------------------------------------------------------------------ evaluation prompt set
def test_stylized_prompt_set_shape():
    ps = load_stylized_prompts()
    assert len(ps) == 24 and len({p["id"] for p in ps}) == 24
    assert all(p["lang"] == "en" and p["style"] in STYLES and p["prompt"] for p in ps)
    assert {p["style"] for p in ps} == set(STYLES) - {"realistic"}
    assert all(sum(q["style"] == s for q in ps) == 3 for s in {p["style"] for p in ps})


def test_exec_evaluate_uses_stylized_prompts_for_stylized_bundle(hd, tmp_path):
    code, ev, out, p = _evaluate(hd, tmp_path, {"mode": "adherence", "num_prompts": 2})
    assert code == 0, (ev, p.stderr)
    rep = json.loads((out / "report.json").read_text())
    assert rep["kind"] == "stylized_prompt_set" and rep["domain"] == "stylized" and rep["samples"] == 2
    assert [r["style"] for r in rep["per_sample"]] == ["anime", "anime"]
    assert rep["per_sample"][0]["prompt_used"].startswith("anime style. ")
    assert ev[-1]["summary"]["videos"] == 2 and "no prompt-adherence" in rep["note"]


def _evaluate(hd, tmp_path, params):
    import subprocess
    import sys
    from test_hd_bundle import env
    job = {"protocol": "nivra.exec/1", "job_type": "ai.evaluate", "job_id": "j-ev", "attempt_id": "a-ev", "params": params,
           "inputs": inputs(hd["dir"]), "output_dir": str(tmp_path / "out_ev"), "limits": {"max_seconds": 600}}
    jp = tmp_path / "ev.json"
    jp.write_text(json.dumps(job))
    p = subprocess.run([sys.executable, "-m", "nivra.cli", "exec", "--job", str(jp)], cwd=str(tmp_path), env=env(),
                       capture_output=True, text=True, timeout=900)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    return p.returncode, ev, Path(job["output_dir"]), p
