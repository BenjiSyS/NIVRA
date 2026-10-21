"""HD model bundle: format, sha verification, text-encoder resolution, exec routing (real subprocess,
real ffmpeg/ffprobe), CLI, ComfyUI node functions and small-bundle regressions. All at tiny scale on CPU."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nivra.inference import hd_bundle as hb
from nivra.inference.bundle import BundleError
from nivra.inference.video_io import probe_mp4
from nivra.models.hd_config import load_hd_config

AI_DIR = Path(__file__).resolve().parent.parent
REPO = AI_DIR.parent
GO_FIELDS = ("schema", "version_id", "weights_sha256", "vae_sha256", "config_name", "architecture_version",
             "tokenizer_version", "param_count")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def hd(tmp_path_factory):
    d = tmp_path_factory.mktemp("hd") / "bundle"
    m = hb.write_hd_bundle(d, "nivra_hd_tiny", seed=1)
    return {"dir": d, "manifest": m}


def inputs(d):
    d = Path(d)
    return {"weights": str(d / "weights.safetensors"), "manifest": str(d / "model_version.json"),
            "vae_weights": str(d / "vae.safetensors")}


def env():
    return dict(os.environ, PYTHONPATH=str(AI_DIR), NIVRA_THREADS="2", PYTHONUTF8="1")


def write_job(workdir, params, ins, name="job", stop_file=True):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    job = {"protocol": "nivra.exec/1", "job_type": "ai.generate", "job_id": "j-" + name, "attempt_id": "a-" + name,
           "params": params, "inputs": ins, "output_dir": str(workdir / ("out_" + name)),
           "stop_file": str(workdir / "STOP"), "limits": {"max_seconds": 600}}
    jp = workdir / f"{name}.json"
    jp.write_text(json.dumps(job))
    return jp, Path(job["output_dir"])


def run(workdir, params, ins, name="job"):
    jp, out = write_job(workdir, params, ins, name)
    p = subprocess.run([sys.executable, "-m", "nivra.cli", "exec", "--job", str(jp)], cwd=str(workdir), env=env(),
                       capture_output=True, text=True, timeout=600)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    return p.returncode, ev, out, p


# ------------------------------------------------------------------ manifest / format
def test_manifest_fields_and_hashes(hd):
    d, m = hd["dir"], json.loads((hd["dir"] / "model_version.json").read_text())
    for k in hb.HD_MANIFEST_FIELDS:
        assert k in m, k
    for k in GO_FIELDS:
        assert m[k] not in (None, ""), k
    assert m["schema"] == "nivra.model_version/1" and m["family"] == "hd" and m["architecture_version"] == "nivra-hd/1"
    assert m["weights_sha256"] == sha(d / "weights.safetensors") and m["vae_sha256"] == sha(d / "vae.safetensors")
    assert m["min_vram_mb"] == 0 and m["default_resolution"] == [96, 64] and m["default_fps"] == 30
    assert m["tokenizer_version"] == m["text_encoder"]["id"] and m["text_encoder"]["kind"] == "stub"
    assert m["text_encoder"]["path"] is None and m["param_count"] == 45896
    assert hb.is_hd_manifest(m) and not hb.is_hd_manifest({**m, "family": "small"}) and not hb.is_hd_manifest({})
    assert not (d / "text_encoder").exists()


def test_min_vram_matches_budget_tool():
    spec = importlib.util.spec_from_file_location("hd_budget_t", AI_DIR / "scripts" / "hd_budget.py")
    bud = importlib.util.module_from_spec(spec)
    sys.modules["hd_budget_t"] = bud
    spec.loader.exec_module(bud)
    cfg = load_hd_config("nivra_medium")
    est = bud.estimate("nivra_medium", True)
    row = [r for r in est["rows"] if r["res"] == "720p"][0]
    expect = (2 * est["params"]["total"] + bud.activation_bytes_inference(est["dit"], row["tokens"], 8192)) / 2 ** 20 \
        + bud.CUDA_OVERHEAD_GB * 1024
    assert hb.estimate_min_vram_mb(cfg) == int(-(-expect // 1))
    assert hb.estimate_min_vram_mb(load_hd_config("nivra_hd_tiny")) == 0
    assert hb.estimate_min_vram_mb(load_hd_config("nivra_large")) > hb.estimate_min_vram_mb(cfg) > 3000


def test_load_dir_files_and_generate(hd, tmp_path):
    pipe = hb.load_hd_bundle(hd["dir"])
    r = pipe.generate("a red ball", seed=2, steps=2, frames=9, width=64, height=40)
    assert r.frames.shape == (9, 40, 64, 3) and r.meta["model_version_id"] == hd["manifest"]["version_id"]
    flat = tmp_path / "flat"                       # the controller hands over the three files individually
    flat.mkdir()
    for f in ("weights.safetensors", "model_version.json", "vae.safetensors"):
        shutil.copy(hd["dir"] / f, flat / f)
    p2 = hb.load_hd_bundle((flat / "weights.safetensors", flat / "model_version.json", flat / "vae.safetensors"))
    r2 = p2.generate("a red ball", seed=2, steps=2, frames=9, width=64, height=40)
    assert (r.frames == r2.frames).all()


def test_sha_mismatch_refused(hd, tmp_path):
    for victim, msg in (("weights.safetensors", "weights"), ("vae.safetensors", "VAE")):
        d = tmp_path / victim.split(".")[0]
        shutil.copytree(hd["dir"], d)
        b = bytearray((d / victim).read_bytes())
        b[-1] ^= 0x55
        (d / victim).write_bytes(bytes(b))
        with pytest.raises(BundleError, match=msg):
            hb.load_hd_bundle(d)
    d = tmp_path / "man"
    shutil.copytree(hd["dir"], d)
    m = json.loads((d / "model_version.json").read_text())
    m["vae_sha256"] = "0" * 64
    (d / "model_version.json").write_text(json.dumps(m))
    with pytest.raises(BundleError, match="VAE"):
        hb.load_hd_bundle(d)
    m["family"] = "small"
    (d / "model_version.json").write_text(json.dumps(m))
    with pytest.raises(BundleError, match="not an HD"):
        hb.load_hd_bundle(d)


def test_exec_refuses_tampered_weights(hd, tmp_path):
    d = tmp_path / "bad"
    shutil.copytree(hd["dir"], d)
    b = bytearray((d / "weights.safetensors").read_bytes())
    b[-1] ^= 0x55
    (d / "weights.safetensors").write_bytes(bytes(b))
    code, ev, _, _ = run(tmp_path / "w", {"prompt": "x", "steps": 1}, inputs(d))
    assert code == 5 and ev[-1]["event"] == "error" and ev[-1]["code"] == "INPUT_INVALID"


def test_large_path_streams_fp16_and_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(hb, "LARGE_PARAM_THRESHOLD", 0)          # force the tensor-by-tensor path on the tiny config
    m = hb.write_hd_bundle(tmp_path / "s", "nivra_hd_tiny", seed=3)
    assert m["weights_dtype"] == "float16"
    from safetensors.torch import load_file
    assert next(iter(load_file(str(tmp_path / "s" / "weights.safetensors")).values())).dtype == torch.float16
    pipe = hb.load_hd_bundle(tmp_path / "s")
    r = pipe.generate("x", steps=2, frames=9, width=64, height=32)
    assert r.frames.shape == (9, 32, 64, 3) and (r.frames.std() >= 0)


# ------------------------------------------------------------------ text encoders
@pytest.fixture(scope="module")
def nivra_ckpt(tmp_path_factory):
    from nivra.text.bpe import BPETokenizer
    from nivra.text.nivra_text import NivraTextConfig, NivraTextModel, save_checkpoint
    tok = BPETokenizer()
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("txt") / "enc"
    save_checkpoint(d, NivraTextModel(NivraTextConfig.preset("tiny", tok.vocab_size)), tok)
    return d


def _cfg64():
    cfg = load_hd_config("nivra_hd_tiny")
    cfg["model"]["text_dim"] = 64
    return cfg


def test_nivra_text_encoder_bundled_and_via_env(nivra_ckpt, tmp_path, monkeypatch):
    out = tmp_path / "b"
    m = hb.write_hd_bundle(out, _cfg64(), text_encoder=f"nivra:{nivra_ckpt}")
    te = m["text_encoder"]
    assert te["kind"] == "nivra" and te["path"] == "text_encoder" and (out / "text_encoder" / "model.safetensors").is_file()
    pipe = hb.load_hd_bundle(out)                                      # shipped inside the bundle directory
    assert pipe.text_encoder.id == te["id"]
    assert pipe.generate("a red ball", steps=2, frames=9, width=32, height=32).frames.shape[0] == 9
    flat = tmp_path / "flat"
    flat.mkdir()
    for f in ("weights.safetensors", "model_version.json", "vae.safetensors"):
        shutil.copy(out / f, flat / f)
    files = (flat / "weights.safetensors", flat / "model_version.json", flat / "vae.safetensors")
    monkeypatch.delenv(hb.ENV_TEXT_DIR, raising=False)
    with pytest.raises(BundleError, match="NIVRA_TEXT_ENCODERS_DIR"):
        hb.load_hd_bundle(files)
    root = tmp_path / "encoders"
    shutil.copytree(nivra_ckpt, root / te["name"])
    monkeypatch.setenv(hb.ENV_TEXT_DIR, str(root))
    assert hb.load_hd_bundle(files).text_encoder.id == te["id"]
    # a modified encoder checkpoint is refused
    w = root / te["name"] / "model.safetensors"
    b = bytearray(w.read_bytes())
    b[-1] ^= 0x33
    w.write_bytes(bytes(b))
    with pytest.raises(BundleError, match="sha256"):
        hb.load_hd_bundle(files)


def test_text_encoder_dim_mismatch_and_bad_spec(nivra_ckpt, tmp_path):
    with pytest.raises(BundleError, match="dim"):
        hb.write_hd_bundle(tmp_path / "x", "nivra_hd_tiny", text_encoder=f"nivra:{nivra_ckpt}")
    with pytest.raises(BundleError, match="bad text encoder"):
        hb.write_hd_bundle(tmp_path / "y", "nivra_hd_tiny", text_encoder="whatever")
    with pytest.raises(BundleError, match="escapes"):
        hb.resolve_text_encoder({"id": "x", "kind": "nivra", "path": "../..", "name": None}, tmp_path)
    with pytest.raises(BundleError, match="unsafe"):
        hb.resolve_text_encoder({"id": "x", "kind": "external", "path": None, "name": "../evil"}, tmp_path,
                                text_encoders_dir=tmp_path)


def test_external_text_encoder_resolved_from_env_only(tmp_path, monkeypatch):
    from nivra.text import external as ext

    class FakeBackend:
        dim = 16

        def __init__(self, directory, device="cpu"):
            pass

        def encode(self, prompts, max_len):
            L = min(max(len(p.split()) for p in prompts) + 1, max_len)
            mask = torch.zeros(len(prompts), L, dtype=torch.bool)
            for i, p in enumerate(prompts):
                mask[i, : min(len(p.split()) + 1, L)] = True
            return torch.ones(len(prompts), L, 16), mask

    monkeypatch.setattr(ext, "TransformersBackend", FakeBackend)
    src = tmp_path / "ext_src"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"pretend weights")
    (src / ext.LICENSE_FILE).write_text(json.dumps({"name": "mini-enc", "license": "Apache-2.0", "source_url": "https://x.invalid",
                                                    "sha256": sha(src / "model.safetensors")}))
    out = tmp_path / "b"
    m = hb.write_hd_bundle(out, "nivra_hd_tiny", text_encoder=f"external:{src}")
    te = m["text_encoder"]
    assert te["kind"] == "external" and te["path"] is None and te["license"] == "Apache-2.0"
    assert not (out / "text_encoder").exists()                         # never copied into the bundle
    monkeypatch.delenv(hb.ENV_TEXT_DIR, raising=False)
    with pytest.raises(BundleError, match="not found"):
        hb.load_hd_bundle(out)
    root = tmp_path / "encoders"
    shutil.copytree(src, root / "mini-enc")
    monkeypatch.setenv(hb.ENV_TEXT_DIR, str(root))
    pipe = hb.load_hd_bundle(out)
    assert pipe.text_encoder.id == te["id"]
    assert pipe.generate("x y", steps=2, frames=9, width=32, height=32).frames.shape[0] == 9


# ------------------------------------------------------------------ exec
def test_exec_hd_end_to_end(hd, tmp_path):
    code, ev, out, p = run(tmp_path, {"prompt": "a red ball rolls", "seed": 4, "steps": 3, "name": "clip"}, inputs(hd["dir"]))
    assert code == 0, (ev, p.stderr)
    info = probe_mp4(out / "clip.mp4")
    s = ev[-1]["summary"]
    assert info["frames"] == s["frames"] == 17 and info["width"] == 96 and info["height"] == 64 and info["fps"] == 30
    assert s["fps"] == 30 and s["width"] == 96 and s["height"] == 64 and s["family"] == "hd"
    assert s["seconds_sampling"] >= 0 and s["seconds_decode"] >= 0 and s["peak_rss_mb"] > 0
    arts = [e for e in ev if e["event"] == "artifact"]
    assert {a["kind"] for a in arts} == {"video", "video_metadata"}
    for a in arts:
        assert hashlib.sha256(Path(a["path"]).read_bytes()).hexdigest() == a["sha256"]
    prog = [e for e in ev if e["event"] == "progress"]
    assert {e["stage"] for e in prog} >= {"text", "sample", "decode", "write"}
    assert [e["step"] for e in prog if e["stage"] == "sample"] == [1, 2, 3]
    fr = [e["fraction"] for e in prog]
    assert fr == sorted(fr) and fr[-1] == 1.0
    meta = json.loads((out / "clip.json").read_text())
    assert meta["frames"] == 17 and meta["steps"] == 3 and meta["mp4"] == "clip.mp4"
    assert ev[-1]["event"] == "result" and sum(e["event"] in ("result", "error") for e in ev) == 1


def test_exec_hd_params(hd, tmp_path):
    code, ev, out, p = run(tmp_path, {"prompt": "waves", "steps": 2, "name": "v", "width": 64, "height": 40, "seconds": 0.5,
                                      "fps": 16, "sampler": "euler", "fp16": False, "offload": "none", "upscale": 1,
                                      "cfg_scale": 1.0}, inputs(hd["dir"]))
    assert code == 0, (ev, p.stderr)
    info = probe_mp4(out / "v.mp4")
    assert (info["frames"], info["width"], info["height"], info["fps"]) == (9, 64, 40, 16)
    assert json.loads((out / "v.json").read_text())["sampler"] == "euler"


def test_exec_hd_param_validation(hd, tmp_path):
    ins = inputs(hd["dir"])
    code, ev, _, _ = run(tmp_path, {"prompt": "x", "upscale": 4}, ins, "up")
    assert code == 2 and ev[-1]["code"] == "INVALID_SPEC" and "upscale" in ev[-1]["message"]
    for bad in ({"sampler": "dpm"}, {"offload": "all"}, {"width": 8}, {"seconds": 0}, {"fp16": "yes"}, {"bogus": 1}):
        code, ev, _, _ = run(tmp_path, {"prompt": "x", **bad}, ins, "bad")
        assert code == 2, bad


def test_exec_hd_stop_file_mid_sampling(hd, tmp_path):
    jp, out = write_job(tmp_path, {"prompt": "x", "steps": 200, "name": "s"}, inputs(hd["dir"]), "stopjob")
    stop = tmp_path / "STOP"
    proc = subprocess.Popen([sys.executable, "-m", "nivra.cli", "exec", "--job", str(jp)], cwd=str(tmp_path), env=env(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    events = []
    for line in proc.stdout:
        e = json.loads(line)
        events.append(e)
        if e["event"] == "progress" and e["stage"] == "sample" and not stop.exists():
            stop.write_text("stop")
    proc.wait(timeout=120)
    assert proc.returncode == 3 and events[-1]["event"] == "error" and events[-1]["code"] == "STOPPED"
    assert not (out / "s.mp4").exists()
    assert 1 <= sum(e.get("stage") == "sample" for e in events) < 200


def test_exec_stop_file_before_start(hd, tmp_path):
    (tmp_path / "STOP").write_text("stop")
    code, ev, _, _ = run(tmp_path, {"prompt": "x"}, inputs(hd["dir"]), "pre")
    assert code == 3


# ------------------------------------------------------------------ small-bundle regression
def test_small_bundle_unchanged(tiny_env, tmp_path):
    d = tiny_env["bundle"]
    code, ev, out, p = run(tmp_path, {"prompt": "a red circle moves up", "steps": 2, "seed": 1, "name": "v", "upscale": 2},
                           inputs(d))
    assert code == 0, (ev, p.stderr)
    assert set(ev[-1]["summary"]) == {"frames", "fps", "seconds_sampling", "seconds_decode", "unknown_words"}
    assert probe_mp4(out / "v.mp4")["frames"] == ev[-1]["summary"]["frames"]
    assert {e["kind"] for e in ev if e["event"] == "artifact"} == {"video", "video_metadata"}
    code, ev, _, _ = run(tmp_path, {"prompt": "x", "width": 64}, inputs(d), "hdonly")
    assert code == 2 and "HD" in ev[-1]["message"]


def test_cli_generate_small_and_hd(tiny_env, hd, tmp_path):
    def cli(*a):
        return subprocess.run([sys.executable, "-m", "nivra.cli", *a], cwd=str(tmp_path), env=env(), capture_output=True,
                              text=True, timeout=600)
    p = cli("generate", "--bundle", str(tiny_env["bundle"]), "--prompt", "a red circle", "--out", str(tmp_path / "s"),
            "--steps", "2", "--upscale", "1")
    assert p.returncode == 0, p.stderr
    assert probe_mp4(tmp_path / "s" / "video.mp4")["frames"] == json.loads(p.stdout.splitlines()[-1])["frames"]
    assert cli("generate", "--bundle", str(tiny_env["bundle"]), "--prompt", "x", "--out", str(tmp_path / "s2"),
               "--sampler", "euler").returncode == 2
    p = cli("generate", "--bundle", str(hd["dir"]), "--prompt", "a red ball", "--out", str(tmp_path / "h"), "--name", "m",
            "--steps", "2", "--width", "64", "--height", "40", "--seconds", "0.5", "--fps", "16", "--sampler", "euler")
    assert p.returncode == 0, p.stderr
    s = json.loads(p.stdout.splitlines()[-1])
    info = probe_mp4(tmp_path / "h" / "m.mp4")
    assert s["frames"] == info["frames"] == 9 and (info["width"], info["height"]) == (64, 40) and s["mp4"] == "m.mp4"
    assert cli("generate", "--bundle", str(hd["dir"]), "--prompt", "x", "--out", str(tmp_path / "h2"), "--upscale", "4").returncode == 2
    assert cli("generate", "--bundle", str(hd["dir"]), "--prompt", "x", "--out", str(tmp_path / "h3"), "--steps", "1",
               "--upscale", "1").returncode == 0


def test_cli_init_hd_bundle(tmp_path):
    p = subprocess.run([sys.executable, "-m", "nivra.cli", "init-hd-bundle", "--config", "nivra_hd_tiny", "--out",
                        str(tmp_path / "b"), "--seed", "5"], cwd=str(tmp_path), env=env(), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout)
    assert r["param_count"] == 45896 and r["min_vram_mb"] == 0
    assert hb.load_hd_bundle(tmp_path / "b").model_version_id == r["version_id"]
    bad = subprocess.run([sys.executable, "-m", "nivra.cli", "init-hd-bundle", "--out", str(tmp_path / "c"), "--text-encoder",
                          "nivra:" + str(tmp_path / "nope")], cwd=str(tmp_path), env=env(), capture_output=True, text=True)
    assert bad.returncode != 0


# ------------------------------------------------------------------ ComfyUI nodes (no ComfyUI installed)
def test_comfyui_nodes_with_hd_bundle(hd, tiny_env, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "integrations" / "comfyui"))
    monkeypatch.delitem(sys.modules, "folder_paths", raising=False)
    from nivra_comfyui import nodes
    assert nodes.folder_paths is None
    models = tmp_path / "models"
    shutil.copytree(hd["dir"], models / "hdtiny")
    shutil.copytree(tiny_env["bundle"], models / "small")
    monkeypatch.setattr(nodes, "_MODELS_DIR", str(models))
    monkeypatch.setattr(nodes, "_output_dir", lambda: str(tmp_path / "out"))
    assert set(nodes._bundles()) == {"hdtiny", "small"}
    # existing workflows still validate: required inputs unchanged, new inputs optional only
    t2v = nodes.NivraTextToVideo.INPUT_TYPES()
    assert set(t2v["required"]) == {"model", "prompt", "seed", "steps", "cfg_scale", "decode_tile"}
    assert set(t2v["optional"]) == {"width", "height", "seconds", "fps", "sampler"}
    (model,) = nodes.NivraModelLoader().load("hdtiny", "cpu")
    assert model.is_hd
    frames, state = nodes.NivraTextToVideo().run(model, "a red ball", 1, 2, 3.0, 0)
    assert frames.shape == (17, 64, 96, 3) and frames.dtype == torch.float32 and 0 <= frames.min() <= frames.max() <= 1
    frames, state = nodes.NivraTextToVideo().run(model, "a red ball", 1, 2, 3.0, 0, width=64, height=64, seconds=0.5, fps=16,
                                                 sampler="euler")
    assert frames.shape == (9, 64, 64, 3)
    f2, st2 = nodes.NivraVideoContinuation().run(model, state, 1, prompt="")
    assert f2.shape[0] > 0 and f2.shape[1:] == (64, 64, 3)
    (path,) = nodes.NivraExportMP4().save(frames, 16, 1, "hd")["result"]
    assert probe_mp4(path)["frames"] == 9
    with pytest.raises(ValueError, match="HD"):
        nodes.NivraImageToVideo().run(model, torch.zeros(1, 8, 8, 3), "x", 0, 1, 3.0)
    (small,) = nodes.NivraModelLoader().load("small", "cpu")
    assert not small.is_hd
    f3, _ = nodes.NivraTextToVideo().run(small, "a red circle moves up", 0, 2, 3.0, 0, width=640, sampler="euler")
    assert f3.ndim == 4 and f3.shape[-1] == 3                           # HD-only inputs are ignored for small models
