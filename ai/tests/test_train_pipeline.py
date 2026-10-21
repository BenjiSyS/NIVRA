"""train-pipeline: captioned clips -> ingest -> BPE -> NivraText -> HC-VAE -> latents -> DiT -> loadable bundle."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nivra.cli import main as cli_main
from nivra.data.ingest.pipeline import ShardWriter
from nivra.distributed.hd.latent_shard import LatentShard
from nivra.inference.hd_bundle import load_hd_bundle
from nivra.training.hd_pipeline import (STAGES, Pipeline, PipelineError, PipelineOptions, fit_stages, scale_stages,
                                        text_config_for)

CAPS = ["a red ball bounces on grass", "an anime girl runs through rain", "a blue cat sleeps on a sofa",
        "two knights fight on a bridge", "a small robot waves hello", "a dragon flies over mountains"]


def make_dataset(root: Path, n=8, h=64, w=96, frames=17) -> Path:
    """An ingest-format dataset (dataset.json + manifest.jsonl + clip shards) built without ffmpeg."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    wr = ShardWriter(root, "train", (frames, h, w, 3), 1 << 24)
    rows = []
    for i in range(n):
        base = rng.integers(0, 255, (1, 1, 1, 3), dtype=np.uint8)
        clip = np.clip(base + rng.integers(-20, 20, (frames, h, w, 3)), 0, 255).astype(np.uint8)
        cid = f"c{i:03d}"
        shard, idx = wr.add(clip, {"id": cid, "caption": CAPS[i % len(CAPS)], "style": "anime", "lang": "en"})
        rows.append({"id": cid, "split": "train", "shard": shard, "index": idx, "caption": CAPS[i % len(CAPS)], "style": "anime",
                     "lang": "en", "bucket": "test", "fps": 8.0, "frames": frames, "width": w, "height": h})
    wr.flush()
    mtext = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (root / "manifest.jsonl").write_bytes(mtext.encode())
    (root / "dataset.json").write_text(json.dumps({
        "format": "nivra-dataset/1", "fingerprint": hashlib.sha256(mtext.encode()).hexdigest(), "bucket": "test", "fps": 8.0,
        "frames": frames, "width": w, "height": h, "clips": n, "train_clips": n, "val_clips": 0, "styles": {"anime": n},
        "shards": wr.files}))
    return root


def opts(tmp_path, dataset, **kw):
    d = dict(config="nivra_hd_tiny", out=str(tmp_path / "run"), dataset=str(dataset), vocab_size=300, text_steps=3,
             vae_steps=4, dit_steps=6, seed=1, device="cpu", save_every=2)
    d.update(kw)
    return PipelineOptions(**d)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("trainds"))


@pytest.fixture(scope="module")
def finished(ds, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    logs = []
    pipe = Pipeline(opts(tmp, ds, install_dir=str(tmp / "install"), name="my-anime"), log=logs.append)
    res = pipe.run()
    return {"tmp": tmp, "res": res, "logs": logs, "pipe": pipe}


def test_end_to_end_produces_a_loadable_bundle(finished):
    r = finished["res"]
    bundle = Path(r["bundle"]["dir"])
    assert bundle == finished["tmp"] / "install" / "models" / "my-anime"          # where the app looks
    m = json.loads((bundle / "model_version.json").read_text())
    assert m["family"] == "hd" and m["domain"] == "stylized" and m["default_style"] == "anime"
    assert m["text_encoder"]["kind"] == "nivra"
    assert (bundle / "text_encoder" / "model.safetensors").exists()
    pipe = load_hd_bundle(bundle, "cpu")
    assert pipe.default_style == "anime"
    # the VAE latent calibration travelled into the bundle config
    assert m["config"]["vae"]["latent_scale"] == r["vae"]["latent_norm"]["latent_scale"]
    assert r["text"]["last_loss"] <= r["text"]["first_loss"] * 1.5


def test_bundle_generates_through_the_executor(finished, tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    from nivra.executor import Emitter, run_job
    import io
    b = Path(finished["res"]["bundle"]["dir"])
    job = {"protocol": "nivra.exec/1", "job_type": "ai.generate", "job_id": "g1",
           "params": {"prompt": "a red ball bounces", "steps": 2, "name": "clip"},
           "inputs": {"weights": str(b / "weights.safetensors"), "manifest": str(b / "model_version.json"),
                      "vae_weights": str(b / "vae.safetensors")}, "output_dir": str(tmp_path / "gen")}
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job))
    buf = io.StringIO()
    assert run_job(jp, Emitter(buf)) == 0, buf.getvalue()
    assert (tmp_path / "gen" / "clip.mp4").stat().st_size > 0


def test_latents_are_in_latent_shard_format(finished):
    lat = finished["res"]["latents"]
    info = json.loads((Path(lat["dir"]) / "dataset.json").read_text())
    assert info["format"] == "nivra-latents-dataset/1" and info["scaled"] is True
    sh = LatentShard(Path(lat["dir"]) / info["shards"][0]["file"])
    m = sh.meta
    assert m["parent_shard_sha256"] == json.loads((finished["pipe"].dataset_dir() / "dataset.json").read_text())["shards"][0]["sha256"]
    assert m["vae_version"] == finished["res"]["vae"]["sha256"] and m["text_encoder_id"] == finished["res"]["text"]["sha256"]
    assert m["stage"]["h"] == 64 and m["stage"]["frames"] == 17


def test_rerun_is_fully_cached_and_option_change_reruns_downstream(finished, ds):
    tmp = finished["tmp"]
    before = {s: json.loads((tmp / "run" / "markers" / f"{s}.json").read_text())["finished_unix"] for s in STAGES}
    logs = []
    Pipeline(opts(tmp, ds, install_dir=str(tmp / "install"), name="my-anime"), log=logs.append).run()
    assert sum(1 for l in logs if str(l).endswith("cached")) == len(STAGES) and not any("start" in str(l) for l in logs)
    logs = []
    Pipeline(opts(tmp, ds, dit_steps=7, install_dir=str(tmp / "install"), name="my-anime"), log=logs.append).run()
    started = [str(l) for l in logs if str(l).endswith("start")]
    assert started == ["[dit] start", "[bundle] start"]                     # tokenizer/text/vae/latents untouched
    after = {s: json.loads((tmp / "run" / "markers" / f"{s}.json").read_text())["finished_unix"] for s in STAGES}
    assert all(after[s] == before[s] for s in ("ingest", "tokenizer", "text", "vae", "latents"))
    assert load_hd_bundle(tmp / "install" / "models" / "my-anime", "cpu")


def test_stage_range_and_missing_prerequisites(ds, tmp_path):
    p = Pipeline(opts(tmp_path, ds, to_stage="tokenizer"), log=lambda *_: None)
    res = p.run()
    assert set(res) == {"ingest", "tokenizer"}
    with pytest.raises(PipelineError, match="no finished result"):
        Pipeline(opts(tmp_path, ds, from_stage="vae"), log=lambda *_: None).plan()
    with pytest.raises(PipelineError):
        Pipeline(opts(tmp_path, ds, from_stage="dit", to_stage="text"))
    with pytest.raises(PipelineError):
        Pipeline(opts(tmp_path, ds, from_stage="nope"))
    with pytest.raises(PipelineError):
        Pipeline(PipelineOptions(out=str(tmp_path / "x")))                  # neither --videos nor --dataset


def test_plan_has_time_estimates_and_cli_plan_only(ds, tmp_path, capsys):
    p = Pipeline(opts(tmp_path, ds), log=lambda *_: None)
    txt = p.format_plan()
    for s in STAGES:
        assert s in txt
    assert "est. time" in txt and "total estimated" in txt and "TFLOP" in txt
    code = cli_main(["train-pipeline", "--config", "nivra_hd_tiny", "--dataset", str(ds), "--out", str(tmp_path / "r2"),
                     "--plan-only"])
    assert code == 0 and "total estimated" in capsys.readouterr().out
    code = cli_main(["train-pipeline", "--dataset", str(ds), "--out", str(tmp_path / "r3"), "--from-stage", "vae", "--to-stage", "dit"])
    assert code == 2


def test_stage_helpers():
    stages = [{"name": "a", "height": 128, "width": 224, "frames": 17, "steps": 30000},
              {"name": "b", "height": 256, "width": 448, "frames": 17, "steps": 50000},
              {"name": "c", "height": 256, "width": 448, "frames": 33, "steps": 20000}]
    assert [s["name"] for s in fit_stages(stages, 256, 448, 17)] == ["a", "b"]
    assert [s["name"] for s in fit_stages(stages, 100, 100, 5)] == ["dataset"]
    sc = scale_stages(stages, 1000)
    assert sum(s["steps"] for s in sc) == 1000 and [s["name"] for s in sc] == ["a", "b", "c"]
    assert sum(s["steps"] for s in scale_stages(stages, 2)) == 2
    assert sum(s["steps"] for s in scale_stages(stages, 1)) == 1
    assert text_config_for(256, 300).d_model == 256 and text_config_for(16, 300).d_model == 16


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_videos_folder_is_ingested(tmp_path):
    vids = tmp_path / "vids"
    vids.mkdir()
    for i, src in enumerate(("testsrc2", "mandelbrot")):
        r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-t", "3", "-f", "lavfi", "-i",
                            f"{src}=s=640x360:r=24", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
                            str(vids / f"v{i}.mp4")], capture_output=True)
        assert r.returncode == 0, r.stderr.decode()
        (vids / f"v{i}.txt").write_text(f"{src} test pattern number {i}", encoding="utf-8")
    (vids / "PROVENANCE.json").write_text(json.dumps({
        "source": "synthetic", "license": "CC0-1.0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "permission": "generated by the test-suite", "collected": "2026-01-01", "contains_people": False, "consent": "n/a"}))
    p = Pipeline(PipelineOptions(config="nivra_hd_tiny", out=str(tmp_path / "run"), videos=str(vids), vocab_size=300,
                                 to_stage="tokenizer", default_style="anime", val_fraction=0.0), log=lambda *_: None)
    res = p.run()
    assert res["ingest"]["clips"] >= 2 and res["tokenizer"]["vocab_size"] >= 260
    with pytest.raises(PipelineError, match="ingest failed"):
        (vids / "PROVENANCE.json").unlink()
        Pipeline(PipelineOptions(config="nivra_hd_tiny", out=str(tmp_path / "run2"), videos=str(vids), to_stage="ingest"),
                 log=lambda *_: None).run()
