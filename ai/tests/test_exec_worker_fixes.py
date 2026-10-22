"""Executor schema limits (prompt length, output name) and ffmpeg fail-fast."""
import os
import shutil

import pytest

from nivra import executor as ex

HD_TYPES = ["ai.generate", "ai.draft", "ai.refine_segment"]


def _base(job_type):
    p = {"prompt": "x"}
    if job_type == "ai.draft":
        p["seconds"] = 8.0
    if job_type == "ai.refine_segment":
        p.update(width=64, height=64, start=0, end=8)
    return p


@pytest.mark.parametrize("job_type", HD_TYPES)
def test_long_prompt_passes_hd_job_types(job_type):
    p = _base(job_type)
    p["prompt"] = "a" * 900
    out = ex._validate(p, ex.SCHEMAS[job_type]["params"])
    assert len(out["prompt"]) == 900


def test_300_char_prompt_passes_draft():
    p = _base("ai.draft")
    p["prompt"] = "b" * 300
    assert ex._validate(p, ex.SCHEMAS["ai.draft"]["params"])["prompt"] == "b" * 300


@pytest.mark.parametrize("job_type", HD_TYPES)
def test_prompt_over_1000_rejected(job_type):
    p = _base(job_type)
    p["prompt"] = "a" * 1001
    with pytest.raises(ex.SpecError):
        ex._validate(p, ex.SCHEMAS[job_type]["params"])


@pytest.mark.parametrize("job_type", ["ai.generate", "ai.draft", "ai.refine_segment", "ai.stitch"])
@pytest.mark.parametrize("name", ["../../evil", "a/b", "a\b", ".hidden", "", "x" * 65, "a b"])
def test_bad_names_rejected(job_type, name):
    p = _base(job_type) if job_type != "ai.stitch" else {"count": 1}
    p["name"] = name
    with pytest.raises(ex.SpecError):
        ex._validate(p, ex.SCHEMAS[job_type]["params"])


@pytest.mark.parametrize("name", ["video", "my_clip-2.final", "_x", "A" * 64])
def test_good_names_accepted(name):
    p = _base("ai.generate")
    p["name"] = name
    assert ex._validate(p, ex.SCHEMAS["ai.generate"]["params"])["name"] == name


def test_defaults_match_pattern():
    import re
    for t in ("ai.generate", "ai.draft", "ai.refine_segment", "ai.stitch"):
        d = ex.SCHEMAS[t]["params"]["name"]["default"]
        assert re.fullmatch(ex.NAME_PATTERN, d)


def test_require_ffmpeg_fails_fast_for_video_types(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    for t in ex.VIDEO_JOB_TYPES:
        with pytest.raises(ex.ToolMissing) as e:
            ex.require_ffmpeg(t)
        assert e.value.code == "FFMPEG_MISSING"
    ex.require_ffmpeg("ai.hd_train")  # not a video job: no error


def test_prepare_ffmpeg_env_prepends_dirs(monkeypatch, tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("NIVRA_FFMPEG", str(d / "ffmpeg.exe"))
    monkeypatch.setenv("NIVRA_FFPROBE", str(d / "ffprobe.exe"))
    monkeypatch.setenv("PATH", "other")
    ex.prepare_ffmpeg_env()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(d)
    assert os.environ["PATH"].endswith("other")
