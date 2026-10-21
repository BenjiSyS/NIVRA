"""Captioning: subtitle cleanup, caption merge, style detection, file handling. No model downloads (models are mocked)."""
from pathlib import Path

import pytest

from nivra.captioning import runner
from nivra.captioning.describe import (clean_caption, detect_style, merge_caption, resolve_style, truncate_words)
from nivra.captioning.frames import frame_times
from nivra.captioning.subtitles import (Stream, parse_cues, split_subtitle_streams, subtitles_to_text)

SRT = """1
00:00:01,000 --> 00:00:03,000
<i>Hello there!</i>

2
00:00:03,500 --> 00:00:05,000
{\\an8}How are&nbsp;you
today?

3
00:00:05,100 --> 00:00:06,000
How are you today?

4
00:00:07,000 --> 00:00:08,000
<font color="#ff0000"></font>

5
00:00:09,000 --> 00:00:10,000
Bye.
"""


def test_srt_cleanup():
    assert subtitles_to_text(SRT) == "Hello there!\nHow are you today?\nBye."


def test_vtt_cleanup():
    vtt = "WEBVTT\n\nNOTE x\n\n00:01.000 --> 00:02.000 align:start\n<c.yellow>Hi</c>\n\n00:02.000 --> 00:03.000\nThere\n"
    assert subtitles_to_text(vtt) == "Hi\nThere"


def test_ass_cleanup():
    ass = ("[Script Info]\nTitle: x\n\n[Events]\n"
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}Hello, world{\\i0}\\Nsecond line\n"
           "Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,Hello, world second line\n"
           "Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,{\\p1}m 0 0 l 100 0 100 100 0 100{\\p0}\n")
    assert parse_cues(ass)[0].startswith("{\\i1}Hello, world")
    assert subtitles_to_text(ass) == "Hello, world second line"


def test_empty_subtitles():
    assert subtitles_to_text("") == ""
    assert subtitles_to_text("1\n00:00:01,000 --> 00:00:02,000\n<b></b>\n") == ""


def test_stream_split_bitmap_vs_text():
    streams = [Stream(0, "video", "h264"), Stream(1, "subtitle", "hdmv_pgs_subtitle"),
               Stream(2, "subtitle", "subrip"), Stream(3, "audio", "aac"), Stream(4, "subtitle", "mov_text")]
    text, bitmap = split_subtitle_streams(streams)
    assert [s.index for s in text] == [2, 4] and [s.index for s in bitmap] == [1]


def test_clean_caption_boilerplate():
    assert clean_caption("The image shows a cartoon cat sitting on a sofa.") == "A cartoon cat sitting on a sofa."
    assert clean_caption("In this image, there is a red car. The image is blurry.") == "A red car. The video is blurry."
    assert clean_caption("This is an image of a dog.").lower().startswith("this is a video of a dog")
    assert "image" not in clean_caption("A man looks at the image on the wall of the image").lower()


def test_truncate_words_whole_sentences():
    t = "One two three. Four five six seven. Eight nine."
    assert truncate_words(t, 8) == "One two three. Four five six seven."
    assert truncate_words(t, 2) == "One two."
    assert truncate_words(t, 0) == ""


def test_detect_style():
    assert detect_style(["An anime girl with blue hair"]) == "anime"
    assert detect_style(["A 3d render of a house", "a cgi house", "a cartoon house"]) == "3d animation"
    assert detect_style(["A pixel art landscape"]) == "pixel art"
    assert detect_style(["A claymation figure"]) == "claymation"
    assert detect_style(["A cel-shaded character"]) == "cel-shaded 3d"
    assert detect_style(["A cartoon dog"]) == "cartoon"
    assert detect_style(["An animated scene with a fox"]) == "2d animation"
    assert detect_style(["A man walking a dog on a street"]) is None  # never guesses anime


def test_resolve_style():
    assert resolve_style("pixel art", ["an anime girl"]) == "pixel art"  # explicit wins
    assert resolve_style("none", ["an anime girl"]) is None
    assert resolve_style("auto", ["an anime girl"]) == "anime"
    assert resolve_style("auto", ["a man"]) is None
    with pytest.raises(ValueError):
        resolve_style("vaporwave", [])


def test_merge_caption_shape_and_length():
    detailed = "The image shows " + " ".join(f"word{i}" for i in range(40)) + ". " + " ".join(f"more{i}" for i in range(60)) + "."
    out = merge_caption("anime", detailed, "A cat on a sofa.", "A dog in a park.")
    assert out.startswith("Anime style. ")
    assert len(out.split()) <= 77
    assert "The scene changes from cat on a sofa to dog in a park." in out
    assert "image" not in out.lower()


def test_merge_no_style_no_change_note():
    out = merge_caption(None, "In this image, a red ball rolls.", "A red ball.", "A red ball.")
    assert out == "A red ball rolls."


def test_merge_keeps_description_when_notes_too_long():
    long_cap = " ".join(f"w{i}" for i in range(30))
    out = merge_caption("cartoon", " ".join(["alpha beta."] * 40), long_cap + " x", "totally different thing " + long_cap)
    assert len(out.split()) <= 77 and out.lower().count("alpha") >= 10


def test_frame_times_avoid_ends_and_even():
    ts = frame_times(100.0, 6)
    assert len(ts) == 6 and ts[0] > 5.0 and ts[-1] < 95.0
    gaps = {round(b - a, 6) for a, b in zip(ts, ts[1:])}
    assert len(gaps) == 1
    assert frame_times(10.0, 1) == [5.0]


# ---------------------------------------------------------------- runner with mocks

class FakeFlorence:
    calls = 0

    def caption_frames(self, frames):
        FakeFlorence.calls += 1
        return ("The image shows an anime girl waving.", "A girl.", "A girl waving at a boy in a park.", ["A girl."])


class FakeWhisper:
    def transcribe(self, wav):
        return ["spoken line one", "spoken line two"]


@pytest.fixture
def patched(monkeypatch):
    FakeFlorence.calls = 0
    monkeypatch.setattr(runner, "extract_frames", lambda v, n, d: [Path("x.jpg")] * n)
    monkeypatch.setattr(runner, "probe_streams", lambda v: [Stream(0, "video", "h264"), Stream(1, "audio", "aac")])
    monkeypatch.setattr(runner, "extract_audio_wav", lambda v, o: True)
    return runner.Lazy(FakeFlorence), runner.Lazy(FakeWhisper)


def _mk(tmp_path, *names):
    out = []
    for n in names:
        p = tmp_path / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        out.append(p)
    return out


def test_naming():
    v = Path("a/b/My Clip.v2.mp4")
    assert runner.caption_path_for(v) == Path("a/b/My Clip.v2.txt")
    assert runner.subtitles_path_for(v) == Path("a/b/My Clip.v2.subtitles.txt")


def test_find_videos_recursive_filters(tmp_path):
    _mk(tmp_path, "a.mp4", "sub/b.MKV", "sub/deep/c.webm", "d.mov", "notes.txt", "e.avi", "sub/f.mp4.txt")
    names = [p.name for p in runner.find_videos(tmp_path)]
    assert names == ["a.mp4", "b.MKV", "c.webm", "d.mov"] or sorted(names) == sorted(["a.mp4", "b.MKV", "c.webm", "d.mov"])
    single = _mk(tmp_path, "z.mp4")[0]
    assert runner.find_videos(single) == [single]


def test_run_writes_caption_and_speech_subtitles(tmp_path, patched):
    (v,) = _mk(tmp_path, "clip.mp4")
    lines = []
    rc = runner.run(tmp_path, runner.Options(), florence=patched[0], whisper=patched[1], out=lines.append)
    assert rc == 0
    cap = (tmp_path / "clip.txt").read_text(encoding="utf-8")
    assert cap.startswith("Anime style. An anime girl waving.") and "The scene changes from girl to girl waving" in cap
    assert (tmp_path / "clip.subtitles.txt").read_text(encoding="utf-8") == "spoken line one\nspoken line two\n"
    assert "1 done (1 with subtitles), 0 skipped (existing), 0 failed" in lines[-1]


def test_skip_existing_and_overwrite(tmp_path, patched):
    _mk(tmp_path, "clip.mp4")
    (tmp_path / "clip.txt").write_text("hand written\n", encoding="utf-8")
    lines = []
    assert runner.run(tmp_path, runner.Options(), florence=patched[0], whisper=patched[1], out=lines.append) == 0
    assert (tmp_path / "clip.txt").read_text(encoding="utf-8") == "hand written\n"
    assert FakeFlorence.calls == 0 and "1 skipped (existing)" in lines[-1]
    assert not (tmp_path / "clip.subtitles.txt").exists()
    assert runner.run(tmp_path, runner.Options(overwrite=True), florence=patched[0], whisper=patched[1], out=lines.append) == 0
    assert (tmp_path / "clip.txt").read_text(encoding="utf-8").startswith("Anime style.")


def test_no_subtitles_when_off_or_no_audio(tmp_path, patched, monkeypatch):
    _mk(tmp_path, "a.mp4")
    runner.run(tmp_path, runner.Options(subtitles="off"), florence=patched[0], whisper=patched[1], out=lambda s: None)
    assert (tmp_path / "a.txt").exists() and not (tmp_path / "a.subtitles.txt").exists()
    (tmp_path / "a.txt").unlink()
    monkeypatch.setattr(runner, "probe_streams", lambda v: [Stream(0, "video", "h264")])
    runner.run(tmp_path, runner.Options(), florence=patched[0], whisper=patched[1], out=lambda s: None)
    assert (tmp_path / "a.txt").exists() and not (tmp_path / "a.subtitles.txt").exists()


def test_embedded_preferred_and_bitmap_reported(tmp_path, patched, monkeypatch):
    _mk(tmp_path, "a.mp4")
    monkeypatch.setattr(runner, "probe_streams", lambda v: [Stream(0, "video", "h264"), Stream(1, "subtitle", "hdmv_pgs_subtitle"),
                                                            Stream(2, "subtitle", "subrip"), Stream(3, "audio", "aac")])
    monkeypatch.setattr(runner, "extract_embedded", lambda v, s: "from the file")
    lines = []
    runner.run(tmp_path, runner.Options(), florence=patched[0], whisper=patched[1], out=lines.append)
    assert (tmp_path / "a.subtitles.txt").read_text(encoding="utf-8") == "from the file\n"
    assert any("bitmap" in l and "unsupported" in l for l in lines)
    assert "embedded" in lines[0]


def test_bitmap_only_falls_back_to_speech_in_auto_not_in_embedded(tmp_path, patched, monkeypatch):
    _mk(tmp_path, "a.mp4")
    monkeypatch.setattr(runner, "probe_streams", lambda v: [Stream(0, "video", "h264"), Stream(1, "subtitle", "dvd_subtitle"),
                                                            Stream(2, "audio", "aac")])
    runner.run(tmp_path, runner.Options(), florence=patched[0], whisper=patched[1], out=lambda s: None)
    assert (tmp_path / "a.subtitles.txt").read_text(encoding="utf-8").startswith("spoken line one")
    (tmp_path / "a.subtitles.txt").unlink()
    runner.run(tmp_path, runner.Options(subtitles="embedded", overwrite=True), florence=patched[0], whisper=patched[1], out=lambda s: None)
    assert not (tmp_path / "a.subtitles.txt").exists()


def test_failure_continues_and_exit_code(tmp_path, patched, monkeypatch):
    _mk(tmp_path, "bad.mp4", "good.mp4")
    real = runner.extract_frames

    def flaky(v, n, d):
        if v.name == "bad.mp4":
            raise runner.CaptionError("corrupt file")
        return real(v, n, d) if False else [Path("x.jpg")] * n
    monkeypatch.setattr(runner, "extract_frames", flaky)
    lines = []
    rc = runner.run(tmp_path, runner.Options(), florence=patched[0], whisper=patched[1], out=lines.append)
    assert rc == 1
    assert (tmp_path / "good.txt").exists() and not (tmp_path / "bad.txt").exists()
    assert any("FAILED: corrupt file" in l and "bad.mp4" in l for l in lines)
    assert "1 done" in lines[-1] and "1 failed" in lines[-1]


def test_missing_path_and_empty_folder(tmp_path):
    assert runner.run(tmp_path / "nope", runner.Options(), out=lambda s: None) == 2
    assert runner.run(tmp_path, runner.Options(), out=lambda s: None) == 0


def test_missing_dependency_message(monkeypatch):
    import builtins
    from nivra.captioning import models
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("transformers"):
            raise ImportError("No module named 'transformers'")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(models.DependencyError) as e:
        models.Florence()
    assert "pip install -r ai/requirements-caption.txt" in str(e.value)


# ---------------------------------------------------------------- batch behaviour

def test_natural_sort_and_progress_format(tmp_path, patched):
    _mk(tmp_path, "clip10.mp4", "clip2.mp4", "a/clip1.mp4", "clip1.mp4")
    assert [p.relative_to(tmp_path).as_posix() for p in runner.find_videos(tmp_path)] == [
        "a/clip1.mp4", "clip1.mp4", "clip2.mp4", "clip10.mp4"]
    lines = []
    runner.run(tmp_path, runner.Options(subtitles="off"), florence=patched[0], whisper=patched[1], out=lines.append)
    import re
    assert re.match(r"\[1/4\] a.clip1\.mp4 \.\.\. done \(\d+ s\)$", lines[0])
    assert lines[3].startswith("[4/4] clip10.mp4 ... done")
    assert "total" in lines[-1] and lines[-1].startswith("summary: 4 done")


def test_limit_continues_on_rerun(tmp_path, patched):
    _mk(tmp_path, *[f"v{i}.mp4" for i in range(1, 6)])
    (tmp_path / "v1.txt").write_text("manual\n", encoding="utf-8")
    opts = runner.Options(subtitles="off", limit=2)
    lines = []
    runner.run(tmp_path, opts, florence=patched[0], whisper=patched[1], out=lines.append)
    assert FakeFlorence.calls == 2  # v1 skipped does not count
    assert (tmp_path / "v2.txt").exists() and (tmp_path / "v3.txt").exists() and not (tmp_path / "v4.txt").exists()
    assert "2 done" in lines[-1] and "2 more need captions" in lines[-1]
    runner.run(tmp_path, opts, florence=patched[0], whisper=patched[1], out=lines.append)
    assert (tmp_path / "v4.txt").exists() and (tmp_path / "v5.txt").exists()
    assert "more need captions" not in lines[-1]


def test_failed_list_written(tmp_path, patched, monkeypatch):
    _mk(tmp_path, "bad.mp4", "ok.mp4")
    monkeypatch.setattr(runner, "extract_frames",
                        lambda v, n, d: (_ for _ in ()).throw(runner.CaptionError("corrupt")) if v.name == "bad.mp4" else [Path("x.jpg")] * n)
    runner.run(tmp_path, runner.Options(subtitles="off"), florence=patched[0], whisper=patched[1], out=lambda s: None)
    txt = (tmp_path / "captions-failed.txt").read_text(encoding="utf-8")
    assert "bad.mp4" in txt and "corrupt" in txt and "ok.mp4" not in txt


def test_ctrl_c_stops_cleanly_no_partial_files(tmp_path, patched, monkeypatch):
    _mk(tmp_path, "a.mp4", "b.mp4", "c.mp4")
    calls = {"n": 0}
    orig = runner.get_subtitles

    def boom(v, mode, w, notes):
        calls["n"] += 1
        if v.name == "b.mp4":
            raise KeyboardInterrupt
        return orig(v, mode, w, notes)
    monkeypatch.setattr(runner, "get_subtitles", boom)
    lines = []
    rc = runner.run(tmp_path, runner.Options(), florence=patched[0], whisper=patched[1], out=lines.append)
    assert rc == 130
    assert (tmp_path / "a.txt").exists() and not (tmp_path / "b.txt").exists() and not (tmp_path / "c.txt").exists()
    assert not list(tmp_path.glob("*.tmp")) and not (tmp_path / "b.subtitles.txt").exists()
    assert "interrupted" in " ".join(lines) and "2 more need captions" in lines[-1]


def test_atomic_write_leaves_no_temp_on_error(tmp_path, monkeypatch):
    def bad_replace(a, b):
        raise OSError("disk full")
    monkeypatch.setattr(runner.os, "replace", bad_replace)
    with pytest.raises(OSError):
        runner.write_text_atomic(tmp_path / "x.txt", "hi")
    assert list(tmp_path.iterdir()) == []
