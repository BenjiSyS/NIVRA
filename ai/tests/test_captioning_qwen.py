"""Qwen3-VL structured captioning: parsing / assembly (pure), runner wiring with a fake model, CLI flags."""
import json
from pathlib import Path

import pytest

from nivra.captioning import runner, structured
from nivra.captioning.runner import Lazy, Options

RAW = json.dumps({
    "style": "anime", "short": "A girl in a red coat runs across a rooftop at dusk while the camera follows her.",
    "subjects": "A teenage girl with short black hair and a red coat.",
    "motion": "She sprints left to right, jumps a gap in the last frames and lands rolling.",
    "camera": "Medium tracking shot that pans with her, ending in a low angle.",
    "setting": "A city rooftop with water tanks and a skyline.",
    "art_style": "Flat cel colours with thick outlines.", "lighting": "Warm orange sunset with long shadows."})


def test_parse_output_tolerates_fences_and_chatter():
    assert structured.parse_output("```json\n" + RAW + "\n```")["style"] == "anime"
    assert structured.parse_output("Sure! Here you go: " + RAW + " Hope it helps {not json}")["style"] == "anime"
    assert structured.parse_output('{"a": "brace } inside string", "b": 1}') == {"a": "brace } inside string", "b": 1}
    assert structured.parse_output("no json here") == {}
    assert structured.parse_output("") == {}


def test_assemble_short_and_dense_shape():
    cap = structured.assemble(RAW)
    short, dense = cap["short"], cap["dense"]
    assert short.startswith("Anime style.") and len(short.split()) <= structured.SHORT_WORDS
    for kw in ("sprints", "tracking shot", "rooftop", "cel colours", "sunset"):
        assert kw in dense
    assert len(dense.split()) > len(short.split())
    assert structured.txt_of(cap) == short + "\n" + dense and "\n" not in short
    rec = structured.record_of(cap, captioner="qwen3vl", model="m", times=[0.5, 1.25])
    assert rec["schema"] == "nivra.caption/1" and rec["fields"]["camera"].startswith("Medium") and rec["frame_times"] == [0.5, 1.25]


def test_short_is_capped_and_style_forced_or_detected():
    long_short = " ".join(["word"] * 60) + "."
    cap = structured.assemble(json.dumps({"style": "cartoon", "short": long_short, "motion": "x moves."}))
    assert len(cap["short"].split()) <= structured.SHORT_WORDS and cap["short"].startswith("Cartoon style.")
    # the user's --style overrides the model; "none" drops the tag; unknown model style falls back to keyword detection
    assert structured.assemble(RAW, "pixel art")["short"].startswith("Pixel art style.")
    assert not structured.assemble(RAW, "none")["short"].lower().startswith("anime style")
    det = structured.assemble(json.dumps({"style": "weird", "short": "A clay figure waves.", "art_style": "claymation look"}))
    assert det["style"] == "claymation"
    # a model that already starts the sentence with the tag does not get it twice
    twice = structured.assemble(json.dumps({"style": "anime", "short": "Anime style. A cat jumps."}))
    assert twice["short"].lower().count("anime style") == 1


def test_speech_and_quotes_never_reach_the_caption():
    raw = json.dumps({"style": "anime", "short": 'A man says "hello there my friend" to a dog.',
                      "subjects": 'A man and a dog. The subtitles read "hello there".',
                      "motion": 'He speaks to the dog. The dog wags its tail.', "camera": "Static."})
    cap = structured.assemble(raw)
    blob = (cap["short"] + " " + cap["dense"]).lower()
    for bad in ("hello", "subtitle", "speaks", "says"):
        assert bad not in blob
    assert "wags its tail" in blob


def test_non_json_answer_still_gives_a_caption_and_empty_fails():
    cap = structured.assemble("A knight walks through a forest. The camera pans up.")
    assert "knight" in cap["short"] and "knight" in cap["dense"]
    with pytest.raises(structured.CaptionError):
        structured.assemble("   ")
    with pytest.raises(structured.CaptionError):
        structured.assemble("{}")


def test_prompt_mentions_frames_times_and_forbids_text():
    p = structured.build_prompt(3, [0.5, 1.5, 2.5])
    assert "These 3 images" in p and "0.5s, 1.5s, 2.5s" in p and "never quote text" in p and '"motion"' in p


class FakeQwen:
    calls = []

    def describe(self, frames, times, **kw):
        FakeQwen.calls.append((len(frames), list(times)))
        return RAW


def _setup(monkeypatch, tmp_path, n=2):
    monkeypatch.setattr(runner, "extract_frames_timed", lambda v, k, d, max_side=768: ([Path("f.jpg")] * k, [0.5 * i for i in range(k)]))
    monkeypatch.setattr(runner, "probe_streams", lambda v: [])
    for i in range(n):
        (tmp_path / f"c{i}.mp4").write_bytes(b"x")
    FakeQwen.calls.clear()


def test_run_with_qwen_writes_txt_and_optional_json(tmp_path, monkeypatch):
    _setup(monkeypatch, tmp_path)
    out = []
    boom = Lazy(lambda: (_ for _ in ()).throw(AssertionError("Florence must not be loaded")))
    rc = runner.run(tmp_path, Options(captioner="qwen3vl", write_json=True, subtitles="off", frames=4), florence=boom,
                    qwen=Lazy(FakeQwen), out=out.append)
    assert rc == 0, out
    assert FakeQwen.calls == [(4, [0.0, 0.5, 1.0, 1.5])] * 2
    txt = (tmp_path / "c0.txt").read_text(encoding="utf-8").splitlines()
    assert len(txt) == 2 and txt[0].startswith("Anime style.") and len(txt[1].split()) > len(txt[0].split())
    rec = json.loads((tmp_path / "c0.json").read_text(encoding="utf-8"))
    assert rec["short"] == txt[0] and rec["dense"] == txt[1] and rec["captioner"] == "qwen3vl"
    # without --json there is no sidecar
    (tmp_path / "c1.txt").unlink()
    (tmp_path / "c1.json").unlink()
    assert runner.run(tmp_path, Options(captioner="qwen3vl", subtitles="off"), florence=boom, qwen=Lazy(FakeQwen), out=out.append) == 0
    assert (tmp_path / "c1.txt").exists() and not (tmp_path / "c1.json").exists()


def test_garbage_model_output_is_a_per_video_failure(tmp_path, monkeypatch):
    _setup(monkeypatch, tmp_path, 1)

    class Empty:
        def describe(self, *a, **k):
            return ""
    out = []
    rc = runner.run(tmp_path, Options(captioner="qwen3vl", subtitles="off"), qwen=Lazy(Empty), out=out.append)
    assert rc == 1 and not (tmp_path / "c0.txt").exists() and any("FAILED" in l for l in out)


def test_choose_captioner(monkeypatch):
    assert runner.choose_captioner("florence", "cuda") == "florence"
    assert runner.choose_captioner("qwen3vl", "cpu") == "qwen3vl"
    assert runner.choose_captioner("auto", "cpu") == "florence"
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert runner.choose_captioner("auto", "auto") == "qwen3vl"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert runner.choose_captioner("auto", "auto") == "florence"
    with pytest.raises(ValueError):
        runner.choose_captioner("gpt", "auto")


def test_cli_flags():
    from nivra import cli
    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    if p is None:
        pytest.skip("no build_parser")
    a = p.parse_args(["caption", "x", "--captioner", "qwen3vl", "--json"])
    assert a.captioner == "qwen3vl" and a.json is True
    assert p.parse_args(["caption", "x"]).captioner == "auto"
    with pytest.raises(SystemExit):
        p.parse_args(["caption", "x", "--captioner", "nope"])


def test_qwen_inputs_expand_one_pad_per_merged_patch():
    import torch
    from nivra.captioning.qwen3vl import Qwen3VL

    class Tok:
        image_token = "<|image_pad|>"

        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, chat_template):
            assert tokenize is False and add_generation_prompt and chat_template == "tmpl"
            n = sum(1 for c in msgs[0]["content"] if c["type"] == "image")
            return "<u>" + "".join("<v><|image_pad|></v>" for _ in range(n)) + msgs[0]["content"][-1]["text"] + "<a>"

        def __call__(self, text, return_tensors):
            ids = torch.arange(len(text.split("|"))).view(1, -1)
            self.last = text
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class Img:
        merge_size = 2

        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.zeros(len(images) * 16, 8), "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images))}

    o = Qwen3VL.__new__(Qwen3VL)
    o.tokenizer, o.image_processor, o.chat_template = Tok(), Img(), "tmpl"
    out = o._inputs([object(), object(), object()], "PROMPT")
    assert o.tokenizer.last.count("<|image_pad|>") == 3 * 4          # grid 1x4x4 / merge^2 = 4 pads per frame
    assert o.tokenizer.last.endswith("PROMPT<a>") and out["pixel_values"].shape[0] == 48 and out["image_grid_thw"].shape == (3, 3)
