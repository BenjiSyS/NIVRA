import numpy as np
import pytest

from nivra.data.captions import caption
from nivra.data.scene import Entity, SceneSpec
from nivra.data.synthetic import (SynthConfig, make_sample, content_hash, fingerprint, manifest, SynthDataset,
                                  GENERATOR_VERSION)
from nivra.text.parser import parse_prompt
from nivra.text.tokenizer import WordTokenizer, default_tokenizer


def test_determinism_and_shape():
    a, b = make_sample(5, 17), make_sample(5, 17)
    assert a.video.dtype == np.uint8 and a.video.shape == (16, 64, 64, 3)
    assert np.array_equal(a.video, b.video)
    assert a.captions == b.captions
    assert content_hash(5, 0, 10) == content_hash(5, 0, 10)
    assert not np.array_equal(make_sample(5, 18).video, a.video)
    assert not np.array_equal(make_sample(6, 17).video, a.video)


def test_fingerprint_depends_on_everything():
    base = fingerprint(1, 0, 100)
    assert base != fingerprint(2, 0, 100) and base != fingerprint(1, 1, 100) and base != fingerprint(1, 0, 101)
    assert base != fingerprint(1, 0, 100, SynthConfig(size=32))
    m = manifest(1, 0, 100)
    assert m["generator_version"] == GENERATOR_VERSION and m["fingerprint"] == base


def test_dark_background_and_objects_present():
    s = make_sample(3, 4)
    assert s.video[0, 0, 0].max() < 30
    assert (s.video.max(axis=3) > 150).any()


def _spec(*ents):
    return SceneSpec([Entity(sh, c, m, x0=0.1 * (i + 1)) for i, (sh, c, m) in enumerate(ents)])


def test_caption_english_swedish():
    s = _spec(("circle", "red", "l2r"))
    assert caption(s, "en") == "a red circle moves from left to right"
    assert caption(s, "sv") == "en röd cirkel rör sig från vänster till höger"
    s = _spec(("square", "blue", "up"), ("square", "blue", "up"))
    assert caption(s, "en") == "two blue squares move up"
    assert caption(s, "sv") == "två blå kvadrater rör sig uppåt"
    s = _spec(("triangle", "yellow", "stationary"))
    assert caption(s, "en") == "a yellow triangle stays still"
    assert caption(s, "sv") == "en gul triangel står still"
    s = _spec(("circle", "white", "stationary"), ("circle", "white", "stationary"), ("circle", "white", "stationary"))
    assert caption(s, "sv") == "tre vita cirklar står stilla"


def test_swedish_ett_word_agreement():
    s = _spec(("circle", "red", "down"))
    assert caption(s, "sv", "object") == "ett rött föremål rör sig nedåt"
    assert caption(s, "en", "object") == "a red object moves down"
    s = _spec(("circle", "green", "down"), ("square", "green", "down"))
    assert caption(s, "sv", "object") == "två gröna föremål rör sig nedåt"
    assert caption(_spec(("circle", "blue", "up")), "sv", "object") == "ett blått föremål rör sig uppåt"
    assert caption(_spec(("circle", "yellow", "up")), "sv", "object") == "ett gult föremål rör sig uppåt"


def test_multi_object_caption_and_parser_roundtrip():
    s = _spec(("circle", "red", "l2r"), ("square", "green", "r2l"), ("triangle", "white", "stationary"))
    assert caption(s, "en") == ("a red circle moves from left to right, a green square moves from right to left "
                                "and a white triangle stays still")
    for i in range(300):
        smp = make_sample(11, i)
        for lang in ("en", "sv"):
            for style in ("shape", "object"):
                p = parse_prompt(smp.captions[lang][style])
                assert p.ok and not p.unknown, (smp.captions[lang][style], p.errors)
                assert p.lang == lang
                want = sorted((e.color, e.motion) for e in smp.spec.entities)
                assert sorted((e.color, e.motion) for e in p.spec.entities) == want
                if style == "shape":
                    assert p.spec.semantic_key() == smp.spec.semantic_key()


def test_parser_reports_errors():
    assert not parse_prompt("a flying banana").ok
    assert parse_prompt("a red circle").errors  # no motion


def test_tokenizer_roundtrip_and_versioning(tmp_path):
    tok = default_tokenizer()
    txt = "en röd cirkel rör sig från vänster till höger"
    ids = tok.encode(txt)
    assert len(ids) == tok.max_len and ids[0] == tok.bos_id
    assert tok.decode(ids) == txt
    assert tok.encode("A RED, circle!! moves up.") == tok.encode("a red circle moves up")
    assert tok.encode("zzz")[1] == tok.unk_id
    tok.save(tmp_path / "t.json")
    tok2 = WordTokenizer.load(tmp_path / "t.json")
    assert tok2.vocab_hash == tok.vocab_hash and tok2.itos == tok.itos
    bad = tok.to_json()
    bad["version"] = "other/9"
    with pytest.raises(ValueError):
        WordTokenizer.from_json(bad)
    bad = tok.to_json()
    bad["itos"] = bad["itos"][:-1] + ["tampered"]
    with pytest.raises(ValueError):
        WordTokenizer.from_json(bad)
    for s in range(20):
        for lg in ("en", "sv"):
            assert not tok.unknown_words(make_sample(1, s).caption(lg))


def test_dataset_view():
    ds = SynthDataset(9, 100, 5)
    assert np.array_equal(ds.sample(2).video, make_sample(9, 102).video)
