"""Style vocabulary: canonical tags, aliases, leading-phrase detection, apply_style."""
import pytest

from nivra.data.styles import (STYLES, STYLIZED_STYLES, apply_style, detect_leading_style, mentions_style,
                               normalize_style, parse_data_style, parse_style_arg, resolve_style,
                               validate_style_list)


@pytest.mark.parametrize("text,canon", [
    ("anime", "anime"), ("Anime", "anime"), ("manga-style", "anime"), ("Manga Style", "anime"),
    ("japanese animation", "anime"), ("3d", "3d animation"), ("3D render", "3d animation"), ("cgi", "3d animation"),
    ("pixar-style", "3d animation"), ("blender", "3d animation"), ("3d animation", "3d animation"),
    ("cartoon", "cartoon"), ("western cartoon", "cartoon"), ("2d", "2d animation"), ("hand-drawn", "2d animation"),
    ("hand drawn", "2d animation"), ("cel-shaded 3d", "cel-shaded 3d"), ("cel shaded 3d", "cel-shaded 3d"),
    ("pixel art", "pixel art"), ("pixel-art", "pixel art"), ("stop-motion", "claymation"), ("clay", "claymation"),
    ("claymation", "claymation"), ("painterly", "painterly"), ("realistic", "realistic"),
    ("anime style", "anime"), ("  ANIME  ", "anime"),
])
def test_normalize_known(text, canon):
    assert normalize_style(text) == canon


@pytest.mark.parametrize("text", ["", "none", "banana", None, 3, "anim", "3d printer"])
def test_normalize_unknown(text):
    assert normalize_style(text) is None


def test_vocabulary_shape():
    assert STYLES[-1] == "realistic" and "realistic" not in STYLIZED_STYLES
    for s in ("anime", "3d animation", "cartoon", "2d animation", "cel-shaded 3d", "pixel art", "claymation", "painterly"):
        assert s in STYLIZED_STYLES
    assert validate_style_list(list(STYLES)) == list(STYLES)
    with pytest.raises(ValueError):
        validate_style_list(["anime", "anime"])
    with pytest.raises(ValueError):
        validate_style_list(["manga-style"])        # canonical tags only
    with pytest.raises(ValueError):
        validate_style_list([])


@pytest.mark.parametrize("caption,style,rest", [
    ("anime style. A girl runs across a rooftop.", "anime", "A girl runs across a rooftop."),
    ("Anime. A girl runs.", "anime", "A girl runs."),
    ("3D render: a robot waves", "3d animation", "a robot waves"),
    ("pixar-style, a toy cowboy rides", "3d animation", "a toy cowboy rides"),     # comma needs 'style'
    ("cel-shaded 3d style; a knight jumps", "cel-shaded 3d", "a knight jumps"),
    ("hand-drawn style | a lantern swings", "2d animation", "a lantern swings"),
    ("Western cartoon style. A dog slips.", "cartoon", "A dog slips."),
])
def test_detect_leading(caption, style, rest):
    assert detect_leading_style(caption) == (style, rest)


@pytest.mark.parametrize("caption", [
    "A girl runs in an anime style world", "anime fans cheer", "anime, a girl runs", "anime style.", "", "cartoonish dog",
    "a 3d printer prints", "banana style. a dog"])
def test_detect_leading_none(caption):
    assert detect_leading_style(caption) == (None, caption)


def test_apply_style():
    assert apply_style("a girl runs", "anime") == "anime style. a girl runs"
    assert apply_style("a girl runs", "Manga-Style") == "anime style. a girl runs"       # alias -> canonical
    assert apply_style("a girl runs", "none") == "a girl runs"
    assert apply_style("a girl runs", None) == "a girl runs"
    assert apply_style("a girl runs", "unspecified") == "a girl runs"
    # prompt already mentions a style alias (case-insensitive, whole word): unchanged
    assert apply_style("A CARTOON dog", "anime") == "A CARTOON dog"
    assert apply_style("a hand-drawn dog", "3d animation") == "a hand-drawn dog"
    assert apply_style("anime girl runs", "cartoon") == "anime girl runs"
    # a style word embedded inside another word is not a mention
    assert apply_style("a cartoonish dog", "anime") == "anime style. a cartoonish dog"
    assert mentions_style("a 3D robot") and not mentions_style("a 3dprinter robot")
    assert apply_style("a girl runs", "realistic") == "realistic style. a girl runs"
    with pytest.raises(ValueError, match="unknown style"):
        apply_style("a girl runs", "banana")


def test_arg_parsers():
    assert parse_style_arg("None") == "none" and parse_style_arg("pixar-style") == "3d animation"
    with pytest.raises(ValueError):
        parse_style_arg("banana")
    with pytest.raises(ValueError):
        parse_style_arg(3)
    assert parse_data_style("none") == "unspecified" and parse_data_style("hand drawn") == "2d animation"
    with pytest.raises(ValueError):
        parse_data_style("banana")
    assert resolve_style(None, [], "anime") == "anime" and resolve_style(None) is None
    assert resolve_style("none", ["anime"], "anime") == "none"
    with pytest.raises(ValueError, match="not supported"):
        resolve_style("cartoon", ["anime"], "anime")
