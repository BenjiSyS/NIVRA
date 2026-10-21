"""text_encoder_archive: safe extraction and use as a network worker's text encoder (no NIVRA_TEXT_ENCODERS_DIR)."""
from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
from pathlib import Path

import pytest
import torch

from nivra.inference import hd_bundle as hb
from nivra.inference.bundle import BundleError
from nivra.inference.te_archive import extract_text_encoder_archive
from nivra.inference.video_io import probe_mp4
from nivra.models.hd_config import load_hd_config
from test_hd_bundle import run


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_tar(path: Path, members, fmt=tarfile.PAX_FORMAT) -> Path:
    """members: (name, bytes | ('symlink', target) | ('dir',))"""
    with tarfile.open(path, "w", format=fmt) as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            if isinstance(data, tuple) and data[0] == "symlink":
                ti.type, ti.linkname = tarfile.SYMTYPE, data[1]
                tf.addfile(ti)
            elif isinstance(data, tuple) and data[0] == "dir":
                ti.type = tarfile.DIRTYPE
                tf.addfile(ti)
            else:
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return path


def tar_dir(src: Path, out: Path) -> Path:
    return make_tar(out, [(p.relative_to(src).as_posix(), p.read_bytes()) for p in sorted(src.rglob("*")) if p.is_file()])


GOOD = [("config.json", b"{}"), ("model.safetensors", b"weights"), ("sub/x.txt", b"x")]


def test_extract_ok_cached_and_atomic(tmp_path):
    a = make_tar(tmp_path / "a.tar", GOOD)
    cache = tmp_path / "te_cache"
    d = extract_text_encoder_archive(a, cache, _sha(b"weights"))
    assert d.parent == cache and d.name == _sha(a.read_bytes())
    assert (d / "model.safetensors").read_bytes() == b"weights" and (d / "sub" / "x.txt").is_file()
    assert [p.name for p in cache.iterdir()] == [d.name]                      # no temp leftovers
    assert extract_text_encoder_archive(a, cache, _sha(b"weights")) == d      # reuse


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/abs/evil.txt", "a\b.txt", "C:evil.txt", "a//b", "./a"])
def test_rejects_bad_names(tmp_path, name):
    a = make_tar(tmp_path / "a.tar", [("model.safetensors", b"w"), (name, b"1")])
    cache = tmp_path / "c"
    with pytest.raises(BundleError):
        extract_text_encoder_archive(a, cache)
    assert not any(cache.iterdir())                                           # nothing published, temp removed
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path.parent / "evil.txt").exists()


def test_rejects_links_dirs_dup_nonweights(tmp_path):
    cache = tmp_path / "c"
    cases = {
        "symlink": GOOD + [("link", ("symlink", "/etc/passwd"))],
        "dir": GOOD + [("d", ("dir",))],
        "dup": GOOD + [("config.json", b"{}")],
        "noweights": [("config.json", b"{}")],
        "empty": [],
    }
    for k, members in cases.items():
        with pytest.raises(BundleError):
            extract_text_encoder_archive(make_tar(tmp_path / f"{k}.tar", members), cache)
    leftovers = [p for p in cache.glob("*")] if cache.exists() else []
    assert leftovers == []                                                   # failed extractions leave nothing behind


def test_rejects_non_tar_and_compressed(tmp_path):
    junk = tmp_path / "junk"
    junk.write_bytes(b"not a tar" * 100)
    with pytest.raises(BundleError):
        extract_text_encoder_archive(junk, tmp_path / "c")
    import gzip
    plain = make_tar(tmp_path / "p.tar", GOOD)
    gz = tmp_path / "p.tar.gz"
    gz.write_bytes(gzip.compress(plain.read_bytes()))
    with pytest.raises(BundleError):
        extract_text_encoder_archive(gz, tmp_path / "c")


def test_sha_mismatch_refused_and_not_cached(tmp_path):
    a = make_tar(tmp_path / "a.tar", GOOD)
    cache = tmp_path / "c"
    with pytest.raises(BundleError, match="sha256"):
        extract_text_encoder_archive(a, cache, _sha(b"other"))
    assert not any(cache.iterdir())
    d = extract_text_encoder_archive(a, cache, None)                          # no sha recorded: accepted
    with pytest.raises(BundleError, match="sha256"):                          # a cached dir is re-verified too
        extract_text_encoder_archive(a, cache, _sha(b"other"))
    assert d.is_dir()


# ------------------------------------------------------------------ integration with the HD bundle loader
@pytest.fixture(scope="module")
def nivra_ckpt(tmp_path_factory):
    from nivra.text.bpe import BPETokenizer
    from nivra.text.nivra_text import NivraTextConfig, NivraTextModel, save_checkpoint
    tok = BPETokenizer()
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("txt") / "enc"
    save_checkpoint(d, NivraTextModel(NivraTextConfig.preset("tiny", tok.vocab_size)), tok)
    return d


@pytest.fixture(scope="module")
def bundle(nivra_ckpt, tmp_path_factory):
    cfg = load_hd_config("nivra_hd_tiny")
    cfg["model"]["text_dim"] = 64
    out = tmp_path_factory.mktemp("b") / "bundle"
    m = hb.write_hd_bundle(out, cfg, text_encoder=f"nivra:{nivra_ckpt}")
    return out, m


def flat_files(bundle_dir: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("weights.safetensors", "model_version.json", "vae.safetensors"):
        shutil.copy(bundle_dir / f, dst / f)
    return (dst / "weights.safetensors", dst / "model_version.json", dst / "vae.safetensors")


def test_archive_dir_precedence(bundle, nivra_ckpt, tmp_path, monkeypatch):
    out, m = bundle
    te = m["text_encoder"]
    monkeypatch.delenv(hb.ENV_TEXT_DIR, raising=False)
    files = flat_files(out, tmp_path / "flat")
    with pytest.raises(BundleError, match="NIVRA_TEXT_ENCODERS_DIR"):         # nothing to find it with
        hb.load_hd_bundle(files)
    arc_dir = extract_text_encoder_archive(tar_dir(out / "text_encoder", tmp_path / "te.tar"), tmp_path / "cache", te["sha256"])
    assert hb.load_hd_bundle(files, text_encoder_dir=arc_dir).text_encoder.id == te["id"]

    # explicit dir > bundle dir: a broken explicit dir is used (and refused) although the bundle dir is fine
    bad = tmp_path / "bad"
    shutil.copytree(out / "text_encoder", bad)
    b = bytearray((bad / "model.safetensors").read_bytes())
    b[-1] ^= 0x55
    (bad / "model.safetensors").write_bytes(bytes(b))
    with pytest.raises(BundleError, match="sha256"):
        hb.load_hd_bundle(out, text_encoder_dir=bad)
    # ... and a good explicit dir wins over a broken bundle dir
    broken_bundle = tmp_path / "broken_bundle"
    shutil.copytree(out, broken_bundle)
    shutil.copy(bad / "model.safetensors", broken_bundle / "text_encoder" / "model.safetensors")
    with pytest.raises(BundleError, match="sha256"):
        hb.load_hd_bundle(broken_bundle)
    assert hb.load_hd_bundle(broken_bundle, text_encoder_dir=arc_dir).text_encoder.id == te["id"]
    # explicit dir > env dir
    root = tmp_path / "envroot"
    shutil.copytree(bad, root / te["name"])
    monkeypatch.setenv(hb.ENV_TEXT_DIR, str(root))
    assert hb.load_hd_bundle(files, text_encoder_dir=arc_dir).text_encoder.id == te["id"]
    with pytest.raises(BundleError, match="sha256"):
        hb.load_hd_bundle(files)


def test_exec_generate_with_archive_input_no_env(bundle, tmp_path, monkeypatch):
    out, m = bundle
    monkeypatch.delenv(hb.ENV_TEXT_DIR, raising=False)
    files = flat_files(out, tmp_path / "in")
    arc = tar_dir(out / "text_encoder", tmp_path / "in" / "archive")
    ins = {"weights": str(files[0]), "manifest": str(files[1]), "vae_weights": str(files[2])}
    params = {"prompt": "a red ball", "steps": 2, "width": 32, "height": 32, "seconds": 0.3, "fps": 8, "name": "v"}
    code, ev, _, p = run(tmp_path / "w1", params, ins, name="noarc")
    assert code == 5 and "NIVRA_TEXT_ENCODERS_DIR" in p.stdout                 # INPUT_INVALID without the archive
    code, ev, o, p = run(tmp_path / "w2", params, dict(ins, text_encoder_archive=str(arc)), name="arc")
    assert code == 0, p.stdout + p.stderr[-2000:]
    assert probe_mp4(o / "v.mp4")
    assert (arc.parent / "te_cache" / _sha(arc.read_bytes()) / "model.safetensors").is_file()
    # a tampered archive is INPUT_INVALID
    bad = tmp_path / "in" / "badarc"
    make_tar(bad, [("config.json", b"{}"), ("model.safetensors", b"nope")])
    code, ev, _, p = run(tmp_path / "w3", params, dict(ins, text_encoder_archive=str(bad)), name="bad")
    assert code == 5 and "sha256" in p.stdout
