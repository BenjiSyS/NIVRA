"""`nivra text-encoder-fetch`: download a licensed external text encoder ONCE, when the owner runs the command.

Nothing here runs implicitly: training, encoding and serving never download anything (external.py refuses URLs).
The catalogue is a closed list with pinned revisions and expected file hashes; the official Hugging Face repo of
umT5-base only ships a pickle (`pytorch_model.bin`), so the file is hash-checked against the pinned LFS digest, read
with ``torch.load(weights_only=True)`` and rewritten as an encoder-only ``model.safetensors`` (what external.py loads).
A `NIVRA_LICENSE.json` {name, license, source_url, sha256, ...} is written next to it.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .external import LICENSE_FILE, hash_weights

CATALOG = {
    "umt5-base": {
        "repo": "google/umt5-base",
        "revision": "0de9394d54f8975e71838d309de1cb496c894ab9",
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/google/umt5-base",   # model card metadata: license apache-2.0
        "source_url": "https://huggingface.co/google/umt5-base",
        "dim": 768,
        "weights": ("pytorch_model.bin", "c48f7036095f1a02ee817d1b131de443bfe126b45ff76133ef5ae685bac9f061", 2368275277),
        "files": ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model"),
        "encoder_class": "UMT5EncoderModel",
    },
}


class FetchError(RuntimeError):
    pass


def default_dir(name: str) -> Path:
    root = os.environ.get("NIVRA_TEXT_ENCODER_DIR")
    if root:
        return Path(root) / name
    models = os.environ.get("NIVRA_MODELS_DIR")
    base = Path(models) if models else Path.home() / ".nivra" / "models"
    return base / "text" / name


def encoder_dir(name: str) -> Path:
    """Where `name` lives locally (does not check that it exists)."""
    return default_dir(name)


def extract_encoder_state(sd: dict) -> dict:
    """Encoder-only state dict (shared embedding + encoder.*) from a UMT5ForConditionalGeneration checkpoint."""
    out = {}
    for k, v in sd.items():
        if k == "shared.weight" or k.startswith("encoder."):
            out[k] = v.detach().clone().contiguous()
    if "shared.weight" not in out and "encoder.embed_tokens.weight" in out:
        out["shared.weight"] = out["encoder.embed_tokens.weight"].clone()
    if "encoder.embed_tokens.weight" not in out and "shared.weight" in out:
        out["encoder.embed_tokens.weight"] = out["shared.weight"].clone()
    if not any(k.startswith("encoder.block.") for k in out):
        raise FetchError("the checkpoint has no encoder blocks")
    return out


def fetch(name: str, dest: Optional[Path] = None, log: Callable[[str], None] = print, cache_dir: Optional[str] = None) -> Path:
    if name not in CATALOG:
        raise FetchError(f"unknown encoder {name!r}; available: {', '.join(sorted(CATALOG))}")
    spec = CATALOG[name]
    dest = Path(dest) if dest else default_dir(name)
    if (dest / LICENSE_FILE).is_file() and (dest / "model.safetensors").is_file():
        log(f"{name} is already installed in {dest}")
        return dest
    try:
        from huggingface_hub import hf_hub_download
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        raise FetchError("text-encoder-fetch needs `huggingface_hub`, `torch` and `safetensors` "
                         "(pip install huggingface_hub safetensors)") from e
    wname, want_sha, want_size = spec["weights"]
    log(f"{name}: {spec['repo']} @ {spec['revision'][:12]}, licence {spec['license']} ({spec['license_url']})")
    log(f"downloading {wname} ({want_size / 1e9:.1f} GB) and {len(spec['files'])} small files; the encoder-only copy "
        f"written to {dest} is about 1.1 GB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=dest.parent))
    try:
        got = {}
        for f in (*spec["files"], wname):
            got[f] = Path(hf_hub_download(spec["repo"], f, revision=spec["revision"], cache_dir=cache_dir))
        import hashlib
        h = hashlib.sha256()
        with open(got[wname], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want_sha:
            raise FetchError(f"{wname}: sha256 {h.hexdigest()[:12]}... differs from the pinned {want_sha[:12]}...; refusing it")
        for f in spec["files"]:
            shutil.copyfile(got[f], tmp / f)
        cfg = json.loads((tmp / "config.json").read_text(encoding="utf-8"))
        cfg["architectures"] = [spec["encoder_class"]]
        cfg["is_encoder_decoder"] = False
        cfg.pop("num_decoder_layers", None)
        (tmp / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        log("converting to an encoder-only model.safetensors (torch.load weights_only=True)")
        sd = torch.load(got[wname], map_location="cpu", weights_only=True)
        enc = extract_encoder_state(sd)
        del sd
        save_file(enc, str(tmp / "model.safetensors"), metadata={"format": "pt", "source": spec["repo"],
                                                                  "source_revision": spec["revision"]})
        sha = hash_weights(tmp)
        (tmp / LICENSE_FILE).write_text(json.dumps({
            "name": name, "license": spec["license"], "source_url": spec["source_url"], "sha256": sha,
            "source_repo": spec["repo"], "source_revision": spec["revision"], "source_file": wname,
            "source_file_sha256": want_sha, "dim": spec["dim"],
            "note": "encoder-only conversion of the original checkpoint; weights unchanged, not Nivra weights"},
            indent=1), encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)       # never leave a half install
        raise
    log(f"installed {name} in {dest}  (id external:{name}@{sha[:12]})")
    return dest
