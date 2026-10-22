"""Opt-in adapter for a separately licensed, locally stored text encoder (e.g. a T5/UMT5-family encoder).

Rules enforced here:
* the encoder lives in a LOCAL directory the owner placed there; nothing is ever downloaded
  (URLs / UNC network paths are refused, `local_files_only=True`, HF offline env vars are set);
* a `NIVRA_LICENSE.json` {name, license, source_url, sha256} must sit next to the weights, and `sha256`
  must equal the real hash of the weight file(s) (single file: its sha256; several: sha256 of the sorted
  per-file hex digests joined by newlines);
* the encoder is identified as `external:<name>@<sha256[:12]>` and is never presented as Nivra weights.

Building the actual network needs the optional `transformers` package. Without it a clear error is raised.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable

import torch

LICENSE_FILE = "NIVRA_LICENSE.json"
REQUIRED_FIELDS = ("name", "license", "source_url", "sha256")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class ExternalEncoderError(RuntimeError):
    pass


def _refuse_remote(path) -> Path:
    s = str(path)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", s) or s.startswith("\\\\") or s.startswith("//"):
        raise ExternalEncoderError(
            f"refusing non-local encoder location {s!r}: external encoders must be a local directory; "
            "Nivra never downloads models")
    p = Path(s)
    if not p.is_dir():
        raise ExternalEncoderError(f"external encoder directory not found: {s}")
    return p


def weight_files(d: Path) -> list[Path]:
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise ExternalEncoderError(f"{d}: no *.safetensors weights found (pickle formats are not accepted)")
    return files


def hash_weights(d: Path) -> str:
    def one(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    digests = [one(p) for p in weight_files(d)]
    return digests[0] if len(digests) == 1 else hashlib.sha256("\n".join(digests).encode()).hexdigest()


def read_license(d: Path) -> dict:
    lp = d / LICENSE_FILE
    if not lp.is_file():
        raise ExternalEncoderError(
            f"{d}: missing {LICENSE_FILE}. The owner must record name, license, source_url and sha256 "
            "for every separately licensed encoder before Nivra will load it.")
    info = json.loads(lp.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_FIELDS if not isinstance(info.get(k), str) or not info[k].strip()]
    if missing:
        raise ExternalEncoderError(f"{lp}: missing/empty fields {missing}")
    if not _SAFE_NAME.match(info["name"]):
        raise ExternalEncoderError(f"{lp}: name must match [A-Za-z0-9._-]+")
    info["sha256"] = info["sha256"].lower()
    return info


class ExternalTextEncoder:
    """TextEncoder-interface wrapper around a backend with `.dim` and `.encode(prompts, max_len)`."""

    def __init__(self, directory: Path, info: dict, backend, max_len: int):
        self.directory = Path(directory)
        self.license = dict(info)
        self.backend = backend
        self.dim = int(backend.dim)
        self.max_len = int(max_len)
        self.id = f"external:{info['name']}@{info['sha256'][:12]}"

    def manifest_entry(self) -> dict:
        """What a model manifest records: never relabelled as Nivra weights."""
        return {"text_encoder": self.id, "origin": "external-third-party", "dim": self.dim,
                "license": self.license["license"], "source_url": self.license["source_url"],
                "sha256": self.license["sha256"], "name": self.license["name"]}

    @torch.no_grad()
    def encode(self, prompts: list[str]):
        if not prompts:
            raise ValueError("encode() needs at least one prompt")
        emb, mask = self.backend.encode(prompts, self.max_len)
        return emb.float().contiguous(), mask.bool()


class TransformersBackend:
    """Builds a T5/UMT5/mT5 encoder (or any AutoModel with an encoder) from local files via `transformers`."""

    def __init__(self, directory: Path, device="cpu", dtype=torch.float32):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import transformers
        except ImportError as e:
            raise ExternalEncoderError(
                "external text encoders need the optional `transformers` package (plus `sentencepiece` for "
                "T5-family tokenizers). Install them yourself, e.g. `pip install transformers sentencepiece`, "
                "and place the encoder files in a local directory. Nivra will never download anything.") from e
        kw = dict(local_files_only=True)
        cfg = transformers.AutoConfig.from_pretrained(str(directory), **kw)
        cls_name = {"t5": "T5EncoderModel", "umt5": "UMT5EncoderModel", "mt5": "MT5EncoderModel"}.get(cfg.model_type)
        if cls_name and hasattr(transformers, cls_name):
            model = getattr(transformers, cls_name).from_pretrained(str(directory), dtype=dtype, **kw)
        else:
            model = transformers.AutoModel.from_pretrained(str(directory), dtype=dtype, **kw)
            if hasattr(model, "get_encoder"):
                model = model.get_encoder()
        self.model = model.eval().to(device)
        self.tok = transformers.AutoTokenizer.from_pretrained(str(directory), **kw)
        self.device = device
        self.dim = int(getattr(cfg, "d_model", None) or getattr(cfg, "hidden_size"))

    @torch.no_grad()
    def encode(self, prompts, max_len):
        t = self.tok(list(prompts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids, att = t["input_ids"].to(self.device), t["attention_mask"].to(self.device)
        out = self.model(input_ids=ids, attention_mask=att).last_hidden_state
        return out.float().cpu(), att.bool().cpu()


def load_external(directory, max_len: int = 226, device="cpu",
                  backend_factory: Callable[[Path], object] | None = None) -> ExternalTextEncoder:
    """Load a local external encoder. `backend_factory(dir)` may be injected (tests / other runtimes)."""
    d = _refuse_remote(directory)
    info = read_license(d)
    actual = hash_weights(d)
    if actual != info["sha256"]:
        raise ExternalEncoderError(
            f"{d}: weights sha256 {actual[:12]}... does not match {LICENSE_FILE} sha256 {info['sha256'][:12]}...")
    backend = backend_factory(d) if backend_factory else TransformersBackend(d, device=device)
    return ExternalTextEncoder(d, info, backend, max_len)
