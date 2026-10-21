"""Nivra HD model bundle: write, verify and load (DiT + HC-VAE + text-encoder reference).

A bundle is a directory (the Go controller stores/passes the three files individually):

    model_version.json    schema "nivra.model_version/1", family "hd" (see HD_MANIFEST_FIELDS)
    weights.safetensors   NivraDiT state dict (float32 or float16)
    vae.safetensors       HC-VAE state dict (+ metadata with its config); the artifact the controller checks
    text_encoder/         OPTIONAL: a NivraText checkpoint directory (config.json, tokenizer.json, model.safetensors)

Text encoder resolution (the worker may only receive the three files above):
    stub      deterministic hashed-word encoder rebuilt from the manifest (tests / dry runs only);
    nivra     <bundle dir>/<text_encoder.path> if that exists, else $NIVRA_TEXT_ENCODERS_DIR/<text_encoder.name>;
              the checkpoint id and the sha256 of its model.safetensors must match the manifest;
    external  $NIVRA_TEXT_ENCODERS_DIR/<text_encoder.name> (or the bundle path when one is recorded); must carry a
              NIVRA_LICENSE.json whose id/hash match. Nothing is ever downloaded.
Embedding the encoder into weights.safetensors was rejected: it would need the tokenizer and config packed in as
well, and would change the DiT artifact hash every time the encoder changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ..distributed import version as ver
from ..models.dit import count_params_meta
from ..models.hd_config import (HD_ARCH_VERSION, build_dit, build_hcvae, dit_config, hd_config_hash, load_hd_config,
                                vae_config)
from ..text.interface import TextEncoder
from ..training.checkpoint import sha256_file
from .bundle import BundleError
from .hd import HDPipeline, StubTextEncoder, frames_for, padded

FAMILY = "hd"
VAE_FILE = "vae.safetensors"
TEXT_DIR = "text_encoder"
ENV_TEXT_DIR = "NIVRA_TEXT_ENCODERS_DIR"
MAX_HD_BYTES = 16 << 30
LARGE_PARAM_THRESHOLD = 20_000_000          # above this, init-hd-bundle streams tensor by tensor
CUDA_OVERHEAD_MB = 0.7 * 1024               # same assumption as ai/scripts/hd_budget.py
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

HD_MANIFEST_FIELDS = (
    "schema", "family", "version_id", "round", "parent_version_id", "architecture_version", "config_name", "config",
    "tokenizer_version", "vocab_hash", "param_count", "vae_param_count", "tensor_shapes", "weights_file",
    "weights_dtype", "weights_sha256", "outer_state_file", "outer_state_sha256", "vae_sha256", "applied_update_ids",
    "created_unix", "min_vram_mb", "default_resolution", "default_seconds", "default_fps", "text_encoder")


def is_hd_manifest(manifest) -> bool:
    return (isinstance(manifest, dict) and manifest.get("family") == FAMILY
            and manifest.get("architecture_version") == HD_ARCH_VERSION)


def read_manifest_if_hd(path) -> Optional[dict]:
    """The manifest dict when `path` is a readable HD manifest, else None (never raises)."""
    try:
        m = ver.read_json_limited(path)
    except Exception:  # noqa: BLE001
        return None
    return m if is_hd_manifest(m) else None


# ===================================================================== VRAM estimate
def _activation_bytes(cfg, n_tokens: int, mlp_chunk: int) -> int:
    """Mirror of ai/scripts/hd_budget.py activation_bytes_inference (parity is asserted in the tests)."""
    N, d = n_tokens, cfg.dim
    c = min(N, mlp_chunk) if mlp_chunk > 0 else N
    fp16 = N * d * 2 * (1 + 1 + 3 + 2 + 1 + 1)
    rope32 = N * d * 4 * 3
    mlp = c * (cfg.mlp_hidden + d) * 2 * 2
    sampler = 5 * N * cfg.latent_channels * 4
    return fp16 + rope32 + mlp + sampler


def estimate_min_vram_mb(cfg: dict) -> int:
    """fp16 DiT-sampling-phase estimate at the config's default resolution/length (fp16 weights resident +
    one block's activations + CUDA overhead, as in the budget tool). 0 for toy models (CPU-capable).
    An estimate, not a measurement; the HC-VAE decode phase runs after the DiT has left the GPU."""
    dc, vc = dit_config(cfg), vae_config(cfg)
    params = count_params_meta(dc)
    if params["total"] < 10_000_000:
        return 0
    inf = cfg["inference"]
    frames = frames_for(float(inf.get("seconds", 8)), int(inf.get("fps", 30)), vc.temporal_factor)
    Hp = padded(int(inf.get("height", 720)), vc.spatial_factor)
    Wp = padded(int(inf.get("width", 1280)), vc.spatial_factor)
    _, Tl, Hl, Wl = vc.latent_shape(frames, Hp, Wp)
    act = _activation_bytes(dc, Tl * Hl * Wl, int(inf.get("mlp_chunk", 0)))
    return int(math.ceil((2 * params["total"] + act) / 2 ** 20 + CUDA_OVERHEAD_MB))


# ===================================================================== streaming safetensors writer
_ST_DTYPES = {torch.float32: ("F32", 4), torch.float16: ("F16", 2)}


def stream_safetensors(path, entries: list, produce: Callable[[str, tuple], torch.Tensor], dtype=torch.float32,
                       metadata: Optional[dict] = None) -> str:
    """Write a safetensors file tensor by tensor (header first, then data), so peak memory is one tensor.
    entries: [(name, shape)]; produce(name, shape) -> tensor of that shape. Returns the file's sha256."""
    if dtype not in _ST_DTYPES:
        raise ValueError("stream_safetensors supports float32 / float16")
    code, size = _ST_DTYPES[dtype]
    header, off = {}, 0
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for name, shape in entries:
        n = size * math.prod(shape)
        header[name] = {"dtype": code, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((-len(hb)) % 8)
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for chunk in (struct.pack("<Q", len(hb)), hb):
            f.write(chunk)
            h.update(chunk)
        for name, shape in entries:
            t = produce(name, tuple(shape))
            if tuple(t.shape) != tuple(shape):
                raise ValueError(f"{name}: produced shape {tuple(t.shape)}, expected {tuple(shape)}")
            b = t.detach().to(dtype).contiguous().cpu().numpy().tobytes()
            f.write(b)
            h.update(b)
    return h.hexdigest()


def _name_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}/{name}".encode()).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


def random_init_tensor(name: str, shape: tuple, seed: int) -> torch.Tensor:
    """Deterministic per-tensor random init for large random-init bundles (plumbing tests, not training)."""
    if len(shape) == 0:
        return torch.zeros(())
    if len(shape) == 1:
        return torch.ones(shape) if name.endswith(".weight") else torch.zeros(shape)
    g = torch.Generator().manual_seed(_name_seed(seed, name))
    return torch.randn(shape, generator=g) / math.sqrt(max(1, math.prod(shape[1:])))


# ===================================================================== text encoder references
def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:128]


def describe_text_encoder(spec: str, cfg: dict, out_dir: Path) -> tuple:
    """Resolve an init-time spec ('stub' | 'nivra:DIR' | 'external:DIR') to (manifest entry, copy-action)."""
    text_dim = int(cfg["model"]["text_dim"])
    if spec == "stub":
        enc = StubTextEncoder(dim=text_dim, max_len=16)
        return ({"id": enc.id, "kind": "stub", "path": None, "name": None, "sha256": None,
                 "license": "none (deterministic test stub, no learned language knowledge)",
                 "dim": enc.dim, "max_len": enc.max_len}, None)
    kind, _, target = spec.partition(":")
    if kind == "nivra" and target:
        src = Path(target)
        meta = json.loads((src / "config.json").read_text(encoding="utf-8"))
        from ..text.nivra_text import NivraTextEncoder
        enc = NivraTextEncoder.from_checkpoint(src)
        if enc.dim != text_dim:
            raise BundleError(f"text encoder {enc.id} has dim {enc.dim}, config {cfg['name']} expects {text_dim}")
        return ({"id": enc.id, "kind": "nivra", "path": TEXT_DIR, "name": _safe_name(enc.id),
                 "sha256": sha256_file(src / "model.safetensors"), "license": "nivra-original (trained from scratch)",
                 "dim": enc.dim, "max_len": enc.max_len, "origin": meta.get("origin", "nivra-original")}, src)
    if kind == "external" and target:
        from ..text.registry import get_text_encoder
        enc = get_text_encoder(spec)
        if enc.dim != text_dim:
            raise BundleError(f"text encoder {enc.id} has dim {enc.dim}, config {cfg['name']} expects {text_dim}")
        e = enc.manifest_entry()
        return ({"id": enc.id, "kind": "external", "path": None, "name": e["name"], "sha256": e["sha256"],
                 "license": e["license"], "dim": enc.dim, "max_len": enc.max_len, "source_url": e["source_url"]}, None)
    raise BundleError(f"bad text encoder spec {spec!r}; use stub | nivra:<dir> | external:<dir>")


def resolve_text_encoder(te: dict, bundle_dir: Optional[Path] = None, text_encoders_dir=None) -> TextEncoder:
    """Rebuild the text encoder a manifest refers to (see module docstring). Raises BundleError."""
    if not isinstance(te, dict) or not isinstance(te.get("id"), str) or te.get("kind") not in ("stub", "nivra", "external"):
        raise BundleError("manifest has no valid text_encoder entry")
    kind = te["kind"]
    if kind == "stub":
        enc = StubTextEncoder(dim=int(te.get("dim", 16)), max_len=int(te.get("max_len", 16)))
    else:
        root = text_encoders_dir or os.environ.get(ENV_TEXT_DIR)
        cands = []
        rel = te.get("path")
        if rel and bundle_dir is not None:
            base = Path(bundle_dir).resolve()
            p = (base / rel).resolve()
            if base != p and base not in p.parents:
                raise BundleError(f"text_encoder.path {rel!r} escapes the bundle directory")
            cands.append(p)
        name = te.get("name")
        if root and name:
            if not _SAFE_NAME.match(str(name)):
                raise BundleError(f"unsafe text_encoder.name {name!r}")
            cands.append(Path(root) / name)
        d = next((c for c in cands if c.is_dir()), None)
        if d is None:
            raise BundleError(f"text encoder {te['id']} ({kind}) not found; looked in "
                              f"{[str(c) for c in cands] or 'nowhere'}; place it at ${ENV_TEXT_DIR}/{name}")
        try:
            if kind == "nivra":
                from ..text.nivra_text import NivraTextEncoder
                if te.get("sha256") and sha256_file(d / "model.safetensors") != te["sha256"]:
                    raise BundleError(f"text encoder {d}: model.safetensors does not match the manifest sha256")
                enc = NivraTextEncoder.from_checkpoint(d, device="cpu")
            else:
                from ..text.external import load_external
                enc = load_external(d, max_len=int(te.get("max_len", 226)), device="cpu")
        except BundleError:
            raise
        except Exception as e:  # noqa: BLE001
            raise BundleError(f"cannot load text encoder {te['id']} from {d}: {e}") from e
    if enc.id != te["id"]:
        raise BundleError(f"text encoder id mismatch: manifest {te['id']!r}, found {enc.id!r}")
    return enc


# ===================================================================== writing
def _vae_metadata(cfg: dict) -> dict:
    return {"kind": "nivra-hcvae", "config": json.dumps(vae_config(cfg).to_dict(), sort_keys=True)}


def write_hd_bundle(out_dir, cfg, dit_state: Optional[dict] = None, vae_state: Optional[dict] = None,
                    text_encoder: str = "stub", seed: int = 0, dtype=None, version_id: Optional[str] = None) -> dict:
    """Write a bundle. With dit_state/vae_state None a random-init one is generated: small models in memory
    with the model's own init, large ones streamed tensor by tensor (peak RAM ~ one tensor)."""
    cfg = load_hd_config(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    te_entry, te_src = describe_text_encoder(text_encoder, cfg, out)
    dc = dit_config(cfg)
    params = count_params_meta(dc)
    with torch.device("meta"):
        dit_meta, vae_meta = build_dit(cfg), build_hcvae(cfg)
    dit_shapes = [(k, tuple(v.shape)) for k, v in dit_meta.state_dict().items()]
    vae_shapes = [(k, tuple(v.shape)) for k, v in vae_meta.state_dict().items()]
    big = params["total"] > LARGE_PARAM_THRESHOLD
    dtype = dtype or (torch.float16 if big else torch.float32)

    if dit_state is None and vae_state is None and not big:
        torch.manual_seed(seed)
        dit, vae = build_dit(cfg), build_hcvae(cfg)
        with torch.no_grad():                       # non-degenerate outputs (the model's own init zeroes some heads)
            for p in list(dit.parameters()) + list(vae.parameters()):
                p.add_(torch.randn_like(p) * 0.02)
        dit_state, vae_state = dit.state_dict(), vae.state_dict()
    if dit_state is not None:
        dprod = lambda n, s: dit_state[n]            # noqa: E731
    else:
        dprod = lambda n, s: random_init_tensor(n, s, seed)             # noqa: E731
    if vae_state is not None:
        vprod = lambda n, s: vae_state[n]            # noqa: E731
    else:
        vprod = lambda n, s: random_init_tensor("vae." + n, s, seed)    # noqa: E731

    wsha = stream_safetensors(out / ver.WEIGHTS_FILE, dit_shapes, dprod, dtype, {"kind": "nivra-dit"})
    vsha = stream_safetensors(out / VAE_FILE, vae_shapes, vprod, dtype, _vae_metadata(cfg))
    if te_src is not None:
        dst = out / TEXT_DIR
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(te_src, dst)
    inf = cfg["inference"]
    manifest = {
        "schema": ver.SCHEMA, "family": FAMILY,
        "version_id": version_id or ver.make_version_id(0, wsha), "round": 0, "parent_version_id": None,
        "architecture_version": HD_ARCH_VERSION, "config_name": cfg["name"], "config": cfg,
        "tokenizer_version": te_entry["id"], "vocab_hash": te_entry["id"],
        "param_count": int(params["total"]), "vae_param_count": int(sum(math.prod(s) for _, s in vae_shapes)),
        "tensor_shapes": {k: list(s) for k, s in dit_shapes},
        "weights_file": ver.WEIGHTS_FILE, "weights_dtype": str(dtype).replace("torch.", ""), "weights_sha256": wsha,
        "outer_state_file": None, "outer_state_sha256": None, "vae_sha256": vsha, "applied_update_ids": [],
        "created_unix": int(time.time()), "min_vram_mb": estimate_min_vram_mb(cfg),
        "default_resolution": [int(inf.get("width", 1280)), int(inf.get("height", 720))],
        "default_seconds": float(inf.get("seconds", 8)), "default_fps": int(inf.get("fps", 30)),
        "text_encoder": te_entry, "config_hash": hd_config_hash(cfg),
    }
    (out / ver.MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


# ===================================================================== loading
def _check_file(path, shapes: Optional[dict], what: str) -> None:
    try:
        ver.check_tensor_file(path, shapes, None, MAX_HD_BYTES)
    except ver.VersionError as e:
        raise BundleError(f"{what}: {e}") from e


def load_hd_bundle_files(weights, manifest, vae_weights, device: str = "cpu", *, bundle_dir=None,
                         text_encoders_dir=None, text_encoder: Optional[TextEncoder] = None,
                         **overrides) -> HDPipeline:
    """Verify sha256 of the DiT and VAE against the manifest (refusing mismatches) and build an HDPipeline.
    overrides are HDPipeline inference options (fp16, int8, offload, attn_chunk, vae_tile, ...)."""
    try:
        m = ver.read_manifest(manifest)
    except (ver.VersionError, OSError) as e:
        raise BundleError(str(e)) from e
    if not is_hd_manifest(m):
        raise BundleError("not an HD bundle manifest (family != 'hd' or wrong architecture_version)")
    weights, vae_weights = Path(weights), Path(vae_weights)
    try:
        _check_file(weights, m.get("tensor_shapes"), "weights")
        if sha256_file(weights) != m["weights_sha256"]:
            raise BundleError("weights sha256 does not match version manifest")
        if not m.get("vae_sha256"):
            raise BundleError("HD manifest has no vae_sha256")
        _check_file(vae_weights, None, "VAE")
        if sha256_file(vae_weights) != m["vae_sha256"]:
            raise BundleError("VAE weights do not match the VAE this model version was built with")
    except OSError as e:
        raise BundleError(f"cannot read bundle file: {e}") from e
    try:
        cfg = load_hd_config(m["config"])
        with safe_open(str(vae_weights), framework="pt") as f:
            vmeta = f.metadata() or {}
        if json.loads(vmeta.get("config", "null")) != vae_config(cfg).to_dict():
            raise BundleError("VAE metadata config does not match the manifest config")
    except BundleError:
        raise
    except Exception as e:  # noqa: BLE001
        raise BundleError(f"invalid HD config/VAE metadata: {e}") from e
    bdir = Path(bundle_dir) if bundle_dir is not None else Path(manifest).resolve().parent
    enc = text_encoder if text_encoder is not None else resolve_text_encoder(m.get("text_encoder"), bdir, text_encoders_dir)
    cuda = str(device).startswith("cuda")
    try:
        dit_sd = load_file(str(weights))
        small_dtype = cuda and overrides.get("fp16", True) and next(iter(dit_sd.values())).dtype == torch.float16
        prev = torch.get_default_dtype()
        if small_dtype:                          # build in fp16 on GPU hosts: no transient fp32 copy in host RAM
            torch.set_default_dtype(torch.float16)
        try:
            dit, vae = build_dit(cfg), build_hcvae(cfg)
        finally:
            torch.set_default_dtype(prev)
        dit.load_state_dict(dit_sd, strict=True)
        del dit_sd
        vae.load_state_dict(load_file(str(vae_weights)), strict=True)
    except (RuntimeError, ValueError, OSError) as e:
        raise BundleError(f"weights do not match architecture: {e}") from e
    for p in list(dit.parameters()) + list(vae.parameters()):
        p.requires_grad_(False)
    if not cuda:
        overrides["offload"] = False             # block offload only makes sense with a GPU
    try:
        return HDPipeline.from_config(cfg, enc, dit=dit, vae=vae, device=device, model_version_id=m["version_id"],
                                      **overrides)
    except (TypeError, ValueError) as e:
        raise BundleError(str(e)) from e


def load_hd_bundle(source, device: str = "cpu", **kw) -> HDPipeline:
    """source: bundle directory, a (weights, manifest, vae_weights) tuple, or a dict with those keys."""
    if isinstance(source, dict):
        return load_hd_bundle_files(source["weights"], source["manifest"], source["vae_weights"], device, **kw)
    if isinstance(source, (tuple, list)):
        w, mf, v = source
        return load_hd_bundle_files(w, mf, v, device, **kw)
    d = Path(source)
    return load_hd_bundle_files(d / ver.WEIGHTS_FILE, d / ver.MANIFEST_FILE, d / VAE_FILE, device, bundle_dir=d, **kw)


# ===================================================================== saving results
def save_hd_result(res, out_dir, name: str, extra_meta: Optional[dict] = None) -> dict:
    """Metadata JSON + generation state next to the MP4 the pipeline already wrote (same layout as small)."""
    out_dir = Path(out_dir)
    meta = dict(res.meta)
    meta.update(mp4=f"{name}.mp4", native_size=[meta["height"], meta["width"]], upscale=1,
                seconds=meta["frames"] / meta["fps"], temporal_position=res.state.temporal_position,
                segments=res.state.segments, family=FAMILY)
    if extra_meta:
        meta.update(extra_meta)
    res.state.save(out_dir / f"{name}.state")
    meta["state_dir"] = f"{name}.state"
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def hd_summary(meta: dict) -> dict:
    keys = ("frames", "fps", "width", "height", "seconds", "seconds_text", "seconds_sampling", "seconds_decode",
            "peak_rss_mb", "cuda_max_allocated_mb", "model_version_id", "text_encoder", "sampler", "steps")
    s = {k: meta[k] for k in keys if k in meta}
    s["family"] = FAMILY
    return s
