"""Safe extraction of a text-encoder archive (artifact kind `text_encoder_archive`).

The Go controller packs a bundle's `text_encoder/` directory into a deterministic, uncompressed tar and ships it
to workers as the job input `text_encoder_archive`. The archive comes over the network, so it is treated as
untrusted: regular files only, plain relative names (no absolute paths, `..`, backslashes, drive letters),
size and count limits, nothing is executed. The extracted directory is cached by the archive's sha256 and
published with an atomic rename, so concurrent jobs never see a half-written encoder.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tarfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Optional

from .bundle import BundleError

MAX_ARCHIVE_BYTES = 4 << 30
MAX_FILES = 64
WEIGHTS_FILE = "model.safetensors"
_BAD_CHARS = re.compile(r"[\\:\x00-\x1f\x7f]")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _safe_member_name(name: str) -> PurePosixPath:
    if not name or len(name) > 200 or _BAD_CHARS.search(name) or name.startswith("/"):
        raise BundleError(f"text encoder archive: unsafe member name {name!r}")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise BundleError(f"text encoder archive: unsafe member name {name!r}")
    return PurePosixPath(*parts)


def extract_text_encoder_archive(archive, cache_root, expect_weights_sha: Optional[str] = None) -> Path:
    """Extract `archive` into `<cache_root>/<archive sha256>` (reusing a previous extraction) and return that
    directory. Raises BundleError for anything unsafe or when model.safetensors does not hash to
    `expect_weights_sha` (the manifest's text_encoder.sha256) when one is given."""
    archive, cache_root = Path(archive), Path(cache_root)
    try:
        asha = _sha256_file(archive)
    except OSError as e:
        raise BundleError(f"cannot read text encoder archive: {e}") from e
    target = cache_root / asha
    if target.is_dir():
        # published only after full verification below, so a present directory is complete; still re-check the hash
        if expect_weights_sha and _sha256_file(target / WEIGHTS_FILE) != expect_weights_sha:
            raise BundleError("text encoder archive: model.safetensors does not match the manifest sha256")
        return target
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp = cache_root / f".tmp-{asha[:12]}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        _extract(archive, tmp)
        wp = tmp / WEIGHTS_FILE
        if not wp.is_file():
            raise BundleError(f"text encoder archive has no {WEIGHTS_FILE}")
        if expect_weights_sha and _sha256_file(wp) != expect_weights_sha:
            raise BundleError("text encoder archive: model.safetensors does not match the manifest sha256")
        try:
            os.rename(tmp, target)
        except OSError:
            if not target.is_dir():          # lost a race to another extractor: theirs is equally valid
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def _extract(archive: Path, dest: Path) -> None:
    try:
        tf = tarfile.open(archive, mode="r:")        # "r:" = uncompressed only
    except (tarfile.TarError, OSError) as e:
        raise BundleError(f"text encoder archive is not a valid uncompressed tar: {e}") from e
    base = dest.resolve()
    seen: set = set()
    total = 0
    with tf:
        try:
            for m in tf:
                if not m.isreg():
                    raise BundleError(f"text encoder archive: member {m.name!r} is not a regular file")
                rel = _safe_member_name(m.name)
                if rel in seen:
                    raise BundleError(f"text encoder archive: duplicate member {m.name!r}")
                seen.add(rel)
                total += m.size
                if len(seen) > MAX_FILES or total > MAX_ARCHIVE_BYTES:
                    raise BundleError("text encoder archive exceeds the file-count or size limit")
                out = (base / Path(*rel.parts)).resolve()
                if base not in out.parents:
                    raise BundleError(f"text encoder archive: {m.name!r} escapes the extraction directory")
                out.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    raise BundleError(f"text encoder archive: cannot read member {m.name!r}")
                with src, open(out, "xb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        except tarfile.TarError as e:
            raise BundleError(f"text encoder archive is corrupt: {e}") from e
        except OSError as e:
            raise BundleError(f"text encoder archive cannot be extracted: {e}") from e
    if not seen:
        raise BundleError("text encoder archive is empty")
