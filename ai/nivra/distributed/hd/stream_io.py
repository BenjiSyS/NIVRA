"""Tensor-by-tensor safetensors I/O for models that do not fit in RAM (runs on the Raspberry Pi controller).

* ``StReader`` parses and validates the header itself (offsets, sizes, dtypes) and reads ONE tensor at a time with
  seek/readinto. It deliberately does not mmap: on Windows a mapped file counts into the working set, so streaming
  four 2 GB deltas through ``safe_open`` would still look like 8 GB of RSS.
* ``StWriter`` writes the header first and the tensors in header order, hashing as it goes (peak = one tensor).
* ``expected_file_size`` / ``check_exact_size`` implement the "upload size must be exactly sum(numel * itemsize) +
  header" rule.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch

MAX_HEADER_BYTES = 64 << 20
DTYPES = {"F32": (torch.float32, 4, np.float32), "F16": (torch.float16, 2, np.float16),
          "BF16": (torch.bfloat16, 2, None)}
CODE_OF = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}


class StreamIOError(Exception):
    pass


def _numel(shape) -> int:
    return math.prod(int(d) for d in shape)


def header_bytes(entries: list, dtype, metadata: Optional[dict] = None) -> bytes:
    """The exact header (padded to 8 bytes) StWriter writes for [(name, shape)] entries."""
    code = CODE_OF[dtype]
    size = DTYPES[code][1]
    header, off = {}, 0
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for name, shape in entries:
        n = size * _numel(shape)
        header[name] = {"dtype": code, "shape": [int(d) for d in shape], "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return hb + b" " * ((-len(hb)) % 8)


def expected_file_size(entries: list, dtype, metadata: Optional[dict] = None) -> int:
    """Exact byte size of the file StWriter produces: 8 + header + sum(numel * itemsize)."""
    size = DTYPES[CODE_OF[dtype]][1]
    return 8 + len(header_bytes(entries, dtype, metadata)) + sum(size * _numel(s) for _, s in entries)


class StReader:
    """Validated random access to the tensors of a safetensors file, one tensor in memory at a time."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            self.size = self.path.stat().st_size
            self._f = open(self.path, "rb")
        except OSError as e:
            raise StreamIOError(f"cannot open {self.path.name}: {e}") from e
        try:
            self._parse()
        except Exception:
            self._f.close()
            raise

    def _parse(self):
        if self.size < 8:
            raise StreamIOError(f"{self.path.name}: too small to be safetensors")
        (hlen,) = struct.unpack("<Q", self._f.read(8))
        if hlen > MAX_HEADER_BYTES or 8 + hlen > self.size:
            raise StreamIOError(f"{self.path.name}: bad header length {hlen}")
        try:
            hdr = json.loads(self._f.read(hlen).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise StreamIOError(f"{self.path.name}: unreadable header: {e}") from e
        if not isinstance(hdr, dict):
            raise StreamIOError(f"{self.path.name}: header is not an object")
        self.header_len = hlen
        self.metadata = hdr.pop("__metadata__", None) or {}
        self.data_start = 8 + hlen
        self.info: dict = {}
        spans = []
        for name, e in hdr.items():
            try:
                code, shape, (a, b) = e["dtype"], [int(d) for d in e["shape"]], [int(o) for o in e["data_offsets"]]
            except Exception as ex:  # noqa: BLE001
                raise StreamIOError(f"{self.path.name}: bad entry for {name!r}") from ex
            if code not in DTYPES:
                raise StreamIOError(f"{self.path.name}: tensor {name!r} has unsupported dtype {code}")
            if any(d < 0 for d in shape) or not 0 <= a <= b or b - a != _numel(shape) * DTYPES[code][1]:
                raise StreamIOError(f"{self.path.name}: tensor {name!r} offsets/shape inconsistent")
            self.info[name] = (code, tuple(shape), a, b)
            spans.append((a, b, name))
        spans.sort()
        pos = 0
        for a, b, name in spans:                      # contiguous, non-overlapping, covering the whole data area
            if a != pos:
                raise StreamIOError(f"{self.path.name}: data offsets are not contiguous at {name!r}")
            pos = b
        if self.data_start + pos != self.size:
            raise StreamIOError(f"{self.path.name}: size {self.size} != header + tensor bytes {self.data_start + pos}")
        self.data_bytes = pos

    # ------------------------------------------------------------------ queries
    def names(self) -> list:
        return list(self.info)

    def shapes(self) -> dict:
        return {k: v[1] for k, v in self.info.items()}

    def dtypes(self) -> dict:
        return {k: v[0] for k, v in self.info.items()}

    def numel(self) -> int:
        return sum(_numel(v[1]) for v in self.info.values())

    # ------------------------------------------------------------------ reads
    def read(self, name: str) -> torch.Tensor:
        """The tensor in its stored dtype (a fresh, writable tensor)."""
        code, shape, a, b = self.info[name]
        tdt, _, npdt = DTYPES[code]
        self._f.seek(self.data_start + a)
        buf = bytearray(b - a)
        if b > a and self._f.readinto(buf) != b - a:
            raise StreamIOError(f"{self.path.name}: truncated tensor {name!r}")
        if npdt is not None:
            t = torch.from_numpy(np.frombuffer(buf, dtype=npdt).copy()) if b > a else torch.empty(0, dtype=tdt)
        else:
            t = torch.frombuffer(buf, dtype=tdt).clone() if b > a else torch.empty(0, dtype=tdt)
        return t.reshape(shape)

    def read_f32(self, name: str) -> torch.Tensor:
        return self.read(name).float()

    def close(self):
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class StWriter:
    """Streaming safetensors writer: header first, then ``write(name, tensor)`` in exactly the declared order."""

    def __init__(self, path, entries: list, dtype=torch.float32, metadata: Optional[dict] = None):
        if dtype not in CODE_OF:
            raise ValueError("StWriter supports float32 / float16 / bfloat16")
        self.path, self.dtype = Path(path), dtype
        self.entries = [(n, tuple(int(d) for d in s)) for n, s in entries]
        self._i = 0
        self._h = hashlib.sha256()
        self._f = open(self.path, "wb")
        hb = header_bytes(self.entries, dtype, metadata)
        for chunk in (struct.pack("<Q", len(hb)), hb):
            self._f.write(chunk)
            self._h.update(chunk)

    def write(self, name: str, t: torch.Tensor) -> None:
        if self._i >= len(self.entries):
            raise StreamIOError("more tensors written than declared")
        en, shape = self.entries[self._i]
        if name != en:
            raise StreamIOError(f"tensor order violated: expected {en!r}, got {name!r}")
        if tuple(t.shape) != shape:
            raise StreamIOError(f"{name}: shape {tuple(t.shape)} != declared {shape}")
        a = t.detach().to(self.dtype).contiguous().cpu()
        b = a.reshape(-1).view(torch.uint8).numpy().tobytes() if a.numel() else b""
        self._f.write(b)
        self._h.update(b)
        self._i += 1

    def close(self) -> str:
        if self._i != len(self.entries):
            self._f.close()
            raise StreamIOError(f"{self.path.name}: wrote {self._i} of {len(self.entries)} tensors")
        self._f.close()
        return self._h.hexdigest()

    def abort(self):
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is not None:
            self.abort()
        return False


def write_tensors(path, tensors: dict, dtype=torch.float32, metadata: Optional[dict] = None) -> str:
    """Convenience: write an in-memory dict through StWriter (deterministic: insertion order)."""
    with StWriter(path, [(k, v.shape) for k, v in tensors.items()], dtype, metadata) as w:
        for k, v in tensors.items():
            w.write(k, v)
        return w.close()


def check_exact_size(path, shapes: dict, dtype_code: str) -> Optional[str]:
    """None if the file holds exactly the declared tensors/shapes/dtype with no extra bytes, else a message.
    Size rule: file size == 8 + header_len + sum(numel * itemsize), no trailing or missing bytes."""
    try:
        with StReader(path) as r:
            if set(r.info) != set(shapes):
                return "tensor names differ from the base"
            for k, (code, shape, _, _) in r.info.items():
                if code != dtype_code:
                    return f"tensor {k} has dtype {code}, expected {dtype_code}"
                if shape != tuple(shapes[k]):
                    return f"tensor {k} has shape {shape}, expected {tuple(shapes[k])}"
            want = r.data_start + sum(_numel(s) * DTYPES[dtype_code][1] for s in shapes.values())
            if r.size != want:
                return f"file is {r.size} bytes, expected {want}"
    except StreamIOError as e:
        return str(e)
    return None


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()
