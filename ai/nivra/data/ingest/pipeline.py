"""Folder -> sharded, provenance-tracked training dataset. See docs/training/DATA_PIPELINE.md."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from . import media
from .quality import DedupIndex, QualityLimits, clip_phash, phash_hex, quality_reject, quality_stats

INGEST_VERSION = 1
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bucket:
    name: str
    width: int
    height: int
    seconds: int
    frames: int          # 1 + 8k rule
    default_fps: int


BUCKETS = {
    "256p2s": Bucket("256p2s", 448, 256, 2, 17, 8),       # 17 frames == 2 s at 8 fps
    "480p4s": Bucket("480p4s", 854, 480, 4, 121, 30),     # 121 frames ~= 4 s at 30 fps
    "720p8s": Bucket("720p8s", 1280, 720, 8, 241, 30),    # 241 frames ~= 8 s at 30 fps
}


@dataclass
class IngestOptions:
    bucket: str = "256p2s"
    fps: float | None = None              # None -> bucket default
    max_clips: int | None = None
    scene_threshold: float | None = 0.3   # None -> no scene detection, fixed-length windows only
    scene_margin: float = 0.1             # seconds trimmed at each side of a detected cut
    shard_bytes: int = 128 * 1024 * 1024
    val_fraction: float = 0.05
    dedup_bits: float = 4.0
    allow_upscale: bool = False
    default_lang: str = "und"
    quality: QualityLimits = field(default_factory=QualityLimits)
    overwrite: bool = False


REQUIRED_PROVENANCE = {"source": str, "license": str, "license_url": str, "permission": str, "collected": str,
                       "contains_people": bool, "consent": str}


def load_provenance(root: Path) -> dict:
    p = root / "PROVENANCE.json"
    if not p.is_file():
        raise IngestError(f"{root}: PROVENANCE.json is required (source, license, license_url, permission, "
                          "collected, contains_people, consent). Refusing to ingest unlabelled data.")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise IngestError(f"{p}: invalid JSON: {e}")
    if not isinstance(d, dict):
        raise IngestError(f"{p}: must be a JSON object")
    may_be_empty = {"license_url", "consent"}
    for k, t in REQUIRED_PROVENANCE.items():
        if k not in d:
            raise IngestError(f"{p}: missing field {k!r}")
        if not isinstance(d[k], t):
            raise IngestError(f"{p}: field {k!r} must be of type {t.__name__}")
        if t is str and not d[k].strip() and k not in may_be_empty:
            raise IngestError(f"{p}: field {k!r} must not be empty")
    if d["contains_people"] and not d["consent"].strip():
        raise IngestError(f"{p}: contains_people is true, so 'consent' must describe the consent obtained")
    return d


def load_captions(root: Path) -> dict[str, dict]:
    """captions.jsonl rows keyed by posix relative path, bare file name and stem (first writer wins)."""
    out: dict[str, dict] = {}
    p = root / "captions.jsonl"
    if p.is_file():
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                f, c = r["file"], r["caption"]
            except (json.JSONDecodeError, KeyError, TypeError):
                raise IngestError(f"{p}:{n}: each line needs JSON keys 'file' and 'caption'")
            row = {"caption": str(c).strip(), "lang": r.get("lang")}
            fp = Path(f)
            for key in (fp.as_posix(), fp.name, fp.stem):
                out.setdefault(key, row)
    return out


def find_caption(video: Path, root: Path, jsonl: dict[str, dict], default_lang: str) -> tuple[str, str] | None:
    txt = video.with_suffix(".txt")
    if txt.is_file():
        c = txt.read_text(encoding="utf-8", errors="replace").strip()
        if c:
            return c, default_lang
    rel = video.relative_to(root).as_posix()
    for key in (rel, video.name, video.stem):
        if key in jsonl and jsonl[key]["caption"]:
            return jsonl[key]["caption"], jsonl[key]["lang"] or default_lang
    return None


class ShardWriter:
    """Fixed-capacity shard buffer (all clips share one shape) -> safetensors files with sha256 recorded."""

    def __init__(self, out: Path, split: str, shape: tuple, shard_bytes: int):
        self.out, self.split, self.shape = out, split, shape
        self.cap = max(1, shard_bytes // int(np.prod(shape)))
        self.buf: np.ndarray | None = None
        self.n = 0
        self.shard_no = 0
        self.files: list[dict] = []

    def current_name(self) -> str:
        return f"shard-{self.split}-{self.shard_no:05d}.safetensors"

    def add(self, clip: np.ndarray) -> tuple[str, int]:
        if self.buf is None:
            self.buf = np.empty((self.cap, *self.shape), dtype=np.uint8)
        name, idx = self.current_name(), self.n
        self.buf[self.n] = clip
        self.n += 1
        if self.n == self.cap:
            self.flush()
        return name, idx

    def flush(self):
        if not self.n:
            return
        name = self.current_name()
        path = self.out / name
        save_file({"videos": self.buf[: self.n]}, str(path), metadata={"nivra": json.dumps({"format": "nivra-clips/1", "split": self.split}, sort_keys=True)})  # one key: safetensors orders several keys randomly
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        self.files.append({"file": name, "split": self.split, "clips": self.n, "bytes": path.stat().st_size,
                           "sha256": h.hexdigest()})
        self.buf, self.n, self.shard_no = None, 0, self.shard_no + 1


def _split_of(rel: str, val_fraction: float) -> str:
    h = int(hashlib.sha256(rel.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def ingest_folder(input_dir, output_dir, opts: IngestOptions | None = None, log=print) -> dict:
    opts = opts or IngestOptions()
    media.need_tools()
    root = Path(input_dir).resolve()
    out = Path(output_dir)
    if opts.bucket not in BUCKETS:
        raise IngestError(f"unknown bucket {opts.bucket!r}; choose from {sorted(BUCKETS)}")
    if not root.is_dir():
        raise IngestError(f"input folder not found: {root}")
    prov = load_provenance(root)                 # refuse before touching the output directory
    b = BUCKETS[opts.bucket]
    fps = float(opts.fps or b.default_fps)
    window = b.frames / fps                       # source seconds consumed per clip
    if out.exists() and any(out.iterdir()):
        if not opts.overwrite:
            raise IngestError(f"output {out} is not empty (use --overwrite)")
        for q in out.iterdir():
            if q.name == "dataset.json" or q.name == "manifest.jsonl" or q.name.startswith("shard-"):
                q.unlink()
    out.mkdir(parents=True, exist_ok=True)

    captions = load_captions(root)
    videos = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
                    key=lambda p: p.relative_to(root).as_posix())
    shape = (b.frames, b.height, b.width, 3)
    writers = {s: ShardWriter(out, s, shape, opts.shard_bytes) for s in ("train", "val")}
    dedup = DedupIndex(opts.dedup_bits)
    manifest: list[dict] = []
    rejects: Counter = Counter()
    rejected_files: list[dict] = []
    per_file_clips: Counter = Counter()
    full = False

    def reject(why: str, rel: str, detail: str = ""):
        rejects[why] += 1
        rejected_files.append({"file": rel, "reason": why, **({"detail": detail} if detail else {})})
        log(f"  reject {rel}: {why} {detail}".rstrip())

    for vp in videos:
        if full:
            break
        rel = vp.relative_to(root).as_posix()
        cap = find_caption(vp, root, captions, opts.default_lang)
        if cap is None:
            reject("no_caption", rel)
            continue
        caption, lang = cap
        try:
            info = media.probe(vp)
        except media.MediaError as e:
            reject("corrupt", rel, str(e))
            continue
        scale = max(b.width / info.width, b.height / info.height)
        if scale > 1.0 and not opts.allow_upscale:
            reject("resolution_too_low", rel, f"{info.width}x{info.height} < {b.width}x{b.height}")
            continue
        if info.duration < window:
            reject("too_short", rel, f"{info.duration:.2f}s < {window:.2f}s")
            continue
        try:
            cuts = media.detect_cuts(vp, opts.scene_threshold) if opts.scene_threshold else []
        except media.MediaError as e:
            reject("corrupt", rel, str(e))
            continue
        scenes = media.scenes_from_cuts(info.duration, cuts)
        split = _split_of(rel, opts.val_fraction)
        for si, (s0, s1) in enumerate(scenes):
            # trim only at detected cuts, not at the file's own start/end
            lo = s0 + (opts.scene_margin if si > 0 else 0.0)
            hi = s1 - (opts.scene_margin if si < len(scenes) - 1 else 0.0)
            t = lo
            while t + window <= hi + 1e-6:
                start = t
                t += window
                if opts.max_clips and len(manifest) >= opts.max_clips:
                    full = True
                    break
                try:
                    clip = media.read_clip(vp, start, b.frames, fps, b.width, b.height)
                except media.MediaError as e:
                    reject("decode_error", rel, f"@{start:.2f}s {e}")
                    continue
                if clip is None:
                    reject("short_read", rel, f"@{start:.2f}s")
                    continue
                st = quality_stats(clip)
                why = quality_reject(st, opts.quality)
                if why:
                    reject(why, rel, f"@{start:.2f}s")
                    continue
                ph = clip_phash(clip)
                if dedup.is_duplicate(ph):
                    di, dd = dedup.nearest(ph)
                    reject("duplicate", rel, f"@{start:.2f}s of {manifest[di]['id']} (dist {dd:.2f})")
                    continue
                dedup.add(ph)
                cid = f"{b.name}-{hashlib.sha256(rel.encode()).hexdigest()[:8]}-{int(round(start * 1000)):07d}"
                shard, idx = writers[split].add(clip)
                manifest.append({
                    "id": cid, "split": split, "shard": shard, "index": idx, "bucket": b.name, "caption": caption,
                    "lang": lang, "source_file": rel, "start": round(start, 3), "end": round(start + window, 3),
                    "fps": fps, "frames": b.frames, "width": b.width, "height": b.height,
                    "source_resolution": [info.width, info.height], "source_codec": info.codec,
                    "scene_index": si, "provenance": prov, "phash": phash_hex(ph), "quality": st})
                per_file_clips[rel] += 1
            if full:
                break
        if per_file_clips[rel] == 0 and not full:
            log(f"  note {rel}: no clips produced")
    for w in writers.values():
        w.flush()
    if not manifest:
        log("warning: no clips were produced")
    # the manifest must reference the same shard names the writers produced (order-stable by construction)
    mtext = "".join(json.dumps(m, sort_keys=True, ensure_ascii=False) + "\n" for m in manifest)
    (out / "manifest.jsonl").write_bytes(mtext.encode("utf-8"))
    fingerprint = hashlib.sha256(mtext.encode("utf-8")).hexdigest()
    shards = sorted((f for w in writers.values() for f in w.files), key=lambda f: f["file"])
    summary = {
        "format": "nivra-dataset/1", "ingest_version": INGEST_VERSION, "fingerprint": fingerprint,
        "bucket": b.name, "fps": fps, "frames": b.frames, "width": b.width, "height": b.height,
        "clips": len(manifest), "train_clips": sum(m["split"] == "train" for m in manifest),
        "val_clips": sum(m["split"] == "val" for m in manifest), "videos_seen": len(videos),
        "rejected": dict(sorted(rejects.items())), "rejected_files": rejected_files,
        "options": {"scene_threshold": opts.scene_threshold, "scene_margin": opts.scene_margin,
                    "val_fraction": opts.val_fraction, "dedup_bits": opts.dedup_bits, "shard_bytes": opts.shard_bytes,
                    "allow_upscale": opts.allow_upscale, "quality": vars(opts.quality)},
        "provenance": prov, "shards": shards}
    (out / "dataset.json").write_text(json.dumps(summary, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return summary
