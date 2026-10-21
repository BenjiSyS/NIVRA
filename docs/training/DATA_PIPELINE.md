# Real-video data pipeline

Contract: `docs/research/NIVRA_720P_DESIGN.md` section 7. Code: `ai/nivra/data/ingest/`, `ai/nivra/data/shards.py`.

```
python -m nivra.cli data-ingest --input <folder> --output <dataset dir> [--bucket 256p2s|480p4s|720p8s]
        [--fps N] [--max-clips N] [--scene-threshold X | --no-scenes] [--shard-mb 128]
        [--val-fraction 0.05] [--dedup-bits 4] [--allow-upscale] [--default-lang und] [--overwrite]
        [--profile stylized|live] [--default-style anime]
python -m nivra.cli data-inspect <dataset dir>      # counts, sample captions, checksum validation (exit 1 if damaged)
```

## Input folder

* videos `.mp4 .mov .mkv .webm` (recursive); caption from `<stem>.txt` or `captions.jsonl` lines
  `{"file": ..., "caption": ..., "lang": ...}` (matched by relative path, file name or stem). No caption: file rejected.
* `PROVENANCE.json` is REQUIRED: `source, license, license_url, permission, collected, contains_people (bool), consent`.
  Missing/invalid file, or `contains_people: true` without a `consent` text, aborts before anything is written. The record is copied into
  every manifest line and into `dataset.json`.

## Steps

1. ffprobe: codec, resolution, fps, duration; no video stream / zero duration / undecodable = `corrupt`.
2. Reject `resolution_too_low` (the cover-scale would need upscaling; override with `--allow-upscale`) and `too_short`.
3. Scene cuts: ffmpeg `select='gt(scene,T)'` on a 160 px downscaled decode (one streaming pass). Windows never straddle a cut
   (0.1 s margin trimmed next to cuts). `--no-scenes` (or a video without cuts) gives fixed-length windows.
4. Each scene is tiled with non-overlapping windows. A window is decoded through an ffmpeg pipe with `fps` resampling, aspect-preserving
   resize (cover) and center crop, straight into one preallocated array.
5. Frame rule 1+8k and sizes: `256p2s` 448x256, 17 frames (8 fps default = 2.1 s); `480p4s` 854x480, 121 frames; `720p8s` 1280x720, 241 frames
   (30 fps default). `--fps` changes the sampling rate; the frame count stays fixed, so the source span is frames/fps.
6. Quality filters on 4x subsampled luma: `black`, `too_bright`, `flat` (blank/single colour), `static` (mean frame difference below limit).
7. Dedup: 8 evenly spaced frames, 64-bit dHash each; a clip is a duplicate when the mean Hamming distance to any kept clip is <= `--dedup-bits`.
   Cross-dataset, in deterministic file order (sorted relative path).
8. Split: `val` if sha256(relative source path) falls in the first `--val-fraction`, so one source never spans train and val.

## Output

* `shard-{train,val}-NNNNN.safetensors`, tensor `videos` uint8 `[N,T,H,W,3]`; sizes limited by `--shard-mb` (at least one clip per shard).
* `manifest.jsonl`: id, split, shard, index, bucket, caption, style, lang, source file (relative), start/end, fps, frames, resolution, source
  resolution/codec, scene index, provenance, phash, quality stats. No timestamps or absolute paths, so re-running gives an identical file.
* `dataset.json`: summary, options, reject counts and reasons, shard list with sha256, `fingerprint` = sha256 of `manifest.jsonl`.

Reading: `ShardDataset(dir, split="train", verify=False)` yields `(uint8 [T,H,W,3], caption, metadata)`; clips are sliced out of the shards one at a
time. `ShardDataset.to_float` converts to `[3,T,H,W]` in [-1,1]. `verify_shards(dir)` checks checksums and the fingerprint.

## Stylized profile (anime / cartoon / 2D / 3D animation)

`data-ingest --profile stylized|live` (default `stylized`) and `--default-style <tag>`. Code: `ai/nivra/data/styles.py`, `ai/nivra/data/ingest/`.

* **Style per video**, first match wins: (1) a leading style phrase in the caption (`anime style. A girl runs ...`; the phrase is removed from the stored
  caption and moved to `style`), (2) `style` in the video's `captions.jsonl` line, (3) optional `style` in `PROVENANCE.json`, (4) `--default-style`,
  else `unspecified`. Values may be a canonical tag or an alias (see `styles.py`: `anime`, `3d animation`, `cartoon`, `2d animation`, `cel-shaded 3d`,
  `pixel art`, `claymation`, `painterly`, `realistic`); an unknown value aborts the ingest (before output is written for the option / PROVENANCE cases).
* Every `manifest.jsonl` row has `style`; `dataset.json` has `styles` (clips per style), `profile`, and the effective quality limits in `options`.
  `data-inspect` reports `per_style`. Datasets from before this change have no `style` and read as `unspecified`.
* **`stylized` filters**: `flat` rejects only near single-colour clips (luma std < 1 instead of 4; big flat areas are normal in cel shading). `static` is
  held-frame aware: animation is drawn on 2s/3s, so the plain mean frame difference is misleading. A clip is `static` only if fewer than 15 % of frame
  pairs change at all, or the pairs that do change differ by less than 0.3 (mean |luma diff|); the mean-difference test is off. Scene threshold `auto`
  = 0.4 (hard cuts score far higher; drawn-on-twos pose jumps should not be split). `--profile live` keeps the original limits (4.0 / 0.35 / 0.3).
  `--scene-threshold X` always wins over the profile default; `--no-scenes` disables scene detection.
* Measured only on synthetic clips (flat-colour squares on flat backgrounds, held on 2s/3s, tests in `ai/tests/test_stylized_ingest.py`): a small shape
  moving on threes is `static` under `live` and kept under `stylized`; a low-contrast flat clip is `flat` under `live` and kept under `stylized`; fully
  static, single-colour, black and over-bright clips are rejected by both. The thresholds are not tuned on real anime or cartoon footage.

## Memory

Only one clip window is decoded at a time and shards use one preallocated buffer. Peak is about one shard buffer plus its serialisation copy plus
one clip: 720p8s clips are 666 MB each, so with 720p8s a shard holds exactly one clip and the peak is around 2x that. Choose `--shard-mb` accordingly.

## Limits

* Dedup is visual near-duplicate detection on 8 frames, not semantic: different windows of an almost static synthetic pattern are treated as copies.
* No audio, no motion-score-based or aesthetic filters, no latent precomputation yet (planned once the HC-VAE exists).
* Quality thresholds are defaults chosen for sanity, not tuned on real footage.
