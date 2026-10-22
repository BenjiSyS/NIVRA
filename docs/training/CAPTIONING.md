# Captioning videos for training

`nivra caption` writes, next to each video and with the same base name:

* `<name>.txt` - a description of what is visible (training caption). The first sentence is a style tag such as
  `Anime style.` when the style is known. With `--captioner qwen3vl` it is two lines: a short caption and a dense paragraph
  (see "Captioners" below); optionally `<name>.json` with the structured fields (`--json`).
* `<name>.subtitles.txt` - the spoken words or embedded subtitle text, plain text, one line per cue. Only written when
  the video has subtitles or speech.

Everything runs locally. Nothing is uploaded.

## Commands

Install once (adds `transformers`, `faster-whisper`, `pillow` to the Nivra Python environment; PyTorch must already be there):

    nivra caption --install

Caption the next 100 videos in a folder (recursive; `.mp4 .mov .mkv .webm`):

    nivra caption D:\videos\batch1 --limit 100

Continue with the next 100 (same command; videos that already have a `.txt` are skipped and do not count towards the limit):

    nivra caption D:\videos\batch1 --limit 100

Other forms:

    nivra caption clip.mp4                                  one file
    nivra caption D:\videos --style anime                   force the style tag
    nivra caption D:\videos --style none --subtitles off    no style tag, no subtitles
    nivra caption D:\videos --overwrite                     replace existing .txt captions
    nivra caption D:\videos --frames 3                      faster (fewer sampled frames)

Without the Go binary: `python -m nivra.cli caption PATH ...` with `PYTHONPATH=ai`.

| Flag | Meaning |
|------|---------|
| `--style auto\|anime\|3d animation\|cartoon\|2d animation\|cel-shaded 3d\|pixel art\|claymation\|painterly\|realistic\|none` | `auto` (default) detects the style from keywords in the model's frame captions and adds no tag if nothing is detected. |
| `--subtitles auto\|embedded\|speech\|off` | `auto`: embedded text subtitles if present, else speech recognition if there is an audio stream. |
| `--frames N` | Frames sampled per video (default 6, evenly spaced, avoiding the first/last 5%). |
| `--device auto\|cpu\|cuda` | `auto` uses CUDA when available (fp16). |
| `--overwrite` | Replace existing `.txt` captions. Without it, existing captions (even hand-written ones) are never touched and the video is skipped. |
| `--limit N` | Caption at most N videos that still need a caption. |
| `--captioner auto\|florence\|qwen3vl` | `auto` (default): `qwen3vl` when CUDA is available, `florence` on CPU-only machines. |
| `--json` | `qwen3vl`: also write `<name>.json` with the short/dense captions and the structured fields. |
| `--model ID_OR_DIR` | A transformers-format model repo id or local directory of the chosen captioner (Florence-2, or `Qwen/Qwen3-VL-2B-Instruct` / `-4B-Instruct`). |

Output while running:

    [12/100] clip12.mp4 ... done (34 s)
    [13/100] clip13.mp4 ... skipped (caption exists)
    [14/100] clip14.mp4 ... FAILED: ffprobe could not read the file: moov atom not found
    summary: 98 done (7 with subtitles), 1 skipped (existing), 1 failed, total 3400 s; 40 more need captions - run the same command again

Videos are processed in natural sort order of their relative path (`clip2` before `clip10`). Failed videos are listed
(path and reason) in `captions-failed.txt` in the folder; they are retried on the next run. Exit code: 0 all ok, 1 at least one
failure or a missing dependency, 2 bad arguments/path, 130 interrupted.

Ctrl+C stops after abandoning the current video. All files are written to a temporary file and renamed, the
`.subtitles.txt` first and the `.txt` last (the `.txt` is the "done" marker), so an interruption never leaves a half-written caption.

## How the description is made

1. `--frames` frames are extracted with ffmpeg (max 768 px side).
2. Florence-2-large (Microsoft, MIT) captions them: `<MORE_DETAILED_CAPTION>` on the middle frame, `<CAPTION>` on the others.
3. A deterministic merge builds `"<style> style. <detailed caption> The scene changes from <earlier> to <later>."`
   (change note only when the first and last frame captions differ), strips boilerplate ("The image shows", "In this image"),
   says "video"/"scene" instead of "image", and stays under 77 words.

Style keywords (anime, cartoon, 3d render/cgi, pixel art, claymation, cel-shaded, painterly, realistic, generic "animated" ->
2d animation) are voted across the frame captions. This is a heuristic on a captioning model's wording: it can miss or
mislabel; pass `--style` when you know the source.

Subtitles: embedded text streams (subrip, ass, mov_text, webvtt, ...) are extracted with ffmpeg and stripped of timing, tags and
consecutive duplicate lines. Bitmap-only streams (PGS/VobSub) are reported as unsupported (no OCR) and, in `auto` mode, speech
recognition is used instead. Speech uses faster-whisper `small` (MIT), language auto, voice-activity filter on.

## Captioners: Florence-2 or Qwen3-VL

`--captioner auto|florence|qwen3vl` (default `auto`):

| | `florence` | `qwen3vl` |
|---|---|---|
| Model | `florence-community/Florence-2-large` (MIT, 0.77 B) | `Qwen/Qwen3-VL-2B-Instruct` (**Apache-2.0**, 2 B; `--model Qwen/Qwen3-VL-4B-Instruct` for more detail) |
| Input | one frame at a time | all `--frames` frames of the clip at once (multi-image input, with their timestamps in the prompt) |
| Motion / camera | only a "scene changes from ... to ..." note | described explicitly, in order over time |
| Output | one caption (<= 77 words) | short caption + dense paragraph (+ optional JSON) |
| Default when | no CUDA GPU (CPU-only machines) | `auto` and CUDA is available, or `--captioner qwen3vl` |

Licence check (2026-10-22): the model cards of `Qwen/Qwen3-VL-2B-Instruct` and `-4B-Instruct` declare `license: apache-2.0` (Hugging Face
API, not gated); the code repository `QwenLM/Qwen3-VL` ships the Apache-2.0 text (https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE). The model
repositories themselves contain no separate LICENSE file, only the model-card metadata. Revision of the 2B pinned in `captioning/__init__.py`:
`89644892e4d85e24eaac8bacfd4f463576704203` (4.26 GB, bf16 safetensors). Avoid Qwen2.5-VL-3B (Qwen research licence, non-commercial).

### What `qwen3vl` writes

    <name>.txt     line 1: short caption, <= 25 words, starts with the style tag:  "Anime style. A girl in a red coat sprints across a rooftop at dusk."
                   line 2: dense paragraph (<= 200 words): subjects and appearance, action and motion over time, camera, setting, art style, lighting
    <name>.json    with --json: {"schema": "nivra.caption/1", "captioner", "model", "style", "short", "dense",
                   "fields": {subjects, motion, camera, setting, art_style, lighting}, "frame_times": [...]}

The model is asked for one JSON object (`structured.py`), parsed tolerantly (code fences and chatter are ignored; if it ignores the format its text becomes
the dense paragraph). `--style` still overrides the style tag; `none` drops it; otherwise the model's own `style` (one of the Nivra styles) or the keyword
vote is used. **Speech never enters the visual caption**: the prompt forbids speech, subtitles and any text, quoted strings are removed and any sentence that
talks about speech / subtitles / dialogue is dropped. Spoken words go to `<name>.subtitles.txt` as before.

The whole `.txt` (both lines) is what `nivra data-ingest` reads as the caption today. The two lengths are in the JSON record (`short` / `dense`) so a trainer can
sample between them (research item 7: mixed caption lengths during training); using the sidecar in ingest/training is not wired yet.

### Speed and memory

**Measured here: nothing with real weights.** This laptop (8 GB RAM, no GPU) has a ~1 MB/s connection to the hub: the 4.26 GB checkpoint needed about an hour,
so the download was abandoned and its partial files deleted (nothing of Qwen3-VL is left in the cache). What was verified instead, with the real tokenizer,
chat template and image processor files of the 2B repo and a random-weights model shrunk to 10 M parameters: frames -> chat template -> one `<|image_pad|>` per
merged patch -> `generate` runs end to end for 3 frames (`ai/tests/test_captioning_qwen.py` covers the rest with a fake model). **Caption quality and speed of the real
model are therefore untested** - run `nivra caption clip.mp4 --captioner qwen3vl --frames 4 --json` once on the target machine and read the result before captioning a dataset.

Estimates (not measurements), from the model sizes and the research note:

| Where | Weights in memory | Per clip (4-6 frames at 448 px) |
|---|---|---|
| RTX 2060 6 GB, fp16 (Turing has no bf16) | 2B: ~4.3 GB + activations, fits only with small frames (the code shrinks frames to 448 px) | a few seconds to ~10 s |
| RTX 2060, 4B | ~8.9 GB in fp16: does not fit; use 2B (4-bit loading is not wired) | - |
| CPU, 16 GB RAM, bf16 | 2B ~4.3 GB | minutes per clip (vision tower + ~400 generated tokens on CPU) |
| CPU, 8 GB RAM (this laptop) | 4.3 GB of weights plus Python, ffmpeg and the OS: marginal; stay with `florence` | - |

Frames are shrunk to 336 px (CPU) / 448 px (GPU) before the model sees them (`--device cpu` selects the smaller size). Weights are cached in the Hugging
Face cache (or `NIVRA_MODELS_DIR`); `NIVRA_QWEN_DTYPE=float32|bfloat16|float16` overrides the dtype (CPU default bf16 to halve memory; GPU default fp16).
`transformers` loads the model class natively (no remote code); no `torchvision` is needed because frames are passed as images, not as a video object.


## Models and versions (verified on this machine)

* Captioning model: `florence-community/Florence-2-large` (MIT), the Hugging Face transformers-format conversion of
  Microsoft's Florence-2-large, loaded with the native `Florence2ForConditionalGeneration` (no `trust_remote_code`).
  Pinned revision `4271c66b88cdbc05735372ec13b2360108de5317`. About 1.5 GB.
* Python 3.14, torch 2.14.1+cpu, transformers 5.20.2, faster-whisper 1.2.1 (CTranslate2 4.8.3), model `small` (~480 MB).
* Models are cached in the standard Hugging Face cache, or in `NIVRA_MODELS_DIR` when that is set.
* The original `microsoft/Florence-2-large` repo needs remote code and is not supported by `--model`.

## Speed

CPU only (8 logical cores): about 20 s per sampled frame (vision encoder dominates), so the default 6 frames is roughly
2-2.5 minutes per video plus model load (~20 s once per run); `--frames 3` roughly halves it. Speech recognition of a short clip
takes ~10 s on top. 100 videos at 6 frames is several hours on CPU; use a CUDA worker machine for large batches.
