# Captioning videos for training

`nivra caption` writes, next to each video and with the same base name:

* `<name>.txt` - a description of what is visible (training caption). The first sentence is a style tag such as
  `Anime style.` when the style is known.
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
| `--model ID_OR_DIR` | A transformers-format Florence-2 repo id or local directory. |

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
