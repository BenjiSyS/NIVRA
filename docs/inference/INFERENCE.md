# Nivra Small inference

Nivra Small turns a short English or Swedish scene description into a 16-frame 64x64 video
(2 s at 8 fps, written as MP4 upscaled 4x to 256x256 by nearest neighbour). It is a research model trained from
random initialisation on procedural data (circles / squares / triangles in 6 colours with 5 motions). It does
**not** generate photographic video and understands only its closed vocabulary (`nivra-wordtok/1`); unknown words
are mapped to UNK and reported.

Run from the repository root (`set PYTHONPATH=ai`, `set NIVRA_THREADS=4`):

```
.venv\Scripts\python.exe -m nivra.cli generate --bundle runs\release\nivra-small-6000 ^
    --prompt "en röd cirkel rör sig från vänster till höger" --out runs\samples --name demo --seed 7 --steps 30 --cfg 3.0
```

## Model bundle

A bundle is a directory with exactly the three files the executor jobs `ai.generate` / `ai.evaluate` take as
`weights`, `manifest`, `vae_weights`:

| file | content |
|---|---|
| `weights.safetensors` | generator weights (EMA weights for a trained model), float32 |
| `model_version.json` | version manifest (`nivra.model_version/1`, see `docs/training/DISTRIBUTED.md` 0.1) |
| `vae.safetensors` | frozen video VAE ("Nivra tokenizer") |

Loading verifies the weights sha256 against the manifest, the VAE sha256 against `vae_sha256`, the architecture and
tokenizer hashes, and the tensor shapes; any mismatch is refused (`BundleError`, executor exit 5).
The released bundle is `runs/release/nivra-small-6000/` (hashes in `docs/research/EXPERIMENTS.md`).
`python -m nivra.cli init-random --vae ... --out ...` writes an untrained bundle (benchmark baseline).

## Text to video

`generate(bundle, prompt, seed, steps, cfg_scale, tile)`: Euler integration of the flow-matching ODE from Gaussian
noise in the 4x8x16x16 latent space (`steps` network evaluations, each with classifier-free guidance = 2 forward
passes batched), then VAE decode to 16 frames. Same seed + same prompt + same bundle + same device/thread count gives
the same video (tested). `tile` > 0 decodes the latent in tiles to reduce peak memory (approximate: GroupNorm
statistics are per tile; measured tiled-decode PSNR 23.2 dB vs 29.5 dB untiled on the VAE test set), off by default.
Each call writes `<name>.mp4`, `<name>.json` (prompt, seed, timings, hardware, peak RSS, unknown words) and a
`<name>.state/` directory (see below).

Prompts: lowercase closed-vocabulary sentences, e.g. `a red circle moves from left to right`,
`two blue squares move from left to right`, `en grön cirkel står still, en lila kvadrat rör sig uppåt och en röd
triangel rör sig nedåt`, `ett rött föremål rör sig från vänster till höger`. Swedish and English are both supported
because both were in the training captions. Measured prompt adherence (it is modest) is in
`docs/research/EXPERIMENTS.md`; do not expect reliable multi-object scenes.

## Continuation

```
python -m nivra.cli continue --bundle runs\release\nivra-small-6000 --state runs\samples\en_red_circle_l2r.state ^
    --out runs\samples --name en_red_circle_l2r_continued --steps 30 --known 2
```

The saved generation state (`nivra.genstate/1`: prompt, seed, model version id, text ids, last latent frames,
temporal position, decoded frames so far) lets a second segment be generated conditioned on the last `--known`
latent frames (1 latent frame = 2 video frames) of the previous one. The known frames are re-imposed (noised along the
flow path) at every Euler step (inpainting-style), and the overlapping frames are dropped, so a 16-frame segment
adds 16 - 2*known frames: with `--known 2` the first continuation gives 16 + 12 = 28 frames (verified with ffprobe).
A state is only accepted by the exact model version that made it. `image_to_video` uses the same mechanism with an
encoded first frame (library API only; tested, no CLI command). Quality of continuation is not benchmarked.

## Executor

`ai.generate` (worker job, schema in `docs/training/DISTRIBUTED.md` 1.7) wraps the same code path:
inputs `weights`, `manifest`, `vae_weights`; outputs `<name>.mp4` + `<name>.json`; artifacts `video`,
`video_metadata`; the stop file is polled between sampling steps (exit 3).

## Nivra HD bundles

`nivra-hd/1` models (HC-VAE + DiT, configs `nivra_hd_tiny`, `nivra_medium`, `nivra_large`) use the same bundle
shape as the small model so the Go controller can store and ship them unchanged (it reads `weights_sha256`,
`vae_sha256`, `config_name`, `architecture_version`, `tokenizer_version`, `param_count`).

| file | content |
| --- | --- |
| `model_version.json` | schema `nivra.model_version/1`, `family: "hd"` |
| `weights.safetensors` | NivraDiT state dict (float32, or float16 for the large configs) |
| `vae.safetensors` | HC-VAE state dict + metadata `{kind, config}` (the artifact the controller sha-checks) |
| `text_encoder/` | optional NivraText checkpoint directory (`config.json`, `tokenizer.json`, `model.safetensors`) |

Manifest fields: `schema`, `family` (`"hd"`), `version_id` (`v000000-<weights sha12>`), `round` (0),
`parent_version_id`, `architecture_version` (`nivra-hd/1`), `config_name`, `config` (the full HD config, with the
VAE calibration), `config_hash`, `tokenizer_version` and `vocab_hash` (both = the text-encoder id), `param_count`
(DiT), `vae_param_count`, `tensor_shapes` (DiT, checked against the file header before hashing/loading),
`weights_file`, `weights_dtype`, `weights_sha256`, `outer_state_file`/`outer_state_sha256` (null),
`vae_sha256`, `applied_update_ids`, `created_unix`, `min_vram_mb` (fp16 DiT-sampling-phase estimate from the budget
formulas at the default resolution: weights + one block's activations + 0.7 GiB CUDA overhead; 0 for models under
10 M parameters, i.e. CPU is fine; Medium 4059; an estimate, never measured on a GPU), `default_resolution`
(`[width, height]`), `default_seconds`, `default_fps`, and `text_encoder`
`{id, kind: stub|nivra|external, path (relative to the bundle or null), name, sha256, license, dim, max_len}`.

`load_hd_bundle(dir | (weights, manifest, vae) | {weights, manifest, vae_weights}, device, **opts)` returns an
`HDPipeline`; it refuses (BundleError, exec exit 5) a weights or VAE sha256 mismatch, a tensor-shape mismatch, a VAE
whose embedded config differs from the manifest, and an unresolvable or mismatching text encoder. The
`HDPipeline` options are `fp16`, `int8`, `offload`, `attn_chunk`, `vae_tile`, ... (on CPU: float32, no offload).

Text encoder resolution (a worker receives only the three files, so the encoder is never inferred from a directory):
* `stub`: rebuilt from the manifest (hashed words, no language knowledge; tests and dry runs only).
* `nivra`: `<bundle dir>/<text_encoder.path>` when the bundle directory is available, otherwise
  `$NIVRA_TEXT_ENCODERS_DIR/<text_encoder.name>`. The checkpoint id and the sha256 of its `model.safetensors` must match
  the manifest.
* `external`: `$NIVRA_TEXT_ENCODERS_DIR/<text_encoder.name>` only, with its `NIVRA_LICENSE.json` (see
  `nivra/text/external.py`); never copied into a bundle and never downloaded.
Embedding the encoder in `weights.safetensors` was rejected: the tokenizer and config would have to be packed in too and
every encoder change would alter the DiT artifact hash the controller tracks.

```
python -m nivra.cli init-hd-bundle --config nivra_hd_tiny --out DIR [--text-encoder stub|nivra:CKPT|external:DIR] [--seed N]
python -m nivra.cli generate --bundle DIR --prompt "..." --out OUT [--width W --height H --seconds S --fps F
       --steps N --cfg C --sampler euler|ab2 --fp16/--no-fp16 --offload none|blocks]
```
`init-hd-bundle` writes a random-init bundle for plumbing tests (not a trained model). Models above 20 M parameters
are written tensor by tensor in float16, so Medium (2.2 GB) needs about 0.4 GB RAM to write (measured). Loading
Medium builds the model in host RAM first (about 3.7 GB float32 on CPU, half of that for a float16 build on CUDA hosts),
which does not fit comfortably next to other work on an 8 GB machine; only the tiny config was run end to end here.
For HD bundles `generate` defaults come from the bundle (Medium 1280x720, 8 s, 30 fps, 30 steps, cfg 5.0);
`--upscale` must be omitted or 1 and `--tile` is ignored. HD-only flags on a small bundle exit 2. The executor
(`ai.generate`, `docs/training/DISTRIBUTED.md` 1.7) and the ComfyUI nodes (Model Loader detects HD bundles; Text-to-Video
has optional `width`/`height`/`seconds`/`fps`/`sampler`, 0 / "model default" = bundle default, ignored for small
models; Video Continuation works, keeping the previous segment's padded frame size; Image-to-Video is small-only) use
the same loader. Continuation of an HD video via the library: `HDPipeline.continue_video`.

## Long videos (`ai.draft`, `ai.refine_segment`, `ai.stitch`)

Contract and controller side: `docs/architecture/LONG_VIDEO.md`. Code: `nivra/inference/longvideo.py` (logic),
`HDPipeline.refine` and `sample_flow(init=, strength=, anchors=)` in `nivra/inference/hd.py`, the streaming
`Mp4Writer` / `RawReader` in `video_io.py`, handlers in `nivra/executor.py`. HD bundles only (others exit 5).

* **`ai.draft`**: params `prompt, seed, style?, sampler?, seconds, fps, draft_width=256, draft_height=144`,
  `segment_frames` (1 + 8k, default 241), optional `steps, cfg_scale, name (draft), fp16, offload`. Plan:
  `F = floor(seconds*fps + 0.5) + 1` (half away from zero, like Go), segment *i* = `[i(S-1), min((i+1)(S-1), F-1)]`
  (`longvideo.segment_plan`, same test vectors as Go's `segmentPlan`). The first segment is `generate`d, every later
  one `continue_video`d (2 known latent frames), frames are streamed into the MP4 as they are produced (memory is
  one segment, not the video) and the last segment is trimmed to F. Artifacts `draft_video` (`<name>.mp4`, F frames at
  draft size) and `draft_plan` (`<name>_plan.json`: `frames, fps, segment_frames, segments[{index,start,end}]`).
  Progress: stage `draft`, fraction over segments, `step`/`total` = segments done / N. The stop file and
  `limits.max_seconds` are polled between sampling steps; a stopped job leaves no partial MP4.
* **`ai.refine_segment`**: inputs `weights, manifest, vae_weights, draft_video`; params `prompt, seed, style?,
  sampler?, width, height, fps, index, start, end, strength=0.55`, optional `steps, cfg_scale, name (segment),
  fp16, offload`. Draft frames `start..end` are read through ffmpeg (accurate seek, bicubic upscale to
  width x height), edge-padded to the VAE multiple and repeated up to 1 + 8k, encoded with the HC-VAE (frame chunks
  are converted to float lazily, so a 1440p segment is never a float tensor), noised to `strength` and sampled over
  the remaining part of the schedule (the first scheduled t <= strength: with `steps=30` and a shifted schedule
  roughly half of them run). The first and last latent frames are replaced by the draft latents (noised along the
  flow path) at every step and restored exactly at the end, so neighbouring segments meet at the draft frame they
  share. Decoded chunks go straight into the writer (CRF 12), padding is trimmed: the output has exactly
  `end - start + 1` frames at width x height. Artifact `segment_video`. `width`/`height` must be even (yuv420p),
  up to 2560 x 1440. Bad range / unreadable draft / range beyond the draft: exit 5.
* **`ai.stitch`**: inputs `segment_000 .. segment_{count-1}` (no model files), params `fps, count,
  crossfade_frames=4 (0 disables), name (video)`. Segments are read one frame at a time through ffmpeg pipes and
  encoded on the fly (CRF 14); memory is O(crossfade window). Every segment but the first drops its first frame
  (the shared boundary), so the total is `sum(n_i) - (N-1)` = F. Seam: the boundary frame B plus `c//2` frames
  before and `c - c//2 - 1` after it (c frames) are blends of both segments with weight `(j+1)/(c+1)` on the next
  segment; a segment has no frames before its start or after its end, so there its boundary frame is held (a
  short still ghost of 1-2 frames, not motion interpolation). `c` shrinks automatically next to very short
  segments. Segments of different sizes or an unreadable file: exit 5; a missing `segment_NNN` or extras: exit 2.

Measured on this CPU (tiny HD config, random weights, 2 threads; plumbing numbers only, not quality or GPU claims):
2560 x 1440, 9 frames, 2 sampling steps: peak RSS 5.6 GB (`generate`, untiled VAE) and 5.6 GB (`refine`, 4 steps,
about 95 s: 47 s encode, 46 s decode). With `vae_tile=16` the peak stays at about 5.5 GB and the time triples: the
decoder's receptive radius is 7 latent cells, so a 16-cell tile still reads a 30-cell window of the 45 x 80 grid.
`vae_tile=8` crashed the interpreter (segfault) in `HCVAE.decode_stream` at this size (not investigated; `hc_vae.py`
is outside this feature). Tiling only pays off on much larger latent grids than the tiny config has.
Not claimed: any visual quality, temporal coherence over minutes, or GPU behaviour (untested).

## Hardware profiles

Measured on this machine (the only hardware used): Intel i5-8250U, 4 cores / 8 threads, 8 GB RAM, no GPU,
Windows 11, torch 2.14.1+cpu, `NIVRA_THREADS=3` while other jobs were running (so timings are noisy, not best case).

| quantity | measured |
|---|---|
| parameters | generator 2,414,352; VAE 471,211 |
| 30-step generation, sampling | 1.3 - 2.6 s per video (10 videos, see `runs/samples/*.json`) |
| VAE decode | 0.03 - 0.05 s |
| peak RSS of `nivra.cli generate` | about 242 MB |
| bundle size on disk | about 9.7 MB generator + 2 MB VAE |
| 24-prompt benchmark, 20 steps (trained model) | 34 s |

The remaining profiles are **projections, not measurements**. Nothing below has been run.

| profile | status | expectation (projected, untested) |
|---|---|---|
| CPU, 8 GB RAM | measured (above) | works; inference is far inside the memory budget |
| RTX 2060 6 GB | projected / untested | model and activations are a few hundred MB; should fit with large margin; FP16 autocast possible |
| 8 / 12 / 16 / 24 GB GPUs | projected / untested | same code path, no change needed; the extra memory would only matter for a larger architecture than Nivra Small |

Precision policy: `NIVRA_DEVICE=cuda` enables `torch.autocast(device_type="cuda", dtype=torch.float16)`;
bf16 is never used; attention uses `F.scaled_dot_product_attention` with the math fallback, so FlashAttention is not
required. The CUDA path (autocast, GradScaler, `NIVRA_DEVICE=cuda` in the executor) has **never been executed**
because no CUDA device is available here; treat it as unverified code. On CPU everything runs in float32.

## Limitations

* Fixed output: 16 frames, 64x64 native (4x nearest-neighbour upscaled in the MP4 only), 8 fps by default.
* Closed vocabulary; Swedish prompts adhere noticeably worse than English ones (see benchmark by language).
* Counts, directions and shapes are frequently wrong with the 6000-step checkpoint; the detector-based benchmark
  quantifies this honestly.
* Reproducibility is per device/thread configuration; bit-identical output across CPU architectures is not claimed
  for generation (it is claimed and tested for the synthetic dataset and shard files only).
