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
