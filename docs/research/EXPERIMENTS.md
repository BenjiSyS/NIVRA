# Nivra Small - experiment log

Every number here was measured on one machine: Intel i5-8250U (4 cores / 8 threads), 8 GB RAM, no GPU, Windows 11,
Python venv with torch 2.14.1+cpu. Wall times are noisy because other jobs (a second trainer, the Go controller
and its workers, tests) shared the CPU during most runs. Raw artefacts live under the gitignored `runs/` directory.
No pretrained weights or external datasets were used; all weights are original and randomly initialised.

Commands assume `set PYTHONPATH=ai` and the project venv (`.venv\Scripts\python.exe`).

## E1. Video VAE ("Nivra tokenizer")

```
python -m nivra.cli train-vae --config nivra_small --run runs/small/vae --steps 2000 --save-every-steps 250
```

| item | value |
|---|---|
| config | `ai/configs/nivra_small.json` (vae: latent 4 ch, c1 32, c2 48, 1 res block; train_vae: batch 8, lr 1e-3, warmup 50, 4000 train clips) |
| parameters | 471,211 |
| latent | 3x16x64x64 -> 4x8x16x16 (spatial /4, temporal /2) |
| steps | 2000 (the run was interrupted once by a network error and resumed from a checkpoint) |
| compute time | about 2,700 s summed over logged steps; median 1.00 s/step (0.77 - 7.3 s/step under load) |
| peak RSS | 998 MB |
| final train loss (10-step mean, step 2000) | total 0.0435, L1 0.0246, MSE 0.0040, motion L1 0.0146, KL 3.54 |
| held-out reconstruction (64 clips, never trained on) | **PSNR 29.48 dB**, MSE 73.3 (0-255 scale), motion L1 2.02 (0-255 scale) |
| encode / decode | 0.012 s / 0.017 s per clip |
| tiled decode (tile 4, 8 clips) | PSNR 23.18 dB (approximate by construction; off by default) |
| scaling factor | 0.9965 |
| VAE sha256 | `e5cd7f7c3b63736896e3ca6475d0979008c2cb068118d9833952f8dd54debe27` |

Source: `runs/small/vae/vae_metrics.json`, `runs/small/vae/log.jsonl`.

## E2. Generator (flow matching, NivraSmall)

```
python -m nivra.cli train --config nivra_small --vae runs/small/vae/export/vae.safetensors ^
      --run runs/small/gen --steps 6000 --save-every-steps 250 --protect 600
```

| item | value |
|---|---|
| config | model: patch 2, dim 128, depth 4, heads 4, mlp ratio 3, 2 text layers; train_gen: batch 8, lr 5e-4, warmup 100, cosine, AdamW wd 0.01, clip 1.0, EMA 0.999, caption dropout 0.1, 8000 training clips |
| parameters | 2,414,352 |
| steps | 6000 (about 6.0 epochs of 8000 clips at batch 8) |
| interruption | the first process died with a network error at about step 2940; training resumed from the step-2750 checkpoint (steps 2751-2940 were recomputed; exact-resume equivalence is a unit test). `steps_run` of the final process: 3250 |
| compute time | about 7,200 s summed over logged 10-step windows; median 1.03 s/step, range 0.50 - 10.4 s/step (contention with other jobs) |
| peak RSS | 883 MB (final process) |
| train loss (10-step mean) | step 10: 2.02, 500: 0.475, 1000: 0.393, 2000: 0.307, 3000: 0.273, 4000: 0.226, 5000: 0.218, 6000: 0.200 |
| held-out flow loss (128 clips, fixed noise, English shape captions) | raw 0.2972, EMA 0.2993 |
| exported model | EMA weights, `runs/small/gen/export/` -> `runs/release/nivra-small-6000/` |

Source: `runs/small/gen/log.jsonl`, `runs/small/gen/final_metrics.json`, `runs/gen_train.log`.

The EMA (decay 0.999) is not better than the raw weights on held-out loss at the last step (0.2993 vs 0.2972); it is
what is exported anyway because sample quality, not loss, was the target and the difference is small.

## E3. Release bundle

`runs/release/nivra-small-6000/` (generator version `v000000-58cfea971826`, 2,414,352 parameters):

| file | sha256 |
|---|---|
| `weights.safetensors` | `58cfea9718261a864127acb46ce1d8fe36446f29e012ff1d2a192b687ad4d28f` |
| `vae.safetensors` | `e5cd7f7c3b63736896e3ca6475d0979008c2cb068118d9833952f8dd54debe27` |
| `model_version.json` | `24fa00de1d34111b32b15df341ee2ed5b8404f3e00d43dc2daa6c7d9ee64b30a` |

The manifest was re-written once after export (same weights, same version id) to add the `tensor_l2` / `weights_l2`
fields introduced by the update-validation hardening; the weights and VAE bytes did not change. The same weights were
also produced independently through `ai.init_model` `from_checkpoint` (EMA) in the exec smoke run (identical weights sha256).

## E4. Detector validation on ground truth (needed to interpret E5)

```
python -m nivra.cli evaluate --mode detector --clips 200 --out runs/eval/detector_gt.json
```

The prompt-adherence scorer (colour / shape / direction / count detection by pixel analysis) was run on 200
ground-truth synthetic clips (seed 99, indices from 5,000,000) against their true scene specs: all-correct rate
1.000, entity recall 1.000, counts correct 1.000, temporal consistent 1.000, zero missing / wrong colour / wrong shape /
wrong direction / invented entities (388 entities). So on this synthetic domain the detector makes no errors; any
score below 1.0 for a generated video is attributable to the video. This says nothing about the detector on other
kinds of video.

## E5. Prompt adherence, trained vs untrained

```
python -m nivra.cli init-random --config nivra_small --vae runs/small/vae/export/vae.safetensors --out runs/eval/bundle_untrained --seed 0
python -m nivra.cli evaluate --mode adherence --bundle runs/eval/bundle_untrained --out runs/eval/bench_untrained.json --seeds 1 --steps 20 --cfg 3.0
python -m nivra.cli evaluate --mode adherence --bundle runs/release/nivra-small-6000 --out runs/eval/bench_trained.json --seeds 1 --steps 20 --cfg 3.0
```

24 benchmark prompts (`ai/configs/benchmark_prompts.json`: 12 English, 12 Swedish; categories single, colour,
direction, count, multi, stationary, generic noun), 1 seed each, 20 Euler steps, CFG 3.0. Untrained = same architecture,
random initialisation (seed 0), same VAE. Untrained took 191 s, trained 34 s for the 24 videos (different load).

| metric (24 videos) | untrained | trained (6000 steps) |
|---|---|---|
| all-correct rate (every entity right: colour, shape, motion, count, nothing invented) | 0.000 | 0.000 |
| entity recall | 0.000 | 0.194 |
| count correct | 0.000 | 0.208 |
| temporally consistent | 0.000 | 0.375 |
| missing entities / video | 1.292 | 0.167 |
| wrong colour / video | 0.167 | 0.125 |
| wrong shape / video | 0.000 | 0.667 |
| wrong direction / video | 0.042 | 0.708 |
| invented entities / video | 0.000 | 0.833 |

Reading it plainly: the untrained model produces essentially empty frames (missing entities 1.29 per video, nothing
invented), the trained model produces objects (missing drops to 0.17) but mostly the wrong shape or direction and often
extra objects. **No video was fully correct; the trained model does not follow prompts reliably.**
Entity recall by category (trained): single 0.50, stationary 0.50, generic noun 0.33, direction 0.25, count 0.11,
multi 0.10, colour 0.00. By language: English 0.32, Swedish 0.06 (so Swedish adherence is clearly worse).
With 24 videos and one seed, differences of a few percentage points are not meaningful.

Sampler sensitivity (trained, 24 prompts, 50 steps): CFG 1.0 -> all-correct 0.083, entity recall 0.167; CFG 6.0 ->
all-correct 0.000, entity recall 0.139 (`runs/eval/bench_trained_steps50_cfg*.json`). Neither is better than
20 steps / CFG 3.0 (recall 0.194) beyond noise; no tuning was done.

Videos: all 24 trained benchmark videos are in `runs/eval/trained_videos/` (with scores in the sidecar JSON).

## E6. Samples

`runs/samples/` (30 steps, CFG 3.0, seed 7, 4x upscale, 8 fps), frame counts checked with
`ffprobe -count_frames`:

* 10 text-to-video MP4s, 16 frames each, 256x256: `en_red_circle_l2r`, `sv_red_circle_l2r`, `en_green_square_up`,
  `sv_yellow_triangle_down`, `sv_blue_circle_r2l`, `en_two_blue_squares`, `sv_two_yellow_triangles`,
  `en_still_green_circle`, `en_multi`, `sv_multi` (5 English, 5 Swedish).
* 1 continuation: `en_red_circle_l2r_continued.mp4` = 28 frames (16 + 12 new, `--known 2`).

Sampling took 1.3 - 2.6 s per video, decode 0.03 - 0.05 s, peak RSS 242 MB. Whether a given sample shows the requested
scene was not hand-labelled; see E5 for the automatic scores.

## E7. DiLoCo simulation (2 real worker processes)

```
python ai/scripts/exec_smoke.py --ckpt runs/small/gen/ckpt_00006000 --vae runs/small/vae/export/vae.safetensors --out runs/smoke
python -m nivra.distributed.simulate --base runs/smoke/init_model/out --out runs/diloco_lr3e-4 --workers 2 --rounds 3 --inner-steps 40 --threads 2 --lr 3e-4
```

Setup: base version created by the real executor job `ai.init_model` with `from_checkpoint` = step-6000 checkpoint
(EMA weights). Each round: 2 separate `python -m nivra.cli exec` `ai.local_train` processes (2 threads each), disjoint
shards (worker seeds 1000 and 1001, 336 clips per round, new range every round, 16 held back for the worker's own
before/after eval), same base version, batch 8, fresh AdamW per call, warmup then constant lr; then one `ai.aggregate`
process (mean of 2 updates, outer SGD lr 0.7, Nesterov momentum 0.9, outer state carried across rounds). Held-out loss =
fixed-noise flow loss on 64 clips from a seed/range never used for training (seed 424242, index 3,000,000).

| run | base | inner lr | inner steps | held-out loss: round 0 (base) / 1 / 2 / 3 |
|---|---|---|---|---|
| A | step 6000 | 3e-4 | 40 | 0.2793 / 0.3024 / 0.3015 / 0.3013 |
| B | step 6000 | 1e-4 | 40 | 0.2793 / 0.2857 / 0.2845 / 0.2845 |
| C | step 6000 | 3e-5 | 40 | 0.2793 / 0.2809 / 0.2802 / 0.2806 |
| D | step 600 (raw, non-EMA weights) | 3e-4 | 60 | 0.5104 / 0.4841 / 0.4849 / 0.4619 |

Findings, stated as measured:
* From the already converged step-6000 model, DiLoCo rounds did **not** improve held-out loss at any tested lr; they made it
  slightly worse (+0.002 to +0.023), with the damage growing with the inner lr. The base was trained with a decayed lr
  (cosine to near zero) while the workers restart AdamW at a constant lr on 40 steps, so noise is expected to dominate.
  Per-worker loss before/after the inner loop on their own held-out clips was also flat or slightly worse.
* From the early step-600 model, 3 rounds lowered held-out loss from 0.5104 to 0.4619 (-9.5 %), so the pipeline does
  transmit useful learning when there is something left to learn. Round 2 was flat (0.4841 -> 0.4849), so progress is
  noisy.
* Total inner steps in D: 3 rounds x 2 workers x 60 = 360 worker steps, wall time about 110 - 130 s per round of workers
  under CPU contention.
* No update was rejected in any run (`rejected: []`), consistent with the validation bounds in
  `docs/training/DISTRIBUTED.md`. These are single runs, no seeds repeated: the 0.002-level differences are within noise.
* Starting from scratch (random init) was not run as a full simulation. A single from-scratch 30-step update was checked
  against the validator: it initially failed the strict per-tensor bound (zero-initialised `ada.*` / `final.*` tensors,
  17 tensors with base norm exactly 0) and passes after adding the per-element slack documented in DISTRIBUTED.md.

## E8. Executor smoke run (every job type through `python -m nivra.cli exec --job job.json`)

`python ai/scripts/exec_smoke.py ...` (report: `runs/smoke/smoke_report.json`). All 10 jobs exited 0 and announced their
artifacts by kind:

| job | kind(s) announced | wall s |
|---|---|---|
| `ai.init_model` (from_checkpoint) | model_version, model_weights, vae_weights | 3.8 |
| `ai.dataset_shard` (16 clips) | dataset_shard, dataset_manifest | 3.6 |
| `ai.local_train` x2 (4 steps, disjoint shards) | update, update_weights | 10.1 / 11.1 |
| `ai.validate_update` | (none; summary valid=true, norm 0.903) | 3.8 |
| `ai.aggregate` (+ heldout eval) | model_version, model_weights, outer_state | 8.4 |
| `ai.generate` | video, video_metadata | 5.0 |
| `ai.evaluate` detector / reconstruction / adherence | evaluation_report | 4.3 / 3.8 / 5.0 |

## E9. Tests

`python -m pytest ai/tests -q` -> `72 passed in 112.75s (0:01:52)`

Covers determinism, EN/SV captions, tokenizer round trip, shapes, forward/backward, parameters change after a step,
checkpoint save/resume equivalence (bit-identical, also across an epoch boundary), corrupted checkpoint refused,
DiLoCo aggregation (mean, Nesterov, trimmed mean, clipping), stale / duplicate / NaN / oversized / zero / outlier updates
rejected, header-before-load and manifest size limits, exec protocol (stop file -> exit 3, bad protocol -> exit 2,
missing input -> exit 5), pinned platform-independent digests of the dataset generator and shard files, MP4 frame
counts via ffprobe. torch 2.6 / aarch64 (Raspberry Pi) was **not** available here, so controller-side compatibility
is by API choice only (no use of post-2.6 torch APIs in these paths), not by execution.

## Failure modes and caveats observed

* A network error interrupted both the VAE run and the generator run; both resumed from checkpoints.
* `nivra.distributed.simulate` originally passed workdir-relative paths to jobs, which resolve against each job's own
  directory; fixed by resolving `--base` / `--out` to absolute paths.
* The first per-tensor update bound (`4 x ||base_t|| + 1e-3`) rejected a legitimate update from a random init (see E7).
* The generator over-produces objects of the wrong shape and direction; prompt adherence is weak (E5).
* GPU / FP16 paths remain unexecuted (no CUDA device).
* Timings are from a contended machine and are not best-case numbers.
