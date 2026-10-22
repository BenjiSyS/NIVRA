# Training Nivra Small

All commands run from the repository root with the project venv and `PYTHONPATH=ai`
(or `pip install -e ai`):

```
set PYTHONPATH=ai
set NIVRA_THREADS=4                 # torch intra-op threads (default 4)
.venv\Scripts\python.exe -m nivra.cli <command> ...
```

Only `torch`, `numpy`, `safetensors` (and `pytest` for tests) are needed. No pretrained weights are
downloaded or used anywhere; every weight is initialised randomly and trained by these commands.

## Pipeline

1. **Data** - `nivra.data.synthetic`: deterministic procedural clips (1-3 shapes, 6 colours, 5 motions,
   dark background) with exact ground-truth scene specs and EN/SV captions. Fully determined by
   `(generator_version="nivra-synth/1", seed, index, SynthConfig)`. A dataset is identified by a
   fingerprint (sha256 of generator version + config + seed + index range). Training indices
   `[0, N)` use seed 1234; held-out indices start at 1,000,000 (never trained on).
2. **VAE ("Nivra tokenizer")** - `train-vae`: 3x16x64x64 -> 4x8x16x16 (spatial x4, temporal x2),
   loss = L1 + MSE + temporal-difference L1 (motion) + small KL. After training the scaling factor
   (1/std of latents over 128 clips) is stored in the VAE file; PSNR / motion error / decode time
   are written to `<run>/vae_metrics.json`.
3. **Generator** - `train`: encodes the training set once with the frozen VAE (cached under
   `runs/<x>/latent_cache`), then trains NivraSmall (flow matching, logit-normal t, v-target =
   noise - x0, 10% empty-caption dropout for CFG, AdamW, linear warmup + cosine, grad clip 1.0,
   EMA 0.999). Captions are drawn per sample from {EN, SV} x {shape noun, generic object noun}
   using the trainer's own RNG, so resume is exact.

```
python -m nivra.cli train-vae --config nivra_small --run runs/small/vae --steps 2000 --save-every-steps 250
python -m nivra.cli train     --config nivra_small --vae runs/small/vae/export/vae.safetensors \
                              --run runs/small/gen --steps 5000 --save-every-steps 250 --protect 600
```
Output of a finished generator run: `<run>/export/` = a *model version* (`weights.safetensors` =
EMA weights, `model_version.json`, `vae.safetensors`, `tokenizer.json`), loadable by
`generate`, `continue`, `evaluate` and the executor.

## Configs
`ai/configs/*.json` (JSON only; PyYAML is not a dependency). `nivra_small.json` is the real
configuration; `nivra_tiny.json` (32x32, 8 frames, ~50k parameters) is used by tests. Sections:
`data`, `vae`, `model`, `train_vae`, `train_gen`. Unknown top-level keys are rejected.

## Checkpoints (no pickle)
Directory `ckpt_<step:08d>/`:

| file | content |
|---|---|
| `model.safetensors`, `ema.safetensors` | weights / EMA weights |
| `optimizer.safetensors` | flattened AdamW state (`state.<param_idx>.<exp_avg/exp_avg_sq/step>`) |
| `rng.safetensors` | torch CPU RNG state (+ CUDA states if present) and the trainer's private generator state |
| `state.json` | python + numpy RNG state, optimizer param-group hyper-parameters, LR-scheduler state, GradScaler state, data-sampler position (epoch, offset, seed), global step |
| `manifest.json` | step, epoch, config, compat block (architecture version, config hash, tokenizer version + vocab hash, dataset fingerprint, VAE sha256), sha256 + size of every file, `manifest_digest` |

* Writes go to `.tmp_ckpt_*` and are published with an atomic `os.replace`.
* Load verifies the manifest digest and every file hash (flip one byte -> `CheckpointError`), then
  refuses an incompatible architecture / tokenizer / dataset fingerprint (`IncompatibleCheckpoint`).
* Retention (`keep_last`, default 3): never deletes the newest *valid* checkpoint, any checkpoint
  containing a `PROTECTED` marker (`--protect STEP,...`), or the `keep_last` newest.
* Save triggers: every `--save-every-steps` steps, every 10 minutes, a `STOP` file in the run
  directory (saves, exits with status "stopped"), SIGINT/SIGTERM/SIGBREAK (graceful save).
* `Trainer.resume()` restores everything above; `ai/tests/test_checkpoint.py` trains k steps, saves,
  resumes in a fresh trainer with a different initialisation and asserts parameters, EMA, Adam
  moments/step counts, scheduler, RNG state and the next batch are bit-identical to an
  uninterrupted run (also across an epoch boundary).

## Mixed precision / GPU (UNTESTED on GPU)
`NIVRA_DEVICE=cuda` enables `torch.autocast(device_type="cuda", dtype=torch.float16)` plus
`torch.amp.GradScaler`; bf16 is never used; attention goes through
`F.scaled_dot_product_attention` (math fallback exists, FlashAttention not required). **These paths
have never been executed** - this machine has no CUDA GPU. Treat them as unverified code.

## Known limitations
* Everything here was measured on a 4-core laptop CPU; see `docs/research/EXPERIMENTS.md`.
* VAE tiled decoding is approximate (GroupNorm statistics are per tile).
* The trainer skips a step if the clipped gradient norm is non-finite (counted in `skipped`).

## Results of the reference run (6000 steps)
Measured numbers, commands and failure modes are in `docs/research/EXPERIMENTS.md`: VAE 471,211 params, held-out PSNR
29.48 dB; generator 2,414,352 params, held-out flow loss 0.297 (raw) / 0.299 (EMA), about 1.0 s/step median on the
4-core CPU (contended). Prompt adherence of the result is weak (entity recall 0.19 vs 0.00 for an untrained model on 24
prompts; no video fully correct).

## Model bundle / release
`runs/small/gen/export/` (written at the end of `train`) is copied to `runs/release/nivra-small-<step>/`
(`weights.safetensors`, `model_version.json`, `vae.safetensors`) - the exact input of the `ai.generate` / `ai.evaluate`
jobs. Hashes of the 6000-step bundle: `EXPERIMENTS.md` E3. Inference: `docs/inference/INFERENCE.md`.

## Helper scripts
* `ai/scripts/exec_smoke.py` - runs every executor job type end-to-end through `python -m nivra.cli exec`.
* `ai/scripts/init_from_ckpt.py` - `ai.init_model` `from_checkpoint` from a generator checkpoint (DiLoCo base).
* `python -m nivra.distributed.simulate --base <version dir> --out <dir> --workers 2 --rounds 3` - real multi-process DiLoCo.

## Distributed-training safety defaults
Update validation always enforces absolute norm bounds, zero-update rejection, JSON size limits and header-first
safetensors checks, with no parameters required (`docs/training/DISTRIBUTED.md` section 0 and 1.4). Consequence for
local experiments: a pseudo-gradient larger than 0.5 x ||base weights|| is rejected (`norm_outlier`).

## Train on your own videos

Goal: a folder of videos you are allowed to use becomes a model that shows up in `Nivra.exe` by itself. Everything runs on
one machine (a 6 GB GPU is enough for `nivra_base`; a CPU works for tiny tests only). Details and numbers:
`docs/architecture/HD_TRAINING.md`.

1. **Put the videos in one folder**, for example `D:\anime-clips`. Add a `PROVENANCE.json` there (source, license, license_url,
   permission, collected, contains_people, consent): ingestion refuses a folder without it, so you always know where the data came from.
2. **Caption them** (needs `ai/requirements-caption.txt`, see `docs/training/CAPTIONING.md`). Captioning is slow on a laptop, so do it in
   batches; each run only touches videos that still lack a caption:

       nivra caption D:\anime-clips --limit 100
       nivra caption D:\anime-clips --limit 100      (again, until it says nothing is left)

   This writes `<video>.txt` next to every video. Read a few and fix wrong ones by hand: the model learns exactly these sentences.
3. **Look at the plan**, then train:

       python -m nivra.cli train-pipeline --config nivra_base --videos D:\anime-clips --out D:\runs\anime-v1 ^
           --name anime-v1 --install-dir "C:\Users\you\Nivra" --plan-only
       python -m nivra.cli train-pipeline --config nivra_base --videos D:\anime-clips --out D:\runs\anime-v1 ^
           --name anime-v1 --install-dir "C:\Users\you\Nivra"

   It prints the stages with rough time estimates, then runs: ingest (cut into 2 s clips, drop duplicates and junk), tokenizer, text
   encoder, video VAE, latents, the diffusion transformer, and the bundle. Defaults are 1500 / 4000 / 8000 steps for text / VAE / DiT;
   raise `--vae-steps` and `--dit-steps` for real quality (days on one 6 GB card, and tens of hours of clips are needed for good
   results; a few hundred clips only teaches the look and a few motions).
4. **Stop and resume freely.** Press Ctrl+C any time and run the same command again: finished stages are skipped (a marker file
   in `RUN\markers` records what each stage was built from) and the VAE/DiT stages continue from their last checkpoint. Changing an
   option (for example `--dit-steps`) re-runs only that stage and the ones after it. `--from-stage` / `--to-stage` run part of the
   chain (for instance `--to-stage latents` overnight, DiT the next day); `--force` re-runs the selected stages.
5. **Use it.** The bundle is written to `<install>\models\anime-v1` (`model_version.json`, `weights.safetensors`, `vae.safetensors`,
   `text_encoder\`). `Nivra.exe` scans `<install>\models`, so the model appears in its model list (Rescan if it is already open; known issue: the app currently ignores the relative `.\models` folder, so copy the bundle to `<data_dir>\models` or load it by its full path). Without
   `--install-dir` the bundle is `RUN\bundle`; copy it into any folder the app scans, or point `--bundle-out` at one.
6. **Or train on the network instead.** Use the finished `RUN\latents` (already encoded, `nivra-latents/1` shards) and the bundle as the
   starting version. Measured end to end on a laptop with the tiny config:

       nivra model import --project main --bundle RUN\bundle          (prints the model version id; run it while the controller is stopped)
       nivra dataset push RUN\latents --kind latents --vae-version <that version id> --url http://CONTROLLER:8740 --project main

   `--url` is required and the token comes from `NIVRA_TOKEN`. `--vae-version` is required for latents because `dataset.json` cannot know
   the id the controller gave the bundle. Mark the workers trusted (Studio, Workers page), then create an HD training run
   (Studio, Training, or `POST /training/runs` with `family: hd`; `docs/architecture/HD_TRAINING.md`). Notes:
   * A dataset needs a validation shard for `eval_gate: pi`. With few videos the ingest may put none aside (the split is by video, 5 %
     by default): `train-pipeline` prints a warning when that happens; pass `--val-fraction 0.2`, add videos, or use `eval_gate: off`.
   * `train` and `eval` jobs need a GPU worker by default. For a tiny config on CPU-only machines set `allow_cpu: true`, `req_vram_mb: 0`
     and a small `req_ram_mb` in the run config.
   * To encode raw shards on the network instead (`nivra dataset push RUN\dataset`, then `POST /datasets/{id}/encode`), upload the
     bundle's `text_encoder\model.safetensors` (artifact kind `text_encoder`) and `tokenizer.json` (kind `tokenizer`) and pass their ids.
     Encode jobs ask for 8192 MB of RAM by default; pass `req_ram_mb` to lower it for small configs.
   * **Generating on other computers needs no encoder setup.** `nivra model import` packs the bundle's `text_encoder/` directory
     (the Nivra-trained encoder) into a deterministic tar artifact (kind `text_encoder_archive`) and the controller hands it to every
     worker that generates, drafts or refines with that model; the worker verifies it against the manifest's `text_encoder.sha256`,
     extracts it safely into its cache and uses it. Only an `external` third-party encoder (licensed weights, never uploaded) still
     has to be placed in `NIVRA_TEXT_ENCODERS_DIR` on each worker, see `docs/training/TEXT_ENCODERS.md`.

Troubleshooting: "no train clips" after ingest means the videos were shorter than 2 s, lower resolution than 448x256 or lacked a caption
(the counts are in `RUN\dataset\dataset.json`, key `rejected`). A `StreamStateTooLarge` error while decoding means the VAE tile size is too small for the output size:
use a larger `vae_tile` (the cache state grows with the number of tiles).

## 2026 improvement pass: what is new in the HD trainer (all config-gated)

Source: `docs/research/IMPROVEMENT_RESEARCH_2026.md` (changes 1-7, 9, 10, 12 and the EMA note). Every option lives in the config's `train`
(or `model` / `vae`) section, is validated by `hd_config.train_options`, and **a config without these keys trains exactly as before**
(loss histories of the old trainer are reproduced to 1e-4 in `ai/tests/test_improve_training.py`; an old bundle loads, generates and
keeps its `hd_config_sha`). Checkpoints written by the old trainer resume with the new one (new groups are created on resume).

| Option (section `train` unless noted) | Meaning | tiny | base | medium / large |
|---|---|---|---|---|
| `vae.temporal_factor` (`vae`) | 4 or 8 (alias for `temporal_down` `[2,3]` / `[1,2,3]`; 32x spatial layout) | 8 | **4** | 8 |
| `vae_stages` | VAE schedule: image AE (T=1) -> low-res clips -> target fine-tune; per stage `image_fraction` mixes T=1 batches into video stages | on | on | on |
| `vae_gan_weight`, `vae_gan_start`, `vae_gan_lr`, `vae_gan_base`, `vae_gan_layers` | hinge patch-GAN (3D, T=1 for images) with the VQGAN adaptive weight, switched on at step `vae_gan_start` | 0.05 | 0.05 | 0.05 |
| `vae.edge_weight` (`vae`) | edge/gradient loss; absent in a stylized config means 0.3, general means 0 | 0.5 | 0.5 | 0.5 |
| `vae_latent_norm` | `channel`: per-channel `[C]` latent shift/scale from encode statistics (needs `nivra-hc-vae/2`); `scalar`: the old pair | channel | channel | channel |
| stage keys `image_fraction`, `buckets` | image-first DiT stage = a stage with `frames: 1`; joint stage = `image_fraction` p of micro-batches are T=1 stills; `buckets` = weighted list of {height,width,frames} | on | on | on |
| `ema_decay`, `ema_warmup` | EMA of the DiT; **the bundle exports the EMA weights when `ema_decay > 0`**; decay ramps (1+n)/(10+n) up to `ema_decay` | 0.99 | 0.9995 | 0.9999 |
| `sra` `{weight, student_layer, teacher_layer, t_ratio, until_step, hidden}` | self-representation alignment: `1 - cos(proj(block[student] at noise t), stop-grad block[teacher] of the EMA model at noise t*t_ratio)`; projector is training-only (checkpointed, never exported) | 0.5 (0->1) | **0.5 (3->8 of 12)** | 0.5 (5->15 of 24) |
| `repa` | REPA with an external teacher (DINOv2, licensed opt-in). **Not implemented**: `enabled: true` is rejected; `RepaHook` in `hd_trainer.py` documents the interface | off | off | off |
| `diffusion_forcing` `{fraction, cond_fraction}` | fraction of DiT micro-batches with an independent timestep per latent frame; `cond_fraction`: batches with clean first / last latent frame(s) as conditioning (needs `model.frame_cond`) | 0.25 / 0 | **0.25 / 0.15** | 0.25 / 0 |
| `model.frame_cond`, `model.fps_cond` (`model`) | v2 DiT: conditioning-frame flag embedding + effective-fps embedding (architecture change: `nivra-dit/2`) | off | **on** | off |
| `fps_dropout` | probability the clip's effective fps is replaced by the learned "unknown" embedding | 0.1 | 0.1 | 0.1 |
| `fp32_attn` | attention logits / softmax in float32 for half inputs | on | on | on |
| `loss_scale` `{init, growth_interval, growth, backoff}` | dynamic loss scaling (CUDA fp16, or any device with `force_loss_scaling: true` for tests) | - | defaults | defaults |

**Order of work.** VAE: `vae_stages` (image autoencoder at T=1, then causal-3D low-resolution short clips with 20 % image batches, then the target
resolution fine-tune), GAN from `vae_gan_start`, then calibration on training clips (per channel when `vae_latent_norm: channel`). The latent
calibration travels with the VAE (`hc_vae.json`, safetensors metadata, bundle manifest `config.vae.latent_shift/latent_scale/latent_norm`); every
consumer calls `vae.to_scaled` / `from_scaled`, which handle both forms. DiT: image-first stage (stills are encoded by the same causal VAE; the
`latents` stage writes extra T=1 shards `lat-still-*` with random frames of every clip, listed under `still_shards` in the latents `dataset.json`), then
joint image + video stages with buckets; the timestep shift is `clamp(sqrt(tokens / shift_base_tokens), 1, 6)` with tokens = T x H x W of the latent;
training (per micro-batch) and inference apply the same `resolution_shift(tokens, base)`, so image batches (few tokens) get shift 1 and a long 256p clip the full shift.

**fp16 on Turing (RTX 20xx has no bf16).** The timestep / fps embedders, the adaLN modulation projection, every LayerNorm / RMSNorm, the modulation
arithmetic and the gate multiply run in float32 under autocast and in a half model (identical to the old code in float32); `train.fp32_attn` /
`inference.attn_fp32` also run the attention logits in float32. The GradScaler is dynamic: an overflow step is skipped, logged
(`fp16 overflow ... optimizer step skipped, loss scale a -> b`), counted in `skipped_steps` and `history[].skipped`, EMA is not advanced, and the scale
never drops below 1; 200 skipped steps in a row raise instead of looping. None of this has run on a GPU yet (this machine has no CUDA device); on CPU it is
exercised with `train.force_loss_scaling` and forced overflows.

**Anime holds.** Ingest (`--profile stylized`, on by default there; `IngestOptions.detect_holds`) decodes each clip window at the SOURCE frame rate
as small gray frames, finds the frame pairs that really change (threshold = max(0.15, 8 % of the clip's 90th-percentile difference)), and stores in
every manifest row `hold` (most common drawing duration in frames: 1 = on ones, 2 = on twos...), `effective_fps` (source fps / mean duration; 2,3,2,3 at
24 fps gives 9.6), `hold_purity` and `hold_fraction`. `model.fps_cond` makes the DiT condition on `effective_fps`; at inference it defaults to the
output fps (`generate(effective_fps=...)` overrides). `IngestOptions.drawing_rate_sampling=True` instead re-reads such a clip at its drawing rate
(every frame a new drawing, the clip spans more source time); it is a Python option only (no CLI flag yet). OCR subtitle / credit filtering is
not implemented: it needs an OCR dependency (PaddleOCR) and this pass installs nothing.

**Frame conditioning for long videos.** For `frame_cond` models `sample_flow` holds the known frames (continuation overlap, refine anchors) CLEAN with
t = 0 and the flag embedding instead of re-noising them along the flow path; older models keep the replacement scheme. `run_draft` / `run_refine`
results report `conditioning: "mask" | "replace"`.

**What a smoke test shows.** `train-pipeline --config nivra_hd_tiny` (8 VAE steps scaled to 4, 6 DiT steps) runs every new path on CPU in about 12 s
and produces a bundle that loads and generates (`ai/tests/test_improve_pipeline.py`). That proves plumbing and exact-resume behaviour, not quality.

**Measured CPU cost of the new options** (this laptop, 4 threads, CPU seconds per optimizer step from `time.process_time`, other jobs were running so
read it as +-10 %; DiT: dim 512, depth 8, 64-channel latent 5x8x14 = 560 tokens, batch 2, grad checkpointing; VAE: channels 16..64, 64 latent channels,
17x128x224, batch 1). DiT baseline 12.8 s: + EMA 1.08x, + SRA with EMA teacher 1.30x, + diffusion forcing 25 % ~1.0x, the whole `nivra_base` recipe (EMA, SRA,
DF + conditioning frames, fp32 attention logits, v2 embeddings) 1.27x. VAE baseline 25.4 s: + patch-GAN (discriminator forward / backward, two extra
gradient passes for the adaptive weight) 1.47x while it is on. fp16 numbers on a real GPU are not measured (no CUDA device here).
