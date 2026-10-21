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
   `text_encoder\`). `Nivra.exe` scans `<install>\models`, so the model appears in its model list (Rescan if it is already open). Without
   `--install-dir` the bundle is `RUN\bundle`; copy it into any folder the app scans, or point `--bundle-out` at one.
6. **Or train on the network instead.** Use the finished `RUN\latents` (already encoded, `nivra-latents/1` shards) and the bundle as the
   starting version: `nivra model import RUN\bundle`, `nivra dataset push RUN\latents --kind latents`, then create an HD training run on
   the controller (`docs/architecture/HD_TRAINING.md`). Workers then train rounds of the same DiT on those latents.

Troubleshooting: "no train clips" after ingest means the videos were shorter than 2 s, lower resolution than 448x256 or lacked a caption
(the counts are in `RUN\dataset\dataset.json`, key `rejected`). A `StreamStateTooLarge` error while decoding means the VAE tile size is too small for the output size:
use a larger `vae_tile` (the cache state grows with the number of tiles).
