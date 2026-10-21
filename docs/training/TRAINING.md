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
