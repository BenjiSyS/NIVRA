# HD distributed training on real data (DiLoCo v2) — contract

Contract between the Python package (`ai/**`) and the controller (`internal/**`, `cmd/**`, `database/**`).
Throughput numbers are *estimates* until measured on real GPUs.

## Critical decisions

1. **The DiT trains on precomputed latents**; raw video never travels per round. The owner ingests locally
   (`data-ingest`), pushes shards to the controller as content-addressed artifacts registered in a **dataset
   registry**, and one **encode pass** (`ai.hd_encode`) turns raw shards into a **latents dataset** bound to a
   frozen VAE version + frozen text encoder (text embeddings and the null embedding for CFG dropout are
   precomputed too). Workers never load the VAE or text encoder for DiT training.
2. **Phases**: tokenizer + NivraText MLM pretraining = single machine. HC-VAE = single machine by default
   (optionally distributed on trusted ≥12 GB workers over raw shards). **DiT = the distributed workload.** VAE and
   text encoder are frozen before latent precompute; changing either needs a new latents dataset.
3. **Worker classes**: CPU (16 GB+ RAM) → no training (too slow); `ai.hd_encode` at CPU speed, captioning, eval.
   6 GB Turing (RTX 2060) → `nivra_base` DiT (~200 M) at 128p/256p (fp16 autocast + GradScaler, grad
   checkpointing, AdamW8bit, batch 2–4 + accumulation). 12–16 GB → base DiT larger batch, or VAE (trusted). 24 GB+
   → Medium (932 M, hard minimum 16 GB). Multi-GPU servers: one job slot per GPU (`internal/worker/agent/slots.go`);
   no in-job DDP in v1.
4. **Update format**: fp16 safetensors deltas, computed in fp32 against the exact weights received. The controller
   keeps an **fp32 master** per version plus a derived **fp16 train copy** that workers download.
5. **Aggregation on the Pi is a two-pass stream, tensor by tensor** (`safe_open`), never loading whole models
   (the current `ai/nivra/distributed/aggregate.py` uses `load_file` and cannot be reused for HD). Peak RAM ≈
   largest tensor × (N+3) × 4 B.
6. **Trusted-only by default** for train, encode and eval jobs. Datasets carry `redistribution:
   owner_devices|any` (default `owner_devices`, forces trusted).
7. **Held-out gate**: `eval_gate = pi | worker | off`; `pi` for models ≤ 300 M (8 fixed (latent, t, noise)
   triples, ~3–4 min on Pi 5), `worker` (trusted `ai.hd_eval`) for Medium.

## Safety constraints

* Each update manifest echoes `run_id, round, round_batch, base_version_id, update_id, attempt_id, target,
  hd_config_sha, stage, inner_steps, lr, shards[{artifact_id, sha256}], samples_seen, delta_dtype`; any mismatch
  with the job params → reject.
* Upload size must be exactly `Σ numel × 2 + header` (budget set to that).
* Duplicate `update_weights` sha256 within a run+round → reject (DB unique index).
* Structural: keys/shapes/dtypes = base manifest; no NaN/Inf; per tensor ‖Δ‖∞ ≤ c·H·lr and ‖Δ‖₂ ≤ c·H·lr·√numel
  (c = `bound_c`, default 3; H = inner steps); `max_rel_norm` kept (configurable); median outlier test + median
  clipping; zero update rejected.
* Encode jobs: a fraction (default 5%) re-encoded on a different trusted device, latent MSE < 1e-3·var. Latent
  shard manifests bind `parent_shard_sha256, vae_version, text_encoder_id, stage`.
* Untrusted mode (opt-in, `redistribution=any` only): `redundancy_frac` re-runs identical work on two devices and
  requires cosine(Δa, Δb) ≥ τ (calibrate τ on trusted GPUs). Bounded-norm backdoors are **not** prevented:
  untrusted HD training is documented as unsafe.
* Version promotion requires the eval gate (`MaxHeldoutIncrease`); a failed gate retries the round.

## Contract

**Artifact kinds**: `dataset_shard`, `dataset_manifest`, `latent_shard`, `model_weights` (fp32 master),
`model_train_weights` (fp16), `model_manifest`, `outer_state` (fp32 momentum), `update_weights` (fp16 delta),
`update` (manifest JSON), `eval_report`, `eval_sample`.

**Dataset registration** (API + CLI, not a job): `nivra dataset push DIR [--kind raw|latents]
[--redistribution owner_devices]` uploads each shard (resumable 8 MB chunks) then
`POST /api/v1/projects/{p}/datasets {name, kind, fingerprint, parent_dataset_id?, encoder:{vae_version_id,
text_encoder_id, stage}, shards:[{idx, split, artifact_id, sha256, clips, bytes}]}`; the controller verifies each
artifact's sha256 and project. `POST .../datasets/{id}/encode {vae_version_id, text_encoder_artifact,
tokenizer_artifact, stage}` creates one `ai.hd_encode` job per raw shard (idempotency key
`enc:{dataset}:{vae}:{idx}`) and advances idempotently into a new latents dataset (same pattern as `generations`).

**`ai.hd_encode`** — inputs `raw_shard, vae_weights, vae_manifest, text_encoder_weights, tokenizer`; params
`{dataset_id, target_dataset_id, shard_idx, shard_sha256, stage{h,w,frames,fps}, latent:"mean"}`; output
`latent_shard`; summary `{clips, sha256, latent_mean, latent_std}`; RAM 8 GB, CPU allowed, trusted.

**`ai.hd_train`** — inputs `base_train_weights, base_manifest, data_shard`×k (never val shards); params
`{update_id, run_id, round, round_batch, base_version_id, target:"dit", hd_config, hd_config_sha, stage,
inner_steps, batch_size, grad_accum, lr, warmup_steps, weight_decay, grad_clip, cfg_dropout, seed,
shards[{artifact_id, sha256}], delta_dtype:"float16"}`; fresh inner optimizer each round (~20 warmup steps);
outputs `update_weights, update`; summary `{loss_first, loss_last, samples_seen, wall_sec, device_class}`.

**Shard assignment** (deterministic): `perm = permute(train_shards, seed = DataSeed ⊕ epoch)`;
`slot = round·W·8 + batch·W + i`; k = `shards_per_job` consecutive entries of `perm` from position
`slot·k mod S` (epoch increments on wrap). Workers keep an LRU cache (`dataset_cache_gb`, default 20).

**`ai.hd_aggregate`** (controller `runPython`) — inputs `base_weights, base_manifest, outer_state?, updates[],
eval_latents?`; params `{outer_lr, momentum, nesterov, version_id, bound_c, inner_steps, lr, eval_gate,
eval_triples_seed}`; summary as today plus `train_weights` output.

**`ai.hd_eval`** — inputs `weights, manifest, val latent_shard`s; params `{version_id, triples_seed, n,
timestep_buckets}`; output `eval_report` (+ optional `eval_sample`); trusted only.

**RunConfig additions**: `family` ("small"|"hd"), `target` ("dit"|"vae"), `hd_config`, `hd_config_sha`,
`dataset_id`, `stage{h,w,frames,fps}`, `vae_version_id`, `text_encoder_id`, `shards_per_job`, `delta_dtype`,
`req_vram_mb`, `eval_gate`, `eval_clips`, `bound_c` (3.0), `redundancy_frac` (0), `retention{keep_last,
keep_every}`, `inner.warmup_steps`, `inner.grad_accum`, `inner.cfg_dropout`.

**HD defaults**: WorkersPerRound 4, MinUpdates 2, RoundTimeoutSec 7200, inner steps 500, batch 4 × accum 2,
lr 1e-4, outer lr 0.7 / momentum 0.9 Nesterov, req_vram_mb 5500 (base) / 22000 (medium), LeaseSeconds 900 with
heartbeat. Validation: `family=hd` requires a latents dataset whose encoder matches `vae_version_id` /
`text_encoder_id`, and a config sha matching the init manifest.

**Migration `0008_hd_training.sql`**: `datasets(id, project_id, name, kind, parent_id, fingerprint, encoder jsonb,
redistribution, status, clip_count, created_at)`; `dataset_shards(dataset_id, idx, split, artifact_id, sha256,
clips, bytes, PK(dataset_id, idx))`; `model_versions` + `train_weights_artifact, family, compat_sha`;
`training_updates` + `content_sha256` with unique `(run_id, round, content_sha256)`.

**Checkpoints/bundles**: version weights use `hd_trainer` tensor names; manifest carries `{family, target,
hd_config_sha, vae_version, text_encoder_id, tokenizer_sha, dataset_fingerprint, stage}`.

**Single machine**: `python -m nivra.cli train-pipeline --config nivra_base --videos DIR --out RUN` — ingest →
BPE → NivraText MLM → VAE → latents → DiT → bundle, each stage resumable with an input-fingerprint marker; latents
use the `latent_shard` format. Hand-off to distributed: `nivra model import BUNDLE` → `nivra dataset push --kind
latents` → RunConfig `init_*`.

Text encoder on the network: `nivra model import` packs a Nivra-trained `text_encoder/` into a deterministic tar artifact (kind `text_encoder_archive`, `model_versions.text_encoder_artifact`, migration 0009); `versionInputs` and long-video generations add it as input role `text_encoder_archive` to `ai.generate`/`ai.draft`/`ai.refine_segment`/`ai.evaluate`, and the worker verifies and extracts it (no `NIVRA_TEXT_ENCODERS_DIR` needed). `external` encoders are never packed. HD training jobs are unchanged: `ai.hd_encode` keeps taking `text_encoder_weights` + `tokenizer`, and DiT training/eval work on precomputed latents/embeddings, so they never need the archive.

### Controller conventions (fixed by the controller implementation)

These are the exact shapes `internal/controller/training/{hd_orchestrator,hd_validators,datasets}.go` produce and check.

**Input roles** (the worker hands inputs to Python as a dict keyed by role, so roles are unique):

| job | roles |
|---|---|
| `ai.hd_train` | `base_train_weights`, `base_manifest`, `data_shard_000`, `data_shard_001`, ... (3 digits, same order as `params.shards`; one entry per distinct shard) |
| `ai.hd_eval` | `weights`, `manifest`, `val_shard_000`, `val_shard_001`, ... |
| `ai.hd_encode` | `raw_shard`, `vae_weights`, `vae_manifest`, `text_encoder_weights`, `tokenizer` |
| `ai.hd_aggregate` | `base_weights`, `base_manifest`, `outer_state` (optional), `updates` = list of `{path, manifest_path, update_id}` (like `ai.aggregate`), `eval_latents` = list of val shard paths (only when `eval_gate=pi`; for `target=vae` these are raw val shards) |

`base_train_weights` is the version's fp16 train copy; version 0 (copied from the imported bundle) has none, so it is the
fp32 master. Workers compute the delta against exactly what they received, in fp32.

**`ai.hd_train` params**: exactly `update_id, run_id, round, round_batch, base_version_id, target, hd_config, hd_config_sha,
stage{h,w,frames,fps}, inner_steps, batch_size, grad_accum, lr, warmup_steps, weight_decay, grad_clip, cfg_dropout, seed,
shards[{artifact_id, sha256}], delta_dtype`. `update_id` is deterministic (UUID from run/round/batch/index); `base_version_id` is the
`version_id` inside the base manifest; `seed = Seed*1000003 + slot`.

**Update manifest** (artifact `update`): must echo `run_id, round, round_batch, base_version_id, update_id, attempt_id, target,
hd_config_sha, stage, inner_steps, lr, shards[], samples_seen, delta_dtype` plus `schema: "nivra.update/<N>"`. `attempt_id` is the
attempt id the worker received. Any difference from the job params (lr compared to 1e-6 relative) rejects the result;
`samples_seen` must be in `1..inner_steps*batch_size*grad_accum`. The result summary carries `loss_first, loss_last, samples_seen,
wall_sec, device_class` (stored as unverified metadata).

**`update_weights`** (fp16 safetensors): must contain exactly the base's tensor names and shapes, every tensor `F16`, tensors tiling the
data area with no gap or trailing byte (file size = `8 + header + 2*sum(numel)`), no NaN/Inf, per tensor `max|d| <= c*H*lr` and
`||d||_2 <= c*H*lr*sqrt(numel)` (`c = bound_c`, `H = inner_steps`), and not all zero. The controller checks all of this itself, by
streaming the file in Go; there is **no** `ai.hd_validate_update` job. The median outlier test and median clipping remain in
`ai.hd_aggregate`, which must also re-check bounds (it receives `bound_c`, `inner_steps`, `lr`).

**Shard assignment** (exact): train shards sorted by `idx`, `S` of them, `k = shards_per_job`. `slot = round*W*8 + batch*W + i`.
Entry `j` (0..k-1) of a job sits at global position `g = slot*k + j`, epoch `e = g / S`, shard `perm_e[g mod S]`. `perm_e` is
Fisher-Yates over `0..S-1` from the end (`for n = S-1 down to 1: m = next() mod (n+1); swap(p[n], p[m])`) driven by splitmix64
seeded with `uint64(DataSeed) xor uint64(e)` (`state += 0x9E3779B97F4A7C15`, then the usual two multiply-xorshift rounds; first
output for seed 0 is `0xe220a8397b1dcdaf`). The controller sends the resulting shards explicitly; Python never recomputes them.

**`ai.hd_aggregate`**: params `{outer_lr, momentum, nesterov: true, version_id, bound_c, inner_steps, lr, eval_gate, eval_triples_seed}`
(`eval_gate` is `pi | worker | off`; Python measures held-out loss only for `pi`). Summary fields used: `new_version_id`, `no_op`,
`applied_update_ids`, `rejected[{update_id, reason}]`, `aggregated_count`, `aggregation_method`, `delta_l2`, `heldout_loss_before`,
`heldout_loss_after`. Output artifact kinds: `model_weights` (fp32 master), `model_train_weights` (fp16), `model_manifest` (the
controller also accepts `model_version`), `outer_state`.

**`ai.hd_eval`**: params `{version_id, triples_seed, n, timestep_buckets}`; artifact `eval_report` = JSON
`{"schema": "nivra.hd_eval/1", "version_id": <params.version_id>, "loss": <float>, "n": <int>}`. With `eval_gate=worker` the controller
evaluates the candidate (and the base once, if its loss is unknown) and promotes only if `loss_new <= loss_base*(1+MaxHeldoutIncrease)`.

**`ai.hd_encode`**: params as in the contract; an audit re-encode (about 5% of shards, only when at least two trusted devices exist)
adds `"audit": true`, runs under key `enc:{target}:{vae}:{idx}:audit` and must run on a different device. `{dataset}` in the
idempotency key is the *target latents dataset* id. Summary `{clips, sha256, latent_mean, latent_std}`: `clips` must equal the raw
shard's registered clips, `sha256` the uploaded `latent_shard`, mean/std finite with std > 0. The audit compares means and stds
(`|diff| <= 0.0317*std`, a necessary condition for MSE < 1e-3*var); the per-element MSE check needs the tensors and is not done.

**Registry details**: `text_encoder_id` of an encoded dataset is the sha256 of the text encoder artifact; `vae_version_id` is a model
version whose manifest has `target: "vae"` (its train copy or master is the VAE) or a full bundle (its `vae_artifact`). A dataset's
`fingerprint` is the client's (dataset.json fingerprint) or the sha256 of its shard list; the same fingerprint cannot be registered twice
per project and kind. Operator uploads: `POST .../artifacts/uploads {kind, sha256, size}` (beginning the same upload again resumes it),
`PUT .../artifacts/uploads/{id}?offset=N` (<= 8 MB, 409 with the authoritative offset on mismatch), `POST .../uploads/{id}/complete`.
Not implemented: `redundancy_frac > 0` (rejected at creation).

### Python executor conventions (`ai/nivra/distributed/hd/`)

* **No `ai.hd_validate_update` job**: update validation is done in Go. `distributed/hd/validate.py` (`validate_hd_update`)
  is the same check in Python; `ai.hd_aggregate` runs it as its first pass and the tests use it.
* **Streaming IO** (`stream_io.py`): `StReader` parses and validates the safetensors header itself (contiguous offsets, exact
  size) and reads one tensor at a time with seek/read, **not** mmap (a mapped 2 GB delta would count into the working
  set on Windows). `StWriter` writes the header first, then the tensors in order, hashing as it goes. The size of an fp16 delta is
  exactly `8 + header + 2*sum(numel)`; the aggregate summary carries `delta_bytes` (exact size of a conforming update for the
  new version) and `expected_delta_bytes(tensor_shapes)` computes it.
* **`ai.hd_train`** (`train_job.py`): accepts the fp16 train copy or the fp32 master as `base_train_weights` (its sha256 must equal
  the manifest's `train_weights_sha256` or `weights_sha256`); `base_version_id` and `hd_config_sha` are checked against the
  base manifest; every `data_shard_NNN` must match `params.shards[i].sha256` and its embedded stage. Fresh optimizer per round
  (`adamw`/`adamw8` from the config), linear warmup to `lr` then constant, fp16 autocast + GradScaler on CUDA, gradient
  checkpointing. delta = received - trained, both fp32, written tensor by tensor as fp16. Manifest = `nivra.update/1` + `family: "hd"`
  + all echo fields (`attempt_id` = the exec job's `attempt_id`, `samples_seen = inner_steps*batch_size*grad_accum`) and
  `update_sha256, update_file, base_weights_sha256, architecture_version, vocab_hash, tensor_shapes, loss_first, loss_last`.
  Summary: `update_id, base_version_id, loss_first, loss_last, samples_seen, wall_sec, device_class, fp16_rel_err`.
  Artifacts: `update`, `update_weights`. `target=vae` is not implemented (exit 2).
* **`ai.hd_aggregate`** (`aggregate.py`): two passes, never a whole model in RAM. Pass 1 per update: header and exact size, sha256,
  echo fields (the entry's `update_id` must match the manifest), finiteness, per-tensor bounds (`bound_c*H*lr`), zero check,
  `max_rel_norm` (default 0.5 of `||base||`), duplicate `update_id` / duplicate content sha256, already-applied ids. Then the median
  outlier test (`max_norm_ratio` x median, needs >= 3 updates) and median clipping (`clip_factor` 2 x median). Pass 2 per tensor: mean
  (n < 3) or coordinate-wise trimmed mean (n >= 3, `trim_ratio` 0.2) of the clipped deltas, outer Nesterov SGD, written simultaneously to
  `weights.safetensors` (fp32 master), `train_weights.safetensors` (fp16) and `outer_state.safetensors` (fp32). The new
  `model_version.json` copies the base manifest and updates `version_id, round, parent_version_id, weights_sha256, weights_dtype,
  train_weights_file/sha256/dtype, outer_state_file/sha256, applied_update_ids, tensor_l2, weights_l2, delta_bytes`.
  Extra optional params (all defaulted): `aggregation, trim_ratio, max_norm_ratio, max_rel_norm, clip_factor, eval_n (8), eval_buckets (4)`.
  Rejections are `{update_id, reason (code), detail}`; codes: `manifest_invalid, size_limit, size_mismatch, checksum_mismatch,
  shape_mismatch, stale_base_version, manifest_mismatch, non_finite, bound_exceeded, zero_update, norm_outlier, duplicate_update`.
  `eval_gate=pi`: the base and the new version are evaluated on the same 8 fixed (latent, t, noise) triples drawn from `eval_latents`
  (`make_triples`: a pure function of `eval_triples_seed` and the shards); summary `heldout_loss_before/after`.
* **`ai.hd_eval`** (`eval_job.py`): the same triples on `weights` (fp16 or fp32) + `manifest`; artifact `eval_report`
  `{schema "nivra.hd_eval/1", version_id, loss, n, per_t}`; summary `{loss, n}`.
* **`latent_shard`** (`latent_shard.py`, produced by `ai.hd_encode`): one safetensors file. Tensors `latents` fp16 `[N,C,T,H,W]`
  (already scaled: `(z - latent_shift) * latent_scale`, posterior mean), `text_emb` fp16 `[N,L,D]`, `text_mask` u8 `[N,L]`,
  `null_emb`/`null_mask` (encoder output of the empty prompt). One metadata key `nivra` = JSON `{format "nivra-latents/1",
  parent_shard_sha256, vae_version, text_encoder_id, text_encoder_name, tokenizer_sha256, stage{h,w,frames,fps}, latent_shift,
  latent_scale, split, dataset_id, target_dataset_id, shard_idx, rows[{id, caption, text, style, lang, split}]}`.
  `vae_version` = sha256 of the VAE weights file, `text_encoder_id` = sha256 of the text encoder weights file (as in the registry).
* **`ai.hd_encode`** (`encode_job.py`): `raw_shard` must carry its own per-clip rows (id, caption, style, lang) in its `nivra` metadata,
  which `data-ingest` now writes (older datasets must be re-ingested; the single-machine pipeline reads `manifest.jsonl` instead).
  `vae_manifest` may be a bundle `model_version.json` or `hc_vae.json`; the NivraText config is read from the weights' metadata
  (`nivra_text_config`, written by new checkpoints) or inferred from the tensor shapes (heads and max_len guessed from the presets for old
  checkpoints). Style-tag dropout (`style_dropout`, default 0.1) is applied once, deterministically per clip id. Optional params:
  `style_dropout, seed, audit`. A second encode on the same machine is byte-identical.
* Bundle manifests written by `write_hd_bundle` now also carry `hd_config_sha` (= `config_hash`); `ai.hd_aggregate` keeps it.

### Single-machine pipeline (`train-pipeline`)

`python -m nivra.cli train-pipeline --config nivra_base (--videos DIR | --dataset DIR) --out RUN [--install-dir DIR | --bundle-out DIR]
[--name N] [--from-stage S] [--to-stage S] [--vae-steps N --dit-steps N --text-steps N --vocab-size N] [--plan-only] [--force]`.
Stages `ingest, tokenizer, text, vae, latents, dit, bundle`; `RUN/markers/<stage>.json` holds the input fingerprint (options + upstream
output fingerprints) and the outputs; a stage is skipped only when both match and the output files exist, so a changed option re-runs that
stage and everything downstream. `vae` and `dit` also resume from their step checkpoints (`RUN/vae_run`, `RUN/dit_run`). The plan with rough
time estimates (FLOP model; 0.12 TFLOP/s CPU, 8 TFLOP/s CUDA sustained; order of magnitude only) is printed first. Latents are encoded at the
largest training stage that fits the dataset; DiT stages larger than the data are dropped. The bundle (`write_hd_bundle`, text encoder
`nivra:RUN/text_encoder` copied in, VAE calibration folded into the config) goes to `<install-dir>/models/<name>`, where the app's
model discovery finds it. `export-hd-bundle --run RUN` re-writes only the last stage.

### `nivra_base` (config `ai/configs/nivra_base.json`)

DiT dim 1024 / depth 12 / 16 heads / MLP x4 / text_dim 256: **212.7 M** parameters (blocks 201.6 M); HC-VAE 64 latent channels,
channels 32-64-128-256-256-512, one res block per level: **59.6 M**; NivraText "small" preset (256-d). 256x448, 17 frames at 8 fps is
14 x 8 x 3 = 336 latent tokens. Training memory estimate (`hd_budget.training_bytes`: fp32 master + grads + optimizer + checkpointed
activations; add about 1.1 GB for the CUDA context and fp16 weight casts; **not measured, there is no GPU here**):

| stage (tokens/sample) | batch 1 / 2 / 4, AdamW fp32 | batch 1 / 2 / 4, AdamW8bit |
|---|---|---|
| 128p 17f (84) | 3.2 / 3.2 / 3.2 GiB | 2.0 / 2.0 / 2.0 GiB |
| 256p 17f (336) | 3.2 / 3.3 / 3.4 GiB | 2.0 / 2.1 / 2.2 GiB |
| 256p 33f (560) | 3.3 / 3.4 / 3.6 GiB | 2.1 / 2.2 / 2.4 GiB |

So 6 GB holds the config default (batch 2 x accum 4, AdamW8bit) with about 3 GB headroom. The VAE (59.6 M, fp32 + AdamW, no
activation checkpointing) is the larger risk: train it on the 128p stage first.

### Measured (CPU, Python 3.14, torch CPU)

* Streaming aggregate vs in-memory reference on `nivra_hd_tiny`: max abs difference < 1e-6 (`test_streaming_aggregate_matches_reference`).
* Aggregate of 4 synthetic 932 M-parameter (Medium-shaped) fp16 deltas + fp32 base: **peak working set 1.47 GB** (limit 4 GB), 350 s wall
  including a sha256 of every file (`ai/scripts/hd_agg_memory.py --config nivra_medium --updates 4`; psutil is not installed, the
  script uses `peak_rss_mb()`, i.e. Windows PeakWorkingSetSize).
* A 50-step honest run on tiny passes the bounds; x1000-scaled, NaN, zero, bit-flipped, wrong-lr, stale-base, replayed and duplicate updates
  are rejected; the fp16 round trip of the delta has relative error < 1e-3 (asserted in the tests).
* **HC-VAE `vae_tile=8` crash at 2560x1440** (exit 139): the root cause is not the tile/radius math. Every spatial tile keeps its own
  causal-cache state between time chunks; 2560x1440 with tile 8 is 8 x 13 = 104 tiles, about 121 MiB each for the tiny VAE (fp32, 22x22-cell
  window) and about 2 GiB each for Medium, and each cache was a *view* that also kept the whole padded chunk input alive. Memory ran
  out inside oneDNN, which does not handle the failed allocation. Fix in `compression/hc_vae.py`: caches are cloned to the frames they need,
  and `decode_stream/encode_tiled(..., max_state_bytes=None)` estimate the total tile state first (`stream_state_bytes_per_tile`, a
  meta-device dry run) and raise `StreamStateTooLarge` (a `ValueError`) when it exceeds 60 % of free memory instead of crashing.
  Regression tests: `ai/tests/test_hd_vae_tilestate.py`. (tile=8 at that size remains impractical for Medium; use a larger tile.)

## Low-bandwidth rounds (int8 deltas, error feedback, weighting, staleness, warm-up)

Research item 11 and the DiLoCo part of item 8 (`docs/research/IMPROVEMENT_RESEARCH_2026.md`). Everything is opt-in; the defaults are the
old behaviour (`delta_dtype: "float16"`).

**`delta_dtype: "int8"`** (run config field, echoed in the job params and in the update manifest; format id `int8-rowscale/1`,
code `ai/nivra/distributed/hd/quant.py`, Go `internal/controller/training/hd_int8.go`). `update_weights` holds, for every base
tensor `NAME` of shape `S`, in the base's order:

| tensor | dtype | shape | content |
|---|---|---|---|
| `NAME` | `I8` | `S` | `q` in `-127..127` (`-128` is rejected) |
| `NAME#scale` | `F32` | `[rows(S)]` | per-row scale `s >= 0`, finite |

`rows(S) = 0` if `numel(S) = 0`, `S[0]` if rank >= 2, else `1` (vectors and scalars share one scale). Dequantised value =
`float32(q) * s` (a float32 multiply on both sides, see the golden fixture below). Quantiser: `s = max|row| / 127`, `q = round(x / s)`; an
all-zero row (or a row whose `s` underflows to 0) has `s = 0, q = 0`. **Exact size** = `8 + header_len + sum(numel) + 4 * sum(rows)`
(no gap, overlap or trailing byte; the controller and the Python validator check names, dtypes, shapes and the size from the header
before reading any data). Content rules: finite non-negative scales, no `-128`, no non-zero `q` under a zero scale, finite
dequantised values, not all zero, and the same bounds as fp16 applied to the **dequantised** values
(`max|d| <= c*H*lr*slack`, `||d||_2 <= c*H*lr*sqrt(numel)*slack`) with slack `1 + 2^-7` instead of fp16's `1 + 2^-9`, because error
feedback can add up to half a quantisation step of the previous round. A base tensor name ending in `#scale` is refused. Size: about
half of fp16 for real shapes (1 byte per element + 4 bytes per row against 2 bytes per element, so
the 213 M `nivra_base` goes from ~430 MB to ~215 MB); the tiny test model, with very small rows, only reaches ~0.63x.

**Error feedback** (worker side, `train_job.py`, `quant.ResidualStore`): the worker quantises `delta + residual` and keeps
`residual = (delta + residual) - dequant(q)` in `<state_dir>/hd_residual/<run>_<target>.safetensors` (+ `.json`; fp32, as large as
the model). `state_dir` = `$NIVRA_STATE_DIR`, else `<worker data dir>/state` derived from the job's work directory, else none (quantisation
without feedback; the Go worker does not set the variable yet, the derivation covers the normal `<data dir>/work/<attempt>` layout).
The residual is committed only after `update.safetensors` and `update.json` are written. It is used only when the round's base is the
**direct successor** of the version it was computed against (`meta.base_version_id == base.parent_version_id`), the update it belongs to is in
the base's `applied_update_ids`, and run id, `hd_config_sha`, tensor shapes and file sha256 match; otherwise (new run, missed rounds, update
rejected or clipped out, rolled-back version, damaged file) it is deleted and the round starts clean. The manifest and the summary say
`error_feedback` and `residual_reset` (reason or null). A retry of the same job quantises from a clean residual.

**Aggregation** (`aggregate.py`): updates are dequantised tensor by tensor while streaming (same peak memory as fp16), clipped, and
combined as a **weighted** mean / trimmed mean. Weight = `samples_seen / median(samples_seen)` capped at 2 (a worker cannot buy influence by
over-claiming) times `staleness_discount` for stale updates. With equal `samples_seen` and no stale update all weights are 1 and the result is
identical to the unweighted aggregate. New `ai.hd_aggregate` params: `delta_dtype`, `weight_by_samples` (true), `max_staleness` (0),
`staleness_discount` (0.5), `outer_lr_warmup_rounds` (0 = off), `outer_lr_warmup_start` (0.3). The summary reports `update_weights`,
`stale_update_ids`, `outer_lr` (effective) and `outer_lr_nominal`.

**Staleness** (aggregator only): `max_staleness = 1` also accepts an update computed against the PARENT of the current base (its
`base_version_id` is then checked against that id), weighted by `staleness_discount`; two or more versions old is always rejected. The
controller does **not** use this yet: `validateHDTrain` still rejects results for an older model version and the round state machine drops
in-flight jobs when a round is promoted, so late updates need orchestration work (hold late jobs across a promotion and load them into the
next round). It is available and tested at the executor level only.

**Outer-lr warm-up**: with `outer_lr_warmup_rounds = K` the effective outer lr after `r` completed rounds is
`outer_lr * (start + (1 - start) * min(1, r / K))`, i.e. 0.3x in round 0 rising to 1x from round K. Run-config fields
`outer_lr_warmup_rounds` and `outer_lr_warmup_start` (default 0.3 when K > 0); the controller passes them to the aggregator, which reads
the round from the base manifest (so it survives a controller restart). To re-warm after a stage change, start a new run.

**Golden fixture shared Go <-> Python**: `ai/tests/fixtures/hd_int8_golden.safetensors` + `.json` (4 tensors: a 2-D one with an all-zero
row, a vector, a 3-D tensor and a scalar). The Python test regenerates the bytes and compares them byte for byte, the Go test
(`TestInt8GoldenFixtureSharedWithPython`) reads the same files and compares sizes, scales, `q`, and per-tensor and total L2 / Linf.
Change the format only together with both sides, then `NIVRA_REGEN_FIXTURES=1 pytest tests/test_hd_int8.py -k golden`.

### Training recipe inside a round

`ai.hd_train` follows `hd_trainer.train_hd` (the config's `train` options through `train_options(cfg)`), using the trainer's own
functions (`rf_loss_ex`, `sample_t_frames`, `make_xt`, `SRAProjector`, `sra_loss`, `fps_dropped`):

* **Diffusion forcing** (`train.diffusion_forcing.fraction`) and **first/last-frame conditioning** (`cond_fraction`, needs `model.frame_cond`).
* **fps conditioning** (`model.fps_cond`): each clip's `effective_fps` (carried into the latent shard rows by `ai.hd_encode` when the raw shard
  rows have it) with the deterministic `train.fps_dropout`.
* **fp32 islands**: `train.fp32_attn`, and a dynamic `GradScaler` with `train.loss_scale` (`force_loss_scaling` for CPU tests). An overflow
  skips the optimizer step and backs the scale off; the round still counts `inner_steps` and the delta stays bounded; 200 consecutive skips
  abort the job. `summary.skipped_steps` and `manifest.recipe.skipped_steps` report it.
* **SRA** (`train.sra`): there is no EMA on a worker (the outer step does the averaging), so the teacher is a **frozen copy of the weights
  received for this round** (one extra model copy in memory and one no-grad forward per micro-batch; it aligns an early block to a deeper
  block of the same starting point at `t_ratio * t`). The projector is training-only and re-initialised every round; it never enters the
  delta. Set `train.sra.weight = 0` for Medium/Large on small GPUs.

The update/echo format is unchanged (the manifest only gains `recipe`, `delta_format`, `error_feedback`, `residual_reset`,
`quant_rel_err`; the Go validator ignores unknown keys). The RNG draw order is unchanged for configs without these options.

`ai.hd_encode` accepts `params.text_encoder_spec = "external:<name>"` (see `docs/training/TEXT_ENCODERS.md`; the `text_encoder_weights` /
`tokenizer` inputs are then optional), loads v1 and v2 HC-VAE directories (`HCVAE_VERSIONS`), always uses the VAE's own `to_scaled` (scalar
or per-channel) and fails if text embeddings are not finite in fp16.

## Feasibility (estimates)

* `nivra_base` at 256p on an RTX 2060: ≈ 0.7 s/sample → a 500 × 8-sample round ≈ 45–60 min; ≈ 400 MB down +
  400 MB up per round (fp16); first round ≈ 1 GB data, then cached.
* 4 × RTX 2060 ≈ 1.5e18 FLOP/day; usable open-domain text-to-video needs ~1e20–1e21 FLOP (months). Narrow the
  domain (one style), train images (T=1) first, then 128p/2 s, then 256p.
* Pushing ~2 GB/round through a free Cloudflare Tunnel may be throttled or against its terms: prefer LAN or
  Tailscale for training workers.
