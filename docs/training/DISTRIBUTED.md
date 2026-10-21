# Nivra distributed training (DiLoCo-style) - executor job schemas

Authoritative contract for the Go controller/worker. Implemented in
`ai/nivra/cli.py` (job dispatch + strict validation) and `ai/nivra/distributed/`.
Protocol envelope: `docs/architecture/EXECUTOR_PROTOCOL.md` (`nivra.exec/1`).

Status: schemas below are implemented and covered by `ai/tests/test_exec_protocol.py`
and `ai/tests/test_distributed.py`. If code and this document ever disagree, the code
plus those tests win and this document is a bug.

## 0. Conventions

* All tensors are **safetensors**. No pickle anywhere. JSON for structure.
* Params are validated **strictly**: unknown keys, wrong types, out-of-range values
  -> `error` event `INVALID_SPEC`, exit 2. Unknown/invalid `inputs` keys -> exit 2.
  Missing / unreadable / checksum-mismatching / incompatible input files ->
  `INPUT_INVALID`, exit 5.
* `inputs` values are absolute (or workdir-relative) file paths. Each job type lists
  required (R) and optional (O) input keys.
* Stop file: checked before work starts and between steps/samples -> `error` event
  `STOPPED`, exit 3.
* `limits.max_seconds` is enforced at the same safe boundaries (treated like a stop:
  `STOPPED`, exit 3). `max_ram_mb` is advisory (not enforced inside the process).
* Device: `NIVRA_DEVICE` (default `cpu`), threads `NIVRA_THREADS` (default 4).
* **Resource limits (always on, no params needed).** Every JSON input (`model_version.json`, `update.json`,
  prompts files) is size-checked *before* it is read: > 1 MiB -> `INPUT_INVALID`, exit 5
  (in `ai.aggregate` an oversized *update* manifest only rejects that update: reason `manifest_too_large`).
  Every safetensors input is opened header-only first (`safe_open`): total file size, declared tensor bytes
  (<= `max_bytes`, default 1 GiB), tensor names, shapes and dtypes (updates must be float32) are compared with the
  base BEFORE any tensor is loaded; a mismatch is rejected without allocating the tensors. Whole files of unknown
  size are never read into memory (hashing is streamed).
* Every output file is also announced with an `artifact` event
  `{kind, path, sha256, bytes}` (path inside `output_dir`).

### 0.1 The "model version" (global generator weights)

A *model version* is the unit that DiLoCo rounds advance. It is the pair
`weights.safetensors` + `model_version.json` (the **version manifest**).
The manifest is the file announced by the artifact of kind `model_version`.

```json
{
  "schema": "nivra.model_version/1",
  "version_id": "v000003-1a2b3c4d5e6f",
  "round": 3,
  "parent_version_id": "v000002-....",
  "architecture_version": "nivra-small/1",
  "config_name": "nivra_small",
  "config": { "...full architecture config object (vae + model + data sections)..." },
  "tokenizer_version": "nivra-wordtok/1",
  "vocab_hash": "<sha256 of canonical vocab>",
  "param_count": 1234567,
  "tensor_shapes": {"<name>": [dims]},
  "tensor_l2": {"<name>": 12.3},
  "weights_l2": 117.2,
  "weights_file": "weights.safetensors",
  "weights_sha256": "<hex>",
  "outer_state_file": "outer_state.safetensors",
  "outer_state_sha256": "<hex or null>",
  "vae_sha256": "<hex of the frozen VAE weights this generator was trained against>",
  "applied_update_ids": ["uuid-a", "uuid-b"],
  "created_unix": 1760000000
}
```

* `tensor_shapes`, `tensor_l2` (per-tensor L2 norm) and `weights_l2` are written by every version this package
  creates (`ai.init_model`, `ai.aggregate`). `ai.validate_update` uses them as the base norms when `inputs.base_weights`
  is not passed. A base manifest that has neither them nor `base_weights` makes every update fail with
  `base_norm_unavailable` (fail closed). Versions made by an older build lack these fields: pass `base_weights`.
* `version_id` is `params.version_id` when supplied by the controller, otherwise
  `v{round:06d}-{weights_sha256[:12]}`.
* `weights.safetensors`: every trainable generator parameter (text encoder +
  transformer), float32, names = module state-dict names.
* The VAE ("Nivra tokenizer") is **frozen** and shared by all versions. It is a
  separate file (`vae.safetensors`) distributed once; its hash is `vae_sha256`.

### 0.2 The "update" (pseudo-gradient)

`update.safetensors`: for each tensor name in the weights, `delta = base - local`
(float32, same shape). Manifest `update.json`:

```json
{
  "schema": "nivra.update/1",
  "update_id": "uuid",
  "base_version_id": "v000002-....",
  "base_weights_sha256": "<hex>",
  "architecture_version": "nivra-small/1",
  "vocab_hash": "<hex>",
  "inner_steps": 30,
  "samples": 240,
  "loss_first": 0.91,
  "loss_last": 0.74,
  "loss_before": 0.88,
  "loss_after": 0.79,
  "delta_l2": 1.234,
  "update_file": "update.safetensors",
  "update_sha256": "<hex>",
  "shard": {"seed": 1, "start": 0, "count": 256, "fingerprint": "<hex>"},
  "wall_seconds": 61.2,
  "device": "cpu"
}
```

`loss_first/last` = mean flow-matching training loss over the first / last 5 inner
steps. `loss_before/after` = fixed-noise held-out loss (last `eval_clips` clips of
the shard, never trained on) before / after the inner loop.

## 1. Job types

### 1.1 `ai.init_model` (controller)

Creates global version 0 (or seeds a distributed run from a single-node checkpoint).

params (all optional unless noted; exactly one of `config` / `from_checkpoint`):
| key | type | default | notes |
|-----|------|---------|-------|
| `config` | string or object | - | architecture config name (`nivra_small`, `nivra_tiny`) or a full config object |
| `seed` | int >= 0 | 0 | init seed (config path only) |
| `from_checkpoint` | string path | - | a Nivra generator checkpoint directory (same as `inputs.from_checkpoint`) |
| `use_ema` | bool | true | with `from_checkpoint`: take EMA weights |
| `vae_checkpoint` | string path | - | VAE checkpoint dir or `vae.safetensors` (same as `inputs.vae_checkpoint`); copied to `out/vae.safetensors` |
| `version_id` | string | derived | |

inputs (O): `from_checkpoint`, `vae_checkpoint` (equivalent to the param forms).

outputs in `output_dir`: `weights.safetensors`, `model_version.json`
(`round: 0`, `applied_update_ids: []`, `outer_state_file: null`),
`vae.safetensors` if a VAE was given.
artifacts: `model_version`, `model_weights`, `vae_weights`.
summary: `{version_id, architecture_version, tokenizer_version, vocab_hash, param_count, weights_sha256}`.

### 1.2 `ai.dataset_shard` (worker)

params: `seed` int>=0 (R), `start` int>=0 (R), `count` int 1..4096 (R),
`frames` int 2..32 (16), `size` int 16..128 (64).
inputs: none. outputs: `shard.safetensors` (`videos` uint8 `[N,T,H,W,3]`),
`shard.json` (manifest with `fingerprint`, `content_sha256`, per-clip captions + specs).
artifacts: `dataset_shard`, `dataset_manifest`. summary: `{count, fingerprint, content_sha256, shard_file_sha256}`.
**Changed:** `content_sha256` is now the sha256 of exactly the raw bytes of the `videos` tensor stored in
`shard.safetensors` (uint8, C-order `[N,T,H,W,3]`; it used to also hash the scene specs). `shard_file_sha256` is the sha256 of
the `shard.safetensors` file itself, which is byte-identical across platforms for the same params (no safetensors
metadata, no timestamps; the generator uses only numpy PCG64 and IEEE-exact float64 ops, no libm transcendentals).
The controller re-runs `ai.dataset_shard` with the same params and compares both digests with the uploaded artifact.
Pinned digests are in `ai/tests/test_exec_protocol.py` (`seed=5,start=10,count=6,frames=8,size=32`).

### 1.3 `ai.local_train` (worker)

params:
| key | type | default | bounds |
|-----|------|---------|--------|
| `update_id` | string | R | 1..128 chars |
| `shard` | object `{seed,start,count}` | R | count 8..4096 |
| `inner_steps` | int | 30 | 1..2000 (H) |
| `batch_size` | int | 8 | 1..64 |
| `lr` | float | 3e-4 | (0, 1e-2] inner AdamW lr, constant after short warmup |
| `weight_decay` | float | 0.0 | [0, 0.2] |
| `grad_clip` | float | 1.0 | (0, 100] |
| `seed` | int | 0 | RNG seed of the inner loop |
| `eval_clips` | int | 16 | 0..256 |

inputs: `base_weights` (R, `weights.safetensors`), `base_manifest` (R, `model_version.json`),
`vae_weights` (R). Hash of `vae_weights` must equal manifest `vae_sha256`
and `base_weights` must match `weights_sha256` (else `INPUT_INVALID`).
outputs: `update.safetensors`, `update.json`. artifacts: `update` (path = `update.json`),
`update_weights`. metrics: `loss` per step, `loss_before`, `loss_after`.
summary: `{update_id, base_version_id, inner_steps, samples, loss_first, loss_last, loss_before, loss_after, delta_l2, update_sha256}`.
`samples = inner_steps * batch_size`. Each inner AdamW is fresh (zero state) every call.

### 1.4 `ai.validate_update` (controller)

params (O): `max_norm_ratio` float>=1 (5.0), `max_bytes` int (1 GiB),
`max_norm` float (none; absolute L2 cap), `max_rel_norm` float>0 (0.5), `peer_norms` list[float] (norms of other updates in the
same round; ratio test uses their median), `expected_base_version_id` string.
inputs: `update` (R, safetensors), `update_manifest` (R, update.json),
`base_manifest` (R), `base_weights` (O; enables an exact shape/dtype check against the real weights,
otherwise the check uses the manifest `tensor_shapes`).
Checks: manifest schema; `update_sha256` matches file; tensor names/shapes/dtypes identical to base;
all values finite; size <= `max_bytes`; L2 norm <= `max_norm_ratio` x median(peer norms)
(if given) and <= `max_norm` (if given); **absolute bounds, always enforced even with 0 peers**:
`||delta|| <= max_rel_norm x ||base||` (`norm_outlier`), per tensor `||delta_t|| <= 4 x ||base_t|| + 1e-3 + 0.05 x sqrt(numel_t)`
(`tensor_norm_outlier`; the per-element term lets zero-initialised tensors such as adaLN-zero / output heads move in
the first rounds: measured on a from-scratch init, the strict `+ 1e-3` rule rejected a legitimate update at
`blocks.0.ada.bias`), `||delta|| >= 1e-8` (`zero_update`); `||base||` / `||base_t||` come from `base_weights` or the
manifest `tensor_l2`; oversized manifest -> exit 5; header checked before tensors are loaded; base_version_id equals the expected / manifest version;
architecture & vocab hash match.
Result is always `status: ok` unless the job itself is malformed;
summary `{valid: bool, reasons: [str], l2_norm, update_id, num_tensors, bytes}`.

### 1.5 `ai.aggregate` (controller; torch + numpy + safetensors only)

params (O):
| key | default | notes |
|-----|---------|-------|
| `outer_lr` | 0.7 | outer SGD lr |
| `momentum` | 0.9 | outer Nesterov momentum |
| `nesterov` | true | |
| `aggregation` | `"auto"` | `auto` = trimmed mean if >=3 accepted updates else mean; or `mean` / `trimmed_mean` |
| `trim_ratio` | 0.2 | per side, at least 1 element each side |
| `max_norm_ratio` | 5.0 | applied via the same checks as validate_update |
| `max_rel_norm` | 0.5 | absolute bound `||delta|| <= max_rel_norm x ||base||` plus the per-tensor bound and zero-update check of `ai.validate_update` (always on) |
| `max_bytes` | 1 GiB | |
| `version_id` | derived | id for the new version |
| `eval_heldout` | false | if true and `inputs.vae_weights` given: compute `heldout_loss_before/after` on a fixed synthetic batch (seed 424242, 16 clips) |

inputs: `base_weights` (R), `base_manifest` (R), `outer_state` (O; safetensors of outer momentum, absent = zeros),
`updates` (R, list of `{ "path": "<update.safetensors>", "manifest_path": "<update.json>" }`, 1..64 entries),
`vae_weights` (O, only for `eval_heldout`).
Algorithm: validate each update; reject (not fatal) with reasons
`stale_base_version`, `duplicate_update_id` (already in `base_manifest.applied_update_ids`, or repeated in this batch),
`manifest_mismatch`, `manifest_invalid`, `manifest_too_large`, `non_finite`, `norm_outlier`, `tensor_norm_outlier`,
`zero_update`, `base_norm_unavailable`, `shape_mismatch`, `size_limit`, `checksum_mismatch`.
Before averaging, every accepted update is clipped to `min(own norm, 2 x median(accepted norms))`
(ids of clipped updates: summary `clipped_update_ids`). `outer_state` is header-checked against the base shapes.
Accepted deltas -> mean (or coordinate-wise trimmed mean) `D`; outer Nesterov:
`m <- momentum*m + D; step = D + momentum*m` (or `m` if not nesterov); `new = base - outer_lr*step`.
outputs: `weights.safetensors`, `outer_state.safetensors`, `model_version.json`
(`round = base.round+1`, `parent_version_id = base.version_id`, `applied_update_ids = base ids + accepted ids`).
artifacts: `model_version`, `model_weights`, `outer_state`.
summary: `{new_version_id, applied_update_ids (this call), all_applied_update_ids, rejected: [{update_id, reason}],
aggregated_count, aggregation_method, delta_l2, heldout_loss_before?, heldout_loss_after?}`.
If **no** update is accepted: `status: ok`, `new_version_id: null`, `no_op: true`, no artifacts.

### 1.6 `ai.evaluate` (worker)

params: `mode` in `adherence` | `reconstruction` | `detector` (R);
`num_prompts` int 1..200 (16, adherence: how many benchmark prompts, taken in file order),
`seeds_per_prompt` int 1..8 (1), `steps` int 2..100 (20), `cfg_scale` float 0..20 (3.0),
`num_clips` int 1..512 (64; reconstruction/detector modes), `eval_seed` int (777),
`prompts_file` string path (O; defaults to the packaged `configs/benchmark_prompts.json`).
inputs: `weights` (R for adherence), `manifest` (R), `vae_weights` (R except detector).
outputs: `report.json`. artifacts: `evaluation_report`. summary: headline metrics
(adherence: `all_correct_rate`, `entity_recall`, ...; reconstruction: `psnr`, `motion_l1`; detector: accuracies).

### 1.7 `ai.generate` (worker)

params: `prompt` string (R, <= 200 chars, words outside the closed vocabulary are mapped to UNK
and reported in the summary), `seed` int (0), `steps` int 1..200 (30), `cfg_scale` float 0..20 (3.0),
`fps` int 1..60 (8), `upscale` int 1..8 (4; nearest-neighbour upscale of the MP4 only),
`tile` int 0..16 (0 = off; latent tile size for tiled decode), `name` string (`video`).
inputs: `weights`, `manifest`, `vae_weights` (all R).
outputs: `<name>.mp4`, `<name>.json` (sidecar). artifacts: `video`, `video_metadata`.
summary: `{frames, fps, seconds_sampling, seconds_decode, unknown_words}`.

## 2. Reference sequence for a distributed round

1. `ai.init_model` -> `model_version.json` v0 (+ `vae.safetensors`).
2. For each worker: `ai.local_train` with `base_*` = current version, a disjoint `shard`,
   unique `update_id`.
3. Controller: `ai.validate_update` (optional, `ai.aggregate` re-validates anyway).
4. `ai.aggregate` with `base_*` = current version, `outer_state` from the previous aggregate,
   `updates` = all accepted worker outputs -> new version. Persist
   `outer_state.safetensors` next to the version.
5. Updates whose `base_version_id` is not the current version are rejected as stale.
