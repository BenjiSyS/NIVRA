# Nivra Executor Protocol (`nivra.exec/1`)

The Go worker agent never imports Python. AI work runs in an isolated
subprocess (`python -m nivra.cli exec --job <job.json>`) that speaks this
protocol. The Controller uses the same protocol for controller-side tasks
(update validation, aggregation).

## 1. Invocation

```
<python> -m nivra.cli exec --job <workdir>/job.json
```

* Working directory: `<workdir>` (created and owned by the caller, one per attempt).
* The caller may set `NIVRA_THREADS` to bound CPU threads and
  `NIVRA_DEVICE` (`cpu`, `cuda`, `cuda:N`) to pin a device.
* The child must not read or write outside `<workdir>`, the paths listed in
  `inputs`, and its own package/cache directory.

## 2. Job file

```json
{
  "protocol": "nivra.exec/1",
  "job_type": "ai.local_train",
  "job_id": "uuid",
  "attempt_id": "uuid",
  "params": { "...": "job-type specific, validated by the executor" },
  "inputs": { "base_weights": "C:/.../base.safetensors" },
  "output_dir": "<workdir>/out",
  "stop_file": "<workdir>/STOP",
  "limits": { "max_seconds": 3600, "max_ram_mb": 4096, "max_vram_mb": 0 }
}
```

Unknown `protocol` values must be rejected (exit code 2).

## 3. Events (stdout, JSON Lines)

One JSON object per line, UTF-8. Anything that is not valid JSON is treated
as a log line. Lines longer than 64 KiB are truncated by the caller.

| `event`    | Fields |
|------------|--------|
| `log`      | `level` (`debug`/`info`/`warn`/`error`), `message` |
| `progress` | `stage`, `fraction` (0..1), optional `step`, `total`, `message` |
| `metric`   | `name`, `value` (number), optional `step` |
| `artifact` | `kind`, `path` (inside `output_dir`), `sha256`, `bytes` |
| `result`   | `status` (`ok`), `summary` (object, <= 32 KiB) |
| `error`    | `code` (`INVALID_SPEC`, `OOM`, `STOPPED`, `INPUT_INVALID`, `INTERNAL`), `message` |

Exactly one `result` or `error` event is emitted last.

## 4. Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success (`result` emitted) |
| 1 | Internal error |
| 2 | Invalid job specification / unsupported protocol |
| 3 | Stopped gracefully after the stop file appeared |
| 4 | Out of memory (CPU or GPU) |
| 5 | Input validation failed (e.g. corrupt or incompatible weights) |

## 5. Stopping

Signals are not portable to Windows, so stopping is file based:

1. The caller creates `stop_file`.
2. The executor checks for it at every safe boundary (between optimizer
   steps, between sampling steps, between samples) and exits with code 3
   after writing any checkpoint that the job type promises.
3. If the child is still alive after the grace period (default 30 s) the
   caller kills the whole process tree (Windows job object / POSIX process
   group).

## 6. Job types

| `job_type` | Runs on | Purpose | Contribution category |
|------------|---------|---------|----------------------|
| `ai.dataset_shard` | worker | Generate a deterministic procedural video shard | preprocessing |
| `ai.local_train` | worker | DiLoCo-style inner loop: H local steps from a base global version, output a pseudo-gradient | training |
| `ai.evaluate` | worker | Prompt-adherence / reconstruction evaluation of a checkpoint | evaluation |
| `ai.generate` | worker | Text-to-video inference, MP4 output | (not credited) |
| `ai.validate_update` | controller | Structural + numerical validation of a submitted update | n/a |
| `ai.aggregate` | controller | Combine validated updates into a new global version | n/a |

Job-type parameter schemas are documented in `ai/nivra/cli.py` and
`docs/training/DISTRIBUTED.md`.
