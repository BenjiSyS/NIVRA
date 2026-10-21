# Nivra HTTP API (v1)

Base path: `/api/v1`. All bodies are JSON (UTF-8). Errors always look like:

```json
{ "error": { "code": "not_found", "message": "human readable" } }
```

Times are RFC 3339 UTC. IDs are UUIDs unless noted.

There are three audiences:

| Audience | Authentication | Routes |
|----------|----------------|--------|
| Public | none | `/health`, `/public/...` |
| Workers | Ed25519 device key → short-lived bearer session | `/enrollment/*`, `/worker/*`, `/artifacts/*` |
| Administrators (Nivra Studio) | username + password → `nivra_session` cookie (HttpOnly, SameSite=Strict) + `X-Nivra-CSRF` header on every non-GET request | `/auth/*`, `/projects/*`, `/system`, `/users`, `/audit` |

Roles: `viewer` (read) < `operator` (control workers, jobs, training) < `admin` (projects, codes, revocation, users).

## Deviations from the specification's suggested contract

* Administrator resources are **project-scoped**: `/api/v1/projects/{pid}/workers` instead of `/api/v1/workers`, etc.
* Worker job routes are keyed by **attempt id** (`/worker/jobs/{attempt}/...`), not job id, because fencing applies per attempt.
* Enrollment returns no long-lived credential: the device keeps its own private key and exchanges signatures for 60-minute sessions (`POST /worker/session`).

---

## 1. Public

### `GET /health`
`200 {"status":"ok","service":"nivra-controller","protocol_version":1,"time":...}` or `503` with `"status":"degraded"` when the database is unreachable.

### `GET /public/projects/{pid}/leaderboard`
Verified contributions of devices that opted in (`public_profile`). No device ids are exposed.

---

## 2. Worker protocol (`nivra.worker/1`)

Every worker `POST` body is an envelope; the server checks `protocol_version`, `message_type`, size (≤256 KiB) and that `device_id`/`project_id` (if present) match the session:

```json
{ "protocol_version": 1, "message_id": "hex", "message_type": "heartbeat",
  "project_id": "...", "device_id": "...", "timestamp_ms": 1760000000000, "payload": { } }
```

`GET` requests send `X-Nivra-Protocol: 1`. Authenticated routes send `Authorization: Bearer <session token>`.

| Route | message_type | Payload → Response |
|-------|--------------|--------------------|
| `POST /enrollment/begin` | `enroll.begin` | `{code, public_key(b64 Ed25519), device{name, contributor_name, os, arch, agent_version}}` → `{enrollment_id, nonce, project_id, project_name, controller_key, controller_signature, expires_at}` |
| `POST /enrollment/complete` | `enroll.complete` | `{enrollment_id, signature}` (device signs `nivra-enroll-proof-v1|id|nonce|project`) → `{device_id, project_id, project_name, device_name}` |
| `POST /worker/session` | `session.open` | `{device_id, timestamp_ms, nonce, signature}` (signs `nivra-session-v1|device|ts|nonce`) → `{session_id, token, expires_at, heartbeat_interval_sec, controller_signature, ...}` |
| `POST /worker/heartbeat` | `heartbeat` | `{state, attempts[{attempt_id, fencing_token, progress, stage}], metrics{...}, capabilities?, agent_version}` → `{server_time, next_heartbeat_sec, pending_commands, paused, invalid_attempts[], session_expires}` |
| `GET /worker/me` | – | → `{device_id, device_name, project_id, project_name, paused, contributions[]}` |
| `GET /worker/jobs/next?wait=25` | – | long poll ≤30 s → `200 JobAssignment` or `204` |
| `POST /worker/jobs/{attempt}/ack` | `job.ack` | `{fencing_token}` |
| `POST /worker/jobs/{attempt}/progress` | `job.progress` | `{fencing_token, state: running|uploading, progress 0..1, stage}` |
| `POST /worker/jobs/{attempt}/complete` | `job.complete` | `{fencing_token, result(json ≤32 KiB), artifact_id?, artifacts{kind: id}?, checksum(sha256 of result), duration_ms}` → `{status: accepted|rejected|validating, reason, duplicate}` |
| `POST /worker/jobs/{attempt}/fail` | `job.fail` | `{fencing_token, code, message, retryable}` |
| `GET /worker/commands` | – | → `{commands:[{id, type, payload, state, created_at, expires_at}]}` (marks queued → delivered) |
| `POST /worker/commands/{id}/ack` | `command.ack` | `{state: acknowledged|executing|completed|failed, result?}` |
| `POST /worker/logs` | `log.batch` | `{lines:[{time_ms, level, service, message}]}` (≤500) |
| `POST /artifacts/uploads` | `upload.begin` | `{kind, size, sha256, attempt_id, fencing_token}` — only for the device's current attempt; at most 4 uploads in flight per device and 16 artifacts per attempt; the media type is assigned by the server from `kind` → `{upload_id, offset, size, chunk_size, state, artifact_id?}` (identical content is deduplicated: `state: complete` immediately) |
| `PUT /artifacts/uploads/{id}?offset=N` | raw bytes ≤8 MiB | → status; `409 offset_mismatch` carries the authoritative offset |
| `GET /artifacts/uploads/{id}` | – | → current offset (resume) |
| `POST /artifacts/uploads/{id}/complete` | `{}` | verifies size + sha256 → `{state: complete, artifact_id}` |
| `GET /artifacts/{id}` | – | file with `Range` support, `X-Nivra-SHA256` header |

Status codes that matter to workers: `401 unauthorized` (open a new session), `403 device_revoked` (stop), `409 stale_lease` (discard the attempt; never retry), `429 rate_limited` (back off).

---

## 3. Administrator API

### Authentication
* `POST /auth/login` `{username, password}` → sets cookie; `{user, csrf_token, expires_at}`. 10 attempts/min per IP and per username.
* `GET /auth/me` → `{user, csrf_token, expires_at}` (use after a page reload).
* `POST /auth/logout`.

### Controller
* `GET /system` → version, uptime, host CPU/RAM/disk, database health and pool, Cloudflare tunnel status, SSE clients, artifact storage, controller identity fingerprint.
* `GET /users` (admin), `GET /audit` (admin, last 300 audit events).

### Projects and enrollment codes
* `GET /projects` → `{projects:[{id, name, status, active_model_version, code_rotation, code_active}]}`
* `POST /projects` (admin) `{name}` → `{project, code}`
* `GET /projects/{pid}/code` (operator) → `{project_id, code: "NV-XXXX-XXXX-XXXX-XXXX", rotation_version, created_at}`
* `POST /projects/{pid}/code/rotate` (admin) → new code (enrolled devices keep working)
* `POST /projects/{pid}/code/close` (admin) → enrollment closed

### Overview and live events
* `GET /projects/{pid}/overview` → `{project, system, workers{online,total,active_gpus}, queue{status:count}, contributions{accepted_jobs,credit}, training, latest_checkpoint}`
* `GET /projects/{pid}/events` — Server-Sent Events. Event types: `hello`, `worker.updated` (full worker view), `worker.metrics` (`{id, metrics, last_seen_at}`), `job.updated` (job view), `command.updated`, `result.finalized`, `training.updated`, `log` (warn/error), `resync` (client must refetch; sent before the server drops a slow client). Keep-alive comments every 15 s.

### Workers
* `GET /projects/{pid}/workers?limit=&offset=` → `{workers:[WorkerView], total}`

  `WorkerView`: `{id, name, contributor_name, os, arch, agent_version, state (idle|busy|degraded|disconnected|suspended|revoked), reported_state, paused, public_profile, last_seen_at, created_at, clock_skew_ms, cpu{model,cores,threads}, gpus[{index,name,vram_mb,...}], ram_mb, vram_mb, ram_budget_mb, vram_budget_mb, executors[], current_task{job_id, attempt_id, job_type, state, progress, stage}|null, metrics{cpu_pct, ram_used_mb, gpu_pct, vram_used_mb, net_rx_kbps, net_tx_kbps}|null, accepted_jobs, credit}`
* `GET /projects/{pid}/workers/{wid}`
* `GET /projects/{pid}/workers/{wid}/metrics?minutes=60` → `{points:[{t,cpu,ram_mb,gpu,vram_mb}]}`
* `POST /projects/{pid}/workers/{wid}/commands` (operator) `{type: pause|resume|stop_task|disconnect|refresh_capabilities}` → `202 CommandView`. `pause`/`resume` also flip the authoritative scheduler flag immediately.
* `GET /projects/{pid}/workers/{wid}/commands`, `GET /projects/{pid}/commands` → `{commands:[{id, device_id, type, state, payload, result, created_at, delivered_at, acknowledged_at, finished_at, expires_at}]}`. The UI shows "requested" (queued/delivered/acknowledged/executing) distinctly from "confirmed" (completed).
* `POST /projects/{pid}/workers/{wid}/rename` (operator) `{name}`
* `POST /projects/{pid}/workers/{wid}/trust` (admin) `{trusted}` — only trusted devices receive distributed model-training jobs (unless a run sets `allow_untrusted`)
* `POST /projects/{pid}/workers/{wid}/revoke` (admin)

### Jobs / queue
* `GET /projects/{pid}/jobs?status=&type=&run=&before=&limit=` → `{jobs:[JobView], counts{status:n}, job_types[]}`

  `JobView`: `{id, job_type, category, status, priority, attempt_count, max_attempts, training_run_id, params, failure_reason, created_at, updated_at, completed_at, current_attempt{id, device_id, device_name, attempt_number, fencing_token, state, progress, stage, lease_expires_at, leased_at, started_at, completed_at, error}|null, result?{validation_status, rejection_reason, payload, artifact_id, submitted_at}}`
* `POST /projects/{pid}/jobs` (operator) `{job_type, params, count?, priority?, max_attempts?, lease_seconds?, req_ram_mb?, req_vram_mb?, req_gpu?, req_os?, idempotency_key?}` → `{job_ids}`
* `GET /projects/{pid}/jobs/{jid}`, `POST .../cancel`, `POST .../retry`

Job statuses: `ready → leased → running → uploading → validating → completed | failed | cancelled` (rejected results return the job to `ready` while attempts remain).

### Contributions and logs
* `GET /projects/{pid}/contributions` → `{by_worker:[{device_id, name, contributor_name, category, jobs, seconds, units, credit}], recent:[...]}` — credit is computed by the controller from verified work only.
* `GET /projects/{pid}/logs?device=&service=&run=&level=info&q=&since=&until=&before_id=&limit=&format=ndjson` → `{logs:[{id, ts, source, device_id, service, level, message, fields}]}`

### Training, models, checkpoints, generation
* `GET /projects/{pid}/training/runs` → `{runs:[Run]}`; `POST` (operator) `{name, vae_artifact, config: RunConfig}`
* `GET /projects/{pid}/training/runs/{rid}`; `POST .../pause|resume|stop|checkpoint`; `POST .../load {checkpoint_id}` (run must be paused, stopped or completed)
* `GET /projects/{pid}/training/runs/{rid}/metrics` → `{series:{name:[{round,step,ts,value}]}}` (names: `heldout_loss_before`, `heldout_loss_after`, `delta_l2`, `accepted_updates`, `rejected_updates`, `update_loss_first_mean`, `update_loss_last_mean`, `round_seconds`, `active_workers`, `outer_lr`)
* `GET /projects/{pid}/training/runs/{rid}/updates` → accepted/applied/invalid/stale updates with worker and metadata
* `GET /projects/{pid}/checkpoints?run=`, `GET /projects/{pid}/models`
* `GET /projects/{pid}/artifacts?kind=`; `POST /projects/{pid}/artifacts?kind=vae_weights` (operator; raw body upload); `GET /projects/{pid}/artifacts/{aid}` (download, Range supported)
* `POST /projects/{pid}/generate` (operator) `{prompt, seed, steps, cfg_scale, fps, upscale, model_version?}` → `{job_id}`; executed by a worker with the AI executor
* `GET /projects/{pid}/generations` → `{generations:[{job: JobView, artifacts:{video, video_metadata}}]}`
* `POST /projects/{pid}/evaluate` `{mode: adherence|reconstruction, num_prompts?, steps?, model_version?}`
* `POST /projects/{pid}/datasets/shards` `{seed, start, count, jobs, frames?, size?}`

`RunConfig`: `{allow_untrusted (default false), max_heldout_increase (default 0.10), init_weights_artifact?, init_manifest_artifact?, architecture: "nivra_tiny", seed, rounds, workers_per_round, min_updates, round_timeout_sec, shard_size, data_seed, outer_lr, momentum, eval_heldout, inner:{inner_steps, batch_size, lr, weight_decay, grad_clip, eval_clips}, req_ram_mb, lease_seconds}`. Run statuses: `created, initializing, running, paused, stopping, stopped, completed, failed`.

---

## 4. Worker local API (loopback only, `127.0.0.1:8741`)

Used by the Worker website. Non-GET requests require header `X-Nivra-Local: 1`; the `Host` must be a loopback name.

* `GET /local/api/status` → `{state, detail, enrolled, running, local_paused, remote_paused, controller_url, project_id, project_name, device_id, device_name, controller_fingerprint, last_heartbeat, task{attempt_id, job_id, job_type, phase, stage, progress, started_at, elapsed_sec, upload, stopping}|null, hardware{host{os, arch, cpu_model, cores, threads, mem_total, mem_available, cpu_pct, disk_total, disk_free}, gpus[], ai{ok, error, python, torch, cuda, cuda_device, cuda_vram_mb, job_types[]}}, settings, executors[], contributions[], completed_this_session, failed_this_session, version}`

  Agent states: `not_enrolled, stopped, connecting, idle, busy, paused, outside_schedule, offline, revoked`.
* `GET /local/api/logs?after=SEQ` → `{logs:[{seq, time, level, service, message}]}`
* `POST /local/api/connect` `{controller_url, code, device_name?, contributor_name?}`
* `POST /local/api/start | pause | resume | stop | stop-task | hardware/refresh`
* `PUT /local/api/settings` `{cpu_threads, gpu_index (-1 = none), ram_budget_mb, vram_budget_mb, disk_cache_mb, bandwidth_kbps, schedule{enabled, start "HH:MM", end "HH:MM"}, auto_start, contributor_name}`
