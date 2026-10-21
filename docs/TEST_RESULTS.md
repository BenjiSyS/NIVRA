# Test Results

Recorded runs with exact commands. Dates: 2026-10-21. Source revisions are git commits in this repository.

## Go test suite (controller, worker, protocol)

Command (Windows 11, Go 1.27.3, PostgreSQL 18.6 local test database):

```bash
export NIVRA_TEST_DATABASE_URL=postgres://nivra:***@127.0.0.1:15432/nivra_test?sslmode=disable
go test -p 1 -count=1 ./...
```

Result at commit `0c7f031`: all packages pass.

| Package | What it covers | Result |
|---------|----------------|--------|
| `internal/controller/scheduler` | lease/ack/progress/complete; duplicate submission credited once; wrong result rejected and requeued (not re-leased to the same device); expired lease fenced and reassigned with a higher token; forged attempt rejected; heartbeat reconciliation releases lost attempts; 12 workers × 30 jobs concurrent leasing with no double assignment; stop requeues without burning an attempt; cancel fences late results | PASS |
| `internal/controller/enrollment` | code format/entropy/normalisation; AES-GCM code bound to project; enrollment with controller signature; session issuance + controller ack signature; replayed session request refused; ±5 min clock skew refused; forged device signature refused; new session revokes previous; code rotation (old code refused, enrolled device unaffected); revoked device refused | PASS |
| `internal/controller/artifacts` | resumable upload with stale-offset rejection and authoritative offset; foreign device refused; incomplete finish refused; content verified; deduplication; wrong checksum rejected; oversize chunk refused | PASS |
| `internal/shared/config` | HTTPS enforced for public controller URLs; loopback-only worker UI; strict config parsing; secret env file | PASS |
| `tests/integration` (e2e) | Part 34 acceptance flow in one process over real HTTP: controller start, admin, project, code display, worker enrollment, worker online, 3 verified jobs + contributions, pause command confirmed and no leasing while paused, resume, stop_task confirmed only after the task stopped and job requeued, disconnect/reconnect, controller restart on same port (code and contributions persist, worker reconnects automatically), revocation | PASS |
| `tests/integration` (fault injection) | controller outage while a task runs (result delivered from the worker outbox after restart); revocation during an active task (job requeued, no credit, worker shows revoked) | PASS (after fixing a worker bug where the revoked state was overwritten) |

## Physical deployment tests (Raspberry Pi 5 controller + Windows laptop worker)

| Test | Result |
|------|--------|
| `install.sh --role controller --yes` on the Pi (postgres:17 container on 127.0.0.1:5433, venv reusing system torch 2.6, migrations 1–3, project created) | PASS |
| Laptop worker enrolled with the Pi through the LAN (SSH tunnel), AI executor detected | PASS |
| 10 × 2M-iteration SHA-256 jobs executed on the laptop, recomputed and accepted by the Pi, 20 credits recorded | PASS |
| Controller graceful stop/restart (`SIGTERM`, "controller stopped cleanly") with worker auto-reconnect | PASS |
| 50 simulated workers, 5 min (see PERFORMANCE.md): 0 errors, 5666 verified jobs, 277 duplicates detected, DB invariants hold | PASS (second run, after fixing session rate limiting) |
| Studio served by the Pi controller (`GET / → 200`) | PASS |

## Web UIs

`cd web && npm run build` (strict `tsc`): PASS. Manually exercised against a real local controller,
worker and PostgreSQL: login, project code copy/rotate, live worker table with pause/resume/disconnect
showing requested → confirmed, rename, revoke, queue creation/cancel, contributions, VAE upload and run
creation, logs live tail; Worker site connect (bad code → friendly error), stop task, settings, change
project. Not exercised: video playback and training charts with real data (no AI runtime in that test).

## ComfyUI

ComfyUI commit `7a28332` (2026-10-20), Python 3.14, torch 2.14.1+cpu, started with `--cpu`.
`custom_nodes/nivra_comfyui` loaded (import time 0.0 s, no errors). Workflows submitted through the
ComfyUI HTTP API:

| Workflow | Bundle | Result |
|----------|--------|--------|
| `nivra_text_to_video_api.json` | random-init `nivra_small` (plumbing test) | success; MP4 16 frames, 256×256, 8 fps |
| `nivra_continuation_api.json` | random-init `nivra_small` | success; MP4 28 frames (16 + 12 new) |

Trained-bundle results: see below when recorded.

## AI package

See `docs/research/EXPERIMENTS.md` (pytest summary, training runs, benchmarks).

## Using the model across computers (generation, not training)

Pi controller (`nivra-pg`, migrations 1–6), workers `benji-laptop` (Windows) and `benjipi-worker` (Pi).
Trained bundle `runs/release/nivra-small-6000` imported through the admin API (same calls as Studio
→ Models → Import bundle) and set as the default model.

| Test | Result |
|------|--------|
| `nivra generate --bundle runs/release/nivra-small-6000 ...` (local, no controller) | PASS — 16 frames 256×256, 2.8 s sampling on the laptop CPU |
| `nivra generate --url <controller> --user admin ...` from the laptop | PASS — job ran on `benjipi-worker` (model downloaded automatically), MP4 downloaded |
| ComfyUI `Nivra Remote Text-to-Video (network)` with `NIVRA_TOKEN` (ComfyUI `7a28332`) | PASS — frames returned into ComfyUI, MP4 exported |
| 4 simultaneous users (1 ComfyUI + 3 CLI with a personal API token) | PASS — requests spread over both workers in parallel (Pi took the ComfyUI job, laptop the three CLI jobs back to back), all 4 MP4s valid (16 frames), CLI users finished after 13 s, 23 s, 34 s |
| API tokens: operator token works without CSRF, viewer token forbidden to create jobs, revoked token refused (`TestAPITokens`) | PASS |
| Swedish prompt with non-ASCII letters end-to-end (`unknown_words: []`) | PASS (after rejecting invalid UTF-8 bodies) |

Quality note: the trained baseline produces the requested colours but blurry shapes and weak motion
(see `docs/research/EXPERIMENTS.md` for the measured prompt-adherence scores).
