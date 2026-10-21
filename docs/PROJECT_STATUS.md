# Project Status

Last updated: 2026-10-21 (end of the first implementation session).
Status values: **Verified complete** · **Implemented, not verified** · **Experimental** · **Pending** · **Blocked**.
Evidence: [TEST_RESULTS.md](TEST_RESULTS.md), [PERFORMANCE.md](PERFORMANCE.md),
[research/EXPERIMENTS.md](research/EXPERIMENTS.md), [research/DISTRIBUTED_RUN_001.md](research/DISTRIBUTED_RUN_001.md),
[DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md).

## Platform

| Feature | Status | Evidence / notes |
|---------|--------|------------------|
| Single `nivra` binary (controller, worker, doctor, repair, admin, generate, loadtest) | Verified | builds for windows/amd64, linux/arm64 (deployed), linux/amd64, darwin/arm64 |
| PostgreSQL schema + checksummed migrations (1–6) | Verified | applied incrementally to the live Pi database |
| Admin accounts (argon2id), cookie sessions + CSRF, roles | Verified | e2e + token tests |
| Personal API tokens (ComfyUI, scripts) | Verified | `TestAPITokens`, real ComfyUI use |
| Project enrollment codes (display, copy, rotate, close) | Verified | enrollment tests, Studio |
| Device enrollment (Ed25519), signed sessions, replay/skew protection, revocation | Verified | enrollment tests, e2e, fault test |
| Durable scheduler (leases, fencing, idempotent results, reassignment, reconciliation) | Verified | scheduler tests, 50/100-worker load tests (0 double-accepts) |
| Reliable commands (pause/resume/stop task/disconnect) with confirmation states | Verified | e2e test, Studio |
| Resumable checksummed artifacts, per-attempt upload budgets, orphan purge | Verified | artifact tests |
| Verified contributions + opt-in public leaderboard | Verified (contributions) / Implemented (leaderboard UI not built) | |
| Telemetry, searchable logs, retention, SSE live dashboard | Verified | Studio tests by the web worker |
| Security review findings F1–F13 | Fixed in code with tests; public exposure still needs owner steps | [security/THREAT_MODEL.md](security/THREAT_MODEL.md) |
| Nivra Studio (all pages) | Verified (most flows) | training charts with real data were exercised via API, not visually |
| Worker website | Verified | |
| Installers `install.sh` (controller verified on Pi), `install.bat`/`install.ps1` | Implemented; Windows installer not run end-to-end | |
| `nivra doctor` / `repair` | Implemented, partially verified | |
| Raspberry Pi deployment | Verified | controller + worker running there during tests |
| Cloudflare public hostnames | **Blocked** on owner (dashboard login + consent) | guide in `deploy/cloudflare/README.md` |
| Autostart on boot | Not configured (by design) | opt-in units in `deploy/systemd/` |
| 50 / 100 worker control-plane load | Verified | PERFORMANCE.md |

## Using the model (generation)

| Feature | Status |
|---------|--------|
| Generate in Studio → executed by a connected worker → MP4 back | Verified (API path; Studio page built) |
| `nivra generate --url` from another computer | Verified |
| `nivra generate --bundle` locally | Verified |
| Import a trained bundle; default model selection | Verified |
| Several users at once (ComfyUI + CLI) spread over workers | Verified (4 simultaneous) |
| ComfyUI local nodes (load, T2V, I2V, continuation, export) | Verified for T2V and continuation; I2V not run in ComfyUI |
| ComfyUI remote network node | Verified |

## AI model

| Feature | Status | Notes |
|---------|--------|-------|
| Procedural bilingual dataset, deterministic across platforms | Verified | byte-identical shards on Windows x86/torch 2.14 and Pi aarch64/torch 2.6 |
| Nivra tokenizer (3D-conv VAE, 471k params) | Verified | PSNR 29.48 dB on held-out clips |
| Nivra Small generator (2.41 M params, rectified flow, factorised attention) | Trained (CPU, 6000 steps) | held-out flow loss 0.297 |
| Checkpoints: safetensors + JSON, atomic, exact resume | Verified | resumed a real interrupted run at step 2750 |
| Prompt adherence | **Weak** | entity recall 0.19 trained vs 0.00 untrained; 0 of 24 benchmark videos fully correct; Swedish worse than English |
| Video continuation (GenerationState) | Implemented, verified mechanically | 28-frame continuation produced; quality not measured separately |
| Distributed DiLoCo training (2 physical workers) | Verified | held-out loss 2.04 → 1.46 over 3 rounds from random init; DiLoCo did **not** improve the converged model |
| 720p / 240-frame / 20-min-on-RTX-2060 targets | Pending | research goal; no GPU worker connected yet |
| Nivra Medium / Large | Pending | |
| Audio, speech, lip-sync | Pending | not started |

## Owner actions required
1. Change the Pi account password (it appeared in chat) and the Nivra admin password
   (`bin/nivra admin passwd --username admin`).
2. Rotate the project code before public use (it was printed in a deployment log).
3. Decide whether to publish via Cloudflare (`deploy/cloudflare/README.md`) and whether to enable autostart.
4. Connect the RTX 2060 PC as a worker to start GPU-scale training.
