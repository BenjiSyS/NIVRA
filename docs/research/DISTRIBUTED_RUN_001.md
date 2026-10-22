# Distributed training run 001 — two physical workers, Raspberry Pi controller

Date: 2026-10-21. Controller `nivra 0.1.0` (deployed commit at the start of the run, restarted once mid-run — see below) on Raspberry Pi 5.
This is the first real multi-worker optimisation of an original Nivra model (Part 21.2 / Milestone F).

## Setup

| Item | Value |
|------|-------|
| Run | `diloco-pi-laptop-1` (id `16218f9a-5a86-4916-b611-e9a05b7f5415`) |
| Architecture | `nivra_small` (2,414,352 generator parameters), random initialisation (`seed` 11) via `ai.init_model` on the Pi |
| VAE | trained single-node VAE, sha256 `e5cd7f7c3b63…debe27` (uploaded as artifact, frozen) |
| Workers | `laptop-worker` (Windows 11, i5-8250U, torch 2.14.1+cpu) and `pi-worker` (the Pi itself, Debian 13 arm64, torch 2.6.0) |
| Algorithm | DiLoCo-style: per round each worker runs H = 20 inner AdamW steps (batch 4, lr 3e-4) from the current global version on a disjoint procedural shard (64 clips), uploads `base − local`; controller validates and applies outer Nesterov SGD (lr 0.7, momentum 0.9) on the mean of the updates |
| Rounds | 3, `workers_per_round` = `min_updates` = 2 |
| Evaluation | `heldout_loss_*`: flow-matching loss on a fixed synthetic batch (seed 424242, 16 clips), computed **by the controller** with the frozen VAE, before and after each aggregation — not reported by workers |

## Results

| Round | Version | Held-out loss before → after | Δ‖update‖ (aggregate L2) | Accepted / rejected | Round wall time |
|------:|--------:|------------------------------|-------------------------:|--------------------:|----------------:|
| 0 | 0 → 1 | 2.0403 → **1.9292** | 3.1755 | 2 / 0 | 260.7 s |
| 1 | 1 → 2 | 1.9292 → **1.8235** | 3.3170 | 2 / 0 | 260.2 s |
| 2 | 2 → 3 | 1.8235 → **1.4546** | 3.0224 | 2 / 0 | 264.4 s |

Per-update records (worker-reported training losses are unverified metadata):

| Round | Worker | Update L2 | Samples | Inner loss first → last | Job wall time |
|------:|--------|----------:|--------:|-------------------------|--------------:|
| 0 | laptop-worker | 3.1699 | 80 | 1.8957 → 1.6042 | 43 s |
| 0 | pi-worker | 3.5259 | 80 | 1.9680 → 1.6374 | 174 s |
| 1 | laptop-worker | 3.3281 | 80 | 1.9433 → 1.7686 | 22 s |
| 1 | pi-worker | 3.4261 | 80 | 1.7656 → 1.6413 | 173 s |
| 2 | laptop-worker | 3.2186 | 80 | 1.7176 → 1.2753 | 31 s |
| 2 | pi-worker | 3.0880 | 80 | 1.8098 → 1.4392 | 178 s |

Checkpoints: one per global version (v0 `init`, v1–v3 `round`, v3 `stop`), all valid with SHA-256
recorded; every update applied exactly once (`training_updates.state = applied`, update ids also
recorded in each version manifest's `applied_update_ids`).

## Fault injected during the run
During round 1 the controller was stopped (`SIGTERM`) and restarted 27 s later while the Pi worker
held a lease. The run resumed from PostgreSQL state, the worker's update was accepted and aggregated,
and rounds 2–3 completed normally. The restart revealed a shutdown stall (SSE/long-poll connections
held graceful shutdown for 15 s), fixed in the commit that added `TestShutdownWithOpenStreams`.

## Interpretation (honest)
* This demonstrates **real distributed optimisation**: two machines with different CPU
  architectures and PyTorch versions trained the same original model, and an independent
  controller-side evaluation shows the global model's held-out loss decreasing every round.
* Scale is tiny (60 inner steps total, 480 training clips); absolute losses are far above the
  single-node model (held-out ≈ 0.30 after 6000 steps), so this run does not produce a useful
  video model by itself. Its purpose was correctness of the distributed pipeline.
* Round time is dominated by the slowest worker (the Pi, ~175 s for 20 steps). A
  straggler-tolerant policy (`min_updates` < `workers_per_round` with a timeout) is implemented but
  was not exercised here.
* With two updates per round, robust aggregation falls back to the mean; trimmed-mean protection
  requires ≥ 3 updates.
