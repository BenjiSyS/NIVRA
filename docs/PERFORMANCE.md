# Performance

All numbers below were measured; nothing is extrapolated unless marked *projected*.

## Control plane on the Raspberry Pi 5

### Setup (2026-10-21)
* Controller: `nivra 0.1.0 (0b166f7)` linux/arm64 on Raspberry Pi 5 16 GB, Debian 13, NVMe root;
  PostgreSQL 17 in Docker (`nivra-pg`). The Pi also runs unrelated services (Docker containers,
  Ollama, InfluxDB, game/web services) during the test.
* Load generator: `nivra loadtest` on the Windows laptop (i5-8250U), connected through an SSH
  tunnel to the Pi's loopback listener — i.e. every request crosses the real LAN.
* Simulated workers speak the real protocol (signed sessions, enveloped messages, heartbeats every
  15 s, long-poll `jobs/next`, ack, complete). They really compute the SHA-256 chains, and the
  **controller recomputes every result before accepting it**. 5 % of results are deliberately
  submitted twice; 2 % of heartbeats force a reconnect (new signed session).
* One real worker (the laptop's `nivra worker`) ran concurrently.
* Workload: 6000 × `builtin.sha256_chain` (50 000 iterations each), 3 concurrent dashboard readers
  polling `/workers`, `/jobs`, `/overview` every 2 s.

### 50 simulated workers, 5 minutes — PASS

| operation | ok | errors | p50 | p95 | p99 | max |
|-----------|---:|---:|---:|---:|---:|---:|
| heartbeat | 994 | 0 | 17.3 ms | 371.9 ms | 566.1 ms | 2.70 s |
| jobs.next (leased) | 5666 | 0 | 151.1 ms | 311.1 ms | 615.7 ms | 5.49 s |
| jobs.ack | 5666 | 0 | 137.1 ms | 287.6 ms | 484.4 ms | 1.98 s |
| jobs.complete (incl. server recomputation) | 5666 | 0 | 203.4 ms | 416.4 ms | 615.9 ms | 1.95 s |
| jobs.complete duplicate | 277 | 0 | 130.3 ms | 282.6 ms | 433.0 ms | 0.60 s |
| session (re)open | 15 | 0 | 12.2 ms | 237.2 ms | 237.2 ms | 0.34 s |
| dashboard /workers | 375 | 0 | 30.0 ms | 391.6 ms | 498.9 ms | 0.66 s |
| dashboard /jobs | 375 | 0 | 155.0 ms | 551.2 ms | 669.0 ms | 0.79 s |
| dashboard /overview | 375 | 0 | 21.3 ms | 487.9 ms | 785.5 ms | 1.04 s |

* Throughput: 5666 verified jobs in 5 min ≈ **18.9 jobs/s** end-to-end (lease → compute → upload →
  server recomputation → accept → credit), CPU-bound on the Pi's verification work.
* All 277 duplicate submissions were answered as duplicates; none were credited.
* Controller RSS after the run: **33 MB**; PostgreSQL database: 26 MB; Pi load average 2.4–4.6.

Post-run database invariants (checked with SQL on the Pi):

| check | result |
|-------|--------|
| accepted results = contributions = completed jobs | 7010 = 7010 = 7010 |
| jobs with more than one accepted result | 0 |
| completed jobs without a contribution | 0 |
| leases left active | 0 |

### First attempt — failed, bug found and fixed
The first 50-worker run (before commit `6260d84`) produced 444 heartbeat and 3106 polling errors:
session opening shared the per-IP enrollment rate limit (10/min), so workers behind one address
(here the tunnel; in reality NAT) could not open sessions. Fixed with a per-device session limit
(12/min) and a separate NAT-friendly per-IP limit; the test above is after the fix.

### 100 simulated workers (stretch), 5 minutes — PASS with 6 transient errors

Same setup, 10 000 jobs queued, 5 dashboard readers, controller `0b166f7`.

| operation | ok | errors | p50 | p95 | p99 | max |
|-----------|---:|---:|---:|---:|---:|---:|
| heartbeat | 1969 | 6 | 88.0 ms | 574.4 ms | 1.28 s | 2.56 s |
| jobs.next (leased) | 9775 | 0 | 220.6 ms | 400.4 ms | 860.6 ms | 5.65 s |
| jobs.ack | 9775 | 0 | 210.0 ms | 375.1 ms | 683.4 ms | 1.46 s |
| jobs.complete | 9775 | 0 | 305.0 ms | 508.2 ms | 861.8 ms | 1.63 s |
| jobs.complete duplicate | 507 | 0 | 197.8 ms | 349.1 ms | 552.2 ms | 1.45 s |
| session (re)open | 31 | 0 | 62.8 ms | 283.5 ms | 299.0 ms | 0.33 s |
| dashboard /workers | 566 | 0 | 95.9 ms | 576.8 ms | 845.3 ms | 2.28 s |
| dashboard /jobs | 566 | 0 | 185.4 ms | 713.0 ms | 925.7 ms | 2.65 s |
| dashboard /overview | 566 | 0 | 87.3 ms | 924.2 ms | 1.02 s | 1.78 s |

* Throughput ≈ **32.6 verified jobs/s**; controller RSS 37 MB.
* Invariants after the run: accepted = contributions = completed = 17 010; 0 double-accepted jobs;
  0 leaked leases.
* Errors: 5 × HTTP 401 and 1 timeout on heartbeats (0.3 %). Root cause: the simulated forced
  reconnects raced with concurrent requests that shared one client; each concurrent 401 opened a new
  session, which revoked the previous one. The same race could hit a real worker (heartbeat, job poll
  and uploads run concurrently). Fixed afterwards with single-flight session refresh in
  `internal/worker/client` plus a regression test (`TestConcurrent401sRefreshSessionOnce`, fails
  before the fix, passes after). The 100-worker run has not been repeated after the fix.

Conclusion: the Pi sustains 50 and 100 concurrently connected workers with bounded memory, no lost
or double-counted results and sub-second p95 latencies on all control operations; worker lists in
the dashboard refresh well within the 1–2 s target at p95 (0.58 s at 100 workers).

## Caveats
* Simulated workers prove the control plane, not distributed GPU training.
* The controller was reached through an SSH tunnel on the LAN, not through Cloudflare; public
  latency will add the tunnel round trip.
* Dashboard readers poll HTTP endpoints; the Studio UI normally receives SSE updates instead.

## Model training and inference
CPU training and inference measurements: [research/EXPERIMENTS.md](research/EXPERIMENTS.md).
