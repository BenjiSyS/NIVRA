# Architecture Decisions

Each decision lists context, choice, alternatives and consequences. Newest last.

## ADR-001 Single Go binary for controller, worker and tools
* **Context**: Controller must run on Raspberry Pi 5 (linux/arm64); workers on Windows, Linux, macOS; contributors should not need toolchains.
* **Decision**: One statically linked Go binary (`nivra`) with subcommands (`controller run`, `worker run`, `doctor`, `repair`, `admin`, `project`, `job`, `migrate`, `loadtest`). `CGO_ENABLED=0`; cross-compiled from any host.
* **Alternatives**: Python/FastAPI controller (heavier runtime and dependency drift on the Pi, weaker concurrency for 50–100 long-polling workers); Rust (cross-compiling for arm64 from Windows needs a linker toolchain; slower iteration).
* **Consequences**: ~15 MB binary, no runtime dependencies for the control plane. The spec's separate `cmd/controller` and `cmd/worker` are replaced by subcommands (documented deviation).

## ADR-002 PostgreSQL as the only database
* **Decision**: PostgreSQL 13+ (Pi: dedicated `postgres:17` container `nivra-pg` on `127.0.0.1:5433`; tests: PostgreSQL 18 on Windows). Embedded, checksummed migrations applied under an advisory lock; modified applied migrations are refused.
* **Why not reuse the existing `angel-pg` container**: it belongs to another application; sharing it would couple upgrades, backups and failure domains.
* **Why not SQLite for dev**: two SQL dialects double the testing surface; a local PostgreSQL is easy to run.

## ADR-003 Durable scheduler with fencing tokens
* **Decision**: Jobs and attempts live in PostgreSQL. Leasing uses `SELECT … FOR UPDATE SKIP LOCKED`; every attempt gets a token from a global sequence; only the job's `current_attempt_id` with the matching token may commit. One result row per attempt (unique) makes submissions idempotent; a partial unique index allows only one accepted result per job. Leases expire via a reaper (advisory-locked) and are extended by heartbeats; unreported attempts are released after a 30 s grace period.
* **Verified by**: `internal/controller/scheduler/scheduler_test.go` (duplicate submission, stale fenced result, reassignment, 12 workers × 30 jobs concurrency, stop/cancel) and `tests/integration/e2e_test.go`.

## ADR-004 Device keys instead of shared secrets for workers
* **Decision**: Each worker generates an Ed25519 key. The project code (80-bit Crockford base32) authorises *enrollment only*; afterwards the device signs a timestamped nonce to obtain a 60-minute bearer session (nonce replay table). The controller signs challenge and session responses with its own Ed25519 identity, which workers pin (TOFU over TLS).
* **Consequences**: Rotating the project code never breaks enrolled devices; revocation is per device; a leaked code cannot impersonate existing devices.

## ADR-005 Recoverable project code storage
* **Context**: Spec requires the permanent code to be visible in Studio, yet discourages plaintext storage.
* **Decision**: Store `HMAC-SHA256(k1, code)` for lookup plus `AES-256-GCM(k2, code, aad=project_id)` for display to authorised users. Keys live in `<data>/secrets/controller-secrets.json` (0600), never in the database. A database dump alone reveals no codes.

## ADR-006 Real-time dashboard via Server-Sent Events
* **Decision**: One SSE stream per dashboard tab, bounded per-client queues (512 events); slow clients are dropped with a `resync` hint rather than blocking publishers. Workers use HTTPS long polling (≤30 s) woken by an in-process notifier.
* **Alternatives**: WebSockets (bidirectional not needed; SSE passes through Cloudflare Tunnel without extra config).

## ADR-007 AI work in isolated Python subprocesses (`nivra.exec/1`)
* **Decision**: The Go worker never imports Python. AI jobs run as `python -m nivra.cli exec --job job.json` with JSON-lines events, file-based stop requests (portable to Windows) and process-tree kill (Windows job objects with KILL_ON_JOB_CLOSE; POSIX process groups + PDEATHSIG). See `docs/architecture/EXECUTOR_PROTOCOL.md`.
* **Consequences**: CPU-only contributors need no Python at all (builtin jobs); crashes in PyTorch cannot take down the agent.

## ADR-008 DiLoCo-style low-communication distributed training
* **Decision**: Workers run H inner AdamW steps from the current global version on disjoint procedural shards and upload a pseudo-gradient (`base − local`). The controller validates (shapes, finiteness, norm bounds vs peers, base version, manifest) and aggregates with an outer Nesterov SGD step, producing a new versioned global model. Stale updates are rejected; update IDs are applied at most once (recorded in the version manifest and `training_updates`).
* **Why**: Workers on home connections cannot sync gradients every step; DiLoCo (Douillard et al., 2023) tolerates infrequent communication.
* **Limits**: Validation cannot prove an update improves the model; worker-reported losses are stored as unverified metadata. Robust aggregation (trimmed mean) only helps with ≥3 updates per round.

## ADR-009 Safetensors + JSON everywhere, no pickle
* **Decision**: Model weights, optimizer state, updates and checkpoints use safetensors plus JSON manifests with SHA-256 for every file; checkpoints are written to a temp directory and atomically renamed.

## ADR-010 Static web UIs embedded in the binary
* **Decision**: React + TypeScript + Vite, two entry pages (Studio, Worker) built into `internal/webui/dist` and embedded with `go:embed`. Strict CSP, no external resources.

## ADR-011 No automatic start on boot by default
* **Decision**: The installer creates the PostgreSQL container with `--restart no` and installs no systemd unit. `deploy/systemd/` contains opt-in units. The owner decides when Nivra should run permanently.
