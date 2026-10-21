# Nivra Threat Model

Scope: controller on a home Raspberry Pi exposed through Cloudflare Tunnel, administrators using Nivra Studio, and public volunteer workers. Status of each control: **Implemented** (code + test), **Implemented, untested**, **Planned**.

## Assets
1. Pi host and its unrelated services (game servers, other databases, Docker containers).
2. Nivra database (accounts, devices, jobs, contributions) and `controller-secrets.json`.
3. Model weights and checkpoints (integrity matters more than secrecy — they are meant to be open).
4. Administrator accounts and sessions.
5. Contributor trust: correct contribution accounting.

## Trust boundaries
* Internet → Cloudflare edge → `cloudflared` on the Pi → `127.0.0.1:8740` (controller listens on loopback only).
* Controller → PostgreSQL on `127.0.0.1:5433` (never exposed).
* Controller → Python subprocess for validation/aggregation of **untrusted** worker uploads.
* Worker → local browser on `127.0.0.1:8741`.

## Threats and controls

| # | Threat | Control | Status |
|---|--------|---------|--------|
| T1 | Brute-forcing the project code | 80-bit codes; per-IP (10/min) and global enrollment rate limits; failed attempts audited | Implemented |
| T2 | Code leaked (posted publicly) | Code only enrolls; devices hold their own Ed25519 keys; code rotation; close enrollment; per-device revocation | Implemented + tested (rotation does not affect enrolled devices; revocation in e2e test) |
| T3 | Replay of a captured session request | Signed timestamp (±5 min) + nonce stored in `used_nonces` | Implemented (unit test pending) |
| T4 | Impersonating the controller to workers | TLS via Cloudflare; controller Ed25519 identity pinned by workers, re-verified on every session | Implemented |
| T5 | Stale or duplicate result submission | Fencing tokens, unique result per attempt, one accepted result per job | Implemented + tested |
| T6 | Fabricated compute | Builtin jobs are recomputed by the controller; dataset shards ≤256 clips are regenerated; training updates are structurally/numerically validated; evaluation/generation results earn 0 credit | Implemented |
| T7 | Poisoned training updates | Shape/dtype/finite checks, norm ratio vs peer median, base-version check, trimmed mean with ≥3 updates | Implemented (statistical defences are partial; a careful attacker within norm bounds can still bias a round) |
| T8 | Malicious artifact uploads (path traversal, huge files, wrong content) | Content-addressed storage named by server-computed SHA-256; size limits; disk reserve; per-device ownership checks; safetensors only (no pickle) | Implemented |
| T9 | Unbounded telemetry/log growth exhausting the Pi | Bounded in-memory buffers with drop counters; retention (metrics 7 d, logs 14 d, audit 365 d) | Implemented |
| T10 | Admin password guessing | argon2id (64 MiB, t=2), constant-time compare, per-IP and per-user login limits, generic error | Implemented |
| T11 | CSRF / clickjacking on Studio | SameSite=Strict HttpOnly cookie, `X-Nivra-CSRF` header on mutations, Origin check, `frame-ancestors 'none'`, CSP | Implemented |
| T12 | DNS-rebinding / CSRF against the local worker UI | Loopback listener, Host-header allow-list, `X-Nivra-Local` header requirement, Origin check | Implemented |
| T13 | Remote code execution via the coordinator | No "run shell" command type exists; job types are a fixed registry; worker updates are not pushed by the controller | Implemented by design |
| T14 | Database exposed to the internet | PostgreSQL bound to 127.0.0.1 inside Docker; Cloudflare routes only to the controller port | Implemented |
| T15 | Secrets in git / logs | `.gitignore` for secrets, data and weights; passwords read from stdin/no-echo; SSH password never used (key auth) | Implemented |
| T16 | Exposed SSH password (appeared in chat) | Deployment used SSH keys only. **Owner action required: change the Pi account password** before enabling public routes | Planned (owner) |
| T17 | Cloudflare Access misconfiguration exposing Studio | Studio has its own authentication regardless; recommended: Cloudflare Access policy on `studio.` hostname, none on `api.` (workers need machine access) | Planned (needs owner's Cloudflare login) |
| T18 | Compromised worker host stealing its device key | Key file 0600 (Windows: profile ACL); revoke device in Studio | Implemented; key encryption at rest (DPAPI/keyring) Planned |
| T19 | Denial of service by many workers | Per-device request rate limit, bounded upload concurrency, long-poll cap 30 s, DB pool cap | Implemented; measured in docs/PERFORMANCE.md |

## Residual risks (honest)
* Training-update validation is heuristic; Nivra does not claim Byzantine-robust training.
* Workers' reported losses/metrics are unverified.
* The worker website is unauthenticated on loopback: any local user/process on that computer can control the worker.
* No automatic security updates of the binary yet (updates are manual; unsigned remote scripts are never executed).

## Security review 2026-10-21 (read-only review, fixes implemented the same day)

| # | Finding | Fix | Test |
|---|---------|-----|------|
| F1 | Uploads unrelated to any job could fill the Pi's disk | Uploads require the device's current attempt + fencing token; ≤4 in flight per device, ≤16 artifacts per attempt; free-space re-check per chunk; unreferenced worker artifacts purged after 24 h (files removed only when no row still uses them) | `TestUploadBudgetsAndIsolation`, `TestOrphanPurge` |
| F2 | Padded manifests could make controller-side Python run out of memory, retried forever | JSON artifacts > 1 MiB rejected in Go before Python; update weights must match the base size ±1 MiB; validator infrastructure failures are retried at most 3 times, then rejected | `TestValidatorFailuresAreBounded` (Python-side size checks: see EXPERIMENTS/DISTRIBUTED docs) |
| F3 | Huge finite updates could destroy the model when fewer than 3 updates are present | Absolute and per-tensor norm bounds relative to the base weights, zero-update rejection and per-update clipping in `ai.validate_update`/`ai.aggregate`; independent controller gate rejects a version whose held-out loss rises > 10 % | Python tests; orchestrator gate |
| F4 | Credit derived from worker-written `samples` | Credit computed from server-issued `inner_steps × batch_size` | code review |
| F5 | Insecure cookie/proxy defaults behind HTTPS | Controller refuses to start with an `https` `public_url` unless `cookie_secure` and `trust_proxy_headers` are set; `__Host-` cookie prefix | `TestHTTPSPublicURLRequiresSecureSettings`, `TestSecureCookieUsesHostPrefix` |
| F6 | Admin API reachable on the worker hostname; per-account login lockout by strangers | `api_hosts`/`studio_hosts` separation; per-account limiter counts failures only | `TestHostSplit` |
| F7 | Worker-chosen content types could run script in the Studio origin | Server assigns media types from `kind`; only `video/mp4`/`image/png` inline; `Content-Security-Policy: sandbox` on artifacts | artifact test (content type ignored) |
| F8 | Log/heartbeat floods | Per-device limits: 20 log lines/s (burst 500), ~2 heartbeats per interval | code review |
| F9 | Unlimited devices could outvote aggregation | Per-project device cap (default 1000); model-training jobs only for admin-trusted devices by default | `TestTrustedJobsOnlyGoToTrustedDevices` |
| F10 | Uploaded shard file never compared | Shard file SHA-256 compared with the controller's recomputation | validator |
| F11 | Cross-project dedupe | Dedupe limited to the same project and uploader | `TestUploadBudgetsAndIsolation` |
| F12 | Spoofable `X-Forwarded-For`; IPv6 rotation | Only `CF-Connecting-IP` from loopback; IPv6 rate keys by /64 | `TestClientIPIgnoresForwardedFor`, `TestRateKeyGroupsIPv6Prefixes` |
| F13 | Device lockout by unauthenticated session floods | Per-device session budget charged only after signature verification | code review |

Status: the blocking findings for public exposure (F1–F5) are fixed in Go; the Python bounds for F2/F3 are
implemented in `ai/` (see the AI package tests). Re-review before enabling the public hostnames.
