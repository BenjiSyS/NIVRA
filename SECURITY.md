# Security Policy

## Reporting a vulnerability

Please report security issues privately to the project owner instead of opening a public issue.
Include affected version (`nivra version`), steps to reproduce and impact. You will get an
acknowledgement and a fix or mitigation plan; please allow time for a release before disclosure.

## Supported versions

Only the latest release receives security fixes during the 0.x research phase.

## Design summary

* Controller listens on loopback; public access only through Cloudflare Tunnel (TLS).
* Administrators: argon2id passwords, HttpOnly SameSite=Strict session cookies, CSRF header, roles.
* Workers: per-device Ed25519 keys, signed short-lived sessions with replay protection, revocation;
  project codes only authorise enrollment.
* Worker results are untrusted: fencing tokens, idempotent submission, server-side recomputation
  where possible, numerical validation of training updates, no credit for unverifiable work.
* No remote shell or arbitrary code execution path exists from controller to worker.
* Secrets live in `<data_dir>/secrets/` with owner-only permissions and are never committed.

Full analysis: [docs/security/THREAT_MODEL.md](docs/security/THREAT_MODEL.md).
