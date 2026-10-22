# Deployment Status

Last updated: 2026-10-21.

## Raspberry Pi controller (`<pi-hostname>`, 192.0.2.10)

| Item | State |
|------|-------|
| Reachable | Yes. SSH key authentication with strict host-key checking; no password was used |
| OS / arch | Debian GNU/Linux 13 (trixie), kernel 6.18.50-rpt, aarch64, 16 GB RAM, 469 GB NVMe (320 GB free at install) |
| Install path | `/home/<user>/Desktop/Nivra` (release package of tracked files + `bin/nivra` linux/arm64) |
| Installed version | see `/home/<user>/Desktop/Nivra/DEPLOYED_VERSION` (`0.1.0`, commit recorded at deploy time) |
| Runtime data | `/home/<user>/NivraData/{controller,worker,postgres,venv,releases,backups,run}` |
| Database | Docker container `nivra-pg`, image `postgres:17`, bound to `127.0.0.1:5433` only, data in `/home/<user>/NivraData/postgres`, restart policy `no` (does not start at boot). Password in `controller/secrets/database.env` (0600) |
| Existing PostgreSQL `angel-pg` (127.0.0.1:5432) | Untouched. Note: the Docker daemon restarted at 15:01 CEST on 2026-10-21 (journal), before any Nivra change on the Pi, which restarted `angel-pg` too; not caused by Nivra |
| Controller config | `/home/<user>/Desktop/Nivra/config/controller.json`: listen `127.0.0.1:8740`, Python `/home/<user>/NivraData/venv/bin/python` (system torch 2.6.0 + safetensors), Cloudflare metrics `127.0.0.1:20241` |
| Secrets | `/home/<user>/NivraData/controller/secrets/`: `controller-secrets.json` (identity fingerprint `d0a3:3ecb:8df3:e10e:974b:39ef:d42b:f4a7`), `database.env`, `initial-admin-password` — back these up; change the admin password |
| Project | "Nivra" with an active enrollment code (shown in Studio top-right; rotate it before public use, it was printed in a deployment log) |
| Pi as worker | `config/worker.json`, data `/home/<user>/NivraData/worker`, enrolled as `<pi-hostname>-worker` |
| Cloudflare | Existing remotely managed tunnel healthy (4 connections). **No Nivra hostnames were added** — needs the owner's Cloudflare dashboard login and consent (see `deploy/cloudflare/README.md`) |
| Autostart | Not configured (by design). Opt-in units in `deploy/systemd/` |
| Backups | Each deploy archives the previous `Desktop/Nivra` to `/home/<user>/NivraData/backups/` and keeps the release tarball in `releases/` |

## Deployment procedure (as executed)

```bash
# on the development machine
bash scripts/deploy-pi.sh            # builds linux/arm64, packages, verifies sha256 on the Pi,
                                     # backs up Desktop/Nivra, unpacks (keeps config/)
# first install only, on the Pi
cd ~/Desktop/Nivra && bash install.sh --role controller --yes
bin/nivra admin create --username admin --password-stdin < <password file>
```

## Verified on the Pi
* Controller start, migrations 1–3, health endpoint, Studio served.
* Worker enrollment from the laptop over the LAN; 10 real jobs; 50- and 100-worker load tests.
* Graceful stop/restart cycles with workers reconnecting automatically.
* Real distributed training run (3 DiLoCo rounds) with a controller restart mid-round.
* Generation job executed by a worker, MP4 stored and served by the Pi.

## Process state at the end of the session
See the final report: test processes started for this session (controller, Pi worker, laptop worker,
SSH tunnel, ComfyUI) are stopped at the end; the database container is left running or stopped as
reported there. **When the controller is stopped, Nivra Studio and the API are offline.**
