# Nivra Quick Start

## Where things live (Raspberry Pi controller)

| What | Path |
|------|------|
| Program, scripts, docs (this folder) | `/home/benji/Desktop/Nivra` |
| Controller configuration | `/home/benji/Desktop/Nivra/config/controller.json` |
| Runtime data (artifacts, checkpoints, logs, backups) | `/home/benji/NivraData/controller` |
| Secrets (keys, database password, initial admin password) — **back up, never share** | `/home/benji/NivraData/controller/secrets/` |
| PostgreSQL data (container `nivra-pg`, 127.0.0.1:5433) | `/home/benji/NivraData/postgres` |
| Python environment for controller-side AI tasks | `/home/benji/NivraData/venv` |
| Release packages and pre-deploy backups | `/home/benji/NivraData/releases`, `/home/benji/NivraData/backups` |
| Installed version | `/home/benji/Desktop/Nivra/DEPLOYED_VERSION` |

## Start and stop the controller (Pi)

```bash
cd ~/Desktop/Nivra
bash start.sh controller        # starts nivra-pg if needed, then the controller (Ctrl+C stops it)
```

Nivra Studio is then at `http://127.0.0.1:8740/` on the Pi. From another computer use an SSH tunnel:

```bash
ssh -L 8740:127.0.0.1:8740 benji@192.168.1.142
```

and open `http://127.0.0.1:8740/`. Sign in as `admin`; the first password is in
`~/NivraData/controller/secrets/initial-admin-password`. Change it:

```bash
cd ~/Desktop/Nivra && bin/nivra admin passwd --username admin
```

To stop the database container as well: `bash start.sh stop-db`.
Nivra does **not** start automatically at boot. To enable that, see `deploy/systemd/`.
Public access through Cloudflare is described in `deploy/cloudflare/README.md`.

## Add a worker (any Windows/Linux computer)

1. Copy or clone this repository (or a release package) to the computer.
2. Windows: run `install.bat` → `2 — Worker`. Linux/macOS: `bash install.sh` → `2 — Worker`.
3. Start it: `start.bat` or `bash start.sh worker`. The Worker website opens at `http://127.0.0.1:8741/`.
4. Enter the controller URL and the **project code** shown in the top-right of Nivra Studio.

## Diagnostics and repair

```bash
bin/nivra doctor      # checks configuration, database, Python/PyTorch, GPU, ports
bin/nivra repair      # recreates missing directories, fixes permissions, applies migrations (keeps data)
```
