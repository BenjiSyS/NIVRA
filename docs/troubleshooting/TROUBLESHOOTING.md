# Troubleshooting

Start with `bin/nivra doctor` (or `install.bat` → 4). It prints `[FAIL]` lines with the fix.

| Symptom | Cause | Fix |
|---------|-------|-----|
| `another Nivra process is already using this data directory (held by pid N)` | A controller/worker is already running on this data dir | Stop it (Ctrl+C in its window) or use a different `data_dir`. The lock is released automatically if the process dies. |
| `listen 127.0.0.1:8740: ... address already in use` | Port in use | `nivra doctor` shows it; change `listen_addr` / `ui_listen_addr`. |
| `no database configured` | Missing `secrets/database.env` | Re-run the installer or set `NIVRA_DATABASE_URL`. |
| `database unreachable` | `nivra-pg` container stopped | `docker start nivra-pg` (or `bash start.sh controller`, which starts it). |
| `migration ... was modified after being applied` | Edited migration file | Restore the original file; add a new migration instead. |
| `project ...: project code ciphertext failed authentication (was the secrets file replaced?)` | `controller-secrets.json` differs from the one that created the database | Restore the original secrets file from backup. Do not delete it. |
| Worker: "That project code was not recognised" | Typo, rotated or closed code | Copy the code again from the top-right of Nivra Studio. |
| Worker state `offline` with "controller unreachable" | Network/tunnel down or controller stopped | The worker retries with backoff; check `https://api.<domain>/api/v1/health`. |
| Worker state `revoked` | An administrator revoked the device | Ask for a new code and use **Change Project**; the old key cannot re-enroll. |
| Worker refuses `http://` URL | Plain HTTP to a public host is blocked | Use `https://` (Cloudflare) or a LAN/loopback address. |
| `controller identity mismatch; refusing to connect` | The controller at that URL presents a different identity key than at enrollment | Verify you are talking to the right server; if the controller was reinstalled with new secrets, re-enroll. |
| AI executor unavailable in Hardware tab | Python/PyTorch missing or `nivra` package not importable | Re-run installer with AI enabled; the probe error is shown verbatim. |
| `OOM` job failures | Model/batch too large for RAM/VRAM budget | Lower the RAM/VRAM budget in Resource settings so the scheduler stops sending such jobs; smaller `batch_size` in the run config. |
| RTX 20-series: very slow or errors with bf16 | Turing has no bf16 | Nivra uses FP16 autocast on CUDA only and never bf16. |
| Job stuck in `validating` | Controller-side Python unavailable | Check `python_path` / `ai_package_dir` in `config/controller.json`; validation retries every 10 s. |
| Studio shows "web interface not built" | Binary built without `web/` assets | `bash scripts/build.sh` (needs Node.js). |
