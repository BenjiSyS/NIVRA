# Development Setup

## Toolchain

| Tool | Version used | Needed for |
|------|--------------|-----------|
| Go | 1.27.3 | controller, worker, CLI |
| Node.js / npm | 26 / 11 | Nivra Studio + Worker website (`web/`) |
| Python | 3.11+ (3.14 on Windows dev, 3.13 on the Pi) | AI package (`ai/`) |
| PyTorch | 2.14 CPU (dev), 2.6 (Pi system package) | AI package |
| PostgreSQL | 13+ (18.6 dev, 17 on the Pi) | controller and database tests |
| ffmpeg | any recent | MP4 export |

## Build

```bash
bash scripts/build.sh                    # web + host binary -> bin/nivra
bash scripts/build.sh --all              # release binaries -> dist/bin/ (+ SHA256SUMS)
bash scripts/build.sh --target linux/arm64 --skip-web
```

## Database for tests

Go database tests run only when `NIVRA_TEST_DATABASE_URL` points to a database whose **name ends in
`_test`** (the harness drops and recreates the `public` schema). A local cluster:

```bash
initdb -D ~/nivra-pg -U nivra --pwfile=<file> --auth=scram-sha-256
pg_ctl -D ~/nivra-pg -o "-p 15432 -c listen_addresses=127.0.0.1" -l ~/nivra-pg.log start
psql -h 127.0.0.1 -p 15432 -U nivra -d postgres -c "CREATE DATABASE nivra_test"
export NIVRA_TEST_DATABASE_URL="postgres://nivra:<password>@127.0.0.1:15432/nivra_test?sslmode=disable"
```

## Tests

```bash
go test -p 1 ./...                      # unit + database + end-to-end (needs the test DB)
cd ai && python -m pytest -q tests      # AI package
cd web && npm run build                 # type-check + build web UIs
```

`-p 1` is required because database test packages share one test database.

`tests/integration/e2e_test.go` starts a real controller and a real worker agent in-process over
HTTP and walks the acceptance flow: enrollment, job execution and validation, contributions,
pause/resume with confirmed commands, stop-task requeue, disconnect/reconnect, controller restart
with persistence, revocation.

## Running locally

```bash
export NIVRA_DATABASE_URL=postgres://...      # or put it in <data_dir>/secrets/database.env
bin/nivra controller init
bin/nivra admin create --username admin
bin/nivra project create --name Dev
bin/nivra controller run                      # http://127.0.0.1:8740
bin/nivra worker run --open                   # http://127.0.0.1:8741
bin/nivra job create --project Dev --type builtin.sha256_chain \
  --params '{"seed":"<64 hex>","iterations":100000}' --count 10
```

## Repository layout

```
cmd/nivra/                 single binary entry point
database/                  embedded SQL migrations + migrator
internal/controller/       api, app (startup), auth, enrollment, projects, workers,
                           scheduler, artifacts, telemetry, training, events, secrets
internal/worker/           agent, client, executor, hardware, identity, localui
internal/shared/           protocol, config, builtin jobs, pyexec, sysinfo, lockfile
internal/loadtest/         simulated-worker load generator
internal/doctor/           nivra doctor / repair
web/                       Nivra Studio + Worker website (React + TypeScript + Vite)
ai/                        original Nivra model research package (PyTorch)
tests/integration/         end-to-end acceptance test
deploy/                    systemd (opt-in), Cloudflare guide
scripts/                   build, deploy, installer logic
docs/                      documentation and progress records
```
