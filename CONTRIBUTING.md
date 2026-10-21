# Contributing to Nivra

Thanks for helping. Two ways to contribute:

1. **Compute** — install a worker (`install.bat` / `bash install.sh` → Worker) and connect it with a
   project code. Only verified work is credited.
2. **Code / research** — pull requests are welcome.

## Pull requests

* Keep changes focused; one topic per PR.
* Run before submitting:
  ```bash
  gofmt -l ./cmd ./internal ./database ./tests   # must print nothing
  go vet ./... && GOOS=linux GOARCH=arm64 go vet ./...
  go test -p 1 ./...                               # with NIVRA_TEST_DATABASE_URL set
  cd ai && python -m pytest -q tests
  cd web && npm run build
  ```
* Database changes: add a new numbered migration; never edit an applied one.
* Protocol changes: update `docs/api/API.md` / `docs/architecture/EXECUTOR_PROTOCOL.md` and bump the
  protocol version for incompatible changes.
* Model/research changes: record runs (exact command, config, measured numbers) in
  `docs/research/EXPERIMENTS.md`. Never report results you did not measure.

## Data and weights

Do not add datasets, model weights or generated media to git. Only contribute data you have the
rights to share, with its source and license recorded.

## Code of conduct

Be respectful and constructive. Harassment is not tolerated.
