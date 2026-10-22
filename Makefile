# Nivra developer tasks. Requires Go 1.27+, Node 20+ (web), Python 3.11+ (ai).
.PHONY: build build-all web test test-go test-ai vet fmt release release-quick ci-local clean

build:            ## Build web + host binary into bin/
	bash scripts/build.sh

build-all:        ## Cross-compile release binaries into dist/bin/
	bash scripts/build.sh --all

web:              ## Build Nivra Studio + Worker websites
	cd web && npm ci && npm run build

vet:
	go vet ./...
	GOOS=linux GOARCH=arm64 go vet ./...

fmt:
	gofmt -w ./cmd ./internal ./database ./tests

test: test-go test-ai

test-go:          ## Needs NIVRA_TEST_DATABASE_URL for database tests (see docs/getting-started/DEVELOPMENT.md)
	go test -p 1 ./...

test-ai:
	cd ai && python -m pytest -q tests

release:          ## Windows zip + Linux arm64 tarball + SHA256SUMS in dist/release/
	bash scripts/package-release.sh

release-quick:    ## Same, reusing the existing web build
	bash scripts/package-release.sh --skip-web

ci-local:         ## Roughly what .github/workflows/ci.yml runs (database tests need NIVRA_TEST_DATABASE_URL)
	go vet ./...
	test -z "$$(gofmt -l ./cmd ./internal ./database ./tests)"
	go test -count=1 -p 1 ./...
	cd ai && python -m pytest -q tests/test_exec_protocol.py tests/test_longvideo.py tests/test_te_archive.py tests/test_hd_budget.py
	cd web && npm run build

clean:
	rm -rf bin dist
