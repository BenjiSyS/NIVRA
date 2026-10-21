# Nivra developer tasks. Requires Go 1.27+, Node 20+ (web), Python 3.11+ (ai).
.PHONY: build build-all web test test-go test-ai vet fmt release clean

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

release:          ## Build a deployable package in dist/
	bash scripts/package-release.sh

clean:
	rm -rf bin dist
