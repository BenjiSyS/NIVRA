#!/usr/bin/env bash
# Build Nivra: web assets (optional) + Go binaries for one or more platforms.
#
#   scripts/build.sh                 # host platform -> bin/nivra[.exe]
#   scripts/build.sh --all           # linux/arm64, linux/amd64, windows/amd64, darwin/arm64 -> dist/bin/
#   scripts/build.sh --target linux/arm64
#   scripts/build.sh --skip-web      # reuse existing internal/webui/dist
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${NIVRA_VERSION:-$(cat VERSION 2>/dev/null || echo 0.1.0-dev)}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
LDFLAGS="-s -w -X nivra/internal/shared/versioning.Version=${VERSION} -X nivra/internal/shared/versioning.Commit=${COMMIT}"

targets=()
skip_web=0
while [ $# -gt 0 ]; do
  case "$1" in
    --all) targets=(linux/arm64 linux/amd64 windows/amd64 darwin/arm64) ;;
    --target) shift; targets+=("$1") ;;
    --skip-web) skip_web=1 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

if [ "$skip_web" -eq 0 ] && [ -f web/package.json ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "==> building web interfaces"
    (cd web && npm ci --no-audit --no-fund && npm run build)
  else
    echo "!! npm not found; using existing internal/webui/dist" >&2
  fi
fi
if [ ! -f internal/webui/dist/studio.html ]; then
  echo "!! warning: web interface not built; the binary will serve a placeholder page" >&2
fi

command -v go >/dev/null 2>&1 || { echo "Go 1.27+ is required to build from source (https://go.dev/dl/)"; exit 1; }

if [ ${#targets[@]} -eq 0 ]; then
  out="bin/nivra"; [ "$(go env GOOS)" = "windows" ] && out="bin/nivra.exe"
  echo "==> go build $(go env GOOS)/$(go env GOARCH) -> $out"
  CGO_ENABLED=0 go build -trimpath -ldflags "$LDFLAGS" -o "$out" ./cmd/nivra
  exit 0
fi

mkdir -p dist/bin
for t in "${targets[@]}"; do
  os="${t%/*}"; arch="${t#*/}"
  out="dist/bin/nivra-${os}-${arch}"; [ "$os" = "windows" ] && out="${out}.exe"
  echo "==> go build $t -> $out"
  CGO_ENABLED=0 GOOS="$os" GOARCH="$arch" go build -trimpath -ldflags "$LDFLAGS" -o "$out" ./cmd/nivra
done
(cd dist/bin && sha256sum nivra-* > SHA256SUMS)
echo "==> checksums written to dist/bin/SHA256SUMS"
