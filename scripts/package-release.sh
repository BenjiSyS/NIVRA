#!/usr/bin/env bash
# Package a Nivra release into dist/release/:
#   nivra-<VERSION>-windows-amd64.zip       Nivra.exe, bin/nivra.exe, ai/, scripts/install.ps1, ...
#   nivra-<VERSION>-linux-arm64.tar.gz      controller: bin/nivra, ai/, deploy/, install.sh, start.sh, ...
#   SHA256SUMS                              checksums of both archives
#
#   scripts/package-release.sh [--skip-web] [--version X.Y.Z]
#
#   --skip-web   reuse the existing web build (internal/webui/dist) instead of building it
#   --version    override the VERSION file (also: NIVRA_VERSION)
#
# Works in Git Bash and Linux. Needs Go (see go.mod), tar, sha256sum, Python 3 (for the zip step),
# and Node.js/npm unless --skip-web is given. The web interface is embedded in the Go binaries.
# Publish: create a GitHub release tagged v<VERSION> and attach all three files; scripts\install.ps1
# downloads the zip of the latest release and verifies it against SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/.."

skip_web=0
VERSION="${NIVRA_VERSION:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-web) skip_web=1 ;;
    --version) shift; VERSION="${1:?--version needs a value}" ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$VERSION" ] || VERSION="$(tr -d '[:space:]' < VERSION 2>/dev/null || true)"
[ -n "$VERSION" ] || VERSION="0.1.0-dev"
case "$VERSION" in *[!A-Za-z0-9._+-]*|"") echo "invalid version: $VERSION" >&2; exit 2 ;; esac
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
LDFLAGS="-s -w -X nivra/internal/shared/versioning.Version=${VERSION} -X nivra/internal/shared/versioning.Commit=${COMMIT}"

command -v go >/dev/null 2>&1 || { echo "Go is required (https://go.dev/dl/)" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum is required" >&2; exit 1; }
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import zipfile' >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3 is required to create the zip" >&2; exit 1; }

# ---- web ----
if [ "$skip_web" -eq 0 ]; then
  command -v npm >/dev/null 2>&1 || { echo "npm not found; install Node.js or use --skip-web with an existing build" >&2; exit 1; }
  echo "==> building web interfaces"
  (cd web
   [ -d node_modules ] || npm ci --no-audit --no-fund
   npm run build)
fi
[ -f internal/webui/dist/studio.html ] || { echo "web interface is not built (internal/webui/dist/studio.html missing)" >&2; exit 1; }

OUT=dist/release
STAGE="$OUT/.stage"
rm -rf "$STAGE"
mkdir -p "$OUT" "$STAGE"
WIN_NAME="nivra-${VERSION}-windows-amd64"
LIN_NAME="nivra-${VERSION}-linux-arm64"
rm -f "$OUT/$WIN_NAME.zip" "$OUT/$LIN_NAME.tar.gz" "$OUT/SHA256SUMS"

# copy_ai <destination>: the Python package without tests, caches, weights.
copy_ai() {
  mkdir -p "$1"
  tar -cf - -C ai \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pt' --exclude='*.safetensors' \
      --exclude='*.npz' --exclude='*.egg-info' --exclude='.pytest_cache' \
      nivra configs scripts pyproject.toml requirements.txt requirements-caption.txt | tar -xf - -C "$1"
}

# ---- Windows: Nivra.exe (GUI) + nivra.exe (CLI) ----
echo "==> go build windows/amd64"
W="$STAGE/$WIN_NAME"
mkdir -p "$W/bin" "$W/scripts"
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "$LDFLAGS" -o "$W/bin/nivra.exe" ./cmd/nivra
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -trimpath \
  -ldflags "$LDFLAGS -H=windowsgui -X main.defaultCommand=app" -o "$W/Nivra.exe" ./cmd/nivra
copy_ai "$W/ai"
cp scripts/install.ps1 "$W/scripts/install.ps1"
cp install.bat start.bat VERSION LICENSE "$W/"
printf '%s\n' "$VERSION" > "$W/VERSION"
cat > "$W/README-RELEASE.txt" <<TXT
Nivra $VERSION (windows-amd64, commit $COMMIT)

Contents: Nivra.exe (the app), bin\nivra.exe (command line), ai\ (Python package), scripts\install.ps1.

Install or update:
  1. Quit Nivra if it is running.
  2. Run Nivra-Setup.bat (it downloads this package and verifies its SHA-256), or extract this zip to a folder and run:
       powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Role app
     The script installs Python, PyTorch and ffmpeg and creates Desktop and Start menu shortcuts.
  3. Open Nivra from the shortcut or Nivra.exe.

Verify the download: compare the SHA-256 of this zip with the SHA256SUMS file of the same release.
  PowerShell:  Get-FileHash nivra-$VERSION-windows-amd64.zip -Algorithm SHA256
TXT

# ---- Linux arm64 controller ----
echo "==> go build linux/arm64"
L="$STAGE/$LIN_NAME"
mkdir -p "$L/bin" "$L/scripts"
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath -ldflags "$LDFLAGS" -o "$L/bin/nivra" ./cmd/nivra
copy_ai "$L/ai"
cp -r deploy "$L/deploy"
cp install.sh start.sh VERSION LICENSE "$L/"
cp scripts/*.sh "$L/scripts/"
printf '%s\n' "$VERSION" > "$L/VERSION"
printf '%s\n' "version=$VERSION" "commit=$COMMIT" "platform=linux/arm64" > "$L/DEPLOYED_VERSION"
chmod +x "$L/bin/nivra" "$L/install.sh" "$L/start.sh" "$L"/scripts/*.sh
cat > "$L/README-RELEASE.txt" <<TXT
Nivra $VERSION controller package (linux-arm64, commit $COMMIT)

Extract, then:  bash install.sh --role controller
Systemd user units are in deploy/systemd/. Web interfaces are embedded in bin/nivra.
Verify the download against the SHA256SUMS file of the same release:  sha256sum -c SHA256SUMS --ignore-missing
TXT

# ---- archives ----
echo "==> creating archives"
"$PY" - "$STAGE/$WIN_NAME" "$OUT/$WIN_NAME.zip" <<'PYEOF'
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
base = os.path.basename(src)
entries = []
for root, dirs, files in os.walk(src):
    dirs.sort()
    for f in sorted(files):
        p = os.path.join(root, f)
        entries.append((p, os.path.relpath(p, src).replace(os.sep, "/")))
entries.sort(key=lambda e: e[1])
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p, rel in entries:
        info = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))  # fixed time: reproducible archive
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        with open(p, "rb") as fh:
            z.writestr(info, fh.read())
print("    %s: %d files" % (dst, len(entries)))
PYEOF
tar --owner=0 --group=0 --numeric-owner -czf "$OUT/$LIN_NAME.tar.gz" -C "$STAGE" "$LIN_NAME"
rm -rf "$STAGE"
(cd "$OUT" && sha256sum --text "$WIN_NAME.zip" "$LIN_NAME.tar.gz" > SHA256SUMS)

echo "==> release $VERSION ($COMMIT) in $OUT/"
(cd "$OUT" && ls -l "$WIN_NAME.zip" "$LIN_NAME.tar.gz" SHA256SUMS && cat SHA256SUMS)
