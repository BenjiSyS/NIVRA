#!/usr/bin/env bash
# Deploy the current source tree to a Raspberry Pi (or other Linux arm64) controller over SSH.
#
#   NIVRA_PI_HOST=203.0.113.5 NIVRA_PI_USER=pi scripts/deploy-pi.sh [--skip-build]
#   scripts/deploy-pi.sh --host 192.0.2.10 --user pi [--path Desktop/Nivra] [--skip-build]
#
# NIVRA_PI_HOST   address or SSH alias of the Pi (required)
# NIVRA_PI_USER   Linux user on the Pi (required)
# NIVRA_PI_PATH   install directory relative to the user's home (default Desktop/Nivra; must end in /Nivra)
# NIVRA_PI_DATA   data directory relative to the user's home (default NivraData)
#
# Uses normal SSH host-key verification and key/agent authentication; it
# never handles passwords. Steps: build linux/arm64 binary, package tracked
# files, verify checksum remotely, back up the existing install, unpack
# (keeping config/),
# record the deployed version. It does not start or restart the controller.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }
PI_HOST_ARG="${NIVRA_PI_HOST:-}"; PI_USER_ARG="${NIVRA_PI_USER:-}"
REMOTE_DIR="${NIVRA_PI_PATH:-Desktop/Nivra}"
REMOTE_DATA="${NIVRA_PI_DATA:-NivraData}"
SKIP_BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) shift; PI_HOST_ARG="${1:-}" ;;
    --user) shift; PI_USER_ARG="${1:-}" ;;
    --path) shift; REMOTE_DIR="${1:-}" ;;
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
if [ -z "$PI_HOST_ARG" ] || [ -z "$PI_USER_ARG" ]; then
  echo "error: set NIVRA_PI_HOST and NIVRA_PI_USER (or pass --host and --user)." >&2
  usage >&2
  exit 2
fi
PI_HOST="${PI_USER_ARG}@${PI_HOST_ARG}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"
SCP="scp -o BatchMode=yes -o StrictHostKeyChecking=yes"

if [ "$SKIP_BUILD" -eq 0 ]; then
  # --skip-web: deploy never reinstalls web dependencies; build the web UI
  # separately (scripts/build.sh) when it changed.
  bash scripts/build.sh --skip-web --target linux/arm64
fi
BIN=dist/bin/nivra-linux-arm64
[ -x "$BIN" ] || [ -f "$BIN" ] || { echo "missing $BIN"; exit 1; }

VERSION="$(cat VERSION)"
COMMIT="$(git rev-parse --short HEAD)"
DIRTY=""; git diff --quiet HEAD || DIRTY="-dirty"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PKG="nivra-${VERSION}-${COMMIT}${DIRTY}-linux-arm64"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/$PKG/bin"
git ls-files -z | grep -zv '^web/' | tar --null -T - -cf - | tar -xf - -C "$STAGE/$PKG"
cp "$BIN" "$STAGE/$PKG/bin/nivra"
chmod +x "$STAGE/$PKG/bin/nivra" "$STAGE/$PKG/install.sh" "$STAGE/$PKG/start.sh" 2>/dev/null || true
printf '%s\n' "version=$VERSION" "commit=$COMMIT$DIRTY" "built=$STAMP" "platform=linux/arm64" > "$STAGE/$PKG/DEPLOYED_VERSION"
tar -C "$STAGE" -czf "$STAGE/$PKG.tar.gz" "$PKG"
SUM="$(sha256sum "$STAGE/$PKG.tar.gz" | cut -d' ' -f1)"
echo "==> package $PKG.tar.gz sha256 $SUM"

$SSH "$PI_HOST" "mkdir -p ~/$REMOTE_DATA/releases ~/$REMOTE_DATA/backups"
$SCP "$STAGE/$PKG.tar.gz" "$PI_HOST:$REMOTE_DATA/releases/"
$SSH "$PI_HOST" bash -s -- "$PKG" "$SUM" "$REMOTE_DIR" "$REMOTE_DATA" "$STAMP" <<'EOF'
set -euo pipefail
PKG="$1"; SUM="$2"; DIR="$HOME/$3"; DATA="$HOME/$4"; STAMP="$5"
cd "$DATA/releases"
echo "$SUM  $PKG.tar.gz" | sha256sum -c -
# Safety: only ever replace a directory named Nivra that is empty or holds a previous Nivra deployment.
[ "$(basename "$DIR")" = "Nivra" ] || { echo "refusing: target directory must be named Nivra"; exit 1; }
if [ -d "$DIR" ] && [ -n "$(ls -A "$DIR")" ] && [ ! -f "$DIR/DEPLOYED_VERSION" ]; then
  echo "refusing: $DIR exists, is not empty and is not a Nivra deployment"; exit 1
fi
if [ -d "$DIR" ]; then
  tar -C "$(dirname "$DIR")" -czf "$DATA/backups/desktop-nivra-$STAMP.tar.gz" "$(basename "$DIR")"
  echo "==> backed up existing $DIR to $DATA/backups/desktop-nivra-$STAMP.tar.gz"
fi
TMP="$(mktemp -d)"
tar -C "$TMP" -xzf "$PKG.tar.gz"
mkdir -p "$DIR"
# Preserve local configuration; replace everything else.
if [ -d "$DIR/config" ]; then cp -a "$DIR/config" "$TMP/$PKG/config.keep"; fi
find "$DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -a "$TMP/$PKG/." "$DIR/"
if [ -d "$DIR/config.keep" ]; then rm -rf "$DIR/config"; mv "$DIR/config.keep" "$DIR/config"; fi
rm -rf "$TMP"
chmod +x "$DIR/bin/nivra" "$DIR/install.sh" "$DIR/start.sh" "$DIR"/scripts/*.sh
"$DIR/bin/nivra" version
# The controller itself makes the nightly database + secrets backup (backup_enabled in config/controller.json);
# it needs pg_dump (apt install postgresql-client). See docs/installation/BACKUP.md.
echo "==> deployed to $DIR"
EOF
rm -rf "$STAGE"
