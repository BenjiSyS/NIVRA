#!/usr/bin/env bash
# Deploy the current source tree to a Raspberry Pi controller over SSH.
#
#   PI_HOST=benji@192.168.1.142 scripts/deploy-pi.sh [--skip-build]
#
# Uses normal SSH host-key verification and key/agent authentication; it
# never handles passwords. Steps: build linux/arm64 binary, package tracked
# files, verify checksum remotely, back up the existing Desktop/Nivra, unpack
# (keeping config/), record the deployed version. It does not start services.
set -euo pipefail
cd "$(dirname "$0")/.."
PI_HOST="${PI_HOST:-benji@192.168.1.142}"
REMOTE_DIR="${REMOTE_DIR:-Desktop/Nivra}"
REMOTE_DATA="${REMOTE_DATA:-NivraData}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"
SCP="scp -o BatchMode=yes -o StrictHostKeyChecking=yes"

if [ "${1:-}" != "--skip-build" ]; then
  bash scripts/build.sh --target linux/arm64
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
echo "==> deployed to $DIR"
EOF
rm -rf "$STAGE"
