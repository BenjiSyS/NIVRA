#!/usr/bin/env bash
# Start Nivra in the foreground. Ctrl+C stops it cleanly (the running task is
# returned to the queue; the database container is left running).
#
#   bash start.sh controller | worker | stop-db
#   bash start.sh            (picks the role that has a config file)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
NIVRA_HOME="${NIVRA_HOME:-$HOME/NivraData}"

case "$(uname -m)" in aarch64|arm64) A=arm64 ;; *) A=amd64 ;; esac
case "$(uname -s)" in Darwin) O=darwin ;; *) O=linux ;; esac
NIVRA=""
for c in bin/nivra "bin/nivra-$O-$A" "dist/bin/nivra-$O-$A"; do
  if [ -x "$c" ]; then NIVRA="$ROOT/$c"; break; fi
done
[ -n "$NIVRA" ] || { echo "Nivra binary not found; run: bash install.sh"; exit 1; }

role="${1:-}"
if [ -z "$role" ]; then
  if [ -f config/controller.json ]; then role=controller; elif [ -f config/worker.json ]; then role=worker; else
    echo "No configuration found; run: bash install.sh"; exit 1; fi
fi

case "$role" in
  controller)
    if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' | grep -qx nivra-pg; then
      if [ "$(docker inspect -f '{{.State.Running}}' nivra-pg)" != "true" ]; then
        echo "==> starting PostgreSQL container nivra-pg"; docker start nivra-pg >/dev/null
        for _ in $(seq 1 60); do docker exec nivra-pg pg_isready -U nivra >/dev/null 2>&1 && break; sleep 1; done
      fi
    fi
    exec "$NIVRA" controller run --config config/controller.json ;;
  worker)
    exec "$NIVRA" worker run --config config/worker.json "${@:2}" ;;
  stop-db)
    docker stop nivra-pg ;;
  *) echo "usage: bash start.sh [controller|worker|stop-db]"; exit 2 ;;
esac
