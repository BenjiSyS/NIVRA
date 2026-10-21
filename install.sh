#!/usr/bin/env bash
# Nivra installer for Linux (including Raspberry Pi OS / Debian ARM64) and macOS.
#
#   bash install.sh                      interactive menu
#   bash install.sh --role controller    non-menu controller install
#   bash install.sh --role worker [--with-ai]
#   bash install.sh --role repair | doctor
#
# Environment overrides: NIVRA_HOME (default ~/NivraData), NIVRA_PG_IMAGE
# (default postgres:17), NIVRA_PG_PORT (default 5433), NIVRA_DATABASE_URL
# (use an existing PostgreSQL instead of a Nivra container).
# The installer never stops unrelated services and never opens firewall ports.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
NIVRA_HOME="${NIVRA_HOME:-$HOME/NivraData}"
PG_IMAGE="${NIVRA_PG_IMAGE:-postgres:17}"
PG_PORT="${NIVRA_PG_PORT:-5433}"
PG_CONTAINER="nivra-pg"
ROLE=""
WITH_AI=0
ASSUME_YES=0

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }
ask()  { # ask "question" default -> REPLY
  if [ "$ASSUME_YES" -eq 1 ]; then REPLY="$2"; return; fi
  read -r -p "$1 [$2]: " REPLY || true; REPLY="${REPLY:-$2}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --role) shift; ROLE="$1" ;;
    --with-ai) WITH_AI=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

# ---------------------------------------------------------------- preflight
preflight() {
  say "Preflight checks"
  OS="$(uname -s)"; ARCH="$(uname -m)"
  case "$OS" in Linux) GOOS=linux ;; Darwin) GOOS=darwin ;; *) die "unsupported OS $OS (use install.bat on Windows)" ;; esac
  case "$ARCH" in aarch64|arm64) GOARCH=arm64 ;; x86_64|amd64) GOARCH=amd64 ;; *) die "unsupported CPU architecture $ARCH" ;; esac
  echo "    platform: $GOOS/$GOARCH"
  if [ "$(id -u)" -eq 0 ]; then warn "running as root is not recommended; Nivra runs fine as a normal user"; fi
  if [ -r /proc/meminfo ]; then
    mem_mb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 ))
    echo "    memory: ${mem_mb} MiB"
    [ "$mem_mb" -lt 1500 ] && warn "less than 1.5 GiB RAM; the controller may be slow"
  fi
  mkdir -p "$NIVRA_HOME"
  free_mb=$(df -Pm "$NIVRA_HOME" | awk 'NR==2 {print $4}')
  echo "    free disk at $NIVRA_HOME: ${free_mb} MiB"
  [ "${free_mb:-0}" -lt 5000 ] && warn "less than 5 GB free disk space for data, checkpoints and artifacts"
  if ! ping -c1 -W2 1.1.1.1 >/dev/null 2>&1 && ! curl -fsS -m 5 https://www.cloudflare.com >/dev/null 2>&1; then
    warn "internet connectivity check failed (OK for an offline LAN-only setup)"
  fi
}

# ---------------------------------------------------------------- binary
find_binary() {
  for c in "bin/nivra" "bin/nivra-$GOOS-$GOARCH" "dist/bin/nivra-$GOOS-$GOARCH"; do
    if [ -x "$c" ] && "$c" version >/dev/null 2>&1; then NIVRA="$ROOT/$c"; return 0; fi
  done
  if command -v go >/dev/null 2>&1; then
    say "Building Nivra from source"
    bash scripts/build.sh
    NIVRA="$ROOT/bin/nivra"; return 0
  fi
  die "no prebuilt binary for $GOOS/$GOARCH and Go is not installed. Download a release package or install Go 1.27+."
}

# ---------------------------------------------------------------- python
setup_python() { # $1 = purpose (controller|worker)
  command -v python3 >/dev/null 2>&1 || { warn "python3 not found; AI features disabled"; PY=""; return; }
  VENV="$NIVRA_HOME/venv"
  if [ ! -x "$VENV/bin/python" ]; then
    say "Creating Python environment at $VENV (reuses system packages such as an existing PyTorch)"
    python3 -m venv --system-site-packages "$VENV" || { warn "python venv failed (install python3-venv)"; PY=""; return; }
  fi
  PY="$VENV/bin/python"
  "$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
  if ! "$PY" -c "import torch" >/dev/null 2>&1; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      say "Installing PyTorch with CUDA support (large download)"
      "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 || warn "CUDA PyTorch install failed"
    else
      say "Installing CPU PyTorch (large download)"
      "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || warn "PyTorch install failed"
    fi
  fi
  "$PY" -m pip install --quiet -r ai/requirements.txt || warn "installing ai/requirements.txt failed"
  "$PY" -m pip install --quiet -r ai/requirements-caption.txt || warn "caption tools did not install (nivra caption --install)"
  if PYTHONPATH="$ROOT/ai" "$PY" -c "import nivra.cli, torch; print('    python AI executor OK: torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then :; else
    warn "AI executor self-test failed; run 'nivra doctor' for details"
  fi
}

# ---------------------------------------------------------------- postgres
setup_database() {
  SECRETS="$NIVRA_HOME/controller/secrets"
  mkdir -p "$SECRETS"; chmod 700 "$SECRETS"
  if [ -n "${NIVRA_DATABASE_URL:-}" ]; then
    umask 077; printf 'NIVRA_DATABASE_URL=%s\n' "$NIVRA_DATABASE_URL" > "$SECRETS/database.env"
    say "Using the PostgreSQL given in NIVRA_DATABASE_URL"; return
  fi
  if [ -f "$SECRETS/database.env" ]; then
    say "Existing database configuration found ($SECRETS/database.env)"
    start_pg_container_if_ours; return
  fi
  command -v docker >/dev/null 2>&1 || die "Docker not found. Install Docker, or set NIVRA_DATABASE_URL to an existing PostgreSQL 13+ database."
  if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
    die "a container named $PG_CONTAINER exists but $SECRETS/database.env is missing; restore it from backup (the installer will not touch existing data)"
  fi
  if (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -qE "[:.]$PG_PORT\s"; then
    die "port $PG_PORT is in use; set NIVRA_PG_PORT to a free port"
  fi
  PG_PASS="$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)"
  mkdir -p "$NIVRA_HOME/postgres"
  say "Starting a dedicated PostgreSQL container ($PG_IMAGE) on 127.0.0.1:$PG_PORT"
  # restart=no: Nivra does not auto-start on boot unless the owner enables it (see deploy/systemd).
  docker run -d --name "$PG_CONTAINER" --restart no \
    -e POSTGRES_USER=nivra -e POSTGRES_PASSWORD="$PG_PASS" -e POSTGRES_DB=nivra \
    -v "$NIVRA_HOME/postgres:/var/lib/postgresql/data" \
    -p "127.0.0.1:$PG_PORT:5432" \
    --health-cmd "pg_isready -U nivra" --health-interval 5s \
    "$PG_IMAGE" >/dev/null
  umask 077
  printf 'NIVRA_DATABASE_URL=postgres://nivra:%s@127.0.0.1:%s/nivra?sslmode=disable\n' "$PG_PASS" "$PG_PORT" > "$SECRETS/database.env"
  unset PG_PASS
  wait_pg
}

start_pg_container_if_ours() {
  command -v docker >/dev/null 2>&1 || return 0
  if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
    if [ "$(docker inspect -f '{{.State.Running}}' "$PG_CONTAINER")" != "true" ]; then
      say "Starting PostgreSQL container $PG_CONTAINER"; docker start "$PG_CONTAINER" >/dev/null
    fi
    wait_pg
  fi
}

wait_pg() {
  for _ in $(seq 1 60); do
    if docker exec "$PG_CONTAINER" pg_isready -U nivra >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  die "PostgreSQL did not become ready (docker logs $PG_CONTAINER)"
}

# ---------------------------------------------------------------- roles
write_config() { # $1 file, $2 content
  if [ -f "$1" ]; then echo "    keeping existing $1"; return; fi
  mkdir -p "$(dirname "$1")"; printf '%s\n' "$2" > "$1"; echo "    wrote $1"
}

install_controller() {
  preflight; find_binary
  setup_database
  setup_python controller
  local cf=""
  if curl -fsS -m 2 http://127.0.0.1:20241/ready >/dev/null 2>&1; then cf="http://127.0.0.1:20241"; fi
  write_config config/controller.json "{
  \"listen_addr\": \"127.0.0.1:8740\",
  \"data_dir\": \"$NIVRA_HOME/controller\",
  \"python_path\": \"${PY:-python3}\",
  \"ai_package_dir\": \"$ROOT/ai\",
  \"cloudflared_metrics_url\": \"$cf\"
}"
  say "Initialising controller (secrets + database schema)"
  "$NIVRA" controller init --config config/controller.json
  if ! "$NIVRA" project list >/dev/null 2>&1 || [ -z "$("$NIVRA" project list 2>/dev/null)" ]; then
    ask "Name of the first project" "Nivra"
    "$NIVRA" project create --name "$REPLY"
  fi
  if [ "$ASSUME_YES" -eq 0 ]; then
    ask "Create an administrator account now? (y/n)" "y"
    if [ "$REPLY" = "y" ]; then
      ask "Administrator username" "admin"
      "$NIVRA" admin create --username "$REPLY" || warn "administrator not created (it may already exist)"
    fi
  fi
  say "Controller installed."
  echo "    Start it with:   bash start.sh controller"
  echo "    Then open:       http://127.0.0.1:8740/  (Nivra Studio)"
  echo "    Data directory:  $NIVRA_HOME/controller  (back up secrets/ and the database)"
}

install_worker() {
  preflight; find_binary
  if [ "$WITH_AI" -eq 0 ] && [ "$ASSUME_YES" -eq 0 ]; then
    ask "Enable AI jobs (installs PyTorch, 1-3 GB download)? (y/n)" "y"
    [ "$REPLY" = "y" ] && WITH_AI=1
  fi
  PY=""
  [ "$WITH_AI" -eq 1 ] && setup_python worker
  write_config config/worker.json "{
  \"ui_listen_addr\": \"127.0.0.1:8741\",
  \"data_dir\": \"$NIVRA_HOME/worker\",
  \"python_path\": \"${PY}\",
  \"ai_package_dir\": \"$ROOT/ai\"
}"
  say "Worker installed."
  echo "    Start it with:   bash start.sh worker"
  echo "    Then open:       http://127.0.0.1:8741/  and enter your project code"
}

menu() {
  cat <<'EOF'

  NIVRA SETUP
  1 — Controller
  2 — Worker
  3 — Repair installation
  4 — System diagnostics
  5 — Exit
EOF
  read -r -p "Choose [1-5]: " choice
  case "$choice" in
    1) ROLE=controller ;; 2) ROLE=worker ;; 3) ROLE=repair ;; 4) ROLE=doctor ;; *) exit 0 ;;
  esac
}

[ -z "$ROLE" ] && menu
case "$ROLE" in
  controller) install_controller ;;
  worker) install_worker ;;
  repair) preflight; find_binary; start_pg_container_if_ours || true; "$NIVRA" repair ;;
  doctor) preflight; find_binary; "$NIVRA" doctor ;;
  *) die "unknown role $ROLE" ;;
esac
