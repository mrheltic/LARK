#!/usr/bin/env bash
# run_gnss_sdr.sh — Launch gnss-sdr with LibreSDR AD9363
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./scripts/run_gnss_sdr.sh                    # GPS L1 real-time
#   ./scripts/run_gnss_sdr.sh multi              # GPS L1 + Galileo E1
#   ./scripts/run_gnss_sdr.sh file <path.dat>    # Process recorded IQ file
#
# Prerequisites:
#   gnss-sdr built with -DENABLE_FMCOMMS2=ON  (see setup.sh)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONF_DIR="$PROJECT_DIR/conf"
DATA_DIR="$PROJECT_DIR/data"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

# Check gnss-sdr is installed
if ! command -v gnss-sdr &>/dev/null; then
    err "gnss-sdr not found. Build it with: ./setup.sh (includes gnss-sdr build)"
fi

# Create data directory
mkdir -p "$DATA_DIR"

MODE="${1:-gps}"

case "$MODE" in
    gps|l1)
        info "Starting gnss-sdr — GPS L1 C/A (real-time AD9363)"
        CONF="$CONF_DIR/gnss_sdr_fmcomms2_gps_l1.conf"
        ;;
    multi|galileo)
        info "Starting gnss-sdr — GPS L1 + Galileo E1 (real-time AD9363)"
        CONF="$CONF_DIR/gnss_sdr_fmcomms2_multi.conf"
        ;;
    file)
        FILE="${2:-}"
        if [[ -z "$FILE" ]]; then
            # Find most recent capture
            FILE=$(ls -t "$DATA_DIR"/gnss_*.dat 2>/dev/null | head -1)
            if [[ -z "$FILE" ]]; then
                err "No IQ file specified and no captures found in $DATA_DIR"
            fi
            warn "Using most recent capture: $FILE"
        fi
        if [[ ! -f "$FILE" ]]; then
            err "File not found: $FILE"
        fi
        info "Starting gnss-sdr — File source: $FILE"
        CONF="$CONF_DIR/gnss_sdr_file_source.conf"
        EXTRA_ARGS="--SignalSource.filename=$FILE"
        ;;
    *)
        echo "Usage: $0 [gps|multi|file <path.dat>]"
        echo ""
        echo "  gps    — GPS L1 C/A real-time from AD9363 (default)"
        echo "  multi  — GPS L1 + Galileo E1 real-time from AD9363"
        echo "  file   — Process recorded IQ file (from gnss_lab.py REC)"
        exit 1
        ;;
esac

if [[ ! -f "$CONF" ]]; then
    err "Config not found: $CONF"
fi

ok "Config: $CONF"
info "Output directory: $DATA_DIR"
echo ""

cd "$PROJECT_DIR"
exec gnss-sdr --config_file="$CONF" ${EXTRA_ARGS:-}
