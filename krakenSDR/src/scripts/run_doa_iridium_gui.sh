#!/usr/bin/env bash
# Launch doa_iridium_burst.py after Heimdall is listening.
# Used by VS Code tasks — avoids starting DoA before ports 5000/5001 are open.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
LARK_ROOT="${LARK_ROOT:-$(cd "$SRC/../.." && pwd)}"
VENV="${LARK_ROOT}/.venv/bin/python3"

_port_listening() {
    ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${1}$"
}

echo "[DoA] Waiting for Heimdall on 127.0.0.1:5000 and :5001 …"
_ready=0
for _i in $(seq 1 90); do
    if _port_listening 5000 && _port_listening 5001; then
        echo "[DoA] Heimdall ports open (${_i}s)."
        _ready=1
        break
    fi
    sleep 1
done
if [[ "$_ready" -eq 0 ]]; then
    echo "[DoA] ERROR: Heimdall not ready after 90s."
    echo "       Run VS Code task: Heimdall: Start — real hardware (Iridium 1626.270 MHz)"
    exit 1
fi

# Stabilise after iq_server bind (delay_sync cal)
sleep 2

if ! pgrep -f 'indoor_1626\.py' >/dev/null 2>&1; then
    echo "[DoA] WARNING: indoor_1626.py TX not running."
    echo "         For lab tests start: LibreSDR: Indoor 1626 MHz — TX IRA burst"
fi

# Kill stale DoA instances (only one KrakenIQ client on port 5000)
if pgrep -f 'doa_iridium_burst\.py' >/dev/null 2>&1; then
    echo "[DoA] Stopping previous doa_iridium_burst instance …"
    pkill -f 'doa_iridium_burst\.py' 2>/dev/null || true
    sleep 1
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Qt5Agg}"
export DISPLAY="${DISPLAY:-:0}"

cd "$SRC"
exec "$VENV" -u apps/doa_iridium/doa_iridium_burst.py \
    --freq 1626.270 \
    --max-sats 1 \
    --fd-max 2000 \
    --multi 6 \
    --snr-min -5 \
    --papr-min 3 \
    "$@"
