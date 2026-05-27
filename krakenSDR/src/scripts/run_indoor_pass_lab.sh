#!/usr/bin/env bash
# Start LibreSDR pass TX + Kraken DoA runner with matched pass-mode settings.
# Waits for Heimdall, starts TX first (~20 s IQ build), then DoA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRAKEN_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
LARK_ROOT="${LARK_ROOT:-$(cd "$KRAKEN_SRC/../.." && pwd)}"
LIBRE_SRC="${LARK_ROOT}/libreSDR/src"
VENV="${LARK_ROOT}/.venv/bin/python3"
TX_GAIN="${TX_GAIN:--50}"
TX_ELEV="${TX_ELEV:-45}"
TX_DUR="${TX_DUR:-90}"
DOA_GUI="${DOA_GUI:-0}"

_port_listening() {
    ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${1}$"
}

echo "══════════════════════════════════════════════════════════════"
echo "  LARK indoor pass lab — LibreSDR TX + Kraken DoA"
echo "  TX: pass mode  gain=${TX_GAIN} dB  elev=${TX_ELEV}°  dur=${TX_DUR}s"
echo "  DoA: INDOOR_TX_MODE=pass from config.py (fd-gate OFF, multi=4)"
echo "══════════════════════════════════════════════════════════════"

echo "[1/4] Waiting for Heimdall (5000/5001) …"
_ready=0
for _i in $(seq 1 60); do
    if _port_listening 5000 && _port_listening 5001; then
        echo "      Heimdall ready (${_i}s)."
        _ready=1
        break
    fi
    sleep 1
done
if [[ "$_ready" -eq 0 ]]; then
    echo "[ERROR] Heimdall not listening. Start task 'Heimdall: Start' first."
    exit 1
fi
sleep 2

echo "[2/4] Stopping previous TX / DoA instances …"
pkill -f 'indoor_1626\.py' 2>/dev/null || true
pkill -f 'iridium_burst_doa_runner\.py' 2>/dev/null || true
sleep 1

echo "[3/4] Starting LibreSDR TX (background) …"
TX_LOG="$(mktemp /tmp/lark_tx_XXXXXX.log)"
cd "$LIBRE_SRC"
nohup "$VENV" -u tx/indoor_1626.py \
    --mode pass --gain "$TX_GAIN" --elev "$TX_ELEV" --dur "$TX_DUR" --cyclic \
    >"$TX_LOG" 2>&1 &
TX_PID=$!
echo "      TX pid=$TX_PID  log=$TX_LOG"

echo "      Waiting for TX streaming (IQ build ~15 s + connect) …"
_tx_ok=0
for _i in $(seq 1 120); do
    if grep -q "Streaming.*looping until Ctrl+C" "$TX_LOG" 2>/dev/null; then
        echo "      TX streaming active (${_i}s)."
        _tx_ok=1
        break
    fi
    if ! kill -0 "$TX_PID" 2>/dev/null; then
        echo "[ERROR] TX process exited early. Log:"
        tail -30 "$TX_LOG"
        exit 1
    fi
    sleep 1
done
if [[ "$_tx_ok" -eq 0 ]]; then
    echo "[ERROR] TX did not reach streaming within 120 s."
    tail -20 "$TX_LOG"
    exit 1
fi
sleep 2

echo "[4/4] Starting DoA runner …"
export PYTHONUNBUFFERED=1
export LARK_ROOT
cd "$KRAKEN_SRC"
DOA_ARGS=(--freq 1626.270)
if [[ "$DOA_GUI" == "0" ]]; then
    DOA_ARGS+=(--no-plot)
else
    export MPLBACKEND="${MPLBACKEND:-Qt5Agg}"
    export DISPLAY="${DISPLAY:-:0}"
fi

echo "      Press Ctrl+C to stop DoA (TX keeps running — kill pid $TX_PID to stop TX)."
exec "$VENV" -u apps/doa_iridium/iridium_burst_doa_runner.py "${DOA_ARGS[@]}" "$@"
