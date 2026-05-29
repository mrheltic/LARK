#!/usr/bin/env bash
# Collect 5-channel Iridium bursts, then run validate_pipeline on the new dataset.
#
# Prerequisites:
#   - Heimdall DAQ running (ports 5000/5001)
#   - LibreSDR TX active for indoor tests (e.g. indoor_1626.py --mode ira --gain -20 --cyclic)
#
# Environment overrides (all optional):
#   COLLECT_SECONDS   max collection time [s]     (default: 120)
#   COLLECT_BURSTS    max bursts                  (default: 1500)
#   VALIDATE_AZ       known TX azimuth [deg]      (default: 0)
#   VALIDATE_EL       known TX elevation [deg]    (default: 60)
#   ALGO              music | capon | bartlett    (default: music)
#   NO_GT             1 = skip TLE annotation     (default: 1)
#   TONE_SNR_MIN      tone SNR gate [dB]          (default: 6)
#   FD_MAX            Doppler gate [Hz], 0=off    (default: 3000)
#   SAVE_PDF          1 = save validation plot    (default: 1)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRAKEN_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
LARK_ROOT="${LARK_ROOT:-$(cd "$KRAKEN_SRC/../.." && pwd)}"
VENV="${LARK_ROOT}/.venv/bin/python3"

COLLECT_SECONDS="${COLLECT_SECONDS:-120}"
COLLECT_BURSTS="${COLLECT_BURSTS:-1500}"
VALIDATE_AZ="${VALIDATE_AZ:-0}"
VALIDATE_EL="${VALIDATE_EL:-60}"
ALGO="${ALGO:-music}"
NO_GT="${NO_GT:-1}"
TONE_SNR_MIN="${TONE_SNR_MIN:-8}"
FD_MAX="${FD_MAX:-800}"
SAVE_PDF="${SAVE_PDF:-1}"
CHECK_HEIMDALL="${CHECK_HEIMDALL:-1}"

_port_listening() {
    ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${1}$"
}

echo "══════════════════════════════════════════════════════════════"
echo "  LARK — collect bursts + validate pipeline"
echo "  Collect : max ${COLLECT_SECONDS}s, up to ${COLLECT_BURSTS} bursts"
echo "  Ground truth : AZ=${VALIDATE_AZ}°  EL=${VALIDATE_EL}°"
echo "  Algorithm : ${ALGO}"
echo "══════════════════════════════════════════════════════════════"

if [[ "$CHECK_HEIMDALL" == "1" ]]; then
    echo "[1/3] Checking Heimdall (5000/5001) …"
    if ! _port_listening 5000 || ! _port_listening 5001; then
        echo "[ERROR] Heimdall not listening. Start task 'Heimdall: Start' first."
        exit 1
    fi
    echo "      Heimdall ready."
else
    echo "[1/3] Skipping Heimdall check (CHECK_HEIMDALL=0)."
fi

COLLECT_LOG="$(mktemp /tmp/lark_collect_XXXXXX.log)"
trap 'rm -f "$COLLECT_LOG"' EXIT

echo "[2/3] Collecting 5-channel burst dataset …"
export PYTHONUNBUFFERED=1
export LARK_ROOT
cd "$KRAKEN_SRC"

COLLECT_ARGS=(
    apps/doa_iridium/collect_iridium_burst_dataset.py
    --max-time-s "$COLLECT_SECONDS"
    --max-bursts "$COLLECT_BURSTS"
    --tone-snr-min "$TONE_SNR_MIN"
    --fd-max "$FD_MAX"
    --gain "$("$VENV" -c "import importlib.util, os; p=os.path.join('$KRAKEN_SRC','apps/doa_iridium/config.py'); s=importlib.util.spec_from_file_location('c',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.GAIN_DB)")"
    --verbose
)
if [[ "$NO_GT" == "1" ]]; then
    COLLECT_ARGS+=(--no-gt)
fi

if ! "$VENV" -u "${COLLECT_ARGS[@]}" "$@" 2>&1 | tee "$COLLECT_LOG"; then
    echo "[ERROR] Collection failed."
    exit 1
fi

NPZ_PATH="$(grep -E '^\[COLLECT\] Saved .+\.npz$' "$COLLECT_LOG" | sed -E 's/^\[COLLECT\] Saved //' | tail -1)"
if [[ -z "$NPZ_PATH" || ! -f "$NPZ_PATH" ]]; then
    echo "[ERROR] Could not find saved .npz path in collector output."
    exit 1
fi

echo "[3/3] Running pipeline validation on ${NPZ_PATH} …"
VALIDATE_ARGS=(
    apps/doa_iridium/validate_pipeline.py
    "$NPZ_PATH"
    --az "$VALIDATE_AZ"
    --el "$VALIDATE_EL"
    --algo "$ALGO"
)
if [[ "$SAVE_PDF" == "1" ]]; then
    REPORT_PATH="${NPZ_PATH%.npz}_report.pdf"
    VALIDATE_ARGS+=(--save-pdf "$REPORT_PATH")
fi

"$VENV" -u "${VALIDATE_ARGS[@]}"

echo
echo "══════════════════════════════════════════════════════════════"
echo "  Done."
echo "  Dataset : $NPZ_PATH"
if [[ "$SAVE_PDF" == "1" ]]; then
    echo "  Report  : $REPORT_PATH"
fi
echo "══════════════════════════════════════════════════════════════"
