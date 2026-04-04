#!/bin/bash
# ============================================================
#  KrakenSDR Docker – Entrypoint
#  Usage: docker run krakensdr:latest [COMMAND]
#
#  Commands:
#    bash              – interactive shell (default)
#    test              – kraken_test.py  (real hardware, 2 ch)
#    fft               – kraken_fft_display.py  (5 ch)
#    synthetic         – kraken_synthetic_test.py  (no hardware needed)
#    dev               – Heimdall bg + interactive shell (/workspace in PYTHONPATH)
#    grc               – GNU Radio Companion (empty workspace, no Heimdall)
#    grc-heimdall      – Heimdall hw + GRC with kraken_doa_main.grc
#    grc-synth         – Heimdall synthetic + GRC with kraken_doa_main.grc
#    doa               – Heimdall bg + N-antenna DoA with widgets (real hardware)
#    doa-synth         – Heimdall synthetic + N-antenna DoA with widgets
#    heimdall          – Heimdall DAQ (real hardware)
#    heimdall-synth    – Heimdall DAQ synthetic mode
# ============================================================
set -e

EXAMPLES="/opt/krakensdr/gr-krakensdr/examples"
FIRMWARE="/opt/krakensdr/heimdall_daq_fw/Firmware"

# ── Kernel / network tuning ─────────────────────────────────────────────────────
# Requires --privileged.  Enlarges the OS socket receive/send buffers so
# that the ZeroMQ pipes between Heimdall and GNU Radio can absorb CPU
# micro-spikes without dropping IQ frames.
# With --network host these sysctl values are written to the HOST kernel.
_tune_kernel() {
    if sysctl -qw net.core.rmem_max=134217728 2>/dev/null; then
        sysctl -qw net.core.wmem_max=134217728
        sysctl -qw net.core.rmem_default=1048576
        sysctl -qw net.core.wmem_default=1048576
        echo "[✓] ZMQ network buffers: rmem/wmem_max = 128 MB"
    else
        echo "[!] Cannot tune network buffers (missing --privileged?)."
    fi
}
_tune_kernel

# ── GNU Radio prefs dir ──────────────────────────────────────────────────────
# GNU Radio looks for /root/.gnuradio/prefs/vmcircbuf_default_factory on first
# start; if missing it logs a warning and falls back to POSIX shm.
# Create the dir and set the preferred buffer type to suppress the warning.
mkdir -p /root/.gnuradio/prefs
if [ ! -f /root/.gnuradio/prefs/vmcircbuf_default_factory ]; then
    echo 'gr::vmcircbuf_sysv_shm_factory' \
        > /root/.gnuradio/prefs/vmcircbuf_default_factory
fi

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          KrakenSDR Docker Environment                ║"
echo "║  GNU Radio 3.10  |  gr-krakensdr  |  Heimdall DAQ    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

if [ -z "$DISPLAY" ]; then
    echo "[!] DISPLAY not set — GUI not available."
else
    echo "[✓] DISPLAY=$DISPLAY"
fi

if lsusb 2>/dev/null | grep -qE "0bda:2838|0bda:2832"; then
    HW_COUNT=$(lsusb | grep -cE "0bda:2838|0bda:2832")
    echo "[✓] RTL-SDR hardware: ${HW_COUNT} dongle(s) detected"
    HW_AVAILABLE=1
else
    echo "[!] RTL-SDR not found — use 'synthetic' for a no-hardware demo"
    HW_AVAILABLE=0
fi
echo ""

# ── Helper: check port LISTEN state without making a TCP connection ──────────
# nc -z triggers accept() in iq_server.out → server closes and re-binds socket
# → brief window where port 5000 is not listening → ConnectionRefusedError.
# ss reads kernel state only, no connection is made.
_port_listening() {
    ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${1}$"
}

# ── Helper: kill all DAQ processes and release ports ────────────────────────
_kill_daq() {
    (cd "$FIRMWARE" 2>/dev/null && bash daq_stop.sh 2>/dev/null) || true
    sleep 1
    # Running as root inside container — no sudo needed.
    # Use pgrep+kill instead of pkill -f (pkill -f self-matches its own cmdline).
    for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
                hw_controller.py delay_sync.py test_data_synthesizer.py; do
        local pids
        pids=$(pgrep -f "$proc" 2>/dev/null || true)
        [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
    done
    for port in 5000 5001 1130; do
        local pids
        pids=$(lsof -ti:"$port" 2>/dev/null || true)
        [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
    done
    # Remove stale POSIX shared memory segments (cause double-free on next start)
    for seg in decimator_out_A decimator_out_B \
               delay_sync_iq_A delay_sync_iq_B \
               delay_sync_hwc_A delay_sync_hwc_B; do
        rm -f "/dev/shm/$seg" 2>/dev/null || true
    done
}

# ── Helper: detach kernel drivers so libusb can claim RTL-SDR interfaces ──────
# With --privileged the container can rmmod host kernel modules.
_detach_rtl_drivers() {
    for mod in dvb_usb_rtl28xxu dvb_usb_v2 rtl2832 rtl2830 r820t; do
        if lsmod 2>/dev/null | grep -q "^${mod} "; then
            echo "[*] Detaching kernel driver: $mod"
            rmmod "$mod" 2>/dev/null || true
        fi
    done
}

# ── Trap: guarantee cleanup when entrypoint exits (Ctrl+C, error, normal) ───
_on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    echo ""
    echo "[*] Container exiting (rc=${rc}) — dumping DAQ logs:"
    _dump_logs 2>/dev/null || true
    _kill_daq 2>/dev/null || true
    exit $rc
}
trap '_on_exit' EXIT INT TERM

# ── Helper: dump DAQ logs for post-mortem diagnostics ───────────────────────
_dump_logs() {
    echo ""
    echo "══ DAQ process status ══════════════════════════════════════"
    for proc in rtl_daq.out rebuffer.out decimate.out delay_sync.py \
                iq_server.out hw_controller.py; do
        if pgrep -f "$proc" >/dev/null 2>&1; then
            echo "  [running] $proc"
        else
            echo "  [stopped] $proc"
        fi
    done
    echo ""
    echo "══ Log files ═══════════════════════════════════════════════"
    for log in rtl_daq rebuffer decimator delay_sync iq_server hwc; do
        local f="${FIRMWARE}/_logs/${log}.log"
        if [ -s "$f" ]; then
            echo "--- ${log}.log (last 10 lines) ---"
            tail -n 10 "$f"
            echo ""
        fi
    done
}

# ── Helper: start Heimdall in background and wait for port 5000 ─────────────
start_heimdall_bg() {
    local mode="$1"   # "hw" or "synth"

    echo "[*] Cleaning up any leftover DAQ processes..."
    _kill_daq 2>/dev/null || true
    sleep 1

    if _port_listening 5000; then
        echo "[v] Heimdall already running on port 5000"
        return 0
    fi

    if [ "$mode" = "hw" ]; then
        if [ "$HW_AVAILABLE" -eq 0 ]; then
            echo "[!] No RTL-SDR detected -- starting in synthetic mode instead"
            mode="synth"
        else
            # Detach host kernel drivers (container has --privileged)
            _detach_rtl_drivers
            echo "[*] Auto-starting Heimdall DAQ (hardware mode)..."
            # Running as root: no sudo needed.
            # Redirect DAQ output to log file — prevents stdout/stderr from
            # interleaving with the wait-loop dots and scrambling the terminal.
            (cd "$FIRMWARE" && bash daq_start_sm.sh >"$FIRMWARE/_logs/daq_start.log" 2>&1) &
        fi
    fi
    if [ "$mode" = "synth" ]; then
        echo "[*] Auto-starting Heimdall DAQ (synthetic mode)..."
        (cd "$FIRMWARE" && bash daq_synthetic_start.sh >"$FIRMWARE/_logs/daq_start.log" 2>&1) &
    fi

    # Ensure Qt XDG runtime dir exists with correct permissions (Qt requires 0700)
    mkdir -p "${XDG_RUNTIME_DIR:-/tmp/runtime-kraken}"
    chmod 700 "${XDG_RUNTIME_DIR:-/tmp/runtime-kraken}"

    # Synthetic mode warms up faster; use a shorter timeout.
    local max_wait=90
    if [ "$mode" = "synth" ]; then
        max_wait=60
    fi

    echo -n "[*] Waiting for Heimdall on port 5000 "
    for i in $(seq 1 "$max_wait"); do
        sleep 1
        if _port_listening 5000; then
            # Wait for Heimdall to finish initial noise-source calibration and
            # reach STATE_TRACK (sync_state=6). iq_server opens port 5000 well
            # before delay_sync completes STATE_IQ_CAL, so a short sleep here
            # avoids the client receiving CAL/DUMMY frames at startup.
            echo " listening! Stabilizing 15s for noise-source calibration..."
            sleep 15
            echo "[✓] Heimdall ready (${i}s)"
            return 0
        fi
        if [ "$i" -gt 20 ]; then
            local iq_alive rtl_alive dec_alive ds_alive
            pgrep -f 'iq_server\.out'  >/dev/null 2>&1 && iq_alive=1  || iq_alive=0
            pgrep -f 'rtl_daq\.out'    >/dev/null 2>&1 && rtl_alive=1 || rtl_alive=0
            pgrep -f 'decimate\.out'   >/dev/null 2>&1 && dec_alive=1 || dec_alive=0
            pgrep -f 'delay_sync\.py'  >/dev/null 2>&1 && ds_alive=1  || ds_alive=0

            if [ "$iq_alive" -eq 0 ] && [ "$rtl_alive" -eq 0 ] && \
               [ "$dec_alive" -eq 0 ] && [ "$ds_alive" -eq 0 ]; then
                echo ""
                echo "[✗] All DAQ processes exited after ${i}s."
                _dump_logs
                exit 1
            fi
            if [ "$rtl_alive" -eq 0 ] && [ "$mode" = "hw" ]; then
                echo ""
                echo "[✗] rtl_daq.out exited after ${i}s — RTL-SDR device error."
                _dump_logs
                exit 1
            fi
        fi
        echo -n "."
    done
    echo ""
    echo "[x] Heimdall did not open port 5000 within ${max_wait} seconds."
    _dump_logs
    exit 1
}

# ── Command dispatch ─────────────────────────────────────────────────────────
case "${1:-bash}" in

    "test")
        echo "[*] Starting: kraken_test.py (2 channels, real hardware)"
        start_heimdall_bg "hw"
        cd "$EXAMPLES"
        python3 kraken_test.py
        ;;

    "fft")
        echo "[*] Starting: kraken_fft_display.py (5 channels, real hardware)"
        start_heimdall_bg "hw"
        cd "$EXAMPLES"
        python3 kraken_fft_display.py
        ;;

    "dev")
        echo "[*] Dev mode: Heimdall running in background + interactive shell"
        echo "    /workspace  ← mounted from host (krakensdr/workspace/)"
        echo "    PYTHONPATH includes /workspace and the gr-krakensdr modules"
        echo "    Examples: $EXAMPLES"
        echo ""
        start_heimdall_bg "hw"
        export PYTHONPATH="/workspace:${PYTHONPATH:-}"
        cd /workspace
        exec /bin/bash
        ;;

    "grc")
        echo "[*] GNU Radio Companion – empty workspace (no Heimdall, no pre-loaded file)"
        echo "    Save .grc files to /workspace → krakensdr/workspace/ on the host"
        echo ""
        export PYTHONPATH="/workspace:${PYTHONPATH:-}"
        export GSETTINGS_BACKEND=memory
        cd /workspace
        exec gnuradio-companion
        ;;

    "grc-heimdall")
        echo "[*] GNU Radio Companion – Heimdall hardware + kraken_doa_main.grc"
        echo "    Opens /workspace/kraken_doa_main.grc automatically."
        echo ""
        start_heimdall_bg "hw"
        export PYTHONPATH="/workspace:${PYTHONPATH:-}"
        export GSETTINGS_BACKEND=memory
        cd /workspace
        exec gnuradio-companion /workspace/kraken_doa_main.grc
        ;;

    "grc-synth")
        echo "[*] GNU Radio Companion – Heimdall synthetic + kraken_doa_main.grc"
        echo "    Opens /workspace/kraken_doa_main.grc automatically."
        echo ""
        start_heimdall_bg "synth"
        export PYTHONPATH="/workspace:${PYTHONPATH:-}"
        export GSETTINGS_BACKEND=memory
        cd /workspace
        exec gnuradio-companion /workspace/kraken_doa_main.grc
        ;;

    "doa")
        echo "[*] KrakenSDR DoA ${NUM_CHANNELS:-2}-antenna (${ARRAY_TYPE:-ULA}) – Heimdall + widget launcher"
        echo "    Runs /workspace/run_doa.py  (env: NUM_CHANNELS, ARRAY_TYPE, CENTER_FREQ, GAIN_DB, ARRAY_DIST)"
        echo ""
        start_heimdall_bg "hw"
        export PYTHONPATH="/workspace:${PYTHONPATH:-}"
        export GSETTINGS_BACKEND=memory
        cd /workspace
        exec python3 /workspace/run_doa.py
        ;;

    "doa-synth")
        echo "[*] KrakenSDR DoA ${NUM_CHANNELS:-2}-antenna (${ARRAY_TYPE:-ULA}) – Heimdall SYNTHETIC + widget launcher"
        echo "    Runs /workspace/run_doa.py  (env: NUM_CHANNELS, ARRAY_TYPE, CENTER_FREQ, GAIN_DB, ARRAY_DIST)"
        echo ""
        start_heimdall_bg "synth"
        export PYTHONPATH="/workspace:${PYTHONPATH:-}"
        export GSETTINGS_BACKEND=memory
        cd /workspace
        exec python3 /workspace/run_doa.py
        ;;

    "synthetic")
        echo "[*] Starting: kraken_synthetic_test.py (no hardware needed)"
        cd "$EXAMPLES"
        python3 kraken_synthetic_test.py
        ;;

    "heimdall")
        echo "[*] Starting: Heimdall DAQ — real hardware mode"
        if [ "$HW_AVAILABLE" -eq 0 ]; then
            echo "[✗] Cannot start: RTL-SDR not found. Use 'heimdall-synth' instead."
            exit 1
        fi
        _detach_rtl_drivers
        cd "$FIRMWARE"
        bash daq_start_sm.sh >/dev/null 2>&1
        echo -n "[*] Waiting for Heimdall on port 5000 "
        for i in $(seq 1 90); do
            sleep 1
            if _port_listening 5000; then
                echo " ready (${i}s)"
                echo "[✓] Heimdall running. Tailing DAQ logs — press Ctrl+C to stop."
                tail -f "$FIRMWARE/_logs/rtl_daq.log" \
                         "$FIRMWARE/_logs/delay_sync.log" \
                         "$FIRMWARE/_logs/iq_server.log" 2>/dev/null | \
                    grep --line-buffered -v "Circular buffer\|race condition\|Likely race" &
                wait
                exit 0
            fi
            echo -n "."
        done
        echo ""
        echo "[✗] Heimdall did not open port 5000 within 90s."
        _dump_logs
        exit 1
        ;;

    "heimdall-synth")
        echo "[*] Starting: Heimdall DAQ — synthetic mode"
        cd "$FIRMWARE"
        bash daq_synthetic_start.sh >/dev/null 2>&1
        echo -n "[*] Waiting for Heimdall on port 5000 "
        for i in $(seq 1 60); do
            sleep 1
            if _port_listening 5000; then
                echo " ready (${i}s)"
                echo "[✓] Heimdall (synthetic) running. Tailing DAQ logs — press Ctrl+C to stop."
                tail -f "$FIRMWARE/_logs/delay_sync.log" \
                         "$FIRMWARE/_logs/iq_server.log" 2>/dev/null &
                wait
                exit 0
            fi
            echo -n "."
        done
        echo ""
        echo "[✗] Heimdall did not open port 5000 within 60s."
        _dump_logs
        exit 1
        ;;

    "stop")
        echo "[*] Stopping Heimdall DAQ..."
        _kill_daq
        ;;

    "bash"|"sh")
        echo "[*] Interactive shell. Available scripts:"
        echo "    python3 $EXAMPLES/kraken_test.py"
        echo "    python3 $EXAMPLES/kraken_fft_display.py"
        echo "    python3 $EXAMPLES/kraken_synthetic_test.py"
        echo ""
        exec /bin/bash
        ;;

    *)
        echo "[!] Unknown command: '${1}'"
        echo "    Valid commands: bash test fft dev grc grc-heimdall grc-synth doa doa-synth synthetic heimdall heimdall-synth stop"
        exit 1
        ;;
esac
