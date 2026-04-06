#!/usr/bin/env bash
# =============================================================================
#  start_heimdall.sh — launch Heimdall DAQ firmware (local, no Docker)
#
#  Replicates the startup sequence from the Docker entrypoint:
#    1. Kernel socket buffer tuning (ZMQ throughput)
#    2. Kill leftover DAQ processes + release ports 5000/5001/1130
#    3. Clean POSIX shared memory segments (prevents double-free)
#    4. Detach RTL-SDR kernel drivers (libusb needs exclusive access)
#    5. Copy project config into firmware directory
#    6. Create required firmware subdirectories (_logs, _data_control)
#    7. Check/install sudoers rule
#    8. Run daq_start_sm.sh (or daq_synthetic_start.sh)
#    9. Wait for port 5000 to confirm Heimdall is ready
#   10. Tail logs (Ctrl+C to stop)
#
#  Usage:
#    bash start_heimdall.sh                # real hardware
#    bash start_heimdall.sh --synthetic    # synthetic (no RTL-SDR)
#
#  Must be run from the LARK workspace root OR with LARK_ROOT set.
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'; CYA='\033[0;36m'; RST='\033[0m'
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
info() { echo -e "${CYA}[→]${RST} $*"; }
warn() { echo -e "${YEL}[!]${RST} $*"; }
fail() { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }

SYNTHETIC=0
[[ "${1:-}" == "--synthetic" ]] && SYNTHETIC=1

# ── Resolve LARK root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LARK_ROOT="${LARK_ROOT:-$(dirname "$(dirname "$SCRIPT_DIR")")}"

# ── Acquire sudo credentials once ───────────────────────────────────────────
# Firmware (chrt -f 99, hw_controller.py) and cleanup of root-owned processes
# require root. Prompt once at startup; the 15-minute credential cache covers
# all subsequent sudo calls so they never block mid-initialization.
if ! sudo -n true 2>/dev/null; then
    info "Heimdall needs elevated privileges (chrt, rmmod, hw_controller)."
    info "Please enter your sudo password (cached for ~15 min):"
    sudo -v 2>/dev/null \
        || warn "sudo authentication skipped — some operations may be unavailable."
fi

# ── 1. Kernel socket buffer tuning ──────────────────────────────────────────
# Enlarges OS socket receive/send buffers so ZeroMQ pipes between Heimdall
# and GNU Radio can absorb CPU micro-spikes without dropping IQ frames.
info "Tuning kernel socket buffers for ZMQ throughput..."
if sudo -n sysctl -qw net.core.rmem_max=134217728 2>/dev/null; then
    sudo -n sysctl -qw net.core.wmem_max=134217728 2>/dev/null
    sudo -n sysctl -qw net.core.rmem_default=1048576 2>/dev/null
    sudo -n sysctl -qw net.core.wmem_default=1048576 2>/dev/null
    ok "ZMQ network buffers: rmem/wmem_max = 128 MB"
else
    warn "Cannot tune network buffers (sudo -n sysctl failed — run 'setup.sh --sudoers' or enter sudo password)."
fi

# ── 2. Kill leftover DAQ processes and release ports ────────────────────────
# Use graceful signal 64 (daq_stop.sh / in-firmware halt) first so libusb can
# close its handles cleanly. Falling back to SIGTERM then SIGKILL only if the
# process does not exit within 4 s. This avoids leaving USB endpoints in HALT
# state (which causes LIBUSB_ERROR_PIPE on the next run).
info "Stopping any leftover DAQ processes..."
_killed=0
for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
            hw_controller.py delay_sync.py test_data_synthesizer.py \
            _iq_server_lark_wrap.sh; do
    pids=$(pgrep -f "$proc" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        warn "Stopping leftover: $proc (pids: $pids)"
        sudo kill -64 $pids 2>/dev/null || true   # graceful halt (daq_stop signal)
        _killed=1
    fi
done
if [[ "$_killed" -eq 1 ]]; then
    # Give processes up to 4 s to exit gracefully so libusb closes USB handles
    for _w in 1 2 3 4; do
        sleep 1
        _any=0
        for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
                    hw_controller.py delay_sync.py test_data_synthesizer.py \
                    _iq_server_lark_wrap.sh; do
            pgrep -f "$proc" >/dev/null 2>&1 && { _any=1; break; }
        done
        [[ "$_any" -eq 0 ]] && break
    done
    # Anything still alive gets SIGTERM, then SIGKILL
    for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
                hw_controller.py delay_sync.py test_data_synthesizer.py \
                _iq_server_lark_wrap.sh; do
        pids=$(pgrep -f "$proc" 2>/dev/null || true)
        [[ -n "$pids" ]] && sudo kill -15 $pids 2>/dev/null || true
    done
    sleep 1
    for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
                hw_controller.py delay_sync.py test_data_synthesizer.py \
                _iq_server_lark_wrap.sh; do
        pids=$(pgrep -f "$proc" 2>/dev/null || true)
        [[ -n "$pids" ]] && { warn "Force-killing $proc"; sudo kill -9 $pids 2>/dev/null || true; }
    done
    sleep 1
fi
for port in 5000 5001 1130; do
    pids=$(sudo lsof -ti:"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        warn "Releasing port $port (pids: $pids)"
        sudo kill -15 $pids 2>/dev/null || true
        sleep 1
        pids=$(sudo lsof -ti:"$port" 2>/dev/null || true)
        [[ -n "$pids" ]] && sudo kill -9 $pids 2>/dev/null || true
    fi
done

# ── 3. Clean POSIX shared memory segments ───────────────────────────────────
# Stale segments from a crashed DAQ run cause double-free on next start.
for seg in decimator_out_A decimator_out_B \
           delay_sync_iq_A delay_sync_iq_B \
           delay_sync_hwc_A delay_sync_hwc_B; do
    [[ -e "/dev/shm/$seg" ]] && { rm -f "/dev/shm/$seg" 2>/dev/null || sudo -n rm -f "/dev/shm/$seg" 2>/dev/null; warn "Removed stale shm: $seg"; }
done

# ── 4. Detach RTL-SDR kernel drivers ────────────────────────────────────────
# dvb_usb_rtl28xxu / rtl2832 hold USB interface 0, causing LIBUSB_ERROR_BUSY
# (-6) inside rtl_daq.out. Must be removed before starting Heimdall.
info "Detaching RTL-SDR kernel drivers..."
_removed=0
for mod in dvb_usb_rtl28xxu dvb_usb_v2 rtl2832 rtl2830 r820t; do
    if lsmod 2>/dev/null | grep -q "^${mod} "; then
        info "  rmmod $mod"
        sudo rmmod "$mod" 2>/dev/null || true
        _removed=1
    fi
done
[[ "$_removed" -eq 0 ]] && ok "No RTL-SDR kernel drivers loaded (OK)."

# ── 4.5. Settle udev after rmmod ─────────────────────────────────────────────
# After rmmod, udev processes device-removal events. A short wait ensures all
# kernel-side USB state is fully released before librtlsdr claims the devices.
# NOTE: USBDEVFS_RESET was removed — it causes re-enumeration which reloads the
# kernel driver via udev, and rmmod during driver init leaves endpoints in STALL
# state (LIBUSB_ERROR_PIPE -9). The Docker entrypoint never resets USB devices;
# it only runs rmmod then starts daq_start_sm.sh directly. Since we already use
# kill -64 (graceful shutdown) instead of kill -9, libusb closes its handles
# properly and no endpoint HALT state is left behind.
if [[ $SYNTHETIC -eq 0 ]]; then
    sleep 2   # let udev finish processing the rmmod removal events
    ok "RTL-SDR drivers detached; udev settled."
fi

# Increase usbfs memory limit before handing control to rtl_daq.out.
# RTL-SDR multi-dongle operation requires >16 MB DMA buffers; the kernel
# default of 16 MB causes ENOMEM in async USB transfer setup.
sudo -n sh -c "echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb" 2>/dev/null \
    && ok "usbfs_memory_mb = 0 (unlimited) for multi-dongle DMA." \
    || warn "Could not set usbfs_memory_mb (non-fatal; run 'setup.sh --sudoers' to enable)."

# ── Find firmware directory ──────────────────────────────────────────────────
# krakenSDR/heimdall is a symlink → ../external/heimdall_daq_fw/Firmware.
# It is the canonical in-tree path; the external/ path is kept as fallback.
find_fw() {
    local candidates=(
        "$LARK_ROOT/krakenSDR/heimdall"
        "$LARK_ROOT/external/heimdall_daq_fw/Firmware"
        "$LARK_ROOT/external/heimdall_daq_fw/Firmware_new"
    )
    for d in "${candidates[@]}"; do
        if [[ -f "$d/daq_start_sm.sh" && -f "$d/_daq_core/rtl_daq.out" ]]; then
            echo "$d"; return 0
        fi
    done
    # fallback: any dir with daq_start_sm.sh (even without binaries)
    for d in "${candidates[@]}"; do
        if [[ -f "$d/daq_start_sm.sh" ]]; then
            warn "Firmware at $d has no compiled binaries — run: make -C $d/_daq_core"
            echo "$d"; return 0
        fi
    done
    return 1
}

FW_DIR="$(find_fw)" || fail "Heimdall firmware not found in any candidate path."
info "Firmware: $FW_DIR"

# ── 5. Copy project config into firmware dir ─────────────────────────────────
CFG="$LARK_ROOT/krakenSDR/src/config/daq_chain_config.ini"
if [[ -f "$CFG" ]]; then
    cp "$CFG" "$FW_DIR/daq_chain_config.ini"
    ok  "Config copied → $FW_DIR/daq_chain_config.ini"
else
    warn "Project config not found ($CFG) — using existing firmware config."
fi

# ── 6. Create required firmware subdirectories ───────────────────────────────
# daq_start_sm.sh writes logs and FIFOs into these directories.
# If they don't exist, mkfifo / log redirection will fail silently.
mkdir -p "$FW_DIR/_logs" "$FW_DIR/_data_control"
ok "Firmware subdirectories ready (_logs, _data_control)."

# ── 7. Check/install sudoers rule ────────────────────────────────────────────
SUDOERS="/etc/sudoers.d/heimdall-daq"
if [[ ! -f "$SUDOERS" ]]; then
    warn "Sudoers NOPASSWD rule not found ($SUDOERS)."
    warn "sudo commands will require your password interactively."
    warn "To install it (once): bash $LARK_ROOT/setup.sh --sudoers"
fi

# ── 8. Select script ─────────────────────────────────────────────────────────
if [[ $SYNTHETIC -eq 1 ]]; then
    SCRIPT="daq_synthetic_start.sh"
    MODE_LABEL="SYNTHETIC (no RTL-SDR)"
    MAX_WAIT=60
else
    SCRIPT="daq_start_sm.sh"
    MODE_LABEL="REAL HARDWARE (RTL-SDR)"
    MAX_WAIT=90
fi

[[ -f "$FW_DIR/$SCRIPT" ]] || fail "Script not found: $FW_DIR/$SCRIPT"
info "Mode: $MODE_LABEL"

# ── 8.5. Wrap iq_server.out — SIGPIPE immunity + auto-restart ────────────────
# ROOT CAUSE of "waiting for Heimdall" crash cascade:
#   iq_server.out receives SIGPIPE when a Python client disconnects abruptly
#   (default handler: kill process, zero log output).
#   iq_server exit → bw_delay_sync_iq FIFO closed → delay_sync.py reads EOF →
#   struct.error crash → fw_delay_sync_hwc closed → hw_controller crashes.
#   Port 5000 disappears; all Python clients hang.
#
# Fix: run iq_server.out inside a bash wrapper with SIGPIPE set to SIG_IGN.
# Ignored disposition is inherited by fork-exec'd children, so iq_server.out
# gets EPIPE from send() instead of SIGPIPE → clean disconnect → reconnect loop.
# The outer while-true provides a restart safety net for any other exit reason.
# The original daq_start_sm.sh is never modified.
_WRAP="$FW_DIR/_iq_server_lark_wrap.sh"
cat > "$_WRAP" << 'WRAP_EOF'
#!/usr/bin/env bash
# Auto-restart wrapper for iq_server.out — GENERATED by start_heimdall.sh
# SIGPIPE is ignored: iq_server survives abrupt client disconnects.
# Max 5 rapid restarts (< 10s each) before giving up to avoid fork bombs.
trap "" PIPE
trap "exit 0" TERM INT HUP
cd "$(dirname "$0")"
_rapid=0
while true; do
    _t0=$(date +%s)
    chrt -f 99 ./_daq_core/iq_server.out
    rc=$?
    _elapsed=$(( $(date +%s) - _t0 ))
    echo "[iq_server] exited (rc=${rc}) after ${_elapsed}s at $(date '+%H:%M:%S')" \
        >> _logs/iq_server.log
    if [[ $_elapsed -lt 10 ]]; then
        (( _rapid++ ))
        if [[ $_rapid -ge 5 ]]; then
            echo "[iq_server] FATAL: 5 rapid exits in a row — DAQ chain is dead, giving up." \
                >> _logs/iq_server.log
            exit 1
        fi
        sleep 2
    else
        _rapid=0    # reset counter if it ran for >10s (genuine restart)
        sleep 1
    fi
done
WRAP_EOF
chmod +x "$_WRAP"

# Patch a copy of the start script to use the wrapper instead of the raw binary.
_PATCHED="$FW_DIR/_lark_${SCRIPT}"
sed 's|chrt -f 99 _daq_core/iq_server\.out 2>_logs/iq_server\.log \&|bash _iq_server_lark_wrap.sh 2>>_logs/iq_server.log \&|' \
    "$FW_DIR/$SCRIPT" > "$_PATCHED"
chmod +x "$_PATCHED"
if grep -q "_iq_server_lark_wrap.sh" "$_PATCHED" 2>/dev/null; then
    SCRIPT="_lark_${SCRIPT}"
    ok "iq_server.out: SIGPIPE immune + auto-restart wrapper applied."
else
    warn "Could not patch iq_server.out line (upstream format changed?) — running original."
    rm -f "$_PATCHED"
fi

# Helper: check port LISTEN state without making a TCP connection.
# nc -z triggers accept() in iq_server.out → server closes and re-binds socket
# → brief window where port 5000 is not listening → ConnectionRefusedError.
# ss reads kernel state only, no connection is made.
_port_listening() {
    ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${1}$"
}

# ── Trap: dump logs and stop all DAQ processes on exit ───────────────────────
# IMPORTANT: Use daq_stop.sh (signal 64) for graceful shutdown so libusb
# closes USB handles properly. This prevents LIBUSB_ERROR_PIPE (-9) on the
# next start. Only fall back to SIGKILL after a 4-second grace period.
_on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    echo ""
    echo "[*] Heimdall exiting (rc=${rc}) — stopping DAQ processes gracefully..."
    # Step 1: graceful halt via signal 64 (daq_stop.sh mechanism)
    (cd "$FW_DIR" 2>/dev/null && sudo bash daq_stop.sh 2>/dev/null) || true
    # Step 2: wait up to 4 s for libusb to close handles
    for _gw in 1 2 3 4; do
        sleep 1
        _alive=0
        for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
                    hw_controller.py delay_sync.py _iq_server_lark_wrap.sh; do
            pgrep -f "$proc" >/dev/null 2>&1 && { _alive=1; break; }
        done
        [[ "$_alive" -eq 0 ]] && break
    done
    # Step 3: anything still running → SIGTERM then SIGKILL
    for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
                hw_controller.py delay_sync.py test_data_synthesizer.py \
                _iq_server_lark_wrap.sh; do
        local pids
        pids=$(pgrep -f "$proc" 2>/dev/null || true)
        [[ -n "$pids" ]] && sudo kill -15 $pids 2>/dev/null || true
    done
    sleep 1
    for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
                hw_controller.py delay_sync.py test_data_synthesizer.py \
                _iq_server_lark_wrap.sh; do
        local pids
        pids=$(pgrep -f "$proc" 2>/dev/null || true)
        [[ -n "$pids" ]] && sudo kill -9 $pids 2>/dev/null || true
    done
    echo "══ Last log lines ══════════════════════════════════════════"
    for log in rtl_daq rebuffer decimator delay_sync iq_server hwc daq_start; do
        local f="${FW_DIR}/_logs/${log}.log"
        if [[ -s "$f" ]]; then
            echo "--- ${log}.log ---"
            tail -n 15 "$f"
            echo ""
        fi
    done
    exit "$rc"
}
trap '_on_exit' EXIT INT TERM

# ── 9. Launch DAQ in background, wait for port 5000 ─────────────────────────
info "Starting Heimdall DAQ firmware (cwd: $FW_DIR) ..."
echo ""
cd "$FW_DIR"
sudo bash "$FW_DIR/$SCRIPT" >"$FW_DIR/_logs/daq_start.log" 2>&1 &

echo -n "[→] Waiting for Heimdall on port 5000 "
for i in $(seq 1 "$MAX_WAIT"); do
    sleep 1
    if _port_listening 5000; then
        # iq_server opens port 5000 before delay_sync completes STATE_IQ_CAL.
        # A short sleep avoids the client receiving CAL/DUMMY frames at startup.
        echo " listening! Stabilising (5s for noise-source calibration)..."
        sleep 5
        ok "Heimdall ready after ${i}s — port 5000 open."
        break
    fi
    if [[ "$i" -gt 20 ]]; then
        # Check if all DAQ processes have already died
        _any_alive=0
        for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
                    hw_controller.py delay_sync.py test_data_synthesizer.py; do
            pgrep -f "$proc" >/dev/null 2>&1 && { _any_alive=1; break; }
        done
        if [[ "$_any_alive" -eq 0 ]]; then
            echo ""
            fail "All DAQ processes exited after ${i}s — check _logs/daq_start.log"
        fi
    fi
    echo -n "."
done

if ! _port_listening 5000; then
    echo ""
    fail "Heimdall did not open port 5000 within ${MAX_WAIT}s."
fi

# ── 10. Tail logs — only lines written AFTER DAQ is fully up ─────────────────
# -n 0: skip historical lines entirely (init USB noise, shm warnings, etc.).
# Only content written after the 15s stabilisation period is shown.
# grep -vE: suppress low-level USB / verbose patterns that are not actionable.
#   failed with -9   = LIBUSB_ERROR_PIPE (transient init stall, auto-recovered)
#   cb transfer status: 5 = LIBUSB_TRANSFER_CANCELLED (normal shutdown flush)
#   rtlsdr_demod_*   = librtlsdr raw register I/O noise
#   r82xx_write_arr  = R820T tuner I2C write noise
#   ERROR setting I2C = librtlsdr I2C stall (benign during init)
#   Allocating.*buffers / Found Rafael = librtlsdr init banner
#   Shared memory not exist = delay_sync waiting for shm (transient on start)
#   IQ adjustment vector    = delay_sync startup dump (not useful at runtime)
#   Delay track statistic   = per-frame verbose stats (use delay_sync.log directly
#                             if you need them)
echo ""
ok "Heimdall running. Tailing logs — Ctrl+C to stop."
echo ""
# -n 0: skip historical lines (init USB noise, shm warnings, etc.).
# Only content written after the 15s stabilisation period is shown.
# _FILTER: suppress low-level USB / verbose patterns that are not actionable.
#   failed with -9   = LIBUSB_ERROR_PIPE (transient init stall, auto-recovered)
#   cb transfer status: 5 = LIBUSB_TRANSFER_CANCELLED (normal shutdown flush)
#   rtlsdr_demod_*   = librtlsdr raw register I/O noise
#   r82xx_write_arr  = R820T tuner I2C write noise
#   ERROR setting I2C = librtlsdr I2C stall (benign during init)
#   Allocating / Found Rafael = librtlsdr init banner
#   Shared memory not exist   = delay_sync wait on shm (transient)
#   IQ adjustment vector      = delay_sync startup dump (not useful at runtime)
#   Delay track statistic     = per-frame verbose stats (always 0 sync fails when OK)
_FILTER='Delay track statistic|Circular buffer|race condition|Likely race'
_FILTER+='|rtlsdr_demod_write_reg|rtlsdr_demod_read_reg|r82xx_write_arr'
_FILTER+='|failed with -9|cb transfer status: [15],|ERROR setting I2C'
_FILTER+='|Allocating.*user-space|Found Rafael Micro|Shared memory not exist'
_FILTER+='|INFO:__main__:IQ adjustment|INFO:__main__:Antenna channel'
_FILTER+='|INFO:__main__:IQ samples per|INFO:__main__:Delay synchronizer'
_FILTER+='|^==>'
tail -n 0 -f \
    "$FW_DIR/_logs/rtl_daq.log" \
    "$FW_DIR/_logs/delay_sync.log" \
    "$FW_DIR/_logs/iq_server.log" 2>/dev/null | \
    grep --line-buffered -vE "$_FILTER" &
wait
