#!/bin/bash
# ============================================================
#  KrakenSDR Docker – Container launcher
#
#  Usage: bash docker/run.sh [COMMAND]
#
#  Commands:
#    grc             – GNU Radio Companion GUI (no Heimdall)
#    grc-heimdall    – Heimdall HW bg + GRC with kraken_doa_main.grc
#    grc-synth       – Heimdall synthetic bg + GRC with kraken_doa_main.grc
#    doa             – Heimdall HW bg + run_doa.py (N-antenna widget)
#    doa-synth       – Heimdall synthetic bg + run_doa.py
#    dev             – Heimdall bg + interactive shell (/workspace in PYTHONPATH)
#    synthetic       – Heimdall synthetic mode (no hardware)
#    test            – kraken_test.py  (2 ch, real hardware)
#    fft             – kraken_fft_display.py  (5 ch, real hardware)
#    heimdall        – Heimdall DAQ (real hardware)
#    heimdall-synth  – Heimdall DAQ (synthetic mode)
#    bash            – interactive shell (no Heimdall)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="krakensdr-docker:latest"
DOCKERHUB_IMAGE="mrheltic/krakensdr-docker:latest"
COMMAND="${1:-synthetic}"

# ── Check image exists — fallback to Docker Hub pull ────────────────────────
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "[!] Local image '$IMAGE' not found."
    echo "    Pulling from Docker Hub: ${DOCKERHUB_IMAGE} …"
    if docker pull "${DOCKERHUB_IMAGE}"; then
        docker tag "${DOCKERHUB_IMAGE}" "$IMAGE"
        echo "[✓] Image pulled and tagged as $IMAGE"
    else
        echo "[✗] Pull failed. Build it locally with:  bash docker/build.sh"
        exit 1
    fi
fi

# ── X11 forwarding ───────────────────────────────────────────────────────────
# With --network host the container uses the host Unix socket directly.
# xhost + disables access control on the X server for local sockets:
# libX11 inside the container connects without a cookie and the server accepts.
# Do NOT set XAUTHORITY inside the container: if set, libX11 sends the
# host-specific cookie which the X server rejects ("no auth protocol").
X11_ARGS=()
QT_PLATFORM=offscreen
GDK_BACKEND=offscreen
if [ -z "$DISPLAY" ]; then
    echo "[!] DISPLAY not set — GUI will run in offscreen mode."
elif [ "$(docker context show 2>/dev/null)" = "desktop-linux" ]; then
    echo "[!] Docker Desktop (VM): add /tmp to File Sharing to enable GUI."
elif [ ! -d "/tmp/.X11-unix" ]; then
    echo "[!] /tmp/.X11-unix not found — GUI offscreen."
else
    xhost + 2>/dev/null || true
    X11_ARGS=(--volume /tmp/.X11-unix:/tmp/.X11-unix:rw)
    QT_PLATFORM=xcb
    GDK_BACKEND=x11
    echo "[✓] X11 ready (DISPLAY=$DISPLAY)"
fi

# ── Kill leftover DAQ processes from previous container runs ─────────────────
# With --network host, surviving DAQ processes hold ports on the HOST:
#   5000 = iq_server.out (IQ data)
#   5001 = hw_controller.py (hardware control)
#   1130 = rtl_daq.out (ZMQ control)
# The new container's pkill cannot see them (different PID namespace).
_killed=0
# Use pgrep+kill instead of pkill -f: pkill -f would self-match when the
# pattern appears in pkill's own command line, killing the sudo parent process.
for proc in rtl_daq.out rebuffer.out decimate.out iq_server.out \
            hw_controller.py delay_sync.py test_data_synthesizer.py; do
    pids=$(sudo pgrep -f "$proc" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[*] Killed leftover host process: $proc (pids: $pids)"
        sudo kill -9 $pids 2>/dev/null || true
        _killed=1
    fi
done
# Also kill by port in case process name doesn't match
for port in 5000 5001 1130; do
    pids=$(sudo lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[*] Releasing port $port (pids: $pids)"
        sudo kill -9 $pids 2>/dev/null || true
        _killed=1
    fi
done
[ "$_killed" -eq 1 ] && sleep 2

# ── Detach RTL-SDR kernel drivers so libusb can claim the interfaces ────────
# The dvb_usb_rtl28xxu / rtl2832 kernel modules hold USB interface 0, causing
# usb_claim_interface error -6 (LIBUSB_ERROR_BUSY) inside the container.
for mod in dvb_usb_rtl28xxu dvb_usb_v2 rtl2832 rtl2830 r820t; do
    if lsmod 2>/dev/null | grep -q "^${mod} "; then
        echo "[*] Detaching kernel driver: $mod"
        sudo rmmod "$mod" 2>/dev/null || true
    fi
done

# ── Detect RTL-SDR dongles (informational only — --privileged grants access) ─
if lsusb 2>/dev/null | grep -qE "0bda:2838|0bda:2832"; then
    RTL_COUNT=$(lsusb | grep -cE "0bda:2838|0bda:2832")
    echo "[*] RTL-SDR detected: ${RTL_COUNT} dongle(s)"
else
    echo "[!] No RTL-SDR detected on host"
fi

# ── PulseAudio (if available) ───────────────────────────────────────────────
PULSE_ARGS=()
PULSE_SOCKET="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/pulse/native"
if [ -S "$PULSE_SOCKET" ]; then
    PULSE_ARGS=(--env "PULSE_SERVER=unix:${PULSE_SOCKET}" --volume "${PULSE_SOCKET}:${PULSE_SOCKET}")
fi

# ── Local data directories ──────────────────────────────────────────────────
mkdir -p "$ROOT_DIR/recordings"
mkdir -p "$ROOT_DIR/workspace"

# ── Launch container ─────────────────────────────────────────────────────────
echo "[*] Starting container: $IMAGE → $COMMAND"
echo ""

docker run --rm -it \
    `# Terminal: force xterm-256color to avoid garbled/scaled output` \
    --env TERM=xterm-256color \
    --env PYTHONUNBUFFERED=1 \
    `# X11 display` \
    --env DISPLAY="${DISPLAY:-:0}" \
    --env QT_X11_NO_MITSHM=1 \
    --env QT_QPA_PLATFORM="${QT_PLATFORM}" \
    --env GDK_BACKEND="${GDK_BACKEND}" \
    --env XDG_RUNTIME_DIR=/tmp/runtime-kraken \
    "${X11_ARGS[@]}" \
    `# Share host IPC namespace: required for X11 MIT-SHM and Qt rendering` \
    --ipc=host \
    `# Host network — Heimdall tcp 5000/5001 reachable without bridge NAT` \
    --network host \
    `# Full privileges: real-time sched, libusb, rmmod, /dev/shm access` \
    `# (matches godsic/krakensdr-containers reference approach)` \
    --privileged \
    `# Unlimited /dev/shm for DAQ ring buffers` \
    --shm-size=0 \
    `# PulseAudio` \
    "${PULSE_ARGS[@]}" \
    `# Persistent data` \
    --volume "${ROOT_DIR}/recordings":/opt/krakensdr/recordings \
    `# workspace/ on host → /workspace in container: edit scripts on the host,` \
    `# run them in the container without rebuilding. Mounted rw to save .grc/.py.` \
    --volume "${ROOT_DIR}/workspace":/workspace \
    `# Mount delay_sync.py from host: correlation threshold lowered 20→10 dB,` \
    `# no STATE_INIT reset on frequency change from the GUI.` \
    --volume "${ROOT_DIR}/heimdall_daq_fw/Firmware/_daq_core/delay_sync.py":/opt/krakensdr/heimdall_daq_fw/Firmware/_daq_core/delay_sync.py:ro \
    `# Mount entrypoint.sh from host: apply fixes without rebuilding the image` \
    `# (e.g. vmcircbuf GNU Radio setting, future runtime tweaks).` \
    --volume "${SCRIPT_DIR}/entrypoint.sh":/entrypoint.sh:ro \
    `# Mount krakensdr_source.py from host: apply Python fixes without rebuild` \
    --volume "${ROOT_DIR}/gr-krakensdr/python/krakensdr/krakensdr_source.py":/usr/lib/python3/dist-packages/gnuradio/krakensdr/krakensdr_source.py:ro \
    `# Mount the optimised DAQ config (sample_rate 1 024 000, initial gain 40.2 dB)` \
    `# Overrides the config baked into the image.` \
    --volume "${SCRIPT_DIR}/config/daq_chain_config.ini":/opt/krakensdr/heimdall_daq_fw/Firmware/daq_chain_config.ini:ro \
    `# Container name` \
    --name "kraken-${COMMAND}" \
    "$IMAGE" "$COMMAND"