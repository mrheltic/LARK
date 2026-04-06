#!/usr/bin/env bash
# =============================================================================
#  LARK – setup.sh
#  Full development environment for KrakenSDR + LibreSDR (Iridium DoA)
#
#  Usage:
#    ./setup.sh              # full install
#    ./setup.sh --deps-only  # apt + pip only (skip Heimdall build)
#    ./setup.sh --heimdall   # only build Heimdall DAQ firmware
#    ./setup.sh --submodules # only clone/update git submodules
#    ./setup.sh --sudoers    # only install the Heimdall sudoers rule
#    ./setup.sh --check      # check environment without installing anything
# =============================================================================
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'; CYA='\033[0;36m'; RST='\033[0m'
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
info() { echo -e "${CYA}[→]${RST} $*"; }
warn() { echo -e "${YEL}[!]${RST} $*"; }
fail() { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }

# ── Project root ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv"
VENV_PY="$VENV/bin/python3"
VENV_PIP="$VENV/bin/pip"

MODE="full"
[[ "${1:-}" == "--deps-only"  ]] && MODE="deps"
[[ "${1:-}" == "--heimdall"   ]] && MODE="heimdall"
[[ "${1:-}" == "--submodules" ]] && MODE="submodules"
[[ "${1:-}" == "--sudoers"    ]] && MODE="sudoers"
[[ "${1:-}" == "--check"      ]] && MODE="check"

# =============================================================================
# 1. System dependencies (apt)
# =============================================================================
install_apt_deps() {
    info "Installing system dependencies (apt)..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        python3 python3-pip python3-venv python3-dev \
        python3-numpy python3-scipy python3-matplotlib \
        libusb-1.0-0-dev \
        libxml2-dev libzmq3-dev \
        python3-pyqt5 pyqt5-dev-tools \
        libsoapysdr-dev soapysdr-module-rtlsdr \
        rtl-sdr \
        libiio-dev python3-libiio libiio-utils \
        libad9361-dev \
        tk-dev python3-tk \
        screen tmux
    ok "System dependencies installed"
}

# =============================================================================
# 2. Python virtual environment + pip dependencies
#
# --system-site-packages lets the venv access apt-installed packages
# (python3-libiio, python3-pyqt5, python3-numpy, etc.) that cannot be
# replaced by pip equivalents.  Packages listed in requirements.txt are
# installed on top inside the venv.
# =============================================================================
create_venv() {
    if [[ ! -x "$VENV_PY" ]]; then
        info "Creating virtual environment: $VENV"
        python3 -m venv --system-site-packages "$VENV"
        ok "Virtual environment created: $VENV"
    else
        ok "Virtual environment already exists: $VENV"
    fi
}

install_pip_deps() {
    create_venv
    info "Installing Python packages into $VENV ..."
    "$VENV_PIP" install --upgrade pip
    "$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"
    ok "Python packages installed"
}

# =============================================================================
# 3. Git submodules
# =============================================================================
init_submodules() {
    info "Initialising git submodules..."

    if [[ ! -d "$SCRIPT_DIR/.git" ]]; then
        warn "Not a git repository — cannot initialise submodules."
        return 0
    fi

    # Sync URLs from .gitmodules → .git/config (required when .git/config
    # does not yet have the submodule entries, e.g. after a fresh clone).
    git -C "$SCRIPT_DIR" submodule sync --recursive

    # Clone / update all registered submodules (shallow, 1 commit depth)
    git -C "$SCRIPT_DIR" submodule update --init --recursive --depth 1 || \
        warn "Submodule update partially failed (network unavailable?). Re-run when online."

    ok "Git submodules initialised"
}

# =============================================================================
# 4. Heimdall DAQ firmware – build (from external/heimdall_daq_fw)
# =============================================================================
build_heimdall() {
    local src="$SCRIPT_DIR/external/heimdall_daq_fw"

    if [[ ! -d "$src" ]] || [[ -z "$(ls -A "$src" 2>/dev/null)" ]]; then
        warn "Submodule external/heimdall_daq_fw is empty."
        warn "Run:  ./setup.sh --submodules   then retry:  ./setup.sh --heimdall"
        return 1
    fi

    # Try Firmware first, then Firmware_new
    local fw=""
    for d in "$src/Firmware" "$src/Firmware_new"; do
        if [[ -d "$d/_daq_core" && -f "$d/_daq_core/Makefile" ]]; then
            fw="$d"
            break
        fi
    done
    if [[ -z "$fw" ]]; then
        warn "No buildable Firmware directory found in heimdall_daq_fw."
        return 1
    fi

    info "Building Heimdall DAQ firmware from ${fw}/_daq_core ..."
    make -C "$fw/_daq_core" clean
    make -C "$fw/_daq_core" -j"$(nproc)" 2>&1 | tail -20
    ok "Heimdall built: $fw/_daq_core"
}

# =============================================================================
# 5. Heimdall sudoers rule (NOPASSWD for DAQ scripts)
#
# Without this rule the VS Code task runs `sudo bash daq_start_sm.sh` in a
# non-interactive panel — sudo cannot prompt for a password and the task
# exits immediately.
# =============================================================================
setup_heimdall_sudoers() {
    local SUDOERS_FILE="/etc/sudoers.d/heimdall-daq"
    local EXT="$SCRIPT_DIR/external/heimdall_daq_fw"
    info "Installing Heimdall sudoers rule ($SUDOERS_FILE) ..."
    # Only covers the in-tree submodule (external/…/Firmware and Firmware_new).
    # Also covers all additional operations in start_heimdall.sh:
    #   sysctl (kernel/net buffer tuning), kill (leftover DAQ pids),
    #   lsof (port checks), rm /dev/shm (stale shm), rmmod (rtlsdr drivers),
    #   tee /sys/bus/usb (USB authorized toggle for LIBUSB_ERROR_PIPE reset),
    #   tee /sys/module/usbcore (usbfs_memory_mb = unlimited for multi-dongle).
    {
        echo "mrheltic ALL=(ALL) NOPASSWD: \\"
        echo "  /bin/bash $EXT/Firmware_new/daq_start_sm.sh, \\"
        echo "  /bin/bash $EXT/Firmware_new/_lark_daq_start_sm.sh, \\"
        echo "  /bin/bash $EXT/Firmware_new/daq_synthetic_start.sh, \\"
        echo "  /bin/bash $EXT/Firmware_new/daq_stop.sh, \\"
        echo "  /bin/bash $EXT/Firmware/daq_start_sm.sh, \\"
        echo "  /bin/bash $EXT/Firmware/_lark_daq_start_sm.sh, \\"
        echo "  /bin/bash $EXT/Firmware/daq_synthetic_start.sh, \\"
        echo "  /bin/bash $EXT/Firmware/daq_stop.sh, \\"
        echo "  /usr/bin/chrt *, \\"
        echo "  /usr/sbin/sysctl *, \\"
        echo "  /usr/bin/kill *, \\"
        echo "  /bin/kill *, \\"
        echo "  /usr/bin/lsof *, \\"
        echo "  /usr/sbin/rmmod *, \\"
        echo "  /usr/bin/python3 *, \\"
        echo "  /bin/sh -c echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb, \\"
        echo "  /usr/bin/tee /proc/sys/vm/drop_caches, \\"
        echo "  /usr/bin/tee /proc/sys/vm/drop_caches *, \\"
        echo "  /usr/bin/tee /sys/module/usbcore/parameters/usbfs_memory_mb, \\"
        echo "  /usr/bin/tee /sys/bus/usb/devices/*/*"
    } | sudo tee "$SUDOERS_FILE" > /dev/null
    sudo chmod 440 "$SUDOERS_FILE"
    ok "Heimdall sudoers rule installed ($SUDOERS_FILE)"
}

# =============================================================================
# 6. Environment check
# =============================================================================
check_env() {
    echo ""
    info "=== LARK environment check ==="
    local ok_count=0 fail_count=0

    chk() {
        local label="$1" cmd="$2"
        if eval "$cmd" &>/dev/null; then
            ok "$label"; (( ok_count++ )) || true
        else
            warn "MISSING: $label"; (( fail_count++ )) || true
        fi
    }

    chk "Python 3"              "python3 --version"
    chk "pip3"                  "pip3 --version"
    chk ".venv"                 "[[ -x '$VENV_PY' ]]"
    chk "numpy  (venv)"         "'$VENV_PY' -c 'import numpy'"
    chk "scipy  (venv)"         "'$VENV_PY' -c 'import scipy'"
    chk "matplotlib (venv)"     "'$VENV_PY' -c 'import matplotlib'"
    chk "pyadi-iio / adi (venv)" "'$VENV_PY' -c 'import adi'"
    chk "libiio / iio (venv)"   "'$VENV_PY' -c 'import iio'"
    chk "PyQt5 (system)"        "python3 -c 'from PyQt5 import Qt'"
    chk "rtl-sdr (rtl_test)"    "command -v rtl_test"
    chk "iio_info"              "command -v iio_info"
    chk "sudoers (heimdall)"    "[[ -f /etc/sudoers.d/heimdall-daq ]]"
    chk "Heimdall firmware" \
        "[[ -f '$SCRIPT_DIR/external/heimdall_daq_fw/Firmware/daq_start_sm.sh' ]] || \
         [[ -f '$SCRIPT_DIR/external/heimdall_daq_fw/Firmware_new/daq_start_sm.sh' ]]"
    chk "Heimdall binaries" \
        "[[ -f '$SCRIPT_DIR/external/heimdall_daq_fw/Firmware/_daq_core/rtl_daq.out' ]] || \
         [[ -f '$SCRIPT_DIR/external/heimdall_daq_fw/Firmware_new/_daq_core/rtl_daq.out' ]]"
    chk "iridium-toolkit" \
        "[[ -f '$SCRIPT_DIR/external/iridium-toolkit/iridium-parser.py' ]]"
    chk "gr-iridium (submodule)" \
        "[[ -d '$SCRIPT_DIR/external/gr-iridium' ]] && \
         [[ -n \"\$(ls -A '$SCRIPT_DIR/external/gr-iridium' 2>/dev/null)\" ]]"

    echo ""
    echo -e "  ${GRN}OK: $ok_count${RST}   ${YEL}Missing: $fail_count${RST}"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
echo ""
echo -e "${CYA}════════════════════════════════════════════════════"
echo -e " LARK – KrakenSDR + LibreSDR setup"
echo -e "════════════════════════════════════════════════════${RST}"
echo ""

case "$MODE" in
    full)
        install_apt_deps
        install_pip_deps
        init_submodules
        setup_heimdall_sudoers
        build_heimdall || warn "Heimdall build failed (maybe offline or already installed). Retry: ./setup.sh --heimdall"
        check_env
        ;;
    deps)
        install_apt_deps
        install_pip_deps
        check_env
        ;;
    heimdall)
        build_heimdall
        ;;
    submodules)
        init_submodules
        ;;
    sudoers)
        setup_heimdall_sudoers
        ;;
    check)
        check_env
        ;;
esac

echo -e "${GRN}Setup complete.${RST}"
echo ""
echo "  Next steps:"
echo "  1. Activate venv:       source .venv/bin/activate"
echo "  2. Check KrakenSDR:     krakenSDR/src/config.py"
echo "  3. Check Heimdall DAQ:  krakenSDR/src/config/daq_chain_config.ini"
echo "  4. Check LibreSDR:      libreSDR/src/config.py"
echo "  5. Run via VS Code:     Ctrl+Shift+P → Tasks: Run Task"
echo ""
