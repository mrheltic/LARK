#!/usr/bin/env bash
# =============================================================================
#  LARK – setup.sh
#  Ambiente di sviluppo completo per KrakenSDR + LibreSDR (Iridium DoA)
#
#  Uso:
#    ./setup.sh              # installazione completa
#    ./setup.sh --deps-only  # solo dipendenze apt/pip (non compila Heimdall)
#    ./setup.sh --heimdall   # solo inizializza e compila Heimdall
#    ./setup.sh --submodules # solo aggiorna i submoduli git
#    ./setup.sh --check      # controlla ambiente senza installare nulla
# =============================================================================
set -euo pipefail

# ── Colori ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'; CYA='\033[0;36m'; RST='\033[0m'
ok()   { echo -e "${GRN}[✓]${RST} $*"; }
info() { echo -e "${CYA}[→]${RST} $*"; }
warn() { echo -e "${YEL}[!]${RST} $*"; }
fail() { echo -e "${RED}[✗]${RST} $*" >&2; exit 1; }

# ── Directory progetto ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="full"
[[ "${1:-}" == "--deps-only"  ]] && MODE="deps"
[[ "${1:-}" == "--heimdall"   ]] && MODE="heimdall"
[[ "${1:-}" == "--submodules" ]] && MODE="submodules"
[[ "${1:-}" == "--check"      ]] && MODE="check"

# =============================================================================
# 1. Dipendenze di sistema (apt)
# =============================================================================
install_apt_deps() {
    info "Installazione dipendenze apt..."
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
    ok "Dipendenze apt installate"
}

# =============================================================================
# 2. Dipendenze Python (pip)
# =============================================================================
install_pip_deps() {
    info "Installazione dipendenze Python..."
    pip3 install --user --upgrade pip
    pip3 install --user -r "$SCRIPT_DIR/requirements.txt"
    ok "Dipendenze Python installate"
}

# =============================================================================
# 3. Submoduli git
# =============================================================================
init_submodules() {
    info "Inizializzazione submoduli git..."

    # Verifica che siamo in un repo git
    if [[ ! -d "$SCRIPT_DIR/.git" ]]; then
        warn "Non sembra un repo git. Inizializzazione..."
        git -C "$SCRIPT_DIR" init
    fi

    # Aggiungi i submoduli se non esistono già
    add_submodule() {
        local url="$1" path="$2"
        if [[ ! -f "$SCRIPT_DIR/$path/.git" && ! -d "$SCRIPT_DIR/$path/.git" ]]; then
            info "Aggiunta submodulo: $path"
            git -C "$SCRIPT_DIR" submodule add --depth 1 "$url" "$path" || \
                warn "Submodulo $path già presente o non raggiungibile."
        fi
    }

    mkdir -p "$SCRIPT_DIR/external"

    add_submodule \
        "https://github.com/krakenrf/heimdall_daq_fw.git" \
        "external/heimdall_daq_fw"

    add_submodule \
        "https://github.com/muccc/iridium-toolkit.git" \
        "external/iridium-toolkit"

    add_submodule \
        "https://github.com/muccc/gr-iridium.git" \
        "external/gr-iridium"

    git -C "$SCRIPT_DIR" submodule update --init --recursive --depth 1 || \
        warn "Aggiornamento submoduli parziale (rete assente?)."

    ok "Submoduli git pronti"
}

# =============================================================================
# 4. Heimdall DAQ firmware – build
# =============================================================================
build_heimdall() {
    local src="$SCRIPT_DIR/external/heimdall_daq_fw"
    local bld="$src/build"

    if [[ ! -d "$src" ]]; then
        warn "Submodulo heimdall_daq_fw non trovato in $src"
        warn "Esegui: ./setup.sh --submodules  prima di --heimdall"
        return 1
    fi

    # Se il submodulo è vuoto (solo .git), fai un checkout
    if [[ ! -f "$src/CMakeLists.txt" ]]; then
        git -C "$SCRIPT_DIR" submodule update --init "$src" || \
            fail "Impossibile inizializzare heimdall_daq_fw"
    fi

    if [[ -f "$bld/daq_server" ]]; then
        ok "Heimdall già compilato: $bld/daq_server"
        return 0
    fi

    info "Compilazione Heimdall DAQ firmware..."
    mkdir -p "$bld"
    cmake -S "$src" -B "$bld" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$bld" --parallel "$(nproc)"
    ok "Heimdall compilato: $bld/daq_server"
}

# =============================================================================
# 5. Check ambiente
# =============================================================================
check_env() {
    echo ""
    info "=== Verifica ambiente LARK ==="
    local ok_count=0 fail_count=0

    chk() {
        local label="$1" cmd="$2"
        if eval "$cmd" &>/dev/null; then
            ok "$label"; (( ok_count++ )) || true
        else
            warn "NON TROVATO: $label"; (( fail_count++ )) || true
        fi
    }

    chk "Python 3"          "python3 --version"
    chk "pip3"              "pip3 --version"
    chk "numpy"             "python3 -c 'import numpy'"
    chk "scipy"             "python3 -c 'import scipy'"
    chk "matplotlib"        "python3 -c 'import matplotlib'"
    chk "pyadi-iio (adi)" \
        "python3 -c 'import adi' || \
         [[ -x '$SCRIPT_DIR/.venv/bin/python3' ]] && '$SCRIPT_DIR/.venv/bin/python3' -c 'import adi'"
    chk "libiio (iio)"      "python3 -c 'import iio'"
    chk "PyQt5"             "python3 -c 'from PyQt5 import Qt'"
    chk "rtl-sdr (rtl_test)" "command -v rtl_test"
    chk "iio_info"          "command -v iio_info"
    chk "Heimdall (daq_start_sm.sh)" \
        "[[ -f \"\$HOME/krakensdr/heimdall_daq_fw/Firmware_new/daq_start_sm.sh\" ]] || \
         [[ -f '$SCRIPT_DIR/external/heimdall_daq_fw/Firmware_new/daq_start_sm.sh' ]]"
    chk "iridium-toolkit" \
        "[[ -f '$SCRIPT_DIR/external/iridium-toolkit/iridium-parser.py' ]] || \
         [[ -f \"\$HOME/krakensdr/iridium-toolkit/iridium-parser.py\" ]]"
    chk "gr-iridium" \
        "[[ -d '$SCRIPT_DIR/external/gr-iridium' ]] || python3 -c 'import iridium' 2>/dev/null || \
         command -v grgsm_decode 2>/dev/null || python3 -c 'import gnuradio' 2>/dev/null"

    echo ""
    echo -e "  ${GRN}OK: $ok_count${RST}   ${YEL}Mancanti: $fail_count${RST}"
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
        build_heimdall || warn "Build Heimdall fallita (forse offline); riprova con rete."
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
    check)
        check_env
        ;;
esac

echo -e "${GRN}Setup completato.${RST}"
echo ""
echo "  Prossimi passi:"
echo "  1. Verifica la config: krakenSDR/src/config.py"
echo "  2. Verifica i parametri Heimdall: krakenSDR/src/config/daq_chain_config.ini"
echo "  3. Verifica la config LibreSDR: libreSDR/src/config.py"
echo "  4. Avvia tutto con VS Code Tasks (Ctrl+Shift+P → Run Task)"
echo ""
