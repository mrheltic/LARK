"""
krakenSDR.heimdall_manager
===========================
Process manager for the Heimdall DAQ firmware (KrakenSDR).

Responsabilità
--------------
* Avviare / fermare Heimdall come:
    a) processo nativo  (``daq_server`` nel PATH o nel submodulo compilato)
    b) container Docker (``krakensdr-docker:latest``, ``network_mode: host``)
       → questo è il caso standard su Linux con il setup docker esistente
* Verificare che le porte TCP siano raggiungibili prima di restituire il
  controllo all'applicazione
* Esporre un context-manager pulito

Struttura attesa del progetto
------------------------------
::

    LARK/
    ├── external/
    │   └── heimdall_daq_fw/           ← git submodule (compilato da setup.sh)
    ├── krakenSDR/
    │   └── src/
    │       ├── config/
    │       │   ├── daq_chain_config.ini   ← template TDMA Iridium
    │       │   └── daq_chain_config_868.ini
    │       └── heimdall_manager.py    ← questo file

Uso rapido — Docker (default)
------------------------------
::

    from heimdall_manager import HeimdallManager

    # Modalità Docker (synthetic = senza hardware RTL-SDR):
    with HeimdallManager(mode="docker-synth") as hdl:
        src = KrakenIQSource(host=hdl.host, port=hdl.data_port)
        ...

    # Modalità Docker (hardware reale):
    with HeimdallManager(mode="docker") as hdl:
        ...

    # Modalità nativa (daq_server binario nel PATH / submodulo):
    with HeimdallManager(mode="native", config="daq_chain_config.ini") as hdl:
        ...
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import logging
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent                     # krakenSDR/src/
_PROJ_ROOT  = _HERE.parent.parent                       # LARK/
_SUBMODULE  = _PROJ_ROOT / "external" / "heimdall_daq_fw"
_BUILD_BIN  = _SUBMODULE / "build" / "daq_server"
_CONFIG_DIR = _HERE / "config"

# Docker compose file: cerca prima in LARK/external/, poi nella home utente
_DOCKER_COMPOSE_CANDIDATES = [
    _PROJ_ROOT / "external" / "heimdall_daq_fw" / "docker" / "docker-compose.yml",
    Path.home() / "krakensdr" / "docker" / "docker-compose.yml",
]

HeimdallMode = Literal["native", "docker", "docker-synth"]


def _find_heimdall_bin() -> Optional[Path]:
    """Cerca ``daq_server`` (build locale o PATH di sistema)."""
    if _BUILD_BIN.exists():
        return _BUILD_BIN
    import shutil
    sys_bin = shutil.which("daq_server")
    if sys_bin:
        return Path(sys_bin)
    return None


def _find_docker_compose() -> Optional[Path]:
    """Trova il docker-compose.yml di Heimdall."""
    for p in _DOCKER_COMPOSE_CANDIDATES:
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HeimdallManager
# ─────────────────────────────────────────────────────────────────────────────

class HeimdallManager:
    """
    Avvia e gestisce il processo Heimdall DAQ (nativo o Docker).

    Parameters
    ----------
    mode : "native" | "docker" | "docker-synth"
        "native"       : avvia ``daq_server`` binario con la config INI
        "docker"       : avvia il container Docker con hardware reale
        "docker-synth" : avvia il container Docker in modalità sintetica
                         (no RTL-SDR hardware, ottimo per sviluppo/test)
    config : str | Path
        [solo mode="native"] Path alla config INI di Heimdall.
        Se è solo un nome file, viene cercata in ``krakenSDR/src/config/``.
    host : str
        Hostname/IP su cui Heimdall ascolta.  Default: ``"127.0.0.1"``.
    data_port : int
        Porta IQ data.  Default: ``5000``.
    ctrl_port : int
        Porta control.  Default: ``5001``.
    verbose : bool
        Se ``True``, stdout/stderr del processo sono visibili.
    docker_compose : Path | None
        Path al docker-compose.yml. Se None cerca automaticamente.
    """

    def __init__(
        self,
        *,
        mode:           HeimdallMode = "docker-synth",
        config:         str | Path   = "daq_chain_config.ini",
        host:           str          = "127.0.0.1",
        data_port:      int          = 5000,
        ctrl_port:      int          = 5001,
        verbose:        bool         = False,
        docker_compose: Optional[Path] = None,
    ) -> None:
        self.mode      = mode
        self.host      = host
        self.data_port = data_port
        self.ctrl_port = ctrl_port
        self.verbose   = verbose

        # Native-mode config
        config = Path(config)
        if not config.is_absolute():
            config = _CONFIG_DIR / config
        self._config = config

        # Docker-mode compose file
        self._compose: Optional[Path] = docker_compose or _find_docker_compose()

        self._proc: Optional[subprocess.Popen] = None
        self._bin:  Optional[Path]             = _find_heimdall_bin()

    # ── Avvio per modalità ────────────────────────────────────────────────────

    def start(self) -> None:
        """Avvia Heimdall secondo la ``mode`` configurata."""
        if self._proc is not None and self._proc.poll() is None:
            logger.warning("[Heimdall] già in esecuzione (PID %d)", self._proc.pid)
            return

        if self.mode == "native":
            self._start_native()
        elif self.mode == "docker":
            self._start_docker(synthetic=False)
        elif self.mode == "docker-synth":
            self._start_docker(synthetic=True)
        else:
            raise ValueError(f"mode non valido: {self.mode!r}")

    def _start_native(self) -> None:
        if self._bin is None:
            raise RuntimeError(
                "Heimdall (daq_server) non trovato.\n"
                "  1. Esegui ./setup.sh --heimdall per compilare\n"
                "  2. Oppure usa mode='docker' / mode='docker-synth'"
            )
        if not self._config.exists():
            raise FileNotFoundError(
                f"Config Heimdall non trovata: {self._config}"
            )
        sink = None if self.verbose else subprocess.DEVNULL
        cmd = [str(self._bin), str(self._config)]
        logger.info("[Heimdall/native] avvio: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdout=sink, stderr=sink,
            preexec_fn=os.setsid,
        )
        logger.info("[Heimdall/native] PID %d", self._proc.pid)

    def _start_docker(self, synthetic: bool) -> None:
        if self._compose is None:
            raise RuntimeError(
                "docker-compose.yml di Heimdall non trovato.\n"
                f"  Cercato in:\n"
                + "\n".join(f"    {p}" for p in _DOCKER_COMPOSE_CANDIDATES)
                + "\n  Clona ~/krakensdr o passa docker_compose=Path('/path/to/docker-compose.yml')"
            )
        service = "heimdall-synth" if synthetic else "heimdall"
        profile = [] if synthetic else ["--profile", "hardware"]
        sink = None if self.verbose else subprocess.DEVNULL
        cmd = (
            ["docker", "compose", "-f", str(self._compose)]
            + profile
            + ["up", "--no-build", "--remove-orphans", service]
        )
        logger.info("[Heimdall/docker] avvio: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdout=sink, stderr=sink,
            cwd=str(self._compose.parent),
            preexec_fn=os.setsid,
        )
        logger.info("[Heimdall/docker/%s] PID %d", service, self._proc.pid)

    # ── Stop ─────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Ferma Heimdall (SIGTERM → attesa 3 s → SIGKILL)."""
        if self._proc is None or self._proc.poll() is not None:
            return
        logger.info("[Heimdall] stop (PID %d)", self._proc.pid)
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("[Heimdall] SIGTERM ignorato → SIGKILL")
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            self._proc.wait()
        except ProcessLookupError:
            pass
        finally:
            self._proc = None
            # Docker: assicura che il container sia down
            if self.mode in ("docker", "docker-synth") and self._compose:
                subprocess.run(
                    ["docker", "compose", "-f", str(self._compose),
                     "down", "--timeout", "3"],
                    capture_output=True, cwd=str(self._compose.parent),
                )

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── Port polling ──────────────────────────────────────────────────────────

    def wait_ready(self, timeout: float = 20.0, poll_interval: float = 0.5) -> bool:
        """Blocca finché le porte TCP sono pronte o scade il timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                raise RuntimeError("[Heimdall] processo terminato inaspettatamente")
            if self._port_open(self.data_port) and self._port_open(self.ctrl_port):
                logger.info(
                    "[Heimdall] porte %d/%d pronte", self.data_port, self.ctrl_port
                )
                return True
            time.sleep(poll_interval)
        logger.error("[Heimdall] timeout attesa porte (%.1f s)", timeout)
        return False

    def _port_open(self, port: int, timeout: float = 0.2) -> bool:
        try:
            with socket.create_connection((self.host, port), timeout=timeout):
                return True
        except OSError:
            return False

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "HeimdallManager":
        self.start()
        if not self.wait_ready():
            self.stop()
            raise RuntimeError("[Heimdall] non pronto entro il timeout")
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = f"PID={self._proc.pid}" if self.is_running() else "fermo"
        return (
            f"HeimdallManager(mode={self.mode!r}, host={self.host!r}, "
            f"data={self.data_port}, ctrl={self.ctrl_port}, state={state})"
        )


Responsabilità
--------------
* Trovare l'eseguibile ``daq_server`` di Heimdall (submodulo → build o PATH di
  sistema)
* Avviare / fermare Heimdall come sottoprocesso
* Verificare che le porte TCP siano raggiungibili prima di restituire il
  controllo all'applicazione
* Esporre un context-manager pulito per i test

Struttura attesa del progetto
------------------------------
::

    LARK/
    ├── external/
    │   └── heimdall_daq_fw/           ← git submodule
    │       └── build/
    │           └── daq_server         ← compilato da setup.sh
    ├── krakenSDR/
    │   └── src/
    │       ├── config/
    │       │   ├── daq_chain_config.ini   ← template TDMA Iridium
    │       │   └── daq_chain_config_868.ini
    │       └── heimdall_manager.py    ← questo file

Uso rapido
----------
::

    from heimdall_manager import HeimdallManager

    # Avvia con la config Iridium (default), attende che le porte siano pronte
    with HeimdallManager() as hdl:
        src = KrakenIQSource(host=hdl.host, port=hdl.data_port)
        ...
    # Heimdall viene fermato all'uscita del with

    # Oppure manuale:
    hdl = HeimdallManager(config="daq_chain_config.ini", verbose=True)
    hdl.start()
    hdl.wait_ready(timeout=10.0)
    # ... usa KrakenIQSource ...
    hdl.stop()
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent                     # krakenSDR/src/
_PROJ_ROOT  = _HERE.parent.parent                       # LARK/
_SUBMODULE  = _PROJ_ROOT / "external" / "heimdall_daq_fw"
_BUILD_BIN  = _SUBMODULE / "build" / "daq_server"
_CONFIG_DIR = _HERE / "config"


def _find_heimdall_bin() -> Optional[Path]:
    """
    Cerca l'eseguibile ``daq_server`` in ordine di priorità:

    1. Build locale del submodule  (``external/heimdall_daq_fw/build/daq_server``)
    2. PATH di sistema             (``which daq_server``)

    Restituisce ``None`` se non trovato.
    """
    if _BUILD_BIN.exists():
        return _BUILD_BIN

    import shutil
    sys_bin = shutil.which("daq_server")
    if sys_bin:
        return Path(sys_bin)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HeimdallManager
# ─────────────────────────────────────────────────────────────────────────────

class HeimdallManager:
    """
    Avvia e gestisce il processo Heimdall DAQ.

    Parameters
    ----------
    config : str | Path
        Path alla config INI di Heimdall.  Se è solo un nome file viene cercata
        in ``krakenSDR/src/config/``.  Default: ``daq_chain_config.ini``.
    host : str
        Hostname/IP su cui Heimdall ascolta.  Default: ``"127.0.0.1"``.
    data_port : int
        Porta IQ data.  Default: ``5000``.
    ctrl_port : int
        Porta control.  Default: ``5001``.
    verbose : bool
        Se ``True`` invia stdout/stderr di Heimdall all'output dello script
        chiamante.  Default: ``False`` (silenzioso).
    """

    def __init__(
        self,
        *,
        config:    str | Path = "daq_chain_config.ini",
        host:      str  = "127.0.0.1",
        data_port: int  = 5000,
        ctrl_port: int  = 5001,
        verbose:   bool = False,
    ) -> None:
        config = Path(config)
        if not config.is_absolute():
            config = _CONFIG_DIR / config
        if not config.exists():
            raise FileNotFoundError(
                f"Heimdall config not found: {config}\n"
                f"Atteso in: {_CONFIG_DIR}"
            )

        self.config    = config
        self.host      = host
        self.data_port = data_port
        self.ctrl_port = ctrl_port
        self.verbose   = verbose

        self._proc: Optional[subprocess.Popen] = None
        self._bin:  Optional[Path]             = _find_heimdall_bin()

    # ── Processo ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Avvia il processo Heimdall."""
        if self._proc is not None and self._proc.poll() is None:
            logger.warning("[Heimdall] già in esecuzione (PID %d)", self._proc.pid)
            return

        if self._bin is None:
            raise RuntimeError(
                "Heimdall (daq_server) non trovato.\n"
                "  1. Esegui ./setup.sh per compilare il submodulo\n"
                "  2. Oppure installa heimdall_daq_fw e aggiungi al PATH"
            )

        cmd = [str(self._bin), str(self.config)]
        sink = None if self.verbose else subprocess.DEVNULL
        logger.info("[Heimdall] avvio: %s", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdout=sink,
            stderr=sink,
            preexec_fn=os.setsid,   # crea un nuovo gruppo di processi per kill pulito
        )
        logger.info("[Heimdall] PID %d", self._proc.pid)

    def stop(self) -> None:
        """Ferma Heimdall con SIGTERM, poi SIGKILL dopo 3 s."""
        if self._proc is None or self._proc.poll() is not None:
            return

        logger.info("[Heimdall] stop (PID %d)", self._proc.pid)
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            self._proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            logger.warning("[Heimdall] SIGTERM ignorato → SIGKILL")
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            self._proc.wait()
        except ProcessLookupError:
            pass   # il processo è già terminato
        finally:
            self._proc = None

    def is_running(self) -> bool:
        """Ritorna ``True`` se il processo Heimdall è attivo."""
        return self._proc is not None and self._proc.poll() is None

    # ── Port polling ──────────────────────────────────────────────────────────

    def wait_ready(self, timeout: float = 15.0, poll_interval: float = 0.25) -> bool:
        """
        Blocca finché entrambe le porte TCP (data + ctrl) sono raggiungibili o
        scade il timeout.

        Ritorna
        -------
        bool
            ``True`` se le porte sono pronte, ``False`` se timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                raise RuntimeError("[Heimdall] processo terminato inaspettatamente")
            if self._port_open(self.data_port) and self._port_open(self.ctrl_port):
                logger.info(
                    "[Heimdall] porte %d/%d pronte", self.data_port, self.ctrl_port
                )
                return True
            time.sleep(poll_interval)
        logger.error("[Heimdall] timeout attesa porte (%.1f s)", timeout)
        return False

    def _port_open(self, port: int, timeout: float = 0.1) -> bool:
        try:
            with socket.create_connection((self.host, port), timeout=timeout):
                return True
        except OSError:
            return False

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "HeimdallManager":
        self.start()
        if not self.wait_ready():
            self.stop()
            raise RuntimeError("[Heimdall] non pronto entro il timeout")
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        state = f"PID={self._proc.pid}" if self.is_running() else "fermo"
        return (
            f"HeimdallManager(config={self.config.name!r}, "
            f"host={self.host!r}, data={self.data_port}, "
            f"ctrl={self.ctrl_port}, state={state})"
        )
