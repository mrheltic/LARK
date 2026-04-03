"""
krakenSDR.heimdall_manager
===========================
Process manager for the Heimdall DAQ firmware (KrakenSDR).

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
