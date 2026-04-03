"""
krakenSDR.heimdall_manager
===========================
Process manager per Heimdall DAQ Firmware (KrakenSDR).

Responsabilità
--------------
* Avviare / fermare Heimdall eseguendo ``daq_start_sm.sh`` dalla cartella
  firmware installata in locale (``~/krakensdr/heimdall_daq_fw/Firmware_new/``
  oppure il submodulo nel progetto una volta compilato)
* Opzionalmente copiare la config INI del progetto nella cartella firmware
  prima dell'avvio (sovrascrive quella di default)
* Verificare che le porte TCP 5000/5001 siano raggiungibili prima di
  restituire il controllo all'applicazione
* Esporre un context-manager pulito

Prerequisiti hardware/software
-------------------------------
* RTL-SDR × N antenne collegati via USB
* Heimdall compilato in ``Firmware_new/`` (tutti i ``.out`` presenti)
* ``daq_chain_config.ini`` con ``out_data_iface_type = eth``  ← già ok

Uso
---
::

    from heimdall_manager import HeimdallManager

    # Avvio con config personalizzata del progetto (Iridium 1626.270 MHz):
    with HeimdallManager(config="daq_chain_config.ini") as hdl:
        src = KrakenIQSource(host=hdl.host, port=hdl.data_port)
        ...

    # Avvio sintetico (no RTL-SDR, usa daq_synthetic_start.sh):
    with HeimdallManager(synthetic=True) as hdl:
        ...

    # Avvio senza sovrascrivere la config nel firmware:
    with HeimdallManager(config=None) as hdl:
        ...
"""

from __future__ import annotations

import os
import shutil
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
_CONFIG_DIR = _HERE / "config"

# Cerca la cartella firmware nell'ordine: installazione utente → submodulo locale
_FIRMWARE_CANDIDATES: list[Path] = [
    Path.home() / "krakensdr" / "heimdall_daq_fw" / "Firmware_new",
    _PROJ_ROOT   / "external"  / "heimdall_daq_fw" / "Firmware_new",
    Path.home() / "krakensdr" / "heimdall_daq_fw" / "Firmware",
    _PROJ_ROOT   / "external"  / "heimdall_daq_fw" / "Firmware",
]


def _find_firmware_dir() -> Optional[Path]:
    """Restituisce la prima cartella firmware con ``daq_start_sm.sh``."""
    for d in _FIRMWARE_CANDIDATES:
        if (d / "daq_start_sm.sh").exists():
            return d
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HeimdallManager
# ─────────────────────────────────────────────────────────────────────────────

class HeimdallManager:
    """
    Avvia e gestisce il processo Heimdall DAQ Firmware (nativo, via script).

    Parameters
    ----------
    config : str | Path | None
        Path alla config INI del progetto da copiare nella cartella firmware
        prima dell'avvio.  Se è solo un nome file, viene cercata in
        ``krakenSDR/src/config/``.  Se ``None``, usa la config già presente
        nella cartella firmware (non sovrascrive).
    host : str
        Hostname su cui Heimdall ascolta (``out_data_iface_type = eth``).
        Default: ``"127.0.0.1"``.
    data_port : int
        Porta IQ server.  Default: ``5000``.
    ctrl_port : int
        Porta hardware controller.  Default: ``5001``.
    synthetic : bool
        Se ``True``, usa ``daq_synthetic_start.sh`` (nessun RTL-SDR necessario).
    firmware_dir : Path | None
        Forza una specifica cartella firmware. Se ``None``, rilevata auto.
    verbose : bool
        Se ``True``, stdout/stderr del processo sono visibili nel terminale.
    """

    def __init__(
        self,
        *,
        config:       str | Path | None = "daq_chain_config.ini",
        host:         str               = "127.0.0.1",
        data_port:    int               = 5000,
        ctrl_port:    int               = 5001,
        synthetic:    bool              = False,
        firmware_dir: Optional[Path]    = None,
        verbose:      bool              = False,
    ) -> None:
        self.host      = host
        self.data_port = data_port
        self.ctrl_port = ctrl_port
        self.synthetic = synthetic
        self.verbose   = verbose

        # Cartella firmware
        self._fw: Path = firmware_dir or _find_firmware_dir() or Path()
        if not (self._fw / "daq_start_sm.sh").exists():
            raise FileNotFoundError(
                "Heimdall DAQ firmware non trovato (daq_start_sm.sh mancante).\n"
                "  Cercato in:\n"
                + "\n".join(f"    {p}" for p in _FIRMWARE_CANDIDATES)
            )

        # Config opzionale da copiare nel firmware prima dell'avvio
        self._config_src: Optional[Path] = None
        if config is not None:
            cfg = Path(config)
            if not cfg.is_absolute():
                cfg = _CONFIG_DIR / cfg
            if cfg.exists():
                self._config_src = cfg
            else:
                logger.warning(
                    "[Heimdall] config non trovata: %s — uso quella nel firmware", cfg
                )

        self._proc: Optional[subprocess.Popen] = None

    # ── Avvio ─────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Avvia Heimdall DAQ firmware."""
        if self._proc is not None and self._proc.poll() is None:
            logger.warning("[Heimdall] già in esecuzione (PID %d)", self._proc.pid)
            return

        # Copia config del progetto nella dir firmware
        if self._config_src is not None:
            dst = self._fw / "daq_chain_config.ini"
            shutil.copy2(self._config_src, dst)
            logger.info("[Heimdall] config copiata: %s → %s", self._config_src, dst)

        start_script = "daq_synthetic_start.sh" if self.synthetic else "daq_start_sm.sh"
        script_path  = self._fw / start_script
        if not script_path.exists():
            raise FileNotFoundError(f"Script non trovato: {script_path}")

        sink = None if self.verbose else subprocess.DEVNULL
        cmd  = ["bash", str(script_path)]
        logger.info("[Heimdall] avvio: %s (cwd=%s)", " ".join(cmd), self._fw)

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._fw),      # ← fondamentale: lo script usa path relativi
            stdout=sink,
            stderr=sink,
            preexec_fn=os.setsid,
        )
        logger.info("[Heimdall] PID %d", self._proc.pid)

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Ferma Heimdall (esegue ``daq_stop.sh`` ufficiale, poi SIGTERM/SIGKILL)."""
        if self._proc is None or self._proc.poll() is not None:
            return
        logger.info("[Heimdall] stop (PID %d)", self._proc.pid)

        # Script di stop ufficiale (killa i sottoprocessi correttamente)
        stop_script = self._fw / "daq_stop.sh"
        if stop_script.exists():
            subprocess.run(
                ["bash", str(stop_script)],
                cwd=str(self._fw),
                capture_output=True,
            )

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

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── Port polling ──────────────────────────────────────────────────────────

    def wait_ready(self, timeout: float = 20.0, poll_interval: float = 0.5) -> bool:
        """Blocca finché le porte TCP 5000/5001 sono pronte o scade il timeout."""
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
        mode  = "synth" if self.synthetic else "hw"
        state = f"PID={self._proc.pid}" if self.is_running() else "fermo"
        return (
            f"HeimdallManager(mode={mode!r}, fw={self._fw.name!r}, "
            f"host={self.host!r}, data={self.data_port}, ctrl={self.ctrl_port}, "
            f"state={state})"
        )
