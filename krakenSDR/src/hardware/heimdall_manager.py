"""
krakenSDR.heimdall_manager
===========================
Process manager for the Heimdall DAQ Firmware (KrakenSDR).

Responsibilities
----------------
* Start / stop Heimdall by running ``daq_start_sm.sh`` from the locally
  installed firmware directory (``~/krakensdr/heimdall_daq_fw/Firmware_new/``
  or the project submodule once compiled).
* Optionally copy the project INI config into the firmware directory
  before start (overwrites the default config).
* Wait until TCP ports 5000/5001 are reachable before returning control
  to the application.
* Expose a clean context-manager interface.

Hardware/software prerequisites
---------------------------------
* RTL-SDR × N antennas connected via USB
* Heimdall compiled in ``Firmware_new/`` (all ``.out`` binaries present)
* ``daq_chain_config.ini`` with ``out_data_iface_type = eth``  ← already set

Usage
-----
::

    from heimdall_manager import HeimdallManager

    # Start with project config (Iridium 1626.270 MHz):
    with HeimdallManager(config="daq_chain_config.ini") as hdl:
        src = KrakenIQSource(host=hdl.host, port=hdl.data_port)
        ...

    # Synthetic start (no RTL-SDR, uses daq_synthetic_start.sh):
    with HeimdallManager(synthetic=True) as hdl:
        ...

    # Start without overwriting the firmware config:
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

# Search for the firmware directory in order: user installation → local submodule
_FIRMWARE_CANDIDATES: list[Path] = [
    Path.home() / "krakensdr" / "heimdall_daq_fw" / "Firmware_new",
    _PROJ_ROOT   / "external"  / "heimdall_daq_fw" / "Firmware_new",
    Path.home() / "krakensdr" / "heimdall_daq_fw" / "Firmware",
    _PROJ_ROOT   / "external"  / "heimdall_daq_fw" / "Firmware",
]


def _find_firmware_dir() -> Optional[Path]:
    """Return the first firmware directory that contains ``daq_start_sm.sh``."""
    for d in _FIRMWARE_CANDIDATES:
        if (d / "daq_start_sm.sh").exists():
            return d
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HeimdallManager
# ─────────────────────────────────────────────────────────────────────────────

class HeimdallManager:
    """
    Start and manage the Heimdall DAQ Firmware process (native, via shell script).

    Parameters
    ----------
    config : str | Path | None
        Path to the project INI config to copy into the firmware directory
        before starting.  A bare filename is resolved relative to
        ``krakenSDR/src/config/``.  ``None`` leaves the existing firmware
        config untouched.
    host : str
        Hostname where Heimdall listens (``out_data_iface_type = eth``).
        Default: ``"127.0.0.1"``.
    data_port : int
        IQ server port.  Default: ``5000``.
    ctrl_port : int
        Hardware controller port.  Default: ``5001``.
    synthetic : bool
        If ``True``, use ``daq_synthetic_start.sh`` (no RTL-SDR required).
    firmware_dir : Path | None
        Force a specific firmware directory. ``None`` = auto-detect.
    verbose : bool
        If ``True``, stdout/stderr of the Heimdall process are visible in
        the terminal.
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

        # Firmware directory
        self._fw: Path = firmware_dir or _find_firmware_dir() or Path()
        if not (self._fw / "daq_start_sm.sh").exists():
            raise FileNotFoundError(
                "Heimdall DAQ firmware not found (daq_start_sm.sh missing).\n"
                "  Searched in:\n"
                + "\n".join(f"    {p}" for p in _FIRMWARE_CANDIDATES)
            )

        # Optional config to copy into the firmware dir before starting
        self._config_src: Optional[Path] = None
        if config is not None:
            cfg = Path(config)
            if not cfg.is_absolute():
                cfg = _CONFIG_DIR / cfg
            if cfg.exists():
                self._config_src = cfg
            else:
                logger.warning(
                    "[Heimdall] config not found: %s — using existing firmware config", cfg
                )

        self._proc: Optional[subprocess.Popen] = None

    # ── Start ─────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Heimdall DAQ firmware."""
        if self._proc is not None and self._proc.poll() is None:
            logger.warning("[Heimdall] already running (PID %d)", self._proc.pid)
            return

        # Copy project config into the firmware directory
        if self._config_src is not None:
            dst = self._fw / "daq_chain_config.ini"
            shutil.copy2(self._config_src, dst)
            logger.info("[Heimdall] config copied: %s → %s", self._config_src, dst)

        start_script = "daq_synthetic_start.sh" if self.synthetic else "daq_start_sm.sh"
        script_path  = self._fw / start_script
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        sink = None if self.verbose else subprocess.DEVNULL
        cmd  = ["sudo", "bash", str(script_path)]
        logger.info("[Heimdall] starting: %s (cwd=%s)", " ".join(cmd), self._fw)

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._fw),      # ← critical: the script uses relative paths
            stdout=sink,
            stderr=sink,
            preexec_fn=os.setsid,
        )
        logger.info("[Heimdall] PID %d", self._proc.pid)

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Stop Heimdall (runs ``daq_stop.sh``, then SIGTERM/SIGKILL)."""
        if self._proc is None or self._proc.poll() is not None:
            return
        logger.info("[Heimdall] stop (PID %d)", self._proc.pid)

        # Official stop script (correctly kills all Heimdall sub-processes)
        stop_script = self._fw / "daq_stop.sh"
        if stop_script.exists():
            subprocess.run(
                ["sudo", "bash", str(stop_script)],
                cwd=str(self._fw),
                capture_output=True,
            )

        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("[Heimdall] SIGTERM ignored → SIGKILL")
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
        """Block until TCP ports 5000/5001 are ready or the timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                raise RuntimeError("[Heimdall] process terminated unexpectedly")
            if self._port_open(self.data_port) and self._port_open(self.ctrl_port):
                logger.info(
                    "[Heimdall] ports %d/%d ready", self.data_port, self.ctrl_port
                )
                return True
            time.sleep(poll_interval)
        logger.error("[Heimdall] timeout waiting for ports (%.1f s)", timeout)
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
            raise RuntimeError("[Heimdall] not ready within timeout")
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        mode  = "synth" if self.synthetic else "hw"
        proc  = self._proc
        state = f"PID={proc.pid}" if proc is not None and proc.poll() is None else "stopped"
        return (
            f"HeimdallManager(mode={mode!r}, fw={self._fw.name!r}, "
            f"host={self.host!r}, data={self.data_port}, ctrl={self.ctrl_port}, "
            f"state={state})"
        )
