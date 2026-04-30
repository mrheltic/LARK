"""
core.burst_pipeline — Burst detection + demodulation pipeline
==============================================================

Connects the burst *detector* to the Iridium *demodulator* and exposes a
single :class:`BurstPipeline` class with a simple ``process()`` interface.

Architecture
------------
    KrakenIQSource → BurstPipeline.process(x, ts_ms)
        ├─ BurstDetector.process(x) → BurstResult
        └─ [if is_burst] IridiumDemod.demod(x, ...) → RAW: line

Usage
-----
    pipeline = BurstPipeline(input_fs=1_024_000, center_freq_hz=1_626_270_000.0)
    for x, ts_ms in frames:
        result, raw_line = pipeline.process(x, ts_ms)
        if raw_line:
            print(raw_line)  # forward to iridium-parser.py
"""

from __future__ import annotations

__all__ = [
    "PipelineResult",
    "BurstPipeline",
]

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .burst       import BurstDetector, BurstResult, PassTracker
from .iridium_demod import IridiumDemod


# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Combined output from one call to :meth:`BurstPipeline.process`."""
    burst:    BurstResult      # detector output (always present)
    raw_line: Optional[str]    # RAW: demodulated line, or None


# ---------------------------------------------------------------------------
class BurstPipeline:
    """
    All-in-one burst detection + DQPSK demodulation pipeline.

    Thread safety
    ~~~~~~~~~~~~~
    The pipeline is **not** thread-safe.  Call :meth:`process` from a single
    thread.  If you need non-blocking operation, wrap it in a worker thread
    and communicate with the UI via :class:`queue.Queue`.

    Parameters
    ----------
    input_fs : int
        IQ sample rate in Hz (must match the Heimdall DAQ configuration).
    center_freq_hz : float
        SDR tuned centre frequency in Hz.  Added to the measured Doppler shift
        to produce the absolute carrier frequency written into the RAW: line.
    burst_snr : float
        Minimum in-band SNR (dB) for burst declaration.
    burst_papr : float
        Minimum in-band PAPR (dB) for burst declaration.
    burst_pwr : float
        Minimum absolute frame power (dBW) for burst declaration (squelch).
    filename : str
        Label placed in the RAW: line filename field.
    demod_enabled : bool
        Set to ``False`` to run only the detector without the demodulator
        (saves CPU when you only need SNR/Doppler metrics).
    """

    def __init__(
        self,
        *,
        input_fs:       int   = 1_024_000,
        center_freq_hz: float = 1_626_270_000.0,
        fft_n:          int   = 512,
        burst_n:        int   = 4096,
        burst_snr:      float = 8.0,
        burst_papr:     float = 5.0,
        burst_pwr:      float = -90.0,
        filename:       str   = "kraken",
        demod_enabled:  bool  = True,
    ) -> None:
        self._center_freq  = center_freq_hz
        self._filename     = filename
        self._demod_en     = demod_enabled
        self._t0           = time.time()   # epoch for relative timestamps

        self._detector = BurstDetector(
            fs         = input_fs,
            fft_n      = fft_n,
            burst_n    = burst_n,
            burst_snr  = burst_snr,
            burst_papr = burst_papr,
            burst_pwr  = burst_pwr,
        )
        self._tracker = PassTracker()

        if demod_enabled:
            self._demod = IridiumDemod(input_fs=input_fs)
        else:
            self._demod = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    @property
    def detector(self) -> BurstDetector:
        """Direct access to the underlying :class:`BurstDetector`."""
        return self._detector

    @property
    def tracker(self) -> PassTracker:
        """Direct access to the :class:`PassTracker`."""
        return self._tracker

    # ------------------------------------------------------------------
    def process(
        self,
        x:          np.ndarray,
        timestamp:  Optional[float] = None,
        filename:   Optional[str]   = None,
    ) -> PipelineResult:
        """
        Process one IQ frame.

        Parameters
        ----------
        x : np.ndarray (complex)
            Raw IQ samples at ``input_fs``.
        timestamp : float or None
            Frame capture time (seconds since epoch).  Defaults to
            ``time.time()``.  Used to derive the RAW: timestamp field
            (milliseconds relative to pipeline creation).
        filename : str or None
            Override the RAW: filename label for this frame only.

        Returns
        -------
        PipelineResult
            ``.burst`` contains the detector output; ``.raw_line`` is the
            demodulated RAW: string or ``None``.
        """
        now = timestamp if timestamp is not None else time.time()
        ts_ms = (now - self._t0) * 1000.0

        # 1. Burst detection (always runs)
        r = self._detector.process(x)
        self._tracker.update(r.doppler_hz, r.is_burst, now)

        raw_line: Optional[str] = None

        # 2. Demodulate only when a burst is present
        if r.is_burst and self._demod is not None:
            fname = filename if filename is not None else self._filename
            try:
                raw_line = self._demod.demod(
                    x             = x,
                    doppler_hz    = r.doppler_hz,
                    timestamp_ms  = ts_ms,
                    center_freq_hz = self._center_freq,
                    filename      = fname,
                )
            except Exception:
                raw_line = None   # demod errors are non-fatal

        return PipelineResult(burst=r, raw_line=raw_line)
