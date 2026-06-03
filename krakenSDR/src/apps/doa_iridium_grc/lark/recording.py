"""
recording.py — Live session recorder for KrakenSDR Iridium DOA.

Output layout (one folder per session):

    session_YYYYMMDD_HHMMSS/
        meta.json           session parameters (freq, fs, gain, array geometry)
        raw_iq.npz          raw CPI frames from Kraken  X: (N, n_ant, cpi_size)
        doa_music.npz       DOA spectra (Kraken-style + full 2D)
            spec2d          (M, n_el, n_az) float32  [dB, peak=0]
            doa_az          (M, n_az)     float32  max-over-elevation az cut
            last_spec2d     (n_el, n_az)  latest 2D spectrum
            last_doa_az     (n_az,)        latest 1D azimuth spectrum (like gr-krakensdr)
            az_deg, el_deg, papr_db, snr_db, t  per estimate
            az_grid_deg, el_grid_deg
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["SessionRecorder", "default_record_dir"]


def default_record_dir() -> str:
    """Default: krakenSDR/data/doa_iridium relative to repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "data", "doa_iridium"))


@dataclass
class SessionRecorder:
    """Accumulate raw Kraken CPI frames and DOA spectra; flush on save()."""

    out_dir: str
    freq_hz: float
    fs: float
    gain_db: float
    n_ant: int
    cpi_size: int
    n_az: int
    n_el: int
    el_min_deg: float = 5.0
    el_max_deg: float = 90.0
    record_raw: bool = True
    checkpoint_s: float = 60.0

    session_dir: str = field(init=False)
    _raw_X: list = field(default_factory=list, init=False)
    _raw_t: list = field(default_factory=list, init=False)
    _spec2d: list = field(default_factory=list, init=False)
    _doa_az: list = field(default_factory=list, init=False)
    _az_deg: list = field(default_factory=list, init=False)
    _el_deg: list = field(default_factory=list, init=False)
    _papr_db: list = field(default_factory=list, init=False)
    _snr_db: list = field(default_factory=list, init=False)
    _doa_t: list = field(default_factory=list, init=False)
    _last_spec2d: np.ndarray | None = field(default=None, init=False)
    _last_doa_az: np.ndarray | None = field(default=None, init=False)
    _last_checkpoint: float = field(default=0.0, init=False)
    _start_t: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.out_dir, f"session_{ts}")
        os.makedirs(self.session_dir, exist_ok=True)
        self._write_meta()

    @property
    def enabled(self) -> bool:
        return True

    def _write_meta(self) -> None:
        meta = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "freq_hz": self.freq_hz,
            "fs_hz": self.fs,
            "gain_db": self.gain_db,
            "n_ant": self.n_ant,
            "cpi_size": self.cpi_size,
            "n_az": self.n_az,
            "n_el": self.n_el,
            "el_min_deg": self.el_min_deg,
            "el_max_deg": self.el_max_deg,
            "record_raw": self.record_raw,
        }
        path = os.path.join(self.session_dir, "meta.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def add_raw_frame(self, X: np.ndarray, timestamp: float | None = None) -> None:
        """Store one Kraken CPI frame (n_ant, cpi_size)."""
        if not self.record_raw:
            return
        self._raw_X.append(np.asarray(X, dtype=np.complex64))
        self._raw_t.append(float(timestamp if timestamp is not None else time.time()))
        self._maybe_checkpoint()

    def add_doa(
        self,
        spec2d: np.ndarray,
        az_deg: float,
        el_deg: float,
        *,
        papr_db: float = 0.0,
        snr_db: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        """Store one DOA estimate and its spectrum."""
        spec = np.asarray(spec2d, dtype=np.float32)
        az_spec = np.max(spec, axis=0).astype(np.float32)  # (n_az,) Kraken-style cut

        self._spec2d.append(spec)
        self._doa_az.append(az_spec)
        self._az_deg.append(float(az_deg))
        self._el_deg.append(float(el_deg))
        self._papr_db.append(float(papr_db))
        self._snr_db.append(float(snr_db))
        self._doa_t.append(float(timestamp if timestamp is not None else time.time()))

        self._last_spec2d = spec
        self._last_doa_az = az_spec
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        if self.checkpoint_s <= 0:
            return
        now = time.time()
        if now - self._last_checkpoint >= self.checkpoint_s:
            self.save(tag="_checkpoint")
            self._last_checkpoint = now

    def save(self, tag: str = "") -> dict[str, str]:
        """Write raw_iq.npz and doa_music.npz. Returns paths written."""
        written: dict[str, str] = {}
        suffix = tag

        if self.record_raw and self._raw_X:
            path = os.path.join(self.session_dir, f"raw_iq{suffix}.npz")
            np.savez_compressed(
                path,
                X=np.stack(self._raw_X, axis=0),
                timestamps=np.array(self._raw_t, dtype=np.float64),
                freq_hz=np.int64(self.freq_hz),
                fs_hz=np.float64(self.fs),
                gain_db=np.float32(self.gain_db),
                cpi_size=np.int32(self.cpi_size),
                n_ant=np.int32(self.n_ant),
            )
            written["raw_iq"] = path
            print(f"[REC] {len(self._raw_X)} raw CPI frames → {path}")

        if self._spec2d:
            az_grid = np.linspace(0.0, 360.0, self.n_az, endpoint=False, dtype=np.float32)
            el_grid = np.linspace(
                self.el_min_deg, self.el_max_deg, self.n_el, dtype=np.float32
            )
            path = os.path.join(self.session_dir, f"doa_music{suffix}.npz")
            payload: dict[str, Any] = dict(
                spec2d=np.stack(self._spec2d, axis=0),
                doa_az=np.stack(self._doa_az, axis=0),
                az_deg=np.array(self._az_deg, dtype=np.float32),
                el_deg=np.array(self._el_deg, dtype=np.float32),
                papr_db=np.array(self._papr_db, dtype=np.float32),
                snr_db=np.array(self._snr_db, dtype=np.float32),
                t=np.array(self._doa_t, dtype=np.float64),
                az_grid_deg=az_grid,
                el_grid_deg=el_grid,
                freq_hz=np.int64(self.freq_hz),
            )
            if self._last_spec2d is not None:
                payload["last_spec2d"] = self._last_spec2d
            if self._last_doa_az is not None:
                payload["last_doa_az"] = self._last_doa_az
            np.savez_compressed(path, **payload)
            written["doa_music"] = path
            print(f"[REC] {len(self._spec2d)} DOA spectra → {path}")

        if not written:
            print("[REC] Nothing to save.")
        else:
            elapsed = time.time() - self._start_t
            print(f"[REC] Session dir: {self.session_dir}  ({elapsed:.0f}s elapsed)")

        return written
