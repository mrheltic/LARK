"""
core.iridium_demod
==================

Python 3 port of the extractor-python demodulation pipeline from iridium-toolkit.

Converts raw complex IQ samples (at any input sample rate) to decoded DQPSK
symbol bits formatted as the RAW: lines accepted by iridium-parser.py.

Pipeline
--------
1. Rational resampling to 1,000,000 sps  (25 000 sym/s × 40 sps/sym)
2. Downmix by caller-supplied Doppler offset
3. Low-pass channel-select filter  (±50 kHz passband)
4. Signal-start detection  (envelope threshold)
5. RRC matched filter  (α = 0.4, 161 taps)
6. Sync word cross-correlation  → exact burst start sample
7. DQPSK demodulation with first-order timing recovery
8. Gray decode  → bit string
9. RAW: line assembly

Usage::

    demod = IridiumDemod(input_fs=1_024_000)
    line  = demod.demod(iq_array, doppler_hz=3200.0, timestamp_ms=841.3,
                        center_freq_hz=1_626_270_000, filename="kraken")
    if line:
        print(line)           # pipe to iridium-parser.py

Credits
-------
Based on extractor-python/demod.py, cut_and_downmix.py, filters.py and
complex_sync_search.py from the iridium-toolkit project
(https://github.com/muccc/iridium-toolkit), originally (c) the iridium-toolkit
contributors, GPLv3+.  Python 3 port by this project.
"""

from __future__ import annotations

import cmath
import itertools
import math
import re
from math import gcd
from typing import Optional, Tuple

import numpy as np
import scipy.signal

# ---------------------------------------------------------------------------
# Iridium protocol constants
# (mirrored from extractor-python/iridium.py and bitsparser.py)
# ---------------------------------------------------------------------------
SYMBOLS_PER_SECOND: int = 25_000
UW_LENGTH:          int = 12
DOWNLINK:           int = 0
UPLINK:             int = 1

UW_DOWNLINK = "022220002002"
UW_UPLINK   = "220002002022"
_LEAD_OUT   = "100101111010110110110011001111"

# QPSK constellation reference symbols (±1±1j)
_S1 = complex(-1, -1)   # "1"
_S0 = -_S1              # "0"  (+1+1j)

# ---------------------------------------------------------------------------
# Root raised-cosine filter
# (Python 3 port of CommPy / extractor-python/filters.rrcosfilter)
# ---------------------------------------------------------------------------

def _rrcosfilter(n_taps: int, alpha: float, ts: float, fs: float) -> np.ndarray:
    """Return a real-valued RRC FIR filter impulse response."""
    T_delta = 1.0 / fs
    h = np.zeros(n_taps, dtype=float)
    for x in range(n_taps):
        t = (x - n_taps / 2) * T_delta
        if t == 0.0:
            h[x] = 1.0 - alpha + (4.0 * alpha / math.pi)
        elif alpha != 0.0 and abs(abs(t) - ts / (4.0 * alpha)) < 1e-12:
            h[x] = (alpha / math.sqrt(2.0)) * (
                (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * alpha))
                + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * alpha))
            )
        else:
            num = (
                math.sin(math.pi * t * (1.0 - alpha) / ts)
                + 4.0 * alpha * (t / ts) * math.cos(math.pi * t * (1.0 + alpha) / ts)
            )
            den = math.pi * t * (1.0 - (4.0 * alpha * t / ts) ** 2) / ts
            h[x] = num / den if abs(den) > 1e-15 else 0.0
    return h


# ---------------------------------------------------------------------------
# Sync-word matched filter (ComplexSyncSearch port)
# ---------------------------------------------------------------------------
_F_SEARCH = 100   # Hz frequency search radius used when pre-computing templates


class _SyncSearch:
    """
    Cross-correlate an IQ signal against a pre-built Iridium sync-word
    matched filter to locate the burst start.

    Parameters
    ----------
    sample_rate : int
        IQ sample rate in Hz (must give integer samples-per-symbol).
    """

    def __init__(self, sample_rate: int) -> None:
        self._fs  = sample_rate
        self._sps = sample_rate // SYMBOLS_PER_SECOND   # e.g. 40 at 1 Msps

        rrc = _rrcosfilter(161, 0.4, 1.0 / SYMBOLS_PER_SECOND, float(sample_rate))

        # Pre-build matched-filter kernels for both directions at offset = 0 Hz
        self._kernels = {
            DOWNLINK: self._make_kernel(16, DOWNLINK, rrc),
            UPLINK:   self._make_kernel(16, UPLINK,   rrc),
        }

    # ------------------------------------------------------------------
    def _make_kernel(
        self, preamble_len: int, direction: int, rrc: np.ndarray
    ) -> np.ndarray:
        """
        Build a conjugate-reversed, RRC-filtered sync-word template.

        The template covers `preamble_len` preamble symbols + 12 UW symbols.
        """
        sps = self._sps
        if direction == DOWNLINK:
            symbols = [_S0] * preamble_len + [
                _S0, _S1, _S1, _S1, _S1, _S0, _S0, _S0, _S1, _S0, _S0, _S1
            ]
        else:   # UPLINK
            symbols = []
            for _ in range(preamble_len // 2):
                symbols += [_S1, _S0]
            symbols += [_S1, _S1, _S0, _S0, _S0, _S1, _S0, _S0, _S1, _S0, _S1, _S1]

        # Upsample: insert (sps-1) zeros between each symbol
        padded = np.zeros(len(symbols) * sps, dtype=complex)
        for k, s in enumerate(symbols):
            padded[k * sps] = s

        # Shape with RRC
        filtered = np.convolve(padded, rrc, "full")

        # Convert to matched-filter kernel: conj(reversed)
        return np.conj(filtered[::-1])

    # ------------------------------------------------------------------
    def estimate_start(
        self, signal: np.ndarray, direction: int
    ) -> Tuple[int, float, float]:
        """
        Cross-correlate *signal* with the matched filter kernel for *direction*.

        Returns
        -------
        start : int
            Estimated start sample of the unique word (after preamble).
        confidence : float
            Correlation peak amplitude (higher = more confident).
        phase_rad : float
            Phase angle of the correlation peak in radians.  Use to correct
            the 4-fold DQPSK phase ambiguity before demodulation.
        """
        kernel = self._kernels[direction]
        c      = scipy.signal.fftconvolve(signal, kernel, "same")
        mid    = int(np.argmax(np.abs(c)))
        conf      = float(np.abs(c[mid]))
        phase_rad = float(np.angle(c[mid]))

        # Mirror the +2*sps compensation in the original ComplexSyncSearch
        # estimate_sync_word_start() that aligns the argmax to the UW start.
        start = mid + 2 * self._sps
        return max(0, start), conf, phase_rad


# ---------------------------------------------------------------------------
# IridiumDemod — public API
# ---------------------------------------------------------------------------
_burst_counter = itertools.count(1)   # global monotonic burst-ID generator


class IridiumDemod:
    """
    Demodulate a raw IQ buffer (containing at most one Iridium burst) and
    return a ``RAW:`` text line compatible with ``iridium-parser.py``.

    Parameters
    ----------
    input_fs : int
        Sample rate of the input IQ array.  KrakenSDR default is 1 024 000.
    """

    _OUT_FS: int = 1_000_000   # target rate after resampling (40 sps/symbol)

    def __init__(self, input_fs: int = 1_024_000) -> None:
        self._input_fs = input_fs
        self._sps      = self._OUT_FS // SYMBOLS_PER_SECOND  # = 40

        # Rational resampling ratio: input_fs → _OUT_FS
        g          = gcd(input_fs, self._OUT_FS)
        self._up   = self._OUT_FS // g          # 125 for 1024000 → 1000000
        self._dn   = input_fs // g              # 128

        # Pre-computed filters
        self._rrc  = _rrcosfilter(161, 0.4, 1.0 / SYMBOLS_PER_SECOND,
                                  float(self._OUT_FS))
        nyq        = self._OUT_FS / 2.0
        self._lpf  = scipy.signal.firwin(151, 50_000.0 / nyq)   # ±50 kHz

        # Sync word matched filter
        self._sync = _SyncSearch(self._OUT_FS)

        # Lead-in before detected signal start (RRC filter warm-up margin)
        self._lead_in = 8 * self._sps  # 8 symbols

    # ------------------------------------------------------------------
    def demod(
        self,
        x:              np.ndarray,
        doppler_hz:     float,
        timestamp_ms:   float = 0.0,
        center_freq_hz: float = 1_626_270_000.0,
        filename:       str   = "kraken",
    ) -> Optional[str]:
        """
        Attempt to demodulate a captured IQ buffer.

        Parameters
        ----------
        x : np.ndarray (complex)
            Raw IQ samples at ``input_fs``.  Should be at least one full
            Iridium simplex slot long (≈ 8.5 ms → ~8700 samples @ 1.024 Msps).
        doppler_hz : float
            Carrier Doppler offset measured by the burst detector [Hz].
            Used to centre the signal at DC before demodulation.
        timestamp_ms : float
            Burst capture timestamp in milliseconds (used in the RAW: line).
        center_freq_hz : float
            SDR tuned centre frequency [Hz].
        filename : str
            Label inserted into the RAW: line (replaces the file path in the
            original extractor).

        Returns
        -------
        str or None
            A ``RAW:`` line ready for ``iridium-parser.py``, or ``None`` if
            demodulation failed (signal too short, no sync word found, etc.).
        """
        # 1. Resample to _OUT_FS ------------------------------------------------
        if self._up != self._dn:
            y = scipy.signal.resample_poly(
                x, self._up, self._dn
            ).astype(np.complex128)
        else:
            y = np.asarray(x, dtype=np.complex128)

        fs = float(self._OUT_FS)

        # 2. Downmix: shift doppler_hz content to DC ----------------------------
        n = np.arange(len(y), dtype=np.float64)
        y = y * np.exp(-1j * 2.0 * math.pi * doppler_hz / fs * n)

        # 3. Channel-select low-pass filter (suppress adjacent Iridium channels)
        y = scipy.signal.fftconvolve(y, self._lpf, "same")

        # 4. Locate signal start by envelope threshold --------------------------
        sig_start = _find_signal_start(y)
        sig_start = max(0, sig_start - self._lead_in)
        y = y[sig_start:]

        min_len = (UW_LENGTH + 100) * self._sps
        if len(y) < min_len:
            return None   # buffer too short

        # 5. RRC matched filter -------------------------------------------------
        y = scipy.signal.fftconvolve(y, self._rrc, "same")

        # 6. Sync word search (both directions; pick best confidence) -----------
        start_dl, conf_dl, phase_dl = self._sync.estimate_start(y, DOWNLINK)
        start_ul, conf_ul, phase_ul = self._sync.estimate_start(y, UPLINK)

        if conf_dl >= conf_ul:
            sync_start = start_dl
            phase_corr = phase_dl
        else:
            sync_start = start_ul
            phase_corr = phase_ul

        # Apply phase correction: rotate signal so preamble aligns with the
        # template reference phase (+1+1j at 45°), resolving the 4-fold
        # DQPSK carrier phase ambiguity and enabling the UW check to pass.
        n_corr = np.arange(len(y), dtype=np.float64)
        y = y * np.exp(-1j * phase_corr)

        # 7. DQPSK demodulate ---------------------------------------------------
        result = _dqpsk_demod(y, sync_start, self._sps)
        if result is None:
            return None

        dataarray, data_str, access_ok, lead_out_ok, confidence, level, nsymbols = result

        # 8. Assemble RAW: line -------------------------------------------------
        freq_hz  = int(round(center_freq_hz + doppler_hz))
        uid      = next(_burst_counter)
        n_data   = max(0, nsymbols - UW_LENGTH)

        raw_line = (
            f"RAW: {filename} {timestamp_ms:012.4f} {freq_hz:10d} "
            f"A:{'OK' if access_ok else 'no'} "
            f"I:{uid:011d} "
            f"{confidence:3.0f}% {level:.5f} {n_data:3d} "
            f"{data_str}"
        )
        return raw_line


# ---------------------------------------------------------------------------
# Module-level helpers  (static functions kept out of the class for clarity)
# ---------------------------------------------------------------------------

def _find_signal_start(y: np.ndarray) -> int:
    """
    Return the first sample index where the envelope exceeds 30 % of the
    frame maximum.  Uses a short smoothing window to suppress noise spikes.
    """
    mag    = np.abs(y)
    kernel = np.bartlett(51)
    kernel /= kernel.sum()
    smooth = np.convolve(mag, kernel, mode="same")
    peak   = float(np.max(smooth))
    if peak < 1e-20:
        return 0
    indices = np.where(smooth > peak * 0.30)[0]
    return int(indices[0]) if len(indices) else 0


def _qpsk(phase_deg: float) -> Tuple[int, float]:
    """
    Map a phase angle (degrees) to a QPSK symbol index (0–3) and the angular
    offset from the nearest ideal symbol point.

    Decision boundaries are at 89.5°, 179.5°, 269.5°, 359.5° (i.e. shifted by
    0.5° vs. exact 90° multiples) so that floating-point values like 179.9999°
    (computed as exp(j·17π) due to IEEE-754 rounding) still map to index 2
    instead of falling below the 180° boundary into index 1.
    """
    phase_deg = phase_deg % 360.0
    sym    = int((phase_deg + 0.5) / 90.0) % 4
    offset = 45.0 - (phase_deg % 90.0)
    return sym, offset


def _dqpsk_demod(
    signal: np.ndarray,
    start: int,
    sps: int,
) -> Optional[Tuple[list, str, bool, bool, float, float, int]]:
    """
    Differential QPSK demodulation with first-order timing recovery.

    Python 3 port of ``Demod.demod()`` from extractor-python/demod.py.

    Parameters
    ----------
    signal : complex array
        IQ at the demodulation sample rate, RRC-filtered.
    start : int
        Index of the sync-word start (from :class:`_SyncSearch`).
    sps : int
        Samples per symbol.

    Returns
    -------
    (dataarray, data_str, access_ok, lead_out_ok, confidence, level, nsymbols)
    or None on failure.
    """
    skip  = 5 * sps                 # skip first 5 symbols for level reference
    sdiff = 2 if sps >= 20 else 1   # timing nudge step
    alpha = 2.0                     # phase-error tolerance in degrees

    # Level reference: mean amplitude over symbols 5…21 after sync word start
    lvl_start = start + skip
    lvl_end   = lvl_start + 16 * sps
    if lvl_end > len(signal):
        return None

    seg   = signal[lvl_start:lvl_end]
    level = float(np.abs(np.mean(seg)))
    lmax  = float(np.max(np.abs(seg)))

    if level < 1e-15 or lmax < 1e-15:
        return None

    errors:  int       = 0
    symbols: list[int] = []
    i      = start
    phase  = 0.0

    while True:
        # --- Timing recovery (first-order feedback, ported from original) ---
        try:
            cur_r  = signal[i].real
            pre_r  = signal[i - sps].real
            post_r = signal[i + sps].real
            cp_r   = signal[i - sdiff].real
            cn_r   = signal[i + sdiff].real

            if pre_r < 0 < post_r and cur_r > 0:
                if cp_r > cur_r > cn_r:
                    i -= sdiff
                elif cp_r < cur_r < cn_r:
                    i += sdiff
            elif pre_r > 0 > post_r and cur_r < 0:
                if cp_r > cur_r > cn_r:
                    i += sdiff
                elif cp_r < cur_r < cn_r:
                    i -= sdiff
            else:
                cur_i  = signal[i].imag
                pre_i  = signal[i - sps].imag
                post_i = signal[i + sps].imag
                cp_i   = signal[i - sdiff].imag
                cn_i   = signal[i + sdiff].imag

                if pre_i < 0 < post_i and cur_i > 0:
                    if cp_i > cur_i > cn_i:
                        i -= sdiff
                    elif cp_i < cur_i < cn_i:
                        i += sdiff
                elif pre_i > 0 > post_i and cur_i < 0:
                    if cp_i > cur_i > cn_i:
                        i += sdiff
                    elif cp_i < cur_i < cn_i:
                        i -= sdiff
        except IndexError:
            pass   # hit array boundary — carry on with current index

        # --- Decision ---
        ang         = cmath.phase(signal[i]) / math.pi * 180.0
        sym, offset = _qpsk(ang + phase)

        if offset > alpha:
            phase += sdiff
        elif offset < -alpha:
            phase -= sdiff

        if abs(offset) > 22.0:
            errors += 1

        symbols.append(sym)
        i += sps

        if i >= len(signal):
            break
        if abs(signal[i]) < lmax / 8.0:
            break   # signal has faded out

    nsymbols = len(symbols)
    if nsymbols < UW_LENGTH + 20:
        return None   # too few symbols to form a useful frame

    # --- Gray decode: differential symbol transitions → bit pairs ------------
    data:      str       = ""
    dataarray: list[int] = []
    oldsym = 0
    for s in symbols:
        bits = (s - oldsym) % 4
        # Map DQPSK transition index to gray-coded bit pair
        #   0 → 00, 1 → 10, 2 → 11, 3 → 01
        if   bits == 0: bits = 0
        elif bits == 1: bits = 2
        elif bits == 2: bits = 3
        else:           bits = 1
        oldsym = s
        data      += str((bits & 2) >> 1) + str(bits & 1)
        dataarray += [(bits & 2) >> 1, bits & 1]

    # --- Unique-word check ---------------------------------------------------
    access    = "".join(str(s) for s in symbols[:UW_LENGTH])
    access_ok = access in (UW_DOWNLINK, UW_UPLINK)

    lead_out_ok = _LEAD_OUT in data
    confidence  = (1.0 - errors / nsymbols) * 100.0

    # --- Format bit string (mirrors original demod.py output) ---------------
    if access_ok:
        data = "<" + data[: UW_LENGTH * 2] + "> " + data[UW_LENGTH * 2 :]
    if lead_out_ok:
        idx  = data.find(_LEAD_OUT)
        data = (
            data[:idx]
            + "[" + data[idx : idx + len(_LEAD_OUT)] + "]"
            + data[idx + len(_LEAD_OUT) :]
        )
    # Insert a space every 32 bits for readability
    data = re.sub(r"([01]{32})", r"\1 ", data)

    return (dataarray, data, access_ok, lead_out_ok, confidence, level, nsymbols)
