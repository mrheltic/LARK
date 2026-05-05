"""
core.tone_extraction
====================
Narrowband tone detection and pilot extraction for preamble-gated DoA.

This module provides a general-purpose, stateless (no I/O, no globals)
API for burst-mode IQ processing.  All app-specific constants (sample rate,
burst length, preamble structure) are passed as explicit arguments so the
same functions work across different modulations and hardware configurations.

Public API
----------
find_preamble_onset(iq, b_start, n_total, *, fs, win, burst_samples,
                    preamble_tone_hz, known_hz, freq_lock_bw)
    → (onset_sample: int, tone_hz: float)
    Joint time × frequency search for the preamble pure-tone onset.

find_tone_onset(iq, b_start, n_total, *, tone_hz, fs, win,
                burst_samples, preamble_samples)
    → onset_sample: int
    Matched-filter (inner-product coherent detector) for tone onset.

extract_pilot_tone(X, fs, tone_hz, bw_hz)
    → X_filtered: ndarray
    FFT-domain band-pass filter preserving inter-antenna phase.

narrowband_filter_fft(X, fs, tone_hz, bw_hz)
    Alias for extract_pilot_tone.

Typical usage (app wrapper)
---------------------------

    from core.tone_extraction import find_preamble_onset, extract_pilot_tone

    # App wraps core function with its own constants:
    def _find_preamble_onset(iq, b_start, n_total, fs=_FS, win=_WIN, known_hz=None):
        return find_preamble_onset(
            iq, b_start, n_total,
            fs=fs, win=win, known_hz=known_hz,
            burst_samples=_BURST_SAMPLES,
            preamble_tone_hz=_PREAMBLE_TONE_HZ,
        )
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "find_preamble_onset",
    "find_tone_onset",
    "extract_pilot_tone",
    "narrowband_filter_fft",
]


# =============================================================================
# find_preamble_onset
# =============================================================================

def find_preamble_onset(
    iq:               np.ndarray,
    b_start:          int,
    n_total:          int,
    *,
    fs:               float,
    win:              int   = 512,
    known_hz:         float | None = None,
    burst_samples:    int,
    preamble_tone_hz: float,
    freq_lock_bw:     float = 2_000.0,
) -> tuple[int, float]:
    """
    Find (preamble_onset_sample, actual_tone_hz) via joint time × frequency search.

    Why joint search
    ----------------
    The IRA preamble is a PURE TONE (all-zero dibits → constant +π/4 rotation
    per symbol → sinusoid at carrier + Rsym/8).  In a `win`-sample FFT block the
    preamble concentrates ≈100× more power into a single bin than wideband data
    symbols (+20 dB), regardless of where the actual LO offset places the tone.
    No knowledge of the LO offset is therefore required.

    Algorithm
    ---------
    Sweep 50 %-overlap windows over

        [b_start − burst_samples … b_start + burst_samples / 2]

    For each window: compute full complex FFT (not rfft — IQ is complex-valued),
    zero DC, restrict candidate bins to ``±freq_lock_bw`` around ``known_hz``
    when that is known.  The global maximum over ALL windows identifies both the
    preamble-window start-time AND the actual tone frequency.

    Parameters
    ----------
    iq               : (N,) complex 1-D array — reference antenna IQ stream.
    b_start          : energy-detector onset in samples.
    n_total          : total samples in ``iq`` (i.e. ``len(iq)``).
    fs               : ADC sample rate [Hz].
    win              : FFT window length [samples].  512 @ 1.024 MSPS ≈ 0.5 ms.
    known_hz         : if not None, restrict frequency search to
                       ``±freq_lock_bw`` around this value (post-lock).
    burst_samples    : expected burst duration in samples (defines search range).
    preamble_tone_hz : nominal preamble tone frequency [Hz] (fallback when the
                       search window is degenerate).
    freq_lock_bw     : half-width [Hz] of the frequency search window when
                       ``known_hz`` is given.  Default 2 kHz.

    Returns
    -------
    (onset, tone_hz) : (int, float)
        onset    — sample index of the preamble start (slightly before best window).
        tone_hz  — detected tone centre frequency [Hz].
    """
    step     = win // 2
    scan_sta = max(0, b_start - burst_samples)
    # Search forward too (½ burst) so late-firing energy detectors still find onset.
    scan_end = min(b_start + burst_samples // 2, n_total - win)

    if scan_end <= scan_sta:
        fallback = float(known_hz if known_hz is not None else preamble_tone_hz)
        return b_start, fallback

    # Full complex FFT: includes negative frequencies (IQ is complex-valued).
    freqs = np.fft.fftfreq(win, d=1.0 / fs)

    # Frequency search mask: full range on first call, ±freq_lock_bw afterwards.
    if known_hz is not None:
        freq_mask: np.ndarray | None = np.abs(freqs - known_hz) <= freq_lock_bw
        freq_mask[0] = False   # always exclude DC
    else:
        freq_mask = None

    best_pwr  = -1.0
    best_pos  = b_start
    best_freq = float(known_hz if known_hz is not None else preamble_tone_hz)

    for pos in range(scan_sta, scan_end, step):
        seg     = iq[pos: pos + win]
        fft_pwr = np.abs(np.fft.fft(seg)) ** 2
        fft_pwr[0] = 0.0   # zero DC (LO leakage)
        search  = fft_pwr * freq_mask if freq_mask is not None else fft_pwr
        pk_bin  = int(np.argmax(search))
        pwr     = float(fft_pwr[pk_bin])
        if pwr > best_pwr:
            best_pwr  = pwr
            best_pos  = pos
            best_freq = float(freqs[pk_bin])

    # Shift slightly back so the full preamble run-up is captured.
    onset = max(scan_sta, best_pos - win // 4)
    return onset, best_freq


# =============================================================================
# find_tone_onset
# =============================================================================

def find_tone_onset(
    iq:               np.ndarray,
    b_start:          int,
    n_total:          int,
    *,
    tone_hz:          float,
    fs:               float,
    win:              int   = 512,
    burst_samples:    int,
    preamble_samples: int,
) -> int:
    """
    Find the sample position with the highest coherent power at ``tone_hz``.

    Uses a DFT matched filter (inner product with the complex exponential) rather
    than FFT magnitude, giving the optimal single-frequency detector regardless
    of bin alignment.  At 1.024 MHz with win=512 the preamble tone at 3125 Hz
    falls between bins → the matched filter captures 100 % of tone energy vs
    ~56 % for the nearest FFT bin.

    Searches

        [b_start − burst_samples … b_start + preamble_samples]

    so the preamble is found regardless of where in the active slot the
    energy detector fired.

    Parameters
    ----------
    iq               : (N,) complex 1-D array — reference antenna IQ stream.
    b_start          : energy-detector onset in samples.
    n_total          : total samples in ``iq``.
    tone_hz          : preamble tone frequency to match [Hz].
    fs               : ADC sample rate [Hz].
    win              : filter length [samples].
    burst_samples    : look-back range before ``b_start`` [samples].
    preamble_samples : look-forward range after ``b_start`` [samples].

    Returns
    -------
    onset : int  Sample index of the detected preamble start.
    """
    t_arr    = np.arange(win, dtype=np.float64)
    template = np.exp(2j * np.pi * tone_hz / fs * t_arr)

    step     = win // 2
    scan_sta = max(0, b_start - burst_samples)
    scan_end = min(b_start + preamble_samples, n_total - win)

    if scan_end <= scan_sta:
        return b_start

    best_pwr = -1.0
    best_pos =  b_start
    for pos in range(scan_sta, scan_end, step):
        seg = iq[pos: pos + win]
        pwr = float(abs(np.dot(np.conj(seg), template)) ** 2)
        if pwr > best_pwr:
            best_pwr = pwr
            best_pos = pos

    # Shift slightly back so the full preamble run-up is included.
    return max(scan_sta, best_pos - win // 4)


# =============================================================================
# extract_pilot_tone  (FFT-domain bandpass filter)
# =============================================================================

def extract_pilot_tone(
    X:       np.ndarray,
    fs:      float,
    tone_hz: float,
    bw_hz:   float = 10_000.0,
) -> np.ndarray:
    """
    Extract a CW pilot tone from multi-channel IQ data via FFT gating.

    For a beacon transmitted at ``LO_freq + tone_hz``, this filters out-of-band
    interference by:

        1. FFT every antenna channel snapshot (N-point).
        2. Zero all bins outside  [tone_hz − bw_hz/2 … tone_hz + bw_hz/2].
        3. IFFT → narrowband IQ, preserving inter-antenna phase.

    Effective SNR gain = 10 · log10(sample_rate / bw_hz)  [dB].
    At 1.024 MSPS with bw_hz=10 kHz → +20 dB noise rejection.

    The inter-antenna phase relationship is preserved:

        ∠ X_k[f] − ∠ X_0[f] = ∠ a_k(az, el) − ∠ a_0(az, el)

    so all DoA algorithms (MUSIC / Capon / Bartlett) benefit directly.

    Parameters
    ----------
    X       : (n_ant, N) complex IQ sampled at ``fs``.
    fs      : ADC sample rate [Hz].
    tone_hz : pilot tone offset from HW LO [Hz].  May be negative.
    bw_hz   : full extraction window width [Hz].  Default 10 kHz.

    Returns
    -------
    X_nb : (n_ant, N) complex, same dtype as X, narrowband around tone_hz.
    """
    n     = X.shape[1]
    freqs = np.fft.fftfreq(n, d=1.0 / fs)        # (N,) Hz
    mask  = np.abs(freqs - tone_hz) <= bw_hz * 0.5

    X_fft          = np.fft.fft(X, axis=1)
    X_gated        = np.zeros_like(X_fft)
    X_gated[:, mask] = X_fft[:, mask]
    return np.fft.ifft(X_gated, axis=1).astype(X.dtype)


# Alias kept for interface consistency with other modules.
narrowband_filter_fft = extract_pilot_tone
