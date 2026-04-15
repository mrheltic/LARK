"""
core.iridium_doa_burst
======================
Single-shot DSP post-processing for Direction-of-Arrival on Iridium TDMA
bursts received by a 5-channel KrakenSDR coherent array.

Problem solved
--------------
Iridium uses TDMA bursts of ~10.44 ms active content every 11.25 ms slot
within a 90 ms super-frame.  The standard KrakenSDR pipeline accumulates the
covariance matrix using an Exponential Moving Average (EMA) over consecutive
frames: this is correct for continuous-wave signals but fails for burst
signals because it averages silent frames (pure noise) together with frames
that contain bursts, degrading the effective SNR of R before MUSIC.

This module provides three pure, stateless functions that:
1. Extract a single burst window from a multi-channel IQ frame
2. Compensate Doppler on all 5 channels in a phase-coherent manner
3. Compute R exclusively over the ~10.44 ms burst window (no EMA)

Pipeline
--------
    (5, 131_072) frame
          │
          ▼  detect_and_extract_burst(frame, threshold_db)
    (5, 10_690) burst  ─── or None (no burst found in frame)
          │
          ▼  compensate_doppler(burst, sample_rate)
    (5, 10_690) compensated,  doppler_hz: float
          │
          ▼  compute_single_shot_covariance(compensated)
     (5, 5) R  ──▶  doa_music(X=compensated, cfg=cfg, R_in=R)

Example usage (post-processing a .npz recording)
-------------------------------------------------
    from core.iridium_doa_burst import (
        detect_and_extract_burst,
        compensate_doppler,
        compute_single_shot_covariance,
    )
    from core.doa_algorithms import ArrayConfig, Geometry, doa_music

    cfg  = ArrayConfig(Nr=5, geometry=Geometry.UCA,
                       radius_lambda=0.358, num_expected_signals=1)
    data = np.load("recording.npz")
    frames = data["frames"]          # shape (n_frames, 5, 131_072), complex128

    for frame in frames:
        burst = detect_and_extract_burst(frame, threshold_db=10.0)
        if burst is None:
            continue
        comp, doppler_hz = compensate_doppler(burst, sample_rate=1_024_000)
        R = compute_single_shot_covariance(comp)
        theta_scan, spec_db = doa_music(X=comp, cfg=cfg,
                                        decorrelation="FBA", R_in=R)
        peak_deg = np.degrees(theta_scan[np.argmax(spec_db)])
        print(f"DoA: {peak_deg:.1f}°   Doppler: {doppler_hz/1e3:+.2f} kHz")

References
----------
* Schmidt, IEEE Trans. Antennas Propagat. 34(3), 1986  — MUSIC
* Harris, Proc. IEEE 66(1), 1978                       — Hann window + parabolic interpolation
* gr-iridium https://github.com/muccc/gr-iridium       — PREAMBLE_LENGTH_LONG = 64 symbols
"""

from __future__ import annotations

import cmath
import math
from typing import Optional, Tuple

import numpy as np

from .burst import MAX_DOP_HZ, PILOT_TONE_OFFSET_HZ, PILOT_TONE_BW_HZ
from .doa_algorithms import covariance

# ---------------------------------------------------------------------------
# System constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE_DEFAULT: int = 1_024_000  # Hz — must match daq_chain_config.ini

# Try to import burst constants from shared.iridium (authoritative source).
# Fall back to hardcoded values confirmed by gr-iridium/lib/iridium.h if
# LARK root is not yet on sys.path at module-import time.
#
#   BURST_TOTAL_SYM = 261 = 8 guard_pre + 64 preamble + 12 UW + 167 data
#                           + 2 tail + 8 guard_post   (gr-iridium confirmed)
#   SYMBOL_RATE     = 25_000 sps                       (gr-iridium confirmed)
#
# Do NOT use TDMA_SLOT_S = 0.00828 from core/burst.py: that value (207 symbols)
# was incorrect.  The active burst spans 261 symbols = 10.44 ms, NOT 8.28 ms.
try:
    import importlib as _il
    _si = _il.import_module("shared.iridium")
    _MAX_DOP_HZ:        float = float(_si.MAX_DOPPLER_HZ)
    _BURST_TOTAL_SYM:   int   = int(_si.BURST_TOTAL_SYM)   # 261 symbols
    _SYMBOL_RATE:       int   = int(_si.SYMBOL_RATE)        # 25_000 sps
    _GUARD_PRE_SYM:     int   = int(_si.GUARD_PRE_SYM)      # 8
    _PREAMBLE_SYM:      int   = int(_si.PREAMBLE_SYM)       # 64
    _UW_SYM:            int   = int(_si.UNIQUE_WORD_SYM)    # 12
except (ImportError, AttributeError):
    _MAX_DOP_HZ      = MAX_DOP_HZ   # 40_000.0 Hz from core/burst.py
    _BURST_TOTAL_SYM = 261          # gr-iridium: 8+64+12+167+2+8
    _SYMBOL_RATE     = 25_000       # gr-iridium: SYMBOLS_PER_SECOND
    _GUARD_PRE_SYM   = 8
    _PREAMBLE_SYM    = 64
    _UW_SYM          = 12

# ---------------------------------------------------------------------------
# Unique-Word constants (from gr-iridium/lib/iridium.h)
# ---------------------------------------------------------------------------
# DQPSK dibit values (0–3) for downlink and uplink sync words.
# Dibit → differential phase: 0 → +π/4, 1 → +3π/4, 2 → −3π/4, 3 → −π/4
_UW_DL: np.ndarray = np.array([0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2], dtype=np.int8)
_UW_UL: np.ndarray = np.array([2, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2], dtype=np.int8)

# Dibit index → expected differential phase increment [rad]
_DIBIT_PHASE_RAD: tuple = (
    math.pi / 4,        # dibit 0
    3.0 * math.pi / 4,  # dibit 1
    -3.0 * math.pi / 4, # dibit 2
    -math.pi / 4,       # dibit 3
)


def _nearest_dibit(diff_phase: float) -> int:
    """Map a differential phase (rad) to the nearest DQPSK dibit (0–3)."""
    # Rotate so that dibit-0 (+π/4) aligns to 0, then quantise in [0, 2π)
    norm = (diff_phase + math.pi / 4) % (2.0 * math.pi)
    idx  = int(norm / (math.pi / 2)) % 4
    # Mapping: quantised 0 → dibit 3, 1 → dibit 0, 2 → dibit 1, 3 → dibit 2
    return (idx - 1) % 4


def _cfo_from_preamble(ch0: np.ndarray, sample_rate: int) -> float:
    """
    Estimate the true Doppler CFO from the IRA preamble pilot tone only.

    The IRA preamble (64 constant dibits=0, Δφ=+π/4 per symbol) generates a
    pure tone at f_carrier + Rs/8 = f_carrier + 3125 Hz.  This function
    scans the first (GUARD_PRE + PREAMBLE) symbols of the burst window
    (widened to handle onset timing jitter), finds the spectral peak, and
    subtracts Rs/8 to return the TRUE carrier Doppler (bias-free).

    Accuracy: ≈ ±(sample_rate / (2 × N_pre)) ≈ ±195 Hz at 1.024 Msps.
    This is better than the full-burst FFT because the preamble tone is
    coherent, while the DQPSK payload spreads ±12.5 kHz and pulls the peak.

    Parameters
    ----------
    ch0 : complex 1-D array   Channel 0 of the burst window (preamble first).
    sample_rate : int

    Returns
    -------
    f_dop : float   True Doppler [Hz] = FFT peak − Rs/8.  Clamped to ±MAX_DOP_HZ.
    """
    pilot_hz = float(_SYMBOL_RATE) / 8.0          # = 3125 Hz  (Rs/8)
    sps_f    = sample_rate / _SYMBOL_RATE           # ≈ 40.96 at 1.024 Msps

    # Wide scan window: include potential guard_pre (silenced but harmless)
    scan_n = min(int((_GUARD_PRE_SYM + _PREAMBLE_SYM) * sps_f), len(ch0))
    if scan_n < 16:
        return 0.0

    x_pre  = ch0[:scan_n]
    N_pre  = len(x_pre)
    window = np.hanning(N_pre)
    spec   = np.fft.fftshift(np.fft.fft(x_pre * window))
    freqs  = np.fft.fftshift(np.fft.fftfreq(N_pre, d=1.0 / sample_rate))
    mag    = np.abs(spec)
    bin_hz = float(sample_rate) / N_pre

    # Pilot sits at f_d + pilot_hz → restrict search band
    f_lo      = -_MAX_DOP_HZ + pilot_hz - 2000.0
    f_hi      =  _MAX_DOP_HZ + pilot_hz + 2000.0
    mask      = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        mask  = np.ones(N_pre, dtype=bool)
    mag_m     = np.where(mask, mag, 0.0)
    pk_idx    = int(np.argmax(mag_m))

    # 3-point parabolic sub-bin interpolation
    delta = 0.0
    if 1 <= pk_idx <= N_pre - 2:
        y_m   = float(mag_m[pk_idx - 1])
        y_0   = float(mag_m[pk_idx])
        y_p   = float(mag_m[pk_idx + 1])
        denom = y_m - 2.0 * y_0 + y_p
        if abs(denom) > 1e-20:
            delta = 0.5 * (y_m - y_p) / denom
            delta = max(-0.5, min(0.5, delta))

    f_peak = float(freqs[pk_idx]) + delta * bin_hz   # = f_d + pilot_hz
    f_dop  = f_peak - pilot_hz                         # = f_d (true Doppler)
    return float(np.clip(f_dop, -_MAX_DOP_HZ, _MAX_DOP_HZ))


# ---------------------------------------------------------------------------
# Function 1 — Time-domain burst extraction
# ---------------------------------------------------------------------------

def detect_and_extract_burst(
    iq_matrix: np.ndarray,
    threshold_db: float = 10.0,
    sample_rate: int = _SAMPLE_RATE_DEFAULT,
) -> Optional[np.ndarray]:
    """
    Locate and extract a single Iridium TDMA burst from a multi-channel IQ frame.

    Computes the power envelope on Channel 0, estimates the noise floor using
    the median (robust because the burst occupies only ~6.5 % of the frame),
    and extracts the temporal window corresponding to BURST_TOTAL_SYM symbols
    starting at the detected onset.

    Parameters
    ----------
    iq_matrix : np.ndarray, shape (5, N), dtype complex
        Coherent 5-channel IQ frame from the KrakenSDR.
        N is typically 131_072 samples (~128 ms at 1.024 Msps).
    threshold_db : float
        Detection threshold in dB above the median noise floor.
        10 dB gives a 10× power margin over noise.
        For weak satellites (low elevation) lower to 6–8 dB.
    sample_rate : int
        Hardware sample rate [Hz].

    Returns
    -------
    np.ndarray, shape (5, N_burst), dtype complex  or  None
        Slice of the frame containing the burst across all 5 channels, or None if:
          - no burst exceeds the threshold in the frame, or
          - the detected onset leaves fewer than N_burst samples before the
            end of the frame (burst truncated at frame boundary).

    Notes
    -----
    N_burst = int(BURST_TOTAL_SYM × sample_rate / SYMBOL_RATE) = 10_690 samples at 1.024 Msps.
    (261 symbols × 1024000 Hz / 25000 sps = 10690.56 → truncated to 10_690)
    The final .copy() is intentional: it prevents the caller from holding the
    entire frame (131_072 × 5 samples) in memory via a NumPy view.
    """
    N_burst = int(_BURST_TOTAL_SYM * sample_rate / _SYMBOL_RATE)
    N       = iq_matrix.shape[1]

    # Instantaneous power envelope on Channel 0 (|x[n]|²)
    power = np.abs(iq_matrix[0]) ** 2   # shape (N,), float64

    # Rectangular smoothing over ~0.2 ms to suppress impulsive noise spikes.
    # At 1.024 Msps: 0.0002 s × 1_024_000 Hz ≈ 205 samples.
    smooth_n = max(1, int(2e-4 * sample_rate))
    kernel   = np.ones(smooth_n, dtype=np.float64) / smooth_n
    smooth   = np.convolve(power, kernel, mode="same")  # shape (N,)

    # Noise floor estimated with the median: robust to the burst being present
    # (burst occupies ~6.5 % of the frame → median falls in the noise).
    noise_floor = float(np.median(smooth))
    if noise_floor < 1e-20:
        return None  # zero-power frame (DAQ in reset or dead channel)

    threshold_linear = noise_floor * (10.0 ** (threshold_db / 10.0))

    above = np.where(smooth > threshold_linear)[0]
    if len(above) == 0:
        return None  # no burst in this frame

    onset = int(above[0])

    # Guard: burst window must not extend past the end of the frame
    if onset + N_burst > N:
        return None

    return iq_matrix[:, onset : onset + N_burst].copy()


# ---------------------------------------------------------------------------
# Function 2 — Multi-channel Doppler compensation (Coarse Frequency Recovery)
# ---------------------------------------------------------------------------

def compensate_doppler(
    burst_matrix: np.ndarray,
    sample_rate: int = _SAMPLE_RATE_DEFAULT,
) -> Tuple[np.ndarray, float]:
    """
    Estimate and remove the Doppler frequency offset from a 5-channel burst.

    Uses the preamble pilot tone (first PREAMBLE_SYM symbols, tone at
    f_carrier + Rs/8 = f_carrier + 3125 Hz) to estimate f_err with sub-bin
    accuracy, then applies the corrective phasor to all 5 channels.

    This is mathematically superior to a full-burst FFT: the preamble IS a
    coherent tone, while the DQPSK payload spreads ±12.5 kHz and would bias
    the peak.  The preamble-only approach is also unbiased: it returns the
    TRUE Doppler (f_d), not f_d + Rs/8.

    Applying the identical phasor to every channel is mathematically mandatory:
    the KrakenSDR is a coherent receiver — all channels see the same carrier
    offset.  A different phasor per channel would corrupt the inter-antenna
    phase differences Δφ = φ_k − φ_0 that encode the DoA information.

    After compensation (f_err = f_d):
      • Carrier is at DC
      • Preamble rotates at +Rs/8 = +3125 Hz  (pilot tone visible at +3125 Hz)
      • UW differentials give the correct DQPSK dibit values  ← key for UW check

    Parameters
    ----------
    burst_matrix : np.ndarray, shape (5, N_burst), dtype complex
        Output of detect_and_extract_burst().
    sample_rate : int
        Hardware sample rate [Hz].

    Returns
    -------
    compensated : np.ndarray, shape (5, N_burst), dtype complex128
        IQ matrix with Doppler removed; inter-antenna phase differences
        are preserved.
    doppler_hz : float
        True estimated Doppler [Hz] (positive = satellite approaching).
        Bias-free (unlike the old full-burst FFT which returned f_d + 3125).
    """
    N_burst = burst_matrix.shape[1]

    # ── Preamble-only CFO estimation (unbiased, high-precision) ──────────────
    f_err = _cfo_from_preamble(burst_matrix[0], sample_rate)  # = true f_d

    # ── Corrective phasor, shape (1, N_burst) to broadcast over 5 channels ──
    # φ(n) = −j·2π·f_err·(n/fs)
    # Same time sequence for all channels (coherent receiver).
    n      = np.arange(N_burst, dtype=np.float64)
    phasor = np.exp(-1j * 2.0 * math.pi * f_err / sample_rate * n)
    phasor = phasor[np.newaxis, :]   # (1, N_burst) → broadcast to (5, N_burst)

    # ── Multiply all channels ────────────────────────────────────────────────
    compensated = (burst_matrix * phasor).astype(np.complex128)

    return compensated, f_err


# ---------------------------------------------------------------------------
# Function 3 — Single-shot spatial covariance (no EMA)
# ---------------------------------------------------------------------------

def compute_single_shot_covariance(
    compensated_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the spatial covariance matrix R over a single burst window.

    R = X · X^H / N

    This is a direct call to doa_algorithms.covariance(), with the explicit
    constraint that NO exponential moving averages (EMA), temporal buffers,
    or data from previous frames are used.  Every call is independent.

    Parameters
    ----------
    compensated_matrix : np.ndarray, shape (5, N_burst), dtype complex
        Output of compensate_doppler().

    Returns
    -------
    R : np.ndarray, shape (5, 5), dtype complex128
        Hermitian spatial covariance matrix, ready to be passed to
        doa_music(..., R_in=R) or scipy.linalg.eigh.

    Notes
    -----
    With N_burst = 10_690 >> Nr² = 25, the ML covariance estimator is
    statistically reliable (samples-to-parameters ratio ~340×).

    Why no EMA:
    The EMA covariance used in the standard pipeline is:
        R_new = α·R_old + (1−α)·R_current
    With α = 0.95 and 128 ms frames, the time constant is ~13 frames ≈ 1.7 s.
    An Iridium burst lasts 8.28 ms; empty frames lower the effective rank of
    R and degrade the MUSIC peak.  The single-shot covariance uses only the
    burst samples: maximum SNR, full rank (5).
    """
    # doa_algorithms.covariance(X) → (X @ X.conj().T) / X.shape[1]
    # No additional logic: the comment above is the contract, not the code.
    return covariance(compensated_matrix)


# ---------------------------------------------------------------------------
# Function 4 — All-bursts extractor (multi-satellite support)
# ---------------------------------------------------------------------------

def detect_and_extract_all_bursts(
    iq_matrix:     np.ndarray,
    threshold_db:  float = 10.0,
    sample_rate:   int   = _SAMPLE_RATE_DEFAULT,
    max_bursts:    int   = 4,
) -> list:
    """
    Find and return ALL Iridium bursts in a multi-channel IQ frame.

    Iridium uses TDMA: multiple satellites may transmit in distinct time
    slots within the same 128 ms KrakenSDR frame.  This function scans
    sequentially: after each burst at offset ``onset``, the search
    continues from ``onset + N_burst + guard`` to find additional bursts.

    Parameters
    ----------
    iq_matrix    : (5, N) complex — coherent 5-channel frame
    threshold_db : dB above median noise floor for detection
    sample_rate  : hardware sample rate [Hz]
    max_bursts   : upper bound on bursts returned per frame

    Returns
    -------
    list of np.ndarray, each shape (5, N_burst), dtype complex
        May be empty.  Order: chronological (by onset time).
    """
    N_burst  = int(_BURST_TOTAL_SYM * sample_rate / _SYMBOL_RATE)
    N        = iq_matrix.shape[1]
    guard    = int(0.001 * sample_rate)   # 1 ms guard gap after each burst

    # Power envelope on ch0 (computed once for the whole frame)
    power    = np.abs(iq_matrix[0]) ** 2
    smooth_n = max(1, int(2e-4 * sample_rate))
    kernel   = np.ones(smooth_n, dtype=np.float64) / smooth_n
    smooth   = np.convolve(power, kernel, mode="same")

    noise_floor = float(np.median(smooth))
    if noise_floor < 1e-20:
        return []

    thr_lin  = noise_floor * (10.0 ** (threshold_db / 10.0))
    results  = []
    search   = 0

    while search < N - N_burst and len(results) < max_bursts:
        above = np.where(smooth[search:] > thr_lin)[0]
        if len(above) == 0:
            break
        onset = search + int(above[0])
        if onset + N_burst > N:
            break
        results.append(iq_matrix[:, onset : onset + N_burst].copy())
        search = onset + N_burst + guard

    return results


# ---------------------------------------------------------------------------
# Function 5 — Burst validation: pilot tone + Unique Word check
# ---------------------------------------------------------------------------

def validate_burst_uw(
    burst: np.ndarray,
    sample_rate: int = _SAMPLE_RATE_DEFAULT,
) -> Tuple[float, float]:
    """
    Validate an Iridium burst by checking the preamble pilot tone and DL UW.

    Call this on the **Doppler-compensated** burst (output of
    :func:`compensate_doppler`).  After compensation with the true Doppler
    f_d (preamble-only estimator):
      • Carrier is at DC
      • Preamble pilot tone is at +Rs/8 = +3125 Hz  ← PILOT_TONE_OFFSET_HZ
      • UW differential phases = Δφ_dibit exactly   ← enables correct decoding

    Parameters
    ----------
    burst : np.ndarray, shape (5, N_burst), dtype complex
        Doppler-compensated 5-channel burst.  Only Channel 0 is used.
        First sample is approximately the preamble onset (guard_pre is
        silence and the power-envelope detector fires at the preamble start).
    sample_rate : int
        Hardware sample rate [Hz].

    Returns
    -------
    pilot_snr_db : float
        SNR [dB] of the preamble pilot tone (+3125 Hz) vs 5–20 kHz noise.
        Values > 6 dB indicate a genuine Iridium IRA preamble.
    uw_score : float
        Fraction of 12 DL-UW dibits that match ``[0,2,2,2,2,0,0,0,2,0,0,2]``
        (range 0.0–1.0; random ≈ 0.25; genuine IRA ≥ 0.67).
        A timing scan over ±4 symbols is performed to handle onset jitter.

    Notes
    -----
    Burst timing (symbols from preamble onset)::

        |←64 preamble→|←12 UW→|←167 data→|←2 tail→|←8 guard_post→|

    The guard_pre (8 silent symbols before the preamble) is NOT captured
    in the burst window because the power detector fires at the preamble
    onset, not at the guard_pre start.
    """
    sps_f = sample_rate / _SYMBOL_RATE          # 40.96 at 1.024 Msps

    x0 = burst[0]   # Channel 0 only
    N  = len(x0)

    # Boundaries based on burst window starting AT preamble (no guard_pre)
    pre_scan_n  = min(int((_GUARD_PRE_SYM + _PREAMBLE_SYM) * sps_f), N)  # wide scan
    preamble_n  = int(_PREAMBLE_SYM * sps_f)                              # nominal end

    # ------------------------------------------------------------------
    # 1. Pilot tone SNR: scan x0[0 : pre_scan_n]
    #    After f_d compensation, preamble pilot sits at +3125 Hz.
    # ------------------------------------------------------------------
    pilot_snr_db: float = 0.0
    if pre_scan_n >= 16:
        pwin   = x0[:pre_scan_n]
        N_pre  = len(pwin)
        fa     = np.abs(np.fft.rfft(pwin * np.hanning(N_pre))) ** 2
        freqs  = np.fft.rfftfreq(N_pre, d=1.0 / sample_rate)

        pilot_mask = np.abs(freqs - PILOT_TONE_OFFSET_HZ) <= PILOT_TONE_BW_HZ
        noise_mask = (freqs >= 5_000.0) & (freqs <= 20_000.0) & ~pilot_mask
        if not np.any(noise_mask):
            noise_mask = freqs >= (freqs[-1] * 0.9)
        if not np.any(noise_mask):
            noise_mask = np.ones(len(freqs), dtype=bool)

        if np.any(pilot_mask) and np.any(noise_mask):
            pilot_power  = float(np.max(fa[pilot_mask]))
            noise_avg    = float(np.mean(fa[noise_mask])) + 1e-30
            pilot_snr_db = float(10.0 * math.log10(pilot_power / noise_avg + 1e-12))

    # ------------------------------------------------------------------
    # 2. UW check with ±4-symbol timing scan for onset-jitter robustness
    #
    #    After f_d compensation (carrier at DC), the differential between
    #    consecutive IQ samples separated by one symbol period is:
    #        diff[k] = x[n_k] × x[n_{k-1}].conj()
    #               ≈ A² × exp(j Δφ_dibit_k)
    #    so _nearest_dibit(angle(diff)) returns the correct dibit directly.
    #
    #    Timing scan: burst onset jitter can place the preamble start up to
    #    ~GUARD_PRE_SYM symbols early in the burst window.  We try offsets
    #    δ ∈ {−4, −3, …, +4} symbols and keep the best UW score.
    # ------------------------------------------------------------------
    best_uw_score = 0.0

    for delta_sym in range(-4, 9):        # scan −4 … +8 symbols
        ref_idx0 = int(round((_PREAMBLE_SYM - 1 + delta_sym + 0.5) * sps_f))
        if ref_idx0 < 0:
            continue

        matches = 0
        valid   = 0
        prev_ok = True
        prev    = x0[ref_idx0] if ref_idx0 < N else None

        for k in range(_UW_SYM):
            sym_idx = int(round((_PREAMBLE_SYM + k + delta_sym + 0.5) * sps_f))
            if sym_idx >= N or prev is None:
                prev_ok = False
                break

            curr = x0[sym_idx]
            if abs(prev) < 1e-20 or abs(curr) < 1e-20:
                prev = curr
                valid += 1
                continue

            diff_phase = cmath.phase(curr * prev.conjugate())
            dibit      = _nearest_dibit(diff_phase)

            if dibit == int(_UW_DL[k]):
                matches += 1
            valid += 1
            prev = curr

        if valid >= _UW_SYM // 2:
            score = float(matches / valid)
            if score > best_uw_score:
                best_uw_score = score

    return pilot_snr_db, best_uw_score
