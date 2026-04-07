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

import math
from typing import Optional, Tuple

import numpy as np

from .burst import MAX_DOP_HZ
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
except (ImportError, AttributeError):
    _MAX_DOP_HZ      = MAX_DOP_HZ   # 40_000.0 Hz from core/burst.py
    _BURST_TOTAL_SYM = 261          # gr-iridium: 8+64+12+167+2+8
    _SYMBOL_RATE     = 25_000       # gr-iridium: SYMBOLS_PER_SECOND


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

    Uses the FFT of Channel 0 only to estimate f_err (Coarse Frequency
    Recovery), then applies the same corrective phasor to all 5 channels.

    Applying the identical phasor to every channel is mathematically mandatory:
    the KrakenSDR is a coherent receiver — all channels see the same carrier
    offset.  A different phasor per channel would corrupt the inter-antenna
    phase differences Δφ = φ_k − φ_0 that encode the DoA information.

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
        Estimated frequency offset [Hz] (positive = satellite approaching).

    Notes
    -----
    The Hann window reduces spectral leakage (sidelobes at −13.3 dB) before
    the FFT.  Three-point parabolic interpolation around the peak brings the
    accuracy to approximately ±48 Hz (half the bin width at 1.024 Msps /
    10_690 samples).  The time index n uses float64: at 40_200 Hz over
    10_690 samples the accumulated phase is ~2141 rad, outside the precision
    range of float32.
    """
    N_burst = burst_matrix.shape[1]
    bin_hz  = sample_rate / N_burst     # larghezza di un bin FFT [Hz]

    # ── Windowed FFT sul Canale 0 ──────────────────────────────────────────
    window  = np.hanning(N_burst)
    x0_win  = burst_matrix[0] * window
    spectrum = np.fft.fftshift(np.fft.fft(x0_win))     # centred at DC
    mag      = np.abs(spectrum)

    # Centred frequency axis [Hz]
    freqs = np.fft.fftshift(np.fft.fftfreq(N_burst, d=1.0 / sample_rate))

    # ── Restrict search to the LEO Doppler band ──────────────────────────────────────
    # Mask ±MAX_DOP_HZ: guards against out-of-band interferers that would
    # otherwise dominate argmax and corrupt the Doppler estimate.
    doppler_mask = np.abs(freqs) <= _MAX_DOP_HZ
    mag_masked   = np.where(doppler_mask, mag, 0.0)

    # ── Main peak ─────────────────────────────────────────────────────────
    pk_idx = int(np.argmax(mag_masked))

    # ── Three-point parabolic sub-bin interpolation ─────────────────────────────────
    # Points: (pk_idx-1, pk_idx, pk_idx+1).  Correction δ ∈ (−0.5, +0.5)
    # minimises the quantisation error introduced by FFT bin discretisation.
    #
    #  δ = 0.5 × (y_{−1} − y_{+1}) / (y_{−1} − 2·y_0 + y_{+1})
    #
    delta = 0.0
    if 1 <= pk_idx <= N_burst - 2:
        y_m = float(mag_masked[pk_idx - 1])
        y_0 = float(mag_masked[pk_idx])
        y_p = float(mag_masked[pk_idx + 1])
        denom = y_m - 2.0 * y_0 + y_p
        if abs(denom) > 1e-20:
            delta = 0.5 * (y_m - y_p) / denom
            delta = max(-0.5, min(0.5, delta))  # clamp for robustness

    f_err = float(freqs[pk_idx]) + delta * bin_hz   # Hz, sub-bin resolution

    # ── Corrective phasor, shape (1, N_burst) to broadcast over 5 channels ──
    # φ(n) = −j·2π·f_err·(n/fs)
    # Same time sequence for all channels (coherent receiver).
    n      = np.arange(N_burst, dtype=np.float64)
    phasor = np.exp(-1j * 2.0 * math.pi * f_err / sample_rate * n)
    phasor = phasor[np.newaxis, :]   # (1, N_burst) → broadcast to (5, N_burst)

    # ── Multiply all channels ─────────────────────────────────────────────────────────
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
