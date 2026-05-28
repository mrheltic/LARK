"""
burst_processing.py — Reusable signal-processing pipeline for Iridium IRA bursts.

Public API
----------
detect_energy_bursts(iq, fs, energy_window, threshold_factor, min_gap_samples)
    Energy-based burst detector.  Returns list of sample offsets.

scan_preamble_tones(iq, fs, nom_tone_hz, scan_bw_hz, n_peaks, min_sep_hz,
                    min_snr_db, dc_guard_hz)
    FFT scan for CW preamble tones.  Returns list of (tone_hz, snr_db).

compute_mf_covariance(X_cal, tone_hz, fs, n_pre, bpf_guard)
    Matched-filter (MF) rank-1 covariance for a CW preamble segment.
    Returns (R_mf, y_mf, mf_snr_db).

apply_bpf_and_normalize(X_win, pre_samples, fs, tone_hz, bpf_bw_hz)
    BPF extraction + amplitude normalization + phase calibration wrapper.
    Returns X_cal (n_ant, pre_samples).

These functions are independent of hardware and config — all parameters are
explicit so they can be unit-tested with synthetic data.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "detect_energy_bursts",
    "scan_preamble_tones",
    "compute_mf_covariance",
    "apply_bpf_and_normalize",
]


# =============================================================================
# 1. Energy-based burst detector
# =============================================================================

def detect_energy_bursts(
    iq: np.ndarray,
    fs: float,
    energy_window: int = 256,
    threshold_factor: float = 3.0,
    min_gap_samples: int | None = None,
) -> list[int]:
    """
    Detect bursts in a single-channel IQ stream by energy thresholding.

    Algorithm
    ---------
    1. Divide `iq` into non-overlapping blocks of `energy_window` samples.
    2. Compute mean power per block.
    3. Use the median power as the noise-floor estimate.
    4. Mark blocks where power > `threshold_factor` × noise_floor as active.
    5. Return the sample index of each rising edge (first active block after
       a quiet block), subject to a minimum gap between consecutive detections.

    Parameters
    ----------
    iq               : (N,) complex IQ samples from one antenna channel.
    fs               : Sample rate [Hz].  Used only to compute default min_gap.
    energy_window    : Block length [samples] for the power estimator.
                       Smaller → finer timing; larger → better SNR estimate.
    threshold_factor : Active-block threshold = threshold_factor × noise_floor.
                       3× for low-SNR indoor; 6× for clean outdoor conditions.
    min_gap_samples  : Minimum distance between consecutive detected bursts
                       [samples].  Default = half of one 90 ms Iridium SF.

    Returns
    -------
    starts : list[int]  — sample offsets of detected burst onsets (rising edges).
    """
    n_blocks = len(iq) // energy_window
    if n_blocks == 0:
        return []

    pwr = np.array(
        [np.mean(np.abs(iq[i * energy_window : (i + 1) * energy_window]) ** 2)
         for i in range(n_blocks)],
        dtype=np.float64,
    )
    noise_floor = float(np.median(pwr)) + 1e-20
    active = pwr > threshold_factor * noise_floor
    edges  = np.diff(active.astype(np.int8), prepend=0)
    starts_blk = np.where(edges > 0)[0]

    if min_gap_samples is None:
        # Default: half of one Iridium 90 ms superframe
        superframe_samples = int(round(0.090 * fs))
        min_gap_samples = max(energy_window, superframe_samples // 2)

    min_gap_blk = max(1, min_gap_samples // energy_window)

    out: list[int] = []
    last = -min_gap_blk - 1
    for blk in starts_blk:
        if blk - last >= min_gap_blk:
            out.append(int(blk * energy_window))
            last = blk
    return out


# =============================================================================
# 2. Preamble-tone FFT scanner
# =============================================================================

def scan_preamble_tones(
    iq: np.ndarray,
    fs: float,
    nom_tone_hz: float,
    scan_bw_hz: float = 45_000.0,
    n_peaks: int = 3,
    min_sep_hz: float = 5_000.0,
    min_snr_db: float = 3.0,
    dc_guard_hz: float = 500.0,
) -> list[tuple[float, float]]:
    """
    FFT scan for CW preamble tones in a burst window.

    Finds up to `n_peaks` frequency peaks inside
    [nom_tone_hz − scan_bw_hz, nom_tone_hz + scan_bw_hz],
    excluding DC artefacts within ±dc_guard_hz of 0 Hz.

    For Iridium IRA: nom_tone_hz = 3125 Hz (preamble tone = fc + Rs/8).
    The returned tone_hz includes the Doppler offset; satellite Doppler =
    tone_hz − nom_tone_hz.

    Parameters
    ----------
    iq           : (N,) complex IQ.  Only the first power-of-2 samples are used.
    fs           : Sample rate [Hz].
    nom_tone_hz  : Centre of the search band [Hz].
    scan_bw_hz   : Half-bandwidth of the search band [Hz].
    n_peaks      : Maximum number of peaks to return.
    min_sep_hz   : Minimum frequency separation between returned peaks [Hz].
    min_snr_db   : Minimum SNR above the median floor to accept a peak [dB].
    dc_guard_hz  : Exclusion zone around DC [Hz].

    Returns
    -------
    peaks : list of (tone_hz, snr_db), sorted by decreasing power.
            Falls back to [(nom_tone_hz, 0.0)] when no valid peak is found.
    """
    N = len(iq)
    if N < 128:
        return [(nom_tone_hz, 0.0)]

    # Zero-pad to the next power of 2 >= N (PySDR frequency-domain chapter:
    # zero-padding interpolates the DFT for finer peak localisation).
    # For the typical preamble window (N=2621): nfft 2048→4096, 500→250 Hz/bin.
    nfft  = max(128, 1 << int(np.ceil(np.log2(max(N, 2)))))
    win   = np.blackman(N)
    Spec  = np.abs(np.fft.fft(iq[:N] * win, n=nfft)) ** 2
    freqs = np.fft.fftfreq(nfft, 1.0 / fs)

    Spec  = np.fft.fftshift(Spec)
    freqs = np.fft.fftshift(freqs)

    lo   = nom_tone_hz - scan_bw_hz
    hi   = nom_tone_hz + scan_bw_hz
    mask = (freqs >= lo) & (freqs <= hi) & (np.abs(freqs) > dc_guard_hz)
    if not np.any(mask):
        return [(nom_tone_hz, 0.0)]

    Sb = Spec[mask].copy()
    fb = freqs[mask]
    noise_floor = float(np.median(Sb)) + 1e-20

    results: list[tuple[float, float]] = []
    for _ in range(n_peaks):
        idx = int(np.argmax(Sb))
        snr = 10.0 * np.log10(max(float(Sb[idx]), 1e-30) / noise_floor)
        if snr < min_snr_db:
            break
        results.append((float(fb[idx]), float(snr)))
        Sb[np.abs(fb - fb[idx]) < min_sep_hz] = 0.0

    return results if results else [(nom_tone_hz, 0.0)]


# =============================================================================
# 3. Matched-filter (MF) covariance
# =============================================================================

def compute_mf_covariance(
    X_cal: np.ndarray,
    tone_hz: float,
    fs: float,
    n_pre: int,
    bpf_guard: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a rank-1 matched-filter covariance matrix for a CW preamble.

    Optimal estimator for a single CW tone in AWGN:

        y_k  = (1/N) Σ_n  x_k[n] · exp(−j·2π·f_tone/fs·n)
        R_mf = y · y^H                      [rank-1]

    λ₁ / (M·σ²) >> 1 when SNR > 0 dB (N samples averaging gain = N).
    Falls back to full sample covariance when MF output is too weak.

    Parameters
    ----------
    X_cal      : (n_ant, N) complex — calibrated, BPF-extracted preamble IQ.
                 Must have N ≥ bpf_guard + n_pre.
    tone_hz    : Exact preamble tone frequency [Hz].
    fs         : Sample rate [Hz].
    n_pre      : Number of preamble samples to use for the MF projection.
    bpf_guard  : Leading samples to skip (BPF impulse-response ringing).

    Returns
    -------
    R_mf    : (n_ant, n_ant) complex — rank-1 MF covariance (or sample cov).
    y_mf    : (n_ant,) complex — per-antenna MF output vector.
    snr_db  : Estimated per-element SNR [dB] from eigenvalue ratio of R_mf.
    """
    if X_cal.shape[1] < bpf_guard + n_pre:
        raise ValueError(
            f"X_cal too short: need {bpf_guard + n_pre} samples, got {X_cal.shape[1]}"
        )

    X_pre = X_cal[:, bpf_guard : bpf_guard + n_pre]
    # Reference vector must account for the bpf_guard offset: the samples at
    # X_cal[:, bpf_guard] correspond to time index bpf_guard, not 0.
    # Using t_vec starting at bpf_guard preserves phase coherence.
    t_vec = np.arange(bpf_guard, bpf_guard + n_pre, dtype=np.float64)
    ref   = np.exp(-2j * np.pi * tone_hz / fs * t_vec)
    y_mf  = (X_pre @ ref) / n_pre          # (n_ant,)
    R_mf  = np.outer(y_mf, y_mf.conj())    # (n_ant, n_ant) rank-1

    # Decide whether to use MF or sample covariance
    mf_power     = float(np.real(np.trace(R_mf)))
    sample_power = float(np.real(np.trace((X_pre @ X_pre.conj().T) / n_pre)))
    if mf_power > 0.01 * sample_power:
        R_out = R_mf
    else:
        R_out = (X_pre @ X_pre.conj().T) / n_pre

    # Per-element SNR from sample covariance (calibration-agnostic).
    # R_mf is rank-1 → noise eigenvalues ≈ 0 → SNR ≈ 160+ dB (meaningless).
    # After amplitude normalization each channel has unit variance, so the
    # absolute eigenvalues are ≈ 1 regardless of signal strength.  The signal
    # is encoded in the eigenvalue SPREAD (λ₁ > mean(λ₂…λ_M)), not in the
    # absolute scale.  Use the eigenvalue ratio (signal-to-interference-plus-
    # noise ratio) which is scale-invariant:
    #   SINR = (λ₁ − σ²_n) / σ²_n   where σ²_n = mean(λ₂…λ_M)
    # This matches snr_uca_db() from doa_uca_2d and gives realistic values:
    # pure noise → SINR ≈ 0 dB, strong coherent signal → SINR > 10 dB.
    R_sample = (X_pre @ X_pre.conj().T) / n_pre
    ev_s     = np.sort(np.maximum(np.linalg.eigvalsh(R_sample), 0.0))[::-1]
    K        = max(1, min(1, R_sample.shape[0] - 1))
    sigma2_n = float(np.mean(ev_s[K:])) + 1e-30
    sinr_lin = max(float(ev_s[0]) - sigma2_n, 1e-30) / max(sigma2_n, 1e-30)
    snr_db   = float(10.0 * np.log10(max(sinr_lin, 1e-10)))

    return R_out, y_mf, snr_db


# =============================================================================
# 4. BPF extraction + amplitude normalization
# =============================================================================

def apply_bpf_and_normalize(
    X_win: np.ndarray,
    pre_samples: int,
    fs: float,
    tone_hz: float,
    bpf_bw_hz: float = 15_000.0,
) -> np.ndarray:
    """
    Apply narrowband BPF and per-channel amplitude normalization.

    Steps:
      1. Extract first ``pre_samples`` columns of X_win.
      2. Apply Hann-windowed soft spectral mask around ``tone_hz ± bpf_bw_hz/2``.
      3. IFFT → narrowband IQ preserving inter-antenna phase.
      4. Normalize each channel to unit RMS (cancels gain imbalance).

    The mask uses a raised-cosine (Hann) taper over the outer 25 % of the
    half-bandwidth as a transition band (PySDR filters chapter).  This gives
    ~32 dB sidelobe rejection vs ~13 dB for a rectangular gate, reducing
    out-of-band interference leakage into the passband.

    Parameters
    ----------
    X_win       : (n_ant, N) complex — raw wideband IQ window.
    pre_samples : Columns to take from X_win before BPF.
    fs          : Sample rate [Hz].
    tone_hz     : Centre of the BPF passband [Hz].
    bpf_bw_hz   : Full width of the BPF passband [Hz].

    Returns
    -------
    X_bpf : (n_ant, pre_samples) complex — BPF + normalized.
    """
    if X_win.shape[1] < pre_samples:
        raise ValueError(
            f"X_win has {X_win.shape[1]} columns, need {pre_samples}"
        )
    X_seg = X_win[:, :pre_samples]
    N     = pre_samples
    freqs = np.fft.fftfreq(N, d=1.0 / fs)

    # Hann-windowed soft spectral mask (PySDR filters chapter).
    # Raised-cosine taper over the outer 25 % of the half-bandwidth.
    half_bw  = bpf_bw_hz * 0.5
    taper_bw = 0.25 * half_bw
    dist     = np.abs(freqs - tone_hz)
    in_pass  = dist <= (half_bw - taper_bw)
    in_taper = (dist > (half_bw - taper_bw)) & (dist <= half_bw)
    w_spec   = np.where(
        in_pass, 1.0,
        np.where(
            in_taper,
            0.5 * (1.0 + np.cos(np.pi * (dist - (half_bw - taper_bw)) / taper_bw)),
            0.0,
        ),
    ).astype(complex)

    X_fft = np.fft.fft(X_seg, axis=1)
    X_nb  = np.fft.ifft(X_fft * w_spec, axis=1).astype(X_win.dtype)

    rms = np.sqrt(np.mean(np.abs(X_nb) ** 2, axis=1, keepdims=True))
    return X_nb / (rms + 1e-20)
