"""
core.doa_uca_2d — 2D DoA for Uniform Circular Arrays
=====================================================

Direction-of-Arrival estimation in 2D (azimuth + elevation) for an N-element
Uniform Circular Array (UCA).

UCA Geometry
------------
    Antenna k at angle φ_k = 2π·k/N clockwise from North.
    Coordinates: p_k_E = r·sin(φ_k), p_k_N = r·cos(φ_k)

Angular Convention
------------------
    azimuth  φ : degrees from North, clockwise (0°=N, 90°=E, 180°=S, 270°=W)
    elevation θ : degrees above horizon (0°=horizon, 90°=zenith)

Implemented Algorithms
----------------------
    • 2D-MUSIC    : P = 1 / ‖E_n^H · a‖²         (super-resolution)
    • 2D-Capon    : P = 1 / (a^H · R^{-1} · a)   (MVDR, adaptive)
    • 2D-Bartlett : P = a^H · R · a               (CBF, robust)

All spectra returned as (n_el, n_az) in dB (peak=0, floor=-40).

References
----------
    • Schmidt 1986 — MUSIC
    • Capon 1969 — MVDR
    • Van Trees 2002 — UCA steering
    • Mathews & Zoltowski 1994 — UCA phase modes
"""

from __future__ import annotations

__all__ = [
    "UcaConfig",
    "find_peak_uca_2d",
    "expected_uca_phase_diffs_deg",
    "phase_residual_deg",
    "doa_phase_fit_uca_2d",
    "extract_pilot_tone",
    "amplitude_normalize_channels",
    "doa_music_uca_2d",
    "doa_bartlett_uca_2d",
    "doa_capon_uca_2d",
    "eigenvalue_spread_uca_db",
    "snr_uca_db",
    "crb_azimuth_deg",
    "CovarianceAccumulatorUca",
]

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# =============================================================================
# UCA configuration
# =============================================================================

@dataclass
class UcaConfig:
    """
    Configuration for an N-element UCA for 2D DoA (azimuth + elevation).

    Parameters
    ----------
    n_ant                : number of antennas             default 5
    radius_lambda        : radius in wavelengths          default 0.5
    n_az                 : azimuth scan points (0…360°)   default 72 → 5° step
    n_el                 : elevation scan points           default 18 → 5° step
    el_min_deg           : minimum elevation [°]           default 5°
    el_max_deg           : maximum elevation [°]           default 90°;
                            lower for indoor (e.g. 45°) to exclude ceiling
    num_expected_signals : expected sources (D in MUSIC subspace split)
                            0 = auto-detect via MDL
    ant0_offset_deg      : physical rotation of antenna 0 from North [°]
                            Calibrate in the field to correct.
    ant_ccw              : True if physical antennas are counter-clockwise
                            (CCW) viewed from above; False = clockwise (CW, default).
                            With CW matrix on CCW array: az_est = 360° − az_true.
    tilt_deg             : tilt of the array plane from horizontal [°].
                            The array normal leans from the zenith towards
                            azimuth ``tilt_az_deg`` by this angle; antennas on
                            that side of the array go down.  Tilting adds
                            vertical baselines (sensitivity ∝ cos el), which
                            improves elevation accuracy near the horizon where
                            a flat UCA is weakest (sensitivity ∝ sin el).
                            0 (default) = flat array, identical to before.
    tilt_az_deg          : azimuth the array normal leans towards [° compass].
    """
    n_ant:                int   = 5
    radius_lambda:        float = 0.5
    n_az:                 int   = 72
    n_el:                 int   = 18
    el_min_deg:           float = 5.0
    el_max_deg:           float = 90.0  # maximum elevation [°]; lower for indoor (e.g. 45°)
    num_expected_signals: int   = 1
    ant0_offset_deg:      float = 0.0   # physical rotation of antenna-0 from North
    ant_ccw:              bool  = False  # True = antennas in counter-clockwise order (CCW)
    tilt_deg:             float = 0.0   # array-plane tilt from horizontal
    tilt_az_deg:          float = 0.0   # azimuth the normal leans towards

    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # ── Geometry ─────────────────────────────────────────────────────────────

    @property
    def positions(self) -> np.ndarray:
        """(n_ant, 3) array: [East, North, Up] coordinates per antenna in wavelengths."""
        k = np.arange(self.n_ant, dtype=np.float64)
        # positive direction = clockwise (CW); ant_ccw=True flips the sign
        sign  = -1.0 if self.ant_ccw else 1.0
        phi_k = np.deg2rad(self.ant0_offset_deg) + sign * 2.0 * np.pi * k / self.n_ant
        east  = self.radius_lambda * np.sin(phi_k)
        north = self.radius_lambda * np.cos(phi_k)
        up    = np.zeros_like(east)
        if self.tilt_deg != 0.0:
            # Rigid rotation of the array plane: the normal leans by tilt_deg
            # towards azimuth tilt_az_deg.  Antennas on that side go down by
            # (p·d)·sin(tilt); horizontal coordinates shrink along d by cos(tilt).
            beta = np.deg2rad(self.tilt_deg)
            t_az = np.deg2rad(self.tilt_az_deg)
            d_e, d_n = np.sin(t_az), np.cos(t_az)       # downhill direction
            along = east * d_e + north * d_n            # p·d
            east  += along * d_e * (np.cos(beta) - 1.0)
            north += along * d_n * (np.cos(beta) - 1.0)
            up     = -along * np.sin(beta)
        return np.column_stack([east, north, up])

    def az_range_deg(self) -> np.ndarray:
        """Azimuth scan grid in degrees: 0° … 360°."""
        return np.linspace(0.0, 360.0, self.n_az, endpoint=False)

    def el_range_deg(self) -> np.ndarray:
        """Elevation scan grid in degrees: el_min … el_max."""
        return np.linspace(self.el_min_deg, self.el_max_deg, self.n_el)

    # ── Steering matrix (pre-computed and cached) ──────────────────────────

    def get_steering_matrix(self) -> np.ndarray:
        """
        Steering matrix (n_ant, n_el × n_az), pre-computed and cached.

        Column index: i_el * n_az + i_az → grid point (el[i_el], az[i_az]).

        Formula (identical to CrossArrayConfig, plus the Up term which is
        zero for a flat array):
            τ_k = 2π · (p_k_E · cosθ · sinφ + p_k_N · cosθ · cosφ + p_k_U · sinθ)
            A_k = exp(j·τ_k)
        """
        key = ('uca2d',
               self.n_ant,
               round(self.radius_lambda, 8),
               self.n_az, self.n_el,
               round(self.el_min_deg, 4),
               round(self.el_max_deg, 4),
               round(self.ant0_offset_deg, 4),
               bool(self.ant_ccw),
               round(self.tilt_deg, 4),
               round(self.tilt_az_deg, 4))
        if key not in self._cache:
            az = np.deg2rad(self.az_range_deg())   # (n_az,)
            el = np.deg2rad(self.el_range_deg())   # (n_el,)

            AZ, EL = np.meshgrid(az, el)           # (n_el, n_az)

            # Direction cosines (ENU)
            u_east  = np.cos(EL) * np.sin(AZ)     # cosθ · sinφ
            u_north = np.cos(EL) * np.cos(AZ)     # cosθ · cosφ
            u_up    = np.sin(EL)                  # sinθ

            # Flatten to (N_grid,) where N_grid = n_el × n_az
            ue = u_east.ravel()
            un = u_north.ravel()
            uu = u_up.ravel()

            # Phase delays (n_ant, N_grid)
            p   = self.positions                   # (n_ant, 3)
            tau = 2.0 * np.pi * (
                p[:, 0:1] * ue[np.newaxis, :]
              + p[:, 1:2] * un[np.newaxis, :]
              + p[:, 2:3] * uu[np.newaxis, :]
            )
            self._cache[key] = np.exp(1j * tau).astype(np.complex128)

        return self._cache[key]

    def invalidate_cache(self) -> None:
        """Force recomputation of the steering matrix on next access."""
        self._cache.clear()

    # ── Compatibility with find_peak_2d from doa_algorithms_3d ──────────────

    @property
    def d_lambda(self) -> float:
        """Alias for CrossArrayConfig compatibility (not used in computation)."""
        return self.radius_lambda


# =============================================================================
# Peak-finding helpers (compatible with doa_algorithms_3d.find_peak_2d)
# =============================================================================

def find_peak_uca_2d(
    spec: np.ndarray,
    cfg:  UcaConfig,
) -> Tuple[float, float, float]:
    """
    Find (azimuth_deg, elevation_deg, papr_db) from a 2D spectrum.

    Uses 2D parabolic interpolation around the peak bin for
    sub-grid accuracy (±half grid step).

    Returns
    -----------
    az_deg  : estimated azimuth  [°, 0…360]
    el_deg  : estimated elevation [°, el_min…el_max]
    papr_db : Peak-to-Average Power Ratio of the spectrum [dB]
    """
    idx       = np.unravel_index(np.argmax(spec), spec.shape)
    i_el, i_az = int(idx[0]), int(idx[1])
    n_el, n_az  = spec.shape

    # Azimuth — periodic (wrap-around)
    az_frac = 0.0
    ym = float(spec[i_el, (i_az - 1) % n_az])
    y0 = float(spec[i_el,  i_az])
    yp = float(spec[i_el, (i_az + 1) % n_az])
    denom_az = ym - 2.0 * y0 + yp
    if abs(denom_az) > 1e-6:
        az_frac = float(np.clip(0.5 * (ym - yp) / denom_az, -0.5, 0.5))

    # Elevation — non-periodic
    el_frac = 0.0
    if 0 < i_el < n_el - 1:
        ym = float(spec[i_el - 1, i_az])
        y0 = float(spec[i_el,     i_az])
        yp = float(spec[i_el + 1, i_az])
        denom_el = ym - 2.0 * y0 + yp
        if abs(denom_el) > 1e-6:
            el_frac = float(np.clip(0.5 * (ym - yp) / denom_el, -0.5, 0.5))

    az_step = 360.0 / n_az
    el_step = (cfg.el_max_deg - cfg.el_min_deg) / max(n_el - 1, 1)

    az_deg = (cfg.az_range_deg()[i_az] + az_frac * az_step) % 360.0
    el_deg = float(
        np.clip(cfg.el_range_deg()[i_el] + el_frac * el_step,
                cfg.el_min_deg, cfg.el_max_deg)
    )

    s_lin  = 10.0 ** (np.clip(spec, -200.0, 0.0) / 10.0)
    mean_v = float(np.mean(s_lin))
    papr   = (10.0 * np.log10(float(np.max(s_lin)) / (mean_v + 1e-15))
              if mean_v > 1e-15 else 0.0)

    return float(az_deg), float(el_deg), float(papr)


def _circular_az_sep_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def expected_uca_phase_diffs_deg(
    az_deg: float,
    el_deg: float,
    cfg: UcaConfig,
) -> np.ndarray:
    """Model inter-antenna phase diffs [°] relative to antenna 0."""
    pos = cfg.positions
    az_r = np.deg2rad(az_deg)
    el_r = np.deg2rad(el_deg)
    tau = 2.0 * np.pi * (
        pos[:, 0] * np.cos(el_r) * np.sin(az_r)
        + pos[:, 1] * np.cos(el_r) * np.cos(az_r)
        + pos[:, 2] * np.sin(el_r)
    )
    tau -= tau[0]
    return np.degrees(tau[1:])


def phase_residual_deg(
    phase_diffs: np.ndarray,
    az_deg: float,
    el_deg: float,
    cfg: UcaConfig,
) -> np.ndarray:
    """Wrap measured − expected inter-antenna phases to (−180°, +180°]."""
    expected = expected_uca_phase_diffs_deg(az_deg, el_deg, cfg)
    return ((phase_diffs - expected + 180.0) % 360.0) - 180.0


def _inter_antenna_phase_error_deg(
    az_deg: float,
    el_deg: float,
    phase_diffs: np.ndarray,
    cfg: UcaConfig,
) -> float:
    """Mean absolute error between measured and UCA-model inter-antenna phases."""
    residual = phase_residual_deg(phase_diffs, az_deg, el_deg, cfg)
    return float(np.mean(np.abs(residual)))


def _resolve_uca_180_ambiguity(
    az_deg: float,
    el_deg: float,
    *,
    phase_diffs: np.ndarray | None = None,
    cfg: UcaConfig | None = None,
    az_hint_deg: float | None = None,
    margin_deg: float = 5.0,
    phase_min_margin_deg: float = 10.0,
) -> float:
    """Pick azimuth side θ vs θ+180° using phase fit, then tracker hint."""
    az_mirror = (az_deg + 180.0) % 360.0
    if (phase_diffs is not None and cfg is not None
            and phase_diffs.size >= cfg.n_ant - 1):
        err_d = _inter_antenna_phase_error_deg(az_deg, el_deg, phase_diffs, cfg)
        err_m = _inter_antenna_phase_error_deg(az_mirror, el_deg, phase_diffs, cfg)
        if err_m + phase_min_margin_deg < err_d:
            return az_mirror
        if err_d + phase_min_margin_deg < err_m:
            return az_deg
    if az_hint_deg is not None:
        d_direct = _circular_az_sep_deg(az_deg, az_hint_deg)
        d_mirror = _circular_az_sep_deg(az_mirror, az_hint_deg)
        if d_mirror + margin_deg < d_direct:
            return az_mirror
    return az_deg


def doa_phase_fit_uca_2d(
    R: np.ndarray,
    cfg: UcaConfig,
    *,
    az_hint_deg: float | None = None,
    el_hint_deg: float | None = None,
    az_window_deg: float = 45.0,
    el_window_deg: float = 20.0,
) -> tuple[float, float, float]:
    """
    Estimate (az, el) by minimising UCA inter-antenna phase mismatch.

    Best for calibrated single-source indoor TX at low SNR; follows motion
    faster than spectral peak-picking when MULTI_BURST_N is small.
    Returns (az_deg, el_deg, mean_abs_phase_error_deg).
    """
    phase_diffs = np.degrees(np.angle(R[1:, 0]))
    el_min = float(cfg.el_min_deg)
    el_max = float(cfg.el_max_deg)

    if az_hint_deg is not None:
        az_c = float(az_hint_deg) % 360.0
        az_vals = np.arange(az_c - az_window_deg, az_c + az_window_deg + 0.1, 2.0)
        az_vals = np.unique(np.mod(az_vals, 360.0))
    else:
        az_vals = np.arange(0.0, 360.0, 4.0)

    if el_hint_deg is not None:
        el_c = float(el_hint_deg)
        el_vals = np.arange(el_c - el_window_deg, el_c + el_window_deg + 0.1, 2.0)
        el_vals = el_vals[(el_vals >= el_min) & (el_vals <= el_max)]
        if el_vals.size == 0:
            el_vals = np.array([el_c])
    else:
        el_vals = np.linspace(el_min, el_max, max(3, int((el_max - el_min) / 3.0) + 1))

    best_err, best_az, best_el = 1e9, 0.0, el_min
    for az in az_vals:
        for el in el_vals:
            err = _inter_antenna_phase_error_deg(float(az), float(el), phase_diffs, cfg)
            if err < best_err:
                best_err, best_az, best_el = err, float(az), float(el)

    for az in np.arange(best_az - 4.0, best_az + 4.1, 0.5):
        for el in np.arange(best_el - 4.0, best_el + 4.1, 0.5):
            azw = float(az) % 360.0
            elw = float(np.clip(el, el_min, el_max))
            err = _inter_antenna_phase_error_deg(azw, elw, phase_diffs, cfg)
            if err < best_err:
                best_err, best_az, best_el = err, azw, elw

    best_az = _resolve_uca_180_ambiguity(
        best_az, best_el,
        phase_diffs=phase_diffs, cfg=cfg,
        az_hint_deg=az_hint_deg,
    )
    return best_az, best_el, float(best_err)


def extract_pilot_tone(
    X:           np.ndarray,
    sample_rate: float,
    tone_hz:     float,
    bw_hz:       float = 10_000.0,
) -> np.ndarray:
    """
    Extract a CW pilot tone from multi-channel IQ data via FFT gating.

    For a LibreSDR beacon transmitted at ``LO_freq + tone_hz``, this filters
    out-of-band interference by:

        1. FFT every channel snapshot (N-point).
        2. Apply a Hann-windowed soft spectral mask around tone_hz ± bw_hz/2.
        3. IFFT → narrowband IQ, preserving inter-antenna phase.

    The mask uses a raised-cosine (Hann) taper over the outer 25 % of the
    half-bandwidth as the transition band.  This gives ~32 dB sidelobe
    rejection vs ~13 dB for a rectangular gate (PySDR filters chapter),
    strongly reducing DQPSK data energy (which starts ~4 kHz from the
    preamble tone) from bleeding into the 8 kHz passband used for MUSIC.

    Effective SNR gain = 10 · log10(sample_rate / bw_hz)  [dB].
    At 1.024 MSPS with bw_hz=10 kHz → +20 dB noise rejection.

    The inter-antenna phase relationship is preserved:

        ∠ X_k[f] − ∠ X_0[f] = ∠ a_k(az, el) − ∠ a_0(az, el)

    so all DoA algorithms (MUSIC / Capon / Bartlett) benefit directly.

    Choosing tone_hz = 100_000 Hz with Heimdall at 1.024 MSPS / CPI 131072:
        bin = 100 000 × 131 072 / 1 024 000 = 12 800  (integer → zero leakage).

    Parameters
    ----------
    X           : (n_ant, N) complex IQ sampled at ``sample_rate`` Hz
    sample_rate : ADC sample rate [Hz]  (Heimdall default: 1_024_000)
    tone_hz     : pilot tone offset from Heimdall LO [Hz]  (may be negative)
    bw_hz       : extraction window width [Hz]  (default 10 kHz)

    Returns
    -------
    X_nb : (n_ant, N) complex, same dtype as X, narrowband around tone_hz
    """
    N     = X.shape[1]
    freqs = np.fft.fftfreq(N, d=1.0 / sample_rate)          # (N,) Hz

    # Hann-windowed soft spectral mask (PySDR filters chapter).
    # Raised-cosine taper over the outer 25 % of the half-bandwidth.
    half_bw  = bw_hz * 0.5
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

    X_fft = np.fft.fft(X, axis=1)
    return np.fft.ifft(X_fft * w_spec, axis=1).astype(X.dtype)


# =============================================================================
# Per-channel amplitude normalisation
# =============================================================================

def amplitude_normalize_channels(X: np.ndarray) -> np.ndarray:
    """
    Normalise each antenna channel to unit RMS power.

    Cancels hardware gain imbalances between KrakenSDR channels
    (up to ~5 dB variation measured on indoor recordings).
    Apply BEFORE computing the sample covariance for DoA.

    .. warning::
        Do NOT use the normalised samples for absolute power measurements
        or SNR calibration — use raw X for those.

    Reference: KrakenSDR signal_processor.py — channel normalisation step.

    Parameters
    ----------
    X : (n_ant, N) complex IQ

    Returns
    -------
    X_n : (n_ant, N) complex, same dtype, each row has unit RMS
    """
    rms = np.sqrt(np.mean(np.abs(X) ** 2, axis=1, keepdims=True))
    return X / (rms + 1e-20)


# =============================================================================
# Covariance helper (internal)
# =============================================================================

def _get_cov(X: np.ndarray, R_in: np.ndarray | None) -> np.ndarray:
    if R_in is not None:
        return np.asarray(R_in, dtype=complex)
    return (X @ X.conj().T) / X.shape[1]


# =============================================================================
# ── Covariance decorrelation (anti-multipath) ──────────────────────────
# =============================================================================

def _circulant_smooth(R: np.ndarray) -> np.ndarray:
    """
    Circulant averaging (UCA spatial smoothing, Mathews & Zoltowski 1994).

    For each circular lag l, compute the average:
        d[l] = (1/N) * sum_{k=0}^{N-1}  R[k, (k+l) mod N]
    and reconstruct the circulant matrix R_c[i,j] = d[(j-i) mod N].

    Equivalent to N overlapping virtual sub-arrays (UCA rotations):
        R_c = (1/N) * sum_k  Π^k · R · (Π^k)^H
    where Π is the cyclic permutation matrix.

    Benefits:
    - Decorrelates coherent sources (multipath) → N/2 coherent copies tolerated
    - Enforces the theoretical circulant structure of the UCA covariance
    - Source: Ita97/2D_MUSIC_DOA (spatial smoothing) adapted for circular UCA
    """
    N = R.shape[0]
    k = np.arange(N)
    # First-lag vector (d[0]..d[N-1]) — average over all starting points
    d = np.empty(N, dtype=complex)
    for lag in range(N):
        d[lag] = np.mean(R[k, (k + lag) % N])
    # Build circulant matrix: R_c[i, j] = d[(j - i) % N]
    rows = [np.roll(d, i) for i in range(N)]
    R_c  = np.array(rows, dtype=complex)
    # Enforce Hermitian symmetry (numerical errors)
    return (R_c + R_c.conj().T) * 0.5


def _fb_average_uca(R: np.ndarray) -> np.ndarray:
    """
    Forward-Backward averaging per UCA: R_fb = 0.5 * (R + J · R* · J).

    J is the anti-diagonal exchange matrix.
    For even ULA it is exact (a(-ψ) = J·a*(ψ) up to scalar phase).
    For N=5 UCA it is approximate but improves robustness to coherent multipath
    by reducing the effective rank of coherent sources.

    Implementation analogous to Ita97/2D_MUSIC_DOA fb=True.
    """
    N = R.shape[0]
    J    = np.eye(N, dtype=complex)[::-1, :]   # anti-diagonal identity
    R_fb = 0.5 * (R + J @ np.conj(R) @ J)
    return (R_fb + R_fb.conj().T) * 0.5  # enforce Hermitian symmetry


def _decor_cov(R: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply decorrelation to the covariance.

    mode: 'none' | 'circulant' | 'fb' | 'both'
        'circulant' = circulant smoothing only  (preferred for Capon)
        'fb'        = forward-backward only     (limited without circulant)
        'both'      = circulant then FB          (default for indoor MUSIC)
    """
    if mode in ("circulant", "both"):
        R = _circulant_smooth(R)
    if mode in ("fb", "both"):
        R = _fb_average_uca(R)
    return R


# =============================================================================
# 2D-MUSIC for UCA
# =============================================================================

def doa_music_uca_2d(
    X:           np.ndarray,
    cfg:         UcaConfig,
    R_in:        np.ndarray | None = None,
    n_snapshots: int | None = None,
    decorr:      str = "none",
) -> np.ndarray:
    """
    2D-MUSIC pseudospectrum for UCA.

    P(φ, θ) = 1 / ‖E_n^H · a(φ, θ)‖²

    NOTE on decorrelation for UCA:
    - Circulant smoothing is correct for URA (Ita97/2D_MUSIC_DOA, sps=True)
      but NOT for UCA: forces R to be cyclically symmetric → eigenvectors
      = DFT vectors → MUSIC spectrum with N-fold symmetry (N-pointed star).
    - For UCA the correct anti-multipath decorrelation is temporal averaging
      (EMA, handled by CovarianceAccumulatorUca) — with alpha=0.97, 33 frames
      of integration decorrelate indoor multipath.
    - decorr='none' is the safe default. Use 'circulant'/'fb' only in
      offline experiments with many guaranteed snapshots.

    Parameters
    ---------
    X           : (n_ant, N_samples) complex
    cfg         : UcaConfig
    R_in        : pre-computed covariance (e.g. EMA); if provided, X is ignored
    n_snapshots : IQ samples (for MDL auto-detect)
    decorr      : 'none' (default) | 'circulant' | 'fb' | 'both'

    Returns
    -----------
    spec : (n_el, n_az) float ndarray  [dB, peak = 0, floor = −40 dB]
    """
    R = _decor_cov(_get_cov(X, R_in), decorr)
    M = R.shape[0]

    # Adaptive diagonal loading scaled to the eigenvalue gap.
    # At high SNR (eig_max/eig_min > 100), loading is negligible (0.1% trace).
    # At low SNR (eig_max/eig_min < 10), loading increases to 5% trace,
    # stabilising the noise subspace without distorting the signal eigenvector.
    ev_raw  = np.sort(np.real(np.linalg.eigvalsh(R)))
    ev_gap  = (ev_raw[-1] / max(ev_raw[0], 1e-20))
    load_frac = np.clip(0.5 / max(np.log10(ev_gap + 1e-10), 0.1), 0.005, 0.05)
    diag_load = load_frac * max(float(np.real(np.trace(R))) / M, 1e-20)
    R = R + diag_load * np.eye(M, dtype=complex)

    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Auto-detect sources via MDL (Wax & Kailath 1985) when num_expected_signals == 0
    if cfg.num_expected_signals == 0:
        if n_snapshots is not None:
            _N = n_snapshots
        elif R_in is None:
            _N = X.shape[1]
        else:
            _N = 1024
        n_sig = _estimate_signal_count_mdl(R, _N, M - 1)
        if n_sig == 0:
            n_sig = 1
    else:
        n_sig = max(1, min(cfg.num_expected_signals, M - 1))

    En = eigenvectors[:, :-n_sig]           # (M, M-n_sig) noise subspace

    A  = cfg.get_steering_matrix()          # (M, N_grid)
    Pa = En.conj().T @ A                   # (M-n_sig, N_grid)

    denom    = np.real(np.sum(np.abs(Pa) ** 2, axis=0))   # (N_grid,)
    pspec    = 1.0 / (denom + 1e-12)
    pspec_db = 10.0 * np.log10(pspec / (float(np.max(pspec)) + 1e-12) + 1e-12)
    return np.clip(pspec_db, -40.0, 0.0).reshape(cfg.n_el, cfg.n_az)


# =============================================================================
# 2D-Bartlett (CBF) for UCA
# =============================================================================

def doa_bartlett_uca_2d(
    X:    np.ndarray,
    cfg:  UcaConfig,
    R_in: np.ndarray | None = None,
) -> np.ndarray:
    """
    Conventional 2D beamformer (delay-and-sum) for UCA.

    P(φ, θ) = a^H(φ, θ) · R · a(φ, θ)

    Most robust when a channel is degraded — degrades gracefully
    by broadening the main lobe instead of failing silently.

    Returns
    -----------
    spec : (n_el, n_az) float ndarray  [dB, peak = 0, floor = −40 dB]
    """
    R = _get_cov(X, R_in)
    A = cfg.get_steering_matrix()              # (M, N_grid)

    pspec    = np.real(np.sum(A.conj() * (R @ A), axis=0))   # (N_grid,)
    pspec    = np.maximum(pspec, 1e-30)
    pspec_db = 10.0 * np.log10(pspec / (float(np.max(pspec)) + 1e-30) + 1e-30)
    return np.clip(pspec_db, -40.0, 0.0).reshape(cfg.n_el, cfg.n_az)


# =============================================================================
# 2D-Capon (MVDR) for UCA
# =============================================================================

def doa_capon_uca_2d(
    X:    np.ndarray,
    cfg:  UcaConfig,
    R_in: np.ndarray | None = None,
    decorr: str = "none",
) -> np.ndarray:
    """
    2D-Capon beamformer (MVDR) for UCA.

    P(φ, θ) = 1 / (a^H(φ, θ) · R^{-1} · a(φ, θ))

    decorr='none' by default (see note in doa_music_uca_2d).
    Circulant smoothing forces N-fold symmetry, making Capon
    equivalent to Bartlett on a degraded covariance.

    Returns
    -----------
    spec : (n_el, n_az) float ndarray  [dB, peak = 0, floor = −40 dB]
    """
    R = _decor_cov(_get_cov(X, R_in), decorr)

    M   = R.shape[0]
    # Adaptive loading based on max eigenvalue
    ev_max = float(abs(np.linalg.eigvalsh(R)[-1]))
    eps    = 1e-4 * max(ev_max, 1e-20)
    R      = R + eps * np.eye(M, dtype=complex)

    # PySDR DoA chapter: "pseudo-inverse tends to work better than a true
    # inverse".  pinv handles any residual near-singularity after diagonal
    # loading without the dangerous identity-matrix fallback.
    R_inv = np.linalg.pinv(R)

    A  = cfg.get_steering_matrix()           # (M, N_grid)
    Qa = R_inv @ A                           # (M, N_grid)

    denom    = np.real(np.sum(A.conj() * Qa, axis=0))   # (N_grid,)
    pspec    = 1.0 / (np.maximum(denom, 1e-30))
    pspec_db = 10.0 * np.log10(pspec / (float(np.max(pspec)) + 1e-30) + 1e-30)
    return np.clip(pspec_db, -40.0, 0.0).reshape(cfg.n_el, cfg.n_az)


# =============================================================================
# ── Signal quality metrics ───────────────────────────────────────────
# =============================================================================

def eigenvalue_spread_uca_db(R: np.ndarray) -> np.ndarray:
    """
    Covariance eigenvalues in dB, descending order.
    Normalised to the smallest (noise floor = 0 dB).
    """
    ev = np.sort(np.maximum(np.linalg.eigvalsh(R), 0.0))[::-1]
    return 10.0 * np.log10(ev / (ev[-1] + 1e-20) + 1e-20)


def snr_uca_db(R: np.ndarray, n_sources: int = 1) -> float:
    """Signal-to-noise ratio estimate [dB] from the covariance eigenvalue spectrum.

    Uses the signal/noise eigenvalue partition (Wax & Kailath 1985, Salama 2025 §4.2):

        σ²_noise = mean(λ_{n_sources+1}, …, λ_M)   (averaged noise eigenvalues)
        SNR      = (λ_1 – σ²_noise) / σ²_noise

    More accurate than the simple λmax/λmin ratio when M > 2, because averaging
    over all noise-subspace eigenvalues suppresses finite-sample estimation noise.

    Parameters
    ----------
    R         : (M, M) sample covariance matrix
    n_sources : assumed number of sources (default 1 for a single Iridium IRA)

    Returns
    -------
    snr_db : estimated SNR in dB (can be negative when source is below noise floor)
    """
    ev = np.sort(np.maximum(np.linalg.eigvalsh(R), 0.0))[::-1]  # descending
    M  = len(ev)
    K  = max(1, min(n_sources, M - 1))
    sigma2_noise = float(np.mean(ev[K:]))
    snr_lin      = (float(ev[0]) - sigma2_noise) / max(sigma2_noise, 1e-30)
    return float(10.0 * np.log10(max(snr_lin, 1e-10)))


def crb_azimuth_deg(
    snr_db:      float,
    n_snapshots: int,
    cfg:         UcaConfig,
    el_deg:      float = 45.0,
) -> float:
    """Cramér-Rao Bound for azimuth estimation on a UCA (Stoica & Nehorai 1990).

    For a single narrowband source at elevation *el_deg* observed with an
    M-element UCA of radius r_λ wavelengths and N temporal snapshots at a
    per-snapshot SNR (linear), the theoretical lower bound on the standard
    deviation of any unbiased azimuth estimator is:

        CRB_φ = (180/π) / [2π · r_λ · cos(el) · √(M · N · SNR_lin)]  [degrees]

    Derivation (Salama 2025 §8.2.1; Van Trees 2002 §8.3)
    ------------------------------------------------------
    Starting from the Fisher Information element for azimuth φ:

        J_φφ = 2N · SNR · Re{ (∂a/∂φ)^H · P_a^⊥ · (∂a/∂φ) }

    where P_a^⊥ = I – a·a^H/M is the projection orthogonal to the steering
    vector a.  For a UCA with φ_k = 2πk/M:

        ∂a_k/∂φ = –j · 2π·r·cos(el)·sin(φ–φ_k) · a_k

        ‖∂a/∂φ‖² = (2π·r·cos(el))² · Σ_k sin²(φ–φ_k) = (2π·r·cos(el))² · M/2

        a^H·(∂a/∂φ) = –j·2π·r·cos(el)·Σ_k sin(φ–2πk/M) = 0   (full-period sum)

    Therefore J_φφ = N · M · SNR · (2π · r_λ · cos(el))²  and  CRB = 1/J_φφ.

    Notes
    -----
    - MUSIC and ESPRIT converge to the CRB asymptotically at high SNR
      (Stoica-Nehorai asymptotic efficiency).
    - At low SNR a "threshold effect" causes all subspace estimators to deviate
      sharply above the CRB; for M=5, N=2600 the threshold is near –5 dB SNR.
    - The formula assumes K=1 source, spatially white noise, and perfect
      array calibration (no gain/phase imbalance).

    Parameters
    ----------
    snr_db      : **per-element** signal-to-noise ratio [dB],
                  i.e. P_source / σ²_noise at a single antenna.
                  If you have the array-gain SNR from snr_uca_db() (which
                  returns M × per-element SNR), subtract 10·log10(M) first.
    n_snapshots : number of IQ samples used to estimate R (= X.shape[1])
    cfg         : UcaConfig — supplies n_ant (M) and radius_lambda (r_λ)
    el_deg      : source elevation [°], 0 = horizon, 90 = zenith  (default 45°)

    Returns
    -------
    crb_deg : CRB standard deviation for azimuth [°]  (always ≥ 0)
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    el_rad  = np.deg2rad(el_deg)
    M = float(cfg.n_ant)
    r = cfg.radius_lambda
    # Fisher Information: J = N · M · SNR · (2π · r_λ · cos(el))²
    J = n_snapshots * M * snr_lin * (2.0 * np.pi * r * np.cos(el_rad)) ** 2
    return float(np.degrees(1.0 / np.sqrt(max(J, 1e-30))))


# =============================================================================
# EMA covariance accumulator for UCA
# =============================================================================

class CovarianceAccumulatorUca:
    """
    Exponential moving-average (EMA) covariance accumulator for UCA.

    R_new = alpha * R_old + (1 - alpha) * R_frame

    alpha = 0 → single-frame only;  alpha close to 1 → long memory.

    For a stationary CW source the high-alpha EMA acts as temporal
    decorrelation for indoor multipath: reflections slowly change phase
    due to thermal/mechanical vibration, while the direct path stays
    stable.  alpha=0.97 gives ~33 frames of memory (~6 s at 5 fps).

    For moving sources lower alpha to 0.50–0.70 to track fast changes.
    """
    def __init__(self, alpha: float = 0.90):
        self.alpha = float(alpha)
        self._R: np.ndarray | None = None
        self.n_updates: int = 0

    def update(self, X: np.ndarray) -> np.ndarray:
        """
        Update with an IQ frame (n_ant × N) and return the EMA covariance.

        Parameters
        ----------
        X : (n_ant, N) complex IQ — should be pre-normalised if desired.
        """
        R_frame = (X @ X.conj().T) / X.shape[1]
        if self._R is None:
            self._R = R_frame.copy()
        else:
            self._R = self.alpha * self._R + (1.0 - self.alpha) * R_frame
        self.n_updates += 1
        return self._R

    def reset(self) -> None:
        """Clear accumulated covariance (e.g. after frequency retune)."""
        self._R = None
        self.n_updates = 0

    @property
    def R(self) -> np.ndarray | None:
        return self._R

    @property
    def is_warm(self) -> bool:
        """True once enough frames have been integrated to trust the EMA."""
        # Time constant tau = 1/(1-alpha) frames; warm after 2*tau
        tau = 1.0 / max(1.0 - self.alpha, 1e-6)
        return self.n_updates >= max(int(2.0 * tau), 2)


# =============================================================================
# Internal: source count estimation (simplified MDL)
# =============================================================================

def _estimate_signal_count_mdl(
    R:          np.ndarray,
    n_snapshots: int,
    max_signals: int = 4,
) -> int:
    """MDL criterion for signal count estimation (Wax & Kailath 1985)."""
    M  = R.shape[0]
    ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
    ev = np.maximum(ev, 1e-30)
    D_max = min(max_signals, M - 1)
    N = float(n_snapshots)
    n = float(M)

    best_k, best_cost = 0, float("inf")
    for k in range(D_max + 1):
        noise_ev = ev[k:]
        m = float(len(noise_ev))
        g_k = float(np.exp(np.mean(np.log(noise_ev))))
        a_k = float(np.mean(noise_ev))
        if a_k < 1e-30 or g_k < 1e-30:
            break
        likelihood = N * m * float(np.log(a_k / g_k))
        penalty    = float(k) * (2.0 * n - float(k))
        cost = likelihood + 0.5 * penalty * np.log(N)
        if cost < best_cost:
            best_cost = cost
            best_k    = k

    return best_k

