#!/usr/bin/env python3
"""
doa_algorithms – DoA signal processing library for KrakenSDR
=============================================================
Provides array configuration, steering vectors, MUSIC / Root-MUSIC,
covariance decorrelation and related signal utilities.

No simulation code; hardware-only.

Ref: Schmidt, IEEE Trans. Antennas Propagat. 34(3), 1986.
     Barabell, ICASSP 1983.
     Tewfik & Hong, IEEE Trans. Signal Processing 40(4), 1992.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# ── Re-export from canonical modules (no duplicate implementations) ────────
from .covariance import (           # noqa: F401
    covariance,
    forward_backward_avg,
    toeplitzify,
    fb_toeplitz,
    apply_decorrelation,
    CovarianceAccumulator,
)
from .signal_quality import (       # noqa: F401
    eigenvalue_spread_db,
    snr_from_covariance,
    coherence_matrix,
)
from .doa_estimators import compute_papr as _compute_papr  # noqa: F401


def papr_db(spectrum_db: np.ndarray) -> float:
    """
    Peak-to-Average Power Ratio of the DoA spectrum [dB].
    High PAPR → sharp MUSIC peak → reliable estimate.
    Rule of thumb: PAPR > 6 dB acceptable, > 12 dB good.
    Delegates to doa_estimators.compute_papr.
    """
    return _compute_papr(spectrum_db)


# =============================================================================
# Array config
# =============================================================================

class Geometry(Enum):
    ULA = "ULA"   # Uniform Linear Array
    UCA = "UCA"   # Uniform Circular Array


@dataclass
class ArrayConfig:
    """
    Antenna array descriptor with internal geometry caches.

    Parameters
    ----------
    Nr                   : number of elements
    geometry             : ULA or UCA
    d_lambda             : [ULA] inter-element spacing as fraction of λ
    radius_lambda        : [UCA] array radius as fraction of λ
    num_expected_signals : number of sources (D in MUSIC subspace split)
    num_scan_points      : angular grid resolution
    """
    Nr:                   int      = 5
    geometry:             Geometry = Geometry.UCA
    d_lambda:             float    = 0.5
    radius_lambda:        float    = 0.5
    num_expected_signals: int      = 1
    num_scan_points:      int      = 360

    # Geometry cache: keyed on parameter values → auto-invalidates on retune.
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def scan_range(self) -> np.ndarray:
        """Scan angles in radians, −π … +π (endpoint excluded), cached."""
        key = self.num_scan_points
        if key not in self._cache:
            self._cache[key] = np.linspace(-np.pi, np.pi, self.num_scan_points,
                                           endpoint=False)
        return self._cache[key]

    def get_steering_matrix(self) -> np.ndarray:
        """
        Full (Nr × num_scan_points) steering matrix, pre-computed and cached.
        Used in the non-VULA path of MUSIC / Capon / ML.
        Cache key includes geometry → auto-invalidates when radius_lambda or
        d_lambda change (e.g. after auto-tune).
        """
        key = ('a', self.Nr, self.geometry,
               round(self.d_lambda, 9), round(self.radius_lambda, 9),
               self.num_scan_points)
        if key not in self._cache:
            theta = self.scan_range()
            if self.geometry == Geometry.ULA:
                k = np.arange(self.Nr)[:, np.newaxis]
                self._cache[key] = np.exp(
                    2j * np.pi * self.d_lambda * k * np.sin(theta[np.newaxis, :]))
            else:  # UCA
                k   = np.arange(self.Nr)[:, np.newaxis]
                phi = 2.0 * np.pi * k / self.Nr
                self._cache[key] = np.exp(
                    2j * np.pi * self.radius_lambda * (
                        np.cos(phi) * np.cos(theta) + np.sin(phi) * np.sin(theta)))
        return self._cache[key]

    def get_vula_a_mat(self) -> np.ndarray:
        """
        Virtual ULA steering matrix (M × num_scan_points) for MUSIC/Capon/ML,
        pre-computed and cached.  M = 2L+1 where L is derived from radius_lambda.
        """
        key = ('va', self.Nr, round(self.radius_lambda, 9), self.num_scan_points)
        if key not in self._cache:
            theta = self.scan_range()
            x = 2.0 * np.pi * self.radius_lambda
            L = int(np.floor(x))
            if 2 * L + 1 > self.Nr:
                L = (self.Nr - 1) // 2
            M = 2 * L + 1
            ms = np.arange(-L, L + 1)[:M, np.newaxis]
            self._cache[key] = np.exp(1j * ms * theta[np.newaxis, :])
        return self._cache[key]


# =============================================================================
# Steering vector
# =============================================================================

def steering(cfg: ArrayConfig, theta: float) -> np.ndarray:
    """
    Steering vector for angle theta [rad].

    ULA : a[k] = exp(j·2π·d·k·sin(θ))
    UCA : a[k] = exp(j·2π·r·cos(θ−φ_k))  where φ_k = 2πk/Nr
    """
    Nr = cfg.Nr
    if cfg.geometry == Geometry.ULA:
        k = np.arange(Nr)
        return np.exp(2j * np.pi * cfg.d_lambda * k * np.sin(theta))
    else:
        k   = np.arange(Nr)
        phi = 2.0 * np.pi * k / Nr
        return np.exp(
            2j * np.pi * cfg.radius_lambda * (
                np.cos(phi) * np.cos(theta) + np.sin(phi) * np.sin(theta)
            )
        )


# =============================================================================
# UCA → VULA (Phase Mode Excitation)
# =============================================================================

_vula_W_cache: dict[tuple[int, float], np.ndarray] = {}


def _vula_transform_matrix(Nr: int, radius_lambda: float) -> np.ndarray:
    """
    Cached (2L+1, Nr) VULA transform matrix W with Wiener regularisation.

    X_vula = W @ X,  R_vula = W @ R_ant @ W^H.

    Wiener regularisation suppresses modes where J_m(x) is near zero
    instead of amplifying noise via 1/J_m.  Critical for r_lambda ~ 0.358
    where J_0(2pi*0.358 = 2.249) ~ 0.08 is close to the first Bessel zero.

    Ref: Tewfik & Hong, IEEE Trans. Signal Processing 40(4), 1992.
    """
    key = (Nr, round(radius_lambda, 8))
    if key in _vula_W_cache:
        return _vula_W_cache[key]

    from scipy.special import jv as bessel_j

    x = 2.0 * np.pi * radius_lambda
    L = int(np.floor(x))
    if 2 * L + 1 > Nr:
        L = (Nr - 1) // 2
    ms = np.arange(-L, L + 1)
    n_idx = np.arange(Nr)

    F = np.exp(2j * np.pi * np.outer(ms, n_idx) / Nr)

    # Wiener-style regularisation:
    #   D[m] = conj(j^m J_m(x)) / (|j^m J_m(x)|^2 + eps)
    # Modes with |J_m| << sqrt(eps) are attenuated, not amplified.
    jm_Jm = np.array(
        [((1j) ** int(m)) * bessel_j(int(m), x) for m in ms], dtype=complex
    )
    power = np.abs(jm_Jm) ** 2
    eps = 0.1 * np.mean(power)
    diag_vals = np.conj(jm_Jm) / (power + eps)

    T = np.diag(diag_vals) @ F / float(Nr)

    # Pre-whitening: W = A^{-1/2} T
    A = T @ T.conj().T
    ev, U = np.linalg.eigh(A)
    A_inv_sqrt = U @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-15))) @ U.conj().T
    W = A_inv_sqrt @ T

    _vula_W_cache[key] = W
    return W


def uca_to_vula(X: np.ndarray, radius_lambda: float) -> np.ndarray:
    """
    Transform N-element UCA IQ samples to (2L+1) virtual ULA samples
    via Phase Mode Excitation with pre-whitening and Wiener regularisation.

    Ref: Tewfik & Hong, IEEE Trans. Signal Processing 40(4), 1992.
    """
    return _vula_transform_matrix(X.shape[0], radius_lambda) @ X


def compute_doa_covariance(
    X:             np.ndarray,
    cfg:           ArrayConfig,
    decorrelation: str = "FBA",
) -> np.ndarray:
    """
    Compute antenna-space sample covariance.

    Note: VULA transformation is now handled internally by each DoA
    algorithm when needed, avoiding numerical instability from the
    Phase Mode Excitation for configurations where J_0(x) ~ 0.
    """
    return covariance(X)


# =============================================================================
# MUSIC with decorrelation
# =============================================================================

def doa_music(
    X:             np.ndarray,
    cfg:           ArrayConfig,
    decorrelation: str               = "FBA",
    R_in:          np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MUSIC pseudospectrum with optional decorrelation.

    Parameters
    ----------
    X             : IQ sample matrix (Nr × N_samples) complex
    cfg           : ArrayConfig
    decorrelation : 'Off' | 'FBA' | 'TOEP' | 'FBTOEP'
    R_in          : pre-computed covariance (e.g. temporal EMA); skips
                    VULA and sample covariance steps when provided

    Returns
    -------
    (theta_scan [rad], pseudospectrum [dB, peak=0, floor=−40])
    """
    theta_scan = cfg.scan_range()
    # UCA + active decorrelation requires VULA: FBA in antenna space
    # doubles the signal rank for UCA (J @ conj(a) != c*a(theta')).
    # In VULA space J @ conj(a) = a(theta), so FBA works correctly.
    use_vula = (cfg.geometry == Geometry.UCA and decorrelation != "Off")

    if R_in is not None:
        if use_vula:
            W = _vula_transform_matrix(cfg.Nr, cfg.radius_lambda)
            R_v = W @ np.asarray(R_in, dtype=complex) @ W.conj().T
            R = apply_decorrelation(R_v, decorrelation)
        else:
            R = apply_decorrelation(
                np.asarray(R_in, dtype=complex).copy(), decorrelation)
        M = R.shape[0]
    else:
        X_proc = uca_to_vula(X, cfg.radius_lambda) if use_vula else X
        M = X_proc.shape[0]
        R = apply_decorrelation(covariance(X_proc), decorrelation)

    _, eigenvectors = np.linalg.eigh(R)
    n_sig = max(1, min(cfg.num_expected_signals, M - 1))
    En    = eigenvectors[:, :-n_sig]
    E_ct  = En @ En.conj().T

    a_mat = cfg.get_vula_a_mat() if use_vula else cfg.get_steering_matrix()

    # Diagonal of a^H · E_ct · a  via matmul+sum — avoids full (M,M,N_scan) tensor
    denom  = np.real(np.sum(a_mat.conj() * (E_ct @ a_mat), axis=0))
    result = 1.0 / (denom + 1e-12)
    result_db = 10.0 * np.log10(result / (np.max(result) + 1e-12) + 1e-12)
    return theta_scan, np.clip(result_db, -40.0, 0.0)


# =============================================================================
# Capon / MVDR beamformer
# =============================================================================

def doa_capon(
    X:             np.ndarray,
    cfg:           ArrayConfig,
    decorrelation: str               = "FBA",
    R_in:          np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Capon / Minimum Variance Distortionless Response (MVDR) beamformer.

    P_MVDR(θ) = 1 / (a^H(θ) · R^{-1} · a(θ))

    Better spatial resolution than the conventional beamformer; sidelobes
    are data-adaptive. Less sensitive to subspace rank than MUSIC.

    Parameters
    ----------
    X             : IQ sample matrix (Nr × N_samples) complex
    cfg           : ArrayConfig
    decorrelation : 'Off' | 'FBA' | 'TOEP' | 'FBTOEP'
    R_in          : pre-computed covariance (temporal EMA)

    Returns
    -------
    (theta_scan [rad], pseudospectrum [dB, peak=0, floor=−40])

    Ref: Capon J., Proc. IEEE 57(8), pp. 1408-1418, 1969.
    """
    theta_scan = cfg.scan_range()
    use_vula = (cfg.geometry == Geometry.UCA and decorrelation != "Off")

    if R_in is not None:
        if use_vula:
            W = _vula_transform_matrix(cfg.Nr, cfg.radius_lambda)
            R_v = W @ np.asarray(R_in, dtype=complex) @ W.conj().T
            R = apply_decorrelation(R_v, decorrelation)
        else:
            R = apply_decorrelation(
                np.asarray(R_in, dtype=complex).copy(), decorrelation)
        M = R.shape[0]
    else:
        X_proc = uca_to_vula(X, cfg.radius_lambda) if use_vula else X
        M = X_proc.shape[0]
        R = apply_decorrelation(covariance(X_proc), decorrelation)

    # Diagonal loading for numerical stability: delta = eps * Tr(R) / M
    eps   = 1e-4 * float(np.real(np.trace(R))) / M
    R_reg = R + eps * np.eye(M, dtype=complex)
    R_inv = np.linalg.inv(R_reg)

    a_mat = cfg.get_vula_a_mat() if use_vula else cfg.get_steering_matrix()

    # P_MVDR(θ) = 1 / (a^H · R^{-1} · a)
    denom     = np.real(np.sum(a_mat.conj() * (R_inv @ a_mat), axis=0))
    result    = np.maximum(1.0 / (denom + 1e-12), 1e-12)
    result_db = 10.0 * np.log10(result / (np.max(result) + 1e-12) + 1e-12)
    return theta_scan, np.clip(result_db, -40.0, 0.0)


# =============================================================================
# Stochastic Maximum Likelihood (Unconditional ML)
# =============================================================================

def doa_ml(
    X:             np.ndarray,
    cfg:           ArrayConfig,
    decorrelation: str               = "FBA",
    R_in:          np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stochastic (Unconditional) Maximum Likelihood DoA spectrum.

    Concentrated likelihood for D sources with a Gaussian stochastic
    signal model.  For a single source the criterion simplifies to:

        ĵ(θ) = p̂(θ) / σ̂²(θ)

    where:
        p̂(θ)  = a^H(θ)·R̂·a(θ) / (a^H(θ)·a(θ))   [signal power estimate]
        σ̂²(θ) = (Tr(R̂) − p̂(θ)) / (M−1)           [noise power estimate]

    Maximising ĵ(θ) is equivalent to the concentrated log-likelihood for a
    single source.  Unlike the beamformer, the denominator sharpens peaks by
    penalising angles where the noise floor is high after the signal is
    "removed".

    Parameters
    ----------
    X             : IQ sample matrix (Nr × N_samples) complex
    cfg           : ArrayConfig
    decorrelation : 'Off' | 'FBA' | 'TOEP' | 'FBTOEP'
    R_in          : pre-computed covariance (temporal EMA)

    Returns
    -------
    (theta_scan [rad], pseudospectrum [dB, peak=0, floor=−40])

    Ref: Stoica P. & Nehorai A., IEEE Trans. ASSP 38(1), pp. 133-149, 1990.
    """
    theta_scan = cfg.scan_range()
    use_vula = (cfg.geometry == Geometry.UCA and decorrelation != "Off")

    if R_in is not None:
        if use_vula:
            W = _vula_transform_matrix(cfg.Nr, cfg.radius_lambda)
            R_v = W @ np.asarray(R_in, dtype=complex) @ W.conj().T
            R = apply_decorrelation(R_v, decorrelation)
        else:
            R = apply_decorrelation(
                np.asarray(R_in, dtype=complex).copy(), decorrelation)
        M = R.shape[0]
    else:
        X_proc = uca_to_vula(X, cfg.radius_lambda) if use_vula else X
        M = X_proc.shape[0]
        R = apply_decorrelation(covariance(X_proc), decorrelation)

    a_mat = cfg.get_vula_a_mat() if use_vula else cfg.get_steering_matrix()

    # a^H R a  (raw beamformed power, one value per scan angle)
    a_Ra   = np.real(np.sum(a_mat.conj() * (R @ a_mat), axis=0))
    # a^H a   (= M for VULA/UCA with unit-modulus elements)
    a_norm = np.real(np.sum(a_mat.conj() * a_mat, axis=0)) + 1e-12

    p_hat  = a_Ra / a_norm                          # signal power estimate
    tr_R   = float(np.real(np.trace(R)))
    sigma2 = np.maximum((tr_R - p_hat) / max(M - 1, 1), 1e-15)

    result    = np.maximum(p_hat / sigma2, 1e-12)
    result_db = 10.0 * np.log10(result / (np.max(result) + 1e-12) + 1e-12)
    return theta_scan, np.clip(result_db, -40.0, 0.0)


# =============================================================================
# Root-MUSIC via VULA
# =============================================================================

def doa_root_music(
    X:             np.ndarray,
    cfg:           ArrayConfig,
    decorrelation: str               = "FBA",
    R_in:          np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    """
    Root-MUSIC via VULA — sub-grid accuracy on UCA.

    Returns
    -------
    (estimated_angle_deg, gaussian_pseudospectrum [dB], confidence_db)

    confidence_db: proximity of root to unit circle in dB (higher = better).

    Ref: Barabell, ICASSP 1983.
    """
    if R_in is not None:
        W = _vula_transform_matrix(cfg.Nr, cfg.radius_lambda)
        R_v = W @ np.asarray(R_in, dtype=complex) @ W.conj().T
        M = R_v.shape[0]
        R = apply_decorrelation(R_v, decorrelation)
    else:
        X_vula = uca_to_vula(X, cfg.radius_lambda)
        M      = X_vula.shape[0]
        R      = apply_decorrelation(covariance(X_vula), decorrelation)

    _, v    = np.linalg.eigh(R)
    n_sig   = max(1, min(cfg.num_expected_signals, M - 1))
    e_noise = v[:, :-n_sig]
    e_ct    = e_noise @ e_noise.conj().T

    p_coeff = np.array(
        [np.trace(e_ct, k) for k in range(M - 1, -(M - 1) - 1, -1)], dtype=complex
    )
    all_roots = np.roots(p_coeff)
    abs_r     = np.abs(all_roots)

    inside_mask = abs_r < 1.0
    if not np.any(inside_mask):
        inside_mask = np.ones(len(abs_r), dtype=bool)

    prox          = np.abs(abs_r - 1.0)
    prox_filtered = np.where(inside_mask, prox, np.inf)
    sel_idx       = np.argsort(prox_filtered)[:n_sig]
    sel_roots     = all_roots[sel_idx]

    best_prox          = float(prox_filtered[sel_idx[-1]])
    confidence_db      = float(np.clip(-20.0 * np.log10(best_prox + 1e-6), 0.0, 40.0))

    degs = np.rad2deg(np.angle(sel_roots)) % 360.0
    best = float(degs[-1])

    theta_scan = cfg.scan_range()
    thetas_deg = np.rad2deg(theta_scan) % 360.0
    sigma_deg  = 2.5
    spec_lin   = np.zeros(len(thetas_deg))
    for d in degs:
        diff = np.minimum(np.abs(thetas_deg - d), 360.0 - np.abs(thetas_deg - d))
        spec_lin += np.exp(-0.5 * (diff / sigma_deg) ** 2)
    spec_db = np.clip(10.0 * np.log10(spec_lin + 1e-12), -40.0, 0.0)

    return best, spec_db, confidence_db


# =============================================================================
# ESPRIT via VULA
# =============================================================================

def doa_esprit(
    X:             np.ndarray,
    cfg:           ArrayConfig,
    decorrelation: str               = "FBA",
    R_in:          np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    """
    ESPRIT (Estimation of Signal Parameters via Rotational Invariance
    Techniques) applied to a Virtual ULA.

    Exploits the shift-invariance of the VULA basis:
        a_2(θ) = e^{jθ} · a_1(θ)
    where sub-array 1 is rows [0..M-2] and sub-array 2 is rows [1..M-1].

    Steps
    -----
    1. Transform UCA → VULA (Phase Mode Excitation)
    2. Apply FBA to decorrelate coherent sources
    3. Eigendecompose R → signal subspace E_s  (D largest eigenvectors)
    4. Partition:  E_s1 = E_s[:-1, :]   E_s2 = E_s[1:, :]
    5. LS rotational operator:  Φ̂ = pinv(E_s1) @ E_s2
    6. Eigenvalues μ_k of Φ̂  →  θ̂_k = arg(μ_k)
    7. Confidence: proximity of |μ_k| to the unit circle

    Returns
    -------
    (estimated_angle_deg, gaussian_pseudospectrum [dB], confidence_db)

    For N=3 / D=1 (active setup):
        E_s  : (3, 1),  Φ̂ is 2×2 → pick eigenvalue closest to |z|=1.

    Ref: Roy R. & Kailath T., IEEE Trans. ASSP 37(7), pp. 984-995, 1989.
    """
    if R_in is not None:
        W = _vula_transform_matrix(cfg.Nr, cfg.radius_lambda)
        R_v = W @ np.asarray(R_in, dtype=complex) @ W.conj().T
        M = R_v.shape[0]
        R = apply_decorrelation(R_v, decorrelation)
    else:
        X_vula = uca_to_vula(X, cfg.radius_lambda)
        M      = X_vula.shape[0]
        R      = apply_decorrelation(covariance(X_vula), decorrelation)

    if M < 2:
        theta_scan = cfg.scan_range()
        return 0.0, np.full(len(theta_scan), -40.0), 0.0

    if cfg.Nr < 5 and cfg.geometry == Geometry.UCA:
        import warnings
        warnings.warn(
            f"ESPRIT via VULA on a {cfg.Nr}-element UCA produces unreliable "
            "results due to Bessel-mode approximation breakdown (need Nr ≥ 5).",
            stacklevel=2,
        )

    _, v   = np.linalg.eigh(R)                        # ascending eigenvalues
    n_sig  = max(1, min(cfg.num_expected_signals, M - 1))
    E_s    = v[:, -n_sig:]                            # (M, n_sig) signal subspace

    # Overlapping sub-array partition (shift-invariance: one VULA element apart)
    E_s1 = E_s[:-1, :]   # (M-1, n_sig)
    E_s2 = E_s[1:,  :]   # (M-1, n_sig)

    # LS rotational operator:  Φ = pinv(E_s1) @ E_s2
    Phi  = np.linalg.lstsq(E_s1, E_s2, rcond=None)[0]  # (n_sig, n_sig)
    mu   = np.linalg.eigvals(Phi)                        # (n_sig,)

    abs_mu = np.abs(mu)
    # Prefer roots on or inside the unit circle
    inside = abs_mu <= 1.0 + 1e-3
    if not np.any(inside):
        inside = np.ones(len(mu), dtype=bool)

    prox     = np.abs(abs_mu - 1.0)
    prox_fil = np.where(inside, prox, np.inf)
    sel      = np.argsort(prox_fil)[:n_sig]
    sel_mu   = mu[sel]

    best_prox = float(prox_fil[sel[-1]])
    conf_db   = float(np.clip(-20.0 * np.log10(best_prox + 1e-6), 0.0, 40.0))

    degs = np.rad2deg(np.angle(sel_mu)) % 360.0
    best = float(degs[-1])

    # Reconstruct Gaussian pseudospectrum centred on estimated angles
    theta_scan = cfg.scan_range()
    thetas_deg = np.rad2deg(theta_scan) % 360.0
    sigma_deg  = 2.5
    spec_lin   = np.zeros(len(thetas_deg))
    for d in degs:
        diff      = np.minimum(np.abs(thetas_deg - d), 360.0 - np.abs(thetas_deg - d))
        spec_lin += np.exp(-0.5 * (diff / sigma_deg) ** 2)
    spec_db = np.clip(10.0 * np.log10(spec_lin + 1e-12), -40.0, 0.0)

    return best, spec_db, conf_db


# =============================================================================# Signal metrics
# =============================================================================

def apply_phase_correction(X: np.ndarray, offsets_deg: list) -> np.ndarray:
    """Per-channel phase correction [degrees]."""
    offs = np.array(offsets_deg[: X.shape[0]], dtype=float)
    return X * np.exp(-1j * np.deg2rad(offs))[:, np.newaxis]


def measure_power_db(X: np.ndarray) -> float:
    """Mean signal power [dBW]."""
    return float(10.0 * np.log10(np.mean(np.abs(X) ** 2) + 1e-20))


def condition_number(R: np.ndarray) -> float:
    """λ_max / λ_min of R — high value indicates strong, detectable signal."""
    ev = np.sort(np.abs(np.linalg.eigvalsh(R)))
    return float(ev[-1] / (ev[0] + 1e-20))



