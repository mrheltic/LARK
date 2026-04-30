"""
core.doa_estimators — Direction-of-Arrival estimation algorithms
=================================================================

Clean, modular implementations of subspace-based DoA estimators.
All functions are pure (no side effects) and operate on covariance matrices.

Algorithms
----------
music           : Multiple Signal Classification (MUSIC)
capon           : Capon beamformer / MVDR
bartlett        : Conventional beamformer
root_music      : Root-MUSIC for ULA (polynomial rooting)
esprit          : ESPRIT for ULA (rotational invariance)

Utilities
---------
subspace_decomposition  : Signal/noise subspace from covariance
estimate_signal_count     : MDL/AIC for automatic model order
peak_interpolation_1d     : Parabolic peak interpolation
peak_interpolation_2d     : 2D parabolic interpolation for DoA grids

References
----------
* Schmidt, IEEE Trans. Antennas Propagat. 34(3), 1986 — MUSIC
* Barabell, ICASSP 1983 — Root-MUSIC
* Roy & Kailath, IEEE Trans. ASSP 37(7), 1989 — ESPRIT
* Capon, Proc. IEEE 57(8), 1969 — MVDR
* Wax & Kailath, IEEE Trans. ASSP 33(2), 1985 — MDL/AIC
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


# =============================================================================
# Subspace decomposition
# =============================================================================

def subspace_decomposition(
    R: np.ndarray,
    n_signals: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eigendecomposition and subspace splitting.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    n_signals : int
        Number of signal sources (must be < M)
    
    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues in descending order (M,)
    E_s : np.ndarray
        Signal subspace eigenvectors (M, n_signals)
    E_n : np.ndarray
        Noise subspace eigenvectors (M, M - n_signals)
    """
    M = R.shape[0]
    if n_signals >= M:
        raise ValueError(f"n_signals ({n_signals}) must be < M ({M})")
    
    eigvals, eigvecs = np.linalg.eigh(R)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    E_s = eigvecs[:, :n_signals]
    E_n = eigvecs[:, n_signals:]
    
    return eigvals, E_s, E_n


def estimate_signal_count(
    R: np.ndarray,
    n_snapshots: int,
    method: Literal["MDL", "AIC"] = "MDL",
) -> int:
    """
    Estimate number of signals using information-theoretic criteria.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    n_snapshots : int
        Number of samples used to estimate R
    method : {'MDL', 'AIC'}
        Criterion to use
    
    Returns
    -------
    n_signals : int
        Estimated number of signals
    
    Reference
    ---------
    Wax & Kailath, IEEE Trans. ASSP 33(2), 1985
    """
    M = R.shape[0]
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(eigvals)[::-1]
    
    # Geometric mean of smallest M-k eigenvalues
    log_eig = np.log(np.maximum(eigvals, 1e-15))
    
    costs = []
    for k in range(M):
        if k == M - 1:
            costs.append(np.inf)
            continue
        
        # Arithmetic mean of noise eigenvalues
        noise_eig = eigvals[k:]
        arithmetic_mean = np.mean(noise_eig)
        # Geometric mean
        geometric_mean = np.exp(np.mean(log_eig[k:]))
        
        likelihood = n_snapshots * (M - k) * np.log(arithmetic_mean / geometric_mean)
        
        if method == "MDL":
            penalty = 0.5 * k * (2 * M - k) * np.log(n_snapshots)
        else:  # AIC
            penalty = k * (2 * M - k)
        
        costs.append(likelihood + penalty)
    
    return int(np.argmin(costs))


# =============================================================================
# 1D DoA Estimators
# =============================================================================

def music(
    R: np.ndarray,
    steering_matrix: np.ndarray,
    n_signals: int = 1,
) -> np.ndarray:
    """
    MUSIC pseudospectrum.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    steering_matrix : np.ndarray
        Steering vectors (M, n_points) for search grid
    n_signals : int
        Number of signals (for subspace dimension)
    
    Returns
    -------
    spectrum : np.ndarray
        MUSIC pseudospectrum (n_points,) in linear scale
    """
    _, _, E_n = subspace_decomposition(R, n_signals)
    
    # P_music = 1 / ||E_n^H @ a||^2
    noise_proj = E_n.conj().T @ steering_matrix
    denom = np.sum(np.abs(noise_proj) ** 2, axis=0)
    
    # Avoid division by zero
    spectrum = 1.0 / np.maximum(denom, 1e-15)
    return spectrum


def capon(
    R: np.ndarray,
    steering_matrix: np.ndarray,
) -> np.ndarray:
    """
    Capon/MVDR beamformer.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    steering_matrix : np.ndarray
        Steering vectors (M, n_points)
    
    Returns
    -------
    spectrum : np.ndarray
        Capon spectrum (n_points,) in linear scale
    """
    R_inv = np.linalg.inv(R)
    
    # P_capon = 1 / (a^H @ R_inv @ a)
    aH_Rinv = steering_matrix.conj().T @ R_inv
    denom = np.sum(aH_Rinv * steering_matrix.T, axis=1)
    
    spectrum = 1.0 / np.maximum(np.real(denom), 1e-15)
    return spectrum


def bartlett(
    R: np.ndarray,
    steering_matrix: np.ndarray,
) -> np.ndarray:
    """
    Conventional Bartlett beamformer.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    steering_matrix : np.ndarray
        Steering vectors (M, n_points)
    
    Returns
    -------
    spectrum : np.ndarray
        Bartlett spectrum (n_points,) in linear scale
    """
    # P_bartlett = a^H @ R @ a
    Ra = R @ steering_matrix
    spectrum = np.sum(steering_matrix.conj() * Ra, axis=0)
    return np.real(spectrum)


def root_music(
    R: np.ndarray,
    n_signals: int,
    d_lambda: float = 0.5,
) -> np.ndarray:
    """
    Root-MUSIC for ULA via polynomial rooting.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    n_signals : int
        Number of signals
    d_lambda : float
        Element spacing in wavelengths
    
    Returns
    -------
    doa_rad : np.ndarray
        DoA estimates in radians (n_signals,)
    
    Reference
    ---------
    Barabell, ICASSP 1983
    """
    from numpy.polynomial import polynomial as P
    
    _, _, E_n = subspace_decomposition(R, n_signals)
    
    # Form polynomial from noise subspace
    M = R.shape[0]
    C = E_n @ E_n.conj().T
    
    # Polynomial coefficients from sum of diagonals
    p_coef = np.zeros(2 * M - 1, dtype=complex)
    for i in range(M):
        for j in range(M):
            p_coef[M - 1 + i - j] += C[i, j]
    
    # Find roots inside unit circle
    roots = np.roots(p_coef[::-1])
    inside = roots[np.abs(roots) < 1]
    
    # Select roots closest to unit circle
    distances = np.abs(np.abs(inside) - 1)
    closest_idx = np.argsort(distances)[:n_signals]
    selected_roots = inside[closest_idx]
    
    # Convert to DOA
    doa_rad = np.arcsin(np.angle(selected_roots) / (2 * np.pi * d_lambda))
    return doa_rad


def esprit(
    R: np.ndarray,
    n_signals: int,
) -> np.ndarray:
    """
    ESPRIT for ULA using rotational invariance.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    n_signals : int
        Number of signals
    
    Returns
    -------
    doa_rad : np.ndarray
        DoA estimates in radians (n_signals,)
    
    Reference
    ---------
    Roy & Kailath, IEEE Trans. ASSP 37(7), 1989
    """
    _, E_s, _ = subspace_decomposition(R, n_signals)
    
    M = R.shape[0]
    # Subarrays: first M-1 and last M-1 elements
    E0 = E_s[:-1, :]   # (M-1, n_signals)
    E1 = E_s[1:, :]    # (M-1, n_signals)
    
    # Solve for rotation matrix
    Phi = np.linalg.lstsq(E0, E1, rcond=None)[0]
    
    # Eigenvalues give phase shifts
    mu = np.angle(np.linalg.eigvals(Phi))
    
    # Convert to DOA (assuming d=lambda/2)
    doa_rad = np.arcsin(mu / np.pi)
    return doa_rad


# =============================================================================
# 2D DoA Estimators (Azimuth + Elevation)
# =============================================================================

def music_2d(
    R: np.ndarray,
    steering_matrix: np.ndarray,
    n_signals: int = 1,
) -> np.ndarray:
    """
    2D MUSIC pseudospectrum for (azimuth, elevation) search.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    steering_matrix : np.ndarray
        Steering vectors (M, n_az * n_el) for 2D grid
    n_signals : int
        Number of signals
    
    Returns
    -------
    spectrum : np.ndarray
        2D MUSIC spectrum reshaped to (n_el, n_az)
    """
    spectrum_1d = music(R, steering_matrix, n_signals)
    # Reshape based on steering matrix dimensions
    n_grid = steering_matrix.shape[1]
    n_el = int(np.sqrt(n_grid))  # Approximate if square grid
    n_az = n_grid // n_el
    return spectrum_1d.reshape(n_el, n_az)


def capon_2d(
    R: np.ndarray,
    steering_matrix: np.ndarray,
) -> np.ndarray:
    """
    2D Capon beamformer for (azimuth, elevation) search.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    steering_matrix : np.ndarray
        Steering vectors (M, n_az * n_el)
    
    Returns
    -------
    spectrum : np.ndarray
        2D Capon spectrum (n_el, n_az)
    """
    spectrum_1d = capon(R, steering_matrix)
    n_grid = steering_matrix.shape[1]
    n_el = int(np.sqrt(n_grid))
    n_az = n_grid // n_el
    return spectrum_1d.reshape(n_el, n_az)


# =============================================================================
# Peak interpolation
# =============================================================================

def peak_interpolation_1d(
    y: np.ndarray,
    idx: int,
) -> Tuple[float, float]:
    """
    Parabolic interpolation of peak location.
    
    Parameters
    ----------
    y : np.ndarray
        1D array of values
    idx : int
        Index of peak sample
    
    Returns
    -------
    peak_offset : float
        Fractional offset from idx (-0.5 to 0.5)
    peak_value : float
        Interpolated peak value
    """
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0, float(y[idx])
    
    ym, y0, yp = y[idx-1], y[idx], y[idx+1]
    denom = ym - 2*y0 + yp
    
    if abs(denom) < 1e-10:
        return 0.0, float(y0)
    
    offset = 0.5 * (ym - yp) / denom
    value = y0 - 0.25 * (ym - yp) * offset
    
    return float(np.clip(offset, -0.5, 0.5)), float(value)


def peak_interpolation_2d(
    Z: np.ndarray,
    row: int,
    col: int,
    periodic_cols: bool = False,
) -> Tuple[float, float, float]:
    """
    2D parabolic peak interpolation.
    
    Parameters
    ----------
    Z : np.ndarray
        2D array (n_rows, n_cols)
    row, col : int
        Peak indices
    periodic_cols : bool
        If True, treat columns as periodic (azimuth wrap)
    
    Returns
    -------
    row_offset : float
        Fractional row offset
    col_offset : float
        Fractional column offset
    peak_value : float
        Interpolated peak value
    """
    n_rows, n_cols = Z.shape
    
    # Row interpolation (elevation - not periodic)
    if 0 < row < n_rows - 1:
        ym, y0, yp = Z[row-1, col], Z[row, col], Z[row+1, col]
        denom = ym - 2*y0 + yp
        row_offset = 0.5 * (ym - yp) / denom if abs(denom) > 1e-10 else 0.0
    else:
        row_offset = 0.0
    
    # Column interpolation (azimuth - may be periodic)
    if periodic_cols:
        cm = (col - 1) % n_cols
        cp = (col + 1) % n_cols
    else:
        if col <= 0 or col >= n_cols - 1:
            col_offset = 0.0
            return row_offset, col_offset, float(Z[row, col])
        cm, cp = col - 1, col + 1
    
    ym, y0, yp = Z[row, cm], Z[row, col], Z[row, cp]
    denom = ym - 2*y0 + yp
    col_offset = 0.5 * (ym - yp) / denom if abs(denom) > 1e-10 else 0.0
    
    # Interpolated value (approximate)
    peak_value = y0
    
    return (
        float(np.clip(row_offset, -0.5, 0.5)),
        float(np.clip(col_offset, -0.5, 0.5)),
        float(peak_value)
    )


# =============================================================================
# Spectrum normalization
# =============================================================================

def normalize_spectrum_db(
    spectrum: np.ndarray,
    floor_db: float = -40.0,
) -> np.ndarray:
    """
    Normalize spectrum to dB with peak at 0 dB.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Linear spectrum values
    floor_db : float
        Minimum dB value (clips below this)
    
    Returns
    -------
    spectrum_db : np.ndarray
        Normalized spectrum in dB
    """
    spectrum = np.maximum(spectrum, 1e-15)
    spectrum_db = 10 * np.log10(spectrum / np.max(spectrum))
    return np.clip(spectrum_db, floor_db, 0.0)


def compute_papr(spectrum_db: np.ndarray) -> float:
    """
    Compute Peak-to-Average Power Ratio from dB spectrum.
    
    Parameters
    ----------
    spectrum_db : np.ndarray
        Spectrum in dB (peak = 0 dB)
    
    Returns
    -------
    papr_db : float
        PAPR in dB
    """
    s_lin = 10 ** (spectrum_db / 10.0)
    mean_lin = np.mean(s_lin)
    return 10 * np.log10(1.0 / (mean_lin + 1e-15))
