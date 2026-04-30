"""
core.covariance — Covariance matrix operations and decorrelation
================================================================

Standardized covariance estimation and preprocessing for DoA algorithms.
All functions are pure NumPy, no side effects, thread-safe.

Functions
---------
covariance              : Sample covariance matrix R = X·X^H / N
forward_backward_avg    : FBA decorrelation for coherent sources
toeplitzify             : Toeplitz rectification
fb_toeplitz             : Combined FB + Toeplitz reconstruction
apply_decorrelation     : Unified decorator dispatcher
spatial_smoothing       : Subarray spatial smoothing (ULA)

References
----------
* Pillai & Kwon, IEEE Trans. ASSP 37(4), 1989 — FBA
* Vallet & Loubaton, ICASSP 2014 — Toeplitz rectification
* Shan et al., IEEE Trans. ASSP 33(4), 1985 — Spatial smoothing
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def covariance(X: np.ndarray) -> np.ndarray:
    """
    Sample covariance matrix: R = X·X^H / N.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_ant, n_samples) complex
    
    Returns
    -------
    R : np.ndarray
        Covariance matrix (n_ant, n_ant) complex
    """
    return (X @ X.conj().T) / X.shape[1]


def sample_covariance(X: np.ndarray, assume_centered: bool = False) -> np.ndarray:
    """
    Sample covariance with optional mean removal.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_ant, n_samples)
    assume_centered : bool
        If True, assume zero-mean data (no subtraction)
    
    Returns
    -------
    R : np.ndarray
        Covariance matrix (n_ant, n_ant)
    """
    if not assume_centered:
        X = X - X.mean(axis=1, keepdims=True)
    return covariance(X)


def forward_backward_avg(R: np.ndarray) -> np.ndarray:
    """
    Forward-Backward Averaging (FBA).
    
    Doubles effective snapshots and decorrelates coherent (multipath) sources:
        R_fb = (R + J·R*·J) / 2
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    
    Returns
    -------
    R_fb : np.ndarray
        FB-averaged covariance (M, M)
    
    Reference
    ---------
    Pillai & Kwon, IEEE Trans. ASSP 37(4), 1989
    """
    M = R.shape[0]
    J = np.fliplr(np.eye(M))
    return (R + J @ R.conj() @ J) / 2.0


def toeplitzify(R: np.ndarray) -> np.ndarray:
    """
    Toeplitz Rectification: averages R along each diagonal.
    
    Exploits shift-invariant structure of ULA/UCA for coherence rejection.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    
    Returns
    -------
    R_toep : np.ndarray
        Toeplitz matrix (M, M)
    
    Reference
    ---------
    Vallet & Loubaton, ICASSP 2014
    """
    import scipy.linalg
    M = R.shape[0]
    c = [np.trace(R, -m) / float(M - m) for m in range(M)]
    return scipy.linalg.toeplitz(c, np.conj(c))


def fb_toeplitz(R: np.ndarray) -> np.ndarray:
    """
    FB + Toeplitz Reconstruction.
    
    Builds complementary Toeplitz forward/backward matrices, then averages
    them with conjugate FB symmetry.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    
    Returns
    -------
    R_fbtoep : np.ndarray
        Reconstructed covariance (M, M)
    
    Reference
    ---------
    Shubair et al., MMS 2016; McDonald & van Wyk, PrimeAsia 2019
    """
    import scipy.linalg
    R_f = scipy.linalg.toeplitz(R[:, 0], R[0, :])
    R_b = scipy.linalg.toeplitz(np.flip(R[:, -1]), np.flip(R[-1, :]))
    return 0.5 * (R_f + R_b.conj())


DecorrelationMethod = Literal["Off", "FBA", "TOEP", "FBTOEP"]


def apply_decorrelation(
    R: np.ndarray,
    method: DecorrelationMethod,
) -> np.ndarray:
    """
    Apply covariance decorrelation method.
    
    Parameters
    ----------
    R : np.ndarray
        Covariance matrix (M, M)
    method : {'Off', 'FBA', 'TOEP', 'FBTOEP'}
        Decorrelation method
    
    Returns
    -------
    R_decorr : np.ndarray
        Processed covariance matrix
    """
    if method == "FBA":
        return forward_backward_avg(R)
    elif method == "TOEP":
        return toeplitzify(R)
    elif method == "FBTOEP":
        return fb_toeplitz(R)
    return R


def spatial_smoothing(
    X: np.ndarray,
    subarray_size: int,
    forward_only: bool = False,
) -> np.ndarray:
    """
    Spatial smoothing for ULA via subarray averaging.
    
    Divides array into overlapping subarrays and averages their covariances.
    Effective for decorrelating coherent multipath.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_ant, n_samples)
    subarray_size : int
        Size of each subarray (must be < n_ant)
    forward_only : bool
        If True, use forward smoothing only (no backward)
    
    Returns
    -------
    R_ss : np.ndarray
        Smoothed covariance (subarray_size, subarray_size)
    
    Reference
    ---------
    Shan et al., IEEE Trans. ASSP 33(4), 1985
    """
    n_ant, n_samples = X.shape
    if subarray_size >= n_ant:
        raise ValueError("subarray_size must be < n_ant")
    
    n_subarrays = n_ant - subarray_size + 1
    R_sum = np.zeros((subarray_size, subarray_size), dtype=complex)
    
    for i in range(n_subarrays):
        X_sub = X[i:i+subarray_size, :]
        R_sub = covariance(X_sub)
        R_sum += R_sub
        
        if not forward_only:
            # Backward smoothing
            X_sub_b = np.flipud(X_sub.conj())
            R_sum += covariance(X_sub_b)
    
    n_avg = n_subarrays if forward_only else 2 * n_subarrays
    return R_sum / n_avg


class CovarianceAccumulator:
    """
    Exponential Moving Average (EMA) accumulator for online covariance.
    
    Maintains running covariance estimate with forgetting factor alpha:
        R_new = alpha * R_old + (1 - alpha) * X·X^H / N
    
    Parameters
    ----------
    alpha : float
        Forgetting factor in [0, 1]. Higher = more memory.
        Typical: 0.90-0.95 for stationary, 0.70-0.80 for moving sources.
    n_ant : int, optional
        Number of antennas (inferred from first update if not given)
    """
    
    def __init__(self, alpha: float = 0.90, n_ant: int | None = None):
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.n_ant = n_ant
        self.R: np.ndarray | None = None
        self.n_updates: int = 0
    
    def update(self, X: np.ndarray) -> np.ndarray:
        """
        Update EMA with new data.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_ant, n_samples)
        
        Returns
        -------
        R : np.ndarray
            Updated covariance estimate (n_ant, n_ant)
        """
        if self.R is None:
            self.n_ant = X.shape[0]
            self.R = covariance(X)
            self.n_updates = 1
        else:
            R_inst = covariance(X)
            self.R = self.alpha * self.R + (1 - self.alpha) * R_inst
            self.n_updates += 1
        
        return self.R
    
    def reset(self) -> None:
        """Reset accumulator state."""
        self.R = None
        self.n_updates = 0
    
    @property
    def is_warm(self) -> bool:
        """True if accumulator has received sufficient updates."""
        if self.n_updates < 2:
            return False
        # Roughly 2*tau updates for EMA to stabilize
        tau = 1.0 / (1.0 - self.alpha) if self.alpha < 0.99 else 100.0
        return self.n_updates >= 2 * tau
