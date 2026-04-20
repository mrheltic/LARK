"""
doa_algorithms_3d — 2D (azimuth + elevation) DoA for cross array
===================================================================

3D space DoA algorithms for the 5-element cross ("+") array.

Cross array layout (East-North ground plane, arm length d·λ):

                 ant2 (North)
                  |
    ant3 ──── ant0 ──── ant1
    (West)    (ctr)     (East)
                  |
                 ant4 (South)

Source convention (satellite over the array):
  azimuth  φ : degrees from North, clockwise
               (0°=N / 90°=E / 180°=S / 270°=W)
  elevation θ : degrees above horizon
               (0°=horizon / 90°=zenith)

The received phase at antenna k for a far-field source at (φ, θ):

    τ_k = 2π · ( p_k_E · cos(θ) · sin(φ)  +  p_k_N · cos(θ) · cos(φ) )

where p_k_E, p_k_N are the East and North coordinates of antenna k in λ.

Steering vector:
    a(φ,θ)  =  [exp(j·τ_0), …, exp(j·τ_4)]^T  ∈ ℂ^5

2D-MUSIC:
    P(φ,θ)  =  1 / (a^H(φ,θ) · E_n · E_n^H · a(φ,θ))
where E_n is the noise subspace of R = E·E^H (D smallest eigenvectors).

2D-Capon (MVDR):
    P(φ,θ)  =  1 / (a^H(φ,θ) · R^{-1} · a(φ,θ))

Why the cross array for satellite DoA
--------------------------------------
A symmetric cross array resolves both azimuth and elevation simultaneously
by exploiting orthogonal apertures (E-W and N-S arms).  With 5 physically
independent elements (no ambiguities at d ≤ 0.5 λ), the 5×5 covariance
matrix provides up to 4 degrees of freedom, sufficient to separate 1–2
satellite sources.

References
----------
* Schmidt R.O., IEEE Trans. Antennas Propagat. 34(3), 1986               — MUSIC
* Capon J., Proc. IEEE 57(8), pp. 1408-1418, 1969                       — MVDR/Capon
* Van Trees H.L., Optimum Array Processing, Wiley 2002, §6.5             — 2D steering
* Pillai S.U. & Kwon B.H., IEEE Trans. ASSP 37(4), 1989                  — FBA
* Vu D.T. et al., IEEE Trans. Signal Process. 58(9), 2010                — 2D subspace methods
* Wax M. & Kailath T., IEEE Trans. ASSP 33(2), 1985                      — AIC/MDL
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


CROSS_ARRAY_CANONICAL_ORDER: tuple[str, ...] = (
    "center", "east", "north", "west", "south"
)

_CROSS_ARRAY_ORDER_ALIASES = {
    "c": "center",
    "ctr": "center",
    "center": "center",
    "centre": "center",
    "e": "east",
    "east": "east",
    "n": "north",
    "north": "north",
    "s": "south",
    "south": "south",
    "w": "west",
    "west": "west",
}

_CROSS_ARRAY_SHORT_LABELS = {
    "center": "C",
    "east": "E",
    "north": "N",
    "west": "W",
    "south": "S",
}


def normalize_cross_array_order(order: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate and normalise a 5-channel cross-array order description."""
    if len(order) != 5:
        raise ValueError(
            f"Cross-array order must contain 5 labels, got {len(order)}: {order!r}"
        )

    norm: list[str] = []
    for label in order:
        key = str(label).strip().lower().replace("-", "_").replace(" ", "")
        if key not in _CROSS_ARRAY_ORDER_ALIASES:
            raise ValueError(f"Unknown cross-array label: {label!r}")
        norm.append(_CROSS_ARRAY_ORDER_ALIASES[key])

    if set(norm) != set(CROSS_ARRAY_CANONICAL_ORDER):
        raise ValueError(
            "Cross-array order must contain center/east/north/west/south exactly once; "
            f"got {norm!r}"
        )
    return tuple(norm)


def reorder_cross_array_channels(
    X: np.ndarray,
    input_order: list[str] | tuple[str, ...],
    *,
    output_order: list[str] | tuple[str, ...] = CROSS_ARRAY_CANONICAL_ORDER,
) -> np.ndarray:
    """Reorder axis 0 of a 5-channel IQ matrix between physical and solver order."""
    in_order = normalize_cross_array_order(input_order)
    out_order = normalize_cross_array_order(output_order)
    if X.shape[0] != len(in_order):
        raise ValueError(
            f"Expected axis 0 length {len(in_order)} for cross-array IQ, got {X.shape[0]}"
        )
    idx = [in_order.index(label) for label in out_order]
    return np.take(X, idx, axis=0)


def short_cross_array_labels(order: list[str] | tuple[str, ...]) -> list[str]:
    """Return short display labels like ['C', 'N', 'E', 'S', 'W']."""
    norm = normalize_cross_array_order(order)
    return [_CROSS_ARRAY_SHORT_LABELS[label] for label in norm]


# =============================================================================
# Cross array configuration
# =============================================================================

@dataclass
class CrossArrayConfig:
    """
    5-element cross array configuration for 2D (azimuth + elevation) DoA.

    Internal solver order is fixed to:
        [center, east, north, west, south]

    If the physical Kraken / Heimdall channel order differs, reorder IQ data
    with :func:`reorder_cross_array_channels` before calling the solver.

    Antenna layout (λ-normalised East-North plane):
        ant0 : center  [ 0,  0]
        ant1 : East    [+d,  0]
        ant2 : North   [ 0, +d]
        ant3 : West    [-d,  0]
        ant4 : South   [ 0, -d]

    Parameters
    ----------
    d_lambda             : arm length [fraction of λ]          default 0.5
    n_az                 : azimuth scan points over 0…360°     default 72 → 5° steps
    n_el                 : elevation scan points over el_min…90° default 18 → 5° steps
    el_min_deg           : minimum elevation angle to scan [°]  default 5°
    num_expected_signals : number of sources (D in MUSIC subspace split)
    """
    d_lambda:             float = 0.5
    n_az:                 int   = 72
    n_el:                 int   = 18
    el_min_deg:           float = 5.0
    num_expected_signals: int   = 1

    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # ── Geometry ──────────────────────────────────────────────────────────────

    @property
    def positions(self) -> np.ndarray:
        """(5, 2) array: [East, North] coordinates per antenna in wavelengths."""
        d = self.d_lambda
        return np.array([
            [ 0.0,  0.0],   # ant0: center
            [+d,    0.0],   # ant1: East
            [ 0.0, +d  ],   # ant2: North
            [-d,    0.0],   # ant3: West
            [ 0.0, -d  ],   # ant4: South
        ], dtype=np.float64)

    def az_range_deg(self) -> np.ndarray:
        """Azimuth scan grid in degrees: 0° … 360°."""
        return np.linspace(0.0, 360.0, self.n_az, endpoint=False)

    def el_range_deg(self) -> np.ndarray:
        """Elevation scan grid in degrees: el_min … 90°."""
        return np.linspace(self.el_min_deg, 90.0, self.n_el)

    # ── Steering matrix (pre-computed + cached) ───────────────────────────────

    def get_steering_matrix(self) -> np.ndarray:
        """
        (5, n_el × n_az) steering matrix, pre-computed and cached.

        Column index: i_el * n_az + i_az  →  grid point (el[i_el], az[i_az]).
        The cache key encodes all shape-determining parameters so it invalidates
        automatically if d_lambda, n_az, n_el, or el_min_deg change.
        """
        key = ('cross3d',
               round(self.d_lambda, 8),
               self.n_az, self.n_el,
               round(self.el_min_deg, 4))
        if key not in self._cache:
            az = np.deg2rad(self.az_range_deg())   # (N_az,)
            el = np.deg2rad(self.el_range_deg())   # (N_el,)

            AZ, EL = np.meshgrid(az, el)           # (N_el, N_az)

            # Direction cosines in the East-North plane
            u_east  = np.cos(EL) * np.sin(AZ)     # cos θ · sin φ
            u_north = np.cos(EL) * np.cos(AZ)     # cos θ · cos φ

            # Flatten to (N_grid,) where N_grid = N_el * N_az
            ue = u_east.ravel();  un = u_north.ravel()

            # Phase delays (5, N_grid)
            p   = self.positions                   # (5, 2)
            tau = 2.0 * np.pi * (
                p[:, 0:1] * ue[np.newaxis, :]
              + p[:, 1:2] * un[np.newaxis, :]
            )
            self._cache[key] = np.exp(1j * tau).astype(np.complex128)

        return self._cache[key]

    def invalidate_cache(self) -> None:
        """Force re-computation of the steering matrix on next access."""
        self._cache.clear()


# =============================================================================
# 2D-MUSIC
# =============================================================================

def doa_music_2d(
    X:    np.ndarray,
    cfg:  CrossArrayConfig,
    R_in: np.ndarray | None = None,
) -> np.ndarray:
    """
    2D-MUSIC pseudospectrum for the cross array.

    P(φ, θ) = 1 / ‖E_n^H · a(φ, θ)‖²

    Parameters
    ----------
    X    : (5, N_samples) complex — IQ matrix (burst window or CW)
    cfg  : CrossArrayConfig
    R_in : optional (5, 5) covariance; if provided X is ignored

    Returns
    -------
    spec : (n_el, n_az) float ndarray  [dB, peak = 0, floor = −40 dB]
    """
    R = _get_cov(X, R_in)

    # Diagonal loading: δ = 1e-4 · Tr(R) / M  for numerical stability
    M = R.shape[0]
    eps = 1e-4 * float(np.real(np.trace(R))) / M
    R = R + eps * np.eye(M, dtype=complex)

    _, eigenvectors = np.linalg.eigh(R)          # ascending eigenvalues
    n_sig = max(1, min(cfg.num_expected_signals, 4))
    En    = eigenvectors[:, :-n_sig]             # (5, 5−n_sig) noise subspace

    A  = cfg.get_steering_matrix()               # (5, N_grid)
    Pa = En.conj().T @ A                         # (5−n_sig, N_grid)

    denom    = np.real(np.sum(np.abs(Pa) ** 2, axis=0))  # (N_grid,)
    pspec    = 1.0 / (denom + 1e-12)
    pspec_db = 10.0 * np.log10(pspec / (np.max(pspec) + 1e-12) + 1e-12)
    return np.clip(pspec_db, -40.0, 0.0).reshape(cfg.n_el, cfg.n_az)


# =============================================================================
# 2D-Capon (MVDR)
# =============================================================================

def doa_capon_2d(
    X:    np.ndarray,
    cfg:  CrossArrayConfig,
    R_in: np.ndarray | None = None,
) -> np.ndarray:
    """
    2D-Capon (MVDR) beamformer spectrum for the cross array.

    P(φ, θ) = 1 / (a^H(φ, θ) · R^{-1} · a(φ, θ))

    Less sensitive to noise-subspace dimension errors than 2D-MUSIC.
    Recommended as cross-check or when the number of sources D is uncertain.

    Parameters
    ----------
    X    : (5, N_samples) complex
    cfg  : CrossArrayConfig
    R_in : optional pre-computed covariance

    Returns
    -------
    spec : (n_el, n_az) float ndarray  [dB, peak = 0, floor = −40 dB]

    Ref: Capon J., Proc. IEEE 57(8), 1969.
    """
    R = _get_cov(X, R_in)

    # Diagonal loading: δ = 1e-4 · Tr(R) / 5  for numerical stability
    eps   = 1e-4 * float(np.real(np.trace(R))) / 5.0
    R_reg = R + eps * np.eye(5, dtype=complex)
    R_inv = np.linalg.inv(R_reg)                # (5, 5)

    A  = cfg.get_steering_matrix()              # (5, N_grid)
    denom    = np.real(np.sum(A.conj() * (R_inv @ A), axis=0))  # (N_grid,)
    pspec    = np.maximum(1.0 / (denom + 1e-12), 1e-12)
    pspec_db = 10.0 * np.log10(pspec / (np.max(pspec) + 1e-12) + 1e-12)
    return np.clip(pspec_db, -40.0, 0.0).reshape(cfg.n_el, cfg.n_az)


# =============================================================================
# Peak finding with sub-grid parabolic interpolation
# =============================================================================

def find_peak_2d(
    spec: np.ndarray,
    cfg:  CrossArrayConfig,
) -> Tuple[float, float, float]:
    """
    Find (azimuth_deg, elevation_deg, papr_db) from a 2D spectrum.

    Uses 2D parabolic interpolation around the peak bin for sub-grid accuracy.
    The interpolation corrects the quantisation introduced by the discrete scan
    grid (step_az × step_el), typically ±half a grid step.

    Returns
    -------
    az_deg  : estimated azimuth  [°, 0…360]
    el_deg  : estimated elevation [°, el_min…90]
    papr_db : peak-to-average power ratio of the spectrum [dB]
    """
    idx       = np.unravel_index(np.argmax(spec), spec.shape)
    i_el, i_az = int(idx[0]), int(idx[1])
    n_el, n_az = spec.shape

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

    # Grid step sizes
    az_step = 360.0 / n_az
    el_step = (90.0 - cfg.el_min_deg) / max(n_el - 1, 1)

    az_deg = (cfg.az_range_deg()[i_az] + az_frac * az_step) % 360.0
    el_deg = float(
        np.clip(cfg.el_range_deg()[i_el] + el_frac * el_step,
                cfg.el_min_deg, 90.0)
    )

    # PAPR of the 2D spectrum
    s_lin  = 10.0 ** (np.clip(spec, -200.0, 0.0) / 10.0)
    mean_v = float(np.mean(s_lin))
    papr   = (10.0 * np.log10(float(np.max(s_lin)) / (mean_v + 1e-15))
              if mean_v > 1e-15 else 0.0)

    return float(az_deg), float(el_deg), float(papr)


# =============================================================================
# Sky-plot coordinate helpers
# =============================================================================

def skyplot_coords(az_deg: float, el_deg: float) -> Tuple[float, float]:
    """
    Convert (az_deg, el_deg) to matplotlib polar plot coordinates.

    The sky plot uses:
        theta_mpl = azimuth in radians (0=North at top, clockwise = –1 direction)
        r_mpl     = 90° − elevation [deg]  (0 = zenith, 90 = horizon)

    Use with:
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
    """
    return float(np.deg2rad(az_deg)), float(90.0 - el_deg)


def make_sky_heatmap_edges(cfg: CrossArrayConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cell-edge arrays for ax.pcolormesh on a polar sky plot.

    Returns
    -------
    theta_edges : (n_az + 1,) in radians   — azimuth cell edges
    r_edges     : (n_el + 1,) in degrees   — radial cell edges (90 − elevation)

    Usage::
        T_e, R_e  = np.meshgrid(theta_edges, r_edges)
        ax.pcolormesh(T_e, R_e, spec_flipped, ...)
    where spec_flipped = np.flipud(spec) to map low elevation → outer ring.
    """
    az_edges = np.linspace(0.0, 2.0 * np.pi, cfg.n_az + 1)
    el_edges = np.linspace(cfg.el_min_deg, 90.0, cfg.n_el + 1)
    r_edges  = 90.0 - el_edges          # low elevation → high r (outer)
    return az_edges, r_edges[::-1]      # flip: near-horizon at edge of plot


# =============================================================================
# Signal quality metrics (same definitions as doa_algorithms.py)
# =============================================================================

def eigenvalue_spread_db(R: np.ndarray) -> np.ndarray:
    """
    Eigenvalues of the 5×5 covariance matrix in dB, sorted descending.

    Normalised against the smallest eigenvalue (noise floor = 0 dB).
    The first eigenvalue is the signal subspace; the remaining four
    represent the noise subspace.  A large gap between λ_0 and the rest
    indicates a strong, localised source and a reliable DoA estimate.
    """
    ev = np.sort(np.abs(np.linalg.eigvalsh(R)))[::-1]
    return 10.0 * np.log10(ev / (ev[-1] + 1e-20) + 1e-20)


def snr_from_covariance(R: np.ndarray) -> float:
    """SNR estimate [dB] from the max/min eigenvalue ratio of R."""
    ev    = np.sort(np.abs(np.linalg.eigvalsh(R)))
    ratio = (ev[-1] - ev[0]) / (ev[0] + 1e-20)
    return float(10.0 * np.log10(max(ratio, 1e-10)))


def coherence_matrix(R: np.ndarray) -> np.ndarray:
    """
    Off-diagonal coherence |ρ_ij| = |R_ij| / sqrt(R_ii · R_jj).

    Diagonal is 1.0; high off-diagonal values indicate coherent channels
    (correlated noise or strong multipath).
    """
    d = np.sqrt(np.real(np.diag(R)) + 1e-30)
    return np.abs(R) / np.outer(d, d)


# =============================================================================
# Exponential moving average covariance (shared with doa_algorithms.py)
# =============================================================================

class CovarianceAccumulator3D:
    """
    EMA covariance accumulator for the 5-element cross array.

    R_new = α · R_old + (1 − α) · R_frame

    α = 0.0 → no memory (single frame)
    α → 1   → very long memory

    For burst signals (Iridium TDMA) this accumulator should be bypassed
    in favour of single-shot covariance on the extracted burst window.
    See core.iridium_doa_burst.compute_single_shot_covariance().
    """
    def __init__(self, alpha: float = 0.90):
        self.alpha = float(alpha)
        self._R:   np.ndarray | None = None

    def update(self, X: np.ndarray) -> np.ndarray:
        """Feed IQ frame (5 × N), return current EMA covariance."""
        R_frame = (X @ X.conj().T) / X.shape[1]
        if self._R is None:
            self._R = R_frame.copy()
        else:
            self._R = self.alpha * self._R + (1.0 - self.alpha) * R_frame
        return self._R

    def reset(self) -> None:
        self._R = None

    @property
    def R(self) -> np.ndarray | None:
        return self._R


# =============================================================================
# Internal helpers
# =============================================================================

def _get_cov(X: np.ndarray, R_in: np.ndarray | None) -> np.ndarray:
    if R_in is not None:
        return np.asarray(R_in, dtype=complex)
    return (X @ X.conj().T) / X.shape[1]


# =============================================================================
# AIC / MDL signal-count estimator
# =============================================================================

def estimate_signal_count(
    R:           np.ndarray,
    n_snapshots: int,
    method:      str = "mdl",
    max_signals: int = 4,
) -> int:
    """
    Estimate the number of signal sources D using AIC or MDL information
    criteria applied to the eigenvalue spectrum of the spatial covariance R.

    Theory (Wax & Kailath, IEEE Trans. ASSP 33(2), 1985)
    ----------------------------------------------------
    For k = 0 … n-1 candidate values of D, define:

        g(k) = geometric mean of {λ_{k+1}, …, λ_n}  (noise eigenvalues)
        a(k) = arithmetic mean of {λ_{k+1}, …, λ_n}

    The likelihood ratio gives

        ℓ(k) = N · (n − k) · ln(a(k) / g(k))   (≥ 0, equals 0 when all equal)

    AIC:  AIC(k)  =  2·ℓ(k) + 2·k·(2n − k)
    MDL:  MDL(k)  =  ℓ(k)   + ½·k·(2n − k)·ln N

    D̂ = argmin_k  AIC(k)  or  argmin_k  MDL(k)

    MDL is consistent (D̂ → D as N → ∞); AIC occasionally over-estimates but
    is more sensitive at low SNR.  MDL is recommended for satellite work.

    Notes
    -----
    * Only the real eigenvalues of a Hermitian R are used (np.linalg.eigvalsh).
    * Eigenvalues are clipped to > 0 before log (Schur positivity not
      guaranteed numerically; small negative values can appear from FBA).
    * At very low SNR (eigenvalue spread < 6 dB) the criteria may return 0.
      The caller should treat 0 as «reject this burst».
    * Tested and verified on the recorded Iridium data: for 17 real bursts with
      eigenvalue spreads 8–19 dB and 10690 snapshots, MDL returns D=1 reliably.

    Parameters
    ----------
    R           : (M, M) Hermitian covariance matrix (M = 5 for cross array)
    n_snapshots : number of IQ samples used to estimate R (= N_burst typically)
    method      : "mdl" (default) or "aic"
    max_signals : upper bound on D (default 4 for M=5)

    Returns
    -------
    D : int in {0, 1, …, max_signals}
    """
    M  = R.shape[0]
    ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]        # descending
    ev = np.maximum(ev, 1e-30)                                 # guard log(0)
    D_max = min(max_signals, M - 1)
    N     = float(n_snapshots)
    n     = float(M)

    best_k    = 0
    best_cost = float("inf")

    for k in range(D_max + 1):
        noise_ev = ev[k:]                          # M − k noise eigenvalues
        m        = float(len(noise_ev))
        g_k      = float(np.exp(np.mean(np.log(noise_ev))))    # geometric mean
        a_k      = float(np.mean(noise_ev))                    # arithmetic mean

        if a_k < 1e-30 or g_k < 1e-30:
            break

        log_ratio = float(np.log(a_k / g_k))      # ≥ 0 by AM-GM inequality
        likelihood = N * m * log_ratio             # = N·(n-k)·ln(a/g)

        penalty_factor = float(k) * (2.0 * n - float(k))
        if method == "aic":
            cost = 2.0 * likelihood + 2.0 * penalty_factor
        else:  # mdl
            cost = likelihood + 0.5 * penalty_factor * np.log(N)

        if cost < best_cost:
            best_cost = cost
            best_k    = k

    return best_k


# =============================================================================
# Per-pass spectrum accumulator
# =============================================================================

class SatellitePassAccumulator:
    """
    Accumulate 2D-MUSIC (or Capon) spectra across a single satellite pass.

    Motivation
    ----------
    Each Iridium burst provides ~10 ms of signal (10 690 IQ samples at
    1.024 Msps).  A typical L-band pass over the sensor produces 15–40
    detectable bursts over 5–12 minutes.  Weighted-averaging the MUSIC
    pseudospectra in the linear domain yields ≈ √N_bursts improvement in
    signal-to-sidelobe ratio, narrowing the Az/El peak and suppressing
    thermal noise and array-calibration artefacts.

    Usage
    -----
    ::

        acc = SatellitePassAccumulator(cfg)
        for burst in pass_bursts:
            spec, az, el, papr = process_burst(burst)
            acc.update(spec, papr_db=papr,
                       new_pass=pass_tracker.update(...))
        az_best, el_best, papr_best = acc.get_best_estimate()

    Notes
    -----
    * ``new_pass=True`` triggers an automatic reset *before* adding the new
      spectrum.
    * Spectra are accumulated in linear power (10^(spec/10)), re-normalised
      to dB for ``get_best_estimate()`` so that ``find_peak_2d`` works
      unchanged.
    * Weight = PAPR in dB (≥ 0).  Bursts with a poorly resolved peak
      (low PAPR) contribute little to the aggregate.
    """

    def __init__(self, cfg: CrossArrayConfig) -> None:
        self._cfg: CrossArrayConfig          = cfg
        self._acc: Optional[np.ndarray]      = None
        self._weight_sum: float              = 0.0
        self._n_bursts: int                  = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Discard all accumulated spectra and restart."""
        self._acc        = None
        self._weight_sum = 0.0
        self._n_bursts   = 0

    # ------------------------------------------------------------------
    def update(
        self,
        spec:         np.ndarray,
        papr_db:      float,
        new_pass:     bool  = False,
        papr_min_db:  float = 0.0,
    ) -> None:
        """
        Add one burst spectrum to the accumulator.

        Parameters
        ----------
        spec         : (n_el, n_az) float — MUSIC pseudospectrum in dB,
                       peak = 0, floor ≈ −40.
        papr_db      : peak-to-average power ratio [dB].  Negative values are
                       treated as 0.
        new_pass     : if True the accumulator is reset before this spectrum is
                       added (new satellite detected by ``PassTracker``).
        papr_min_db  : minimum PAPR threshold [dB].  Bursts below this value
                       are silently skipped (not accumulated, n_bursts unchanged).
                       Default 0.0 = accept all.
        """
        if new_pass:
            self.reset()

        if papr_db < papr_min_db:
            return

        spec_lin = 10.0 ** (np.clip(spec, -200.0, 0.0) / 10.0)
        # Squared PAPR weighting: high-quality bursts contribute quadratically
        # more than marginal ones (e.g. PAPR 3 dB gets 9× more weight than 1 dB).
        w = float(max(papr_db, 0.0)) ** 2

        if self._acc is None:
            self._acc = np.zeros(spec_lin.shape, dtype=np.float64)

        self._acc        += w * spec_lin.astype(np.float64)
        self._weight_sum += w
        self._n_bursts   += 1

    # ------------------------------------------------------------------
    def get_accumulated_spectrum_db(self) -> Optional[np.ndarray]:
        """
        Return the weighted-average spectrum normalised to peak = 0 dB.

        Returns ``None`` if no spectra have been accumulated yet.
        """
        if self._acc is None or self._weight_sum < 1e-15:
            return None
        avg_lin = self._acc / self._weight_sum
        peak    = float(np.max(avg_lin))
        if peak < 1e-30:
            return None
        avg_db = 10.0 * np.log10(avg_lin / peak + 1e-15)
        return np.clip(avg_db, -40.0, 0.0)

    # ------------------------------------------------------------------
    def get_best_estimate(self) -> Tuple[float, float, float]:
        """
        Return ``(az_deg, el_deg, papr_db)`` of the dominant peak in the
        accumulated spectrum.

        Falls back to ``(0.0, 0.0, 0.0)`` if the accumulator is empty.
        """
        spec_db = self.get_accumulated_spectrum_db()
        if spec_db is None:
            return 0.0, 0.0, 0.0
        return find_peak_2d(spec_db, self._cfg)

    # ------------------------------------------------------------------
    @property
    def n_bursts(self) -> int:
        """Number of burst spectra accumulated since the last reset."""
        return self._n_bursts

    @property
    def is_empty(self) -> bool:
        """``True`` if no spectra have been accumulated yet."""
        return self._acc is None
