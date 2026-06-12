"""
track_filter.py — Kalman filter + RTS smoother for per-burst DOA tracks.

Per-burst DOA estimates are noisy but nearly white (validated az MAD ≈ 3.8°,
el MAD ≈ 5.5° against TLE), while the true satellite trajectory is smooth
(Iridium peak angular rate ≈ 0.6°/s).  Smoothing along the track therefore
removes most of the random error and leaves only the systematic part.

Why filter in unit-vector space
-------------------------------
Filtering azimuth/elevation directly is fragile: azimuth wraps at 0°/360°
and changes arbitrarily fast near the zenith.  Instead the state is the
East-North-Up *unit vector* of the direction of arrival plus its time
derivative (6 states, constant-velocity model).  The measurement — the unit
vector of the measured (az, el) — is then a *linear* function of the state,
so a plain Kalman filter suffices (no EKF), and wrap/zenith issues vanish.

Offline we always run the full forward Kalman pass followed by a
Rauch–Tung–Striebel (RTS) backward pass, which conditions every estimate on
the *whole* track rather than just the past.

Conventions (same as the rest of the pipeline):
    azimuth   0° = geographic North, increasing clockwise (compass)
    elevation degrees above the horizon
    ENU       u = [East, North, Up]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "azel_to_unit",
    "unit_to_azel",
    "kalman_smooth_track",
    "kalman_smooth_track_robust",
    "SmoothedTrack",
]

# Defaults from the validated reference session (sigma ≈ 1.4826 × MAD).
DEFAULT_SIGMA_AZ_DEG = 5.6
DEFAULT_SIGMA_EL_DEG = 8.2

# Process noise: white angular acceleration [rad/s²].  Iridium passes reach
# ~0.01 rad/s angular rate building up over ~100 s (peak ~1e-4 rad/s²).
# Tuned on the reference session WITH the robust outlier handling enabled
# (without gating/rejection the optimum was 5e-5 — the heavy-tailed outliers
# pushed it looser than the physics).
DEFAULT_SIGMA_ACC = 2e-5

# Innovation gate: chi-square threshold (3 dof).  11.3 ≈ 99th percentile —
# measurements whose normalised innovation exceeds it are skipped (the filter
# coasts on the prediction), so single outliers cannot yank the track.
DEFAULT_GATE_CHI2 = 11.3


# =============================================================================
# Angle ↔ unit-vector conversions
# =============================================================================

def azel_to_unit(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """(az, el) [deg] → ENU unit vectors, shape (..., 3)."""
    az = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(el_deg, dtype=np.float64))
    return np.stack([
        np.cos(el) * np.sin(az),   # East
        np.cos(el) * np.cos(az),   # North
        np.sin(el),                # Up
    ], axis=-1)


def unit_to_azel(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ENU vectors (..., 3) → (az, el) [deg]; az in [0, 360). Normalizes u."""
    u = np.asarray(u, dtype=np.float64)
    n = np.linalg.norm(u, axis=-1, keepdims=True)
    n = np.where(n > 0.0, n, 1.0)
    e, no, up = (u / n)[..., 0], (u / n)[..., 1], (u / n)[..., 2]
    az = np.rad2deg(np.arctan2(e, no)) % 360.0
    el = np.rad2deg(np.arcsin(np.clip(up, -1.0, 1.0)))
    return az, el


def _tangent_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Local tangent unit vectors at direction ``u`` (one vector, shape (3,)):
    e_az (direction of increasing azimuth) and e_el (increasing elevation).
    """
    e, no, up = u
    cos_el = float(np.hypot(e, no))
    if cos_el < 1e-9:                       # zenith: any horizontal pair works
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    # d(u)/d(az) / cos(el) and d(u)/d(el), both unit length.
    e_az = np.array([no, -e, 0.0]) / cos_el
    e_el = np.array([-e * up, -no * up, cos_el**2]) / cos_el
    e_el /= np.linalg.norm(e_el)
    return e_az, e_el


# =============================================================================
# Kalman filter + RTS smoother
# =============================================================================

@dataclass
class SmoothedTrack:
    """Output of kalman_smooth_track (all arrays have length n_points)."""
    t: np.ndarray            # input times [s]
    az_deg: np.ndarray       # smoothed azimuth [deg]
    el_deg: np.ndarray       # smoothed elevation [deg]
    sigma_az_deg: np.ndarray  # 1σ posterior azimuth uncertainty [deg]
    sigma_el_deg: np.ndarray  # 1σ posterior elevation uncertainty [deg]
    outlier: np.ndarray | None = None  # True where the measurement was rejected


def _transition(dt: float, sigma_acc: float) -> tuple[np.ndarray, np.ndarray]:
    """State transition F and process noise Q for one constant-velocity step."""
    I3 = np.eye(3)
    F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
    q = sigma_acc ** 2
    Q = q * np.block([
        [dt**4 / 4.0 * I3, dt**3 / 2.0 * I3],
        [dt**3 / 2.0 * I3, dt**2 * I3],
    ])
    return F, Q


def _measurement_cov(u_meas: np.ndarray, sigma_az: float, sigma_el: float) -> np.ndarray:
    """
    3×3 measurement covariance for a unit-vector observation.

    Angular errors live in the tangent plane at the measured direction:
    sigma_az·cos(el) along e_az and sigma_el along e_el (small-angle chord
    approximation).  The radial term must NOT be made small: the "radial"
    direction is taken at the *noisy* measurement, so a stiff radial
    constraint would inject large fictitious corrections (the |u| = 1
    constraint is enforced by renormalization instead).
    """
    e_az, e_el = _tangent_basis(u_meas)
    cos_el = float(np.hypot(u_meas[0], u_meas[1]))
    s_az = max(sigma_az * cos_el, 1e-4)
    radial = max(s_az, sigma_el)
    return (
        s_az**2 * np.outer(e_az, e_az)
        + sigma_el**2 * np.outer(e_el, e_el)
        + radial**2 * np.outer(u_meas, u_meas)
    )


def kalman_smooth_track(
    t: np.ndarray,
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    *,
    sigma_az_deg: float = DEFAULT_SIGMA_AZ_DEG,
    sigma_el_deg: float = DEFAULT_SIGMA_EL_DEG,
    sigma_acc: float = DEFAULT_SIGMA_ACC,
    weights: np.ndarray | None = None,
    gate_chi2: float | None = DEFAULT_GATE_CHI2,
) -> SmoothedTrack:
    """
    Smooth one DOA track with a forward Kalman pass + RTS backward pass.

    Parameters
    ----------
    t            : burst times [s], strictly increasing (not necessarily uniform)
    az_deg       : measured azimuths [deg, compass]
    el_deg       : measured elevations [deg]
    sigma_az_deg : 1σ per-burst azimuth noise (defaults from the validated
                   reference session, ≈ 1.4826 × MAD)
    sigma_el_deg : 1σ per-burst elevation noise
    sigma_acc    : process noise — white angular acceleration [rad/s²].
                   Larger → follows measurements more, smooths less.
    weights      : optional per-point quality weights (e.g. linear SNR);
                   the measurement covariance is scaled by median(w)/w_i,
                   clipped to [1/10, 10] so no single point dominates.
    gate_chi2    : innovation gate (chi-square, 3 dof); measurements above it
                   are skipped and flagged in ``outlier``.  None disables.

    Returns
    -------
    SmoothedTrack with smoothed az/el, 1σ uncertainties, and the gate mask.
    Tracks with fewer than 3 points are returned unsmoothed.
    """
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    z = azel_to_unit(az_deg, el_deg)                    # (n, 3) measurements
    s_az = np.deg2rad(sigma_az_deg)
    s_el = np.deg2rad(sigma_el_deg)

    if n < 3:
        sig_az = np.full(n, sigma_az_deg)
        sig_el = np.full(n, sigma_el_deg)
        az, el = unit_to_azel(z)
        return SmoothedTrack(t, az, el, sig_az, sig_el, np.zeros(n, bool))
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing")

    if weights is None:
        r_scale = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64)
        r_scale = np.clip(np.median(w[w > 0]) / np.maximum(w, 1e-12), 0.1, 10.0)

    H = np.hstack([np.eye(3), np.zeros((3, 3))])        # measure u only

    # Forward Kalman pass, storing what RTS needs.
    x = np.concatenate([z[0], np.zeros(3)])
    P = np.diag([s_el**2] * 3 + [(0.02) ** 2] * 3)      # generous initial rate
    xs_f = np.empty((n, 6))                             # filtered means
    Ps_f = np.empty((n, 6, 6))                          # filtered covariances
    xs_p = np.empty((n, 6))                             # predicted means
    Ps_p = np.empty((n, 6, 6))                          # predicted covariances
    Fs = np.empty((n, 6, 6))
    gated = np.zeros(n, dtype=bool)

    for k in range(n):
        if k == 0:
            F = np.eye(6)
            x_pred, P_pred = x, P
        else:
            F, Q = _transition(float(t[k] - t[k - 1]), sigma_acc)
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
        Fs[k], xs_p[k], Ps_p[k] = F, x_pred, P_pred

        R = _measurement_cov(z[k], s_az, s_el) * r_scale[k]
        S = H @ P_pred @ H.T + R
        innov = z[k] - H @ x_pred
        # Innovation gate: never gate the first point (no prior to trust yet).
        if gate_chi2 is not None and k > 0 \
                and float(innov @ np.linalg.solve(S, innov)) > gate_chi2:
            gated[k] = True
            x, P = x_pred.copy(), P_pred
        else:
            K = P_pred @ H.T @ np.linalg.solve(S, np.eye(3))
            x = x_pred + K @ innov
            P = (np.eye(6) - K @ H) @ P_pred
        # Keep the direction part on the unit sphere.
        x[:3] /= np.linalg.norm(x[:3])
        xs_f[k], Ps_f[k] = x, P

    # RTS backward pass.
    xs_s = xs_f.copy()
    Ps_s = Ps_f.copy()
    for k in range(n - 2, -1, -1):
        G = Ps_f[k] @ Fs[k + 1].T @ np.linalg.solve(Ps_p[k + 1], np.eye(6))
        xs_s[k] = xs_f[k] + G @ (xs_s[k + 1] - xs_p[k + 1])
        Ps_s[k] = Ps_f[k] + G @ (Ps_s[k + 1] - Ps_p[k + 1]) @ G.T

    az_s, el_s = unit_to_azel(xs_s[:, :3])

    # Project the position covariance back to angular 1σ.
    sig_az = np.empty(n)
    sig_el = np.empty(n)
    for k in range(n):
        u = xs_s[k, :3] / np.linalg.norm(xs_s[k, :3])
        e_az, e_el = _tangent_basis(u)
        Pu = Ps_s[k, :3, :3]
        cos_el = max(float(np.hypot(u[0], u[1])), 1e-6)
        sig_az[k] = np.rad2deg(np.sqrt(max(e_az @ Pu @ e_az, 0.0)) / cos_el)
        sig_el[k] = np.rad2deg(np.sqrt(max(e_el @ Pu @ e_el, 0.0)))

    return SmoothedTrack(t, az_s, el_s, sig_az, sig_el, gated)


def _angular_sep_deg(az_a, el_a, az_b, el_b) -> np.ndarray:
    """Great-circle separation [deg] between two az/el series."""
    dots = np.sum(azel_to_unit(az_a, el_a) * azel_to_unit(az_b, el_b), axis=-1)
    return np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))


def kalman_smooth_track_robust(
    t: np.ndarray,
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    *,
    k_mad: float = 4.0,
    **kwargs,
) -> SmoothedTrack:
    """
    Two-pass robust smoothing: smooth, reject outliers, re-smooth.

    The Kalman filter is a least-squares estimator, so heavy-tailed DOA
    outliers (sidelobe picks, wrong-tone bursts) still pull the trajectory
    even with the innovation gate.  This wrapper smooths once, flags points
    whose great-circle residual to the smoothed track exceeds
    ``k_mad × 1.4826 × MAD``, re-smooths using inliers only, and fills the
    smoothed value at outlier times by interpolating the inlier solution.

    Accepts the same keyword arguments as :func:`kalman_smooth_track`
    (``weights`` is subset along with the inliers).  ``outlier`` in the
    returned track marks the rejected points.
    """
    t = np.asarray(t, dtype=np.float64)
    az_deg = np.asarray(az_deg, dtype=np.float64)
    el_deg = np.asarray(el_deg, dtype=np.float64)

    first = kalman_smooth_track(t, az_deg, el_deg, **kwargs)
    if len(t) < 10:
        return first

    resid = _angular_sep_deg(az_deg, el_deg, first.az_deg, first.el_deg)
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    bad = resid > max(k_mad * sigma, 1.0)
    if not bad.any():
        return first
    keep = ~bad
    if keep.sum() < 10:
        return first

    sub_kwargs = dict(kwargs)
    if sub_kwargs.get("weights") is not None:
        sub_kwargs["weights"] = np.asarray(sub_kwargs["weights"])[keep]
    sm = kalman_smooth_track(t[keep], az_deg[keep], el_deg[keep], **sub_kwargs)

    # Evaluate at every input time: interpolate the inlier solution through
    # unit-vector space (then renormalise) so azimuth wrap is handled.
    u_in = azel_to_unit(sm.az_deg, sm.el_deg)
    u_all = np.column_stack([np.interp(t, t[keep], u_in[:, i]) for i in range(3)])
    az_all, el_all = unit_to_azel(u_all)
    sig_az = np.interp(t, t[keep], sm.sigma_az_deg)
    sig_el = np.interp(t, t[keep], sm.sigma_el_deg)
    outlier = bad.copy()
    if sm.outlier is not None:
        outlier[keep] |= sm.outlier
    return SmoothedTrack(t, az_all, el_all, sig_az, sig_el, outlier)
