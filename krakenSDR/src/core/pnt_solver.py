"""
pnt_solver.py — Solve for the *observer's* position from Iridium bursts.

The DOA pipeline answers "where is the satellite, given where I am".  This
inverts it: given where the satellites are — from `core/broadcast_ephemeris.py`,
decoded off the air, no TLE and no network — solve for where the receiver is.
That is the practical point of doing direction finding on Iridium at all: a
GNSS-independent position fix from signals of opportunity.

Two observables, with very different power:

* **Doppler.**  Measured against a real pass, the carrier offset matches the
  ephemeris prediction to ~46 Hz (MAD) once a single receiver LO offset is
  removed.  At 1626 MHz, moving the observer 1 km changes the Doppler by about
  14 Hz north / 6.5 Hz east over a pass, so this is a strong constraint, and its
  cost surface has one clean minimum over hundreds of km.
* **Angle of arrival.**  Roughly 4° of error at ~800 km slant range is ~55 km of
  cross-range, so angles alone give a poor fix — measured at 150 km on the
  reference session.  They earn their place by supplying the *initial guess*
  (in closed form, from a single burst) and an independent geometric check.

Unknowns are ``[lat, lon, delta_f]``.  Altitude is held fixed: a planar UCA
barely constrains height, and on a known-height fix the vertical is the one
thing a map already tells you.  ``delta_f`` — the receiver's LO offset, ~+3.3 kHz
(2 ppm) on this hardware — is one unknown shared by every observation in the
session, so it is heavily over-determined and does not compete with position.

This module is deliberately free of I/O, skyfield and any ephemeris source: it
takes satellite states as plain arrays.  The satellite's state does not depend on
the observer, so the caller evaluates it once per epoch and the solver then
iterates over candidate positions with pure vector algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C_LIGHT_KM_S = 299_792.458
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_MEAN_R_KM = 6371.0088


# ── geometry ─────────────────────────────────────────────────────────────────

def observer_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Geodetic (WGS-84) -> ECEF [km]."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    h = alt_m / 1000.0
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([
        (n + h) * np.cos(lat) * np.cos(lon),
        (n + h) * np.cos(lat) * np.sin(lon),
        (n * (1.0 - WGS84_E2) + h) * np.sin(lat),
    ])


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """(3, 3) matrix whose rows are the local East, North, Up unit vectors."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    sla, cla, slo, clo = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([
        [-slo, clo, 0.0],
        [-sla * clo, -sla * slo, cla],
        [cla * clo, cla * slo, sla],
    ])


def predicted_doppler(sat_pos: np.ndarray, sat_vel: np.ndarray, obs: np.ndarray,
                      f0_hz: float) -> np.ndarray:
    """Doppler shift [Hz] of a transmitter at (pos, vel) seen from ``obs``.

    Everything is in ECEF, so ``sat_vel`` is velocity *in the rotating frame* and
    the range rate below already accounts for Earth rotation — no extra term.
    Computed analytically rather than by differencing range over a time step,
    which smooths the curve by hundreds of Hz at coarse steps.
    """
    d = sat_pos - obs
    rho = np.linalg.norm(d, axis=-1)
    range_rate = np.sum(d * sat_vel, axis=-1) / rho
    return -range_rate * f0_hz / C_LIGHT_KM_S


def predicted_los_enu(sat_pos: np.ndarray, obs: np.ndarray, lat_deg: float,
                      lon_deg: float) -> np.ndarray:
    """Unit line-of-sight vectors in the observer's ENU frame, shape (..., 3)."""
    d = sat_pos - obs
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return d @ enu_basis(lat_deg, lon_deg).T


def azel_to_enu(az_deg, el_deg) -> np.ndarray:
    """(az, el) [deg] -> unit vectors (..., 3) in East/North/Up.

    Mirrors ``shared.geo_utils.az_el_to_unit`` but vectorised; kept here so this
    module stays importable without the repo-root package on the path.
    """
    az, el = np.deg2rad(np.asarray(az_deg)), np.deg2rad(np.asarray(el_deg))
    ce = np.cos(el)
    return np.stack([ce * np.sin(az), ce * np.cos(az), np.sin(el)], axis=-1)


def initial_guess_from_burst(sat_pos: np.ndarray, az_deg: float, el_deg: float,
                             ) -> tuple[float, float]:
    """Closed-form observer guess from ONE burst: satellite position + (az, el).

    A measured elevation fixes how far the observer is from the sub-satellite
    point, because the satellite's altitude is known from its own position:

        gamma = arccos( R * cos(el) / (R + h) ) - el

    and the measured azimuth fixes the direction, so the observer sits at
    angular distance ``gamma`` from the sub-satellite point along bearing
    ``az + 180``.  With ~4 deg of elevation error this lands a few hundred km
    out, which is comfortably inside the basin of the Doppler cost surface.
    """
    r = float(np.linalg.norm(sat_pos))
    lat_s = np.arcsin(np.clip(sat_pos[2] / r, -1.0, 1.0))
    lon_s = np.arctan2(sat_pos[1], sat_pos[0])

    el = np.deg2rad(el_deg)
    gamma = np.arccos(np.clip(EARTH_MEAN_R_KM * np.cos(el) / r, -1.0, 1.0)) - el
    gamma = max(float(gamma), 0.0)

    brg = np.deg2rad(az_deg + 180.0)
    lat = np.arcsin(np.sin(lat_s) * np.cos(gamma)
                    + np.cos(lat_s) * np.sin(gamma) * np.cos(brg))
    lon = lon_s + np.arctan2(np.sin(brg) * np.sin(gamma) * np.cos(lat_s),
                             np.cos(gamma) - np.sin(lat_s) * np.sin(lat))
    return float(np.rad2deg(lat)), float((np.rad2deg(lon) + 540.0) % 360.0 - 180.0)


# ── observations and solution ────────────────────────────────────────────────

@dataclass
class Observations:
    """Bursts already paired with the transmitting satellite's state.

    ``sat_pos``/``sat_vel`` are ECEF (N, 3) in km and km/s, evaluated at each
    burst's epoch.  How they were obtained — decoded IRA frames or SGP4 — is
    none of the solver's business.
    """

    cfo_hz: np.ndarray            # (N,) measured carrier offset
    sat_pos: np.ndarray           # (N, 3)
    sat_vel: np.ndarray           # (N, 3)
    az_deg: np.ndarray | None = None
    el_deg: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.cfo_hz = np.asarray(self.cfo_hz, dtype=np.float64)
        self.sat_pos = np.asarray(self.sat_pos, dtype=np.float64)
        self.sat_vel = np.asarray(self.sat_vel, dtype=np.float64)
        if self.az_deg is not None:
            self.az_deg = np.asarray(self.az_deg, dtype=np.float64)
            self.el_deg = np.asarray(self.el_deg, dtype=np.float64)

    def __len__(self) -> int:
        return int(self.cfo_hz.size)


@dataclass
class PntSolution:
    lat: float
    lon: float
    alt_m: float
    delta_f_hz: float
    n_obs: int
    converged: bool
    doppler_rms_hz: float
    angular_rms_deg: float
    cov: np.ndarray | None = None
    sigma_major_km: float = float("nan")
    sigma_minor_km: float = float("nan")
    sigma_bearing_deg: float = float("nan")
    # One entry per frequency-offset group (a single entry unless `groups` was used).
    delta_f_per_group_hz: list[float] = field(default_factory=list)

    def error_km(self, lat_true: float, lon_true: float) -> tuple[float, float, float]:
        """(total, north, east) great-circle-ish error against a known truth [km]."""
        km_per_deg = np.deg2rad(1.0) * EARTH_MEAN_R_KM
        north = (self.lat - lat_true) * km_per_deg
        east = (self.lon - lon_true) * km_per_deg * np.cos(np.deg2rad(lat_true))
        return float(np.hypot(north, east)), float(north), float(east)


def cov_to_ellipse_en(cov: np.ndarray, lat_deg: float,
                      ) -> tuple[float, float, float]:
    """lat/lon covariance -> (semi-major, semi-minor [km], major-axis bearing).

    The covariance is in degrees; the two axes are scaled differently by the
    cos(lat) convergence of meridians, so it must be mapped into a local metric
    frame *before* the eigen-decomposition, not after.
    """
    km_per_deg = np.deg2rad(1.0) * EARTH_MEAN_R_KM
    scale = np.diag([km_per_deg, km_per_deg * np.cos(np.deg2rad(lat_deg))])
    c_en = scale @ cov[:2, :2][::-1, ::-1] @ scale.T   # [lat, lon] -> [north, east]
    vals, vecs = np.linalg.eigh(c_en)
    vals = np.clip(vals, 0.0, None)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    major, minor = np.sqrt(vals[0]), np.sqrt(vals[1])
    bearing = (np.rad2deg(np.arctan2(vecs[1, 0], vecs[0, 0]))) % 360.0
    return float(major), float(minor), float(bearing)


def solve_position(obs: Observations, *, x0: tuple[float, float],
                   alt_m: float = 0.0, f0_hz: float = 1_626_270_000.0,
                   mode: str = "joint", sigma_f_hz: float = 68.0,
                   sigma_ang_deg: float = 4.0, delta_f0_hz: float = 0.0,
                   f_scale: float = 3.0, max_nfev: int = 200,
                   groups: np.ndarray | None = None) -> PntSolution:
    """Robust least-squares fix.  ``mode`` is 'doppler', 'angles' or 'joint'.

    Both residual blocks are divided by their own sigma so the two observables
    are combined on equal, dimensionless footing and ``f_scale`` means the same
    thing for each: a soft_l1 knee measured in standard deviations.  That
    robustness is not optional — the DOA peak list mixes in ghosts and bursts
    from other satellites, which appear as gross Doppler outliers.

    ``groups`` optionally splits the frequency offset into one unknown per group
    (0-based indices, one per observation) instead of a single global one.  With
    one group per satellite this measures each transmitter's own frequency error
    rather than forcing it into the position: on the reference session the four
    satellites differ by ~50 Hz (0.03 ppm), the same pattern appears under two
    independent ephemerides, and letting them float improves the fix by ~20%.
    ``PntSolution.delta_f_hz`` then reports the mean over groups.
    """
    from scipy.optimize import least_squares

    if mode not in ("doppler", "angles", "joint"):
        raise ValueError(f"unknown mode {mode!r}")
    use_dopp = mode in ("doppler", "joint")
    use_ang = mode in ("angles", "joint")
    if use_ang and obs.az_deg is None:
        raise ValueError(f"mode {mode!r} needs az/el in the observations")

    u_meas = azel_to_enu(obs.az_deg, obs.el_deg) if use_ang else None
    sigma_ang = np.deg2rad(sigma_ang_deg)

    # In angles-only mode the LO offset does not appear in any residual, so
    # carrying it as a free parameter leaves the Jacobian rank-deficient and the
    # covariance uninvertible. Solve for position alone and pass it through.
    fit_df = use_dopp
    if groups is None:
        gidx, n_df = None, 1
    else:
        gidx = np.asarray(groups, dtype=int)
        n_df = int(gidx.max()) + 1 if gidx.size else 1
    if not fit_df:
        n_df = 0

    def residuals(x: np.ndarray) -> np.ndarray:
        lat, lon = float(x[0]), float(x[1])
        o = observer_ecef(lat, lon, alt_m)
        parts = []
        if use_dopp:
            df = x[2:][gidx] if gidx is not None else float(x[2])
            pred = predicted_doppler(obs.sat_pos, obs.sat_vel, o, f0_hz) + df
            parts.append((obs.cfo_hz - pred) / sigma_f_hz)
        if use_ang:
            u_pred = predicted_los_enu(obs.sat_pos, o, lat, lon)
            parts.append(((u_meas - u_pred) / sigma_ang).ravel())
        return np.concatenate(parts)

    x_init = np.array([x0[0], x0[1]] + [delta_f0_hz] * n_df, dtype=np.float64)
    res = least_squares(residuals, x_init, loss="soft_l1", f_scale=f_scale,
                        x_scale=[0.01, 0.01] + [1000.0] * n_df,
                        max_nfev=max_nfev)

    lat, lon = float(res.x[0]), float(res.x[1])
    df_all = res.x[2:] if fit_df else np.array([delta_f0_hz])
    df = float(np.mean(df_all))
    o = observer_ecef(lat, lon, alt_m)

    dopp_rms = float("nan")
    if use_dopp:
        df_obs = df_all[gidx] if gidx is not None else df
        r = obs.cfo_hz - (predicted_doppler(obs.sat_pos, obs.sat_vel, o, f0_hz) + df_obs)
        dopp_rms = float(np.sqrt(np.mean(r ** 2)))
    ang_rms = float("nan")
    if use_ang:
        u_pred = predicted_los_enu(obs.sat_pos, o, lat, lon)
        # |u_meas - u_pred| is the chord; for small angles it is the angle in rad.
        ang_rms = float(np.rad2deg(np.sqrt(np.mean(
            np.sum((u_meas - u_pred) ** 2, axis=-1)))))

    cov = None
    major = minor = bearing = float("nan")
    try:
        dof = max(1, res.fun.size - res.x.size)
        s2 = 2.0 * res.cost / dof
        cov = np.linalg.inv(res.jac.T @ res.jac) * s2
        major, minor, bearing = cov_to_ellipse_en(cov, lat)
    except np.linalg.LinAlgError:
        pass

    return PntSolution(lat=lat, lon=lon, alt_m=alt_m, delta_f_hz=df,
                       n_obs=len(obs), converged=bool(res.success),
                       doppler_rms_hz=dopp_rms, angular_rms_deg=ang_rms, cov=cov,
                       sigma_major_km=major, sigma_minor_km=minor,
                       sigma_bearing_deg=bearing,
                       delta_f_per_group_hz=df_all.tolist() if fit_df else [])


def associate_by_doppler(cfo_hz: np.ndarray, sat_states: dict, obs_ecef_km: np.ndarray,
                         f0_hz: float = 1_626_270_000.0, tol_hz: float = 1500.0,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Assign each burst to the satellite whose predicted Doppler fits best.

    ``sat_states`` maps satellite id -> (pos, vel) arrays already evaluated at
    every burst epoch, with NaN where that satellite has no arc covering it.

    This replaces the DOA track clustering for PNT purposes, and is strictly
    better for it: measured against the broadcast ephemeris the Doppler residual
    of correctly-assigned bursts has a MAD of ~46 Hz while satellites sit tens of
    kHz apart, so the assignment is essentially unambiguous — whereas the angle
    tracks demonstrably mix satellites together (~20% of the peaks in the longest
    track of the reference session belong to a different one).

    The unknown LO offset is removed as the median of the winning residuals
    before the tolerance is applied, so no prior calibration is needed.  The
    observer position enters only through a ~14 Hz/km sensitivity: an initial
    guess good to 100 km perturbs the decision by ~1 kHz, far below the
    separation between satellites.

    Returns (sat_id per burst, residual per burst); id is -1 where no satellite
    fits within ``tol_hz``.
    """
    ids = sorted(sat_states)
    n = int(np.asarray(cfo_hz).size)
    resid = np.full((len(ids), n), np.inf)
    for i, sid in enumerate(ids):
        pos, vel = sat_states[sid]
        ok = np.isfinite(pos).all(axis=-1) & np.isfinite(vel).all(axis=-1)
        if not ok.any():
            continue
        pred = predicted_doppler(pos[ok], vel[ok], obs_ecef_km, f0_hz)
        resid[i, ok] = cfo_hz[ok] - pred

    best = np.argmin(np.abs(resid), axis=0)
    best_res = resid[best, np.arange(n)]
    finite = np.isfinite(best_res)
    if not finite.any():
        return np.full(n, -1), best_res
    lo = float(np.median(best_res[finite]))

    # Re-decide once with the LO offset removed: it is common to every satellite,
    # so leaving it in biases the comparison toward whichever sits on that side.
    best = np.argmin(np.abs(resid - lo), axis=0)
    best_res = resid[best, np.arange(n)] - lo

    out = np.array([ids[b] for b in best])
    out[~np.isfinite(best_res) | (np.abs(best_res) > tol_hz)] = -1
    return out, best_res
