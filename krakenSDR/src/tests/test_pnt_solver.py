"""Tests for core/pnt_solver.py — solving for the observer's own position.

The properties that matter: the forward model must be right (checked against an
independent construction), a noise-free solve must recover the truth, the error
must fall as sigma_f/sqrt(N), and the closed-form single-burst guess must land
inside the Doppler cost basin.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.pnt_solver import (
    C_LIGHT_KM_S,
    EARTH_MEAN_R_KM,
    Observations,
    associate_by_doppler,
    azel_to_enu,
    cov_to_ellipse_en,
    enu_basis,
    initial_guess_from_burst,
    observer_ecef,
    predicted_doppler,
    predicted_los_enu,
    solve_position,
)

F0 = 1_626_270_000.0
MU_EARTH = 398600.4418
R_ORBIT_KM = 7158.0

# Biot / Sophia Antipolis — the reference session's observer.
LAT0, LON0, ALT0 = 43.614642858726036, 7.071836433649546, 372.0


MEAN_MOTION = np.sqrt(MU_EARTH / R_ORBIT_KM**3)          # ~100 min period
# Phase that puts the satellite at the observer's latitude mid-window (t=200 s).
PASS_PHASE = np.deg2rad(LAT0) - MEAN_MOTION * 200.0
# Ground track 3 deg west of the observer.  Chosen to match the conditioning of
# the real reference pass (14.2 Hz/km north, 6.5 Hz/km east); see
# test_overhead_pass_is_east_degenerate for why "directly overhead" is a trap.
PASS_LON_DEG = 4.0


def _pass_states(t: np.ndarray, lon_deg: float = PASS_LON_DEG,
                 phase: float = PASS_PHASE):
    """A polar-orbit pass over roughly the right place, in ECEF.

    Earth rotation is ignored: these tests need a self-consistent (pos, vel)
    pair in one frame, not a faithful orbit.
    """
    n = MEAN_MOTION
    th = n * t + phase
    lam = np.deg2rad(lon_deg)
    # Orbit plane through the poles at longitude `lam`.
    pos = R_ORBIT_KM * np.stack(
        [np.cos(th) * np.cos(lam), np.cos(th) * np.sin(lam), np.sin(th)], axis=-1)
    vel = R_ORBIT_KM * n * np.stack(
        [-np.sin(th) * np.cos(lam), -np.sin(th) * np.sin(lam), np.cos(th)], axis=-1)
    return pos, vel


def _observations(t, lat=LAT0, lon=LON0, delta_f=3335.0, noise_hz=0.0, seed=0,
                  noise_ang_deg=0.0, **kw):
    pos, vel = _pass_states(t, **kw)
    o = observer_ecef(lat, lon, ALT0)
    cfo = predicted_doppler(pos, vel, o, F0) + delta_f
    u = predicted_los_enu(pos, o, lat, lon)
    el = np.rad2deg(np.arcsin(np.clip(u[:, 2], -1, 1)))
    az = np.rad2deg(np.arctan2(u[:, 0], u[:, 1])) % 360.0
    if noise_hz or noise_ang_deg:
        rng = np.random.default_rng(seed)
        cfo = cfo + rng.normal(0.0, noise_hz, cfo.shape)
        az = az + rng.normal(0.0, noise_ang_deg, az.shape)
        el = el + rng.normal(0.0, noise_ang_deg, el.shape)
    return Observations(cfo_hz=cfo, sat_pos=pos, sat_vel=vel, az_deg=az, el_deg=el)


# ── geometry primitives ──────────────────────────────────────────────────────

def test_observer_ecef_matches_known_values():
    """Equator and pole are the two places the WGS-84 formula is checkable by hand."""
    eq = observer_ecef(0.0, 0.0, 0.0)
    assert eq == pytest.approx([6378.137, 0.0, 0.0], abs=1e-6)
    pole = observer_ecef(90.0, 0.0, 0.0)
    assert pole[2] == pytest.approx(6356.752314, abs=1e-5)  # semi-minor axis
    assert np.linalg.norm(pole[:2]) < 1e-9


def test_enu_basis_is_orthonormal_and_up_is_radial():
    b = enu_basis(LAT0, LON0)
    assert b @ b.T == pytest.approx(np.eye(3), abs=1e-12)
    up = observer_ecef(LAT0, LON0, 0.0)
    assert b[2] @ (up / np.linalg.norm(up)) == pytest.approx(1.0, abs=1e-4)


def test_azel_enu_roundtrip():
    az, el = np.array([0.0, 90.0, 187.0, 300.0]), np.array([5.0, 45.0, 80.0, 12.0])
    u = azel_to_enu(az, el)
    assert np.rad2deg(np.arcsin(u[:, 2])) == pytest.approx(el)
    assert np.rad2deg(np.arctan2(u[:, 0], u[:, 1])) % 360.0 == pytest.approx(az)


def test_predicted_doppler_sign_and_magnitude():
    """Closing range must give a positive shift, and LEO tops out near 40 kHz."""
    o = observer_ecef(0.0, 0.0, 0.0)
    pos = np.array([R_ORBIT_KM, 0.0, 0.0])
    approaching = predicted_doppler(pos, np.array([-3.0, 0.0, 0.0]), o, F0)
    assert approaching > 0
    receding = predicted_doppler(pos, np.array([3.0, 0.0, 0.0]), o, F0)
    assert receding == pytest.approx(-approaching)
    # A full 7.5 km/s radial closure is the physical ceiling.
    assert abs(predicted_doppler(pos, np.array([-7.5, 0.0, 0.0]), o, F0)) == \
        pytest.approx(7.5 * F0 / C_LIGHT_KM_S, rel=1e-9)


def test_predicted_los_enu_is_consistent_with_azel():
    t = np.arange(0.0, 200.0, 10.0)
    obs = _observations(t)
    o = observer_ecef(LAT0, LON0, ALT0)
    u = predicted_los_enu(obs.sat_pos, o, LAT0, LON0)
    assert u == pytest.approx(azel_to_enu(obs.az_deg, obs.el_deg), abs=1e-9)


# ── initial guess ────────────────────────────────────────────────────────────

def test_initial_guess_is_exact_without_angle_error():
    t = np.array([100.0])
    obs = _observations(t)
    lat, lon = initial_guess_from_burst(obs.sat_pos[0], obs.az_deg[0], obs.el_deg[0])
    km_per_deg = np.deg2rad(1.0) * EARTH_MEAN_R_KM
    err = np.hypot((lat - LAT0) * km_per_deg,
                   (lon - LON0) * km_per_deg * np.cos(np.deg2rad(LAT0)))
    # A spherical-Earth construction against a WGS-84 truth: tens of km is the
    # honest floor here, and it only ever has to seed the optimiser.
    assert err < 50.0


def test_initial_guess_lands_inside_the_doppler_basin():
    """With realistic 4 deg angle error the guess must still be usable."""
    t = np.arange(0.0, 300.0, 5.0)
    rng = np.random.default_rng(3)
    km_per_deg = np.deg2rad(1.0) * EARTH_MEAN_R_KM
    errs = []
    for _ in range(20):
        obs = _observations(t)
        i = int(np.argmax(obs.el_deg))
        lat, lon = initial_guess_from_burst(
            obs.sat_pos[i], obs.az_deg[i] + rng.normal(0, 4.0),
            obs.el_deg[i] + rng.normal(0, 4.0))
        errs.append(np.hypot((lat - LAT0) * km_per_deg,
                             (lon - LON0) * km_per_deg * np.cos(np.deg2rad(LAT0))))
    assert np.median(errs) < 300.0


# ── solving ──────────────────────────────────────────────────────────────────

def test_solve_recovers_truth_without_noise():
    t = np.arange(0.0, 400.0, 2.0)
    obs = _observations(t)
    sol = solve_position(obs, x0=(LAT0 + 1.0, LON0 + 1.0), alt_m=ALT0,
                         f0_hz=F0, mode="doppler")
    assert sol.converged
    total, _, _ = sol.error_km(LAT0, LON0)
    assert total < 0.1, f"{total:.3f} km"
    assert sol.delta_f_hz == pytest.approx(3335.0, abs=1.0)


def test_solve_recovers_lo_offset_and_position_together():
    """The LO offset must not be confusable with position."""
    t = np.arange(0.0, 400.0, 2.0)
    for df in (-5000.0, 0.0, 3335.0, 12000.0):
        obs = _observations(t, delta_f=df)
        sol = solve_position(obs, x0=(LAT0 + 0.5, LON0 - 0.5), alt_m=ALT0,
                             f0_hz=F0, mode="doppler")
        assert sol.error_km(LAT0, LON0)[0] < 0.5
        assert sol.delta_f_hz == pytest.approx(df, abs=2.0)


def test_error_scales_with_noise_over_sqrt_n():
    """The headline claim: position error follows sigma_f / sqrt(N)."""
    t = np.arange(0.0, 400.0, 2.0)
    out = {}
    for sigma in (50.0, 100.0, 300.0, 600.0):
        errs = []
        for seed in range(6):
            obs = _observations(t, noise_hz=sigma, seed=seed)
            sol = solve_position(obs, x0=(LAT0 + 0.5, LON0 + 0.5), alt_m=ALT0,
                                 f0_hz=F0, mode="doppler", sigma_f_hz=sigma)
            errs.append(sol.error_km(LAT0, LON0)[0])
        out[sigma] = float(np.median(errs))

    assert out[50.0] < out[600.0]
    # A 12x increase in noise should give roughly a 12x increase in error;
    # allow generous slack for the geometry-dependent conditioning.
    ratio = out[600.0] / max(out[50.0], 1e-9)
    assert 4.0 < ratio < 40.0, out
    # At the noise actually measured on this receiver (~68 Hz MAD-equivalent)
    # a 200-burst pass has to give a km-class fix, or the premise is wrong.
    assert out[100.0] < 3.0, out


def test_uncertainty_ellipse_covers_the_error():
    """An honest 1-sigma must actually contain the truth most of the time."""
    t = np.arange(0.0, 400.0, 2.0)
    inside = 0
    for seed in range(20):
        obs = _observations(t, noise_hz=100.0, seed=seed)
        sol = solve_position(obs, x0=(LAT0 + 0.5, LON0 + 0.5), alt_m=ALT0,
                             f0_hz=F0, mode="doppler", sigma_f_hz=100.0)
        total, _, _ = sol.error_km(LAT0, LON0)
        if np.isfinite(sol.sigma_major_km) and total < 3.0 * sol.sigma_major_km:
            inside += 1
    assert inside >= 16, f"only {inside}/20 within 3 sigma"


def test_overhead_pass_is_east_degenerate():
    """A pass straight overhead barely constrains the across-track axis.

    Doppler positioning from a single pass is nearly symmetric about the ground
    track, so an observer sitting on it gets almost no across-track information —
    the textbook ambiguity. Worth pinning: it explains why the fix quality
    depends on geometry, and why the reference session (ground track ~3 deg to
    the west) is a favourable case rather than a lucky one.
    """
    t = np.arange(0.0, 400.0, 2.0)
    o = observer_ecef(LAT0, LON0, ALT0)
    km_per_deg = np.deg2rad(1.0) * EARTH_MEAN_R_KM

    def sensitivity(lon_orbit):
        pos, vel = _pass_states(t, lon_deg=lon_orbit)
        f0 = predicted_doppler(pos, vel, o, F0)
        out = []
        for dn, de in ((10.0, 0.0), (0.0, 10.0)):
            o2 = observer_ecef(LAT0 + dn / km_per_deg,
                               LON0 + de / (km_per_deg * np.cos(np.deg2rad(LAT0))),
                               ALT0)
            d = predicted_doppler(pos, vel, o2, F0) - f0
            out.append(float((d - d.mean()).std() / 10.0))   # Hz per km
        return out

    n_over, e_over = sensitivity(LON0)          # ground track through the observer
    n_off, e_off = sensitivity(PASS_LON_DEG)    # the geometry used elsewhere here

    assert e_over < 1.0, "an overhead pass should be east-degenerate"
    assert e_off > 4.0, "the off-track geometry must actually constrain east"
    assert n_over > 10.0 and n_off > 10.0, "north is well constrained either way"


def test_angles_only_is_much_worse_than_doppler():
    """The reason this module exists: angles alone do not make a useful fix."""
    t = np.arange(0.0, 400.0, 2.0)
    obs = _observations(t, noise_hz=68.0, noise_ang_deg=4.0, seed=1)
    kw = dict(x0=(LAT0 + 0.5, LON0 + 0.5), alt_m=ALT0, f0_hz=F0,
              sigma_f_hz=68.0, sigma_ang_deg=4.0)
    e_ang = solve_position(obs, mode="angles", **kw).error_km(LAT0, LON0)[0]
    e_dop = solve_position(obs, mode="doppler", **kw).error_km(LAT0, LON0)[0]
    assert e_dop < e_ang


def test_solve_is_robust_to_gross_outliers():
    """Ghost peaks and other satellites show up as huge Doppler residuals."""
    t = np.arange(0.0, 400.0, 2.0)
    obs = _observations(t, noise_hz=68.0, seed=2)
    obs.cfo_hz[::7] += 15000.0
    sol = solve_position(obs, x0=(LAT0 + 0.5, LON0 + 0.5), alt_m=ALT0,
                         f0_hz=F0, mode="doppler", sigma_f_hz=68.0)
    assert sol.error_km(LAT0, LON0)[0] < 20.0


def test_per_group_frequency_offsets():
    """Each transmitter can have its own frequency error; forcing one global
    offset pushes that error into the position instead."""
    t = np.arange(0.0, 300.0, 3.0)
    o = observer_ecef(LAT0, LON0, ALT0)
    offsets = [3400.0, 3455.0]          # ~55 Hz apart, as measured on real data
    pos_l, vel_l, cfo_l, grp = [], [], [], []
    for k, (lon_orbit, off) in enumerate(zip((PASS_LON_DEG, 16.0), offsets)):
        p, v = _pass_states(t, lon_deg=lon_orbit)
        pos_l.append(p)
        vel_l.append(v)
        cfo_l.append(predicted_doppler(p, v, o, F0) + off)
        grp.append(np.full(t.size, k))
    obs = Observations(cfo_hz=np.concatenate(cfo_l), sat_pos=np.vstack(pos_l),
                       sat_vel=np.vstack(vel_l))
    kw = dict(x0=(LAT0 + 0.3, LON0 + 0.3), alt_m=ALT0, f0_hz=F0, mode="doppler")

    grouped = solve_position(obs, groups=np.concatenate(grp), **kw)
    assert grouped.delta_f_per_group_hz == pytest.approx(offsets, abs=2.0)
    assert grouped.error_km(LAT0, LON0)[0] < 0.5

    single = solve_position(obs, **kw)
    assert len(single.delta_f_per_group_hz) == 1
    assert grouped.error_km(LAT0, LON0)[0] < single.error_km(LAT0, LON0)[0]


def test_solve_rejects_unknown_mode():
    obs = _observations(np.arange(0.0, 50.0, 5.0))
    with pytest.raises(ValueError, match="unknown mode"):
        solve_position(obs, x0=(LAT0, LON0), mode="magic")


def test_angles_mode_requires_angles():
    obs = _observations(np.arange(0.0, 50.0, 5.0))
    obs.az_deg = None
    with pytest.raises(ValueError, match="needs az/el"):
        solve_position(obs, x0=(LAT0, LON0), mode="angles")


# ── association ──────────────────────────────────────────────────────────────

def test_associate_by_doppler_separates_satellites():
    t = np.arange(0.0, 300.0, 3.0)
    a_pos, a_vel = _pass_states(t, lon_deg=7.0, phase=-0.35)
    b_pos, b_vel = _pass_states(t, lon_deg=20.0, phase=0.20)
    o = observer_ecef(LAT0, LON0, ALT0)

    lo = 3335.0
    cfo_a = predicted_doppler(a_pos, a_vel, o, F0) + lo
    cfo_b = predicted_doppler(b_pos, b_vel, o, F0) + lo
    truth = np.array([73] * t.size + [89] * t.size)
    cfo = np.concatenate([cfo_a, cfo_b])
    states = {73: (np.vstack([a_pos, a_pos]), np.vstack([a_vel, a_vel])),
              89: (np.vstack([b_pos, b_pos]), np.vstack([b_vel, b_vel]))}

    ids, resid = associate_by_doppler(cfo, states, o, f0_hz=F0)
    assert (ids == truth).mean() > 0.95
    assert np.median(np.abs(resid)) < 50.0


def test_associate_flags_unexplained_bursts():
    """A burst matching nothing must come back as -1, not be forced onto a satellite."""
    t = np.arange(0.0, 100.0, 5.0)
    pos, vel = _pass_states(t)
    o = observer_ecef(LAT0, LON0, ALT0)
    cfo = predicted_doppler(pos, vel, o, F0)
    cfo[0] += 40000.0
    ids, _ = associate_by_doppler(cfo, {73: (pos, vel)}, o, f0_hz=F0)
    assert ids[0] == -1
    assert (ids[1:] == 73).all()


# ── covariance ───────────────────────────────────────────────────────────────

def test_cov_to_ellipse_handles_meridian_convergence():
    """A degree of longitude is shorter than a degree of latitude at 43 N."""
    cov = np.diag([1e-4, 1e-4, 1.0])          # equal in degrees...
    major, minor, _ = cov_to_ellipse_en(cov, 43.6)
    assert major > minor                       # ... so unequal in km
    assert minor / major == pytest.approx(np.cos(np.deg2rad(43.6)), rel=1e-3)
