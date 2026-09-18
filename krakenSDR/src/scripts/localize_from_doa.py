#!/usr/bin/env python3
"""
localize_from_doa.py — Estimate the OBSERVER's own position from measured DoA.

This closes the loop of the whole project. The normal pipeline assumes the
observer location is known and predicts each satellite's direction from its TLE
("I know where I am -> I predict where the satellite is"). Here we invert it:

    known satellite positions (TLE/SGP4) + measured directions to them
        -> solve for where the receiver is.

This is the "why" of doing DoA on Iridium: a proof of concept that Iridium
bursts can be used as *signals of opportunity* for GNSS-independent positioning
(LEO-PNT). Accuracy is coarse (km-scale) by design — the point is that it works.

Method
------
For each accepted burst we have a measured (az, el) and, from the pipeline's
track clustering + `groundtruth_matches.json`, the satellite it belongs to. The
satellite's true direction as seen from a *candidate* observer position is the
existing forward model `IridiumCatalogue.sat_azel`. We search the observer
(lat, lon) that best reproduces all measured directions, in a robust non-linear
least-squares sense, comparing unit line-of-sight vectors (so azimuth wrap-around
and the cos(el) weighting are handled naturally). Altitude is held at an assumed
value (a 2-D "known height" fix) because elevation is the weak axis of a planar
UCA and barely constrains height.

Satellite identity is taken as given (it is a separately solved sub-problem in
the pipeline). The track clustering that provides `track_ids` is observer-
independent (it clusters measured az/el/CFO), so only the *label* of each track
comes from the ground-truth match table.

Usage (run from krakenSDR/src/):
    python3 scripts/localize_from_doa.py \
        ../data/doa_iridium/session_20260605_110716 --x0-offset-km 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)                      # krakenSDR/src/
_ROOT = os.path.dirname(os.path.dirname(_SRC))     # LARK project root
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.iridium_groundtruth import (  # noqa: E402
    OBSERVER_ALT,
    OBSERVER_LAT,
    OBSERVER_LON,
    load_session_window,
)
from shared.iridium_tle import load_catalogue, use_session_tle  # noqa: E402

_EARTH_R_KM = 6371.0088


# ─────────────────────────────────────────────────────────────────────────────
# Small geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def azel_to_unit(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Vectorised (az, el) [deg] -> (..., 3) unit vectors [East, North, Up]."""
    az = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(el_deg, dtype=np.float64))
    ce = np.cos(el)
    return np.stack([ce * np.sin(az), ce * np.cos(az), np.sin(el)], axis=-1)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points [km]."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * _EARTH_R_KM * np.arcsin(np.sqrt(a)))


def latlon_to_enu_m(lat, lon, lat0, lon0):
    """Local East/North offset [m] of (lat, lon) relative to (lat0, lon0)."""
    east = np.deg2rad(lon - lon0) * _EARTH_R_KM * 1000.0 * np.cos(np.deg2rad(lat0))
    north = np.deg2rad(lat - lat0) * _EARTH_R_KM * 1000.0
    return float(east), float(north)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_measurements(session_dir: str, matches_path: str, min_el: float,
                      min_peaks: int, sat_filter: set[str] | None) -> list[dict]:
    """One clean entry per satellite track: {t_rel, az, el, sat, n}.

    Uses the per-track median direction (``doa_az_med`` / ``doa_el_med``) from
    ``groundtruth_matches.json`` — these are internally consistent with the
    satellite association — joined with ``tracks.json`` for the track's time
    window. This is far cleaner than the raw per-peak directions, whose
    ``track_ids`` numbering does not line up with the match table.
    """
    d = json.load(open(matches_path, encoding="utf-8"))
    tracks_path = os.path.join(session_dir, "tracks.json")
    tracks = {int(t["id"]): t
              for t in json.load(open(tracks_path, encoding="utf-8"))["tracks"]}
    out: list[dict] = []
    for m in d.get("matches", []):
        sat = m.get("satellite")
        tid = m.get("track_id")
        if not sat or tid is None:
            continue
        tr = tracks.get(int(tid))
        if tr is None:
            continue
        el = float(m["doa_el_med"])
        npk = int(m.get("track_n_peaks", 0))
        if el < min_el or npk < min_peaks:
            continue
        if sat_filter and sat not in sat_filter:
            continue
        out.append({
            "t_rel": 0.5 * (float(tr["t_start"]) + float(tr["t_end"])),
            "az": float(m["doa_az_med"]),
            "el": el,
            "sat": sat,
            "n": npk,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inverse solve
# ─────────────────────────────────────────────────────────────────────────────

def build_groups(meas, cat, t0):
    """Group tracks by satellite; precompute skyfield times, LoS, weights.

    Each track is weighted by sqrt(n_bursts) (a longer track has a more reliable
    median direction), normalised to unit mean so `f_scale` stays in radians.
    """
    wmean = float(np.mean(np.sqrt(np.clip([m["n"] for m in meas], 1, 60))))
    by_sat: dict[str, list[dict]] = {}
    for m in meas:
        by_sat.setdefault(m["sat"], []).append(m)

    groups = []
    for sat_name, rows in by_sat.items():
        sat = cat[sat_name]
        dts = [t0 + timedelta(seconds=r["t_rel"]) for r in rows]
        t_sf = cat._ts.from_datetimes(dts)  # vector Time
        u_meas = azel_to_unit([r["az"] for r in rows], [r["el"] for r in rows])
        w = np.sqrt(np.clip([r["n"] for r in rows], 1, 60)) / wmean
        groups.append({
            "sat": sat, "t_sf": t_sf, "u_meas": u_meas, "w": w,
            "n_tracks": len(rows), "n_bursts": int(sum(r["n"] for r in rows)),
        })
    return groups


def make_residuals(groups, cat, alt_fixed, alt_free):
    def residuals(x):
        lat, lon = float(x[0]), float(x[1])
        alt = float(x[2]) if alt_free else alt_fixed
        observer = cat._wgs84.latlon(lat, lon, elevation_m=alt)
        chunks = []
        for g in groups:
            topo = (g["sat"] - observer).at(g["t_sf"])
            el_a, az_a, _ = topo.altaz()
            u_pred = azel_to_unit(az_a.degrees, el_a.degrees)
            chunks.append((g["w"][:, None] * (g["u_meas"] - u_pred)).ravel())
        return np.concatenate(chunks)
    return residuals


def solve(groups, cat, x0, alt_fixed, alt_free, f_scale):
    from scipy.optimize import least_squares

    resid = make_residuals(groups, cat, alt_fixed, alt_free)
    res = least_squares(resid, x0, loss="soft_l1", f_scale=f_scale,
                        x_scale=[1.0, 1.0, 1000.0][:len(x0)])
    # Angular residual RMS: |u_meas-u_pred| ~= angular error in rad for small err.
    ang_rms_deg = float(np.rad2deg(np.sqrt(np.mean(res.fun ** 2)) * np.sqrt(1.5)))
    # Parameter covariance from the Jacobian (Gauss-Newton approximation).
    cov = None
    try:
        dof = max(1, res.fun.size - res.x.size)
        s2 = float(2 * res.cost / dof)
        cov = np.linalg.inv(res.jac.T @ res.jac) * s2
    except np.linalg.LinAlgError:
        pass
    return res, ang_rms_deg, cov


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def cov_to_ellipse_en(cov, lat):
    """1-sigma covariance -> ellipse in the local East/North plane.

    Returns dict with matplotlib params (metres, angle from +East) plus the
    major/minor semi-axes [km] and the compass bearing of the major axis [deg].
    """
    m_per_deg = np.deg2rad(1) * _EARTH_R_KM * 1000.0
    P = cov[:2, :2][[1, 0]][:, [1, 0]]                   # reorder [lat,lon]->[lon,lat]
    J = np.diag([m_per_deg * np.cos(np.deg2rad(lat)), m_per_deg])  # [lon,lat]->[E,N]
    cov_en = J @ P @ J.T
    vals, vecs = np.linalg.eigh(cov_en)                  # ascending
    vals = np.clip(vals, 0.0, None)
    major = vecs[:, 1]                                    # [E, N] of major axis
    return {
        "width_m": 2 * np.sqrt(vals[1]),
        "height_m": 2 * np.sqrt(vals[0]),
        "angle_deg": float(np.degrees(np.arctan2(major[1], major[0]))),
        "major_km": float(np.sqrt(vals[1]) / 1000.0),
        "minor_km": float(np.sqrt(vals[0]) / 1000.0),
        "major_bearing_deg": float(np.degrees(np.arctan2(major[0], major[1])) % 180.0),
    }


def make_plot(out_png, est_lat, est_lon, cov, x0_lat, x0_lon,
              true_lat, true_lon, n_meas, n_sats, err_km):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    # Everything in local ENU metres, true observer at the origin.
    e_est, n_est = latlon_to_enu_m(est_lat, est_lon, true_lat, true_lon)
    e_x0, n_x0 = latlon_to_enu_m(x0_lat, x0_lon, true_lat, true_lon)

    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.scatter([0], [0], marker="*", s=260, color="#1a9850",
               edgecolor="k", zorder=5, label="True observer (GNSS)")
    ax.scatter([e_est], [n_est], marker="o", s=90, color="#d73027",
               edgecolor="k", zorder=5, label=f"DoA estimate ({err_km:.1f} km)")
    ax.scatter([e_x0], [n_x0], marker="x", s=90, color="#4575b4",
               zorder=4, label="Initial guess")
    ax.annotate("", xy=(e_est, n_est), xytext=(e_x0, n_x0),
                arrowprops=dict(arrowstyle="->", color="#4575b4", lw=1.2, alpha=0.6))

    # 1-sigma error ellipse from the lat/lon covariance (converted to metres).
    if cov is not None:
        el = cov_to_ellipse_en(cov, true_lat)
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((e_est, n_est), el["width_m"], el["height_m"],
                             angle=el["angle_deg"], fc="#d73027", ec="#d73027",
                             alpha=0.15, zorder=2, label="1σ ellipse"))

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Observer self-positioning from Iridium DoA\n"
                 f"{n_meas} bursts, {n_sats} satellites — error {err_km:.1f} km")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: method validation + noise sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_self_test(meas, cat, t0, args, out_png):
    """Validate the inversion on synthetic data using the real geometry.

    (1) Noise-free measurements (SGP4 directions at the true position) must
        recover the true position to ~0 km — proves the method + code.
    (2) Add zero-mean Gaussian angular noise at several levels and report the
        resulting position error — isolates the geometric sensitivity (GDOP)
        of THIS satellite geometry from the real-data systematic errors.
    """
    rng = np.random.default_rng(args.seed)
    d = args.x0_offset_km / 111.0
    brg = np.deg2rad(args.x0_bearing_deg)
    x0 = [OBSERVER_LAT + d * np.cos(brg),
          OBSERVER_LON + d * np.sin(brg) / np.cos(np.deg2rad(OBSERVER_LAT))]

    # Truth directions for the real tracks (noise-free reference).
    truth = []
    for m in meas:
        az_p, el_p, _ = cat.sat_azel(m["sat"], OBSERVER_LAT, OBSERVER_LON,
                                     args.alt_m, t0 + timedelta(seconds=m["t_rel"]))
        truth.append((az_p, el_p))

    def solve_with(az_el):
        pert = [dict(m, az=a, el=e) for m, (a, e) in zip(meas, az_el)]
        res, _, _ = solve(build_groups(pert, cat, t0), cat, list(x0),
                          args.alt_m, False, args.f_scale)
        return haversine_km(res.x[0], res.x[1], OBSERVER_LAT, OBSERVER_LON)

    err0 = solve_with(truth)
    print(f"[self-test] noise-free recovery error: {err0:.4f} km "
          f"({'PASS' if err0 < 1.0 else 'FAIL'})")

    sigmas = [0.0, 1.0, 2.0, 3.3, 5.0, 8.0, 12.0]
    means, meds = [], []
    for s in sigmas:
        errs = []
        for _ in range(args.mc_trials):
            noisy = [(a + rng.normal(0, s), e + rng.normal(0, s)) for a, e in truth]
            errs.append(solve_with(noisy))
        errs = np.array(errs)
        means.append(errs.mean())
        meds.append(np.median(errs))
        print(f"  σ={s:4.1f}°  median={np.median(errs):6.1f} km  mean={errs.mean():6.1f} km")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(sigmas, meds, "o-", color="#d73027", label="median error")
    ax.plot(sigmas, means, "s--", color="#4575b4", alpha=0.7, label="mean error")
    ax.axvline(3.3, color="0.5", ls=":", lw=1)
    ax.text(3.35, ax.get_ylim()[1] * 0.92, "pipeline el 3.3°", fontsize=8, color="0.4")
    ax.set_xlabel("Per-measurement angular noise σ [deg]")
    ax.set_ylabel("Horizontal position error [km]")
    ax.set_title(f"Positioning sensitivity to DoA error\n"
                 f"({len(meas)} tracks, {len({m['sat'] for m in meas})} satellites, "
                 f"real geometry, {args.mc_trials} trials)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\n  wrote {out_png}")
    return {"noise_free_km": err0, "sigmas": sigmas,
            "median_km": meds, "mean_km": means}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Estimate observer position from DoA")
    p.add_argument("session_dir")
    p.add_argument("--matches", default="groundtruth_matches.json",
                   help="track->satellite table (relative to session_dir)")
    p.add_argument("--min-el", type=float, default=15.0,
                   help="Discard tracks whose median elevation is below this [deg]")
    p.add_argument("--min-peaks", type=int, default=10,
                   help="Discard tracks shorter than this many bursts")
    p.add_argument("--x0-offset-km", type=float, default=150.0,
                   help="Start the search this far from the true position [km]")
    p.add_argument("--x0-bearing-deg", type=float, default=45.0,
                   help="Bearing of the initial-guess offset [deg from N]")
    p.add_argument("--alt-m", type=float, default=OBSERVER_ALT,
                   help="Assumed observer altitude [m] (held fixed unless --alt-free)")
    p.add_argument("--alt-free", action="store_true",
                   help="Also solve for altitude (poorly constrained)")
    p.add_argument("--f-scale", type=float, default=0.15,
                   help="soft_l1 robust scale on unit-vector residuals")
    p.add_argument("--sat", default="",
                   help="Comma-separated satellite subset, e.g. 'IRIDIUM 100,IRIDIUM 133'")
    p.add_argument("--self-test", action="store_true",
                   help="Validate on synthetic data (noise-free + noise sweep) instead")
    p.add_argument("--mc-trials", type=int, default=30,
                   help="Monte-Carlo trials per noise level in --self-test")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --self-test")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    session_dir = os.path.abspath(args.session_dir.rstrip("/"))
    matches_path = os.path.join(session_dir, args.matches)
    for pth in (matches_path, os.path.join(session_dir, "tracks.json")):
        if not os.path.isfile(pth):
            sys.exit(f"Missing required file: {pth}")

    sat_filter = {s.strip() for s in args.sat.split(",") if s.strip()} or None

    use_session_tle(session_dir)                 # freeze SGP4 elements per session
    cat = load_catalogue()
    t0, _t1, _meta = load_session_window(session_dir)

    meas = load_measurements(session_dir, matches_path, args.min_el,
                             args.min_peaks, sat_filter)
    if not meas:
        sys.exit("No matched tracks above the gates — nothing to solve.")

    if args.self_test:
        out_png = os.path.join(session_dir, "localization_selftest.png")
        return run_self_test(meas, cat, t0, args, out_png)

    groups = build_groups(meas, cat, t0)
    n_tracks = sum(g["n_tracks"] for g in groups)
    n_bursts = sum(g["n_bursts"] for g in groups)
    n_sats = len(groups)
    print(f"[localize] {n_tracks} tracks / {n_bursts} bursts / {n_sats} satellites "
          f"(min_el={args.min_el}°, min_peaks={args.min_peaks})")

    # Initial guess: deliberately offset from the truth (no peeking).
    d = args.x0_offset_km / 111.0
    brg = np.deg2rad(args.x0_bearing_deg)
    x0_lat = OBSERVER_LAT + d * np.cos(brg)
    x0_lon = OBSERVER_LON + d * np.sin(brg) / np.cos(np.deg2rad(OBSERVER_LAT))
    x0 = [x0_lat, x0_lon] + ([args.alt_m] if args.alt_free else [])

    res, ang_rms_deg, cov = solve(groups, cat, x0, args.alt_m, args.alt_free,
                                  args.f_scale)
    est_lat, est_lon = float(res.x[0]), float(res.x[1])
    est_alt = float(res.x[2]) if args.alt_free else args.alt_m

    err_km = haversine_km(est_lat, est_lon, OBSERVER_LAT, OBSERVER_LON)
    err0_km = haversine_km(x0_lat, x0_lon, OBSERVER_LAT, OBSERVER_LON)
    # Decompose the error into North (cross-range) and East components.
    e_err, n_err = latlon_to_enu_m(est_lat, est_lon, OBSERVER_LAT, OBSERVER_LON)
    ellipse = cov_to_ellipse_en(cov, OBSERVER_LAT) if cov is not None else None

    result = {
        "session": os.path.basename(session_dir),
        "n_tracks": n_tracks,
        "n_bursts": n_bursts,
        "n_satellites": n_sats,
        "satellites": sorted({m["sat"] for m in meas}),
        "min_el_deg": args.min_el,
        "est_lat": est_lat,
        "est_lon": est_lon,
        "est_alt_m": est_alt,
        "true_lat": OBSERVER_LAT,
        "true_lon": OBSERVER_LON,
        "error_km": round(err_km, 2),
        "error_north_km": round(n_err / 1000.0, 2),
        "error_east_km": round(e_err / 1000.0, 2),
        "initial_error_km": round(err0_km, 2),
        "sigma_major_km": None if ellipse is None else round(ellipse["major_km"], 1),
        "sigma_minor_km": None if ellipse is None else round(ellipse["minor_km"], 1),
        "sigma_major_bearing_deg": None if ellipse is None else round(ellipse["major_bearing_deg"], 0),
        "angular_residual_rms_deg": round(ang_rms_deg, 2),
        "alt_free": args.alt_free,
        "converged": bool(res.success),
    }
    print(json.dumps(result, indent=2))
    print(f"\n  initial error : {err0_km:8.1f} km")
    print(f"  final  error : {err_km:8.2f} km "
          f"(N {n_err/1000:+.1f} km, E {e_err/1000:+.1f} km)")
    if ellipse is not None:
        print(f"  1σ ellipse   : {ellipse['major_km']:.1f} x {ellipse['minor_km']:.1f} km, "
              f"major axis bearing {ellipse['major_bearing_deg']:.0f}° "
              f"(weak-observability direction)")

    out_json = os.path.join(session_dir, "localization.json")
    json.dump(result, open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"\n  wrote {out_json}")

    if not args.no_plot:
        out_png = os.path.join(session_dir, "localization.png")
        make_plot(out_png, est_lat, est_lon, cov, x0_lat, x0_lon,
                  OBSERVER_LAT, OBSERVER_LON, n_bursts, n_sats, err_km)
        print(f"  wrote {out_png}")

    return result


if __name__ == "__main__":
    main()
