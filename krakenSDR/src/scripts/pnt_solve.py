#!/usr/bin/env python3
"""
pnt_solve.py — Fix the receiver's own position from Iridium bursts.

This is the inverse of everything else in the repo: instead of assuming the
observer is known and predicting where the satellites are, it takes the
satellites as known and solves for the observer.  With ``--ephemeris ira`` the
satellite positions come from ``<session>/broadcast_ephemeris.json``, decoded
from the downlink itself, so the fix uses no TLE, no network and no host clock —
a self-contained GNSS-independent position.

Design notes worth knowing before reading the code:

* **Bursts, not tracks.**  The DOA track clustering is not used.  Measured
  against the broadcast ephemeris, correctly-assigned bursts have a Doppler
  residual with a MAD of ~46 Hz while satellites sit tens of kHz apart, so
  per-burst assignment by Doppler is close to unambiguous — whereas the angle
  tracks demonstrably mix satellites (about a fifth of the peaks in the longest
  track of the reference session belong to a different satellite).
* **Blind start by vote.**  Nothing external says which satellite a burst came
  from, nor roughly where we are.  Every (satellite, burst) pair yields a
  closed-form observer guess; correct pairings all vote for the same place while
  wrong ones scatter, so the densest cell of a coarse lat/lon histogram is the
  seed.  Candidates are then ranked by how many of *all* peaks they explain, not
  by the residual of the subset they happened to assign — that subset-residual
  score rewards solutions that explain almost nothing, and picking it sent an
  early version 1200 km off.
* **Iterate assign/solve.**  Assignment depends weakly on position (~14 Hz/km
  against inter-satellite separations of tens of kHz), so two passes converge.
* **Time matters as much as ephemeris.**  On the reference session the host
  clock is 1.29 s late, which drags a satellite ~10 km along track — and Iridium
  orbits being near-polar, almost due north.  Uncorrected, that alone turns a
  2.7 km fix into a 7.3 km one.  The IRA path takes the correction from IBC;
  ``--clock-offset`` exists so the SGP4 baseline can be compared fairly.

Usage (run from krakenSDR/src/):
    python3 scripts/pnt_solve.py <session> --ephemeris ira --mode joint
    python3 scripts/pnt_solve.py <session> --ephemeris ira --per-satellite
    python3 scripts/pnt_solve.py <session> --ablation --json-out pnt.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.broadcast_ephemeris import ShortArc  # noqa: E402
from core.pnt_solver import (  # noqa: E402
    EARTH_MEAN_R_KM,
    Observations,
    associate_by_doppler,
    initial_guess_from_burst,
    observer_ecef,
    solve_position,
)
from core.recording import session_frame_times  # noqa: E402
from scripts.iridium_groundtruth import (  # noqa: E402
    OBSERVER_LAT,
    OBSERVER_LON,
)

F0_HZ = 1_626_270_000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve for the observer position")
    p.add_argument("session_dir")
    p.add_argument("--subdir", default="doa_multi_music",
                   help="Reprocess subdir holding doa_multi.jsonl")
    p.add_argument("--ephemeris", choices=("ira", "tle"), default="ira",
                   help="ira = decoded from the air (default); tle = SGP4, for comparison")
    p.add_argument("--mode", choices=("doppler", "angles", "joint"), default="joint")
    p.add_argument("--ablation", action="store_true",
                   help="Run all three modes and print the comparison table")
    p.add_argument("--per-satellite", action="store_true",
                   help="Also solve from each satellite alone (single-pass fixes)")
    p.add_argument("--sigma-f", type=float, default=68.0,
                   help="Doppler measurement sigma [Hz] (default: measured 68)")
    p.add_argument("--per-sat-df", action="store_true",
                   help="Fit one frequency offset per satellite instead of one "
                        "global receiver LO offset. The four satellites of the "
                        "reference session differ by ~50 Hz (0.03 ppm) and the "
                        "same pattern shows up under two independent ephemerides, "
                        "so this is a real transmitter effect, not overfitting.")
    p.add_argument("--sigma-ang", type=float, default=4.0,
                   help="Angle measurement sigma [deg]")
    p.add_argument("--assoc-tol", type=float, default=1500.0,
                   help="Max |Doppler residual| to accept an assignment [Hz]")
    p.add_argument("--min-el", type=float, default=0.0, help="Elevation gate [deg]")
    p.add_argument("--max-peak-rank", type=int, default=0,
                   help="Highest DOA peak index to use, 0 = strongest tone only "
                        "(default). Secondary peaks are mostly ghosts: measured "
                        "against the broadcast ephemeris, 24%% of rank-0 peaks are "
                        "Doppler outliers versus 83%% of rank-1 and 75%% of rank-2, "
                        "and including them costs ~1 km of accuracy.")
    p.add_argument("--clock-offset", type=float, default=None,
                   help="Override the host-clock correction [s]. The IRA path takes "
                        "it from IBC; SGP4 has no such source, so pass it here to "
                        "compare the two ephemerides on equal timing.")
    p.add_argument("--x0", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="Explicit initial guess (default: closed form, blind)")
    p.add_argument("--x0-offset-km", type=float, default=0.0,
                   help="Displace the initial guess by this much to bearing 45 deg")
    p.add_argument("--alt", type=float, default=0.0,
                   help="Assumed observer altitude [m] (held fixed). Defaults to "
                        "sea level on purpose: the fix is supposed to claim no "
                        "prior knowledge of where the receiver is, and altitude "
                        "is part of that. It costs almost nothing -- on the "
                        "reference session the true 372 m gives 871 m of error "
                        "and 0 m gives 884 m, so the prior buys 13 m.")
    p.add_argument("--truth", nargs=2, type=float, metavar=("LAT", "LON"),
                   default=[OBSERVER_LAT, OBSERVER_LON],
                   help="Known position, for scoring only")
    p.add_argument("--json-out", default="", help="Write the result JSON here")
    p.add_argument("--plot", default="", help="Write a summary figure here")
    p.add_argument("--plot-figsize", default="16,4.6", metavar="W,H",
                   help="Summary-figure size in inches. The default suits a "
                        "screen; use something like 9,3 when the panels are to "
                        "be split apart and printed.")
    p.add_argument("--gui", action="store_true",
                   help="Interactive replay: scrub listening time and watch the "
                        "fix converge (apps/doa_iridium/pnt_replay.py)")
    p.add_argument("--gui-steps", type=int, default=48,
                   help="Timeline resolution for --gui (one solve per step)")
    return p.parse_args(argv)


# ── data ─────────────────────────────────────────────────────────────────────

def load_peaks(session: str, subdir: str, clock_offset_s: float, min_el: float,
               max_peak_rank: int = 0):
    """DOA peaks as (t_abs, cfo, az, el, snr) — no track clustering.

    Absolute epochs come from the frame write times, corrected by the clock
    offset measured against Iridium system time; the ``t`` field in the JSONL is
    only relative to frame 0.

    ``max_peak_rank`` keeps only the strongest tones per CPI. This matters: the
    secondary peaks are overwhelmingly ghosts (see ADR 001 — they are usually
    elevation harmonics of the same satellite), and they are the main source of
    the heavy residual tails.
    """
    path = os.path.join(session, subdir, "doa_multi.jsonl")
    frame_ts = session_frame_times(session)
    if frame_ts is None:
        raise SystemExit(f"No frame timestamps for {session}")
    fs_hz = _session_fs(session)

    t, cfo, az, el, snr = [], [], [], [], []
    n_no_b0 = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fi = int(rec["frame"])
            if fi >= len(frame_ts):
                continue
            epoch = float(frame_ts[fi]) - clock_offset_s
            b0s = rec.get("b0_per_peak")
            for k, peak in enumerate(rec["peaks"]):
                if k > max_peak_rank or k >= len(rec["cfo_per_peak"]):
                    continue
                if peak[1] < min_el:
                    continue
                # A burst's epoch is its CPI's epoch plus its start inside the
                # CPI -- the convention remap_epochs() applies to the decoded
                # IRA/IBC frames.  Omitting it leaves the bursts ~23 ms early
                # against the ephemeris on the reference session, which alone
                # moved the fix by 0.2 km along track.
                if b0s is not None and k < len(b0s) and b0s[k] >= 0:
                    t.append(epoch + b0s[k] / fs_hz)
                else:
                    n_no_b0 += 1
                    t.append(epoch)
                cfo.append(float(rec["cfo_per_peak"][k]))
                az.append(float(peak[0]))
                el.append(float(peak[1]))
                snr.append(float(rec["snr_per_peak"][k])
                           if k < len(rec["snr_per_peak"]) else 0.0)
    if n_no_b0:
        print(f"[warn] {n_no_b0} peaks have no b0_per_peak: epoch = CPI start "
              f"(run scripts/add_burst_offsets.py on {subdir})", file=sys.stderr)
    return (np.array(t), np.array(cfo), np.array(az), np.array(el), np.array(snr))


def _session_fs(session: str) -> float:
    try:
        with open(os.path.join(session, "meta.json"), encoding="utf-8") as fh:
            return float(json.load(fh).get("fs_hz", 1_024_000.0))
    except (OSError, ValueError):
        return 1_024_000.0


class IraSource:
    """Ephemeris decoded from the air, read back from broadcast_ephemeris.json."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        self.clock_offset_s = float(d.get("clock_applied_s") or 0.0)
        self.arcs = {
            int(s["sat_id"]): [
                ShortArc(t0=a["t0"], scale=a["scale"],
                         coeffs=np.array(a["coeffs"]), n_used=a["n_used"],
                         residual_km=a["residual_km"], t_start=a["t_start"],
                         t_end=a["t_end"])
                for a in s["arcs"]]
            for s in d["satellites"]}
        self.label = {int(s["sat_id"]): f"sat:{int(s['sat_id']):03d}"
                      for s in d["satellites"]}
        for row in d.get("tle_validation", []):
            self.label[int(row["sat_id"])] += f" ({row['tle_name']})"

    @property
    def satellites(self) -> list[int]:
        return sorted(self.arcs)

    def states(self, sat_id: int, epochs: np.ndarray):
        epochs = np.asarray(epochs, dtype=np.float64)
        pos = np.full((epochs.size, 3), np.nan)
        vel = np.full((epochs.size, 3), np.nan)
        for arc in self.arcs.get(sat_id, []):
            m = (epochs >= arc.t_start) & (epochs <= arc.t_end)
            if m.any():
                pos[m], vel[m] = arc.state(epochs[m])
        return pos, vel


class TleSource:
    """SGP4 ephemeris — the comparison baseline, needs the session TLE snapshot."""

    def __init__(self, session: str, names: list[str] | None = None):
        from shared.iridium_tle import load_catalogue, use_session_tle
        self._cat = load_catalogue(str(use_session_tle(session)))
        self._names = names or list(self._cat._by_name)
        self.clock_offset_s = 0.0
        self.label = {i: n for i, n in enumerate(self._names)}

    @property
    def satellites(self) -> list[int]:
        return list(range(len(self._names)))

    def states(self, sat_id: int, epochs: np.ndarray):
        from skyfield.framelib import itrs
        epochs = np.asarray(epochs, dtype=np.float64)
        sat = self._cat._by_name[self._names[sat_id]]
        times = self._cat._ts.from_datetimes(
            [datetime.fromtimestamp(float(x), tz=timezone.utc) for x in epochs])
        geo = sat.at(times)
        r, v = geo.frame_xyz_and_velocity(itrs)
        return r.km.T, v.km_per_s.T


# ── solving ──────────────────────────────────────────────────────────────────

def _states_for_all(src, sats, t):
    return {s: src.states(s, t) for s in sats}


def solve_once(src, sats, data, x0, args, mode, sat_filter=None, n_iter=2):
    """Assign bursts to satellites, solve, repeat.  Returns (solution, ids)."""
    t, cfo, az, el, _ = data
    states = _states_for_all(src, sats, t)
    lat, lon = x0
    sol = ids = None
    for _ in range(n_iter):
        ids, _res = associate_by_doppler(cfo, states, observer_ecef(lat, lon, args.alt),
                                         f0_hz=F0_HZ, tol_hz=args.assoc_tol)
        keep = ids >= 0
        if sat_filter is not None:
            keep &= ids == sat_filter
        if keep.sum() < 10:
            return None, ids
        pos = np.full((int(keep.sum()), 3), np.nan)
        vel = np.full_like(pos, np.nan)
        for j, i in enumerate(np.flatnonzero(keep)):
            p, v = states[int(ids[i])]
            pos[j], vel[j] = p[i], v[i]
        good = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
        if good.sum() < 10:
            return None, ids
        idx = np.flatnonzero(keep)[good]
        obs = Observations(cfo_hz=cfo[idx], sat_pos=pos[good], sat_vel=vel[good],
                           az_deg=az[idx], el_deg=el[idx])
        groups = None
        if args.per_sat_df and sat_filter is None:
            present = sorted(set(ids[idx].tolist()))
            groups = np.array([present.index(int(s)) for s in ids[idx]])
        sol = solve_position(obs, x0=(lat, lon), alt_m=args.alt, f0_hz=F0_HZ,
                             mode=mode, sigma_f_hz=args.sigma_f,
                             sigma_ang_deg=args.sigma_ang, groups=groups,
                             delta_f0_hz=sol.delta_f_hz if sol else 0.0)
        lat, lon = sol.lat, sol.lon
    return sol, ids


def blind_seeds(src, data, *, el_min=20.0, bin_deg=2.0, n_seeds=3):
    """Candidate starting positions, with no prior and no satellite labels.

    Every (satellite, burst) pair gives a closed-form observer guess.  Pairs that
    happen to be correct all vote for the same place; wrong pairings scatter over
    the globe, because a satellite's geometry only admits one observer for a
    given (az, el).  So the densest cell of a coarse lat/lon histogram is the
    answer — on the reference session the winning cell collects a third of all
    votes and its centroid lands ~90 km from the truth, comfortably inside the
    basin of the Doppler cost surface.

    Only high-elevation bursts vote: their guesses are less sensitive to the
    elevation error, which is the weaker DOA axis.
    """
    t, _cfo, az, el, _snr = data
    guesses = []
    for s in src.satellites:
        pos, _ = src.states(s, t)
        ok = np.isfinite(pos).all(axis=1) & (el > el_min)
        for i in np.flatnonzero(ok):
            guesses.append(initial_guess_from_burst(pos[i], az[i], el[i]))
    if not guesses:
        return []

    g = np.array(guesses)
    lat_edges = np.arange(-90.0, 90.0 + bin_deg, bin_deg)
    lon_edges = np.arange(-180.0, 180.0 + bin_deg, bin_deg)
    hist, _, _ = np.histogram2d(g[:, 0], g[:, 1], bins=[lat_edges, lon_edges])

    out = []
    for flat in np.argsort(hist, axis=None)[::-1][:n_seeds]:
        a, b = np.unravel_index(flat, hist.shape)
        if hist[a, b] <= 0:
            break
        m = ((g[:, 0] >= lat_edges[a]) & (g[:, 0] < lat_edges[a + 1])
             & (g[:, 1] >= lon_edges[b]) & (g[:, 1] < lon_edges[b + 1]))
        out.append((float(g[m, 0].mean()), float(g[m, 1].mean())))
    return out


def _consensus(src, data, sol, args, mode) -> int:
    """How many of ALL peaks the solution explains, not just the assigned ones.

    Scoring a candidate by the residual of its own assigned subset is a trap: a
    badly-placed solution that only manages to assign a handful of bursts scores
    better than a good one that explains most of them.  Counting inliers over
    every peak is the consensus criterion that avoids it — measured with the
    observable the mode actually uses.
    """
    from core.pnt_solver import azel_to_enu, predicted_doppler, predicted_los_enu

    t, cfo, az, el, _snr = data
    o = observer_ecef(sol.lat, sol.lon, args.alt)
    best = np.full(t.size, np.inf)
    u_meas = azel_to_enu(az, el) if mode == "angles" else None

    for s in src.satellites:
        pos, vel = src.states(s, t)
        ok = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
        if not ok.any():
            continue
        if mode == "angles":
            u_pred = predicted_los_enu(pos[ok], o, sol.lat, sol.lon)
            r = np.rad2deg(np.linalg.norm(u_meas[ok] - u_pred, axis=-1))
        else:
            r = np.abs(cfo[ok] - (predicted_doppler(pos[ok], vel[ok], o, F0_HZ)
                                  + sol.delta_f_hz))
        best[ok] = np.minimum(best[ok], r)

    tol = 3.0 * (args.sigma_ang if mode == "angles" else args.sigma_f)
    return int((best < tol).sum())


def blind_solve(src, data, args, mode):
    """No prior on position and no satellite labels: seed by vote, keep the
    candidate that explains the most bursts."""
    sats = src.satellites
    seeds = [tuple(args.x0)] if args.x0 else blind_seeds(src, data)
    if not seeds:
        raise SystemExit("No usable bursts to seed the search.")

    if args.x0_offset_km:
        km_per_deg = np.deg2rad(1.0) * EARTH_MEAN_R_KM
        d = args.x0_offset_km / np.sqrt(2.0)
        seeds = [(la + d / km_per_deg,
                  lo + d / (km_per_deg * np.cos(np.deg2rad(la)))) for la, lo in seeds]

    best = None
    for x0 in seeds:
        sol, ids = solve_once(src, sats, data, x0, args, mode)
        if sol is None:
            continue
        score = _consensus(src, data, sol, args, mode)
        if best is None or score > best[0]:
            best = (score, sol, ids, x0)
    if best is None:
        raise SystemExit("No solution: nothing could be associated to a satellite.")
    print(f"  seed {best[3][0]:.3f} N {best[3][1]:.3f} E -> "
          f"{best[0]} / {data[0].size} peaks explained")
    return best[1], best[2], best[3]


def _report(sol, truth, label=""):
    total, north, east = sol.error_km(*truth)
    print(f"  {label:<26s} n={sol.n_obs:5d}  err {total:7.2f} km "
          f"(N {north:+7.2f}, E {east:+7.2f})  "
          f"sigma {sol.sigma_major_km:6.2f}x{sol.sigma_minor_km:.2f} km  "
          f"dopp_rms {sol.doppler_rms_hz:7.1f} Hz  df {sol.delta_f_hz:+8.1f} Hz")
    return {"label": label, "n_obs": sol.n_obs, "lat": sol.lat, "lon": sol.lon,
            "error_km": round(total, 3), "error_north_km": round(north, 3),
            "error_east_km": round(east, 3),
            "sigma_major_km": round(sol.sigma_major_km, 3),
            "sigma_minor_km": round(sol.sigma_minor_km, 3),
            "delta_f_hz": round(sol.delta_f_hz, 2),
            "doppler_rms_hz": round(sol.doppler_rms_hz, 2),
            "angular_rms_deg": round(sol.angular_rms_deg, 3),
            "converged": sol.converged}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = args.session_dir.rstrip("/")

    if args.ephemeris == "ira":
        path = os.path.join(session, "broadcast_ephemeris.json")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found — run scripts/decode_ephemeris.py first")
        src = IraSource(path)
        print(f"[eph] broadcast: {len(src.satellites)} satellites decoded from the air, "
              f"clock corrected by {src.clock_offset_s:+.3f} s")
    else:
        names = None
        ira_path = os.path.join(session, "broadcast_ephemeris.json")
        if os.path.isfile(ira_path):
            names = sorted({r["tle_name"]
                            for r in json.load(open(ira_path, encoding="utf-8"))
                            .get("tle_validation", [])}) or None
        src = TleSource(session, names)
        print(f"[eph] SGP4 baseline: {len(src.satellites)} satellites")

    clock = src.clock_offset_s if args.clock_offset is None else args.clock_offset
    data = load_peaks(session, args.subdir, clock, args.min_el, args.max_peak_rank)
    print(f"[data] {data[0].size} DOA peaks (rank <= {args.max_peak_rank}), "
          f"clock corrected by {clock:+.3f} s")

    truth = tuple(args.truth)
    modes = ("angles", "doppler", "joint") if args.ablation else (args.mode,)
    results = []
    print(f"\n[fix] truth {truth[0]:.5f} N, {truth[1]:.5f} E "
          f"(scoring only — not used by the solver)")
    main_sol = None
    for mode in modes:
        sol, ids, x0 = blind_solve(src, data, args, mode)
        results.append(_report(sol, truth, f"all sats / {mode}"))
        results[-1]["mode"] = mode
        if mode == args.mode or len(modes) == 1:
            main_sol, main_ids = sol, ids

    if args.per_satellite:
        print("\n[fix] single satellite, single pass:")
        sats = src.satellites
        for s in sats:
            sol, _ = solve_once(src, sats, data, (main_sol.lat, main_sol.lon),
                                args, args.mode, sat_filter=s)
            if sol is not None:
                r = _report(sol, truth, src.label.get(s, str(s)))
                r["sat"] = src.label.get(s, str(s))
                results.append(r)

    out = {"session": os.path.basename(session), "ephemeris": args.ephemeris,
           "mode": args.mode, "n_peaks": int(data[0].size),
           "sigma_f_hz": args.sigma_f, "truth": list(truth),
           "clock_offset_s": clock, "results": results}
    path = args.json_out or os.path.join(session, "pnt_solution.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[fix] wrote {path}")

    if args.plot:
        _plot(args.plot, src, data, main_sol, main_ids, truth, args)
        print(f"[fix] wrote {args.plot}")

    if args.gui:
        # Imported late: the viewer pulls in matplotlib widgets, and this script
        # must stay usable headless.
        from apps.doa_iridium.pnt_replay import PntReplayViewer
        PntReplayViewer(session, src, data, main_ids, alt_m=args.alt,
                        sigma_f=args.sigma_f, mode=args.mode,
                        per_sat_df=args.per_sat_df, truth=truth,
                        steps=args.gui_steps).run()
    return 0


def _plot(path, src, data, sol, ids, truth, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.pnt_solver import predicted_doppler

    t, cfo, az, el, _ = data
    fw, fh = (float(v) for v in str(args.plot_figsize).split(","))
    fig, ax = plt.subplots(1, 3, figsize=(fw, fh))

    # 1. Doppler residual vs time, per satellite.
    o = observer_ecef(sol.lat, sol.lon, args.alt)
    for s in src.satellites:
        m = ids == s
        if not m.any():
            continue
        pos, vel = src.states(s, t[m])
        ok = np.isfinite(pos).all(axis=1)
        r = cfo[m][ok] - (predicted_doppler(pos[ok], vel[ok], o, F0_HZ) + sol.delta_f_hz)
        ax[0].plot((t[m][ok] - t.min()) / 60.0, r, ".", ms=3,
                   label=src.label.get(s, str(s)))
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(xlabel="time [min]", ylabel="Doppler residual [Hz]",
              title=f"residual at the fix (rms {sol.doppler_rms_hz:.0f} Hz)",
              ylim=(-4 * args.sigma_f * 5, 4 * args.sigma_f * 5))
    ax[0].legend(fontsize=7, markerscale=3)

    # 2. Cost surface around the solution.
    km_per_deg = np.deg2rad(1.0) * EARTH_MEAN_R_KM
    span = np.linspace(-40.0, 40.0, 41)
    states = _states_for_all(src, src.satellites, t)
    keep = ids >= 0
    pos = np.full((int(keep.sum()), 3), np.nan)
    vel = np.full_like(pos, np.nan)
    for j, i in enumerate(np.flatnonzero(keep)):
        p, v = states[int(ids[i])]
        pos[j], vel[j] = p[i], v[i]
    good = np.isfinite(pos).all(axis=1)
    pos, vel = pos[good], vel[good]
    cf = cfo[np.flatnonzero(keep)[good]]
    Z = np.empty((span.size, span.size))
    for a, dn in enumerate(span):
        for b, de in enumerate(span):
            oo = observer_ecef(sol.lat + dn / km_per_deg,
                               sol.lon + de / (km_per_deg * np.cos(np.deg2rad(sol.lat))),
                               args.alt)
            r = cf - predicted_doppler(pos, vel, oo, F0_HZ)
            Z[a, b] = np.sqrt(np.mean((r - np.median(r)) ** 2))
    im = ax[1].contourf(span, span, Z, levels=20)
    ax[1].plot(0, 0, "w+", ms=12, mew=2, label="fix")
    total, north, east = sol.error_km(*truth)
    ax[1].plot(-north, -east, "rx", ms=10, mew=2, label="truth")
    ax[1].set(xlabel="north offset [km]", ylabel="east offset [km]",
              title="Doppler cost surface")
    ax[1].legend(fontsize=8)
    fig.colorbar(im, ax=ax[1], label="rms residual [Hz]")

    # 3. Sky coverage of the peaks that were used.
    axp = fig.add_subplot(1, 3, 3, projection="polar")
    ax[2].remove()
    for s in src.satellites:
        m = ids == s
        if m.any():
            axp.plot(np.deg2rad(az[m]), 90.0 - el[m], ".", ms=3,
                     label=src.label.get(s, str(s)))
    axp.set_theta_zero_location("N")
    axp.set_theta_direction(-1)
    axp.set_rmax(90)
    axp.set_title(f"assigned peaks\nfix error {total:.2f} km", fontsize=10)

    fig.suptitle(f"{os.path.basename(args.session_dir.rstrip('/'))} — "
                 f"ephemeris: {args.ephemeris}, mode: {args.mode}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
