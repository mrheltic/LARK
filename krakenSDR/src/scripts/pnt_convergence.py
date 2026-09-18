#!/usr/bin/env python3
"""
pnt_convergence.py — Headless figure: how the position fix converges with
listening time.

A signal-of-opportunity fix is not instantaneous.  Starting from a blind
closed-form guess, the solver walks in as bursts accumulate and satellites rise
and set, then stops improving once the error becomes systematic rather than
statistical.  This script renders that story as a static figure for print,
reusing exactly the computation behind ``pnt_solve.py --gui``.

Two panels:

  left   the walk-in, in local East/North kilometres, from the blind seed to the
         converged fix, with the final 1-sigma ellipse and the true position
  right  absolute error and 1-sigma major axis against listening time

Usage (run from krakenSDR/src/):

    python3 scripts/pnt_convergence.py ../data/doa_iridium/session_.../ \
        --ephemeris ira --mode doppler --per-sat-df --out /tmp/pnt_convergence
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.pnt_solver import (  # noqa: E402
    EARTH_MEAN_R_KM,
    Observations,
    solve_position,
)
from scripts.pnt_solve import (  # noqa: E402
    F0_HZ,
    IraSource,
    TleSource,
    blind_solve,
    load_peaks,
    parse_args as pnt_parse_args,
)

KM_PER_DEG = np.deg2rad(1.0) * EARTH_MEAN_R_KM


def enu_km(lat, lon, lat0, lon0):
    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    east = (lon - lon0) * KM_PER_DEG * np.cos(np.deg2rad(lat0))
    north = (lat - lat0) * KM_PER_DEG
    return east, north


def cumulative_fixes(src, data, ids, x0, args, steps: int, min_bursts: int):
    """Solve once per cumulative time cut, exactly as the interactive replay does."""
    t, cfo, az, el, _snr = data
    keep = ids >= 0
    order = np.argsort(t[keep])
    idx = np.flatnonzero(keep)[order]
    t, cfo, az, el, ids = t[idx], cfo[idx], az[idx], el[idx], ids[idx]

    pos = np.full((t.size, 3), np.nan)
    vel = np.full((t.size, 3), np.nan)
    for s in np.unique(ids):
        m = ids == s
        pos[m], vel[m] = src.states(int(s), t[m])
    good = np.isfinite(pos).all(1) & np.isfinite(vel).all(1)
    t, cfo, az, el, ids, pos, vel = (a[good] for a in (t, cfo, az, el, ids, pos, vel))

    counts = np.unique(np.linspace(min(min_bursts, t.size), t.size, steps).astype(int))
    sols = []
    print(f"[conv] {counts.size} cumulative fixes over {t.size} bursts, "
          f"{len(np.unique(ids))} satellites")
    for k, n in enumerate(counts):
        obs = Observations(cfo_hz=cfo[:n], sat_pos=pos[:n], sat_vel=vel[:n],
                           az_deg=az[:n], el_deg=el[:n])
        groups = None
        if args.per_sat_df:
            present = sorted({int(s) for s in ids[:n]})
            groups = np.array([present.index(int(s)) for s in ids[:n]])
        sols.append(solve_position(obs, x0=x0, alt_m=args.alt, f0_hz=F0_HZ,
                                   mode=args.mode, sigma_f_hz=args.sigma_f,
                                   groups=groups))
        if k % 10 == 0:
            print(f"  {k}/{counts.size}", end="\r", flush=True)
    print(f"  {counts.size}/{counts.size} done")

    elapsed = np.array([t[int(n) - 1] - t[0] for n in counts])
    return sols, counts, elapsed


def make_figure(sols, elapsed, x0, truth, out_base: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import thesis_style as ts

    ts.apply(plt)
    C = ts.SERIES

    final = sols[-1]
    lat0, lon0 = truth
    trail_e, trail_n = enu_km([s.lat for s in sols], [s.lon for s in sols], lat0, lon0)
    seed_e, seed_n = enu_km([x0[0]], [x0[1]], lat0, lon0)
    err = np.array([s.error_km(*truth)[0] for s in sols])
    sig = np.array([s.sigma_major_km for s in sols])

    fig, (axm, axc) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ── left: the walk-in ────────────────────────────────────────────────────
    axm.plot(trail_e, trail_n, "-", color="0.55", lw=1.0, zorder=2)
    sc = axm.scatter(trail_e, trail_n, c=elapsed / 60.0, s=16,
                     cmap=ts.sequential_cmap_visible(), zorder=3)
    axm.plot(0, 0, "*", color="k", ms=13, label="true position", zorder=5)
    fe, fn = enu_km([final.lat], [final.lon], lat0, lon0)
    maj, mino = final.sigma_major_km, final.sigma_minor_km
    # sigma_bearing_deg is a compass bearing (CW from north); matplotlib's
    # Ellipse angle is CCW from the east axis.
    axm.add_patch(Ellipse((fe[0], fn[0]), 2 * maj, 2 * mino,
                          angle=90.0 - final.sigma_bearing_deg,
                          fill=False, ec=C[0], lw=1.4, zorder=4,
                          label=r"final $1\sigma$"))
    # The first fixes are hundreds of km out and would flatten the scale, so the
    # panel is framed on the converged part and the seed is annotated off-scale.
    near = err < 10.0
    if near.any():
        span = max(np.abs(trail_e[near]).max(), np.abs(trail_n[near]).max(),
                   maj) * 1.6 + 0.5
    else:
        span = max(np.abs(trail_e).max(), np.abs(trail_n).max()) * 1.1
    axm.set_xlim(-span, span)
    axm.set_ylim(-span, span)
    seed_km = float(np.hypot(seed_e[0], seed_n[0]))
    axm.annotate(f"blind seed\n{seed_km:.0f} km away",
                 xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8,
                 color=ts.INK_SOFT, va="bottom", ha="left")
    axm.set_xlabel("East [km]")
    axm.set_ylabel("North [km]")
    axm.set_aspect("equal")
    axm.grid(alpha=0.3)
    axm.legend(fontsize=8, loc="upper right")
    axm.set_title("Convergence of the fix (converged region)", fontsize=10)
    cb = fig.colorbar(sc, ax=axm, fraction=0.046, pad=0.03)
    cb.set_label("listening time [min]", fontsize=8)

    # ── right: convergence ───────────────────────────────────────────────────
    axc.plot(elapsed / 60.0, err, "-", color=C[1], lw=1.6,
             label="absolute error")
    axc.plot(elapsed / 60.0, sig, "-", color=C[0], lw=1.6,
             label=r"$1\sigma$ major axis")
    axc.axhline(err[-1], color=ts.INK_MUTED, lw=0.8)
    axc.annotate(f"final {err[-1]:.2f} km", xy=(elapsed[-1] / 60.0, err[-1]),
                 xytext=(-4, 6), textcoords="offset points",
                 va="bottom", ha="right", fontsize=8, color=ts.INK_SOFT)
    axc.set_yscale("log")
    axc.set_xlabel("listening time [min]")
    axc.set_ylabel("[km]")
    axc.grid(alpha=0.3, which="both")
    axc.legend(fontsize=8)
    axc.set_title("Error against listening time", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
        print(f"  wrote {out_base}.{ext}")
    plt.close(fig)

    # a small machine-readable trace beside the figure
    np.savez_compressed(f"{out_base}.npz", elapsed_s=elapsed, error_km=err,
                        sigma_major_km=sig, east_km=trail_e, north_km=trail_n,
                        seed_lat=x0[0], seed_lon=x0[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", default=None,
                    help="output basename (default <session>/pnt_convergence)")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--min-bursts", type=int, default=15)
    own, rest = ap.parse_known_args(argv)
    args = pnt_parse_args(rest)

    session = args.session_dir.rstrip("/")
    if args.ephemeris == "ira":
        path = os.path.join(session, "broadcast_ephemeris.json")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found — run scripts/decode_ephemeris.py first")
        src = IraSource(path)
        print(f"[eph] broadcast: {len(src.satellites)} satellites decoded from the air, "
              f"clock corrected by {src.clock_offset_s:+.3f} s")
    else:
        src = TleSource(session)
        print(f"[eph] SGP4 baseline: {len(src.satellites)} satellites")

    clock = src.clock_offset_s if args.clock_offset is None else args.clock_offset
    data = load_peaks(session, args.subdir, clock, args.min_el, args.max_peak_rank)
    print(f"[data] {data[0].size} DOA peaks (rank <= {args.max_peak_rank})")

    sol, ids, x0 = blind_solve(src, data, args, args.mode)
    print(f"[fix] converged {sol.error_km(*args.truth)[0]:.2f} km from truth")

    sols, _counts, elapsed = cumulative_fixes(
        src, data, ids, x0, args, own.steps, own.min_bursts)

    out_base = own.out or os.path.join(session, "pnt_convergence")
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    make_figure(sols, elapsed, x0, tuple(args.truth), out_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
