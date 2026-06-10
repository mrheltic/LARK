#!/usr/bin/env python3
"""
plot_track_vs_tle.py — Measured DOA trajectory vs real (SGP4/TLE) satellite track.

For each satellite with enough DOA peaks assigned by Doppler, draws:
  * a polar sky plot (N up, E right, radius = 90 − el): the TLE trajectory
    rise→set as a time-coloured line, the DOA peaks as scatter with the same
    time colormap — temporal correspondence is readable at a glance;
  * az(t), el(t) and Doppler(t) panels, measured vs predicted.

Peak→satellite assignment is the same unique-Doppler rule used by
eval_doa_accuracy.py (±dopp_tol after receiver LO offset removal).

Usage:
    python3 scripts/plot_track_vs_tle.py session_dir/
    python3 scripts/plot_track_vs_tle.py session_dir/ --subdir doa_multi_capon
    python3 scripts/plot_track_vs_tle.py session_dir/ --sat "IRIDIUM 125" --show
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.eval_doa_accuracy import estimate_lo, load_peaks  # noqa: E402
from scripts.fit_array_cal import _interp_track, build_sat_tracks  # noqa: E402
from scripts.iridium_groundtruth import (  # noqa: E402
    OBSERVER_ALT,
    OBSERVER_LAT,
    OBSERVER_LON,
    load_session_window,
)

# Dark theme (matches offline_viz.py / run_doa.py)
BG = "#1a1d27"
BG2 = "#21253a"
C_BDR = "#3b4263"
C_MUT = "#8891b0"
C_TEXT = "#d8dae8"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DOA trajectory vs TLE ground truth")
    p.add_argument("session_dir")
    p.add_argument("--subdir", default="doa_multi_music",
                   help="Subdir with doa_multi.jsonl (default: doa_multi_music)")
    p.add_argument("--sat", default="",
                   help='Only this satellite (e.g. "IRIDIUM 125"); default: all')
    p.add_argument("--min-peaks", type=int, default=30,
                   help="Min assigned peaks to plot a satellite")
    p.add_argument("--dopp-tol", type=float, default=2_000.0,
                   help="Doppler tolerance [Hz] for unique satellite assignment")
    p.add_argument("--lo-offset", type=float, default=None,
                   help="Receiver LO offset [Hz] (default: auto-estimate)")
    p.add_argument("--out-dir", default="",
                   help="Output dir for PNGs (default: <session>/plots)")
    p.add_argument("--show", action="store_true", help="Open interactive windows")
    # build_sat_tracks() expects observer location + el_min on the namespace.
    p.set_defaults(lat=OBSERVER_LAT, lon=OBSERVER_LON, alt=OBSERVER_ALT, el_min=5.0)
    return p.parse_args(argv)


def wrap180(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def assign_peaks(peaks: list[dict], tracks: list[dict], *,
                 dopp_tol: float, lo_offset: float) -> dict[int, list[dict]]:
    """{track_index: [peak + az/el/dop predicted]} — unique Doppler match only.

    Keyed by track index, not satellite name: build_sat_tracks returns one
    entry per PASS, and a satellite can pass more than once in a session.
    """
    by_track: dict[int, list[dict]] = {}
    for pk in peaks:
        cfo = pk["cfo_hz"] - lo_offset
        matches = []
        for ti, trk in enumerate(tracks):
            c = _interp_track(trk, pk["t_rel"])
            if c is None:
                continue
            dop, az, el = c
            if abs(cfo - dop) < dopp_tol:
                matches.append((ti, dop, az, el))
        if len(matches) != 1:
            continue
        ti, dop, az, el = matches[0]
        by_track.setdefault(ti, []).append(
            {**pk, "az_pred": az, "el_pred": el, "dop_pred": dop})
    return by_track


def _style_axes(ax):
    ax.set_facecolor(BG2)
    for s in ax.spines.values():
        s.set_color(C_BDR)
    ax.tick_params(colors=C_MUT, labelsize=8)
    ax.xaxis.label.set_color(C_MUT)
    ax.yaxis.label.set_color(C_MUT)
    ax.grid(color=C_BDR, alpha=0.5, lw=0.5)


def plot_satellite(sat: str, pts: list[dict], trk: dict, *,
                   lo_offset: float, subdir: str, out_dir: str, show: bool) -> str:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    t_m = np.array([p["t_rel"] for p in pts]) / 60.0      # min since session t0
    az_m = np.array([p["az"] for p in pts])
    el_m = np.array([p["el"] for p in pts])
    cfo_m = np.array([p["cfo_hz"] for p in pts]) - lo_offset
    az_p = np.array([p["az_pred"] for p in pts])
    el_p = np.array([p["el_pred"] for p in pts])

    # TLE track restricted to the observed window (±60 s margin)
    sel = (trk["t_rel"] >= pts[0]["t_rel"] - 60.0) & \
          (trk["t_rel"] <= pts[-1]["t_rel"] + 60.0)
    tt = np.asarray(trk["t_rel"])[sel] / 60.0
    taz = np.asarray(trk["az_deg"])[sel]
    tel = np.asarray(trk["el_deg"])[sel]
    tdop = np.asarray(trk["doppler_hz"])[sel]

    az_res = wrap180(az_m - az_p)
    el_res = el_m - el_p
    az_mad = float(np.median(np.abs(az_res - np.median(az_res))))
    el_mad = float(np.median(np.abs(el_res - np.median(el_res))))

    cmap = plt.get_cmap("plasma")
    t_lo, t_hi = float(tt[0]), float(tt[-1])
    norm = plt.Normalize(t_lo, t_hi)

    fig = plt.figure(figsize=(13, 7), facecolor=BG)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0],
                          hspace=0.45, wspace=0.25,
                          left=0.05, right=0.97, top=0.88, bottom=0.08)

    # ── Sky plot ────────────────────────────────────────────────────────────
    ax_sky = fig.add_subplot(gs[:, 0], polar=True)
    ax_sky.set_facecolor(BG2)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)          # compass: E to the right
    ax_sky.set_rlim(0, 85)
    ax_sky.set_rgrids([30, 60], labels=["60°", "30°"], color=C_MUT, fontsize=8)
    ax_sky.set_thetagrids([0, 90, 180, 270], labels=["N", "E", "S", "W"],
                          color=C_TEXT, fontsize=10)
    ax_sky.grid(color=C_BDR, alpha=0.6, lw=0.6)
    ax_sky.spines["polar"].set_color(C_BDR)

    # TLE trajectory as time-coloured line
    th = np.deg2rad(taz)
    r = 90.0 - tel
    segs = np.stack([np.column_stack([th[:-1], r[:-1]]),
                     np.column_stack([th[1:], r[1:]])], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.2, alpha=0.9)
    lc.set_array(tt[:-1])
    ax_sky.add_collection(lc)
    ax_sky.plot(th[0], r[0], "^", color=cmap(norm(t_lo)), ms=9,
                mec=C_TEXT, mew=0.6, zorder=5)
    ax_sky.plot(th[-1], r[-1], "v", color=cmap(norm(t_hi)), ms=9,
                mec=C_TEXT, mew=0.6, zorder=5)

    # DOA peaks, same colormap
    sc = ax_sky.scatter(np.deg2rad(az_m), 90.0 - el_m, c=t_m, cmap=cmap,
                        norm=norm, s=22, edgecolors=BG, linewidths=0.4,
                        zorder=6)
    cb = fig.colorbar(sc, ax=ax_sky, pad=0.1, fraction=0.04)
    cb.set_label("min since session start", color=C_MUT, fontsize=8)
    cb.ax.tick_params(colors=C_MUT, labelsize=8)
    cb.outline.set_edgecolor(C_BDR)

    # ── Time panels ─────────────────────────────────────────────────────────
    # Azimuth on the unwrapped predicted branch (no 0/360 jumps)
    taz_u = np.rad2deg(np.unwrap(np.deg2rad(taz)))
    az_p_u = np.interp(t_m, tt, taz_u)
    az_m_u = az_p_u + az_res                # measured on the same branch

    panels = [
        ("Azimuth [°]", (tt, taz_u), (t_m, az_m_u),
         (float(taz_u.min()) - 25.0, float(taz_u.max()) + 25.0)),
        ("Elevation [°]", (tt, tel), (t_m, el_m),
         (0.0, float(tel.max()) + 15.0)),
        ("Doppler [kHz]", (tt, tdop / 1e3), (t_m, cfo_m / 1e3), None),
    ]
    for i, (label, (tx, py), (mx, my), ylim) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 1])
        _style_axes(ax)
        ax.plot(tx, py, color=C_MUT, lw=1.6, label="TLE (SGP4)")
        ax.scatter(mx, my, c=mx, cmap=cmap, norm=norm, s=14,
                   edgecolors=BG, linewidths=0.3, label="DOA", zorder=5)
        ax.set_ylabel(label, fontsize=9)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if i == 2:
            ax.set_xlabel("min since session start", fontsize=9)
        if i == 0:
            ax.legend(loc="best", fontsize=8, facecolor=BG2,
                      edgecolor=C_BDR, labelcolor=C_TEXT)

    fig.suptitle(
        f"{sat} — DOA trajectory vs TLE   ({subdir}, {len(pts)} burst)\n"
        f"residuals: az {np.median(az_res):+.1f}° (MAD {az_mad:.1f}°)   "
        f"el {np.median(el_res):+.1f}° (MAD {el_mad:.1f}°)   "
        f"LO {lo_offset:+.0f} Hz",
        color=C_TEXT, fontsize=11,
    )

    out_path = os.path.join(
        out_dir, f"track_{sat.replace(' ', '_')}_{subdir}.png")
    fig.savefig(out_path, dpi=140, facecolor=BG)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session_dir = os.path.abspath(args.session_dir.rstrip("/"))
    jsonl = os.path.join(session_dir, args.subdir, "doa_multi.jsonl")
    if not os.path.isfile(jsonl):
        sys.exit(f"No JSONL at {jsonl}")
    out_dir = args.out_dir or os.path.join(session_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    t0, t1, _meta = load_session_window(session_dir)
    peaks = load_peaks(jsonl)
    print(f"{len(peaks)} DOA peaks from {jsonl}")

    tracks = build_sat_tracks(t0, t1 + timedelta(seconds=30), args)
    lo = float(args.lo_offset) if args.lo_offset is not None \
        else estimate_lo(peaks, tracks)
    print(f"LO offset: {lo:+.0f} Hz")

    by_track = assign_peaks(peaks, tracks, dopp_tol=args.dopp_tol, lo_offset=lo)

    # A satellite can pass more than once: number passes per name.
    n_passes = {}
    for ti in sorted(by_track):
        name = tracks[ti]["name"]
        n_passes[name] = n_passes.get(name, 0) + 1
    seen: dict[str, int] = {}

    plotted = 0
    for ti, pts in sorted(by_track.items(), key=lambda kv: -len(kv[1])):
        sat = tracks[ti]["name"]
        if args.sat and sat != args.sat:
            continue
        if len(pts) < args.min_peaks:
            continue
        pts.sort(key=lambda p: p["t_rel"])
        seen[sat] = seen.get(sat, 0) + 1
        name = sat if n_passes[sat] == 1 else f"{sat} pass{seen[sat]}"
        path = plot_satellite(
            name, pts, tracks[ti], lo_offset=lo,
            subdir=args.subdir, out_dir=out_dir, show=args.show)
        print(f"  {name:24s} {len(pts):4d} burst → {path}")
        plotted += 1

    if plotted == 0:
        print("No satellite reached --min-peaks; nothing plotted.")


if __name__ == "__main__":
    main()
