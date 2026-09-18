#!/usr/bin/env python3
"""
doa_error_vs_elev.py — DOA error binned against true satellite elevation.

eval_doa_accuracy.py reports pooled azimuth/elevation error statistics; it
discards the true elevation of each burst, which is the abscissa needed to show
*where* in a pass the accuracy is lost.  This script keeps it.

Two mechanisms degrade accuracy at low elevation and they are not separable from
a pooled median: the link budget worsens with slant range (measured on its own by
link_budget_snr.py) and a planar UCA loses elevation observability as the
manifold derivative falls off with sin(el).  Binning the error against elevation
is what makes the combined effect visible.

Usage:
    python3 scripts/doa_error_vs_elev.py session_dir/ --out figdir --cache c.json
    python3 scripts/doa_error_vs_elev.py --replot --cache c.json --out figdir
"""

from __future__ import annotations

import argparse
import json
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

from scripts.eval_doa_accuracy import load_peaks                  # noqa: E402
from scripts.fit_array_cal import _interp_track, build_sat_tracks  # noqa: E402
from scripts.iridium_groundtruth import (                          # noqa: E402
    OBSERVER_ALT,
    OBSERVER_LAT,
    OBSERVER_LON,
    load_session_window,
)
from shared.iridium_tle import use_session_tle                     # noqa: E402

# Same edges as link_budget_snr.py: the two figures sit in the same section and
# have to be read against each other bin for bin.
ELEV_BINS = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30),
             (30, 40), (40, 50), (50, 60), (60, 75), (75, 90)]

# Configurations overlaid on each panel: (subdir, label).
CONFIGS = [
    ("doa_multi_music", "single burst"),
    ("doa_multi_music_covB16", "multi-burst, $B=16$"),
]

EL_MIN_GATE_DEG = 10.0   # el_min_gate_deg in doa_config.toml


# ─────────────────────────────────────────────────────────────────────────────
def collect(jsonl_path: str, tracks: list[dict], *,
            dopp_tol: float, lo_offset: float) -> list[dict]:
    """Per-peak records keeping the true elevation, not just the error.

    Mirrors the Doppler association of eval_doa_accuracy.evaluate(): a peak is
    kept only when exactly one predicted track matches its CFO, so an ambiguous
    burst contributes to neither configuration.
    """
    peaks = load_peaks(jsonl_path)
    rows: list[dict] = []
    for pk in peaks:
        cfo = pk["cfo_hz"] - lo_offset
        matches = []
        for trk in tracks:
            c = _interp_track(trk, pk["t_rel"])
            if c is None:
                continue
            dop, az, el = c
            if abs(cfo - dop) < dopp_tol:
                matches.append((trk["name"], az, el))
        if len(matches) != 1:
            continue
        name, az_t, el_t = matches[0]
        rows.append({
            "sat": name,
            "el_true": el_t,
            "az_true": az_t,
            "d_az": (pk["az"] - az_t + 180.0) % 360.0 - 180.0,
            "d_el": pk["el"] - el_t,
            "papr_db": pk["papr_db"],
        })
    return rows


def estimate_lo(jsonl_path: str, tracks: list[dict]) -> float:
    """Median signed Doppler residual under a loose gate (as eval_doa_accuracy)."""
    resid = []
    for pk in load_peaks(jsonl_path):
        cands = [_interp_track(trk, pk["t_rel"]) for trk in tracks]
        diffs = [pk["cfo_hz"] - c[0] for c in cands if c is not None]
        if diffs:
            d = min(diffs, key=abs)
            if abs(d) < 5_000.0:
                resid.append(d)
    return float(np.median(resid)) if len(resid) >= 10 else 0.0


def bin_stats(rows: list[dict], key: str) -> list[dict]:
    """Median and 10/90 percentiles of |error| per elevation bin."""
    el = np.array([r["el_true"] for r in rows])
    err = np.abs(np.array([r[key] for r in rows]))
    out = []
    for lo, hi in ELEV_BINS:
        m = (el >= lo) & (el < hi)
        if m.sum() == 0:
            out.append(dict(el_bin=[lo, hi], n=0))
            continue
        out.append(dict(el_bin=[lo, hi], n=int(m.sum()),
                        p10=round(float(np.percentile(err[m], 10)), 2),
                        p50=round(float(np.median(err[m])), 2),
                        p90=round(float(np.percentile(err[m], 90)), 2)))
    return out


def summarise(per_config: dict) -> dict:
    """Per-bin statistics plus the pooled medians, for cross-checking the table."""
    summary = {}
    for subdir, rows in per_config.items():
        az = np.abs(np.array([r["d_az"] for r in rows]))
        el = np.abs(np.array([r["d_el"] for r in rows]))
        gated = [r for r in rows if r["el_true"] >= EL_MIN_GATE_DEG]
        az_g = np.abs(np.array([r["d_az"] for r in gated]))
        el_g = np.abs(np.array([r["d_el"] for r in gated]))
        summary[subdir] = dict(
            n=len(rows),
            n_above_gate=len(gated),
            az_medae_all=round(float(np.median(az)), 2) if len(az) else None,
            el_medae_all=round(float(np.median(el)), 2) if len(el) else None,
            az_medae_gated=round(float(np.median(az_g)), 2) if len(az_g) else None,
            el_medae_gated=round(float(np.median(el_g)), 2) if len(el_g) else None,
            az_bins=bin_stats(rows, "d_az"),
            el_bins=bin_stats(rows, "d_el"),
        )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
def plot(per_config: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        sys.path.insert(0, _HERE)
        import thesis_style
        thesis_style.apply(plt)
        c_single, c_multi = thesis_style.SERIES[1], thesis_style.SERIES[0]
        ink_soft, grid = thesis_style.INK_SOFT, thesis_style.GRID
    except Exception:            # style module is optional
        c_single, c_multi = "#eb6834", "#2a78d6"
        ink_soft, grid = "#52514e", "#e3e2de"

    os.makedirs(out_dir, exist_ok=True)
    colours = {"doa_multi_music": c_single, "doa_multi_music_covB16": c_multi}
    centres = np.array([(lo + hi) / 2 for lo, hi in ELEV_BINS])

    # Only bins that actually hold data; this session tops out below 60 deg.
    el_max = max(r["el_true"] for rows in per_config.values() for r in rows)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharex=True, sharey=True)

    for ax, key, name in ((axes[0], "d_az", "azimuth"),
                          (axes[1], "d_el", "elevation")):
        # Everything left of the gate is discarded by the pipeline; shade it so
        # the reader does not read the first bin as a result.
        ax.axvspan(0, EL_MIN_GATE_DEG, color=grid, alpha=0.7, lw=0, zorder=0)

        for subdir, label in CONFIGS:
            rows = per_config.get(subdir)
            if not rows:
                continue
            st = bin_stats(rows, key)
            ok = [i for i, b in enumerate(st) if b["n"] >= 5]
            if not ok:
                continue
            x = centres[ok]
            p50 = np.array([st[i]["p50"] for i in ok])
            col = colours[subdir]
            # 10-90 band for the reference configuration only, or the panel
            # becomes unreadable.
            if subdir == "doa_multi_music_covB16":
                p10 = np.array([st[i]["p10"] for i in ok])
                p90 = np.array([st[i]["p90"] for i in ok])
                ax.fill_between(x, p10, p90, color=col, alpha=0.15, lw=0,
                                zorder=1, label="10th–90th pct, $B=16$")
            ax.plot(x, p50, "o-", color=col, ms=3.5, lw=1.3, zorder=3,
                    label=f"median, {label}")

        ax.axvline(EL_MIN_GATE_DEG, color=ink_soft, lw=0.9, ls=":", zorder=2)
        ax.set_xlabel("true satellite elevation [deg]")
        ax.set_title(f"{name} error", fontsize=9.5)
        ax.set_xlim(0, min(90, 5 * np.ceil(el_max / 5) + 5))
        ax.set_yscale("log")
        ax.set_ylim(0.4, 40)
        ax.set_yticks([0.5, 1, 2, 5, 10, 20])
        ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
        ax.tick_params(axis="y", which="minor", left=False)

    axes[0].set_ylabel("absolute error [deg]")
    axes[0].annotate("gate", xy=(EL_MIN_GATE_DEG - 1.2, 24), fontsize=7,
                     ha="right", va="center", color=ink_soft, rotation=90)
    axes[1].legend(fontsize=7, loc="upper right", frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"doa_error_vs_elev.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"  wrote doa_error_vs_elev.png/.pdf in {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("session_dir", nargs="?", default="")
    p.add_argument("--out", default=".", help="figure output directory")
    p.add_argument("--cache", default="", help="JSON cache of the per-peak records")
    p.add_argument("--replot", action="store_true",
                   help="redraw from the cache, without reading the session")
    p.add_argument("--dopp-tol", type=float, default=2_000.0,
                   help="Doppler tolerance [Hz] for unique satellite assignment")
    p.add_argument("--el-min", type=float, default=5.0,
                   help="min elevation [deg] for pass prediction (not the gate)")
    p.add_argument("--lo-offset", type=float, default=None,
                   help="receiver LO offset [Hz] (default: auto-estimate)")
    p.set_defaults(lat=OBSERVER_LAT, lon=OBSERVER_LON, alt=OBSERVER_ALT)
    args = p.parse_args(argv)

    if args.replot:
        if not args.cache or not os.path.isfile(args.cache):
            p.error("--replot needs an existing --cache")
        per_config = json.load(open(args.cache, encoding="utf-8"))["per_config"]
    else:
        if not args.session_dir:
            p.error("session directory required without --replot")
        session_dir = os.path.abspath(args.session_dir.rstrip("/"))
        use_session_tle(session_dir)      # freeze ground-truth elements
        t0, t1, _meta = load_session_window(session_dir)
        tracks = build_sat_tracks(t0, t1 + timedelta(seconds=30), args)

        per_config = {}
        lo = args.lo_offset
        for subdir, label in CONFIGS:
            jsonl = os.path.join(session_dir, subdir, "doa_multi.jsonl")
            if not os.path.isfile(jsonl):
                print(f"  skip {subdir}: no doa_multi.jsonl")
                continue
            if lo is None:
                # One LO estimate for the whole session; the offset is a receiver
                # property, not a per-configuration one.
                lo = estimate_lo(jsonl, tracks)
                print(f"LO offset: {lo:+.0f} Hz")
            rows = collect(jsonl, tracks, dopp_tol=args.dopp_tol, lo_offset=lo)
            print(f"  {subdir}: {len(rows)} assigned peaks")
            per_config[subdir] = rows

        if args.cache:
            with open(args.cache, "w", encoding="utf-8") as f:
                json.dump({"session": session_dir, "lo_offset_hz": lo,
                           "per_config": per_config}, f)
            print(f"  wrote cache {args.cache}")

    summary = summarise(per_config)
    print(json.dumps(summary, indent=2))
    if args.cache:
        with open(args.cache + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    plot(per_config, args.out)
    return summary


if __name__ == "__main__":
    main()
