#!/usr/bin/env python3
"""
smooth_tracks.py — Kalman/RTS smoothing of clustered DOA tracks (+ TLE eval).

Reads a doa_multi.jsonl produced by reprocess_session.py (peaks already carry
``track_ids`` from core/track_clusterer.py), smooths every sufficiently long
track with the unit-vector Kalman filter + RTS smoother from
core/track_filter.py, and writes ``tracks_smoothed.json`` next to the JSONL.

If TLE data is available it also evaluates raw vs smoothed accuracy against
the SGP4-predicted directions (same unique-Doppler assignment as
eval_doa_accuracy.py), so the improvement is quantified on real data.

Usage:
    python3 scripts/smooth_tracks.py session_dir/ --subdir doa_multi_music
    python3 scripts/smooth_tracks.py session_dir/ --no-eval   # smoothing only
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

from core.track_filter import (  # noqa: E402
    DEFAULT_SIGMA_ACC,
    DEFAULT_SIGMA_AZ_DEG,
    DEFAULT_SIGMA_EL_DEG,
    kalman_smooth_track,
    kalman_smooth_track_robust,
)
from scripts.eval_doa_accuracy import estimate_lo, load_peaks  # noqa: E402
from scripts.fit_array_cal import _interp_track, build_sat_tracks  # noqa: E402
from scripts.iridium_groundtruth import (  # noqa: E402
    OBSERVER_ALT,
    OBSERVER_LAT,
    OBSERVER_LON,
    load_session_window,
)
from shared.iridium_tle import use_session_tle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kalman/RTS smoothing of DOA tracks")
    p.add_argument("session_dir")
    p.add_argument("--subdir", default="doa_multi_music",
                   help="Subdir with doa_multi.jsonl (default: doa_multi_music)")
    p.add_argument("--min-peaks", type=int, default=30,
                   help="Minimum peaks for a track to be smoothed (default: 30)")
    p.add_argument("--sigma-az", type=float, default=DEFAULT_SIGMA_AZ_DEG,
                   help="Per-burst azimuth 1-sigma [deg]")
    p.add_argument("--sigma-el", type=float, default=DEFAULT_SIGMA_EL_DEG,
                   help="Per-burst elevation 1-sigma [deg]")
    p.add_argument("--sigma-acc", type=float, default=DEFAULT_SIGMA_ACC,
                   help="Process noise: angular acceleration 1-sigma [rad/s^2]")
    p.add_argument("--no-robust", action="store_true",
                   help="Plain KF/RTS: no outlier rejection, no SNR weights")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip the TLE accuracy comparison")
    p.add_argument("--dopp-tol", type=float, default=2_000.0,
                   help="Doppler tolerance [Hz] for unique satellite assignment")
    p.add_argument("--el-min", type=float, default=5.0,
                   help="Min satellite elevation [deg] for pass prediction")
    p.add_argument("--lo-offset", type=float, default=None,
                   help="Receiver LO offset [Hz] (default: auto-estimate)")
    p.add_argument("--json-out", default="", help="Write eval summary JSON here")
    # build_sat_tracks() expects the observer location on the namespace.
    p.set_defaults(lat=OBSERVER_LAT, lon=OBSERVER_LON, alt=OBSERVER_ALT)
    return p.parse_args(argv)


def collect_tracks(jsonl_path: str, min_peaks: int) -> list[dict]:
    """
    Group JSONL peaks by clusterer track id.

    Returns one dict per track (id, t, az, el, cfo, snr arrays sorted by
    time, duplicate timestamps deduplicated keeping the strongest peak).
    """
    by_id: dict[int, list[tuple[float, float, float, float, float, float]]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tids = row.get("track_ids") or []
            cfo_pp = row.get("cfo_per_peak") or []
            snr_pp = row.get("snr_per_peak") or []
            for pi, pk in enumerate(row["peaks"]):
                tid = int(tids[pi]) if pi < len(tids) else -1
                if tid < 0:
                    continue
                cfo = float(cfo_pp[pi]) if pi < len(cfo_pp) else float(row["cfo_hz"])
                snr = float(snr_pp[pi]) if pi < len(snr_pp) else float(row["snr_db"])
                by_id.setdefault(tid, []).append(
                    (float(row["t"]), float(pk[0]), float(pk[1]),
                     cfo, float(pk[3]), snr)
                )

    tracks: list[dict] = []
    for tid, pts in sorted(by_id.items()):
        if len(pts) < min_peaks:
            continue
        pts.sort(key=lambda x: (x[0], -x[4]))     # by time, strongest first
        dedup = [pts[0]]
        for pt in pts[1:]:
            if pt[0] > dedup[-1][0]:
                dedup.append(pt)
        arr = np.asarray(dedup)
        tracks.append({
            "id": tid,
            "t": arr[:, 0],
            "az": arr[:, 1],
            "el": arr[:, 2],
            "cfo": arr[:, 3],
            "snr_db": arr[:, 5],
        })
    return tracks


def _stats(err: np.ndarray) -> dict:
    return {
        "bias_deg": round(float(np.median(err)), 1),
        "mad_deg": round(float(np.median(np.abs(err - np.median(err)))), 1),
        "rms_deg": round(float(np.sqrt(np.mean(err ** 2))), 1),
    }


def evaluate_vs_tle(tracks: list[dict], sat_tracks: list[dict], *,
                    dopp_tol: float, lo_offset: float) -> dict:
    """Raw vs smoothed error stats at the same (uniquely assigned) points."""
    az_raw, el_raw, az_sm, el_sm = [], [], [], []
    per_sat: dict[str, int] = {}
    for trk in tracks:
        for i in range(len(trk["t"])):
            cfo = trk["cfo"][i] - lo_offset
            matches = []
            for st in sat_tracks:
                c = _interp_track(st, trk["t"][i])
                if c is None:
                    continue
                dop, az, el = c
                if abs(cfo - dop) < dopp_tol:
                    matches.append((st["name"], az, el))
            if len(matches) != 1:
                continue
            name, az_t, el_t = matches[0]
            per_sat[name] = per_sat.get(name, 0) + 1
            az_raw.append((trk["az"][i] - az_t + 180.0) % 360.0 - 180.0)
            el_raw.append(trk["el"][i] - el_t)
            az_sm.append((trk["az_smooth"][i] - az_t + 180.0) % 360.0 - 180.0)
            el_sm.append(trk["el_smooth"][i] - el_t)

    summary: dict = {
        "n_assigned": len(az_raw),
        "lo_offset_hz": round(lo_offset, 0),
        "n_sats": len(per_sat),
        "per_sat": per_sat,
    }
    if az_raw:
        summary["raw"] = {"az": _stats(np.asarray(az_raw)),
                          "el": _stats(np.asarray(el_raw))}
        summary["smoothed"] = {"az": _stats(np.asarray(az_sm)),
                               "el": _stats(np.asarray(el_sm))}
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    session_dir = os.path.abspath(args.session_dir.rstrip("/"))
    subdir = os.path.join(session_dir, args.subdir)
    jsonl_path = os.path.join(subdir, "doa_multi.jsonl")
    if not os.path.isfile(jsonl_path):
        sys.exit(f"No JSONL at {jsonl_path}")

    tracks = collect_tracks(jsonl_path, args.min_peaks)
    print(f"{len(tracks)} tracks with >= {args.min_peaks} peaks")

    n_out = 0
    for trk in tracks:
        kw = dict(sigma_az_deg=args.sigma_az, sigma_el_deg=args.sigma_el,
                  sigma_acc=args.sigma_acc)
        if args.no_robust:
            sm = kalman_smooth_track(trk["t"], trk["az"], trk["el"],
                                     gate_chi2=None, **kw)
        else:
            # No SNR weighting: validated on the reference session, the DOA
            # error is manifold-dominated, not noise-dominated, so weighting
            # by SNR makes the stats slightly WORSE (az MAD 2.5° → 2.9°).
            sm = kalman_smooth_track_robust(trk["t"], trk["az"], trk["el"],
                                            **kw)
        trk["az_smooth"] = sm.az_deg
        trk["el_smooth"] = sm.el_deg
        trk["sigma_az"] = sm.sigma_az_deg
        trk["sigma_el"] = sm.sigma_el_deg
        trk["outlier"] = sm.outlier if sm.outlier is not None \
            else np.zeros(len(trk["t"]), bool)
        n_out += int(trk["outlier"].sum())
    if not args.no_robust:
        n_tot = sum(len(trk["t"]) for trk in tracks)
        print(f"Rejected {n_out}/{n_tot} outlier points")

    out_path = os.path.join(subdir, "tracks_smoothed.json")
    payload = {
        "meta": {
            "source": os.path.basename(jsonl_path),
            "min_peaks": args.min_peaks,
            "sigma_az_deg": args.sigma_az,
            "sigma_el_deg": args.sigma_el,
            "sigma_acc": args.sigma_acc,
            "robust": not args.no_robust,
        },
        "tracks": [
            {
                "id": trk["id"],
                "n_peaks": len(trk["t"]),
                "t": [round(v, 3) for v in trk["t"]],
                "az_raw": [round(v, 2) for v in trk["az"]],
                "el_raw": [round(v, 2) for v in trk["el"]],
                "az_smooth": [round(v, 2) for v in trk["az_smooth"]],
                "el_smooth": [round(v, 2) for v in trk["el_smooth"]],
                "sigma_az": [round(v, 2) for v in trk["sigma_az"]],
                "sigma_el": [round(v, 2) for v in trk["sigma_el"]],
                "cfo_hz": [round(v, 0) for v in trk["cfo"]],
                "outlier": [bool(v) for v in trk["outlier"]],
            }
            for trk in tracks
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Wrote {out_path}")

    if args.no_eval:
        return {}

    use_session_tle(session_dir)   # freeze ground-truth elements per session
    t0, t1, _meta = load_session_window(session_dir)
    sat_tracks = build_sat_tracks(t0, t1 + timedelta(seconds=30), args)
    lo = float(args.lo_offset) if args.lo_offset is not None \
        else estimate_lo(load_peaks(jsonl_path), sat_tracks)
    print(f"LO offset: {lo:+.0f} Hz")

    summary = evaluate_vs_tle(tracks, sat_tracks,
                              dopp_tol=args.dopp_tol, lo_offset=lo)
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()
