#!/usr/bin/env python3
"""
eval_doa_accuracy.py — Per-burst DOA accuracy against TLE ground truth.

Reads a doa_multi.jsonl produced by reprocess_session.py, assigns every DOA
peak to a satellite by Doppler (unique match within --dopp-tol after removing
the receiver LO offset), and reports azimuth/elevation error statistics
against the SGP4-predicted direction at the burst time.

Unlike groundtruth_matches.json (track-level medians), this gives a per-burst
error distribution — the right metric for comparing pipeline parameters.

Usage:
    python3 scripts/eval_doa_accuracy.py session_dir/ --subdir doa_multi_music
    python3 scripts/eval_doa_accuracy.py session_dir/ --jsonl path/to/doa_multi.jsonl
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

from scripts.fit_array_cal import _interp_track, build_sat_tracks  # noqa: E402
from scripts.iridium_groundtruth import (  # noqa: E402
    OBSERVER_ALT,
    OBSERVER_LAT,
    OBSERVER_LON,
    load_session_window,
)
from shared.iridium_tle import use_session_tle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-burst DOA accuracy vs TLE")
    p.add_argument("session_dir")
    p.add_argument("--subdir", default="doa_multi_music",
                   help="Subdir with doa_multi.jsonl (default: doa_multi_music)")
    p.add_argument("--jsonl", default="", help="Explicit JSONL path (overrides --subdir)")
    p.add_argument("--dopp-tol", type=float, default=2_000.0,
                   help="Doppler tolerance [Hz] for unique satellite assignment")
    p.add_argument("--el-min", type=float, default=5.0,
                   help="Min satellite elevation [deg] for pass prediction")
    p.add_argument("--lo-offset", type=float, default=None,
                   help="Receiver LO offset [Hz] (default: auto-estimate)")
    p.add_argument("--json-out", default="", help="Write summary JSON here")
    # build_sat_tracks() expects the observer location on the namespace.
    p.set_defaults(lat=OBSERVER_LAT, lon=OBSERVER_LON, alt=OBSERVER_ALT)
    return p.parse_args(argv)


def load_peaks(jsonl_path: str) -> list[dict]:
    """One entry per DOA peak: t, az, el, papr, cfo."""
    out: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cfo_pp = row.get("cfo_per_peak") or []
            for pi, pk in enumerate(row["peaks"]):
                out.append({
                    "t_rel": float(row["t"]),
                    "az": float(pk[0]),
                    "el": float(pk[1]),
                    "papr_db": float(pk[3]),
                    "cfo_hz": float(cfo_pp[pi]) if pi < len(cfo_pp) else float(row["cfo_hz"]),
                })
    return out


def estimate_lo(peaks: list[dict], tracks: list[dict]) -> float:
    """Median signed residual to the nearest predicted Doppler (loose gate)."""
    resid = []
    for pk in peaks:
        cands = [_interp_track(trk, pk["t_rel"]) for trk in tracks]
        diffs = [pk["cfo_hz"] - c[0] for c in cands if c is not None]
        if diffs:
            d = min(diffs, key=abs)
            if abs(d) < 5_000.0:
                resid.append(d)
    return float(np.median(resid)) if len(resid) >= 10 else 0.0


def evaluate(peaks: list[dict], tracks: list[dict], *,
             dopp_tol: float, lo_offset: float) -> dict:
    az_err: list[float] = []
    el_err: list[float] = []
    per_sat: dict[str, int] = {}
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
        d_az = (pk["az"] - az_t + 180.0) % 360.0 - 180.0
        az_err.append(d_az)
        el_err.append(pk["el"] - el_t)
        per_sat[name] = per_sat.get(name, 0) + 1

    az_a, el_a = np.asarray(az_err), np.asarray(el_err)
    summary = {
        "n_peaks": len(peaks),
        "n_assigned": int(len(az_a)),
        "lo_offset_hz": round(lo_offset, 0),
        "n_sats": len(per_sat),
        "per_sat": per_sat,
    }
    if len(az_a):
        # bias = typical signed error; MedAE = typical |error| (robust);
        # MAE/RMSE include the tail — RMSE ≫ MedAE flags heavy outliers.
        summary.update({
            "az_bias_deg": round(float(np.median(az_a)), 1),
            "az_medae_deg": round(float(np.median(np.abs(az_a))), 1),
            "az_mae_deg": round(float(np.mean(np.abs(az_a))), 1),
            "az_mad_deg": round(float(np.median(np.abs(az_a - np.median(az_a)))), 1),
            "az_rms_deg": round(float(np.sqrt(np.mean(az_a ** 2))), 1),
            "el_bias_deg": round(float(np.median(el_a)), 1),
            "el_medae_deg": round(float(np.median(np.abs(el_a))), 1),
            "el_mae_deg": round(float(np.mean(np.abs(el_a))), 1),
            "el_mad_deg": round(float(np.median(np.abs(el_a - np.median(el_a)))), 1),
            "el_rms_deg": round(float(np.sqrt(np.mean(el_a ** 2))), 1),
        })
    return summary


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    session_dir = os.path.abspath(args.session_dir.rstrip("/"))
    jsonl_path = args.jsonl or os.path.join(session_dir, args.subdir, "doa_multi.jsonl")
    if not os.path.isfile(jsonl_path):
        sys.exit(f"No JSONL at {jsonl_path}")

    use_session_tle(session_dir)   # freeze ground-truth elements per session
    t0, t1, _meta = load_session_window(session_dir)
    peaks = load_peaks(jsonl_path)
    print(f"{len(peaks)} DOA peaks from {jsonl_path}")

    tracks = build_sat_tracks(t0, t1 + timedelta(seconds=30), args)
    lo = float(args.lo_offset) if args.lo_offset is not None \
        else estimate_lo(peaks, tracks)
    print(f"LO offset: {lo:+.0f} Hz")

    summary = evaluate(peaks, tracks, dopp_tol=args.dopp_tol, lo_offset=lo)
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()
