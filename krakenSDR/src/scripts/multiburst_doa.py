#!/usr/bin/env python3
"""
multiburst_doa.py — DOA from multi-burst covariance (beyond rank-1).

A single Iridium burst yields a rank-1 matched-filter covariance, for which
MUSIC, Capon and Bartlett share the same argmax.  But consecutive bursts of
the *same satellite* (same clusterer track) carry independent noise — and,
as the satellite moves, decorrelating ground multipath.  Averaging

    R_i = (1/B) Σ_{j ∈ window(i)} ŷ_j · ŷ_jᴴ        (ŷ = y / ‖y‖)

over a sliding window of B bursts makes the noise subspace estimable, so
subspace methods finally differ from plain beamforming.

Requires ``y_per_peak`` in doa_multi.jsonl (sessions reprocessed after
June 2026; re-run apps/doa_iridium/reprocess_session.py for older outputs).

Usage:
    # Sweep window sizes, report az/el accuracy vs TLE per window:
    python3 scripts/multiburst_doa.py <session_dir> --windows 1 2 4 8 16

    # Write a derived result set (same JSONL format → all existing tools work):
    python3 scripts/multiburst_doa.py <session_dir> --window 8 --write
    python3 scripts/eval_doa_accuracy.py  <session_dir> --subdir doa_multi_music_covB8
    python3 scripts/smooth_tracks.py      <session_dir> --subdir doa_multi_music_covB8
    python3 scripts/plot_track_vs_tle.py  <session_dir> --subdir doa_multi_music_covB8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from apps.doa_iridium.run_doa import load_config  # noqa: E402
from apps.doa_iridium.run_doa_offline import _build_uca  # noqa: E402
from core.multiburst import collect_track_peaks, reestimate  # noqa: E402
from scripts.eval_doa_accuracy import estimate_lo, evaluate  # noqa: E402
from scripts.fit_array_cal import build_sat_tracks  # noqa: E402
from scripts.iridium_groundtruth import (  # noqa: E402
    OBSERVER_ALT,
    OBSERVER_LAT,
    OBSERVER_LON,
    load_session_window,
)
from shared.iridium_tle import use_session_tle  # noqa: E402

_DEF_CONFIG = os.path.join(_SRC, "apps", "doa_iridium", "doa_config.toml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-burst covariance DOA")
    p.add_argument("session_dir")
    p.add_argument("--subdir", default="doa_multi_music",
                   help="Subdir with doa_multi.jsonl (default: doa_multi_music)")
    p.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                   help="Window sizes B to sweep (bursts per covariance)")
    p.add_argument("--window", type=int, default=None,
                   help="Single window size (use with --write)")
    p.add_argument("--write", action="store_true",
                   help="Write <subdir>_covB<N>/doa_multi.jsonl with the "
                        "re-estimated peaks (needs --window)")
    p.add_argument("--algo", default="music",
                   choices=["music", "capon", "bartlett"],
                   help="DOA algorithm for the re-estimation (default: music)")
    p.add_argument("--mdl", action="store_true",
                   help="Auto-detect source count via MDL instead of fixing 1")
    p.add_argument("--max-span-s", type=float, default=12.0,
                   help="Max window time span [s] — limits steering smear "
                        "(satellite moves ~0.5 deg/s)")
    p.add_argument("--config", default=_DEF_CONFIG, help="doa_config.toml path")
    p.add_argument("--dopp-tol", type=float, default=2_000.0)
    p.add_argument("--el-min", type=float, default=5.0)
    p.add_argument("--lo-offset", type=float, default=None)
    # build_sat_tracks() expects the observer location on the namespace.
    p.set_defaults(lat=OBSERVER_LAT, lon=OBSERVER_LON, alt=OBSERVER_ALT)
    return p.parse_args(argv)


def load_rows(jsonl_path: str) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sweep(args, by_tid, sat_tracks, lo) -> dict[int, dict]:
    cfg = load_config(args.config)
    uca = _build_uca(cfg)
    results: dict[int, dict] = {}
    for B in args.windows:
        peaks = reestimate(by_tid, window=B, max_span_s=args.max_span_s,
                           uca=uca, algo=args.algo, mdl=args.mdl)
        ev = [{"t_rel": p["t"], "az": p["az"], "el": p["el"],
               "papr_db": p["papr_db"], "cfo_hz": p["cfo_hz"]} for p in peaks]
        s = evaluate(ev, sat_tracks, dopp_tol=args.dopp_tol, lo_offset=lo)
        results[B] = s
        print(f"B={B:3d}  n={s['n_assigned']:5d}  "
              f"az {s.get('az_bias_deg', '—'):+5.1f}±{s.get('az_mad_deg', 0):4.1f} "
              f"(rms {s.get('az_rms_deg', 0):4.1f})   "
              f"el {s.get('el_bias_deg', 0):+5.1f}±{s.get('el_mad_deg', 0):4.1f} "
              f"(rms {s.get('el_rms_deg', 0):4.1f})")
    return results


def write_derived(args, rows, by_tid) -> None:
    cfg = load_config(args.config)
    uca = _build_uca(cfg)
    B = args.window
    peaks = reestimate(by_tid, window=B, max_span_s=args.max_span_s,
                       uca=uca, algo=args.algo, mdl=args.mdl)

    new_rows = [dict(r) for r in rows]
    for p in peaks:
        row = new_rows[p["row_idx"]]
        pk = list(row["peaks"][p["peak_idx"]])
        pk[0], pk[1], pk[3] = round(p["az"], 2), round(p["el"], 2), \
            round(p["papr_db"], 2)
        row["peaks"][p["peak_idx"]] = pk

    session_dir = os.path.abspath(args.session_dir.rstrip("/"))
    out_sub = f"{args.subdir}_covB{B}"
    out_dir = os.path.join(session_dir, out_sub)
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "doa_multi.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in new_rows:
            row.pop("y_per_peak", None)       # derived set: keep it light
            f.write(json.dumps(row) + "\n")
    meta = {"derived_from": args.subdir, "window_bursts": B,
            "max_span_s": args.max_span_s, "algo": args.algo,
            "mdl": args.mdl}
    with open(os.path.join(out_dir, "multiburst_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_jsonl} (track_ids preserved from {args.subdir})")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session_dir = os.path.abspath(args.session_dir.rstrip("/"))
    jsonl_path = os.path.join(session_dir, args.subdir, "doa_multi.jsonl")
    if not os.path.isfile(jsonl_path):
        sys.exit(f"No JSONL at {jsonl_path}")
    if args.write and args.window is None:
        sys.exit("--write requires --window N")

    rows = load_rows(jsonl_path)
    by_tid = collect_track_peaks(rows)
    n_pk = sum(len(v) for v in by_tid.values())
    print(f"{n_pk} tracked peaks in {len(by_tid)} tracks from {jsonl_path}")

    if args.write:
        write_derived(args, rows, by_tid)
        return

    use_session_tle(session_dir)   # freeze ground-truth elements per session
    t0, t1, _meta = load_session_window(session_dir)
    sat_tracks = build_sat_tracks(t0, t1 + timedelta(seconds=30), args)
    flat = [{"t_rel": p["t"], "cfo_hz": p["cfo_hz"]}
            for pts in by_tid.values() for p in pts]
    lo = float(args.lo_offset) if args.lo_offset is not None \
        else estimate_lo(flat, sat_tracks)
    print(f"LO offset: {lo:+.0f} Hz   algo={args.algo}  "
          f"max_span={args.max_span_s:.0f}s")
    sweep(args, by_tid, sat_tracks, lo)


if __name__ == "__main__":
    main()
