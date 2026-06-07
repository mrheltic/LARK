#!/usr/bin/env python3
"""
batch_reprocess.py — Run multi-peak DOA reprocess for MUSIC, Capon, and Bartlett.

Each algorithm writes to its own session subdirectory:
  doa_multi_music/, doa_multi_capon/, doa_multi_bartlett/
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from apps.doa_iridium_grc.reprocess_session import (  # noqa: E402
    DEFAULT_ALGOS,
    reprocess_session,
    parse_args as reprocess_parse_args,
)
from apps.doa_iridium_grc.run_doa import load_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch reprocess session with MUSIC, Capon, and Bartlett",
    )
    p.add_argument("session_dir", help="Path to session_YYYYMMDD_HHMMSS/")
    p.add_argument("--config", default=os.path.join(_HERE, "doa_config.toml"))
    p.add_argument(
        "--algos", nargs="+", choices=list(DEFAULT_ALGOS),
        default=list(DEFAULT_ALGOS),
        help="Algorithms to run (default: all three)",
    )
    p.add_argument("--mode", choices=["indoor", "outdoor"])
    p.add_argument("--k-peaks", type=int, default=3)
    p.add_argument("--el-min", type=float, default=10.0,
                   help="Minimum elevation [°] to accept a DOA peak")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--save-spec", action="store_true")
    p.add_argument("--track-gap-s", type=float, default=30.0)
    p.add_argument("--min-track-len", type=int, default=10,
                   help="Min peaks to keep a track (default 10)")
    p.add_argument("--max-az-deg", type=float, default=25.0)
    p.add_argument("--max-el-deg", type=float, default=15.0)
    p.add_argument("--max-cfo-hz", type=float, default=12000.0)
    p.add_argument("--recluster-only", action="store_true",
                   help="Re-run clustering only on existing JSONL (no IQ reprocess)")
    p.add_argument("--phase-cal", action="store_true")
    p.add_argument("--cal-file", metavar="PATH")
    p.add_argument("--phase-offs", metavar="DEG_LIST")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def print_comparison_table(summaries: list[dict], *, recluster: bool = False) -> None:
    if not summaries:
        return
    print("\n" + "=" * 72)
    print("Recluster summary" if recluster else "Algorithm comparison")
    print("=" * 72)
    if recluster:
        print(f"{'Algo':<10} {'Tracks':>7} {'Long≥30':>8} {'Long≥50':>8}")
        print("-" * 72)
        for s in summaries:
            tr_path = s.get("tracks", "")
            long30 = long50 = 0
            if tr_path and os.path.isfile(tr_path):
                import json
                with open(tr_path, encoding="utf-8") as f:
                    tr = json.load(f).get("tracks", [])
                long30 = sum(1 for t in tr if t.get("n_peaks", 0) >= 30)
                long50 = sum(1 for t in tr if t.get("n_peaks", 0) >= 50)
            print(
                f"{s.get('algo', '?'):<10} {s.get('n_tracks', 0):>7} "
                f"{long30:>8} {long50:>8}"
            )
    else:
        print(f"{'Algo':<10} {'Bursts':>7} {'Tracks':>7} {'Long≥30':>8} {'MeanPAPR':>9} {'Time[s]':>8}")
        print("-" * 72)
        for s in summaries:
            print(
                f"{s['algo']:<10} {s['n_bursts']:>7} {s['n_tracks']:>7} "
                f"{s['n_long_tracks']:>8} {s['mean_papr_db']:>8.1f}dB "
                f"{s['elapsed_s']:>8.0f}"
            )
    print("=" * 72)


def main() -> None:
    args = parse_args()
    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        print(f"Not a session directory: {session_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    summaries: list[dict] = []

    for algo in args.algos:
        print(f"\n{'─' * 60}\nRunning {algo.upper()}\n{'─' * 60}")
        rp_args = reprocess_parse_args([session_dir])
        rp_args.algo = algo
        rp_args.out_subdir = f"doa_multi_{algo}"
        rp_args.k_peaks = args.k_peaks
        rp_args.el_min = args.el_min
        rp_args.frame_start = args.frame_start
        rp_args.frame_stride = args.frame_stride
        rp_args.max_frames = args.max_frames
        rp_args.save_spec = args.save_spec
        rp_args.track_gap_s = args.track_gap_s
        rp_args.min_track_len = args.min_track_len
        rp_args.max_az_deg = args.max_az_deg
        rp_args.max_el_deg = args.max_el_deg
        rp_args.max_cfo_hz = args.max_cfo_hz
        rp_args.recluster_only = args.recluster_only
        rp_args.phase_cal = args.phase_cal
        rp_args.cal_file = args.cal_file
        rp_args.phase_offs = args.phase_offs
        rp_args.verbose = args.verbose
        if args.mode:
            rp_args.mode = args.mode

        summary = reprocess_session(session_dir, cfg=cfg, args=rp_args)
        summaries.append(summary)

    print_comparison_table(summaries, recluster=args.recluster_only)
    if args.recluster_only:
        if all(s.get("n_tracks", 0) == 0 for s in summaries):
            sys.exit(1)
        return
    if all(s["n_bursts"] == 0 for s in summaries):
        sys.exit(1)


if __name__ == "__main__":
    main()
