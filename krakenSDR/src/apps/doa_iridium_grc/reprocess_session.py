#!/usr/bin/env python3
"""
reprocess_session.py — Full offline multi-peak DOA reprocess from raw CPI frames.

Writes session_.../doa_multi/burst_NNNNNN.npz, doa_multi.jsonl, and tracks.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from apps.doa_iridium_grc.run_doa import (  # noqa: E402
    _PROFILES,
    _load_cal,
    load_config,
)
from apps.doa_iridium_grc.lark.multi_peak import process_cpi_for_multi  # noqa: E402
from apps.doa_iridium_grc.lark.recording import (  # noqa: E402
    count_session_raw_frames,
    iter_session_raw_frames,
    _atomic_savez_compressed,
)
from apps.doa_iridium_grc.lark.track_clusterer import cluster_from_jsonl  # noqa: E402
from apps.doa_iridium_grc.run_doa_offline import (  # noqa: E402
    _build_uca,
    _phase_offsets,
)
from core.doa_uca_2d import UcaConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reprocess session raw/ → doa_multi/ with top-K peaks + tracks",
    )
    p.add_argument("session_dir", help="Path to session_YYYYMMDD_HHMMSS/")
    p.add_argument("--config", default=os.path.join(_HERE, "doa_config.toml"))
    p.add_argument("--mode", choices=["indoor", "outdoor"])
    p.add_argument("--algo", choices=["music", "capon", "bartlett"])
    p.add_argument("--k-peaks", type=int, default=3, help="Max peaks per burst")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--save-spec", action="store_true",
                   help="Store spec2d in each burst_*.npz (larger files)")
    p.add_argument("--track-gap-s", type=float, default=30.0,
                   help="Max gap [s] before closing a track")
    p.add_argument("--min-track-len", type=int, default=5,
                   help="Min peaks to keep a track (shorter → outlier id -1)")
    p.add_argument("--phase-cal", action="store_true")
    p.add_argument("--cal-file", metavar="PATH")
    p.add_argument("--phase-offs", metavar="DEG_LIST")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def reprocess_session(
    session_dir: str,
    *,
    cfg: dict | None = None,
    args: argparse.Namespace | None = None,
) -> dict:
    """Run multi-peak reprocess; return summary dict."""
    if args is None:
        args = parse_args([session_dir])

    session_dir = os.path.abspath(session_dir)
    if cfg is None:
        cfg = load_config(args.config)

    meta_path = os.path.join(session_dir, "meta.json")
    meta: dict = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cfg = {**cfg, "hardware": {**cfg["hardware"],
                                   "freq_mhz": float(meta.get("freq_hz",
                                                               cfg["hardware"]["freq_mhz"] * 1e6)) / 1e6}}
        if args.mode is None and meta.get("mode"):
            cfg = {**cfg, "algorithm": {**cfg["algorithm"], "mode": meta["mode"]}}
        if args.algo is None and meta.get("algo"):
            cfg = {**cfg, "algorithm": {**cfg["algorithm"], "algo": meta["algo"]}}

    if args.mode:
        cfg = {**cfg, "algorithm": {**cfg["algorithm"], "mode": args.mode}}
    if args.algo:
        cfg = {**cfg, "algorithm": {**cfg["algorithm"], "algo": args.algo}}
    if args.phase_cal:
        cfg = {**cfg, "array": {**cfg["array"], "use_phase_cal": True}}

    alg = cfg["algorithm"]
    profile = _PROFILES[alg["mode"]]
    uca = _build_uca(cfg)
    phase_offs = _phase_offsets(cfg, args)
    thr = cfg["thresholds"]
    fs_hz = float(meta.get("fs_hz", 1_024_000))
    cpi_size_meta = int(meta.get("cpi_size", cfg["hardware"]["cpi_size"]))

    total = count_session_raw_frames(session_dir)
    start = max(0, args.frame_start)
    stride = max(1, args.frame_stride)
    max_frames = args.max_frames if args.max_frames > 0 else max(
        0, (total - start + stride - 1) // stride,
    )

    out_dir = os.path.join(session_dir, "doa_multi")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "doa_multi.jsonl")
    if os.path.isfile(jsonl_path):
        os.remove(jsonl_path)

    print(f"Reprocess: {total} CPI in raw/  start={start} stride={stride} "
          f"max_frames={max_frames}  k={args.k_peaks}  mode={alg['mode']}")

    burst_idx = 0
    json_rows: list[dict] = []
    t0 = time.time()
    n_seen = 0

    frame_iter = iter_session_raw_frames(
        session_dir, start=start, stride=stride, max_frames=max_frames,
    )
    for fi, X in frame_iter:
        n_seen += 1
        cpi_size = int(X.shape[1])
        t_rel = fi * cpi_size / fs_hz

        rec = process_cpi_for_multi(
            X,
            frame_idx=fi,
            cfg=cfg,
            profile=profile,
            uca=uca,
            phase_offs=phase_offs,
            algo=alg["algo"],
            thr=thr,
            k_peaks=args.k_peaks,
            timestamp=t_rel,
        )
        if rec is None:
            if args.verbose and n_seen % 500 == 0:
                print(f"  … {n_seen} CPI, {burst_idx} bursts", file=sys.stderr)
            continue

        spec2d = rec.pop("spec2d")
        peaks = rec["peaks"]
        npz_payload = {
            "t": np.float64(rec["t"]),
            "frame": np.int32(fi),
            "cfo_hz": np.float32(rec["cfo_hz"]),
            "tone_hz": np.float32(rec["tone_hz"]),
            "snr_db": np.float32(rec["snr_db"]),
            "papr_db_global": np.float32(rec["papr_db_global"]),
            "peaks": peaks,
        }
        if args.save_spec:
            npz_payload["spec2d"] = spec2d

        burst_path = os.path.join(out_dir, f"burst_{burst_idx:06d}.npz")
        _atomic_savez_compressed(burst_path, **npz_payload)

        row = {
            "burst_idx": burst_idx,
            "frame": fi,
            "t": rec["t"],
            "cfo_hz": rec["cfo_hz"],
            "tone_hz": rec["tone_hz"],
            "snr_db": rec["snr_db"],
            "papr_db_global": rec["papr_db_global"],
            "peaks": peaks.tolist(),
            "track_ids": [],
        }
        json_rows.append(row)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        burst_idx += 1
        if args.verbose and burst_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  … {burst_idx} bursts from {n_seen} CPI ({elapsed:.0f}s)",
                  file=sys.stderr)

    elapsed = time.time() - t0
    print(f"Processed {n_seen} CPI in {elapsed:.0f}s → {burst_idx} multi-bursts")

    tracks_path = os.path.join(session_dir, "tracks.json")
    tracks: list[dict] = []
    if burst_idx > 0:
        tracks = cluster_from_jsonl(
            jsonl_path,
            tracks_path,
            max_gap_s=args.track_gap_s,
            min_track_len=args.min_track_len,
            meta={
                "session": os.path.basename(session_dir),
                "k_peaks": args.k_peaks,
                "n_bursts": burst_idx,
                "mode": alg["mode"],
                "algo": alg["algo"],
            },
        )
        print(f"Tracks: {len(tracks)} satellites/passages → {tracks_path}")
        for tr in tracks[:10]:
            print(f"  T{tr['id']:2d}  n={tr['n_peaks']:4d}  "
                  f"az={tr['az_range']}  cfo={tr['cfo_range_hz']} Hz")
        if len(tracks) > 10:
            print(f"  … and {len(tracks) - 10} more tracks")
    else:
        print("No bursts detected — tracks.json not written", file=sys.stderr)

    return {
        "n_cpi": n_seen,
        "n_bursts": burst_idx,
        "n_tracks": len(tracks),
        "doa_multi_dir": out_dir,
        "jsonl": jsonl_path,
        "tracks": tracks_path if burst_idx > 0 else "",
        "elapsed_s": elapsed,
    }


def main() -> None:
    args = parse_args()
    summary = reprocess_session(args.session_dir, args=args)
    if summary["n_bursts"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
