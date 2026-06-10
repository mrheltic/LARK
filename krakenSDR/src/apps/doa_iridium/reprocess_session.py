#!/usr/bin/env python3
"""
reprocess_session.py — Full offline multi-peak DOA reprocess from raw CPI frames.

Writes session_.../<out_subdir>/burst_NNNNNN.npz, doa_multi.jsonl, waterfall.npz,
and tracks.json.
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

from apps.doa_iridium.run_doa import (  # noqa: E402
    _PROFILES,
    load_config,
)
from core.multi_peak import process_cpi_for_multi  # noqa: E402
from core.recording import (  # noqa: E402
    count_session_raw_frames,
    iter_session_raw_frames,
    session_frame_times,
    _atomic_savez_compressed,
)
from core.track_clusterer import cluster_from_jsonl  # noqa: E402
from apps.doa_iridium.run_doa_offline import (  # noqa: E402
    _build_uca,
    _phase_offsets,
)

DEFAULT_ALGOS = ("music", "capon", "bartlett")


def default_out_subdir(algo: str | None = None) -> str:
    """Default output directory name for a given algorithm."""
    if algo:
        return f"doa_multi_{algo.lower()}"
    return "doa_multi"


def resolve_out_subdir(args: argparse.Namespace, algo: str | None = None) -> str:
    if getattr(args, "out_subdir", None):
        return args.out_subdir
    effective_algo = algo or getattr(args, "algo", None)
    return default_out_subdir(effective_algo)


def list_multi_dirs(session_dir: str) -> list[str]:
    """Return available doa_multi* directories in a session, sorted."""
    session_dir = session_dir.rstrip("/")
    if not os.path.isdir(session_dir):
        return []
    found: list[str] = []
    for name in sorted(os.listdir(session_dir)):
        path = os.path.join(session_dir, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("doa_multi"):
            continue
        jsonl = os.path.join(path, "doa_multi.jsonl")
        if os.path.isfile(jsonl):
            found.append(name)
    return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reprocess session raw/ → doa_multi/ with top-K peaks + tracks",
    )
    p.add_argument("session_dir", help="Path to session_YYYYMMDD_HHMMSS/")
    p.add_argument("--config", default=os.path.join(_HERE, "doa_config.toml"))
    p.add_argument("--mode", choices=["indoor", "outdoor"])
    p.add_argument("--algo", choices=list(DEFAULT_ALGOS))
    p.add_argument("--out-subdir", metavar="NAME",
                   help="Output subdir under session (default: doa_multi or doa_multi_<algo>)")
    p.add_argument("--k-peaks", type=int, default=3, help="Max peaks per burst")
    p.add_argument("--el-min", type=float, default=10.0,
                   help="Minimum elevation [°] to accept a DOA peak")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--save-spec", action="store_true",
                   help="Store spec2d in each burst_*.npz (larger files)")
    p.add_argument("--track-gap-s", type=float, default=30.0,
                   help="Max gap [s] before closing a track")
    p.add_argument("--min-track-len", type=int, default=5,
                   help="Min peaks to keep a track (shorter → outlier id -1)")
    p.add_argument("--max-az-deg", type=float, default=25.0,
                   help="Max azimuth separation [°] for track association")
    p.add_argument("--max-el-deg", type=float, default=15.0,
                   help="Max elevation separation [°] for track association")
    p.add_argument("--max-cfo-hz", type=float, default=12000.0,
                   help="Max CFO separation [Hz] for track association")
    p.add_argument("--recluster-only", action="store_true",
                   help="Re-run clustering on existing JSONL without reprocessing IQ")
    p.add_argument("--phase-cal", action="store_true")
    p.add_argument("--cal-file", metavar="PATH")
    p.add_argument("--phase-offs", metavar="DEG_LIST")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def recluster_session(
    session_dir: str,
    *,
    args: argparse.Namespace,
) -> dict:
    """Re-run clustering on an existing JSONL without reprocessing IQ data."""
    session_dir = os.path.abspath(session_dir)
    out_subdir = resolve_out_subdir(args)
    out_dir = os.path.join(session_dir, out_subdir)
    jsonl_path = os.path.join(out_dir, "doa_multi.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"No JSONL found at {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    tracks_path = os.path.join(out_dir, "tracks.json")
    print(
        f"Recluster → {out_subdir}/  "
        f"min_track_len={args.min_track_len}  max_az={getattr(args, 'max_az_deg', 25.0):.0f}°  "
        f"max_el={getattr(args, 'max_el_deg', 15.0):.0f}°  "
        f"max_cfo={getattr(args, 'max_cfo_hz', 12000.0):.0f}Hz  "
        f"gap={args.track_gap_s:.0f}s"
    )
    tracks = cluster_from_jsonl(
        jsonl_path,
        tracks_path,
        max_gap_s=args.track_gap_s,
        min_track_len=args.min_track_len,
        max_az_deg=getattr(args, "max_az_deg", 25.0),
        max_el_deg=getattr(args, "max_el_deg", 15.0),
        max_cfo_hz=getattr(args, "max_cfo_hz", 12000.0),
    )
    print(f"Tracks: {len(tracks)} → {tracks_path}")
    long = sum(1 for tr in tracks if tr.get("n_peaks", 0) >= 30)
    for tr in tracks[:10]:
        print(f"  T{tr['id']:2d}  n={tr['n_peaks']:4d}  az={tr['az_range']}")
    if len(tracks) > 10:
        print(f"  … and {len(tracks) - 10} more")
    return {"n_tracks": len(tracks), "n_long_tracks": long, "out_subdir": out_subdir}


def reprocess_session(
    session_dir: str,
    *,
    cfg: dict | None = None,
    args: argparse.Namespace | None = None,
) -> dict:
    """Run multi-peak reprocess; return summary dict."""
    if args is None:
        args = parse_args([session_dir])

    if getattr(args, "recluster_only", False):
        result = recluster_session(session_dir, args=args)
        out_subdir = result["out_subdir"]
        out_dir = os.path.join(os.path.abspath(session_dir), out_subdir)
        algo = out_subdir.replace("doa_multi_", "") if out_subdir.startswith("doa_multi_") else "music"
        if out_subdir == "doa_multi":
            algo = getattr(args, "algo", None) or "music"
        return {
            **result,
            "n_cpi": 0,
            "n_bursts": 0,
            "elapsed_s": 0.0,
            "doa_multi_dir": out_dir,
            "jsonl": os.path.join(out_dir, "doa_multi.jsonl"),
            "tracks": os.path.join(out_dir, "tracks.json"),
            "mean_papr_db": 0.0,
            "algo": algo,
            "recluster_only": True,
        }

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
    el_min_deg = float(getattr(args, "el_min", 5.0))
    fs_hz = float(meta.get("fs_hz", 1_024_000))

    total = count_session_raw_frames(session_dir)
    start = max(0, args.frame_start)
    stride = max(1, args.frame_stride)
    max_frames = args.max_frames if args.max_frames > 0 else max(
        0, (total - start + stride - 1) // stride,
    )

    out_subdir = resolve_out_subdir(args, alg["algo"])
    out_dir = os.path.join(session_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "doa_multi.jsonl")
    if os.path.isfile(jsonl_path):
        os.remove(jsonl_path)

    print(f"Reprocess → {out_subdir}/  {total} CPI in raw/  start={start} "
          f"stride={stride} max_frames={max_frames}  k={args.k_peaks}  "
          f"algo={alg['algo']}  el_min={el_min_deg:.0f}°  mode={alg['mode']}")

    burst_idx = 0
    az_slices: list[np.ndarray] = []
    burst_times: list[float] = []
    mean_papr_sum = 0.0
    mean_papr_n = 0
    t0 = time.time()
    n_seen = 0

    frame_iter = iter_session_raw_frames(
        session_dir, start=start, stride=stride, max_frames=max_frames,
    )
    # Wall-clock per frame: live recording drops CPIs when processing lags,
    # so fi × CPI duration drifts from real time (minutes over a session)
    # and would misalign the TLE ground truth.
    frame_ts = session_frame_times(session_dir)

    for fi, X in frame_iter:
        n_seen += 1
        cpi_size = int(X.shape[1])
        if frame_ts is not None and fi < len(frame_ts):
            t_rel = float(frame_ts[fi] - frame_ts[0])
        else:
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
            el_min_deg=el_min_deg,
            timestamp=t_rel,
        )
        if rec is None:
            if args.verbose and n_seen % 500 == 0:
                print(f"  … {n_seen} CPI, {burst_idx} bursts", file=sys.stderr)
            continue

        spec2d = rec.pop("spec2d")
        peaks = rec["peaks"]
        az_slice = np.max(spec2d, axis=0).astype(np.float32) if spec2d.ndim == 2 else np.zeros(360, dtype=np.float32)
        az_slices.append(az_slice)
        burst_times.append(float(rec["t"]))

        for row in peaks:
            mean_papr_sum += float(row[3])
            mean_papr_n += 1

        npz_payload = {
            "t": np.float64(rec["t"]),
            "frame": np.int32(fi),
            "cfo_hz": np.float32(rec["cfo_hz"]),
            "tone_hz": np.float32(rec["tone_hz"]),
            "snr_db": np.float32(rec["snr_db"]),
            "papr_db_global": np.float32(rec["papr_db_global"]),
            "peaks": peaks,
            "cfo_per_peak": np.asarray(rec["cfo_per_peak"], dtype=np.float32),
            "snr_per_peak": np.asarray(rec["snr_per_peak"], dtype=np.float32),
            "az_slice": az_slice,
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
            "cfo_per_peak": [round(float(c), 1) for c in rec["cfo_per_peak"]],
            "snr_per_peak": [round(float(s), 2) for s in rec["snr_per_peak"]],
            "track_ids": [],
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        burst_idx += 1
        if args.verbose and burst_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  … {burst_idx} bursts from {n_seen} CPI ({elapsed:.0f}s)",
                  file=sys.stderr)

    elapsed = time.time() - t0
    print(f"Processed {n_seen} CPI in {elapsed:.0f}s → {burst_idx} multi-bursts")

    if burst_idx > 0:
        wf = np.stack(az_slices, axis=0)
        wf_path = os.path.join(out_dir, "waterfall.npz")
        _atomic_savez_compressed(
            wf_path,
            waterfall=wf,
            t=np.asarray(burst_times, dtype=np.float64),
            algo=np.array(alg["algo"]),
        )
        print(f"Waterfall: {wf.shape} → {wf_path}")

    tracks_path = os.path.join(out_dir, "tracks.json")
    tracks: list[dict] = []
    if burst_idx > 0:
        tracks = cluster_from_jsonl(
            jsonl_path,
            tracks_path,
            max_gap_s=args.track_gap_s,
            min_track_len=args.min_track_len,
            max_az_deg=getattr(args, "max_az_deg", 25.0),
            max_el_deg=getattr(args, "max_el_deg", 15.0),
            max_cfo_hz=getattr(args, "max_cfo_hz", 12000.0),
            meta={
                "session": os.path.basename(session_dir),
                "out_subdir": out_subdir,
                "k_peaks": args.k_peaks,
                "el_min_deg": el_min_deg,
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

    mean_papr = mean_papr_sum / max(mean_papr_n, 1)
    long_tracks = sum(1 for tr in tracks if tr.get("n_peaks", 0) >= 30)

    return {
        "n_cpi": n_seen,
        "n_bursts": burst_idx,
        "n_tracks": len(tracks),
        "n_long_tracks": long_tracks,
        "mean_papr_db": mean_papr,
        "algo": alg["algo"],
        "out_subdir": out_subdir,
        "doa_multi_dir": out_dir,
        "jsonl": jsonl_path,
        "tracks": tracks_path if burst_idx > 0 else "",
        "elapsed_s": elapsed,
    }


def main() -> None:
    args = parse_args()
    summary = reprocess_session(args.session_dir, args=args)
    if summary.get("recluster_only"):
        return
    if summary["n_bursts"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
