#!/usr/bin/env python3
"""
iridium_groundtruth.py — Iridium satellite passes from TLE for a DOA session.

Fetches Iridium NEXT TLEs from Celestrak, predicts all visible passes during the
session window, computes Az/El/Doppler per satellite, and saves groundtruth
that can be compared against DOA estimates.

Usage:
    # Pass prediction for a session (uses meta.json timestamps)
    python3 scripts/iridium_groundtruth.py session_20260605_110716/

    # Pass prediction for a time window at explicit location
    python3 scripts/iridium_groundtruth.py --t0 "2026-06-05T11:00" --t1 "2026-06-05T12:00"

    # Match DOA tracks against predicted satellites
    python3 scripts/iridium_groundtruth.py session_20260605_110716/ --match

    # Export ground-truth for clean_tracks.json
    python3 scripts/iridium_groundtruth.py session_20260605_110716/ --clean-only

Coordinates: 43.61464°N, 7.07184°E  (Biot / Sophia Antipolis, 372 m AMSL)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)             # krakenSDR/src/
_ROOT = os.path.dirname(os.path.dirname(_SRC))  # LARK project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Default observer: Biot / Sophia Antipolis
OBSERVER_LAT = 43.614642858726036
OBSERVER_LON = 7.071836433649546
OBSERVER_ALT = 372.0   # m AMSL


def parse_isotime(s: str) -> datetime:
    """Parse ISO 8601 to UTC datetime, with fallbacks."""
    # Try Python's built-in parser first (handles fractional seconds, timezone)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    for fmt in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]:
        try:
            dt = datetime.strptime(s[:19], fmt) if len(s) >= 19 else datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {s!r}")


def load_session_window(session_dir: str) -> tuple[datetime, datetime, dict]:
    """Extract time window and metadata from a session directory."""
    meta_path = os.path.join(session_dir, "meta.json")
    state_path = os.path.join(session_dir, "state.json")

    meta = {}
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))

    # Preferred clock: raw frame write times (unambiguous epochs).  The
    # 'created' field of older sessions is LOCAL time without offset, and
    # CPI-count timing drifts because live recording drops frames — both
    # break TLE alignment by minutes to hours.
    from core.recording import session_frame_times

    ts = session_frame_times(session_dir)
    if ts is not None and len(ts) >= 2:
        cpi_dur = float(meta.get("cpi_size", 131072)) / float(meta.get("fs_hz", 1_024_000.0))
        t0 = datetime.fromtimestamp(float(ts[0]) - cpi_dur, tz=timezone.utc)
        t1 = datetime.fromtimestamp(float(ts[-1]), tz=timezone.utc)
        return t0, t1, meta

    t0_str = meta.get("created", "")
    if not t0_str:
        raise ValueError(f"No 'created' field in {meta_path}")

    t0 = parse_isotime(t0_str)

    # Try to get end time from state.json, else from meta duration
    t1 = t0 + timedelta(minutes=40)  # default: 40 min
    if os.path.isfile(state_path):
        state = json.load(open(state_path))
        elapsed = state.get("elapsed_s", 0)
        if elapsed > 0:
            t1 = t0 + timedelta(seconds=elapsed)

    return t0, t1, meta


def predict_passes(
    t0: datetime,
    t1: datetime,
    lat: float = OBSERVER_LAT,
    lon: float = OBSERVER_LON,
    alt: float = OBSERVER_ALT,
    el_min_deg: float = 5.0,
    force_download: bool = False,
) -> list[dict]:
    """Predict all Iridium NEXT passes in [t0, t1] above el_min_deg."""
    from shared.iridium_tle import load_catalogue

    cat = load_catalogue(force_download=force_download)
    return cat.predict_passes(
        lat=lat, lon=lon, alt=alt,
        t0_utc=t0, t1_utc=t1,
        el_min_deg=el_min_deg,
        step_s=20.0,
    )


def compute_track(
    sat_name: str,
    sat_norad: int,
    t0: datetime,
    t1: datetime,
    lat: float = OBSERVER_LAT,
    lon: float = OBSERVER_LON,
    alt: float = OBSERVER_ALT,
    step_s: float = 1.0,
    force_download: bool = False,
) -> dict:
    """Compute high-resolution Az/El/Doppler track for one satellite."""
    from shared.iridium_tle import load_catalogue

    cat = load_catalogue(force_download=force_download)
    return cat.doppler_track(
        sat_name_or_id=sat_name,
        lat=lat, lon=lon, alt=alt,
        t0_utc=t0, t1_utc=t1,
        freq_hz=1_626_270_000.0,
        step_s=step_s,
    )


def match_doa_to_satellites(
    doa_tracks: list[dict],
    sat_passes: list[dict],
    session_t0: datetime,
    az_tol_deg: float = 30.0,
    cfo_tol_hz: float = 15000.0,
    lo_offset_hz: float = 0.0,
) -> list[dict]:
    """Match DOA track clusters to predicted satellite passes.

    lo_offset_hz: receiver LO frequency offset subtracted from measured CFOs
    before comparison (the Kraken TCXO ppm error biases every CFO by the same
    amount; estimate it as the median signed residual of a first match pass).
    """
    from shared.iridium_tle import load_catalogue

    # Load catalogue once for all matches
    cat = load_catalogue()

    matches = []
    for trk in doa_tracks:
        t_start = session_t0 + timedelta(seconds=trk["t_start"])
        t_end = session_t0 + timedelta(seconds=trk["t_end"])
        median_az = (trk["az_range"][0] + trk["az_range"][1]) / 2
        median_el = (trk["el_range"][0] + trk["el_range"][1]) / 2
        median_cfo = (trk["cfo_range_hz"][0] + trk["cfo_range_hz"][1]) / 2 - lo_offset_hz

        best = None
        best_score = float("inf")
        for sat in sat_passes:
            # Time overlap check
            rise = parse_isotime(sat["rise_utc"])
            sset = parse_isotime(sat["set_utc"])
            if t_end < rise or t_start > sset:
                continue

            # Azimuth at track midpoint
            t_mid = t_start + (t_end - t_start) / 2
            try:
                sat_az, sat_el, _ = cat.sat_azel(
                    sat["name"], OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT, t_mid
                )
            except Exception:
                sat_az = (sat["rise_az_deg"] + sat["set_az_deg"]) / 2
                sat_el = sat["culm_el_deg"]

            az_diff = min(abs(sat_az - median_az), 360 - abs(sat_az - median_az))
            el_diff = abs(sat_el - median_el)
            
            # Predicted CFO at track midpoint (faster: use sat track)
            try:
                cfo_track = cat.doppler_track(
                    sat["name"], OBSERVER_LAT, OBSERVER_LON, OBSERVER_ALT,
                    t_start, t_end, step_s=10.0,
                )
                mid_idx = len(cfo_track["doppler_hz"]) // 2
                pred_cfo = cfo_track["doppler_hz"][mid_idx] if mid_idx < len(cfo_track["doppler_hz"]) else 0
            except Exception:
                pred_cfo = median_cfo

            cfo_diff = abs(median_cfo - pred_cfo)

            if az_diff > az_tol_deg:
                continue
            if cfo_diff > cfo_tol_hz:
                continue

            score = az_diff * 2 + cfo_diff / 1000
            if score < best_score:
                best_score = score
                best = {
                    "satellite": sat["name"],
                    "norad_id": sat["norad_id"],
                    "az_error_deg": round(az_diff, 1),
                    "el_error_deg": round(el_diff, 1),
                    "cfo_error_hz": round(cfo_diff, 0),
                    "cfo_resid_hz": round(median_cfo - pred_cfo, 0),
                    "score": round(best_score, 1),
                    "doa_az_med": round(median_az, 1),
                    "doa_el_med": round(median_el, 1),
                    "doa_cfo_med": round(median_cfo, 0),
                    "sat_az_mid": round(sat_az, 1),
                    "sat_el_mid": round(sat_el, 1),
                    "sat_pred_cfo": round(pred_cfo, 0),
                    "sat_rise": sat["rise_utc"],
                    "sat_set": sat["set_utc"],
                    "sat_max_el": sat["max_el_deg"],
                    "track_id": trk["id"],
                    "track_n_peaks": trk["n_peaks"],
                    "track_dur_s": round(trk["t_end"] - trk["t_start"], 1),
                }

        if best is not None:
            matches.append(best)
        else:
            matches.append({
                "satellite": None,
                "track_id": trk["id"],
                "track_n_peaks": trk["n_peaks"],
                "track_az_med": round(median_az, 1),
                "track_el_med": round(median_el, 1),
                "track_cfo_med": round(median_cfo, 0),
                "track_dur_s": round(trk["t_end"] - trk["t_start"], 1),
                "unmatched": True,
            })

    return matches


def main():
    p = argparse.ArgumentParser(
        description="Iridium satellite pass prediction and ground-truth for DOA sessions"
    )
    p.add_argument("session_dir", nargs="?", default="",
                   help="Path to session_.../ directory (reads meta.json)")
    p.add_argument("--lat", type=float, default=OBSERVER_LAT,
                   help="Observer latitude [°N] (default: Biot/Sophia Antipolis)")
    p.add_argument("--lon", type=float, default=OBSERVER_LON,
                   help="Observer longitude [°E]")
    p.add_argument("--alt", type=float, default=OBSERVER_ALT,
                   help="Observer altitude [m AMSL]")
    p.add_argument("--t0", help="Start time ISO 8601 (overrides session meta)")
    p.add_argument("--t1", help="End time ISO 8601")
    p.add_argument("--el-min", type=float, default=5.0, help="Min elevation [deg]")
    p.add_argument("--force-download", action="store_true",
                   help="Force fresh TLE download")
    p.add_argument("--match", action="store_true",
                   help="Match DOA tracks (clean_tracks.json) to satellites")
    p.add_argument("--clean-only", action="store_true",
                   help="Use clean_tracks.json instead of original tracks.json")
    p.add_argument("--az-tol", type=float, default=30.0,
                   help="Azimuth tolerance for matching [deg]")
    p.add_argument("--cfo-tol", type=float, default=15000.0,
                   help="CFO tolerance for matching [Hz]")
    args = p.parse_args()

    # Determine time window
    meta = {}
    if args.session_dir:
        session = args.session_dir.rstrip("/")
        if os.path.isdir(session):
            t0, t1, meta = load_session_window(session)
            print(f"Session: {os.path.basename(session)}")
            print(f"  Window: {t0.isoformat()} → {t1.isoformat()} "
                  f"({(t1-t0).total_seconds()/60:.0f} min)")
    elif args.t0 and args.t1:
        t0 = parse_isotime(args.t0)
        t1 = parse_isotime(args.t1)
    else:
        # Default: next 6 hours
        t0 = datetime.now(timezone.utc)
        t1 = t0 + timedelta(hours=6)

    print(f"Observer: {args.lat:.4f}°N, {args.lon:.4f}°E, {args.alt:.0f}m")
    print(f"Window:   {t0.isoformat(timespec='seconds')} → "
          f"{t1.isoformat(timespec='seconds')}\n")

    # Predict passes
    passes = predict_passes(
        t0=t0, t1=t1,
        lat=args.lat, lon=args.lon, alt=args.alt,
        el_min_deg=args.el_min,
        force_download=args.force_download,
    )
    print(f"Predicted passes (el ≥ {args.el_min:.0f}°): {len(passes)}\n")

    for i, sat in enumerate(passes[:30]):
        rise = parse_isotime(sat["rise_utc"])
        sset = parse_isotime(sat["set_utc"])
        r_str = rise.strftime("%H:%M:%S")
        s_str = sset.strftime("%H:%M:%S")
        print(f"{i:3d}. {sat['name']:20s}  "
              f"↑{sat['max_el_deg']:5.1f}°  "
              f"az:{sat['rise_az_deg']:6.1f}→{sat['culm_az_deg']:6.1f}→{sat['set_az_deg']:6.1f}  "
              f"{r_str}→{s_str}  {sat['duration_s']:5.0f}s")

    # Save groundtruth
    if args.session_dir and os.path.isdir(args.session_dir):
        out_path = os.path.join(args.session_dir, "groundtruth.json")
        gt = {
            "session": os.path.basename(args.session_dir),
            "observer": {"lat": args.lat, "lon": args.lon, "alt_m": args.alt},
            "window_utc": {"t0": t0.isoformat(), "t1": t1.isoformat()},
            "n_passes": len(passes),
            "passes": [{
                "name": s["name"],
                "norad_id": s["norad_id"],
                "rise_utc": s["rise_utc"],
                "culmination_utc": s["culmination_utc"],
                "set_utc": s["set_utc"],
                "rise_az_deg": s["rise_az_deg"],
                "culm_az_deg": s["culm_az_deg"],
                "culm_el_deg": s["culm_el_deg"],
                "set_az_deg": s["set_az_deg"],
                "max_el_deg": s["max_el_deg"],
                "duration_s": s["duration_s"],
            } for s in passes],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2)
        print(f"\nSaved groundtruth to {out_path}")

    # Match against DOA tracks if requested
    if args.match and args.session_dir:
        session = args.session_dir.rstrip("/")
        tracks_file = os.path.join(session, "clean_tracks.json" if args.clean_only else "tracks.json")
        if not os.path.isfile(tracks_file):
            print(f"\nNo tracks file at {tracks_file} — run clean_session.py first or use --match without --clean-only")
            return

        with open(tracks_file) as f:
            tracks_data = json.load(f)
        doa_tracks = tracks_data.get("tracks", tracks_data) if isinstance(tracks_data, dict) else tracks_data
        if isinstance(doa_tracks, list):
            pass
        elif isinstance(doa_tracks, dict):
            doa_tracks = list(doa_tracks.values())

        print(f"\nMatching {len(doa_tracks)} DOA tracks against {len(passes)} satellite passes...")
        matches = match_doa_to_satellites(doa_tracks, passes, t0, args.az_tol, args.cfo_tol)

        # Receiver LO offset: the Kraken TCXO ppm error shifts every measured
        # CFO by the same amount.  Estimate it as the median signed residual
        # over the matched tracks, then re-match with the bias removed — this
        # tightens cfo_error and can recover matches lost to the CFO gate.
        lo_offset_hz = 0.0
        resids = [m["cfo_resid_hz"] for m in matches if m.get("satellite")]
        if len(resids) >= 3:
            lo_offset_hz = float(np.median(resids))
            ppm = lo_offset_hz / 1626.27  # 1 ppm = 1626.27 Hz at 1626.27 MHz
            print(f"  LO offset estimate: {lo_offset_hz:+.0f} Hz ({ppm:+.2f} ppm) "
                  f"from {len(resids)} matched tracks")
            if abs(lo_offset_hz) > 200.0:
                print("  Re-matching with LO correction applied…")
                matches = match_doa_to_satellites(
                    doa_tracks, passes, t0, args.az_tol, args.cfo_tol,
                    lo_offset_hz=lo_offset_hz,
                )

        matched = [m for m in matches if m.get("satellite")]
        unmatched = [m for m in matches if m.get("unmatched")]

        print(f"  Matched:   {len(matched)}")
        print(f"  Unmatched: {len(unmatched)}")

        if matched:
            print(f"\n{'SATELLITE':20s} {'DOA az':>6s} {'SAT az':>6s} {'Δaz':>5s} {'DOA el':>6s} {'SAT el':>6s} {'CFO err':>8s} {'N':>4s}")
            print("-" * 80)
            for m in sorted(matched, key=lambda x: -x["track_n_peaks"])[:20]:
                print(f"{m['satellite']:20s} {m['doa_az_med']:6.1f}° {m['sat_az_mid']:6.1f}° {m['az_error_deg']:5.1f}° "
                      f"{m['doa_el_med']:6.1f}° {m['sat_el_mid']:6.1f}° {m['cfo_error_hz']:8.0f}Hz {m['track_n_peaks']:4d}")

        # Save matches
        match_path = os.path.join(session, "groundtruth_matches.json")
        with open(match_path, "w", encoding="utf-8") as f:
            json.dump({
                "n_matched": len(matched),
                "n_unmatched": len(unmatched),
                "lo_offset_hz": round(lo_offset_hz, 0),
                "matches": matches,
            }, f, indent=2)
        print(f"Saved matches to {match_path}")


if __name__ == "__main__":
    main()
