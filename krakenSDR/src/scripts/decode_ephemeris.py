#!/usr/bin/env python3
"""
decode_ephemeris.py — Build a satellite ephemeris from the downlink itself.

Turns the output of ``iridium-parser.py`` into ``<session>/broadcast_ephemeris.json``:
per-satellite short-arc fits, the receiver clock offset measured against Iridium
system time, and (optionally) a check against the frozen TLE snapshot.

No network access, no TLE catalogue and no host clock are needed to *produce*
the ephemeris — the TLE is used only by ``--validate-vs-tle``, as a yardstick.

Full pipeline, from a recorded session (run from krakenSDR/src/):

    # 1. export CPIs as single-channel cf32 (see scripts/export_iq.py)
    python3 scripts/export_iq.py <session> --ant 0 --out /tmp/cpi

    # 2. demodulate (gr-iridium; needs iridium-extractor on PATH)
    for f in /tmp/cpi/*.cf32; do
        iridium-extractor -f cf32_le -r 1000000 -c 1626270000 "$f"
    done | grep '^RAW:' > /tmp/frames.bits

    # 3. parse bits into frames  (-o line, or nothing is printed)
    python3 external/iridium-toolkit/iridium-parser.py -o line \
        /tmp/frames.bits > /tmp/frames.parsed

    # 4. this script
    python3 scripts/decode_ephemeris.py <session> \
        --parsed /tmp/frames.parsed --index /tmp/cpi/index.json --validate-vs-tle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.broadcast_ephemeris import (  # noqa: E402
    IraEphemeris,
    estimate_clock_offset,
    filter_plausible,
    load_export_index,
    parse_decodes,
    remap_epochs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a broadcast ephemeris from decoded Iridium frames")
    p.add_argument("session_dir")
    p.add_argument("--parsed", required=True, help="iridium-parser.py -o line output")
    p.add_argument("--index", required=True, help="index.json from export_iq.py")
    p.add_argument("--out", default="", help="Output JSON (default: <session>/broadcast_ephemeris.json)")
    p.add_argument("--arc-deg", type=int, default=5, help="Short-arc polynomial degree")
    p.add_argument("--arc-gap-s", type=float, default=120.0,
                   help="Time gap that starts a new pass [s]")
    p.add_argument("--min-fixes", type=int, default=10,
                   help="Ignore satellites with fewer plausible fixes")
    p.add_argument("--keep-host-clock", action="store_true",
                   help="Do not correct epochs with the IBC-measured clock offset")
    p.add_argument("--validate-vs-tle", action="store_true",
                   help="Score each decoded satellite against the session TLE snapshot")
    return p.parse_args(argv)


def validate_vs_tle(session_dir: str, fixes_by_sat: dict[int, np.ndarray]) -> list[dict]:
    """Nearest TLE satellite to each decoded sat_id, and the runner-up.

    The runner-up matters more than the winner: identification is only
    meaningful if the margin is large.  A few thousand km of separation between
    first and second says the match cannot be coincidence.
    """
    from skyfield.framelib import itrs

    from shared.iridium_tle import load_catalogue, use_session_tle

    cat = load_catalogue(str(use_session_tle(session_dir)))
    out = []
    for sat_id, arr in sorted(fixes_by_sat.items()):
        t, pos = arr[:, 0], arr[:, 1:]
        times = cat._ts.from_datetimes(
            [datetime.fromtimestamp(float(x), tz=timezone.utc) for x in t])
        scored = []
        for name, sat in cat._by_name.items():
            r = sat.at(times).frame_xyz(itrs).km.T
            scored.append((float(np.median(np.linalg.norm(r - pos, axis=1))), name))
        scored.sort()
        out.append({
            "sat_id": int(sat_id),
            "n_fixes": int(t.size),
            "tle_name": scored[0][1],
            "residual_km": round(scored[0][0], 3),
            "runner_up": scored[1][1],
            "runner_up_km": round(scored[1][0], 1),
            "margin": round(scored[1][0] / max(scored[0][0], 1e-9), 1),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = args.session_dir.rstrip("/")

    fixes, times = parse_decodes(args.parsed)
    index = load_export_index(args.index)
    fixes = remap_epochs(fixes, index)
    times = remap_epochs(times, index)
    print(f"[decode] {len(fixes)} IRA fixes, {len(times)} IBC times (epochs remapped)")

    clock_offset, clock_mad, n_clock = estimate_clock_offset(times)
    if n_clock:
        print(f"[clock] host is {clock_offset:+.3f} s off Iridium system time "
              f"(MAD {clock_mad:.3f} s, n={n_clock})")
    else:
        print("[clock] no IBC time frames decoded — host clock left uncorrected")

    # Correct the recording clock before anything geometric happens: an
    # uncorrected second of error moves the satellite 7.5 km along track.
    applied = 0.0
    if n_clock and not args.keep_host_clock and np.isfinite(clock_offset):
        applied = clock_offset
        for f in fixes:
            f.t -= applied

    plausible = filter_plausible(fixes)
    print(f"[decode] {len(plausible)}/{len(fixes)} positions pass the orbital "
          f"radius test ({100 * len(plausible) / max(len(fixes), 1):.0f}%)")

    by_sat: dict[int, list] = {}
    for f in plausible:
        by_sat.setdefault(f.sat_id, []).append(f)
    kept = {s: g for s, g in by_sat.items() if len(g) >= args.min_fixes}
    dropped = sorted(set(by_sat) - set(kept))
    if dropped:
        print(f"[decode] ignoring {len(dropped)} satellite(s) with <"
              f"{args.min_fixes} fixes: {', '.join(f'sat:{s:03d}' for s in dropped)}")

    eph = IraEphemeris.from_fixes([f for g in kept.values() for f in g],
                                  deg=args.arc_deg, gap_s=args.arc_gap_s)

    sats = []
    for sat_id in eph.satellites:
        arcs = [{"t_start": round(a.t_start, 3), "t_end": round(a.t_end, 3),
                 "n_used": a.n_used, "residual_km": round(a.residual_km, 3),
                 "t0": a.t0, "scale": a.scale, "coeffs": a.coeffs.tolist()}
                for a in eph.arcs[sat_id]]
        sats.append({"sat_id": sat_id, "n_fixes": len(kept[sat_id]), "arcs": arcs})
        for a in eph.arcs[sat_id]:
            print(f"  sat:{sat_id:03d}  {a.n_used:4d} pts  {a.t_end - a.t_start:6.0f} s"
                  f"  fit residual {a.residual_km:.2f} km")

    result = {
        "session": os.path.basename(session),
        "source": os.path.basename(args.parsed),
        "n_ira_ok": len(fixes),
        "n_plausible": len(plausible),
        "clock_offset_s": None if not n_clock else round(clock_offset, 4),
        "clock_mad_s": None if not n_clock else round(clock_mad, 4),
        "clock_n_ibc": n_clock,
        "clock_applied_s": round(applied, 4),
        "arc_deg": args.arc_deg,
        "satellites": sats,
    }

    if args.validate_vs_tle:
        arrs = {s: np.array([[f.t, *f.pos_km] for f in g]) for s, g in kept.items()}
        result["tle_validation"] = validate_vs_tle(session, arrs)
        print("\n[validate] decoded sat_id -> nearest TLE satellite")
        for row in result["tle_validation"]:
            print(f"  sat:{row['sat_id']:03d} {row['n_fixes']:4d} fixes  "
                  f"{row['tle_name']:<14s} {row['residual_km']:6.2f} km   "
                  f"(runner-up {row['runner_up']} at {row['runner_up_km']:.0f} km, "
                  f"{row['margin']:.0f}x)")

    out_path = args.out or os.path.join(session, "broadcast_ephemeris.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"\n[decode] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
