#!/usr/bin/env python3
"""Predict Iridium NEXT passes. Usage: python3 predict_passes.py --hours 12 --min-el 15"""
import argparse, sys
from datetime import datetime, timedelta, timezone
try:
    from skyfield.api import load, wgs84
except ImportError:
    print("Install: pip3 install skyfield"); sys.exit(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=12)
    p.add_argument("--lat", type=float, default=45.0)
    p.add_argument("--lon", type=float, default=9.0)
    p.add_argument("--min-el", type=float, default=15.0)
    args = p.parse_args()

    print("Fetching Iridium NEXT TLE...")
    sat_list = load.tle_file(
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle",
        reload=True,
    )
    print(f"Loaded {len(sat_list)} satellites")

    ts = load.timescale()
    obs = wgs84.latlon(args.lat, args.lon)
    t0 = ts.now()
    dt = timedelta(hours=args.hours)
    t1 = ts.utc(
        (datetime.now(timezone.utc) + dt).year,
        (datetime.now(timezone.utc) + dt).month,
        (datetime.now(timezone.utc) + dt).day,
        (datetime.now(timezone.utc) + dt).hour,
        (datetime.now(timezone.utc) + dt).minute,
    )

    print(f"Window: {t0.utc_strftime('%H:%M')} → {t1.utc_strftime('%H:%M')} UTC  "
          f"loc={args.lat:.1f}°N {args.lon:.1f}°E  min_el={args.min_el}°\n")

    passes = []
    for sat in sat_list:
        try:
            t_ev, ev = sat.find_events(obs, t0, t1, altitude_degrees=args.min_el)
        except Exception:
            continue
        if len(t_ev) == 0:
            continue

        # Group events into passes: rise(0) → culminate(1) → set(2)
        rises = []; culminates = []; sets = []
        for tj, ej in zip(t_ev, ev):
            if ej == 0: rises.append(tj)
            elif ej == 1: culminates.append(tj)
            elif ej == 2: sets.append(tj)

        # Pair each rise with the next set
        for ri, r_t in enumerate(rises):
            # Find the culminate between this rise and the next set
            if ri < len(sets):
                s_t = sets[ri]
                # Find max elevation at culminate
                max_el = 0; max_t = r_t
                for c_t in culminates:
                    if r_t.tt < c_t.tt < s_t.tt:
                        diff = sat - obs
                        topo = diff.at(c_t)
                        el_deg, az_deg, _ = topo.altaz()
                        e = float(el_deg.degrees)
                        if e > max_el:
                            max_el = e; max_t = c_t

                dur = (s_t.tt - r_t.tt) * 86400
                passes.append({
                    "name": sat.name.strip(),
                    "id": sat.model.satnum,
                    "rise": r_t.utc_strftime('%H:%M'),
                    "max": max_t.utc_strftime('%H:%M'),
                    "set": s_t.utc_strftime('%H:%M'),
                    "el": max_el,
                    "dur": dur,
                })

    passes.sort(key=lambda x: x["rise"])

    print(f"{'Satellite':<25} {'ID':>6} {'Rise':>6} {'Max':>6} {'Set':>6} {'El':>5} {'Dur':>6}")
    print("-" * 68)
    for p in passes[:40]:
        m = " ★" if p["el"] > 60 else (" ↑" if p["el"] > 30 else "  ")
        print(f"{p['name']:<25} {p['id']:>6} {p['rise']:>6} {p['max']:>6} "
              f"{p['set']:>6} {p['el']:>4.0f}° {p['dur']:>5.0f}s{m}")

    high = [p for p in passes if p["el"] > 50]
    if high:
        print(f"\n★★★ Best passes (el > 50°): {len(high)}")
        for p in high[:10]:
            print(f"  {p['name']:<25} {p['rise']}→{p['set']} max {p['el']:.0f}° @{p['max']}")

    print(f"\nTotal: {len(passes)} passes")

if __name__ == "__main__":
    main()
