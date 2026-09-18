#!/usr/bin/env python3
"""
decode_yield.py — Does the array help *decode*, not just point?

A single-antenna receiver (the commercial Iridium-timing/location products are
one) sees whatever arrives at one element.  LARK has five, plus a direction
estimate for each burst, so it can steer.  This measures whether that translates
into more decoded frames on exactly the same recording, comparing the streams
produced by ``scripts/export_iq.py``:

    --ant N      one element: the single-antenna baseline
    --sum        flat phase-calibrated sum — i.e. a beam pointed at zenith
    --beamform   MRC weights from the measured DOA (full array gain)

Measured on the reference session, over the CPIs that carry weights: beamforming
decodes **+31%** more IRA frames than a single element, and every satellite
gains. A flat sum decodes **31% fewer** — it is not "coherent combining", it is a
zenith beam, and a satellite at 30-60° elevation is partly cancelled by the array
factor. Array gain requires steering, not summing.

Fair comparison needs care on two points, both handled here:

* **Same CPIs.**  Every mode exports the same frames, and each decoded frame is
  mapped back to its CPI through ``index.json`` (sample offset // cpi_size), so
  the comparison is per-CPI rather than per-file.
* **Beamforming only applies where a DOA exists.**  Only 1807 of 23595 CPIs have
  an accepted burst and therefore weights; elsewhere ``export_iq.py`` falls back
  to the sum, which is *worse* than one element. Mixing those in inverts the
  conclusion — over all CPIs beamforming scores −9%, over the CPIs it actually
  applies to, +31%. ``--doa-only`` selects the honest scope.

Steering at the strongest peak might have been expected to suppress other
satellites sharing a CPI. It does not: on this session all four gain
(+20% to +46%), so the extra SNR outweighs the spatial selectivity.

Usage (run from krakenSDR/src/):
    python3 scripts/decode_yield.py <session> \
        --mode ant0:/tmp/m_ant0 --mode sum:/tmp/m_sum --mode beamform:/tmp/m_bf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.broadcast_ephemeris import (  # noqa: E402
    filter_plausible,
    load_export_index,
    parse_decodes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare decode yield across combining modes")
    p.add_argument("session_dir")
    p.add_argument("--mode", action="append", required=True, metavar="NAME:DIR",
                   help="Combining mode and its export dir (repeatable). The "
                        "parsed file is expected at <DIR>.parsed")
    p.add_argument("--subdir", default="doa_multi_music",
                   help="Reprocess subdir, used to find which CPIs have a DOA")
    p.add_argument("--doa-only", action="store_true",
                   help="Restrict to CPIs that carry beamforming weights")
    p.add_argument("--json-out", default="")
    return p.parse_args(argv)


def cpi_of(chunk_map: dict, chunk: str, t_chunk_s: float, fs: float, cpi: int):
    """Which recorded CPI a decoded frame came from, or None."""
    entry = chunk_map.get(chunk)
    if entry is None:
        return None
    k = int(t_chunk_s * fs // cpi)
    frames = entry["frames"]
    return int(frames[k]) if 0 <= k < len(frames) else None


def load_mode(export_dir: str, parsed_path: str):
    index = load_export_index(os.path.join(export_dir, "index.json"))
    fs, cpi = float(index["fs_hz"]), int(index["cpi_size"])
    chunk_map = {os.path.splitext(c["file"])[0]: c for c in index["chunks"]}

    fixes, times = parse_decodes(parsed_path)
    plausible = {id(f) for f in filter_plausible(fixes)}

    per_cpi_ira = Counter()
    per_cpi_good = Counter()
    # Keyed by (cpi, sat) so the per-satellite table can honour the same scope
    # filter as the totals — otherwise it silently reports over all CPIs.
    sats = Counter()
    for f in fixes:
        c = cpi_of(chunk_map, f.chunk, f.t, fs, cpi)
        if c is None:
            continue
        per_cpi_ira[c] += 1
        if id(f) in plausible:
            per_cpi_good[c] += 1
            sats[(c, f.sat_id)] += 1
    per_cpi_ibc = Counter()
    for tm in times:
        c = cpi_of(chunk_map, tm.chunk, tm.t_rx, fs, cpi)
        if c is not None:
            per_cpi_ibc[c] += 1

    n_cpi = sum(len(c["frames"]) for c in index["chunks"])
    return {"ira": per_cpi_ira, "good": per_cpi_good, "ibc": per_cpi_ibc,
            "sats": sats, "n_ira": len(fixes), "n_ibc": len(times),
            "n_cpi": n_cpi}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = args.session_dir.rstrip("/")

    doa_cpis = set()
    jsonl = os.path.join(session, args.subdir, "doa_multi.jsonl")
    if os.path.isfile(jsonl):
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                doa_cpis.add(int(json.loads(line)["frame"]))
    print(f"[yield] {len(doa_cpis)} CPIs carry a DOA estimate (and thus weights)")

    modes = {}
    for spec in args.mode:
        name, _, d = spec.partition(":")
        modes[name] = load_mode(d, d.rstrip("/") + ".parsed")

    sel = doa_cpis if args.doa_only else None
    scope = "CPIs with a DOA" if args.doa_only else "all CPIs"
    print(f"\n[yield] scope: {scope}")
    print(f"  {'mode':<12s} {'IRA':>7s} {'IRA ok pos':>11s} {'IBC':>6s} "
          f"{'CPIs hit':>9s} {'IRA per 100 CPI':>16s}")

    rows = []
    for name, m in modes.items():
        def total(counter):
            return sum(v for k, v in counter.items() if sel is None or k in sel)
        hit = len({k for k in m["ira"] if sel is None or k in sel})
        n_ira, n_good, n_ibc = total(m["ira"]), total(m["good"]), total(m["ibc"])
        # Rate over every CPI in scope, not just the ones that decoded — otherwise
        # the ratio is trivially 1 and says nothing about yield.
        denom = len(sel) if sel is not None else m["n_cpi"]
        print(f"  {name:<12s} {n_ira:7d} {n_good:11d} {n_ibc:6d} {hit:9d} "
              f"{100.0 * n_ira / max(denom, 1):16.2f}")
        m["sats_in_scope"] = Counter()
        for (c, s), v in m["sats"].items():
            if sel is None or c in sel:
                m["sats_in_scope"][s] += v
        rows.append({"mode": name, "ira": n_ira, "ira_good_pos": n_good,
                     "ibc": n_ibc, "cpis_hit": hit,
                     "sats": {str(k): v
                              for k, v in m["sats_in_scope"].most_common()}})

    base = rows[0]["mode"]
    print(f"\n[yield] relative to '{base}':")
    for r in rows[1:]:
        for key, label in (("ira", "IRA"), ("ira_good_pos", "IRA w/ good pos"),
                           ("ibc", "IBC")):
            b = rows[0][key]
            gain = 100.0 * (r[key] - b) / max(b, 1)
            print(f"  {r['mode']:<12s} {label:<16s} {r[key]:6d} vs {b:6d}  "
                  f"{gain:+6.1f}%")

    print(f"\n[yield] plausible IRA positions per satellite ({scope}):")
    all_sats = sorted({s for m in modes.values() for s in m["sats_in_scope"]})
    print("  " + "sat".ljust(9) + "".join(f"{n:>12s}" for n in modes))
    for s in all_sats:
        print(f"  sat:{s:03d}  " + "".join(
            f"{modes[n]['sats_in_scope'].get(s, 0):>12d}" for n in modes))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"session": os.path.basename(session), "scope": scope,
                       "n_doa_cpis": len(doa_cpis), "modes": rows}, fh, indent=1)
        print(f"\n[yield] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
