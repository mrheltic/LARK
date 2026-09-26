#!/usr/bin/env python3
"""
add_burst_offsets.py — Add ``b0_per_peak`` to an existing doa_multi.jsonl.

A burst's epoch is its CPI's epoch plus the sample offset at which the burst
starts inside the CPI.  ``remap_epochs()`` applies that to the decoded IRA/IBC
frames, and ``reprocess_session.py`` now records it per peak, but JSONL files
written before it did carry only the CPI epoch.  This script recovers the offset
from the raw samples: it re-runs the energy detector and the tone scan on each
CPI and, for every recorded peak, keeps the burst start whose tone matches the
recorded carrier offset.

The original file is kept as ``doa_multi.jsonl.bak``.

    python3 scripts/add_burst_offsets.py <session> [--subdir doa_multi_music]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from apps.doa_iridium.run_doa import _PROFILES  # noqa: E402
from core.burst_processing import detect_energy_bursts, scan_preamble_tones  # noqa: E402

_ARGS: dict = {}


def _burst_starts(item):
    fi, cfos = item
    a = _ARGS
    X = np.load(os.path.join(a["session"], "raw", f"frame_{fi:06d}.npy"))
    if a["channel"] is not None:
        X = X[a["channel"]:a["channel"] + 1]
    if a["mix_hz"]:
        n = np.arange(X.shape[1], dtype=np.float64)
        X = X * np.exp(-2j * np.pi * (a["mix_hz"] / a["fs"]) * n).astype(np.complex64)
    prof = a["profile"]
    cand = []                                           # (b0, tone_hz) of every scanned tone
    for b0 in detect_energy_bursts(X, a["fs"], threshold_factor=prof["energy_threshold"]):
        if X.shape[1] - b0 < a["min_len"]:
            continue
        for f, snr in scan_preamble_tones(X[:, b0:b0 + a["window"]], a["fs"],
                                          nom_tone_hz=prof["tone_nom_hz"],
                                          scan_bw_hz=prof["scan_bw_hz"], n_peaks=3,
                                          min_snr_db=prof["min_snr_db"],
                                          dc_guard_hz=prof["dc_guard_hz"]):
            cand.append((b0, f - prof["tone_nom_hz"]))
    out = []
    for c in cfos:
        best = min(cand, key=lambda x: abs(x[1] - c), default=None)
        out.append(int(best[0]) if best is not None and abs(best[1] - c) < a["tol_hz"] else -1)
    return fi, out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session_dir")
    p.add_argument("--subdir", default="doa_multi_music")
    p.add_argument("--mode", default="outdoor", choices=list(_PROFILES))
    p.add_argument("--mix-hz", type=float, default=0.0,
                   help="Same --mix-hz the subdir was reprocessed with")
    p.add_argument("--channel", type=int, default=None,
                   help="Single-channel front end (tones_by_antennas-style subdirs)")
    p.add_argument("--window", type=int, default=3000)
    p.add_argument("--tol-hz", type=float, default=50.0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    session = os.path.abspath(args.session_dir.rstrip("/"))
    path = os.path.join(session, args.subdir, "doa_multi.jsonl")
    with open(os.path.join(session, "meta.json"), encoding="utf-8") as fh:
        fs = float(json.load(fh).get("fs_hz", 1_024_000.0))
    _ARGS.update(session=session, fs=fs, profile=_PROFILES[args.mode], mix_hz=args.mix_hz,
                 channel=args.channel, window=args.window, min_len=2800, tol_hz=args.tol_hz)

    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    items = [(int(r["frame"]), list(r["cfo_per_peak"])) for r in rows]
    with Pool(args.workers, initializer=_ARGS.update, initargs=(dict(_ARGS),)) as pool:
        found = dict(pool.map(_burst_starts, items, chunksize=16))

    n_pk = n_ok = 0
    for r in rows:
        r["b0_per_peak"] = found[int(r["frame"])]
        n_pk += len(r["b0_per_peak"])
        n_ok += sum(b >= 0 for b in r["b0_per_peak"])
    if not os.path.isfile(path + ".bak"):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    b = np.array([x for r in rows for x in r["b0_per_peak"] if x >= 0])
    print(f"[b0] {args.subdir}: {n_ok}/{n_pk} peaks matched; burst start median "
          f"{np.median(b) / fs * 1e3:.1f} ms, mean {b.mean() / fs * 1e3:.1f} ms into the CPI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
