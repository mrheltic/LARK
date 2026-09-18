#!/usr/bin/env python3
"""
export_iq.py — Export recorded 5-channel CPIs as single-channel cf32 for gr-iridium.

The DOA pipeline never demodulates: it locks onto the preamble CW tone and throws
the payload away.  But the IRA (Ring Alert) payload carries the *satellite's own
ECEF position*, and IBC carries Iridium system time — exactly the ephemeris and
clock a self-contained PNT solution needs, broadcast in the clear.  Decoding them
requires handing the IQ to a real demodulator (gr-iridium), which is single
channel, while our recordings are (n_ant, cpi_size) complex64 per CPI.

This script bridges the two.  Three combining modes:

    --ant N      single antenna — the baseline a one-antenna receiver sees
    --sum        phase-calibrated sum of all elements
    --beamform   weights from the measured DOA (the array's full gain, up to
                 ~7 dB for 5 elements) — needs a reprocessed doa_multi.jsonl

Comparing the decode yield of the three is the point: a commercial single-antenna
receiver cannot do the third one.

Do not read ``--sum`` as "coherent combining that must beat one element".  A flat
sum of a UCA is a beam pointed at **zenith**: only a signal arriving straight
down adds in phase, and a satellite at 30-60° elevation is partly cancelled by
the array factor.  Measured on the reference session it decodes roughly 40%
*fewer* bursts than a single antenna.  It is included precisely because that is
the instructive control — array gain requires steering, not just summing.

Timing
------
CPIs are dropped whenever live processing lags, so consecutive frame_*.npy files
are NOT necessarily adjacent in time, and concatenating them compresses the time
axis.  We concatenate anyway (one extractor invocation per CPI would take hours)
and instead write an ``index.json`` recording the true epoch of every CPI in every
chunk.  A decode at sample offset s in a chunk maps back exactly:

    k        = s // cpi_size            # which CPI inside the chunk
    t_true   = epoch[k] + (s % cpi_size) / fs_hz

so absolute timing survives concatenation — and, as a bonus, this recovers the
*sub-CPI* burst time that the DOA path discards (all peaks in a CPI currently
share one timestamp).

The cost of concatenating is bounded: a discontinuity only corrupts a burst
straddling it.  A burst is 261 symbols at 25 ksps = 10.4 ms against a 64 ms CPI,
and roughly 30% of CPI boundaries have a gap, so the expected loss is ~5% of
bursts — irrelevant when IRA repeats every 90 ms.

Resampling
----------
gr-iridium refuses any sample rate not divisible by 100 kHz, and the KrakenSDR
records at 1.024 MS/s.  We resample to 1.0 MS/s, which is exact and convenient:
125/128 turns a 65536-sample CPI into exactly 64000 samples, so the index
arithmetic above stays integer.  Resampling is done per CPI, not per chunk, so
the polyphase filter never runs across a time discontinuity.

Usage (run from krakenSDR/src/):
    python3 scripts/export_iq.py ../data/doa_iridium/session_.../ --ant 0 --out /tmp/cpi
    python3 scripts/export_iq.py ../data/doa_iridium/session_.../ --beamform --out /tmp/bf
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import resample_poly

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.recording import (  # noqa: E402
    count_session_raw_frames,
    iter_session_raw_frames,
    session_frame_times,
)

_DEFAULT_CAL = os.path.join(_SRC, "apps", "doa_iridium", "cal_tle.npz")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export session CPIs as single-channel cf32_le for gr-iridium")
    p.add_argument("session_dir")
    p.add_argument("--out", required=True, help="Output directory for .cf32 chunks")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--ant", type=int, default=None, metavar="N",
                      help="Export antenna N only (default: 0)")
    mode.add_argument("--sum", action="store_true",
                      help="Phase-calibrated coherent sum of all elements")
    mode.add_argument("--beamform", action="store_true",
                      help="MRC weights from the measured DOA (needs --subdir)")
    p.add_argument("--cal-file", default=_DEFAULT_CAL,
                   help="Phase calibration npz (for --sum / --beamform)")
    p.add_argument("--no-cal", action="store_true",
                   help="Skip phase calibration (diagnostic only)")
    p.add_argument("--subdir", default="doa_multi_music",
                   help="Reprocess subdir holding doa_multi.jsonl (for --beamform)")
    p.add_argument("--chunk-cpis", type=int, default=500,
                   help="CPIs per output file (default: 500 = 32 s)")
    p.add_argument("--fs-out", type=float, default=1_000_000.0,
                   help="Output sample rate [Hz]; gr-iridium requires a multiple "
                        "of 100 kHz (default: 1e6). 0 = keep the native rate.")
    p.add_argument("--start", type=int, default=0, help="First frame index")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    return p.parse_args(argv)


def load_phase_offsets(cal_file: str, n_ant: int) -> np.ndarray:
    """Per-antenna phase offsets [deg] from a cal npz; zeros if unavailable."""
    offs = np.zeros(n_ant, dtype=float)
    if not cal_file or not os.path.isfile(cal_file):
        print(f"[cal] {cal_file!r} not found — using zeros (uncalibrated).")
        return offs
    data = np.load(cal_file)
    vals = np.asarray(data["phase_offsets_deg"], dtype=float)
    offs[: min(n_ant, vals.size)] = vals[:n_ant]
    print(f"[cal] {os.path.basename(cal_file)}: "
          + " ".join(f"{v:+.1f}" for v in offs) + " deg")
    return offs


def load_beamform_weights(session_dir: str, subdir: str) -> dict[int, np.ndarray]:
    """frame index -> unit-norm MRC weight vector, from the strongest peak.

    ``y_per_peak`` in doa_multi.jsonl is the matched-filter array response for
    that peak, already phase-calibrated (core/multi_peak.py writes it that way),
    so it *is* the steering vector estimate — no need to re-derive it from
    (az, el).  Where a CPI holds several satellites we take the highest-SNR
    peak: one output stream can only point one way.
    """
    path = os.path.join(session_dir, subdir, "doa_multi.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found — run reprocess_session.py first, or use --ant/--sum")
    weights: dict[int, np.ndarray] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            ys = rec.get("y_per_peak") or []
            snrs = rec.get("snr_per_peak") or []
            if not ys:
                continue
            best = int(np.argmax(snrs)) if len(snrs) == len(ys) else 0
            y = np.asarray(ys[best], dtype=np.float64)      # (n_ant, 2) [re, im]
            w = y[:, 0] + 1j * y[:, 1]
            norm = np.linalg.norm(w)
            if norm > 0:
                weights[int(rec["frame"])] = (w / norm).astype(np.complex128)
    print(f"[beamform] {len(weights)} CPIs with DOA weights from {subdir}")
    return weights


def resample_ratio(fs_in: float, fs_out: float) -> tuple[int, int]:
    """Exact up/down integers for fs_in -> fs_out (1.024e6 -> 1e6 gives 125/128)."""
    from math import gcd
    up, down = int(round(fs_out)), int(round(fs_in))
    g = gcd(up, down)
    return up // g, down // g


def combine(X: np.ndarray, args: argparse.Namespace, offsets_deg: np.ndarray,
            weights: dict[int, np.ndarray] | None, frame_idx: int) -> np.ndarray:
    """(n_ant, N) complex64 -> (N,) complex64, per the selected mode."""
    if args.sum or args.beamform:
        Xc = X * np.exp(-1j * np.deg2rad(offsets_deg))[:, None]
    else:
        Xc = X

    if args.beamform:
        w = weights.get(frame_idx) if weights else None
        if w is None:
            # No DOA for this CPI: fall back to the coherent sum rather than
            # dropping it, so the three modes cover the same CPIs.
            return (Xc.sum(axis=0) / np.sqrt(Xc.shape[0])).astype(np.complex64)
        return (np.conj(w) @ Xc).astype(np.complex64)
    if args.sum:
        return (Xc.sum(axis=0) / np.sqrt(Xc.shape[0])).astype(np.complex64)
    return Xc[args.ant if args.ant is not None else 0].astype(np.complex64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = args.session_dir.rstrip("/")
    meta = json.load(open(os.path.join(session, "meta.json"), encoding="utf-8"))
    fs_hz = float(meta.get("fs_hz", 1_024_000.0))
    freq_hz = float(meta.get("freq_hz", 1_626_270_000.0))
    n_ant = int(meta.get("n_ant", 5))
    cpi_size = int(meta.get("cpi_size", 65536))

    total = count_session_raw_frames(session)
    if total == 0:
        print(f"No raw/frame_*.npy in {session}", file=sys.stderr)
        return 1

    needs_cal = (args.sum or args.beamform) and not args.no_cal
    offsets_deg = (load_phase_offsets(args.cal_file, n_ant) if needs_cal
                   else np.zeros(n_ant))
    weights = load_beamform_weights(session, args.subdir) if args.beamform else None

    up, down = (resample_ratio(fs_hz, args.fs_out) if args.fs_out > 0 else (1, 1))
    fs_out = fs_hz * up / down
    cpi_out = cpi_size * up // down
    if (up, down) != (1, 1):
        if cpi_size * up % down:
            print(f"[warn] {cpi_size} samples do not resample to an integer at "
                  f"{up}/{down}; sample-offset timing will drift.")
        print(f"[resample] {fs_hz/1e6:.3f} -> {fs_out/1e6:.3f} MS/s "
              f"({up}/{down}), {cpi_size} -> {cpi_out} samples/CPI")

    frame_ts = session_frame_times(session)
    if frame_ts is None:
        print("[warn] no timestamps.csv and no frame mtimes — epochs will be NaN")

    mode = ("beamform" if args.beamform else "sum" if args.sum
            else f"ant{args.ant if args.ant is not None else 0}")
    os.makedirs(args.out, exist_ok=True)

    chunks: list[dict] = []
    buf: list[np.ndarray] = []
    epochs: list[float] = []
    frames: list[int] = []
    n_written = 0

    def flush() -> None:
        nonlocal buf, epochs, frames, n_written
        if not buf:
            return
        name = f"{mode}_{frames[0]:06d}.cf32"
        np.concatenate(buf).tofile(os.path.join(args.out, name))
        chunks.append({"file": name, "first_frame": frames[0],
                       "frames": frames, "epochs": epochs})
        n_written += 1
        buf, epochs, frames = [], [], []

    for fi, X in iter_session_raw_frames(session, start=args.start,
                                         max_frames=args.max_frames):
        y = combine(np.asarray(X), args, offsets_deg, weights, fi)
        if (up, down) != (1, 1):
            # Per CPI, never across chunks: consecutive CPIs may not be adjacent
            # in time, and a polyphase filter must not run over that seam.
            y = resample_poly(y, up, down).astype(np.complex64)
        buf.append(y)
        frames.append(int(fi))
        epochs.append(float(frame_ts[fi]) if frame_ts is not None
                      and fi < len(frame_ts) else float("nan"))
        if len(buf) >= args.chunk_cpis:
            flush()
            print(f"  ... {n_written} chunks, frame {fi}", end="\r", flush=True)
    flush()
    print()

    index = {
        "session": os.path.basename(session),
        "mode": mode,
        "fs_hz_native": fs_hz,
        "fs_hz": fs_out,            # rate of the exported .cf32 — use this one
        "freq_hz": freq_hz,
        "cpi_size": cpi_out,        # samples per CPI in the exported stream
        "cpi_size_native": cpi_size,
        "resample": [up, down],
        "phase_cal": needs_cal,
        "chunks": chunks,
    }
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)

    n_cpi = sum(len(c["frames"]) for c in chunks)
    print(f"[export] {n_cpi} CPIs -> {n_written} chunk(s) in {args.out} (mode={mode})")
    print(f"[export] extractor: iridium-extractor -f cf32_le "
          f"-r {int(fs_out)} -c {int(freq_hz)} <chunk>.cf32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
