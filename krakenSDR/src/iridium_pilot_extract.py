#!/usr/bin/env python3
"""
iridium_pilot_extract.py — Iridium pilot-tone extractor (text output)
======================================================================

Scans a baseband IQ recording, demodulates every detected Iridium burst
and writes a structured text report focused on the **pilot tone** (the
16-symbol preamble that precedes the Unique Word).

The pilot tone is the only part of an Iridium burst that is:
  - Fully known a-priori (16 × S₀ = +1+1j at 45°)
  - Phase-coherent → directly usable as DoA spatial snapshot
  - Free of data uncertainty → clean SNR/phase measurement

For each burst the report contains:
  1. RF detection metadata  (time, freq, SNR, Doppler)
  2. Pilot tone analysis    (amplitude, phase, freq residual, pilot SNR)
  3. Protocol structure     (preamble / UW / data sample offsets)
  4. Protocol decode        (IRA / IBC / IRI / IMS via iridium-parser.py)
  5. Per-symbol pilot table (phase and amplitude of each of the 16 symbols)
  6. Payload hex dump

By default only Ring-Alert related bursts are analysed in detail
(types IRA  = ring-alert with satellite position + paging,
        IBC  = broadcast channel / paging,
        ALL  = all decoded bursts when no ring-alert found).

Usage
-----
    python3 iridium_pilot_extract.py                      # auto-detect WAV
    python3 iridium_pilot_extract.py path/to/rec.wav
    python3 iridium_pilot_extract.py --all                # report all bursts
    python3 iridium_pilot_extract.py --out my_report.txt
    python3 iridium_pilot_extract.py --limit 300          # scan first 300 s
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from hardware.file_iq_source import FileIQSource
from core.burst_pipeline import BurstPipeline
from core.iridium_demod import (
    IridiumDemod, DemodDebug, DOWNLINK, UPLINK,
    UW_LENGTH, UW_DOWNLINK, UW_UPLINK, PREAMBLE_LENGTH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SPS          = 40        # samples per symbol @ 1 Msps
_SYM_RATE     = 25_000    # symbols / second
_PILOT_REF    = 45.0      # expected phase of S₀ in degrees
_PILOT_NSYMS  = 16        # preamble length in symbols
_RING_TYPES   = {"IRA", "IBC"}   # parser prefixes considered ring-alert

# Find iridium-parser relative to project root
_LARK_ROOT  = Path(_HERE).parent.parent
_PARSER_EXE = str(_LARK_ROOT / "external" / "iridium-toolkit" / "iridium-parser.py")
_VENV_PYTHON = str(_LARK_ROOT / ".venv" / "bin" / "python3")
if not Path(_VENV_PYTHON).exists():
    _VENV_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Pilot tone analysis
# ---------------------------------------------------------------------------

def _analyse_pilot(d: DemodDebug) -> dict:
    """
    Compute pilot-tone (preamble) metrics from a DemodDebug object.

    Returns a dict with keys:
        valid        : bool — pilot samples exist
        amp_mean     : float — mean |IQ| of 16 symbol decisions
        amp_std      : float
        amp_cv_pct   : float — coefficient of variation (%)
        phase_mean   : float — mean phase (degrees)
        phase_std    : float — phase noise (degrees)
        phase_offset : float — phase_mean − 45°  (ideal is 0)
        freq_resid_hz: float — residual carrier freq offset (Hz)
        pilot_snr_db : float — pilot SNR estimate (dB)
        sym_phases   : list[float]  — per-symbol phase (deg)
        sym_amps     : list[float]  — per-symbol amplitude
        pream_start  : int   — sample index of preamble start in iq_1m
        uw_start     : int   — sync_start (sample index)
    """
    sps         = d.sps
    sync_start  = d.sync_start
    pream_start = max(0, sync_start - _PILOT_NSYMS * sps)
    sig         = d.iq_1m

    # Collect 16 symbol-decision samples from preamble
    sym_phases, sym_amps = [], []
    for k in range(_PILOT_NSYMS):
        idx = pream_start + k * sps
        if idx >= len(sig):
            break
        s = sig[idx]
        sym_amps.append(float(abs(s)))
        sym_phases.append(float(math.degrees(math.atan2(s.imag, s.real))) % 360.0)

    if not sym_amps:
        return {"valid": False}

    amp_arr   = np.array(sym_amps)
    phase_arr = np.array(sym_phases)

    # Unwrap phases for slope (freq offset) estimation
    ph_rad    = np.unwrap([math.radians(p) for p in sym_phases])
    if len(ph_rad) >= 2:
        slope_per_sym = float(np.polyfit(range(len(ph_rad)), ph_rad, 1)[0])
        freq_resid_hz = slope_per_sym * _SYM_RATE / (2.0 * math.pi)
    else:
        freq_resid_hz = 0.0

    amp_mean = float(np.mean(amp_arr))
    amp_std  = float(np.std(amp_arr))

    # Wrap-aware phase mean and std
    ph_wrapped  = ph_rad - np.round(ph_rad / (2.0 * math.pi)) * 2.0 * math.pi
    phase_mean  = float(math.degrees(np.mean(ph_wrapped)))
    phase_std   = float(math.degrees(np.std(ph_wrapped)))
    phase_offset = phase_mean - _PILOT_REF

    # SNR: signal = amp_mean², noise estimated from amplitude fluctuation
    noise_var = amp_std ** 2 + 1e-30
    sig_var   = amp_mean ** 2
    pilot_snr = 10.0 * math.log10(sig_var / noise_var) if noise_var > 1e-30 else 0.0

    return {
        "valid":         True,
        "amp_mean":      amp_mean,
        "amp_std":       amp_std,
        "amp_cv_pct":    100.0 * amp_std / (amp_mean + 1e-30),
        "phase_mean":    phase_mean,
        "phase_std":     phase_std,
        "phase_offset":  phase_offset,
        "freq_resid_hz": freq_resid_hz,
        "pilot_snr_db":  pilot_snr,
        "sym_phases":    sym_phases,
        "sym_amps":      sym_amps,
        "pream_start":   pream_start,
        "uw_start":      sync_start,
    }


# ---------------------------------------------------------------------------
# iridium-parser integration
# ---------------------------------------------------------------------------

def _parse_raw_lines(raw_lines: List[str]) -> dict[str, str]:
    """
    Pass RAW: lines to iridium-parser.py and return a dict
    mapping timestamp_int (int ms) → parser output line.
    """
    if not raw_lines:
        return {}

    mapping: dict[int, str] = {}
    try:
        proc = subprocess.run(
            [_VENV_PYTHON, _PARSER_EXE, "--uw-ec", "--harder"],
            input="\n".join(raw_lines) + "\n",
            capture_output=True, text=True, timeout=120,
        )
        for line in proc.stdout.splitlines():
            parts = line.split()
            # Both RAW: and decoded lines have timestamp at field index 2
            if len(parts) >= 3:
                try:
                    ts_key = int(float(parts[2]))
                    mapping[ts_key] = line.strip()
                except ValueError:
                    pass
    except Exception as e:
        print(f"  [WARNING] iridium-parser failed: {e}", file=sys.stderr)
    return mapping


def _classify_parser_line(line: Optional[str]) -> str:
    """Return message type prefix: IRA / IBC / IRI / IMS / ERR / ---."""
    if line is None:
        return "---"
    for prefix in ("IRA:", "IBC:", "IRI:", "IMS:", "STL:", "RAW:"):
        if line.startswith(prefix):
            return prefix.rstrip(":")
    return "ERR"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan(
    wav_path: Path,
    snr:      float = 12.0,
    papr:     float = 8.0,
    pwr:      float = -95.0,
    limit_s:  float = float("inf"),
    frame_sz: int   = 131_072,
) -> List[dict]:
    """Scan recording and return list of burst dicts with DemodDebug."""
    src = FileIQSource(wav_path, frame_size=frame_sz)
    src.start()
    fs = src.sample_rate

    demod = IridiumDemod(input_fs=int(fs))
    pipe  = BurstPipeline(
        input_fs       = int(fs),
        center_freq_hz = 1_626_270_000.0,
        burst_snr      = snr,
        burst_papr     = papr,
        burst_pwr      = pwr,
        demod_enabled  = False,
        filename       = wav_path.stem,
    )
    T0     = pipe._t0
    bursts = []
    total  = src.total_samples
    done   = 0

    print(f"[PILOT] Scanning {wav_path.name}  "
          f"({src.duration_s:.0f} s @ {fs/1e6:.3f} MS/s) …")

    while True:
        frame = src.get_frame(timeout=0)
        if frame is None:
            break
        t_s = src.elapsed_s - frame_sz / fs
        if t_s > limit_s:
            break
        x    = frame[0].astype(np.complex128)
        pr   = pipe.process(x, timestamp=T0 + t_s)
        done += frame_sz

        pct = min(done / total, 1.0)
        bar = int(pct * 40)
        print(f"\r  [{'#'*bar:<40}] {pct*100:5.1f}%  {len(bursts)} bursts",
              end="", flush=True)

        if not pr.burst.is_burst:
            continue

        dbg = demod.demod_full(
            x              = x,
            doppler_hz     = pr.burst.doppler_hz,
            timestamp_ms   = t_s * 1000.0,
            center_freq_hz = 1_626_270_000.0,
            filename       = wav_path.stem,
            snr_db         = pr.burst.burst_snr_db,
        )
        bursts.append({
            "t_s":      t_s,
            "snr_db":   pr.burst.burst_snr_db,
            "dop_hz":   pr.burst.doppler_hz,
            "debug":    dbg,
        })

    src.stop()
    n_ok  = sum(1 for b in bursts if b["debug"] and b["debug"].access_ok)
    print(f"\n[PILOT] Done — {len(bursts)} bursts  {n_ok} A:OK  "
          f"({100*n_ok//max(len(bursts),1)}%)")
    return bursts


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_W = 72   # line width


def _rule(char="═"):
    return char * _W


def _header_block(wav: Path, bursts: List[dict], args) -> str:
    n_ok    = sum(1 for b in bursts if b["debug"] and b["debug"].access_ok)
    n_demod = sum(1 for b in bursts if b["debug"])
    lines = [
        _rule("═"),
        "  LARK — Iridium Pilot Tone Extractor",
        f"  Recording  : {wav.name}",
        f"  File path  : {wav}",
        f"  Run date   : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        _rule("─"),
        f"  Bursts detected : {len(bursts)}",
        f"  Demodulated     : {n_demod}  ({100*n_demod//max(len(bursts),1)}%)",
        f"  A:OK (UW match) : {n_ok}  ({100*n_ok//max(n_demod,1)}% of demod)",
        f"  Detection SNR   : ≥ {args.snr} dB  PAPR ≥ {args.papr} dB",
        f"  Filter mode     : {'ALL bursts' if args.all else 'Ring-Alert only (IRA / IBC)'}",
        _rule("─"),
        "",
        "  PILOT TONE EXPLAINED",
        "  ─────────────────────",
        "  Each Iridium burst begins with a 16-symbol preamble (the 'pilot')",
        "  constructed from S₀ = +1+1j symbols at ideal phase 45°.",
        "  After carrier-phase correction the preamble should be a tight",
        "  cluster near 45° in the IQ plane. Deviations indicate:",
        "    • Phase offset   → residual carrier phase error",
        "    • Phase noise    → SNR / modulation quality",
        "    • Phase slope    → residual Doppler / freq offset",
        "  For DoA: the pilot IQ snapshot (16 complex samples per antenna)",
        "  is the ideal input to the MUSIC / ESPRIT spatial spectrum.",
        _rule("═"),
        "",
    ]
    return "\n".join(lines)


def _burst_report(idx: int, b: dict, parser_line: Optional[str]) -> str:
    d        = b["debug"]
    msg_type = _classify_parser_line(parser_line)
    t_s      = b["t_s"]
    snr      = b["snr_db"]
    dop_khz  = b["dop_hz"] / 1e3
    freq_mhz = (1_626_270_000.0 + b["dop_hz"]) / 1e6

    ok_str   = "A:OK ✓" if (d and d.access_ok) else "A:no ✗"
    dir_str  = ("DL↓" if d.direction == DOWNLINK else "UL↑") if d else "—"

    lines = [
        "",
        _rule("═"),
        f"  BURST #{idx:03d}   type={msg_type:4s}   {ok_str}   dir={dir_str}",
        _rule("─"),
        f"  Time        : {t_s:>10.3f} s  from recording start",
        f"  Frequency   : {freq_mhz:.3f} MHz  (base 1626.270 MHz)",
        f"  Doppler Δf  : {dop_khz:+.2f} kHz",
        f"  Burst SNR   : {snr:.1f} dB",
    ]

    if d is None:
        lines += [
            "",
            "  [demodulation failed — burst too short or sync not found]",
        ]
        lines.append(_rule("─"))
        return "\n".join(lines)

    # ── Pilot tone analysis ─────────────────────────────────────────────────
    pt = _analyse_pilot(d)

    lines += ["", "  ── PILOT TONE ANALYSIS  (16-symbol preamble) ──"]
    if pt["valid"]:
        stable = "YES ✓" if pt["amp_cv_pct"] < 15 else "NO  ✗"
        locked = "YES ✓" if abs(pt["phase_offset"]) < 10 else f"NO ({pt['phase_offset']:+.1f}°)"
        lines += [
            f"  Amplitude  mean : {pt['amp_mean']:.4f}   std : {pt['amp_std']:.4f}"
            f"   CV={pt['amp_cv_pct']:.1f}%   stable={stable}",
            f"  Phase      mean : {pt['phase_mean']:+7.2f}°   std : {pt['phase_std']:.2f}°"
            f"   (ideal 45.00°)",
            f"  Phase offset    : {pt['phase_offset']:+.2f}°   carrier lock={locked}",
            f"  Freq residual   : {pt['freq_resid_hz']:+.1f} Hz  (after Doppler correction)",
            f"  Pilot SNR est.  : {pt['pilot_snr_db']:.1f} dB",
        ]
        lines += ["", "  Per-symbol pilot table  (S₀ expected at 45°, amplitude ~const)"]
        lines += ["  Sym   Phase     Amp    Δphase    OK?"]
        lines += ["  ───   ──────   ──────  ──────   ─────"]
        for k, (ph, am) in enumerate(zip(pt["sym_phases"], pt["sym_amps"])):
            delta = ph - _PILOT_REF
            ok    = "✓" if abs(delta) < 22.5 else "✗"
            lines.append(
                f"  #{k:02d}   {ph:+7.2f}°  {am:.4f}  {delta:+7.2f}°   {ok}"
            )
    else:
        lines.append("  [pilot analysis unavailable — preamble out of buffer range]")

    # ── RF Frame structure ──────────────────────────────────────────────────
    lines += ["", "  ── RF FRAME STRUCTURE  (@ 1 Msps, 40 sps/sym) ──"]
    if pt["valid"]:
        ps  = pt["pream_start"]
        uws = pt["uw_start"]
        ds  = uws + UW_LENGTH * _SPS
        lines += [
            f"  Preamble start  : sample {ps:>6d}   t={ps/1e6*1e3:>7.3f} ms",
            f"  UW start        : sample {uws:>6d}   t={uws/1e6*1e3:>7.3f} ms",
            f"  Data start      : sample {ds:>6d}   t={ds/1e6*1e3:>7.3f} ms",
            f"  Num symbols     : {d.nsymbols}",
            f"  Data bits       : {len(d.dataarray)} bits  ({len(d.dataarray)//8} bytes)",
        ]
    else:
        lines += [
            f"  UW start        : sample {d.sync_start}",
            f"  Num symbols     : {d.nsymbols}",
        ]

    # ── Unique Word check ───────────────────────────────────────────────────
    lines += ["", "  ── UNIQUE WORD (UW) CHECK ──"]
    uw_got = "".join(str(s) for s in d.symbols[:UW_LENGTH])
    uw_exp = UW_DOWNLINK if d.direction == DOWNLINK else UW_UPLINK
    uw_ok_str = "✓ MATCH" if d.access_ok else "✗ MISMATCH"
    lines += [
        f"  Direction : {'DOWNLINK ↓' if d.direction == DOWNLINK else 'UPLINK ↑'}",
        f"  Got       : {uw_got}",
        f"  Expected  : {uw_exp}",
        f"  Status    : {uw_ok_str}  ({UW_LENGTH * 2} bits, {UW_LENGTH} QPSK symbols)",
        f"  Confidence: {d.confidence:.0f}%",
    ]

    # ── Protocol decode ─────────────────────────────────────────────────────
    lines += ["", "  ── PROTOCOL DECODE (iridium-parser.py) ──"]
    if parser_line:
        wrapped = textwrap.fill(
            parser_line, width=_W - 4,
            initial_indent="  ", subsequent_indent="      ",
        )
        lines.append(wrapped)
    else:
        lines.append(f"  [no decode available — A:no or parser error]")

    # ── Payload hex dump ────────────────────────────────────────────────────
    bits    = d.dataarray[UW_LENGTH * 2:]   # skip UW bits
    n_bytes = len(bits) // 8
    if n_bytes > 0:
        lines += ["", "  ── PAYLOAD HEX DUMP  (after UW, MSB first) ──"]
        byte_row = []
        for i in range(n_bytes):
            byte_val = 0
            for b_idx in range(8):
                pos = i * 8 + b_idx
                if pos < len(bits):
                    byte_val = (byte_val << 1) | bits[pos]
            byte_row.append(f"{byte_val:02X}")
            if len(byte_row) == 16:
                offset = (i // 16) * 16
                lines.append(f"  0x{offset:04X}: {' '.join(byte_row)}")
                byte_row = []
        if byte_row:
            offset = (n_bytes // 16) * 16
            lines.append(f"  0x{offset:04X}: {' '.join(byte_row)}")

    # ── Raw bit string ───────────────────────────────────────────────────────
    if d.raw_line:
        lines += ["", "  ── RAW: LINE ──"]
        wrapped = textwrap.fill(
            d.raw_line, width=_W - 4,
            initial_indent="  ", subsequent_indent="  ",
        )
        lines.append(wrapped)

    lines.append(_rule("─"))
    return "\n".join(lines)


def _summary_table(bursts: List[dict], parser_map: dict[str, str]) -> str:
    lines = [
        "",
        _rule("═"),
        "  SUMMARY TABLE",
        _rule("─"),
        "   #   Time(s)   Freq(MHz)   SNR(dB)  Dop(kHz) Dir  A:OK  Type   PilotPhase°  PilotCV%",
        "  ──  ────────  ──────────  ───────  ────────  ───  ────  ─────  ──────────  ────────",
    ]
    for idx, b in enumerate(bursts, 1):
        d      = b["debug"]
        t_s    = b["t_s"]
        freq   = (1_626_270_000.0 + b["dop_hz"]) / 1e6
        dop    = b["dop_hz"] / 1e3
        snr    = b["snr_db"]
        acc    = "OK" if (d and d.access_ok) else "no"
        dir_s  = ("DL" if d.direction == DOWNLINK else "UL") if d else "--"

        # find parser type
        ptype = "---"
        if d and d.raw_line:
            try:
                ts_key = int(float(d.raw_line.split()[2]))
                if ts_key in parser_map:
                    ptype = _classify_parser_line(parser_map[ts_key])
            except (ValueError, IndexError):
                pass

        # pilot metrics
        if d:
            pt      = _analyse_pilot(d)
            ph_str  = f"{pt['phase_mean']:+7.2f}°" if pt["valid"] else "  ---   "
            cv_str  = f"{pt['amp_cv_pct']:5.1f}%"  if pt["valid"] else " ---  "
        else:
            ph_str, cv_str = "  ---   ", " ---  "

        lines.append(
            f"  {idx:3d}  {t_s:8.3f}  {freq:10.3f}  {snr:7.1f}"
            f"  {dop:+8.2f}  {dir_s:3s}   {acc:2s}   {ptype:5s}  {ph_str}  {cv_str}"
        )
    lines += [_rule("═"), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def _auto_wav() -> Path:
    candidates = [
        Path(_HERE).parent.parent / "krakenSDR" / "recordings",
        Path(_HERE).parent / "recordings",
        Path(_HERE) / "recordings",
    ]
    for d in candidates:
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            return wavs[0]
    raise FileNotFoundError(
        "No WAV file found. Pass the path as a positional argument."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file",    nargs="?", default=None, metavar="FILE")
    ap.add_argument("--snr",   type=float, default=12.0)
    ap.add_argument("--papr",  type=float, default=8.0)
    ap.add_argument("--limit", type=float, default=float("inf"), metavar="S",
                    help="Scan only first N seconds")
    ap.add_argument("--out",   type=str,   default=None, metavar="FILE",
                    help="Output text file (default: <wav>.pilot_report.txt)")
    ap.add_argument("--all",   action="store_true",
                    help="Report ALL bursts (default: ring-alert only)")
    args = ap.parse_args()

    wav = Path(args.file) if args.file else _auto_wav()
    if not wav.exists():
        sys.exit(f"[ERROR] File not found: {wav}")

    out_path = Path(args.out) if args.out else wav.with_suffix(".pilot_report.txt")

    # ── Scan ──────────────────────────────────────────────────────────────
    bursts = scan(wav, snr=args.snr, papr=args.papr, limit_s=args.limit)
    if not bursts:
        sys.exit("[PILOT] No bursts detected.")

    # ── Run parser ────────────────────────────────────────────────────────
    print("[PILOT] Running iridium-parser.py …")
    raw_lines = [
        b["debug"].raw_line
        for b in bursts
        if b["debug"] and b["debug"].raw_line
    ]
    parser_map = _parse_raw_lines(raw_lines)

    # ── Decide which bursts to report ─────────────────────────────────────
    def _burst_parser_type(b: dict) -> str:
        d = b["debug"]
        if d and d.raw_line:
            try:
                ts_key = int(float(d.raw_line.split()[2]))
                pl = parser_map.get(ts_key)
                return _classify_parser_line(pl)
            except (ValueError, IndexError):
                pass
        return "---"

    if args.all:
        report_bursts = list(enumerate(bursts, 1))
    else:
        # Keep only ring-alert types; fall back to all A:OK if none found
        ra_list = [
            (i, b) for i, b in enumerate(bursts, 1)
            if _burst_parser_type(b) in _RING_TYPES
        ]
        if ra_list:
            report_bursts = ra_list
            print(f"[PILOT] Found {len(ra_list)} ring-alert bursts (IRA/IBC)")
        else:
            # No IRA/IBC decoded → report all A:OK bursts
            report_bursts = [
                (i, b) for i, b in enumerate(bursts, 1)
                if b["debug"] and b["debug"].access_ok
            ]
            print(f"[PILOT] No ring-alert decoded; reporting {len(report_bursts)} A:OK bursts")
            if not report_bursts:
                report_bursts = list(enumerate(bursts, 1))
                print("[PILOT] No A:OK bursts either; reporting all bursts")

    # ── Assemble report ───────────────────────────────────────────────────
    print(f"[PILOT] Writing report ({len(report_bursts)} bursts) → {out_path}")

    report_parts = [_header_block(wav, bursts, args)]
    report_parts.append(_summary_table(bursts, parser_map))

    for idx, b in report_bursts:
        d = b["debug"]
        parser_line = None
        if d and d.raw_line:
            try:
                ts_key = int(float(d.raw_line.split()[2]))
                parser_line = parser_map.get(ts_key)
            except (ValueError, IndexError):
                pass
        report_parts.append(_burst_report(idx, b, parser_line))

    full_report = "\n".join(report_parts) + "\n"

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(full_report)

    print(f"[PILOT] Done.  Report saved to:  {out_path}")

    # Also print summary to terminal
    print()
    print(_summary_table(bursts, parser_map))


if __name__ == "__main__":
    main()
