#!/usr/bin/env python3
"""
tx/indoor_1626.py — Indoor Iridium-band test transmitter at 1626.270 MHz.

======================================================================
  IMPORTANT LEGAL AND SAFETY NOTICE
======================================================================
Transmitting in the Iridium L-band (1616–1626.5 MHz) without a licence
is ILLEGAL in Italy (and most countries).
Use a wired connection (TX → attenuator ≥ 30 dB → RX) ONLY.

Modes
-----
  ira   Static IRA bursts (CFO ≈ 0 Hz, no Doppler chirp).
        Use with DoA runner DOPPLER_GATE_HZ = 3000 in config.py.

  pass  Simulated LEO satellite pass (Doppler chirp ±28–40 kHz).
        Use with DoA runner DOPPLER_GATE_HZ = 0 in config.py.

Usage
-----
    python3 tx/indoor_1626.py --dry-run
    python3 tx/indoor_1626.py --mode pass --cyclic
    python3 tx/indoor_1626.py --mode pass --gain -50 --elev 45 --dur 90 --cyclic
    python3 tx/indoor_1626.py --mode ira --gain -50 --cyclic
    python3 tx/indoor_1626.py --mode ira -n 1
    python3 tx/indoor_1626.py --mode ira --gain -40 -n 8

Satellite emulation link budget — calibrated from outdoor field measurement
---------------------------------------------------------------------------
  MEASURED (outdoor, 27-05-2026, single antenna, 800 kHz BW):
    Burst SNR in band  : median 9.2 dB / mean 9.9 dB  (range 6–15 dB)

  INDOOR SIMULATION (−60 dB TX gain, d ≈ 1 m):
    Expected burst SNR : ≈ 11 dB  (matches outdoor measurement)

  Gain guide:
    −60 dB → real satellite equivalent  (SNR ≈ 10 dB, recommended default)
    −50 dB → +10 dB margin for multipath-heavy environments
    −40 dB → high-SNR diagnosis / antenna alignment (SNR ≈ 21 dB)
    −70 dB → worst-case simulation (low-elevation satellite, SNR ≈ 1 dB)
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tx import transmit_ira, transmit_pass
from hw.ad9363 import DEFAULT_URI

# Iridium Ring Alert / Simplex channel (confirmed by gr-iridium + iridium-toolkit)
_IRIDIUM_FREQ_HZ: int = 1_626_270_000   # 1626.270 MHz

_DEFAULT_GAIN_DB: float = -60.0
_DEFAULT_MAX_ELEV_DEG: float = 45.0
_DEFAULT_PASS_DUR_S: float = 90.0
_DEFAULT_IRA_SLOTS: int = 20


def _print_safety_banner(mode: str) -> None:
    mode_label = {
        "ira":  "IRA burst TX  (static CFO, no Doppler)",
        "pass": "LEO pass simulation  (IRA + Doppler chirp)",
    }[mode]
    print("=" * 68)
    print("  LibreSDR — INDOOR 1626 MHz TEST")
    print(f"  Mode: {mode_label}")
    print("=" * 68)
    print("  LEGAL WARNING: Iridium band (1616–1626.5 MHz) is licensed.")
    print("  Use a wired connection (TX → attenuator ≥ 30 dB → RX) ONLY.")
    print(f"  Default gain: {_DEFAULT_GAIN_DB:+.0f} dB = satellite-equivalent power.")
    print("=" * 68)
    print()


def _check_gain(gain_db: float, *, dry_run: bool) -> None:
    if gain_db > -20.0 and not dry_run:
        print(f"[ERROR] Gain {gain_db:+.1f} dB exceeds the indoor safety limit of "
              f"−20 dB for the wired 1626 MHz test.")
        print("        The KrakenSDR front-end may saturate. Use −20 dB or lower.")
        print("        Override is NOT provided — reduce the gain.")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Indoor 1626 MHz Iridium test transmitter "
            "(cable + attenuator required)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["ira", "pass"], default="ira",
        help="ira = static IRA bursts (CFO≈0); pass = simulated LEO pass with Doppler",
    )
    p.add_argument("--gain", type=float, default=_DEFAULT_GAIN_DB,
                   help="TX attenuation [dB].  −60 dB = real satellite SNR @ 1 m.")
    p.add_argument("--sat-id", type=int, default=47,
                   help="Satellite ID encoded in IRA frame payload (0–127)")
    p.add_argument("--beam-id", type=int, default=3,
                   help="Beam ID encoded in IRA frame payload (0–47)")
    p.add_argument("--cyclic", action="store_true",
                   help="Repeat the TX buffer until Ctrl+C")
    p.add_argument("--uri", default=DEFAULT_URI,
                   help="LibreSDR IIO URI")
    p.add_argument("--dry-run", action="store_true",
                   help="Print config and exit without transmitting")

    # ── IRA-mode options ──────────────────────────────────────────────────
    ira = p.add_argument_group("IRA mode (--mode ira)")
    ira.add_argument("-n", "--n-slots", type=int, default=_DEFAULT_IRA_SLOTS,
                     help="Number of IRA TDMA slots to transmit (90 ms each)")

    # ── Pass-mode options ─────────────────────────────────────────────────
    pas = p.add_argument_group("Pass mode (--mode pass)")
    pas.add_argument("--elev", type=float, default=_DEFAULT_MAX_ELEV_DEG,
                     help="Maximum elevation angle of the simulated pass [°]")
    pas.add_argument("--dur", type=float, default=_DEFAULT_PASS_DUR_S,
                     help="Pass duration [s]; use --cyclic to loop continuously")

    args = p.parse_args()

    _print_safety_banner(args.mode)
    _check_gain(args.gain, dry_run=args.dry_run)

    if args.mode == "ira":
        transmit_ira(
            freq_hz=_IRIDIUM_FREQ_HZ,
            gain_db=args.gain,
            n_slots=args.n_slots,
            sat_id=args.sat_id,
            beam_id=args.beam_id,
            cyclic=args.cyclic,
            uri=args.uri,
            dry_run=args.dry_run,
        )
        return

    transmit_pass(
        freq_hz=_IRIDIUM_FREQ_HZ,
        gain_db=args.gain,
        max_elev_deg=args.elev,
        pass_dur_s=args.dur,
        snr_db=100.0,   # noise-free generator — hardware channel adds natural AWGN
        sat_id=args.sat_id,
        beam_id=args.beam_id,
        cyclic=args.cyclic,
        uri=args.uri,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
