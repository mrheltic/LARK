#!/usr/bin/env python3
"""
tx/indoor_1626.py — Indoor Iridium-band test at 1626.270 MHz (Ring Alert channel).

======================================================================
  IMPORTANT LEGAL AND SAFETY NOTICE
======================================================================
Transmitting in the Iridium L-band (1616–1626.5 MHz) without a licence
is ILLEGAL in Italy (and most countries).

Usage
-----
    python3 tx/indoor_1626.py --dry-run           # config check only
    python3 tx/indoor_1626.py --gain -60          # satellite-equivalent SNR (default)
    python3 tx/indoor_1626.py --gain -60 --cyclic # continuous loop
    python3 tx/indoor_1626.py -n 1                # single burst (quick test)
    python3 tx/indoor_1626.py --gain -40 -n 8    # 8 bursts at high power (diagnose only)

Satellite emulation link budget (gain = −60 dB, d ≈ 1 m)
---------------------------------------------------------
  AD9363 max output at 1626 MHz : +3 dBm
  TX hardware gain               : −60 dB
  TX antenna port power          : −57 dBm
  Free-space path loss at 1 m   : 38.7 dB  (FSPL = 20·log₁₀(4πd/λ))
  Received power at KrakenSDR   : −96 dBm

  Real Iridium NEXT at 45° elevation (slant range ≈ 1100 km):
  FSPL                           : 157.5 dB
  Typical Rx signal (0 dBi ant.) : −95 to −90 dBm

  → −60 dB TX at 1 m ≈ real satellite SNR (≈5 dB margin).
  For higher SNR (diagnose/align): use −40 dB.
  For weaker signal (worst-case sat, low elevation): use −70 dB.
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tx import transmit_ira
from hw.ad9363 import DEFAULT_URI

# Iridium Ring Alert / Simplex channel (confirmed by gr-iridium + iridium-toolkit)
_IRIDIUM_FREQ_HZ: int = 1_626_270_000   # 1626.270 MHz

# −60 dB → received power ≈ −96 dBm at KrakenSDR (d ≈ 1 m) = real satellite SNR.
# Use −40 dB for high-SNR alignment / diagnosis; −70 dB for worst-case simulation.
# The safety guard below rejects anything above −20 dB.
_DEFAULT_GAIN_DB: float = -30.0


def _print_safety_banner() -> None:
    print("=" * 68)
    print("  LibreSDR — INDOOR 1626 MHz TEST  (WIRED / NEAR-FIELD ONLY)")
    print("=" * 68)
    print("  LEGAL WARNING: Iridium band (1616–1626.5 MHz) is licensed.") 
    print("  Use a wired connection (TX → attenuator ≥ 30 dB → RX) ONLY.")
    print("  Power levels are set to the absolute minimum by default.")
    print("=" * 68)
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Indoor 1626 MHz Iridium IRA test (cable + attenuator required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gain",    type=float, default=_DEFAULT_GAIN_DB,
                   help="TX attenuation [dB].  −40 dB confirmed for 1–3 m indoor test.")
    p.add_argument("-n", "--n-slots", type=int, default=20,
                   help="Number of IRA slots to transmit")
    p.add_argument("--sat-id",  type=int,   default=47,
                   help="Satellite ID in payload (0–127)")
    p.add_argument("--beam-id", type=int,   default=3,
                   help="Beam ID in payload (0–47)")
    p.add_argument("--cyclic",  action="store_true",
                   help="Repeat buffer continuously until Ctrl+C")
    p.add_argument("--uri",     default=DEFAULT_URI,
                   help="LibreSDR IIO URI")
    p.add_argument("--dry-run", action="store_true",
                   help="Print config and exit without transmitting")
    args = p.parse_args()

    _print_safety_banner()

    # Safety guard: refuse gains that could overload the KrakenSDR front-end.
    # The AD9363 maximum output power is around +5 dBm at 0 dB setting.
    # Even over a 30 dB attenuator, −20 dB gives −45 dBm at the RX input,
    # which is already above the KrakenSDR optimal point. Reject > −20 dB.
    if args.gain > -20.0 and not args.dry_run:
        print(f"[ERROR] Gain {args.gain:+.1f} dB exceeds the indoor safety limit of "
              f"−20 dB for the wired 1626 MHz test.")
        print("        The KrakenSDR front-end may saturate. Use −20 dB or lower.")
        print("        Override is NOT provided — reduce the gain.")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
