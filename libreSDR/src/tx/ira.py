#!/usr/bin/env python3
"""
tx/ira.py — Authentic Iridium IRA burst transmitter.

Transmits π/4-DQPSK IRA bursts at any frequency via LibreSDR (AD9363).

Signal parameters (authentic Iridium IRA)
------------------------------------------
  Modulation : π/4-DQPSK
  Symbol rate: 25 000 sps
  RRC β      : 0.4  (gr-iridium reference)
  Burst       : 64 preamble + 12 UW + 167 data + 2 tail = 245 symbols
  Preamble   : all-zero dibits → pure CW tone at fc + Rs/8 = fc + 3125 Hz
  TDMA period: 90 ms per slot

Default frequency is 868.1 MHz (ISM band) for safe lab and indoor tests.
Use --freq 1626270000 for real Iridium band (cable + ≥ 30 dB attenuator ONLY).

Usage
-----
    python3 tx/ira.py                          # 868.1 MHz, 4 bursts → stop
    python3 tx/ira.py --cyclic                 # loop forever (Ctrl+C to stop)
    python3 tx/ira.py --freq 1626270000 -n 1  # single burst at Iridium freq
    python3 tx/ira.py --freq 868100000 -n 8   # 8 bursts at 868 MHz
    python3 tx/ira.py --gain -20               # higher power
    python3 tx/ira.py --dry-run                # show config, no TX

WARNING: transmitting at 1616–1626.5 MHz (Iridium band) without a licence
         is ILLEGAL. Use cable + ≥ 30 dB attenuator between TX and RX.

Gain guidance
-------------
    -60 dB : near-zero power (sanity check only)
    -30 dB : safe for bench tests ≤ 2 m (cable + 20 dB attenuator)
    -20 dB : typical indoor KrakenSDR test at 1–3 m
      0 dB : maximum power — NEVER in lab without cable + strong attenuator
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Authentic Iridium IRA burst transmitter via LibreSDR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq",    type=int,   default=868_100_000,
                   help="TX carrier frequency [Hz]  (default: 868.1 MHz ISM)")
    p.add_argument("--gain",    type=float, default=-30.0,
                   help="TX attenuation [dB]  (0=max power)")
    p.add_argument("-n", "--n-slots", type=int, default=4,
                   help="Number of IRA slots to transmit")
    p.add_argument("--sat-id",  type=int,   default=47,
                   help="Satellite ID encoded in frame payload (0–127)")
    p.add_argument("--beam-id", type=int,   default=3,
                   help="Beam ID encoded in frame payload (0–47)")
    p.add_argument("--cyclic",  action="store_true",
                   help="Repeat buffer continuously until Ctrl+C")
    p.add_argument("--uri",     default=DEFAULT_URI,
                   help="LibreSDR IIO URI")
    p.add_argument("--dry-run", action="store_true",
                   help="Print config and exit without transmitting")
    args = p.parse_args()

    transmit_ira(
        freq_hz=args.freq,
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
