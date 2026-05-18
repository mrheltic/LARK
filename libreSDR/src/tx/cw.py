#!/usr/bin/env python3
"""
tx/cw.py — CW / pilot-tone beacon for KrakenSDR DoA tests.

Transmits a continuous-wave tone via LibreSDR (AD9363).

With --pilot-offset 100000 (default) the tone lands at LO + 100 kHz, which
falls on FFT bin 12800 of the Heimdall DAQ (1.024 MSPS / 131072-sample CPI,
7.8125 Hz/bin → zero spectral leakage). This offset also avoids DC artefacts
from the AD9363 mixer.

Usage
-----
    python3 tx/cw.py                            # 868.1 MHz, pilot tone +100 kHz
    python3 tx/cw.py --freq 868100000           # explicit frequency
    python3 tx/cw.py --pilot-offset 0           # plain carrier at DC
    python3 tx/cw.py --gain -10                 # higher power (careful!)
    python3 tx/cw.py --uri ip:192.168.2.1       # non-default IIO URI
    python3 tx/cw.py --dry-run                  # print config, no TX

Gain guidance
-------------
    -60 dB : near-zero power (sanity check only)
    -30 dB : safe for bench tests ≤ 2 m (cable + 20 dB attenuator)
    -20 dB : typical indoor KrakenSDR loopback test at 1–3 m
    -10 dB : outdoor test at > 10 m (verify no ADC saturation first)
      0 dB : maximum power — NEVER in lab without cable + strong attenuator
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tx import transmit_cw
from hw.ad9363 import DEFAULT_URI


def main() -> None:
    p = argparse.ArgumentParser(
        description="CW / pilot-tone beacon for KrakenSDR DoA tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq",         type=int,   default=868_100_000,
                   help="TX LO frequency [Hz]")
    p.add_argument("--gain",         type=float, default=-20.0,
                   help="TX attenuation [dB]  (0=max power, start at -30 in lab)")
    p.add_argument("--pilot-offset", type=int,   default=100_000,
                   help="Tone offset above LO [Hz].  0 = plain carrier.")
    p.add_argument("--uri",          default=DEFAULT_URI,
                   help="LibreSDR IIO URI")
    p.add_argument("--dry-run",      action="store_true",
                   help="Print config and exit without transmitting")
    args = p.parse_args()

    transmit_cw(
        freq_hz=args.freq,
        gain_db=args.gain,
        pilot_offset_hz=args.pilot_offset,
        uri=args.uri,
        cyclic=True,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
