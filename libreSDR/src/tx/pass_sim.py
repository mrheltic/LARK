#!/usr/bin/env python3
"""
tx/pass_sim.py — Simulated Iridium LEO satellite pass transmitter.

Generates and transmits a realistic Iridium satellite pass:
  • IRA bursts every 90 ms (authentic TDMA)
  • Orbital Doppler model: h = 780 km, up to ±40 kHz @ 1626 MHz
  • Configurable maximum elevation and pass duration
  • Optional AWGN (default: noise-free for clean DoA tests)

Typical indoor integration test
---------------------------------
    Terminal 1 (TX):
        python3 tx/pass_sim.py --freq 868100000 --gain -20 --cyclic

    Terminal 2 (RX + DoA):
        python3 krakenSDR/src/apps/doa_test_868/doa_test_868_burst.py --freq 868100000

The KrakenSDR will see:
  • IRA bursts every 90 ms  (same TDMA as a real satellite)
  • Preamble tone at fc + 3125 Hz  (identical to the real satellite)
  • Realistic Doppler chirp over the pass arc
  • Power envelope following the RHCP patch pattern (cos²(elevation))

Usage
-----
    python3 tx/pass_sim.py                              # 868.1 MHz, 60 s, 45° elev
    python3 tx/pass_sim.py --freq 868100000 --elev 60  # 60° max elevation
    python3 tx/pass_sim.py --dur 120                   # 2-minute pass
    python3 tx/pass_sim.py --cyclic                    # loop pass buffer until Ctrl+C
    python3 tx/pass_sim.py --dry-run --save pass.iq    # generate IQ only, no TX
    python3 tx/pass_sim.py --snr 15                    # add AWGN at 15 dB SNR

WARNING: transmitting in the Iridium band (1616–1626.5 MHz) without a
         licence is ILLEGAL. Use ISM band (868 / 433 MHz) or a wired RF
         connection (cable + ≥ 30 dB attenuator) for lab tests.
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tx import transmit_pass
from hw.ad9363 import DEFAULT_URI


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simulated Iridium LEO pass transmitter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq",    type=int,   default=868_100_000,
                   help="TX carrier frequency [Hz]  (default: 868.1 MHz ISM)")
    p.add_argument("--gain",    type=float, default=-20.0,
                   help="TX attenuation [dB]")
    p.add_argument("--elev",    type=float, default=45.0,
                   help="Maximum elevation of the simulated pass [°]")
    p.add_argument("--dur",     type=float, default=60.0,
                   help="Simulation duration [s]")
    p.add_argument("--snr",     type=float, default=100.0,
                   help="SNR [dB] of generated signal.  100 = noise-free.")
    p.add_argument("--sat-id",  type=int,   default=47,
                   help="Satellite ID in payload")
    p.add_argument("--beam-id", type=int,   default=3,
                   help="Beam ID in payload")
    p.add_argument("--cyclic",  action="store_true",
                   help="Repeat pass buffer in hardware loop until Ctrl+C")
    p.add_argument("--uri",     default=DEFAULT_URI,
                   help="LibreSDR IIO URI")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate IQ but do not transmit")
    p.add_argument("--save",    default=None,
                   help="Save raw complex64 IQ to this file path")
    args = p.parse_args()

    transmit_pass(
        freq_hz=args.freq,
        gain_db=args.gain,
        max_elev_deg=args.elev,
        pass_dur_s=args.dur,
        snr_db=args.snr,
        sat_id=args.sat_id,
        beam_id=args.beam_id,
        cyclic=args.cyclic,
        uri=args.uri,
        dry_run=args.dry_run,
        save_iq=args.save,
    )


if __name__ == "__main__":
    main()
