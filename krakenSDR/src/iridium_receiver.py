#!/usr/bin/env python3
"""
iridium_receiver.py – KrakenSDR Iridium burst receiver / RAW: line producer
=============================================================================

Connects to Heimdall, detects Iridium L-band TDMA bursts with the FFT-based
:class:`BurstDetector`, and DQPSK-demodulates each burst frame with
:class:`IridiumDemod`.  Each successfully decoded burst is emitted as a
``RAW:`` text line on **stdout**, ready to pipe directly into
``iridium-parser.py`` for full protocol decoding.

Typical usage
~~~~~~~~~~~~~
::

    # Pipe into iridium-parser for decoded messages
    cd krakensdr/workspace
    python3 pysdr_doa/iridium_receiver.py | \\
        python3 ../iridium-toolkit/iridium-parser.py -p

    # Or capture raw lines to a file for offline analysis
    python3 pysdr_doa/iridium_receiver.py --freq 1626270000 > session.bits
    cat session.bits | python3 ../iridium-toolkit/iridium-parser.py

    # Verbose mode (also shows detector metrics on stderr)
    python3 pysdr_doa/iridium_receiver.py -v

Options
~~~~~~~
--freq HZ       SDR centre frequency (default: 1 626 270 000 Hz)
--gain DB       IF gain (default: config.py GAIN_DB)
--snr DB        Burst SNR threshold (default: 8 dB)
--papr DB       Burst PAPR threshold (default: 5 dB)
--power DBW     Absolute power squelch (default: -90 dBW)
--channel N     Heimdall IQ channel index (default: 0)
--no-demod      Disable DQPSK — only print burst detection events to stderr
-v / --verbose  Print detector metrics on stderr for every burst frame
--host IP       Heimdall host (default: config.py HEIMDALL_HOST)
--port N        Heimdall data port (default: config.py HEIMDALL_PORT)

Notes
~~~~~
* stdout carries only ``RAW:`` lines so the output is clean for piping.
* All informational / diagnostic messages go to **stderr**.
* Ctrl-C exits cleanly and prints a summary to stderr.
* The receiver runs indefinitely; Heimdall reconnects automatically if the
  connection drops.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ── Resolve package root ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import config as C
from kraken_iq_source import KrakenIQSource
from core.burst_pipeline import BurstPipeline

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _eprint(*args, **kwargs):
    """Print to stderr (keeps stdout clean for RAW: line piping)."""
    print(*args, file=sys.stderr, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── CLI ───────────────────────────────────────────────────────────────────
    _ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ap.add_argument("--freq",     type=float, default=1_626_270_000.0, metavar="HZ",
                     help="Centre frequency [Hz]  (default: 1626270000)")
    _ap.add_argument("--gain",     type=float, default=None, metavar="DB",
                     help="IF gain [dB]  (default: config.py GAIN_DB)")
    _ap.add_argument("--snr",      type=float, default=8.0, metavar="DB",
                     help="Burst SNR threshold [dB]  (default: 8)")
    _ap.add_argument("--papr",     type=float, default=5.0, metavar="DB",
                     help="Burst PAPR threshold [dB]  (default: 5)")
    _ap.add_argument("--power",    type=float, default=-90.0, metavar="DBW",
                     help="Absolute power squelch [dBW]  (default: -90)")
    _ap.add_argument("--channel",  type=int,   default=0, metavar="N",
                     help="Heimdall channel index  (default: 0)")
    _ap.add_argument("--no-demod", action="store_true",
                     help="Disable DQPSK demodulation (detection only)")
    _ap.add_argument("--host",     type=str,   default=None, metavar="IP",
                     help="Heimdall host  (default: config.py HEIMDALL_HOST)")
    _ap.add_argument("--port",     type=int,   default=None, metavar="N",
                     help="Heimdall data port  (default: config.py HEIMDALL_PORT)")
    _ap.add_argument("-v", "--verbose", action="store_true",
                     help="Print detector metrics on stderr for each burst frame")
    args = _ap.parse_args()

    freq_hz    = args.freq
    gain_db    = args.gain if args.gain is not None else float(C.GAIN_DB)
    burst_snr  = args.snr
    burst_papr = args.papr
    burst_pwr  = args.power
    ch_idx     = args.channel
    verbose    = args.verbose
    demod_en   = not args.no_demod
    host       = args.host if args.host else C.HEIMDALL_HOST
    port       = args.port if args.port else C.HEIMDALL_PORT
    fs         = float(C.SAMPLE_RATE_HZ)

    # ── Banner ────────────────────────────────────────────────────────────────
    _eprint(f"[IRD-RX] KrakenSDR Iridium Receiver")
    _eprint(f"[IRD-RX] Frequency   : {freq_hz/1e6:.4f} MHz")
    _eprint(f"[IRD-RX] Gain        : {gain_db} dB   Channel: {ch_idx}")
    _eprint(f"[IRD-RX] Thresholds  : SNR ≥ {burst_snr} dB   PAPR ≥ {burst_papr} dB   pwr ≥ {burst_pwr} dBW")
    _eprint(f"[IRD-RX] Demod       : {'ENABLED (RAW: lines → stdout)' if demod_en else 'DISABLED (detection only)'}")
    _eprint(f"[IRD-RX] Capture time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    _eprint()

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = BurstPipeline(
        input_fs       = int(fs),
        center_freq_hz = freq_hz,
        burst_snr      = burst_snr,
        burst_papr     = burst_papr,
        burst_pwr      = burst_pwr,
        demod_enabled  = demod_en,
    )

    # ── Connect to Heimdall ───────────────────────────────────────────────────
    kraken = KrakenIQSource(
        host         = host,
        port         = port,
        ctrl_port    = C.HEIMDALL_CTRL,
        num_channels = C.N_ANTENNAS,
        freq_hz      = freq_hz,
        gain_db      = gain_db,
        verbose      = C.VERBOSE_FRAMES,
    )
    kraken.start()
    _eprint(f"[IRD-RX] Connecting to Heimdall at {host}:{port} …", end=" ", flush=True)
    time.sleep(2.0)
    _eprint("connected" if kraken.is_connected else "not reachable — will retry")

    # ── Main loop ─────────────────────────────────────────────────────────────
    _WARMUP       = 3
    warmup_n      = 0
    frames_total  = 0
    bursts_total  = 0
    decoded_total = 0

    try:
        while True:
            frame = kraken.get_frame(timeout=0.3)
            if frame is None:
                continue

            frames_total += 1
            X = frame.astype(np.complex128)
            if C.HW_NUM_SAMPLES > 0 and X.shape[1] > C.HW_NUM_SAMPLES:
                X = X[:, :C.HW_NUM_SAMPLES]
            x = X[min(ch_idx, X.shape[0] - 1), :]

            if warmup_n < _WARMUP:
                warmup_n += 1
                _eprint(f"[IRD-RX] Warmup {warmup_n}/{_WARMUP} …", end="\r")
                continue

            now = time.time()
            pr  = pipeline.process(x, timestamp=now)

            if not pr.burst.is_burst:
                continue

            bursts_total += 1
            sign = "+" if pr.burst.doppler_hz >= 0 else ""

            if verbose or not demod_en:
                _pilot = (
                    f"  PILOT {pr.burst.pilot_snr_db:+.1f} dB "
                    f"{'[\u2713]' if pr.burst.pilot_detected else '[ ]'}"
                )
                _eprint(
                    f"[BURST #{bursts_total:05d}]"
                    f"  \u0394f {sign}{pr.burst.doppler_hz/1e3:+.2f} kHz"
                    f"  SNR {pr.burst.burst_snr_db:.1f} dB"
                    f"{_pilot}"
                    f"  PAPR {pr.burst.burst_papr_db:.1f} dB"
                    f"  pwr {pr.burst.abs_pwr_db:.1f} dBW"
                    f"  pass #{pipeline.tracker.pass_count}"
                )

            if pr.raw_line is not None:
                decoded_total += 1
                print(pr.raw_line, flush=True)
                if verbose:
                    _eprint(f"  → {pr.raw_line[:90]}{'…' if len(pr.raw_line) > 90 else ''}")
            elif demod_en and verbose:
                _eprint("  → demod: no sync word found")

    except KeyboardInterrupt:
        _eprint()
        _eprint(f"[IRD-RX] Stopped by user.")
    finally:
        kraken.stop()
        tracker = pipeline.tracker
        _eprint(
            f"[IRD-RX] Summary: frames={frames_total}"
            f"  bursts={bursts_total}"
            f"  decoded={decoded_total}"
            f"  passes={tracker.pass_count}"
        )


if __name__ == "__main__":
    main()
