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
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

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
_ARGS = _ap.parse_args()

FREQ_HZ   = _ARGS.freq
GAIN_DB   = _ARGS.gain if _ARGS.gain is not None else float(C.GAIN_DB)
BURST_SNR  = _ARGS.snr
BURST_PAPR = _ARGS.papr
BURST_PWR  = _ARGS.power
CH_IDX     = _ARGS.channel
VERBOSE    = _ARGS.verbose
DEMOD_EN   = not _ARGS.no_demod
HOST       = _ARGS.host if _ARGS.host else C.HEIMDALL_HOST
PORT       = _ARGS.port if _ARGS.port else C.HEIMDALL_PORT
FS         = float(C.SAMPLE_RATE_HZ)

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALISE
# ═══════════════════════════════════════════════════════════════════════════════

def _eprint(*args, **kwargs):
    """Print to stderr (keeps stdout clean for RAW: line piping)."""
    print(*args, file=sys.stderr, **kwargs)


_eprint(f"[IRD-RX] KrakenSDR Iridium Receiver")
_eprint(f"[IRD-RX] Frequency   : {FREQ_HZ/1e6:.4f} MHz")
_eprint(f"[IRD-RX] Gain        : {GAIN_DB} dB   Channel: {CH_IDX}")
_eprint(f"[IRD-RX] Thresholds  : SNR ≥ {BURST_SNR} dB   PAPR ≥ {BURST_PAPR} dB   pwr ≥ {BURST_PWR} dBW")
_eprint(f"[IRD-RX] Demod       : {'ENABLED (RAW: lines → stdout)' if DEMOD_EN else 'DISABLED (detection only)'}")
_eprint(f"[IRD-RX] Capture time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
_eprint()

# Build pipeline
_pipeline = BurstPipeline(
    input_fs       = int(FS),
    center_freq_hz = FREQ_HZ,
    burst_snr      = BURST_SNR,
    burst_papr     = BURST_PAPR,
    burst_pwr      = BURST_PWR,
    demod_enabled  = DEMOD_EN,
)

# Connect to Heimdall
_kraken = KrakenIQSource(
    host        = HOST,
    port        = PORT,
    ctrl_port   = C.HEIMDALL_CTRL,
    num_channels= C.N_ANTENNAS,
    freq_hz     = FREQ_HZ,
    gain_db     = GAIN_DB,
    verbose     = C.VERBOSE_FRAMES,
)
_kraken.start()
_eprint(f"[IRD-RX] Connecting to Heimdall at {HOST}:{PORT} …", end=" ", flush=True)
time.sleep(2.0)
_eprint("connected" if _kraken.is_connected else "not reachable — will retry")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

_WARMUP  = 3           # discard first N frames while Heimdall settles
_warmup_n = 0
_frames_total  = 0
_bursts_total  = 0
_decoded_total = 0

try:
    while True:
        frame = _kraken.get_frame(timeout=0.3)
        if frame is None:
            continue

        _frames_total += 1
        X = frame.astype(np.complex128)
        if C.HW_NUM_SAMPLES > 0 and X.shape[1] > C.HW_NUM_SAMPLES:
            X = X[:, :C.HW_NUM_SAMPLES]
        x = X[min(CH_IDX, X.shape[0] - 1), :]

        # Warmup: let Heimdall settle before burst search
        if _warmup_n < _WARMUP:
            _warmup_n += 1
            _eprint(f"[IRD-RX] Warmup {_warmup_n}/{_WARMUP} …", end="\r")
            continue

        now = time.time()
        pr  = _pipeline.process(x, timestamp=now)

        if not pr.burst.is_burst:
            continue

        _bursts_total += 1
        sign = "+" if pr.burst.doppler_hz >= 0 else ""

        if VERBOSE or not DEMOD_EN:
            _eprint(
                f"[BURST #{_bursts_total:05d}]"
                f"  Δf {sign}{pr.burst.doppler_hz/1e3:+.2f} kHz"
                f"  SNR {pr.burst.burst_snr_db:.1f} dB"
                f"  PAPR {pr.burst.burst_papr_db:.1f} dB"
                f"  pwr {pr.burst.abs_pwr_db:.1f} dBW"
                f"  pass #{_pipeline.tracker.pass_count}"
            )

        if pr.raw_line is not None:
            _decoded_total += 1
            # RAW: line → stdout (piped to iridium-parser.py)
            print(pr.raw_line, flush=True)
            if VERBOSE:
                _eprint(f"  → {pr.raw_line[:90]}{'…' if len(pr.raw_line) > 90 else ''}")
        elif DEMOD_EN and VERBOSE:
            _eprint("  → demod: no sync word found")

except KeyboardInterrupt:
    _eprint()
    _eprint(f"[IRD-RX] Stopped by user.")
finally:
    _kraken.stop()
    tracker = _pipeline.tracker
    _eprint(
        f"[IRD-RX] Summary: frames={_frames_total}"
        f"  bursts={_bursts_total}"
        f"  decoded={_decoded_total}"
        f"  passes={tracker.pass_count}"
    )
