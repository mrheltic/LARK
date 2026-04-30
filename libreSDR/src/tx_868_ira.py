#!/usr/bin/env python3
"""
tx_868_ira.py — Headless IRA burst transmitter at 868 MHz via LibreSDR (AD9363)

Transmits authentic Iridium IRA-structure π/4-DQPSK bursts at 868 MHz
(ISM band) without any GUI.  Designed for SSH / headless operation.

Signal structure (authentic Iridium IRA):
  - Modulation  : π/4-DQPSK  (RRC β=0.4, 25 ksps — matches gr-iridium)
  - Preamble    : 64 symbols, all-zero dibits → pure tone at fc + Rs/8 = fc + 3125 Hz
  - UW          : 12 absolute BPSK symbols (from iridium.h UW_DL[])
  - Data        : 167+2 symbols, rate-1/2 K=7 convolutional code
  - TDMA period : one IRA slot per 90 ms superframe

   ┌──────────────────────────────────────────────────────────────────┐
   │  TX @ 868 MHz  ──(cable + attenuator)──►  KrakenSDR 868 MHz RX  │
   │     tx_868_ira.py                             doa_test_868_burst.py │
   └──────────────────────────────────────────────────────────────────┘

Usage
-----
    python3 tx_868_ira.py                    # loop forever (Ctrl+C to stop)
    python3 tx_868_ira.py --n 1              # single burst then exit
    python3 tx_868_ira.py --gain -20         # higher power
    python3 tx_868_ira.py --uri ip:192.168.2.1
    python3 tx_868_ira.py --demo             # dry-run (print config, no TX)

Hardware requirements
---------------------
    LibreSDR / AD9363 reachable at DEFAULT_URI (ip:192.168.1.10 by default).
    pyadi-iio:  pip install pyadi-iio
    scipy:      pip install scipy numpy

Gain guidance (start conservative)
------------------------------------
    -60 dB : near-zero power (sanity check only)
    -30 dB : safe for bench tests up to 2 m (cable + 20 dB attenuator)
    -20 dB : SNR ≈ 30 dB at 0.5 m; indoor tests (direct cable preferred)
    -10 dB : field test at > 5 m — ensure no ADC saturation (check λ1/λ2)
      0 dB : max power — never in lab without cable + strong attenuator
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy import signal as sp_signal

# ── Library path ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from iridium.realistic_sim import (
    generate_ira_burst,
    generate_rrc_filter,
    SYMBOL_RATE,
    SPS,
    SAMPLE_RATE as IRA_BASE_RATE,   # 250 000 Hz (SYMBOL_RATE × SPS)
    RRC_BETA,
    RRC_NUM_TAPS,
    IRA_PREAMBLE_SYMS,
    IRA_BURST_SYMS,
    SUPERFRAME_S,
)

# ── Hardware defaults ─────────────────────────────────────────────────────────
DEFAULT_URI      = "ip:192.168.1.10"
DEFAULT_FREQ_HZ  = 868_100_000   # 868.1 MHz ISM
TX_SAMPLE_RATE   = 1_000_000     # 1 MSPS — practical limit for AD9363 over Ethernet
TX_RF_BANDWIDTH  = 200_000       # 200 kHz RF channel
DEFAULT_GAIN_DB  = -30.0         # safe start for bench tests

# Upsampling: 250 kHz → 1 MHz = ×4 (exact integer ratio)
_IRA_UPS = TX_SAMPLE_RATE // IRA_BASE_RATE  # 4

# Preamble tone offset: Rs/8 = 3125 Hz (detectable by gr-iridium / doa_test_868_burst.py)
_PREAMBLE_TONE_HZ = SYMBOL_RATE // 8  # 3 125 Hz


# =============================================================================
# IQ generation
# =============================================================================

def build_slot(rrc: np.ndarray, frame_count: int = 0,
               sat_id: int = 47, beam_id: int = 3) -> np.ndarray:
    """
    Generate one IRA TDMA slot upsampled to TX_SAMPLE_RATE.

    Returns
    -------
    numpy.ndarray of complex64, length = SUPERFRAME_S × TX_SAMPLE_RATE = 90 000 samples.
    The slot includes one IRA burst (9.8 ms) followed by silence (80.2 ms).
    """
    slot_iq, _ = generate_ira_burst(rrc, sat_id=sat_id, beam_id=beam_id,
                                     frame_count=frame_count)
    # Resample from 250 kHz to 1 MHz (integer ×4, no aliasing)
    up = sp_signal.resample_poly(slot_iq, _IRA_UPS, 1)
    # Scale to 80 % DAC full-scale (AD9363 via pyadi-iio: |sample| ≤ 2^14)
    peak = float(np.max(np.abs(up))) + 1e-12
    up_scaled = (up / peak * 0.8 * (2 ** 14)).astype(np.complex64)

    # Pad to exactly one superframe (90 000 samples at 1 MSPS)
    sf_len = int(round(SUPERFRAME_S * TX_SAMPLE_RATE))
    if len(up_scaled) >= sf_len:
        return up_scaled[:sf_len]
    buf = np.zeros(sf_len, dtype=np.complex64)
    buf[: len(up_scaled)] = up_scaled
    return buf


# =============================================================================
# Hardware helpers
# =============================================================================

def _connect(uri: str):
    try:
        import adi
    except ImportError:
        print("[ERROR] pyadi-iio not installed. Install with:  pip install pyadi-iio")
        sys.exit(1)
    print(f"  Connecting to {uri} ...", end=" ", flush=True)
    for attempt in range(3):
        try:
            sdr = adi.Pluto(uri)
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            print("OK")
            return sdr
        except OSError as e:
            if e.errno == 16 and attempt < 2:
                print(f"busy ({attempt+1}/3), waiting 3 s ...", end=" ", flush=True)
                time.sleep(3)
            else:
                host = uri.split(":", 1)[-1] if ":" in uri else uri
                print(f"FAILED\n[ERROR] {e}")
                print(f"  Check connection:  ping {host}")
                print(f"  IIO diagnostics:   iio_info -u {uri}")
                sys.exit(1)
        except Exception as e:
            print(f"FAILED\n[ERROR] {e}")
            sys.exit(1)


def _configure_tx(sdr, freq_hz: int, gain_db: float) -> None:
    sdr.sample_rate           = int(TX_SAMPLE_RATE)
    sdr.tx_rf_bandwidth       = int(TX_RF_BANDWIDTH)
    sdr.tx_lo                 = int(freq_hz)
    sdr.tx_hardwaregain_chan0 = float(gain_db)
    print(f"  Frequency   : {freq_hz / 1e6:.3f} MHz")
    print(f"  Sample rate : {TX_SAMPLE_RATE / 1e6:.1f} MSPS")
    print(f"  TX gain     : {gain_db:+.1f} dB")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Headless IRA burst TX at 868 MHz for KrakenSDR DoA tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uri",     default=DEFAULT_URI,
                   help="LibreSDR IIO URI")
    p.add_argument("--freq",    type=int,   default=DEFAULT_FREQ_HZ,
                   help="TX centre frequency [Hz]")
    p.add_argument("--gain",    type=float, default=DEFAULT_GAIN_DB,
                   help="TX attenuation [dB]  (0=max, -30=safe bench, -60=very low)")
    p.add_argument("-n", "--num-slots", type=int, default=0,
                   help="Slots to transmit then stop  (0 = loop forever)")
    p.add_argument("--sat-id",  type=int,   default=47, help="Satellite ID in burst (0-127)")
    p.add_argument("--beam-id", type=int,   default=3,  help="Beam ID in burst (0-47)")
    p.add_argument("--demo",    action="store_true",
                   help="Dry-run: print config and exit without connecting")
    args = p.parse_args()

    print("=" * 55)
    print("  LibreSDR — IRA burst TX at 868 MHz (headless)")
    print("=" * 55)
    print(f"  Frequency         : {args.freq / 1e6:.3f} MHz")
    print(f"  Symbol rate       : {SYMBOL_RATE / 1e3:.0f} ksps (β={RRC_BETA}, authentic Iridium)")
    print(f"  Preamble tone     : +{_PREAMBLE_TONE_HZ} Hz  (64 syms, all-zero dibits)")
    print(f"  Burst structure   : {IRA_BURST_SYMS} active syms per 90 ms slot")
    print(f"  TX gain           : {args.gain:+.1f} dB")
    print(f"  Slots             : {'∞ (loop)' if args.num_slots == 0 else args.num_slots}")

    if args.demo:
        print("\n  [DEMO]  No transmission.")
        return

    # ── Build RRC filter and a pre-rendered slot buffer ───────────────────
    print("\n  Building IRA burst...", end=" ", flush=True)
    rrc  = generate_rrc_filter(RRC_BETA, SPS, RRC_NUM_TAPS)
    slot = build_slot(rrc, frame_count=0, sat_id=args.sat_id, beam_id=args.beam_id)
    dur_ms = len(slot) / TX_SAMPLE_RATE * 1e3
    print(f"  {len(slot)} samples  ({dur_ms:.0f} ms per slot)")

    sdr = _connect(args.uri)
    _configure_tx(sdr, args.freq, args.gain)
    sdr.tx_cyclic_buffer = False

    # Destroy any leftover buffer before starting
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

    print("\n  Transmitting IRA bursts. Press Ctrl+C to stop.")
    slot_idx = 0
    try:
        while args.num_slots == 0 or slot_idx < args.num_slots:
            # Regenerate the slot every time so frame_count advances
            # (makes each burst unique and allows UW + data validation)
            if slot_idx > 0:
                slot = build_slot(rrc, frame_count=slot_idx,
                                  sat_id=args.sat_id, beam_id=args.beam_id)
            t0 = time.monotonic()
            sdr.tx(slot)
            elapsed = time.monotonic() - t0
            remaining = SUPERFRAME_S - elapsed
            if remaining > 0.005:
                time.sleep(remaining)
            slot_idx += 1
            if slot_idx % 20 == 0:
                print(f"  {slot_idx} slots transmitted  "
                      f"({slot_idx * SUPERFRAME_S:.1f} s elapsed)")
    except KeyboardInterrupt:
        print(f"\n  Stop after {slot_idx} slots.")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
