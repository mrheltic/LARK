#!/usr/bin/env python3
"""
tx_868_libresdr.py — CW / pilot-tone beacon at 868 MHz via LibreSDR (AD9363).

Transmits a continuous-wave tone at 868.1 MHz (or any ISM frequency) as a
coherent phase reference for KrakenSDR DoA tests with a 5-element UCA.

Pilot-tone mode (recommended)
------------------------------
Pass --pilot-offset N (default 100 000 Hz) to shift the TX tone N Hz above
the LO.  The KrakenSDR side then extracts that narrow-band tone via FFT gating,
rejecting broadband noise and gaining ~20 dB of effective SNR:

    SNR gain = 10·log10(sample_rate / pilot_bw)  ≈ 20 dB
               (at 1.024 MSPS / 10 kHz extraction window)

100 kHz is chosen so it falls exactly on FFT bin 12 800 of the Heimdall DAQ
(1 024 000 Hz rate, 131 072-sample CPI → bin resolution 7.8125 Hz,
12 800 × 7.8125 = 100 000.0 Hz → zero spectral leakage).

KrakenSDR config.py must have PILOT_TONE_OFFSET_HZ = 100_000 (or the same
value as --pilot-offset, default) and PILOT_TONE_ENABLED = True.

Usage
-----
    python3 tx_868_libresdr.py                         # CW with 100 kHz pilot offset
    python3 tx_868_libresdr.py --freq 865197800        # match Heimdall FREQ_HZ
    python3 tx_868_libresdr.py --gain -10              # use -10 dB for longer range
    python3 tx_868_libresdr.py --pilot-offset 0        # plain carrier (DC, no offset)
    python3 tx_868_libresdr.py --dry-run               # print config, no TX

Warning: always use TX → RX via coaxial cable + attenuator in lab.
         Start with gain=-30 dB and work up to avoid saturating the KrakenSDR.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# ── Default parameters ───────────────────────────────────────────────────────
DEFAULT_URI              = "ip:192.168.1.10"
DEFAULT_FREQ_HZ          = 868_100_000   # [Hz]  868.1 MHz ISM
TX_SAMPLE_RATE           = 1_000_000     # 1 MSPS
TX_RF_BANDWIDTH          = 200_000       # 200 kHz
DEFAULT_GAIN_DB          = -15.0         # [dB] TX attenuation  (0=max power)
#                                          #  Field tests: -10 to -15 dB gives a
#                                          #  clean signal at 1–5 m with the KrakenSDR.
#                                          #  Start at -30 dB in lab, step up slowly.
DEFAULT_PILOT_OFFSET_HZ  = 100_000       # [Hz] tone shift above LO
#                                          #  0 = plain carrier at DC (plain CW mode)
#                                          #  100 000 Hz lands on exact FFT bin 12800
#                                          #  at Heimdall 1.024 MSPS / CPI 131072.
#                                          #  Must match PILOT_TONE_OFFSET_HZ in
#                                          #  krakenSDR/src/apps/doa_test_868/config.py

# CW buffer duration (repeated in hardware loop)
_BUF_DURATION_S = 0.10   # 100 ms → 100 000 samples at 1 MSPS


# =============================================================================
# Helpers
# =============================================================================

def _make_cw(
    duration_s:  float = _BUF_DURATION_S,
    offset_hz:   float = 0.0,
) -> np.ndarray:
    """
    Generate a CW IQ buffer.

    offset_hz = 0    : pure carrier at DC (LO frequency).  I=const, Q=0.
    offset_hz = 100k : tone at +100 kHz above LO.  Complex exponential.

    The inter-channel phase relationships received by the KrakenSDR are the
    same regardless of offset_hz, so all DoA algorithms work identically.
    A non-zero offset avoids hardware DC artefacts and enables narrow-band
    pilot-tone extraction at the receiver (see extract_pilot_tone() in
    doa_uca_2d.py).
    """
    n = int(round(TX_SAMPLE_RATE * duration_s))
    t = np.arange(n, dtype=np.float64)
    if offset_hz == 0.0:
        # Pure carrier — save compute, same as exp(j·0)
        iq = np.ones(n, dtype=np.complex64) * (0.9 * 2 ** 14)
    else:
        iq = (0.9 * 2 ** 14 * np.exp(
            2j * np.pi * offset_hz / TX_SAMPLE_RATE * t
        )).astype(np.complex64)
    return iq


def _connect(uri: str):
    """Connect to LibreSDR via pyadi-iio."""
    try:
        import adi
    except ImportError:
        print("[ERROR] pyadi-iio not installed. Install with:")
        print("  pip install pyadi-iio")
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
        except OSError as exc:
            if exc.errno == 16 and attempt < 2:   # Device busy
                print(f"busy ({attempt+1}/3), waiting 3 s ...", end=" ", flush=True)
                time.sleep(3)
            else:
                print(f"FAILED\n[ERROR] {exc}")
                host = uri.split(":", 1)[-1] if ":" in uri else uri
                print(f"  Check connection:  ping {host}")
                print(f"  IIO diagnostics:   iio_info -u {uri}")
                sys.exit(1)
        except Exception as exc:
            print(f"FAILED\n[ERROR] {exc}")
            sys.exit(1)


def _configure_tx(sdr, freq_hz: int, gain_db: float) -> None:
    """Configure the AD9363 TX channel."""
    sdr.sample_rate           = int(TX_SAMPLE_RATE)
    sdr.tx_rf_bandwidth       = int(TX_RF_BANDWIDTH)
    sdr.tx_lo                 = int(freq_hz)
    sdr.tx_hardwaregain_chan0 = float(gain_db)
    print(f"  Frequency : {freq_hz / 1e6:.3f} MHz")
    print(f"  MSPS      : {TX_SAMPLE_RATE / 1e6:.1f}")
    print(f"  TX gain   : {gain_db:+.1f} dB  (0=max power, negative=attenuation)")


def _try_dds_cw(uri: str, sdr=None, scale: float = 0.9,
                pilot_offset_hz: int = 0):
    """
    Configure the AD9363 hardware DDS for a CW tone at pilot_offset_hz.

    The FPGA DDS core generates the signal internally — no software buffer
    push loop needed.  The tone persists as long as the IIO context is open.

    IMPORTANT: the returned iio.Context MUST stay alive in the caller scope;
    if garbage-collected the DDS may reset to zero.

    pilot_offset_hz = 0       : DC carrier (tone at LO frequency)
    pilot_offset_hz = 100_000 : tone at LO + 100 kHz

    Returns
    -------
    Active iio.Context if DDS configured, None if not available.
    """
    try:
        import iio
    except ImportError:
        return None

    # Reuse the internal adi.Pluto context when possible (same connection)
    ctx = getattr(sdr, "_ctx", None)
    _ctx_owned = False
    if ctx is None:
        try:
            ctx = iio.Context(uri)
            _ctx_owned = True
        except Exception:
            return None

    _DDS_NAMES = [
        "cf-ad9361-dds-core-lpc",
        "cf-ad9361-dds-core-hpc",
        "axi-ad9361-dds-lpc",
        "axi-ad9361-dds-hpc",
    ]
    dds = None
    for name in _DDS_NAMES:
        dds = ctx.find_device(name)
        if dds is not None:
            break
    if dds is None:
        return None

    # TX1 DDS channels: altvoltage0/1 = I tone-1/tone-2, altvoltage2/3 = Q tone-1/tone-2
    # Generating a complex tone at +f Hz:  I = cos(2πft),  Q = sin(2πft)
    #   altvoltage0: freq=f,  phase=0°,    scale=scale   (I, tone 1)
    #   altvoltage1: off
    #   altvoltage2: freq=f,  phase=90000, scale=scale   (Q, tone 1, 90° relative to I)
    #   altvoltage3: off
    # For DC (pilot_offset_hz=0): freq=0 → DC; quadrature phase has no effect.
    freq_mdeg_cfgs = [
        ("altvoltage0", int(pilot_offset_hz), 0,      scale),  # TX1_I tone-1
        ("altvoltage1", 0,                   0,      0.0  ),  # TX1_I tone-2 off
        ("altvoltage2", int(pilot_offset_hz), 90000,  scale),  # TX1_Q tone-1 (+90°)
        ("altvoltage3", 0,                   0,      0.0  ),  # TX1_Q tone-2 off
    ]
    try:
        for ch_name, freq, phase_mdeg, sc in freq_mdeg_cfgs:
            ch = dds.find_channel(ch_name, is_output=True)
            if ch is None:
                return None
            try: ch.attrs["frequency"].value = str(freq)
            except Exception: pass
            try: ch.attrs["phase"].value = str(phase_mdeg)
            except Exception: pass
            try: ch.attrs["scale"].value = f"{sc:.6f}"
            except Exception: pass
            try: ch.attrs["raw"].value = "1" if sc > 0 else "0"
            except Exception: pass
        return ctx if _ctx_owned else True
    except Exception:
        return None


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="CW / pilot-tone beacon at 868 MHz for KrakenSDR DoA tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uri",          default=DEFAULT_URI,
                   help="LibreSDR IIO URI")
    p.add_argument("--freq",         type=int,   default=DEFAULT_FREQ_HZ,
                   help="TX centre frequency [Hz]")
    p.add_argument("--gain",         type=float, default=DEFAULT_GAIN_DB,
                   help="TX attenuation [dB]  (0=max power, -89.75=min).  "
                        "Start at -30 dB in lab!")
    p.add_argument("--pilot-offset", type=int,   default=DEFAULT_PILOT_OFFSET_HZ,
                   help="Tone offset above LO [Hz].  "
                        "0=plain DC carrier.  "
                        "100000=pilot tone mode (recommended; matches KrakenSDR config).")
    p.add_argument("--dry-run",      action="store_true",
                   help="Print config and exit without transmitting")
    args = p.parse_args()

    print("=" * 55)
    print("  LibreSDR — CW beacon 868 MHz")
    print("=" * 55)

    pilot_offset = int(args.pilot_offset)
    tx_buf = _make_cw(offset_hz=float(pilot_offset))

    mode_str = (f"pilot tone +{pilot_offset/1e3:.0f} kHz above LO"
                if pilot_offset != 0 else "plain carrier at DC (0 Hz offset)")
    print(f"  TX mode    : {mode_str}")
    print(f"  Buffer     : {len(tx_buf)} samples  "
          f"({len(tx_buf) / TX_SAMPLE_RATE * 1e3:.0f} ms, repeated)")

    if args.dry_run:
        print("  [DRY RUN]  No transmission.")
        return

    sdr = _connect(args.uri)
    _configure_tx(sdr, args.freq, args.gain)

    # ── TX strategy ───────────────────────────────────────────────────────────
    # Hardware cyclic DMA over Ethernet is unreliable: DMA transmits once then stops.
    # Attempt 1: hardware DDS (tone generated on FPGA, no software buffer needed).
    # Attempt 2: software push loop (non-cyclic, rate-limited by DMA/TCP flow control).
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

    # _dds_ctx MUST stay alive in this scope — do NOT rename or del!
    # If garbage-collected, the IIO context closes and the DDS resets.
    _dds_ctx = _try_dds_cw(args.uri, sdr=sdr, scale=0.9,
                            pilot_offset_hz=pilot_offset)
    dds_ok   = _dds_ctx is not None

    print()
    if dds_ok:
        print(f"  [DDS]  Hardware tone active at +{pilot_offset/1e3:.0f} kHz.  "
              "No software buffer needed.")
    else:
        # Non-cyclic: sdr.tx() blocks until the DMA consumes the previous buffer,
        # so the loop runs at roughly one buffer per DMA period automatically.
        sdr.tx_cyclic_buffer = False
        sdr.tx(tx_buf)
        print("  [SW]  CW active via software push loop.")

    print("  Press Ctrl+C to stop.")
    try:
        while True:
            if dds_ok:
                # Keepalive: prevents iiod from closing the idle connection
                time.sleep(2.0)
                try:
                    _ = sdr.tx_lo
                except Exception:
                    pass
            else:
                # Re-push: sdr.tx() blocks until DMA has consumed the previous
                # buffer, so this loop runs at the natural DMA rate.
                sdr.tx(tx_buf)
    except KeyboardInterrupt:
        print("\n  Stop.")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
