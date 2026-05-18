"""
tx — LibreSDR Iridium / CW transmitter API.

High-level one-call functions for the most common TX scenarios.
All functions block until transmission is complete (or until Ctrl+C for
cyclic modes).

Quick reference
---------------
    from tx import transmit_cw, transmit_ira, transmit_pass

    # CW pilot-tone beacon for KrakenSDR DoA tests at 868 MHz:
    transmit_cw(freq_hz=868_100_000, gain_db=-20, pilot_offset_hz=100_000)

    # 4 IRA bursts at 1626 MHz (cable + ≥ 30 dB attenuator, lab only!):
    transmit_ira(freq_hz=1_626_270_000, gain_db=-60, n_slots=4)

    # Simulate a 60 s LEO pass at 45° max elevation:
    transmit_pass(freq_hz=868_100_000, gain_db=-20, max_elev_deg=45, pass_dur_s=60)

Available scripts (command-line entry points)
---------------------------------------------
    tx/cw.py            CW / pilot-tone beacon
    tx/ira.py           IRA burst transmitter (868 / 1626 MHz)
    tx/pass_sim.py      Full simulated LEO pass with Doppler chirp
    tx/indoor_1626.py   Indoor 1626 MHz test (cable/near-field, lab only)
    tx/gui.py           GUI: 868 MHz TX + real-time loopback display
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import signal as sp_signal

# -- Path resolution --------------------------------------------------------
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # libreSDR/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hw.ad9363 import Ad9363, DEFAULT_URI, DEFAULT_SAMPLE_RATE as _TX_RATE
from iridium.realistic_sim import (
    generate_ira_burst,
    generate_rrc_filter,
    SYMBOL_RATE,
    SPS,
    SAMPLE_RATE as _IRA_BASE_RATE,
    RRC_BETA,
    RRC_NUM_TAPS,
    IRA_PREAMBLE_SYMS,
    IRA_BURST_SYMS,
    SUPERFRAME_S,
    simulate_iridium_pass,
)

# Upsampling: 250 kHz (IRA base) → 1 MHz (AD9363 TX) = ×4
_IRA_UPS: int = _TX_RATE // _IRA_BASE_RATE  # 4


# =============================================================================
# Buffer builders (pure DSP, no hardware access)
# =============================================================================

def build_ira_slot(
    rrc: np.ndarray,
    frame_count: int = 0,
    sat_id: int = 47,
    beam_id: int = 3,
) -> np.ndarray:
    """
    Generate one IRA TDMA slot upsampled to 1 MSPS.

    Returns a complex64 array scaled for the AD9363 DAC
    (peak ≈ 0.8 × 16384 = 13107).  Length = SUPERFRAME_S × 1e6 = 90000 samples.
    """
    slot_iq, _ = generate_ira_burst(rrc, sat_id=sat_id, beam_id=beam_id,
                                     frame_count=frame_count)
    upsampled = sp_signal.resample_poly(slot_iq, _IRA_UPS, 1)
    # Scale to 80 % DAC full-scale
    peak = float(np.max(np.abs(upsampled))) + 1e-12
    return (upsampled / peak * 0.8 * 2**14).astype(np.complex64)


def build_ira_buffer(
    n_slots: int = 4,
    sat_id: int = 47,
    beam_id: int = 3,
) -> np.ndarray:
    """
    Generate a TX buffer containing `n_slots` consecutive IRA TDMA slots.

    Slot spacing = SUPERFRAME_S = 90 ms (as in the real Iridium system).
    Each slot is exactly one superframe long at 1 MSPS (90 000 samples).
    """
    rrc = generate_rrc_filter(RRC_BETA, SPS, RRC_NUM_TAPS)
    sf_samples = int(round(SUPERFRAME_S * _TX_RATE))   # 90 000
    tx_buf = np.zeros(sf_samples * n_slots, dtype=np.complex64)
    for i in range(n_slots):
        slot = build_ira_slot(rrc, frame_count=i, sat_id=sat_id, beam_id=beam_id)
        n = min(len(slot), sf_samples)
        tx_buf[i * sf_samples: i * sf_samples + n] = slot[:n]
    return tx_buf


def build_cw_buffer(
    offset_hz: float = 0.0,
    duration_s: float = 0.1,
    sample_rate: int = _TX_RATE,
) -> np.ndarray:
    """
    Generate a CW IQ buffer.

    offset_hz = 0       : pure carrier at DC (I=1, Q=0).
    offset_hz = 100_000 : complex tone at LO + 100 kHz (pilot mode).

    The buffer is intended to be repeated in hardware via transmit_cyclic().
    """
    n = int(round(sample_rate * duration_s))
    if offset_hz == 0.0:
        iq = np.ones(n, dtype=np.complex64) * (0.9 * 2**14)
    else:
        t = np.arange(n, dtype=np.float64)
        iq = (0.9 * 2**14 * np.exp(
            2j * np.pi * offset_hz / sample_rate * t
        )).astype(np.complex64)
    return iq


# =============================================================================
# High-level one-call transmit functions
# =============================================================================

def transmit_cw(
    freq_hz: int = 868_100_000,
    gain_db: float = -20.0,
    pilot_offset_hz: int = 100_000,
    uri: str = DEFAULT_URI,
    cyclic: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Transmit a CW / pilot-tone beacon.

    Tries hardware DDS first (no software buffer needed); falls back to a
    software push loop.

    Args:
        freq_hz:          TX LO frequency [Hz].
        gain_db:          TX attenuation [dB].  Start at −30 dB in lab!
        pilot_offset_hz:  Tone offset from LO [Hz].
                          0 = plain carrier.
                          100_000 = KrakenSDR pilot-tone default (recommended).
        uri:              LibreSDR IIO URI (default ip:192.168.1.10).
        cyclic:           If True, run until Ctrl+C.
        dry_run:          Print config and exit without transmitting.
    """
    print("=" * 55)
    print("  LibreSDR — CW beacon")
    print("=" * 55)
    print(f"  Frequency      : {freq_hz / 1e6:.3f} MHz")
    mode_str = (f"pilot tone +{pilot_offset_hz / 1e3:.0f} kHz"
                if pilot_offset_hz else "plain carrier at DC")
    print(f"  TX mode        : {mode_str}")
    print(f"  TX gain        : {gain_db:+.1f} dB")

    tx_buf = build_cw_buffer(offset_hz=float(pilot_offset_hz))

    if dry_run:
        print("  [DRY RUN]  No transmission.")
        return

    with Ad9363.connect(uri) as sdr:
        sdr.configure_tx(freq_hz, gain_db)

        # Prefer hardware DDS (no Python overhead, tone is rock-stable)
        dds_ok = sdr.configure_dds_cw(offset_hz=pilot_offset_hz)
        print()
        if dds_ok:
            print(f"  [DDS] Hardware tone active at +{pilot_offset_hz / 1e3:.0f} kHz.")
        else:
            sdr._sdr.tx_cyclic_buffer = False
            sdr._sdr.tx(tx_buf)
            print("  [SW] CW active via software push loop.")

        print("  Press Ctrl+C to stop.")
        try:
            while True:
                import time as _time
                if dds_ok:
                    _time.sleep(2.0)
                    try:
                        _ = sdr._sdr.tx_lo   # keepalive: prevents iiod closing idle connection
                    except Exception:
                        pass
                else:
                    sdr._sdr.tx(tx_buf)     # re-push at natural DMA rate
        except KeyboardInterrupt:
            print("\n  Stop.")


def transmit_ira(
    freq_hz: int = 868_100_000,
    gain_db: float = -30.0,
    n_slots: int = 4,
    sat_id: int = 47,
    beam_id: int = 3,
    cyclic: bool = False,
    uri: str = DEFAULT_URI,
    dry_run: bool = False,
) -> None:
    """
    Transmit authentic Iridium IRA bursts (π/4-DQPSK, β=0.4, 25 ksps).

    Args:
        freq_hz:  TX LO frequency [Hz].
                  Default: 868.1 MHz (ISM, safe for lab).
                  For actual Iridium: 1_626_270_000 (cable + ≥ 30 dB attenuator!).
        gain_db:  TX attenuation [dB].  Start at −60 dB for first connection.
        n_slots:  Number of IRA slots to transmit (spacing = 90 ms each).
        sat_id:   Satellite ID encoded in the frame payload (0–127).
        beam_id:  Beam ID encoded in the frame payload (0–47).
        cyclic:   If True, repeat the buffer continuously until Ctrl+C.
        uri:      LibreSDR IIO URI.
        dry_run:  Print config without transmitting.
    """
    print("=" * 60)
    print("  LibreSDR — Iridium IRA burst TX")
    print("=" * 60)
    print(f"  Frequency   : {freq_hz / 1e6:.3f} MHz")
    print(f"  Preamble    : {IRA_PREAMBLE_SYMS} syms → tone at +{SYMBOL_RATE // 8} Hz")
    print(f"  Burst syms  : {IRA_BURST_SYMS}  (preamble + UW + data + tail)")
    print(f"  TDMA period : {SUPERFRAME_S * 1e3:.0f} ms/slot")
    print(f"  TX gain     : {gain_db:+.0f} dB")
    print(f"  Slots       : {n_slots}{'  [CYCLIC]' if cyclic else ''}")

    print(f"\n  Generating {n_slots} IRA slot(s)...")
    tx_buf = build_ira_buffer(n_slots=n_slots, sat_id=sat_id, beam_id=beam_id)

    dur_ms = len(tx_buf) / _TX_RATE * 1e3
    peak   = float(np.max(np.abs(tx_buf)))
    print(f"  TX buffer   : {len(tx_buf)} samples  ({dur_ms:.0f} ms)  "
          f"DAC peak = {peak:.0f}/16384")

    if dry_run:
        print("  [DRY RUN]  No transmission.")
        return

    with Ad9363.connect(uri) as sdr:
        sdr.configure_tx(freq_hz, gain_db)
        print()
        if cyclic:
            sdr.transmit_cyclic(tx_buf)
        else:
            import time as _time
            print(f"  Transmitting {n_slots} slot(s)...")
            t0 = _time.time()
            sdr.transmit_once(tx_buf)
            print(f"  Completed in {(_time.time() - t0) * 1e3:.0f} ms")

    print("Done.")


def transmit_pass(
    freq_hz: int = 868_100_000,
    gain_db: float = -20.0,
    max_elev_deg: float = 45.0,
    pass_dur_s: float = 60.0,
    snr_db: float = 100.0,
    sat_id: int = 47,
    beam_id: int = 3,
    cyclic: bool = False,
    uri: str = DEFAULT_URI,
    dry_run: bool = False,
    save_iq: str | None = None,
) -> None:
    """
    Generate and transmit a realistic simulated Iridium LEO satellite pass.

    The IQ signal includes:
    - IRA bursts every 90 ms (authentic TDMA structure)
    - Doppler chirp modelled as LEO orbital physics (h = 780 km)
    - AWGN at the requested SNR (default 100 dB ≈ noise-free)

    Args:
        freq_hz:      TX carrier frequency [Hz].
        gain_db:      TX attenuation [dB].
        max_elev_deg: Maximum elevation angle of the simulated pass [°].
        pass_dur_s:   Duration to generate/transmit [s].
        snr_db:       SNR of the generated signal [dB].
                      100 = effectively noise-free (useful for clean DoA tests).
        sat_id:       Satellite ID in payload.
        beam_id:      Beam ID in payload.
        cyclic:       If True, cycle the buffer until Ctrl+C.
        uri:          LibreSDR IIO URI.
        dry_run:      Generate IQ but do not transmit.
        save_iq:      Path to save the raw complex64 IQ file (None = don't save).
    """
    print("=" * 60)
    print("  LibreSDR — Simulated Iridium LEO pass")
    print("=" * 60)
    print(f"  Frequency   : {freq_hz / 1e6:.3f} MHz")
    print(f"  Max elev    : {max_elev_deg:.0f}°")
    print(f"  Duration    : {pass_dur_s:.0f} s")
    print(f"  SNR         : {snr_db:.0f} dB")
    print(f"  TX gain     : {gain_db:+.0f} dB")

    print("\n  Generating IQ ...")
    iq_base, burst_log, _, _ = simulate_iridium_pass(
        duration_s=pass_dur_s,
        carrier_hz=float(freq_hz),
        max_elev_deg=max_elev_deg,
        snr_db=snr_db,
        sat_id=sat_id,
        beam_id=beam_id,
    )
    print(f"  {len(burst_log)} bursts generated, "
          f"{len(iq_base)} samples ({len(iq_base) / _IRA_BASE_RATE:.1f} s @ 250 kHz)")

    # Resample 250 kHz → 1 MHz (×4)
    print("  Resampling 250 kHz → 1 MHz ...", end=" ", flush=True)
    iq_tx = sp_signal.resample_poly(iq_base, _IRA_UPS, 1)
    # Scale for DAC
    peak = float(np.max(np.abs(iq_tx))) + 1e-12
    iq_tx = (iq_tx / peak * 0.8 * 2**14).astype(np.complex64)
    print("OK")
    print(f"  TX buffer   : {len(iq_tx)} samples  "
          f"({len(iq_tx) / _TX_RATE:.1f} s @ 1 MSPS)  "
          f"DAC peak = {float(np.max(np.abs(iq_tx))):.0f}/16384")

    if save_iq:
        iq_tx.tofile(save_iq)
        print(f"  IQ saved → {save_iq}")

    if dry_run:
        print("  [DRY RUN]  No transmission.")
        return

    with Ad9363.connect(uri) as sdr:
        sdr.configure_tx(freq_hz, gain_db)
        print()
        if cyclic:
            sdr.transmit_cyclic(iq_tx)
        else:
            print("  Transmitting (one-shot, full pass) ...")
            import time as _time
            t0 = _time.time()
            sdr.transmit_once(iq_tx)
            print(f"  Completed in {(_time.time() - t0):.1f} s")

    print("Done.")


__all__ = [
    "transmit_cw",
    "transmit_ira",
    "transmit_pass",
    "build_ira_buffer",
    "build_cw_buffer",
    "build_ira_slot",
]
