#!/usr/bin/env python3
"""
iridium_burst_gen — Iridium-like burst generator library.

Generates baseband π/4-DQPSK bursts with Root Raised-Cosine pulse shaping.
Used as a pure DSP library by iridium_tx_libresdr.py.

Iridium system parameters (public specifications):
  Band          : 1616–1626.5 MHz  (L-band)
  Channel width : ~41.667 kHz
  Symbol rate   : 25 ksps
  Modulation    : π/4-DQPSK
  Pulse shape   : Root Raised Cosine (β=0.35)
  TDMA frame    : ~90 ms, up to 4 time slots
"""

import numpy as np
from scipy import signal as sp_signal


# ── Iridium-like parameters ──────────────────────────────────────────────
SYMBOL_RATE = 25_000          # 25 ksps
SAMPLES_PER_SYMBOL = 8        # oversampling
SAMPLE_RATE = SYMBOL_RATE * SAMPLES_PER_SYMBOL  # 200 kHz
RRC_BETA = 0.35               # RRC roll-off
RRC_NUM_TAPS = 101            # RRC filter length

# Burst structure (in symbols)
PREAMBLE_LEN = 64             # preamble symbols (alternating pattern)
UNIQUE_WORD_LEN = 12          # unique word symbols (frame synchronization)
HEADER_LEN = 12               # burst header
PAYLOAD_LEN = 156             # data payload
GUARD_SYMBOLS = 20            # guard time between bursts (silence)

# Known Unique Word (12 dibits → 24 bits). These dibits, applied after a
# 64-symbol all-zero preamble (ending at absolute phase 0°), produce the
# π/4-DQPSK symbol-index sequence "022220002002" = UW_DOWNLINK recognised
# by iridium-parser / bitsparser and validated by iridium_doa_burst.validate_burst_uw.
#
# Mapping (burst_gen transmitter → receiver dibit):
#   bit pair (0,0) → phase +π/4      → dibit 0
#   bit pair (0,1) → phase +3π/4     → dibit 1
#   bit pair (1,0) → phase  −π/4     → dibit 3
#   bit pair (1,1) → phase −3π/4     → dibit 2   ← used for '2' in UW_DL
#
# UW_DL = [0,2,2,2,2,0,0,0,2,0,0,2]  ↔  pairs (0,0)(1,1)(1,1)(1,1)(1,1)(0,0)(0,0)(0,0)(1,1)(0,0)(0,0)(1,1)
UNIQUE_WORD_BITS = np.array([
    0, 0,  1, 1,  1, 1,
    1, 1,  1, 1,  0, 0,
    0, 0,  0, 0,  1, 1,
    0, 0,  0, 0,  1, 1,
], dtype=np.int8)


def generate_rrc_filter(beta, sps, num_taps):
    """Generate Root Raised Cosine (RRC) filter."""
    t = np.arange(num_taps) - (num_taps - 1) // 2
    t = t / sps  # normalize to symbol period

    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + \
                  4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den

    h /= np.sqrt(np.sum(h**2))  # normalize energy
    return h


def dqpsk_modulate(bits):
    """
    DQPSK modulation: maps bit pairs into differential phase rotations.

    Dibit → phase rotation mapping:
      00 → +π/4
      01 → +3π/4
      10 → -π/4
      11 → -3π/4
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    dibits = bits.reshape(-1, 2)
    num_symbols = len(dibits)

    # Map dibit → phase rotation (Gray coding)
    phase_map = {
        (0, 0): np.pi / 4,
        (0, 1): 3 * np.pi / 4,
        (1, 0): -np.pi / 4,
        (1, 1): -3 * np.pi / 4,
    }

    # Initial reference phase
    phase = 0.0
    symbols = np.zeros(num_symbols, dtype=np.complex128)

    for i in range(num_symbols):
        dibit_key = (int(dibits[i, 0]), int(dibits[i, 1]))
        delta_phase = phase_map[dibit_key]
        phase += delta_phase
        symbols[i] = np.exp(1j * phase)

    return symbols


def generate_preamble(length):
    """
    Generate Iridium preamble: all-zero dibits produce a +π/4 rotation per
    symbol, resulting in a pilot tone at +Rs/8 = +3 125 Hz above the carrier.
    After `length` symbols the cumulative phase returns to 0° (mod 2π for
    length that is a multiple of 8), so the last preamble symbol acts as a
    valid DQPSK reference (absolute phase = 0°) for the UW decoder.
    """
    return np.zeros(length * 2, dtype=np.int8)


def generate_burst(payload_bits=None, burst_type="data"):
    """
    Generate a single Iridium-like burst. 

    Structure:
      [Preamble | Unique Word | Header | Payload]

    Args:
        payload_bits: payload bits (if None, generated randomly)
        burst_type: "data" (simplex data) or "ring_alert" (shorter burst)

    Returns:
        symbols: array of complex DQPSK symbols
        sections: dict with indices of each burst section
    """
    if burst_type == "ring_alert":
        preamble_len = 32   # Ring Alert has a shorter preamble
        payload_len = 48
    else:
        preamble_len = PREAMBLE_LEN
        payload_len = PAYLOAD_LEN

    # Preamble
    preamble_bits = generate_preamble(preamble_len)

    # Unique Word
    uw_bits = UNIQUE_WORD_BITS.copy()

    # Header (simplified: burst type + channel)
    header_bits = np.random.randint(0, 2, HEADER_LEN * 2).astype(np.int8)

    # Payload
    if payload_bits is None:
        payload_bits = np.random.randint(0, 2, payload_len * 2).astype(np.int8)
    else:
        payload_bits = np.array(payload_bits, dtype=np.int8)

    # Concatenate all bits
    all_bits = np.concatenate([preamble_bits, uw_bits, header_bits, payload_bits])

    # DQPSK modulate
    symbols = dqpsk_modulate(all_bits)

    sections = {
        "preamble": (0, preamble_len),
        "unique_word": (preamble_len, preamble_len + UNIQUE_WORD_LEN),
        "header": (preamble_len + UNIQUE_WORD_LEN,
                   preamble_len + UNIQUE_WORD_LEN + HEADER_LEN),
        "payload": (preamble_len + UNIQUE_WORD_LEN + HEADER_LEN,
                    len(symbols)),
    }

    return symbols, sections


def apply_pulse_shaping(symbols, rrc_filter, sps):
    """
    Apply RRC pulse shaping: upsampling + filtering.

    1. Insert zeros between symbols (upsampling)
    2. Convolve with RRC filter
    """
    # Upsampling: insert sps-1 zeros between each symbol
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols

    # Filter with RRC
    shaped = np.convolve(upsampled, rrc_filter, mode="same")
    return shaped


def add_frequency_offset(signal, freq_offset_hz, sample_rate):
    """Add a frequency offset (simulates Doppler shift or channel offset)."""
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(1j * 2 * np.pi * freq_offset_hz * t)


def generate_tdma_frame(num_bursts=4, burst_types=None, freq_offsets=None):
    """
    Generate a TDMA frame with multiple bursts, Iridium-like.

    An Iridium frame is ~90 ms and contains up to 4 timeslots.

    Args:
        num_bursts: number of bursts in the frame
        burst_types: list of burst types (default: all "data")
        freq_offsets: per-burst frequency offsets (Hz), to simulate FDMA

    Returns:
        frame_signal: complex IQ signal of the entire frame
        burst_info: list of dicts with info for each burst
    """
    if burst_types is None:
        burst_types = ["data"] * num_bursts
    if freq_offsets is None:
        freq_offsets = [0.0] * num_bursts

    rrc = generate_rrc_filter(RRC_BETA, SAMPLES_PER_SYMBOL, RRC_NUM_TAPS)

    frame_signal = np.array([], dtype=np.complex128)
    burst_info = []

    for i in range(num_bursts):
        # Generate burst
        symbols, sections = generate_burst(burst_type=burst_types[i])

        # Pulse shaping
        shaped = apply_pulse_shaping(symbols, rrc, SAMPLES_PER_SYMBOL)

        # Frequency offset (FDMA)
        if freq_offsets[i] != 0.0:
            shaped = add_frequency_offset(shaped, freq_offsets[i], SAMPLE_RATE)

        # Guard time (silence between bursts)
        guard = np.zeros(GUARD_SYMBOLS * SAMPLES_PER_SYMBOL, dtype=np.complex128)

        # Save burst info
        start_sample = len(frame_signal)
        frame_signal = np.concatenate([frame_signal, shaped, guard])

        burst_info.append({
            "index": i,
            "type": burst_types[i],
            "start_sample": start_sample,
            "end_sample": start_sample + len(shaped),
            "num_symbols": len(symbols),
            "freq_offset": freq_offsets[i],
            "sections": sections,
        })

    return frame_signal, burst_info





def save_iq_file(signal: np.ndarray, filepath: str) -> None:
    """Save IQ signal to a binary complex64 file (standard SDR format)."""
    signal.astype(np.complex64).tofile(filepath)


# ── kept for backwards compat placeholder ─────────────────────────────────────
def _plot_burst_stub(*_args, **_kwargs):  # pragma: no cover
    raise RuntimeError(
        "plot_burst() has been removed from iridium_burst_gen. "
        "Visualisation is out of scope for the TX library."
    )


def _deleted_stub(*_args, **_kwargs):
    raise RuntimeError(
        "add_channel_effects() has been removed from iridium_burst_gen. "
        "Channel simulation is out of scope for the TX library."
    )


# Tombstone names so that old callers get a clear error instead of AttributeError
plot_burst = _plot_burst_stub
add_channel_effects = _deleted_stub


