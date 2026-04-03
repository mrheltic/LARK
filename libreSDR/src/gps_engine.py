#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gps_engine.py — GPS L1 C/A TX Encoder / Signal Generator

Provides TX-side building blocks for GPS L1 C/A signal generation:
  - Gold code (C/A PRN) generation for all 32 GPS satellites
  - Navigation message encoder (subframes 1-5 with parity)
  - GPS baseband signal generator (code × nav × carrier)

For RX decoding (acquisition, tracking, telemetry, PVT), use gnss-sdr:
  python3 scripts/gnss_decoder.py --live    # real-time from AD9363
  python3 scripts/gnss_decoder.py --file X  # from IQ recording

Hardware target: LibreSDR (Zynq7020 + AD9363)

References:
  - IS-GPS-200 (GPS Interface Control Document)
  - gnss-sdr: https://github.com/gnss-sdr/gnss-sdr
"""

import numpy as np
from typing import Tuple, Optional, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GPS_L1_FREQ       = 1575.42e6       # GPS L1 center frequency (Hz)
GALILEO_E1_FREQ   = 1575.42e6       # Galileo E1 center frequency (Hz)
GLONASS_L1_FREQ   = 1602.00e6       # GLONASS L1 center frequency (Hz)
BEIDOU_B1_FREQ    = 1561.098e6      # BeiDou B1I center frequency (Hz)

CA_CODE_RATE      = 1.023e6         # C/A code chip rate (chips/s)
CA_CODE_LEN       = 1023            # C/A code length (chips)
NAV_BIT_RATE      = 50              # Navigation message bit rate (bps)
CHIPS_PER_BIT     = 20              # C/A code epochs per navigation bit
CODE_PERIOD_S     = 1e-3            # One code period = 1 ms

LIGHT_SPEED       = 299792458.0     # Speed of light (m/s)
GPS_PI            = 3.1415926535898 # GPS-defined value of pi
MU_EARTH          = 3.986005e14     # Earth gravitational constant (m^3/s^2)
OMEGA_E           = 7.2921151467e-5 # Earth rotation rate (rad/s)

# GPS C/A code G2 tap assignments for PRN 1–32 (IS-GPS-200 Table 3-Ia)
G2_TAPS = {
     1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),
     5: (1, 9),   6: (2, 10),  7: (1, 8),   8: (2, 9),
     9: (3, 10), 10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7),  14: (7, 8),  15: (8, 9),  16: (9, 10),
    17: (1, 4),  18: (2, 5),  19: (3, 6),  20: (4, 7),
    21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7),  26: (6, 8),  27: (7, 9),  28: (8, 10),
    29: (1, 6),  30: (2, 7),  31: (3, 8),  32: (4, 9),
}

# GPS TLM preamble
GPS_TLM_PREAMBLE      = 0x8B
GPS_TLM_PREAMBLE_BITS = np.array([1, 0, 0, 0, 1, 0, 1, 1], dtype=np.int8)


# ═════════════════════════════════════════════════════════════════════════════
# GPS C/A Gold Code Generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_ca_code(prn: int) -> np.ndarray:
    """Generate the 1023-chip GPS C/A Gold code for a given PRN (1-32).

    Uses the standard Gold code construction:
      G1 polynomial: 1 + x^3 + x^10
      G2 polynomial: 1 + x^2 + x^3 + x^6 + x^8 + x^9 + x^10
      Output = G1[10] XOR G2[tap1] XOR G2[tap2]

    Returns: bipolar float32 array of length 1023 (+1.0 / -1.0)
    """
    if prn < 1 or prn > 32:
        raise ValueError(f"PRN must be 1..32, got {prn}")

    tap1, tap2 = G2_TAPS[prn]

    # Initialize both shift registers to all ones
    g1 = np.ones(10, dtype=np.int8)
    g2 = np.ones(10, dtype=np.int8)

    code = np.zeros(CA_CODE_LEN, dtype=np.float32)

    for i in range(CA_CODE_LEN):
        # Output chip: G1[9] XOR G2[tap1-1] XOR G2[tap2-1]
        g2_out = g2[tap1 - 1] ^ g2[tap2 - 1]
        chip = g1[9] ^ g2_out
        code[i] = 1.0 - 2.0 * chip   # 0 → +1, 1 → -1

        # G1 feedback: taps 3, 10 (indices 2, 9)
        g1_fb = g1[2] ^ g1[9]
        g1[1:] = g1[:-1]
        g1[0] = g1_fb

        # G2 feedback: taps 2, 3, 6, 8, 9, 10 (indices 1,2,5,7,8,9)
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g2[1:] = g2[:-1]
        g2[0] = g2_fb

    return code


def generate_all_ca_codes() -> Dict[int, np.ndarray]:
    """Generate C/A codes for all 32 GPS PRNs."""
    return {prn: generate_ca_code(prn) for prn in range(1, 33)}


def upsample_code(code: np.ndarray, samples_per_chip: int) -> np.ndarray:
    """Upsample C/A code by repeating each chip (rectangular pulse shaping)."""
    return np.repeat(code, samples_per_chip)


# ═════════════════════════════════════════════════════════════════════════════
# Navigation Message Encoder
# ═════════════════════════════════════════════════════════════════════════════

def _int_to_bits(val: int, nbits: int) -> np.ndarray:
    """Convert integer to binary array (MSB first)."""
    bits = np.zeros(nbits, dtype=np.int8)
    for i in range(nbits):
        bits[nbits - 1 - i] = (val >> i) & 1
    return bits


def _bits_to_int(bits, signed=False) -> int:
    """Convert bit array to integer."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    if signed and len(bits) > 0 and bits[0]:
        val -= (1 << len(bits))
    return val


def _gps_parity(d: np.ndarray, D29_star: int, D30_star: int) -> np.ndarray:
    """Compute 6 parity bits for a 24-bit GPS nav data word.

    Implements IS-GPS-200 Table 20-XIV parity equations.

    Args:
        d: 24 source data bits (d1..d24), already complemented if needed
        D29_star: bit 29 of previous word
        D30_star: bit 30 of previous word

    Returns: 6 parity bits [D25..D30]
    """
    D25 = (D29_star ^ d[0] ^ d[1] ^ d[2] ^ d[4] ^ d[5] ^
           d[9] ^ d[10] ^ d[11] ^ d[12] ^ d[13] ^
           d[16] ^ d[17] ^ d[19] ^ d[21]) & 1
    D26 = (D30_star ^ d[1] ^ d[2] ^ d[3] ^ d[5] ^ d[6] ^
           d[10] ^ d[11] ^ d[12] ^ d[13] ^ d[14] ^
           d[17] ^ d[18] ^ d[20] ^ d[22]) & 1
    D27 = (D29_star ^ d[0] ^ d[2] ^ d[3] ^ d[4] ^ d[6] ^ d[7] ^
           d[11] ^ d[12] ^ d[13] ^ d[14] ^ d[15] ^
           d[18] ^ d[19] ^ d[21] ^ d[23]) & 1
    D28 = (D30_star ^ d[1] ^ d[3] ^ d[4] ^ d[5] ^ d[7] ^ d[8] ^
           d[12] ^ d[13] ^ d[14] ^ d[15] ^ d[16] ^
           d[19] ^ d[20] ^ d[22]) & 1
    D29 = (D30_star ^ d[0] ^ d[2] ^ d[4] ^ d[5] ^ d[6] ^ d[8] ^ d[9] ^
           d[13] ^ d[14] ^ d[15] ^ d[16] ^ d[17] ^
           d[20] ^ d[21] ^ d[23]) & 1
    D30 = (D29_star ^ d[2] ^ d[4] ^ d[5] ^ d[7] ^ d[8] ^ d[9] ^ d[10] ^
           d[12] ^ d[14] ^ d[18] ^ d[21] ^ d[22] ^ d[23]) & 1

    return np.array([D25, D26, D27, D28, D29, D30], dtype=np.int8)


def encode_nav_word(data_24: np.ndarray, D29_star: int, D30_star: int) -> np.ndarray:
    """Encode a 30-bit GPS navigation word from 24 data bits + 6 parity bits.

    If D30_star == 1, complements data bits before parity computation
    (IS-GPS-200 §20.3.5).
    """
    d = np.array(data_24, dtype=np.int8).copy()
    if D30_star:
        d = 1 - d
    parity = _gps_parity(d, D29_star, D30_star)
    word = np.concatenate([data_24, parity])
    return word.astype(np.int8)


def encode_tlm_word(tlm_message: int = 0,
                    D29_star: int = 0, D30_star: int = 0) -> np.ndarray:
    """Encode TLM (Telemetry) word — Word 1 of every subframe.

    Preamble: 10001011 (0x8B) in bits 1-8.
    """
    data = np.zeros(24, dtype=np.int8)
    data[0:8] = _int_to_bits(GPS_TLM_PREAMBLE, 8)
    data[8:22] = _int_to_bits(tlm_message & 0x3FFF, 14)
    return encode_nav_word(data, D29_star, D30_star)


def encode_how_word(tow: int, subframe_id: int, alert: int = 0,
                    antispoof: int = 0,
                    D29_star: int = 0, D30_star: int = 0) -> np.ndarray:
    """Encode HOW (Handover Word) — Word 2 of every subframe.

    Args:
        tow: Time of Week count (6-second epochs, 17 bits stored as TOW >> 2)
        subframe_id: Subframe ID (1-5)
    """
    data = np.zeros(24, dtype=np.int8)
    tow_17 = tow & 0x1FFFF
    data[0:17] = _int_to_bits(tow_17, 17)
    data[17] = alert & 1
    data[18] = antispoof & 1
    data[19:22] = _int_to_bits(subframe_id & 0x7, 3)
    return encode_nav_word(data, D29_star, D30_star)


def encode_subframe(subframe_id: int, tow: int,
                    data_words: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Encode a complete 300-bit GPS subframe (10 words × 30 bits).

    Args:
        subframe_id: 1-5
        tow: Time of Week count for HOW
        data_words: 8 arrays of 24 data bits each (words 3-10).
                    Defaults to zeros.

    Returns: 300-element int8 array
    """
    if data_words is None:
        data_words = [np.zeros(24, dtype=np.int8) for _ in range(8)]

    subframe = np.zeros(300, dtype=np.int8)

    D29s, D30s = 0, 0

    # Word 1: TLM
    tlm = encode_tlm_word(D29_star=D29s, D30_star=D30s)
    subframe[0:30] = tlm
    D29s, D30s = int(tlm[28]), int(tlm[29])

    # Word 2: HOW
    how = encode_how_word(tow, subframe_id, D29_star=D29s, D30_star=D30s)
    subframe[30:60] = how
    D29s, D30s = int(how[28]), int(how[29])

    # Words 3-10
    for i in range(8):
        word = encode_nav_word(data_words[i], D29s, D30s)
        offset = 60 + i * 30
        subframe[offset:offset + 30] = word
        D29s, D30s = int(word[28]), int(word[29])

    return subframe


def encode_nav_frame(tow_start: int = 0) -> np.ndarray:
    """Encode a complete GPS navigation frame (5 subframes = 1500 bits = 30 s)."""
    frame = np.zeros(1500, dtype=np.int8)
    for sf_id in range(1, 6):
        tow = tow_start + (sf_id - 1)
        sf = encode_subframe(sf_id, tow)
        frame[(sf_id - 1) * 300 : sf_id * 300] = sf
    return frame


def nav_bits_to_bipolar(nav_bits: np.ndarray) -> np.ndarray:
    """Convert 0/1 navigation bits to ±1 bipolar format."""
    return 1.0 - 2.0 * nav_bits.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# GPS Signal Generator (TX Encoder)
# ═════════════════════════════════════════════════════════════════════════════

def generate_gps_baseband(prn: int, samp_rate: float, duration_s: float,
                          nav_bits: Optional[np.ndarray] = None,
                          doppler_hz: float = 0.0,
                          code_phase_chips: float = 0.0,
                          amplitude: float = 1.0) -> np.ndarray:
    """Generate a GPS L1 C/A complex baseband signal.

    This mirrors gnss-sdr's signal_generator block.

    Args:
        prn: Satellite PRN number (1-32)
        samp_rate: Sample rate (Hz)
        duration_s: Duration (seconds)
        nav_bits: Navigation data bits (0/1). Auto-generated if None.
        doppler_hz: Doppler frequency shift (Hz)
        code_phase_chips: Initial code phase offset (chips)
        amplitude: Signal amplitude (0.0-1.0)

    Returns: complex64 baseband signal array
    """
    n_samples = int(samp_rate * duration_s)
    ca_code = generate_ca_code(prn)

    if nav_bits is None:
        n_bits = int(np.ceil(duration_s * NAV_BIT_RATE)) + 1
        frame = encode_nav_frame(tow_start=100)
        # Repeat frame to fill duration
        n_frames = int(np.ceil(n_bits / 1500)) + 1
        nav_bits = np.tile(frame, n_frames)[:n_bits]

    nav_bipolar = nav_bits_to_bipolar(nav_bits)

    # Time array
    t = np.arange(n_samples, dtype=np.float64) / samp_rate

    # C/A code chip indices
    chip_phase = (t * CA_CODE_RATE + code_phase_chips) % CA_CODE_LEN
    code_signal = ca_code[chip_phase.astype(int)]

    # Navigation data modulation: each bit spans 20 ms (20 code epochs)
    samples_per_bit = int(samp_rate / NAV_BIT_RATE)
    nav_signal = np.zeros(n_samples, dtype=np.float32)
    for i in range(min(len(nav_bipolar), (n_samples // samples_per_bit) + 1)):
        s = i * samples_per_bit
        e = min(s + samples_per_bit, n_samples)
        nav_signal[s:e] = nav_bipolar[i]

    # Spread-spectrum: code × data
    spread = code_signal * nav_signal

    # Doppler shift
    carrier = np.exp(1j * 2 * np.pi * doppler_hz * t).astype(np.complex64)

    signal = (amplitude * spread * carrier).astype(np.complex64)
    return signal


# ═════════════════════════════════════════════════════════════════════════════
