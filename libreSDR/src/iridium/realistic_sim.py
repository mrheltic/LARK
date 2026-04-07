#!/usr/bin/env python3
"""
iridium_realistic_sim.py — Faithful simulation of Iridium downlink from a single satellite

Implements the REAL Iridium system parameters from public sources:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Sources: gr-iridium (muccc/gr-iridium), iridium-toolkit         │
  │          (muccc/iridium-toolkit), SDR community analysis        │
  │          Motorola patents / public ITU filings                  │
  └─────────────────────────────────────────────────────────────────┘

PHYSICAL PARAMETERS:
  - Modulation:        π/4-DQPSK  (Differential QPSK rotated 45°)
  - Symbol rate:       25.000 sps (confirmed by gr-iridium)
  - RRC roll-off β:    0.4        (confirmed by gr-iridium/community)
  - Operating band:    1616 – 1626.5 MHz (L-band)
  - Base frequency:    1,615,604,164 Hz  (from iridium-toolkit util.py)
  - Channel spacing:   41,667 Hz  (FDMA, from iridium-toolkit)

TDMA / IRA BURST STRUCTURE (Ring Alert):
  - Superframe:        90 ms  →  8 slots  →  1 slot = 11.25 ms = 281 symbols
  - Guard pre-burst:   8 symbols  (0.32 ms)
  - Preamble run-in:   32 symbols (1.28 ms) — constant +45° phase rotation (all 0x00)
                       → produces a tone at  fc + 3125 Hz  (Rs/8)
                       → THIS IS THE SIGNAL TO DETECT
  - Unique Word (UW):  12 symbols (24 bits) — from gr-iridium README footnote 2
                       ("12-symbol BPSK Iridium sync word")
  - Frame data:        167 symbols (334 bits) — length confirmed by gr-iridium output
                       (179 symbols output = 12 UW + 167 data)
  - Guard post-burst:  8 symbols  (0.32 ms)
  - Silence:           54 symbols to complete the 281-symbol slot

LEO DOPPLER MODEL:
  - Orbit: circular at h = 780 km (inclination 86.4°, Iridium classic/NEXT)
  - Orbital velocity: ~7464 m/s   (derived from μ_Earth/r_orbit)
  - Max Doppler:       ±40.2 kHz   (at 1621 MHz, overhead pass)
  - Chirp rate:        ~386 Hz/s   (maximum at closest approach, overhead pass)
  - Geometric model:   hyperbolic → r(t) = √(r_min² + v_sat² × (t−t_ca)²)
    where t_ca = instant of closest approach (maximum elevation)

Usage:
  python3 scripts/iridium_realistic_sim.py                   # IQ file + plot
  python3 scripts/iridium_realistic_sim.py --pass-dur 30     # 30 s pass
  python3 scripts/iridium_realistic_sim.py --elev 45         # max elev. 45°
  python3 scripts/iridium_realistic_sim.py --pass-dur 60     # transmit via LibreSDR (TX on by default)
  python3 scripts/iridium_realistic_sim.py --no-tx --save out.iq  # generate IQ without TX
  python3 scripts/iridium_realistic_sim.py --detect          # show correlator

LEGAL NOTE: transmitting in the Iridium band (1616-1626.5 MHz) without a licence
            is illegal. Use a wired RF connection (TX→attenuator→RX) or an
            authorised ISM frequency for lab tests.
"""

import argparse
import sys
import os
import time
import numpy as np
from scipy import signal as sp_signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical constants ──────────────────────────────────────────────────────────
C          = 2.99792458e8    # speed of light [m/s]
MU_EARTH   = 3.986004418e14  # Earth gravitational parameter [m³/s²]
R_EARTH    = 6.3710e6        # mean Earth radius [m]

# ── Iridium orbital parameters ──────────────────────────────────────────────────
IRIDIUM_ALT_M     = 780_000.0   # altitude [m]
IRIDIUM_INCL_DEG  = 86.4        # inclination [°]
# Derived orbital velocity: v = √(μ/r)
_r_orbit = R_EARTH + IRIDIUM_ALT_M
V_SAT     = np.sqrt(MU_EARTH / _r_orbit)   # ≈ 7464 m/s

# ── Physical signal parameters (all sources agree) ────────────────────────
SYMBOL_RATE  = 25_000      # [sps] — confirmed by gr-iridium
SPS          = 10          # samples/symbol — gr-iridium default (250 ksps total)
SAMPLE_RATE  = SYMBOL_RATE * SPS   # 250,000 Hz
RRC_BETA     = 0.4         # RRC roll-off — from gr-iridium / community analysis
RRC_NUM_TAPS = 11 * SPS + 1  # 111 taps per β=0.4, sps=10

# ── Iridium frequency plan (from iridium-toolkit/util.py) ────────────────────────
# frequency = 1_615_604_164 + (FA + 8×SB) × 41_667 + offset
IRDM_FREQ_BASE    = 1_615_604_164   # [Hz]
IRDM_CHAN_SPACING  = 41_667          # [Hz]
# IRA channel computed from iridium-toolkit formula (SB=0, FA=8)
IRDM_IRA_FREQ     = IRDM_FREQ_BASE + 8 * IRDM_CHAN_SPACING   # ≈ 1.616 GHz (formula only)
# Actual Iridium Ring Alert / Simplex channel used by gr-iridium and iridium-toolkit
IRDM_RING_ALERT_HZ = 1_626_270_000   # 1626.270 MHz — matches SIMPLEX_RING_CH_HZ

# ── TDMA structure ────────────────────────────────────────────────────────────
# Superframe = 90 ms, 8 slots, 1 slot = 281.25 symbols → rounded to 281
SUPERFRAME_S      = 0.090     # 90 ms
SLOTS_PER_FRAME   = 8
SLOT_SYMS         = int(SUPERFRAME_S * SYMBOL_RATE / SLOTS_PER_FRAME)  # 281

# ── IRA burst structure ───────────────────────────────────────────────────────
# Fonte: gr-iridium README (footnote 2: "12-symbol BPSK Iridium sync word")
#        iridium-toolkit: output frame length 179 symbols (12 UW + 167 data)
IRA_GUARD_SYMS     = 8    # silence before burst
IRA_PREAMBLE_SYMS  = 64   # PREAMBLE_LENGTH_LONG — gr-iridium/include/iridium/iridium.h
IRA_UW_SYMS        = 12   # UW_LENGTH — gr-iridium/include/iridium/iridium.h
IRA_DATA_SYMS      = 167  # payload — gr-iridium output: 179 total = UW+data
IRA_TAIL_SYMS      = 2    # tail bits (flush encoder K=7)
IRA_POST_GUARD     = 8    # silence after burst
# Total burst symbols (excluding silence):
IRA_BURST_SYMS     = IRA_PREAMBLE_SYMS + IRA_UW_SYMS + IRA_DATA_SYMS + IRA_TAIL_SYMS
# = 64 + 12 + 167 + 2 = 245 symbols → 9.80 ms

# The remainder of a slot is filled with silence:
IRA_SLOT_SILENCE   = SLOT_SYMS - IRA_GUARD_SYMS - IRA_BURST_SYMS - IRA_POST_GUARD
# = 281 - 8 - 245 - 8 = 20 silence symbols post-burst

# ── IRA Unique Word — from gr-iridium/include/iridium/iridium.h ─────────────────
# UW_DL[] = { 0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2 }
# Absolute BPSK symbols: quadrant 0 = NE (45°), quadrant 2 = SW (225°)
# IMPORTANT: the UW does NOT pass through the π/4-DQPSK differential encoder.
#            It is inserted directly as absolute phases in the burst.
_UW_DL_QUADRANTS = np.array([0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2], dtype=np.int8)
UW_DL_SYMBOLS = np.exp(1j * (_UW_DL_QUADRANTS * np.pi / 2 + np.pi / 4))  # 45° or 225°

# ── π/4-DQPSK mapping ──────────────────────────────────────────────────────
# Gray coding: dibit → phase increment (Δφ)
# (0,0) → +π/4     (0,1) → +3π/4
# (1,0) → -π/4     (1,1) → -3π/4
_DQPSK_MAP = {
    (0, 0): +np.pi / 4,
    (0, 1): +3 * np.pi / 4,
    (1, 0): -np.pi / 4,
    (1, 1): -3 * np.pi / 4,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. π/4-DQPSK MODULATION
# ═══════════════════════════════════════════════════════════════════════════════

def pi4_dqpsk_modulate(bits: np.ndarray, initial_phase: float = np.pi / 4) -> np.ndarray:
    """
    Modulatore π/4-DQPSK fedele alle specifiche Iridium.

    The π/4 variant differs from classic DQPSK by the initial phase π/4,
    which causes the transmitted constellation to alternate between two QPSK
    grids rotated by 45°. This prevents transitions through the origin →
    better RF amplifier efficiency in operational conditions.

    Args:
        bits:          Bit array (integers 0/1), must have even length.
        initial_phase: Initial phase in rad (default π/4 for the π/4 variant)

    Returns:
        Array of normalised complex symbols (|s|=1)
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    dibits = bits.reshape(-1, 2)
    num_symbols = len(dibits)

    phase = float(initial_phase)
    symbols = np.zeros(num_symbols, dtype=np.complex128)
    for i in range(num_symbols):
        delta = _DQPSK_MAP[(int(dibits[i, 0]), int(dibits[i, 1]))]
        phase += delta
        symbols[i] = np.exp(1j * phase)
    return symbols


def pi4_dqpsk_demodulate_differential(samples: np.ndarray) -> np.ndarray:
    """
    Demodulatore differenziale: estrae la variazione di fase tra simboli
    consecutivi.  Restituisce un array complesso s[n]×conj(s[n−1]).
    Usato sia per il correlatore del preamble sia per la decodifica.
    """
    if len(samples) < 2:
        return np.array([], dtype=np.complex128)
    return samples[1:] * np.conj(samples[:-1])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RRC FILTER (Root Raised Cosine, β=0.4)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rrc_filter(beta: float, sps: int, num_taps: int) -> np.ndarray:
    """
    Genera il filtro RRC con formula esatta (non approssimata).
    β=0.4 is the value used for Iridium in gr-iridium.
    """
    t = (np.arange(num_taps) - (num_taps - 1) // 2) / float(sps)
    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif abs(abs(2.0 * beta * ti) - 1.0) < 1e-10:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta))
                   + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IRA BURST GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _iridium_scrambler(length: int, seed: int = 0x4A2C) -> np.ndarray:
    """
    Scrambler LFSR semplificato per il payload Iridium.
    The exact polynomial is not publicly documented; we use a 16-bit LFSR
    that produces a pseudo-random sequence with good statistical properties.

    NOTE: the real Iridium payload uses a convolutional code + specific scrambling.
          Here we only use scrambling for spectral density realism. The payload
          structure does not affect preamble/UW detection.
    """
    state = seed & 0xFFFF
    seq = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        bit = ((state >> 15) ^ (state >> 4) ^ (state >> 2) ^ (state >> 1)) & 1
        seq[i] = bit
        state = ((state << 1) | bit) & 0xFFFF
    return seq


def conv_encode_rate_half(bits: np.ndarray) -> np.ndarray:
    """
    Codificatore convoluzionale rate 1/2, constraint length K=7.
    Polinomi generatori (ottale): G0=0171, G1=0133 (standard NASA/CCSDS).
    G0 = 1111001b = 0x79  →  1 + D² + D³ + D⁴ + D⁶
    G1 = 1011011b = 0x5B  →  1 + D + D³ + D⁴ + D⁶

    Ogni bit in ingresso produce 2 bit in uscita (interleaved G0, G1).
    Note: the encoder is initialised with shift register = 0 (trellis flushed
    adding K-1=6 tail bits).
    """
    K = 7
    G0 = 0b1111001   # 0x79
    G1 = 0b1011011   # 0x5B
    # Append K-1 tail bits (zeros) to flush the shift register
    padded = np.concatenate([bits, np.zeros(K - 1, dtype=np.uint8)])
    shift_reg = 0
    output = np.zeros(len(padded) * 2, dtype=np.uint8)
    for i, b in enumerate(padded):
        shift_reg = ((shift_reg >> 1) | (int(b) << (K - 1))) & ((1 << K) - 1)
        out0 = bin(shift_reg & G0).count('1') % 2
        out1 = bin(shift_reg & G1).count('1') % 2
        output[2 * i]     = out0
        output[2 * i + 1] = out1
    return output


def generate_ira_frame_bits(sat_id: int = 0, beam_id: int = 0,
                             frame_count: int = 0) -> np.ndarray:
    """
    Genera i bit del frame IRA codificati con convoluzionale rate 1/2 K=7.

    Struttura payload pre-codifica:
      ┌──────────────────┬──────────────────┬──────────────────────┬──────┐
      │ sat_id  (8 bit)  │ beam_id (6 bit)  │ frame_cnt (16 bit)   │ dati │
      └──────────────────┴──────────────────┴──────────────────────┴──────┘
    Total payload is {(IRA_DATA_SYMS+IRA_TAIL_SYMS)*2} bits (338 bits),
    corresponding to {(IRA_DATA_SYMS+IRA_TAIL_SYMS)*2 // 2} information bits
    after rate 1/2 coding (K-1=6 tail bits already included in the output).
    """
    # 338 total coded bits from the frame (data + tail symbols × 2 bits/symbol)
    total_coded_bits = (IRA_DATA_SYMS + IRA_TAIL_SYMS) * 2   # 338

    # How many information bits (pre-encoding) produce 338 coded bits?
    # conv_encode padding adds K-1=6 tail bits, therefore:
    # info_bits × 2 + (K-1) × 2 = total_coded_bits
    # info_bits = (total_coded_bits // 2) - (K - 1) = 169 - 6 = 163
    K = 7
    info_bits = total_coded_bits // 2 - (K - 1)   # = 163

    # Fixed header (30 bits)
    header = np.zeros(30, dtype=np.uint8)
    for i in range(8):
        header[7 - i] = (sat_id >> i) & 1
    for i in range(6):
        header[13 - i] = (beam_id >> i) & 1
    for i in range(16):
        header[29 - i] = (frame_count >> i) & 1

    # Scrambled payload for the remaining bits
    payload_bits_count = info_bits - len(header)
    payload = _iridium_scrambler(payload_bits_count, seed=frame_count & 0xFFFF)
    info = np.concatenate([header, payload]).astype(np.uint8)

    # Convolutional encoding rate 1/2, K=7
    coded = conv_encode_rate_half(info)
    # conv_encode_rate_half produces info_bits*2 + (K-1)*2 = 169*2 — already included
    # Truncate/pad to total_coded_bits for safety
    if len(coded) > total_coded_bits:
        coded = coded[:total_coded_bits]
    elif len(coded) < total_coded_bits:
        coded = np.concatenate([coded, np.zeros(total_coded_bits - len(coded), dtype=np.uint8)])
    return coded


def generate_ira_burst(rrc_filter: np.ndarray,
                       sat_id: int = 0,
                       beam_id: int = 0,
                       frame_count: int = 0) -> tuple:
    """
    Genera un burst IRA completo con pulse shaping RRC.

    Struttura trasmessa (in simboli) — da gr-iridium/include/iridium/iridium.h:
      [Guard(8)] [Preamble(64)] [UW(12 BPSK assoluti)] [Data(169 conv. coded)] [Guard(8)] [Silence(20)]
        = slot di 281 simboli = 11.24 ms @ 25 ksps (≈ 90 ms / 8 slot)

    Il Unique Word usa fasi assolute BPSK (45° o 225°) — NON il codificatore
    differenziale π/4-DQPSK — come da UW_DL[] in iridium.h di gr-iridium.
    I dati sono codificati con convoluzionale rate 1/2, K=7 (NASA/CCSDS).

    Returns:
        (slot_iq, markers) where markers is a dict with the sample indices of
        inizio/fine di ogni sezione.
    """
    # ── Section 1: Guard pre-burst (silence) ──────────────────────────────
    guard_pre_syms  = np.zeros(IRA_GUARD_SYMS, dtype=np.complex128)

    # ── Section 2: Preamble run-in (64 symbols) ─────────────────────────────
    # All dibits 0x00 → Δφ = +π/4 per symbol → tone @ +Rs/8 Hz from carrier
    # This is the "signature" tone that allows burst detection in RF
    preamble_bits = np.zeros(IRA_PREAMBLE_SYMS * 2, dtype=np.uint8)
    preamble_syms = pi4_dqpsk_modulate(preamble_bits, initial_phase=np.pi / 4)

    # ── Section 3: Unique Word (12 absolute BPSK symbols) ───────────────────
    # UW_DL[] from iridium.h: {0,2,2,2,2,0,0,0,2,0,0,2}, quadrant 0=45°, 2=225°
    # The UW does NOT go through the differential modulator: it is inserted
    # directly as absolute phases. This is the expected behaviour per gr-iridium.
    uw_syms = UW_DL_SYMBOLS.copy()   # 12 complex symbols, |s|=1

    # ── Section 4: Frame data (167+2 symbols, conv. code rate 1/2 K=7) ──────
    frame_bits = generate_ira_frame_bits(sat_id, beam_id, frame_count)
    # Initial phase for data continues from the last UW symbol (absolute)
    data_syms = pi4_dqpsk_modulate(frame_bits,
                                    initial_phase=float(np.angle(uw_syms[-1])))

    # ── Pulse shaping (upsampling + RRC filtering) ────────────────────────
    def pulse_shape(symbols):
        up = np.zeros(len(symbols) * SPS, dtype=np.complex128)
        up[::SPS] = symbols
        return np.convolve(up, rrc_filter, mode="same")

    burst_syms = np.concatenate([preamble_syms, uw_syms, data_syms])
    burst_iq   = pulse_shape(burst_syms)

    # Prepend/append guard intervals (silence, no filter)
    guard_iq  = np.zeros(IRA_GUARD_SYMS * SPS, dtype=np.complex128)
    silence_n = IRA_SLOT_SILENCE * SPS if IRA_SLOT_SILENCE > 0 else 0
    silence_iq = np.zeros(silence_n, dtype=np.complex128)

    slot_iq = np.concatenate([guard_iq, burst_iq, guard_iq, silence_iq])

    # Sample indices of each section (relative to slot start)
    g_pre = IRA_GUARD_SYMS * SPS
    pm_s  = g_pre
    pm_e  = pm_s + IRA_PREAMBLE_SYMS * SPS
    uw_s  = pm_e
    uw_e  = uw_s + IRA_UW_SYMS * SPS
    da_s  = uw_e
    da_e  = da_s + (IRA_DATA_SYMS + IRA_TAIL_SYMS) * SPS

    markers = {
        "guard_pre":  (0, g_pre),
        "preamble":   (pm_s, pm_e),
        "uw":         (uw_s, uw_e),
        "data":       (da_s, da_e),
        "guard_post": (da_e, da_e + IRA_GUARD_SYMS * SPS),
    }
    return slot_iq, markers


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LEO DOPPLER MODEL — real orbital physics
# ═══════════════════════════════════════════════════════════════════════════════

class IridiumLEODoppler:
    """
    Geometric Doppler model for an Iridium satellite pass.

    The model approximates satellite motion as uniform rectilinear motion
    at constant altitude (valid for observation windows < 5 minutes).
    The effect of Earth's rotation is neglected (error < 1%).

    Reference geometry:
      - Ground station at (0, 0)
      - Satellite moves along the x-axis at altitude h = 780 km
      - t = 0: closest approach (moment of maximum elevation)
      - For t < 0: satellite approaching (positive Doppler, freq increases)
      - For t > 0: satellite receding (negative Doppler, freq decreases)

    The instantaneous range is:
        r(t) = √(r_min² + v_sat² × (t − t_ca)²)

    where r_min = h / sin(E_max) is the minimum range (slant range at closest approach).

    The varying range gives:
        ṙ(t) = v_sat² × (t − t_ca) / r(t)

    The Doppler shift is:
        Δf(t) = −f₀ × ṙ(t) / c    [Hz]

    Typical values for Iridium at 1621 MHz:
      - Overhead pass (E_max = 90°): Doppler from +40.2 kHz to −40.2 kHz
      - 45° pass:                    from +31.3 kHz to −31.3 kHz
      - Max chirp rate (overhead):   ≈ 386 Hz/s at closest approach
      - Max chirp rate (45° pass):   ≈ 273 Hz/s at closest approach
    """

    def __init__(self, carrier_hz: float, max_elev_deg: float = 90.0,
                 t_closest_approach: float = 0.0):
        self.f0    = carrier_hz
        self.E_max = np.radians(max_elev_deg)
        self.t_ca  = t_closest_approach

        # Minimum slant range = distance at closest approach
        # For E_max = 90°: r_min = h;  for E_max < 90°: r_min > h
        sin_e = np.sin(self.E_max)
        if sin_e < 1e-4:
            sin_e = 1e-4   # avoid division by zero at 0° elevation
        self.r_min = IRIDIUM_ALT_M / sin_e

        # Pass statistics
        self.v_sat  = V_SAT
        self.doppler_max_hz = self.f0 * self.v_sat / C  # ≈ 40.2 kHz at 1621 MHz
        chirp_hz_per_s = self.f0 * self.v_sat ** 2 / (C * self.r_min)
        print(f"  [Doppler] r_min={self.r_min/1000:.1f} km | "
              f"Δf_max=±{self.doppler_at_t(-1e6)/1e3:.1f} kHz | "
              f"chirp={chirp_hz_per_s:.1f} Hz/s  (closest approach)")

    def range_rate(self, t: float) -> float:
        """Radial velocity [m/s] at time t. Sign: >0 = moving away."""
        dt = t - self.t_ca
        r  = np.sqrt(self.r_min ** 2 + self.v_sat ** 2 * dt ** 2)
        return self.v_sat ** 2 * dt / r

    def doppler_at_t(self, t: float) -> float:
        """Doppler shift [Hz] at time t.
        Positive = satellite approaching (received freq > transmitted freq).
        Negative = satellite receding.
        """
        return -self.f0 * self.range_rate(t) / C

    def elevation_deg(self, t: float) -> float:
        """Elevation angle [°] above the horizon at time t."""
        dt = t - self.t_ca
        horiz = abs(self.v_sat * dt)
        # Flat-Earth approximation (valid for observations < 2000 km horizontal)
        el_rad = np.arctan2(IRIDIUM_ALT_M, horiz)
        return np.degrees(el_rad)

    def pass_half_duration_s(self, min_elev_deg: float = 5.0) -> float:
        """
        Duration from closest approach to satellite set (minimum elevation).
        Total pass duration = 2 × this value.
        """
        min_el_rad = np.radians(min_elev_deg)
        horiz_at_min = IRIDIUM_ALT_M / np.tan(min_el_rad)
        return horiz_at_min / self.v_sat


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FULL-PASS IQ GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_iridium_pass(
    duration_s: float = 60.0,
    carrier_hz: float = float(IRDM_RING_ALERT_HZ),
    max_elev_deg: float = 45.0,
    snr_db: float = 15.0,
    sat_id: int = 47,
    beam_id: int = 3,
    t_closest: float = None,
) -> tuple:
    """
    Generate the IQ signal of an Iridium satellite transmitting IRA bursts
    during a pass with realistic Doppler effect.

    The satellite transmits one IRA burst every 90 ms (1 burst per superframe
    on the assigned simplex channel). Doppler changes gradually with
    the pass geometry.

    Args:
        duration_s:    Total simulation duration [s]
        carrier_hz:    Transmitted carrier frequency [Hz]
        max_elev_deg:  Maximum pass elevation [°]
        snr_db:        SNR in dB (added as AWGN at the end)
        sat_id:        Satellite ID (0-127)
        beam_id:       Beam ID (0-47)
        t_closest:     Instant of closest approach [s] (default: midpoint of sim)

    Returns:
        (iq_samples, burst_log, doppler_model, rrc_filter)
          iq_samples: complex64 array of the entire simulation
          burst_log:  list of dicts with info for each burst
          doppler_model, rrc_filter: oggetti per uso esterno
    """

    if t_closest is None:
        t_closest = duration_s / 2.0

    rrc = generate_rrc_filter(RRC_BETA, SPS, RRC_NUM_TAPS)
    doppler = IridiumLEODoppler(carrier_hz, max_elev_deg, t_closest)

    # Number of slots in the simulation period
    n_slots = int(np.ceil(duration_s / SUPERFRAME_S))
    total_samples = int(duration_s * SAMPLE_RATE)
    iq_out = np.zeros(total_samples, dtype=np.complex128)

    burst_log = []

    for slot_idx in range(n_slots):
        t_burst  = slot_idx * SUPERFRAME_S          # burst start [s]
        t_center = t_burst + SUPERFRAME_S / 2       # burst centre [s]
        el_deg   = doppler.elevation_deg(t_center)

        # Compute Doppler at burst centre (constant over the entire burst:
        # Doppler variation WITHIN a 9 ms burst is < 3.5 Hz → negligible)
        f_doppler = doppler.doppler_at_t(t_center)

        # Generate burst IQ @ base freq. (no Doppler applied yet)
        slot_iq, markers = generate_ira_burst(rrc, sat_id, beam_id, slot_idx)

        # Apply Doppler offset to burst (frequency shift):
        # s_rx(t) = s_tx(t) × exp(j × 2π × Δf_doppler × t)
        n_slot = len(slot_iq)
        # Use the absolute time of burst start for correct phase
        t_abs_start = t_burst
        t_vec = t_abs_start + np.arange(n_slot) / SAMPLE_RATE
        slot_iq *= np.exp(1j * 2 * np.pi * f_doppler * t_vec)

        # Insert into the global timeline
        start_sample = int(t_burst * SAMPLE_RATE)
        end_sample   = start_sample + n_slot
        if end_sample > total_samples:
            end_sample  = total_samples
            n_slot      = end_sample - start_sample
            slot_iq     = slot_iq[:n_slot]

        iq_out[start_sample:end_sample] += slot_iq

        burst_log.append({
            "slot_idx":       slot_idx,
            "t_start_s":      t_burst,
            "t_center_s":     t_center,
            "elevation_deg":  el_deg,
            "doppler_hz":     f_doppler,
            "start_sample":   start_sample,
            "markers":        {k: (v[0] + start_sample, v[1] + start_sample)
                               for k, v in markers.items()},
        })

    # Normalise and add AWGN
    peak = np.max(np.abs(iq_out))
    if peak > 0:
        iq_out /= peak

    if snr_db < 100:   # SNR=100 means ideal signal without noise
        sig_power   = np.mean(np.abs(iq_out) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(total_samples) + 1j * np.random.randn(total_samples)
        )
        iq_out += noise

    return iq_out.astype(np.complex64), burst_log, doppler, rrc


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PREAMBLE + UW CORRELATOR (burst detection)
# ═══════════════════════════════════════════════════════════════════════════════

def build_preamble_template(rrc: np.ndarray) -> np.ndarray:
    """
    Build the IQ template for the preamble run-in (already filtered with RRC).
    Used for cross-correlation.
    """
    pm_bits = np.zeros(IRA_PREAMBLE_SYMS * 2, dtype=np.uint8)
    pm_syms = pi4_dqpsk_modulate(pm_bits, initial_phase=np.pi / 4)
    up = np.zeros(len(pm_syms) * SPS, dtype=np.complex128)
    up[::SPS] = pm_syms
    return np.convolve(up, rrc, mode="same")


def detect_preamble(iq: np.ndarray, rrc: np.ndarray,
                    doppler_search_hz: float = 45_000.0,
                    doppler_step_hz: float = 500.0,
                    threshold_factor: float = 0.35) -> list:
    """
    Preamble correlation detector with Doppler compensation.

    Algorithm:
      1. Scan a Doppler offset grid (−45 kHz … +45 kHz, step 500 Hz)
      2. For each offset: mix the signal to compensate the Doppler
      3. Compute cross-correlation with the preamble template
      4. Search for peaks above threshold

    This is essentially what gr-iridium does to find bursts:
    it searches for energy in the already channelised FDMA channel (Doppler handled
    by the polyphase filter bank), then passes to the QPSK demodulator.

    Args:
        iq:                  Input IQ samples
        rrc:                 RRC filter for the template
        doppler_search_hz:   Doppler search range [Hz]
        doppler_step_hz:     Doppler grid step [Hz]
        threshold_factor:    Threshold as fraction of the global peak

    Returns:
        List of dicts: {sample_idx, doppler_hz, corr_peak, t_s}
    """
    template = build_preamble_template(rrc)
    t = np.arange(len(iq)) / SAMPLE_RATE
    template_len = len(template)

    detections = []
    max_corr_global = 0.0

    freqs = np.arange(-doppler_search_hz, doppler_search_hz + doppler_step_hz,
                      doppler_step_hz)

    print(f"  [Detector] Doppler scan: {len(freqs)} offsets "
          f"({-doppler_search_hz/1e3:.0f}…+{doppler_search_hz/1e3:.0f} kHz, "
          f"passo {doppler_step_hz:.0f} Hz) …", end=" ", flush=True)
    t0 = time.time()

    # Efficient computation: FFT-based cross-correlation
    n_fft = len(iq) + template_len - 1
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))   # next power of 2

    corr_max = np.zeros(len(iq))  # correlation peak for each sample

    for f_d in freqs:
        # Doppler compensation: rotate signal by −f_d
        iq_comp = iq * np.exp(-1j * 2 * np.pi * f_d * t)

        # Cross-correlation via FFT (faster than direct convolution)
        X = np.fft.fft(iq_comp, n=n_fft)
        T = np.fft.fft(np.conj(template[::-1]), n=n_fft)  # matched filter
        corr = np.abs(np.fft.ifft(X * T))[:len(iq)]

        # Update the maximum for each sample across all Doppler offsets
        np.maximum(corr_max, corr, out=corr_max)

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    # Adaptive threshold: 99.9th percentile + factor
    if np.max(corr_max) > 0:
        threshold = np.max(corr_max) * threshold_factor
    else:
        return detections

    # Find peaks above threshold (with minimum distance = 1 slot)
    min_dist = int(SUPERFRAME_S * SAMPLE_RATE * 0.5)
    peak_locs, _ = sp_signal.find_peaks(corr_max, height=threshold,
                                         distance=min_dist)

    for loc in peak_locs:
        # Estimate Doppler at detection point via refined search
        best_corr = 0.0
        best_dop  = 0.0
        for f_d in freqs:
            seg = iq[loc:loc + template_len]
            if len(seg) < template_len:
                break
            t_seg = loc / SAMPLE_RATE + np.arange(template_len) / SAMPLE_RATE
            seg_comp = seg * np.exp(-1j * 2 * np.pi * f_d * t_seg)
            c = abs(np.dot(np.conj(template), seg_comp))
            if c > best_corr:
                best_corr = c
                best_dop  = f_d

        detections.append({
            "sample_idx": int(loc),
            "t_s":        loc / SAMPLE_RATE,
            "doppler_hz": best_dop,
            "corr_peak":  float(corr_max[loc]),
        })

    return detections


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ANALYSIS PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_analysis(iq: np.ndarray, burst_log: list, doppler_model: IridiumLEODoppler,
                  detections: list = None, save_path: str = None):
    """Generate a 4-panel diagnostic plot."""
    duration_s = len(iq) / SAMPLE_RATE
    t_ms = np.arange(len(iq)) / SAMPLE_RATE * 1000

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Iridium IRA Burst — LEO Pass Simulation  "
        f"(sps=10, Rs=25 ksps, RRC β=0.4, π/4-DQPSK)",
        fontsize=13, fontweight="bold"
    )

    # 1) Time-domain envelope
    ax = axes[0, 0]
    ax.plot(t_ms, np.abs(iq), linewidth=0.3, color="steelblue", alpha=0.8)
    for b in burst_log[:20]:   # max 20 bursts
        pm = b["markers"]["preamble"]
        uw  = b["markers"]["uw"]
        t_pm_s = pm[0] / SAMPLE_RATE * 1000
        t_pm_e = pm[1] / SAMPLE_RATE * 1000
        ax.axvspan(t_pm_s, t_pm_e, alpha=0.25, color="lime",  zorder=3)
        t_uw_s = uw[0] / SAMPLE_RATE * 1000
        t_uw_e = uw[1] / SAMPLE_RATE * 1000
        ax.axvspan(t_uw_s, t_uw_e, alpha=0.3, color="orange", zorder=3)
    # Colour legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="lime",   alpha=0.5, label="Preamble"),
                        Patch(color="orange", alpha=0.5, label="UW")],
               fontsize=8, loc="upper right")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal envelope (first 20 bursts)")
    ax.set_xlim([0, min(duration_s * 1000, 5000)])  # show max 5 s
    ax.grid(True, alpha=0.3)

    # 2) Doppler trajectory during the pass
    ax = axes[0, 1]
    t_pass = np.linspace(0, duration_s, 500)
    doppler_traj = [doppler_model.doppler_at_t(t) / 1000 for t in t_pass]
    elev_traj    = [doppler_model.elevation_deg(t) for t in t_pass]
    ax.plot(t_pass, doppler_traj, "b-", linewidth=1.5, label="Doppler (kHz)")
    ax2 = ax.twinx()
    ax2.plot(t_pass, elev_traj, "r--", linewidth=1.0, alpha=0.6, label="Elevation (°)")
    ax2.set_ylabel("Elevation (°)", color="red", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="red")
    # Overlay measured Doppler for each burst
    if burst_log:
        t_bursts  = [b["t_center_s"] for b in burst_log]
        d_bursts  = [b["doppler_hz"] / 1000 for b in burst_log]
        ax.scatter(t_bursts, d_bursts, s=6, c="steelblue", zorder=5, alpha=0.7,
                   label="Simul. burst")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Doppler shift (kHz)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.set_title(f"LEO Doppler trajectory (h=780 km, E_max={doppler_model.E_max*180/np.pi:.0f}°)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # 3) Preamble spectrum (should show the tone at +3125 Hz)
    ax = axes[1, 0]
    if burst_log:
        pm_m = burst_log[0]["markers"]["preamble"]
        seg  = iq[pm_m[0]:pm_m[1]]
        if len(seg) > 0:
            nfft = max(len(seg), 512) * 4
            f_ax = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / SAMPLE_RATE)) / 1000
            spec = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(seg, n=nfft))) + 1e-12)
            spec -= np.max(spec)
            ax.plot(f_ax, spec, linewidth=0.7, color="darkgreen")
            ax.axvline(x=3.125, color="red", linestyle="--",
                       linewidth=0.8, alpha=0.8, label="Expected tone +3125 Hz")
    ax.set_xlabel("Frequency relative to carrier (kHz)")
    ax.set_ylabel("PSD (dB, norm.)")
    ax.set_title("Preamble run-in spectrum (tone @ +Rs/8 = +3125 Hz)")
    ax.set_xlim([-30, 30])
    ax.set_ylim([-50, 5])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Scatter: detections vs. generated bursts
    ax = axes[1, 1]
    if burst_log:
        t_b = [b["t_center_s"] for b in burst_log]
        d_b = [b["doppler_hz"] / 1000 for b in burst_log]
        ax.scatter(t_b, d_b, s=20, c="steelblue", alpha=0.7, label="Generated bursts",
                   zorder=3)
    if detections:
        t_d  = [d["t_s"] for d in detections]
        d_d  = [d["doppler_hz"] / 1000 for d in detections]
        ax.scatter(t_d, d_d, s=40, c="red", marker="x", linewidths=1.5,
                   label="Detected bursts (correlator)", zorder=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated Doppler (kHz)")
    ax.set_title("Generated vs. detected bursts (correlator)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path if save_path else "iridium_realistic_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TRANSMISSION VIA LIBRESDR (pyadi-iio)
# ═══════════════════════════════════════════════════════════════════════════════

def _resample_to_hw(iq: np.ndarray, target_sps: int = 1_000_000) -> np.ndarray:
    """Resample from SAMPLE_RATE (250 kHz) to target_sps (default 1 MHz)."""
    from math import gcd
    g = gcd(target_sps, SAMPLE_RATE)
    up, down = target_sps // g, SAMPLE_RATE // g
    return sp_signal.resample_poly(iq, up, down).astype(np.complex64)


def transmit_via_libresdr(iq: np.ndarray, uri: str, center_freq_hz: int,
                           tx_gain_db: float = -60.0, cyclic: bool = False):
    """
    Transmit IQ samples via LibreSDR (AD9363) using pyadi-iio.
    Identical interface to PlutoSDR (same firmware).

    SAFETY:
      - Default gain: −60 dB (very low)
      - Increase ONLY after verifying reception on a wired cable
      - DO NOT transmit in the Iridium band (1616-1626 MHz) without a licence
    """
    try:
        import adi
    except ImportError:
        print("[TX ERROR] pyadi-iio not installed. Install with: pip install pyadi-iio")
        return False

    # Resample to 1 MSPS (AD9363 minimum)
    TX_RATE = 1_000_000
    print(f"  Resampling {SAMPLE_RATE/1e3:.0f} kHz → {TX_RATE/1e6:.1f} MSPS …",
          end=" ", flush=True)
    tx_iq = _resample_to_hw(iq, TX_RATE)
    # Scale for DAC (range ±2^14)
    mx = np.max(np.abs(tx_iq))
    if mx > 0:
        tx_iq = tx_iq / mx * 0.9 * 2**14
    print("OK")

    print(f"  Connecting to {uri} …", end=" ", flush=True)
    try:
        sdr = adi.Pluto(uri)
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    print("OK")

    sdr.sample_rate = int(TX_RATE)
    sdr.tx_rf_bandwidth = int(TX_RATE)
    sdr.tx_lo = int(center_freq_hz)
    sdr.tx_hardwaregain_chan0 = float(tx_gain_db)

    # AD9363 hardware buffer limit via network (Ethernet IIO):
    # buffers > ~2^20 samples cause BrokenPipe on the DMA pipeline.
    # For cyclic mode we use an integer multiple of slots (281 symbols × SPS),
    # max 2^20 samples ≈ 1 s at 1 MSPS. This ensures the Iridium burst
    # (period 90 ms = 90000 samples) is fully represented at least once
    # per hardware cycle.
    HW_BUF_MAX = 2**20   # 1,048,576 samples ≈ 1.05 s @ 1 MSPS

    if cyclic:
        # Trim to nearest superframe multiple below HW_BUF_MAX
        slot_samples_hw = int(SUPERFRAME_S * TX_RATE)   # 90000 samples
        n_slots_fit = max(1, HW_BUF_MAX // slot_samples_hw)
        cyclic_len = n_slots_fit * slot_samples_hw
        cyclic_iq = tx_iq[:cyclic_len].copy()
        cycle_ms = cyclic_len / TX_RATE * 1000
        print(f"  Cycle: {cyclic_len} samples ({cycle_ms:.0f} ms, "
              f"{n_slots_fit} superframe) — ripetuto in loop")
    else:
        cyclic_iq = None

    print(f"  TX → f={center_freq_hz/1e6:.3f} MHz | "
          f"SR={TX_RATE/1e6:.1f} MSPS | gain={tx_gain_db:+.0f} dB | "
          f"{'CICLICO' if cyclic else 'ONE-SHOT'}")

    if cyclic:
        sdr.tx_cyclic_buffer = True
        sdr.tx(cyclic_iq)
        print("  Cyclic transmission active. Ctrl+C to stop …")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()
        finally:
            sdr.tx_destroy_buffer()
    else:
        # One-shot: the AD9363 DMA (over IIO/Ethernet) cannot handle buffers larger than
        # ~2^20 samples — anything bigger causes BrokenPipe on _txbuf.push().
        # Cap to HW_BUF_MAX and warn the user if truncation occurs.
        if len(tx_iq) > HW_BUF_MAX:
            print(f"  [WARN] IQ block ({len(tx_iq)} samples, "
                  f"{len(tx_iq)/TX_RATE:.1f} s) exceeds HW limit "
                  f"({HW_BUF_MAX} samples, {HW_BUF_MAX/TX_RATE:.1f} s). "
                  f"Truncating. Use --cyclic for longer passes.")
            tx_iq = tx_iq[:HW_BUF_MAX]
        sdr.tx_buffer_size = len(tx_iq)
        sdr.tx(tx_iq)
        print(f"  Transmitted: {len(tx_iq)} samples ({len(tx_iq)/TX_RATE*1000:.1f} ms)")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Realistic Iridium downlink simulation (IRA bursts, π/4-DQPSK, LEO Doppler)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 30 s pass, save IQ and plot
  python3 iridium_realistic_sim.py --pass-dur 30 --save iridium.iq

  # 60° elevation pass, high SNR, burst detection
  python3 iridium_realistic_sim.py --elev 60 --snr 25 --detect

  # Transmit in loop via LibreSDR at 100 MHz (safe test frequency)
  python3 iridium_realistic_sim.py --tx-freq 100e6 --tx-gain -55 --cyclic

  # Disable TX (IQ simulation only)
  python3 iridium_realistic_sim.py --no-tx --no-save --no-plot --pass-dur 60
"""
    )
    parser.add_argument("--pass-dur",  type=float, default=60.0,
                        metavar="SECONDS",
                        help="Total simulation duration [s] (default: 60)")
    parser.add_argument("--elev",      type=float, default=45.0,
                        metavar="DEGREES",
                        help="Maximum pass elevation [°] (default: 45). "
                             "90° = overhead pass; 10° = low horizon pass")
    parser.add_argument("--snr",       type=float, default=15.0,
                        metavar="dB",
                        help="AWGN channel SNR [dB] (default: 15). "
                             "Use 100 for ideal signal without noise")
    parser.add_argument("--carrier",   type=float, default=float(IRDM_RING_ALERT_HZ),
                        metavar="HZ",
                        help=f"Simulated carrier frequency [Hz] "
                             f"(default: {IRDM_RING_ALERT_HZ} Hz = Iridium Ring Alert)")
    parser.add_argument("--sat-id",    type=int, default=47,
                        metavar="ID",
                        help="Simulated Iridium satellite ID (0-127, default: 47)")
    parser.add_argument("--beam-id",   type=int, default=3,
                        metavar="ID",
                        help="Spot beam ID (0-47, default: 3)")
    parser.add_argument("--save",      type=str, default=None,
                        metavar="FILE.IQ",
                        help="Save IQ samples as complex64 (default: iridium_pass.iq)")
    parser.add_argument("--no-save",   action="store_true",
                        help="Skip saving the IQ file")
    parser.add_argument("--save-plot", type=str, default=None,
                        metavar="FILE.PNG",
                        help="PNG plot output path (default: iridium_realistic_plot.png)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--detect",    action="store_true",
                        help="Run preamble correlator (slow: scans Doppler)")
    parser.add_argument("--no-tx",     action="store_true",
                        help="Disable TX via LibreSDR (default: TX always active)")
    parser.add_argument("--tx-uri",    type=str, default="ip:192.168.1.10",
                        metavar="URI",
                        help="URI IIO del LibreSDR (default: ip:192.168.1.10)")
    parser.add_argument("--tx-freq",   type=float, default=float(IRDM_RING_ALERT_HZ),
                        metavar="HZ",
                        help=f"LibreSDR TX carrier frequency [Hz] "
                             f"(default: {IRDM_RING_ALERT_HZ} Hz = Iridium Ring Alert)")
    parser.add_argument("--tx-gain",   type=float, default=-60.0,
                        metavar="dB",
                        help="TX gain in dB, range −90…0  (default: −60)")
    parser.add_argument("--cyclic",    action="store_true",
                        help="Cyclic TX transmission until Ctrl+C")
    args = parser.parse_args()

    print("=" * 68)
    print("  Iridium Downlink Simulation — Single Satellite")
    print("=" * 68)
    print(f"  Pass duration:     {args.pass_dur:.0f} s")
    print(f"  Max elevation:     {args.elev:.0f}°")
    print(f"  Channel SNR:       {args.snr:.0f} dB")
    print(f"  Simul. carrier:    {args.carrier/1e6:.4f} MHz")
    print(f"  SAT_ID/BEAM_ID:    {args.sat_id}/{args.beam_id}")
    print()
    print(f"  Modulazione:       π/4-DQPSK")
    print(f"  Symbol rate:       {SYMBOL_RATE/1e3:.0f} ksps")
    print(f"  Sample rate sim.:  {SAMPLE_RATE/1e3:.0f} ksps  ({SPS} sps)")
    print(f"  RRC β:             {RRC_BETA}")
    print(f"  Superframe:        {SUPERFRAME_S*1000:.0f} ms  "
          f"({SLOTS_PER_FRAME} slots/frame, {SLOT_SYMS} sym/slot)")
    print(f"  Burst structure:   "
          f"guard({IRA_GUARD_SYMS}) + preamble({IRA_PREAMBLE_SYMS}) + "
          f"UW({IRA_UW_SYMS} BPSK) + data({IRA_DATA_SYMS}+{IRA_TAIL_SYMS} conv K=7) + "
          f"guard({IRA_POST_GUARD}) + silence({IRA_SLOT_SILENCE})")
    print(f"  Preamble tone:     +{SYMBOL_RATE//8} Hz from carrier "
          f"(= Rs/8 = constant +π/4/symbol)")
    n_bursts_expected = int(args.pass_dur / SUPERFRAME_S)
    n_samples_total   = int(args.pass_dur * SAMPLE_RATE)
    size_mb = n_samples_total * 8 / 1e6   # complex64 = 8 bytes
    print(f"  Bursts generated:  {n_bursts_expected}")
    print(f"  Total samples:     {n_samples_total/1e6:.2f} M  ({size_mb:.1f} MB)")
    print()

    # ── Pass simulation ────────────────────────────────────────────────────
    print("  Generating pass IQ …", end=" ", flush=True)
    t0 = time.time()
    iq, burst_log, doppler, rrc = simulate_iridium_pass(
        duration_s  = args.pass_dur,
        carrier_hz  = args.carrier,
        max_elev_deg= args.elev,
        snr_db      = args.snr,
        sat_id      = args.sat_id,
        beam_id     = args.beam_id,
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s  →  {len(iq)/1e6:.2f} M samples")

    # Print burst statistics
    if burst_log:
        d_vals = [b["doppler_hz"] for b in burst_log]
        e_vals = [b["elevation_deg"] for b in burst_log]
        print(f"\n  Burst log ({len(burst_log)} bursts):")
        print(f"    Doppler range:     {min(d_vals)/1e3:+.1f} … {max(d_vals)/1e3:+.1f} kHz")
        print(f"    Elevation range:   {min(e_vals):.1f}° … {max(e_vals):.1f}°")
        print(f"    Closest approach:  t={doppler.t_ca:.1f}s  "
              f"(Δf=0 Hz, elev={doppler.E_max*180/np.pi:.0f}°)")
        print(f"    Doppler max:       ±{doppler.doppler_at_t(-1e6)/1e3:.1f} kHz")
        # Show first 5 and last 5
        print()
        print("    idx | t_center [s] | Elev [°] | Doppler [kHz]")
        print("    " + "─" * 46)
        show = list(range(min(4, len(burst_log)))) + \
               (["..."] if len(burst_log) > 8 else []) + \
               list(range(max(4, len(burst_log) - 4), len(burst_log)))
        for idx in show:
            if idx == "...":
                print(f"    {'...':<5}  {'...':<12}  {'...':<8}  {'...'}")
                continue
            b = burst_log[idx]
            print(f"    {b['slot_idx']:<5}  {b['t_center_s']:<12.2f}  "
                  f"{b['elevation_deg']:<8.1f}  {b['doppler_hz']/1e3:+.2f}")

    # ── Preamble correlator (optional, slow) ──────────────────────────────
    detections = []
    if args.detect:
        print(f"\n  Preamble detection (correlator):")
        # Analyse only the first 10 seconds for speed
        seg_len = min(len(iq), int(10.0 * SAMPLE_RATE))
        detections = detect_preamble(iq[:seg_len], rrc,
                                      doppler_search_hz=45_000,
                                      doppler_step_hz=1000)
        print(f"  → {len(detections)} bursts detected in the first 10 s")
        for d in detections[:6]:
            print(f"     t={d['t_s']:.3f}s  Δf={d['doppler_hz']/1e3:+.1f} kHz  "
                  f"corr_peak={d['corr_peak']:.2f}")

    # ── Save IQ file ──────────────────────────────────────────────────────
    if not args.no_save:
        out_iq = args.save if args.save else "iridium_pass.iq"
        iq.astype(np.complex64).tofile(out_iq)
        print(f"\n  IQ file saved: {out_iq}")
        print(f"    Samples:     {len(iq)}  |  SR: {SAMPLE_RATE} Hz")
        print(f"    Format:      complex64 (I float32 + Q float32 interleaved)")
        print(f"    Lettura:     np.fromfile('{out_iq}', dtype=np.complex64)")
        print(f"    gr-iridium:  iridium-extractor -c {int(args.carrier)} "
              f"-r {SAMPLE_RATE} -f float {out_iq}")

    # ── Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n  Generating plot …", end=" ", flush=True)
        plot_analysis(iq, burst_log, doppler, detections,
                      save_path=args.save_plot)
        print("OK")

    # ── Transmission via LibreSDR ─────────────────────────────────────────
    if not args.no_tx:
        print(f"\n  Transmitting via LibreSDR ({args.tx_uri})")
        print(f"  FREQ: {args.tx_freq/1e6:.4f} MHz  |  GAIN: {args.tx_gain:+.0f} dB")
        transmit_via_libresdr(
            iq             = iq,
            uri            = args.tx_uri,
            center_freq_hz = int(args.tx_freq),
            tx_gain_db     = args.tx_gain,
            cyclic         = args.cyclic,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
