"""
shared.iridium
==============
Single source of truth for all Iridium L-band physical and protocol constants.

Both krakenSDR (receiver/DoA) and libreSDR (transmitter) import from here so
there is zero risk of parameter drift between the RX and TX side.

Sources
-------
* gr-iridium    https://github.com/muccc/gr-iridium
* iridium-toolkit https://github.com/muccc/iridium-toolkit  (util.py, demod.py)
* ITU-R M.1031  – Iridium radio interface specification
* ETSI EN 300 461 – Iridium TDMA frame structure
* Motorola / Iridium LLC public SIS-ICD
"""

from __future__ import annotations

import math

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT: float = 2.99792458e8     # speed of light [m/s]
MU_EARTH: float = 3.986004418e14  # Earth gravitational parameter [m³/s²]
R_EARTH: float = 6.3710e6         # mean Earth radius [m]

# ─────────────────────────────────────────────────────────────────────────────
# Orbital parameters (Iridium classic & NEXT)
# ─────────────────────────────────────────────────────────────────────────────
ORBIT_ALTITUDE_M: float = 780_000.0   # nominal LEO altitude [m]
ORBIT_INCL_DEG:   float = 86.4        # orbital inclination [°]
_R_ORBIT = R_EARTH + ORBIT_ALTITUDE_M
V_SAT: float = math.sqrt(MU_EARTH / _R_ORBIT)   # ≈ 7464 m/s

# ─────────────────────────────────────────────────────────────────────────────
# RF frequency plan (from iridium-toolkit util.py)
# ─────────────────────────────────────────────────────────────────────────────
#   f_n = BASE_FREQ + n × CHANNEL_SPACING   (n = 0 … 239)
#   Downlink band: 1621.35 – 1626.5 MHz
#   Uplink band:   1616.0  – 1621.35 MHz
BASE_FREQ_HZ:      int = 1_615_604_164     # Iridium channel 0 centre [Hz]
CHANNEL_SPACING_HZ: int = 41_667          # FDMA channel spacing [Hz]
NUM_CHANNELS:       int = 240             # total FDMA channels

# Frequently-used named channels
SIMPLEX_RING_CH_HZ: float = 1_626_270_000.0   # ch ~252: ring-alert (most active DL)
SIMPLEX_NEXT_CH_HZ: float = 1_626_104_000.0   # Iridium NEXT paging
DUPLEX_DL_1621_HZ:  float = 1_621_500_000.0   # duplex DL centre 1
DUPLEX_DL_1623_HZ:  float = 1_623_500_000.0   # duplex DL centre 2

DOWNLINK_CHANNELS: dict[str, float] = {
    "Simplex 1626.270 MHz [ring alerts – most active]": SIMPLEX_RING_CH_HZ,
    "Simplex 1626.104 MHz [NEXT paging]":               SIMPLEX_NEXT_CH_HZ,
    "Duplex DL 1621.5 MHz":                             DUPLEX_DL_1621_HZ,
    "Duplex DL 1623.5 MHz":                             DUPLEX_DL_1623_HZ,
}

# ─────────────────────────────────────────────────────────────────────────────
# Modulation
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL_RATE: int   = 25_000    # symbol rate [sps] – confirmed by gr-iridium
RRC_BETA:    float = 0.4       # RRC roll-off factor – confirmed by gr-iridium

# Samples-per-symbol reference values
SPS_GRIB:    int = 10          # gr-iridium / iridium_realistic_sim default (250 ksps)
SPS_TX:      int = 8           # iridium_burst_gen (200 ksps, loopback)

SAMPLE_RATE_GRIB: int = SYMBOL_RATE * SPS_GRIB   # 250_000 Hz
SAMPLE_RATE_TX:   int = SYMBOL_RATE * SPS_TX     # 200_000 Hz

# ─────────────────────────────────────────────────────────────────────────────
# TDMA frame structure (IRA – Ring Alert, Simplex downlink  )
# ─────────────────────────────────────────────────────────────────────────────
SUPERFRAME_S:    float = 0.090       # super-frame period [s]
SLOTS_PER_FRAME: int   = 8
TDMA_PERIOD_S:   float = SUPERFRAME_S / SLOTS_PER_FRAME   # = 0.01125 s / slot
TDMA_SLOT_SYM:   int   = int(SUPERFRAME_S * SYMBOL_RATE / SLOTS_PER_FRAME)  # = 281

# Burst structure [symbols]
#   Sources: gr-iridium/lib/iridium.h, iridium-toolkit, confirmed in realistic_sim.py
#
#   PREAMBLE_LENGTH_LONG  = 64  (gr-iridium/lib/iridium.h)
#   UW_LENGTH             = 12  (gr-iridium/lib/iridium.h)
#   Data payload inferred from iridium-toolkit output: 179 total frame = 12 UW + 167 data
#   IRA_TAIL_SYM = 2  tail bits to flush the K=7 convolutional encoder
#
#   Timeline within one 281-symbol TDMA slot:
#     8 guard_pre + 64 preamble + 12 UW + 167 data + 2 tail + 8 guard_post = 261
#     + 20 silence  = 281  ✓
GUARD_PRE_SYM:   int = 8     # guard before burst (silence)
PREAMBLE_SYM:    int = 64    # run-in: constant +π/4 rotation → tone at Rs/8 = 3125 Hz
                              # source: gr-iridium PREAMBLE_LENGTH_LONG
UNIQUE_WORD_SYM: int = 12    # synchronisation word (24 bits, BPSK-like dibits)
                              # source: gr-iridium UW_LENGTH
FRAME_DATA_SYM:  int = 167   # payload (334 bits: header + BCH + data)
                              # source: iridium-toolkit output 179 total = 12+167
IRA_TAIL_SYM:    int = 2     # tail symbols to flush K=7 convolutional encoder
GUARD_POST_SYM:  int = 8     # guard after burst (silence)
SILENCE_SYM:     int = 20    # inter-slot padding: TDMA_SLOT_SYM minus all above = 20

BURST_TOTAL_SYM: int = (GUARD_PRE_SYM + PREAMBLE_SYM +
                         UNIQUE_WORD_SYM + FRAME_DATA_SYM +
                         IRA_TAIL_SYM + GUARD_POST_SYM)

# Unique Word bit pattern (24 bits → 12 DQPSK dibits)
# Source: gr-iridium iridium.py, extractor-python, iridium-toolkit
UNIQUE_WORD_DOWNLINK: str = "022220002002"   # dibit string
UNIQUE_WORD_UPLINK:   str = "220002002022"

# ─────────────────────────────────────────────────────────────────────────────
# Doppler / propagation
# ─────────────────────────────────────────────────────────────────────────────
MAX_DOPPLER_HZ:   float = 40_200.0   # max |Doppler| at L-band from LEO [Hz]
DOPPLER_RATE_HZ_S: float = 386.0     # max rate of change [Hz/s] at closest approach
NEW_PASS_DELTA_HZ: float = 12_000.0  # Doppler jump that marks a new satellite [Hz]
PASS_TIMEOUT_S:    float = 6.0       # silence that marks end of satellite pass [s]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute channel centre frequency by index
# ─────────────────────────────────────────────────────────────────────────────

def channel_freq_hz(n: int) -> float:
    """Return the centre frequency [Hz] of Iridium FDMA channel *n* (0-based)."""
    return float(BASE_FREQ_HZ + n * CHANNEL_SPACING_HZ)
