# =============================================================================
#  apps/doa_iridium — 2D DoA on Iridium IRA bursts at 1626.270 MHz
#
#  Hardware
#  --------
#  TX  : LibreSDR (AD9363) running tx/indoor_1626.py
#        — indoor lab tests at 1626.270 MHz (cable + ≥30 dB attenuator)
#        — outdoor mode: no TX, the sky transmits
#  RX  : KrakenSDR 5-channel RHCP UCA, Heimdall DAQ on TCP:5000
# =============================================================================
from __future__ import annotations

import os as _os
import sys as _sys

# ── Hardware base (re-export) ─────────────────────────────────────────────────
_SRC = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..")
)
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from config_hw import *   # noqa: F401, F403

# ═══════════════════════════════════════════════════════════════════════════════
# Array geometry
# ═══════════════════════════════════════════════════════════════════════════════

N_ANTENNAS     = 5
GEOMETRY       = "UCA"

RADIUS_LAMBDA  = 0.4253
# UCA radius in wavelengths for λ/2 element spacing:
#   R = (λ/2) / (2·sin(π/5)) ≈ 0.4253 λ
# At 1626.270 MHz (λ ≈ 18.43 cm):  physical radius ≈ 7.84 cm

ANT_CCW = False
# False = elements CLOCKWISE (CW) viewed from above — standard KrakenSDR UCA.
# Set True if azimuth estimates are mirrored (East ↔ West swapped).

ANT0_OFFSET_DEG = 0.0
# Rotation of antenna 0 from geographic North [°].
# Re-calibrate:  python3 doa_iridium_burst.py --calibrate 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# RF / Frequency
# ═══════════════════════════════════════════════════════════════════════════════

FREQ_HZ = 1_626_270_000
# Iridium Ring Alert channel (1626.270 MHz).

GAIN_DB = 45
# IF gain [dB] applied to all KrakenSDR channels.  RTL-SDR max ≈ 49.6 dB.
# Indoor −60 dB TX at ~1 m:  45–49 dB
# Indoor −20 dB TX at ~1 m:  30–35 dB  (avoid saturation)
# Outdoor real Iridium:      30–40 dB

# ═══════════════════════════════════════════════════════════════════════════════
# 2D DoA algorithm
# ═══════════════════════════════════════════════════════════════════════════════

DOA_ALGORITHM  = "MUSIC"
# "MUSIC" / "CAPON" / "ROOT-MUSIC" / "MFBA-MUSIC" / "UNITARY-ESPRIT"

MUSIC_DECORR   = "none"
# Covariance decorrelation: 'none' / 'fb' / 'circulant'.
# 'none' — default. Multi-burst sample covariance is already full rank.

NUM_SIGNALS = 0
# Expected sources PER SATELLITE for MUSIC subspace split.
# 0 = auto-MDL (Wax & Kailath 1985).  Peak selection via pick_doa_peak_uca_2d.

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-satellite detection
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SATELLITES = 1
# Maximum simultaneously tracked satellites.
# Iridium L-band: typically 1–3 above horizon.  Indoor single-TX: set to 1.

MULTI_BURST_N = 20
# Number of preamble bursts accumulated per DoA estimate.
# Indoor: 20 → ~1 estimate every 10s.  Outdoor satellite: 10–15.

DOPPLER_SCAN_BW_HZ = 45_000
# Half-width [Hz] of FFT scan around preamble tone (3125 Hz).

DOPPLER_GATE_HZ = 3_000
# CFO rejection gate [Hz].  Peaks with |cfo| > DOPPLER_GATE_HZ discarded.
# Indoor TX: 3000 Hz  (static TX, TCXO ±1 kHz)
# Outdoor satellite: 0 (disabled)

CFO_TRACK_MAX_JUMP_HZ = 4_000
# Max CFO jump [Hz] between consecutive bursts for same tracker.

SAT_MIN_SEP_HZ = 5_000
# Minimum Doppler separation [Hz] to treat peaks as distinct satellites.

SAT_TIMEOUT_S = 8.0
# Seconds without accepted bursts before removing a tracker.

SAT_COLORS = ["#f4a431", "#4ecdc4", "#a78bfa"]
# Colours [amber, teal, violet] for the 3 satellite slots.

# ═══════════════════════════════════════════════════════════════════════════════
# 2D scan grid
# ═══════════════════════════════════════════════════════════════════════════════

N_AZ       = 360       # azimuth grid points (1° step)
N_EL       = 86        # elevation grid points [5°, 90°] with 1° step
EL_MIN_DEG = 5.0       # lower bound: patches are directional
EL_MAX_DEG = 90.0      # upper bound: satellite can pass through zenith

INDOOR_EL_MAX_DEG = 40.0
# Indoor MUSIC grid cap.  Excludes el > 40° (ceiling reflections at 60–80°).
# Active only when DOPPLER_GATE_HZ > 0.

INDOOR_EL_PREF_MAX_DEG = 22.0
# Preferred elevation upper bound for pick_doa_peak scoring (TX at 10–20°).

INDOOR_EL_PREF_MIN_DEG = 10.0
# Penalise peaks below this elevation (horizon / aliasing).

AZ_FREEZE_EL_DEG = 75.0
# Above this elevation the projected UCA aperture collapses.

# ═══════════════════════════════════════════════════════════════════════════════
# EMA smoothing
# ═══════════════════════════════════════════════════════════════════════════════

COV_ALPHA = 0.85
# Burst-level covariance EMA weight.  τ = 1/(1−α) burst time constant.

AZ_SMOOTH_ALPHA = 0.50
# Circular EMA on per-burst azimuth.  τ = 1/(1−α) ≈ 2 updates.
# Indoor stationary TX: 0.70–0.90  (stable, slow)
# Outdoor satellite:     0.50       (tracks 0.5–1°/s motion)

EL_SMOOTH_ALPHA = 0.50
# Linear EMA on per-burst elevation.

# ═══════════════════════════════════════════════════════════════════════════════
# Hardware phase calibration
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_PHASE_OFFSETS_DEG = [0.0, 24.71, -86.64, 8.72, -3.4]
# Per-channel phase offset [°].  Channel 0 is reference (always 0.0).
# Re-run:  python3 doa_iridium_burst.py --calibrate 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# Quality gates
# ═══════════════════════════════════════════════════════════════════════════════

SQUELCH_ENABLED      = True
SQUELCH_THRESHOLD_DB = -60.0

EIG_SPREAD_MIN_DB = 0.5
# Minimum per-burst eigenvalue spread [dB].  Very permissive; SINR gate is primary.

EIG_SN_GAP_MIN_DB = 0.0
# Disabled post-BPF.

SNR_INST_MIN_DB = -5.0
# Minimum per-burst SINR [dB] from eigenvalue ratio of sample covariance.
# After amplitude normalisation, SINR = (λ₁ − σ²_n) / σ²_n is scale-invariant.
# −5 dB: accepts weak indoor signals; pure noise ≈ 0 dB.

PAPR_INST_MIN_DB = 4.0
# Minimum MUSIC PAPR [dB] for accumulated multi-burst DoA.
# Indoor: 4.0  →  Outdoor clear sky: 8.0–12.0

# ═══════════════════════════════════════════════════════════════════════════════
# Preamble / pilot tone
# ═══════════════════════════════════════════════════════════════════════════════

PILOT_TONE_ENABLED   = False    # False = IRA burst mode
PILOT_TONE_OFFSET_HZ = 3_125   # Iridium IRA preamble tone offset (Rs/8)
PREAMBLE_BPF_BW_HZ   = 8_000   # BPF bandwidth around preamble tone [Hz]
SAMPLE_RATE_HZ       = 1_024_000   # KrakenSDR / Heimdall DAQ rate [Hz]

# ═══════════════════════════════════════════════════════════════════════════════
# Gate parameters
# ═══════════════════════════════════════════════════════════════════════════════

AZ_OUTLIER_ENABLED      = True
AZ_OUTLIER_MAX_DEV_DEG  = 90.0
AZ_OUTLIER_MIN_HISTORY  = 15

AZ_PICK_HINT_MIN_HISTORY = 4
# Burst history before using tracker azimuth median in pick_doa_peak.

PHASE_COHERENCE_ENABLED     = False
PHASE_COHERENCE_MAX_JUMP_DEG = 180.0

# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

HISTORY_LEN        = 50          # samples kept in sliding history plots
UPDATE_INTERVAL_MS = 300         # matplotlib animation refresh [ms]
