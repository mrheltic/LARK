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
# At  868 MHz     (λ ≈ 34.56 cm):  physical radius ≈ 14.7 cm

ANT_CCW = False
# False = elements are CLOCKWISE (CW) viewed from above — standard KrakenSDR UCA.
# The steering vector uses φ_k = 2πk/N (k=0 at North, k=1 at 72° clockwise).
# Set True if azimuth estimates are mirrored (East ↔ West swapped).

ANT0_OFFSET_DEG = -91.7  # auto-cal 2026-05-21 20:49 median_az=91.7° tx_az=0.0°
# Rotation of antenna 0 from geographic North [°].
# Re-calibrate after any physical change:
#   python3 doa_iridium_burst.py --calibrate 0.0
# (place TX at true North, run until completion)

# ═══════════════════════════════════════════════════════════════════════════════
# RF / Frequency
# ═══════════════════════════════════════════════════════════════════════════════

FREQ_HZ = 1_626_270_000
# Iridium Ring Alert channel (1626.270 MHz).
# For ISM 868 MHz indoor lab: FREQ_HZ = 868_100_000

GAIN_DB = 45
# IF gain [dB] applied to all KrakenSDR channels.  RTL-SDR max ≈ 49.6 dB.
# Indoor −60 dB TX at ~1 m:  45–49 dB  (satellite-equivalent SNR)
# Indoor −40 dB TX at ~1 m:  35–40 dB  (high-SNR diagnosis)
# Outdoor real Iridium:      30–40 dB  (ground level −70…−50 dBm)

# ═══════════════════════════════════════════════════════════════════════════════
# 2D DoA algorithm
# ═══════════════════════════════════════════════════════════════════════════════

DOA_ALGORITHM  = "MUSIC"
# "MUSIC"          — best PAPR with forward-backward decorrelation (default)
# "CAPON"          — MVDR: less sensitive to array calibration errors
# "ROOT-MUSIC"     — polynomial super-resolution, excellent for single source
# "MFBA-MUSIC"     — Modified Forward-Backward Averaging MUSIC
# "UNITARY-ESPRIT" — real-valued, fast, good for high-elevation short passes

MUSIC_DECORR   = "none"
# Covariance decorrelation strategy:  'none'  /  'fb'  /  'circulant'.
# 'none' — default.  The multi-burst sample covariance is already full rank;
#          indoor multipath phases vary burst-to-burst, providing natural
#          decorrelation.  Synthetic test: PAPR=38.8 dB with 'none' vs
#          35.3 dB with 'fb'.
# 'fb'   — forward-backward averaging.  Use only when multipath is strongly
#          coherent (specular reflection with fixed delay).
# 'circulant' — DO NOT use for UCA: enforces N-fold symmetry (star-shaped spectrum).

NUM_SIGNALS = 0
# Expected sources PER SATELLITE for MUSIC subspace split.
# 0 = auto-MDL (Wax & Kailath 1985): automatic from eigenvalue ratios.
#     Recommended for indoor where direct + reflected paths vary.
#     Peak selection via pick_doa_peak_uca_2d for multi-peak handling.
# 1 = force K=1 (single path).  Use only in very clean environments.

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-satellite detection
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SATELLITES = 1
# Maximum simultaneously tracked satellites.
# Iridium L-band: typically 1–3 satellites above horizon.
# Indoor single-TX: set to 1.

MULTI_BURST_N = 15
# Number of preamble bursts accumulated per DoA estimate.
# Each burst contributes ~2500 preamble IQ samples; the sample covariance
#   R = X_big @ X_big^H / (N × n_pre)   with  X_big = hstack(X_pre)
# is full-rank for N ≥ 5, allowing MUSIC to resolve direct + reflected paths.
# With 30 bursts: ~75k snapshots → eigenvalue spread 2–3×, PAPR > 20 dB.
# Indoor single-TX: 30 bursts → ~15 s per estimate at 50% acceptance.
# Outdoor satellite: reduce to 10–15 for faster tracking (< 5 s).

DOPPLER_SCAN_BW_HZ = 45_000
# Half-width [Hz] of the FFT scan around the nominal preamble tone (3125 Hz).
# Covers ±40 kHz — maximum Iridium LEO Doppler at 1626 MHz.

DOPPLER_GATE_HZ = 2_000
# CFO (Doppler − 3125 Hz) rejection gate [Hz].
# Peaks with |cfo_hz| > DOPPLER_GATE_HZ are discarded before BPF.
# Indoor TX (−60 dB): 2000 Hz  (static TX, fd≈0, TCXO ±1 kHz)
# Outdoor satellite:  0       (disabled, full ±45 kHz pass)

CFO_TRACK_MAX_JUMP_HZ = 2000
# Maximum allowed CFO jump between consecutive bursts for the same tracker.
# 2000 Hz reduces fd_rej (~20%) from gate-edge FFT spurs.

CFO_EMA_ALPHA = 0.96
# Tracker CFO EMA weight.  τ ≈ 25 bursts for stable tone lock.

DOPPLER_XZ_ENABLED = False
# Doppler zero-crossing detection (useful for outdoor LEO passes).
# Disabled for indoor static TX.

SAT_MIN_SEP_HZ = 5_000
# Minimum Doppler separation [Hz] to treat two FFT peaks as distinct satellites.
# < 5 kHz → same satellite;  ≥ 5 kHz → separate satellite.

SAT_TIMEOUT_S = 8.0
# Seconds without accepted bursts before removing a satellite tracker.

IQ_GET_FRAME_TIMEOUT_S = 15.0
# Frame queue timeout [s].  CPI ~4 MB: after DAQ restart need > 5 s.

HEIMDALL_RECV_TIMEOUT_S = 45.0
# Heimdall socket recv timeout [s].  Avoids "timed out" during delay_sync cal.

SAT_COLORS = ["#f4a431", "#4ecdc4", "#a78bfa"]
# Colours [amber, teal, violet] for the 3 satellite slots in the polar plot.

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
# Preferred elevation upper bound for pick_doa_peak scoring.
# TX at 10–20°; ceiling reflections at 40–80° are heavily penalised above this.

INDOOR_EL_PREF_MIN_DEG = 10.0
# Penalise peaks below this elevation (horizon / aliasing).

AZ_FREEZE_EL_DEG = 75.0
# Above this elevation the projected UCA aperture shrinks by cos(el) ≤ 0.26,
# making azimuth unreliable.  The EMA phasor freezes above this threshold.

# ═══════════════════════════════════════════════════════════════════════════════
# EMA smoothing
# ═══════════════════════════════════════════════════════════════════════════════

COV_ALPHA = 0.85
# Burst-level covariance EMA weight.  τ = 1/(1−α) burst time constant.
# Indoor stationary TX:  0.85–0.90  (τ ≈ 7 bursts ≈ 600 ms)
# Outdoor satellite:     0.50–0.70  (faster tracking)

AZ_SMOOTH_ALPHA = 0.70
# Circular EMA on per-burst azimuth.  τ = 1/(1−α) ≈ 3.3 updates.
# Indoor stationary TX: 0.90–0.95  (max stability, slow response)
# Outdoor satellite:     0.50–0.70  (tracks 0.5–1°/s motion)

EL_SMOOTH_ALPHA = 0.70
# Linear EMA on per-burst elevation.  Same τ as azimuth for balanced response.

# ═══════════════════════════════════════════════════════════════════════════════
# Hardware phase calibration
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_PHASE_OFFSETS_DEG = [0.0, 39.21, -95.7, 12.68, -16.57]  # auto-cal 2026-05-21 20:49 from 1398 bursts (117 snapshots) az=0.0°
# Per-channel phase offset [°].  Channel 0 is reference (always 0.0).
# Auto-calibrated 2026-05-21 18:26 from 1442 bursts at az=0.0°.
# Re-run:  python3 doa_iridium_burst.py --calibrate 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# Quality gates
# ═══════════════════════════════════════════════════════════════════════════════

SQUELCH_ENABLED      = True
SQUELCH_THRESHOLD_DB = -60.0

EIG_SPREAD_MIN_DB = 0.5
# Minimum per-burst eigenvalue spread [dB].  Very permissive; SINR gate is more
# selective.  Raise to 2.5 in clean outdoor LOS.

EIG_SN_GAP_MIN_DB = 0.0
# Disabled post-BPF.

SNR_INST_MIN_DB = -3.0
# Minimum per-burst SINR [dB] from eigenvalue ratio of sample covariance.
# After amplitude normalisation each channel has unit variance → eigenvalues
# are all ≈ 1 regardless of signal power.  SINR = (λ₁ − σ²_n) / σ²_n is
# scale-invariant: pure noise ≈ 0 dB, strong source > 10 dB.
# Indoor −60 dB: single burst often −1…+2 dB; averaging over N bursts raises SNR.
# Override at runtime: --snr-min <value>

PAPR_INST_MIN_DB = 4.0
# Minimum MUSIC PAPR [dB] for the accumulated multi-burst DoA.
# With multi-burst sample covariance:
#   direct + 1 moderate reflection : PAPR ≈ 8–15 dB
#   direct only (clean LOS)        : PAPR ≈ 20–40 dB
#   pure noise                     : PAPR ≈ 0–1 dB
# 4 dB: reject noise, accept real signals.
# Indoor: 4.0  →  Outdoor clear sky: 8.0–12.0

# ═══════════════════════════════════════════════════════════════════════════════
# Preamble / pilot tone
# ═══════════════════════════════════════════════════════════════════════════════

PILOT_TONE_ENABLED   = False    # False = IRA burst mode (preamble tone auto-detected)
PILOT_TONE_OFFSET_HZ = 3_125   # Iridium IRA preamble tone offset (Rs/8)
PREAMBLE_BPF_BW_HZ   = 8_000   # BPF bandwidth around preamble tone [Hz]
# 8 kHz gives ~21 dB SNR gain vs full band; rejects DQPSK data energy.
SAMPLE_RATE_HZ       = 1_024_000   # KrakenSDR / Heimdall DAQ rate [Hz]

# ═══════════════════════════════════════════════════════════════════════════════
# Gate parameters
# ═══════════════════════════════════════════════════════════════════════════════

AZ_OUTLIER_ENABLED      = True
# Reject multi-modal MUSIC azimuth jumps once tracker has history.
AZ_OUTLIER_MAX_DEV_DEG  = 35.0
AZ_OUTLIER_MIN_HISTORY  = 6

AZ_PICK_HINT_MIN_HISTORY = 4
# Burst history length before using tracker azimuth median in pick_doa_peak.

PHASE_COHERENCE_ENABLED     = True
PHASE_COHERENCE_MAX_JUMP_DEG = 55.0
# Indoor −60 dB: typical jumps < 35°; 55° rejects outliers without blocking lock.

# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

HISTORY_LEN        = 100         # samples kept in sliding history plots
UPDATE_INTERVAL_MS = 300         # matplotlib animation refresh [ms]
