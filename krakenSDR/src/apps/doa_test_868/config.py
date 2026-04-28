# =============================================================================
#  apps/doa_test_868 — 2D DoA test at 868 MHz (azimuth + elevation)
#
#  Validates 2D-MUSIC / 2D-Capon / 2D-Bartlett on the KrakenSDR 5-element UCA
#  in the ISM 868 MHz band.
#
#  Recommended hardware setup
#  --------------------------
#  TX (beacon) : LibreSDR AD9363  →  tx_868_libresdr.py  (CW tone or burst)
#                or any 868 MHz LoRa / Arduino beacon
#  RX (array)  : KrakenSDR 5-ch  →  Heimdall DAQ active on TCP:5000
#  TX position : known azimuth and distance; elevation estimated from geometry
#
#  Data flow
#  ---------
#  Heimdall DAQ → TCP:5000 ──► KrakenIQSource ──► pilot-tone extractor
#    ──► amplitude normalisation ──► EMA covariance ──► 2D-MUSIC (UCA)
#    ──► polar compass + heat-map + az/el history
#
#  Quick configuration
#  -------------------
#  1. Set FREQ_HZ to match the LibreSDR TX frequency.
#  2. Set RADIUS_LAMBDA from the physical UCA radius:
#       r [m] / λ [m]   where  λ = 300e6 / FREQ_HZ
#  3. Calibrate ANT0_OFFSET_DEG with the TX at a known angle (see below).
#  4. If using the pilot-tone feature, set PILOT_TONE_OFFSET_HZ to the same
#     value as --pilot-offset passed to tx_868_libresdr.py (default 100 000 Hz).
#
#  Field calibration of ANT0_OFFSET_DEG
#  -------------------------------------
#  - Place TX due North of the array centre.
#  - Run the script; read the median azimuth estimate.
#  - Set ANT0_OFFSET_DEG = −(measured_az) to shift the estimate to 0°.
#
#  Data-driven tuning notes  (from recordings doa_data_20260427_*.npz)
#  ------------------------------------------------------------------
#  Best session (184244): az_mean=45.4°, snr=31 dB, papr=5.8 dB, 32% valid.
#  High az_std (79°) points to indoor multipath as the dominant error source.
#  Raising IF gain to 40 dB and enabling pilot-tone extraction (+20 dB SNR
#  selectivity) is expected to raise the valid-frame ratio above 80%.
# =============================================================================
from __future__ import annotations

import os as _os
import sys as _sys

# ── Hardware base (re-export) ─────────────────────────────────────────────────
_SRC = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from config_hw import *   # noqa: F401, F403

# ── Array geometry ────────────────────────────────────────────────────────────
N_ANTENNAS     = 5        # number of populated KrakenSDR channels
GEOMETRY       = "UCA"    # Uniform Circular Array — do not change for this test

RADIUS_LAMBDA  = 0.4253
# Physical UCA radius as a fraction of wavelength.
#
#  Formula: R = d / (2·sin(π/N))  where d = inter-element spacing, N = antennas.
#  With λ/2 spacing and N=5:
#    R = (λ/2) / (2·sin(36°)) = 0.5 / (2 · 0.5878) ≈ 0.4253λ  ← correct for KrakenSDR
#
#  Case 1 — λ/2 element spacing (KrakenSDR 868 MHz default):
#    RADIUS_LAMBDA = 0.4253
#
#  Case 2 — known physical radius (e.g. 12.5 cm at 868 MHz, λ=34.56 cm):
#    RADIUS_LAMBDA = 0.125 / (300e6 / 868e6) ≈ 0.362

ANT_CCW = True
# True  — antennas physically arranged counter-clockwise (CCW) viewed from above.
# False — clockwise (CW).
# With CCW antennas and CW steering: az_est = 360° − az_real for az ≠ 0°/180°.

ANT0_OFFSET_DEG = 0.0
# Rotation of antenna 0 relative to geographic North [degrees].
# Used to align the DoA estimate with the compass.
# Measure on the field with TX at a known angle (e.g. 0° = North → ant0 points North).

# ── RF / Frequency ────────────────────────────────────────────────────────────
FREQ_HZ   = 868_100_000   # [Hz] — LibreSDR tx_868_gui.py default (868.1 MHz ISM)
#                          #  BURST mode: π/4-DQPSK IRA preamble tone at +Rs/8 = +3125 Hz
#                          #  CW pilot mode:  set PILOT_TONE_OFFSET_HZ = 100_000
GAIN_DB   = 40            # KrakenSDR IF gain [dB].
#                          #  From field recordings SNR~31 dB at GAIN=20 but only 32% valid
#                          #  frames.  Raising to 40 dB ensures all frames clear squelch.
#                          #  Lower back to 20–30 dB if ADC saturates (check eig values).

# ── 2D DoA algorithm ─────────────────────────────────────────────────────────
DOA_ALGORITHM  = "MUSIC"
# "MUSIC"    — subspace super-resolution (recommended for CW beacon).
#               Temporal EMA (COV_ALPHA=0.97) acts as indoor multipath decorrelation.
# "CAPON"    — MVDR: good at low SNR but may lock onto multipath mean direction.
# "BARTLETT" — conventional beamformer: robust fallback, lower resolution.

NUM_SIGNALS    = 1        # expected simultaneous sources
#                          # 0 = auto-detect via MDL (slower, useful in complex RF env.)

# ── 2D scan grid ─────────────────────────────────────────────────────────────
N_AZ       = 180          # azimuth scan points (180 → 2° step)
N_EL       = 36           # elevation scan points (36 → 2.5° step from 0° to 90°)
EL_MIN_DEG = 0.0          # minimum elevation [°] — 0° for ground-level ISM beacons

# ── Covariance EMA accumulation ──────────────────────────────────────────────
COV_ALPHA  = 0.97         # EMA weight  (time constant τ = 1/(1-α) frames)
#                          # 0.97 → ~33 frames of memory.
#                          # CW source stays stable; reflections slowly drift in phase →
#                          # temporal averaging is the correct multipath decorrelation for UCA.
#                          # Lower to 0.50–0.70 for moving sources.

# ── Squelch ───────────────────────────────────────────────────────────────────
SQUELCH_ENABLED      = True
SQUELCH_THRESHOLD_DB = -60.0
# Minimum mean IQ power [dBW] below which the frame is discarded.
# From field data: valid frames at GAIN=20 had power ≈ −50…−40 dBW.
# −60 dBW is a conservative floor; raise to −50 if noise bursts cause ghost estimates.

EIG_SPREAD_MIN_DB = 2.5
# Minimum eigenvalue spread (λ_max / λ_noise in dB) to declare signal present.
# Lowered from 3.0 → 2.5 (data-driven, burst session 20260428, N=3101):
#   all 3101 detected bursts had λ1 ≥ 2.5 dB even at TX gain = −40 dB (SNR≈0 dB).
#   Valid bursts (PAPR≥4 dB) showed λ1 ≈7 dB.  The 2.5 dB floor rejects only
#   pure-noise frames while accepting weak-signal preamble captures.

# ── Pilot tone extraction ─────────────────────────────────────────────────────
SAMPLE_RATE_HZ        = 1_024_000
# Heimdall DAQ sample rate [Hz] — must match daq_chain_config_868.ini sample_rate.
# Used to compute FFT bin indices for pilot-tone extraction.

PILOT_TONE_ENABLED    = True
# Enable narrow-band extraction of the LibreSDR CW pilot tone.
# When enabled, each IQ frame is bandpass-filtered around PILOT_TONE_OFFSET_HZ
# before the covariance is computed.  This rejects broadband noise and interference,
# yielding ~20 dB SNR improvement (10·log10(sample_rate / PILOT_TONE_BW_HZ)).
# Requires LibreSDR to be transmitting at  LO_freq + PILOT_TONE_OFFSET_HZ.
# Set PILOT_TONE_ENABLED = False when using a non-LibreSDR beacon (e.g. Arduino).

PILOT_TONE_OFFSET_HZ  = 100_000
# Pilot tone offset from the Heimdall LO centre frequency [Hz].
# 100 kHz is chosen for exact FFT bin alignment at 1.024 MSPS / CPI 131072:
#   bin = 100 000 × 131 072 / 1 024 000 = 12 800  (integer → zero spectral leakage)
# Must match --pilot-offset passed to tx_868_libresdr.py.

PILOT_TONE_BW_HZ      = 15_000
# Extraction bandwidth [Hz].  Wider = higher tolerance for TX frequency drift;
# narrower = more noise rejection.  Raised from 10 kHz → 15 kHz (2026-04-28):
# at 868.1 MHz the AD9363 TCXO drift can reach ±3–4 kHz; 15 kHz ensures the
# tone stays within the extraction window.  SNR penalty vs 10 kHz: <1 dB.

# ── Per-channel amplitude normalisation ──────────────────────────────────────
AMPLITUDE_NORMALIZE = True
# Normalise each KrakenSDR channel to unit RMS power before covariance.
# Cancels between-channel gain imbalance (up to ~5 dB measured on hardware).
# Does not affect phase relationships (DoA accuracy is preserved).
# Disable for hardware calibration sessions.

# ── Display ───────────────────────────────────────────────────────────────────
UPDATE_INTERVAL_MS = 200   # animation refresh interval [ms]  (200 ms ≈ 5 fps)
HISTORY_LEN        = 60    # rolling history length [frames]

# ── High-elevation adaptive algorithm ────────────────────────────────────────
HIGH_EL_THRESHOLD_DEG = 50.0
# Above this elevation [degrees] MUSIC loses azimuth resolution because
# cos(el) → 0 reduces inter-channel phase spread → nearly flat MUSIC spectrum.
# Transition to HIGH_EL_ALGO automatically above this threshold.
# Also used as a fallback when PAPR drops below _PAPR_FLAT_DB.

HIGH_EL_ALGO = "BARTLETT"
# Algorithm used when el_est ≥ HIGH_EL_THRESHOLD_DEG.
# BARTLETT: conventional beamformer — degrades gracefully, works at all elevations.
# CAPON:    intermediate; better resolution than Bartlett, less stable than MUSIC.

# ── Covariance decorrelation (anti-multipath) ─────────────────────────────────
MUSIC_DECORR = "none"
# Pre-processing decorrelation for 2D-MUSIC on UCA.
# IMPORTANT: 'circulant' and 'fb' BREAK 2D-MUSIC on UCA:
#   circulant smoothing forces R to be cyclically symmetric →
#   eigenvectors = DFT vectors → N-fold star pattern in MUSIC spectrum → PAPR~0 dB.
# For UCA the correct multipath decorrelation is temporal EMA (COV_ALPHA=0.97).
# Leave 'none' unless running offline synthetic experiments.
CAPNT_DECORR = "none"
# Same restriction applies to Capon: circulant forces N-fold symmetry
# in R^{-1} → Capon degrades to Bartlett on a circulant covariance.

