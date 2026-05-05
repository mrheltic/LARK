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

ANT_CCW = False
# True  — antennas physically arranged counter-clockwise (CCW) viewed from above.
# False — clockwise (CW).
# With CCW antennas and CW steering: az_est = 360° − az_real for az ≠ 0°/180°.
#
# Verification: place TX at a known azimuth (e.g. 90° East).
#   If az_est ≈  90° → ANT_CCW is correct.
#   If az_est ≈ 270° → flip ANT_CCW (mirror ambiguity due to CW/CCW mismatch).

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
# "MUSIC"        — subspace super-resolution.  PAPR 17-20 dB with nsig=2 indoors.
#                  With MUSIC_DECORR='fb' and NUM_SIGNALS=2 handles indoor multipath.
#                  RECOMMENDED for this setup (validated on real data 2026-05-04).
# "CAPON"        — MVDR: better at low SNR but can lock onto the multipath mean direction.
# "BARTLETT"     — conventional beamformer: PAPR max ~6 dB with 5 antennas (beamwidth 72°)
#                  → always falls below PAPR_INST_MIN_DB threshold. Do not use.
# "ROOT-MUSIC"   — polynomial rooting approach: high resolution, good for UCA.
# "UNITARY-ESPRIT"— real-valued processing: computationally efficient.
# "MFBA-MUSIC"   — Modified Forward-Backward Averaging: improved covariance estimate.

MUSIC_DECORR   = "fb"
# Decorrelation applied to covariance before MUSIC (and Capon).
# 'none'      — no decorrelation (safe default, relies on EMA only)
# 'fb'        — Forward-Backward averaging: improves MUSIC with coherent multipath.
#               SAFE for UCA; reduces the impact of ceiling/wall reflections.
# 'circulant' — for offline debug only (forces cyclic symmetry: valid only for pure UCA).
# 'both'      — circulant + FB: for offline analysis only.

NUM_SIGNALS    = 2        # sorgenti attese simultanee
# INDOOR: use 2 — multipath creates a second coherent signal component
# (wall reflection) that inflates λ₂ relative to λ₃-λ₅.  With nsig=1, MUSIC
# places λ₂ in the noise subspace → flat spectrum → PAPR < 5 dB → bursts rejected.
# With nsig=2: PAPR 12-20 dB, az stable at 234° across all bursts (validated on data).
# OUTDOOR / LOS: use 1 (single source, no multipath).
# 0 = MDL auto-detect (not recommended: estimates N=4 with N_snapshots=2621)

# ── 2D scan grid ─────────────────────────────────────────────────────────────
N_AZ       = 360          # azimuth scan points (360 → 1° step)
N_EL       = 72        # elevation scan points (61 points over [EL_MIN, EL_MAX])
EL_MIN_DEG = 5.0
# Minimum grid elevation [°].
# 5° excludes floor reflections that appear at el≈0° and lead to
# wrong az estimates (data 20260504: Mode A with λ1=41.5 dB, az=171° instead of 234°).
# With EL_MIN=0° the MUSIC peak at el=0° was consistently the highest.
EL_MAX_DEG = 65.0
# Maximum scan grid elevation [°].
# IMPORTANT: set above the maximum expected TX elevation.
# With EL_MAX=35°: burst_data_20260504_161052.npz (493 bursts, -30 dB atten.):
#   geometric fit on ch3/ch4 (stable, std≈13°) → az≈301°, el≈44°.
#   The true elevation is OUTSIDE the grid [5°,35°] → MUSIC finds the
#   second local maximum at az≈170°/el≈5° (boundary ghost, NOT a reflection).
#   46% of bursts saturated at the ceiling el=35°; the rest at floor el=5°.
# 65° gives ~20° margin above geometrically estimated el≈44°.
# If after the new run el stabilises below 35°, lower to 50° (↑resolution).
# With N_EL=61 and range [5–65°]: step = 60/60 = 1.0°.

# ── Covariance EMA accumulation ──────────────────────────────────────────────
COV_ALPHA  = 0.95         # EMA weight  (time constant τ = 1/(1-α) frames)
#                          # 0.95 → ~20 valid-burst frames of memory ≈ 3-4 s
#                          # INDOOR: use HIGH α (0.95-0.98) to average over more
#                          # reflections and get a stable direction estimate.
#                          # OUTDOOR / moving TX: use LOW α (0.50-0.70).

AZ_SMOOTH_ALPHA = 0.60
# Circular EMA smoothing on the per-burst az angle estimate.
# τ = 1 / (1 − α) valid-preamble bursts.
# 0.60 → τ≈2.5 bursts — optimal with outlier gate active.
# Lower to 0.40-0.50 if the TX moves quickly (faster tracking).

EL_SMOOTH_ALPHA = 0.50
# Linear EMA smoothing on the per-burst elevation estimate.
# τ = 1 / (1 − α) valid-preamble bursts.
# 0.50 → τ=2 bursts, 90% settle-time: ~4 bursts ≈ 0.36 s @ 90ms/burst.
# LOWERED from 0.85 (2026-05-05): at 0.85 the EMA took 14 bursts (1.2s)
# to follow an elevation jump — so at az=54° (99 bursts) the EMA was still
# settling while in the az=234° segment the value was stable.
# Theoretical el_sigma at SNR=20dB with 5 antennas is 0.2-0.4° → little smoothing needed.
# The visible el instability comes from geometrically different multipath in the
# two pointing directions, NOT from noise → lower alpha = more responsive.
# The el EMA updates ONLY when the burst is not on a boundary (floor or ceiling)
# to prevent floor/ceiling ghosts from dragging the estimate.

# ── Hardware phase calibration ────────────────────────────────────────────────
CHANNEL_PHASE_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
# Per-channel phase offset correction [degrees], relative to channel 0 (reference).
# Computed on 2026-05-04 from burst_data_20260504_121242_iq.npz (26 stable bursts,
# source at ~234° az / ~46° el, BPF at -8595 Hz).
# To recalibrate: python3 doa_test_868_burst.py --calibrate <source_az>
# Compensates for cable-length differences and ADC input imbalances.
# WITHOUT calibration: instantaneous MUSIC PAPR drops from 33 dB to ~14-17 dB
# for ±10-15° hardware errors, reducing preamble detection rate.
#
# Auto-calibration procedure:
#   1. Place TX at a KNOWN azimuth (e.g. 20.0°) with good line-of-sight.
#   2. Run:  python3 doa_test_868_burst.py --calibrate 20.0
#   3. The script prints the computed CHANNEL_PHASE_OFFSETS_DEG values.
#   4. Copy them here and restart.
#
# Manual calibration (when TX azimuth is unknown):
#   1. Run with demo mode to confirm the array geometry is correct.
#   2. Compare displayed phase_diffs against expected geometry phases.
#   3. Iterate until az_median matches ground truth.

# ── Squelch ───────────────────────────────────────────────────────────────────
SQUELCH_ENABLED      = True
SQUELCH_THRESHOLD_DB = -55.0
# Minimum mean IQ power [dBW] below which the frame is discarded.
# INDOOR: -55 dBW is a good compromise for small rooms (TX nearby → strong signal).
# If you see too many "NO SIGNAL" with TX on, lower to -60 or -65.
# If you see false signals (ghosts) with TX off, raise to -50.

EIG_SPREAD_MIN_DB = 0.5
# Minimum eigenvalue spread (λ_max / λ_noise in dB) to declare signal present.
# IMPORTANT: in indoor environments multipath invalidates the classical subspace
# separation: all eigenvalues appear similar.  Very low threshold to avoid losing
# valid bursts; separation is then handled by the MUSIC PAPR threshold.
#   INDOOR small room: 0.5 dB  (lets almost everything through, PAPR does the filtering)
#   OUTDOOR / LOS:     2.5-3.0 dB  (signal well separated from noise)

EIG_INST_MAX_DB = 50.0
# Raised from 41→50 dB: synthetic demo generates λ1≈47 dB (perfect rank-1 after BPF)
# while real hardware saturated ADC shows λ1≈44-48 dB. 50 dB leaves 3 dB margin.
# Maximum acceptable dominant eigenvalue (λ₁) before discarding the burst.
# When λ₁ exceeds this threshold the signal is strong enough to saturate the ADC.
# Calibration 2026-05-04: saturated ADC → λ₁=41.5 dB (wrong az 171°).
#                         normal ADC    → λ₁=21.6-37.9 dB (correct az 234°).
# RAISED from 38 to 41 dB (2026-05-05): the good session (620 bursts) had λ₁ max=37.9 dB,
# just below the old limit. With GAIN=40 and TX nearby, λ₁ fluctuates up to 38-40 dB
# without saturation → the gate at 38 rejected EVERYTHING, giving only ~12 bursts in 3 min.
# 41.0 dB: 3.5 dB above the good-session max, 0.5 dB below the known saturation point.
# If the incorrect-az problem reappears, lower to 39 and reduce GAIN_DB to 35.
# Do NOT lower below 38 dB.

EIG_SN_GAP_MIN_DB = 0.0
# Minimum signal/noise subspace boundary gap [dB] = λ₂/λ₃ ratio in dB.
# ⚠️  POST-BPF: this gate is DISABLED (0.0) in burst mode.
# Rationale: the pre-BPF calibration (20260504) showed:
#   az≈0° (ghost):    gap ≈ 1.9 dB → MUSIC unreliable
#   az≈54° (real):    gap ≈ 9.9 dB → MUSIC reliable
# But after BPF preamble-tone extraction, the eigenvalue noise floor
# collapses to 0 (λ3 ≈ λ4 ≈ λ5 ≈ 1e-20). The λ2/λ3 ratio becomes unstable:
# with a SINGLE source (rank-1 expected), ev[1] and ev[2] are both numerical
# artefacts → the gap can randomly be 0–20 dB → rejects valid bursts.
# The PAPR gate (≥8 dB) already handles preamble/noise separation post-BPF
# reliably. EIG_SN_GAP post-BPF is redundant and harmful.
# To re-enable (raw IQ environments without BPF): set to 3.0-5.0.

PAPR_INST_MIN_DB = 8.0
# Minimum MUSIC PAPR to accept a preamble burst as valid.
# Intentionally low threshold: selection of "good" bursts is delegated to the
# floor/ceiling gates (el_min+step/2, el_max-step/2) which prevent ghosts from
# updating the az/el EMA — but bursts are still recorded for analysis.
# Raising the threshold drastically reduces the number of recorded bursts (~32% fewer
# already at 10 dB), degrading the EMA estimate due to lack of samples.
# BARTLETT with 5 ants: PAPR max ~6 dB → must stay ≤ 8 dB when using Bartlett.
# OUTDOOR / clean LOS (calibrated array): can raise to 12-20 dB for precision.
# Override at runtime: python3 doa_test_868_burst.py --papr-min 8

# ── Pilot tone extraction ─────────────────────────────────────────────────────
# ⚠️  IMPORTANT: PILOT_TONE_ENABLED should be enabled ONLY in CW mode (continuous tone).
# In BURST mode (IRA) the pilot tone is at +3125 Hz in the preamble and is handled
# automatically by the burst-specific code.  Enabling PILOT_TONE_ENABLED in burst
# mode extracts a 100 kHz band containing only noise → SNR collapses.
SAMPLE_RATE_HZ        = 1_024_000
PILOT_TONE_ENABLED    = False   # ← False for BURST mode, True only for CW mode
PILOT_TONE_OFFSET_HZ  = 100_000 # ← Used only when PILOT_TONE_ENABLED=True (CW mode)
PILOT_TONE_BW_HZ      = 15_000  # ← Used only when PILOT_TONE_ENABLED=True (CW mode)

PREAMBLE_BPF_BW_HZ = 10_000
# BPF bandwidth in burst mode.
# With BW=10 kHz: SNR gain ≈ 10·log10(1024000/10000) ≈ +20 dB; wider window
# handles inaccuracies in tone frequency detection
# (FFT resolution 2000 Hz on 512-sample window → uncertainty ±1000 Hz).
# Do not go below 4000 Hz.

TONE_SEARCH_BW_HZ = 100_000
# Search window for the actual IRA preamble tone frequency [Hz].
# TX (LibreSDR) and RX (KrakenSDR) use independent oscillators → typical LO
# offset ±10–50 kHz at 868 MHz.  The code searches the FFT peak in
# [+3125 - TONE_SEARCH_BW_HZ/2 … +3125 + TONE_SEARCH_BW_HZ/2] and uses that
# frequency for the BPF.  100 kHz covers ±50 kHz offset (>50 ppm at 868 MHz).
# If TX is calibrated or shares clock: lower to 20_000 for better precision.

# ── Per-channel amplitude normalisation ──────────────────────────────────────
AMPLITUDE_NORMALIZE = True
# Normalise each KrakenSDR channel to unit RMS power before covariance.
# Cancels between-channel gain imbalance (up to ~5 dB measured on hardware).
# Does not affect phase relationships (DoA accuracy is preserved).
# Disable for hardware calibration sessions.

# ── Enhanced preprocessing options ────────────────────────────────────────────
# ⚠️ WARNING: advanced preprocessing can CORRUPT the signal if not configured
# correctly.  For indoor use, it is generally better to leave it disabled
# and rely on the temporal EMA (COV_ALPHA) for multipath.
ENABLE_ENHANCED_PREPROCESSING = False
# Enables advanced preprocessing techniques based on research papers:
# - Spatial smoothing to decorrelate coherent signals (indoor reflections)
# - Modified Forward-Backward Averaging (MFBA) for UCA
# - Adaptive filtering to suppress interference
# - Outlier rejection for robust statistics

APPLY_SPATIAL_SMOOTHING = False
# Applies spatial smoothing to decorrelate coherent signals.
# INDOOR: can be useful but requires tuning.  Start with False.

APPLY_MFBA = False
# Applies Modified Forward-Backward Averaging to improve covariance estimation.

APPLY_ADAPTIVE_FILTERING = False
# ⚠️ DANGEROUS: adaptive filtering subtracts the mean of the other channels,
# completely cancelling a signal coherent across all channels (such as our
# IRA burst).  Always leave False for this application.

APPLY_OUTLIER_REJECTION = False
# Applies statistical outlier rejection.  Useful only with strong impulsive noise.

# ── SNR-adaptive algorithm switching ──────────────────────────────────────────
SNR_ADAPTIVE_ENABLED = True
SNR_HIGH_DB = 10.0
# Above this SNR, use the user-selected algorithm (MUSIC / Capon).
# MUSIC at high SNR delivers the best resolution.
SNR_LOW_DB  = 4.0
# Below this SNR, force BARTLETT (most robust, degrades gracefully).
# Between SNR_LOW and SNR_HIGH: use CAPON as a compromise.
# Set SNR_ADAPTIVE_ENABLED = False to always use DOA_ALGORITHM.

# ── Phase coherence gating ──────────────────────────────────────────────────
PHASE_COHERENCE_ENABLED = False
PHASE_COHERENCE_MAX_JUMP_DEG = 120.0
# DISABLED (2026-05-04): the gate compares phase_diffs_inst (from R_inst,
# single burst, std≈90° in indoor multipath) against the phase_hist median
# (now populated only by stable multi-burst R_avg). The mismatch causes false
# rejection of valid bursts → "very few bursts accepted".
# Re-enable only when bursts are consistently high quality (outdoor/LOS).

# ── Circular azimuth outlier rejection ───────────────────────────────────────
AZ_OUTLIER_ENABLED = True
AZ_OUTLIER_MAX_DEV_DEG = 45.0
# Reject an az estimate that deviates more than this from the
# running circular median.  Catches the wild 200° jumps
# observed in indoor multipath (az_std ≈ 80° without gating).
# 45° is conservative; tighten to 30° once array is calibrated.
AZ_OUTLIER_MIN_HISTORY = 5
# Minimum number of accepted estimates before outlier rejection kicks in.

# ── Multi-burst / multi-frame accumulation ──────────────────────────────────
MULTI_BURST_N = 1
# Number of valid bursts to accumulate before running DoA.
# With BPF active (+22 dB SNR), every single burst is already excellent quality.
# LOWERED from 3 to 1 (2026-05-05): with N=3, just 2 failed bursts out of 3 can
# stall the DoA output for several seconds. With BPF, N=1 is sufficient.
# Increase to 2-3 in heavy-multipath environments (deep indoor) to gain
# ~4-5 dB extra averaging at the cost of additional latency.
USE_EMA_FOR_DOA_BELOW_SNR = 6.0
# When instantaneous SNR is below this, use the EMA covariance R for
# DoA instead of the single-frame R_inst.  R_EMA has much lower noise
# at the cost of tracking speed.

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
# NOTE: MUSIC_DECORR is also defined above (section "2D DoA algorithm").
# The value below was "none" and silently overrode the "fb" value defined
# above, because Python uses the last assignment.  Bug removed.
# The effective value is now the one at the top of the DoA section: "fb".
# ─────────────────────────────────────────────────────────────────────────────
# WARNING on circulant smoothing: 'circulant' forces R to be cyclically
# symmetric → eigenvectors = DFT → N-fold star pattern → PAPR≈0 dB on UCA.
# Do NOT use 'circulant' or 'both' on UCA.  'fb' is SAFE (non-circulant).
CAPNT_DECORR = "none"
# Same restriction applies to Capon: circulant forces N-fold symmetry
# in R^{-1} → Capon degrades to Bartlett on a circulant covariance.

