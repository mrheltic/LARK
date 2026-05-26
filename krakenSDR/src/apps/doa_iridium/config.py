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
# Antenna model: Taoglas GPDF6010.A — all-band GNSS stacked patch, RHCP via HC125A hybrid coupler.
#  • Spec'd range: 1176–1602 MHz (max listed: GLONASS G1 at 1602 MHz).
#    Iridium at 1626.270 MHz is ~24 MHz beyond the datasheet; gain likely ~1–2 dBic
#    (vs 4.0 dBic at 1575 MHz and 2.7 dBic at 1602 MHz).  VSWR probably still < 2:1.
#  • RHCP requires HC125A hybrid coupler on each element; without couplers the
#    output is linearly polarised → amplitude modulation across UCA elements → DoA bias.
#  • Phase Center Offset (PCO) ≈ +1.0 cm above ground plane at 1575–1602 MHz.
#    Vertical PCO at el=45° → ~0.7 cm path ≈ 0.038λ → ~14° phase smear in elevation.
#    Effect on azimuth: negligible (symmetric).  Elevation estimates biased +2–5°.
#  • Ground plane ≥ Ø100 mm required; at R=78 mm UCA, adjacent elements are ~38 mm
#    edge-to-edge → significant mutual coupling and ground-plane overlap.
#    Empirical phase calibration (CHANNEL_PHASE_OFFSETS_DEG) absorbs this effect.

RADIUS_LAMBDA  = 0.4253
# UCA radius in wavelengths for λ/2 element spacing:
#   R = (λ/2) / (2·sin(π/5)) ≈ 0.4253 λ
# At 1626.270 MHz (λ ≈ 18.43 cm):  physical radius ≈ 7.84 cm

ANT_CCW = False
# False = elements CLOCKWISE (CW) viewed from above — standard KrakenSDR UCA.
# Set True if azimuth estimates are mirrored (East ↔ West swapped).

ANT0_OFFSET_DEG = 0.0
# Rotation of antenna 0 from geographic North [°].
# Re-calibrate:  python3 iridium_burst_doa_runner.py --calibrate 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# RF / Frequency
# ═══════════════════════════════════════════════════════════════════════════════

FREQ_HZ = 1_626_270_000
# Iridium Ring Alert channel (1626.270 MHz).

GAIN_DB = 40
# IF gain [dB] applied to all KrakenSDR channels.  RTL-SDR max ≈ 49.6 dB.
# Valid discrete steps near this range: 37.2, 38.6, 40.2, 42.1, 43.4 dB.
# GPDF6010.A at 1626 MHz (24 MHz above datasheet spec) has ~1–2 dBic gain
# vs 4.0 dBic at 1575 MHz → use 2–3 dB more gain to compensate.
# Outdoor: 38–42.  Lower to 35 if ADC overdrive flags appear.

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

MAX_SATELLITES = 3
# Maximum simultaneously tracked satellites.
# Iridium L-band: typically 1–3 above horizon.  Indoor single-TX: set to 1.

MULTI_BURST_N = 4
# Number of preamble bursts accumulated per DoA estimate.
# Each burst contributes ~519 preamble IQ samples at 1.024 MSPS;
# N=4 → 2076 snapshots, fully overdetermined for a 5-element array.
# Iridium IRA superframe period = 90 ms → N bursts span N×90 ms.
# A satellite at low elevation moves ≈ 2°/s in azimuth, so:
#   N=4  → 360 ms window → 0.72° smearing → MUSIC FWHM ~1°  (correct)
#   N=12 → 1.08 s         → 2.2° smearing → MUSIC FWHM ~3°  (borderline)
#   N=100→ 9 s            → 18° smearing  → completely unusable
# Indoor static single-source: 16–24 recommended (no motion, more SNR averaging).
# Outdoor satellite: 4–6.

DOPPLER_SCAN_BW_HZ = 45_000
# Half-width [Hz] of FFT scan around preamble tone (3125 Hz).

DOPPLER_GATE_HZ = 0
# CFO rejection gate [Hz].  Peaks with |cfo| > DOPPLER_GATE_HZ discarded.
# Indoor TX: 3000 Hz  (static TX, TCXO ±1 kHz)
# Outdoor satellite: 0 (disabled)

# ─── TLE-aided elevation prior ───────────────────────────────────────────────
TLE_EL_PRIOR_ENABLED   = False
# Apply a Gaussian dB penalty to the MUSIC 2D spectrum along the elevation
# axis, centred on the TLE-predicted satellite elevation.  This corrects the
# systematic +30° EL bias caused by the flat UCA having minimal phase slope
# in the elevation dimension.  Automatically disabled in indoor/demo modes.
# Keep disabled by default until Doppler/LO offset is calibrated; a bad
# Doppler match can force the spectrum toward the wrong elevation.

TLE_DOPPLER_LO_OFFSET_HZ = 0.0
# Constant LO offset [Hz] used for TLE Doppler matching:
# effective_doppler_hz = measured_cfo_hz - TLE_DOPPLER_LO_OFFSET_HZ.
# Example: if measured CFO is shifted by +4 kHz vs predicted Doppler,
# set this to +4000.

TLE_EL_PRIOR_SIGMA_DEG = 20.0
# Gaussian sigma [°] of the elevation prior.  Smaller = tighter constraint.
# 20° → ±20° costs 4.3 dB,  ±50° costs 43 dB (strong suppression).
# Widen to 30° if TLE Doppler match is unreliable (LO offset > 10 kHz).

CFO_TRACK_MAX_JUMP_HZ = 2_500
# Max CFO jump [Hz] between consecutive bursts for same tracker.
# Outdoor satellites change Doppler smoothly (tens of Hz between bursts).
# 2.5 kHz rejects cross-satellite hopping while preserving true tracks.

SAT_MIN_SEP_HZ = 8_000
# Minimum Doppler separation [Hz] to treat peaks as distinct satellites.

SAT_TIMEOUT_S = 12.0
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

INDOOR_EL_PREF_MAX_DEG = 22.0
# Preferred elevation upper bound for pick_doa_peak scoring.
# Keep wider indoor range to avoid artificial lock around ~15-20°.

INDOOR_EL_PREF_MIN_DEG = 5.0
# Penalise peaks below this elevation (horizon / aliasing), but keep floor low
# enough to track realistic indoor setups.

INDOOR_SINGLE_SOURCE_RELAX_GATES = True
# Indoor near-field with a single source is multipath-heavy: relaxing AZ/phase
# continuity gates improves acceptance and prevents tracking stalls.

AZ_FREEZE_EL_DEG = 60.0
# Above this elevation the projected UCA aperture collapses.

# ═══════════════════════════════════════════════════════════════════════════════
# EMA smoothing
# ═══════════════════════════════════════════════════════════════════════════════

COV_ALPHA = 0.93
# Burst-level covariance EMA weight.  τ = 1/(1−α) burst time constant.
# 0.93 → τ ≈ 14 burst-times ≈ 90 s at 1.83 bursts/s, multi=12.

AZ_SMOOTH_ALPHA = 0.40
# Circular EMA on per-burst azimuth.  Outdoor: lower α tracks moving target.
# 0.55 → τ ≈ 2.2 estimates.  With N=4 at ~2 est/s: time constant ≈ 1.1 s.
# Satellite at el=20° moves ~2°/s → EMA lag ≈ 2.2°.  0.70 gives 3.3x more lag.

EL_SMOOTH_ALPHA = 0.40
# Linear EMA on per-burst elevation.  Same rationale as AZ_SMOOTH_ALPHA.

# ═══════════════════════════════════════════════════════════════════════════════
# Hardware phase calibration
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_PHASE_OFFSETS_DEG = [0.0, 24.71, -86.64, 8.72, -3.4]
# Per-channel phase offset [°].  Channel 0 is reference (always 0.0).
# Re-run:  python3 iridium_burst_doa_runner.py --calibrate 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# Quality gates
# ═══════════════════════════════════════════════════════════════════════════════

SQUELCH_ENABLED      = True
SQUELCH_THRESHOLD_DB = -60.0

EIG_SPREAD_MIN_DB = 0.5
# Minimum per-burst eigenvalue spread [dB].  Very permissive; SINR gate is primary.

EIG_SN_GAP_MIN_DB = 0.0
# Disabled post-BPF.

SNR_INST_MIN_DB = -10.0   # per-burst eigenvalue SNR: unreliable at weak signal; PAPR gate is the real filter
# Minimum per-burst SINR [dB] from eigenvalue ratio of sample covariance.
# After amplitude normalisation, SINR = (λ₁ − σ²_n) / σ²_n is scale-invariant.
# Outdoor clear sky: 5–10 dB.  3 dB = permissive (first lock / debug).  −5 dB: indoor/weak only.

PAPR_INST_MIN_DB = 4.0   # multi-burst PAPR gate; outdoor satellite signal is weak
# Minimum MUSIC PAPR [dB] for accumulated multi-burst DoA.
# With MULTI_BURST_N=4 the observed mean is ~29 dB; setting 22 dB rejects the
# bottom ~20% low-quality estimates where multipath dominates.
# Indoor: 4.0  →  Outdoor clear sky: 22.0  →  4.0 = permissive debug mode

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
AZ_OUTLIER_MAX_DEV_DEG  = 45.0
AZ_OUTLIER_MIN_HISTORY  = 8
AZ_OUTLIER_RELOCK_STREAK = 12
# If AZ_OUTLIER rejects this many bursts in a row, force tracker re-lock.
# Prevents freeze after abrupt antenna/TX direction changes.

AZ_PICK_HINT_MIN_HISTORY = 4
# Burst history before using tracker azimuth median in pick_doa_peak.

PHASE_COHERENCE_ENABLED     = False
PHASE_COHERENCE_MAX_JUMP_DEG = 55.0
PHASE_COHERENCE_RELOCK_STREAK = 12
# If phase coherence rejects this many bursts in a row, reset phase baseline.

GATE_RELOCK_BYPASS_BURSTS = 12
# After a relock event, temporarily bypass AZ/phase gates for N accepted bursts
# so the tracker can converge to the new direction without wiping plot history.

# Doppler zero-crossing event diagnostics.
# Keep disabled for indoor static TX because CFO can jitter around zero and
# produce frequent sign flips that are not real pass apex events.
DOPPLER_XZ_ENABLED = True
DOPPLER_XZ_MIN_ABS_HZ = 2000.0
DOPPLER_XZ_MIN_JUMP_HZ = 4000.0
DOPPLER_XZ_DEADTIME_S = 30.0
# Outdoor: Doppler changes sign near pass apogee; deadtime 30s avoids duplicates.

# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

HISTORY_LEN        = 100         # samples kept in sliding history plots
UPDATE_INTERVAL_MS = 300         # matplotlib animation refresh [ms]

SKYPLOT_LOBE_WIDTH_DEG = 12.0
SKYPLOT_LOBE_ALPHA = 0.10
# Fill opacity for skyplot lobes.

# ═══════════════════════════════════════════════════════════════════════════════
# OUTDOOR SATELLITE TRACKING — override these after indoor calibration
# ═══════════════════════════════════════════════════════════════════════════════
# For outdoor passes (real Iridium satellites), copy these values over the
# defaults above, or use CLI overrides:
#   --fd-max 0 --gain 38 --multi 4 --snr-min -10 --papr-min 4
#
#   DOPPLER_GATE_HZ       = 0          # disabled (full ±45 kHz scan)
#   CFO_TRACK_MAX_JUMP_HZ = 2_500      # reject cross-satellite Doppler hops (N×90ms jumps)
#   GAIN_DB               = 38         # GPDF6010.A passive: ~1–2 dBic at 1626 MHz; keep gain high
#   MULTI_BURST_N         = 4          # 4 bursts × 90 ms = 360 ms window; satellite moves < 1°
#   AZ_SMOOTH_ALPHA       = 0.40       # fast EMA for moving target (~2°/s azimuth rate)
#   EL_SMOOTH_ALPHA       = 0.40
#   INDOOR_EL_MAX_DEG     = 90         # full elevation range
#   SNR_INST_MIN_DB       = -10        # accept weak signals (8-bit ADC, ~42 dB dynamic range)
#   DOPPLER_XZ_ENABLED    = True       # zero-crossing = max elevation event
#   MAX_SATELLITES        = 3          # up to 3 satellites simultaneously
