# =============================================================================
#  apps/doa_iridium — 2D DoA on Iridium IRA bursts at 1626.270 MHz
#
#  Hardware setup
#  --------------
#  TX (optional): LibreSDR AD9363  →  tx/indoor_1626.py  (indoor lab only)
#  RX           : KrakenSDR 5-ch UCA RHCP, Heimdall DAQ on TCP:5000
#
#  What to change
#  ---------------
#  1. SCENARIO          — select operating mode (see below)
#  2. FREQ_HZ / GAIN_DB — only if changing frequency or gain
#  3. Calibration       — auto-updated by --calibrate, do not edit manually
#  4. Optional overrides — uncomment the section at the bottom for fine-tuning
# =============================================================================
from __future__ import annotations

import os as _os
import sys as _sys

_SRC = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from config_hw import *   # noqa: F401, F403  (HEIMDALL_HOST/PORT, SAMPLE_RATE_HZ …)

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO — single variable to switch between operating modes
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO = "indoor_ira"
# "indoor_ira"  — Static TX (LibreSDR --mode ira, CFO ≈ 0 Hz).
#                 Single burst DoA, Doppler gate ±3 kHz.
#                 Use with:  python3 tx/indoor_1626.py --mode ira --gain -50 --cyclic
#
# "indoor_pass" — TX simulates LEO pass (LibreSDR --mode pass, Doppler chirp ±28 kHz).
#                 Short averaging (4 bursts ≈ 0.5 s), Doppler gate OFF.
#                 Use with:  python3 tx/indoor_1626.py --mode pass --gain -50 --cyclic
#
# "outdoor"     — Real Iridium satellite (no TX required).
#                 Short averaging (4 bursts), Doppler gate OFF, up to 3 simultaneous
#                 satellites, multi-channel IRA scan.

# ═══════════════════════════════════════════════════════════════════════════════
# RF / Frequency
# ═══════════════════════════════════════════════════════════════════════════════

FREQ_HZ = 1_626_270_000      # Iridium Ring Alert channel (1626.270 MHz)
GAIN_DB = 40                  # RTL-SDR gain [dB]. Indoor: 40-45; outdoor: 35-37.

# ═══════════════════════════════════════════════════════════════════════════════
# UCA geometry
# ═══════════════════════════════════════════════════════════════════════════════

N_ANTENNAS      = 5
GEOMETRY        = "UCA"
RADIUS_LAMBDA   = 0.4253    # UCA radius in wavelengths → R ≈ 7.84 cm at 1626 MHz
ANT_CCW         = False     # False = CW (clockwise from above); True = CCW
ANT0_OFFSET_DEG = 0.0       # Antenna-0 rotation from geographic North [°]

# ═══════════════════════════════════════════════════════════════════════════════
# DoA algorithm
# ═══════════════════════════════════════════════════════════════════════════════

DOA_ALGORITHM = "MUSIC"
# "MUSIC"            — Recommended.  Best resolution for UCA.
# "CAPON"/"BARTLETT" — More robust but lower resolution.
# "PHASE-FIT"        — Indoor-calibrated only; most reactive.
# "ROOT-MUSIC" / "MFBA-MUSIC" / "UNITARY-ESPRIT" — Experimental.

MUSIC_DECORR = "none"
# 'none' is mandatory for UCA: fb/circulant decorrelation distorts R and shifts the peak by 180°.

NUM_SIGNALS = 1
# Sources per satellite for MUSIC subspace.  1 = force rank-1 (stable).
# 0 = auto-MDL: not recommended (MDL tends to return k=4 on a rank-1 signal).

# ═══════════════════════════════════════════════════════════════════════════════
# Hardware calibration  (auto-updated by:  python3 iridium_burst_doa_runner.py --calibrate AZ)
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_PHASE_OFFSETS_DEG = [0.0, 54.95, 137.24, 133.58, 48.31]  # auto-cal 2026-05-29 11:37 from 65 bursts (13 snapshots) az=0.0° el=60.0°
DOA_AZ_OFFSET_DEG = 0.00   # frame offset from --calibrate (do not edit manually)
DOA_EL_OFFSET_DEG = 0.00   # reset: offsets 126/20 were from fb-decorr + el_max=40 clipping
CAL_REFERENCE_AZ_DEG = 0.00
CAL_REFERENCE_EL_DEG = 60.00

# ═══════════════════════════════════════════════════════════════════════════════
# DoA grid / display  (fixed — independent of scenario)
# ═══════════════════════════════════════════════════════════════════════════════

N_AZ          = 360     # Azimuth grid points [1°/step]
N_EL          = 86      # Elevation grid points (5°–90°, 1°/step)
EL_MIN_DEG    = 5.0     # Minimum displayed elevation [°]
EL_MAX_DEG    = 90.0    # Maximum displayed elevation [°]  (reduced by scenario for indoor)
COV_ALPHA     = 0.93    # Default covariance EMA weight per burst (overridden by scenario)
HISTORY_LEN   = 100     # Tracker history buffer length

# ═══════════════════════════════════════════════════════════════════════════════
# Scenario profiles
# ═══════════════════════════════════════════════════════════════════════════════
# Operational parameters for each scenario, applied by _load_scenario_profile()
# in the runner at startup.  Priority: CLI argument > explicit config.py variable
# > scenario default.

SCENARIO_PROFILES: dict[str, dict] = {
    "indoor_ira": dict(
        INDOOR_TX_MODE               = "ira",
        MAX_SATELLITES               = 1,
        MULTI_BURST_N                = 1,        # SINGLE-BURST: indoor multipath destroys
                                                  # rank-1 structure even with N=2 (2026-06-02).
                                                  # Tracker EMA (α=0.88) provides temporal smoothing.
        DOPPLER_GATE_HZ              = 3_000,
        CFO_TRACK_MAX_JUMP_HZ        = 2_500,
        PAPR_INST_MIN_DB             = 3.0,
        SNR_INST_MIN_DB              = -3.0,
        AZ_SMOOTH_ALPHA              = 0.88,
        EL_SMOOTH_ALPHA              = 0.65,
        COV_ALPHA                    = 0.95,
        ENERGY_DETECT_THRESHOLD      = 3.0,
        INDOOR_EL_MAX_DEG            = 75.0,   # tilted array: TX at ~60° elevation
        INDOOR_EL_PREF_MAX_DEG       = 70.0,
        INDOOR_EL_PREF_MIN_DEG       = 45.0,
        AZ_OUTLIER_ENABLED           = True,
        AZ_OUTLIER_MAX_DEV_DEG       = 30.0,   # wider tolerance for indoor multipath
        AZ_OUTLIER_MIN_HISTORY       = 8,
        AZ_OUTLIER_RELOCK_STREAK     = 40,    # was 20; wait longer before relock
        PHASE_COHERENCE_ENABLED      = True,
        GATE_RELOCK_BYPASS_BURSTS    = 12,
        DOPPLER_XZ_ENABLED           = False,
        IRA_SCAN_OFFSETS_HZ          = [0],
        SAT_MIN_SEP_HZ               = 8_000,
        SAT_TIMEOUT_S                = 12.0,
        INDOOR_SINGLE_BURST_FALLBACK = True,
        PAPR_SINGLE_BURST_MIN_DB     = 2.0,
        INDOOR_IRA_TONE_LOCK_BW_HZ        = 150.0,
        INDOOR_IRA_TONE_EMA_ALPHA          = 0.92,
        INDOOR_IRA_STATIC_PHASE_MODEL      = True,
        AZ_PICK_HINT_MIN_HISTORY           = 3,
        INDOOR_PHASE_SCORE_WEIGHT          = 0.45,
        INDOOR_AZ_HINT_SCORE_WEIGHT        = 0.55,
        INDOOR_EL_HINT_SCORE_WEIGHT        = 0.20,
        INDOOR_UCA_MIRROR_MARGIN_DEG       = 5.0,
        INDOOR_MIRROR_PHASE_MIN_MARGIN_DEG = 10.0,
        PHASE_SCORE_MIN_SNR_DB             = 2.0,
    ),
    "indoor_pass": dict(
        INDOOR_TX_MODE               = "pass",
        MAX_SATELLITES               = 1,
        MULTI_BURST_N                = 4,
        DOPPLER_GATE_HZ              = 0,
        CFO_TRACK_MAX_JUMP_HZ        = 25_000,
        PAPR_INST_MIN_DB             = 1.0,
        SNR_INST_MIN_DB              = -3.0,
        AZ_SMOOTH_ALPHA              = 0.45,
        EL_SMOOTH_ALPHA              = 0.45,
        COV_ALPHA                    = 0.93,
        ENERGY_DETECT_THRESHOLD      = 3.0,
        INDOOR_EL_MAX_DEG            = 40.0,
        INDOOR_EL_PREF_MAX_DEG       = 35.0,
        INDOOR_EL_PREF_MIN_DEG       = 8.0,
        AZ_OUTLIER_ENABLED           = True,
        AZ_OUTLIER_MAX_DEV_DEG       = 45.0,
        AZ_OUTLIER_MIN_HISTORY       = 5,
        AZ_OUTLIER_RELOCK_STREAK     = 12,
        PHASE_COHERENCE_ENABLED      = False,
        GATE_RELOCK_BYPASS_BURSTS    = 8,
        DOPPLER_XZ_ENABLED           = False,
        IRA_SCAN_OFFSETS_HZ          = [0],
        SAT_MIN_SEP_HZ               = 8_000,
        SAT_TIMEOUT_S                = 10.0,
        INDOOR_SINGLE_BURST_FALLBACK = True,
        PAPR_SINGLE_BURST_MIN_DB     = 0.5,
        INDOOR_IRA_TONE_LOCK_BW_HZ        = 3_000.0,
        INDOOR_IRA_TONE_EMA_ALPHA          = 0.50,
        INDOOR_IRA_STATIC_PHASE_MODEL      = False,
        AZ_PICK_HINT_MIN_HISTORY           = 4,
        INDOOR_PHASE_SCORE_WEIGHT          = 0.35,
        INDOOR_AZ_HINT_SCORE_WEIGHT        = 0.25,
        INDOOR_EL_HINT_SCORE_WEIGHT        = 0.15,
        INDOOR_UCA_MIRROR_MARGIN_DEG       = 10.0,
        INDOOR_MIRROR_PHASE_MIN_MARGIN_DEG = 15.0,
        PHASE_SCORE_MIN_SNR_DB             = 2.0,
    ),
    "outdoor": dict(
        INDOOR_TX_MODE               = "pass",   # "pass" signals no Doppler gate; real satellite
        MAX_SATELLITES               = 3,
        MULTI_BURST_N                = 2,        # outdoor signals weak → reduce stack to avoid corruption
        DOPPLER_GATE_HZ              = 0,
        CFO_TRACK_MAX_JUMP_HZ        = 2_500,
        PAPR_INST_MIN_DB             = 2.5,      # lowered: outdoor weak signals have lower PAPR
        SNR_INST_MIN_DB              = -5.0,
        AZ_SMOOTH_ALPHA              = 0.45,
        EL_SMOOTH_ALPHA              = 0.45,
        COV_ALPHA                    = 0.93,
        ENERGY_DETECT_THRESHOLD      = 2.0,   # outdoor signals ~-31dB need low threshold
        INDOOR_EL_MAX_DEG            = 90.0,
        INDOOR_EL_PREF_MAX_DEG       = 85.0,
        INDOOR_EL_PREF_MIN_DEG       = 5.0,
        AZ_OUTLIER_ENABLED           = True,
        AZ_OUTLIER_MAX_DEV_DEG       = 55.0,
        AZ_OUTLIER_MIN_HISTORY       = 5,
        AZ_OUTLIER_RELOCK_STREAK     = 12,
        PHASE_COHERENCE_ENABLED      = False,
        GATE_RELOCK_BYPASS_BURSTS    = 8,
        DOPPLER_XZ_ENABLED           = True,
        DOPPLER_XZ_MIN_ABS_HZ        = 2000.0,
        DOPPLER_XZ_MIN_JUMP_HZ       = 4000.0,
        DOPPLER_XZ_DEADTIME_S        = 30.0,
        # Empirical IRA channel offsets (outdoor session 2026-05-27 / 2026-06-02):
        IRA_SCAN_OFFSETS_HZ          = [
            -210_000, -148_000, -107_000, -65_000, -23_000, +19_000,
            +60_000, +102_000, +143_000, +185_000, +227_000, +268_000,
            +310_000, +352_000,
        ],
        SAT_MIN_SEP_HZ               = 8_000,
        SAT_TIMEOUT_S                = 12.0,
        INDOOR_SINGLE_BURST_FALLBACK = False,
        PAPR_SINGLE_BURST_MIN_DB     = 2.0,
        INDOOR_IRA_TONE_LOCK_BW_HZ        = 10_000.0,
        INDOOR_IRA_TONE_EMA_ALPHA          = 0.50,
        INDOOR_IRA_STATIC_PHASE_MODEL      = False,
        AZ_PICK_HINT_MIN_HISTORY           = 4,
        INDOOR_PHASE_SCORE_WEIGHT          = 0.0,
        INDOOR_AZ_HINT_SCORE_WEIGHT        = 0.25,
        INDOOR_EL_HINT_SCORE_WEIGHT        = 0.10,
        INDOOR_UCA_MIRROR_MARGIN_DEG       = 15.0,
        INDOOR_MIRROR_PHASE_MIN_MARGIN_DEG = 20.0,
        PHASE_SCORE_MIN_SNR_DB             = 5.0,
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Optional overrides — uncomment to override scenario defaults for fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════

# MULTI_BURST_N          = 8      # bursts to average for DoA (scenario: ira=32, pass/outdoor=4)
# PAPR_INST_MIN_DB       = 3.0    # minimum MUSIC PAPR gate [dB]
# SNR_INST_MIN_DB        = -3.0   # minimum per-element SNR gate [dB]
# DOPPLER_GATE_HZ        = 3000   # Doppler acceptance window [Hz]  (0 = disabled)
# MAX_SATELLITES         = 1      # maximum tracked satellites  (scenario: indoor=1, outdoor=3)
# AZ_OUTLIER_MAX_DEV_DEG = 22.0   # azimuth outlier rejection threshold [°]

