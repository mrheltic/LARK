# =============================================================================
#  apps/doa — ISM band direction-finding configuration
#
#  Covers the 868 MHz / 433 MHz ISM scenarios (LoRa trackers, custom beacons)
#  using a 5-element UCA on the KrakenSDR.
#
#  Hardware constants (Heimdall address, N_ANTENNAS, SAMPLE_RATE_HZ, …) are
#  re-exported from config_hw.py so scripts can import a single module:
#
#    import config as C    →   C.FREQ_HZ, C.HEIMDALL_HOST, C.DOA_ALGORITHM …
#
#  Apply a named profile at launch to switch parameter sets without editing
#  this file:
#
#    LARK_PROFILE=ism_868             python3 apps/doa/doa_runner.py
#    LARK_PROFILE=outdoor_5ant_fast   python3 apps/doa/doa_runner.py
#    python3 apps/doa/doa_runner.py --profile outdoor_5ant_precision
#
#  See krakenSDR/src/profiles.py for all available profiles and their values.
# =============================================================================
from __future__ import annotations

import os as _os
import sys as _sys

# ── Hardware base ─────────────────────────────────────────────────────────────
HEIMDALL_HOST  = "127.0.0.1"
HEIMDALL_PORT  = 5000
HEIMDALL_CTRL  = 5001
N_ANTENNAS     = 5
SAMPLE_RATE_HZ = 1_024_000
HW_NUM_SAMPLES = 0

# ── Antenna array geometry ────────────────────────────────────────────────────
GEOMETRY       = "UCA"          # "UCA" = uniform circular array  (recommended, full 360°)
#                                # "ULA" = uniform linear array

RADIUS_LAMBDA  = 0.358          # [UCA] array radius in fractions of λ
#                                #  KrakenSDR 5-ant ring @ 865 MHz: physical radius ~12.5 cm
#                                #  λ @ 865 MHz ≈ 34.7 cm  →  12.5/34.7 ≈ 0.360λ

D_LAMBDA       = 0.5            # [ULA] inter-element spacing in fractions of λ
#                                #  Only used when GEOMETRY = "ULA"

# ── RF / Radio ────────────────────────────────────────────────────────────────
FREQ_HZ        = 865.21e6       # carrier frequency [Hz] — standard LoRaWAN 868 downlink
GAIN_DB        = 15             # IF gain [dB] applied to all channels
#                                #  Valid RTL-SDR gain steps (dB):
#                                #  0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4,
#                                #  15.7, 16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7,
#                                #  32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
#                                #  43.9, 44.5, 48.0, 49.6
#                                #  Typical outdoor at 15 m: 15–25 dB.

# ── DoA algorithm ─────────────────────────────────────────────────────────────
SCAN_POINTS    = 360            # angular scan resolution (points across 0° … 360°)
NUM_SIGNALS    = 1              # number of simultaneous signal sources expected

DOA_ALGORITHM  = "MUSIC"        # "MUSIC"      — subspace pseudospectrum (RECOMMENDED)
#                                # "ROOT-MUSIC" — polynomial roots on VULA, sub-grid accuracy
#                                # "CAPON"      — MVDR beamformer  P = 1 / (a^H R⁻¹ a)
#                                # "ML"         — stochastic maximum-likelihood, sharp peaks
#                                # "ESPRIT"     — rotational invariance on VULA, no angle scan
#                                # NOTE: CAPON/ML are sensitive to TOEP/FBTOEP on a 5-ant UCA

# ── Covariance decorrelation ──────────────────────────────────────────────────
DECORRELATION  = "FBA"          # "Off"    — no decorrelation
#                                # "FBA"    — Forward-Backward Averaging       (RECOMMENDED)
#                                #            operates in VULA space; works well for UCA
#                                # "TOEP"   — Toeplitz rectification
#                                # "FBTOEP" — FBA + Toeplitz
#                                # NOTE: TOEP/FBTOEP require a Toeplitz R_vula.  With 5 antennas
#                                # and r_lambda ≈ 0.358 the VULA is not accurately Toeplitz
#                                # (J₀(2.25) ≈ 0.08, first zero at 2.40).  Use FBA only.

# ── Covariance accumulation (EMA) ─────────────────────────────────────────────
COV_ALPHA      = 0.95           # exponential moving-average weight on consecutive frames
#                                #  τ = 1/(1−α) frames: α=0.80 → ~5 fr, α=0.95 → ~20 fr
#                                #  Fixed source       : 0.90–0.96
#                                #  Moving source      : 0.50–0.70

# ── Angle smoothing ───────────────────────────────────────────────────────────
ANGLE_SMOOTH_ALPHA = 0.80       # circular EMA on the final DoA estimate
#                                #  Operates on complex phasors → correct 0°/360° wrap.
#                                #  Fixed source  : 0.70–0.85
#                                #  Moving source : 0.30–0.50
#                                #  0.0 = disabled (raw per-frame estimate)

# ── Channel amplitude normalisation ──────────────────────────────────────────
AMPLITUDE_NORMALIZE  = True     # normalise each channel to unit power before covariance
#                                #  Cancels hardware gain imbalance between channels.
#                                #  Do NOT use normalised data for HW calibration runs.

# ── Squelch ───────────────────────────────────────────────────────────────────
SQUELCH_ENABLED      = True     # skip DoA processing when signal power is below threshold
SQUELCH_THRESHOLD_DB = -60.0    # absolute power threshold [dBW] on raw un-normalised X
#                                #  Indoor @ 40 dB gain typically reads −30…−10 dBW.
#                                #  Raise if ghost estimates appear when band is idle.

# ── Per-channel phase calibration ─────────────────────────────────────────────
PHASE_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
#                                #  Per-antenna hardware phase offset correction [°].
#                                #  len must equal N_ANTENNAS.
#                                #  Measure: point array at a known far-field source,
#                                #  read off-diagonal phases from the coherence panel,
#                                #  set phi[k] = −arg(R[0,k]) for each k ≠ 0.

# ── Display / animation ───────────────────────────────────────────────────────
INTERVAL_MS    = 80             # milliseconds between Matplotlib animation frames
#                                #  80 ms ≈ 12 fps.  Lower = smoother, higher CPU.
VERBOSE_FRAMES = 0              # print Heimdall diagnostics every N frames (0 = silent)

# =============================================================================
# Profile override (LARK_PROFILE env var or --profile CLI flag)
# =============================================================================
#  The LARK_PROFILE env var is applied at import time so even module-level
#  config reads (e.g. doa_runner.py) see the correct values.
#  The --profile CLI flag in individual scripts calls apply_profile(name, C)
#  AFTER argparse, overriding these module attributes directly.
# Profile override: removed (profiles.py was deleted). Set parameters directly.
