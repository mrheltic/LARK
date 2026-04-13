# =============================================================================
#  MOVED — This file's content has been reorganised into apps/doa/config.py
#
#  DoA algorithm, array geometry, squelch, and display parameters now live in:
#    krakenSDR/src/apps/doa/config.py
#
#  This stub is kept only so any stale import of "config_doa" produces a
#  clear error rather than an obscure AttributeError elsewhere.
# =============================================================================
raise ImportError(
    "config_doa.py has been moved.  "
    "Edit krakenSDR/src/apps/doa/config.py for DoA parameters."
)


# ── DoA algorithm ─────────────────────────────────────────────────────────────
SCAN_POINTS    = 360            # angular scan resolution (points across −180° … +180°)
NUM_SIGNALS    = 1              # number of simultaneous signal sources expected

DOA_ALGORITHM  = "MUSIC"        # "MUSIC"      — subspace pseudospectrum (RECOMMENDED)
#                                #               robust to decorrelation method and N
#                                # "ROOT-MUSIC" — polynomial roots on VULA, sub-grid precision
#                                # "CAPON"      — MVDR beamformer  P = 1 / (a^H R⁻¹ a)
#                                # "ML"         — stochastic maximum-likelihood, sharp peaks
#                                # "ESPRIT"     — rotational invariance on VULA, no angle scan
#                                # NOTE: CAPON / ML are sensitive to TOEP/FBTOEP on 5-ant UCA

# ── Covariance decorrelation ──────────────────────────────────────────────────
DECORRELATION  = "FBA"          # "Off"    — no decorrelation
#                                # "FBA"    — Forward-Backward Averaging       (RECOMMENDED)
#                                #            works correctly in the VULA space for UCA
#                                # "TOEP"   — Toeplitz rectification
#                                # "FBTOEP" — FBA + Toeplitz
#                                # NOTE: TOEP/FBTOEP require a Toeplitz R_vula.  With 5 antennas
#                                # and r_lambda=0.358 the VULA is not accurately Toeplitz
#                                # (J₀(2.25)≈0.08, first zero at 2.40).  Use FBA only.

# ── Covariance accumulation (EMA) ─────────────────────────────────────────────
COV_ALPHA      = 0.95           # exponential moving average weight on consecutive frames
#                                #  α = 0.80 → time constant ~5 frames (~0.6 s @ 9 fps)
#                                #  Fixed source       : raise to 0.90–0.95
#                                #  Moving source      : lower to 0.50–0.70
#                                #  Outdoor (less multipath) : 0.80 works well

# ── Angle smoothing ───────────────────────────────────────────────────────────
ANGLE_SMOOTH_ALPHA = 0.80       # circular EMA weight applied to the final DoA estimate
#                                #  Operates on complex phasors → correct 0°/360° wrap.
#                                #  Fixed source       : 0.70–0.85
#                                #  Moving source      : 0.30–0.50
#                                #  0.0 = disabled (raw per-frame estimate)

# ── Channel amplitude normalisation ──────────────────────────────────────────
AMPLITUDE_NORMALIZE  = True     # normalise each channel to unit power before covariance
#                                #  Cancels hardware gain imbalance between channels.
#                                #  (4.6 dB measured on indoor_001 recording)
#                                #  NOTE: do NOT use normalised samples for HW calibration;
#                                #  use only for DoA estimation.

# ── Squelch ───────────────────────────────────────────────────────────────────
SQUELCH_ENABLED      = True     # skip DoA processing when signal is below threshold
SQUELCH_THRESHOLD_DB = -60.0    # absolute power threshold [dBW] on raw un-normalised X
#                                #  −60 dBW: conservative floor.
#                                #  Indoor 865 MHz at 40 dB gain typically reads −30…−10 dBW.
#                                #  Raise if ghost estimates appear when band is idle.

# ── Per-channel phase calibration ─────────────────────────────────────────────
PHASE_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
#                                #  Per-antenna hardware phase offset correction [degrees].
#                                #  Length must equal N_ANTENNAS (5).
#                                #  How to measure: point array at a known far-field source,
#                                #  read off-diagonal phases from the coherence panel H, then
#                                #  set phi[k] = −arg(R[0, k]) for each k ≠ 0.

# ── Display / animation ───────────────────────────────────────────────────────
INTERVAL_MS    = 80             # milliseconds between Matplotlib animation frames
#                                #  80 ms ≈ 12 fps.  Lower = smoother but heavier CPU.
VERBOSE_FRAMES = 0              # print Heimdall diagnostics every N frames (0 = silent)
