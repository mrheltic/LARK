# =============================================================================
#  apps/doa_iridium — 2D DoA on real Iridium IRA bursts (1626.270 MHz)
#
#  Setup hardware
#  --------------
#  TX  : LibreSDR (AD9363) running tx_iridium_libresdr.py
#        — uses ISM 868 MHz band for indoor lab tests (no licence required)
#        — switch to 1626.270 MHz ONLY when wired (coax + 30 dB attenuator)
#          or with a valid licence for the 1616-1626.5 MHz Iridium band.
#  RX  : KrakenSDR 5-channel, RHCP patch antenna array (UCA layout).
#        Heimdall DAQ active on TCP:5000.
#
#  Indoor testing (ISM 868 MHz)
#  ----------------------------
#    TX  → LibreSDR  @ 868.1 MHz, gain -20 dB, IRA burst mode
#    RX  → KrakenSDR @ 868.1 MHz, gain 30 dB
#    Set FREQ_HZ = 868_100_000,  GAIN_DB = 30
#
#  Outdoor / real satellite testing
#  ---------------------------------
#    When testing with the real Iridium network:
#    - Set FREQ_HZ = 1_626_270_000
#    - Set GAIN_DB = 25-35 (Iridium ground level: -70…-50 dBm)
#    - No TX: the sky transmits
#    - The RHCP patch is matched to Iridium polarisation → +3 dB vs LHCP
#
#  TDMA burst parameters (Iridium IRA)
#  ------------------------------------
#    Symbol rate   : 25 000 sps
#    Preamble syms : 64   → pure tone at fc + Rs/8 = fc + 3125 Hz  ← keystone for DoA
#    Burst syms    : 245  (preamble + UW + data + tail)
#    Superframe    : 90 ms (one IRA slot per TDMA superframe slot)
#    Modulation    : π/4-DQPSK, RRC β=0.4, gr-iridium compatible
#
#  2D DoA algorithm
#  -----------------
#  MUSIC + 2D search over azimuth [AZ 0°-360°] and elevation [EL 5°-90°].
#  Elevation range 5°-90° brackets indoor tests (source a few metres away at
#  el ≈ 10°-60°) and outdoor satellite passes (el rises from horizon to 90°).
#
#  Patch antenna note
#  ------------------
#  RHCP patch antennas are directional: they have a main lobe above the horizon.
#  Expected gain pattern: +3 dBi at boresight (zenith), −3 dBi at 30° elevation,
#  pattern rolls off at the horizon.  EL_MIN_DEG = 5° is a safe lower bound.
#
#  Path setup notes
#  ----------------
#  This config.py is imported by doa_iridium_burst.py via:
#      sys.path.insert(0, os.path.dirname(__file__))   # this folder
#      sys.path.insert(0, <root>/krakenSDR/src)         # hardware layer
#      import config as C
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

# ── Array geometry ────────────────────────────────────────────────────────────
N_ANTENNAS     = 5
GEOMETRY       = "UCA"

RADIUS_LAMBDA  = 0.4253
# UCA radius in wavelengths for a KrakenSDR 5-element array with λ/2 spacing:
#   R = (λ/2) / (2·sin(π/5)) ≈ 0.4253λ
#
# For 1626.270 MHz (λ = 18.43 cm):
#   Physical radius = 0.4253 × 18.43 cm ≈ 7.84 cm ≈ 78 mm
#
# For 868 MHz (λ = 34.56 cm):
#   Physical radius = 0.4253 × 34.56 cm ≈ 14.7 cm ≈ 147 mm
#
# If your patch-antenna UCA uses a different physical radius r [m]:
#   RADIUS_LAMBDA = r / (speed_of_light / FREQ_HZ)
#   e.g. 10 cm @ 1626 MHz: RADIUS_LAMBDA = 0.10 / 0.1843 ≈ 0.543

ANT_CCW = False
# False = antenne disposte in senso ORARIO (CW) viste dall'alto — standard KrakenSDR UCA.
# Il vettore di sterzatura usa phi_k = 2πk/N (k=0 a Nord, k=1 a 72° orario = ENE, ...).
# Imposta True se le stime az sono specchiate (Est↔Ovest invertiti).

ANT0_OFFSET_DEG = 0.0
# Rotation of antenna 0 from geographic North [°].
# Calibrate: place TX due North → read median az → set ANT0_OFFSET_DEG = -median_az.

# ── RF / Frequency ────────────────────────────────────────────────────────────
# REAL IRIDIUM (1626.270 MHz — requires licence, wired test, or outdoor sky RX):
FREQ_HZ = 1_626_270_000
#
# INDOOR LAB (ISM 868 MHz — LibreSDR TX, no licence required):
# FREQ_HZ = 868_100_000

GAIN_DB = 30
# IF gain [dB] applied to all KrakenSDR channels.
#
# Indoor 868 MHz test (LibreSDR TX at 0.5–2 m):     30–40 dB
# Outdoor 1626 MHz (real Iridium from sky, patch):  30–40 dB
#   Iridium signal at ground level: -70…-50 dBm; KrakenSDR noise figure ≈ 5-7 dB.
#   Start at 30 dB; increase to 35 if bursts are missed; reduce if ADC clips.

# ── 2D DoA algorithm ─────────────────────────────────────────────────────────
DOA_ALGORITHM  = "MUSIC"
# "MUSIC"          — INDOOR default: best PAPR post-BPF with forward-backward decorr.
# "CAPON"          — MVDR: good at lower SNR, less sensitive to array errors.
# "ROOT-MUSIC"     — polynomial super-resolution, excellent for single source.
# "MFBA-MUSIC"     — Modified Forward-Backward Averaging MUSIC: best for Iridium
#                    (natural RHCP polarisation + LEO geometry → strong single source).
# "UNITARY-ESPRIT" — real-valued, fast, excellent for high-elevation short passes.

MUSIC_DECORR   = "none"
# 'none' / 'fb' / 'circulant'.
# 'none' is REQUIRED for burst-gated DoA with matched-filter covariance.
#   The MF covariance R_mf = y_mf * y_mf^H is rank-1.  Applying FB
#   averaging (R_fb = (R + J R^* J) / 2) inflates the apparent rank to 2.
#   With n_sig=1, the FB-image eigenvector leaks into the MUSIC noise
#   subspace, preventing a spatial null at the source direction and
#   collapsing PAPR to 2-4 dB (effectively rejecting 98% of valid bursts).
#   decorr='none' preserves the rank-1 structure → PAPR >> 20 dB.
# 'fb' is appropriate only for continuous non-gated IQ with sample
#   covariance (full-rank R from many snapshots) — NOT for burst mode.

NUM_SIGNALS = 1
# Sorgenti attese PER SATELLITE per l'algoritmo MUSIC (split sottospazio S/N).
# Lascia a 1: ogni satellite IRA è una singola sorgente coerente.  Con MULTI_BURST_N≥3
# e decorr='fb' l'algoritmo è robusto anche con piccolo multipath residuo.

# ── Multi-satellite detection ─────────────────────────────────────────────────
MAX_SATELLITES = 3
# Numero massimo di satelliti tracciati contemporaneamente.
# In vista di Iridium (1626 MHz) ci sono tipicamente 1–3 satelliti sopra l'orizzonte.
# Riduci a 1 per test indoor con un solo TX LibreSDR.

DOPPLER_SCAN_BW_HZ = 45_000
# Semi-ampiezza della scansione FFT intorno al tono nominale (fc + 3125 Hz) [Hz].
# Copre ±40 kHz (Doppler massimo Iridium LEO a 1626 MHz).
# A 868 MHz il Doppler max è ±21 kHz: la finestra ±45 kHz è comunque sicura.

SAT_MIN_SEP_HZ = 5_000
# Separazione minima Doppler [Hz] per considerare due picchi FFT come satelliti distinti.
# < 5 kHz → stesso satellite (o ghost); ≥ 5 kHz → satellite separato.

SAT_TIMEOUT_S = 8.0
# Secondi senza burst accettati prima di eliminare un tracker satllite.
# Iridium 90 ms burst rate → 1 burst/90 ms → SAT_TIMEOUT_S = 8 = ~88 burst miss streak.

SAT_COLORS = ["#f4a431", "#4ecdc4", "#a78bfa"]
# Colori [amber, teal, viola] per i 3 possibili satelliti nel grafico polar.

# ── 2D scan grid ─────────────────────────────────────────────────────────────
N_AZ       = 360          # azimuth points (1° step)
N_EL       = 86           # elevation points:  [5°, 90°] with 1° step → 86 points
EL_MIN_DEG = 5.0
# 5° lower bound: patches are directional — signal below horizon is spurious.
# In outdoor satellite mode the satellite is always > 5° when above the horizon gate.
EL_MAX_DEG = 90.0
# 90° upper bound: satellite can pass through zenith (elevation = 90°).
# For indoor tests where TX is e.g. on a bench at 20–60° elevation, this is fine.

# ── Covariance EMA accumulation ──────────────────────────────────────────────
COV_ALPHA = 0.50
# Burst-level EMA weight.
# 0.50 → τ = 2 valid-burst time constant (≈180 ms at 1 burst/90 ms).
#
# INDOOR LibreSDR TX (stationary):  0.70–0.90
# OUTDOOR satellite (fast Doppler): 0.30–0.50  — satellite moves ≈ 1°/s → fast α.
# Set to 0.0 for fresh covariance every burst (max noise, no memory).
# MULTI_BURST_N = 3 (3-burst average before DoA) partly compensates low α.

AZ_SMOOTH_ALPHA = 0.50
# Circular EMA on per-burst azimuth angle estimate.
# 0.50 → τ = 2 bursts ≈ 180 ms — suited to satellite motion (1°/s typ.).
# INDOOR stationary: raise to 0.70 for smoother display.

EL_SMOOTH_ALPHA = 0.50
# Linear EMA on per-burst elevation estimate.
# Iridium passes change elevation at ≈ 0.5–2°/s → 0.50 tracks well.

# ── Hardware phase calibration ────────────────────────────────────────────────
CHANNEL_PHASE_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
# Per-channel phase offset [°], channel 0 is the reference (always 0.0).
# Run --calibrate <az_deg> with TX at a known azimuth to compute automatically.
# Example (TX at North = 0°): python3 doa_iridium_burst.py --calibrate 0.0

# ── Squelch ───────────────────────────────────────────────────────────────────
SQUELCH_ENABLED      = True
SQUELCH_THRESHOLD_DB = -60.0
# Lower than 868 MHz indoor because Iridium path is much longer.
# For real Iridium -70…-60 dBW is normal range from sky.
# For LibreSDR indoor: -55 dBW as with the 868 test.

EIG_SPREAD_MIN_DB = 0.5
# Very permissive: let SNR gate do the final selection.
# Raise to 2.5 in clean outdoor LOS conditions.

EIG_SN_GAP_MIN_DB = 0.0
# Disabled post-BPF (same rationale as 868 burst mode — see comments there).

SNR_INST_MIN_DB = 5.0
# Minimum per-element SNR [dB] from the covariance eigenvalue ratio to accept
# a preamble burst as containing a real signal.
# The SNR check is calibration-agnostic (depends only on eigenvalues, not on
# the steering manifold), so it works correctly on uncalibrated hardware where
# MUSIC PAPR would fail due to inter-channel phase offsets.
# 5 dB: rank-1 R_mf from IRA preamble gives SNR >> 20 dB; a noise-only R
# (from energy-detector false alarm) gives SNR < 0 dB.
# Lower threshold (1–3 dB) for very weak outdoor signals; raise to 10+ dB
# to tighten false-alarm rate after hardware calibration.
# Override at runtime: --snr-min <value>

PAPR_INST_MIN_DB = 3.0
# Minimum MUSIC PAPR [dB] to accept a multi-burst averaged DoA result.
# Used only for the final multi-burst stage (instant gate now uses SNR_INST_MIN_DB).
# 3 dB is permissive enough for uncalibrated hardware where inter-channel phase
# offsets shift y_mf off the UCA steering manifold, degrading MUSIC PAPR.
# After running --calibrate, raise back to 8–12 dB for stricter quality control.
# Override at runtime: --papr-min <value>

# ── Pilot tone / preamble ─────────────────────────────────────────────────────
PILOT_TONE_ENABLED   = False    # False = IRA burst mode (preamble tone auto-detected)
PILOT_TONE_OFFSET_HZ = 3_125   # Iridium IRA preamble tone offset (Rs/8)
PREAMBLE_BPF_BW_HZ   = 15_000  # BPF bandwidth around preamble tone [Hz]
#                                # 15 kHz: ~6× symbol rate, accepts ±5 kHz LO offset.
#                                # Narrower (8-10 kHz) gives more SNR, less LO-offset tolerance.
SAMPLE_RATE_HZ       = 1_024_000   # KrakenSDR / Heimdall DAQ rate [Hz]

# ── Gate parameters ───────────────────────────────────────────────────────────
AZ_OUTLIER_ENABLED      = True
AZ_OUTLIER_MAX_DEV_DEG  = 45.0   # reject burst if az differs > 45° from recent median
AZ_OUTLIER_MIN_HISTORY  = 5      # minimum bursts before outlier gate activates

PHASE_COHERENCE_ENABLED     = True
PHASE_COHERENCE_MAX_JUMP_DEG = 120.0
# Phase coherence gate: reject burst if inter-channel phase diffs jump > threshold
# from the previous accepted burst. ONLY active on calibrated hardware (_has_cal True).
# Without calibration, phase diffs are dominated by noise at SNR 5-15 dB (indoor) and
# would reject most bursts.  Once calibrated, tighten to 30-60° for best rejection.

# ── Multi-burst accumulation ──────────────────────────────────────────────────
MULTI_BURST_N = 3
# Average N burst covariance matrices before computing DoA.
# Iridium SNR:  outdoor real = -70…-50 dBm → SNR can be 0-10 dB with GAIN=30.
# Averaging 3 bursts = +4.8 dB SNR gain ≈ 270 ms latency (3 × 90 ms).
# For indoor LibreSDR tests (high SNR): 1 or 2 is sufficient.

# ── Display ───────────────────────────────────────────────────────────────────
HISTORY_LEN        = 100         # samples kept in sliding history plots
UPDATE_INTERVAL_MS = 200         # matplotlib animation refresh [ms]
