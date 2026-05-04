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
DOA_ALGORITHM  = "BARTLETT"
# "MUSIC"        — subspace super-resolution (raccomandato per CW beacon outdoor).
# "CAPON"        — MVDR: buono a basso SNR ma può bloccarsi sulla direzione media del multipath.
# "BARTLETT"     — beamformer convenzionale: ROBUSTO al multipath indoor, risoluzione inferiore
#                    ma molto più stabile in ambiente indoor con riflessioni.  Consigliato per stanze.
# "ROOT-MUSIC"   — approccio polynomial rooting: alta risoluzione, buono per UCA.
# "UNITARY-ESPRIT"— elaborazione a valori reali: efficiente computazionalmente.
# "MFBA-MUSIC"   — Modified Forward-Backward Averaging: stima covarianza migliorata.
#
# INDOOR (stanza piccola, molto multipath):  BARTLETT  ← più stabile, meno jitter
# OUTDOOR / LOS (linea di vista libera):     MUSIC     ← massima risoluzione

MUSIC_DECORR   = "fb"
# Decorrelation applied to covariance before MUSIC (and Capon).
# 'none'      — nessuna decorrelazione (default sicuro, usa solo EMA)
# 'fb'        — Forward-Backward averaging: migliora MUSIC con multipath coerente
#               SICURO per UCA; riduce l'impatto delle riflessioni dal soffitto.
# 'circulant' — solo per debug offline (forza simmetria ciclica: valida solo per UCA pura).
# 'both'      — circulant + FB: solo per analisi offline.

NUM_SIGNALS    = 1        # sorgenti attese simultanee

# ── 2D scan grid ─────────────────────────────────────────────────────────────
N_AZ       = 180          # azimuth scan points (180 → 2° step)
N_EL       = 36           # elevation scan points (36 → 2.5° step from 0° to 90°)
EL_MIN_DEG = 0.0          # minimum elevation [°] — 0° for ground-level ISM beacons

# ── Covariance EMA accumulation ──────────────────────────────────────────────
COV_ALPHA  = 0.95         # EMA weight  (time constant τ = 1/(1-α) frames)
#                          # 0.95 → ~20 valid-burst frames of memory ≈ 3-4 s
#                          # INDOOR: usare α ALTO (0.95-0.98) per mediare su più
#                          # riflessioni e ottenere direzione stabile.
#                          # OUTDOOR / TX in movimento: usare α BASSO (0.50-0.70).

AZ_SMOOTH_ALPHA = 0.50
# Circular EMA smoothing on the per-burst az angle estimate.
# τ = 1 / (1 − α) valid-preamble bursts.
# 0.50 → τ≈2 burst → più reattivo ora che il BPF preambolo dà fasi più pulite.
# Alzare a 0.70-0.80 se l'azimuth è ancora troppo jitter.

# ── Hardware phase calibration ────────────────────────────────────────────────
CHANNEL_PHASE_OFFSETS_DEG = [0.0, 0.0, 0.0, 0.0, 0.0]
# Per-channel phase offset correction [degrees], relative to channel 0 (reference).
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
# INDOOR: -55 dBW è un buon compromesso per stanze piccole (TX vicino → segnale forte).
# Se vedi troppi "NO SIGNAL" con TX acceso, abbassa a -60 o -65.
# Se vedi falsi segnali (ghost) con TX spento, alza a -50.

EIG_SPREAD_MIN_DB = 0.5
# Minimum eigenvalue spread (λ_max / λ_noise in dB) to declare signal present.
# IMPORTANTE: in ambienti indoor il multipath invalida la separazione classica dei
# sottospazi: tutti gli autovalori appaiono simili.  Soglia molto bassa per non
# perdere burst validi; la separazione viene poi fatta dalla soglia PAPR del MUSIC.
#   INDOOR piccola stanza: 0.5 dB  (lascia passare quasi tutto, filtra il PAPR)
#   OUTDOOR / LOS:         2.5-3.0 dB  (segnale ben separato dal rumore)

PAPR_INST_MIN_DB = 8.0
# Minimum MUSIC PAPR to accept a preamble burst as valid.
# INDOOR (multipath, SNR basso): abbassare a 6-8 dB.
# OUTDOOR / LOS (array calibrato): 12-20 dB.
# Override a runtime con: python3 doa_test_868_burst.py --papr-min 6

# ── Pilot tone extraction ─────────────────────────────────────────────────────
# ⚠️  IMPORTANTE: PILOT_TONE_ENABLED va abilitato SOLO in modalità CW (tono continuo).
# In modalità BURST (IRA) il tono pilota è a +3125 Hz nel preamble e viene gestito
# automaticamente dal codice burst-specifico.  Abilitare PILOT_TONE_ENABLED in burst
# mode fa estrarre una banda a 100 kHz che contiene solo rumore → SNR crolla.
SAMPLE_RATE_HZ        = 1_024_000
PILOT_TONE_ENABLED    = False   # ← False per BURST mode, True solo per CW mode
PILOT_TONE_OFFSET_HZ  = 100_000 # ← Usato solo se PILOT_TONE_ENABLED=True (CW mode)
PILOT_TONE_BW_HZ      = 15_000  # ← Usato solo se PILOT_TONE_ENABLED=True (CW mode)

PREAMBLE_BPF_BW_HZ = 10_000
# Larghezza di banda del filtro BPF in burst mode.
# Con BW=10 kHz: SNR gain ≈ 10·log10(1024000/10000) ≈ +20 dB; finestra più ampia
# gestisce eventuali imprecisioni nella rilevazione della frequenza del tono
# (risoluzione FFT 2000 Hz su finestra 512 camp. → incertezza ±1000 Hz).
# Non scendere sotto 4000 Hz.

TONE_SEARCH_BW_HZ = 100_000
# Finestra di ricerca della frequenza reale del tono IRA preamble [Hz].
# TX (LibreSDR) e RX (KrakenSDR) usano oscillatori indipendenti → offset LO
# tipico ±10–50 kHz a 868 MHz.  Il codice cerca il picco FFT in
# [+3125 - TONE_SEARCH_BW_HZ/2 … +3125 + TONE_SEARCH_BW_HZ/2] e usa quella
# frequenza per il BPF.  100 kHz copre ±50 kHz offset (>50 ppm a 868 MHz).
# Se il TX è calibrato o usa stesso clock: abbassare a 20_000 per più precisione.

# ── Per-channel amplitude normalisation ──────────────────────────────────────
AMPLITUDE_NORMALIZE = True
# Normalise each KrakenSDR channel to unit RMS power before covariance.
# Cancels between-channel gain imbalance (up to ~5 dB measured on hardware).
# Does not affect phase relationships (DoA accuracy is preserved).
# Disable for hardware calibration sessions.

# ── Enhanced preprocessing options ────────────────────────────────────────────
# ⚠️ ATTENZIONE: il preprocessing avanzato può CORROMPERE il segnale se non
# configurato correttamente.  Per indoor, è generalmente meglio lasciarlo
# disabilitato e affidarsi all'EMA temporale (COV_ALPHA) per il multipath.
ENABLE_ENHANCED_PREPROCESSING = False
# Abilita tecniche avanzate di preprocessing basate sui paper scientifici:
# - Spatial smoothing per decorrelare segnali coerenti (riflessioni indoor)
# - Modified Forward-Backward Averaging (MFBA) per UCA
# - Adaptive filtering per sopprimere interferenze
# - Outlier rejection per statistica robusta

APPLY_SPATIAL_SMOOTHING = False
# Applica spatial smoothing per decorrelare segnali coerenti.
# INDOOR: può essere utile ma richiede tuning.  Inizia con False.

APPLY_MFBA = False
# Applica Modified Forward-Backward Averaging per migliorare la stima della covarianza.

APPLY_ADAPTIVE_FILTERING = False
# ⚠️ PERICOLOSO: il filtraggio adattivo sottrae la media degli altri canali,
# annullando completamente un segnale coerente su tutti i canali (come il nostro
# burst IRA).  Lascia sempre False per questa applicazione.

APPLY_OUTLIER_REJECTION = False
# Applica reiezione statistica di outlier.  Utile solo per rumore impulsivo forte.

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
PHASE_COHERENCE_ENABLED = True
PHASE_COHERENCE_MAX_JUMP_DEG = 60.0
# Reject a frame if ANY inter-channel phase difference jumps by more
# than this from the running circular median.  Indoor multipath causes
# sudden phase flips on individual bursts → this gates them out.
# 60° works well for stationary TX; widen to 90° for moving TX.

# ── Circular azimuth outlier rejection ───────────────────────────────────────
AZ_OUTLIER_ENABLED = True
AZ_OUTLIER_MAX_DEV_DEG = 45.0
# Reject an az estimate that deviates more than this from the
# running circular median.  This catches the wild 200° jumps
# observed in indoor multipath (az_std ≈ 80° without gating).
# 45° is conservative; tighten to 30° once array is calibrated.
AZ_OUTLIER_MIN_HISTORY = 5
# Minimum number of accepted estimates before outlier rejection kicks in.

# ── Multi-burst / multi-frame accumulation ──────────────────────────────────
MULTI_BURST_N = 3
# Number of valid bursts to accumulate before running DoA.
# Ora che il BPF preambolo è attivo (+22 dB SNR), ogni singolo burst è molto
# più stabile: 3 burst bilanciano stabilità (4.8 dB extra) e latenza (~270 ms).
# OUTDOOR: 1-2 per reattività. INDOOR pesante multipath: 4-5.
# Set to 1 to disable (per-burst DoA as before).
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

