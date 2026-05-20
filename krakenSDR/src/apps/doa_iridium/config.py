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

ANT0_OFFSET_DEG = -333.0
# Rotation of antenna 0 from geographic North [°].
# Calibrate: place TX due North → read median az → set ANT0_OFFSET_DEG = -median_az.

# ── RF / Frequency ────────────────────────────────────────────────────────────
# REAL IRIDIUM (1626.270 MHz — requires licence, wired test, or outdoor sky RX):
FREQ_HZ = 1_626_270_000
#
# INDOOR LAB (ISM 868 MHz — LibreSDR TX, no licence required):
# FREQ_HZ = 868_100_000

GAIN_DB = 49
# IF gain [dB] applied to all KrakenSDR channels.  RTL-SDR max ≈ 49.6 dB.
#
# Indoor 1626 MHz (LibreSDR TX −60 dB at ~1 m, satellite-equivalent SNR): 49 dB (max)
# Indoor 1626 MHz (LibreSDR TX −40 dB at ~1 m, high-SNR diagnosis):       35–40 dB
# Outdoor 1626 MHz (real Iridium from sky, RHCP patch):                    30–40 dB
#   Iridium signal at ground level: −70…−50 dBm; KrakenSDR NF ≈ 5–7 dB.
#   Start at 30 dB; increase to 40 if bursts are missed; reduce if ADC clips.

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
#
# 'none' — INDOOR default con sample covariance.
#   Con il nuovo pipeline R_sample = X_big @ X_big^H / N (covarianza campionaria
#   da MULTI_BURST_N finestre di preambolo):
#   - R_sample è già full-rank → nessuna decorrelazione aggiuntiva necessaria
#   - Le fasi del multipath indoor variano burst a burst (randomizzate dall'ambiente)
#     → burst accumulati incoherenti → R_sample risolve automaticamente diretto+riflesso
#   - Test simulato: PAPR=38.8 dB con 'none', PAPR=35.3 dB con 'fb' → 'none' migliore
#
# 'fb' — usare solo se si sospetta multipath fortemente coerente (riflessione speculare
#   con ritardo fisso e fase stabile tra burst, es. metallo piano vicino).
#
# 'circulant' — NON usare per UCA: forza simmetria N-fold (spettro a stella).

NUM_SIGNALS = 0
# Sorgenti attese PER SATELLITE per l'algoritmo MUSIC (split sottospazio S/N).
# 0 = auto-MDL (Wax & Kailath 1985): stima automatica dal rapporto autovalori.
#   Consigliato per indoor dove il numero di cammini (diretto + riflessi) varia.
#   MDL rileva 1 sorgente in LOS pulita, 2 con un riflesso forte, ecc.
# 1 = forza K=1 (un solo cammino). Usare solo se l'ambiente è molto pulito.

# ── Multi-satellite detection ─────────────────────────────────────────────────
MAX_SATELLITES = 1
# Numero massimo di satelliti tracciati contemporaneamente.
# In vista di Iridium (1626 MHz) ci sono tipicamente 1–3 satelliti sopra l'orizzonte.
# Riduci a 1 per test indoor con un solo TX LibreSDR.  ← ATTUALE: indoor single-TX.

MULTI_BURST_N = 60
# Numero di burst preamble da accumulare per ogni stima DoA.
#
# NUOVO PIPELINE: per ogni burst viene salvata X_pre (finestra IQ del preambolo).
# Dopo MULTI_BURST_N burst, la covarianza campionaria viene calcolata come:
#   R_sample = X_big @ X_big^H / (N * n_pre)   con X_big = hstack(X_pre_i)
# Questo è il modo matematicamente corretto per gestire il multipath indoor:
#   - Con N=20 burst × 2552 campioni = 51040 snapshot IQ
#   - R_sample è full-rank → MUSIC funziona correttamente
#   - PAPR >> 20 dB atteso anche con multipath moderato
#
# INDOOR con satellite visibile (default): 20
#   Il satellite Iridium riduce la hit-rate del TX a ~14 burst/10s (fd-gate
#   blocca satellite ma la scansione multi-peak deve trovare il TX fra picchi
#   più forti). Con N=20: prima stima in ~14s.
#
# INDOOR senza satellite (TX solo): 55
#   Tutti i burst windows riconoscono il TX → hit-rate ~11/s → prima stima in 5s
#
# OUTDOOR satellite rapido (velocità > 1°/s): ridurre a 5-10

DOPPLER_SCAN_BW_HZ = 45_000
# Semi-ampiezza della scansione FFT intorno al tono nominale (fc + 3125 Hz) [Hz].
# Copre ±40 kHz (Doppler massimo Iridium LEO a 1626 MHz).
# A 868 MHz il Doppler max è ±21 kHz: la finestra ±45 kHz è comunque sicura.

DOPPLER_GATE_HZ = 5_000
# Gate sul CFO (Doppler - 3125 Hz) dopo la scansione FFT [Hz].
# I picchi con |cfo_hz| > DOPPLER_GATE_HZ vengono scartati prima del BPF.
#
# INDOOR TX LibreSDR (default): 5000 Hz
#   Il TX è a fd≈0 Hz (cfo≈0). I satelliti Iridium hanno cfo≈−24kHz che
#   vengono bloccati → il tracker segue solo il TX statico → fasi stabili.
#   Copre l'errore TCXO tipico dell'AD9363 (±1-2 kHz).
#
# OUTDOOR satellite: 0  (gate disabilitato, accetta tutti i Doppler)
#   Iridium LEO: cfo ≈ ±40 kHz → serve finestra ampia.
#
# Override a runtime: --fd-max <hz>

CFO_TRACK_MAX_JUMP_HZ = 2000
# Max salto CFO consentito [Hz] tra due burst accettati dello stesso tracker.
# Riduce i "salti" verso picchi FFT spurii dentro la finestra Doppler.
# 0 = disabilitato (default robusto: massima probabilita' di lock iniziale).
# INDOOR TX statico: 1500 Hz — con Doppler scan su preamble-only il CFO e'
#   stabile entro ±200 Hz (TCXO drift); salti > 1.5 kHz sono casi spurii.
# OUTDOOR satellite dinamico:  0 (disabilitato) o >=6000 Hz

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

AZ_FREEZE_EL_DEG = 75.0
# Above this elevation [°] the projected UCA aperture shrinks by cos(el) ≤ 0.26,
# making azimuth estimation unreliable (MUSIC spectrum becomes nearly flat).
# The azimuth EMA phasor is frozen above this threshold and resumed below it.

# ── Covariance EMA accumulation ──────────────────────────────────────────────
COV_ALPHA = 0.85
# Burst-level EMA weight.
# τ = 1/(1-α) burst time constant.
#
# INDOOR LibreSDR TX (stationary, current mode):  0.85–0.90
#   τ=6.7 bursts ≈ 600 ms → suppresses burst-to-burst multipath fluctuations.
#   The instantaneous el→az coupling seen when moving the TX is multipath-induced
#   (confirmed lag-0 cross-correlation): higher α averages it out.
# OUTDOOR satellite pass:  0.50–0.70
#   Satellite moves ≈ 0.5–1°/s; az changes < 1° in 600 ms → 0.85 still OK.
#   Use 0.50 for very fast passes near zenith (el > 60°, az rate > 2°/s).
# Set to 0.0 for fresh covariance every burst (max noise, no memory).
# MULTI_BURST_N=3 partly compensates low α by averaging 8 snapshots.

AZ_SMOOTH_ALPHA = 0.70
# Circular EMA on per-burst azimuth angle estimate.
# 0.70 → τ = 3.3 bursts ≈ 300 ms.
# INDOOR stationary TX: 0.70 suppresses el→az coupling artefacts from multipath.
# OUTDOOR satellite (fast pass): lower to 0.50 to track az rate ≈ 1–2°/s.

EL_SMOOTH_ALPHA = 0.70
# Linear EMA on per-burst elevation estimate.
# 0.70 → τ = 3.3 bursts ≈ 300 ms.
# Indoor TX (stationary): 0.70 smooths noisy el estimates (multipath-induced scatter).
# Outdoor satellite: el changes at ≈ 0.5–2°/s → 0.70 (6.6° lag max) is acceptable.

# ── Hardware phase calibration ────────────────────────────────────────────────
CHANNEL_PHASE_OFFSETS_DEG = [0.0, 24.71, -86.64, 8.72, -3.4]  # auto-cal 2026-05-19 11:33 from 1389 bursts (118 snapshots) az=0.0°
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

SNR_INST_MIN_DB = -1.0
# Minimum SINR [dB] from the eigenvalue ratio of the per-burst sample covariance.
# After amplitude normalization each channel has unit variance → the absolute
# eigenvalues are all ≈ 1 regardless of signal strength.  The signal manifests
# as λ₁ > mean(λ₂…λ_M), and the SINR = (λ₁ − σ²_n) / σ²_n is scale-invariant:
# pure noise → SINR ≈ 0 dB;  strong coherent source → SINR > 10 dB.
# 2 dB: rejects pure noise bursts, accepts low-SNR indoor signals.
# Raise to 5–8 dB in clean outdoor LOS conditions.
# Override at runtime: --snr-min <value>

PAPR_INST_MIN_DB = 2.0
# Minimum MUSIC PAPR [dB] to accept a DoA result.
#
# Con la covarianza campionaria (R_sample da N burst × n_pre snapshot):
#   - Diretto + 1 riflessione moderata: PAPR ≈ 8-15 dB (risolti da MUSIC)
#   - Solo diretto (LOS puro):          PAPR ≈ 20-40 dB
#   - Rumore puro:                      PAPR ≈ 0-1 dB
# 4 dB: rifiuta falsi allarmi (rumore), accetta segnali reali indoor.
#
# INDOOR (default): 4.0
# OUTDOOR cielo aperto: 8.0-12.0

# ── Pilot tone / preamble ─────────────────────────────────────────────────────
PILOT_TONE_ENABLED   = False    # False = IRA burst mode (preamble tone auto-detected)
PILOT_TONE_OFFSET_HZ = 3_125   # Iridium IRA preamble tone offset (Rs/8)
PREAMBLE_BPF_BW_HZ   = 8_000  # BPF bandwidth around preamble tone [Hz]
#                                # 8 kHz: narrower filter gives more SNR gain (≈21 dB vs full band)
#                                # and rejects DQPSK data energy.  With Doppler scan on preamble
#                                # only, detected CFO is accurate to ~250 Hz → 8 kHz is safe.
#                                # Wider (15 kHz) was needed with full-window Doppler scan;
#                                # with preamble-only scan, 8 kHz is sufficient and more selective.
SAMPLE_RATE_HZ       = 1_024_000   # KrakenSDR / Heimdall DAQ rate [Hz]

# ── Gate parameters ───────────────────────────────────────────────────────────
AZ_OUTLIER_ENABLED      = True
AZ_OUTLIER_MAX_DEV_DEG  = 45.0   # reject burst if az differs > 45° from recent median
AZ_OUTLIER_MIN_HISTORY  = 5      # minimum bursts before outlier gate activates

PHASE_COHERENCE_ENABLED     = True
PHASE_COHERENCE_MAX_JUMP_DEG = 180.0
# Phase coherence gate: reject burst if inter-channel phase diffs jump > threshold
# from the previous accepted burst. ONLY active on calibrated hardware (_has_cal True).
# Without calibration, phase diffs are dominated by noise at SNR 5-15 dB (indoor) and
# would reject most bursts.
# INDOOR multipath: multipath changes the effective steering direction burst-to-burst
#   → phase diffs can jump 90-150° between accepted bursts even for a real signal.
#   180° = effectively disabled (max possible circular jump). Safe fallback.
# OUTDOOR clear sky: tighten to 30-60° once system is stable.

# ── Multi-burst accumulation ──────────────────────────────────────────────────
# (see MULTI_BURST_N definition above — set to 1 for indoor multipath immunity)

# ── Display ───────────────────────────────────────────────────────────────────
HISTORY_LEN        = 100         # samples kept in sliding history plots
UPDATE_INTERVAL_MS = 200         # matplotlib animation refresh [ms]
