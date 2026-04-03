# =============================================================================
#  KrakenSDR DoA – Central configuration
#  Edit THIS FILE to change all system parameters.
#  Hardware-only mode: no simulation fallback.
# =============================================================================

# ── Heimdall connection ───────────────────────────────────────────────────────
#  Heimdall runs inside the Docker container with network_mode: host,
#  so ports 5000/5001 are directly reachable at 127.0.0.1.
HEIMDALL_HOST  = "127.0.0.1"   # Heimdall server IP
HEIMDALL_PORT  = 5000           # IQ data port
HEIMDALL_CTRL  = 5001           # control port

# ── Antenna array ─────────────────────────────────────────────────────────────
N_ANTENNAS     = 3              # number of KrakenSDR antennas

GEOMETRY       = "UCA"          # "UCA" = circular (recommended, full 360° coverage)
#                                # "ULA" = linear

RADIUS_LAMBDA  = 0.289          # [UCA] array radius in fractions of lambda
#                                #  KrakenSDR 3-ant @ 868 MHz → radius ~9.98 cm → 0.289λ
#                                #  KrakenSDR 5-ant @ 868 MHz → radius ~12.35 cm → 0.358λ

D_LAMBDA       = 0.5            # [ULA] inter-element spacing in fractions of lambda
#                                #  (used only when GEOMETRY = "ULA")

# ── RF ────────────────────────────────────────────────────────────────────────
FREQ_HZ        = 865.21e6          # carrier frequency [Hz]
#                                #  Examples: 433e6 (433 MHz), 868e6 (868 MHz),
#                                #            915e6 (915 MHz), 2.4e9 (Wi-Fi 2.4 GHz)

SAMPLE_RATE_HZ = 1.024e6        # ADC sample rate [Hz] — must match daq_chain_config.ini
#                                #  docker/config/daq_chain_config.ini: sample_rate = 1024000

GAIN_DB        = 15             # IF gain [dB]
#                                #  Can be a single value (same for all channels)
#                                #  or a list per channel: [30.0, 30.0, 30.0]
#                                #  Valid RTL-SDR gain values:
#                                #  0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4,
#                                #  15.7, 16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7,
#                                #  32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
#                                #  43.9, 44.5, 48.0, 49.6

# ── MUSIC algorithm ───────────────────────────────────────────────────────────
SCAN_POINTS    = 360            # angular scan resolution (points from -180° to +180°)
NUM_SIGNALS    = 1              # number of simultaneous expected sources

# ── Samples per frame ─────────────────────────────────────────────────────────
HW_NUM_SAMPLES  = 0             # IQ samples used per frame for MUSIC
#                                #  0 = use all samples sent by Heimdall
#                                #  N > 0 = use only the first N samples
#                                #  Heimdall typically sends 512–2048 samples
#                                #  Recommended values: 512, 1024, 2048

# ── Display ───────────────────────────────────────────────────────────────────
INTERVAL_MS    = 80             # ms between animation frames (80 ms ≈ 12 fps)
VERBOSE_FRAMES = 0              # print Heimdall diagnostics every N frames (0 = silent)

# ── DoA improvements (from krakenrf/krakensdr_doa and EttusResearch/gr-doa) ───
#
#  These parameters optimize estimation for indoor circular arrays with
#  coherent signals (multipath, reflections).
#
#  ── Indoor analysis results (2026-03-24) ─────────────────────────────
#  SNR 22 dB, cond(R) 196  → strong signal, no power issues
#  angle σ = 28.3°         → instability caused by multipath, not noise
#  phase off-diag σ = 1.6r → severe multipath changes R structure each frame
#  imbalance = 4.6 dB      → unbalanced channel → AMPLITUDE_NORMALIZE enabled
#  ──────────────────────────────────────────────────────────────────────
#
#  ── Outdoor 5-antenna setup (current config) ─────────────────────────
#  Meno multipath → cond(R) atteso < 20, σ angolare < 5°
#  ROOT-MUSIC + FBTOEP: massima precisione sub-griglia con N=5
#  COV_ALPHA ridotto a 0.80 per risposta più rapida
#  GAIN_DB: controllare pannello F (PAPR) — se > 15 dB abbassare il gain
#  Calibrare PHASE_OFFSETS_DEG sul campo prima delle misure
#  ──────────────────────────────────────────────────────────────────────

DOA_ALGORITHM  = "MUSIC"        # "MUSIC" | "ROOT-MUSIC" | "CAPON" | "ML" | "ESPRIT"
#                                #  MUSIC      : subspace pseudospectrum — RACCOMANDATO,
#                                #               robusto a ogni decorrelazione e N
#                                #  ROOT-MUSIC : radici polinomiali su VULA — precisione sub-griglia
#                                #  CAPON      : MVDR beamformer (P = 1/a^H R^{-1} a)
#                                #  ML         : maximum likelihood stocastico, picchi nitidi
#                                #  ESPRIT     : invarianza rotazionale su VULA — no scan angolare
#                                #  NOTA: Capon/ML sono sensibili a TOEP/FBTOEP su UCA 5-ant;

DECORRELATION  = "FBA"          # Covariance decorrelation method
#                                #  "Off"    : nessuna decorrelazione
#                                #  "FBA"    : Forward-Backward Averaging — RACCOMANDATO
#                                #             funziona correttamente nello spazio VULA per UCA
#                                #  "TOEP"   : Toeplitz Rectification
#                                #  "FBTOEP" : FBA + Toeplitz
#                                #  NOTA: TOEP/FBTOEP richiedono R_vula Toeplitz; con 5 antenne
#                                #  e r_lambda=0.358 la VULA non e' accuratamente Toeplitz
#                                #  (J_0(2.25) ~ 0.08, primo zero a 2.40).  Usare solo FBA.

COV_ALPHA      = 0.95            # Covariance EMA between consecutive frames
#                                #  0.80 → costante di tempo ~5 frame (~0.6 s @ 9 fps)
#                                #  Outdoor: meno multipath → meno memoria necessaria,
#                                #  risposta più rapida a variazioni di angolo.
#                                #  Per sorgente fissa: alzare a 0.90–0.95.
#                                #  Per sorgente in moto: abbassare a 0.5–0.7.

ANGLE_SMOOTH_ALPHA = 0.80        # Circular EMA on estimated angle (final output)
#                                #  Opera su fasori: gestisce correttamente il wrap 0°/360°.
#                                #  0.55 → ~2 frame di damping — più reattivo outdoor.
#                                #  Per sorgente in moto: 0.3–0.5.
#                                #  Per sorgente fissa: 0.7–0.85.
#                                #  0.0 = disabilitato.

AMPLITUDE_NORMALIZE  = True     # Normalize each channel amplitude to unit power
#                                #  before covariance computation.
#                                #  Cancels the effect of unbalanced HW gain
#                                #  (4.6 dB measured on indoor_001).
#                                #  NOTE: do not save normalized samples for HW
#                                #  calibration; use only for DoA.

SQUELCH_ENABLED      = True     # Skip DoA processing when signal is too weak
SQUELCH_THRESHOLD_DB = -60.0    # Absolute power threshold [dBW] measured on raw X
#                                #  (before amplitude normalization).
#                                #  -60 dBW is conservative; indoor 865 MHz signal
#                                #  typically reads -30...-10 dBW at gain 40 dB.

PHASE_OFFSETS_DEG = [0.0, 0.0, 0.0]
#                                # Per-channel hardware phase correction [degrees]
#                                # Deve avere N_ANTENNAS elementi (ora 5).
#                                # Misura: phi[k] = -arg(R[0,k]) sul pannello H,
#                                # puntando le antenne verso una sorgente in campo lontano.
