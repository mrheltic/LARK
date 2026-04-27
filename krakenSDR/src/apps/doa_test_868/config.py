# =============================================================================
#  apps/doa_test_868 — Test DoA 2D (azimuth + elevazione) a 868 MHz
#
#  Script di test per validare gli algoritmi DoA 2D-MUSIC / 2D-Capon /
#  2D-Bartlett sull'UCA a 5 antenne del KrakenSDR in banda ISM 868 MHz.
#
#  Setup fisico consigliato
#  ------------------------
#  TX (beacon):  LibreSDR (AD9363)  →  tx_868_libresdr.py  (tono CW o burst)
#                oppure Arduino/LoRa modulo a 868 MHz
#  RX (array):   KrakenSDR 5-ch     →  Heimdall DAQ attivo su TCP:5000
#  Posizione TX: nota (angolo e distanza misurati), elevazione stimata
#
#  Flusso dati
#  -----------
#  Heimdall DAQ → TCP:5000 ──► KrakenIQSource ──► EMA covariance
#    ──► 2D-MUSIC (UCA) ──► sky-plot + mappa calore + storia (azimuth/elevazione)
#
#  Configurazione rapida
#  ---------------------
#  Adatta FREQ_HZ alla frequenza del tuo beacon TX.
#  Imposta RADIUS_LAMBDA in base al raggio fisico dell'UCA:
#    r [m] / λ [m]  dove λ = 300e6 / FREQ_HZ
#  Esegui il calibration offset (ANT0_OFFSET_DEG) allineando il risultato
#  con una sorgente di riferimento a direzione nota.
#
#  Per la calibrazione iniziale:
#    - Posiziona il TX a 0° Nord rispetto al centro dell'array
#    - Esegui lo script, verifica che il picco MUSIC cada vicino a 0° azimuth
#    - Regola ANT0_OFFSET_DEG in gradi finché non è allineato
# =============================================================================
from __future__ import annotations

import os as _os
import sys as _sys

# ── Hardware base (re-export) ─────────────────────────────────────────────────
_SRC = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from config_hw import *   # noqa: F401, F403

# ── Geometria array ───────────────────────────────────────────────────────────
N_ANTENNAS     = 5        # numero di antenne KrakenSDR popolate
GEOMETRY       = "UCA"    # Uniform Circular Array — non modificare per questo test

RADIUS_LAMBDA  = 0.4253
# Raggio fisico dell'UCA in frazioni di λ.
#
#  Formula: R = d / (2·sin(π/N))  dove d = distanza inter-elemento e N = numero antenne.
#  Con spacing d = λ/2 e N = 5:
#    R = (λ/2) / (2·sin(36°)) = 0.5 / (2 · 0.5878) ≈ 0.4253λ  ← VALORE CORRETTO
#
#  Caso 1 — spacing λ/2 tra elementi adiacenti (default KrakenSDR 868 MHz):
#    RADIUS_LAMBDA = 0.4253
#
#  Caso 2 — raggio fisico noto (es. 12.5 cm a 868 MHz, λ=34.56 cm):
#    RADIUS_LAMBDA = 0.125 / (300e6 / 868e6) = 0.362

ANT_CCW = True
# True  — antenne fisicamente disposte in senso anti-orario (CCW) guardando dall'alto
# False — senso orario (CW)
# Con antenne CCW e matrice CW: az_stimato = 360° − az_reale per az ≠ 0/180°.

ANT0_OFFSET_DEG = 0.0
# Offset angolare dell'antenna 0 rispetto al Nord fisico [gradi].
# Usato per allineare il risultato DoA con la bussola.
# Calibra sul campo con TX a direzione nota (es. 0° = Nord → ant0 deve puntare Nord).

# ── RF / Frequenza ────────────────────────────────────────────────────────────
FREQ_HZ   = 865_197_800   # [Hz] — Arduino beacon rilevato automaticamente a 865.1978 MHz
GAIN_DB   = 20            # Guadagno IF [dB] — aumenta se il segnale è debole

# ── Algoritmo DoA 2D ──────────────────────────────────────────────────────────
DOA_ALGORITHM  = "MUSIC"
# "MUSIC"    — super-risoluzione con circulant smoothing attivo (MUSIC_DECORR='both').
#               Ora stabile indoor: il circulant averaging decorrela il multipath
#               coerente (N sub-array virtuali per rotazione UCA).
#               Con multipath perfettamente coerente e MUSIC+circulant: err_az~0°.
#               PREFERITO per test indoor CW (Arduino beacon).
# "CAPON"    — MVDR: buono per SNR basso, ma con multipath coerente forte
#               può puntare alla direzione media delle riflessioni invece che
#               alla sorgente. Utile come fallback se MUSIC instabile.
# "BARTLETT" — beamformer convenzionale: più robusto, risoluzione bassa

NUM_SIGNALS    = 1        # sorgenti attese (D per il sottospazio noise di MUSIC)
#                          # 0 = auto-detect via MDL (più lento, utile in ambienti caotici)

# ── Griglia di scansione 2D ───────────────────────────────────────────────────
N_AZ       = 180          # punti di scansione azimuth (180 → step 2°)
N_EL       = 36           # punti di scansione elevazione (36 → step 2.5° da 0° a 90°)
#                          # era 18 → 5° — troppo grossolano per variazioni indoor (es.
#                          # sorgente alzata di 0.5m a 2m di distanza = Δel≈14° ~ 5.6 step)
EL_MIN_DEG = 0.0          # elevazione minima [°] — 0.0 per beacon a terra (ISM 868 MHz)

# ── Accumulazione covarianza (EMA) ─────────────────────────────────────────────
COV_ALPHA  = 0.97         # α = EMA weight  (τ = 1/(1-α) frame)
#                          # 0.97 → ~33 frame di memoria = decorrelazione temporale
#                          # del multipath indoor (il segnale CW rimane stazionario,
#                          # le riflessioni cambiano fase nel tempo per vibrazioni/calore).
#                          # Questa è la strategia di decorrelazione corretta per UCA.

# ── Squelch ───────────────────────────────────────────────────────────────────
SQUELCH_ENABLED      = True
SQUELCH_THRESHOLD_DB = -55.0
# Livello minimo di potenza media [dB] sotto cui il frame viene scartato.
# Abbassa se il segnale è debole, alza per filtrare il rumore ambientale.

EIG_SPREAD_MIN_DB = 4.0
# Spread minimo del primo autovalore rispetto al piano del rumore [dB].
# NOTA: dopo circulant smoothing il primo autovalore si riduce leggermente
# (la struttura circolante distribuisce potenza su più lags). Soglia ridotta
# da 6 a 4 dB per non scartare frame validi post-decorrelazione.
# Abbassa a 2 dB se il segnale è debolissimo.

# ── Display ───────────────────────────────────────────────────────────────────
UPDATE_INTERVAL_MS = 200   # intervallo di aggiornamento del plot [ms]
HISTORY_LEN        = 60    # campioni di storia per i grafici azimuth/elevazione

# ── Algoritmo adattivo ad alta elevazione ────────────────────────────────────
HIGH_EL_THRESHOLD_DEG = 50.0
# Sopra questa elevazione [gradi] MUSIC perde risoluzione perché cos(el)→0
# riduce le differenze di fase inter-canale → steering vector quasi indipendenti
# dall'azimuth → spettro MUSIC piatto.
# Abbassato da 70° a 50° (dall'analisi sui dati reali: degradazione inizia ~55°).
# Sopra soglia: passa automaticamente a HIGH_EL_ALGO.
# NOTA: anche sotto soglia lo script fa auto-fallback se PAPR < _PAPR_FLAT_DB.

HIGH_EL_ALGO = "BARTLETT"
# Algoritmo usato quando el_est ≥ HIGH_EL_THRESHOLD_DEG.
# BARTLETT: beamformer convenzionale, più robusto agli errori di fase HW.
# CAPON:    compromesso intermedio.

# ── Decorrelazione covarianza (anti-multipath) ────────────────────────────────
MUSIC_DECORR = "none"
# Decorrelazione pre-processing per MUSIC su UCA.
# IMPORTANTE: 'circulant' e 'fb' NON funzionano per UCA con MUSIC:
#   il circulant smoothing forza R ad essere ciclicamente simmetrica →
#   autovettori = vettori DFT → spettro MUSIC con simmetria N-fold
#   (stella a 5 punte) e PAPR~0 dB. Usare solo 'none' con UCA.
# La decorrelazione corretta per CW indoor è l'EMA temporale (COV_ALPHA=0.97).
# Lasciare 'none' tranne per esperimenti offline con dataset sintetici.
CAPNT_DECORR = "none"
# Per Capon la situazione è identica: circulant forza N-fold symmetry
# anche nell'inversa R^{-1} → Capon si degrada a Bartlett su R circolante.
