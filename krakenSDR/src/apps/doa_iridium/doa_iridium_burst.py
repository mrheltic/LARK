#!/usr/bin/env python3
"""
doa_iridium_burst.py — Multi-satellite burst-gated 2D DoA on Iridium IRA
=========================================================================
DoA 2D (azimuth 0-360°, elevation 5-90°) sui burst IRA Iridium; supporta
la ricezione simultanea di fino a 3 satelliti differenti (discriminati per
offset Doppler del tono di preambolo), con tracker indipendente per ciascuno
e colori distinti sul grafico polare.

Setup hardware
--------------
  Indoor  (ISM 868 MHz, nessuna licenza):
    TX → LibreSDR AD9363 @ 868.1 MHz  (tx_iridium_realsim.py)
    RX → KrakenSDR 5-ch UCA RHCP @ 868.1 MHz  (Heimdall DAQ)
  Outdoor (real Iridium):
    RX → KrakenSDR @ 1626.270 MHz  (nessun TX richiesto)

Come funziona il rilevamento multi-satellite
--------------------------------------------
Ogni burst IRA inizia con 64 simboli di preambolo (tutti dibit 0,0) che
generano un tono puro a fc + Rs/8 = fc + 3125 Hz.  Due satelliti in vista
hanno Doppler diversi (satellite a elevazione 60° ≈ ±10 kHz; satellite a
horizon = ±40 kHz a 1626 MHz), quindi i loro toni di preambolo cadono a
frequenze diverse nel riferimento del ricevitore.

Il modulo `_scan_doppler_peaks()` esegue una scansione FFT di larghezza
±DOPPLER_SCAN_BW_HZ (default ±45 kHz) intorno al tono nominale 3125 Hz su
ogni finestra burst, restituendo fino a MAX_SATELLITES picchi.  Per ogni
picco viene applicato un BPF stretto, poi eseguito il DoA 2D-MUSIC.  Il
risultato viene associato al satellite più vicino (per CFO) nel registro dei
tracker attivi.

Utilizzo
--------
    python3 doa_iridium_burst.py              # hardware reale (Heimdall attivo)
    python3 doa_iridium_burst.py --demo       # simulazione 2 satelliti sintetici
    python3 doa_iridium_burst.py --demo --n-demo-sats 3
    python3 doa_iridium_burst.py --freq 1626.270
    python3 doa_iridium_burst.py --calibrate 45.0
    python3 doa_iridium_burst.py --out-dir /tmp/doa_iridium
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
_DATA_DIR = os.path.normpath(os.path.join(_SRC, "..", "data", "doa_iridium"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

import socket as _socket
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import config as C
from hardware.kraken_iq_source import KrakenIQSource
from core.doa_uca_2d import (
    UcaConfig, CovarianceAccumulatorUca,
    doa_music_uca_2d, doa_capon_uca_2d, doa_bartlett_uca_2d,
    doa_root_music_uca_2d, doa_unitary_esprit_uca_2d, doa_mfba_music_uca_2d,
    find_peak_uca_2d, eigenvalue_spread_uca_db, snr_uca_db,
    extract_pilot_tone, amplitude_normalize_channels,
    crb_azimuth_deg, estimate_signal_count_mdl,
)
from core.doa_algorithms import apply_phase_correction as _apply_phase_correction
from core.tracking import KalmanAngular, KalmanScalar
from core.gates import circ_median_deg
from core.tone_extraction import (
    find_preamble_onset as _fpo_core,
    find_tone_onset    as _fto_core,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#1a1d27"; BG2   = "#21253a"; BG3  = "#2a2f47"
C_BDR = "#3b4263"; C_MUT = "#8891b0"; C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIO  = "#a78bfa"; C_ROSE = "#f16b6f"; C_LIME  = "#6dd97d"

# 3 colori distinti per i 3 satellite tracker (override da C.SAT_COLORS se presente)
_SAT_COLORS = list(getattr(C, "SAT_COLORS", ["#f4a431", "#4ecdc4", "#a78bfa"]))

# ── Parametri IRA Iridium (da gr-iridium / iridium-toolkit) ──────────────────
_SYMBOL_RATE   = 25_000           # sps
_SPS_BASE      = 10               # campioni/simbolo @ 250 kHz
_IRA_SAMPLE_RATE = _SYMBOL_RATE * _SPS_BASE   # 250 000 Hz
_TX_SAMPLE_RATE  = 1_000_000      # AD9363 TX rate
_IRA_UPS         = _TX_SAMPLE_RATE // _IRA_SAMPLE_RATE   # = 4

_PREAMBLE_SYMS = 64
_BURST_SYMS    = 245              # 64 pream + 12 UW + 167 data + 2 tail
_SUPERFRAME_S  = 0.090            # 90 ms tra burst IRA dello stesso satellite

# Tono di preambolo: dibit (0,0) → +π/4 per simbolo → puro tono @ fc + Rs/8
_PREAMBLE_TONE_HZ = _SYMBOL_RATE // 8   # = 3125 Hz

_FS = float(getattr(C, "SAMPLE_RATE_HZ", 1_024_000))

_BURST_SAMPLES = int(round(_BURST_SYMS    * _SPS_BASE * _IRA_UPS * _FS / _TX_SAMPLE_RATE))
_PRE_SAMPLES   = int(round(_PREAMBLE_SYMS * _SPS_BASE * _IRA_UPS * _FS / _TX_SAMPLE_RATE))
_SF_SAMPLES    = int(round(_SUPERFRAME_S  * _FS))

_WINDOW_SAMPLES = min(_BURST_SAMPLES + 512, _SF_SAMPLES - 256)

_ENERGY_WIN    = 256
_TONE_SCAN_WIN = 512
_PAPR_INST_MIN_DB = 8.0
_SPEC_EMA         = 0.25

# ── Demo parametri default (2 satelliti) ─────────────────────────────────────
_DEMO_SATS = [
    dict(az=45.0,   el=30.0, doppler=+8_000,  snr_db=15.0),  # Sat-0 amber
    dict(az=200.0,  el=55.0, doppler=-14_000, snr_db=12.0),  # Sat-1 teal
    dict(az=310.0,  el=20.0, doppler=+28_000, snr_db=10.0),  # Sat-2 violet
]

# =============================================================================
# DoA algorithm dispatcher (evita if/elif ripetuto)
# =============================================================================


def _check_narrowband(freq_hz: float, cfg: UcaConfig) -> None:
    """Print a one-time validation of the narrowband array assumption.

    The narrowband assumption (Salama 2025 §2.3) requires that the signal
    bandwidth B satisfies:

        B × τ_max  ≪  1

    where τ_max = 2r / c is the maximum propagation delay across the array
    aperture (diameter / speed of light).

    For Iridium IRA the relevant bandwidth is the preamble symbol rate
    (25 kSps); the 15 kHz BPF used in the pipeline makes it even narrower.
    At 1626 MHz with the 5-element KrakenSDR UCA (r ≈ 7.8 cm):

        τ_max ≈ 2 × 0.078 / 3e8 ≈ 0.52 ns
        B × τ_max = 25 000 × 0.52e-9 ≈ 1.3e-5  ≪  1  ✓

    The narrowband assumption is more than satisfied; wideband ISSM processing
    (Salama 2025 §7.2) is not needed for this setup.
    """
    lam_m   = 3e8 / freq_hz                   # wavelength [m]
    r_m     = cfg.radius_lambda * lam_m        # UCA physical radius [m]
    tau_max = 2.0 * r_m / 3e8                  # max delay across diameter [s]
    B       = float(_SYMBOL_RATE)              # Iridium preamble BW ≈ symbol rate
    ratio   = B * tau_max
    ok      = ratio < 0.01
    status  = "✓  narrowband" if ok else "⚠  wideband effects possible"
    print(
        f"[NARROWBAND CHECK] B×τmax = {ratio:.2e}  {status}\n"
        f"  diameter = {2*r_m*100:.1f} cm, τmax = {tau_max*1e9:.2f} ns, "
        f"BW = {B/1e3:.0f} kHz @ {freq_hz/1e6:.3f} MHz"
    )


def _run_doa_algo(X_cal: np.ndarray, R: np.ndarray,
                  cfg: UcaConfig, algo: str) -> np.ndarray:
    decorr = getattr(C, "MUSIC_DECORR", "none")
    if algo == "capon":
        return doa_capon_uca_2d(X_cal, cfg, R_in=R, decorr=decorr)
    if algo == "bartlett":
        return doa_bartlett_uca_2d(X_cal, cfg, R_in=R)
    if algo == "root-music":
        return doa_root_music_uca_2d(R, cfg)[0]
    if algo == "unitary-esprit":
        return doa_unitary_esprit_uca_2d(R, cfg)[0]
    if algo == "mfba-music":
        return doa_mfba_music_uca_2d(R, cfg)[0]
    return doa_music_uca_2d(X_cal, cfg, R_in=R, decorr=decorr)


# =============================================================================
# Doppler scan — trova tutti i toni di preambolo IRA nella finestra burst
# =============================================================================

def _scan_doppler_peaks(
    iq: np.ndarray,
    fs: float,
    nom_tone_hz: float,
    scan_bw_hz: float,
    n_peaks: int,
    min_sep_hz: float,
    min_snr_db: float = 6.0,
) -> list[tuple[float, float]]:
    """
    Scansione FFT intorno a nom_tone_hz ± scan_bw_hz.

    Restituisce una lista di (tone_hz, snr_db) ordinata per potenza decrescente,
    con al massimo n_peaks picchi separati di almeno min_sep_hz.

    tone_hz è l'offset dal carrier (= vero Doppler + 3125 Hz);
    il Doppler del satellite è tone_hz - 3125 Hz.
    """
    N = len(iq)
    if N < 128:
        return [(nom_tone_hz, 0.0)]

    # Scelgo la potenza di 2 più vicina per FFT veloce
    nfft = max(128, 1 << int(np.floor(np.log2(N))))
    win  = np.blackman(nfft)
    seg  = iq[:nfft] * win
    Spec = np.abs(np.fft.fft(seg, n=nfft)) ** 2
    freqs = np.fft.fftfreq(nfft, 1.0 / fs)

    # Rearrange: positivo va nella zona corretta
    # Voglio la forma "dc al centro" per semplicità
    Spec  = np.fft.fftshift(Spec)
    freqs = np.fft.fftshift(freqs)

    # Maschera: solo la banda di ricerca
    lo = nom_tone_hz - scan_bw_hz
    hi = nom_tone_hz + scan_bw_hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return [(nom_tone_hz, 0.0)]

    Sb = Spec[mask].copy()
    fb = freqs[mask]
    noise_floor = float(np.median(Sb)) + 1e-20

    results: list[tuple[float, float]] = []
    for _ in range(n_peaks):
        idx = int(np.argmax(Sb))
        snr = 10.0 * np.log10(float(Sb[idx]) / noise_floor)
        if snr < min_snr_db:
            break
        results.append((float(fb[idx]), float(snr)))
        # Annulla finestra intorno al picco
        null = np.abs(fb - fb[idx]) < min_sep_hz
        Sb[null] = 0.0

    return results if results else [(nom_tone_hz, 0.0)]


# =============================================================================
# Per-satellite tracker
# =============================================================================

def _make_sat_tracker(
    sat_id: int, color: str, cfo_hz: float,
    n_el: int, n_az: int, hist_len: int, multi_n: int,
    el_mid: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        sat_id      = sat_id,
        color       = color,
        cfo_hz      = cfo_hz,          # offset Doppler corrente (EMA)
        az_kf       = KalmanAngular(q=5.0, r=20.0),
        el_kf       = KalmanScalar(q=2.0,  r=8.0),
        az_ema      = 0.0,
        el_ema      = el_mid,
        az_phasor   = np.exp(0j),
        az_hist     = collections.deque(maxlen=hist_len),
        el_hist     = collections.deque(maxlen=hist_len),
        snr_hist    = collections.deque(maxlen=hist_len),
        cfo_hist    = collections.deque(maxlen=hist_len),
        spec2d      = np.full((n_el, n_az), -40.0),
        R_batch     = collections.deque(maxlen=multi_n),
        burst_count = 0,
        last_seen   = time.monotonic(),
        az_init     = True,
        papr_db     = 0.0,
        snr_db      = 0.0,
        az_deg      = 0.0,
        el_deg      = el_mid,
        no_doa      = True,
    )


def _find_or_create_tracker(
    satellites: dict, cfo_hz: float,
    sat_colors: list, min_sep_hz: float, max_sats: int,
    n_el: int, n_az: int, hist_len: int, multi_n: int, el_mid: float,
) -> SimpleNamespace | None:
    """
    Cerca tra i tracker attivi quello con CFO più vicino a cfo_hz.
    Se la distanza è < min_sep_hz × 2 → aggiorna il CFO e restituisce il tracker.
    Altrimenti crea un nuovo tracker (se non si supera max_sats).
    """
    best_id, best_dist = None, float("inf")
    for sid, trk in satellites.items():
        d = abs(trk.cfo_hz - cfo_hz)
        if d < best_dist:
            best_dist, best_id = d, sid

    if best_id is not None and best_dist < min_sep_hz * 2.0:
        trk = satellites[best_id]
        trk.cfo_hz = 0.90 * trk.cfo_hz + 0.10 * cfo_hz  # EMA
        return trk

    if len(satellites) >= max_sats:
        return None   # troppi satelliti, ignora questo picco

    new_id = (max(satellites.keys()) + 1) if satellites else 0
    color  = sat_colors[new_id % len(sat_colors)]
    trk    = _make_sat_tracker(new_id, color, cfo_hz, n_el, n_az, hist_len, multi_n, el_mid)
    satellites[new_id] = trk
    return trk


def _prune_trackers(satellites: dict, timeout_s: float) -> None:
    now  = time.monotonic()
    dead = [sid for sid, trk in satellites.items() if now - trk.last_seen > timeout_s]
    for sid in dead:
        del satellites[sid]


def _update_tracker(
    trk: SimpleNamespace,
    az_doa: float, el_doa: float,
    papr_db: float, snr_db: float,
    cfo_hz: float, spec2d: np.ndarray,
    az_alpha: float, el_alpha: float,
    el_min: float, el_max: float,
) -> None:
    el_step = max(0.1, (el_max - el_min) / 90.0)
    on_floor   = el_doa < el_min + el_step
    on_ceiling = el_doa > el_max - el_step

    # Azimuth EMA (phasore circolare)
    phasor_new = np.exp(1j * np.deg2rad(az_doa))
    if trk.az_init:
        trk.az_phasor = phasor_new
        trk.az_kf.update(az_doa)
        trk.az_init   = False
    else:
        trk.az_phasor = az_alpha * trk.az_phasor + (1.0 - az_alpha) * phasor_new
    trk.az_ema = float(np.degrees(np.angle(trk.az_phasor)) % 360.0)
    trk.az_kf.update(az_doa)

    if not on_floor and not on_ceiling:
        trk.el_ema = el_alpha * trk.el_ema + (1.0 - el_alpha) * el_doa
    trk.el_kf.update(el_doa)

    trk.az_hist.append(trk.az_ema)
    trk.el_hist.append(trk.el_ema)
    trk.snr_hist.append(snr_db)
    trk.cfo_hist.append(cfo_hz)
    trk.cfo_hz  = 0.85 * trk.cfo_hz + 0.15 * cfo_hz
    trk.spec2d  = spec2d
    trk.papr_db = papr_db
    trk.snr_db  = snr_db
    trk.az_deg  = trk.az_ema
    trk.el_deg  = trk.el_ema
    trk.no_doa  = False
    trk.burst_count += 1
    trk.last_seen    = time.monotonic()


# =============================================================================
# Helper: heimdall check
# =============================================================================

def _check_heimdall(host: str, port: int) -> bool:
    try:
        s = _socket.create_connection((host, port), timeout=2.0)
        s.close()
        return True
    except OSError:
        return False

_circ_median = circ_median_deg


# =============================================================================
# Global shared state (unico lock per thread)
# =============================================================================

def _make_state(n_az: int, n_el: int) -> SimpleNamespace:
    return SimpleNamespace(
        # dome complessivo (proiezione sul satellite dominante o somma)
        az_spec     = np.full(n_az, -40.0),
        spec2d      = np.full((n_el, n_az), -40.0),
        # per retrocompatibilità con calibrazione
        eig_db      = np.zeros(5),
        phase_diffs = np.zeros(4),
        phase_hist  = [collections.deque(maxlen=C.HISTORY_LEN) for _ in range(4)],
        energy_hist = collections.deque(maxlen=C.HISTORY_LEN),
        snr_hist    = collections.deque(maxlen=C.HISTORY_LEN),
        burst_n     = 0,
        frame_n     = 0,
        no_signal   = True,
        lock        = threading.Lock(),
        running     = True,
        # Latest per-burst metrics (updated by acquisition thread)
        crb_az_deg  = float("inf"),  # CRB azimuth standard deviation [°]
        mdl_k       = 1,             # MDL-estimated number of sources
        # ── Satellite tracker registry (dict {sat_id: namespace}) ──────────────
        satellites  = {},     # aggiornato dal thread acq, letto dal thread UI
        sat_colors  = list(_SAT_COLORS),
        # ── Calibrazione (usa solo sat_id=0) ────────────────────────────────────
        R_cal       = None,
        n_cal_bursts= 0,
        # ── Recording ────────────────────────────────────────────────────────────
        rec_enabled  = True,
        rec_iq_enabled = False,
        rec_t        = [], rec_az    = [], rec_el    = [],
        rec_papr     = [], rec_snr   = [], rec_eig   = [],
        rec_phase    = [], rec_R     = [], rec_has_sig= [],
        rec_sat_cfo  = [], rec_X     = [],
        # Cramér-Rao Bound (Salama 2025 §8.2.1) and MDL source count per burst
        rec_crb      = [], rec_mdl_k = [],
    )


# =============================================================================
# Energy detector
# =============================================================================

def _detect_bursts(iq: np.ndarray, threshold_factor: float = 6.0) -> list[int]:
    n_blocks = len(iq) // _ENERGY_WIN
    if n_blocks == 0:
        return []
    pwr = np.array([np.mean(np.abs(iq[i*_ENERGY_WIN:(i+1)*_ENERGY_WIN])**2)
                    for i in range(n_blocks)])
    noise_floor = float(np.median(pwr)) + 1e-20
    active = pwr > threshold_factor * noise_floor
    edges  = np.diff(active.astype(np.int8), prepend=0)
    starts = np.where(edges > 0)[0]
    min_gap = max(1, _SF_SAMPLES // 2 // _ENERGY_WIN)
    out: list[int] = []
    last = -min_gap - 1
    for blk in starts:
        if blk - last >= min_gap:
            out.append(int(blk * _ENERGY_WIN))
            last = blk
    return out


# =============================================================================
# Preamble-onset wrappers
# =============================================================================

def _find_preamble_onset(iq, b_start, n_total,
                          fs=_FS, win=_TONE_SCAN_WIN, known_hz=None):
    return _fpo_core(iq, b_start, n_total, fs=fs, win=win, known_hz=known_hz,
                     burst_samples=_BURST_SAMPLES,
                     preamble_tone_hz=float(_PREAMBLE_TONE_HZ))


def _find_tone_onset(iq, b_start, n_total,
                     tone_hz=float(_PREAMBLE_TONE_HZ),
                     fs=_FS, win=_TONE_SCAN_WIN):
    return _fto_core(iq, b_start, n_total, tone_hz=tone_hz, fs=fs, win=win,
                     burst_samples=_BURST_SAMPLES, preamble_samples=_PRE_SAMPLES)


# =============================================================================
# Demo: genera frame sintetico con N_sats satelliti a diversi az/el/Doppler
# =============================================================================

def _demo_frame(rng: np.random.Generator, cfg: UcaConfig,
                demo_params: list[dict]) -> np.ndarray:
    """Genera un frame multi-satellite sintetico (senza hardware)."""
    N = _SF_SAMPLES
    X = np.zeros((cfg.n_ant, N), dtype=np.complex128)
    pos = cfg.positions   # (n_ant, 2)  [Est, Nord]

    for sat in demo_params:
        az_r   = np.deg2rad(sat["az"])
        el_r   = np.deg2rad(sat["el"])
        fd     = sat["doppler"]          # Doppler Hz
        snr_lin = 10 ** (sat["snr_db"] / 10.0)

        # Steering vector (planare, 2D)
        tau = 2 * np.pi * (pos[:, 0] * np.cos(el_r) * np.sin(az_r)
                           + pos[:, 1] * np.cos(el_r) * np.cos(az_r))

        t = np.arange(N, dtype=np.float64)
        tone = np.exp(2j * np.pi * (_PREAMBLE_TONE_HZ + fd) / _FS * t)

        b0     = 512    # offset burst all'interno del frame
        pre_len = _PRE_SAMPLES
        b_end  = min(b0 + pre_len, N)

        for k in range(cfg.n_ant):
            X[k, b0:b_end] += (
                np.sqrt(snr_lin)
                * np.exp(1j * tau[k])
                * tone[b0:b_end]
            )

    # AWGN
    X += (rng.standard_normal((cfg.n_ant, N)) + 1j*rng.standard_normal((cfg.n_ant, N))) / np.sqrt(2)
    return X


# =============================================================================
# Acquisition + DoA thread
# =============================================================================

def _acq_loop(
    src, cfg: UcaConfig, algo: str,
    acc: CovarianceAccumulatorUca, S: SimpleNamespace,
    demo: bool = False,
    demo_params: list[dict] | None = None,
) -> None:
    rng = np.random.default_rng(42)
    _fs = _FS

    _phase_offs = list(getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * cfg.n_ant))
    _phase_offs = (_phase_offs + [0.0] * cfg.n_ant)[:cfg.n_ant]
    _has_cal    = any(o != 0.0 for o in _phase_offs)
    _az_alpha   = float(getattr(C, "AZ_SMOOTH_ALPHA",   0.50))
    _el_alpha   = float(getattr(C, "EL_SMOOTH_ALPHA",   0.50))
    el_mid      = (cfg.el_min_deg + cfg.el_max_deg) / 2.0

    _papr_min        = float(getattr(C, "PAPR_INST_MIN_DB", _PAPR_INST_MIN_DB))
    _eig_min         = float(getattr(C, "EIG_SPREAD_MIN_DB", 0.5))
    _multi_n         = max(1, int(getattr(C, "MULTI_BURST_N",       3)))
    _scan_bw         = float(getattr(C, "DOPPLER_SCAN_BW_HZ",   45_000))
    _min_sep         = float(getattr(C, "SAT_MIN_SEP_HZ",        5_000))
    _sat_timeout     = float(getattr(C, "SAT_TIMEOUT_S",          8.0))
    _max_sats        = int(getattr(C, "MAX_SATELLITES",            3))
    _bpf_bw          = float(getattr(C, "PREAMBLE_BPF_BW_HZ",  15_000))
    _hist_len        = int(getattr(C, "HISTORY_LEN",              100))

    _buf: list[np.ndarray] = []
    _buf_len  = 0
    _min_buf  = _SF_SAMPLES + _WINDOW_SAMPLES + 2048

    _no_burst_streak = 0
    _diag_t0   = time.monotonic()
    _cnt_det = _cnt_eig = _cnt_papr = _cnt_acc = 0

    _tone_known: dict[int, float] = {}   # sat_id → last known tone_hz

    demo_params = demo_params or _DEMO_SATS[:2]

    while S.running:
        # ── Acquisisci frame ──────────────────────────────────────────────────
        if demo:
            frame = _demo_frame(rng, cfg, demo_params)
            time.sleep(_SUPERFRAME_S)
        else:
            try:
                frame = src.get_iq_online()
            except Exception:
                continue
            if frame is None or frame.shape[1] < 512:
                continue

        with S.lock:
            S.frame_n += 1

        _buf.append(frame)
        _buf_len += frame.shape[1]
        if _buf_len < _min_buf:
            continue

        X_stream = np.hstack(_buf)
        _buf.clear(); _buf_len = 0
        n_total  = X_stream.shape[1]

        # ── Energy detection su ch0 ───────────────────────────────────────────
        bursts = _detect_bursts(X_stream[0])
        if not bursts:
            pwr_db = float(10 * np.log10(np.mean(np.abs(X_stream[0])**2) + 1e-20))
            with S.lock:
                S.energy_hist.append(pwr_db)
                S.no_signal = True
            _no_burst_streak += 1
            if _no_burst_streak == 10:
                print(
                    "[WARN] 10 frame consecutivi senza burst IRA rilevato.\n"
                    "  Indoor: assicurarsi che il LibreSDR TX sia attivo\n"
                    "         (python3 tx_iridium_realsim.py --cyclic)\n"
                    "  Outdoor: attendere una passata Iridium (controllare TLE)\n"
                    "  Test: aggiungere --demo per simulare il segnale."
                )
            continue
        _no_burst_streak = 0
        _cnt_det += len(bursts)

        # ── Diagnostica periodica ─────────────────────────────────────────────
        now = time.monotonic()
        if now - _diag_t0 >= 10.0:
            sats_str = ", ".join(
                f"S{sid}:az={trk.az_deg:.0f}°/el={trk.el_deg:.0f}°/fd={trk.cfo_hz:+.0f}Hz"
                for sid, trk in S.satellites.items()
            )
            print(
                f"[DIAG] det={_cnt_det} eig_rej={_cnt_eig} "
                f"papr_rej={_cnt_papr} acc={_cnt_acc} | "
                f"sats={len(S.satellites)}: {sats_str}"
            )
            _diag_t0 = now
            _cnt_det = _cnt_eig = _cnt_papr = _cnt_acc = 0

        pwr_db = float(10 * np.log10(np.mean(np.abs(X_stream[0])**2) + 1e-20))

        for b_start in bursts:
            b_end = min(b_start + _WINDOW_SAMPLES, n_total)
            if b_end - b_start < _PRE_SAMPLES:
                continue

            X_win = X_stream[:, b_start:b_end]   # (n_ant, window)
            X0    = X_win[0]

            # ── Scansione FFT per tutti i toni di preambolo in questa finestra ─
            peaks = _scan_doppler_peaks(
                X0, _fs,
                nom_tone_hz = float(_PREAMBLE_TONE_HZ),
                scan_bw_hz  = _scan_bw,
                n_peaks     = _max_sats,
                min_sep_hz  = _min_sep,
                min_snr_db  = 6.0,
            )

            for tone_hz, peak_snr_db in peaks:
                cfo_hz = tone_hz - _PREAMBLE_TONE_HZ

                # ── BPF stretto intorno a questo tono ─────────────────────────
                X_bpf = extract_pilot_tone(X_win, _fs, tone_hz=tone_hz, bw_hz=_bpf_bw)
                X_bpf = amplitude_normalize_channels(X_bpf)

                if _has_cal:
                    X_cal = _apply_phase_correction(X_bpf, _phase_offs)
                else:
                    X_cal = X_bpf

                # ── Covarianza istantanea + gate eigenvalue ────────────────────
                try:
                    R_inst = (X_cal @ X_cal.conj().T) / X_cal.shape[1]
                    eig    = eigenvalue_spread_uca_db(R_inst)
                    snr    = snr_uca_db(R_inst)
                except Exception:
                    continue

                if eig[0] < _eig_min:
                    _cnt_eig += 1
                    continue

                # ── DoA istantaneo (PAPR gate) ────────────────────────────────
                try:
                    spec2d_inst = _run_doa_algo(X_cal, R_inst, cfg, algo)
                    az_i, el_i, papr_i = find_peak_uca_2d(spec2d_inst, cfg)
                except Exception:
                    continue

                if papr_i < _papr_min:
                    _cnt_papr += 1
                    continue

                # ── Associa al tracker ────────────────────────────────────────
                with S.lock:
                    trk = _find_or_create_tracker(
                        S.satellites, cfo_hz,
                        S.sat_colors, _min_sep, _max_sats,
                        cfg.n_el, cfg.n_az, _hist_len, _multi_n, el_mid,
                    )
                if trk is None:
                    continue

                trk.R_batch.append(R_inst)
                if len(trk.R_batch) < _multi_n:
                    continue

                # ── DoA media su _multi_n burst ───────────────────────────────
                R_avg = np.mean(list(trk.R_batch), axis=0)
                try:
                    spec2d = _run_doa_algo(X_cal, R_avg, cfg, algo)
                    az_doa, el_doa, papr_doa = find_peak_uca_2d(spec2d, cfg)
                except Exception:
                    continue

                if papr_doa < _papr_min:
                    continue

                _cnt_acc += 1
                acc.update(R_avg)   # per calibrazione

                phase_diffs = np.degrees(np.angle(R_avg[1:, 0]))

                # CRB for azimuth (Salama 2025 §8.2.1; Stoica & Nehorai 1990).
                # Uses the DoA-averaged covariance SNR and the number of IQ
                # samples in the BPF output window as the snapshot count.
                # crb_azimuth_deg expects per-element SNR; snr_uca_db returns
                # the array-gain SNR = M × per-element, so subtract 10·log10(M).
                n_snaps  = X_cal.shape[1]
                snr_per_element_db = snr - 10.0 * np.log10(float(cfg.n_ant))
                crb_deg_ = crb_azimuth_deg(snr_per_element_db, n_snaps, cfg,
                                           el_deg=el_doa)

                # MDL source-count estimate on the averaged covariance.
                # For Iridium IRA we expect K=1; deviations indicate multipath
                # or a second satellite within the Doppler search window.
                mdl_k = estimate_signal_count_mdl(R_avg, n_snaps,
                                                  max_signals=cfg.n_ant - 1)

                with S.lock:
                    _update_tracker(
                        trk, az_doa, el_doa, papr_doa, snr,
                        cfo_hz, spec2d, _az_alpha, _el_alpha,
                        cfg.el_min_deg, cfg.el_max_deg,
                    )
                    S.no_signal = False
                    S.burst_n  += 1
                    S.energy_hist.append(pwr_db)
                    S.snr_hist.append(float(snr))
                    S.eig_db    = eig
                    S.crb_az_deg = float(crb_deg_)
                    S.mdl_k      = int(mdl_k)
                    S.phase_diffs = phase_diffs
                    for i in range(4):
                        S.phase_hist[i].append(float(phase_diffs[i]))
                    # Spec2d globale: max su tutti i satellite tracker
                    if S.satellites:
                        S.spec2d = np.max([t.spec2d for t in S.satellites.values()], axis=0)
                        S.az_spec = S.spec2d.max(axis=0)

                    if S.rec_enabled:
                        S.rec_t.append(time.time())
                        S.rec_az.append(float(trk.az_deg))
                        S.rec_el.append(float(trk.el_deg))
                        S.rec_papr.append(float(papr_doa))
                        S.rec_snr.append(float(snr))
                        S.rec_eig.append(eig.copy())
                        S.rec_phase.append(phase_diffs.copy())
                        S.rec_R.append(R_avg.copy())
                        S.rec_has_sig.append(True)
                        S.rec_sat_cfo.append(float(cfo_hz))
                        S.rec_crb.append(float(crb_deg_))
                        S.rec_mdl_k.append(int(mdl_k))
                        if S.rec_iq_enabled:
                            S.rec_X.append(X_cal[:, :_PRE_SAMPLES].copy())

                # Rimuovi tracker scaduti
                with S.lock:
                    _prune_trackers(S.satellites, _sat_timeout)


# =============================================================================
# Build UI — multi-satellite
# =============================================================================

def _build_ui(S: SimpleNamespace, cfg: UcaConfig,
              algo: str, freq_hz: int) -> None:
    n_ant  = cfg.n_ant
    n_sig  = cfg.num_expected_signals
    el_min = cfg.el_min_deg
    el_max = cfg.el_max_deg
    n_az   = cfg.n_az
    H      = C.HISTORY_LEN
    az_rad = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    MAX_SATS = int(getattr(C, "MAX_SATELLITES", 3))
    sat_colors = list(getattr(C, "SAT_COLORS", _SAT_COLORS))

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             left=0.05, right=0.97, top=0.95, bottom=0.05,
                             hspace=0.32, wspace=0.30)

    # ── [0,0]  Skyplot polare ─────────────────────────────────────────────────
    ax_sky = fig.add_subplot(gs[0, 0], projection="polar", facecolor=BG2)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90)
    ax_sky.set_rticks([10, 30, 60, 90])
    ax_sky.tick_params(colors=C_MUT, labelsize=7)
    ax_sky.set_facecolor(BG2)
    for sp in ax_sky.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_sky.grid(color=C_BDR, lw=0.5, alpha=0.4)
    ax_sky.set_title(f"DoA skyplot — {MAX_SATS} sat max",
                      color=C_TEXT, fontsize=9, pad=12)

    # Spettro di sfondo (proiezione az)
    _r0 = np.zeros(n_az + 1)
    sky_spec_fill, = ax_sky.fill(_r0, _r0, color=C_BLUE, alpha=0.12)
    sky_spec_line, = ax_sky.plot([], [], "-", color=C_BLUE, lw=0.8, alpha=0.45)

    # Pre-allocazione marker per MAX_SATS satelliti
    sky_dots  = []
    sky_texts = []
    for i in range(MAX_SATS):
        col = sat_colors[i % len(sat_colors)]
        dot, = ax_sky.plot([], [], "o", color=col, ms=11, zorder=7)
        txt  = ax_sky.text(0.5, 0.5, f"S{i}", transform=ax_sky.transAxes,
                            ha="center", va="center", color=col,
                            fontsize=7, visible=False)
        sky_dots.append(dot)
        sky_texts.append(txt)

    lbl_nosig = ax_sky.text(np.pi/2, 45, "in attesa burst…",
                              ha="center", va="center", color=C_MUT,
                              fontsize=9)

    # ── [0,1]  Az/El history — per satellite ─────────────────────────────────
    ax_hist = fig.add_subplot(gs[0, 1], facecolor=BG2)
    ax_hist.set_facecolor(BG2)
    ax_hist.set_title("Storico Az / El per satellite", color=C_TEXT, fontsize=9)
    ax_hist.set_xlabel("Burst recenti →", color=C_MUT, fontsize=8)
    ax_hist.set_ylabel("Angolo [°]", color=C_MUT, fontsize=8)
    ax_hist.set_xlim(0, H); ax_hist.set_ylim(-5, 375)
    ax_hist.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_hist.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_hist.grid(color=C_BDR, lw=0.4, alpha=0.4)

    hist_az_lines, hist_el_lines = [], []
    for i in range(MAX_SATS):
        col = sat_colors[i % len(sat_colors)]
        la, = ax_hist.plot([], [], "-",  color=col,  lw=1.5, label=f"S{i} Az")
        le, = ax_hist.plot([], [], "--", color=col,  lw=1.0, label=f"S{i} El")
        hist_az_lines.append(la)
        hist_el_lines.append(le)
    ax_hist.legend(loc="upper left", fontsize=6, ncol=2,
                    facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # ── [0,2]  Eigenvalue bar ──────────────────────────────────────────────────
    ax_eig = fig.add_subplot(gs[0, 2], facecolor=BG2)
    ax_eig.set_facecolor(BG2)
    ax_eig.set_title("Eigenvalue spread (λ dB)", color=C_TEXT, fontsize=9)
    ax_eig.set_xlim(-0.5, n_ant - 0.5)
    ax_eig.set_xticks(range(n_ant))
    ax_eig.set_xticklabels([f"λ{i+1}" for i in range(n_ant)],
                             fontsize=8, color=C_MUT)
    ax_eig.set_ylim(-3, 42)
    ax_eig.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_eig.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_eig.grid(axis="y", color=C_BDR, lw=0.5, alpha=0.4, zorder=1)
    ax_eig.axhline(0, color=C_MUT, lw=0.8, ls="--", alpha=0.5, zorder=2)
    if 0 < n_sig < n_ant:
        ax_eig.axvline(n_sig - 0.5, color=C_ROSE, lw=1.0, ls=":", alpha=0.75)
    eig_cols = [C_AMBER if i < n_sig else "#5a618a" for i in range(n_ant)]
    eig_bars = ax_eig.bar(range(n_ant), np.zeros(n_ant),
                           color=eig_cols, edgecolor=BG2, linewidth=0.6, zorder=3)
    txt_info  = ax_eig.text(n_ant/2, 39,  "", ha="center", va="top",
                             color=C_TEXT, fontsize=8)
    txt_crb   = ax_eig.text(n_ant/2, 33, "", ha="center", va="top",
                             color=C_TEAL, fontsize=7)
    txt_sats  = ax_eig.text(n_ant/2, 10, "satelliti: 0", ha="center", va="top",
                             color=C_LIME, fontsize=8)
    txt_bursts= ax_eig.text(n_ant/2, 4,  "burst: 0",     ha="center", va="top",
                             color=C_MUT,  fontsize=8)

    # ── [1,0]  2D MUSIC heatmap ────────────────────────────────────────────────
    ax_2d = fig.add_subplot(gs[1, 0], facecolor=BG2)
    ax_2d.set_facecolor(BG2)
    ax_2d.set_title("2D MUSIC  az–el  (tutti i satelliti)", color=C_TEXT, fontsize=9)
    ax_2d.set_xlabel("Azimuth [°]", color=C_MUT, fontsize=8)
    ax_2d.set_ylabel("Elevazione [°]", color=C_MUT, fontsize=8)
    ax_2d.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_2d.spines.values():
        sp.set_edgecolor(C_BDR)
    hm_img = ax_2d.imshow(
        np.full((cfg.n_el, cfg.n_az), -40.0), aspect="auto", origin="lower",
        extent=[0, 360, el_min, el_max], vmin=-40, vmax=0, cmap="plasma",
    )
    hm_peaks = [ax_2d.plot([], [], c=sat_colors[i % len(sat_colors)],
                             marker="+", ms=12, mew=2, ls="", zorder=5)[0]
                for i in range(MAX_SATS)]
    plt.colorbar(hm_img, ax=ax_2d, fraction=0.046, pad=0.04,
                  label="dB", location="right")

    # ── [1,1]  Doppler CFO per satellite ──────────────────────────────────────
    ax_cfo = fig.add_subplot(gs[1, 1], facecolor=BG2)
    ax_cfo.set_facecolor(BG2)
    ax_cfo.set_title("Doppler CFO per satellite", color=C_TEXT, fontsize=9)
    ax_cfo.set_xlabel("Burst recenti →", color=C_MUT, fontsize=8)
    ax_cfo.set_ylabel("CFO = fd [Hz]", color=C_MUT, fontsize=8)
    ax_cfo.set_xlim(0, H)
    ax_cfo.axhline(0, color=C_BDR, lw=0.6)
    ax_cfo.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_cfo.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_cfo.grid(color=C_BDR, lw=0.4, alpha=0.4)
    cfo_lines = []
    for i in range(MAX_SATS):
        cl, = ax_cfo.plot([], [], "-", color=sat_colors[i % len(sat_colors)],
                           lw=1.5, label=f"S{i}")
    cfo_lines.append(cl)
    # fix: rebuild properly
    cfo_lines = []
    for i in range(MAX_SATS):
        cl, = ax_cfo.plot([], [], "-", color=sat_colors[i % len(sat_colors)],
                           lw=1.5, label=f"S{i} fd")
        cfo_lines.append(cl)
    ax_cfo.legend(loc="upper left", fontsize=6, facecolor=BG3,
                   edgecolor=C_BDR, labelcolor=C_TEXT)

    # ── [1,2]  Phase diffs ─────────────────────────────────────────────────────
    ax_ph = fig.add_subplot(gs[1, 2], facecolor=BG2)
    ax_ph.set_facecolor(BG2)
    ax_ph.set_title("ΔΦ  CH1..4 – CH0", color=C_TEXT, fontsize=9)
    ax_ph.set_xlabel("Burst recenti →", color=C_MUT, fontsize=8)
    ax_ph.set_ylabel("ΔΦ [°]", color=C_MUT, fontsize=8)
    ax_ph.set_xlim(0, H); ax_ph.set_ylim(-185, 185)
    ax_ph.axhline(0, color=C_BDR, lw=0.6)
    ax_ph.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_ph.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_ph.grid(color=C_BDR, lw=0.4, alpha=0.4)
    _ph_colors = [C_BLUE, C_TEAL, C_AMBER, C_VIO]
    ph_lines = [ax_ph.plot([], [], "-", color=_ph_colors[i], lw=1.2,
                             label=f"ΔΦ CH{i+1}")[0] for i in range(4)]
    ax_ph.legend(loc="upper left", fontsize=6, facecolor=BG3,
                  edgecolor=C_BDR, labelcolor=C_TEXT)

    fig.suptitle(
        f"KrakenSDR UCA {n_ant}-ant CW RHCP — {algo.upper()} @ {freq_hz/1e6:.3f} MHz  "
        f"(IRA +{_PREAMBLE_TONE_HZ} Hz  r={cfg.radius_lambda:.4f}λ  "
        f"el=[{el_min:.0f}°,{el_max:.0f}°]  max {MAX_SATS} sat)",
        color=C_TEXT, fontsize=8, y=0.98,
    )

    def _update(_):
        with S.lock:
            sats_snap = {sid: SimpleNamespace(
                az_deg=t.az_deg, el_deg=t.el_deg,
                az_hist=list(t.az_hist), el_hist=list(t.el_hist),
                cfo_hist=list(t.cfo_hist), cfo_hz=t.cfo_hz,
                papr_db=t.papr_db, snr_db=t.snr_db,
                no_doa=t.no_doa, color=t.color,
            ) for sid, t in S.satellites.items()}
            spec2d   = S.spec2d.copy()
            az_spec  = S.az_spec.copy()
            eig      = S.eig_db.copy()
            ph_hist  = [list(q) for q in S.phase_hist]
            no_sig   = S.no_signal
            n_burst  = S.burst_n
            enrg_h   = list(S.energy_hist)
            crb_snap = float(S.crb_az_deg)
            mdl_k_snap = int(S.mdl_k)

        sats_sorted = sorted(sats_snap.values(), key=lambda t: 0 if not t.no_doa else 1)

        # ── skyplot spec fill ─────────────────────────────────────────────────
        _lin = 10 ** (np.clip(az_spec, -40, 0) / 10.0)
        _lin /= _lin.max() + 1e-12
        _r   = _lin * 80.0
        _az_ext = np.r_[az_rad, az_rad[0]]
        _r_ext  = np.r_[_r, _r[0]]
        sky_spec_line.set_data(_az_ext, _r_ext)
        sky_spec_fill.set_xy(np.column_stack([_az_ext, _r_ext]))

        # ── dotplot per satellite ─────────────────────────────────────────────
        has_any = False
        for i, dot in enumerate(sky_dots):
            if i < len(sats_sorted) and not sats_sorted[i].no_doa:
                t   = sats_sorted[i]
                r   = float(np.clip(90.0 - t.el_deg, 0, 90))
                th  = np.deg2rad(t.az_deg)
                dot.set_data([th], [r])
                dot.set_visible(True)
                sky_texts[i].set_text(
                    f"S{i}\naz={t.az_deg:.0f}°\nel={t.el_deg:.0f}°\n"
                    f"fd={t.cfo_hz/1e3:+.1f}k"
                )
                # posiziona label leggermente spostata
                sky_texts[i].set_position((th + 0.15, r + 6))
                sky_texts[i].set_ha("left")
                sky_texts[i].set_va("bottom")
                sky_texts[i].set_visible(True)
                sky_texts[i].set_transform(ax_sky.transData)
                has_any = True
            else:
                dot.set_visible(False)
                sky_texts[i].set_visible(False)
        lbl_nosig.set_visible(not has_any)

        # ── heatmap 2D ────────────────────────────────────────────────────────
        hm_img.set_data(spec2d)
        hm_img.set_clim(-40, 0)
        for i, pk in enumerate(hm_peaks):
            if i < len(sats_sorted) and not sats_sorted[i].no_doa:
                t = sats_sorted[i]
                pk.set_data([t.az_deg], [t.el_deg])
                pk.set_visible(True)
            else:
                pk.set_visible(False)

        # ── az/el history ─────────────────────────────────────────────────────
        for i, (la, le) in enumerate(zip(hist_az_lines, hist_el_lines)):
            if i < len(sats_sorted) and not sats_sorted[i].no_doa:
                t  = sats_sorted[i]
                xa = np.arange(len(t.az_hist))
                xe = np.arange(len(t.el_hist))
                la.set_data(xa, t.az_hist)
                le.set_data(xe, t.el_hist)
            else:
                la.set_data([], [])
                le.set_data([], [])

        # ── eigenvalue bars ───────────────────────────────────────────────────
        if len(eig) == n_ant:
            for bar, v in zip(eig_bars, eig):
                bar.set_height(float(np.clip(v, 0, 42)))
        n_active = len([t for t in sats_snap.values() if not t.no_doa])
        snr0 = sats_sorted[0].snr_db if sats_sorted else 0.0
        txt_info.set_text(f"SNR={snr0:.1f} dB  K̂={mdl_k_snap}")
        # CRB: Cramér-Rao Bound for azimuth (Salama 2025 §8.2.1)
        if crb_snap < 100.0:
            txt_crb.set_text(f"CRB_az ≥ {crb_snap:.2f}°  (Salama §8.2.1)")
        else:
            txt_crb.set_text("")
        txt_sats.set_text(f"satelliti attivi: {n_active}/{MAX_SATS}")
        txt_bursts.set_text(f"burst totali: {n_burst}")

        # ── CFO per satellite ─────────────────────────────────────────────────
        for i, cl in enumerate(cfo_lines):
            if i < len(sats_sorted) and sats_sorted[i].cfo_hist:
                ch = sats_sorted[i].cfo_hist
                cl.set_data(np.arange(len(ch)), list(ch))
            else:
                cl.set_data([], [])

        # ── phase diffs ───────────────────────────────────────────────────────
        for i, (line, q) in enumerate(zip(ph_lines, ph_hist)):
            line.set_data(np.arange(len(q)), q)

        return (sky_spec_line, sky_spec_fill,
                *sky_dots, *sky_texts, lbl_nosig,
                hm_img, *hm_peaks,
                *hist_az_lines, *hist_el_lines,
                *eig_bars, txt_info, txt_crb, txt_sats, txt_bursts,
                *cfo_lines, *ph_lines)

    ani = animation.FuncAnimation(  # noqa: F841
        fig, _update,
        interval=C.UPDATE_INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
    )
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        S.running = False


# =============================================================================
# Auto-calibrazione (usa il satellite con sat_id=0, il primo rilevato)
# =============================================================================

def _run_calibration(
    acc: CovarianceAccumulatorUca, S: SimpleNamespace,
    cfg: UcaConfig, known_az_deg: float,
) -> None:
    import datetime, re
    print(f"\n[CAL] Aspetto convergenza EMA (TX az={known_az_deg:.1f}°)…")
    t0 = time.time()
    while time.time() - t0 < 120.0:
        time.sleep(1.0)
        if not S.running:
            break
        if acc.is_warm and acc.R is not None:
            break
        print(f"  [{time.time()-t0:5.1f}s]  {acc.n_updates} burst validi", end="\r")
    print()
    if acc.R is None:
        print("[CAL] Nessun burst IRA valido. Controlla TX / Heimdall."); return

    ev, V = np.linalg.eigh(acc.R)
    v = V[:, -1]
    v = v * np.exp(-1j * np.angle(v[0]))

    az_rad = np.deg2rad(known_az_deg)
    pos    = cfg.positions
    tau    = 2*np.pi*(pos[:,0]*np.sin(az_rad) + pos[:,1]*np.cos(az_rad))
    tau   -= tau[0]

    hw_offsets = np.degrees(np.angle(v) - tau)
    hw_offsets = (hw_offsets + 180) % 360 - 180
    hw_offsets[0] = 0.0

    print(f"\n[CAL] Offsets hardware da {acc.n_updates} burst:")
    print(f"  ┌─ Copia in config.py ──────────────────────────────────────────")
    print(f"  │  CHANNEL_PHASE_OFFSETS_DEG = {hw_offsets.round(2).tolist()}")
    print(f"  └───────────────────────────────────────────────────────────────")

    cfg_path = os.path.join(_HERE, "config.py")
    try:
        with open(cfg_path) as f: txt = f.read()
        if "CHANNEL_PHASE_OFFSETS_DEG" in txt:
            new_val = f"CHANNEL_PHASE_OFFSETS_DEG = {hw_offsets.round(2).tolist()}"
            new_cmt = (f"  # auto-cal {datetime.datetime.now():%Y-%m-%d %H:%M} "
                       f"da {acc.n_updates} burst az={known_az_deg:.1f}°")
            txt2 = re.sub(
                r"CHANNEL_PHASE_OFFSETS_DEG = \[.*?\]",
                new_val + new_cmt, txt,
            )
            if txt2 != txt:
                with open(cfg_path, "w") as f: f.write(txt2)
                print("[CAL] config.py aggiornato. Riavviare per applicare.")
    except Exception as e:
        print(f"[CAL] Aggiornamento fallito ({e}) — modifica config.py manualmente.")


# =============================================================================
# Recording
# =============================================================================

def _save_recording(S: SimpleNamespace, out_dir: str, tag: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"doa_iridium_{ts}{tag}")
    payload = dict(
        t        = np.array(S.rec_t),
        az_deg   = np.array(S.rec_az),
        el_deg   = np.array(S.rec_el),
        papr_db  = np.array(S.rec_papr),
        snr_db   = np.array(S.rec_snr),
        eig_db   = np.array(S.rec_eig),
        phase_diff = np.array(S.rec_phase),
        R        = np.array(S.rec_R),
        has_sig  = np.array(S.rec_has_sig),
        sat_cfo_hz = np.array(S.rec_sat_cfo),
        # Cramér-Rao Bound and MDL source count per burst (Salama 2025 §8.2.1, §4.2.4)
        crb_az_deg = np.array(S.rec_crb),
        mdl_k      = np.array(S.rec_mdl_k),
        freq_hz  = np.array([C.FREQ_HZ]),
    )
    np.savez_compressed(base + ".npz", **payload)
    print(f"[REC] Salvato {base}.npz  ({len(S.rec_t)} burst)")
    if S.rec_iq_enabled and S.rec_X:
        np.savez_compressed(base + "_iq.npz", X=np.array(S.rec_X))
        print(f"[REC] Salvato {base}_iq.npz  ({len(S.rec_X)} finestre IQ)")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            f"DoA 2D multi-satellite su burst IRA Iridium "
            f"(tono preambolo +{_PREAMBLE_TONE_HZ} Hz) — KrakenSDR UCA CW RHCP"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq",    type=float, default=C.FREQ_HZ / 1e6,
                   help="Frequenza RF centrale [MHz]")
    p.add_argument("--gain",    type=float, default=C.GAIN_DB)
    p.add_argument("--radius",  type=float, default=C.RADIUS_LAMBDA,
                   help="Raggio UCA in lunghezze d'onda")
    p.add_argument("--offset",  type=float, default=C.ANT0_OFFSET_DEG,
                   help="Offset antenna-0 rispetto al Nord [°]")
    p.add_argument("--algo",
                   choices=["music","capon","bartlett",
                             "root-music","unitary-esprit","mfba-music"],
                   default=C.DOA_ALGORITHM.lower())
    p.add_argument("--nsig",    type=int,   default=C.NUM_SIGNALS,
                   help="Sorgenti attese per satellite (MUSIC subspace D)")
    p.add_argument("--alpha",   type=float, default=C.COV_ALPHA,
                   help="EMA covarianza tra burst (0=niente memoria)")
    p.add_argument("--demo",    action="store_true",
                   help="Simulazione sintetica (nessun hardware)")
    p.add_argument("--n-demo-sats", type=int, default=2, choices=[1,2,3],
                   help="Numero di satelliti simulati in modalità --demo")
    p.add_argument("--max-sats",type=int,   default=int(getattr(C,"MAX_SATELLITES",3)),
                   help="Max satelliti tracciati (sovrascrive config)")
    p.add_argument("--papr-min",type=float,
                   default=float(getattr(C,"PAPR_INST_MIN_DB",_PAPR_INST_MIN_DB)),
                   help="PAPR minimo [dB] per accettare un burst")
    p.add_argument("--multi",   type=int,
                   default=int(getattr(C,"MULTI_BURST_N",3)),
                   help="Burst da mediare per DoA")
    p.add_argument("--calibrate", type=float, default=None, metavar="AZ_DEG",
                   help="Auto-calibra offset HW con TX noto a az=AZ_DEG")
    p.add_argument("--out-dir", default=_DATA_DIR, metavar="DIR")
    p.add_argument("--no-rec",  action="store_true")
    p.add_argument("--save-iq", action="store_true")
    args = p.parse_args()

    freq_hz = int(args.freq * 1e6)
    C.FREQ_HZ          = freq_hz
    C.PAPR_INST_MIN_DB = args.papr_min
    C.MULTI_BURST_N    = args.multi
    C.MAX_SATELLITES   = args.max_sats

    cfg = UcaConfig(
        n_ant=C.N_ANTENNAS, radius_lambda=args.radius,
        n_az=C.N_AZ, n_el=C.N_EL,
        el_min_deg=C.EL_MIN_DEG,
        el_max_deg=float(getattr(C, "EL_MAX_DEG", 90.0)),
        num_expected_signals=args.nsig,
        ant0_offset_deg=args.offset,
        ant_ccw=C.ANT_CCW,   # False = CW (orario)
    )
    acc = CovarianceAccumulatorUca(alpha=args.alpha)
    S   = _make_state(cfg.n_az, cfg.n_el)
    S.rec_enabled    = not args.no_rec
    S.rec_iq_enabled = args.save_iq

    ccw_str = "CCW" if C.ANT_CCW else "CW (orario)"
    print("=" * 66)
    print(f"  DoA IRIDIUM MULTI-SATELLITE  —  {args.algo.upper()}  @  {freq_hz/1e6:.3f} MHz")
    if freq_hz < 1_000_000_000:
        print(f"  [MODO] Indoor lab ISM  — TX LibreSDR richiesto")
    else:
        print(f"  [MODO] Real Iridium outdoor  (solo RX)")
    print(f"  UCA: {cfg.n_ant} ant  {ccw_str}  r={args.radius:.4f}λ  offset={args.offset:.1f}°  RHCP")
    print(f"  Heimdall: {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}")
    print(f"  Tono preambolo: +{_PREAMBLE_TONE_HZ} Hz  |  burst: {_BURST_SYMS} sym  |  SF: {int(_SUPERFRAME_S*1000)} ms")
    print(f"  Max satelliti: {args.max_sats}  "
          f"multi={args.multi}  papr_min={args.papr_min:.0f} dB")
    scan_bw = int(getattr(C, "DOPPLER_SCAN_BW_HZ", 45_000))
    sep_hz  = int(getattr(C, "SAT_MIN_SEP_HZ", 5_000))
    print(f"  Doppler scan: ±{scan_bw/1e3:.0f} kHz  separazione min: {sep_hz/1e3:.0f} kHz")
    phase_offs = getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0]*cfg.n_ant)
    if any(o != 0.0 for o in phase_offs):
        print(f"  HW phase cal: {[f'{o:.1f}' for o in phase_offs]} °")
    else:
        print("  HW phase cal: non calibrato — eseguire --calibrate <az_deg>")
    print("=" * 66)
    _check_narrowband(freq_hz, cfg)

    if not args.demo:
        if not _check_heimdall(C.HEIMDALL_HOST, C.HEIMDALL_PORT):
            print(f"\n[ERRORE] Heimdall DAQ non raggiungibile a "
                  f"{C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}.")
            print("  Avviare prima Heimdall (task 'Heimdall: Start'), poi riprovare.")
            print("  Per test offline: aggiungere --demo")
            sys.exit(1)
        src = KrakenIQSource(C.HEIMDALL_HOST, C.HEIMDALL_PORT, C.N_ANTENNAS)
    else:
        src        = None
        demo_params = _DEMO_SATS[:args.n_demo_sats]
        print(f"[DEMO] {args.n_demo_sats} satelliti sintetici attivi:")
        for i, d in enumerate(demo_params):
            print(f"  S{i}: az={d['az']:.0f}°  el={d['el']:.0f}°  "
                  f"fd={d['doppler']:+.0f} Hz  SNR={d['snr_db']:.0f} dB  "
                  f"({_SAT_COLORS[i]})")
        print()

    if args.calibrate is not None:
        acq_thread = threading.Thread(
            target=_acq_loop,
            args=(src, cfg, args.algo, acc, S, args.demo,
                  _DEMO_SATS[:1] if args.demo else None),
            daemon=True,
        )
        acq_thread.start()
        try:
            _run_calibration(acc, S, cfg, args.calibrate)
        finally:
            S.running = False
        return

    def _auto_save_loop() -> None:
        last_n = 0
        while S.running:
            time.sleep(5.0)
            with S.lock:
                n = len(S.rec_t)
            if n - last_n >= 200:
                with S.lock:
                    snap = SimpleNamespace(**vars(S))
                _save_recording(snap, args.out_dir, tag="_autosave")
                last_n = n

    demo_params_arg = _DEMO_SATS[:args.n_demo_sats] if args.demo else None
    acq_thread = threading.Thread(
        target=_acq_loop,
        args=(src, cfg, args.algo, acc, S, args.demo, demo_params_arg),
        daemon=True,
    )
    acq_thread.start()
    if S.rec_enabled and args.out_dir:
        threading.Thread(target=_auto_save_loop, daemon=True).start()

    try:
        _build_ui(S, cfg, args.algo, freq_hz)
    except KeyboardInterrupt:
        pass
    finally:
        S.running = False
        if S.rec_enabled and S.rec_t and args.out_dir:
            with S.lock:
                snap = SimpleNamespace(**vars(S))
            _save_recording(snap, args.out_dir)


if __name__ == "__main__":
    main()
