#!/usr/bin/env python3
"""
doa_test_868_burst.py — Burst-gated 2D DoA at 868 MHz on KrakenSDR 5-element UCA
==================================================================================
Designed for use with LibreSDR tx_868_gui.py in BURST mode (π/4-DQPSK IRA).

Burst detection & pilot extraction
-----------------------------------
- LibreSDR transmits one IRA TDMA slot every 90 ms (SUPERFRAME period).
- The 64-symbol preamble uses all-zero dibits → constant +π/4 rotation per
  symbol → pure tone at carrier + Rs/8 = carrier + 3125 Hz.
- This file detects each burst using a short-window energy gate, extracts
  the preamble portion, and runs extract_pilot_tone() at 3125 Hz.
- DoA covariance is computed fresh per burst → no history contamination.
- EMA smoothing is applied across successive burst estimates only.

Key parameters (matched to LibreSDR realistic_sim.py)
-------------------------------------------------------
  Symbol rate   : 25 000 sps
  Burst symbols : 245  (pre + UW + data + tail)
  Preamble syms : 64    → tone at +3125 Hz
  SUPERFRAME    : 90 ms (one burst per slot)
  TX sample rate: 1 000 000 Hz (4× upsampled from 250 kHz base)
  RX sample rate: 1 024 000 Hz (Heimdall DAQ)

Usage
-----
    python3 doa_test_868_burst.py              # real hardware
    python3 doa_test_868_burst.py --demo       # synthetic IRA at 45° (no HW)
    python3 doa_test_868_burst.py --algo capon
    python3 doa_test_868_burst.py --out-dir /tmp/doa_burst
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
_DATA_DIR = os.path.normpath(os.path.join(_SRC, "..", "data", "doa_868"))
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
    enhanced_preprocessing as enhanced_preprocessing_doa,
)
from core.doa_algorithms import apply_phase_correction as _apply_phase_correction

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
C_BDR = "#3b4263"; C_MUT = "#8891b0"; C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIO  = "#a78bfa"; C_ROSE = "#f16b6f"; C_LIME  = "#6dd97d"

# ── IRA burst parameters (from libreSDR/src/iridium/realistic_sim.py) ─────────
_SYMBOL_RATE     = 25_000          # symbols/s
_SPS_TX          = 10              # samples/symbol at base rate (250 kHz)
_IRA_UPS         = 4               # TX upsampling to 1 MHz
_IRA_SAMPLE_RATE = _SYMBOL_RATE * _SPS_TX        # 250 000 Hz
_TX_SAMPLE_RATE  = _IRA_SAMPLE_RATE * _IRA_UPS   # 1 000 000 Hz

_PREAMBLE_SYMS   = 64
_BURST_SYMS      = 245
_SUPERFRAME_S    = 0.090           # 90 ms between bursts

# Preamble tone: all-zero dibits → +π/4 per symbol = Rs/8 = 3125 Hz
_PREAMBLE_TONE_HZ = _SYMBOL_RATE // 8   # 3125 Hz

# Heimdall DAQ sample rate (from config)
_FS = float(getattr(C, "SAMPLE_RATE_HZ", 1_024_000))

# Burst duration in Heimdall samples (accounting for 1 MHz TX → 1.024 MHz RX)
_BURST_SAMPLES  = int(round(_BURST_SYMS * _SPS_TX * _IRA_UPS * _FS / _TX_SAMPLE_RATE))
_PRE_SAMPLES    = int(round(_PREAMBLE_SYMS * _SPS_TX * _IRA_UPS * _FS / _TX_SAMPLE_RATE))
_SF_SAMPLES     = int(round(_SUPERFRAME_S * _FS))   # superframe gap (min. burst spacing)

# Burst extraction window: a bit wider than the burst to catch the preamble
_WINDOW_SAMPLES  = min(_BURST_SAMPLES + 512, _SF_SAMPLES - 256)

# Energy detector: integrate over this many samples
_ENERGY_WIN      = 256
# Tone-scan window for preamble-onset localisation
_TONE_SCAN_WIN   = 512   # points per FFT block (≈0.5 ms @ 1.024 MHz)
# PAPR and display smoothing
# Lowered from 3.0 → 2.0 (data-driven: burst session 20260428 shows PAPR≈4.4 dB
# when preamble correctly aligned; 2.0 is a safe floor at –20 dB TX gain).
_PAPR_MIN_DB     = 2.0
_PAPR_FLAT_DB    = 2.0
# Instantaneous MUSIC PAPR threshold for EMA gate.
# Genuine preamble window (calibrated array):   papr_inst ≥ 29 dB  (SNR 0-10 dB).
# With typical hardware phase errors ≤ ±15°:    papr_inst ≥ 14.6 dB
# DATA section / noise:                         papr_inst ≤ 12.1 dB
# Threshold 12 dB leaves ≥ 2.5 dB margin above DATA ceiling while still accepting
# preamble captures with up to ±20° hardware phase errors (min papr = 12.4 dB).
# After hardware calibration (see CHANNEL_PHASE_OFFSETS_DEG in config.py), raise
# back to 20 dB for a cleaner gate.
# NOTE: soglia dipende dall'algoritmo:
#   MUSIC/Capon:    12 dB (super-resolution → picchi molto netti)
#   Bartlett:       2 dB  (beamformer convenzionale → picchi più larghi)
_PAPR_INST_MIN_DB_MUSIC    = 12.0
_PAPR_INST_MIN_DB_BARTLETT = 2.0
_SPEC_EMA                  = 0.20  # faster update than CW: each burst is ~90ms


def _check_heimdall(host: str, port: int) -> bool:
    try:
        s = _socket.create_connection((host, port), timeout=2.0)
        s.close()
        return True
    except OSError:
        return False


def _circ_median(angles_deg: np.ndarray) -> float:
    """Circular median of azimuth values in [0..360°]."""
    if len(angles_deg) == 0:
        return 0.0
    a = np.deg2rad(angles_deg)
    mean_ang = np.angle(np.mean(np.exp(1j * a)))
    centred   = np.degrees(np.angle(np.exp(1j * (a - mean_ang))))
    return float((np.median(centred) + np.degrees(mean_ang)) % 360)


def _make_state(n_az: int, n_el: int) -> SimpleNamespace:
    return SimpleNamespace(
        az_spec    = np.full(n_az, -40.0),
        spec2d     = np.full((n_el, n_az), -40.0),
        az_deg     = 0.0,
        az_median  = 0.0,
        el_deg     = 0.0,
        phase_diffs= np.zeros(4),
        papr_db    = 0.0,
        snr_db     = 0.0,
        eig_db     = np.zeros(5),
        burst_n    = 0,        # total bursts detected since start
        frame_n    = 0,        # total CPI frames consumed
        az_hist    = collections.deque(maxlen=C.HISTORY_LEN),
        el_hist    = collections.deque(maxlen=C.HISTORY_LEN),
        snr_hist   = collections.deque(maxlen=C.HISTORY_LEN),
        phase_hist = [collections.deque(maxlen=C.HISTORY_LEN) for _ in range(4)],
        energy_hist= collections.deque(maxlen=C.HISTORY_LEN),
        az_phasor  = np.exp(0j),  # circular EMA phasor for az smoothing
        no_signal  = True,
        no_doa     = True,
        lock       = threading.Lock(),
        running    = True,
        rec_enabled= True,
        rec_t       = [],
        rec_az      = [],
        rec_el      = [],
        rec_papr    = [],
        rec_snr     = [],
        rec_eig     = [],
        rec_phase   = [],
        rec_R       = [],
        rec_has_sig = [],
    )


# =============================================================================
# Burst detection helper
# =============================================================================

def _detect_bursts(iq: np.ndarray, threshold_factor: float = 6.0) -> list[int]:
    """
    Return start-sample indices of energy bursts in a 1D IQ stream.

    Strategy:
    - Compute short-window RMS power (non-overlapping blocks of _ENERGY_WIN).
    - Background level = median of all blocks.
    - A block is "active" when power > threshold_factor × background.
    - Merge contiguous active blocks; return the start of the first active block.
    - Enforce minimum gap of _SF_SAMPLES / 2 between detections.
    """
    n_blocks = len(iq) // _ENERGY_WIN
    if n_blocks == 0:
        return []

    pwr = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        seg = iq[i * _ENERGY_WIN: (i + 1) * _ENERGY_WIN]
        pwr[i] = float(np.mean(np.abs(seg) ** 2))

    noise_floor = float(np.median(pwr)) + 1e-20
    active = pwr > threshold_factor * noise_floor

    # Find rising edges (0→1 transitions)
    edges = np.diff(active.astype(np.int8), prepend=0)
    starts_blk = np.where(edges > 0)[0]

    # Convert to sample indices; enforce minimum spacing
    min_gap_blk = max(1, _SF_SAMPLES // 2 // _ENERGY_WIN)
    detections: list[int] = []
    last_blk = -min_gap_blk - 1
    for blk in starts_blk:
        if blk - last_blk >= min_gap_blk:
            detections.append(int(blk * _ENERGY_WIN))
            last_blk = blk
    return detections


# =============================================================================
# Preamble-tone onset localisation
# =============================================================================

def _find_tone_onset(
    iq:      np.ndarray,     # channel-0 samples, 1D complex
    b_start: int,            # energy-detector burst onset (sample index)
    n_total: int,            # total samples available in iq
    tone_hz: float = float(_PREAMBLE_TONE_HZ),
    fs:      float = _FS,
    win:     int   = _TONE_SCAN_WIN,
) -> int:
    """
    Scan a neighbourhood around the energy-detector onset and return the
    sample position with the highest FFT power at ``tone_hz``.

    Why: the IRA slot is  Guard(silence) → Preamble(tone@3125 Hz) → Data(wideband).
    The wideband energy detector can fire at ANY point in the active slot.  If
    it fires on the DATA portion the fixed extraction ``X[:, :_PRE_SAMPLES]``
    captures wideband signal → near-flat MUSIC spectrum → low PAPR → rejected.

    This function searches [b_start − _BURST_SAMPLES, b_start + _PRE_SAMPLES]
    (≈ one full IRA slot backwards + one preamble forwards) so the preamble is
    found regardless of when the detector fired.  The 3125 Hz tone power in a
    512-sample FFT block is ~17 dB above wideband data floor, making the
    distinction reliable even at SNR ≈ 0 dB.
    """
    # Matched-filter template: optimally detects a pure sinusoid at tone_hz
    # regardless of FFT bin alignment.  At 1.024 MHz, 3125 Hz falls between
    # bins 1 (2000 Hz) and 2 (4000 Hz) for win=512 → the closest bin holds
    # only ~56% of the tone energy.  The matched filter captures 100%.
    t_arr    = np.arange(win, dtype=np.float64)
    template = np.exp(2j * np.pi * tone_hz / fs * t_arr)

    step     = win // 2                              # 50 % overlap
    scan_sta = max(0, b_start - _BURST_SAMPLES)      # look back up to one full burst
    scan_end = min(b_start + _PRE_SAMPLES, n_total - win)

    if scan_end <= scan_sta:
        return b_start   # not enough context — fall back to raw onset

    best_pwr = -1.0
    best_pos =  b_start
    for pos in range(scan_sta, scan_end, step):
        seg = iq[pos: pos + win]
        pwr = float(abs(np.dot(np.conj(seg), template)) ** 2)
        if pwr > best_pwr:
            best_pwr = pwr
            best_pos = pos

    # Shift slightly back so the full preamble run-up is included.
    return max(scan_sta, best_pos - win // 4)


# =============================================================================
# Acquisition + DoA thread
# =============================================================================

def _circ_distance_burst(a_deg: float, b_deg: float) -> float:
    """Shortest angular distance between two azimuths [0..360°]."""
    d = abs(a_deg - b_deg) % 360.0
    return min(d, 360.0 - d)


def _select_algo_burst(user_algo: str, snr_db: float) -> str:
    """SNR-adaptive algorithm selection for burst mode."""
    if not getattr(C, "SNR_ADAPTIVE_ENABLED", False):
        return user_algo
    snr_high = getattr(C, "SNR_HIGH_DB", 10.0)
    snr_low  = getattr(C, "SNR_LOW_DB", 4.0)
    if snr_db >= snr_high:
        return user_algo
    elif snr_db <= snr_low:
        return "bartlett"
    else:
        return "capon" if user_algo != "bartlett" else "bartlett"


def _acq_loop(
    src, cfg: UcaConfig, algo: str,
    acc: CovarianceAccumulatorUca, S: SimpleNamespace,
    demo: bool = False,
) -> None:
    rng     = np.random.default_rng(42)
    pos     = cfg.positions
    _fs     = _FS

    _pilot_bw    = float(getattr(C, "PILOT_TONE_BW_HZ", 5_000))
    _amp_norm    = getattr(C, "AMPLITUDE_NORMALIZE", True)

    _phase_offs  = list(getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * cfg.n_ant))
    _phase_offs  = (_phase_offs + [0.0] * cfg.n_ant)[: cfg.n_ant]
    _has_cal     = any(o != 0.0 for o in _phase_offs)
    _az_alpha    = float(getattr(C, "AZ_SMOOTH_ALPHA", 0.50))

    _no_burst_streak = 0

    # Multi-burst covariance accumulation
    _multi_n       = max(1, getattr(C, "MULTI_BURST_N", 1))
    _R_accum       = None
    _accum_count   = 0
    _use_ema_snr   = getattr(C, "USE_EMA_FOR_DOA_BELOW_SNR", 6.0)

    # Phase coherence gating
    _phase_coh_en  = getattr(C, "PHASE_COHERENCE_ENABLED", False)
    _phase_max_jmp = getattr(C, "PHASE_COHERENCE_MAX_JUMP_DEG", 60.0)
    _ph_coh_hist   = [collections.deque(maxlen=20) for _ in range(cfg.n_ant - 1)]

    # Az outlier rejection
    _az_outlier_en    = getattr(C, "AZ_OUTLIER_ENABLED", False)
    _az_outlier_max   = getattr(C, "AZ_OUTLIER_MAX_DEV_DEG", 45.0)
    _az_outlier_min_n = getattr(C, "AZ_OUTLIER_MIN_HISTORY", 5)

    _buf: list[np.ndarray] = []
    _buf_len = 0
    _min_buf = _SF_SAMPLES + _WINDOW_SAMPLES + 2048

    while S.running:
        # ── Get IQ frame ────────────────────────────────────────────────────
        if demo:
            demo_az = np.deg2rad(45.0)
            demo_el = np.deg2rad(10.0)
            gain_off  = np.array([1.0, 0.92, 1.08, 0.95, 1.03])
            phase_off = np.deg2rad([0.0, 5.0, -8.0, 12.0, -3.0])
            tau = 2 * np.pi * (pos[:, 0] * np.cos(demo_el) * np.sin(demo_az)
                               + pos[:, 1] * np.cos(demo_el) * np.cos(demo_az))

            N_frame = _SF_SAMPLES
            t = np.arange(N_frame, dtype=np.float64)
            burst_start = 512
            preamble_tone = np.exp(2j * np.pi * _PREAMBLE_TONE_HZ / _fs * t)

            X = np.zeros((cfg.n_ant, N_frame), dtype=np.complex128)
            snr_lin = 10 ** (18.0 / 10.0)
            for k in range(cfg.n_ant):
                channel_phase = tau[k] + phase_off[k]
                X[k, burst_start: burst_start + _PRE_SAMPLES] += (
                    gain_off[k] * np.exp(1j * channel_phase)
                    * preamble_tone[burst_start: burst_start + _PRE_SAMPLES]
                    * np.sqrt(snr_lin)
                )
            X += ((rng.standard_normal((cfg.n_ant, N_frame))
                   + 1j * rng.standard_normal((cfg.n_ant, N_frame))) / np.sqrt(2))

            frame = X
            time.sleep(_SUPERFRAME_S)
        else:
            frame = src.get_frame(timeout=2.0)
            if frame is None:
                continue
            frame = frame.astype(np.complex128)

        if frame.shape[0] != cfg.n_ant:
            continue

        with S.lock:
            S.frame_n += 1

        _buf.append(frame)
        _buf_len += frame.shape[1]

        if _buf_len < _min_buf:
            continue

        X_stream = np.concatenate(_buf, axis=1)
        n_total  = X_stream.shape[1]

        _keep = _WINDOW_SAMPLES + 512
        if n_total > _keep:
            _buf = [X_stream[:, -_keep:]]
            _buf_len = _keep
        else:
            _buf = [X_stream]
            _buf_len = n_total

        bursts = _detect_bursts(X_stream[0])

        pwr = float(np.mean(np.abs(X_stream) ** 2))
        pwr_db = 10 * np.log10(pwr + 1e-20)

        if not bursts:
            with S.lock:
                S.energy_hist.append(pwr_db)
                S.no_signal = True
            _no_burst_streak += 1
            if _no_burst_streak == 10:
                print("[WARN] 10 frames without burst — check TX is in IRA/BURST mode")
            continue

        _no_burst_streak = 0

        for b_start in bursts:
            b_end = min(b_start + _WINDOW_SAMPLES, n_total)
            if b_end - b_start < _PRE_SAMPLES:
                continue

            tone_start = _find_tone_onset(X_stream[0], b_start, n_total)
            tone_end = min(tone_start + _PRE_SAMPLES, n_total)
            if tone_end - tone_start < _PRE_SAMPLES // 2:
                continue

            X_pre = X_stream[:, tone_start: tone_end]
            pwr_db = float(10 * np.log10(np.mean(np.abs(X_pre) ** 2) + 1e-20))

            # ── Pilot tone extraction ─────────────────────────────────────────
            if C.PILOT_TONE_ENABLED:
                X_proc = extract_pilot_tone(
                    X_pre, C.SAMPLE_RATE_HZ, C.PILOT_TONE_OFFSET_HZ, C.PILOT_TONE_BW_HZ
                )
            else:
                X_proc = X_pre

            if C.AMPLITUDE_NORMALIZE:
                X_proc = amplitude_normalize_channels(X_proc)

            if getattr(C, "ENABLE_ENHANCED_PREPROCESSING", False):
                X_proc = enhanced_preprocessing_doa(
                    X_proc, cfg,
                    sample_rate=C.SAMPLE_RATE_HZ,
                    center_freq=C.FREQ_HZ,
                    apply_spatial_smoothing=getattr(C, "APPLY_SPATIAL_SMOOTHING", True),
                    apply_mfba=getattr(C, "APPLY_MFBA", True),
                    apply_adaptive_filtering=getattr(C, "APPLY_ADAPTIVE_FILTERING", False),
                    apply_outlier_rejection=getattr(C, "APPLY_OUTLIER_REJECTION", False)
                )

            if _has_cal:
                X_cal = _apply_phase_correction(X_proc, _phase_offs)
            else:
                X_cal = X_proc

            try:
                R_inst = (X_cal @ X_cal.conj().T) / X_cal.shape[1]
                snr    = snr_uca_db(R_inst)
                eig    = eigenvalue_spread_uca_db(R_inst)

                # Stage 1: eigenspread gate
                if eig[0] < C.EIG_SPREAD_MIN_DB:
                    with S.lock:
                        S.snr_db    = snr
                        S.eig_db    = eig
                        S.no_signal = True
                        S.no_doa    = True
                        S.energy_hist.append(pwr_db)
                        S.snr_hist.append(snr)
                        S.burst_n += 1
                    continue

                # ── Multi-burst covariance accumulation ──────────────────────
                if _R_accum is None:
                    _R_accum = R_inst.copy()
                else:
                    _R_accum += R_inst
                _accum_count += 1

                if _accum_count < _multi_n:
                    with S.lock:
                        S.snr_db    = snr
                        S.eig_db    = eig
                        S.no_signal = False
                        S.no_doa    = True
                        S.energy_hist.append(pwr_db)
                        S.snr_hist.append(snr)
                        S.burst_n += 1
                    continue

                R_multi = _R_accum / _accum_count
                _R_accum = None
                _accum_count = 0

                # At low SNR, prefer R_EMA for DoA
                if snr < _use_ema_snr and acc.is_warm and acc.R is not None:
                    R_doa = acc.R
                else:
                    R_doa = R_multi

                # Stage 2: DoA with SNR-adaptive algorithm
                _use_algo = _select_algo_burst(algo, snr)
                if _use_algo == "capon":
                    spec2d_inst = doa_capon_uca_2d(X_cal, cfg, R_in=R_doa,
                                                    decorr=getattr(C, "CAPNT_DECORR", "none"))
                elif _use_algo == "bartlett":
                    spec2d_inst = doa_bartlett_uca_2d(X_cal, cfg, R_in=R_doa)
                else:
                    spec2d_inst = doa_music_uca_2d(X_cal, cfg, R_in=R_doa,
                                                   decorr=getattr(C, "MUSIC_DECORR", "none"))

                az_inst, el_inst, papr_inst = find_peak_uca_2d(spec2d_inst, cfg)
                _papr_threshold = _PAPR_INST_MIN_DB_BARTLETT if _use_algo == "bartlett" else _PAPR_INST_MIN_DB_MUSIC

                is_preamble = (papr_inst >= _papr_threshold)

                if not is_preamble:
                    R = acc.R if acc.R is not None else R_doa
                    with S.lock:
                        S.snr_db    = snr
                        S.eig_db    = eig
                        S.no_signal = True
                        S.no_doa    = True
                        S.energy_hist.append(pwr_db)
                        S.snr_hist.append(snr)
                        S.burst_n += 1
                    continue

                # ── Valid preamble burst ──────────────────────────────────────
                R = acc.update(X_cal)
                phase_diffs = np.degrees(np.angle(R[1:, 0]))

                # ── Phase coherence gating ────────────────────────────────────
                if _phase_coh_en:
                    _coherent = True
                    for _i, _p in enumerate(phase_diffs):
                        hist = _ph_coh_hist[_i]
                        if len(hist) >= 3:
                            med = _circ_median(np.array(hist))
                            if _circ_distance_burst(float(_p), med) > _phase_max_jmp:
                                _coherent = False
                                break
                    if not _coherent:
                        with S.lock:
                            S.snr_db  = snr
                            S.eig_db  = eig
                            S.no_signal = False
                            S.no_doa    = True
                            S.energy_hist.append(pwr_db)
                            S.snr_hist.append(snr)
                            S.burst_n += 1
                        continue
                    for _i, _p in enumerate(phase_diffs):
                        _ph_coh_hist[_i].append(float(_p))

                # Circular EMA on az angle
                az_ph     = np.exp(1j * np.deg2rad(az_inst))
                S.az_phasor = _az_alpha * S.az_phasor + (1.0 - _az_alpha) * az_ph
                az        = float(np.degrees(np.angle(S.az_phasor)) % 360.0)
                el_est    = el_inst
                spec2d    = spec2d_inst
                az_spec   = np.max(spec2d, axis=0)
                papr      = papr_inst

            except Exception as exc:
                print(f"[DoA] burst #{S.burst_n+1}: {exc}")
                continue

            has_signal = True
            has_doa    = True

            # ── Az outlier rejection ──────────────────────────────────────
            if _az_outlier_en:
                with S.lock:
                    n_hist = len(S.az_hist)
                    az_med = S.az_median
                if n_hist >= _az_outlier_min_n:
                    if _circ_distance_burst(az, az_med) > _az_outlier_max:
                        has_doa = False

            if not has_doa:
                with S.lock:
                    S.snr_db    = snr
                    S.eig_db    = eig
                    S.no_signal = False
                    S.no_doa    = True
                    S.energy_hist.append(pwr_db)
                    S.snr_hist.append(snr)
                    S.burst_n += 1
                continue

            with S.lock:
                S.az_spec     = (1 - _SPEC_EMA) * S.az_spec + _SPEC_EMA * az_spec
                S.spec2d      = (1 - _SPEC_EMA) * S.spec2d  + _SPEC_EMA * spec2d
                S.az_deg      = az
                S.el_deg      = el_est
                S.phase_diffs = phase_diffs
                S.papr_db     = papr
                S.snr_db      = snr
                S.eig_db      = eig
                S.no_signal   = False
                S.no_doa      = False
                S.energy_hist.append(pwr_db)
                S.el_hist.append(el_est)
                S.snr_hist.append(snr)
                for _i, _p in enumerate(phase_diffs):
                    S.phase_hist[_i].append(float(_p))
                S.az_hist.append(az)
                if len(S.az_hist) >= 3:
                    S.az_median = _circ_median(np.array(S.az_hist))
                else:
                    S.az_median = az
                if S.rec_enabled:
                    S.rec_t.append(time.time())
                    S.rec_az.append(float(az))
                    S.rec_el.append(float(el_est))
                    S.rec_papr.append(float(papr))
                    S.rec_snr.append(float(snr))
                    S.rec_eig.append(eig.copy())
                    S.rec_phase.append(phase_diffs.copy())
                    S.rec_R.append(np.array(R, dtype=np.complex128).copy())
                    S.rec_has_sig.append(True)
                S.burst_n += 1


# =============================================================================
# Auto-save
# =============================================================================

def _autosave_loop(
    S: SimpleNamespace, freq_hz: int, algo: str,
    out_dir: str | None, rec_every: int,
) -> None:
    last_n = 0
    while S.running:
        time.sleep(5)
        with S.lock:
            n = len(S.rec_t)
        if n >= last_n + rec_every and S.running:
            _save_recording(S, freq_hz, algo, out_dir=out_dir, label="burst_checkpoint")
            last_n = n


def _save_recording(
    S: SimpleNamespace,
    freq_hz: int,
    algo: str,
    out_dir: str | None = None,
    label: str = "burst_data",
) -> str | None:
    import datetime
    n = len(S.rec_t)
    if n == 0:
        print("[REC] No bursts recorded — file not saved.")
        return None
    if out_dir is None:
        out_dir = _DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{label}_{ts}.npz")
    np.savez_compressed(
        path,
        timestamps  = np.array(S.rec_t,       dtype=np.float64),
        az_deg      = np.array(S.rec_az,       dtype=np.float32),
        el_deg      = np.array(S.rec_el,       dtype=np.float32),
        papr_db     = np.array(S.rec_papr,     dtype=np.float32),
        snr_db      = np.array(S.rec_snr,      dtype=np.float32),
        eig_db      = np.array(S.rec_eig,      dtype=np.float32),
        phase_diffs = np.array(S.rec_phase,    dtype=np.float32),
        R_real      = np.real(np.array(S.rec_R, dtype=np.complex128)),
        R_imag      = np.imag(np.array(S.rec_R, dtype=np.complex128)),
        has_signal  = np.array(S.rec_has_sig,  dtype=bool),
        freq_hz     = np.int64(freq_hz),
        algo        = np.bytes_(algo.encode()),
        burst_mode  = np.bytes_(b"IRA_PREAMBLE"),
        preamble_tone_hz = np.int32(_PREAMBLE_TONE_HZ),
    )
    n_sig = int(np.sum(S.rec_has_sig))
    print(f"[REC] {n} bursts ({n_sig} valid, {n_sig*100//max(n,1)}%) → {path}")
    return path


# =============================================================================
# UI
# =============================================================================

def _build_and_run_ui(S: SimpleNamespace, cfg: UcaConfig,
                      algo: str, freq_hz: int) -> None:
    el_min  = cfg.el_min_deg
    el_max  = 90.0
    az_rad  = np.deg2rad(cfg.az_range_deg())
    n_az    = cfg.n_az
    n_el    = len(cfg.el_range_deg())
    H       = C.HISTORY_LEN

    fig = plt.figure(figsize=(17, 9), facecolor=BG)
    fig.canvas.manager.set_window_title(
        f"DoA BURST 868 MHz — {algo.upper()}  (IRA preamble @ +{_PREAMBLE_TONE_HZ} Hz)")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[1.4, 1.0],
                           left=0.05, right=0.97,
                           top=0.93, bottom=0.07,
                           hspace=0.42, wspace=0.33)

    # ── [0,0]  Polar azimuth compass ─────────────────────────────────────────
    ax_pol = fig.add_subplot(gs[0, 0], projection="polar", facecolor=BG2)
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_ylim(0, 1)
    ax_pol.set_yticks([])
    ax_pol.tick_params(colors=C_MUT, labelsize=7)
    ax_pol.set_facecolor(BG2)
    for sp in ax_pol.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_pol.set_title("Azimuth DoA  (burst)", color=C_TEXT, fontsize=9, pad=10)

    _z = np.zeros(n_az + 1)
    spec_line, = ax_pol.plot(np.r_[az_rad, az_rad[0]], _z,
                              "-", color=C_TEAL, lw=1.2, alpha=0.7)
    ax_pol.fill(np.r_[az_rad, az_rad[0]], _z, color=C_TEAL, alpha=0.12)
    arrow_line, = ax_pol.plot([0, 0], [0, 0.92], "-", color=C_LIME, lw=2.5)
    arrow_dot,  = ax_pol.plot([0], [0.92], "o",  color=C_LIME, ms=8, zorder=6)
    inst_line,  = ax_pol.plot([0, 0], [0, 0.78], "-", color=C_AMBER,
                               lw=1.2, alpha=0.55, zorder=4)
    txt_az = ax_pol.text(0, 0, "—°", ha="center", va="center",
                          color=C_LIME, fontsize=15, fontweight="bold")
    txt_nosig = ax_pol.text(
        0.5, 0.5, "NO BURST", transform=ax_pol.transAxes,
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=C_ROSE, alpha=0.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=C_ROSE, alpha=0.0),
        zorder=10)

    # ── [0,1]  2D az × el map ─────────────────────────────────────────────────
    ax_2d = fig.add_subplot(gs[0, 1], facecolor=BG2)
    ax_2d.set_facecolor(BG2)
    ax_2d.set_xlabel("Azimuth [°]", color=C_MUT, fontsize=8)
    ax_2d.set_ylabel("Elevation [°]", color=C_MUT, fontsize=8)
    ax_2d.set_title("2D spectrum  az × el  (preamble gate)", color=C_TEXT, fontsize=9)
    ax_2d.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_2d.spines.values():
        sp.set_edgecolor(C_BDR)
    im_2d = ax_2d.imshow(
        np.zeros((n_el, n_az)),
        origin="lower", extent=[0, 360, el_min, el_max],
        aspect="auto", cmap="plasma", vmin=0.0, vmax=1.0,
        interpolation="bilinear",
    )
    xh_v, = ax_2d.plot([0, 0],   [el_min, el_max], "--", color=C_LIME, lw=0.9, alpha=0.7)
    xh_h, = ax_2d.plot([0, 360], [el_min, el_min], "--", color=C_LIME, lw=0.9, alpha=0.7)
    peak_dot, = ax_2d.plot([0], [el_min], "o", color=C_LIME, ms=6, zorder=6)

    # ── [0,2]  Eigenvalues + quality ─────────────────────────────────────────
    ax_q = fig.add_subplot(gs[0, 2], facecolor=BG2)
    ax_q.set_facecolor(BG2)
    ax_q.set_title("Eigenvalues + quality", color=C_TEXT, fontsize=9)
    ax_q.set_xlabel("Channel", color=C_MUT, fontsize=8)
    ax_q.set_ylabel("Spread [dB]", color=C_MUT, fontsize=8)
    ax_q.set_xlim(-0.5, 4.5); ax_q.set_xticks(range(5))
    ax_q.set_ylim(-3, 35)
    ax_q.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_q.spines.values():
        sp.set_edgecolor(C_BDR)
    bars = ax_q.bar(range(5), np.zeros(5),
                    color=[C_BLUE, C_TEAL, C_AMBER, C_VIO, C_ROSE],
                    edgecolor=BG2, linewidth=0.5, zorder=3)
    ax_q.axhline(0, color=C_BDR, lw=0.8, zorder=2)
    ax_q.grid(axis="y", color=C_BDR, lw=0.5, alpha=0.4, zorder=1)
    txt_snr   = ax_q.text(2, 32, "SNR: — dB",       ha="center", va="top", color=C_TEXT,  fontsize=9)
    txt_papr  = ax_q.text(2, 29, "PAPR: — dB",      ha="center", va="top", color=C_AMBER, fontsize=9)
    txt_el_q  = ax_q.text(2, 26, "El:  — °",        ha="center", va="top", color=C_TEAL,  fontsize=9)
    txt_burst = ax_q.text(2, 23, "bursts: 0",       ha="center", va="top", color=C_LIME,  fontsize=8)
    txt_frame = ax_q.text(2, 20, "frames: 0",       ha="center", va="top", color=C_MUT,   fontsize=8)
    txt_rec   = ax_q.text(2, 17, "rec: 0",          ha="center", va="top", color=C_ROSE,  fontsize=8)

    # ── [1,0]  Azimuth history ────────────────────────────────────────────────
    ax_az = fig.add_subplot(gs[1, 0], facecolor=BG2)
    ax_az.set_facecolor(BG2)
    ax_az.set_title("Azimuth history  (per burst)", color=C_TEXT, fontsize=9)
    ax_az.set_xlabel("Recent bursts →", color=C_MUT, fontsize=8)
    ax_az.set_ylabel("Az [°]", color=C_MUT, fontsize=8)
    ax_az.set_xlim(0, H); ax_az.set_ylim(0, 360)
    ax_az.set_yticks([0, 90, 180, 270, 360])
    ax_az.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_az.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_az.grid(color=C_BDR, lw=0.4, alpha=0.4)
    az_line,     = ax_az.plot([], [], "-",  color=C_LIME, lw=1.4)
    az_med_line, = ax_az.plot([], [], "--", color=C_LIME, lw=0.8, alpha=0.45)

    # ── [1,1]  Burst energy history ───────────────────────────────────────────
    ax_en = fig.add_subplot(gs[1, 1], facecolor=BG2)
    ax_en.set_facecolor(BG2)
    ax_en.set_title("Preamble power history", color=C_TEXT, fontsize=9)
    ax_en.set_xlabel("Recent bursts →", color=C_MUT, fontsize=8)
    ax_en.set_ylabel("Power [dBFS]", color=C_MUT, fontsize=8)
    ax_en.set_xlim(0, H); ax_en.set_ylim(-80, 0)
    ax_en.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_en.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_en.grid(color=C_BDR, lw=0.4, alpha=0.4)
    en_line, = ax_en.plot([], [], "-", color=C_AMBER, lw=1.4)

    # ── [1,2]  Inter-channel phase differences ────────────────────────────────
    ax_ph = fig.add_subplot(gs[1, 2], facecolor=BG2)
    ax_ph.set_facecolor(BG2)
    ax_ph.set_title("ΔΦ  CH1..4 – CH0  (from covariance matrix)", color=C_TEXT, fontsize=9)
    ax_ph.set_xlabel("Recent bursts →", color=C_MUT, fontsize=8)
    ax_ph.set_ylabel("ΔΦ [°]", color=C_MUT, fontsize=8)
    ax_ph.set_xlim(0, H); ax_ph.set_ylim(-185, 185)
    ax_ph.axhline(0, color=C_BDR, lw=0.6)
    ax_ph.set_yticks([-180, -90, 0, 90, 180])
    ax_ph.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_ph.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_ph.grid(color=C_BDR, lw=0.4, alpha=0.4)
    _ph_colors = [C_BLUE, C_TEAL, C_AMBER, C_VIO]
    ph_lines = [ax_ph.plot([], [], "-", color=_ph_colors[i], lw=1.2,
                            label=f"ΔΦ CH{i+1}−CH0")[0] for i in range(4)]
    ax_ph.legend(loc="upper right", fontsize=6, facecolor=BG3,
                  edgecolor=C_BDR, labelcolor=C_TEXT)

    fig.suptitle(
        f"KrakenSDR UCA 5-ant  —  BURST {algo.upper()}  @  {freq_hz/1e6:.3f} MHz  "
        f"(r={cfg.radius_lambda:.4f}λ  ant0={cfg.ant0_offset_deg:.0f}°  "
        f"preamble tone +{_PREAMBLE_TONE_HZ} Hz  SF={int(_SUPERFRAME_S*1000)} ms)",
        color=C_TEXT, fontsize=8, y=0.98,
    )

    def _update(_):
        with S.lock:
            az_s    = S.az_spec.copy()
            s2d     = S.spec2d.copy()
            az      = S.az_deg
            az_med  = S.az_median
            el      = S.el_deg
            phase   = S.phase_diffs.copy()
            papr    = S.papr_db
            snr     = S.snr_db
            eig     = S.eig_db.copy()
            no_sig  = S.no_signal
            no_doa  = S.no_doa
            bn      = S.burst_n
            fn      = S.frame_n
            n_rec   = len(S.rec_t)
            az_h    = list(S.az_hist)
            en_h    = list(S.energy_hist)
            ph_h    = [list(q) for q in S.phase_hist]

        # Polar compass
        lin = 10 ** (np.clip(az_s, -40, 0) / 10.0)
        lin /= (lin.max() + 1e-12)
        spec_line.set_data(np.r_[az_rad, az_rad[0]], np.r_[lin, lin[0]])
        th_med = np.deg2rad(az_med)
        arrow_line.set_data([th_med, th_med], [0, 0.92])
        arrow_dot.set_data([th_med], [0.92])
        txt_az.set_text(f"{az_med:.0f}°")
        th_inst = np.deg2rad(az)
        inst_line.set_data([th_inst, th_inst], [0, 0.78])

        # 2D heatmap
        lin2 = 10 ** (np.clip(s2d, -40, 0) / 10.0)
        lin2 /= (lin2.max() + 1e-12)
        im_2d.set_data(lin2)
        xh_v.set_xdata([az, az]);  xh_h.set_ydata([el, el])
        peak_dot.set_data([az], [el])

        # Quality panel
        for bar, v in zip(bars, eig):
            bar.set_height(float(v))
        txt_snr.set_text(f"SNR:  {snr:+.1f} dB")
        txt_papr.set_text(f"PAPR: {papr:.1f} dB")
        txt_el_q.set_text(f"El:   {el:.0f}°")
        txt_burst.set_text(f"bursts: {bn}")
        txt_frame.set_text(f"frames: {fn}")
        txt_rec.set_text(f"rec: {n_rec}")

        if no_sig:
            txt_nosig.set_text("NO BURST"); txt_nosig.set_color(C_ROSE)
            txt_nosig.get_bbox_patch().set_edgecolor(C_ROSE); a = 0.85
        elif no_doa:
            txt_nosig.set_text("DIR ?"); txt_nosig.set_color(C_AMBER)
            txt_nosig.get_bbox_patch().set_edgecolor(C_AMBER); a = 0.75
        else:
            a = 0.0
        txt_nosig.set_alpha(a)
        txt_nosig.get_bbox_patch().set_alpha(a * 0.6)
        arrow_line.set_alpha(0.15 if no_sig else (0.55 if no_doa else 1.0))
        arrow_dot.set_alpha(0.15 if no_sig else (0.55 if no_doa else 1.0))

        # Azimuth history
        if az_h:
            xs = np.arange(len(az_h))
            az_line.set_data(xs, az_h)
            az_med_line.set_data([0, H], [az_med, az_med])
        else:
            az_line.set_data([], []); az_med_line.set_data([], [])

        # Energy history
        if en_h:
            xs = np.arange(len(en_h))
            en_line.set_data(xs, en_h)
        else:
            en_line.set_data([], [])

        # Phase history
        for line, ph_data in zip(ph_lines, ph_h):
            if ph_data:
                line.set_data(np.arange(len(ph_data)), ph_data)
            else:
                line.set_data([], [])

        return (spec_line, arrow_line, arrow_dot, inst_line, txt_az, txt_nosig,
                im_2d, xh_v, xh_h, peak_dot,
                *bars, txt_snr, txt_papr, txt_el_q, txt_burst, txt_frame, txt_rec,
                az_line, az_med_line, en_line, *ph_lines)

    ani = animation.FuncAnimation(   # noqa: F841
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
# Main
# =============================================================================

def _run_calibration(
    acc: CovarianceAccumulatorUca,
    S: SimpleNamespace,
    cfg: UcaConfig,
    known_az_deg: float,
) -> None:
    """
    Auto-calibrate hardware phase offsets from EMA R at known TX azimuth.

    Method:
      - Wait until the EMA accumulator is warm (≥ 2·tau valid preamble bursts).
      - The dominant eigenvector v of R_EMA = a_hw(θ) = a(θ) ⊙ hw_offsets,
        where a(θ) is the theoretical steering vector and hw_offsets are
        cable/ADC phase imbalances.
      - Compute expected geometry phases for known_az_deg (assumes el≈0°).
      - hw_offset_k = angle(v_k) − geometry_phase_k  (channel 0 = reference).
      - Print resulting CHANNEL_PHASE_OFFSETS_DEG for the user to copy into config.
    """
    import datetime
    print(f"\n[CAL] Waiting for EMA convergence (TX az={known_az_deg:.1f}°)...")
    print(f"      Collecting until acc is warm (~{2/(1-acc.alpha):.0f} valid bursts) ...")

    timeout_s = 120.0
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(1.0)
        if not S.running:
            break
        if acc.is_warm and acc.R is not None:
            break
        n = acc.n_updates
        print(f"  [{time.time()-t0:5.1f}s]  valid preamble bursts so far: {n}", end="\r")
    print()

    if acc.R is None:
        print("[CAL] ERROR: no valid preamble bursts received.  Check TX / Heimdall.")
        return

    R_ema = acc.R
    # Dominant eigenvector ≈ actual hardware array response vector
    ev, V = np.linalg.eigh(R_ema)
    v = V[:, -1]  # largest eigenvalue → signal subspace
    v = v * np.exp(-1j * np.angle(v[0]))  # channel 0 = reference (phase 0)

    # Theoretical geometry phases for known TX at el≈0°
    az_rad = np.deg2rad(known_az_deg)
    pos    = cfg.positions  # (n_ant, 2) in wavelengths
    # tau[k] = 2π * (x_k*sin(az) + y_k*cos(az)) for el=0
    tau = 2 * np.pi * (pos[:, 0] * np.sin(az_rad) + pos[:, 1] * np.cos(az_rad))
    tau -= tau[0]  # normalise to channel 0

    # hw_offset = measured_phase − geometry_phase
    measured_phase = np.angle(v)  # already normalised (v[0] → 0°)
    hw_offsets_deg = np.degrees(measured_phase - tau)
    # Wrap to [-180, 180]
    hw_offsets_deg = (hw_offsets_deg + 180) % 360 - 180
    hw_offsets_deg[0] = 0.0  # reference channel

    print(f"\n[CAL] Hardware phase offsets computed from {acc.n_updates} preamble bursts:")
    print(f"      Phase meas:     {np.degrees(measured_phase).round(1).tolist()}")
    print(f"      Geom phases:    {np.degrees(tau).round(1).tolist()}")
    print(f"\n  ┌─ Copy to config.py ───────────────────────────────────────────")
    print(f"  │  CHANNEL_PHASE_OFFSETS_DEG = {hw_offsets_deg.round(2).tolist()}")
    print(f"  └───────────────────────────────────────────────────────────────")

    # Offer to auto-update config.py
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
        with open(cfg_path, "r") as f:
            txt = f.read()
        old_key = "CHANNEL_PHASE_OFFSETS_DEG"
        if old_key in txt:
            import re
            new_val = f"CHANNEL_PHASE_OFFSETS_DEG = {hw_offsets_deg.round(2).tolist()}"
            new_comment = (f"  # auto-calibrated {datetime.datetime.now():%Y-%m-%d %H:%M} "
                           f"from {acc.n_updates} bursts at az={known_az_deg:.1f}°")
            txt2 = re.sub(
                r"CHANNEL_PHASE_OFFSETS_DEG = \[.*?\]",
                new_val + new_comment,
                txt,
            )
            if txt2 != txt:
                with open(cfg_path, "w") as f:
                    f.write(txt2)
                print(f"\n[CAL] config.py updated automatically. Restart to apply.")
            else:
                print(f"\n[CAL] Could not auto-update config.py — edit manually.")
    except Exception as e:
        print(f"\n[CAL] Auto-update failed ({e}) — edit config.py manually.")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            f"Burst-gated 2D DoA at 868 MHz — KrakenSDR 5-element UCA  "
            f"(IRA preamble pilot at +{_PREAMBLE_TONE_HZ} Hz)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq",     type=float, default=C.FREQ_HZ / 1e6,
                   help="RF centre frequency [MHz] — must match LibreSDR TX")
    p.add_argument("--gain",     type=float, default=C.GAIN_DB,
                   help="IF gain [dB]")
    p.add_argument("--radius",   type=float, default=C.RADIUS_LAMBDA,
                   help="UCA radius in wavelengths")
    p.add_argument("--offset",   type=float, default=C.ANT0_OFFSET_DEG,
                   help="Antenna-0 offset from North [deg]")
    p.add_argument("--algo",     choices=["music", "capon", "bartlett"],
                   default=C.DOA_ALGORITHM.lower(), help="DoA algorithm")
    p.add_argument("--nsig",     type=int,   default=C.NUM_SIGNALS,
                   help="Expected signal sources")
    p.add_argument("--alpha",    type=float, default=C.COV_ALPHA,
                   help="EMA covariance smoothing across bursts (0=off, 0.9=heavy)")
    p.add_argument("--demo",     action="store_true",
                   help="Inject synthetic IRA burst at az=45° (no hardware required)")
    p.add_argument("--calibrate", type=float, default=None, metavar="AZ_DEG",
                   help=(
                       "Auto-calibrate hardware phase offsets for known TX azimuth. "
                       "Collects EMA until convergence, computes per-channel offsets, "
                       "prints updated CHANNEL_PHASE_OFFSETS_DEG values and exits."
                   ))
    p.add_argument("--out-dir",  default=None, metavar="DIR",
                   help="Directory to write .npz recordings")
    p.add_argument("--no-rec",   action="store_true",
                   help="Disable recording")
    p.add_argument("--rec-every", type=int, default=200, metavar="N",
                   help="Auto-save checkpoint every N bursts (0=only at exit)")
    args = p.parse_args()

    freq_hz = int(args.freq * 1e6)
    out_dir = getattr(args, "out_dir", None)

    cfg = UcaConfig(
        n_ant=C.N_ANTENNAS, radius_lambda=args.radius,
        n_az=C.N_AZ, n_el=C.N_EL, el_min_deg=C.EL_MIN_DEG,
        num_expected_signals=args.nsig, ant0_offset_deg=args.offset,
        ant_ccw=C.ANT_CCW,
    )
    # Alpha=0 → fresh covariance each burst; moderate alpha → EMA smoothing
    acc = CovarianceAccumulatorUca(alpha=args.alpha)
    S   = _make_state(cfg.n_az, cfg.n_el)
    S.rec_enabled = not args.no_rec

    print("=" * 58)
    print(f"  DoA BURST — {args.algo.upper()}  @  {freq_hz/1e6:.3f} MHz")
    print(f"  UCA: {cfg.n_ant} ant  r={args.radius:.3f}λ  offset={args.offset:.1f}°")
    print(f"  Heimdall: {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}")
    print(f"  Preamble pilot: +{_PREAMBLE_TONE_HZ} Hz  (IRA Rs/8)")
    print(f"  Burst: {_BURST_SYMS} sym  preamble: {_PREAMBLE_SYMS} sym  "
          f"SF: {int(_SUPERFRAME_S*1000)} ms")
    phase_offs = getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * cfg.n_ant)
    if any(o != 0.0 for o in phase_offs):
        print(f"  HW phase cal:  {[f'{o:.1f}' for o in phase_offs]} deg")
    else:
        print("  HW phase cal:  uncalibrated — run --calibrate <az_deg>")
    _multi_n = max(1, getattr(C, "MULTI_BURST_N", 1))
    if _multi_n > 1:
        print(f"  Multi-burst:   accumulate {_multi_n} bursts before DoA (~{10*np.log10(_multi_n):.1f} dB gain)")
    if getattr(C, "SNR_ADAPTIVE_ENABLED", False):
        print(f"  SNR-adaptive:  BARTLETT<{C.SNR_LOW_DB:.0f}dB / CAPON / {args.algo.upper()}>{C.SNR_HIGH_DB:.0f}dB")
    if getattr(C, "PHASE_COHERENCE_ENABLED", False):
        print(f"  Phase gate:    ±{C.PHASE_COHERENCE_MAX_JUMP_DEG:.0f}°")
    if getattr(C, "AZ_OUTLIER_ENABLED", False):
        print(f"  Az outlier:    ±{C.AZ_OUTLIER_MAX_DEV_DEG:.0f}° from median")
    burst_dir_str = out_dir or os.path.dirname(os.path.abspath(__file__))
    if S.rec_enabled:
        print(f"  Recording every {args.rec_every} bursts → {burst_dir_str}")
    else:
        print("  Recording: DISABLED")
    print("=" * 58)

    if not args.demo and not _check_heimdall(C.HEIMDALL_HOST, C.HEIMDALL_PORT):
        print(f"\n[ERROR] Heimdall not reachable at {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}")
        print("  Start Heimdall first, or use --demo to test without hardware.")
        sys.exit(1)

    if args.demo:
        src = None
        print("  DEMO: synthetic IRA burst at 45° azimuth")
    else:
        src = KrakenIQSource(
            host=C.HEIMDALL_HOST, port=C.HEIMDALL_PORT,
            ctrl_port=C.HEIMDALL_CTRL, num_channels=C.N_ANTENNAS,
            freq_hz=freq_hz, gain_db=args.gain,
        )
        src.start()
        print("  Heimdall connected.")

    threading.Thread(
        target=_acq_loop,
        args=(src, cfg, args.algo, acc, S),
        kwargs={"demo": args.demo},
        daemon=True,
    ).start()

    # ── Auto-calibration mode ─────────────────────────────────────────────────
    if args.calibrate is not None:
        _run_calibration(acc, S, cfg, args.calibrate)
        S.running = False
        if src is not None:
            src.stop()
        return

    if S.rec_enabled and args.rec_every > 0:
        threading.Thread(
            target=_autosave_loop,
            args=(S, freq_hz, args.algo, out_dir, args.rec_every),
            daemon=True,
        ).start()

    _build_and_run_ui(S, cfg, algo=args.algo, freq_hz=freq_hz)

    S.running = False
    if src is not None:
        src.stop()
    if S.rec_enabled:
        _save_recording(S, freq_hz=freq_hz, algo=args.algo, out_dir=out_dir)
    print("Session ended.")


if __name__ == "__main__":
    main()
