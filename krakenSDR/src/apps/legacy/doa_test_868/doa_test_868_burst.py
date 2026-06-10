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
# ── Modular LEGO components ───────────────────────────────────────────────────
from core.tracking import KalmanAngular, KalmanScalar
from core.gates import circ_median_deg
from core.tone_extraction import (
    find_preamble_onset as _fpo_core,
    find_tone_onset as _fto_core,
)

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
_PAPR_INST_MIN_DB = 12.0
_SPEC_EMA        = 0.20  # faster update than CW: each burst is ~90ms


def _check_heimdall(host: str, port: int) -> bool:
    try:
        s = _socket.create_connection((host, port), timeout=2.0)
        s.close()
        return True
    except OSError:
        return False


# Circular median for azimuth — implementation lives in core.gates.
_circ_median = circ_median_deg


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
        el_ema     = (C.EL_MIN_DEG + C.EL_MAX_DEG) / 2.0,  # linear EMA for elevation (init: grid midpoint)
        no_signal  = True,
        no_doa     = True,
        lock       = threading.Lock(),
        running    = True,
        # ── New telemetry fields ─────────────────────────────
        cfo_hist     = collections.deque(maxlen=C.HISTORY_LEN),   # CFO offset [Hz]
        last_iq_env  = np.zeros(64, dtype=np.float32),             # burst envelope CH0
        kf_az_hist   = collections.deque(maxlen=C.HISTORY_LEN),   # Kalman az estimate
        kf_el_hist   = collections.deque(maxlen=C.HISTORY_LEN),   # Kalman el estimate
        # ── Kalman filters (core.tracking) ───────────────────
        az_kf        = KalmanAngular(q=5.0, r=20.0),              # azimuth circular KF
        el_kf        = KalmanScalar(q=2.0,  r=8.0),               # elevation linear KF
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
        rec_X       = [],          # raw preamble IQ windows (n_ant × n_samp)
        rec_iq_enabled = True,    # set by --save-iq CLI flag
        burst_acc   = [],          # multi-burst covariance accumulation buffer
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
# Preamble-tone onset localisation  — thin wrappers around core.tone_extraction
# =============================================================================
# The core algorithms are in core/tone_extraction.py (general-purpose, fully
# parametric).  These local wrappers bind the IRA-specific module-level constants
# (_BURST_SAMPLES, _PRE_SAMPLES, _PREAMBLE_TONE_HZ) so _acq_loop keeps the same
# compact call signature.

def _find_preamble_onset(
    iq:       np.ndarray,
    b_start:  int,
    n_total:  int,
    fs:       float = _FS,
    win:      int   = _TONE_SCAN_WIN,
    known_hz: float | None = None,
) -> tuple:
    """Find (onset_sample, tone_hz).  See core.tone_extraction.find_preamble_onset."""
    return _fpo_core(
        iq, b_start, n_total,
        fs=fs, win=win, known_hz=known_hz,
        burst_samples=_BURST_SAMPLES,
        preamble_tone_hz=float(_PREAMBLE_TONE_HZ),
    )


def _find_tone_onset(
    iq:      np.ndarray,
    b_start: int,
    n_total: int,
    tone_hz: float = float(_PREAMBLE_TONE_HZ),
    fs:      float = _FS,
    win:     int   = _TONE_SCAN_WIN,
) -> int:
    """Find onset sample with highest coherent power at tone_hz.
    See core.tone_extraction.find_tone_onset."""
    return _fto_core(
        iq, b_start, n_total,
        tone_hz=tone_hz, fs=fs, win=win,
        burst_samples=_BURST_SAMPLES,
        preamble_samples=_PRE_SAMPLES,
    )


# =============================================================================
# Acquisition + DoA thread
# =============================================================================

def _acq_loop(
    src, cfg: UcaConfig, algo: str,
    acc: CovarianceAccumulatorUca, S: SimpleNamespace,
    demo: bool = False,
) -> None:
    rng     = np.random.default_rng(42)
    pos     = cfg.positions
    _fs     = _FS

    # Pilot extraction parameters
    _pilot_bw    = float(getattr(C, "PILOT_TONE_BW_HZ", 5_000))
    _amp_norm    = getattr(C, "AMPLITUDE_NORMALIZE", True)

    # Hardware phase calibration (per-channel offset in degrees, ch0 = reference 0°)
    _phase_offs  = list(getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * cfg.n_ant))
    _phase_offs  = (_phase_offs + [0.0] * cfg.n_ant)[: cfg.n_ant]  # pad/clip
    _has_cal     = any(o != 0.0 for o in _phase_offs)
    # Circular EMA alpha for per-burst az angle smoothing
    _az_alpha    = float(getattr(C, "AZ_SMOOTH_ALPHA", 0.50))
    # Linear EMA alpha for elevation smoothing (same gate logic as az)
    _el_alpha    = float(getattr(C, "EL_SMOOTH_ALPHA", 0.70))
    # Pre-computed elevation grid step for floor/ceiling boundary detection
    _el_step     = (cfg.el_max_deg - cfg.el_min_deg) / max(cfg.n_el - 1, 1)

    # Outlier rejection gate (AZ_OUTLIER_ENABLED in config)
    _az_outlier_enabled  = bool(getattr(C, "AZ_OUTLIER_ENABLED", True))
    _az_outlier_max_dev  = float(getattr(C, "AZ_OUTLIER_MAX_DEV_DEG", 45.0))
    _az_outlier_min_hist = int(getattr(C, "AZ_OUTLIER_MIN_HISTORY", 5))
    _az_reject_streak    = 0      # resets EMA after N consecutive rejections
    _AZ_RESET_AFTER      = 3     # force-accept after this many consecutive rejects
    _az_phasor_init      = True   # True until first accepted burst seeds phasor

    # Phase coherence gate (PHASE_COHERENCE_ENABLED in config)
    _phase_coh_enabled   = bool(getattr(C, "PHASE_COHERENCE_ENABLED", True))
    _phase_coh_max_jump  = float(getattr(C, "PHASE_COHERENCE_MAX_JUMP_DEG", 60.0))

    # Signal/noise subspace gap gate (EIG_SN_GAP_MIN_DB in config)
    _eig_sn_gap_min = float(getattr(C, "EIG_SN_GAP_MIN_DB", 0.0))

    # Consecutive-no-burst counter — warns user if TX is likely in CW mode
    _no_burst_streak = 0

    # Multi-burst accumulation (MULTI_BURST_N covariance matrices averaged before DoA)
    _multi_n = max(1, int(getattr(C, "MULTI_BURST_N", 1)))
    _R_batch: collections.deque = collections.deque(maxlen=_multi_n)
    _papr_min = float(getattr(C, "PAPR_INST_MIN_DB", _PAPR_INST_MIN_DB))

    # Tone frequency lock: None until first burst passes PAPR gate.
    # After lock, _find_preamble_onset restricts freq search to ±2 kHz
    # around this value so low-SNR noise spikes can't steal the onset.
    _cfo_ema_hz: float | None = None

    # Phase-diffs phasor EMA: circular-mean smoothing over accepted bursts.
    # Only updated on has_doa=True bursts so floor/outlier reflections
    # (which have different spatial phases) do not corrupt the display.
    _phase_diffs_phasor: np.ndarray = np.zeros(4, dtype=complex)

    # ── Gate rejection diagnostic counters ───────────────────────────────────
    # Printed every _DIAG_INTERVAL_S seconds so the user can see which gate
    # is most active.  Format: "DIAG: detected=N eig_spread=N papr=N sn_gap=N
    #                                  floor=N ceil=N outlier=N accepted=N"
    _DIAG_INTERVAL_S = 10.0
    _diag_t0     = time.monotonic()
    _cnt_detect  = 0   # bursts passing energy detector
    _cnt_eig_lo  = 0   # Stage 1: eig spread too low
    _cnt_papr    = 0   # Stage 2: PAPR too low (not a preamble)
    _cnt_sn_gap  = 0   # Stage 2b: λ2/λ3 gap too low
    _cnt_floor   = 0   # floor boundary reject
    _cnt_outlier = 0   # az outlier reject
    _cnt_accepted= 0   # bursts accepted into _R_batch (→ DoA runs after N)
    _cnt_doa_out = 0   # full DoA outputs produced
    _papr_hist:  list[float] = []   # PAPR of ALL computed windows (threshold calibration)

    # Streaming sample buffer: accumulate CPI frames until we have enough
    # for reliable burst detection (≥ 1 full SUPERFRAME).
    _buf: list[np.ndarray] = []
    _buf_len = 0
    _min_buf = _SF_SAMPLES + _WINDOW_SAMPLES + 2048  # need ≥ superframe + burst

    while S.running:
        # ── Get IQ frame ────────────────────────────────────────────────────
        if demo:
            # Synthetic: inject one IRA burst at demo_az every 90 ms.
            # Silence + burst + silence fills one synthetic "CPI".
            demo_az = np.deg2rad(45.0)
            demo_el = np.deg2rad(10.0)
            gain_off  = np.array([1.0, 0.92, 1.08, 0.95, 1.03])
            phase_off = np.deg2rad([0.0, 5.0, -8.0, 12.0, -3.0])
            tau = 2 * np.pi * (pos[:, 0] * np.cos(demo_el) * np.sin(demo_az)
                               + pos[:, 1] * np.cos(demo_el) * np.cos(demo_az))

            N_frame = _SF_SAMPLES
            t = np.arange(N_frame, dtype=np.float64)
            # Place burst at sample 512
            burst_start = 512
            burst_len   = _BURST_SAMPLES
            pre_len     = _PRE_SAMPLES
            # Preamble tone at +3125 Hz
            preamble_tone = np.exp(2j * np.pi * _PREAMBLE_TONE_HZ / _fs * t)

            X = np.zeros((cfg.n_ant, N_frame), dtype=np.complex128)
            snr_lin = 10 ** (18.0 / 10.0)   # 18 dB SNR main path
            for k in range(cfg.n_ant):
                channel_phase = tau[k] + phase_off[k]
                # Preamble (pure tone with array phase)
                X[k, burst_start: burst_start + pre_len] += (
                    gain_off[k] * np.exp(1j * channel_phase)
                    * preamble_tone[burst_start: burst_start + pre_len]
                    * np.sqrt(snr_lin)
                )
            # Simulated multipath reflection: +30° offset, -6 dB (12 dB SNR)
            # Inflates λ2 so EIG_SN_GAP gate behaves like real indoor conditions.
            refl_az = demo_az + np.deg2rad(30.0)
            refl_el = np.deg2rad(8.0)
            tau_r = 2 * np.pi * (pos[:, 0] * np.cos(refl_el) * np.sin(refl_az)
                                 + pos[:, 1] * np.cos(refl_el) * np.cos(refl_az))
            snr_refl = 10 ** (12.0 / 10.0)   # -6 dB relative to main
            for k in range(cfg.n_ant):
                X[k, burst_start: burst_start + pre_len] += (
                    gain_off[k] * np.exp(1j * (tau_r[k] + phase_off[k]))
                    * preamble_tone[burst_start: burst_start + pre_len]
                    * np.sqrt(snr_refl)
                )
            # Add white noise everywhere
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

        # ── Accumulate into sliding-window buffer ────────────────────────────
        _buf.append(frame)
        _buf_len += frame.shape[1]

        if _buf_len < _min_buf:
            continue   # not enough samples yet

        # Flatten to one contiguous array for detection
        X_stream = np.concatenate(_buf, axis=1)
        n_total  = X_stream.shape[1]

        # Sliding-window trim: keep the last _WINDOW_SAMPLES as overlap so
        # bursts that straddle frame boundaries are never missed.
        _keep = _WINDOW_SAMPLES + 512
        if n_total > _keep:
            _buf = [X_stream[:, -_keep:]]
            _buf_len = _keep
        else:
            _buf = [X_stream]
            _buf_len = n_total

        # ── Burst detection on channel 0 (reference) ─────────────────────────
        bursts = _detect_bursts(X_stream[0])

        if not bursts:
            # No burst in this block — update energy display anyway
            pwr = float(np.mean(np.abs(X_stream) ** 2))
            with S.lock:
                S.energy_hist.append(10 * np.log10(pwr + 1e-20))
                S.no_signal = True
            _no_burst_streak += 1
            if _no_burst_streak == 10:
                print("[WARN] 10 consecutive frames with no burst detected.\n"
                      "       Make sure LibreSDR is in IRA/BURST mode (not CW).\n"
                      "       Use task 'LibreSDR: TX 868 MHz — GUI BURST' or\n"
                      "       pass --mode ira to tx_868_gui.py")
            continue

        _no_burst_streak = 0

        # ── Periodic gate-rejection diagnostics ────────────────────────────
        _cnt_detect += len(bursts)
        _now = time.monotonic()
        if _now - _diag_t0 >= _DIAG_INTERVAL_S:
            _elapsed = _now - _diag_t0
            _diag_t0 = _now
            _ph = _papr_hist if _papr_hist else [0.0]
            _papr_mean = float(np.mean(_ph))
            _papr_med  = float(np.median(_ph))
            _papr_min5 = float(np.percentile(_ph, 5))
            print(
                f"[DIAG] {_elapsed:.0f}s | "
                f"detected={_cnt_detect} "
                f"eig_lo={_cnt_eig_lo} "
                f"papr_rej={_cnt_papr} "
                f"(μ={_papr_mean:.1f} med={_papr_med:.1f} p5={_papr_min5:.1f} thr≥{_papr_min:.0f}dB) "
                f"sn_gap={_cnt_sn_gap} "
                f"floor={_cnt_floor} "
                f"outlier={_cnt_outlier} "
                f"accepted={_cnt_accepted} "
                f"doa_out={_cnt_doa_out}"
            )
            _cnt_detect = _cnt_eig_lo = _cnt_papr = _cnt_sn_gap = 0
            _cnt_floor  = _cnt_outlier = _cnt_accepted = _cnt_doa_out = 0
            _papr_hist = []

        for b_start in bursts:
            b_end = min(b_start + _WINDOW_SAMPLES, n_total)
            if b_end - b_start < _PRE_SAMPLES:
                continue   # too close to end of buffer

            # ── Joint time×frequency search: find preamble onset + real tone Hz ──
            # No prior knowledge of the LO offset is needed: the preamble
            # pure tone has the highest per-bin FFT power in the burst,
            # regardless of where the actual frequency lands.
            tone_start, _tone_hz = _find_preamble_onset(
                X_stream[0], b_start, n_total,
                known_hz=_cfo_ema_hz,   # None on first burst → unconstrained
            )
            # Print LO offset on first successful detection (diagnostic)
            if not getattr(_acq_loop, "_offset_printed", False):
                _lo_off_khz = (_tone_hz - _PREAMBLE_TONE_HZ) / 1_000.0
                print(f"[INFO] Preamble tone auto-detected @ {_tone_hz/1e3:.2f} kHz  "
                      f"(LO offset = {_lo_off_khz:+.1f} kHz  "
                      f"= {_lo_off_khz*1e3/868_100:.0f} ppm)")
                _acq_loop._offset_printed = True
                # Lock immediately so all subsequent searches are constrained.
                # Do NOT wait for PAPR gate — first unconstrained detection is
                # the most reliable (real signal present after energy trigger).
                _cfo_ema_hz = _tone_hz
                print(f"[INFO] Tone frequency locked at {_tone_hz/1e3:.2f} kHz  (auto-lock)")
            tone_end = min(tone_start + _PRE_SAMPLES, n_total)
            if tone_end - tone_start < _PRE_SAMPLES // 2:
                continue   # not enough preamble to process

            X_pre = X_stream[:, tone_start: tone_end]

            pwr_db = float(10 * np.log10(np.mean(np.abs(X_pre) ** 2) + 1e-20))

            # ── Preamble narrowband BPF ────────────────────────────────────────
            # Filter at the AUTO-DETECTED tone frequency (handles LO offset).
            # Gain SNR ≈ 10·log10(FS / BPF_BW) ≈ +22 dB vs raw preamble.
            # In CW mode (PILOT_TONE_ENABLED=True), filter at the configured offset.
            _bpf_ok = True   # assume BPF captured the tone; overridden below if not
            if getattr(C, "PILOT_TONE_ENABLED", False):
                X_proc = extract_pilot_tone(
                    X_pre, float(C.SAMPLE_RATE_HZ),
                    tone_hz=float(C.PILOT_TONE_OFFSET_HZ),
                    bw_hz=float(C.PILOT_TONE_BW_HZ),
                )
            else:
                # Burst mode: BPF at detected preamble tone (auto LO-offset corrected)
                _bpf_bw = float(getattr(C, "PREAMBLE_BPF_BW_HZ", 10_000.0))
                X_proc = extract_pilot_tone(
                    X_pre, float(C.SAMPLE_RATE_HZ),
                    tone_hz=_tone_hz,
                    bw_hz=_bpf_bw,
                )
                # Safety check: if BPF output power < 5% of raw input the tone
                # was not captured (e.g. aliased or very below noise floor).
                # Fall back to unfiltered X_pre so DoA can still gate on PAPR.
                _bpf_pwr = float(np.mean(np.abs(X_proc) ** 2))
                _raw_pwr = float(np.mean(np.abs(X_pre)  ** 2)) + 1e-30
                _bpf_ok  = (_bpf_pwr / _raw_pwr >= 0.05)
                if not _bpf_ok:   # < 5% of raw power → BPF missed tone
                    X_proc = X_pre

            # ── Amplitude normalization ───────────────────────────────────────
            if C.AMPLITUDE_NORMALIZE:
                X_proc = amplitude_normalize_channels(X_proc)

            # ── Enhanced preprocessing (optional) ─────────────────────────────
            # Apply advanced preprocessing techniques based on research papers
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

            # ── Hardware phase calibration ────────────────────────────────────
            # Compensates cable-length differences and ADC phase imbalances that
            # shift the steering vectors relative to theoretical UCA positions.
            # Without calibration, instantaneous MUSIC PAPR drops from ≥29 dB to
            # ~14-17 dB at ±10-15° error, limiting preamble detection rate.
            # Configure CHANNEL_PHASE_OFFSETS_DEG in config.py (run --calibrate).
            if _has_cal:
                X_cal = _apply_phase_correction(X_proc, _phase_offs)
            else:
                X_cal = X_proc

            # ── Instantaneous covariance + quality gate ───────────────────────
            try:
                R_inst = (X_cal @ X_cal.conj().T) / X_cal.shape[1]
                snr    = snr_uca_db(R_inst)
                eig    = eigenvalue_spread_uca_db(R_inst)

                # Phase diffs from R_inst — always computed so the plot is always live
                phase_diffs_inst = np.degrees(np.angle(R_inst[1:, 0]))

                # Stage 1 fast-reject: skip MUSIC if eigenspread too low
                if eig[0] < C.EIG_SPREAD_MIN_DB:
                    _cnt_eig_lo += 1
                    with S.lock:
                        S.snr_db  = snr
                        S.eig_db  = eig
                        S.no_signal = True
                        S.no_doa    = True
                        S.energy_hist.append(pwr_db)
                        S.snr_hist.append(snr)
                        # Phase NOT updated: R_inst at near-noise eigenspread is pure noise.
                        S.burst_n += 1
                    continue

                # Stage 1b: ADC saturation advisory — λ1 > EIG_INST_MAX_DB.
                # NOTE: after BPF pilot extraction the noise floor eigenvalue is
                # near-zero, so eigenvalue SPREAD is always large (40-55+ dB).
                # This gate is therefore WARNING-ONLY — never rejects a burst.
                # Use raw IQ amplitude clipping to detect true ADC saturation.
                _eig_max = float(getattr(C, "EIG_INST_MAX_DB", 50.0))
                if eig[0] > _eig_max:
                    if not getattr(_acq_loop, "_sat_warned", False):
                        print(
                            f"[INFO] λ1={eig[0]:.1f} dB (high eigenspread — normal after BPF)")
                        _acq_loop._sat_warned = True

                # Stage 2: instantaneous DoA on R_inst — per-burst DoA estimator.
                # For a genuine preamble window (rank-1 after pilot BPF), MUSIC
                # gives papr_inst ≥ 12 dB even with ±20° HW phase errors.
                # For DATA/noise windows the BPF output is near-isotropic →
                # papr_inst ≤ 12 dB → rejected.  This separation makes DoA
                # the SIGNAL DETECTOR and DOA ESTIMATOR simultaneously.
                
                # Apply selected algorithm based on user preference
                if algo == "capon":
                    spec2d_inst = doa_capon_uca_2d(X_cal, cfg, R_in=R_inst,
                                                    decorr=getattr(C, "CAPNT_DECORR", "none"))
                elif algo == "bartlett":
                    spec2d_inst = doa_bartlett_uca_2d(X_cal, cfg, R_in=R_inst)
                elif algo == "root-music":
                    spec2d_inst = doa_root_music_uca_2d(R_inst, cfg)[0]  # Get spectrum from tuple
                elif algo == "unitary-esprit":
                    spec2d_inst = doa_unitary_esprit_uca_2d(R_inst, cfg)[0]  # Get spectrum from tuple
                elif algo == "mfba-music":
                    spec2d_inst = doa_mfba_music_uca_2d(R_inst, cfg)[0]  # Get spectrum from tuple
                else:  # Default to MUSIC
                    spec2d_inst = doa_music_uca_2d(X_cal, cfg, R_in=R_inst,
                                                   decorr=getattr(C, "MUSIC_DECORR", "none"))
                
                az_inst, el_inst, papr_inst = find_peak_uca_2d(spec2d_inst, cfg)
                _papr_hist.append(float(papr_inst))   # track all, not just accepted

                is_preamble = (papr_inst >= _papr_min)

                if is_preamble:
                    # Update tone frequency EMA only when the BPF actually captured
                    # the tone (_bpf_ok). If the fallback (raw X_pre) was used the
                    # returned _tone_hz is unreliable → skip EMA to avoid poisoning.
                    if _bpf_ok and _cfo_ema_hz is not None:
                        _cfo_ema_hz = 0.95 * _cfo_ema_hz + 0.05 * _tone_hz

                if not is_preamble:
                    _cnt_papr += 1
                    # Non-preamble window: skip DoA.
                    # Phase NOT updated: wideband data/noise window has isotropic
                    # R_inst phases (std≈90°) → pollutes phase display.
                    with S.lock:
                        S.snr_db  = snr
                        S.eig_db  = eig
                        S.no_signal = True
                        S.no_doa    = True
                        S.energy_hist.append(pwr_db)
                        S.snr_hist.append(snr)
                        S.burst_n += 1
                    continue

                # ── Stage 2b: signal/noise subspace gap gate (EIG_SN_GAP_MIN_DB) ──
                # Checks λ₂/λ₃ ratio in dB (eig[1]-eig[2]).  With NUM_SIGNALS=2,
                # a small gap means λ₂ ≈ λ₃ → noise contaminates signal subspace →
                # MUSIC spectrum is nearly flat → argmax returns bin 0 (az=0°).
                # Data 20260504: az≈0° ghosts have gap=1.9 dB; true az≈54° have 9.9 dB.
                # Bursts failing this gate are NOT added to _R_batch (skipped cleanly).
                # NOTE: we do NOT update phase_hist here.  These bursts have low
                # signal/noise separation → R_inst phases are pure noise.  Adding them
                # would corrupt the phase display (turned it completely random after fix).
                if _eig_sn_gap_min > 0.0 and len(eig) >= 3:
                    _sn_gap = float(eig[1] - eig[2])
                    if _sn_gap < _eig_sn_gap_min:
                        _cnt_sn_gap += 1
                        with S.lock:
                            S.snr_db  = snr
                            S.eig_db  = eig
                            S.no_signal = True
                            S.no_doa    = True
                            S.energy_hist.append(pwr_db)
                            S.snr_hist.append(snr)
                            S.burst_n += 1
                        continue

                # ── Valid preamble burst ──────────────────────────────────────
                # Multi-burst accumulation: average N covariance matrices before DoA
                _cnt_accepted += 1
                _R_batch.append(R_inst)
                if len(_R_batch) < _multi_n:
                    # Accumulate EMA while waiting for N bursts.
                    # Phase NOT updated yet: single-burst R_inst phase is noisier
                    # than the R_avg we'll get when the batch is full.
                    acc.update(X_cal)
                    with S.lock:
                        S.snr_db = snr; S.eig_db = eig
                        S.energy_hist.append(pwr_db); S.snr_hist.append(snr)
                        S.burst_n += 1
                    continue
                # Average the accumulated batch → lower noise floor by ~5 dB (N=3)
                R_avg = np.mean(list(_R_batch), axis=0)

                # Update R_EMA for temporal multipath decorrelation / phase display
                R = acc.update(X_cal)

                # For DoA, run algorithm again on R_avg (averaged over N bursts)
                # This reduces the noise floor by ~10*log10(N) dB.
                if _multi_n > 1:
                    if algo == "capon":
                        spec2d_avg = doa_capon_uca_2d(X_cal, cfg, R_in=R_avg,
                                                       decorr=getattr(C, "CAPNT_DECORR", "none"))
                    elif algo == "bartlett":
                        spec2d_avg = doa_bartlett_uca_2d(X_cal, cfg, R_in=R_avg)
                    else:
                        spec2d_avg = doa_music_uca_2d(X_cal, cfg, R_in=R_avg,
                                                       decorr=getattr(C, "MUSIC_DECORR", "none"))
                    _,  _, papr_avg = find_peak_uca_2d(spec2d_avg, cfg)
                    spec2d_inst = spec2d_avg

                # ── Marginal-then-conditional peak extraction ──────────────────
                # A flat UCA has a coupling ridge in the 2D MUSIC spectrum: the
                # joint 2D argmax can slide up the ridge to the el-ceiling instead
                # of staying at the true peak.  Fix: estimate az from the 1D
                # marginal (max over el, more robust), then estimate el from the
                # conditional slice at that az.  papr_inst unchanged (from 2D).
                _az_marg  = np.max(spec2d_inst, axis=0)   # (n_az,)
                _az_bin_m = int(np.argmax(_az_marg))
                az_inst   = float(cfg.az_range_deg()[_az_bin_m])
                _el_slice = spec2d_inst[:, _az_bin_m]     # (n_el,) at true az
                _el_bin_m = int(np.argmax(_el_slice))
                el_inst   = float(cfg.el_range_deg()[_el_bin_m])

                # ── Outlier gate (AZ_OUTLIER) ────────────────────────────────────────
                # Reject az estimates that deviate too much from the running median.
                # Protects the EMA from wild multipath jumps.
                # Exception: after _AZ_RESET_AFTER consecutive rejects, the array has
                # likely moved → force-accept to let the EMA re-lock (rotation support).
                az_ph    = np.exp(1j * np.deg2rad(az_inst))
                _accept   = True
                _el_clamp = False   # True when el hits floor or ceiling (az still valid)

                # Floor-boundary guard: el at bottom grid point.
                # Previously this rejected the entire burst on the assumption that
                # el≈el_min implies a ground reflection with wrong az.  However data
                # shows PAPR-confirmed preamble bursts (≥8 dB) consistently land at
                # the floor when the TX is at low elevation (~5-6°) — these are valid
                # direct-path bursts, not reflections.  Az from the 1D marginal is
                # reliable regardless of el position (marginal = max over all el rows).
                # The outlier gate acts as a backstop against any wild az values.
                # → Treat floor exactly like ceiling: clamp el, accept az.
                if el_inst <= cfg.el_min_deg + _el_step * 0.5:
                    _el_clamp = True
                    _cnt_floor += 1   # keep counting for diagnostics
                # Ceiling-boundary guard: el at top grid point → el unreliable.
                # Since az is now estimated via 1D marginal (independent of el),
                # do NOT reject the burst — just skip the el_ema update.
                if el_inst >= cfg.el_max_deg - _el_step * 0.5:
                    _el_clamp = True
                if _az_outlier_enabled and len(S.az_hist) >= _az_outlier_min_hist:
                    _med = _circ_median(np.array(S.az_hist))
                    _dev = float(abs(((az_inst - _med + 180) % 360) - 180))
                    if _dev > _az_outlier_max_dev:
                        _az_reject_streak += 1
                        if _az_reject_streak >= _AZ_RESET_AFTER:
                            # Likely array rotation — force re-init phasor to new direction
                            _az_reject_streak = 0
                            _az_phasor_init = True   # next accept seeds phasor cold
                        else:
                            _accept = False
                            _cnt_outlier += 1

                # ── Phase coherence gate ─────────────────────────────────────────────
                # Reject if ANY channel's phase diff jumps by more than the threshold.
                # Catches sudden per-burst multipath inversion while allowing the
                # genuine slow phase drift during array rotation.
                if _accept and _phase_coh_enabled:
                    _ph_hist_len = len(S.phase_hist[0])
                    if _ph_hist_len >= 3:
                        for _gi in range(4):
                            _window = list(S.phase_hist[_gi])[-5:]
                            _ph_med = float(np.degrees(np.angle(
                                np.mean(np.exp(1j * np.deg2rad(_window))))))
                            _ph_dev = float(abs(
                                ((phase_diffs_inst[_gi] - _ph_med + 180) % 360) - 180))
                            if _ph_dev > _phase_coh_max_jump:
                                _accept = False
                                break

                if _accept:
                    _az_reject_streak = 0
                    if _az_phasor_init:
                        # Cold start: seed phasor directly from first valid estimate.
                        # Avoids blending with the 0° initial value which would
                        # keep az_deg near 0° for several bursts before converging.
                        S.az_phasor = az_ph
                        _az_phasor_init = False
                    else:
                        S.az_phasor = _az_alpha * S.az_phasor + (1.0 - _az_alpha) * az_ph
                    # El linear EMA — only updated from non-ceiling, non-boundary estimates
                    if not _el_clamp:
                        S.el_ema = _el_alpha * S.el_ema + (1.0 - _el_alpha) * el_inst
                az        = float(np.degrees(np.angle(S.az_phasor)) % 360.0)
                el_est    = S.el_ema   # always valid (initialised to grid midpoint)
                spec2d    = spec2d_inst
                az_spec   = np.max(spec2d, axis=0)
                papr      = papr_inst

                # phase_diffs from R_avg (multi-burst average) for display
                phase_diffs = np.degrees(np.angle(R_avg[1:, 0]))

            except Exception as exc:
                print(f"[DoA] burst #{S.burst_n+1}: {exc}")
                continue

            # has_signal = valid preamble detected (papr_inst ≥ threshold)
            # has_doa    = preamble passed all gates (boundary + outlier)
            has_signal = True
            has_doa    = _accept    # False for boundary/outlier-rejected bursts
            _cnt_doa_out += 1

            # Phasor EMA for phase display: only update on has_doa bursts.
            # Floor/outlier-rejected bursts have different spatial phases
            # (multipath, reflections) → must not pollute the live display.
            _PHASE_EMA = 0.15   # ~6 burst smoothing window
            if has_doa:
                _ph_new = np.exp(1j * np.deg2rad(phase_diffs))
                if np.all(_phase_diffs_phasor == 0):
                    _phase_diffs_phasor = _ph_new
                else:
                    _phase_diffs_phasor = (1 - _PHASE_EMA) * _phase_diffs_phasor + _PHASE_EMA * _ph_new
                _phase_diffs_smooth = np.degrees(np.angle(_phase_diffs_phasor))

            with S.lock:
                S.az_spec     = (1 - _SPEC_EMA) * S.az_spec + _SPEC_EMA * az_spec
                S.spec2d      = (1 - _SPEC_EMA) * S.spec2d  + _SPEC_EMA * spec2d
                S.az_deg      = az
                S.el_deg      = el_est   # smoothed EMA (or raw on first burst)
                S.papr_db     = papr
                S.snr_db      = snr
                S.eig_db      = eig
                S.no_signal   = False
                S.no_doa      = not has_doa   # DIR? when boundary/outlier rejected
                S.energy_hist.append(pwr_db)
                S.el_hist.append(el_inst)  # raw per-burst for scatter display
                S.snr_hist.append(snr)
                if has_doa:
                    S.phase_diffs = _phase_diffs_smooth   # EMA-smoothed, has_doa only
                    for _i, _p in enumerate(phase_diffs):
                        S.phase_hist[_i].append(float(_p))
                    S.az_hist.append(az_inst)  # raw MUSIC peak → gate uses reactive median
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
                    if S.rec_iq_enabled:
                        S.rec_X.append(X_pre.copy())
                # ── CFO tracking ─────────────────────────────────────────────
                S.cfo_hist.append(float(_tone_hz - _PREAMBLE_TONE_HZ))
                # ── IQ burst envelope (CH0, down-sampled to 64 pts) ──────────
                _env = np.abs(X_pre[0])
                _ds  = max(1, len(_env) // 64)
                _smp = _env[::_ds][:64].astype(np.float32)
                _pk  = float(np.max(_smp)) + 1e-12
                S.last_iq_env = _smp / _pk        # normalise 0..1
                # ── Scalar Kalman smoother  (az circular, el linear) ─────────
                # KalmanAngular / KalmanScalar from core.tracking — handle
                # cold-start, angular wrap, and Q/R noise configuration.
                _kf_az = S.az_kf.update(az_inst)
                _kf_el = S.el_kf.update(el_inst)
                S.kf_az_hist.append(_kf_az)
                S.kf_el_hist.append(_kf_el)
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
    # Save raw preamble IQ windows if --save-iq was used
    if S.rec_iq_enabled and S.rec_X:
        iq_path = path.replace(".npz", "_iq.npz")
        try:
            # Build array: shape (N_bursts, n_ant, n_samp) — pad to max length
            max_samp = max(x.shape[1] for x in S.rec_X)
            iq_arr = np.zeros((len(S.rec_X), S.rec_X[0].shape[0], max_samp),
                               dtype=np.complex64)
            for _k, _x in enumerate(S.rec_X):
                iq_arr[_k, :, :_x.shape[1]] = _x.astype(np.complex64)
            np.savez_compressed(iq_path, preamble_iq=iq_arr,
                                         sample_rate=np.float64(_FS),
                                         preamble_tone_hz=np.int32(_PREAMBLE_TONE_HZ))
            print(f"[REC] IQ raw  → {iq_path}  ({len(S.rec_X)} bursts, shape {iq_arr.shape})")
        except Exception as _e:
            print(f"[REC] IQ save failed: {_e}")
    n_sig = int(np.sum(S.rec_has_sig))
    print(f"[REC] {n} bursts ({n_sig} valid, {n_sig*100//max(n,1)}%) → {path}")
    return path


# =============================================================================
# UI
# =============================================================================

def _build_and_run_ui(S: SimpleNamespace, cfg: UcaConfig,
                      algo: str, freq_hz: int) -> None:
    el_min  = cfg.el_min_deg
    el_max  = cfg.el_max_deg          # use actual grid max (not hardcoded 90)
    az_rad  = np.deg2rad(cfg.az_range_deg())
    n_az    = cfg.n_az
    n_el    = len(cfg.el_range_deg())
    n_ant   = cfg.n_ant
    n_sig   = cfg.num_expected_signals
    H       = C.HISTORY_LEN

    fig = plt.figure(figsize=(17, 9), facecolor=BG)
    fig.canvas.manager.set_window_title(
        f"DoA BURST 868 MHz — {algo.upper()}  (IRA preamble @ +{_PREAMBLE_TONE_HZ} Hz)")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[1.4, 1.0],
                           left=0.05, right=0.97,
                           top=0.93, bottom=0.07,
                           hspace=0.45, wspace=0.35)

    # ── [0,0]  Skyplot  (az × el polar, zenith = centre, horizon = edge) ────
    ax_sky = fig.add_subplot(gs[0, 0], projection="polar", facecolor=BG2)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_ylim(0, 90)           # r = 90 − el_deg  (0 → zenith, 90 → horizon)
    ax_sky.set_yticks([15, 30, 45, 60, 75])
    ax_sky.set_yticklabels(["75°", "60°", "45°", "30°", "15°"], fontsize=6, color=C_MUT)
    ax_sky.tick_params(colors=C_MUT, labelsize=7)
    ax_sky.set_facecolor(BG2)
    for sp in ax_sky.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_sky.set_title("Skyplot  (N↑ CW,  centre = zenith)", color=C_TEXT, fontsize=9, pad=10)
    ax_sky.grid(color=C_BDR, lw=0.5, alpha=0.35)
    # MUSIC spectrum lobes (Az 1-D marginal, max over el) — radar style
    _az_ext = np.r_[az_rad, az_rad[0]]
    _lobe_init = np.zeros(n_az + 1)
    sky_spec_line, = ax_sky.plot(_az_ext, _lobe_init,
                                  "-", color=C_TEAL, lw=1.0, alpha=0.65, zorder=2)
    sky_spec_fill  = ax_sky.fill(_az_ext, _lobe_init,
                                  color=C_TEAL, alpha=0.12, zorder=1)[0]
    # History scatter (recency-coloured)
    sky_scat = ax_sky.scatter([], [], c=[], cmap="plasma",
                               vmin=0.0, vmax=1.0, s=14, alpha=0.55, zorder=3)
    # Kalman track
    sky_kf_line, = ax_sky.plot([], [], "-", color=C_LIME, lw=1.8, alpha=0.65, zorder=4)
    # EMA estimate — radial spoke + dot
    sky_arrow, = ax_sky.plot([0, 0], [0, 45], "-", color=C_LIME, lw=2.2, zorder=5)
    sky_dot,   = ax_sky.plot([0], [45], "o",  color=C_LIME, ms=9, zorder=6) 
    txt_sky_az = ax_sky.text(0, 0, "—°", ha="center", va="center",
                              color=C_LIME, fontsize=14, fontweight="bold")
    txt_sky_nosig = ax_sky.text(
        0.5, 0.5, "NO BURST", transform=ax_sky.transAxes,
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=C_ROSE, alpha=0.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=C_ROSE, alpha=0.0),
        zorder=10)
    txt_sky_el = ax_sky.text(
        0.5, 0.02, "El: —°", transform=ax_sky.transAxes,
        ha="center", va="bottom", color=C_TEAL, fontsize=9)

    # ── [0,1]  2D az × el spectrum ────────────────────────────────────────────
    ax_2d = fig.add_subplot(gs[0, 1], facecolor=BG2)
    ax_2d.set_facecolor(BG2)
    ax_2d.set_xlabel("Azimuth [°]", color=C_MUT, fontsize=8)
    ax_2d.set_ylabel("Elevation [°]", color=C_MUT, fontsize=8)
    ax_2d.set_title("2D MUSIC spectrum  az × el", color=C_TEXT, fontsize=9)
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

    # ── [0,2]  IQ burst oscilloscope  (CH0 envelope of last valid burst) ─────
    _IQ_PTS = 64
    ax_iq = fig.add_subplot(gs[0, 2], facecolor=BG2)
    ax_iq.set_facecolor(BG2)
    ax_iq.set_xlabel("Time  [sample index, downsampled]", color=C_MUT, fontsize=8)
    ax_iq.set_ylabel("Amplitude  [normalised]", color=C_MUT, fontsize=8)
    ax_iq.set_title("Last burst — IQ envelope  (CH0)", color=C_TEXT, fontsize=9)
    ax_iq.set_xlim(0, _IQ_PTS - 1)
    ax_iq.set_ylim(0, 1.05)
    ax_iq.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_iq.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_iq.grid(color=C_BDR, lw=0.4, alpha=0.35)
    _iq_xs = np.arange(_IQ_PTS)
    iq_line, = ax_iq.plot(_iq_xs, np.zeros(_IQ_PTS), "-",
                           color=C_TEAL, lw=1.4, alpha=0.9, zorder=3)
    iq_fill_ref = [ax_iq.fill_between(_iq_xs, np.zeros(_IQ_PTS),
                                       color=C_TEAL, alpha=0.10, zorder=2)]
    txt_iq_snr  = ax_iq.text(0.97, 0.94, "SNR: — dB",  transform=ax_iq.transAxes,
                               ha="right", va="top", color=C_TEXT,  fontsize=9)
    txt_iq_papr = ax_iq.text(0.97, 0.80, "PAPR: — dB", transform=ax_iq.transAxes,
                               ha="right", va="top", color=C_AMBER, fontsize=9)
    txt_iq_az   = ax_iq.text(0.97, 0.66, "Az: —°",     transform=ax_iq.transAxes,
                               ha="right", va="top", color=C_LIME,  fontsize=9)
    txt_iq_cfo  = ax_iq.text(0.97, 0.52, "CFO: — Hz",  transform=ax_iq.transAxes,
                               ha="right", va="top", color=C_VIO,   fontsize=9)

    # ── [1,0]  Az + El joint history  (twin y-axis) ──────────────────────────
    ax_az = fig.add_subplot(gs[1, 0], facecolor=BG2)
    ax_az.set_facecolor(BG2)
    ax_az.set_title("Az  +  El  history  (per valid burst)", color=C_TEXT, fontsize=9)
    ax_az.set_xlabel("Recent valid bursts →", color=C_MUT, fontsize=8)
    ax_az.set_ylabel("Az [°]", color=C_LIME, fontsize=8)
    ax_az.set_xlim(0, H); ax_az.set_ylim(0, 360)
    ax_az.set_yticks([0, 90, 180, 270, 360])
    ax_az.tick_params(axis="y", colors=C_LIME, labelsize=7)
    ax_az.tick_params(axis="x", colors=C_MUT,  labelsize=7)
    for sp in ax_az.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_az.grid(color=C_BDR, lw=0.4, alpha=0.4, zorder=1)
    az_line,     = ax_az.plot([], [], "-",  color=C_LIME, lw=1.0, alpha=0.45, zorder=3, label="az")
    az_med_line, = ax_az.plot([], [], "--", color=C_LIME, lw=0.8, alpha=0.35, zorder=2)
    kf_az_line,  = ax_az.plot([], [], "-",  color=C_LIME, lw=2.2, alpha=0.90, zorder=5, label="kf_az")
    ax_el_tw = ax_az.twinx()
    ax_el_tw.set_facecolor(BG2)
    ax_el_tw.set_ylim(el_min, el_max)
    ax_el_tw.set_ylabel("El [°]", color=C_TEAL, fontsize=8)
    ax_el_tw.tick_params(axis="y", colors=C_TEAL, labelsize=7)
    ax_el_tw.spines["right"].set_edgecolor(C_TEAL)
    el_line,     = ax_el_tw.plot([], [], "-",  color=C_TEAL, lw=1.0, alpha=0.45, zorder=3, label="el")
    el_ema_line, = ax_el_tw.plot([], [], "--", color=C_TEAL, lw=0.8, alpha=0.35, zorder=2)
    kf_el_line,  = ax_el_tw.plot([], [], "-",  color=C_TEAL, lw=2.0, alpha=0.90, zorder=5, label="kf_el")

    # ── [1,1]  Eigenvalue profile  (signal ↔ noise subspace) ─────────────────
    ax_eig = fig.add_subplot(gs[1, 1], facecolor=BG2)
    ax_eig.set_facecolor(BG2)
    ax_eig.set_title("Eigenvalue profile  (signal ↔ noise subspace)", color=C_TEXT, fontsize=9)
    ax_eig.set_xlabel("Rank  (λ₁ ≥ λ₂ ≥ … ≥ λ_N)", color=C_MUT, fontsize=8)
    ax_eig.set_ylabel("Spread above noise floor [dB]", color=C_MUT, fontsize=8)
    ax_eig.set_xlim(-0.5, n_ant - 0.5)
    ax_eig.set_xticks(range(n_ant))
    ax_eig.set_xticklabels([f"λ{i+1}" for i in range(n_ant)], fontsize=8, color=C_MUT)
    ax_eig.set_ylim(-3, 42)
    ax_eig.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_eig.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_eig.grid(axis="y", color=C_BDR, lw=0.5, alpha=0.4, zorder=1)
    ax_eig.axhline(0, color=C_MUT, lw=0.8, ls="--", alpha=0.5, zorder=2)
    # vertical line separating signal (first n_sig) from noise subspace
    if 0 < n_sig < n_ant:
        ax_eig.axvline(n_sig - 0.5, color=C_ROSE, lw=1.0, ls=":", alpha=0.75, zorder=3)
        ax_eig.text(n_sig - 0.5, 39, f"D={n_sig}",
                    ha="center", va="top", color=C_ROSE, fontsize=7)
    eig_cols = [C_AMBER if i < n_sig else "#5a618a" for i in range(n_ant)]
    eig_bars = ax_eig.bar(range(n_ant), np.zeros(n_ant),
                           color=eig_cols, edgecolor=BG2, linewidth=0.6, zorder=3)
    txt_snr_eig  = ax_eig.text(n_ant / 2, 39, "SNR: — dB",
                                ha="center", va="top", color=C_TEXT,  fontsize=9)
    txt_papr_eig = ax_eig.text(n_ant / 2, 33, "PAPR: — dB",
                                ha="center", va="top", color=C_AMBER, fontsize=9)
    txt_el_eig   = ax_eig.text(n_ant / 2, 27, "El: —°",
                                ha="center", va="top", color=C_TEAL,  fontsize=9)
    txt_burst    = ax_eig.text(n_ant / 2, 21, "bursts: 0",
                                ha="center", va="top", color=C_LIME,  fontsize=8)
    txt_frame    = ax_eig.text(n_ant / 2, 15, "frames: 0",
                                ha="center", va="top", color=C_MUT,   fontsize=8)
    txt_rec      = ax_eig.text(n_ant / 2,  9, "rec: 0",
                                ha="center", va="top", color=C_ROSE,  fontsize=8)

    # ── [1,2]  Inter-channel phase differences + CFO tracker ─────────────────
    ax_ph = fig.add_subplot(gs[1, 2], facecolor=BG2)
    ax_ph.set_facecolor(BG2)
    ax_ph.set_title("ΔΦ  CH1..4 – CH0  +  CFO tracker", color=C_TEXT, fontsize=9)
    ax_ph.set_xlabel("Recent valid bursts →", color=C_MUT, fontsize=8)
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
    ax_ph.legend(loc="upper left", fontsize=6, facecolor=BG3,
                  edgecolor=C_BDR, labelcolor=C_TEXT)
    # CFO twin-axis
    ax_cfo = ax_ph.twinx()
    ax_cfo.set_ylim(-3000, 3000)
    ax_cfo.set_ylabel("CFO [Hz]", color=C_VIO, fontsize=8)
    ax_cfo.tick_params(axis="y", colors=C_VIO, labelsize=7)
    ax_cfo.spines["right"].set_edgecolor(C_VIO)
    ax_cfo.axhline(0, color=C_VIO, lw=0.5, ls="--", alpha=0.30, zorder=1)
    cfo_line, = ax_cfo.plot([], [], "-", color=C_VIO, lw=1.5, alpha=0.80, zorder=5)

    fig.suptitle(
        f"KrakenSDR UCA {n_ant}-ant  —  {algo.upper()}  @  {freq_hz/1e6:.3f} MHz  "
        f"(r={cfg.radius_lambda:.4f}λ  el=[{el_min:.0f}°,{el_max:.0f}°]  "
        f"preamble +{_PREAMBLE_TONE_HZ} Hz  D={n_sig})",
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
            el_h    = list(S.el_hist)
            ph_h    = [list(q) for q in S.phase_hist]
            kf_az_h = list(S.kf_az_hist)
            kf_el_h = list(S.kf_el_hist)
            cfo_h   = list(S.cfo_hist)
            iq_env  = S.last_iq_env.copy()

        # ── Skyplot MUSIC lobes ────────────────────────────────────────────────
        _lin_s = 10 ** (np.clip(az_s, -40, 0) / 10.0)
        _lin_s /= (_lin_s.max() + 1e-12)
        _lobe_r = _lin_s * 80.0                     # scale to 80° radius
        _az_ext = np.r_[az_rad, az_rad[0]]
        _r_ext  = np.r_[_lobe_r, _lobe_r[0]]
        sky_spec_line.set_data(_az_ext, _r_ext)
        sky_spec_fill.set_xy(np.column_stack([_az_ext, _r_ext]))

        # ── Skyplot ────────────────────────────────────────────────────────────
        sky_r     = float(np.clip(90.0 - el, 0.0, 90.0))
        sky_theta = np.deg2rad(az)
        sky_arrow.set_data([sky_theta, sky_theta], [0.0, sky_r])
        sky_dot.set_data([sky_theta], [sky_r])
        txt_sky_az.set_text(f"{az_med:.0f}°")
        txt_sky_el.set_text(f"El: {el:.1f}°")
        n_pts = min(len(az_h), len(el_h))
        if n_pts > 0:
            pts_th = np.deg2rad(np.array(az_h[-n_pts:], dtype=float))
            pts_r  = np.clip(90.0 - np.array(el_h[-n_pts:], dtype=float), 0.0, 90.0)
            cols   = np.linspace(0.0, 1.0, n_pts)
            sky_scat.set_offsets(np.c_[pts_th, pts_r])
            sky_scat.set_array(cols)
        n_kf = min(len(kf_az_h), len(kf_el_h))
        if n_kf > 0:
            kf_th = np.deg2rad(np.array(kf_az_h[-n_kf:], dtype=float))
            kf_r  = np.clip(90.0 - np.array(kf_el_h[-n_kf:], dtype=float), 0.0, 90.0)
            sky_kf_line.set_data(kf_th, kf_r)
        else:
            sky_kf_line.set_data([], [])
        # NO-SIGNAL overlay on skyplot
        if no_sig:
            txt_sky_nosig.set_text("NO BURST"); txt_sky_nosig.set_color(C_ROSE)
            txt_sky_nosig.get_bbox_patch().set_edgecolor(C_ROSE); _a = 0.85
        elif no_doa:
            txt_sky_nosig.set_text("DIR ?"); txt_sky_nosig.set_color(C_AMBER)
            txt_sky_nosig.get_bbox_patch().set_edgecolor(C_AMBER); _a = 0.75
        else:
            _a = 0.0
        txt_sky_nosig.set_alpha(_a)
        txt_sky_nosig.get_bbox_patch().set_alpha(_a * 0.6)
        sky_arrow.set_alpha(0.15 if no_sig else (0.55 if no_doa else 1.0))
        sky_dot.set_alpha(0.15 if no_sig else (0.55 if no_doa else 1.0))

        # ── 2D heatmap ─────────────────────────────────────────────────────────
        lin2 = 10 ** (np.clip(s2d, -40, 0) / 10.0)
        lin2 /= (lin2.max() + 1e-12)
        im_2d.set_data(lin2)
        xh_v.set_xdata([az, az]);  xh_h.set_ydata([el, el])
        peak_dot.set_data([az], [el])

        # ── IQ burst oscilloscope ──────────────────────────────────────────────
        if len(iq_env) == _IQ_PTS:
            iq_line.set_ydata(iq_env)
            iq_fill_ref[0].remove()
            iq_fill_ref[0] = ax_iq.fill_between(_iq_xs, iq_env,
                                                  color=C_TEAL, alpha=0.10, zorder=2)
        txt_iq_snr.set_text(f"SNR:  {snr:+.1f} dB")
        txt_iq_papr.set_text(f"PAPR: {papr:.1f} dB")
        txt_iq_az.set_text(f"Az:   {az_med:.0f}°")
        _last_cfo = cfo_h[-1] if cfo_h else float("nan")
        txt_iq_cfo.set_text(f"CFO:  {_last_cfo:+.0f} Hz" if np.isfinite(_last_cfo) else "CFO: — Hz")

        # ── Az + El history (twin y-axis) + Kalman ─────────────────────────────
        if az_h:
            xs = np.arange(len(az_h))
            az_line.set_data(xs, az_h)
            az_med_line.set_data([0, H], [az_med, az_med])
        else:
            az_line.set_data([], []); az_med_line.set_data([], [])
        if el_h:
            xs_el = np.arange(len(el_h))
            el_line.set_data(xs_el, el_h)
            el_ema_line.set_data([0, H], [el, el])
        else:
            el_line.set_data([], []); el_ema_line.set_data([], [])
        if kf_az_h:
            kf_az_line.set_data(np.arange(len(kf_az_h)), kf_az_h)
        else:
            kf_az_line.set_data([], [])
        if kf_el_h:
            kf_el_line.set_data(np.arange(len(kf_el_h)), kf_el_h)
        else:
            kf_el_line.set_data([], [])

        # ── Eigenvalue profile ─────────────────────────────────────────────────
        for bar, v in zip(eig_bars, eig):
            bar.set_height(float(max(v, 0.0)))
        txt_snr_eig.set_text(f"SNR:  {snr:+.1f} dB")
        txt_papr_eig.set_text(f"PAPR: {papr:.1f} dB")
        txt_el_eig.set_text(f"El:   {el:.1f}°")
        txt_burst.set_text(f"bursts: {bn}")
        txt_frame.set_text(f"frames: {fn}")
        txt_rec.set_text(f"rec: {n_rec}")

        # ── Phase history + CFO ────────────────────────────────────────────────
        for line, ph_data in zip(ph_lines, ph_h):
            if ph_data:
                line.set_data(np.arange(len(ph_data)), ph_data)
            else:
                line.set_data([], [])
        if cfo_h:
            cfo_line.set_data(np.arange(len(cfo_h)), cfo_h)
        else:
            cfo_line.set_data([], [])

        return (sky_spec_line, sky_spec_fill,
                sky_scat, sky_kf_line, sky_arrow, sky_dot,
                txt_sky_az, txt_sky_nosig, txt_sky_el,
                im_2d, xh_v, xh_h, peak_dot,
                iq_line, txt_iq_snr, txt_iq_papr, txt_iq_az, txt_iq_cfo,
                az_line, az_med_line, kf_az_line, el_line, el_ema_line, kf_el_line,
                *eig_bars, txt_snr_eig, txt_papr_eig, txt_el_eig,
                txt_burst, txt_frame, txt_rec,
                *ph_lines, cfo_line)

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
    p.add_argument("--save-iq",  action="store_true",
                   help="Save raw preamble IQ windows to *_iq.npz for offline re-processing")
    p.add_argument("--rec-every", type=int, default=200, metavar="N",
                   help="Auto-save checkpoint every N bursts (0=only at exit)")
    p.add_argument("--papr-min", type=float,
                   default=float(getattr(C, "PAPR_INST_MIN_DB", _PAPR_INST_MIN_DB)),
                   help="Min instantaneous PAPR [dB] to accept a preamble burst. "
                        "Lower this (e.g. 8) for indoor / low-SNR operation.")
    args = p.parse_args()

    freq_hz = int(args.freq * 1e6)
    out_dir = getattr(args, "out_dir", None)

    cfg = UcaConfig(
        n_ant=C.N_ANTENNAS, radius_lambda=args.radius,
        n_az=C.N_AZ, n_el=C.N_EL, el_min_deg=C.EL_MIN_DEG,
        el_max_deg=float(getattr(C, "EL_MAX_DEG", 90.0)),
        num_expected_signals=args.nsig, ant0_offset_deg=args.offset,
        ant_ccw=C.ANT_CCW,
    )
    # Alpha=0 → fresh covariance each burst; moderate alpha → EMA smoothing
    acc = CovarianceAccumulatorUca(alpha=args.alpha)
    S   = _make_state(cfg.n_az, cfg.n_el)
    S.rec_enabled    = not args.no_rec
    S.rec_iq_enabled = args.save_iq
    # Override PAPR threshold at runtime (useful for indoor testing)
    C.PAPR_INST_MIN_DB = args.papr_min   # type: ignore[attr-defined]

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
