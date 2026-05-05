#!/usr/bin/env python3
"""
doa_test_868_realtime.py — Real-time 2D DoA at 868 MHz on KrakenSDR 5-element UCA
==================================================================================
Receives IQ from Heimdall DAQ and estimates the azimuth of a CW beacon using
2D-MUSIC.  Display shows:
  - Polar compass: MUSIC spectrum collapsed to azimuth + direction arrow
  - 2D heatmap: azimuth × elevation spectrum with crosshair at peak
  - Quality panel: eigenvalues, SNR, PAPR, estimated azimuth + elevation
  - Rolling history: azimuth, elevation, inter-channel phase differences

Pilot-tone mode
---------------
With PILOT_TONE_ENABLED=True in config.py, a narrow-band FFT gate is applied
around PILOT_TONE_OFFSET_HZ before computing the covariance.  This rejects
broadband noise, gaining ~20 dB of effective SNR and greatly improving PAPR.
The LibreSDR must transmit at  LO_freq + PILOT_TONE_OFFSET_HZ (default 100 kHz)
via  python3 tx_868_libresdr.py --pilot-offset 100000.

Usage
-----
    python3 doa_test_868_realtime.py              # hardware (Heimdall)
    python3 doa_test_868_realtime.py --demo       # synthetic signal at 45° (no HW)
    python3 doa_test_868_realtime.py --algo capon
    python3 doa_test_868_realtime.py --offset 45  # antenna-0 calibration offset
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
from core.gates import circ_median_deg

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
C_BDR = "#3b4263"; C_MUT = "#8891b0"; C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIO  = "#a78bfa"; C_ROSE = "#f16b6f"; C_LIME  = "#6dd97d"

_PAPR_MIN_DB  = 4.0   # below this threshold with low eigenvalue: direction unreliable
_PAPR_FLAT_DB = 3.0   # below this threshold MUSIC is flat → auto-fallback to Bartlett
_SPEC_EMA     = 0.10  # display spectrum temporal smoothing (lower = more stable)

# Circular median — implementation lives in core.gates.
_circ_median = circ_median_deg


def _circ_distance(a_deg: float, b_deg: float) -> float:
    """Shortest angular distance between two azimuths [0..360°]."""
    d = abs(a_deg - b_deg) % 360.0
    return min(d, 360.0 - d)


def _select_algo_by_snr(user_algo: str, snr_db: float) -> str:
    """SNR-adaptive algorithm selection."""
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


def _check_phase_coherence(
    phase_diffs: np.ndarray,
    phase_history: list[collections.deque],
    max_jump_deg: float,
) -> bool:
    """Return True if phase diffs are coherent with recent history."""
    min_hist = 3
    for i, ph in enumerate(phase_diffs):
        hist = phase_history[i]
        if len(hist) < min_hist:
            continue
        med = _circ_median(np.array(hist))
        if _circ_distance(float(ph), med) > max_jump_deg:
            return False
    return True


# =============================================================================
# Heimdall pre-flight
# =============================================================================

def _check_heimdall(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except OSError:
        return False


# =============================================================================
# Stato condiviso
# =============================================================================

def _make_state(n_az: int, n_el: int) -> SimpleNamespace:
    return SimpleNamespace(
        az_spec    = np.full(n_az, -40.0),           # 1D azimuth spectrum [dB]
        spec2d     = np.full((n_el, n_az), -40.0),   # 2D spectrum (n_el, n_az) [dB]
        az_deg     = 0.0,                            # instantaneous azimuth estimate [°]
        az_median  = 0.0,                            # circular median of last N valid frames [°]
        el_deg     = 0.0,                            # estimated elevation [°]
        phase_diffs= np.zeros(4),                    # angle(R[1:5,0]) [°]
        papr_db    = 0.0,
        snr_db     = 0.0,
        eig_db     = np.zeros(5),
        az_hist    = collections.deque(maxlen=C.HISTORY_LEN),
        el_hist    = collections.deque(maxlen=C.HISTORY_LEN),
        snr_hist   = collections.deque(maxlen=C.HISTORY_LEN),
        phase_hist = [collections.deque(maxlen=C.HISTORY_LEN) for _ in range(4)],
        frame_n    = 0,
        no_signal  = True,    # True = no signal detected (EIG below threshold)
        no_doa     = True,    # True = signal present but direction unreliable (PAPR low)
        lock       = threading.Lock(),
        running    = True,
        rec_enabled= True,    # False when --no-rec is passed
        # ── Recording buffers (accumulated for the full session) ──────────────
        rec_t       = [],   # UNIX timestamp float64
        rec_az      = [],   # estimated azimuth [°]
        rec_el      = [],   # estimated elevation [°]
        rec_papr    = [],   # PAPR [dB]
        rec_snr     = [],   # SNR [dB]
        rec_eig     = [],   # list of (5,) float: eigenvalue spreads [dB]
        rec_phase   = [],   # list of (4,) float: ΔΦ CH1..4 vs CH0 [°]
        rec_R       = [],   # list of (5,5) complex128: EMA covariance
        rec_has_sig = [],   # bool: True = valid signal + reliable direction
    )


# =============================================================================
# Thread acquisizione + DoA
# =============================================================================

def _acq_loop(src, cfg: UcaConfig, algo: str,
              acc: CovarianceAccumulatorUca, S: SimpleNamespace,
              demo: bool = False) -> None:
    rng      = np.random.default_rng(42)
    # Demo source at 45° azimuth (East-NorthEast), 10° elevation — matches
    # typical indoor LibreSDR bench test at 45° from array North.
    demo_az  = np.deg2rad(45.0)
    demo_el  = np.deg2rad(10.0)
    pos      = cfg.positions

    # Pilot-tone extraction parameters (from config)
    _pilot_enabled = getattr(C, "PILOT_TONE_ENABLED",   False)
    _pilot_hz      = float(getattr(C, "PILOT_TONE_OFFSET_HZ", 100_000))
    _pilot_bw      = float(getattr(C, "PILOT_TONE_BW_HZ",     10_000))
    _amp_norm      = getattr(C, "AMPLITUDE_NORMALIZE",  True)
    _sample_rate   = float(getattr(C, "SAMPLE_RATE_HZ", 1_024_000))

    while S.running:
        # ── IQ frame ─────────────────────────────────────────────────────────
        if demo:
            N = 65536
            tau = 2 * np.pi * (pos[:, 0] * np.cos(demo_el) * np.sin(demo_az)
                                + pos[:, 1] * np.cos(demo_el) * np.cos(demo_az))
            # Demo: pure pilot tone at demo_az + per-channel gain imbalance +
            # random phase offsets to simulate realistic hardware conditions.
            phase_offsets = np.deg2rad([0.0, 5.0, -8.0, 12.0, -3.0])
            gain_offsets  = np.array([1.0, 0.92, 1.08, 0.95, 1.03])
            # Pilot tone is at PILOT_TONE_OFFSET_HZ in the demo spectrum
            t = np.arange(N, dtype=np.float64)
            pilot_phasor = np.exp(2j * np.pi * _pilot_hz / _sample_rate * t)
            snr_linear = 10 ** (15.0 / 10.0)   # 15 dB SNR on the pilot
            s = pilot_phasor * np.sqrt(snr_linear)
            X = (
                np.outer(
                    gain_offsets * np.exp(1j * (tau + phase_offsets)),
                    s
                )
                + (rng.standard_normal((cfg.n_ant, N))
                   + 1j * rng.standard_normal((cfg.n_ant, N))) / np.sqrt(2)
            )
            X = X.astype(np.complex128)
            time.sleep(0.05)
        else:
            frame = src.get_frame(timeout=2.0)
            if frame is None:
                continue
            X = frame.astype(np.complex128)

        if X.shape[0] != cfg.n_ant:
            continue

        pwr_db = float(10 * np.log10(np.mean(np.abs(X)**2) + 1e-20))
        if C.SQUELCH_ENABLED and pwr_db < C.SQUELCH_THRESHOLD_DB:
            continue

        try:
            # ── Pilot tone extraction ─────────────────────────────────────────
            if C.PILOT_TONE_ENABLED:
                X_proc = extract_pilot_tone(
                    X, float(C.SAMPLE_RATE_HZ),
                    tone_hz=float(C.PILOT_TONE_OFFSET_HZ),
                    bw_hz=float(getattr(C, 'PILOT_TONE_BW_HZ', 10_000.0)),
                )
            else:
                X_proc = X

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

            # ── EMA covariance ────────────────────────────────────────────────
            R = acc.update(X_proc)

            # High-elevation fallback: at cos(el)→0 inter-channel phase spread
            # shrinks → MUSIC spectrum becomes flat → switch to Bartlett.
            _el_now    = S.el_deg
            _use_algo  = algo
            if _el_now >= C.HIGH_EL_THRESHOLD_DEG and algo in ("music", "capon"):
                _use_algo = C.HIGH_EL_ALGO.lower()

            if _use_algo == "capon":
                spec2d = doa_capon_uca_2d(X_proc, cfg, R_in=R,
                                          decorr=getattr(C, "CAPNT_DECORR", "none"))
            elif _use_algo == "bartlett":
                spec2d = doa_bartlett_uca_2d(X_proc, cfg, R_in=R)
            elif _use_algo == "root-music":
                spec2d = doa_root_music_uca_2d(R, cfg)[0]  # Get spectrum from tuple
            elif _use_algo == "unitary-esprit":
                spec2d = doa_unitary_esprit_uca_2d(R, cfg)[0]  # Get spectrum from tuple
            elif _use_algo == "mfba-music":
                spec2d = doa_mfba_music_uca_2d(R, cfg)[0]  # Get spectrum from tuple
            else:
                spec2d = doa_music_uca_2d(X_proc, cfg, R_in=R,
                                          decorr=getattr(C, "MUSIC_DECORR", "none"))

            # Collapse elevation scan → 1-D azimuth spectrum (max over elevation)
            az_spec = np.max(spec2d, axis=0)

            az, el_est, papr = find_peak_uca_2d(spec2d, cfg)

            # Auto-fallback to Bartlett when spectrum is flat (high-el or low PAPR)
            if papr < _PAPR_FLAT_DB and _use_algo != "bartlett":
                spec2d_b = doa_bartlett_uca_2d(X_proc, cfg, R_in=R)
                az_b, el_b, papr_b = find_peak_uca_2d(spec2d_b, cfg)
                if papr_b > papr:
                    spec2d, az, el_est, papr = spec2d_b, az_b, el_b, papr_b
                    az_spec = np.max(spec2d, axis=0)

            eig         = eigenvalue_spread_uca_db(R)
            snr         = snr_uca_db(R)
            phase_diffs = np.degrees(np.angle(R[1:, 0]))   # (4,) ΔΦ CH1..4 vs CH0
        except Exception as exc:
            print(f"[DoA] frame #{S.frame_n+1}: {exc}")
            continue

        # Signal detection: eigenvalue spread is robust to elevation angle
        # (unlike PAPR, which drops at high el where the spectrum flattens).
        has_signal = float(eig[0]) >= C.EIG_SPREAD_MIN_DB
        has_doa    = has_signal and papr >= _PAPR_MIN_DB

        with S.lock:
            S.az_spec    = (1 - _SPEC_EMA) * S.az_spec + _SPEC_EMA * az_spec
            S.spec2d     = (1 - _SPEC_EMA) * S.spec2d  + _SPEC_EMA * spec2d
            S.az_deg     = az
            S.el_deg     = el_est
            S.phase_diffs= phase_diffs
            S.papr_db    = papr
            S.snr_db     = snr
            S.eig_db     = eig
            S.no_signal  = not has_signal
            S.no_doa     = not has_doa
            S.el_hist.append(el_est)
            S.snr_hist.append(snr)
            for _i, _p in enumerate(phase_diffs):
                S.phase_hist[_i].append(float(_p))
            if has_doa:
                S.az_hist.append(az)
                if len(S.az_hist) >= 3:
                    S.az_median = _circ_median(np.array(S.az_hist))
                else:
                    S.az_median = az
            # ── Recording (every frame, valid or not) ─────────────────────────
            if S.rec_enabled:
                S.rec_t.append(time.time())
                S.rec_az.append(float(az))
                S.rec_el.append(float(el_est))
                S.rec_papr.append(float(papr))
                S.rec_snr.append(float(snr))
                S.rec_eig.append(eig.copy())
                S.rec_phase.append(phase_diffs.copy())
                S.rec_R.append(np.array(R, dtype=np.complex128).copy())
                S.rec_has_sig.append(bool(has_doa))
            S.frame_n += 1


# =============================================================================
# UI: 6 panels — polar compass, 2D heatmap, eigenvalues, az/el history, phases
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
        f"DoA CW {freq_hz/1e6:.3f} MHz — {algo.upper()}")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[1.4, 1.0],
                           left=0.05, right=0.97,
                           top=0.93, bottom=0.07,
                           hspace=0.42, wspace=0.33)

    # ── [0,0]  Bussola polare azimuth ─────────────────────────────────────────
    ax_pol = fig.add_subplot(gs[0, 0], projection="polar", facecolor=BG2)
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_ylim(0, 1)
    ax_pol.set_yticks([])
    ax_pol.tick_params(colors=C_MUT, labelsize=7)
    ax_pol.set_facecolor(BG2)
    for sp in ax_pol.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_pol.set_title("Azimuth DoA", color=C_TEXT, fontsize=9, pad=10)

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
        0.5, 0.5, "NO SIGNAL", transform=ax_pol.transAxes,
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=C_ROSE, alpha=0.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=C_ROSE, alpha=0.0),
        zorder=10)

    # ── [0,1]  Mappa 2D az × el ───────────────────────────────────────────────
    ax_2d = fig.add_subplot(gs[0, 1], facecolor=BG2)
    ax_2d.set_facecolor(BG2)
    ax_2d.set_xlabel("Azimuth [°]", color=C_MUT, fontsize=8)
    ax_2d.set_ylabel("Elevation [°]", color=C_MUT, fontsize=8)
    ax_2d.set_title("2D spectrum  az × el", color=C_TEXT, fontsize=9)
    ax_2d.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_2d.spines.values():
        sp.set_edgecolor(C_BDR)
    im_2d = ax_2d.imshow(
        np.zeros((n_el, n_az)),
        origin="lower",
        extent=[0, 360, el_min, el_max],
        aspect="auto",
        cmap="plasma",
        vmin=0.0, vmax=1.0,
        interpolation="bilinear",
    )
    xh_v, = ax_2d.plot([0, 0],     [el_min, el_max], "--", color=C_LIME, lw=0.9, alpha=0.7)
    xh_h, = ax_2d.plot([0, 360],   [el_min, el_min], "--", color=C_LIME, lw=0.9, alpha=0.7)
    peak_dot, = ax_2d.plot([0], [el_min], "o", color=C_LIME, ms=6, zorder=6)

    # ── [0,2]  Eigenvalues + readout ──────────────────────────────────────────
    ax_q = fig.add_subplot(gs[0, 2], facecolor=BG2)
    ax_q.set_facecolor(BG2)
    ax_q.set_title("Eigenvalues + quality", color=C_TEXT, fontsize=9)
    ax_q.set_xlabel("Canale", color=C_MUT, fontsize=8)
    ax_q.set_ylabel("Spread [dB]", color=C_MUT, fontsize=8)
    ax_q.set_xlim(-0.5, 4.5)
    ax_q.set_xticks(range(5))
    ax_q.set_ylim(-3, 35)
    ax_q.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_q.spines.values():
        sp.set_edgecolor(C_BDR)
    bars = ax_q.bar(range(5), np.zeros(5),
                    color=[C_BLUE, C_TEAL, C_AMBER, C_VIO, C_ROSE],
                    edgecolor=BG2, linewidth=0.5, zorder=3)
    ax_q.axhline(0, color=C_BDR, lw=0.8, zorder=2)
    ax_q.grid(axis="y", color=C_BDR, lw=0.5, alpha=0.4, zorder=1)
    txt_snr   = ax_q.text(2, 32, "SNR: — dB",   ha="center", va="top",
                           color=C_TEXT,  fontsize=9)
    txt_papr  = ax_q.text(2, 29, "PAPR: — dB",  ha="center", va="top",
                           color=C_AMBER, fontsize=9)
    txt_el_q  = ax_q.text(2, 26, "El:  — °",    ha="center", va="top",
                           color=C_TEAL,  fontsize=9)
    txt_inst  = ax_q.text(2, 23, "inst az: —°", ha="center", va="top",
                           color=C_AMBER, fontsize=8, alpha=0.75)
    txt_frame = ax_q.text(2, 20, "frame: 0",    ha="center", va="top",
                           color=C_MUT, fontsize=8)
    txt_rec   = ax_q.text(2, 17, "rec: 0",      ha="center", va="top",
                           color=C_ROSE, fontsize=8)

    # ── [1,0]  Storia azimuth rolling ─────────────────────────────────────────
    ax_az = fig.add_subplot(gs[1, 0], facecolor=BG2)
    ax_az.set_facecolor(BG2)
    ax_az.set_title("Azimuth history", color=C_TEXT, fontsize=9)
    ax_az.set_xlabel("Recent frames →", color=C_MUT, fontsize=8)
    ax_az.set_ylabel("Az [°]", color=C_MUT, fontsize=8)
    ax_az.set_xlim(0, H)
    ax_az.set_ylim(0, 360)
    ax_az.set_yticks([0, 90, 180, 270, 360])
    ax_az.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_az.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_az.grid(color=C_BDR, lw=0.4, alpha=0.4)
    az_line,     = ax_az.plot([], [], "-",  color=C_LIME,       lw=1.4)
    az_med_line, = ax_az.plot([], [], "--", color=C_LIME,       lw=0.8, alpha=0.45)

    # ── [1,1]  Elevation history ──────────────────────────────────────────────
    ax_el = fig.add_subplot(gs[1, 1], facecolor=BG2)
    ax_el.set_facecolor(BG2)
    ax_el.set_title("Elevation history", color=C_TEXT, fontsize=9)
    ax_el.set_xlabel("Recent frames →", color=C_MUT, fontsize=8)
    ax_el.set_ylabel("El [°]", color=C_MUT, fontsize=8)
    ax_el.set_xlim(0, H)
    ax_el.set_ylim(el_min, el_max)
    ax_el.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_el.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_el.grid(color=C_BDR, lw=0.4, alpha=0.4)
    el_line, = ax_el.plot([], [], "-", color=C_TEAL, lw=1.4)

    # ── [1,2]  Differenze di fase inter-canale ────────────────────────────────
    ax_ph = fig.add_subplot(gs[1, 2], facecolor=BG2)
    ax_ph.set_facecolor(BG2)
    ax_ph.set_title("ΔΦ  CH1..4 – CH0  (from covariance matrix)", color=C_TEXT, fontsize=9)
    ax_ph.set_xlabel("Recent frames →", color=C_MUT, fontsize=8)
    ax_ph.set_ylabel("ΔΦ [°]", color=C_MUT, fontsize=8)
    ax_ph.set_xlim(0, H)
    ax_ph.set_ylim(-185, 185)
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
        f"KrakenSDR UCA 5-ant  —  {algo.upper()}  @  {freq_hz/1e6:.3f} MHz  "
        f"(r={cfg.radius_lambda:.4f}λ  ant0={cfg.ant0_offset_deg:.0f}°  "
        f"{'CCW' if cfg.ant_ccw else 'CW'})",
        color=C_TEXT, fontsize=9, y=0.98,
    )

    # ── Funzione di aggiornamento ──────────────────────────────────────────────
    def _update(_):
        with S.lock:
            az_s     = S.az_spec.copy()
            s2d      = S.spec2d.copy()
            az       = S.az_deg
            az_med   = S.az_median
            el       = S.el_deg
            phase    = S.phase_diffs.copy()
            papr     = S.papr_db
            snr      = S.snr_db
            eig      = S.eig_db.copy()
            no_sig   = S.no_signal
            no_doa   = S.no_doa
            fn       = S.frame_n
            n_rec    = len(S.rec_t)
            az_h     = list(S.az_hist)
            el_h     = list(S.el_hist)
            ph_h     = [list(q) for q in S.phase_hist]

        # Bussola polare ───────────────────────────────────────────────────────
        lin = 10 ** (np.clip(az_s, -40, 0) / 10.0)
        lin /= (lin.max() + 1e-12)
        spec_line.set_data(np.r_[az_rad, az_rad[0]], np.r_[lin, lin[0]])
        th_med  = np.deg2rad(az_med)
        arrow_line.set_data([th_med, th_med], [0, 0.92])
        arrow_dot.set_data([th_med], [0.92])
        txt_az.set_text(f"{az_med:.0f}°")
        th_inst = np.deg2rad(az)
        inst_line.set_data([th_inst, th_inst], [0, 0.78])

        # Heatmap 2D ───────────────────────────────────────────────────────────
        lin2 = 10 ** (np.clip(s2d, -40, 0) / 10.0)
        lin2 /= (lin2.max() + 1e-12)
        im_2d.set_data(lin2)
        xh_v.set_xdata([az, az])
        xh_h.set_ydata([el, el])
        peak_dot.set_data([az], [el])

        # Autovalori + readout ─────────────────────────────────────────────────
        for bar, v in zip(bars, eig):
            bar.set_height(float(v))
        txt_snr.set_text(f"SNR:  {snr:+.1f} dB")
        txt_papr.set_text(f"PAPR: {papr:.1f} dB")
        txt_el_q.set_text(f"El:   {el:.0f}°")
        txt_inst.set_text(f"inst az: {az:.0f}°")
        txt_frame.set_text(f"frame: {fn}")
        txt_rec.set_text(f"rec: {n_rec}")

        # Overlay stato segnale: NO SIGNAL (rosso) / DIR? (ambra) / ok (nascosto)
        if no_sig:
            txt_nosig.set_text("NO SIGNAL")
            txt_nosig.set_color(C_ROSE)
            txt_nosig.get_bbox_patch().set_edgecolor(C_ROSE)
            a = 0.85
        elif no_doa:
            txt_nosig.set_text("DIR ?")
            txt_nosig.set_color(C_AMBER)
            txt_nosig.get_bbox_patch().set_edgecolor(C_AMBER)
            a = 0.75
        else:
            a = 0.0
        txt_nosig.set_alpha(a)
        txt_nosig.get_bbox_patch().set_alpha(a * 0.6)
        arrow_line.set_alpha(0.15 if no_sig else (0.55 if no_doa else 1.0))
        arrow_dot.set_alpha(0.15 if no_sig else (0.55 if no_doa else 1.0))
        inst_line.set_alpha(0.08 if no_sig else (0.30 if no_doa else 0.55))

        # Azimuth history ──────────────────────────────────────────────────────
        if az_h:
            xs = np.arange(len(az_h))
            az_line.set_data(xs, az_h)
            az_med_line.set_data([0, H], [az_med, az_med])
        else:
            az_line.set_data([], [])
            az_med_line.set_data([], [])

        # Elevazione history ───────────────────────────────────────────────────
        if el_h:
            xs = np.arange(len(el_h))
            el_line.set_data(xs, el_h)
        else:
            el_line.set_data([], [])

        # Phase differences history ────────────────────────────────────────────
        for line, ph_data in zip(ph_lines, ph_h):
            if ph_data:
                xs = np.arange(len(ph_data))
                line.set_data(xs, ph_data)
            else:
                line.set_data([], [])

        return (spec_line, arrow_line, arrow_dot, inst_line, txt_az, txt_nosig,
                im_2d, xh_v, xh_h, peak_dot,
                *bars, txt_snr, txt_papr, txt_el_q, txt_inst, txt_frame, txt_rec,
                az_line, az_med_line, el_line, *ph_lines)

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
# Salvataggio recording
# =============================================================================

def _save_recording(
    S: SimpleNamespace,
    freq_hz: int,
    algo: str,
    out_dir: str | None = None,
    label: str = "doa_data",
) -> str | None:
    """
    Save the accumulated session buffer to a compressed .npz file.

    Parameters
    ----------
    out_dir : directory to write into (default: same directory as this script).
    label   : filename prefix, e.g.  "doa_data" → doa_data_20260428_153201.npz
              Use "doa_checkpoint" for periodic auto-saves.
    """
    import datetime
    n = len(S.rec_t)
    if n == 0:
        print("[REC] No frames recorded — file not saved.")
        return None
    if out_dir is None:
        out_dir = _DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{label}_{ts}.npz")
    np.savez_compressed(
        path,
        timestamps  = np.array(S.rec_t,       dtype=np.float64),   # (N,)
        az_deg      = np.array(S.rec_az,       dtype=np.float32),   # (N,)
        el_deg      = np.array(S.rec_el,       dtype=np.float32),   # (N,)
        papr_db     = np.array(S.rec_papr,     dtype=np.float32),   # (N,)
        snr_db      = np.array(S.rec_snr,      dtype=np.float32),   # (N,)
        eig_db      = np.array(S.rec_eig,      dtype=np.float32),   # (N, 5)
        phase_diffs = np.array(S.rec_phase,    dtype=np.float32),   # (N, 4) [deg]
        R_real      = np.real(np.array(S.rec_R, dtype=np.complex128)),  # (N, 5, 5)
        R_imag      = np.imag(np.array(S.rec_R, dtype=np.complex128)),  # (N, 5, 5)
        has_signal  = np.array(S.rec_has_sig,  dtype=bool),          # (N,)
        freq_hz     = np.int64(freq_hz),
        algo        = np.bytes_(algo.encode()),
    )
    n_sig = int(np.sum(S.rec_has_sig))
    print(f"[REC] {n} frames ({n_sig} valid, {n_sig*100//max(n,1)}%) → {path}")
    return path


# =============================================================================
# Periodic auto-save
# =============================================================================

def _autosave_loop(
    S: SimpleNamespace, freq_hz: int, algo: str,
    out_dir: str | None, rec_every: int,
) -> None:
    """
    Background thread: write a checkpoint .npz every *rec_every* new frames.

    The checkpoint overwrites the previous file (same basename with a fresh
    timestamp) so disk space stays bounded.  Full session file is always
    written at clean exit by main().
    """
    last_n = 0
    while S.running:
        time.sleep(5)   # poll every 5 s
        with S.lock:
            n = len(S.rec_t)
        if n >= last_n + rec_every and S.running:
            _save_recording(S, freq_hz, algo, out_dir=out_dir, label="doa_checkpoint")
            last_n = n


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="CW DoA at 868 MHz — KrakenSDR 5-element UCA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq",   type=float, default=C.FREQ_HZ / 1e6, help="RF centre frequency [MHz]")
    p.add_argument("--gain",   type=float, default=C.GAIN_DB,        help="IF gain [dB]")
    p.add_argument("--radius", type=float, default=C.RADIUS_LAMBDA,  help="UCA radius in wavelengths")
    p.add_argument("--offset", type=float, default=C.ANT0_OFFSET_DEG,
                   help="Antenna-0 offset from North [deg]")
    p.add_argument("--algo",   choices=["music", "capon", "bartlett"],
                   default=C.DOA_ALGORITHM.lower(), help="DoA algorithm")
    p.add_argument("--nsig",   type=int,   default=C.NUM_SIGNALS,    help="Expected signal sources")
    p.add_argument("--alpha",  type=float, default=C.COV_ALPHA,      help="EMA covariance factor")
    p.add_argument("--demo",   action="store_true",
                   help="Synthetic CW at az=45° (no hardware required)")
    p.add_argument("--out-dir",  default=None, metavar="DIR",
                   help="Directory to write .npz recordings "
                        "(default: same directory as this script)")
    p.add_argument("--no-rec",   action="store_true",
                   help="Disable recording (no .npz written)")
    p.add_argument("--rec-every", type=int, default=300, metavar="N",
                   help="Auto-save checkpoint every N frames; 0 = only at exit")
    args = p.parse_args()

    freq_hz = int(args.freq * 1e6)

    cfg = UcaConfig(
        n_ant=C.N_ANTENNAS, radius_lambda=args.radius,
        n_az=C.N_AZ, n_el=C.N_EL, el_min_deg=C.EL_MIN_DEG,
        num_expected_signals=args.nsig, ant0_offset_deg=args.offset,
        ant_ccw=C.ANT_CCW,
    )
    acc = CovarianceAccumulatorUca(alpha=args.alpha)
    S   = _make_state(cfg.n_az, cfg.n_el)
    S.rec_enabled = not args.no_rec
    out_dir = getattr(args, "out_dir", None)

    pilot_str = (
        f"pilot-tone +{C.PILOT_TONE_OFFSET_HZ/1e3:.0f} kHz  (bw {C.PILOT_TONE_BW_HZ/1e3:.0f} kHz)"
        if getattr(C, "PILOT_TONE_ENABLED", False) and getattr(C, "PILOT_TONE_OFFSET_HZ", 0) != 0
        else "broadband (no pilot tone)"
    )
    print("=" * 58)
    print(f"  DoA CW — {args.algo.upper()}  @  {freq_hz/1e6:.3f} MHz")
    print(f"  UCA: {cfg.n_ant} ant  r={args.radius:.3f}λ  offset={args.offset:.1f}°")
    print(f"  Heimdall: {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}")
    print(f"  Mode: {pilot_str}")
    print(f"  Amplitude normalise: {getattr(C, 'AMPLITUDE_NORMALIZE', True)}")
    if getattr(C, "SNR_ADAPTIVE_ENABLED", False):
        print(f"  SNR-adaptive algo: BARTLETT<{C.SNR_LOW_DB:.0f}dB / CAPON / {args.algo.upper()}>{C.SNR_HIGH_DB:.0f}dB")
    if getattr(C, "PHASE_COHERENCE_ENABLED", False):
        print(f"  Phase coherence gate: ±{C.PHASE_COHERENCE_MAX_JUMP_DEG:.0f}°")
    if getattr(C, "AZ_OUTLIER_ENABLED", False):
        print(f"  Az outlier rejection: ±{C.AZ_OUTLIER_MAX_DEV_DEG:.0f}° from median")
    if S.rec_enabled:
        rec_dir_str = out_dir or os.path.dirname(os.path.abspath(__file__))
        print(f"  Recording: every {args.rec_every} frames → {rec_dir_str}")
    else:
        print("  Recording: DISABLED (--no-rec)")

    if not args.demo and not _check_heimdall(C.HEIMDALL_HOST, C.HEIMDALL_PORT):
        print(f"\n[ERROR] Heimdall not reachable at {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}")
        print("  Start Heimdall first, or use --demo to test without hardware.")
        sys.exit(1)

    if args.demo:
        src = None
        print("  DEMO: synthetic CW source at 45° azimuth")
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

    # Auto-save background thread
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
