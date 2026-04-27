#!/usr/bin/env python3
"""
doa_test_868_realtime.py — DoA CW @ 868 MHz su UCA KrakenSDR 5 antenne
=======================================================================
Riceve IQ da Heimdall DAQ e stima l'azimuth del beacon CW con MUSIC 2D.
Il display mostra:
  - Bussola polare: spettro MUSIC collassato sull'azimuth + freccia stima
  - Pannello qualità: autovalori, SNR, PAPR, azimuth stimato

Utilizzo:
    python3 doa_test_868_realtime.py              # hardware (Heimdall)
    python3 doa_test_868_realtime.py --demo        # senza hardware
    python3 doa_test_868_realtime.py --algo capon
    python3 doa_test_868_realtime.py --offset 45  # calibrazione ant0
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
    find_peak_uca_2d, eigenvalue_spread_uca_db, snr_uca_db,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
C_BDR = "#3b4263"; C_MUT = "#8891b0"; C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIO  = "#a78bfa"; C_ROSE = "#f16b6f"; C_LIME  = "#6dd97d"

_PAPR_MIN_DB  = 5.0   # sotto questa soglia (con eig basso): direzione inaffidabile
_PAPR_FLAT_DB = 3.0   # sotto questa soglia MUSIC è piatto → auto-fallback Bartlett
_SPEC_EMA     = 0.10  # smoothing temporale dello spettro (più basso = display stabile)


def _circ_median(angles_deg: np.ndarray) -> float:
    """
    Mediana circolare degli azimuth [0..360°].
    Ruota al centro della distribuzione, calcola la mediana scalare, deruota.
    Più robusta degli outlier rispetto alla media circolare.
    """
    if len(angles_deg) == 0:
        return 0.0
    a = np.deg2rad(angles_deg)
    # Centro: angolo della media circolare
    mu = float(np.angle(np.mean(np.exp(1j * a))))
    # Residui in [-π, π] rispetto al centro
    residui = np.angle(np.exp(1j * (a - mu)))
    return float(np.degrees(mu + np.median(residui)) % 360.0)


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
        az_spec    = np.full(n_az, -40.0),           # spettro 1D azimuth [dB]
        spec2d     = np.full((n_el, n_az), -40.0),   # spettro 2D (n_el, n_az) [dB]
        az_deg     = 0.0,                            # stima istantanea
        az_median  = 0.0,                            # mediana circolare ultimi N frame
        el_deg     = 0.0,                            # elevazione stimata [°]
        phase_diffs= np.zeros(4),                    # angle(R[1:5,0]) in gradi
        papr_db    = 0.0,
        snr_db     = 0.0,
        eig_db     = np.zeros(5),
        az_hist    = collections.deque(maxlen=C.HISTORY_LEN),
        el_hist    = collections.deque(maxlen=C.HISTORY_LEN),
        snr_hist   = collections.deque(maxlen=C.HISTORY_LEN),
        phase_hist = [collections.deque(maxlen=C.HISTORY_LEN) for _ in range(4)],
        frame_n    = 0,
        no_signal  = True,    # True = nessun segnale (EIG < soglia)
        no_doa     = True,    # True = segnale presente ma direzione inaffidabile (PAPR basso)
        lock       = threading.Lock(),
        running    = True,
        # ── Recording buffers (accumulati per tutta la sessione) ──────────────
        rec_t       = [],   # timestamp UNIX float64
        rec_az      = [],   # azimuth stimato [°]
        rec_el      = [],   # elevazione stimata [°]
        rec_papr    = [],   # PAPR [dB]
        rec_snr     = [],   # SNR [dB]
        rec_eig     = [],   # list of (5,) float: eigenvalue spreads [dB]
        rec_phase   = [],   # list of (4,) float: ΔΦ CH1..4 vs CH0 [°]
        rec_R       = [],   # list of (5,5) complex128: covariance EMA
        rec_has_sig = [],   # bool: frame con segnale valido
    )


# =============================================================================
# Thread acquisizione + DoA
# =============================================================================

def _acq_loop(src, cfg: UcaConfig, algo: str,
              acc: CovarianceAccumulatorUca, S: SimpleNamespace,
              demo: bool = False) -> None:
    rng      = np.random.default_rng(42)
    demo_az  = np.deg2rad(90.0)
    demo_el  = np.deg2rad(5.0)
    pos      = cfg.positions

    while S.running:
        # ── IQ frame ─────────────────────────────────────────────────────────
        if demo:
            N = 65536
            tau = 2 * np.pi * (pos[:, 0] * np.cos(demo_el) * np.sin(demo_az)
                                + pos[:, 1] * np.cos(demo_el) * np.cos(demo_az))
            s = np.exp(1j * (2 * np.pi * 0.05 * np.arange(N) + rng.uniform(0, 2*np.pi)))
            X = (np.exp(1j * tau)[:, None] * s[None, :] * np.sqrt(10**(15/10))
                 + (rng.standard_normal((cfg.n_ant, N))
                    + 1j * rng.standard_normal((cfg.n_ant, N))) / np.sqrt(2))
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
            R = acc.update(X)
            # Ad alta elevazione (cos(el)→0 → |Δτ|<<2π) passa a Bartlett:
            #   1. se el_est del frame precedente supera HIGH_EL_THRESHOLD_DEG
            #   2. oppure se lo spettro risulta piatto (PAPR < _PAPR_FLAT_DB)
            _el_now = S.el_deg   # stima dell'iterazione precedente (thread-safe: lettura float atomica)
            if _el_now >= C.HIGH_EL_THRESHOLD_DEG and algo in ("music", "capon"):
                _use_algo = C.HIGH_EL_ALGO.lower()
            else:
                _use_algo = algo
            if _use_algo == "capon":
                spec2d = doa_capon_uca_2d(X, cfg, R_in=R,
                                          decorr=getattr(C, "CAPNT_DECORR", "none"))
            elif _use_algo == "bartlett":
                spec2d = doa_bartlett_uca_2d(X, cfg, R_in=R)
            else:
                spec2d = doa_music_uca_2d(X, cfg, R_in=R,
                                          decorr=getattr(C, "MUSIC_DECORR", "none"))

            # Collassa la scansione elevazione → spettro 1D azimuth (max su el)
            az_spec = np.max(spec2d, axis=0)       # (n_az,)

            az, el_est, papr = find_peak_uca_2d(spec2d, cfg)

            # Auto-fallback Bartlett se spettro piatto:
            # elevazione alta → cos(θ)→0 → steering vector quasi costante in az
            # → MUSIC/Capon producono spettro omogeneo → PAPR crolla.
            # Bartlett (beamformer) degrada con lobo largo ma mantiene un picco.
            if papr < _PAPR_FLAT_DB and _use_algo != "bartlett":
                spec2d_b = doa_bartlett_uca_2d(X, cfg, R_in=R)
                az_b, el_b, papr_b = find_peak_uca_2d(spec2d_b, cfg)
                if papr_b > papr:
                    spec2d, az, el_est, papr = spec2d_b, az_b, el_b, papr_b
                    az_spec = np.max(spec2d, axis=0)

            eig = eigenvalue_spread_uca_db(R)
            snr = snr_uca_db(R)
            phase_diffs = np.degrees(np.angle(R[1:, 0]))   # (4,) ΔΦ CH1..4 vs CH0 [°]
        except Exception as exc:
            print(f"[DoA] frame #{S.frame_n+1}: {exc}")
            continue

        # Rilevazione segnale: usa l'autovalore (robusto all'elevazione),
        # non il PAPR che crolla quando cos(el)→0 rende lo spettro piatto.
        has_signal = float(eig[0]) >= C.EIG_SPREAD_MIN_DB
        # Affidabilità direzione: richiede anche un picco marcato nello spettro.
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
            if has_doa:   # solo quando la direzione è affidabile (eig + PAPR OK)
                S.az_hist.append(az)
                if len(S.az_hist) >= 3:
                    S.az_median = _circ_median(np.array(S.az_hist))
                else:
                    S.az_median = az
            # ── Recording (ogni frame, valido o meno) ─────────────────────────
            S.rec_t.append(time.time())
            S.rec_az.append(float(az))
            S.rec_el.append(float(el_est))
            S.rec_papr.append(float(papr))
            S.rec_snr.append(float(snr))
            S.rec_eig.append(eig.copy())
            S.rec_phase.append(phase_diffs.copy())
            S.rec_R.append(np.array(R, dtype=np.complex128).copy())
            S.rec_has_sig.append(bool(has_doa))   # True = segnale + direzione affidabile
            S.frame_n += 1


# =============================================================================
# UI: 6 pannelli — bussola, heatmap 2D, autovalori, storia az/el, fasi
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
    ax_2d.set_ylabel("Elevazione [°]", color=C_MUT, fontsize=8)
    ax_2d.set_title("Spettro 2D  az × el", color=C_TEXT, fontsize=9)
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

    # ── [0,2]  Autovalori + readout ───────────────────────────────────────────
    ax_q = fig.add_subplot(gs[0, 2], facecolor=BG2)
    ax_q.set_facecolor(BG2)
    ax_q.set_title("Autovalori + qualità", color=C_TEXT, fontsize=9)
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
    ax_az.set_title("Storia  azimuth", color=C_TEXT, fontsize=9)
    ax_az.set_xlabel("Frame recenti →", color=C_MUT, fontsize=8)
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

    # ── [1,1]  Storia elevazione rolling ─────────────────────────────────────
    ax_el = fig.add_subplot(gs[1, 1], facecolor=BG2)
    ax_el.set_facecolor(BG2)
    ax_el.set_title("Storia  elevazione", color=C_TEXT, fontsize=9)
    ax_el.set_xlabel("Frame recenti →", color=C_MUT, fontsize=8)
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
    ax_ph.set_title("ΔΦ  CH1..4 – CH0  (da matrice di covarianza)", color=C_TEXT, fontsize=9)
    ax_ph.set_xlabel("Frame recenti →", color=C_MUT, fontsize=8)
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

def _save_recording(S: SimpleNamespace, freq_hz: int, algo: str) -> None:
    """Salva su .npz il buffer dati accumulato durante la sessione."""
    import datetime
    n = len(S.rec_t)
    if n == 0:
        print("[REC] Nessun frame acquisito — file non salvato.")
        return
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"doa_data_{ts}.npz")
    np.savez_compressed(
        path,
        timestamps  = np.array(S.rec_t,       dtype=np.float64),   # (N,)
        az_deg      = np.array(S.rec_az,       dtype=np.float32),   # (N,)
        el_deg      = np.array(S.rec_el,       dtype=np.float32),   # (N,)
        papr_db     = np.array(S.rec_papr,     dtype=np.float32),   # (N,)
        snr_db      = np.array(S.rec_snr,      dtype=np.float32),   # (N,)
        eig_db      = np.array(S.rec_eig,      dtype=np.float32),   # (N, 5)
        phase_diffs = np.array(S.rec_phase,    dtype=np.float32),   # (N, 4) gradi
        R_real      = np.real(np.array(S.rec_R, dtype=np.complex128)),  # (N, 5, 5)
        R_imag      = np.imag(np.array(S.rec_R, dtype=np.complex128)),  # (N, 5, 5)
        has_signal  = np.array(S.rec_has_sig,  dtype=bool),          # (N,)
        freq_hz     = np.int64(freq_hz),
        algo        = np.bytes_(algo.encode()),
    )
    print(f"[REC] {n} frame salvati → {path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="DoA CW 868 MHz — KrakenSDR UCA 5 antenne",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq",   type=float, default=C.FREQ_HZ / 1e6, help="Freq RF [MHz]")
    p.add_argument("--gain",   type=float, default=C.GAIN_DB,        help="IF gain [dB]")
    p.add_argument("--radius", type=float, default=C.RADIUS_LAMBDA,  help="Raggio UCA in λ")
    p.add_argument("--offset", type=float, default=C.ANT0_OFFSET_DEG,
                   help="Offset ant0 dal Nord [°]")
    p.add_argument("--algo",   choices=["music", "capon", "bartlett"],
                   default=C.DOA_ALGORITHM.lower(), help="Algoritmo DoA")
    p.add_argument("--nsig",   type=int,   default=C.NUM_SIGNALS,    help="Sorgenti attese")
    p.add_argument("--alpha",  type=float, default=C.COV_ALPHA,      help="EMA covarianza")
    p.add_argument("--demo",   action="store_true",
                   help="Segnale CW sintetico a az=90° (nessun hardware)")
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

    print(f"  DoA CW — {args.algo.upper()}  @  {freq_hz/1e6:.3f} MHz")
    print(f"  UCA: {cfg.n_ant} ant  r={args.radius:.3f}λ  offset={args.offset:.1f}°")
    print(f"  Heimdall: {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}")

    if not args.demo and not _check_heimdall(C.HEIMDALL_HOST, C.HEIMDALL_PORT):
        print(f"\n[ERRORE] Heimdall non raggiungibile su {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}")
        print("  Avvia Heimdall prima, oppure usa --demo per testare senza hardware.")
        sys.exit(1)

    if args.demo:
        src = None
        print("  DEMO: CW sintetico a 90° Est")
    else:
        src = KrakenIQSource(
            host=C.HEIMDALL_HOST, port=C.HEIMDALL_PORT,
            ctrl_port=C.HEIMDALL_CTRL, num_channels=C.N_ANTENNAS,
            freq_hz=freq_hz, gain_db=args.gain,
        )
        src.start()
        print("  Heimdall connesso.")

    threading.Thread(
        target=_acq_loop,
        args=(src, cfg, args.algo, acc, S),
        kwargs={"demo": args.demo},
        daemon=True,
    ).start()

    _build_and_run_ui(S, cfg, algo=args.algo, freq_hz=freq_hz)

    S.running = False
    if src is not None:
        src.stop()
    _save_recording(S, freq_hz=freq_hz, algo=args.algo)
    print("Stop.")


if __name__ == "__main__":
    main()
