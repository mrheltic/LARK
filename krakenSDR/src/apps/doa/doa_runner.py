#!/usr/bin/env python3
"""
KrakenSDR DoA N-antenna – standalone application
=================================================
Direction-of-Arrival estimation without GNU Radio dependencies.
Uses KrakenIQSource (TCP) + doa_algorithms.py for signal processing
and matplotlib for visualisation.

Configuration via environment variables or config.py defaults:
  NUM_CHANNELS  number of antennas/channels  (default: config.N_ANTENNAS)
  ARRAY_TYPE    'ULA' or 'UCA'               (default: config.GEOMETRY)
  CENTER_FREQ   centre frequency in MHz       (default: config.FREQ_HZ/1e6)
  GAIN_DB       RF gain in dB                 (default: config.GAIN_DB)
  ARRAY_DIST    antenna spacing in metres     (ULA: overrides d_lambda)

Launch:
    python3 krakenSDR/src/run_doa.py
    NUM_CHANNELS=5 ARRAY_TYPE=UCA CENTER_FREQ=868 python3 krakenSDR/src/run_doa.py

UI Layout
---------
  Row 0:   freq slider | gain | array_dist | estimated range
  Row 1-2: CH0 FFT (col 0-2) | Polar MUSIC spectrum widget (col 3-4)
  Row 3:   Compass (col 0-1) | 2D Map (col 2-3) | quality gauge (col 4)
  Row 4:   Bearing history (col 0-2) | Calibration (col 3-4)

Phase calibration
-----------------
Calibration corrects the hardware phase offset between the RF branches.
The offset is saved to /workspace/.doa_calibration.json and reloaded
automatically on the next start.

Phase consistency (2-element ULA only)
--------------------------------------
For 2-antenna ULA: deviation between MUSIC bearing and the bearing
expected from the cross-correlation phase.  Values <= 5 deg indicate
a well-calibrated array with a dominant source.
"""

from __future__ import annotations

import collections
import ctypes
import json
import math
import os
import signal
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if sys.platform.startswith("linux"):
    try:
        ctypes.cdll.LoadLibrary("libX11.so").XInitThreads()
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

import config as C
from hardware.kraken_iq_source import KrakenIQSource
from core.doa_algorithms import (
    ArrayConfig, Geometry,
    doa_music, doa_root_music, doa_capon, doa_ml, doa_esprit,
    apply_phase_correction,
    measure_power_db, snr_from_covariance, papr_db,
    condition_number, CovarianceAccumulator,
    apply_decorrelation, covariance,
)
from ui.theme import (
    apply_mpl_style,
    BG, BG2, BG3, BORDER, DIM,
    BLUE, TEAL, AMBER, VIOLET, ROSE, LIME,
    TEXT, MUTED,
)

# ── Config from env vars (override config.py defaults) ────────────────────────
NUM_CHANNELS = int(os.environ.get("NUM_CHANNELS",  str(C.N_ANTENNAS)))
ARRAY_TYPE   = os.environ.get("ARRAY_TYPE",  C.GEOMETRY).upper()
CENTER_FREQ  = float(os.environ.get("CENTER_FREQ", str(C.FREQ_HZ / 1e6))) * 1e6
GAIN_DB      = float(os.environ.get("GAIN_DB",  str(C.GAIN_DB)))

if "ARRAY_DIST" in os.environ:            # metres → lambda-normalised
    _dist_m  = float(os.environ["ARRAY_DIST"])
    _lam_m   = 3e8 / CENTER_FREQ
    D_LAMBDA = _dist_m / _lam_m
    R_LAMBDA = _dist_m / _lam_m
else:
    D_LAMBDA = C.D_LAMBDA
    R_LAMBDA = C.RADIUS_LAMBDA

_CAL_FILE = os.path.join(_SRC, ".doa_calibration.json")


def main() -> None:
    apply_mpl_style()

    GEOM = Geometry.UCA if ARRAY_TYPE == "UCA" else Geometry.ULA
    cfg  = ArrayConfig(
        Nr=NUM_CHANNELS, geometry=GEOM,
        d_lambda=D_LAMBDA, radius_lambda=R_LAMBDA,
        num_expected_signals=C.NUM_SIGNALS,
        num_scan_points=C.SCAN_POINTS,
    )
    theta_scan = cfg.scan_range()

    HIST  = 120
    FFT_N = 512
    _flat     = np.full(C.SCAN_POINTS, -40.0)
    _fft_freqs = np.fft.fftshift(np.fft.fftfreq(FFT_N)) * C.SAMPLE_RATE_HZ / 1e3
    _x_hist   = np.arange(HIST)
    WARMUP    = max(10, int(1.0 / max(0.01, 1.0 - C.COV_ALPHA)))

    # ── Persistent state ──────────────────────────────────────────────────────
    S = SimpleNamespace(
        angle_phasor = complex(1.0, 0.0),
        last_fft     = np.full(FFT_N, -80.0),
        warmup_count = 0,
        cov_acc      = CovarianceAccumulator(alpha=C.COV_ALPHA),
        h_angle      = collections.deque([float("nan")] * HIST, maxlen=HIST),
        h_snr        = collections.deque([0.0] * HIST, maxlen=HIST),
        est_deg      = 0.0,
        cal_offset   = 0.0,
        fps          = 0.0,
        t_last       = time.time(),
    )

    # Load saved calibration
    try:
        with open(_CAL_FILE) as _f:
            _cal = json.load(_f)
        S.cal_offset = float(_cal.get("offset_deg", 0.0))
    except Exception:
        pass

    # ── Heimdall connection ────────────────────────────────────────────────────
    kraken = KrakenIQSource(
        host=C.HEIMDALL_HOST, port=C.HEIMDALL_PORT, ctrl_port=C.HEIMDALL_CTRL,
        num_channels=NUM_CHANNELS, freq_hz=CENTER_FREQ, gain_db=GAIN_DB,
    )
    kraken.start()

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.05, right=0.97, top=0.92, bottom=0.12,
        hspace=0.48, wspace=0.40,
    )

    ax_music = fig.add_subplot(gs[0, 0], polar=True)
    ax_comp  = fig.add_subplot(gs[0, 1], polar=True)
    ax_hist  = fig.add_subplot(gs[0, 2])
    ax_fft   = fig.add_subplot(gs[1, 0])
    ax_snr   = fig.add_subplot(gs[1, 1])
    ax_info  = fig.add_subplot(gs[1, 2])

    # ── Style helpers ──────────────────────────────────────────────────────────
    def _sty(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_color(BORDER); sp.set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=7)
        if title:  ax.set_title(title,  color=TEXT,  fontsize=8.5, pad=5, fontweight="semibold")
        if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=7)
        if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=7)

    def _sty_pol(ax, title=""):
        ax.set_facecolor(BG2)
        ax.spines["polar"].set_color(BORDER)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        if title: ax.set_title(title, color=TEXT, fontsize=8.5, pad=8, fontweight="semibold")

    # ── Panel A: MUSIC pseudospectrum ──────────────────────────────────────────
    _sty_pol(ax_music, "Pseudospectrum")
    ax_music.set_ylim([-40, 2]); ax_music.set_rlabel_position(45)
    ax_music.set_yticks([-30, -20, -10, 0])
    ax_music.set_yticklabels(["-30", "-20", "-10", "0"], fontsize=6, color=MUTED)
    line_spec,  = ax_music.plot(theta_scan, _flat.copy(), color=BLUE, linewidth=1.5)
    line_est_m, = ax_music.plot([0, 0], [-40, 2], color=TEAL, linewidth=2.0, alpha=0.85)
    txt_algo    = ax_music.text(0.5, -0.07, "", transform=ax_music.transAxes,
                                ha="center", fontsize=8, color=TEXT)

    # ── Panel B: compass ───────────────────────────────────────────────────────
    _sty_pol(ax_comp, "DoA Compass")
    ax_comp.set_yticks([])
    ax_comp.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax_comp.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                             color=MUTED, fontsize=8)
    ax_comp.set_ylim([0, 1])
    ax_comp.plot(np.linspace(0, 2*np.pi, 360), np.ones(360)*0.92, color=BORDER, linewidth=0.8)
    needle,   = ax_comp.plot([0, 0], [0, 0.85], color=TEAL, linewidth=3.5)
    needle_b, = ax_comp.plot([0, 0], [0, 0.38], color=TEAL, linewidth=2.0, alpha=0.30)
    txt_est    = ax_comp.text(0.5, -0.07, "", transform=ax_comp.transAxes,
                              ha="center", va="top", fontsize=14, fontweight="bold",
                              color=TEAL,
                              bbox=dict(facecolor=BG3, edgecolor=BORDER, boxstyle="round,pad=0.4"))
    txt_status = ax_comp.text(0.02, 1.05, "", transform=ax_comp.transAxes,
                               ha="left", va="bottom", fontsize=8, fontweight="bold")
    txt_fps    = ax_comp.text(0.98, 1.05, "", transform=ax_comp.transAxes,
                               ha="right", va="bottom", fontsize=7, color=DIM)

    # ── Panel C: Bearing history ───────────────────────────────────────────────
    _sty(ax_hist, "Bearing History", "", "deg")
    ax_hist.set_xlim(0, HIST - 1); ax_hist.set_ylim(-5, 365)
    ax_hist.set_yticks(range(0, 361, 90))
    line_ahist, = ax_hist.plot(_x_hist, list(S.h_angle), color=TEAL, linewidth=1.4)
    txt_sigma   = ax_hist.text(0.02, 0.96, "", transform=ax_hist.transAxes,
                               fontsize=7.5, color=TEAL, va="top")

    # ── Panel D: IQ Spectrum ───────────────────────────────────────────────────
    _sty(ax_fft, "IQ Spectrum  (CH0)", "offset [kHz]", "dB")
    line_fft, = ax_fft.plot(_fft_freqs, S.last_fft.copy(), color=BLUE, linewidth=0.9)
    ax_fft.set_xlim(_fft_freqs[0], _fft_freqs[-1]); ax_fft.set_ylim(-65, 5)
    ax_fft.axvline(0, color=ROSE, linewidth=0.8, linestyle="--", alpha=0.55)

    # ── Panel E: SNR history ───────────────────────────────────────────────────
    _sty(ax_snr, "SNR History", "", "dB")
    ax_snr.set_xlim(0, HIST - 1); ax_snr.set_ylim(-2, 36)
    ax_snr.axhline(10, color=VIOLET, linewidth=0.8, linestyle=":", alpha=0.5, label="10 dB")
    line_snr, = ax_snr.plot(_x_hist, list(S.h_snr), color=VIOLET, linewidth=1.4)
    ax_snr.legend(fontsize=6.5, labelcolor=MUTED, framealpha=0, loc="upper right")

    # ── Panel F: System info ───────────────────────────────────────────────────
    _sty(ax_info, "System Info")
    ax_info.axis("off")
    txt_info = ax_info.text(0.05, 0.92, "", transform=ax_info.transAxes,
                            fontsize=8, color=TEXT, va="top", fontfamily="monospace")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"KrakenSDR DoA  │  {NUM_CHANNELS}-ant {ARRAY_TYPE}"
        f"  │  {C.DOA_ALGORITHM}/{C.DECORRELATION}"
        f"  │  {CENTER_FREQ/1e6:.3f} MHz",
        color=TEXT, fontsize=10, fontweight="semibold", y=0.97,
    )

    # ── Calibration buttons ────────────────────────────────────────────────────
    _BTN_Y = 0.015; _BTN_H = 0.048
    ax_btn_cal = fig.add_axes([0.35, _BTN_Y, 0.12, _BTN_H])
    ax_btn_rst = fig.add_axes([0.48, _BTN_Y, 0.10, _BTN_H])
    txt_cal    = fig.text(0.60, 0.030, f"Offset: {S.cal_offset:.1f}°",
                          color=AMBER, fontsize=8, va="center",
                          bbox=dict(facecolor=BG3, edgecolor=BORDER, boxstyle="round,pad=0.3"))
    btn_cal = Button(ax_btn_cal, "Set Zero",  color=BG3, hovercolor="#3a4060")
    btn_rst = Button(ax_btn_rst, "Reset Cal", color=BG3, hovercolor="#3a4060")
    for _b in (btn_cal, btn_rst):
        _b.label.set_color(TEXT); _b.label.set_fontsize(8)

    def _save_cal():
        try:
            with open(_CAL_FILE, "w") as _f:
                json.dump({"offset_deg": S.cal_offset, "freq_hz": CENTER_FREQ}, _f)
        except Exception:
            pass

    def _on_set_zero(_):
        S.cal_offset = S.est_deg
        txt_cal.set_text(f"Offset: {S.cal_offset:.1f}°")
        txt_cal.set_color(LIME)
        S.angle_phasor = complex(1.0, 0.0)
        _save_cal()

    def _on_reset_cal(_):
        S.cal_offset = 0.0
        txt_cal.set_text("Offset: 0.0°")
        txt_cal.set_color(AMBER)
        S.angle_phasor = complex(1.0, 0.0)
        _save_cal()

    btn_cal.on_clicked(_on_set_zero)
    btn_rst.on_clicked(_on_reset_cal)

    # ── Animation update ───────────────────────────────────────────────────────
    def update(_frame):
        frame = kraken.get_frame(timeout=0.005)
        if frame is None:
            return

        nr = min(NUM_CHANNELS, frame.shape[0])
        X  = frame[:nr, :].astype(np.complex128)
        if C.HW_NUM_SAMPLES > 0:
            X = X[:, :C.HW_NUM_SAMPLES]

        # Phase correction
        ph_offs = (C.PHASE_OFFSETS_DEG + [0.0] * nr)[:nr]
        X = apply_phase_correction(X, ph_offs)

        # Amplitude normalise
        if C.AMPLITUDE_NORMALIZE:
            pwr = np.sqrt(np.maximum(np.mean(np.abs(X) ** 2, axis=1, keepdims=True), 1e-15))
            X   = X / pwr

        # Squelch gate
        pwr_dbw = float(measure_power_db(X[0]))
        if C.SQUELCH_ENABLED and pwr_dbw < C.SQUELCH_THRESHOLD_DB:
            txt_status.set_text("SQUELCH"); txt_status.set_color(DIM)
            return

        # FFT (CH0 preview)
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft(X[0, :FFT_N], FFT_N)))
        fft_db  = 20 * np.log10(np.maximum(fft_mag / FFT_N, 1e-10))
        S.last_fft = 0.7 * S.last_fft + 0.3 * fft_db

        # Covariance EMA + decorrelation
        R      = S.cov_acc.update(covariance(X))
        R_proc = apply_decorrelation(R, C.DECORRELATION)

        S.warmup_count += 1
        if S.warmup_count < WARMUP:
            txt_status.set_text("WARM UP"); txt_status.set_color(AMBER)
            return

        # ── DoA estimation ────────────────────────────────────────────────────
        algo = C.DOA_ALGORITHM.upper()
        spec = _flat.copy()
        est_rad = 0.0

        if algo == "MUSIC":
            th, sp    = doa_music(R_proc, cfg)
            idx       = int(np.argmax(sp)); spec = sp; est_rad = float(th[idx])
        elif algo == "ROOT-MUSIC":
            est_rad, spec, _ = doa_root_music(R_proc, cfg)
            est_rad = float(est_rad)
        elif algo == "CAPON":
            th, sp    = doa_capon(R_proc, cfg)
            idx       = int(np.argmax(sp)); spec = sp; est_rad = float(th[idx])
        elif algo == "ML":
            th, sp    = doa_ml(R_proc, cfg)
            idx       = int(np.argmax(sp)); spec = sp; est_rad = float(th[idx])
        elif algo == "ESPRIT":
            est_rad, spec, _ = doa_esprit(R_proc, cfg)
            est_rad = float(est_rad)

        # Circular EMA on phasor
        alpha        = float(C.ANGLE_SMOOTH_ALPHA)
        S.angle_phasor = alpha * S.angle_phasor + (1.0 - alpha) * np.exp(1j * est_rad)
        smooth_rad   = float(np.angle(S.angle_phasor))
        smooth_deg   = (float(np.degrees(smooth_rad)) - S.cal_offset) % 360.0
        S.est_deg    = smooth_deg

        snr_db = float(snr_from_covariance(R, cfg.Nr - cfg.num_expected_signals))
        S.h_angle.append(smooth_deg)
        S.h_snr.append(snr_db)

        # FPS
        now    = time.time()
        S.fps  = 0.9 * S.fps + 0.1 / max(0.001, now - S.t_last)
        S.t_last = now

        # ── Plot updates ───────────────────────────────────────────────────────
        ang_arr   = np.array(list(S.h_angle))
        valid     = ang_arr[~np.isnan(ang_arr)]
        sigma_deg = float(np.std(valid)) if len(valid) > 1 else 0.0

        line_spec.set_ydata(spec)
        line_est_m.set_xdata([smooth_rad, smooth_rad])
        txt_algo.set_text(f"{C.DOA_ALGORITHM}/{C.DECORRELATION}  {smooth_deg:.1f}°")

        needle.set_xdata([smooth_rad, smooth_rad])
        needle_b.set_xdata([smooth_rad + np.pi, smooth_rad + np.pi])
        txt_est.set_text(f"{smooth_deg:.1f}°")

        ok = snr_db >= 8.0
        txt_status.set_text("LOCK" if ok else "SEARCH")
        txt_status.set_color(TEAL if ok else AMBER)
        txt_fps.set_text(f"{S.fps:.1f} fps")

        line_ahist.set_ydata(list(S.h_angle))
        txt_sigma.set_text(f"σ = {sigma_deg:.1f}°")

        line_fft.set_ydata(S.last_fft)
        line_snr.set_ydata(list(S.h_snr))

        txt_info.set_text(
            f"Heimdall:   {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}\n"
            f"Antennas:   {NUM_CHANNELS}  [{ARRAY_TYPE}]\n"
            f"Frequency:  {CENTER_FREQ/1e6:.3f} MHz\n"
            f"Gain:       {GAIN_DB} dB\n"
            f"Power CH0:  {pwr_dbw:.1f} dBW\n"
            f"SNR:        {snr_db:.1f} dB\n"
            f"Bearing:    {smooth_deg:.1f}°\n"
            f"σ bearing:  {sigma_deg:.1f}°\n"
            f"Algorithm:  {C.DOA_ALGORITHM}\n"
            f"Decorr:     {C.DECORRELATION}\n"
            f"Calibr.:    {S.cal_offset:.1f}°"
        )

    # ── Event handlers ─────────────────────────────────────────────────────────
    def _on_close(_evt):
        kraken.stop()

    fig.canvas.mpl_connect("close_event", _on_close)

    def _sig_handler(*_):
        kraken.stop()
        plt.close("all")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    ani = animation.FuncAnimation(
        fig, update,
        interval=max(40, C.INTERVAL_MS),
        blit=False,
        cache_frame_data=False,
    )

    print(f"[run_doa] {NUM_CHANNELS}-ant {ARRAY_TYPE}  "
          f"{CENTER_FREQ/1e6:.3f} MHz  gain={GAIN_DB} dB  "
          f"algo={C.DOA_ALGORITHM}/{C.DECORRELATION}")
    plt.show()


if __name__ == "__main__":
    main()
