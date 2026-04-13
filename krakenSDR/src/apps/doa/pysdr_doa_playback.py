#!/usr/bin/env python3
"""
DoA Playback – KrakenSDR
=========================
Reads a .npz recording saved by pysdr_doa_realtime.py and replays it
offline through the same DoA pipeline, displaying all 8 panels.

Usage:
    python3 pysdr_doa/pysdr_doa_playback.py recordings/kraken_20260325_123456.npz
    python3 pysdr_doa/pysdr_doa_playback.py          # opens a file-picker dialog

Playback controls (bottom strip):
  ⏸/▶  – Play / Pause
  ⏮    – Rewind to frame 0
  ×0.5 / ×1 / ×2 / ×4  – speed (frames per animation tick)
  [──────────────]       – Frame scrub slider

Scientific references same as pysdr_doa_realtime.py.
"""

from __future__ import annotations

import os
import sys
import json
import time
import collections

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config as C
from core.doa_algorithms import (
    ArrayConfig, Geometry,
    doa_music, doa_root_music, doa_capon, doa_ml, doa_esprit,
    apply_phase_correction,
    measure_power_db, snr_from_covariance, papr_db,
    condition_number, eigenvalue_spread_db, coherence_matrix,
    CovarianceAccumulator,
)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider


# =============================================================================
# File selection
# =============================================================================


def main() -> None:

    if len(sys.argv) > 1:
        rec_path = os.path.abspath(sys.argv[1])
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            _root = tk.Tk()
            _root.withdraw()
            _rec_dir = os.path.normpath(os.path.join(_SRC, "..", "..", "recordings"))
            rec_path = filedialog.askopenfilename(
                title="Open KrakenSDR recording",
                initialdir=_rec_dir if os.path.isdir(_rec_dir) else ".",
                filetypes=[("KrakenSDR recording", "*.npz"), ("All files", "*.*")],
            )
            _root.destroy()
            if not rec_path:
                print("[PB] No file selected.")
                sys.exit(0)
        except Exception:
            print("Usage:  python3 pysdr_doa_playback.py <recording.npz>")
            sys.exit(1)

    if not os.path.isfile(rec_path):
        print(f"[PB] File not found: {rec_path}")
        sys.exit(1)


    # =============================================================================
    # Load recording
    # =============================================================================

    print(f"[PB] Loading  {rec_path} …", end=" ", flush=True)
    _data      = np.load(rec_path, allow_pickle=False)
    frames_all = _data["frames"].astype(np.complex128)   # (N, ant, samples)
    timestamps = _data["timestamps"]                      # (N,)  float64
    N_TOTAL    = int(frames_all.shape[0])

    if N_TOTAL == 0:
        print("recording is empty."); sys.exit(1)
    print(f"{N_TOTAL} frames  ({frames_all.shape[1]} ant  {frames_all.shape[2]} samples/frame)")

    # JSON sidecar (optional but expected when saved by realtime recorder)
    meta: dict = {}
    _meta_path = rec_path.replace(".npz", ".json")
    if os.path.isfile(_meta_path):
        with open(_meta_path) as _f:
            meta = json.load(_f)
        print(f"[PB] freq={meta.get('freq_hz',0)/1e6:.4f} MHz  "
              f"algo={meta.get('algo','?')}  decorr={meta.get('decorr','?')}  "
              f"n_frames={meta.get('n_frames','?')}  dur={meta.get('duration_s',0):.1f} s")


    # =============================================================================
    # Configuration – prefer metadata, fall back to config.py
    # =============================================================================

    FREQ_HZ    = float(meta.get("freq_hz",        C.FREQ_HZ))
    FS         = float(meta.get("sample_rate_hz", C.SAMPLE_RATE_HZ))
    N_ANT      = int(  meta.get("n_antennas",     frames_all.shape[1]))
    ALGO       = meta.get("algo",   C.DOA_ALGORITHM)
    DECORR     = meta.get("decorr", C.DECORRELATION)
    COV_ALPHA  = float(meta.get("cov_alpha",      C.COV_ALPHA))
    GEOM_STR   = meta.get("geometry",             C.GEOMETRY)
    R_LAMBDA   = float(meta.get("radius_lambda",  C.RADIUS_LAMBDA))
    D_LAMBDA   = float(meta.get("d_lambda",       C.D_LAMBDA))
    NUM_SIG    = int(  meta.get("n_signals",      C.NUM_SIGNALS))
    SCAN_PTS   = int(  meta.get("scan_points",    C.SCAN_POINTS))
    CAL_OFFSET = float(meta.get("cal_offset_deg", 0.0))

    _ANG_ALPHA  = C.ANGLE_SMOOTH_ALPHA
    _PHASE_OFFS = C.PHASE_OFFSETS_DEG
    _SQ_EN      = C.SQUELCH_ENABLED
    _SQ_THR     = C.SQUELCH_THRESHOLD_DB

    _GEOM = Geometry.UCA if GEOM_STR.upper() == "UCA" else Geometry.ULA
    cfg   = ArrayConfig(
        Nr                   = N_ANT,
        geometry             = _GEOM,
        d_lambda             = D_LAMBDA,
        radius_lambda        = R_LAMBDA,
        num_expected_signals = NUM_SIG,
        num_scan_points      = SCAN_PTS,
    )
    theta_scan = cfg.scan_range()

    HIST       = 120
    FFT_N      = 512
    INTERVAL   = max(40, C.INTERVAL_MS)

    _flat      = np.full(SCAN_PTS, -40.0)
    _fft_freqs = np.fft.fftshift(np.fft.fftfreq(FFT_N)) * FS / 1e3   # kHz
    _x_hist    = np.arange(HIST)


    # =============================================================================
    # Running state (reset on scrub/rewind)
    # =============================================================================

    _cov_acc      = CovarianceAccumulator(alpha=COV_ALPHA)
    _angle_phasor = complex(1.0, 0.0)
    _last_fft     = np.full(FFT_N, -80.0)

    _h_angle = collections.deque([np.nan] * HIST, maxlen=HIST)
    _h_papr  = collections.deque([0.0]    * HIST, maxlen=HIST)
    _h_snr   = collections.deque([0.0]    * HIST, maxlen=HIST)
    _h_ph    = [collections.deque([0.0] * HIST, maxlen=HIST)
                for _ in range(N_ANT - 1)]


    def _reset_accumulators() -> None:
        """Flush all running state – call after scrub or rewind."""
        global _angle_phasor, _last_fft
        _cov_acc.reset()
        _angle_phasor = complex(1.0, 0.0)
        _last_fft     = np.full(FFT_N, -80.0)
        _h_angle.clear(); _h_angle.extend([np.nan] * HIST)
        _h_papr.clear();  _h_papr.extend([0.0] * HIST)
        _h_snr.clear();   _h_snr.extend([0.0] * HIST)
        for _d in _h_ph:
            _d.clear(); _d.extend([0.0] * HIST)


    # =============================================================================
    # Playback state
    # =============================================================================

    _pb = {
        "idx":     0.0,      # float frame pointer
        "playing": True,
        "speed":   1.0,      # frames to advance per animation tick
        "cal":     CAL_OFFSET,
    }
    _slider_drag = [False]


    # ---------------------------------------------------------------------------
    # Colour palette  (same Nord-inspired scheme as realtime)
    # ---------------------------------------------------------------------------
    BG        = "#1a1d27"
    BG2       = "#21253a"
    BG3       = "#2a2f47"
    C_BORDER  = "#3b4263"
    C_DIM     = "#4e5680"
    C_BLUE    = "#5ea4e0"
    C_TEAL    = "#4ecdc4"
    C_AMBER   = "#f4a431"
    C_VIOLET  = "#a78bfa"
    C_ROSE    = "#f16b6f"
    C_LIME    = "#6dd97d"
    C_SKY     = "#93c5fd"
    C_TEXT    = "#d8dae8"
    C_MUTED   = "#8891b0"

    # Aliases
    C_CYAN  = C_BLUE
    C_GREEN = C_TEAL
    C_ORG   = C_AMBER
    C_MAG   = C_VIOLET
    C_RED   = C_ROSE
    C_GRID  = C_BORDER


    matplotlib.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         8,
        "axes.titlesize":    8.5,
        "axes.labelsize":    7.5,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   6.5,
        "figure.facecolor":  "#1a1d27",
        "axes.facecolor":    "#21253a",
        "axes.edgecolor":    "#3b4263",
        "axes.grid":         True,
        "grid.color":        "#3b4263",
        "grid.linewidth":    0.5,
        "grid.alpha":        0.7,
        "xtick.color":       "#8891b0",
        "ytick.color":       "#8891b0",
        "text.color":        "#d8dae8",
    })

    # =============================================================================
    # Figure layout  (identical grid to realtime, bottom increased for controls)
    # =============================================================================

    fig = plt.figure(figsize=(20, 10.5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        left=0.04, right=0.97,
        top=0.93,  bottom=0.13,
        hspace=0.50, wspace=0.40,
        width_ratios=[1.25, 1.25, 1.1, 1.4],
        height_ratios=[1.4, 1.0],
    )

    ax_music  = fig.add_subplot(gs[0, 0], polar=True)
    ax_comp   = fig.add_subplot(gs[0, 1], polar=True)
    ax_hist_a = fig.add_subplot(gs[0, 2])
    ax_eig    = fig.add_subplot(gs[0, 3])
    ax_coh    = fig.add_subplot(gs[1, 0])
    ax_pq     = fig.add_subplot(gs[1, 1])
    ax_fft    = fig.add_subplot(gs[1, 2])
    ax_phase  = fig.add_subplot(gs[1, 3])


    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(C_DIM)
        ax.tick_params(colors="gray", labelsize=7)
        ax.yaxis.set_tick_params(labelcolor="gray")
        ax.xaxis.set_tick_params(labelcolor="gray")
        ax.grid(color=C_GRID, linewidth=0.5, alpha=0.7)
        if title:   ax.set_title(title,   color=C_TEXT, fontsize=8.5, pad=4)
        if xlabel:  ax.set_xlabel(xlabel, color=C_TEXT, fontsize=7.5)
        if ylabel:  ax.set_ylabel(ylabel, color=C_TEXT, fontsize=7.5)


    def _style_polar(ax, title=""):
        ax.set_facecolor(BG)
        ax.spines["polar"].set_color(C_DIM)
        ax.tick_params(colors="gray", labelsize=7)
        for gl in ax.yaxis.get_gridlines():
            gl.set_color(C_GRID); gl.set_linewidth(0.5)
        for gl in ax.xaxis.get_gridlines():
            gl.set_color(C_GRID); gl.set_linewidth(0.5)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        if title:
            ax.set_title(title, color=C_TEXT, fontsize=8.5, pad=8)


    # ── A: MUSIC pseudospectrum ───────────────────────────────────────────────────
    _style_polar(ax_music, f"Pseudospectrum  [{ALGO} / {DECORR}]")
    ax_music.set_ylim([-40, 2])
    ax_music.set_rlabel_position(45)
    ax_music.set_yticks([-40, -20, -10, 0])
    ax_music.set_yticklabels(["-40", "-20", "-10", "0"], fontsize=6, color=C_MUTED)

    line_spec,  = ax_music.plot(theta_scan, _flat.copy(), color=C_BLUE, linewidth=1.5)
    line_est_m, = ax_music.plot([0, 0], [-40, 2], color=C_TEAL, linewidth=2.0, alpha=0.85)
    txt_music   = ax_music.text(0.5, -0.07, "", transform=ax_music.transAxes,
                                 ha="center", fontsize=8, color=C_TEXT)

    # ── B: DoA compass ────────────────────────────────────────────────────────────
    _style_polar(ax_comp, "DoA Compass")
    ax_comp.set_yticks([])
    ax_comp.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax_comp.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                             color=C_MUTED, fontsize=8)
    ax_comp.set_ylim([0, 1])
    ax_comp.plot(np.linspace(0, 2 * np.pi, 360), np.ones(360) * 0.92,
                 color=C_BORDER, linewidth=0.8)

    needle,   = ax_comp.plot([0, 0], [0, 0.85], color=C_TEAL, linewidth=3.5)
    needle_b, = ax_comp.plot([0, 0], [0, 0.38], color=C_TEAL, linewidth=2.0, alpha=0.30)

    # Uncertainty wedge (filled arc showing ±σ around estimated direction)
    _unc_theta = np.linspace(0, 0.01, 40)
    _unc_r     = np.concatenate([[0], np.ones(38) * 0.92, [0]])
    unc_fill,  = ax_comp.fill(_unc_theta, _unc_r, color=C_TEAL, alpha=0.0)

    txt_est     = ax_comp.text(
        0.5, -0.07, "", transform=ax_comp.transAxes,
        ha="center", va="top", fontsize=15, fontweight="bold", color=C_TEAL,
        bbox=dict(facecolor=BG3, edgecolor=C_BORDER, boxstyle="round,pad=0.45"),
    )
    txt_status  = ax_comp.text(0.02, 1.05, "", transform=ax_comp.transAxes,
                                ha="left",   va="bottom", fontsize=8.5, fontweight="bold")
    txt_fps     = ax_comp.text(0.98, 1.05, "", transform=ax_comp.transAxes,
                                ha="right",  va="bottom", fontsize=7.5, color=C_DIM)
    txt_metrics = ax_comp.text(0.5,  1.05, "", transform=ax_comp.transAxes,
                                ha="center", va="bottom", fontsize=7, color=C_MUTED)

    # ── C: Angle history ──────────────────────────────────────────────────────────
    _style(ax_hist_a, "Bearing History", "", "deg")
    ax_hist_a.set_xlim(0, HIST - 1)
    ax_hist_a.set_ylim(-5, 365)
    ax_hist_a.set_yticks(range(0, 361, 45))
    for _deg in [0, 90, 180, 270, 360]:
        ax_hist_a.axhline(_deg, color=C_BORDER, linewidth=0.5)
    line_ahist, = ax_hist_a.plot(_x_hist, list(_h_angle), color=C_TEAL, linewidth=1.4)
    txt_sigma   = ax_hist_a.text(0.02, 0.96, "", transform=ax_hist_a.transAxes,
                                  fontsize=7.5, color=C_TEAL, va="top")

    # ── D: Eigenvalue spread ──────────────────────────────────────────────────────
    _style(ax_eig, "Eigenvalues  [dB]", "", "dB")
    ax_eig.set_xlim(-0.5, N_ANT - 0.5)
    ax_eig.set_xticks(range(N_ANT))
    ax_eig.set_xticklabels([f"\u03bb{i}" for i in range(N_ANT)], color=C_MUTED, fontsize=7.5)
    _eig_colors = [C_BLUE, C_AMBER, C_VIOLET, C_TEAL, C_ROSE][:N_ANT]
    bars_eig    = ax_eig.bar(range(N_ANT), [0.0] * N_ANT,
                              color=_eig_colors, edgecolor="none", alpha=0.85)
    ax_eig.axvline(N_ANT - NUM_SIG - 0.5,
                   color=C_ROSE, linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"noise / signal  (d={NUM_SIG})")
    ax_eig.legend(fontsize=6.5, labelcolor=C_MUTED, framealpha=0.0, loc="upper right")
    txt_cond = ax_eig.text(0.5, 0.96, "", transform=ax_eig.transAxes,
                            ha="center", va="top", fontsize=7.5, color=C_MUTED)

    # ── E: Coherence matrix ───────────────────────────────────────────────────────
    _style(ax_coh, f"Coherence  |\u03bc|  (N={N_ANT})")
    ax_coh.set_xticks(range(N_ANT)); ax_coh.set_yticks(range(N_ANT))
    ax_coh.set_xticklabels([f"ch{k}" for k in range(N_ANT)], color=C_MUTED, fontsize=7.5)
    ax_coh.set_yticklabels([f"ch{k}" for k in range(N_ANT)], color=C_MUTED, fontsize=7.5)
    im_coh = ax_coh.imshow(np.eye(N_ANT), cmap="Blues", vmin=0, vmax=1,
                            aspect="equal", interpolation="nearest")
    _cb = plt.colorbar(im_coh, ax=ax_coh, fraction=0.04, pad=0.04)
    _cb.ax.tick_params(colors=C_MUTED, labelsize=6)
    _cb.outline.set_edgecolor(C_BORDER)
    _coh_txts = [
        [ax_coh.text(j, i, "", ha="center", va="center",
                      fontsize=9, color=C_TEXT, fontweight="bold")
         for j in range(N_ANT)]
        for i in range(N_ANT)
    ]
    txt_coh_lbl = ax_coh.text(0.5, -0.15, "", transform=ax_coh.transAxes,
                                ha="center", fontsize=7, color=C_MUTED)

    # ── F: PAPR + SNR history ─────────────────────────────────────────────────────
    _style(ax_pq, "PAPR & SNR  [dB]", "", "dB")
    ax_pq.set_xlim(0, HIST - 1)
    ax_pq.set_ylim(-2, 36)
    ax_pq.axhline(6,  color=C_AMBER,  linewidth=0.8, linestyle=":",  alpha=0.6,
                  label="PAPR 6 dB")
    ax_pq.axhline(12, color=C_AMBER,  linewidth=0.8, linestyle="--", alpha=0.45,
                  label="PAPR 12 dB")
    ax_pq.axhline(10, color=C_VIOLET, linewidth=0.8, linestyle=":",  alpha=0.5,
                  label="SNR 10 dB")
    line_papr, = ax_pq.plot(_x_hist, list(_h_papr), color=C_AMBER,  linewidth=1.4, label="PAPR")
    line_snr,  = ax_pq.plot(_x_hist, list(_h_snr),  color=C_VIOLET, linewidth=1.0,
                             alpha=0.85, label="SNR")
    ax_pq.legend(fontsize=6.5, labelcolor=C_MUTED, framealpha=0.0,
                 loc="upper right", ncol=2)
    txt_pq = ax_pq.text(0.02, 0.96, "", transform=ax_pq.transAxes,
                         fontsize=7.5, color=C_MUTED, va="top")

    # ── G: IQ FFT spectrum ────────────────────────────────────────────────────────
    _style(ax_fft, "IQ Spectrum", "offset [kHz]", "dB")
    line_fft, = ax_fft.plot(_fft_freqs, _last_fft.copy(), color=C_BLUE, linewidth=0.9)
    ax_fft.set_xlim(_fft_freqs[0], _fft_freqs[-1])
    ax_fft.set_ylim(-65, 5)
    ax_fft.axvline(0, color=C_ROSE, linewidth=0.8, linestyle="--", alpha=0.55,
                   label="carrier")
    ax_fft.legend(fontsize=6.5, labelcolor=C_MUTED, framealpha=0.0)
    txt_fft_pk   = ax_fft.text(0.02, 0.96, "", transform=ax_fft.transAxes,
                                 fontsize=7, color=C_MUTED, va="top")
    txt_fft_freq = ax_fft.text(0.98, 0.96, f"{FREQ_HZ/1e6:.4f} MHz",
                                transform=ax_fft.transAxes,
                                ha="right", fontsize=7, color=C_AMBER, va="top")

    # ── H: Off-diagonal phase history ─────────────────────────────────────────────
    _style(ax_phase, "Phase Stability  arg(R)", "", "rad")
    ax_phase.set_xlim(0, HIST - 1)
    ax_phase.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax_phase.axhline(0, color=C_BORDER, linewidth=0.5)
    ax_phase.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_phase.set_yticklabels(["-\u03c0", "-\u03c0/2", "0", "\u03c0/2", "\u03c0"], fontsize=6.5, color=C_MUTED)
    _ph_colors = [C_AMBER, C_VIOLET, C_TEAL, C_SKY]
    _ph_lines  = [
        ax_phase.plot(_x_hist, list(_h_ph[k]), color=_ph_colors[k],
                       linewidth=1.1, label=f"ang R[0,{k+1}]")[0]
        for k in range(N_ANT - 1)
    ]
    ax_phase.legend(fontsize=6.5, labelcolor=C_MUTED, framealpha=0.0, loc="upper right")
    txt_ph_lbl = ax_phase.text(0.02, 0.96, "", transform=ax_phase.transAxes,
                                fontsize=7, color=C_TEAL, va="top")


    # =============================================================================
    # Global title
    # =============================================================================
    _rec_name = os.path.basename(rec_path)
    _dur_s    = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    fig.suptitle(
        f"KrakenSDR DoA  PLAYBACK  |  {_rec_name}  |  "
        f"{N_ANT}-ant {GEOM_STR}  |  {ALGO} / {DECORR}  |  "
        f"{FREQ_HZ/1e6:.4f} MHz  |  {N_TOTAL} frames  |  {_dur_s:.1f} s",
        color=C_TEXT, fontsize=10, fontweight="semibold", y=0.977,
    )


    # =============================================================================
    # Playback control widgets
    # =============================================================================

    # ---------------------------------------------------------------------------
    # Bottom control strip   (bottom=0.13 leaves enough room for two rows)
    # Row 1  y=0.060:  Play/Pause  Rewind  |  Speed buttons
    # Row 2  y=0.020:  Frame scrub slider (full width)  +  counter badge
    # ---------------------------------------------------------------------------
    _BTN_Y1 = 0.060   # top button row bottom
    _BTN_Y2 = 0.020   # slider row bottom
    _BTN_H  = 0.040   # button height
    _SLD_H  = 0.022   # slider height

    _ax_play   = fig.add_axes([0.04,  _BTN_Y1, 0.090, _BTN_H])
    _ax_rew    = fig.add_axes([0.136, _BTN_Y1, 0.070, _BTN_H])
    # speed buttons with separator gap
    _ax_s05    = fig.add_axes([0.230, _BTN_Y1, 0.056, _BTN_H])
    _ax_s1     = fig.add_axes([0.290, _BTN_Y1, 0.056, _BTN_H])
    _ax_s2     = fig.add_axes([0.350, _BTN_Y1, 0.056, _BTN_H])
    _ax_s4     = fig.add_axes([0.410, _BTN_Y1, 0.056, _BTN_H])
    # slider spans the full bottom row
    _ax_slider = fig.add_axes([0.04,  _BTN_Y2, 0.855, _SLD_H])
    # counter badge to the right of the slider
    _ax_ctr_x  = 0.905

    _btn_play = Button(_ax_play,  "Pause",    color=BG3,       hovercolor="#3a4060")
    _btn_rew  = Button(_ax_rew,   "Rewind",   color=BG3,       hovercolor="#3a4060")
    _btn_s05  = Button(_ax_s05,   "x 0.5",   color="#2a2318",  hovercolor="#403525")
    _btn_s1   = Button(_ax_s1,    "x 1",     color="#1a2a1a",  hovercolor="#273d27")
    _btn_s2   = Button(_ax_s2,    "x 2",     color="#2a2318",  hovercolor="#403525")
    _btn_s4   = Button(_ax_s4,    "x 4",     color="#2a2318",  hovercolor="#403525")

    for _b in (_btn_play, _btn_rew, _btn_s05, _btn_s1, _btn_s2, _btn_s4):
        _b.label.set_color(C_TEXT)
        _b.label.set_fontsize(8.5)

    _slider = Slider(
        _ax_slider, "", 0, max(N_TOTAL - 1, 1),
        valinit=0, valstep=1,
        color=C_TEAL, initcolor=C_TEAL, track_color=BG3,
    )
    _slider.label.set_color(C_TEXT)
    _slider.valtext.set_color(C_MUTED)
    _slider.valtext.set_fontsize(7)

    _txt_pb = fig.text(
        _ax_ctr_x, 0.031, f"0 / {N_TOTAL - 1}",
        color=C_TEXT, fontsize=8, va="center",
        bbox=dict(facecolor=BG3, edgecolor=C_BORDER, boxstyle="round,pad=0.3"),
    )


    def _on_play_pause(_):
        _pb["playing"] = not _pb["playing"]
        _btn_play.label.set_text(
            "Pause" if _pb["playing"] else "Play"
        )


    def _on_rewind(_):
        _pb["idx"]     = 0.0
        _pb["playing"] = True
        _btn_play.label.set_text("Pause")
        _reset_accumulators()
        _slider.eventson = False
        _slider.set_val(0)
        _slider.eventson = True


    def _set_speed(v: float) -> None:
        _pb["speed"] = v
        _btn_s1.ax.set_facecolor("#1a2a1a"  if abs(v - 1.0) < 1e-3 else "#2a2318")
        _btn_s05.ax.set_facecolor("#1a2a1a" if abs(v - 0.5) < 1e-3 else "#2a2318")
        _btn_s2.ax.set_facecolor("#1a2a1a"  if abs(v - 2.0) < 1e-3 else "#2a2318")
        _btn_s4.ax.set_facecolor("#1a2a1a"  if abs(v - 4.0) < 1e-3 else "#2a2318")


    def _on_slider_changed(val):
        if _slider_drag[0]:
            new_idx      = max(0, min(N_TOTAL - 1, int(round(val))))
            _pb["idx"]   = float(new_idx)
            _reset_accumulators()


    def _on_slider_press(_evt):
        _slider_drag[0] = True


    def _on_slider_release(_evt):
        _slider_drag[0] = False


    _btn_play.on_clicked(_on_play_pause)
    _btn_rew.on_clicked(_on_rewind)
    _btn_s05.on_clicked(lambda _: _set_speed(0.5))
    _btn_s1.on_clicked( lambda _: _set_speed(1.0))
    _btn_s2.on_clicked( lambda _: _set_speed(2.0))
    _btn_s4.on_clicked( lambda _: _set_speed(4.0))
    _slider.on_changed(_on_slider_changed)
    fig.canvas.mpl_connect("button_press_event",   _on_slider_press)
    fig.canvas.mpl_connect("button_release_event", _on_slider_release)

    _set_speed(1.0)   # highlight ×1 at startup


    # =============================================================================
    # DoA pipeline  (offline version – no warmup gate, no hardware retune)
    # =============================================================================

    def _pipeline_pb(X: np.ndarray) -> dict:
        """
        Process one pre-recorded IQ frame through the full DoA pipeline.
        Identical to the realtime pipeline minus the warmup gate and auto-tune.
        """
        global _angle_phasor, _last_fft

        _empty = {
            "spec":     _flat.copy(), "est_deg": 0.0,
            "papr":     0.0,          "snr":     0.0,
            "cond":     1.0,          "ev_db":   np.zeros(N_ANT),
            "coh":      np.eye(N_ANT),
            "fft_db":   _last_fft.copy(),
            "phase_od": [0.0] * (N_ANT - 1),
            "squelched": True,
        }

        # 1. Phase correction
        if any(o != 0.0 for o in _PHASE_OFFS):
            X = apply_phase_correction(X, _PHASE_OFFS)

        # 2. Power squelch
        if _SQ_EN and measure_power_db(X) < _SQ_THR:
            return _empty

        # 3. IQ FFT
        _hann    = np.hanning(FFT_N)
        _fft_acc = np.zeros(FFT_N)
        for _k in range(N_ANT):
            _seg      = (X[_k, :FFT_N] if X.shape[1] >= FFT_N
                         else np.pad(X[_k], (0, FFT_N - X.shape[1])))
            _fft_acc += np.abs(np.fft.fft(_seg * _hann)) ** 2
        _last_fft  = np.fft.fftshift(10.0 * np.log10(_fft_acc / N_ANT + 1e-20))
        _last_fft -= np.max(_last_fft)

        # 4. Covariance EMA
        R_ant = _cov_acc.update((X @ X.conj().T) / X.shape[1])

        # 5. DoA estimation (algorithms handle VULA conversion internally)
        if ALGO == "ROOT-MUSIC":
            est_deg, spec, papr_val = doa_root_music(
                X, cfg, decorrelation=DECORR, R_in=R_ant)
        elif ALGO == "ESPRIT":
            est_deg, spec, papr_val = doa_esprit(
                X, cfg, decorrelation=DECORR, R_in=R_ant)
        elif ALGO == "CAPON":
            _, spec = doa_capon(X, cfg, decorrelation=DECORR, R_in=R_ant)
            est_deg  = float(np.rad2deg(theta_scan[int(np.argmax(spec))]) % 360.0)
            papr_val = papr_db(spec)
        elif ALGO == "ML":
            _, spec = doa_ml(X, cfg, decorrelation=DECORR, R_in=R_ant)
            est_deg  = float(np.rad2deg(theta_scan[int(np.argmax(spec))]) % 360.0)
            papr_val = papr_db(spec)
        else:   # MUSIC (default)
            _, spec = doa_music(X, cfg, decorrelation=DECORR, R_in=R_ant)
            est_deg  = float(np.rad2deg(theta_scan[int(np.argmax(spec))]) % 360.0)
            papr_val = papr_db(spec)

        # 6. Circular EMA smoothing
        if _ANG_ALPHA > 0.0:
            _new_ph       = np.exp(1j * np.deg2rad(est_deg))
            _angle_phasor = _ANG_ALPHA * _angle_phasor + (1.0 - _ANG_ALPHA) * _new_ph
            est_deg       = float(np.rad2deg(np.angle(_angle_phasor)) % 360.0)

        # 7. Auxiliary metrics
        snr_val  = snr_from_covariance(R_ant)
        cond_val = condition_number(R_ant)
        ev_db    = eigenvalue_spread_db(R_ant)
        coh      = coherence_matrix(R_ant)
        phase_od = [
            float(np.angle(R_ant[0, k + 1])) if N_ANT > k + 1 else 0.0
            for k in range(N_ANT - 1)
        ]

        return {
            "spec":     spec,  "est_deg":  est_deg,
            "papr":     float(papr_val),
            "snr":      float(snr_val),
            "cond":     float(cond_val),
            "ev_db":    ev_db, "coh":      coh,
            "fft_db":   _last_fft.copy(),
            "phase_od": phase_od,
            "squelched": False,
        }


    # =============================================================================
    # Animation update
    # =============================================================================

    _fps_t = [time.time()]
    _fps_v = [0.0]

    # Fractional frame counter to handle ×0.5 speed (advance 0.5 per tick)
    _frame_accum = [0.0]


    def update(_anim_i):
        # FPS measurement
        _now = time.time()
        _dt  = max(_now - _fps_t[0], 1e-6)
        _fps_t[0] = _now
        _fps_v[0] = 0.9 * _fps_v[0] + 0.1 / _dt

        cur_idx = max(0, min(N_TOTAL - 1, int(_pb["idx"])))

        # Advance playback pointer
        if _pb["playing"]:
            _pb["idx"]      = min(_pb["idx"] + 1.0, float(N_TOTAL - 1))
            if _pb["idx"] >= N_TOTAL - 1:
                _pb["playing"]       = False
                _btn_play.label.set_text("Play")

        # Sync slider (suppress slider callback while updating programmatically)
        if not _slider_drag[0]:
            _slider.eventson = False
            _slider.set_val(cur_idx)
            _slider.eventson = True

        # Timestamp label
        _rec_ts = float(timestamps[cur_idx]) if cur_idx < len(timestamps) else 0.0
        _ts_str = time.strftime("%H:%M:%S", time.localtime(_rec_ts)) if _rec_ts > 0 else ""
        _txt_pb.set_text(f"{cur_idx} / {N_TOTAL - 1}")

        # Process frame
        X = frames_all[cur_idx].astype(np.complex128)
        try:
            d = _pipeline_pb(X)
        except Exception as exc:
            txt_status.set_text(f"ERR {exc}")
        cal_rad = np.deg2rad(_pb["cal"])

        # ── A: Pseudospectrum ─────────────────────────────────────────────────────
        line_spec.set_xdata(theta_scan - cal_rad)
        line_spec.set_ydata(d["spec"])
        line_est_m.set_xdata([est_rad, est_rad])
        if sq:
            line_spec.set_color(C_DIM)
            txt_music.set_text("SQUELCH"); txt_music.set_color(C_DIM)
        else:
            line_spec.set_color(C_BLUE)
            txt_music.set_text(f"{display_deg:.1f}\u00b0   PAPR {d['papr']:.1f} dB")
            txt_music.set_color(C_BLUE)

        # ── B: Compass ────────────────────────────────────────────────────────────
        if sq:
            needle.set_color(C_DIM); needle_b.set_color(C_DIM)
            unc_fill.set_xy(np.column_stack([np.zeros(40), np.zeros(40)]))
            unc_fill.set_alpha(0.0)
            txt_est.set_text("---"); txt_est.set_color(C_DIM)
        else:
            col = C_ROSE if d["papr"] < 6 else (C_AMBER if d["papr"] < 12 else C_TEAL)
            needle.set_color(col); needle_b.set_color(col)
            needle.set_data([est_rad, est_rad], [0, 0.85])
            needle_b.set_data([est_rad + np.pi, est_rad + np.pi], [0, 0.38])
            txt_est.set_text(f"{display_deg:.1f}\u00b0"); txt_est.set_color(col)

        state_str = f"\u25b6 PLAY  {_ts_str}" if _pb["playing"] else f"\u23f8 PAUSED  {_ts_str}"
        txt_status.set_text(state_str)
        txt_status.set_color(C_LIME if _pb["playing"] else C_AMBER)
        txt_fps.set_text(f"{_fps_v[0]:.0f} fps")
        txt_metrics.set_text(
            f"SNR {d['snr']:.0f}  \u2502  PAPR {d['papr']:.0f}  \u2502  "
            f"\u03ba {d['cond']:.0f}  \u2502  x{_pb['speed']:.1f}  \u2502  "
            f"fr {cur_idx}/{N_TOTAL-1}"
        )

        # ── C: Angle history ──────────────────────────────────────────────────────
        _h_angle.append(display_deg if not sq else np.nan)
        line_ahist.set_ydata(list(_h_angle))

        _ang_arr = np.array([v for v in _h_angle if not np.isnan(v)])
        if len(_ang_arr) > 4:
            _sin_m  = np.mean(np.sin(np.deg2rad(_ang_arr)))
            _cos_m  = np.mean(np.cos(np.deg2rad(_ang_arr)))
            _R_mean = np.sqrt(_sin_m ** 2 + _cos_m ** 2)
            _sigma  = min(float(np.rad2deg(np.sqrt(-2.0 * np.log(max(_R_mean, 1e-9))))), 180.0)
            _col    = C_ROSE if _sigma > 20 else (C_AMBER if _sigma > 8 else C_TEAL)
            txt_sigma.set_text(f"\u03c3 = {_sigma:.1f}\u00b0"); txt_sigma.set_color(_col)

            # Update uncertainty wedge on compass
            if not sq and _sigma > 0.5:
                _sigma_rad = np.deg2rad(min(_sigma, 90.0))
                _w_theta = np.linspace(est_rad - _sigma_rad, est_rad + _sigma_rad, 40)
                _w_r     = np.concatenate([[0], np.ones(38) * 0.92, [0]])
                unc_fill.set_xy(np.column_stack([_w_theta, _w_r]))
                unc_fill.set_facecolor(_col)
                unc_fill.set_alpha(0.12)
            elif not sq:
                unc_fill.set_alpha(0.0)
        else:
            txt_sigma.set_text("\u03c3 = ---"); txt_sigma.set_color(C_DIM)
            unc_fill.set_alpha(0.0)

        # ── D: Eigenvalue spread ──────────────────────────────────────────────────
        if not sq:
            _ev = d["ev_db"]
            for _i, _bar in enumerate(bars_eig):
                _bar.set_height(float(_ev[_i]) if _i < len(_ev) else 0.0)
            ax_eig.set_ylim(
                bottom=min(0.0, float(np.min(_ev)) - 1),
                top=max(float(np.max(_ev)) + 2, 5.0),
            )
            txt_cond.set_text(f"\u03ba = {d['cond']:.0f}")

        # ── E: Coherence matrix ───────────────────────────────────────────────────
        if not sq:
            im_coh.set_data(d["coh"])
            for _i in range(N_ANT):
                for _j in range(N_ANT):
                    _coh_txts[_i][_j].set_text(f"{d['coh'][_i, _j]:.2f}")
            _mask  = 1 - np.eye(N_ANT)
            _mu_od = float(np.mean(d["coh"] * _mask))
            _qual  = "high" if _mu_od > 0.6 else ("moderate" if _mu_od > 0.3 else "low")
            txt_coh_lbl.set_text(f"|\u03bc| = {_mu_od:.3f}  ({_qual})")

        # ── F: PAPR + SNR history ─────────────────────────────────────────────────
        _h_papr.append(d["papr"]); _h_snr.append(d["snr"])
        line_papr.set_ydata(list(_h_papr))
        line_snr.set_ydata(list(_h_snr))
        txt_pq.set_text(f"PAPR {d['papr']:.1f} dB   SNR {d['snr']:.1f} dB")

        # ── G: IQ FFT ─────────────────────────────────────────────────────────────
        if not sq:
            line_fft.set_ydata(d["fft_db"])
            _pk_bin = int(np.argmax(d["fft_db"]))
            _pk_kHz = float(_fft_freqs[_pk_bin])
            txt_fft_pk.set_text(f"peak  {_pk_kHz:+.2f} kHz")

        # ── H: Off-diagonal phase ─────────────────────────────────────────────────
        for _k in range(N_ANT - 1):
            _h_ph[_k].append(d["phase_od"][_k])
            _ph_lines[_k].set_ydata(list(_h_ph[_k]))
        _stds    = [float(np.std(list(_h_ph[_k]))) for _k in range(N_ANT - 1)]
        _std_max = max(_stds) if _stds else 0.0
        _qual    = "stable" if _std_max < 0.3 else ("moderate" if _std_max < 0.7 else "unstable")
        _col     = C_TEAL if _std_max < 0.3 else (C_AMBER if _std_max < 0.7 else C_ROSE)
        txt_ph_lbl.set_text(f"\u03c3\u03c6 = {_std_max:.3f} rad  ({_qual})")
        txt_ph_lbl.set_color(_col)



    # =============================================================================
    # Launch
    # =============================================================================

    ani = animation.FuncAnimation(
        fig, update,
        interval=INTERVAL,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
