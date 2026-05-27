#!/usr/bin/env python3
"""
Offline DoA visualiser — replay a .npz recording with the same GUI panels
as the real-time system, plus a time slider to scrub through the data.

Usage:
    python3 offline_visualiser.py data/doa_iridium/doa_iridium_20260526_155627.npz
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path: sys.path.insert(0, _SRC)
if _HERE not in sys.path: sys.path.insert(0, _HERE)

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#1a1d27"; BG2   = "#21253a"; BG3  = "#2a2f47"
C_BDR = "#3b4263"; C_MUT = "#8891b0"; C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIO  = "#a78bfa"; C_ROSE = "#f16b6f"; C_LIME  = "#6dd97d"


def main():
    p = argparse.ArgumentParser(description="Offline DoA visualiser")
    p.add_argument("input", help="Path to .npz recording")
    p.add_argument("--window", type=int, default=200, help="Visible points in history")
    args = p.parse_args()

    d = np.load(args.input, allow_pickle=True)
    az = d['az_deg']; el = d['el_deg']; papr = d['papr_db']; snr = d['snr_db']
    cfo = d['sat_cfo_hz']; t = d['t']; phase = d['phase_diff']
    n = len(az)
    t_rel = t - t[0]

    print(f"Loaded {n} bursts from {os.path.basename(args.input)}")
    print(f"Duration: {t_rel[-1]:.0f}s  Az: {np.median(az):.0f}±{np.std(az):.0f}°  El: {np.median(el):.0f}±{np.std(el):.0f}°")

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG, dpi=100)
    fig.patch.set_facecolor(BG)

    # ── Layout ──────────────────────────────────────────────────────────────
    gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[19, 1],
                                  left=0.02, right=0.99, top=0.97, bottom=0.04, hspace=0.05)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0, 0],
                                               width_ratios=[0.9, 1.3], wspace=0.04)
    # ── Skyplot (left) ──────────────────────────────────────────────────────
    ax_sky = fig.add_subplot(gs_top[0, 0], projection="polar", facecolor=BG2)
    ax_sky.set_theta_zero_location("N"); ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90); ax_sky.set_rticks([15, 30, 60, 90])
    ax_sky.set_yticklabels(["75°", "60°", "30°", "0°"], fontsize=7, color=C_MUT)
    ax_sky.tick_params(colors=C_MUT, labelsize=7); ax_sky.set_facecolor(BG2)
    for sp in ax_sky.spines.values(): sp.set_edgecolor(C_BDR)
    ax_sky.grid(color=C_BDR, lw=0.5, alpha=0.4)
    ax_sky.set_title("Skyplot", color=C_TEXT, fontsize=11, pad=15)
    sky_trail, = ax_sky.plot([], [], "o", color=C_AMBER, ms=3, alpha=0.3, zorder=5)
    sky_dot, = ax_sky.plot([], [], "o", color=C_AMBER, ms=14, zorder=8, mec="white", mew=2)
    sky_lbl = ax_sky.text(0, 0, "", ha="left", va="bottom", color=C_AMBER, fontsize=8, fontweight="bold", zorder=9)

    # ── Right panels ────────────────────────────────────────────────────────
    gs_right = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_top[0, 1],
                                                  height_ratios=[2, 2, 1.5, 1.5], hspace=0.4)

    # Az/El vs time
    ax_hist = fig.add_subplot(gs_right[0, 0], facecolor=BG2)
    ax_hist.set_facecolor(BG2)
    ax_hist.set_title("Azimuth / Elevation vs time", color=C_TEXT, fontsize=9)
    ax_hist.set_xlabel("Time [s]", color=C_MUT, fontsize=7)
    ax_hist.set_ylabel("Angle [°]", color=C_MUT, fontsize=7)
    ax_hist.set_ylim(-5, 375); ax_hist.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_hist.spines.values(): sp.set_edgecolor(C_BDR)
    ax_hist.grid(color=C_BDR, lw=0.3, alpha=0.4)
    line_az, = ax_hist.plot([], [], "-", color=C_AMBER, lw=1.5, label="Az")
    line_el, = ax_hist.plot([], [], "-", color=C_TEAL, lw=1.2, label="El")
    line_cursor = ax_hist.axvline(0, color="white", lw=0.8, alpha=0.6)
    ax_hist.legend(loc="upper right", fontsize=7, facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # CFO vs time
    ax_cfo = fig.add_subplot(gs_right[1, 0], facecolor=BG2)
    ax_cfo.set_facecolor(BG2)
    ax_cfo.set_title("CFO (Doppler) vs time", color=C_TEXT, fontsize=9)
    ax_cfo.set_xlabel("Time [s]", color=C_MUT, fontsize=7)
    ax_cfo.set_ylabel("CFO [kHz]", color=C_MUT, fontsize=7)
    ax_cfo.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_cfo.spines.values(): sp.set_edgecolor(C_BDR)
    ax_cfo.grid(color=C_BDR, lw=0.3, alpha=0.4)
    ax_cfo.axhline(0, color=C_BDR, lw=0.6)
    line_cfo, = ax_cfo.plot([], [], "-", color=C_BLUE, lw=1.2)
    line_cfo_cursor = ax_cfo.axvline(0, color="white", lw=0.8, alpha=0.6)

    # PAPR + SINR vs time
    ax_qual = fig.add_subplot(gs_right[2, 0], facecolor=BG2)
    ax_qual.set_facecolor(BG2)
    ax_qual.set_title("PAPR / SINR vs time", color=C_TEXT, fontsize=8)
    ax_qual.set_xlabel("Time [s]", color=C_MUT, fontsize=7)
    ax_qual.set_ylabel("dB", color=C_MUT, fontsize=7)
    ax_qual.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_qual.spines.values(): sp.set_edgecolor(C_BDR)
    ax_qual.grid(color=C_BDR, lw=0.3, alpha=0.4)
    line_papr, = ax_qual.plot([], [], "-", color=C_AMBER, lw=1.0, label="PAPR")
    line_sinr, = ax_qual.plot([], [], "-", color=C_TEAL, lw=1.0, label="SINR")
    ax_qual.legend(loc="upper right", fontsize=6, facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # Phase diffs vs time
    ax_ph = fig.add_subplot(gs_right[3, 0], facecolor=BG2)
    ax_ph.set_facecolor(BG2)
    ax_ph.set_title("ΔΦ CH1..4 – CH0 vs time", color=C_TEXT, fontsize=8)
    ax_ph.set_xlabel("Time [s]", color=C_MUT, fontsize=7)
    ax_ph.set_ylabel("ΔΦ [°]", color=C_MUT, fontsize=7)
    ax_ph.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_ph.spines.values(): sp.set_edgecolor(C_BDR)
    ax_ph.grid(color=C_BDR, lw=0.3, alpha=0.4)
    _phc = [C_BLUE, C_TEAL, C_AMBER, C_VIO]
    ph_lines = [ax_ph.plot([], [], "-", color=_phc[i], lw=0.8, alpha=0.8, label=f"CH{i+1}")[0] for i in range(4)]
    ax_ph.legend(loc="upper right", fontsize=5, ncol=4, facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    fig.suptitle(f"Offline Playback — {os.path.basename(args.input)}  ({n} bursts, {t_rel[-1]:.0f}s)",
                  color=C_TEXT, fontsize=8, y=0.995)

    # ── Time slider ─────────────────────────────────────────────────────────
    ax_slider = fig.add_subplot(gs_outer[1, 0], facecolor=BG2)
    ax_slider.set_facecolor(BG2)
    slider = Slider(ax_slider, "Time", 0, n-1, valinit=0, valstep=1, color=C_AMBER)
    slider.label.set_color(C_TEXT); slider.valtext.set_color(C_MUT)

    # ── Update function ─────────────────────────────────────────────────────
    def update(idx):
        idx = int(idx)
        lo = max(0, idx - args.window); hi = min(n, idx + 1)
        t_win = t_rel[lo:hi]; az_win = az[lo:hi]; el_win = el[lo:hi]
        cfo_win = cfo[lo:hi] / 1000; papr_win = papr[lo:hi]; snr_win = snr[lo:hi]
        t_now = t_rel[idx]

        # Skyplot: trail + current position
        trail_n = min(50, idx)
        if trail_n > 0:
            tr_az = az[max(0, idx-trail_n):idx+1]
            tr_el = el[max(0, idx-trail_n):idx+1]
            sky_trail.set_data(np.deg2rad(tr_az), 90 - tr_el)
        else:
            sky_trail.set_data([], [])
        sky_dot.set_data([np.deg2rad(az[idx])], [90 - el[idx]])
        sky_lbl.set_text(f"az={az[idx]:.0f}°\nel={el[idx]:.0f}°")
        sky_lbl.set_position((np.deg2rad(az[idx]) + 0.15, 90 - el[idx] + 6))
        sky_lbl.set_transform(ax_sky.transData)

        # History lines
        line_az.set_data(t_win, az_win); line_el.set_data(t_win, el_win)
        line_cursor.set_xdata([t_now, t_now])
        ax_hist.set_xlim(max(0, t_now - args.window * (t_rel[-1]/n)*3), min(t_rel[-1], t_now + (t_rel[-1]/n)*10))

        # CFO
        line_cfo.set_data(t_win, cfo_win)
        line_cfo_cursor.set_xdata([t_now, t_now])
        ax_cfo.set_xlim(*ax_hist.get_xlim())

        # Quality
        line_papr.set_data(t_win, papr_win); line_sinr.set_data(t_win, snr_win)
        ax_qual.set_xlim(*ax_hist.get_xlim())

        # Phase
        for i in range(4):
            ph_lines[i].set_data(t_win, phase[lo:hi, i])
        ax_ph.set_xlim(*ax_hist.get_xlim())

    slider.on_changed(update)
    update(0)
    plt.show()

if __name__ == "__main__":
    main()
