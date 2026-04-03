#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: KrakenSDR pysdr DoA – Real-Time
# Author: KrakenSDR pysdr-DoA translation
# Description: GRC translation of pysdr_doa_realtime.py. Identical pipeline: KrakenIQSource → phase-correction → squelch → amplitude-normalise → EMA covariance → MUSIC/Root-MUSIC/Capon/ML/ESPRIT → circular-EMA angle → 8 matplotlib panels + Record/Calibration buttons.

# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore


def snipfcn_realtime_doa_snippet(self):
    # =======================================================================
    # pysdr_doa_realtime.py — GRC translation
    # Startup config dialog → 8-panel matplotlib DoA dashboard.
    # All mutable shared state lives in _S (SimpleNamespace) to avoid the
    # exec()-scope pitfall where `global` declarations are ignored.
    # =======================================================================
    import os as _os, sys as _sys, time as _time, json as _json
    import threading as _threading, collections as _collections
    from types import SimpleNamespace as _NS

    _WORKSPACE = '/workspace'
    _PYSDR_DIR = _os.path.join(_WORKSPACE, 'pysdr_doa')
    for _p in (_WORKSPACE, _PYSDR_DIR):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    # ── Imports ────────────────────────────────────────────────────────────
    import numpy as _np
    import matplotlib as _mpl
    _mpl.use('Qt5Agg')
    import matplotlib.pyplot as _plt
    import matplotlib.animation as _animation
    import matplotlib.gridspec as _gridspec
    from matplotlib.widgets import Button as _Button

    from doa_algorithms import (
        ArrayConfig as _ArrayConfig, Geometry as _Geometry,
        doa_music as _doa_music, doa_root_music as _doa_root_music,
        doa_capon as _doa_capon, doa_ml as _doa_ml, doa_esprit as _doa_esprit,
        apply_phase_correction as _apply_phase_correction,
        measure_power_db as _measure_power_db,
        snr_from_covariance as _snr_from_cov,
        papr_db as _papr_db,
        condition_number as _cond_num,
        eigenvalue_spread_db as _eig_spread,
        coherence_matrix as _coh_mat,
        CovarianceAccumulator as _CovAcc,
    )
    from kraken_iq_source import KrakenIQSource as _KrakenIQSource

    # ── Colour palette ─────────────────────────────────────────────────────
    BG       = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
    C_BORDER = "#3b4263"; C_DIM = "#4e5680"
    C_BLUE   = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER  = "#f4a431"
    C_VIOLET = "#a78bfa"; C_ROSE  = "#f16b6f"; C_LIME   = "#6dd97d"
    C_SKY    = "#93c5fd"; C_TEXT  = "#d8dae8"; C_MUTED  = "#8891b0"

    # ══════════════════════════════════════════════════════════════════════
    # STARTUP CONFIG DIALOG (tkinter, dark-themed)
    # ══════════════════════════════════════════════════════════════════════
    import tkinter as _tk
    import tkinter.ttk as _ttk

    _CFG = {}   # filled by the dialog; keys match the GRC variable names

    def _run_config_dialog():
        root = _tk.Tk()
        root.title("KrakenSDR DoA — Configuration")
        root.configure(bg="#1a1d27")
        root.resizable(False, False)

        TK_BG   = "#1a1d27"; TK_BG2 = "#21253a"; TK_BG3 = "#2a2f47"
        TK_FG   = "#d8dae8"; TK_MUT = "#8891b0"; TK_ACC = "#5ea4e0"
        TK_BTN  = "#3b4263"; TK_GRN = "#6dd97d"; TK_RED = "#f16b6f"
        TK_FONT = ("Segoe UI", 9)
        TK_HEAD = ("Segoe UI", 10, "bold")

        style = _ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", background=TK_BG, foreground=TK_FG, font=TK_FONT,
                        fieldbackground=TK_BG2, selectbackground=TK_ACC,
                        selectforeground=TK_BG, troughcolor=TK_BG3,
                        bordercolor=TK_BTN, darkcolor=TK_BG2, lightcolor=TK_BG2)
        style.configure("TLabel",  background=TK_BG,  foreground=TK_FG)
        style.configure("TEntry",  fieldbackground=TK_BG2, foreground=TK_FG, insertcolor=TK_FG)
        style.configure("TCombobox", fieldbackground=TK_BG2, foreground=TK_FG,
                        selectbackground=TK_ACC, arrowcolor=TK_FG)
        style.map("TCombobox", fieldbackground=[("readonly", TK_BG2)])
        style.configure("TScale",  background=TK_BG, troughcolor=TK_BG3, sliderlength=14)
        style.configure("TFrame",  background=TK_BG)
        style.configure("TSeparator", background=TK_BTN)
        style.configure("Accent.TButton", background=TK_GRN, foreground=TK_BG,
                        font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Cancel.TButton", background=TK_RED, foreground=TK_BG,
                        font=("Segoe UI", 10, "bold"), padding=6)

        # ── helpers ──────────────────────────────────────────────────────
        def _lbl(parent, text, col=TK_MUT, **kw):
            return _ttk.Label(parent, text=text, foreground=col, **kw)

        def _section(parent, text):
            f = _ttk.Frame(parent); f.pack(fill="x", padx=12, pady=(10,2))
            _ttk.Label(f, text=f"  {text}  ", background=TK_BG3, foreground=TK_ACC,
                       font=("Segoe UI", 9, "bold")).pack(side="left")
            _ttk.Separator(f, orient="horizontal").pack(side="left", fill="x", expand=True, padx=4)

        def _row(parent):
            f = _ttk.Frame(parent); f.pack(fill="x", padx=16, pady=3)
            return f

        # ── title ─────────────────────────────────────────────────────────
        hdr = _ttk.Frame(root); hdr.pack(fill="x", padx=0, pady=0)
        _tk.Label(hdr, text="KrakenSDR  DoA  Real-Time",
                  bg=TK_BG3, fg=TK_ACC, font=("Segoe UI", 13, "bold"),
                  pady=10).pack(fill="x")
        _tk.Label(root, text="Configure before starting — all settings can be changed live during run",
                  bg=TK_BG, fg=TK_MUT, font=("Segoe UI", 8)).pack(pady=(2,0))

        # ── RF / Hardware ─────────────────────────────────────────────────
        _section(root, "RF / Hardware")
        r = _row(root)
        _lbl(r, "Centre Freq (MHz)").pack(side="left")
        _v_freq = _tk.StringVar(value=str(float(self.freq_hz)/1e6))
        _ttk.Entry(r, textvariable=_v_freq, width=12).pack(side="left", padx=(6,14))
        _lbl(r, "Gain (dB)").pack(side="left")
        _v_gain = _tk.StringVar(value=str(float(self.gain_db)))
        _ttk.Entry(r, textvariable=_v_gain, width=8).pack(side="left", padx=6)

        r2 = _row(root)
        _lbl(r2, "Host").pack(side="left")
        _v_host = _tk.StringVar(value=self.heimdall_host)
        _ttk.Entry(r2, textvariable=_v_host, width=16).pack(side="left", padx=(6,14))
        _lbl(r2, "Port").pack(side="left")
        _v_port = _tk.StringVar(value=str(self.heimdall_port))
        _ttk.Entry(r2, textvariable=_v_port, width=8).pack(side="left", padx=(6,14))
        _lbl(r2, "Sample Rate (MHz)").pack(side="left")
        _v_fs = _tk.StringVar(value=str(float(self.sample_rate_hz)/1e6))
        _ttk.Entry(r2, textvariable=_v_fs, width=8).pack(side="left", padx=6)

        # ── Array geometry ────────────────────────────────────────────────
        _section(root, "Array Geometry")
        r = _row(root)
        _lbl(r, "Antennas").pack(side="left")
        _v_nr = _tk.StringVar(value=str(self.n_antennas))
        _ttk.Combobox(r, textvariable=_v_nr, values=["2","3","4","5"],
                      state="readonly", width=4).pack(side="left", padx=(6,14))
        _lbl(r, "Geometry").pack(side="left")
        _v_geom = _tk.StringVar(value=self.geometry)
        _ttk.Combobox(r, textvariable=_v_geom, values=["UCA","ULA"],
                      state="readonly", width=6).pack(side="left", padx=(6,14))
        _lbl(r, "Radius/λ (UCA)").pack(side="left")
        _v_rlam = _tk.StringVar(value=str(self.radius_lambda))
        _ttk.Entry(r, textvariable=_v_rlam, width=8).pack(side="left", padx=(6,14))
        _lbl(r, "d/λ (ULA)").pack(side="left")
        _v_dlam = _tk.StringVar(value=str(self.d_lambda))
        _ttk.Entry(r, textvariable=_v_dlam, width=8).pack(side="left", padx=6)

        # ── DoA Algorithm ─────────────────────────────────────────────────
        _section(root, "DoA Algorithm")
        r = _row(root)
        _lbl(r, "Algorithm").pack(side="left")
        _v_algo = _tk.StringVar(value=self.doa_algorithm)
        _ttk.Combobox(r, textvariable=_v_algo,
                      values=["MUSIC","ROOT-MUSIC","CAPON","ML","ESPRIT"],
                      state="readonly", width=12).pack(side="left", padx=(6,14))
        _lbl(r, "Decorrelation").pack(side="left")
        _v_decorr = _tk.StringVar(value=self.decorrelation)
        _ttk.Combobox(r, textvariable=_v_decorr,
                      values=["Off","FBA","TOEP","FBTOEP"],
                      state="readonly", width=10).pack(side="left", padx=(6,14))
        _lbl(r, "Sources (D)").pack(side="left")
        _v_nsig = _tk.StringVar(value=str(self.num_signals))
        _ttk.Combobox(r, textvariable=_v_nsig, values=["1","2","3","4"],
                      state="readonly", width=4).pack(side="left", padx=(6,14))
        _lbl(r, "Scan Points").pack(side="left")
        _v_scan = _tk.StringVar(value=str(self.scan_points))
        _ttk.Combobox(r, textvariable=_v_scan,
                      values=["90","180","360","720","1440"],
                      state="readonly", width=6).pack(side="left", padx=6)

        # ── Signal processing ─────────────────────────────────────────────
        _section(root, "Signal Processing")
        r = _row(root)
        _lbl(r, "Cov EMA α").pack(side="left")
        _v_cov = _tk.DoubleVar(value=float(self.cov_alpha))
        _ttk.Scale(r, variable=_v_cov, from_=0.0, to=0.99,
                   orient="horizontal", length=120).pack(side="left", padx=(4,2))
        _lbl_cov = _ttk.Label(r, width=5)
        _lbl_cov.pack(side="left", padx=(0,14))
        def _upd_cov(*_): _lbl_cov.config(text=f"{_v_cov.get():.2f}")
        _v_cov.trace_add("write", _upd_cov); _upd_cov()

        _lbl(r, "Angle smooth α").pack(side="left")
        _v_ang = _tk.DoubleVar(value=float(self.angle_smooth_alpha))
        _ttk.Scale(r, variable=_v_ang, from_=0.0, to=0.99,
                   orient="horizontal", length=120).pack(side="left", padx=(4,2))
        _lbl_ang = _ttk.Label(r, width=5)
        _lbl_ang.pack(side="left")
        def _upd_ang(*_): _lbl_ang.config(text=f"{_v_ang.get():.2f}")
        _v_ang.trace_add("write", _upd_ang); _upd_ang()

        r2 = _row(root)
        _lbl(r2, "Amp. Normalise").pack(side="left")
        _v_ampn = _tk.StringVar(value="On" if int(self.amplitude_normalize) else "Off")
        _ttk.Combobox(r2, textvariable=_v_ampn, values=["On","Off"],
                      state="readonly", width=5).pack(side="left", padx=(6,14))
        _lbl(r2, "Squelch").pack(side="left")
        _v_sq = _tk.StringVar(value="Enabled" if int(self.squelch_enabled) else "Disabled")
        _ttk.Combobox(r2, textvariable=_v_sq, values=["Enabled","Disabled"],
                      state="readonly", width=10).pack(side="left", padx=(6,14))
        _lbl(r2, "Squelch Thr (dBW)").pack(side="left")
        _v_sqthr = _tk.StringVar(value=str(float(self.squelch_threshold_db)))
        _ttk.Entry(r2, textvariable=_v_sqthr, width=8).pack(side="left", padx=6)

        # ── Phase offsets ─────────────────────────────────────────────────
        _section(root, "Phase Offsets (deg, one per antenna)")
        r = _row(root)
        _lbl(r, "Offsets").pack(side="left")
        _v_phase = _tk.StringVar(value=str(self.phase_offsets_deg))
        _ttk.Entry(r, textvariable=_v_phase, width=40).pack(side="left", padx=6)

        # ── Buttons ───────────────────────────────────────────────────────
        _ttk.Separator(root, orient="horizontal").pack(fill="x", padx=12, pady=10)
        bf = _ttk.Frame(root); bf.pack(pady=(0,12))
        _cancelled = [False]

        def _on_start():
            try:
                _CFG["freq_hz"]             = float(_v_freq.get()) * 1e6
                _CFG["gain_db"]             = float(_v_gain.get())
                _CFG["heimdall_host"]        = _v_host.get().strip()
                _CFG["heimdall_port"]        = int(_v_port.get())
                _CFG["sample_rate_hz"]       = float(_v_fs.get()) * 1e6
                _CFG["n_antennas"]           = int(_v_nr.get())
                _CFG["geometry"]             = _v_geom.get().strip().upper()
                _CFG["radius_lambda"]        = float(_v_rlam.get())
                _CFG["d_lambda"]             = float(_v_dlam.get())
                _CFG["doa_algorithm"]        = _v_algo.get().strip().upper()
                _CFG["decorrelation"]        = _v_decorr.get().strip().upper()
                _CFG["num_signals"]          = int(_v_nsig.get())
                _CFG["scan_points"]          = int(_v_scan.get())
                _CFG["cov_alpha"]            = float(_v_cov.get())
                _CFG["angle_smooth_alpha"]   = float(_v_ang.get())
                _CFG["amplitude_normalize"]  = _v_ampn.get() == "On"
                _CFG["squelch_enabled"]      = _v_sq.get() == "Enabled"
                _CFG["squelch_threshold_db"] = float(_v_sqthr.get())
                import ast
                _CFG["phase_offsets_deg"]    = list(ast.literal_eval(_v_phase.get()))
                root.destroy()
            except Exception as exc:
                _tk.messagebox.showerror("Input Error", str(exc), parent=root)

        def _on_cancel():
            _cancelled[0] = True; root.destroy()

        _ttk.Button(bf, text="  ▶  Start DoA  ", style="Accent.TButton",
                    command=_on_start).pack(side="left", padx=8)
        _ttk.Button(bf, text="  ✕  Cancel  ", style="Cancel.TButton",
                    command=_on_cancel).pack(side="left", padx=8)

        root.update_idletasks()
        w, h = root.winfo_reqwidth(), root.winfo_reqheight()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        root.lift(); root.focus_force()
        root.mainloop()
        if _cancelled[0]: raise SystemExit(0)

    _run_config_dialog()

    # ══════════════════════════════════════════════════════════════════════
    # Apply config (dialog values override GRC widget values)
    # ══════════════════════════════════════════════════════════════════════
    _FREQ_HZ    = _CFG.get("freq_hz",       self.freq_hz)
    _GAIN_DB    = _CFG.get("gain_db",        self.gain_db)
    _HOST       = _CFG.get("heimdall_host",  self.heimdall_host)
    _PORT       = _CFG.get("heimdall_port",  self.heimdall_port)
    _FS         = _CFG.get("sample_rate_hz", self.sample_rate_hz)
    _NR         = _CFG.get("n_antennas",     self.n_antennas)
    _GEOM_STR   = _CFG.get("geometry",       self.geometry)
    _R_LAMBDA   = _CFG.get("radius_lambda",  self.radius_lambda)
    _D_LAMBDA   = _CFG.get("d_lambda",       self.d_lambda)
    _SCAN_PTS   = _CFG.get("scan_points",    self.scan_points)
    _N_SIG      = _CFG.get("num_signals",    self.num_signals)
    _PHASE_OFFS = _CFG.get("phase_offsets_deg", self.phase_offsets_deg)
    _HW_NSAMP   = self.hw_num_samples
    _INTERVAL   = max(40, int(self.interval_ms))

    HIST  = 120
    FFT_N = 512

    # ── Array + config ─────────────────────────────────────────────────────
    _GEOM_E = _Geometry.UCA if _GEOM_STR.upper() == "UCA" else _Geometry.ULA
    _cfg = _ArrayConfig(
        Nr                   = _NR,
        geometry             = _GEOM_E,
        d_lambda             = _D_LAMBDA,
        radius_lambda        = _R_LAMBDA,
        num_expected_signals = _N_SIG,
        num_scan_points      = _SCAN_PTS,
    )
    _theta_scan = _cfg.scan_range()
    _flat       = _np.full(_SCAN_PTS, -40.0)
    _fft_freqs  = _np.fft.fftshift(_np.fft.fftfreq(FFT_N)) * _FS / 1e3
    _x_hist     = _np.arange(HIST)

    # ── Mutable shared state (SimpleNamespace avoids exec global() issues) ─
    _S = _NS(
        # pipeline running values
        angle_phasor   = complex(1.0, 0.0),
        last_fft       = _np.full(FFT_N, -80.0),
        warmup_count   = 0,
        cov_acc        = _CovAcc(alpha=_CFG.get("cov_alpha", self.cov_alpha)),
        # histories
        h_angle  = _collections.deque([_np.nan] * HIST, maxlen=HIST),
        h_papr   = _collections.deque([0.0] * HIST,     maxlen=HIST),
        h_snr    = _collections.deque([0.0] * HIST,     maxlen=HIST),
        h_ph     = [_collections.deque([0.0]*HIST, maxlen=HIST) for _ in range(_NR-1)],
        # autotune
        peak_offset_hist = _collections.deque(maxlen=15),
        last_retune_t    = 0.0,
        active_freq_hz   = float(_FREQ_HZ),
        # recording
        rec_active    = False,
        rec_buffer    = [],
        rec_ts        = [],
        rec_start_t   = 0.0,
        # display state
        est_deg       = 0.0,
        fps           = 0.0,
        t_last        = _time.time(),
        cal_offset    = 0.0,
        retuning      = False,
        warming       = True,
    )
    _WARMUP = max(12, int(1.0 / (1.0 - float(_CFG.get("cov_alpha", self.cov_alpha)))))

    # ── Heimdall connection ────────────────────────────────────────────────
    _kraken = _KrakenIQSource(
        host         = _HOST,
        port         = _PORT,
        ctrl_port    = self.heimdall_ctrl,
        num_channels = _NR,
        freq_hz      = _FREQ_HZ,
        gain_db      = _GAIN_DB,
        verbose      = 0,
    )
    _kraken.start()

    # ── Recording dir ──────────────────────────────────────────────────────
    _REC_DIR = _os.path.normpath(_os.path.join(_PYSDR_DIR, '..', '..', 'recordings'))
    _os.makedirs(_REC_DIR, exist_ok=True)

    # ── Auto-tune constants ────────────────────────────────────────────────
    _AUTOTUNE_THRESH_KHZ = 5.0
    _AUTOTUNE_COOLDOWN   = 8.0

    # ── Figure layout (identical to pysdr_doa_realtime.py) ────────────────
    _mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": C_BORDER, "axes.grid": True,
        "grid.color": C_BORDER, "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "text.color": C_TEXT,
    })

    _fig = _plt.figure(figsize=(20, 10.5), facecolor=BG)
    _fig.patch.set_facecolor(BG)
    _gs = _gridspec.GridSpec(
        2, 4, figure=_fig,
        left=0.04, right=0.97, top=0.93, bottom=0.11,
        hspace=0.50, wspace=0.40,
        width_ratios=[1.25, 1.25, 1.1, 1.4],
        height_ratios=[1.4, 1.0],
    )
    _ax_music  = _fig.add_subplot(_gs[0, 0], polar=True)
    _ax_comp   = _fig.add_subplot(_gs[0, 1], polar=True)
    _ax_hist_a = _fig.add_subplot(_gs[0, 2])
    _ax_eig    = _fig.add_subplot(_gs[0, 3])
    _ax_coh    = _fig.add_subplot(_gs[1, 0])
    _ax_pq     = _fig.add_subplot(_gs[1, 1])
    _ax_fft    = _fig.add_subplot(_gs[1, 2])
    _ax_phase  = _fig.add_subplot(_gs[1, 3])

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_color(C_BORDER); sp.set_linewidth(0.8)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        if title:  ax.set_title(title, color=C_TEXT, fontsize=8.5, pad=5, fontweight="semibold")
        if xlabel: ax.set_xlabel(xlabel, color=C_MUTED, fontsize=7)
        if ylabel: ax.set_ylabel(ylabel, color=C_MUTED, fontsize=7)

    def _style_polar(ax, title=""):
        ax.set_facecolor(BG2)
        ax.spines["polar"].set_color(C_BORDER); ax.spines["polar"].set_linewidth(0.8)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        if title: ax.set_title(title, color=C_TEXT, fontsize=8.5, pad=8, fontweight="semibold")

    # A: MUSIC pseudospectrum
    _style_polar(_ax_music, "Pseudospectrum  [algo / decorr]")
    _ax_music.set_ylim([-40, 2])
    _ax_music.set_rlabel_position(45)
    _ax_music.set_yticks([-30, -20, -10, 0])
    _ax_music.set_yticklabels(["-30", "-20", "-10", "0"], fontsize=6, color=C_MUTED)
    _line_spec,  = _ax_music.plot(_theta_scan, _flat.copy(), color=C_BLUE, linewidth=1.5)
    _line_est_m, = _ax_music.plot([0, 0], [-40, 2], color=C_TEAL, linewidth=2.0, alpha=0.85)
    _txt_music   = _ax_music.text(0.5, -0.07, "", transform=_ax_music.transAxes,
                                  ha="center", fontsize=8, color=C_TEXT)

    # B: Compass
    _style_polar(_ax_comp, "DoA Compass")
    _ax_comp.set_yticks([])
    _ax_comp.set_xticks(_np.linspace(0, 2*_np.pi, 8, endpoint=False))
    _ax_comp.set_xticklabels(["N","NE","E","SE","S","SW","W","NW"], color=C_MUTED, fontsize=8)
    _ax_comp.set_ylim([0, 1])
    _ax_comp.plot(_np.linspace(0, 2*_np.pi, 360), _np.ones(360)*0.92, color=C_BORDER, linewidth=0.8)
    _needle,   = _ax_comp.plot([0, 0], [0, 0.85], color=C_TEAL, linewidth=3.5)
    _needle_b, = _ax_comp.plot([0, 0], [0, 0.38], color=C_TEAL, linewidth=2.0, alpha=0.30)
    _unc_theta = _np.linspace(0, 0.01, 40)
    _unc_r     = _np.concatenate([[0], _np.ones(38)*0.92, [0]])
    _unc_fill, = _ax_comp.fill(_unc_theta, _unc_r, color=C_TEAL, alpha=0.0)
    _txt_est    = _ax_comp.text(0.5, -0.07, "", transform=_ax_comp.transAxes,
                                ha="center", va="top", fontsize=15, fontweight="bold", color=C_TEAL,
                                bbox=dict(facecolor=BG3, edgecolor=C_BORDER, boxstyle="round,pad=0.45"))
    _txt_status  = _ax_comp.text(0.02, 1.05, "", transform=_ax_comp.transAxes,
                                 ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    _txt_fps     = _ax_comp.text(0.98, 1.05, "", transform=_ax_comp.transAxes,
                                 ha="right", va="bottom", fontsize=7.5, color=C_DIM)
    _txt_metrics = _ax_comp.text(0.5, 1.05, "", transform=_ax_comp.transAxes,
                                 ha="center", va="bottom", fontsize=7, color=C_MUTED)

    # C: Angle history
    _style(_ax_hist_a, "Bearing History", "", "deg")
    _ax_hist_a.set_xlim(0, HIST-1); _ax_hist_a.set_ylim(-5, 365)
    _ax_hist_a.set_yticks(range(0, 361, 90))
    for _dg in [0, 90, 180, 270, 360]:
        _ax_hist_a.axhline(_dg, color=C_BORDER, linewidth=0.5)
    _line_ahist, = _ax_hist_a.plot(_x_hist, list(_S.h_angle), color=C_TEAL, linewidth=1.4)
    _txt_sigma   = _ax_hist_a.text(0.02, 0.96, "", transform=_ax_hist_a.transAxes,
                                   fontsize=7.5, color=C_TEAL, va="top")

    # D: Eigenvalue spread
    _style(_ax_eig, "Eigenvalues  [dB]", "", "dB")
    _ax_eig.set_xlim(-0.5, _NR - 0.5); _ax_eig.set_xticks(range(_NR))
    _ax_eig.set_xticklabels([f"\u03bb{i}" for i in range(_NR)], color=C_MUTED, fontsize=7.5)
    _eig_colors = [C_BLUE, C_AMBER, C_VIOLET, C_TEAL, C_ROSE][:_NR]
    _bars_eig   = _ax_eig.bar(range(_NR), [0.0]*_NR, color=_eig_colors, edgecolor="none", alpha=0.85)
    _ax_eig.axvline(_NR - _N_SIG - 0.5, color=C_ROSE, linewidth=1.2, linestyle="--", alpha=0.7,
                    label=f"signal/noise (d={_N_SIG})")
    _ax_eig.legend(fontsize=6.5, labelcolor=C_MUTED, framealpha=0.0, loc="upper right")
    _txt_cond = _ax_eig.text(0.5, 0.96, "", transform=_ax_eig.transAxes,
                             ha="center", va="top", fontsize=7.5, color=C_MUTED)

    # E: Coherence matrix
    _style(_ax_coh, f"Coherence  |\u03bc|  (N={_NR})")
    _ax_coh.set_xticks(range(_NR)); _ax_coh.set_yticks(range(_NR))
    _ax_coh.set_xticklabels([f"ch{k}" for k in range(_NR)], color=C_MUTED, fontsize=7.5)
    _ax_coh.set_yticklabels([f"ch{k}" for k in range(_NR)], color=C_MUTED, fontsize=7.5)
    _im_coh = _ax_coh.imshow(_np.eye(_NR), cmap="Blues", vmin=0, vmax=1,
                             aspect="equal", interpolation="nearest")
    _cb = _plt.colorbar(_im_coh, ax=_ax_coh, fraction=0.04, pad=0.04)
    _cb.ax.tick_params(colors=C_MUTED, labelsize=6); _cb.outline.set_edgecolor(C_BORDER)
    _coh_txts = [[_ax_coh.text(j, i, "", ha="center", va="center",
                               fontsize=9, color=C_TEXT, fontweight="bold")
                  for j in range(_NR)] for i in range(_NR)]
    _txt_coh_lbl = _ax_coh.text(0.5, -0.15, "", transform=_ax_coh.transAxes,
                                 ha="center", fontsize=7, color=C_MUTED)

    # F: PAPR + SNR
    _style(_ax_pq, "PAPR & SNR  [dB]", "", "dB")
    _ax_pq.set_xlim(0, HIST-1); _ax_pq.set_ylim(-2, 36)
    _ax_pq.axhline(6,  color=C_AMBER,  linewidth=0.8, linestyle=":",  alpha=0.6, label="PAPR 6 dB")
    _ax_pq.axhline(12, color=C_AMBER,  linewidth=0.8, linestyle="--", alpha=0.45, label="PAPR 12 dB")
    _ax_pq.axhline(10, color=C_VIOLET, linewidth=0.8, linestyle=":",  alpha=0.5, label="SNR 10 dB")
    _line_papr, = _ax_pq.plot(_x_hist, list(_S.h_papr), color=C_AMBER,  linewidth=1.4, label="PAPR")
    _line_snr,  = _ax_pq.plot(_x_hist, list(_S.h_snr),  color=C_VIOLET, linewidth=1.0, alpha=0.85, label="SNR")
    _ax_pq.legend(fontsize=6.5, labelcolor=C_MUTED, framealpha=0.0, loc="upper right", ncol=2)
    _txt_pq = _ax_pq.text(0.02, 0.96, "", transform=_ax_pq.transAxes, fontsize=7.5, color=C_MUTED, va="top")

    # G: IQ FFT
    _style(_ax_fft, "IQ Spectrum", "offset [kHz]", "dB")
    _line_fft,   = _ax_fft.plot(_fft_freqs, _S.last_fft.copy(), color=C_BLUE, linewidth=0.9)
    _ax_fft.set_xlim(_fft_freqs[0], _fft_freqs[-1]); _ax_fft.set_ylim(-65, 5)
    _ax_fft.axvline(0, color=C_ROSE, linewidth=0.8, linestyle="--", alpha=0.55, label="carrier")
    _ax_fft.legend(fontsize=6.5, labelcolor=C_MUTED, framealpha=0.0)
    _txt_fft_pk   = _ax_fft.text(0.02, 0.96, "", transform=_ax_fft.transAxes, fontsize=7, color=C_MUTED, va="top")
    _txt_fft_freq = _ax_fft.text(0.98, 0.96, f"{_S.active_freq_hz/1e6:.4f} MHz",
                                 transform=_ax_fft.transAxes, ha="right", fontsize=7, color=C_AMBER, va="top")

    # H: Phase stability
    _style(_ax_phase, "Phase Stability  arg(R)", "", "rad")
    _ax_phase.set_xlim(0, HIST-1); _ax_phase.set_ylim(-_np.pi-0.3, _np.pi+0.3)
    _ax_phase.axhline(0, color=C_BORDER, linewidth=0.5)
    _ax_phase.set_yticks([-_np.pi, -_np.pi/2, 0, _np.pi/2, _np.pi])
    _ax_phase.set_yticklabels(["-\u03c0","-\u03c0/2","0","\u03c0/2","\u03c0"], fontsize=6.5, color=C_MUTED)
    _ph_colors = [C_AMBER, C_VIOLET, C_TEAL, C_SKY]
    _ph_lines  = [_ax_phase.plot(_x_hist, list(_S.h_ph[k]), color=_ph_colors[k],
                                 linewidth=1.1, label=f"ang R[0,{k+1}]")[0]
                  for k in range(_NR - 1)]
    _ax_phase.legend(fontsize=6.5, labelcolor=C_MUTED, framealpha=0.0, loc="upper right")
    _txt_ph_lbl = _ax_phase.text(0.02, 0.96, "", transform=_ax_phase.transAxes,
                                 fontsize=7, color=C_TEAL, va="top")

    # Title (updates dynamically via _txt_title)
    _txt_title = _fig.suptitle(
        f"KrakenSDR DoA  \u2502  {_NR}-ant {_GEOM_E.value}  \u2502  "
        f"{_CFG.get('doa_algorithm', self.doa_algorithm)} / {_CFG.get('decorrelation', self.decorrelation)}"
        f"  \u2502  {_S.active_freq_hz/1e6:.3f} MHz",
        color=C_TEXT, fontsize=10.5, fontweight="semibold", y=0.977,
    )

    # ── Bottom controls ──────────────────────────────────────────────────
    _BTN_Y = 0.020; _BTN_H = 0.044; _BADGE_Y = 0.027

    _ax_btn_cal = _fig.add_axes([0.38, _BTN_Y, 0.115, _BTN_H])
    _ax_btn_rst = _fig.add_axes([0.50, _BTN_Y, 0.090, _BTN_H])
    _txt_cal    = _fig.text(0.598, _BADGE_Y, "Offset: 0.0\u00b0",
                            color=C_AMBER, fontsize=8.5, va="center",
                            bbox=dict(facecolor=BG3, edgecolor=C_BORDER, boxstyle="round,pad=0.35"))
    _btn_cal = _Button(_ax_btn_cal, "Set Zero",  color=BG3, hovercolor="#3a4060")
    _btn_rst = _Button(_ax_btn_rst, "Reset Cal", color=BG3, hovercolor="#3a4060")
    for _b in (_btn_cal, _btn_rst):
        _b.label.set_color(C_TEXT); _b.label.set_fontsize(8.5)

    def _reset_angle_state():
        _S.angle_phasor  = complex(1.0, 0.0)
        _S.warmup_count  = 0
        _S.h_angle.clear(); _S.h_angle.extend([_np.nan] * HIST)

    def _on_set_zero(_):
        _S.cal_offset = _S.est_deg
        _txt_cal.set_text(f"Offset: {_S.cal_offset:.1f}\u00b0")
        _txt_cal.set_color(C_LIME); _reset_angle_state()

    def _on_reset_cal(_):
        _S.cal_offset = 0.0
        _txt_cal.set_text("Offset: 0.0\u00b0")
        _txt_cal.set_color(C_AMBER); _reset_angle_state()

    _btn_cal.on_clicked(_on_set_zero)
    _btn_rst.on_clicked(_on_reset_cal)

    _ax_btn_rec  = _fig.add_axes([0.04,  _BTN_Y, 0.115, _BTN_H])
    _ax_btn_stop = _fig.add_axes([0.162, _BTN_Y, 0.095, _BTN_H])
    _txt_rec = _fig.text(0.265, _BADGE_Y, "", color=C_ROSE, fontsize=8.5, va="center",
                         bbox=dict(facecolor=BG3, edgecolor=C_BORDER, boxstyle="round,pad=0.35"),
                         visible=False)
    _btn_rec  = _Button(_ax_btn_rec,  "Record", color="#2a1a1a", hovercolor="#3d2020")
    _btn_stop = _Button(_ax_btn_stop, "Stop",   color=BG3,       hovercolor="#3a4060")
    for _b in (_btn_rec, _btn_stop):
        _b.label.set_color(C_TEXT); _b.label.set_fontsize(8.5)

    def _save_recording():
        if not _S.rec_buffer: _txt_rec.set_visible(False); return
        try:
            frames    = _np.stack([f.astype(_np.complex64) for f in _S.rec_buffer], axis=0)
            ts        = _np.array(_S.rec_ts, dtype=_np.float64)
            stamp     = _time.strftime("%Y%m%d_%H%M%S", _time.localtime(ts[0]))
            name      = f"kraken_{stamp}"
            npz_path  = _os.path.join(_REC_DIR, name + ".npz")
            json_path = _os.path.join(_REC_DIR, name + ".json")
            _np.savez_compressed(npz_path, frames=frames, timestamps=ts)
            meta = {
                "freq_hz": _S.active_freq_hz, "sample_rate_hz": _FS,
                "n_antennas": _NR, "cal_offset_deg": _S.cal_offset,
                "algo": self.doa_algorithm, "decorr": self.decorrelation,
                "cov_alpha": float(self.cov_alpha),
                "geometry": _GEOM_STR, "radius_lambda": _cfg.radius_lambda,
                "d_lambda": _cfg.d_lambda, "n_frames": int(len(_S.rec_buffer)),
                "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
            }
            with open(json_path, "w") as _fh: _json.dump(meta, _fh, indent=2)
            _txt_rec.set_text(f"Saved: {name}  ({len(_S.rec_buffer)} fr)")
            _txt_rec.set_color(C_LIME); _txt_rec.set_visible(True)
            print(f"[REC] Saved {npz_path}  ({len(_S.rec_buffer)} frames)")
        except Exception as exc:
            _txt_rec.set_text(f"Save error: {exc}"); _txt_rec.set_color(C_ROSE); _txt_rec.set_visible(True)
        finally:
            _S.rec_buffer = []; _S.rec_ts = []

    def _on_rec_start(_):
        if _S.rec_active: return
        _S.rec_active = True; _S.rec_buffer = []; _S.rec_ts = []; _S.rec_start_t = _time.time()
        _txt_rec.set_text("REC  00:00  (0 fr)"); _txt_rec.set_color(C_ROSE)
        _txt_rec.get_bbox_patch().set(facecolor=BG3, edgecolor=C_BORDER); _txt_rec.set_visible(True)

    def _on_rec_stop(_):
        if not _S.rec_active: return
        _S.rec_active = False
        _txt_rec.set_text(f"Saving ...  ({len(_S.rec_buffer)} fr)"); _txt_rec.set_color(C_AMBER)
        _txt_rec.get_bbox_patch().set(facecolor=BG3, edgecolor=C_BORDER)
        _threading.Thread(target=_save_recording, daemon=True).start()

    _btn_rec.on_clicked(_on_rec_start)
    _btn_stop.on_clicked(_on_rec_stop)

    # ── Pipeline helpers ─────────────────────────────────────────────────
    def _get_X():
        frame = _kraken.get_frame(timeout=0.005)
        if frame is None: return None, False
        nr = min(_NR, frame.shape[0])
        X  = frame[:nr, :].astype(_np.complex128)
        if _HW_NSAMP > 0: X = X[:, :_HW_NSAMP]
        return X, True

    def _try_autotune():
        if len(_S.peak_offset_hist) < 10: return
        if (_time.time() - _S.last_retune_t) < _AUTOTUNE_COOLDOWN: return
        median_hz = float(_np.median(list(_S.peak_offset_hist)))
        if abs(median_hz) < _AUTOTUNE_THRESH_KHZ * 1e3: return
        new_hz = _S.active_freq_hz + median_hz
        ratio  = new_hz / _S.active_freq_hz
        _cfg.radius_lambda *= ratio; _cfg.d_lambda *= ratio
        _kraken.set_frequency(new_hz)
        _S.active_freq_hz = new_hz; _S.last_retune_t = _time.time()
        _S.cov_acc.reset(); _S.peak_offset_hist.clear(); _reset_angle_state()
        _S.warming = True; _S.warmup_count = 0; _S.retuning = True
        print(f"[AutoTune] → {_S.active_freq_hz/1e6:.4f} MHz  (Δ={median_hz/1e3:+.2f} kHz)")

    def _pipeline(X):
        # Read live GRC widget values on every call so changes take effect without restart
        _ALGO      = self.doa_algorithm
        _DECORR    = self.decorrelation
        _SQ_EN     = bool(self.squelch_enabled)
        _SQ_THR    = float(self.squelch_threshold_db)
        _AMP_NORM  = bool(self.amplitude_normalize)
        _ANG_ALPHA = float(self.angle_smooth_alpha)
        empty = {
            "spec": _flat.copy(), "est_deg": _S.est_deg,
            "papr": 0.0, "snr": 0.0, "cond": 1.0,
            "ev_db": _np.zeros(_NR), "coh": _np.eye(_NR),
            "fft_db": _S.last_fft.copy(),
            "phase_od": [0.0]*(_NR-1), "squelched": False, "warmup_left": 0,
        }
        # 1. Phase correction
        if any(o != 0.0 for o in _PHASE_OFFS):
            X = _apply_phase_correction(X, _PHASE_OFFS)
        # 2. Squelch
        if _SQ_EN and _measure_power_db(X) < _SQ_THR:
            empty["squelched"] = True; return empty
        # 3. Amplitude normalisation
        if _AMP_NORM:
            pwr = _np.sqrt(_np.mean(_np.abs(X)**2, axis=1, keepdims=True)) + 1e-15
            X   = X / pwr
        # 3.5. IQ FFT
        _hann  = _np.hanning(FFT_N)
        _fa    = _np.zeros(FFT_N)
        for _k in range(_NR):
            seg  = X[_k, :FFT_N] if X.shape[1] >= FFT_N else _np.pad(X[_k], (0, FFT_N - X.shape[1]))
            _fa += _np.abs(_np.fft.fft(seg * _hann))**2
        _S.last_fft  = _np.fft.fftshift(10.0 * _np.log10(_fa / _NR + 1e-20))
        _S.last_fft -= _np.max(_S.last_fft)
        _pk_bin      = int(_np.argmax(_S.last_fft))
        _pk_hz       = float(_np.fft.fftshift(_np.fft.fftfreq(FFT_N))[_pk_bin] * _FS)
        _fft_papr    = 10.0 * _np.log10(_np.max(_fa) / (_np.mean(_fa) + 1e-12) + 1e-12)
        if _fft_papr > 6.0: _S.peak_offset_hist.append(_pk_hz)
        else: _S.peak_offset_hist.clear()
        # 4. Covariance EMA
        _S.cov_acc.alpha = float(self.cov_alpha)   # live slider
        if _kraken.last_header.delay_sync_flag == 0:
            _S.cov_acc.reset(); empty["fft_db"] = _S.last_fft.copy()
            empty["squelched"] = True; return empty
        R_ant = _S.cov_acc.update((X @ X.conj().T) / X.shape[1])
        # 5. DoA
        if _ALGO == "ROOT-MUSIC":
            est_deg, spec, papr = _doa_root_music(X, _cfg, decorrelation=_DECORR, R_in=R_ant)
        elif _ALGO == "ESPRIT":
            est_deg, spec, papr = _doa_esprit(X, _cfg, decorrelation=_DECORR, R_in=R_ant)
        elif _ALGO == "CAPON":
            _, spec = _doa_capon(X, _cfg, decorrelation=_DECORR, R_in=R_ant)
            est_deg = float(_np.rad2deg(_theta_scan[int(_np.argmax(spec))]) % 360.0)
            papr    = _papr_db(spec)
        elif _ALGO == "ML":
            _, spec = _doa_ml(X, _cfg, decorrelation=_DECORR, R_in=R_ant)
            est_deg = float(_np.rad2deg(_theta_scan[int(_np.argmax(spec))]) % 360.0)
            papr    = _papr_db(spec)
        else:  # MUSIC
            _, spec = _doa_music(X, _cfg, decorrelation=_DECORR, R_in=R_ant)
            est_deg = float(_np.rad2deg(_theta_scan[int(_np.argmax(spec))]) % 360.0)
            papr    = _papr_db(spec)
        # Warmup gate
        if _S.warming:
            _S.warmup_count += 1
            if _S.warmup_count >= _WARMUP:
                _S.warming = False; _reset_angle_state()
            empty["squelched"]   = True
            empty["warmup_left"] = max(0, _WARMUP - _S.warmup_count)
            empty.update({
                "snr": _snr_from_cov(R_ant), "cond": _cond_num(R_ant),
                "ev_db": _eig_spread(R_ant), "coh": _coh_mat(R_ant),
                "fft_db": _S.last_fft.copy(),
                "phase_od": [float(_np.angle(R_ant[0, k+1])) if _NR > k+1 else 0.0
                             for k in range(_NR-1)],
            })
            return empty
        # Circular EMA
        if _ANG_ALPHA > 0.0:
            _S.angle_phasor = _ANG_ALPHA * _S.angle_phasor + (1.0 - _ANG_ALPHA) * _np.exp(1j * _np.deg2rad(est_deg))
            est_deg         = float(_np.rad2deg(_np.angle(_S.angle_phasor)) % 360.0)
        # Metrics
        return {
            "spec": spec, "est_deg": est_deg,
            "papr": float(papr), "snr": float(_snr_from_cov(R_ant)),
            "cond": float(_cond_num(R_ant)), "ev_db": _eig_spread(R_ant),
            "coh": _coh_mat(R_ant), "fft_db": _S.last_fft.copy(),
            "phase_od": [float(_np.angle(R_ant[0, k+1])) if _NR > k+1 else 0.0
                         for k in range(_NR-1)],
            "squelched": False,
        }

    # ── Animation update (identical to pysdr_doa_realtime.py update()) ───
    def _update(_):
        _S.cov_acc.alpha = float(self.cov_alpha)   # live slider
        now  = _time.time()
        dt   = max(now - _S.t_last, 1e-6)
        _S.t_last = now
        _S.fps    = 0.9 * _S.fps + 0.1 / dt

        X, is_hw = _get_X()
        if X is not None and _S.rec_active:
            _S.rec_buffer.append(X.astype(_np.complex64)); _S.rec_ts.append(_time.time())
            el = _time.time() - _S.rec_start_t
            _txt_rec.set_text(f"REC  {int(el//60):02d}:{int(el%60):02d}  ({len(_S.rec_buffer)} fr)")

        if X is None:
            _txt_status.set_text("\u25cb waiting"); _txt_status.set_color(C_ROSE)
            _txt_fps.set_text(f"{_S.fps:.0f} fps"); return

        try:    d = _pipeline(X)
        except Exception as exc:
            _txt_status.set_text(f"ERR {exc}"); _txt_status.set_color(C_ROSE); return

        _S.est_deg = d["est_deg"]
        disp_deg  = (d["est_deg"] - _S.cal_offset) % 360.0
        est_rad   = _np.deg2rad(disp_deg)
        sq        = d["squelched"]
        warming   = d.get("warmup_left", 0) > 0
        cal_rad   = _np.deg2rad(_S.cal_offset)

        # A
        _line_spec.set_xdata(_theta_scan - cal_rad); _line_spec.set_ydata(d["spec"])
        _line_est_m.set_xdata([est_rad, est_rad])
        if sq:
            _line_spec.set_color(C_DIM)
            if warming: _txt_music.set_text(f"Stabilising... ({d['warmup_left']})"); _txt_music.set_color(C_AMBER)
            else:        _txt_music.set_text("SQUELCH"); _txt_music.set_color(C_DIM)
        else:
            _line_spec.set_color(C_BLUE)
            _txt_music.set_text(f"{disp_deg:.1f}\u00b0   PAPR {d['papr']:.1f} dB"); _txt_music.set_color(C_BLUE)

        # B
        if sq:
            _needle.set_color(C_DIM); _needle_b.set_color(C_DIM)
            _unc_fill.set_xy(_np.column_stack([_np.zeros(40), _np.zeros(40)])); _unc_fill.set_alpha(0.0)
            if warming: _txt_est.set_text("..."); _txt_est.set_color(C_AMBER)
            else:       _txt_est.set_text("---"); _txt_est.set_color(C_DIM)
        else:
            col = C_ROSE if d["papr"] < 6 else (C_AMBER if d["papr"] < 12 else C_TEAL)
            _needle.set_color(col); _needle_b.set_color(col)
            _needle.set_data([est_rad, est_rad], [0, 0.85])
            _needle_b.set_data([est_rad+_np.pi, est_rad+_np.pi], [0, 0.38])
            _txt_est.set_text(f"{disp_deg:.1f}\u00b0"); _txt_est.set_color(col)

        if warming: _txt_status.set_text(f"WARMUP {d['warmup_left']}"); _txt_status.set_color(C_AMBER)
        else:       _txt_status.set_text("\u25cf LIVE" if is_hw else "\u25cb OFFLINE")
        _txt_status.set_color(C_LIME if (is_hw and not warming) else (C_AMBER if warming else C_ROSE))
        _txt_fps.set_text(f"{_S.fps:.0f} fps")
        _txt_metrics.set_text(f"SNR {d['snr']:.0f}  \u2502  PAPR {d['papr']:.0f}  \u2502  \u03ba {d['cond']:.0f}")

        # C
        _S.h_angle.append(disp_deg if not sq else _np.nan)
        _line_ahist.set_ydata(list(_S.h_angle))
        ang_arr = _np.array([v for v in _S.h_angle if not _np.isnan(v)])
        if len(ang_arr) > 4:
            sin_m = _np.mean(_np.sin(_np.deg2rad(ang_arr))); cos_m = _np.mean(_np.cos(_np.deg2rad(ang_arr)))
            R_mean = _np.sqrt(sin_m**2 + cos_m**2)
            sigma  = min(float(_np.rad2deg(_np.sqrt(-2.0 * _np.log(R_mean + 1e-10)))), 180.0)
            col    = C_ROSE if sigma > 20 else (C_AMBER if sigma > 8 else C_TEAL)
            _txt_sigma.set_text(f"\u03c3 = {sigma:.1f}\u00b0"); _txt_sigma.set_color(col)
            if not sq and sigma > 0.5:
                sr = _np.deg2rad(min(sigma, 90.0))
                wt = _np.linspace(est_rad - sr, est_rad + sr, 40)
                wr = _np.concatenate([[0], _np.ones(38)*0.92, [0]])
                _unc_fill.set_xy(_np.column_stack([wt, wr]))
                _unc_fill.set_facecolor(col); _unc_fill.set_alpha(0.12)
            elif not sq: _unc_fill.set_alpha(0.0)
        else:
            _txt_sigma.set_text("\u03c3 = ---"); _txt_sigma.set_color(C_DIM); _unc_fill.set_alpha(0.0)

        # D
        if not sq or warming:
            ev = d["ev_db"]
            for i, bar in enumerate(_bars_eig): bar.set_height(float(ev[i]) if i < len(ev) else 0.0)
            _ax_eig.set_ylim(bottom=min(0.0, float(_np.min(ev))-1), top=max(float(_np.max(ev))+2, 5.0))
            _txt_cond.set_text(f"\u03ba = {d['cond']:.0f}")

        # E
        if not sq or warming:
            _im_coh.set_data(d["coh"])
            for i in range(_NR):
                for j in range(_NR): _coh_txts[i][j].set_text(f"{d['coh'][i,j]:.2f}")
            mask  = 1 - _np.eye(_NR)
            mu_od = float(_np.mean(d["coh"] * mask))
            qual  = "high" if mu_od > 0.6 else ("moderate" if mu_od > 0.3 else "low")
            _txt_coh_lbl.set_text(f"|\u03bc| = {mu_od:.3f}  ({qual})")

        # F
        _S.h_papr.append(d["papr"]); _S.h_snr.append(d["snr"])
        _line_papr.set_ydata(list(_S.h_papr)); _line_snr.set_ydata(list(_S.h_snr))
        _txt_pq.set_text(f"PAPR {d['papr']:.1f} dB   SNR {d['snr']:.1f} dB")

        # G
        if not sq or warming:
            _line_fft.set_ydata(d["fft_db"])
            pk_bin = int(_np.argmax(d["fft_db"]))
            pk_kHz = float(_fft_freqs[pk_bin])
            warn   = "  [offset detected]" if abs(pk_kHz) > _AUTOTUNE_THRESH_KHZ else ""
            _txt_fft_pk.set_text(f"peak  {pk_kHz:+.2f} kHz{warn}")
            _txt_fft_freq.set_text(f"{_S.active_freq_hz/1e6:.4f} MHz")
            if _S.retuning: _txt_fft_freq.set_color(C_ROSE); _S.retuning = False
            else:           _txt_fft_freq.set_color(C_AMBER)
            _try_autotune()

        # H
        for k in range(_NR - 1):
            _S.h_ph[k].append(d["phase_od"][k]); _ph_lines[k].set_ydata(list(_S.h_ph[k]))
        stds    = [float(_np.std(list(_S.h_ph[k]))) for k in range(_NR - 1)]
        std_max = max(stds) if stds else 0.0
        qual    = "stable" if std_max < 0.3 else ("moderate" if std_max < 0.7 else "unstable")
        col     = C_TEAL if std_max < 0.3 else (C_AMBER if std_max < 0.7 else C_ROSE)
        _txt_ph_lbl.set_text(f"\u03c3\u03c6 = {std_max:.3f} rad  ({qual})"); _txt_ph_lbl.set_color(col)

    # ── Launch animation ──────────────────────────────────────────────────
    self._rt_ani = _animation.FuncAnimation(
        _fig, _update, interval=_INTERVAL, blit=False, cache_frame_data=False)

    def _on_close(_):
        _kraken.stop()
        if _S.rec_active: _on_rec_stop(None)

    _fig.canvas.mpl_connect("close_event", _on_close)
    _plt.show(block=False)

    print(f"[GRC-RT] DoA window open  {self.doa_algorithm}/{self.decorrelation}  "
          f"{_NR}-ant {_GEOM_E.value}  {_S.active_freq_hz/1e6:.3f} MHz")


def snippets_main_after_init(tb):
    snipfcn_realtime_doa_snippet(tb)

from gnuradio import qtgui

class pysdr_doa_realtime(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "KrakenSDR pysdr DoA – Real-Time", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("KrakenSDR pysdr DoA – Real-Time")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "pysdr_doa_realtime")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.squelch_threshold_db = squelch_threshold_db = -60
        self.squelch_enabled = squelch_enabled = 1
        self.scan_points = scan_points = 360
        self.sample_rate_hz = sample_rate_hz = 1.024e6
        self.radius_lambda = radius_lambda = 0.289
        self.phase_offsets_deg = phase_offsets_deg = [0.0, 0.0, 0.0]
        self.num_signals = num_signals = 1
        self.n_antennas = n_antennas = 3
        self.interval_ms = interval_ms = 80
        self.hw_num_samples = hw_num_samples = 0
        self.heimdall_port = heimdall_port = 5000
        self.heimdall_host = heimdall_host = "127.0.0.1"
        self.heimdall_ctrl = heimdall_ctrl = 5001
        self.geometry = geometry = "UCA"
        self.gain_db = gain_db = 15
        self.freq_hz = freq_hz = 865.21e6
        self.doa_algorithm = doa_algorithm = "MUSIC"
        self.decorrelation = decorrelation = "FBA"
        self.d_lambda = d_lambda = 0.5
        self.cov_alpha = cov_alpha = 0.95
        self.angle_smooth_alpha = angle_smooth_alpha = 0.80
        self.amplitude_normalize = amplitude_normalize = 1

        ##################################################
        # Blocks
        ##################################################
        self._squelch_threshold_db_range = Range(-100, 0, 1, -60, 200)
        self._squelch_threshold_db_win = RangeWidget(self._squelch_threshold_db_range, self.set_squelch_threshold_db, "Squelch Thr. (dBW)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._squelch_threshold_db_win)
        # Create the options list
        self._squelch_enabled_options = [0, 1]
        # Create the labels list
        self._squelch_enabled_labels = ['Disabled', 'Enabled']
        # Create the combo box
        self._squelch_enabled_tool_bar = Qt.QToolBar(self)
        self._squelch_enabled_tool_bar.addWidget(Qt.QLabel("Squelch" + ": "))
        self._squelch_enabled_combo_box = Qt.QComboBox()
        self._squelch_enabled_tool_bar.addWidget(self._squelch_enabled_combo_box)
        for _label in self._squelch_enabled_labels: self._squelch_enabled_combo_box.addItem(_label)
        self._squelch_enabled_callback = lambda i: Qt.QMetaObject.invokeMethod(self._squelch_enabled_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._squelch_enabled_options.index(i)))
        self._squelch_enabled_callback(self.squelch_enabled)
        self._squelch_enabled_combo_box.currentIndexChanged.connect(
            lambda i: self.set_squelch_enabled(self._squelch_enabled_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._squelch_enabled_tool_bar)
        # Create the options list
        self._scan_points_options = [90, 180, 360, 720, 1440]
        # Create the labels list
        self._scan_points_labels = ['90 (4°/step)', '180 (2°/step)', '360 (1°/step)', '720 (0.5°/step)', '1440 (0.25°/step)']
        # Create the combo box
        self._scan_points_tool_bar = Qt.QToolBar(self)
        self._scan_points_tool_bar.addWidget(Qt.QLabel("Scan Points" + ": "))
        self._scan_points_combo_box = Qt.QComboBox()
        self._scan_points_tool_bar.addWidget(self._scan_points_combo_box)
        for _label in self._scan_points_labels: self._scan_points_combo_box.addItem(_label)
        self._scan_points_callback = lambda i: Qt.QMetaObject.invokeMethod(self._scan_points_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._scan_points_options.index(i)))
        self._scan_points_callback(self.scan_points)
        self._scan_points_combo_box.currentIndexChanged.connect(
            lambda i: self.set_scan_points(self._scan_points_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._scan_points_tool_bar)
        self._radius_lambda_range = Range(0.05, 1.0, 0.001, 0.289, 200)
        self._radius_lambda_win = RangeWidget(self._radius_lambda_range, self.set_radius_lambda, "Radius / lambda", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._radius_lambda_win)
        # Create the options list
        self._num_signals_options = [1, 2, 3, 4]
        # Create the labels list
        self._num_signals_labels = ['1', '2', '3', '4']
        # Create the combo box
        self._num_signals_tool_bar = Qt.QToolBar(self)
        self._num_signals_tool_bar.addWidget(Qt.QLabel("Sources (D)" + ": "))
        self._num_signals_combo_box = Qt.QComboBox()
        self._num_signals_tool_bar.addWidget(self._num_signals_combo_box)
        for _label in self._num_signals_labels: self._num_signals_combo_box.addItem(_label)
        self._num_signals_callback = lambda i: Qt.QMetaObject.invokeMethod(self._num_signals_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._num_signals_options.index(i)))
        self._num_signals_callback(self.num_signals)
        self._num_signals_combo_box.currentIndexChanged.connect(
            lambda i: self.set_num_signals(self._num_signals_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._num_signals_tool_bar)
        # Create the options list
        self._n_antennas_options = [2, 3, 4, 5]
        # Create the labels list
        self._n_antennas_labels = ['2', '3', '4', '5']
        # Create the combo box
        self._n_antennas_tool_bar = Qt.QToolBar(self)
        self._n_antennas_tool_bar.addWidget(Qt.QLabel("Antennas" + ": "))
        self._n_antennas_combo_box = Qt.QComboBox()
        self._n_antennas_tool_bar.addWidget(self._n_antennas_combo_box)
        for _label in self._n_antennas_labels: self._n_antennas_combo_box.addItem(_label)
        self._n_antennas_callback = lambda i: Qt.QMetaObject.invokeMethod(self._n_antennas_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._n_antennas_options.index(i)))
        self._n_antennas_callback(self.n_antennas)
        self._n_antennas_combo_box.currentIndexChanged.connect(
            lambda i: self.set_n_antennas(self._n_antennas_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._n_antennas_tool_bar)
        # Create the options list
        self._geometry_options = ['UCA', 'ULA']
        # Create the labels list
        self._geometry_labels = ['UCA (circular)', 'ULA (linear)']
        # Create the combo box
        self._geometry_tool_bar = Qt.QToolBar(self)
        self._geometry_tool_bar.addWidget(Qt.QLabel("Geometry" + ": "))
        self._geometry_combo_box = Qt.QComboBox()
        self._geometry_tool_bar.addWidget(self._geometry_combo_box)
        for _label in self._geometry_labels: self._geometry_combo_box.addItem(_label)
        self._geometry_callback = lambda i: Qt.QMetaObject.invokeMethod(self._geometry_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._geometry_options.index(i)))
        self._geometry_callback(self.geometry)
        self._geometry_combo_box.currentIndexChanged.connect(
            lambda i: self.set_geometry(self._geometry_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._geometry_tool_bar)
        self._gain_db_range = Range(0, 49.6, 0.1, 15, 200)
        self._gain_db_win = RangeWidget(self._gain_db_range, self.set_gain_db, "Gain (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._gain_db_win)
        self._freq_hz_range = Range(80e6, 1700e6, 100e3, 865.21e6, 200)
        self._freq_hz_win = RangeWidget(self._freq_hz_range, self.set_freq_hz, "Centre Frequency (Hz)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._freq_hz_win)
        # Create the options list
        self._doa_algorithm_options = ['MUSIC', 'ROOT-MUSIC', 'CAPON', 'ML', 'ESPRIT']
        # Create the labels list
        self._doa_algorithm_labels = ['MUSIC', 'ROOT-MUSIC', 'CAPON', 'ML', 'ESPRIT']
        # Create the combo box
        self._doa_algorithm_tool_bar = Qt.QToolBar(self)
        self._doa_algorithm_tool_bar.addWidget(Qt.QLabel("DoA Algorithm" + ": "))
        self._doa_algorithm_combo_box = Qt.QComboBox()
        self._doa_algorithm_tool_bar.addWidget(self._doa_algorithm_combo_box)
        for _label in self._doa_algorithm_labels: self._doa_algorithm_combo_box.addItem(_label)
        self._doa_algorithm_callback = lambda i: Qt.QMetaObject.invokeMethod(self._doa_algorithm_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._doa_algorithm_options.index(i)))
        self._doa_algorithm_callback(self.doa_algorithm)
        self._doa_algorithm_combo_box.currentIndexChanged.connect(
            lambda i: self.set_doa_algorithm(self._doa_algorithm_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._doa_algorithm_tool_bar)
        # Create the options list
        self._decorrelation_options = ['Off', 'FBA', 'TOEP', 'FBTOEP']
        # Create the labels list
        self._decorrelation_labels = ['Off', 'FBA (forward-backward)', 'TOEP (Toeplitz)', 'FBTOEP (FBA+Toep)']
        # Create the combo box
        self._decorrelation_tool_bar = Qt.QToolBar(self)
        self._decorrelation_tool_bar.addWidget(Qt.QLabel("Decorrelation" + ": "))
        self._decorrelation_combo_box = Qt.QComboBox()
        self._decorrelation_tool_bar.addWidget(self._decorrelation_combo_box)
        for _label in self._decorrelation_labels: self._decorrelation_combo_box.addItem(_label)
        self._decorrelation_callback = lambda i: Qt.QMetaObject.invokeMethod(self._decorrelation_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._decorrelation_options.index(i)))
        self._decorrelation_callback(self.decorrelation)
        self._decorrelation_combo_box.currentIndexChanged.connect(
            lambda i: self.set_decorrelation(self._decorrelation_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._decorrelation_tool_bar)
        self._d_lambda_range = Range(0.1, 2.0, 0.01, 0.5, 200)
        self._d_lambda_win = RangeWidget(self._d_lambda_range, self.set_d_lambda, "d / lambda", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._d_lambda_win)
        self._cov_alpha_range = Range(0.0, 0.99, 0.01, 0.95, 200)
        self._cov_alpha_win = RangeWidget(self._cov_alpha_range, self.set_cov_alpha, "Cov. EMA α", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._cov_alpha_win)
        self._angle_smooth_alpha_range = Range(0.0, 0.99, 0.01, 0.80, 200)
        self._angle_smooth_alpha_win = RangeWidget(self._angle_smooth_alpha_range, self.set_angle_smooth_alpha, "Angle smooth α", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._angle_smooth_alpha_win)
        # Create the options list
        self._amplitude_normalize_options = [0, 1]
        # Create the labels list
        self._amplitude_normalize_labels = ['Off', 'On']
        # Create the combo box
        self._amplitude_normalize_tool_bar = Qt.QToolBar(self)
        self._amplitude_normalize_tool_bar.addWidget(Qt.QLabel("Amp. Normalize" + ": "))
        self._amplitude_normalize_combo_box = Qt.QComboBox()
        self._amplitude_normalize_tool_bar.addWidget(self._amplitude_normalize_combo_box)
        for _label in self._amplitude_normalize_labels: self._amplitude_normalize_combo_box.addItem(_label)
        self._amplitude_normalize_callback = lambda i: Qt.QMetaObject.invokeMethod(self._amplitude_normalize_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._amplitude_normalize_options.index(i)))
        self._amplitude_normalize_callback(self.amplitude_normalize)
        self._amplitude_normalize_combo_box.currentIndexChanged.connect(
            lambda i: self.set_amplitude_normalize(self._amplitude_normalize_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._amplitude_normalize_tool_bar)



    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "pysdr_doa_realtime")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_squelch_threshold_db(self):
        return self.squelch_threshold_db

    def set_squelch_threshold_db(self, squelch_threshold_db):
        self.squelch_threshold_db = squelch_threshold_db

    def get_squelch_enabled(self):
        return self.squelch_enabled

    def set_squelch_enabled(self, squelch_enabled):
        self.squelch_enabled = squelch_enabled
        self._squelch_enabled_callback(self.squelch_enabled)

    def get_scan_points(self):
        return self.scan_points

    def set_scan_points(self, scan_points):
        self.scan_points = scan_points
        self._scan_points_callback(self.scan_points)

    def get_sample_rate_hz(self):
        return self.sample_rate_hz

    def set_sample_rate_hz(self, sample_rate_hz):
        self.sample_rate_hz = sample_rate_hz

    def get_radius_lambda(self):
        return self.radius_lambda

    def set_radius_lambda(self, radius_lambda):
        self.radius_lambda = radius_lambda

    def get_phase_offsets_deg(self):
        return self.phase_offsets_deg

    def set_phase_offsets_deg(self, phase_offsets_deg):
        self.phase_offsets_deg = phase_offsets_deg

    def get_num_signals(self):
        return self.num_signals

    def set_num_signals(self, num_signals):
        self.num_signals = num_signals
        self._num_signals_callback(self.num_signals)

    def get_n_antennas(self):
        return self.n_antennas

    def set_n_antennas(self, n_antennas):
        self.n_antennas = n_antennas
        self._n_antennas_callback(self.n_antennas)

    def get_interval_ms(self):
        return self.interval_ms

    def set_interval_ms(self, interval_ms):
        self.interval_ms = interval_ms

    def get_hw_num_samples(self):
        return self.hw_num_samples

    def set_hw_num_samples(self, hw_num_samples):
        self.hw_num_samples = hw_num_samples

    def get_heimdall_port(self):
        return self.heimdall_port

    def set_heimdall_port(self, heimdall_port):
        self.heimdall_port = heimdall_port

    def get_heimdall_host(self):
        return self.heimdall_host

    def set_heimdall_host(self, heimdall_host):
        self.heimdall_host = heimdall_host

    def get_heimdall_ctrl(self):
        return self.heimdall_ctrl

    def set_heimdall_ctrl(self, heimdall_ctrl):
        self.heimdall_ctrl = heimdall_ctrl

    def get_geometry(self):
        return self.geometry

    def set_geometry(self, geometry):
        self.geometry = geometry
        self._geometry_callback(self.geometry)

    def get_gain_db(self):
        return self.gain_db

    def set_gain_db(self, gain_db):
        self.gain_db = gain_db

    def get_freq_hz(self):
        return self.freq_hz

    def set_freq_hz(self, freq_hz):
        self.freq_hz = freq_hz

    def get_doa_algorithm(self):
        return self.doa_algorithm

    def set_doa_algorithm(self, doa_algorithm):
        self.doa_algorithm = doa_algorithm
        self._doa_algorithm_callback(self.doa_algorithm)

    def get_decorrelation(self):
        return self.decorrelation

    def set_decorrelation(self, decorrelation):
        self.decorrelation = decorrelation
        self._decorrelation_callback(self.decorrelation)

    def get_d_lambda(self):
        return self.d_lambda

    def set_d_lambda(self, d_lambda):
        self.d_lambda = d_lambda

    def get_cov_alpha(self):
        return self.cov_alpha

    def set_cov_alpha(self, cov_alpha):
        self.cov_alpha = cov_alpha

    def get_angle_smooth_alpha(self):
        return self.angle_smooth_alpha

    def set_angle_smooth_alpha(self, angle_smooth_alpha):
        self.angle_smooth_alpha = angle_smooth_alpha

    def get_amplitude_normalize(self):
        return self.amplitude_normalize

    def set_amplitude_normalize(self, amplitude_normalize):
        self.amplitude_normalize = amplitude_normalize
        self._amplitude_normalize_callback(self.amplitude_normalize)




def main(top_block_cls=pysdr_doa_realtime, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    snippets_main_after_init(tb)
    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
