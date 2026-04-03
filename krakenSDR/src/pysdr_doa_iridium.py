#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: KrakenSDR pysdr DoA – Iridium L-Band
# Author: KrakenSDR pysdr-DoA — Iridium L-Band
# Description: Iridium L-band passive DoA via KrakenSDR.
Key differences from pysdr_doa_realtime:
  • Burst detection (Iridium TDMA energy threshold + PAPR gate)
  • Doppler frequency estimation + per-burst correction
  • No cross-burst covariance EMA: each burst → fresh R
    (satellite moves ~1–2°/s, temporal averaging blurs spat. covariance)
  • Panel H replaced by Doppler S-curve for pass tracking
  • Within-pass circular EMA on bearing estimate
  • Pass detection via Doppler discontinuity / timeout

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


def snipfcn_iridium_doa_snippet(self):
    import os as _os, sys as _sys, time as _time, json as _json
    import collections as _collections
    from types import SimpleNamespace as _NS

    _WORKSPACE = '/workspace'
    _PYSDR_DIR = _os.path.join(_WORKSPACE, 'pysdr_doa')
    for _p in (_WORKSPACE, _PYSDR_DIR):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    import numpy as _np
    import matplotlib as _mpl
    _mpl.use('Qt5Agg')
    import matplotlib.pyplot as _plt
    import matplotlib.animation as _animation
    import matplotlib.gridspec as _gridspec
    from matplotlib.widgets import Button as _Button
    import matplotlib.colors as _mcol

    from doa_algorithms import (
        ArrayConfig as _ArrayConfig, Geometry as _Geometry,
        doa_music as _doa_music, doa_root_music as _doa_root_music,
        doa_capon as _doa_capon, doa_ml as _doa_ml, doa_esprit as _doa_esprit,
        apply_phase_correction as _apply_phase_correction,
        snr_from_covariance as _snr_from_cov,
        papr_db as _papr_db,
        condition_number as _cond_num,
        eigenvalue_spread_db as _eig_spread,
        coherence_matrix as _coh_mat,
    )
    from kraken_iq_source import KrakenIQSource as _KrakenIQSource
    import config as _C

    # ── Colour palette ──────────────────────────────────────────────────────
    BG      = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
    C_BORDER= "#3b4263"; C_DIM = "#4e5680"
    C_BLUE  = "#5ea4e0"; C_TEAL  = "#4ecdc4"; C_AMBER  = "#f4a431"
    C_VIOLET= "#a78bfa"; C_ROSE  = "#f16b6f"; C_LIME   = "#6dd97d"
    C_TEXT  = "#d8dae8"; C_MUTED = "#8891b0"

    # ── Iridium L-band constants ─────────────────────────────────────────────
    # Ref: ITU-R M.1031, ETSI EN 300 461, Iridium LLC SIS-ICD
    # Downlink: 1621.35 – 1626.5 MHz, carrier spacing 41.667 kHz
    # Frame: 90 ms TDMA, 8 downlink access slots (1 slot = 8.28 ms)
    # Modulation: DQPSK, 25 kbps, bandwidth ~31.5 kHz null-to-null
    # Orbit: 780 km LEO, orbital velocity ~7560 m/s
    # Max Doppler at 1626 MHz: Δf = v_r/c × f ≈ 37 kHz (±_MAX_DOP_HZ)
    _IRD_CHANS_HZ = {
        "Simplex 1626.270":   1626270000.0,   # ring alerts, most active
        "NEXT ring 1626.104": 1626104000.0,   # Iridium NEXT paging bursts
        "Duplex DL 1621.5":   1621500000.0,   # duplex downlink centre
        "Duplex DL 1623.5":   1623500000.0,
    }
    _TDMA_FRAME_S  = 0.090     # [s] Iridium TDMA frame period
    _MAX_DOP_HZ    = 40000.0   # [Hz] max Doppler for Iridium LEO at L-band
    _NEW_PASS_HZ   = 12000.0   # [Hz] Doppler jump → new satellite
    _PASS_TIMEOUT_S= 6.0       # [s]  no burst → pass ended

    # ── GRC widget reads ─────────────────────────────────────────────────────
    _NR        = int(self.n_antennas)
    _FREQ_HZ   = float(self.iridium_freq_hz)
    _GEOM_STR  = str(self.geometry)
    _R_LAMBDA  = float(self.radius_lambda)
    _PHASE_OFFS= [0.0] * _NR
    _FS        = float(_C.SAMPLE_RATE_HZ)
    HIST       = 120          # rolling history depth (angle + PAPR + burst mask)
    HIST_D     = 200          # Doppler history depth (to show full S-curve)
    FFT_N      = 512          # IQ FFT bins (spectrum panel)
    BURST_N    = 4096         # burst-detect FFT bins (fine freq resolution)
    _SCAN_PTS  = 360
    _WARMUP    = 3            # Heimdall settle frames before burst search
    _INTERVAL  = max(40, int(_TDMA_FRAME_S * 1000))  # ≈90 ms ≈ 1 TDMA frame

    # ══════════════════════════════════════════════════════════════════════════
    # STARTUP CONFIGURATION DIALOG (tkinter, dark Nord theme)
    # ══════════════════════════════════════════════════════════════════════════
    import tkinter as _tk
    import tkinter.ttk as _ttk
    import tkinter.messagebox as _tmsg

    def _run_iridium_dialog():
        root = _tk.Tk()
        root.title("KrakenSDR — Iridium L-Band DoA")
        root.configure(bg="#1a1d27"); root.resizable(False, False)
        TK_BG="#1a1d27"; TK_BG2="#21253a"; TK_BG3="#2a2f47"
        TK_FG="#d8dae8"; TK_ACC="#5ea4e0"
        TK_FONT=("Segoe UI", 9)

        sty = _ttk.Style(root); sty.theme_use("clam")
        sty.configure(".", background=TK_BG, foreground=TK_FG, font=TK_FONT,
                      fieldbackground=TK_BG2, selectbackground=TK_ACC,
                      selectforeground=TK_BG, troughcolor=TK_BG3,
                      bordercolor=TK_BG3, darkcolor=TK_BG2, lightcolor=TK_BG2)
        for w in ("TLabel","TFrame","TLabelframe","TLabelframe.Label"):
            sty.configure(w, background=TK_BG, foreground=TK_FG)
        sty.configure("TEntry",    fieldbackground=TK_BG2, foreground=TK_FG, insertcolor=TK_FG)
        sty.configure("TCombobox", fieldbackground=TK_BG2, foreground=TK_FG)
        sty.map("TCombobox", fieldbackground=[("readonly", TK_BG2)])
        sty.configure("TButton",  background=TK_BG3, foreground=TK_FG, relief="flat",
                      font=("Segoe UI", 10, "bold"), padding="8 4")
        sty.map("TButton", background=[("active", "#3d4675")])
        sty.configure("TCheckbutton", background=TK_BG, foreground=TK_FG)
        _p = {"padx": 10, "pady": 4}

        # Title
        hdr = _ttk.Frame(root, padding="14 10 14 4"); hdr.grid(row=0, column=0, sticky="ew")
        _ttk.Label(hdr, text="KrakenSDR  ·  Iridium L-Band DoA",
                   font=("Segoe UI", 14, "bold"), foreground=TK_ACC).pack(anchor="w")
        _ttk.Label(hdr, text="Passive TDMA burst DoA  —  \u03bb\u226518.4 cm @ 1626 MHz  —  RTL-SDR R820T2 / E4000",
                   foreground="#8891b0", font=TK_FONT).pack(anchor="w")
        _ttk.Separator(root).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 4))

        # ── RF / channel section ────────────────────────────────────────────
        fr_rf = _ttk.LabelFrame(root, text=" RF / Iridium Channel ", padding="12 6")
        fr_rf.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 6))
        chan_names = list(_IRD_CHANS_HZ.keys())
        chan_var = _tk.StringVar(value=chan_names[0])
        freq_var = _tk.StringVar(value=str(int(_FREQ_HZ)))
        gain_var = _tk.StringVar(value=str(int(_C.GAIN_DB)))

        _ttk.Label(fr_rf, text="Channel preset").grid(row=0, column=0, sticky="w", **_p)
        cb_chan = _ttk.Combobox(fr_rf, textvariable=chan_var, values=chan_names,
                                width=26, state="readonly")
        cb_chan.grid(row=0, column=1, **_p)

        _ttk.Label(fr_rf, text="Frequency (Hz)").grid(row=1, column=0, sticky="w", **_p)
        e_freq = _ttk.Entry(fr_rf, textvariable=freq_var, width=14)
        e_freq.grid(row=1, column=1, sticky="w", **_p)

        def _chan_sel(e=None):
            sel = chan_var.get()
            if sel in _IRD_CHANS_HZ:
                freq_var.set(str(int(_IRD_CHANS_HZ[sel])))
        cb_chan.bind("<<ComboboxSelected>>", _chan_sel)

        _ttk.Label(fr_rf, text="IF Gain (dB)").grid(row=2, column=0, sticky="w", **_p)
        e_gain = _ttk.Entry(fr_rf, textvariable=gain_var, width=6)
        e_gain.grid(row=2, column=1, sticky="w", **_p)

        _ttk.Label(fr_rf,
                   text="\u2139  R820T2: 24 MHz \u2013 1766 MHz  \u2714  (1626 MHz is within range)\n"
                        "   Recommended gain: 30\u201345 dB; verify AGC off in daq_chain_config.ini\n"
                        "   Sample rate: 1024000 (default) or 2048000 for wider view",
                   foreground=TK_ACC, font=("Segoe UI", 8)).grid(
                   row=3, column=0, columnspan=2, sticky="w", **_p)

        # ── Array section ───────────────────────────────────────────────────
        fr_arr = _ttk.LabelFrame(root, text=" Antenna Array ", padding="12 6")
        fr_arr.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 6))
        nr_var   = _tk.StringVar(value=str(_NR))
        geom_var = _tk.StringVar(value=_GEOM_STR)
        rlam_var = _tk.StringVar(value=f"{_R_LAMBDA:.3f}")

        _ttk.Label(fr_arr, text="N antennas").grid(row=0, column=0, sticky="w", **_p)
        _ttk.Combobox(fr_arr, textvariable=nr_var, values=["3","4","5"],
                      width=4, state="readonly").grid(row=0, column=1, sticky="w", **_p)

        _ttk.Label(fr_arr, text="Geometry").grid(row=1, column=0, sticky="w", **_p)
        _ttk.Combobox(fr_arr, textvariable=geom_var, values=["UCA","ULA"],
                      width=5, state="readonly").grid(row=1, column=1, sticky="w", **_p)

        _ttk.Label(fr_arr, text="Radius / d  (\u03bb)").grid(row=2, column=0, sticky="w", **_p)
        _ttk.Entry(fr_arr,  textvariable=rlam_var, width=8).grid(row=2, column=1, sticky="w", **_p)

        _ttk.Label(fr_arr,
                   text="\u2022 5-ant UCA  r=12.4 cm  \u2192  r/\u03bb\u22480.673  @ 1626 MHz  (VULA M=9, L=4)\n"
                        "\u2022 3-ant UCA  r= 7.2 cm  \u2192  r/\u03bb\u22480.390  @ 1626 MHz  (VULA M=3, L=1)",
                   foreground=TK_ACC, font=("Segoe UI", 8)).grid(
                   row=3, column=0, columnspan=2, sticky="w", **_p)

        # ── DoA & burst section ─────────────────────────────────────────────
        fr_doa = _ttk.LabelFrame(root, text=" DoA Algorithm & Burst Detection ", padding="12 6")
        fr_doa.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 6))
        algo_var   = _tk.StringVar(value=str(self.doa_algorithm))
        decorr_var = _tk.StringVar(value=str(self.decorrelation))
        dop_var    = _tk.IntVar(value=int(self.doppler_correct))
        bsnr_var   = _tk.StringVar(value=str(float(self.burst_snr_threshold_db)))
        bpwr_var   = _tk.StringVar(value=str(float(self.burst_power_threshold_db)))
        bpapr_var  = _tk.StringVar(value=str(float(self.min_burst_papr_db)))

        _ttk.Label(fr_doa, text="Algorithm").grid(row=0, column=0, sticky="w", **_p)
        _ttk.Combobox(fr_doa, textvariable=algo_var, width=14,
                      values=["ROOT-MUSIC","MUSIC","CAPON","ML","ESPRIT"],
                      state="readonly").grid(row=0, column=1, **_p)

        _ttk.Label(fr_doa, text="Decorrelation").grid(row=1, column=0, sticky="w", **_p)
        _ttk.Combobox(fr_doa, textvariable=decorr_var, width=8,
                      values=["FBA","Off","TOEP","FBTOEP"],
                      state="readonly").grid(row=1, column=1, sticky="w", **_p)

        _ttk.Checkbutton(fr_doa, text="Doppler frequency correction  [exp(\u2212j2\u03c0 f\u1d30 n/fs)]",
                         variable=dop_var).grid(row=2, column=0, columnspan=2, sticky="w", **_p)

        _ttk.Label(fr_doa, text="Burst SNR min (dB)").grid(row=3, column=0, sticky="w", **_p)
        _ttk.Entry(fr_doa, textvariable=bsnr_var, width=6).grid(row=3, column=1, sticky="w", **_p)

        _ttk.Label(fr_doa, text="Burst PAPR min (dB)").grid(row=4, column=0, sticky="w", **_p)
        _ttk.Entry(fr_doa, textvariable=bpapr_var, width=6).grid(row=4, column=1, sticky="w", **_p)

        _ttk.Label(fr_doa, text="Power floor (dBW)").grid(row=5, column=0, sticky="w", **_p)
        _ttk.Entry(fr_doa, textvariable=bpwr_var, width=6).grid(row=5, column=1, sticky="w", **_p)

        _ttk.Label(fr_doa,
                   text="\u2139  ROOT-MUSIC: sub-grid accuracy, D=1 always (single satellite per channel)\n"
                        "   FBA: forward-backward averaging compensates multipath at low elevation\n"
                        "   Burst SNR \u22658 dB typical for good passes; lower for weak/horizon",
                   foreground=TK_ACC, font=("Segoe UI", 8)).grid(
                   row=6, column=0, columnspan=2, sticky="w", **_p)

        # ── Buttons ─────────────────────────────────────────────────────────
        result = [None]
        def _start():
            try:
                result[0] = {
                    "freq_hz":    float(freq_var.get()),
                    "gain_db":    float(gain_var.get()),
                    "n_ant":      int(nr_var.get()),
                    "geom":       geom_var.get(),
                    "r_lam":      float(rlam_var.get()),
                    "algo":       algo_var.get(),
                    "decorr":     decorr_var.get(),
                    "dop_corr":   bool(dop_var.get()),
                    "burst_snr":  float(bsnr_var.get()),
                    "burst_papr": float(bpapr_var.get()),
                    "burst_pwr":  float(bpwr_var.get()),
                }
                root.destroy()
            except ValueError as e:
                _tmsg.showerror("Input error", str(e), parent=root)
        def _cancel():
            root.destroy(); raise SystemExit(0)

        bf = _ttk.Frame(root, padding="14 0 14 14"); bf.grid(row=5, column=0, sticky="ew")
        _ttk.Button(bf, text="\u25b6  Start Iridium DoA", command=_start).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        _ttk.Button(bf, text="\u2715  Cancel", command=_cancel).pack(
            side="left", fill="x", expand=True)

        root.update_idletasks()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w2, h2 = root.winfo_width() // 2, root.winfo_height() // 2
        root.geometry(f"+{sw//2-w2}+{sh//2-h2}")
        root.lift(); root.focus_force(); root.mainloop()
        return result[0]

    _CFG = _run_iridium_dialog()
    if _CFG is None:
        raise SystemExit(0)

    # Apply dialog overrides
    _FREQ_HZ    = _CFG["freq_hz"]
    _GAIN_DB    = _CFG["gain_db"]
    _NR         = _CFG["n_ant"]
    _GEOM_STR   = _CFG["geom"]
    _R_LAMBDA   = _CFG["r_lam"]
    _ALGO       = _CFG["algo"]
    _DECORR     = _CFG["decorr"]
    _DO_DOP     = _CFG["dop_corr"]
    _BURST_SNR  = _CFG["burst_snr"]
    _BURST_PAPR = _CFG["burst_papr"]
    _BURST_PWR  = _CFG["burst_pwr"]
    _PHASE_OFFS = [0.0] * _NR

    # ── Array configuration ─────────────────────────────────────────────────
    _GEOM_E     = _Geometry.UCA if _GEOM_STR.upper() == "UCA" else _Geometry.ULA
    _cfg = _ArrayConfig(
        Nr=_NR, geometry=_GEOM_E,
        d_lambda=0.5, radius_lambda=_R_LAMBDA,
        num_expected_signals=1,   # always 1: one satellite per channel
        num_scan_points=_SCAN_PTS,
    )
    _theta_scan = _cfg.scan_range()
    _flat       = _np.full(_SCAN_PTS, -40.0)
    _fft_freqs  = _np.fft.fftshift(_np.fft.fftfreq(FFT_N)) * _FS / 1e3  # kHz

    # ── Heimdall connection ─────────────────────────────────────────────────
    _kraken = _KrakenIQSource(
        host=_C.HEIMDALL_HOST, port=_C.HEIMDALL_PORT,
        ctrl_port=_C.HEIMDALL_CTRL,
        num_channels=_NR, freq_hz=_FREQ_HZ, gain_db=_GAIN_DB,
        verbose=_C.VERBOSE_FRAMES,
    )
    _kraken.start()
    print(f"[IRD] Connecting Heimdall \u2192 {_FREQ_HZ/1e6:.4f} MHz ...", end=" ", flush=True)
    _time.sleep(2.0)
    print("connected" if _kraken.is_connected else "not reachable \u2013 waiting")
    print(f"[IRD] {_ALGO}/{_DECORR}  DopCorr={'ON' if _DO_DOP else 'OFF'}"
          f"  burst_SNR={_BURST_SNR}dB  burst_PAPR={_BURST_PAPR}dB")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL PROCESSING HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _get_X():
        """Fetch one IQ frame from Heimdall. Returns (ndarray, is_hw) or (None, False)."""
        frame = _kraken.get_frame(timeout=0.05)
        if frame is None:
            return None, False
        X = frame.astype(_np.complex128)
        if _C.HW_NUM_SAMPLES > 0 and X.shape[1] > _C.HW_NUM_SAMPLES:
            X = X[:, :_C.HW_NUM_SAMPLES]
        return X, True


    def _detect_burst(X):
        """
        Detect Iridium TDMA burst via sum-channel spectral analysis.

        Algorithm
        ---------
        1. Non-coherent energy sum across all channels (immune to phase offsets
           between antennas — avoids constructive/destructive interference bias).
        2. Compute power spectrum with BURST_N-bin FFT + Hann window.
        3. Define signal band: |f| <= _MAX_DOP_HZ (Doppler-widened Iridium channel).
        4. Noise reference: |f| > 2.5 × _MAX_DOP_HZ (well outside Iridium band).
        5. In-band SNR = peak / noise_mean [dB] — calibration-free detection.
        6. In-band PAPR = peak / in-band mean [dB] — narrows to burst shape.
        7. Accept if: SNR >= _BURST_SNR, PAPR >= _BURST_PAPR, abs_power >= _BURST_PWR.

        Returns
        -------
        (is_burst, doppler_hz, burst_snr_db, burst_papr_db)
          doppler_hz   : FFT peak frequency relative to Heimdall centre [Hz]
          burst_snr_db : in-band peak vs out-of-band mean [dB]
          burst_papr_db: in-band peak vs in-band mean [dB]
        """
        N   = min(X.shape[1], BURST_N)
        win = _np.hanning(N)
        fa  = _np.zeros(BURST_N)
        for _k in range(_NR):
            seg = X[_k, :N] * win
            fa += _np.abs(_np.fft.fft(seg, n=BURST_N)) ** 2
        fa         = _np.fft.fftshift(fa)
        freqs_hz   = _np.fft.fftshift(_np.fft.fftfreq(BURST_N)) * _FS
        sig_mask   = _np.abs(freqs_hz) <= _MAX_DOP_HZ
        noise_mask = _np.abs(freqs_hz) >  _MAX_DOP_HZ * 2.5
        if not _np.any(noise_mask):
            noise_mask = ~sig_mask

        noise_avg     = float(_np.mean(fa[noise_mask])) + 1e-30
        sig_fa        = fa * sig_mask
        pk_bin        = int(_np.argmax(sig_fa))
        doppler_hz    = float(freqs_hz[pk_bin])
        pk_power      = float(fa[pk_bin])
        burst_snr_db  = float(10.0 * _np.log10(pk_power / noise_avg + 1e-12))
        in_band       = fa[sig_mask]
        burst_papr_db = float(10.0 * _np.log10(
            _np.max(in_band) / (_np.mean(in_band) + 1e-12) + 1e-12))
        abs_pwr_db    = float(10.0 * _np.log10(_np.mean(_np.abs(X) ** 2) + 1e-15))
        is_burst = (burst_snr_db  >= _BURST_SNR  and
                    burst_papr_db >= _BURST_PAPR and
                    abs_pwr_db    >= _BURST_PWR)
        return is_burst, doppler_hz, burst_snr_db, burst_papr_db


    def _doppler_correct(X, doppler_hz):
        """
        Doppler frequency correction via coherent phase rotation.

        X_corr[n] = X[n] * exp(-j2pi f_D n / fs)

        Removes the common Doppler carrier offset from all channels.
        After correction, inter-element phase differences encode only spatial
        information (bearing), not temporal frequency drift.

        Critical for ROOT-MUSIC / ESPRIT: an uncompensated Doppler shifts the
        steering vector phase by 2pi*f_D/fs per sample, rotating the signal
        subspace and degrading DoA precision proportionally to burst duration.
        For f_D=20 kHz, fs=1.024 MHz: phase ramp = 2pi*0.0195 rad/sample.
        Over 8191 samples (8 ms burst): total phase = 1000 rad → completely
        destroys subspace structure without correction.

        Ref: Van Trees H.L., "Optimum Array Processing", Wiley 2002, ch. 9.
        """
        t    = _np.arange(X.shape[1]) / _FS
        corr = _np.exp(-2j * _np.pi * doppler_hz * t)
        return X * corr[_np.newaxis, :]


    def _pipeline_burst(X, doppler_hz):
        """
        Per-burst DoA pipeline. Fresh covariance for every detected burst.

        Design rationale for no cross-burst covariance EMA
        ---------------------------------------------------
        Iridium satellites orbit at 780 km altitude with ~7560 m/s velocity.
        At closest approach the angular velocity reaches ~0.5 deg/s (near
        overhead) to ~1.5 deg/s (low elevation passes).  Even at 0.5 deg/s
        and 90 ms/frame the satellite moves ~0.045 deg between consecutive
        bursts.  Over 20 frames (1.8 s) it moves ~0.9 deg – well within
        array resolution – so gentle angle EMA is valid WITHIN a pass.
        However, TEMPORALLY averaging the covariance matrix across bursts
        blurs the spatial correlation structure because successive R matrices
        encode slightly different steering vectors.  This reduces MUSIC peak
        sharpness and degrades ROOT-MUSIC root proximity to the unit circle.
        Therefore: fresh R = X @ X.H / N for every burst.

        Steps
        -----
        1. Phase offset calibration (hardware imbalance correction)
        2. Doppler correction: remove common carrier offset (see _doppler_correct)
        3. Amplitude normalisation: unit per-channel RMS (removes gain imbalance)
        4. IQ FFT: spectrum panel + verify Doppler marker aligns with peak
        5. Sample covariance R = X @ X.H / N (no EMA)
        6. DoA: algorithm selected by live widget
        7. Quality metrics: papr, snr, cond, eigenvalues, coherence
        """
        _ALGO_L   = self.doa_algorithm
        _DECORR_L = self.decorrelation
        _DO_D     = bool(self.doppler_correct)

        # 1. Phase calibration
        if any(o != 0.0 for o in _PHASE_OFFS):
            X = _apply_phase_correction(X, _PHASE_OFFS)

        # 2. Doppler correction (only if significant)
        if _DO_D and abs(doppler_hz) > 100.0:
            X = _doppler_correct(X, doppler_hz)

        # 3. Amplitude normalisation
        pwr = _np.sqrt(_np.mean(_np.abs(X) ** 2, axis=1, keepdims=True)) + 1e-15
        X   = X / pwr

        # 4. IQ FFT
        win    = _np.hanning(FFT_N)
        fa     = _np.zeros(FFT_N)
        for _k in range(_NR):
            seg  = X[_k, :FFT_N] if X.shape[1] >= FFT_N else _np.pad(X[_k], (0, FFT_N - X.shape[1]))
            fa  += _np.abs(_np.fft.fft(seg * win)) ** 2
        fft_db  = _np.fft.fftshift(10.0 * _np.log10(fa / _NR + 1e-20))
        fft_db -= _np.max(fft_db)

        # 5. Fresh per-burst covariance (NO EMA)
        R_ant = (X @ X.conj().T) / X.shape[1]

        # 6. DoA
        if _ALGO_L == "ROOT-MUSIC":
            est_deg, spec, pv = _doa_root_music(X, _cfg, decorrelation=_DECORR_L, R_in=R_ant)
        elif _ALGO_L == "ESPRIT":
            est_deg, spec, pv = _doa_esprit(X, _cfg, decorrelation=_DECORR_L, R_in=R_ant)
        elif _ALGO_L == "CAPON":
            _, spec    = _doa_capon(X, _cfg, decorrelation=_DECORR_L, R_in=R_ant)
            est_deg    = float(_np.rad2deg(_theta_scan[int(_np.argmax(spec))]) % 360.0)
            pv         = _papr_db(spec)
        elif _ALGO_L == "ML":
            _, spec    = _doa_ml(X, _cfg, decorrelation=_DECORR_L, R_in=R_ant)
            est_deg    = float(_np.rad2deg(_theta_scan[int(_np.argmax(spec))]) % 360.0)
            pv         = _papr_db(spec)
        else:   # MUSIC
            _, spec    = _doa_music(X, _cfg, decorrelation=_DECORR_L, R_in=R_ant)
            est_deg    = float(_np.rad2deg(_theta_scan[int(_np.argmax(spec))]) % 360.0)
            pv         = _papr_db(spec)

        return {
            "spec":     spec,
            "est_deg":  est_deg,
            "papr":     float(pv),
            "snr":      float(_snr_from_cov(R_ant)),
            "cond":     float(_cond_num(R_ant)),
            "ev_db":    _eig_spread(R_ant),
            "coh":      _coh_mat(R_ant),
            "fft_db":   fft_db,
            "phase_od": [float(_np.angle(R_ant[0, k+1])) if _NR > k+1 else 0.0
                         for k in range(_NR - 1)],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # SHARED MUTABLE STATE  (SimpleNamespace avoids exec-scope global issues)
    # ══════════════════════════════════════════════════════════════════════════
    _S = _NS(
        angle_phasor  = complex(1.0, 0.0),
        last_fft      = _np.full(FFT_N, -80.0),
        est_deg       = 0.0,
        cal_offset    = 0.0,
        fps           = 0.0,
        t_last        = _time.time(),
        warming       = True,
        warmup_count  = 0,
        # Burst / pass counters
        burst_count   = 0,
        pass_count    = 0,
        last_burst_t  = 0.0,
        doppler_prev  = None,
        # Rolling histories
        h_angle  = _collections.deque([_np.nan] * HIST,   maxlen=HIST),
        h_papr   = _collections.deque([0.0]    * HIST,   maxlen=HIST),
        h_snr    = _collections.deque([0.0]    * HIST,   maxlen=HIST),
        h_burst  = _collections.deque([0]      * HIST,   maxlen=HIST),
        h_dop    = _collections.deque([0.0]    * HIST_D, maxlen=HIST_D),
        h_dop_q  = _collections.deque([0]      * HIST_D, maxlen=HIST_D),
    )

    def _is_new_pass(dop_hz):
        """Heuristic: Doppler jump > _NEW_PASS_HZ Hz OR gap > _PASS_TIMEOUT_S."""
        if _S.doppler_prev is None:
            return True
        if _time.time() - _S.last_burst_t > _PASS_TIMEOUT_S:
            return True
        return abs(dop_hz - _S.doppler_prev) > _NEW_PASS_HZ

    def _reset_pass():
        _S.angle_phasor = complex(1.0, 0.0)
        _S.h_angle.clear(); _S.h_angle.extend([_np.nan] * HIST)

    def _on_set_zero(_):
        _S.cal_offset = _S.est_deg; _reset_pass()

    def _on_reset_cal(_):
        _S.cal_offset = 0.0; _reset_pass()

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE LAYOUT  (3 × 4 GridSpec — 8 panels)
    # ══════════════════════════════════════════════════════════════════════════
    _plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": C_BORDER, "axes.labelcolor": C_MUTED,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "text.color": C_TEXT, "grid.color": C_BORDER, "grid.alpha": 0.4,
    })
    _fig = _plt.figure(figsize=(20, 11), facecolor=BG)
    _fig.subplots_adjust(left=0.04, right=0.97, top=0.90, bottom=0.06,
                         hspace=0.52, wspace=0.35)
    _txt_title = _fig.suptitle(
        f"KrakenSDR  \u00b7  Iridium L-Band DoA  \u00b7  {_FREQ_HZ/1e6:.4f} MHz"
        f"  \u00b7  {_NR}-ant {_GEOM_STR}  \u00b7  {_ALGO}/{_DECORR}",
        fontsize=13, fontweight="bold", color=C_TEXT)

    _gs = _gridspec.GridSpec(3, 4, figure=_fig,
                             height_ratios=[1.0, 0.75, 0.80],
                             width_ratios=[1.2, 1.0, 1.0, 1.3])

    # ── A: MUSIC pseudospectrum (polar) ───────────────────────────────────
    _ax_spec = _fig.add_subplot(_gs[0, 0], polar=True)
    _ax_spec.set_facecolor(BG2)
    _ax_spec.set_theta_zero_location("N"); _ax_spec.set_theta_direction(-1)
    _ax_spec.set_ylim(-40, 0); _ax_spec.set_yticks([-40, -30, -20, -10, 0])
    _ax_spec.tick_params(colors=C_MUTED, labelsize=7); _ax_spec.set_rlabel_position(45)
    _ax_spec.grid(True, color=C_BORDER, alpha=0.4)
    _ax_spec.set_title(f"Pseudospectrum  [{_ALGO}/{_DECORR}]",
                       color=C_MUTED, fontsize=8, pad=8)
    _line_spec,  = _ax_spec.plot(_theta_scan, _flat, color=C_BLUE, linewidth=1.4)
    _line_est_m, = _ax_spec.plot([0, 0], [-40, 0], color=C_ROSE, linewidth=2.0, alpha=0.8)
    _txt_music   = _ax_spec.text(0, -20, "", ha="center", va="center",
                                 color=C_BLUE, fontsize=8, fontweight="bold")

    # ── B: Compass + Doppler indicator ───────────────────────────────────
    _ax_cmp = _fig.add_subplot(_gs[0, 1], polar=True)
    _ax_cmp.set_facecolor(BG2)
    _ax_cmp.set_theta_zero_location("N"); _ax_cmp.set_theta_direction(-1)
    _ax_cmp.set_ylim(0, 1); _ax_cmp.set_yticks([])
    _ax_cmp.set_xticks(_np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    _ax_cmp.set_xticklabels(["N","NE","E","SE","S","SW","W","NW"],
                             color=C_MUTED, fontsize=7)
    _ax_cmp.grid(True, color=C_BORDER, alpha=0.3)
    _ax_cmp.set_title("Bearing  +  Doppler", color=C_MUTED, fontsize=8, pad=8)
    _needle,   = _ax_cmp.plot([0, 0], [0, 0.85], color=C_TEAL, linewidth=2.5,
                               solid_capstyle="round")
    _needle_b, = _ax_cmp.plot([_np.pi, _np.pi], [0, 0.38], color=C_TEAL,
                               linewidth=1.5, alpha=0.6)
    _unc_fill  = _ax_cmp.fill([], [], color=C_TEAL, alpha=0.08)[0]
    _txt_est   = _ax_cmp.text(0, -0.28, "---", ha="center", va="center",
                               color=C_TEAL, fontsize=14, fontweight="bold",
                               transform=_ax_cmp.transData)
    _txt_dop_b = _ax_cmp.text(0.5, 0.02, "Doppler: --- kHz",
                               ha="center", va="bottom", fontsize=8,
                               color=C_AMBER, transform=_ax_cmp.transAxes)

    # ── C: Bearing history + burst activity shading ────────────────────
    _ax_hist = _fig.add_subplot(_gs[0, 2:])
    _ax_hist.set_facecolor(BG2)
    _ax_hist.set_xlim(0, HIST - 1); _ax_hist.set_ylim(-5, 375)
    _ax_hist.axhline(0, color=C_BORDER, lw=0.5)
    _ax_hist.axhline(360, color=C_BORDER, lw=0.5)
    _ax_hist.set_title("Bearing history — Iridium burst activity (amber shading)",
                       color=C_MUTED, fontsize=8)
    _ax_hist.set_ylabel("Bearing (\u00b0)", color=C_MUTED, fontsize=7)
    _ax_hist.set_xlabel("Frames", color=C_MUTED, fontsize=7)
    _ax_hist.grid(True, axis="y", color=C_BORDER, alpha=0.3, linestyle="--")
    _x_hist       = _np.arange(HIST)
    _line_ahist,  = _ax_hist.plot(_x_hist, list(_S.h_angle), color=C_TEAL, linewidth=1.6)
    _txt_sigma    = _ax_hist.text(0.98, 0.92, "\u03c3 = ---",
                                  transform=_ax_hist.transAxes,
                                  ha="right", fontsize=8, color=C_DIM)
    _txt_burst_lbl= _ax_hist.text(0.02, 0.92, "Bursts: 0  |  Passes: 0",
                                  transform=_ax_hist.transAxes,
                                  ha="left", fontsize=8, color=C_AMBER)

    # ── D: Eigenvalue spread ──────────────────────────────────────────────
    _ax_eig = _fig.add_subplot(_gs[1, 0])
    _ax_eig.set_facecolor(BG2)
    _ax_eig.set_title("Eigenvalue spread", color=C_MUTED, fontsize=8)
    _ax_eig.set_xlabel("k", color=C_MUTED, fontsize=7)
    _ax_eig.set_ylabel("dB above noise", color=C_MUTED, fontsize=7)
    _ax_eig.grid(True, color=C_BORDER, alpha=0.4)
    _bars_eig = _ax_eig.bar(_np.arange(_NR), _np.zeros(_NR),
                             color=[C_TEAL]+[C_DIM]*(_NR-1), width=0.7)
    _txt_cond = _ax_eig.text(0.98, 0.92, "\u03ba = ---",
                              transform=_ax_eig.transAxes, ha="right",
                              fontsize=8, color=C_MUTED)

    # ── E: Coherence matrix ───────────────────────────────────────────────
    _ax_coh = _fig.add_subplot(_gs[1, 1])
    _ax_coh.set_facecolor(BG2)
    _ax_coh.set_title("Coherence  |\u03c1|", color=C_MUTED, fontsize=8)
    _ax_coh.set_xticks(_np.arange(_NR)); _ax_coh.set_yticks(_np.arange(_NR))
    _ax_coh.tick_params(colors=C_MUTED, labelsize=6)
    _cm = _mcol.LinearSegmentedColormap.from_list(
        "ird_cm", [BG2, BG3, C_VIOLET, C_BLUE, C_TEAL])
    _im_coh = _ax_coh.imshow(_np.eye(_NR), vmin=0, vmax=1, cmap=_cm, aspect="auto")
    _coh_txts = [[_ax_coh.text(j, i, "---", ha="center", va="center",
                                fontsize=6, color=C_TEXT)
                  for j in range(_NR)] for i in range(_NR)]
    _txt_coh_lbl = _ax_coh.text(0.5, -0.22, "|\u03bc| = ---",
                                 transform=_ax_coh.transAxes,
                                 ha="center", fontsize=7, color=C_MUTED)

    # ── F: PAPR / SNR per-burst history ──────────────────────────────────
    _ax_pq = _fig.add_subplot(_gs[1, 2])
    _ax_pq.set_facecolor(BG2)
    _ax_pq.set_title("PAPR / SNR  [dB/burst]", color=C_MUTED, fontsize=8)
    _ax_pq.set_xlim(0, HIST - 1); _ax_pq.set_ylim(0, 30)
    _ax_pq.grid(True, color=C_BORDER, alpha=0.3)
    _ax_pq.set_xlabel("Frames", color=C_MUTED, fontsize=7)
    _line_papr, = _ax_pq.plot(_x_hist, list(_S.h_papr), color=C_AMBER,
                               lw=1.4, label="PAPR")
    _line_snr,  = _ax_pq.plot(_x_hist, list(_S.h_snr),  color=C_VIOLET,
                               lw=1.0, alpha=0.85, label="SNR")
    _ax_pq.legend(fontsize=7, facecolor=BG3, edgecolor=C_BORDER,
                  labelcolor=C_TEXT, loc="upper left")
    _txt_pq = _ax_pq.text(0.98, 0.92, "---", transform=_ax_pq.transAxes,
                           ha="right", fontsize=7, color=C_MUTED)

    # ── G: IQ Spectrum — full L-band view with Doppler marker ─────────
    _ax_fft = _fig.add_subplot(_gs[2, :3])
    _ax_fft.set_facecolor(BG2)
    _ax_fft.set_title(
        "IQ Spectrum  —  L-band centred at tuned frequency"
        "  (amber band = Iridium Doppler window \u00b140 kHz)",
        color=C_MUTED, fontsize=8)
    _ax_fft.set_xlabel("\u0394f from centre (kHz)", color=C_MUTED, fontsize=7)
    _ax_fft.set_ylabel("Norm. power (dB)", color=C_MUTED, fontsize=7)
    _ax_fft.set_xlim(_fft_freqs[0], _fft_freqs[-1]); _ax_fft.set_ylim(-45, 5)
    _ax_fft.grid(True, color=C_BORDER, alpha=0.4)
    # Shaded Iridium Doppler window
    _ax_fft.axvspan(-_MAX_DOP_HZ/1e3, _MAX_DOP_HZ/1e3, color=C_AMBER, alpha=0.06)
    _ax_fft.axvline(-_MAX_DOP_HZ/1e3, color=C_AMBER, lw=0.6, alpha=0.4, linestyle=":")
    _ax_fft.axvline( _MAX_DOP_HZ/1e3, color=C_AMBER, lw=0.6, alpha=0.4, linestyle=":")
    _ax_fft.text(min(_MAX_DOP_HZ/1e3+1.5, _fft_freqs[-1]-5), -42,
                 f"\u00b1{_MAX_DOP_HZ/1e3:.0f} kHz", color=C_AMBER, fontsize=6)
    _line_fft,    = _ax_fft.plot(_fft_freqs, _S.last_fft, color=C_BLUE, lw=0.9)
    _line_dop_mk, = _ax_fft.plot([0, 0], [-45, 5], color=C_ROSE,
                                  lw=1.2, alpha=0.8, linestyle="--")
    _txt_fft_pk   = _ax_fft.text(0.98, 0.88, "Doppler: ---",
                                  transform=_ax_fft.transAxes, ha="right",
                                  fontsize=7, color=C_ROSE)
    _txt_fft_freq = _ax_fft.text(0.02, 0.88, f"{_FREQ_HZ/1e6:.4f} MHz",
                                  transform=_ax_fft.transAxes, ha="left",
                                  fontsize=8, color=C_AMBER, fontweight="bold")

    # ── H: Doppler S-curve track — satellite pass visualisation ──────
    # The classic Doppler S-curve uniquely identifies a satellite pass:
    #   positive Doppler → approaching;  zero crossing = TCA (closest approach);
    #   negative Doppler → receding.  Scatter colour = estimated bearing.
    _x_hist_d  = _np.arange(HIST_D)
    _ax_dop    = _fig.add_subplot(_gs[2, 3])
    _ax_dop.set_facecolor(BG2)
    _ax_dop.set_title(
        "Doppler S-curve  (+ = approach, 0 = TCA, \u2212 = recede)",
        color=C_MUTED, fontsize=8)
    _ax_dop.set_xlim(0, HIST_D - 1)
    _ax_dop.set_ylim(-_MAX_DOP_HZ/1e3, _MAX_DOP_HZ/1e3)
    _ax_dop.axhline(0, color=C_BORDER, lw=1.0, linestyle="--", alpha=0.7)
    _ax_dop.grid(True, color=C_BORDER, alpha=0.3)
    _ax_dop.set_xlabel("Frames (recent)", color=C_MUTED, fontsize=7)
    _ax_dop.set_ylabel("kHz", color=C_MUTED, fontsize=7)
    _line_dop, = _ax_dop.plot(_x_hist_d, list(_S.h_dop), color=C_AMBER, lw=1.2)
    _scat_dop  = _ax_dop.scatter([], [], c=[], cmap="hsv", s=20,
                                  vmin=0, vmax=360, zorder=3, alpha=0.85)
    _txt_tca       = _ax_dop.text(0.98, 0.90, "TCA: ---",
                                   transform=_ax_dop.transAxes, ha="right",
                                   fontsize=7, color=C_MUTED)
    _txt_pass_az   = _ax_dop.text(0.02, 0.90, "Pass az: ---",
                                   transform=_ax_dop.transAxes, ha="left",
                                   fontsize=7, color=C_TEAL)

    # ── Status / button bar ───────────────────────────────────────────────
    _txt_status  = _fig.text(0.005, 0.960, "\u25cf INIT", fontsize=9,
                              fontweight="bold", color=C_DIM, va="top")
    _txt_fps     = _fig.text(0.200, 0.960, "--- fps", fontsize=8,
                              color=C_MUTED, va="top")
    _txt_metrics = _fig.text(0.380, 0.960, "", fontsize=8,
                              color=C_MUTED, va="top")
    _ax_b0 = _fig.add_axes([0.730, 0.955, 0.080, 0.030])
    _ax_b1 = _fig.add_axes([0.820, 0.955, 0.080, 0.030])
    for _ax_b in (_ax_b0, _ax_b1):
        _ax_b.set_facecolor(BG3)
        for sp in _ax_b.spines.values(): sp.set_edgecolor(C_BORDER)
    _btn_zero = _Button(_ax_b0, "Set North", color=BG3, hovercolor=C_BORDER)
    _btn_cal  = _Button(_ax_b1, "\u21ba Reset",   color=BG3, hovercolor=C_BORDER)
    _btn_zero.label.set_color(C_TEXT); _btn_cal.label.set_color(C_TEXT)
    _btn_zero.on_clicked(_on_set_zero); _btn_cal.on_clicked(_on_reset_cal)

    # ══════════════════════════════════════════════════════════════════════════
    # ANIMATION UPDATE
    # ══════════════════════════════════════════════════════════════════════════

    def _fft_bg(X):
        """Update IQ spectrum panel from raw X (runs every frame, burst or not)."""
        win    = _np.hanning(FFT_N); fa = _np.zeros(FFT_N)
        for _k in range(_NR):
            seg = X[_k, :FFT_N] if X.shape[1] >= FFT_N else _np.pad(X[_k], (0, FFT_N - X.shape[1]))
            fa += _np.abs(_np.fft.fft(seg * win)) ** 2
        fft_db  = _np.fft.fftshift(10.0 * _np.log10(fa / _NR + 1e-20))
        fft_db -= _np.max(fft_db)
        _S.last_fft = fft_db
        _line_fft.set_ydata(fft_db)

    def _refresh_doppler_panel():
        """Refresh Doppler S-curve panel and detect TCA (time of closest approach)."""
        dop_arr = _np.array(list(_S.h_dop))
        q_arr   = _np.array(list(_S.h_dop_q))
        az_arr  = _np.array(list(_S.h_angle))
        _line_dop.set_ydata(dop_arr)
        # Scatter: detected burst frames, colour = bearing.
        # h_dop_q has HIST_D elements; h_angle only has HIST — align by offset.
        b_idx = _np.where(q_arr == 1)[0]
        if len(b_idx):
            az_off  = len(dop_arr) - len(az_arr)   # HIST_D - HIST
            b_valid = b_idx[b_idx >= az_off]
            if len(b_valid):
                az_b = az_arr[b_valid - az_off]
                az_b = _np.where(_np.isnan(az_b), 0.0, az_b)
                _scat_dop.set_offsets(_np.column_stack([b_valid, dop_arr[b_valid]]))
                _scat_dop.set_array(az_b)
        # TCA detection: downward zero-crossing (positive → negative slope)
        recent = dop_arr[max(0, len(dop_arr) - 40):]
        if len(recent) >= 4:
            signs = _np.sign(recent)
            cross = _np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
            if len(cross):
                fa = len(recent) - 1 - cross[-1]
                _txt_tca.set_text(f"TCA: {fa} fr ago")
                _txt_tca.set_color(C_LIME if fa < 5 else C_AMBER)
            elif len(recent) and recent[-1] > 200:
                _txt_tca.set_text("TCA: approaching \u2192")
                _txt_tca.set_color(C_BLUE)
            elif len(recent) and recent[-1] < -200:
                _txt_tca.set_text("TCA: receded \u2190")
                _txt_tca.set_color(C_DIM)
            else:
                _txt_tca.set_text("TCA: ---"); _txt_tca.set_color(C_DIM)

    def _update(_):
        _S.fps    = 0.9 * _S.fps + 0.1 / max(_time.time() - _S.t_last, 1e-6)
        _S.t_last = _time.time()
        _txt_fps.set_text(f"{_S.fps:.1f} fps")

        X, _is_hw = _get_X()
        if X is None:
            _txt_status.set_text("\u25cb waiting for Heimdall")
            _txt_status.set_color(C_ROSE); return

        # Warmup: let Heimdall settle before burst search
        if _S.warming:
            _S.warmup_count += 1
            if _S.warmup_count >= _WARMUP:
                _S.warming = False
            _txt_status.set_text(f"WARMUP {_S.warmup_count}/{_WARMUP}")
            _txt_status.set_color(C_AMBER)
            _fft_bg(X); return

        # ── Burst detection ────────────────────────────────────────────────
        is_burst, doppler_hz, burst_snr, burst_papr = _detect_burst(X)

        # Update histories unconditionally
        _S.h_burst.append(1 if is_burst else 0)
        _S.h_dop.append(doppler_hz / 1e3)
        _S.h_dop_q.append(1 if is_burst else 0)
        _fft_bg(X)
        _refresh_doppler_panel()
        _line_dop_mk.set_xdata([doppler_hz / 1e3, doppler_hz / 1e3])

        if not is_burst:
            _S.h_papr.append(0.0); _S.h_snr.append(0.0); _S.h_angle.append(_np.nan)
            wait = _time.time() - _S.last_burst_t
            _txt_status.set_text(
                f"\u25cb scanning  \u00b7  last burst {wait:.0f} s ago"
                f"  \u00b7  \u0394f {doppler_hz/1e3:+.1f} kHz"
                if _S.last_burst_t > 0 else "\u25cb scanning for Iridium burst \u2026")
            _txt_status.set_color(C_DIM)
            _txt_burst_lbl.set_text(f"Bursts: {_S.burst_count}  |  Passes: {_S.pass_count}")
            _txt_fft_pk.set_text(f"\u0394f {doppler_hz/1e3:+.2f} kHz  (no burst)")
            return

        # ── New satellite / pass detection ─────────────────────────────────
        if _is_new_pass(doppler_hz):
            _S.pass_count += 1; _reset_pass()
        _S.burst_count += 1
        _S.last_burst_t = _time.time()
        _S.doppler_prev = doppler_hz

        # ── Per-burst DoA pipeline ─────────────────────────────────────────
        try:
            d = _pipeline_burst(X, doppler_hz)
        except Exception as exc:
            _txt_status.set_text(f"ERR {exc}"); _txt_status.set_color(C_ROSE); return

        # Within-pass circular EMA on bearing
        _ANG_A = float(self.angle_smooth_alpha)
        if _ANG_A > 0.0:
            _S.angle_phasor = (_ANG_A * _S.angle_phasor +
                               (1.0 - _ANG_A) * _np.exp(1j * _np.deg2rad(d["est_deg"])))
            est_deg = float(_np.rad2deg(_np.angle(_S.angle_phasor)) % 360.0)
        else:
            est_deg = d["est_deg"]
        _S.est_deg = est_deg
        disp_deg   = (est_deg - _S.cal_offset) % 360.0
        est_rad    = _np.deg2rad(disp_deg)
        cal_rad    = _np.deg2rad(_S.cal_offset)

        _S.h_angle.append(disp_deg)
        _S.h_papr.append(d["papr"]); _S.h_snr.append(d["snr"])

        # ── A: Pseudospectrum ──────────────────────────────────────────────
        _line_spec.set_xdata(_theta_scan - cal_rad)
        _line_spec.set_ydata(d["spec"]); _line_spec.set_color(C_BLUE)
        _line_est_m.set_xdata([est_rad, est_rad])
        _txt_music.set_text(f"{disp_deg:.1f}\u00b0\nPAPR {d['papr']:.1f} dB")
        _txt_music.set_color(C_BLUE)

        # ── B: Compass + Doppler ───────────────────────────────────────────
        col = C_ROSE if d["papr"] < 6 else (C_AMBER if d["papr"] < 12 else C_TEAL)
        _needle.set_color(col); _needle_b.set_color(col)
        _needle.set_data([est_rad, est_rad], [0, 0.85])
        _needle_b.set_data([est_rad + _np.pi, est_rad + _np.pi], [0, 0.38])
        _txt_est.set_text(f"{disp_deg:.1f}\u00b0"); _txt_est.set_color(col)
        sign    = "+" if doppler_hz >= 0 else ""
        dop_col = C_LIME if doppler_hz > 500 else (C_ROSE if doppler_hz < -500 else C_AMBER)
        dop_lbl = ("approach \u2197" if doppler_hz > 200
                   else "\u2198 receding" if doppler_hz < -200 else "\u2022 TCA")
        _txt_dop_b.set_text(f"Doppler {sign}{doppler_hz/1e3:.2f} kHz  ({dop_lbl})")
        _txt_dop_b.set_color(dop_col)

        # Uncertainty cone
        ang_valid = [v for v in _S.h_angle if not _np.isnan(v)]
        if len(ang_valid) > 4:
            sm = _np.mean(_np.sin(_np.deg2rad(ang_valid)))
            cm = _np.mean(_np.cos(_np.deg2rad(ang_valid)))
            Rm = _np.sqrt(sm**2 + cm**2)
            sigma = min(float(_np.rad2deg(_np.sqrt(-2.0 * _np.log(max(Rm, 1e-9))))), 180.0)
            _txt_sigma.set_text(f"\u03c3 = {sigma:.1f}\u00b0")
            _txt_sigma.set_color(
                C_ROSE if sigma > 20 else (C_AMBER if sigma > 8 else C_TEAL))
            if sigma > 0.5:
                sr = _np.deg2rad(min(sigma, 90.0))
                wt = _np.linspace(est_rad - sr, est_rad + sr, 40)
                wr = _np.concatenate([[0], _np.ones(38) * 0.90, [0]])
                _unc_fill.set_xy(_np.column_stack([wt, wr]))
                _unc_fill.set_facecolor(col); _unc_fill.set_alpha(0.10)
            else:
                _unc_fill.set_alpha(0.0)
        else:
            _txt_sigma.set_text("\u03c3 = ---"); _unc_fill.set_alpha(0.0)

        # ── C: Bearing history + burst activity shading ────────────────
        _line_ahist.set_ydata(list(_S.h_angle))
        _ax_hist.collections.clear()
        bdata = list(_S.h_burst)
        if any(b == 1 for b in bdata):
            _ax_hist.fill_between(_x_hist, 0, 370,
                                  where=[b == 1 for b in bdata],
                                  color=C_AMBER, alpha=0.10, linewidth=0)
        _txt_burst_lbl.set_text(f"Bursts: {_S.burst_count}  |  Passes: {_S.pass_count}")

        # ── D: Eigenvalue spread ───────────────────────────────────────────
        ev = d["ev_db"]
        for i, bar in enumerate(_bars_eig):
            bar.set_height(float(ev[i]) if i < len(ev) else 0.0)
        _ax_eig.set_ylim(bottom=min(0.0, float(_np.min(ev)) - 1),
                          top=max(float(_np.max(ev)) + 2, 5.0))
        _txt_cond.set_text(f"\u03ba = {d['cond']:.0f}")

        # ── E: Coherence matrix ────────────────────────────────────────────
        _im_coh.set_data(d["coh"])
        for i in range(_NR):
            for j in range(_NR):
                _coh_txts[i][j].set_text(f"{d['coh'][i,j]:.2f}")
        mask  = 1 - _np.eye(_NR)
        mu_od = float(_np.mean(d["coh"] * mask))
        _txt_coh_lbl.set_text(
            f"|\u03bc| = {mu_od:.3f}  "
            f"({'high' if mu_od > 0.6 else 'moderate' if mu_od > 0.3 else 'low'})")

        # ── F: PAPR / SNR history ──────────────────────────────────────────
        _line_papr.set_ydata(list(_S.h_papr))
        _line_snr.set_ydata(list(_S.h_snr))
        _txt_pq.set_text(f"PAPR {d['papr']:.1f} dB   SNR {d['snr']:.1f} dB")

        # ── G: IQ spectrum (Doppler marker already updated above) ─────────
        _line_fft.set_ydata(d["fft_db"])
        _txt_fft_pk.set_text(
            f"Doppler {sign}{doppler_hz/1e3:.2f} kHz  "
            f"SNR {burst_snr:.1f} dB  PAPR {burst_papr:.1f} dB")

        # ── H: Doppler track already refreshed via _refresh_doppler_panel ─
        # Update pass-mean bearing label
        recent_az = [v for v in list(_S.h_angle)[-40:] if not _np.isnan(v)]
        if len(recent_az) >= 2:
            mean_az = float(_np.rad2deg(
                _np.angle(_np.mean(_np.exp(1j * _np.deg2rad(recent_az))))) % 360.0)
            _txt_pass_az.set_text(f"Pass az \u2248 {mean_az:.0f}\u00b0")
            _txt_pass_az.set_color(C_TEAL)
        else:
            _txt_pass_az.set_text("Pass az: ---")
            _txt_pass_az.set_color(C_DIM)

        # ── Status bar ─────────────────────────────────────────────────────
        _txt_status.set_text(
            f"\u25cf BURST #{_S.burst_count}"
            f"  \u00b7  {sign}{doppler_hz/1e3:.2f} kHz  ({dop_lbl})"
            f"  \u00b7  SNR {burst_snr:.1f} dB"
            f"  \u00b7  {disp_deg:.1f}\u00b0")
        _txt_status.set_color(C_LIME)
        _txt_metrics.set_text(
            f"PAPR {d['papr']:.0f} dB  \u2502  SNR {d['snr']:.0f} dB"
            f"  \u2502  \u03ba {d['cond']:.0f}"
            f"  \u2502  Pass #{_S.pass_count}")

    # ── Launch ────────────────────────────────────────────────────────────
    self._ird_ani = _animation.FuncAnimation(
        _fig, _update, interval=_INTERVAL, blit=False, cache_frame_data=False)

    def _on_close(_):
        _kraken.stop()
    _fig.canvas.mpl_connect("close_event", _on_close)
    _plt.show(block=False)

    print(f"[IRD] Iridium DoA running  {_FREQ_HZ/1e6:.4f} MHz"
          f"  {_NR}-ant {_GEOM_STR}  r/\u03bb={_R_LAMBDA:.3f}"
          f"  {_ALGO}/{_DECORR}"
          f"  DopCorr={'ON' if _DO_DOP else 'OFF'}"
          f"  BurstSNR={_BURST_SNR}dB"
          f"  BurstPAPR={_BURST_PAPR}dB")


def snippets_main_after_init(tb):
    snipfcn_iridium_doa_snippet(tb)

from gnuradio import qtgui

class pysdr_doa_iridium(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "KrakenSDR pysdr DoA – Iridium L-Band", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("KrakenSDR pysdr DoA – Iridium L-Band")
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

        self.settings = Qt.QSettings("GNU Radio", "pysdr_doa_iridium")

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
        self.radius_lambda = radius_lambda = 0.50
        self.n_antennas = n_antennas = 5
        self.min_burst_papr_db = min_burst_papr_db = 5.0
        self.iridium_freq_hz = iridium_freq_hz = 1626270000.0
        self.heimdall_port = heimdall_port = 5000
        self.heimdall_host = heimdall_host = "127.0.0.1"
        self.heimdall_ctrl = heimdall_ctrl = 5001
        self.geometry = geometry = "UCA"
        self.doppler_correct = doppler_correct = 1
        self.doa_algorithm = doa_algorithm = "ROOT-MUSIC"
        self.decorrelation = decorrelation = "FBA"
        self.burst_snr_threshold_db = burst_snr_threshold_db = 8.0
        self.burst_power_threshold_db = burst_power_threshold_db = -90
        self.angle_smooth_alpha = angle_smooth_alpha = 0.65

        ##################################################
        # Blocks
        ##################################################
        self._radius_lambda_range = Range(0.15, 1.50, 0.01, 0.50, 200)
        self._radius_lambda_win = RangeWidget(self._radius_lambda_range, self.set_radius_lambda, "Array radius (λ)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._radius_lambda_win)
        # Create the options list
        self._n_antennas_options = [3, 4, 5]
        # Create the labels list
        self._n_antennas_labels = ['3', '4', '5']
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
        self._min_burst_papr_db_range = Range(2.0, 18.0, 0.5, 5.0, 200)
        self._min_burst_papr_db_win = RangeWidget(self._min_burst_papr_db_range, self.set_min_burst_papr_db, "Min burst PAPR (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._min_burst_papr_db_win)
        # Create the options list
        self._iridium_freq_hz_options = [1626270000.0, 1626104000.0, 1621500000.0, 1623500000.0]
        # Create the labels list
        self._iridium_freq_hz_labels = ['Simplex 1626.270 MHz', 'NEXT ring 1626.104 MHz', 'Duplex DL 1621.5 MHz', 'Duplex DL 1623.5 MHz']
        # Create the combo box
        self._iridium_freq_hz_tool_bar = Qt.QToolBar(self)
        self._iridium_freq_hz_tool_bar.addWidget(Qt.QLabel("Iridium Channel" + ": "))
        self._iridium_freq_hz_combo_box = Qt.QComboBox()
        self._iridium_freq_hz_tool_bar.addWidget(self._iridium_freq_hz_combo_box)
        for _label in self._iridium_freq_hz_labels: self._iridium_freq_hz_combo_box.addItem(_label)
        self._iridium_freq_hz_callback = lambda i: Qt.QMetaObject.invokeMethod(self._iridium_freq_hz_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._iridium_freq_hz_options.index(i)))
        self._iridium_freq_hz_callback(self.iridium_freq_hz)
        self._iridium_freq_hz_combo_box.currentIndexChanged.connect(
            lambda i: self.set_iridium_freq_hz(self._iridium_freq_hz_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._iridium_freq_hz_tool_bar)
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
        # Create the options list
        self._doppler_correct_options = [0, 1]
        # Create the labels list
        self._doppler_correct_labels = ['Off', 'On']
        # Create the combo box
        self._doppler_correct_tool_bar = Qt.QToolBar(self)
        self._doppler_correct_tool_bar.addWidget(Qt.QLabel("Doppler Correction" + ": "))
        self._doppler_correct_combo_box = Qt.QComboBox()
        self._doppler_correct_tool_bar.addWidget(self._doppler_correct_combo_box)
        for _label in self._doppler_correct_labels: self._doppler_correct_combo_box.addItem(_label)
        self._doppler_correct_callback = lambda i: Qt.QMetaObject.invokeMethod(self._doppler_correct_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._doppler_correct_options.index(i)))
        self._doppler_correct_callback(self.doppler_correct)
        self._doppler_correct_combo_box.currentIndexChanged.connect(
            lambda i: self.set_doppler_correct(self._doppler_correct_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._doppler_correct_tool_bar)
        # Create the options list
        self._doa_algorithm_options = ['ROOT-MUSIC', 'MUSIC', 'CAPON', 'ML', 'ESPRIT']
        # Create the labels list
        self._doa_algorithm_labels = ['ROOT-MUSIC', 'MUSIC', 'CAPON', 'ML', 'ESPRIT']
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
        self._decorrelation_options = ['FBA', 'Off', 'TOEP', 'FBTOEP']
        # Create the labels list
        self._decorrelation_labels = ['FBA', 'Off', 'TOEP', 'FBTOEP']
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
        self._burst_snr_threshold_db_range = Range(3.0, 25.0, 0.5, 8.0, 200)
        self._burst_snr_threshold_db_win = RangeWidget(self._burst_snr_threshold_db_range, self.set_burst_snr_threshold_db, "Burst SNR threshold (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._burst_snr_threshold_db_win)
        self._burst_power_threshold_db_range = Range(-100, -40, 1, -90, 200)
        self._burst_power_threshold_db_win = RangeWidget(self._burst_power_threshold_db_range, self.set_burst_power_threshold_db, "Burst power floor (dBW)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._burst_power_threshold_db_win)
        self._angle_smooth_alpha_range = Range(0.0, 0.95, 0.01, 0.65, 200)
        self._angle_smooth_alpha_win = RangeWidget(self._angle_smooth_alpha_range, self.set_angle_smooth_alpha, "Angle smooth α (per-pass)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._angle_smooth_alpha_win)



    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "pysdr_doa_iridium")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_radius_lambda(self):
        return self.radius_lambda

    def set_radius_lambda(self, radius_lambda):
        self.radius_lambda = radius_lambda

    def get_n_antennas(self):
        return self.n_antennas

    def set_n_antennas(self, n_antennas):
        self.n_antennas = n_antennas
        self._n_antennas_callback(self.n_antennas)

    def get_min_burst_papr_db(self):
        return self.min_burst_papr_db

    def set_min_burst_papr_db(self, min_burst_papr_db):
        self.min_burst_papr_db = min_burst_papr_db

    def get_iridium_freq_hz(self):
        return self.iridium_freq_hz

    def set_iridium_freq_hz(self, iridium_freq_hz):
        self.iridium_freq_hz = iridium_freq_hz
        self._iridium_freq_hz_callback(self.iridium_freq_hz)

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

    def get_doppler_correct(self):
        return self.doppler_correct

    def set_doppler_correct(self, doppler_correct):
        self.doppler_correct = doppler_correct
        self._doppler_correct_callback(self.doppler_correct)

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

    def get_burst_snr_threshold_db(self):
        return self.burst_snr_threshold_db

    def set_burst_snr_threshold_db(self, burst_snr_threshold_db):
        self.burst_snr_threshold_db = burst_snr_threshold_db

    def get_burst_power_threshold_db(self):
        return self.burst_power_threshold_db

    def set_burst_power_threshold_db(self, burst_power_threshold_db):
        self.burst_power_threshold_db = burst_power_threshold_db

    def get_angle_smooth_alpha(self):
        return self.angle_smooth_alpha

    def set_angle_smooth_alpha(self, angle_smooth_alpha):
        self.angle_smooth_alpha = angle_smooth_alpha




def main(top_block_cls=pysdr_doa_iridium, options=None):

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
