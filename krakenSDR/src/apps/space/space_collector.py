#!/usr/bin/env python3
"""
space_collector.py — KrakenSDR 5-channel burst collector
=========================================================
Connects to the Heimdall DAQ server, detects Iridium TDMA bursts across
all 5 channels, and saves the raw multi-channel burst windows to disk for
offline 3D DoA analysis.

This script is the data-collection companion to space_doa_realtime.py
and space_doa_playback.py.  It focuses on:

  1. Reliable burst detection (energy + PAPR on Channel 0)
  2. Multi-channel extraction (burst window across all 5 antennas)
  3. Doppler estimation per burst (no correction here — stored raw)
  4. Immediate flush to disk (.npz + .json sidecar)
  5. Live matplotlib display showing signal quality per antenna

Saved format
------------
  <outdir>/kraken_space_YYYYMMDD_HHMMSS.npz
    bursts      : (N, 5, N_burst)  complex64   — raw burst IQ per antenna
    timestamps  : (N,)             float64     — burst onset [ms] from session start
    doppler_hz  : (N,)             float64     — coarse Doppler estimate [Hz]
    snr_db      : (N,)             float32     — burst SNR [dB]

  <outdir>/kraken_space_YYYYMMDD_HHMMSS.json
    freq_hz, sample_rate_hz, gain_db, n_antennas, burst_threshold_db,
    n_bursts, duration_s, timestamp_utc

Usage
-----
    python3 space_collector.py
    python3 space_collector.py --freq 1626.27 --gain 20 --limit 200
    python3 space_collector.py --out /mnt/ssd/captures --threshold 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import collections
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)   # app-local config.py takes priority

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Button

import config as C
from hardware.kraken_iq_source import KrakenIQSource
from core.doa_algorithms_3d import (
    CROSS_ARRAY_CANONICAL_ORDER,
    normalize_cross_array_order,
    short_cross_array_labels,
)
from core.iridium_doa_burst import (
    detect_and_extract_burst,
    compensate_doppler,
)
from core.burst import IRD_CHANS

# ── Colour palette ───────────────────────────────────────────────────────────
BG       = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
C_BORDER = "#3b4263"; C_DIM = "#4e5680"
C_BLUE   = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIOLET = "#a78bfa"; C_ROSE  = "#f16b6f"; C_LIME  = "#6dd97d"
C_TEXT   = "#d8dae8"; C_MUTED = "#8891b0"

_ANT_COLORS = [C_BLUE, C_TEAL, C_AMBER, C_VIOLET, C_ROSE]
_INPUT_ORDER = normalize_cross_array_order(
    getattr(C, "ANTENNA_INPUT_ORDER", CROSS_ARRAY_CANONICAL_ORDER)
)
_INPUT_SHORT = short_cross_array_labels(_INPUT_ORDER)
_INPUT_TICKS = [f"ch{k}/{label}" for k, label in enumerate(_INPUT_SHORT)]

# =============================================================================
# Config dialog
# =============================================================================

def _run_config_dialog() -> dict:
    import tkinter as tk
    import tkinter.ttk as ttk
    import tkinter.messagebox as msgbox
    import ast

    CFG = {}

    _freq_presets = {k.split()[0] + " " + k.split()[1]: v / 1e6 for k, v
                     in IRD_CHANS.items()}
    _freq_labels  = list(_freq_presets.keys())

    root = tk.Tk()
    root.title("KrakenSDR — Space Collector")
    root.configure(bg=BG)
    root.resizable(False, False)

    TK_FG = "#d8dae8"; TK_BG = BG; TK_BG2 = BG2; TK_BG3 = BG3
    TK_ACC = C_BLUE; TK_BTN = C_BORDER
    TK_FONT = ("Segoe UI", 9); TK_HEAD = ("Segoe UI", 10, "bold")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=TK_BG, foreground=TK_FG, font=TK_FONT,
                    fieldbackground=TK_BG2, selectbackground=TK_ACC,
                    selectforeground=TK_BG, troughcolor=TK_BG3,
                    bordercolor=TK_BTN, darkcolor=TK_BG2, lightcolor=TK_BG2)
    style.configure("TLabel",    background=TK_BG,  foreground=TK_FG)
    style.configure("TEntry",    fieldbackground=TK_BG2, foreground=TK_FG, insertcolor=TK_FG)
    style.configure("TCombobox", fieldbackground=TK_BG2, foreground=TK_FG)
    style.map("TCombobox", fieldbackground=[("readonly", TK_BG2)])
    style.configure("TFrame",    background=TK_BG)
    style.configure("TSeparator", background=TK_BTN)
    style.configure("Accent.TButton", background=C_LIME, foreground=TK_BG,
                    font=("Segoe UI", 10, "bold"), padding=6)
    style.configure("Cancel.TButton", background=C_ROSE, foreground=TK_BG,
                    font=("Segoe UI", 10, "bold"), padding=6)

    def _lbl(parent, text, col=TK_MUTED if False else C_MUTED, **kw):
        return ttk.Label(parent, text=text, foreground=col, **kw)

    def _section(parent, text):
        f = ttk.Frame(parent); f.pack(fill="x", padx=12, pady=(10, 2))
        ttk.Label(f, text=f"  {text}  ",
                  background=TK_BG3, foreground=TK_ACC,
                  font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Separator(f, orient="horizontal").pack(side="left", fill="x", expand=True, padx=4)

    def _row(parent):
        f = ttk.Frame(parent); f.pack(fill="x", padx=16, pady=3); return f

    # Title
    tk.Label(root, text="KrakenSDR  Space Collector",
             bg=TK_BG3, fg=TK_ACC, font=("Segoe UI", 13, "bold"),
             pady=10).pack(fill="x")
    tk.Label(root,
             text="Records Iridium burst windows from all 5 antennas for offline 3D DoA",
             bg=TK_BG, fg=C_MUTED, font=("Segoe UI", 8)).pack(pady=(2, 0))

    # ── RF / Hardware ─────────────────────────────────────────────────────────
    _section(root, "RF / Hardware")
    r = _row(root)
    _lbl(r, "Centre Freq (MHz)").pack(side="left")
    _v_freq = tk.StringVar(value=str(C.FREQ_HZ / 1e6))
    ttk.Entry(r, textvariable=_v_freq, width=12).pack(side="left", padx=(6, 6))
    _lbl(r, "or preset:").pack(side="left")
    _v_preset = tk.StringVar(value="")
    cb = ttk.Combobox(r, textvariable=_v_preset, values=_freq_labels,
                      state="readonly", width=14)
    cb.pack(side="left", padx=4)

    def _on_preset(*_):
        v = _v_preset.get()
        if v in _freq_presets:
            _v_freq.set(str(_freq_presets[v]))

    cb.bind("<<ComboboxSelected>>", _on_preset)

    r2 = _row(root)
    _lbl(r2, "Gain (dB)").pack(side="left")
    _v_gain = tk.StringVar(value=str(C.GAIN_DB))
    ttk.Entry(r2, textvariable=_v_gain, width=8).pack(side="left", padx=(6, 14))
    _lbl(r2, "Host").pack(side="left")
    _v_host = tk.StringVar(value=C.HEIMDALL_HOST)
    ttk.Entry(r2, textvariable=_v_host, width=14).pack(side="left", padx=(6, 14))
    _lbl(r2, "Port").pack(side="left")
    _v_port = tk.StringVar(value=str(C.HEIMDALL_PORT))
    ttk.Entry(r2, textvariable=_v_port, width=8).pack(side="left", padx=6)

    # ── Capture settings ──────────────────────────────────────────────────────
    _section(root, "Capture Settings")
    r3 = _row(root)
    _lbl(r3, "Burst threshold (dB)").pack(side="left")
    _v_thr = tk.StringVar(value="10")
    ttk.Entry(r3, textvariable=_v_thr, width=8).pack(side="left", padx=(6, 14))
    _lbl(r3, "Max bursts (0=∞)").pack(side="left")
    _v_max  = tk.StringVar(value="0")
    ttk.Entry(r3, textvariable=_v_max, width=8).pack(side="left", padx=(6, 14))
    _lbl(r3, "Max time s (0=∞)").pack(side="left")
    _v_time = tk.StringVar(value="0")
    ttk.Entry(r3, textvariable=_v_time, width=8).pack(side="left", padx=6)

    r4 = _row(root)
    _lbl(r4, "Output directory").pack(side="left")
    _default_rec = os.path.normpath(os.path.join(_SRC, "..", "..", "recordings"))
    _v_out = tk.StringVar(value=_default_rec)
    ttk.Entry(r4, textvariable=_v_out, width=40).pack(side="left", padx=6)

    # ── Buttons ───────────────────────────────────────────────────────────────
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=12, pady=10)
    bf = ttk.Frame(root); bf.pack(pady=(0, 12))
    _cancelled = [False]

    def _on_start():
        try:
            CFG["freq_hz"]            = float(_v_freq.get()) * 1e6
            CFG["gain_db"]            = float(_v_gain.get())
            CFG["heimdall_host"]      = _v_host.get().strip()
            CFG["heimdall_port"]      = int(_v_port.get())
            CFG["burst_threshold_db"] = float(_v_thr.get())
            CFG["max_bursts"]         = int(_v_max.get())
            CFG["max_time_s"]         = float(_v_time.get())
            CFG["out_dir"]            = _v_out.get().strip()
            root.destroy()
        except Exception as exc:
            msgbox.showerror("Input Error", str(exc), parent=root)

    def _on_cancel():
        _cancelled[0] = True; root.destroy()

    ttk.Button(bf, text="  ▶  Start collecting  ", style="Accent.TButton",
               command=_on_start).pack(side="left", padx=8)
    ttk.Button(bf, text="  ✕  Cancel  ", style="Cancel.TButton",
               command=_on_cancel).pack(side="left", padx=8)

    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    root.lift(); root.focus_force()
    root.mainloop()

    if _cancelled[0]:
        raise SystemExit(0)
    return CFG


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # ── Parse args (allow headless override) ──────────────────────────────────
    parser = argparse.ArgumentParser(
        description="KrakenSDR space collector — saves 5-channel burst IQ to .npz"
    )
    parser.add_argument("--freq",      type=float, help="Centre frequency [MHz]")
    parser.add_argument("--gain",      type=float, help="IF gain [dB]")
    parser.add_argument("--host",      type=str,   help="Heimdall host")
    parser.add_argument("--port",      type=int,   help="Heimdall port")
    parser.add_argument("--threshold", type=float, default=None, help="Burst threshold [dB]")
    parser.add_argument("--limit",     type=int,   default=0,    help="Max bursts (0=∞)")
    parser.add_argument("--time",      type=float, default=0.0,  help="Max session time [s] (0=∞)")
    parser.add_argument("--out",       type=str,   default=None, help="Output directory")
    parser.add_argument("--no-gui",    action="store_true",      help="Run without display (headless)")
    parser.add_argument("--profile",   type=str,   default=None, metavar="NAME",
                        help="Named config profile (e.g. iridium_1626). Overrides LARK_PROFILE env var.")
    args = parser.parse_args()

    # ── Profile (must run before any C.* read) ────────────────────────────────
    if args.profile:
        from profiles import apply_profile
        apply_profile(args.profile, C)

    # If any required param missing from CLI, show dialog
    if args.freq is None or args.gain is None:
        CFG = _run_config_dialog()
    else:
        CFG = {
            "freq_hz":            args.freq * 1e6,
            "gain_db":            args.gain,
            "heimdall_host":      args.host or C.HEIMDALL_HOST,
            "heimdall_port":      args.port or C.HEIMDALL_PORT,
            "burst_threshold_db": args.threshold if args.threshold is not None else 10.0,
            "max_bursts":         args.limit,
            "max_time_s":         args.time,
            "out_dir":            args.out or os.path.normpath(
                                      os.path.join(_SRC, "..", "..", "recordings")),
        }
    if args.threshold is not None:
        CFG["burst_threshold_db"] = args.threshold
    if args.limit:
        CFG["max_bursts"] = args.limit
    if args.out:
        CFG["out_dir"] = args.out

    FREQ_HZ   = float(CFG["freq_hz"])
    GAIN_DB   = float(CFG["gain_db"])
    HOST      = CFG["heimdall_host"]
    PORT      = int(CFG["heimdall_port"])
    THRESHOLD = float(CFG.get("burst_threshold_db", 10.0))
    MAX_BURSTS = int(CFG.get("max_bursts", 0))
    MAX_TIME   = float(CFG.get("max_time_s", 0.0))
    OUT_DIR    = CFG["out_dir"]

    os.makedirs(OUT_DIR, exist_ok=True)
    FS = C.SAMPLE_RATE_HZ

    # ── Session file ──────────────────────────────────────────────────────────
    stamp     = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"kraken_space_{stamp}"
    npz_path  = os.path.join(OUT_DIR, base_name + ".npz")
    json_path = os.path.join(OUT_DIR, base_name + ".json")

    print(f"[COL] freq={FREQ_HZ/1e6:.4f} MHz  gain={GAIN_DB} dB  "
          f"threshold={THRESHOLD} dB  out={npz_path}")

    # ── Shared state ──────────────────────────────────────────────────────────
    FFT_N  = 512
    HIST   = 100

    class S:
        lock         = threading.Lock()
        running      = True
        bursts       : list = []          # list of (5, N_burst) complex64
        timestamps   : list = []          # float64  ms from session start
        doppler_list : list = []          # float64  Hz
        snr_list     : list = []          # float32  dB
        t_start      = time.time()
        # display
        last_fft    = np.full(FFT_N, -80.0)
        ch_powers   = np.zeros(5)
        h_snr       = collections.deque([0.0] * HIST, maxlen=HIST)
        h_dop       = collections.deque([0.0] * HIST, maxlen=HIST)
        h_burst_t   = collections.deque([np.nan] * HIST, maxlen=HIST)
        frame_count = 0
        fps         = 0.0
        t_last_fps  = time.time()

    # ── Heimdall ──────────────────────────────────────────────────────────────
    kraken = KrakenIQSource(
        host         = HOST,
        port         = PORT,
        ctrl_port    = C.HEIMDALL_CTRL,
        num_channels = 5,
        freq_hz      = FREQ_HZ,
        gain_db      = GAIN_DB,
        verbose      = 0,
    )
    kraken.start()

    # ── Acquisition thread ────────────────────────────────────────────────────
    def _acq_loop():
        while S.running:
            frame = kraken.get_frame(timeout=0.010)
            if frame is None:
                continue

            X = frame[:5].astype(np.complex128)
            t_now = (time.time() - S.t_start) * 1000.0   # ms

            # Channel FFT (ch 0) for display
            win  = np.hanning(FFT_N)
            seg  = X[0, :FFT_N] if X.shape[1] >= FFT_N else np.pad(X[0], (0, FFT_N - X.shape[1]))
            spec = np.fft.fftshift(np.fft.fft(seg * win))
            fft_db = np.clip(20.0 * np.log10(np.abs(spec) + 1e-12), -80.0, 0.0)

            # Channel powers (dB relative to ch0)
            ch_pwr = np.array([float(10.0 * np.log10(np.mean(np.abs(X[k]) ** 2) + 1e-30))
                               for k in range(5)])

            # Burst detection
            burst = detect_and_extract_burst(X, threshold_db=THRESHOLD,
                                              sample_rate=int(FS))
            if burst is not None:
                _, dop_hz = compensate_doppler(burst, sample_rate=int(FS))
                noise_var  = float(np.median(np.abs(X[0]) ** 2))
                sig_var    = float(np.mean(np.abs(burst[0]) ** 2))
                snr_db     = float(10.0 * np.log10(max(sig_var / (noise_var + 1e-30), 1e-10)))

                with S.lock:
                    S.bursts.append(burst.astype(np.complex64))
                    S.timestamps.append(t_now)
                    S.doppler_list.append(dop_hz)
                    S.snr_list.append(snr_db)
                    S.h_snr.append(snr_db)
                    S.h_dop.append(dop_hz / 1e3)   # kHz
                    S.h_burst_t.append(t_now / 1000.0)  # s

                n_now = len(S.bursts)
                print(f"[COL] Burst #{n_now:4d}  t={t_now/1000:.1f}s  "
                      f"SNR={snr_db:.1f}dB  Dop={dop_hz/1e3:+.2f}kHz")

                # Stop conditions
                if MAX_BURSTS > 0 and n_now >= MAX_BURSTS:
                    S.running = False
                if MAX_TIME > 0 and (time.time() - S.t_start) >= MAX_TIME:
                    S.running = False

            with S.lock:
                S.last_fft  = fft_db
                S.ch_powers = ch_pwr
                S.frame_count += 1
                now = time.time()
                if now - S.t_last_fps >= 1.0:
                    S.fps = S.frame_count / (now - S.t_last_fps)   # approx
                    S.frame_count = 0; S.t_last_fps = now

    acq_thread = threading.Thread(target=_acq_loop, daemon=True)
    acq_thread.start()

    # ── Save helper ────────────────────────────────────────────────────────────
    def _save():
        with S.lock:
            if not S.bursts:
                print("[COL] No bursts collected — nothing saved.")
                return
            frames_arr = np.stack(S.bursts, axis=0)
            ts_arr     = np.array(S.timestamps, dtype=np.float64)
            dop_arr    = np.array(S.doppler_list, dtype=np.float64)
            snr_arr    = np.array(S.snr_list,     dtype=np.float32)
        np.savez_compressed(npz_path,
                            bursts=frames_arr, timestamps=ts_arr,
                            doppler_hz=dop_arr, snr_db=snr_arr)
        meta = {
            "freq_hz": FREQ_HZ, "sample_rate_hz": FS, "gain_db": GAIN_DB,
            "n_antennas": 5, "burst_threshold_db": THRESHOLD,
            "antenna_input_order": list(_INPUT_ORDER),
            "solver_channel_order": list(CROSS_ARRAY_CANONICAL_ORDER),
            "n_bursts": int(len(S.bursts)),
            "duration_s": float(time.time() - S.t_start),
            "timestamp_utc": stamp,
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[COL] Saved {npz_path}  ({len(S.bursts)} bursts)")

    # ── Headless mode ─────────────────────────────────────────────────────────
    if args.no_gui:
        try:
            print("[COL] Running headless. Press Ctrl-C to stop.")
            while S.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        S.running = False
        time.sleep(0.3)
        _save()
        return

    # ── GUI ───────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": C_BORDER, "axes.grid": True,
        "grid.color": C_BORDER, "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "text.color": C_TEXT,
    })

    fig = plt.figure(figsize=(18, 8.5), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.97, top=0.92, bottom=0.13,
        hspace=0.55, wspace=0.38,
        width_ratios=[1.0, 1.8, 1.4],
        height_ratios=[1.0, 1.0],
    )
    ax_pwr  = fig.add_subplot(gs[0, 0])   # channel powers
    ax_fft  = fig.add_subplot(gs[0, 1])   # IQ spectrum ch0
    ax_snr  = fig.add_subplot(gs[0, 2])   # SNR history
    ax_dop  = fig.add_subplot(gs[1, 0])   # Doppler history
    ax_tl   = fig.add_subplot(gs[1, 1])   # burst timeline scatter
    ax_stat = fig.add_subplot(gs[1, 2])   # stats text

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_color(C_BORDER); sp.set_linewidth(0.8)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        if title:  ax.set_title(title, color=C_TEXT, fontsize=8.5, pad=5, fontweight="semibold")
        if xlabel: ax.set_xlabel(xlabel, color=C_MUTED, fontsize=7)
        if ylabel: ax.set_ylabel(ylabel, color=C_MUTED, fontsize=7)

    # A: Channel powers
    _style(ax_pwr, "Channel Powers", "", "dBW")
    ax_pwr.set_xlim(-0.5, 4.5); ax_pwr.set_xticks(range(5))
    ax_pwr.set_xticklabels(_INPUT_TICKS, color=C_MUTED, fontsize=7.5)
    ax_pwr.set_ylim(-80, 0)
    bars_pwr = ax_pwr.bar(range(5), [-60.0] * 5, color=_ANT_COLORS, edgecolor="none", alpha=0.85)

    # B: IQ spectrum
    _style(ax_fft, "IQ Spectrum  ch0", "offset [kHz]", "dB")
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(FFT_N)) * FS / 1e3
    line_fft, = ax_fft.plot(fft_freqs, np.full(FFT_N, -80.0), color=C_BLUE, linewidth=0.9)
    ax_fft.set_xlim(fft_freqs[0], fft_freqs[-1]); ax_fft.set_ylim(-80, 5)
    ax_fft.axvline(0, color=C_ROSE, linewidth=0.8, linestyle="--", alpha=0.6)
    txt_freq = ax_fft.text(0.98, 0.96, f"{FREQ_HZ/1e6:.4f} MHz",
                           transform=ax_fft.transAxes, ha="right", fontsize=7, color=C_AMBER, va="top")

    # C: SNR history
    _style(ax_snr, "Burst SNR  [dB]", "burst #", "dB")
    ax_snr.set_xlim(0, HIST - 1); ax_snr.set_ylim(-2, 40)
    ax_snr.axhline(10, color=C_ROSE,  linewidth=0.7, linestyle=":", alpha=0.6)
    ax_snr.axhline(20, color=C_TEAL,  linewidth=0.7, linestyle=":", alpha=0.4)
    x_hist = np.arange(HIST)
    line_snr, = ax_snr.plot(x_hist, list(S.h_snr), color=C_TEAL, linewidth=1.4)

    # D: Doppler history
    _style(ax_dop, "Doppler  [kHz]", "burst #", "kHz")
    ax_dop.set_xlim(0, HIST - 1); ax_dop.set_ylim(-45, 45)
    ax_dop.axhline(0, color=C_BORDER, linewidth=0.5)
    line_dop, = ax_dop.plot(x_hist, list(S.h_dop), color=C_AMBER, linewidth=1.4)

    # E: Burst timeline scatter
    _style(ax_tl, "Burst Timeline", "time [s]", "SNR [dB]")
    ax_tl.set_xlim(0, 60); ax_tl.set_ylim(-2, 40)
    scat_tl = ax_tl.scatter([], [], c=[], cmap="plasma",
                            vmin=5, vmax=35, s=30, alpha=0.8)
    txt_tl  = ax_tl.text(0.02, 0.96, "", transform=ax_tl.transAxes,
                         fontsize=7, color=C_MUTED, va="top")

    # F: Stats text
    _style(ax_stat, "Session Info")
    ax_stat.set_axis_off()
    txt_stats = ax_stat.text(
        0.05, 0.95, "Waiting for first burst…",
        transform=ax_stat.transAxes,
        color=C_TEXT, fontsize=8.5, va="top",
        fontfamily="DejaVu Sans Mono",
        bbox=dict(facecolor=BG3, edgecolor=C_BORDER, boxstyle="round,pad=0.5"),
    )

    fig.suptitle(
        f"KrakenSDR  Space Collector  │  {FREQ_HZ/1e6:.4f} MHz  │  {GAIN_DB} dB",
        color=C_TEXT, fontsize=10.5, fontweight="semibold", y=0.977,
    )

    # ── Bottom buttons ─────────────────────────────────────────────────────────
    BTN_Y = 0.022; BTN_H = 0.052
    ax_btn_save = fig.add_axes((0.04,  BTN_Y, 0.10, BTN_H))
    ax_btn_stop = fig.add_axes((0.155, BTN_Y, 0.08, BTN_H))
    btn_save = Button(ax_btn_save, "Save now", color=BG3, hovercolor="#3a4060")
    btn_stop = Button(ax_btn_stop, "Stop",     color=BG3, hovercolor="#3a4060")
    for b in (btn_save, btn_stop):
        b.label.set_color(C_TEXT); b.label.set_fontsize(8.5)

    def _on_save(_): threading.Thread(target=_save, daemon=True).start()
    def _on_stop(_): S.running = False

    btn_save.on_clicked(_on_save)
    btn_stop.on_clicked(_on_stop)

    # ── Animation ─────────────────────────────────────────────────────────────
    def _update(_):
        if not S.running:
            return

        with S.lock:
            fft_now   = S.last_fft.copy()
            pwr_now   = S.ch_powers.copy()
            snr_hist  = list(S.h_snr)
            dop_hist  = list(S.h_dop)
            bt_hist   = list(S.h_burst_t)
            n_bursts  = len(S.bursts)
            fps_now   = S.fps
            all_snrs  = list(S.snr_list)
            all_ts    = list(S.timestamps)
            all_dops  = list(S.doppler_list)

        # A: powers
        for bar, p in zip(bars_pwr, pwr_now):
            bar.set_height(max(p, -80))
        ax_pwr.set_ylim(min(pwr_now) - 5, max(pwr_now) + 5)

        # B: FFT
        line_fft.set_ydata(fft_now)

        # C: SNR history
        line_snr.set_ydata(snr_hist)

        # D: Doppler history
        line_dop.set_ydata(dop_hist)

        # E: Scatter timeline
        if all_ts:
            ts_s = np.array(all_ts) / 1000.0
            snrs = np.array(all_snrs)
            ax_tl.set_xlim(max(0, ts_s[-1] - 120), max(60, ts_s[-1] + 5))
            scat_tl.set_offsets(np.column_stack([ts_s, snrs]))
            scat_tl.set_array(snrs)
        elapsed = time.time() - S.t_start
        txt_tl.set_text(f"Total: {n_bursts}  |  {elapsed:.0f}s elapsed")

        # F: Stats
        last_dop = f"{all_dops[-1]/1e3:+.2f} kHz" if all_dops else "---"
        last_snr = f"{all_snrs[-1]:.1f} dB"         if all_snrs else "---"
        avg_snr  = f"{np.mean(all_snrs):.1f} dB"    if all_snrs else "---"
        size_kb  = sum(b.nbytes for b in S.bursts) // 1024 if S.bursts else 0
        txt_stats.set_text(
            f"  Bursts collected:  {n_bursts}\n"
            f"  Session time:      {elapsed:.0f} s\n"
            f"  Buffer size:       {size_kb} kB\n"
            f"  Last SNR:          {last_snr}\n"
            f"  Last Doppler:      {last_dop}\n"
            f"  Average SNR:       {avg_snr}\n"
            f"  Frame rate:        {fps_now:.1f} fps\n\n"
            f"  Output:\n"
            f"  {os.path.basename(npz_path)}"
        )

    ani = animation.FuncAnimation(fig, _update, interval=150, cache_frame_data=False)

    def _on_close(_):
        S.running = False
        _save()

    fig.canvas.mpl_connect("close_event", _on_close)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    S.running = False
    time.sleep(0.3)
    _save()


if __name__ == "__main__":
    main()
