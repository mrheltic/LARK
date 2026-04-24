#!/usr/bin/env python3
"""
space_doa_realtime.py — KrakenSDR 3D Space DoA (real-time)
===========================================================
Real-time 2D direction-of-arrival (azimuth + elevation) for satellite signals
using a 5-element cross array and the KrakenSDR / Heimdall DAQ back-end.

Unlike the 1D azimuth-only scripts (doa_runner.py), this script:
  • Scans the full upper hemisphere (az 0–360°, el 5–90°)
  • Uses a cross ("+") array geometry with orthogonal E-W and N-S apertures
  • Runs 2D-MUSIC or 2D-Capon on each frame or burst window
  • Renders a polar sky-plot (zenithal equidistant projection)

Modes
-----
  BURST  : detect each Iridium TDMA burst → single-shot R → 2D DoA
            (best accuracy; aligns with satellite pass geometry)
  CW     : exponential moving average of continuous frames → 2D DoA
            (useful for test tones, reference beacons, or tuning)

Display — 8 panels (2 × 4 GridSpec)
-------------------------------------
  Row 0: [Sky plot (polar)]  [2D heatmap (rect)]  [Az + El history]  [Eigenvalues]
  Row 1: [Coherence matrix]  [PAPR + SNR history]  [IQ FFT ch0]  [Phase stability]

Usage
-----
    python3 space_doa_realtime.py
    python3 space_doa_realtime.py --freq 1626.27 --gain 20 --algo music
    python3 space_doa_realtime.py --algo capon --mode cw --n_az 90 --n_el 27
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import collections
import threading
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
_ROOT = os.path.dirname(os.path.dirname(_SRC))    # LARK/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# Always re-insert _HERE at 0: Python may have already added it further down
# the list when the script launched, which would let krakenSDR/src/config.py
# shadow the app-local apps/space/config.py.
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Button

import config as C
from hardware.kraken_iq_source import KrakenIQSource
from core.iridium_doa_burst import (
    detect_and_extract_burst,
    detect_and_extract_all_bursts,
    compensate_doppler,
    compute_single_shot_covariance,
    validate_burst_uw,
    narrowband_filter_burst,
)
from core.doa_algorithms_3d import (
    CROSS_ARRAY_CANONICAL_ORDER,
    CrossArrayConfig,
    CovarianceAccumulator3D,
    SatellitePassAccumulator,
    doa_music_2d,
    doa_capon_2d,
    find_peak_2d,
    make_sky_heatmap_edges,
    skyplot_coords,
    eigenvalue_spread_db,
    snr_from_covariance,
    coherence_matrix,
    normalize_cross_array_order,
    reorder_cross_array_channels,
    short_cross_array_labels,
)
from core.burst import IRD_CHANS, BurstDetector, PassTracker

try:
    from shared.observer import get_observer as _get_observer
    _HAS_OBSERVER = True
except ImportError:
    _HAS_OBSERVER = False

# ── Palette ───────────────────────────────────────────────────────────────────
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
_SOLVER_SHORT = short_cross_array_labels(CROSS_ARRAY_CANONICAL_ORDER)

# Spectrum EMA (temporal smoothing of 2D MUSIC map)
_SPEC_EMA_ALPHA   = 0.35   # blend fraction per new burst (0=frozen, 1=no memory)
_SPEC_MIN_PAPR_DB = 3.0    # minimum PAPR [dB] to accept a burst into the EMA

# UW validation gate
# ≥ 0.67 means at least 8/12 UW dibits match the DL pattern.
# Set to 0.0 to disable the filter (accept all power-detected bursts).
_UW_SCORE_MIN     = 0.4    # 5/12 UW dibits; random match ≈ 0.25; real-HW bursts score 0.4–0.75 (lowered from 0.5)

# Eigenvalue spread gate: minimum dB spread between largest and smallest eigenvalue.
# Bursts below this threshold have insufficient spatial SNR for reliable DoA.
# Recording analysis shows genuine Iridium bursts: spread 5-19 dB (mean ~12 dB).
# Bursts with spread < 8 dB increase DoA error and should be skipped.
_EIG_SPREAD_MIN_DB = 6.0   # dB; real-HW max spread ~7.5 dB (lowered from 8.0)

# Minimum PAPR to accumulate a burst into the SatellitePassAccumulator.
# Bursts below this threshold still update the EMA spectrum and history
# panels but are excluded from the weighted accumulation used for az_smooth.
_PAPR_MIN_DB_ACC   = 1.0   # dB

# GT Doppler matching: only annotate bursts on the ring-alert FDMA channel.
# compensate_doppler() returns total CFO = f_FDMA_offset + f_Doppler.
# For 1626.270 MHz ring-alert bursts f_FDMA_offset ≈ 0; max Iridium Doppler ≈ ±42 kHz.
# Bursts with |CFO| > this limit are on adjacent FDMA channels (≥±80 kHz offset)
# and carry no meaningful GT information from our TLE predictions.
_GT_RING_BW_HZ    = 50_000  # Hz — accept GT match only if |f_dop| < 50 kHz

# Channel health monitor: a channel is flagged faulty if its EMA power is more
# than _CH_FAULT_DB below the median of all channels.  The EMA window is 8 frames.
_CH_FAULT_DB  = 10.0   # dB below median → fault
_CH_PWR_EMA   = 0.125  # 1 / 8 frames


# =============================================================================
# Config Dialog
# =============================================================================

def _run_config_dialog() -> dict:
    import tkinter as tk
    import tkinter.ttk as ttk
    import tkinter.messagebox as msgbox

    _freq_presets = {k: v / 1e6 for k, v in IRD_CHANS.items()}

    CFG = {}
    root = tk.Tk()
    root.title("KrakenSDR — Space DoA (real-time)")
    root.configure(bg=BG)
    root.resizable(False, False)

    TK_FG = "#d8dae8"; TK_BG = BG; TK_BG2 = BG2; TK_BG3 = BG3
    TK_ACC = C_BLUE
    TK_FONT = ("Segoe UI", 9); TK_HEAD = ("Segoe UI", 10, "bold")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=TK_BG, foreground=TK_FG, font=TK_FONT,
                    fieldbackground=TK_BG2, selectbackground=TK_ACC,
                    selectforeground=TK_BG, troughcolor=TK_BG3,
                    bordercolor=C_BORDER, darkcolor=TK_BG2, lightcolor=TK_BG2)
    for w in ("TLabel", "TFrame", "TCheckbutton"):
        style.configure(w, background=TK_BG, foreground=TK_FG)
    style.configure("TEntry",    fieldbackground=TK_BG2, foreground=TK_FG, insertcolor=TK_FG)
    style.configure("TCombobox", fieldbackground=TK_BG2, foreground=TK_FG)
    style.map("TCombobox", fieldbackground=[("readonly", TK_BG2)])
    style.configure("Accent.TButton", background=C_LIME, foreground=TK_BG,
                    font=("Segoe UI", 10, "bold"), padding=6)
    style.configure("Cancel.TButton", background=C_ROSE, foreground=TK_BG,
                    font=("Segoe UI", 10, "bold"), padding=6)

    def _lbl(parent, text, **kw):
        return ttk.Label(parent, text=text, foreground=C_MUTED, **kw)

    def _section(parent, text):
        f = ttk.Frame(parent); f.pack(fill="x", padx=12, pady=(10, 2))
        ttk.Label(f, text=f"  {text}  ",
                  background=TK_BG3, foreground=TK_ACC,
                  font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Separator(f, orient="horizontal").pack(
            side="left", fill="x", expand=True, padx=4)

    def _row(parent):
        f = ttk.Frame(parent); f.pack(fill="x", padx=16, pady=3); return f

    # Title
    tk.Label(root, text="KrakenSDR  Space DoA  —  Real-time",
             bg=TK_BG3, fg=TK_ACC, font=("Segoe UI", 13, "bold"),
             pady=10).pack(fill="x")
    tk.Label(root, text="2D-MUSIC / 2D-Capon on cross array  |  azimuth + elevation",
             bg=TK_BG, fg=C_MUTED, font=("Segoe UI", 8)).pack(pady=(2, 0))

    # ── RF / Hardware ─────────────────────────────────────────────────────────
    _section(root, "RF / Hardware")
    r = _row(root)
    _lbl(r, "Centre Freq (MHz)").pack(side="left")
    _v_freq = tk.StringVar(value=str(C.FREQ_HZ / 1e6))
    ttk.Entry(r, textvariable=_v_freq, width=12).pack(side="left", padx=(6, 6))
    _lbl(r, "preset:").pack(side="left")
    _v_preset = tk.StringVar(value="")
    cb_freq = ttk.Combobox(r, textvariable=_v_preset,
                           values=list(_freq_presets.keys()),
                           state="readonly", width=24)
    cb_freq.pack(side="left", padx=4)

    def _on_preset(*_):
        v = _v_preset.get()
        if v in _freq_presets:
            _v_freq.set(str(_freq_presets[v]))

    cb_freq.bind("<<ComboboxSelected>>", _on_preset)

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

    # ── Cross Array ───────────────────────────────────────────────────────────
    _section(root, "Cross Array Geometry")
    r3 = _row(root)
    _lbl(r3, "Arm length d/λ").pack(side="left")
    _v_d = tk.StringVar(value="0.5")
    ttk.Entry(r3, textvariable=_v_d, width=8).pack(side="left", padx=(6, 14))
    _lbl(r3, "Az scan pts").pack(side="left")
    _v_naz = tk.StringVar(value="72")
    ttk.Entry(r3, textvariable=_v_naz, width=7).pack(side="left", padx=(6, 14))
    _lbl(r3, "El scan pts").pack(side="left")
    _v_nel = tk.StringVar(value="18")
    ttk.Entry(r3, textvariable=_v_nel, width=7).pack(side="left", padx=(6, 14))
    _lbl(r3, "El min (°)").pack(side="left")
    _v_elmin = tk.StringVar(value="5")
    ttk.Entry(r3, textvariable=_v_elmin, width=7).pack(side="left", padx=6)

    # ── Algorithm ─────────────────────────────────────────────────────────────
    _section(root, "DoA Algorithm")
    r4 = _row(root)
    _lbl(r4, "Algorithm").pack(side="left")
    _v_algo = tk.StringVar(value="2D-MUSIC")
    ttk.Combobox(r4, textvariable=_v_algo,
                 values=["2D-MUSIC", "2D-CAPON"],
                 state="readonly", width=12).pack(side="left", padx=(6, 14))
    _lbl(r4, "Mode").pack(side="left")
    _v_mode = tk.StringVar(value="BURST")
    ttk.Combobox(r4, textvariable=_v_mode,
                 values=["BURST", "CW"],
                 state="readonly", width=10).pack(side="left", padx=(6, 14))
    _lbl(r4, "# signals (D)  [0=auto-MDL]").pack(side="left")
    _v_nsig = tk.StringVar(value="0")
    ttk.Entry(r4, textvariable=_v_nsig, width=5).pack(side="left", padx=6)

    # ── Burst settings ────────────────────────────────────────────────────────
    _section(root, "Burst / Signal Processing")
    r5 = _row(root)
    _lbl(r5, "Burst threshold (dB)").pack(side="left")
    _v_thr = tk.StringVar(value="10")
    ttk.Entry(r5, textvariable=_v_thr, width=8).pack(side="left", padx=(6, 14))
    _lbl(r5, "CW EMA α").pack(side="left")
    _v_alpha = tk.StringVar(value="0.90")
    ttk.Entry(r5, textvariable=_v_alpha, width=8).pack(side="left", padx=(6, 14))
    _v_fba = tk.BooleanVar(value=True)
    ttk.Checkbutton(r5, text="Forward-Backward avg",
                    variable=_v_fba).pack(side="left", padx=6)

    # ── Calibration ───────────────────────────────────────────────────────────
    _section(root, "Phase Offsets  (° per antenna — fine calibration)")
    r6 = _row(root)
    _ph_vars = [tk.StringVar(value="0.0") for _ in range(5)]
    for k, (v, col) in enumerate(zip(_ph_vars, _ANT_COLORS)):
        tk.Label(r6, text=f"ch{k}/{_INPUT_SHORT[k]}", fg=col, bg=BG,
                 font=("Segoe UI", 8)).pack(side="left")
        ttk.Entry(r6, textvariable=v, width=6).pack(side="left", padx=(2, 8))

    # ── Buttons ───────────────────────────────────────────────────────────────
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=12, pady=10)
    bf = ttk.Frame(root); bf.pack(pady=(0, 12))
    _cancelled = [False]

    def _on_start():
        try:
            CFG["freq_hz"]            = float(_v_freq.get()) * 1e6
            CFG["gain_db"]            = float(_v_gain.get())
            CFG["host"]               = _v_host.get().strip()
            CFG["port"]               = int(_v_port.get())
            CFG["d_lambda"]           = float(_v_d.get())
            CFG["n_az"]               = int(_v_naz.get())
            CFG["n_el"]               = int(_v_nel.get())
            CFG["el_min_deg"]         = float(_v_elmin.get())
            CFG["algo"]               = _v_algo.get().upper().replace("-", "_")
            CFG["mode"]               = _v_mode.get().upper()
            _nsig_raw = _v_nsig.get().strip().lower()
            CFG["n_signals"]          = 0 if _nsig_raw in ("0", "auto", "") else int(_nsig_raw)
            CFG["burst_threshold_db"] = float(_v_thr.get())
            CFG["cov_alpha"]          = float(_v_alpha.get())
            CFG["use_fba"]            = bool(_v_fba.get())
            CFG["phase_offsets"]      = [float(v.get()) for v in _ph_vars]
            root.destroy()
        except Exception as exc:
            msgbox.showerror("Input Error", str(exc), parent=root)

    def _on_cancel():
        _cancelled[0] = True; root.destroy()

    ttk.Button(bf, text="  ▶  Start DoA  ", style="Accent.TButton",
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
    parser = argparse.ArgumentParser(
        description="KrakenSDR real-time 3D space DoA (2D-MUSIC / 2D-Capon)")
    parser.add_argument("--freq",      type=float, help="Centre freq [MHz]")
    parser.add_argument("--gain",      type=float, help="IF gain [dB]")
    parser.add_argument("--host",      type=str,   help="Heimdall host")
    parser.add_argument("--port",      type=int,   help="Heimdall port")
    parser.add_argument("--algo",      type=str,   default="music",
                        choices=["music", "capon"], help="DoA algorithm")
    parser.add_argument("--mode",      type=str,   default="burst",
                        choices=["burst", "cw"],   help="Burst gate vs CW EMA")
    parser.add_argument("--n_az",      type=int,   default=72,  help="Az scan points")
    parser.add_argument("--n_el",      type=int,   default=18,  help="El scan points")
    parser.add_argument("--d_lambda",  type=float, default=None, help="Arm length [λ] (default: config D_LAMBDA)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Burst threshold [dB]")
    parser.add_argument("--profile",   type=str,   default=None, metavar="NAME",
                        help="Named config profile (e.g. iridium_1626). Overrides LARK_PROFILE env var.")
    args = parser.parse_args()

    # ── Profile (must run before any C.* read) ────────────────────────────────
    if args.profile:
        from profiles import apply_profile
        apply_profile(args.profile, C)

    # Resolve d_lambda: explicit CLI wins, then profile/config default
    _d_lambda = args.d_lambda if args.d_lambda is not None else float(C.D_LAMBDA)

    if args.freq is None or args.gain is None:
        CFG = _run_config_dialog()
    else:
        CFG = {
            "freq_hz":            args.freq * 1e6,
            "gain_db":            args.gain,
            "host":               args.host or C.HEIMDALL_HOST,
            "port":               args.port or C.HEIMDALL_PORT,
            "d_lambda":           _d_lambda,
            "n_az":               args.n_az,
            "n_el":               args.n_el,
            "el_min_deg":         5.0,
            "algo":               args.algo.upper(),
            "mode":               args.mode.upper(),
            "n_signals":          0,   # 0 = auto MDL
            "burst_threshold_db": args.threshold,
            "cov_alpha":          0.90,
            "use_fba":            True,
            "phase_offsets":      [0.0] * 5,
        }

    FREQ_HZ   = float(CFG["freq_hz"])
    GAIN_DB   = float(CFG["gain_db"])
    HOST      = CFG["host"]
    PORT      = int(CFG["port"])
    MODE      = CFG.get("mode", "BURST")
    ALGO      = CFG.get("algo", "2D_MUSIC")
    THRESHOLD = float(CFG.get("burst_threshold_db", 10.0))
    COV_ALPHA = float(CFG.get("cov_alpha", 0.90))
    USE_FBA   = bool(CFG.get("use_fba", True))
    PH_OFF    = np.deg2rad(np.array(CFG.get("phase_offsets", [0.0] * 5),
                                    dtype=np.float64))
    if PH_OFF.shape != (5,):
        raise ValueError("phase_offsets must contain 5 values in physical channel order")

    # Auto-load latest phase calibration if no manual offsets were configured
    _ph_latest  = Path(_ROOT, "krakenSDR", "calibration", "phase_offsets_latest.json")
    PH_AZ_BIAS  = 0.0   # systematic az offset [rad] from last phase-cal (applied as az_zero_offset)
    if _ph_latest.is_file() and not any(v != 0.0 for v in CFG.get("phase_offsets", [0.0] * 5)):
        try:
            with open(_ph_latest) as _f:
                _ph_data = json.load(_f)
            _ph_input = _ph_data.get("phase_offsets_deg_input_order")
            if _ph_input and len(_ph_input) == 5:
                PH_OFF = np.deg2rad(np.array(_ph_input, dtype=np.float64))
                print(f"[CAL] Auto-loaded phase offsets from {_ph_latest.name}  "
                      f"(Az {_ph_data.get('az_mae_before_deg','?')}°→"
                      f"{_ph_data.get('az_mae_after_deg','?')}°  "
                      f"El {_ph_data.get('el_mae_before_deg','?')}°→"
                      f"{_ph_data.get('el_mae_after_deg','?')}°)")
            _az_bias = _ph_data.get("az_bias_deg")
            if _az_bias is not None:
                PH_AZ_BIAS = np.deg2rad(float(_az_bias))
                print(f"[CAL] Auto-applied az_bias={float(_az_bias):.2f}° as initial az_zero_offset")
        except Exception as _e:
            print(f"[CAL] Warning: could not load {_ph_latest.name}: {_e}")
    FS        = float(C.SAMPLE_RATE_HZ)

    cfg = CrossArrayConfig(
        d_lambda             = float(CFG["d_lambda"]),
        n_az                 = int(CFG["n_az"]),
        n_el                 = int(CFG["n_el"]),
        el_min_deg           = float(CFG.get("el_min_deg", 5.0)),
        num_expected_signals = int(CFG.get("n_signals", 1)),
    )

    # ── Shared state ──────────────────────────────────────────────────────────
    HIST      = 120
    FFT_N     = 512
    N_ANT     = 5

    class S:
        lock     = threading.Lock()
        running  = True
        # DoA results
        az_deg   = 0.0;  el_deg   = 45.0
        papr_db  = 0.0;  snr_db   = 0.0
        spec_2d  = np.full((cfg.n_el, cfg.n_az), -40.0)
        R_now    = np.eye(N_ANT, dtype=complex)
        ev_db    = np.zeros(N_ANT)
        coh_mat  = np.eye(N_ANT)
        last_fft = np.full(FFT_N, -80.0)
        # history
        h_az   = collections.deque([0.0]  * HIST, maxlen=HIST)
        h_el   = collections.deque([45.0] * HIST, maxlen=HIST)
        h_papr = collections.deque([0.0]  * HIST, maxlen=HIST)
        h_dop  = collections.deque([0.0]  * HIST, maxlen=HIST)  # Doppler [kHz]
        # recording
        rec_bursts     : list = []
        rec_timestamps : list = []
        rec_az         : list = []
        rec_el         : list = []
        recording      = False
        t_start        = time.time()
        frame_count    = 0
        burst_count    = 0
        pass_count     = 0      # satellite passes detected by PassTracker
        uw_score       = 0.0    # last accepted burst UW correlation [0–1]
        pilot_snr_db   = 0.0    # preamble pilot SNR of last accepted burst [dB]
        ph_offsets     = PH_OFF.copy()  # live-adjustable
        az_zero_offset = PH_AZ_BIAS     # auto-set from last phase-cal az_bias_deg
        # Accumulated (EMA) spectrum for stable display
        spec_ema  = np.full((cfg.n_el, cfg.n_az), -40.0)
        az_smooth = 0.0
        el_smooth = 45.0
        # Channel health monitoring
        ch_pwr_ema  : np.ndarray = np.ones(5, dtype=np.float64)  # linear power EMA
        ch_fault    : list = []   # list of faulty channel labels (human-readable)
        # Real-time satellite visibility (background thread, updated every 15 s)
        visible_sats   : list = []
        # Per-burst ground-truth via Doppler matching (during recording)
        rec_gt_name    : list = []
        rec_gt_az      : list = []   # [deg]
        rec_gt_el      : list = []   # [deg]
        rec_gt_norad   : list = []
        # GT match counters (for the status bar)
        rec_gt_matched : int  = 0    # bursts with valid GT this session
        rec_gt_by_sat  : dict = {}   # {sat_name: matched_count}

    # ── Covariance accumulator (CW mode) ─────────────────────────────────────
    accum = CovarianceAccumulator3D(alpha=COV_ALPHA)

    # ── Per-pass spectrum accumulator ────────────────────────────────────────
    pass_acc = SatellitePassAccumulator(cfg)

    # ── BurstDetector: per-frame RF metrics (PAPR, SNR, Doppler, pilot) ─────
    _burst_n = max(4096, int(261 * FS / 25_000))   # 10690 at 1.024 Msps
    bd = BurstDetector(
        fs        = FS,
        burst_n   = _burst_n,
        burst_snr = 4.0,    # loose — just for per-frame metrics
        burst_papr= 2.0,
    )
    # ── PassTracker: detect satellite transitions ────────────────────────────
    pt = PassTracker()

    # ── Heimdall ──────────────────────────────────────────────────────────────
    kraken = KrakenIQSource(
        host         = HOST,
        port         = PORT,
        ctrl_port    = C.HEIMDALL_CTRL,
        num_channels = N_ANT,
        freq_hz      = FREQ_HZ,
        gain_db      = GAIN_DB,
        verbose      = 0,
    )
    kraken.start()

    # ── DoA function ──────────────────────────────────────────────────────────
    _doa_fn = doa_music_2d if "MUSIC" in ALGO else doa_capon_2d

    def _apply_cal(X: np.ndarray) -> np.ndarray:
        """Apply per-channel phase offsets in the physical input order."""
        offsets = S.ph_offsets
        return X * np.exp(1j * offsets[:, np.newaxis])

    def _fba(R: np.ndarray) -> np.ndarray:
        """Forward-backward averaging: R_fba = (R + J R* J) / 2."""
        J = np.fliplr(np.eye(N_ANT, dtype=complex))
        return (R + J @ R.conj() @ J) / 2.0

    # ── Acquisition thread ────────────────────────────────────────────────────
    def _acq_loop():
        while S.running:
            frame = kraken.get_frame(timeout=0.010)
            if frame is None:
                continue

            X_input = _apply_cal(frame[:N_ANT].astype(np.complex128))
            X = reorder_cross_array_channels(X_input, _INPUT_ORDER)

            # IQ spectrum (ch 0) for display
            win  = np.hanning(FFT_N)
            seg  = X_input[0, :FFT_N] if X_input.shape[1] >= FFT_N else np.pad(X_input[0], (0, FFT_N - X_input.shape[1]))
            fft  = np.fft.fftshift(np.fft.fft(seg * win))
            fft_db = np.clip(20.0 * np.log10(np.abs(fft) + 1e-12), -80.0, 0.0)

            # ── Per-frame RF metrics (always, regardless of burst) ────────────
            bd_result  = bd.process(X_input[0])
            rf_papr    = bd_result.burst_papr_db
            rf_dop_khz = bd_result.doppler_hz / 1000.0

            R              = None
            got_burst      = False
            n_bursts_found = 0
            spec_best           = None;  R_best    = None
            az_best   = 0.0;  el_best   = 45.0
            papr_best = -999.0; snr_best = 0.0
            ev_best   = np.zeros(N_ANT);  coh_best = np.eye(N_ANT)
            burst_input_for_rec = None
            uw_score_best = 0.0;  pilot_snr_best = 0.0
            dop_hz_best   = 0.0

            if MODE == "BURST":
                bursts_all = detect_and_extract_all_bursts(
                    X_input, threshold_db=THRESHOLD, sample_rate=int(FS)
                )
                for bi in bursts_all:
                    b_r           = reorder_cross_array_channels(bi, _INPUT_ORDER)
                    comp_i, f_dop_i = compensate_doppler(b_r, sample_rate=int(FS))

                    # ── Narrowband filter (28 kHz → one FDMA channel) ─────────
                    comp_i = narrowband_filter_burst(comp_i, sample_rate=int(FS))

                    # ── UW validation gate ────────────────────────────────────
                    pilot_snr_i, uw_score_i = validate_burst_uw(
                        comp_i, sample_rate=int(FS)
                    )
                    if uw_score_i < _UW_SCORE_MIN:
                        continue   # reject: UW pattern mismatch

                    R_i       = compute_single_shot_covariance(comp_i)
                    if USE_FBA:
                        R_i = _fba(R_i)

                    # ── Eigenvalue spread gate ────────────────────────────────
                    ev_i    = eigenvalue_spread_db(R_i)
                    spread_i = float(ev_i[0] - ev_i[-1]) if len(ev_i) > 1 else 0.0
                    if spread_i < _EIG_SPREAD_MIN_DB:
                        continue   # reject: insufficient spatial SNR

                    spec_i = _doa_fn(comp_i, cfg, R_in=R_i)
                    az_i, el_i, papr_i = find_peak_2d(spec_i, cfg)
                    snr_i  = snr_from_covariance(R_i)
                    if papr_i > papr_best:
                        spec_best = spec_i;  R_best    = R_i
                        az_best   = az_i;    el_best   = el_i
                        papr_best = papr_i;  snr_best  = snr_i
                        burst_input_for_rec = bi
                        uw_score_best  = uw_score_i
                        pilot_snr_best = pilot_snr_i
                        dop_hz_best    = f_dop_i
                        ev_best        = ev_i
                        coh_best       = coherence_matrix(R_i)
                    n_bursts_found += 1
                if n_bursts_found > 0:
                    got_burst = True
                    R = R_best
            else:  # CW
                R = accum.update(X)
                if USE_FBA:
                    R = _fba(R)
                got_burst = True

            # ── PassTracker: detect new satellite pass ────────────────────────
            new_pass = pt.update(bd_result.doppler_hz, bd_result.is_burst)

            # ── Pass accumulator: weighted average of accepted burst spectra ──
            if got_burst and MODE == "BURST" and spec_best is not None:
                pass_acc.update(spec_best, papr_db=papr_best, new_pass=new_pass,
                                papr_min_db=_PAPR_MIN_DB_ACC)

            # ── Compute DoA outside the lock ──────────────────────────────────
            if got_burst and R is not None:
                if MODE == "BURST":
                    spec = spec_best
                    az   = (az_best - np.rad2deg(S.az_zero_offset)) % 360.0
                    el   = el_best;  papr = papr_best;  snr = snr_best
                    ev   = ev_best;  coh  = coh_best
                else:
                    spec = _doa_fn(X, cfg, R_in=R)
                    az, el, papr = find_peak_2d(spec, cfg)
                    az   = (az - np.rad2deg(S.az_zero_offset)) % 360.0
                    snr  = snr_from_covariance(R)
                    ev   = eigenvalue_spread_db(R)
                    coh  = coherence_matrix(R)
            else:
                spec = None

            # ── Channel health check (power imbalance, runs every frame) ──────
            _pwr_lin = np.mean(np.abs(X_input) ** 2, axis=1)   # (5,) linear

            # ── Update shared state (always, every frame) ─────────────────────
            with S.lock:
                S.ch_pwr_ema = (_CH_PWR_EMA * _pwr_lin
                                + (1.0 - _CH_PWR_EMA) * S.ch_pwr_ema)
                _pdb = 10.0 * np.log10(S.ch_pwr_ema + 1e-30)
                _med = float(np.median(_pdb))
                _fault_new = [_INPUT_ORDER[i] for i, p in enumerate(_pdb)
                              if _med - p > _CH_FAULT_DB]
                if _fault_new != S.ch_fault:
                    S.ch_fault = _fault_new
                    if _fault_new:
                        print(f"[DOA] ⚠  Channel fault detected: "
                              f"{', '.join(_fault_new)} "
                              f"(>{_CH_FAULT_DB:.0f} dB below median)."
                              f"  Per-ch dB: {dict(zip(_INPUT_ORDER, _pdb.round(1)))}")
                S.last_fft    = fft_db
                S.frame_count += 1
                # RF PAPR and Doppler scroll every frame — even between bursts
                S.h_papr.append(rf_papr)
                S.h_dop.append(rf_dop_khz)
                if new_pass:
                    S.pass_count += 1
                    S.spec_ema[:] = -40.0         # new satellite → reset EMA map
                if got_burst and spec is not None:
                    S.spec_2d  = spec
                    S.az_deg   = az;  S.el_deg   = el
                    S.papr_db  = papr; S.snr_db  = snr
                    S.ev_db    = ev;   S.coh_mat = coh
                    S.R_now    = R.copy()
                    S.h_az.append(az)
                    S.h_el.append(el)
                    S.burst_count  += max(1, n_bursts_found)
                    S.uw_score      = uw_score_best
                    S.pilot_snr_db  = pilot_snr_best
                    if rf_papr >= _SPEC_MIN_PAPR_DB:
                        S.spec_ema = ((1.0 - _SPEC_EMA_ALPHA) * S.spec_ema
                                      + _SPEC_EMA_ALPHA * spec)
                        # Use SatellitePassAccumulator for stable smooth Az/El
                        _az_e, _el_e, _ = pass_acc.get_best_estimate()
                        if pass_acc.n_bursts >= 2:
                            S.az_smooth = (_az_e - np.rad2deg(S.az_zero_offset)) % 360.0
                            S.el_smooth = _el_e
                        else:
                            # First burst of a new pass: use raw single-burst DoA
                            S.az_smooth = az
                            S.el_smooth = el
                    if S.recording and burst_input_for_rec is not None:
                        S.rec_bursts.append(burst_input_for_rec.astype(np.complex64))
                        S.rec_timestamps.append(
                            (time.time() - S.t_start) * 1000.0)
                        S.rec_az.append(az)
                        S.rec_el.append(el)
                        # Real-time GT: match this burst to the closest visible
                        # satellite by Doppler (threshold 8 kHz).  Since Iridium
                        # uses TDMA each burst comes from exactly one satellite;
                        # simultaneous satellites differ by 20-40 kHz.
                        # Guard: only match bursts on the ring-alert FDMA channel
                        # (|CFO| < 50 kHz). Bursts at ±185/±313 kHz are on adjacent
                        # Iridium FDMA channels — their CFO includes the channel
                        # offset, so TLE Doppler comparison is meaningless.
                        _vsat = S.visible_sats  # already holding lock → safe read
                        if _vsat and abs(dop_hz_best) < _GT_RING_BW_HZ:
                            _bs = min(_vsat, key=lambda s: abs(s["doppler_hz"] - dop_hz_best))
                            if abs(_bs["doppler_hz"] - dop_hz_best) < 8000:
                                S.rec_gt_name.append(_bs["name"])
                                S.rec_gt_az.append(_bs["az_deg"])
                                S.rec_gt_el.append(_bs["el_deg"])
                                S.rec_gt_norad.append(_bs.get("norad_id", 0))
                                S.rec_gt_matched += 1
                                S.rec_gt_by_sat[_bs["name"]] = (
                                    S.rec_gt_by_sat.get(_bs["name"], 0) + 1
                                )
                            else:
                                S.rec_gt_name.append("")
                                S.rec_gt_az.append(float("nan"))
                                S.rec_gt_el.append(float("nan"))
                                S.rec_gt_norad.append(0)
                        else:
                            S.rec_gt_name.append("")
                            S.rec_gt_az.append(float("nan"))
                            S.rec_gt_el.append(float("nan"))
                            S.rec_gt_norad.append(0)
                else:
                    S.h_az.append(float("nan"))
                    S.h_el.append(float("nan"))

    acq_thread = threading.Thread(target=_acq_loop, daemon=True)
    acq_thread.start()

    # ── Satellite tracker thread (visible_now every 15 s) ─────────────────────
    def _sat_tracker_loop():
        # Load once at startup; the catalogue auto-uses 24h TLE cache.
        _trk_cat = None
        _trk_lat = _trk_lon = _trk_alt = None
        if _HAS_OBSERVER:
            try:
                _trk_lat, _trk_lon, _trk_alt = _get_observer(interactive=False)
            except Exception as _e:
                print(f"[TRACKER] Cannot read observer position: {_e}")
        if _trk_lat is not None:
            try:
                from shared.iridium_tle import load_catalogue as _load_cat
                _trk_cat = _load_cat()
                print(f"[TRACKER] TLE catalogue loaded — {len(_trk_cat.satellites)} satellites")
            except Exception as _e:
                print(f"[TRACKER] Failed to load TLE catalogue: {_e}")

        while S.running:
            if _trk_cat is not None:
                try:
                    _vis = _trk_cat.visible_now(
                        _trk_lat, _trk_lon, _trk_alt, el_min_deg=5.0
                    )
                    with S.lock:
                        S.visible_sats = _vis
                    if _vis:
                        _names = ", ".join(
                            f"{sv['name'].replace('IRIDIUM ','#')}"
                            f" az={sv['az_deg']:.0f}° el={sv['el_deg']:.0f}°"
                            f" dop={sv['doppler_hz']/1e3:+.1f}kHz"
                            for sv in _vis
                        )
                        print(f"[TRACKER] {len(_vis)} visible: {_names}")
                except Exception as _e:
                    print(f"[TRACKER] visible_now error: {_e}")
            time.sleep(15.0)

    sat_tracker_thread = threading.Thread(target=_sat_tracker_loop, daemon=True)
    sat_tracker_thread.start()

    # ── GUI setup ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": C_BORDER, "axes.grid": True,
        "grid.color": C_BORDER, "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "text.color": C_TEXT,
    })

    fig = plt.figure(figsize=(22, 10.5), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        left=0.04, right=0.98, top=0.93, bottom=0.10,
        hspace=0.48, wspace=0.38,
    )

    ax_sky  = fig.add_subplot(gs[0, 0], polar=True)   # sky plot
    ax_heat = fig.add_subplot(gs[0, 1])                # 2D rect heatmap
    ax_hist = fig.add_subplot(gs[0, 2])                # az + el history
    ax_eig  = fig.add_subplot(gs[0, 3])                # eigenvalues
    ax_coh  = fig.add_subplot(gs[1, 0])                # coherence matrix
    ax_pq   = fig.add_subplot(gs[1, 1])                # PAPR + SNR
    ax_fft  = fig.add_subplot(gs[1, 2])                # IQ FFT
    ax_ph   = fig.add_subplot(gs[1, 3])                # phase stability

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_color(C_BORDER); sp.set_linewidth(0.8)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        if title:  ax.set_title(title, color=C_TEXT, fontsize=8.5, pad=5, fontweight="semibold")
        if xlabel: ax.set_xlabel(xlabel, color=C_MUTED, fontsize=7)
        if ylabel: ax.set_ylabel(ylabel, color=C_MUTED, fontsize=7)

    # Panel 0: Sky plot (polar) ────────────────────────────────────────────────
    ax_sky.set_facecolor(BG2)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)         # clockwise = East
    ax_sky.set_rlim(0, 90)
    ax_sky.set_rlabel_position(112.5)
    ax_sky.tick_params(colors=C_MUTED, labelsize=6.5)
    ax_sky.set_rticks([0, 30, 60, 90])
    ax_sky.set_yticklabels(["90°", "60°", "30°", "0°"], color=C_MUTED, fontsize=6)
    _az_ticks_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    ax_sky.set_xticks(np.deg2rad(_az_ticks_deg))
    ax_sky.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                            color=C_MUTED, fontsize=6.5)
    ax_sky.set_title("Sky Plot  (zenithal ≡ equidistant)",
                     color=C_TEXT, fontsize=8.5, pad=10, fontweight="semibold")
    ax_sky.grid(color=C_BORDER, linewidth=0.5, alpha=0.6)
    ax_sky.spines["polar"].set_color(C_BORDER)

    # Sky pcolormesh (spectrum heat)
    th_edges, r_edges = make_sky_heatmap_edges(cfg)
    T_e, R_e = np.meshgrid(th_edges, r_edges)
    # initial blank spectrum
    _blank = np.full((cfg.n_el, cfg.n_az), -40.0)
    sky_mesh = ax_sky.pcolormesh(T_e, R_e, np.flipud(_blank),
                                  cmap="inferno", vmin=-40, vmax=0, shading="flat")

    # Peak marker
    sky_peak, = ax_sky.plot([], [], "o", color=C_LIME, markersize=8,
                             markeredgecolor=BG, markeredgewidth=1.5,
                             zorder=5, label="peak")
    sky_peak_outer, = ax_sky.plot([], [], "o", color="none", markersize=14,
                                   markeredgecolor=C_LIME, markeredgewidth=1.0,
                                   zorder=4)
    sky_az_line,    = ax_sky.plot([0, 0], [0, 90], color=C_LIME,
                                   linewidth=0.6, alpha=0.4, zorder=3)

    txt_sky = ax_sky.text(
        0.5, -0.07, "Az: ---°   El: ---°   PAPR: --- dB",
        transform=ax_sky.transAxes, ha="center", fontsize=7.5,
        color=C_TEXT, fontweight="semibold",
    )

    # Panel 1: 2D rect heatmap ─────────────────────────────────────────────────
    _style(ax_heat, "2D Spectrum  (Az × El)", "Azimuth [°]", "Elevation [°]")
    az_c  = cfg.az_range_deg()
    el_c  = cfg.el_range_deg()
    heat_img = ax_heat.imshow(
        _blank, aspect="auto", origin="lower",
        extent=[az_c[0], az_c[-1], el_c[0], el_c[-1]],
        cmap="inferno", vmin=-40, vmax=0,
    )
    heat_vline = ax_heat.axvline(0.0, color=C_LIME, linewidth=1.0, alpha=0.7)
    heat_hline = ax_heat.axhline(el_c[-1], color=C_LIME, linewidth=1.0, alpha=0.7)
    txt_heat   = ax_heat.text(0.02, 0.97, "", transform=ax_heat.transAxes,
                               fontsize=7, color=C_TEXT, va="top",
                               bbox=dict(facecolor=BG3, edgecolor="none", alpha=0.7))

    # Panel 2: Az + El history ─────────────────────────────────────────────────
    _style(ax_hist, "Az + El History", "frame", "")
    ax_hist.set_xlim(0, HIST - 1)
    ax_hist_el = ax_hist.twinx()  # twin y for elevation
    ax_hist_el.set_facecolor(BG2)
    ax_hist_el.tick_params(colors=C_VIOLET, labelsize=7)
    ax_hist_el.set_ylabel("Elevation [°]", color=C_VIOLET, fontsize=7)
    ax_hist.set_ylim(0, 360); ax_hist.set_ylabel("Azimuth [°]", color=C_AMBER, fontsize=7)
    ax_hist_el.set_ylim(0, 90); ax_hist.set_xlim(0, HIST - 1)
    x_h = np.arange(HIST)
    line_az, = ax_hist.plot(x_h, [0.0] * HIST, color=C_AMBER, linewidth=1.4, label="Az")
    line_el, = ax_hist_el.plot(x_h, [45.0] * HIST, color=C_VIOLET, linewidth=1.4, label="El")
    ax_hist.legend(handles=[line_az, line_el], loc="upper right",
                   fontsize=6.5, framealpha=0.4, facecolor=BG3, edgecolor=C_BORDER)

    # Panel 3: Eigenvalues ─────────────────────────────────────────────────────
    _style(ax_eig, "Eigenvalues  (noise floor = 0 dB)", "", "dB")
    ax_eig.set_xlim(-0.5, N_ANT - 0.5)
    ax_eig.set_xticks(range(N_ANT))
    ax_eig.set_xticklabels([f"λ{k}" for k in range(N_ANT)], color=C_MUTED, fontsize=7)
    ax_eig.set_ylim(-2, 40)
    bars_eig = ax_eig.bar(range(N_ANT), [0.0] * N_ANT,
                          color=[C_BLUE, C_TEAL, C_TEAL, C_TEAL, C_BORDER],
                          edgecolor="none", alpha=0.9)
    ax_eig.axhline(0, color=C_BORDER, linewidth=0.6)
    txt_eig = ax_eig.text(0.98, 0.97, "", transform=ax_eig.transAxes,
                          ha="right", fontsize=7, color=C_MUTED, va="top")

    # Panel 4: Coherence matrix ────────────────────────────────────────────────
    _style(ax_coh, "Coherence  |ρ_{ij}|", "", "")
    coh_img = ax_coh.imshow(np.eye(N_ANT), cmap="viridis", vmin=0, vmax=1,
                             aspect="equal", origin="lower")
    ax_coh.set_xticks(range(N_ANT))
    ax_coh.set_yticks(range(N_ANT))
    ax_coh.set_xticklabels(_SOLVER_SHORT, fontsize=6, color=C_MUTED)
    ax_coh.set_yticklabels(_SOLVER_SHORT, fontsize=6, color=C_MUTED)
    fig.colorbar(coh_img, ax=ax_coh, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # Panel 5: PAPR + SNR ──────────────────────────────────────────────────────
    _style(ax_pq, "RF PAPR  +  Doppler", "frame", "PAPR [dB]")
    ax_pq.set_xlim(0, HIST - 1); ax_pq.set_ylim(-2, 25)
    ax_pq.axhline(_SPEC_MIN_PAPR_DB, color=C_TEAL, linewidth=0.6, linestyle=":", alpha=0.6)
    ax_pq_snr = ax_pq.twinx()
    ax_pq_snr.set_facecolor(BG2); ax_pq_snr.set_ylim(-50, 50)
    ax_pq_snr.tick_params(colors=C_VIOLET, labelsize=7)
    ax_pq_snr.set_ylabel("Doppler [kHz]", color=C_VIOLET, fontsize=7)
    ax_pq_snr.axhline(0, color=C_VIOLET, linewidth=0.5, linestyle="--", alpha=0.4)
    line_papr, = ax_pq.plot(x_h, [0.0] * HIST, color=C_TEAL, linewidth=1.4, label="PAPR")
    line_snr,  = ax_pq_snr.plot(x_h, [0.0] * HIST, color=C_VIOLET, linewidth=1.4, label="Dop")
    ax_pq.set_ylabel("RF PAPR [dB]", color=C_TEAL, fontsize=7)
    ax_pq.legend(handles=[line_papr, line_snr], loc="upper right",
                 fontsize=6.5, framealpha=0.4, facecolor=BG3, edgecolor=C_BORDER)

    # Panel 6: IQ FFT ──────────────────────────────────────────────────────────
    _style(ax_fft, "IQ Spectrum  ch0", "offset [kHz]", "dB")
    freq_axis = np.fft.fftshift(np.fft.fftfreq(FFT_N)) * FS / 1e3
    line_fft,  = ax_fft.plot(freq_axis, np.full(FFT_N, -80.0), color=C_BLUE, linewidth=0.9)
    ax_fft.set_xlim(freq_axis[0], freq_axis[-1]); ax_fft.set_ylim(-80, 5)
    ax_fft.axvline(0, color=C_ROSE, linewidth=0.7, linestyle="--", alpha=0.5)

    # Panel 7: Phase stability ─────────────────────────────────────────────────
    _style(ax_ph, "Phase  (off-diag |R|)", "ant pair", "dB")
    _pairs = ["01", "02", "03", "04", "12", "13", "14", "23", "24", "34"]
    ax_ph.set_xlim(-0.5, len(_pairs) - 0.5)
    ax_ph.set_xticks(range(len(_pairs)))
    ax_ph.set_xticklabels(_pairs, fontsize=5.5, color=C_MUTED, rotation=45)
    bars_ph = ax_ph.bar(range(len(_pairs)), [0.0] * len(_pairs),
                        color=C_VIOLET, edgecolor="none", alpha=0.8)

    fig.suptitle(
        f"KrakenSDR  Space DoA  —  {FREQ_HZ/1e6:.4f} MHz  │  "
        f"{ALGO.replace('_', '-')}  │  MODE={MODE}  │  "
        f"d={cfg.d_lambda}λ  {cfg.n_az}×{cfg.n_el} grid",
        color=C_TEXT, fontsize=10, fontweight="semibold", y=0.980,
    )

    # ── Buttons ───────────────────────────────────────────────────────────────
    BTN_Y = 0.018; BTN_H = 0.046
    ax_rec     = fig.add_axes((0.04,  BTN_Y, 0.08, BTN_H))
    ax_az_zero = fig.add_axes((0.13,  BTN_Y, 0.10, BTN_H))
    ax_cal_rst = fig.add_axes((0.24,  BTN_Y, 0.10, BTN_H))
    ax_save    = fig.add_axes((0.355, BTN_Y, 0.08, BTN_H))

    # ── Recording / satellite status bar ──────────────────────────────────────
    ax_status = fig.add_axes((0.45, BTN_Y, 0.535, BTN_H))
    ax_status.set_facecolor(BG3)
    ax_status.set_xticks([]); ax_status.set_yticks([])
    for _sp in ax_status.spines.values():
        _sp.set_color(C_BORDER); _sp.set_linewidth(0.6)
    txt_status = ax_status.text(
        0.012, 0.5,
        "Ready — press ● Record to start a session",
        transform=ax_status.transAxes,
        ha="left", va="center", fontsize=7.5,
        color=C_MUTED, fontweight="normal",
    )

    btn_rec     = Button(ax_rec,     "● Record",    color=BG3, hovercolor="#3a4060")
    btn_az_zero = Button(ax_az_zero, "Set Az Zero", color=BG3, hovercolor="#3a4060")
    btn_cal_rst = Button(ax_cal_rst, "Reset Cal",   color=BG3, hovercolor="#3a4060")
    btn_save    = Button(ax_save,    "Save .npz",   color=BG3, hovercolor="#3a4060")
    for b in (btn_rec, btn_az_zero, btn_cal_rst, btn_save):
        b.label.set_color(C_TEXT); b.label.set_fontsize(8.0)

    def _on_rec(_):
        with S.lock:
            if not S.recording:
                # Starting a new session: clear any previous burst buffer
                S.rec_bursts.clear()
                S.rec_timestamps.clear()
                S.rec_az.clear()
                S.rec_el.clear()
                S.rec_gt_name.clear()
                S.rec_gt_az.clear()
                S.rec_gt_el.clear()
                S.rec_gt_norad.clear()
                S.rec_gt_matched = 0
                S.rec_gt_by_sat.clear()
                S.t_start = time.time()
            S.recording = not S.recording
        btn_rec.label.set_text("■ Stop rec" if S.recording else "● Record")
        btn_rec.color = "#4b1f1f" if S.recording else BG3
        fig.canvas.draw_idle()

    def _on_az_zero(_):
        with S.lock:
            S.az_zero_offset = np.deg2rad(S.az_deg) % (2 * np.pi)
            S.spec_ema[:] = -40.0  # reset EMA after az re-zero

    def _on_cal_rst(_):
        with S.lock:
            S.az_zero_offset = 0.0
            S.ph_offsets[:] = 0.0
            S.spec_ema[:] = -40.0  # reset EMA after calibration reset

    def _on_save(_):
        _do_save()

    btn_rec.on_clicked(_on_rec)
    btn_az_zero.on_clicked(_on_az_zero)
    btn_cal_rst.on_clicked(_on_cal_rst)
    btn_save.on_clicked(_on_save)

    # ── Save recording ─────────────────────────────────────────────────────────
    _rec_dir = os.path.normpath(os.path.join(_SRC, "..", "..", "recordings"))

    def _do_save():
        from datetime import datetime, timezone, timedelta
        os.makedirs(_rec_dir, exist_ok=True)
        stamp     = time.strftime("%Y%m%d_%H%M%S")
        utc_iso   = datetime.now(timezone.utc).isoformat()
        base  = f"kraken_space_doa_{stamp}"
        with S.lock:
            if not S.rec_bursts:
                print("[DOA] No data recorded."); return
            frames_arr = np.stack(S.rec_bursts, axis=0)
            ts_arr     = np.array(S.rec_timestamps, dtype=np.float64)
            az_arr     = np.array(S.rec_az,         dtype=np.float32)
            el_arr     = np.array(S.rec_el,         dtype=np.float32)
            gt_name_arr  = list(S.rec_gt_name)
            gt_az_arr    = np.array(S.rec_gt_az,    dtype=np.float32)
            gt_el_arr    = np.array(S.rec_gt_el,    dtype=np.float32)
            gt_norad_arr = np.array(S.rec_gt_norad, dtype=np.int32)
            n_saved    = len(frames_arr)
            dur_s      = float(time.time() - S.t_start)
        npz_p  = os.path.join(_rec_dir, base + ".npz")
        json_p = os.path.join(_rec_dir, base + ".json")

        # ── Ground-truth satellite positions ──────────────────────────────────
        # Primary: real-time Doppler-based matching collected during recording
        # Fallback: post-hoc angle-based matching (if no live GT was collected)
        npz_extra: dict = {}
        _meta_gt: dict  = {}
        n_rt_gt = int(np.sum(np.isfinite(gt_az_arr)))
        if n_rt_gt > 0:
            npz_extra = {
                "gt_az_deg":     gt_az_arr,
                "gt_el_deg":     gt_el_arr,
                "gt_sat_name":   np.array(gt_name_arr, dtype="U32"),
                "gt_norad_id":   gt_norad_arr,
            }
            _meta_gt = {
                "ground_truth_source":    "doppler_realtime",
                "ground_truth_n_matched": n_rt_gt,
            }
            print(f"[DOA] Ground-truth: {n_rt_gt}/{n_saved} bursts matched via Doppler")
        elif _HAS_OBSERVER:
            try:
                _lat, _lon, _alt = _get_observer(interactive=False)
                from shared.satellite_tracker import match_bursts_by_angle
                t0_utc = datetime.fromisoformat(utc_iso)
                gt = match_bursts_by_angle(
                    ts_arr, az_arr, el_arr, t0_utc, _lat, _lon, _alt,
                    verbose=True,
                )
                npz_extra = {
                    "gt_az_deg":     gt["az_deg"],
                    "gt_el_deg":     gt["el_deg"],
                    "gt_sat_name":   gt["sat_name"],
                    "gt_norad_id":   gt["norad_id"],
                    "gt_doppler_hz": gt["doppler_hz"],
                }
                _meta_gt = {
                    "ground_truth_source":    gt["source"],
                    "ground_truth_n_matched": gt["n_matched"],
                }
            except Exception as exc:
                print(f"[DOA] Ground-truth annotation skipped: {exc}")

        np.savez_compressed(npz_p, bursts=frames_arr, timestamps=ts_arr,
                            **npz_extra)
        _meta = {
            "freq_hz": FREQ_HZ, "sample_rate_hz": FS, "gain_db": GAIN_DB,
            "n_antennas": N_ANT, "algo": ALGO, "mode": MODE,
            "d_lambda": cfg.d_lambda, "n_az": cfg.n_az, "n_el": cfg.n_el,
            "el_min_deg": cfg.el_min_deg, "n_signals": cfg.num_expected_signals,
            "antenna_input_order": list(_INPUT_ORDER),
            "solver_channel_order": list(CROSS_ARRAY_CANONICAL_ORDER),
            "phase_offsets_deg_input_order": list(np.rad2deg(S.ph_offsets)),
            "burst_threshold_db": THRESHOLD,
            "n_frames": n_saved,
            "duration_s": dur_s,
            "timestamp_utc": utc_iso,
            **(_meta_gt),
        }
        if _HAS_OBSERVER:
            try:
                _lat2, _lon2, _alt2 = _get_observer(interactive=False)
                _meta["observer_lat"] = round(_lat2, 7)
                _meta["observer_lon"] = round(_lon2, 7)
                _meta["observer_alt_m"] = round(_alt2, 1)
            except Exception:
                pass
        with open(json_p, "w") as f:
            json.dump(_meta, f, indent=2)
        n_saved = len(S.rec_bursts)
        # Clear buffer so that pressing Save again doesn't produce a duplicate.
        # Also stop recording so the user knows a fresh session is needed.
        with S.lock:
            S.rec_bursts.clear()
            S.rec_timestamps.clear()
            S.rec_az.clear()
            S.rec_el.clear()
            S.rec_gt_name.clear()
            S.rec_gt_az.clear()
            S.rec_gt_el.clear()
            S.rec_gt_norad.clear()
            S.rec_gt_matched = 0
            S.rec_gt_by_sat.clear()
            S.recording = False
        btn_rec.label.set_text("● Record")
        btn_rec.color = BG3
        fig.canvas.draw_idle()
        print(f"[DOA] Saved {npz_p}  ({n_saved} frames)")

    # ── Animation ─────────────────────────────────────────────────────────────
    _sat_artists: list = []   # matplotlib artists for visible-satellite overlay

    def _update(_):
        with S.lock:
            spec   = S.spec_2d.copy()
            spec_e = S.spec_ema.copy()
            az     = S.az_deg;   el    = S.el_deg
            az_s   = S.az_smooth; el_s  = S.el_smooth
            papr   = S.papr_db;  snr   = S.snr_db
            ev     = S.ev_db.copy()
            coh    = S.coh_mat.copy()
            fft_d  = S.last_fft.copy()
            h_az   = list(S.h_az);   h_el   = list(S.h_el)
            h_papr = list(S.h_papr); h_dop  = list(S.h_dop)
            n_b    = S.burst_count;  n_f    = S.frame_count
            n_pass = S.pass_count
            uw_sc  = S.uw_score
            p_snr  = S.pilot_snr_db
            rec      = S.recording
            r_n      = len(S.rec_bursts)
            r_gt     = S.rec_gt_matched
            r_by_sat = dict(S.rec_gt_by_sat)
            r_tstart = S.t_start
            R      = S.R_now.copy()
            ch_fault = list(S.ch_fault)
            vis_sats = list(S.visible_sats)

        # Sky plot — use EMA-smoothed spectrum and smoothed peak position
        sky_mesh.set_array(np.flipud(spec_e).ravel())
        t_peak, r_peak = skyplot_coords(az_s, el_s)
        sky_peak.set_data([t_peak], [r_peak])
        sky_peak_outer.set_data([t_peak], [r_peak])
        sky_az_line.set_data([t_peak, t_peak], [0, 90])
        _n_vis = len(vis_sats)
        txt_sky.set_text(
            f"Az: {az_s:6.1f}°   El: {el_s:5.1f}°   PAPR: {papr:.1f} dB"
            f"  [Sat #{n_pass}]  ({_n_vis} vis.)"
        )

        # Visible satellite overlay on sky plot (refreshed every animation frame)
        for _art in _sat_artists:
            try:
                _art.remove()
            except Exception:
                pass
        _sat_artists.clear()
        for sv in vis_sats:
            _th = np.deg2rad(sv["az_deg"])
            _r  = 90.0 - sv["el_deg"]
            _m, = ax_sky.plot([_th], [_r], "s", color=C_AMBER,
                              markersize=7, zorder=6, markeredgecolor=BG, markeredgewidth=0.8)
            _dop_khz = sv.get("doppler_hz", 0) / 1000.0
            _label = sv["name"].replace("IRIDIUM ", "#") + f"\n{_dop_khz:+.0f}kHz"
            _t = ax_sky.text(_th, min(_r + 7, 89), _label,
                             color=C_AMBER, fontsize=5.0, ha="center",
                             va="bottom", zorder=7)
            _sat_artists.extend([_m, _t])

        # 2D heatmap — accumulated EMA spectrum (stable hot zone)
        heat_img.set_data(spec_e)
        heat_vline.set_xdata([az_s, az_s])
        heat_hline.set_ydata([el_s, el_s])
        txt_heat.set_text(f"Az={az_s:.1f}°  El={el_s:.1f}°  (raw {az:.0f}°/{el:.0f}°)")

        # Az + El history
        line_az.set_ydata(h_az);  line_el.set_ydata(h_el)

        # Eigenvalues
        for bar, val in zip(bars_eig, ev):
            bar.set_height(max(val, 0))
        spread = float(ev[0]) - float(ev[-1]) if len(ev) > 1 else 0.0
        txt_eig.set_text(f"spread={spread:.0f}dB  UW:{uw_sc:.2f}  Pilot:{p_snr:.0f}dB")
        bars_eig[0].set_color(C_BLUE if spread > 5.0 else C_DIM)

        # Coherence
        coh_img.set_data(coh)

        # PAPR + SNR
        line_papr.set_ydata(h_papr); line_snr.set_ydata(h_dop)

        # FFT
        line_fft.set_ydata(fft_d)

        # Phase stability (off-diag |R| pairs)
        pairs = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
        R_ab  = [float(abs(R[i, j])) for i, j in pairs]
        max_v = max(R_ab) if max(R_ab) > 1e-10 else 1.0
        for bar, v in zip(bars_ph, R_ab):
            bar.set_height(v / max_v)

        # Channel fault overlay on FFT panel
        if ch_fault:
            ax_fft.set_title(
                f"IQ Spectrum ch0  ⚠ FAULT: {', '.join(ch_fault)}",
                color="#ff6060", fontsize=8,
            )
        else:
            ax_fft.set_title("IQ Spectrum  ch0", color=C_TEXT, fontsize=8)

        # Recording / satellite status bar
        if rec:
            _el = time.time() - r_tstart
            _mm, _ss = divmod(int(_el), 60)
            _pct = 100 * r_gt // r_n if r_n else 0
            _sat_parts = "  │  ".join(
                f"#{k.replace('IRIDIUM ', '')}: {v}"
                for k, v in sorted(r_by_sat.items(), key=lambda x: -x[1])
            )
            if not _sat_parts:
                _sat_parts = "(no GT — TLE tracker initialising)"
            txt_status.set_text(
                f"⏺ REC  {r_n} bursts  │  GT {r_gt}/{r_n} ({_pct}%)  │  {_sat_parts}  │  {_mm:02d}:{_ss:02d}"
            )
            txt_status.set_color(C_ROSE)
            ax_status.set_facecolor("#2d1010")
        elif r_n > 0:
            # Stopped but not yet saved
            _pct = 100 * r_gt // r_n if r_n else 0
            txt_status.set_text(
                f"{r_n} bursts buffered  │  GT {r_gt}/{r_n} ({_pct}%)  │  press Save .npz"
            )
            txt_status.set_color(C_AMBER)
            ax_status.set_facecolor(BG3)
        else:
            # Idle: show visible satellites
            if vis_sats:
                _vis_str = "  │  ".join(
                    f"#{sv['name'].replace('IRIDIUM ', '')}"
                    f" az={sv['az_deg']:.0f}° el={sv['el_deg']:.0f}°"
                    f" {sv.get('doppler_hz', 0)/1e3:+.0f}kHz"
                    for sv in vis_sats[:5]
                )
                txt_status.set_text(f"Ready  │  {len(vis_sats)} visible: {_vis_str}")
                txt_status.set_color(C_TEAL)
            else:
                txt_status.set_text("Ready — press ● Record  (TLE tracker starting…)")
                txt_status.set_color(C_MUTED)
            ax_status.set_facecolor(BG3)

    ani = animation.FuncAnimation(fig, _update, interval=C.INTERVAL_MS,
                                   cache_frame_data=False)

    def _on_close(_):
        S.running = False
        _do_save()   # no-op if buffer is empty; avoids data-loss on window close

    fig.canvas.mpl_connect("close_event", _on_close)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        S.running = False


if __name__ == "__main__":
    main()
