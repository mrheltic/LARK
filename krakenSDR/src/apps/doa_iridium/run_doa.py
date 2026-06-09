#!/usr/bin/env python3
"""
run_doa.py  —  Simple modular live DOA pipeline for Iridium IRA bursts.

No GNU Radio required: KrakenIQSource (TCP) + numpy DSP + core DOA algorithms.

Pipeline (7 stages, each a named function call)
────────────────────────────────────────────────
  1. Source       KrakenIQSource.get_frame()         →  (n_ant, N) IQ
  2. Detection    detect_energy_bursts()             →  burst start index
  3. Tone scan    scan_preamble_tones()              →  tone_hz (= 3125 + Doppler)
  4. BPF          apply_bpf_and_normalize()          →  (n_ant, window) narrowband IQ
  5. Calibration  apply_phase_correction()           →  phase-aligned IQ
  6. Covariance   compute_mf_covariance() + EMA      →  R_ema (n_ant×n_ant)
  7. DOA          doa_*_uca_2d() + EMA smoothing     →  az_ema, el_ema

Usage (headless — JSON to stdout):
    python3 run_doa.py

Usage (custom config):
    python3 run_doa.py --config doa_config.toml

Usage (live UI):
    python3 run_doa.py --gui

Usage (record raw Kraken IQ + DOA spectra):
    python3 run_doa.py --record data/doa_iridium
    python3 run_doa.py --record /tmp/doa --no-record-raw   # spectra only

CLI flags override the TOML config for quick experiments:
    python3 run_doa.py --gain 36.4 --algo capon --mode indoor --gui
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.normpath(os.path.join(_HERE, "..", ".."))   # krakenSDR/src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── TOML loader (Python ≥3.11 built-in; install 'tomli' for older versions) ──
try:
    import tomllib                  # Python 3.11+  
except ImportError:
    try:
        import tomli as tomllib     # pip install tomli
    except ImportError:
        tomllib = None              # will use defaults only

# ── Dark colour palette (same family as the old runner) ───────────────────────
BG     = "#1a1d27"
BG2    = "#21253a"
BG3    = "#2a2f47"
C_BDR  = "#3b4263"
C_MUT  = "#8891b0"
C_TEXT = "#d8dae8"
C_BLUE  = "#5ea4e0"
C_TEAL  = "#4ecdc4"
C_AMBER = "#f4a431"
C_VIO   = "#a78bfa"
C_ROSE  = "#f16b6f"
C_LIME  = "#6dd97d"

# ── Detection mode profiles ───────────────────────────────────────────────────
_PROFILES: dict[str, dict] = {
    "indoor": dict(
        tone_nom_hz=3125.0, scan_bw_hz=3_000.0,  bpf_bw_hz=8_000.0,
        dc_guard_hz=200.0,  min_snr_db=2.0,       energy_threshold=3.0,
    ),
    "outdoor": dict(
        tone_nom_hz=3125.0, scan_bw_hz=45_000.0, bpf_bw_hz=15_000.0,
        dc_guard_hz=500.0,  min_snr_db=3.0,       energy_threshold=2.0,
    ),
}

# ── Default configuration (mirrors doa_config.toml) ──────────────────────────
_DEFAULTS: dict = {
    "hardware": {
        "daq_ip": "localhost", "port": 5000, "ctrl_port": 5001,
        "freq_mhz": 1626.27, "gain_db": 30.0,
        "cpi_size": 131072, "pre_samples": 2621, "window_samples": 3000, "bpf_guard": 128,
    },
    "array": {
        "n_ant": 5, "radius_lambda": 0.4253,
        "ant0_offset_deg": 0.0, "ant_ccw": False,
        "use_phase_cal": False, "cal_file": "",
    },
    "algorithm": {
        "algo": "music", "mode": "outdoor",
        "n_az": 360, "n_el": 86,
    },
    "thresholds": {
        "snr_min_db": 3.0, "papr_min_db": 2.0,
        "cov_alpha": 0.93, "az_ema_alpha": 0.88, "el_ema_alpha": 0.65,
    },
    "output": {"json_file": "", "debug_dir": ""},
    "recording": {
        "enabled": False, "dir": "", "record_raw": True, "checkpoint_s": 60.0,
        "consolidate_on_exit": False,
    },
    "ui": {"update_interval_ms": 300, "history_len": 120},
}


# =============================================================================
# Configuration helpers
# =============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(config_path: str) -> dict:
    cfg = _deep_merge({}, _DEFAULTS)
    if config_path and os.path.exists(config_path):
        if tomllib is None:
            print("[config] tomllib not available — using defaults. "
                  "Install 'tomli' or upgrade to Python 3.11+.")
        else:
            with open(config_path, "rb") as f:
                cfg = _deep_merge(cfg, tomllib.load(f))
    elif config_path:
        print(f"[config] {config_path!r} not found — using defaults.")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iridium DOA live pipeline (no GNU Radio required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=os.path.join(_HERE, "doa_config.toml"),
                   metavar="FILE", help="TOML config file")
    p.add_argument("--gui",  action="store_true", help="Open live matplotlib UI")
    # Hardware overrides
    p.add_argument("--host",  metavar="IP",  help="Heimdall DAQ IP")
    p.add_argument("--freq",  type=float, metavar="HZ",  help="Centre frequency [Hz]")
    p.add_argument("--gain",  type=float, metavar="DB",  help="IF gain [dB]")
    # Array overrides
    p.add_argument("--n-ant",      type=int,   metavar="N",   help="Number of antennas")
    p.add_argument("--cal-file",   metavar="PATH",             help="Calibration .npz")
    p.add_argument("--ant0-offset",type=float, metavar="DEG", help="Ant-0 offset from North [°]")
    p.add_argument("--ccw",        action="store_true",        help="Antennae are CCW")
    # Algorithm overrides
    p.add_argument("--algo", choices=["music", "capon", "bartlett"], help="DOA algorithm")
    p.add_argument("--mode", choices=["indoor", "outdoor"],          help="Detection profile")
    # Output
    p.add_argument("--out",       metavar="FILE", help="JSON output file (additionally to stdout)")
    p.add_argument("--debug-dir", metavar="DIR",  help="Save intermediate data per frame")
    p.add_argument("--record",    metavar="DIR",  help="Record session (raw IQ + doa_music.npz)")
    p.add_argument("--no-record-raw", action="store_true",
                   help="With --record: save DOA spectra only, skip raw CPI frames")
    p.add_argument("--record-checkpoint", type=float, metavar="SEC",
                   help="Flush recording checkpoint every N seconds")
    p.add_argument("--record-consolidate", action="store_true",
                   help="Build raw_iq.npz on exit (slow for long sessions)")
    p.add_argument("--phase-cal", action="store_true",
                   help="Enable hardware phase calibration (cal_file)")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


def _apply_cli(cfg: dict, args: argparse.Namespace) -> dict:
    hw = cfg["hardware"]; arr = cfg["array"]; alg = cfg["algorithm"]
    out = cfg["output"]; rec = cfg["recording"]
    if args.host is not None:        hw["daq_ip"]          = args.host
    if args.freq is not None:        hw["freq_mhz"]        = args.freq / 1e6
    if args.gain is not None:        hw["gain_db"]         = args.gain
    if args.n_ant is not None:       arr["n_ant"]          = args.n_ant
    if args.cal_file is not None:    arr["cal_file"]       = args.cal_file
    if args.ant0_offset is not None: arr["ant0_offset_deg"]= args.ant0_offset
    if args.ccw:                     arr["ant_ccw"]        = True
    if getattr(args, "phase_cal", False):
        arr["use_phase_cal"] = True
    if args.algo is not None:        alg["algo"]           = args.algo
    if args.mode is not None:        alg["mode"]           = args.mode
    if args.out is not None:         out["json_file"]      = args.out
    if args.debug_dir is not None:   out["debug_dir"]      = args.debug_dir
    if getattr(args, "record", None):
        rec["enabled"] = True
        rec["dir"] = args.record
    if getattr(args, "no_record_raw", False):
        rec["record_raw"] = False
    if getattr(args, "record_checkpoint", None) is not None:
        rec["checkpoint_s"] = args.record_checkpoint
    if getattr(args, "record_consolidate", False):
        rec["consolidate_on_exit"] = True
    return cfg


def _load_cal(cal_file: str, n_ant: int) -> list[float]:
    if not cal_file:
        return [0.0] * n_ant
    try:
        data = np.load(cal_file)
        offs = list(data["phase_offsets_deg"])
        offs += [0.0] * max(0, n_ant - len(offs))
        return [float(x) for x in offs[:n_ant]]
    except Exception as exc:
        print(f"[cal] Could not load {cal_file!r}: {exc} — using zeros.")
        return [0.0] * n_ant


# =============================================================================
# Shared state
# =============================================================================

def make_state(history_len: int = 120) -> dict:
    """Thread-safe dictionary shared between pipeline and UI threads."""
    return {
        "lock":    threading.Lock(),
        "running": True,
        # Latest estimates
        "az_raw": np.nan, "el_raw": np.nan,
        "az_ema": np.nan, "el_ema": np.nan,
        # History ring-buffers
        "az_hist": collections.deque(maxlen=history_len),
        "el_hist": collections.deque(maxlen=history_len),
        "t_hist":  collections.deque(maxlen=history_len),
        # Diagnostic data for UI panels
        "spec2d":    None,   # (n_el, n_az) MUSIC spectrum [dB]
        "R_ema":     None,   # (n_ant, n_ant) EMA covariance
        "eigenvalues": None, # (n_ant,) sorted descending
        "phase_meas":  None, # (n_ant-1,) measured inter-ant phase diffs [°]
        "phase_exp":   None, # (n_ant-1,) expected phase diffs at current DOA [°]
        # Scalar diagnostics
        "tone_hz": 0.0, "snr_db": 0.0, "papr_db": 0.0, "cfo_hz": 0.0,
        "burst_count": 0, "frame_count": 0,
    }


# =============================================================================
# Pipeline thread
# =============================================================================

def pipeline_thread(state: dict, cfg: dict, out_file=None, verbose: bool = False,
                    recorder=None):
    """
    Acquisition + processing loop.  Runs in a background daemon thread.
    Writes results to stdout (always) and optionally to out_file.
    Updates state dict under lock for the UI thread.
    """
    from hardware.kraken_iq_source import KrakenIQSource
    from core.burst_processing import (
        detect_energy_bursts, scan_preamble_tones,
        apply_bpf_and_normalize, compute_mf_covariance,
    )
    from core.pipeline_debug import PipelineDebugSaver
    from core.doa_uca_2d import (
        UcaConfig,
        doa_music_uca_2d, doa_capon_uca_2d, doa_bartlett_uca_2d,
        find_peak_uca_2d,
    )
    from core.doa_algorithms import apply_phase_correction

    hw   = cfg["hardware"]
    arr  = cfg["array"]
    alg  = cfg["algorithm"]
    thr  = cfg["thresholds"]
    out  = cfg["output"]

    fs             = 1_024_000.0
    n_ant          = arr["n_ant"]
    pre_samples    = hw["pre_samples"]
    window_samples = hw["window_samples"]
    bpf_guard      = hw["bpf_guard"]
    algo           = alg["algo"].upper()
    profile        = _PROFILES[alg["mode"]]
    radius_lambda  = arr["radius_lambda"]
    use_phase_cal  = bool(arr.get("use_phase_cal", False))
    phase_offs     = _load_cal(arr["cal_file"], n_ant) if use_phase_cal else [0.0] * n_ant
    has_cal        = use_phase_cal and any(p != 0.0 for p in phase_offs)

    uca = UcaConfig(
        n_ant=n_ant, radius_lambda=radius_lambda,
        n_az=alg["n_az"], n_el=alg["n_el"],
        el_min_deg=5.0, el_max_deg=90.0,
        ant0_offset_deg=arr["ant0_offset_deg"],
        ant_ccw=arr["ant_ccw"],
        num_expected_signals=1,
    )

    debug_dir = out.get("debug_dir", "")
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    cov_alpha = thr["cov_alpha"]
    az_alpha  = thr["az_ema_alpha"]
    el_alpha  = thr["el_ema_alpha"]

    # Precompute antenna angular positions for expected phase diff computation
    _phi_k = 2 * np.pi * np.arange(1, n_ant) / n_ant   # (n_ant-1,), CW from North

    src = KrakenIQSource(
        host=hw["daq_ip"], port=hw["port"], ctrl_port=hw["ctrl_port"],
        num_channels=n_ant, freq_hz=hw["freq_mhz"] * 1e6,
        gain_db=hw["gain_db"], queue_size=4,
        verbose=5 if verbose else 0,
    )
    src.start()

    R_ema  = None
    az_ema = el_ema = None
    frame_idx = 0

    print(_banner(cfg))

    try:
        while state["running"]:
            frame = src.get_frame(timeout=1.0)
            if frame is None:
                continue

            X = frame[:n_ant, :]
            frame_idx += 1
            t_frame = time.time()
            with state["lock"]:
                state["frame_count"] = frame_idx

            if recorder is not None:
                recorder.add_raw_frame(X, t_frame)

            dbg = PipelineDebugSaver(debug_dir, frame_idx) if debug_dir else None
            if dbg and dbg.enabled:
                dbg.save("raw_iq", X)

            # ── Stage 1: Energy burst detection ──────────────────────────────
            burst_starts = detect_energy_bursts(
                X[0], fs,
                threshold_factor=profile["energy_threshold"],
            )
            if dbg and dbg.enabled:
                dbg.save("burst_starts", np.asarray(burst_starts, dtype=np.int64))
            if not burst_starts:
                continue

            b0   = burst_starts[0]
            bend = min(b0 + window_samples, X.shape[1])
            if bend - b0 < pre_samples + bpf_guard:
                continue

            # ── Stage 2: Preamble tone scan ───────────────────────────────────
            tones = scan_preamble_tones(
                X[0, b0:bend], fs,
                nom_tone_hz=profile["tone_nom_hz"],
                scan_bw_hz=profile["scan_bw_hz"],
                min_snr_db=profile["min_snr_db"],
                dc_guard_hz=profile["dc_guard_hz"],
            )
            if not tones:
                continue
            tone_hz, tone_snr = tones[0]
            if dbg and dbg.enabled:
                dbg.save("tones", [{"tone_hz": float(t), "snr_db": float(s)} for t, s in tones])

            # ── Stage 3: BPF + amplitude normalisation ────────────────────────
            X_win = X[:, b0:bend]
            try:
                X_bpf = apply_bpf_and_normalize(
                    X_win, window_samples, fs, tone_hz, profile["bpf_bw_hz"]
                )
            except ValueError:
                continue

            if dbg and dbg.enabled:
                dbg.save("bpf", X_bpf)

            # ── Stage 4: Hardware phase calibration ───────────────────────────
            X_cal = apply_phase_correction(X_bpf, phase_offs) if has_cal else X_bpf
            if dbg and dbg.enabled:
                dbg.save("phase_corrected", X_cal)

            # ── Stage 5: Matched-filter covariance ────────────────────────────
            try:
                R_mf, _, snr_db = compute_mf_covariance(
                    X_cal, tone_hz, fs, pre_samples, bpf_guard
                )
            except ValueError:
                continue

            if snr_db < thr["snr_min_db"]:
                continue

            R_ema = R_mf.copy() if R_ema is None else (
                cov_alpha * R_ema + (1.0 - cov_alpha) * R_mf
            )

            if dbg and dbg.enabled:
                dbg.save("R_mf", R_mf)
                dbg.save("R_ema", R_ema)

            # ── Stage 6: 2D DOA ───────────────────────────────────────────────
            if algo == "CAPON":
                spec = doa_capon_uca_2d(X_cal, uca, R_in=R_ema, decorr="none")
            elif algo == "BARTLETT":
                spec = doa_bartlett_uca_2d(X_cal, uca, R_in=R_ema)
            else:
                spec = doa_music_uca_2d(X_cal, uca, R_in=R_ema)

            az_raw, el_raw, papr = find_peak_uca_2d(spec, uca)

            if papr < thr["papr_min_db"]:
                continue

            if dbg and dbg.enabled:
                dbg.save("spec2d", spec)

            # ── Stage 7: EMA smoothing ────────────────────────────────────────
            if az_ema is None:
                az_ema, el_ema = az_raw, el_raw
            else:
                d_az   = ((az_raw - az_ema + 180.0) % 360.0) - 180.0
                az_ema = (az_ema + az_alpha * d_az) % 360.0
                el_ema += el_alpha * (el_raw - el_ema)

            # ── Derived diagnostics for UI panels ────────────────────────────
            eigvals = np.sort(np.real(np.linalg.eigvalsh(R_ema)))[::-1]

            # Measured inter-antenna phase diffs from cross-correlation column
            phase_meas = np.degrees(np.angle(R_ema[1:, 0]))

            # Expected phase diffs: UCA steering at current DOA estimate
            az_r = np.radians(az_ema)
            el_r = np.radians(el_ema)
            phase_exp = np.degrees(
                2.0 * np.pi * radius_lambda * np.cos(el_r) *
                (np.cos(_phi_k - az_r) - np.cos(-az_r))
            )

            t_now = time.time()

            # ── Update shared state (UI reads this under lock) ─────────────────
            with state["lock"]:
                state["az_raw"]    = az_raw
                state["el_raw"]    = el_raw
                state["az_ema"]    = az_ema
                state["el_ema"]    = el_ema
                state["az_hist"].append(az_ema)
                state["el_hist"].append(el_ema)
                state["t_hist"].append(t_now)
                state["spec2d"]    = spec
                state["R_ema"]     = R_ema.copy()
                state["eigenvalues"] = eigvals
                state["phase_meas"]  = phase_meas
                state["phase_exp"]   = phase_exp
                state["tone_hz"]   = tone_hz
                state["snr_db"]    = snr_db
                state["papr_db"]   = float(papr)
                state["cfo_hz"]    = tone_hz - profile["tone_nom_hz"]
                state["burst_count"] += 1

            n = state["burst_count"]
            doa_result = {
                "t": round(t_now, 3), "n": n, "frame": frame_idx,
                "az": round(az_ema, 1), "el": round(el_ema, 1),
                "az_raw": round(az_raw, 1), "el_raw": round(el_raw, 1),
                "snr_db": round(snr_db, 1), "papr_db": round(float(papr), 1),
                "cfo_hz": round(tone_hz - profile["tone_nom_hz"], 0),
                "tone_hz": round(tone_hz, 1),
                "algo": algo,
            }
            if dbg and dbg.enabled:
                dbg.save("doa_result", doa_result)

            if recorder is not None:
                recorder.add_doa(
                    spec, az_ema, el_ema,
                    papr_db=float(papr), snr_db=snr_db, timestamp=t_now,
                )

            rec = json.dumps(doa_result)
            print(rec, flush=True)
            if out_file:
                print(rec, file=out_file, flush=True)

            if verbose:
                print(f"  [{n:4d}] az={az_ema:6.1f}° el={el_ema:5.1f}° "
                      f"snr={snr_db:.1f}dB cfo={tone_hz - profile['tone_nom_hz']:+.0f}Hz",
                      file=sys.stderr)

    finally:
        src.stop()
        if recorder is not None:
            recorder.save()


def _banner(cfg: dict) -> str:
    hw = cfg["hardware"]; alg = cfg["algorithm"]; arr = cfg["array"]
    rec = cfg.get("recording", {})
    cal = "off"
    if arr.get("use_phase_cal"):
        cal = arr.get("cal_file", "") or "(zeros)"
    rec_line = ""
    if rec.get("enabled"):
        raw = "raw IQ + " if rec.get("record_raw", True) else ""
        rec_line = f"  Record: {raw}doa_music.npz\n"
    return (
        f"\n{'═'*56}\n"
        f"  Iridium DOA  —  {alg['algo'].upper()} / {alg['mode']}\n"
        f"  Freq  : {hw['freq_mhz']:.3f} MHz     Gain : {hw['gain_db']:.1f} dB\n"
        f"  Source: {hw['daq_ip']}:{hw['port']}\n"
        f"  Phase : {cal}\n"
        f"{rec_line}"
        f"  Ctrl+C to stop\n"
        f"{'═'*56}"
    )


# =============================================================================
# Live UI (matplotlib)
# =============================================================================

def _build_ui(state: dict, cfg: dict):
    """
    Build the matplotlib figure and return (fig, animation).

    Layout  (16×9, dark theme)
    ──────────────────────────
      Left  40 %   Skyplot (polar, N-up clockwise)
      Right 60 %   ┬ Az/El timeseries
                   ├ 2D MUSIC heatmap
                   └ Eigenvalues | Phase residuals | Status
    """
    import matplotlib
    for _backend in ("Qt5Agg", "TkAgg", "Qt6Agg"):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.animation as animation

    alg_cfg = cfg["algorithm"]
    ui_cfg  = cfg["ui"]
    n_ant   = cfg["array"]["n_ant"]
    n_az    = alg_cfg["n_az"]
    n_el    = alg_cfg["n_el"]
    hist    = ui_cfg["history_len"]

    # ── Global rcParams for dark theme ────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG2,
        "axes.edgecolor":   C_BDR,
        "axes.labelcolor":  C_MUT,
        "text.color":       C_TEXT,
        "xtick.color":      C_MUT,
        "ytick.color":      C_MUT,
        "grid.color":       C_BDR,
        "grid.alpha":       0.45,
        "axes.titlesize":   9,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
    })

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("Iridium DOA")
    except Exception:
        pass

    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[4, 6], wspace=0.04,
        left=0.04, right=0.97, top=0.95, bottom=0.05,
    )

    # ─────────────────────── Left: Skyplot ────────────────────────────────────
    ax_sky = fig.add_subplot(gs_outer[0, 0], polar=True)
    ax_sky.set_facecolor(BG2)
    ax_sky.set_theta_zero_location("N")   # North at top
    ax_sky.set_theta_direction(-1)         # clockwise
    ax_sky.set_rlim(0, 88)
    ax_sky.set_rticks([15, 30, 45, 60, 75])
    ax_sky.set_yticklabels(["75°", "60°", "45°", "30°", "15°"],
                            color=C_MUT, fontsize=7)
    ax_sky.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                            color=C_TEXT, fontsize=8)
    ax_sky.grid(color=C_BDR, linewidth=0.5, alpha=0.7)
    ax_sky.spines["polar"].set_edgecolor(C_BDR)
    ax_sky.set_title("Skyplot", color=C_TEXT, fontsize=10, pad=10)

    trail_sc   = ax_sky.scatter([], [], s=20, c=[], cmap="plasma",
                                 vmin=0, vmax=1, alpha=0.55, zorder=2)
    current_sc = ax_sky.scatter([], [], s=130, color=C_AMBER,
                                 marker="o", zorder=3,
                                 edgecolors="white", linewidths=1.2)
    wait_txt   = ax_sky.text(
        0.5, 0.5, "Waiting for bursts…",
        ha="center", va="center",
        color=C_MUT, fontsize=9, transform=ax_sky.transAxes,
    )

    # ─────────────────────── Right panel ─────────────────────────────────────
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_outer[0, 1],
        height_ratios=[3, 4, 3], hspace=0.44,
    )

    # ── Row 0: Az/El history ──────────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs_right[0])
    ax_hist.set_xlim(0, hist)
    ax_hist.set_ylim(0, 360)
    ax_hist.set_ylabel("Azimuth [°]", color=C_BLUE, fontsize=8)
    ax_hist.set_xlabel("Burst #", color=C_MUT, fontsize=8)
    ax_hist.set_title("Az / El History", color=C_TEXT, fontsize=9, loc="left")
    ax_hist.grid(True)
    ax_hist.spines[:].set_edgecolor(C_BDR)

    ax_el = ax_hist.twinx()
    ax_el.set_ylim(0, 90)
    ax_el.set_ylabel("Elevation [°]", color=C_AMBER, fontsize=8)
    ax_el.tick_params(colors=C_MUT)
    ax_el.spines[:].set_edgecolor(C_BDR)

    (az_line,) = ax_hist.plot([], [], color=C_BLUE,  lw=1.8, zorder=3)
    (el_line,) = ax_el.plot([],   [], color=C_AMBER, lw=1.8, zorder=3)
    ax_hist.tick_params(axis="y", colors=C_BLUE)

    # ── Row 1: 2D MUSIC heatmap ───────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs_right[1])
    ax_heat.set_facecolor(BG)
    _blank = np.full((n_el, n_az), -40.0)
    im_heat = ax_heat.imshow(
        _blank, origin="lower", aspect="auto",
        extent=[0, 360, 5, 90], cmap="plasma", vmin=-40, vmax=0,
    )
    ax_heat.set_xlabel("Azimuth [°]",   color=C_MUT, fontsize=8)
    ax_heat.set_ylabel("Elevation [°]", color=C_MUT, fontsize=8)
    ax_heat.set_title("2D MUSIC Spectrum", color=C_TEXT, fontsize=9, loc="left")
    ax_heat.tick_params(colors=C_MUT)
    ax_heat.spines[:].set_edgecolor(C_BDR)
    peak_cross = ax_heat.plot([], [], "w+", ms=16, mew=2, zorder=5)[0]
    cb = fig.colorbar(im_heat, ax=ax_heat, fraction=0.018, pad=0.01)
    cb.set_label("dB", color=C_MUT, fontsize=7)
    cb.ax.tick_params(colors=C_MUT, labelsize=6)

    # ── Row 2: 3 sub-panels ───────────────────────────────────────────────────
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_right[2], wspace=0.40,
    )

    # Panel A: Eigenvalue profile
    ax_eig = fig.add_subplot(gs_bot[0])
    ax_eig.set_title("Eigenvalues", color=C_TEXT, fontsize=9, loc="left")
    ax_eig.set_xlabel("Index", color=C_MUT, fontsize=7)
    ax_eig.set_ylabel("Relative power", color=C_MUT, fontsize=7)
    ax_eig.grid(axis="y")
    ax_eig.spines[:].set_edgecolor(C_BDR)
    _eig_colors = [C_BLUE] + [C_MUT] * (n_ant - 1)
    eig_bars = ax_eig.bar(range(n_ant), [0.0] * n_ant,
                           color=_eig_colors, width=0.65)
    ax_eig.set_xticks(range(n_ant))
    ax_eig.set_ylim(0, 1.08)

    # Panel B: Phase residuals (measured − expected) per antenna
    ax_ph = fig.add_subplot(gs_bot[1])
    ax_ph.set_title("Phase residuals CH1–4", color=C_TEXT, fontsize=9, loc="left")
    ax_ph.set_xlabel("Channel", color=C_MUT, fontsize=7)
    ax_ph.set_ylabel("Meas − Exp [°]", color=C_MUT, fontsize=7)
    ax_ph.set_xticks(range(1, n_ant))
    ax_ph.set_ylim(-180, 180)
    ax_ph.axhline(0, color=C_MUT, lw=0.8, ls="--", alpha=0.6)
    ax_ph.grid(axis="y")
    ax_ph.spines[:].set_edgecolor(C_BDR)
    ph_bars = ax_ph.bar(range(1, n_ant), [0.0] * (n_ant - 1),
                         color=C_TEAL, width=0.55, alpha=0.85)

    # Panel C: Status text
    ax_stat = fig.add_subplot(gs_bot[2])
    ax_stat.set_facecolor(BG3)
    ax_stat.set_xlim(0, 1)
    ax_stat.set_ylim(0, 1)
    ax_stat.axis("off")
    ax_stat.spines[:].set_edgecolor(C_BDR)
    stat_txt = ax_stat.text(
        0.08, 0.92, "—",
        transform=ax_stat.transAxes,
        color=C_TEXT, fontsize=9, va="top", family="monospace",
        linespacing=1.75,
    )

    # ── Animation callback ────────────────────────────────────────────────────
    def _update(_fn):
        with state["lock"]:
            az_ema   = state["az_ema"]
            el_ema   = state["el_ema"]
            az_h     = list(state["az_hist"])
            el_h     = list(state["el_hist"])
            spec2d   = state["spec2d"]
            eigvals  = state["eigenvalues"]
            ph_meas  = state["phase_meas"]
            ph_exp   = state["phase_exp"]
            tone_hz  = state["tone_hz"]
            snr_db   = state["snr_db"]
            papr_db  = state["papr_db"]
            cfo_hz   = state["cfo_hz"]
            n_bursts = state["burst_count"]
            n_frames = state["frame_count"]

        has_data = not np.isnan(az_ema)

        # Skyplot
        wait_txt.set_visible(not has_data)
        if has_data and az_h:
            n_h     = len(az_h)
            theta_h = np.radians(az_h)
            r_h     = 90.0 - np.asarray(el_h)
            trail_sc.set_offsets(np.c_[theta_h, r_h])
            trail_sc.set_array(np.linspace(0, 1, n_h))
            current_sc.set_offsets([[np.radians(az_ema), 90.0 - el_ema]])
        else:
            trail_sc.set_offsets(np.empty((0, 2)))
            current_sc.set_offsets(np.empty((0, 2)))

        # Az/El history
        if az_h:
            x = np.arange(len(az_h))
            az_line.set_data(x, az_h)
            el_line.set_data(x, el_h)
            ax_hist.set_xlim(0, max(hist, len(az_h)))

        # 2D MUSIC heatmap
        if spec2d is not None:
            im_heat.set_data(spec2d)
            peak = float(np.max(spec2d))
            im_heat.set_clim(peak - 40.0, peak)
            if has_data:
                peak_cross.set_data([az_ema], [el_ema])
            else:
                peak_cross.set_data([], [])

        # Eigenvalue profile (normalised to max)
        if eigvals is not None:
            max_ev = max(float(eigvals[0]), 1e-12)
            for bar, ev in zip(eig_bars, eigvals):
                bar.set_height(max(float(ev), 0.0) / max_ev)

        # Phase residuals
        if ph_meas is not None and ph_exp is not None:
            residuals = ((ph_meas - ph_exp + 180.0) % 360.0) - 180.0
            for bar, res in zip(ph_bars, residuals):
                res = float(res)
                bar.set_y(min(0.0, res))
                bar.set_height(abs(res))

        # Status text
        if has_data:
            stat_txt.set_text(
                f"Az    {az_ema:7.1f} °\n"
                f"El    {el_ema:7.1f} °\n"
                f"SNR   {snr_db:7.1f} dB\n"
                f"PAPR  {papr_db:7.1f} dB\n"
                f"CFO   {cfo_hz:+7.0f} Hz\n"
                f"Tone  {tone_hz:7.0f} Hz\n"
                f"────────────────\n"
                f"Bursts  {n_bursts:6d}\n"
                f"Frames  {n_frames:6d}"
            )

    ani = animation.FuncAnimation(
        fig, _update,
        interval=ui_cfg["update_interval_ms"],
        blit=False,
        cache_frame_data=False,
    )
    return fig, ani


# =============================================================================
# Entry point
# =============================================================================

def _make_recorder(cfg: dict):
    rec_cfg = cfg.get("recording", {})
    if not rec_cfg.get("enabled"):
        return None
    from core.recording import SessionRecorder, default_record_dir

    hw = cfg["hardware"]
    alg = cfg["algorithm"]
    out_dir = rec_cfg.get("dir") or default_record_dir()
    return SessionRecorder(
        out_dir=out_dir,
        freq_hz=hw["freq_mhz"] * 1e6,
        fs=1_024_000.0,
        gain_db=hw["gain_db"],
        n_ant=cfg["array"]["n_ant"],
        cpi_size=hw.get("cpi_size", 131072),
        n_az=alg["n_az"],
        n_el=alg["n_el"],
        record_raw=bool(rec_cfg.get("record_raw", True)),
        checkpoint_s=float(rec_cfg.get("checkpoint_s", 60.0)),
        consolidate_on_exit=bool(rec_cfg.get("consolidate_on_exit", False)),
        mode=str(alg.get("mode", "")),
        algo=str(alg.get("algo", "")),
    )


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = _apply_cli(cfg, args)

    state    = make_state(history_len=cfg["ui"]["history_len"])
    out_path = cfg["output"].get("json_file", "")
    out_file = open(out_path, "w") if out_path else None
    recorder = _make_recorder(cfg)

    def _on_exit(signum, _frame):
        if state["running"]:
            state["running"] = False
            print("\n[stop] Shutting down — saving recording (please wait, do not press Ctrl+C again)…",
                  flush=True)
        else:
            print("\n[stop] Save in progress — please wait…", flush=True)

    signal.signal(signal.SIGINT,  _on_exit)
    signal.signal(signal.SIGTERM, _on_exit)

    t = threading.Thread(
        target=pipeline_thread,
        args=(state, cfg, out_file, args.verbose, recorder),
        name="doa-pipeline",
        daemon=False,
    )
    t.start()

    if args.gui:
        fig, ani = _build_ui(state, cfg)
        import matplotlib.pyplot as plt
        plt.show()
        state["running"] = False
    else:
        try:
            while state["running"]:
                time.sleep(0.5)
        except KeyboardInterrupt:
            state["running"] = False

    t.join()
    if recorder is not None:
        recorder.save_done.wait(timeout=600.0)

    if out_file:
        out_file.close()

    n = state["burst_count"]
    f = state["frame_count"]
    print(f"\nDone — {n} bursts in {f} frames.", file=sys.stderr)


if __name__ == "__main__":
    main()
