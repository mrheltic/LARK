#!/usr/bin/env python3
"""
algo_compare_offline.py — Offline algorithm comparison on recorded burst_data_*.npz
======================================================================================

Loads the latest (or specified) burst_data_*_iq.npz recording, replays covariance
matrices through all DoA algorithms and produces a comprehensive comparison report
(statistics table + multi-panel plot saved as PDF/PNG).

Algorithms compared
-------------------
  • MUSIC-FB   — forward-backward spatial smoothing (multipath-robust)
  • MUSIC-SS   — forward-only spatial smoothing
  • Capon/MVDR — minimum-variance distortionless response
  • Bartlett   — conventional beamforming (CBF), no null steering
  • Root-MUSIC — polynomial rooting, az-only
  • Unitary-ESPRIT — real-domain ESPRIT on UCA phase modes, az-only

Usage
-----
    python3 algo_compare_offline.py                        # use latest burst file
    python3 algo_compare_offline.py --file path/to.npz    # specific file
    python3 algo_compare_offline.py --demo                 # synthetic benchmark (no file)
    python3 algo_compare_offline.py --out /tmp/compare.pdf
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import config as C
from core.doa_uca_2d import (
    UcaConfig, doa_music_uca_2d, doa_capon_uca_2d, doa_bartlett_uca_2d,
    doa_root_music_uca_2d, doa_unitary_esprit_uca_2d,
    find_peak_uca_2d, eigenvalue_spread_uca_db,
)
from core.doa_algorithms import apply_phase_correction

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
C_BDR = "#3b4263"; C_MUT = "#8891b0"; C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIO  = "#a78bfa"; C_ROSE = "#f16b6f"; C_LIME  = "#6dd97d"
C_ORG  = "#f58a42"; C_PINK = "#e879a0"

ALGO_DEFS = [
    # (key,       label,            color,   supports_2d, fb_mode)
    ("music_fb",  "MUSIC-FB",       C_LIME,  True,  "fb"),
    ("music_ss",  "MUSIC-SS",       C_TEAL,  True,  "ss"),
    ("capon",     "Capon / MVDR",   C_AMBER, True,  None),
    ("bartlett",  "Bartlett / CBF", C_BLUE,  True,  None),
    ("root_music","Root-MUSIC",     C_VIO,   True,  None),
    ("esprit",    "U-ESPRIT",       C_ROSE,  True,  None),
]


# =============================================================================
# Synthetic benchmark dataset
# =============================================================================

def _make_synthetic(
    n_bursts: int = 400, snr_db: float = 20.0,
    az_true: float = 54.0, el_true: float = 44.0,
) -> tuple:
    """Generate synthetic R matrices for ground-truth benchmarking."""
    cfg = UcaConfig(
        n_ant=5, radius_lambda=C.RADIUS_LAMBDA,
        n_az=C.N_AZ, n_el=C.N_EL,
        el_min_deg=C.EL_MIN_DEG, el_max_deg=float(getattr(C, "EL_MAX_DEG", 65.0)),
        num_expected_signals=C.NUM_SIGNALS,
        ant0_offset_deg=float(C.ANT0_OFFSET_DEG), ant_ccw=bool(C.ANT_CCW),
    )
    rng = np.random.default_rng(42)
    pos = cfg.positions
    az_r = np.deg2rad(az_true); el_r = np.deg2rad(el_true)
    tau = 2 * np.pi * (pos[:, 0] * np.cos(el_r) * np.sin(az_r)
                       + pos[:, 1] * np.cos(el_r) * np.cos(az_r))
    a = np.exp(1j * tau)  # (5,) steering vector
    snr_lin = 10 ** (snr_db / 10.0)

    N_snap = 64  # snapshots per burst (= _PRE_SAMPLES equivalent)
    R_list = []
    for _ in range(n_bursts):
        s = (rng.standard_normal(N_snap) + 1j * rng.standard_normal(N_snap)) / np.sqrt(2)
        noise = (rng.standard_normal((5, N_snap)) + 1j * rng.standard_normal((5, N_snap))) / np.sqrt(2)
        X = np.sqrt(snr_lin) * np.outer(a, s) + noise
        R = (X @ X.conj().T) / N_snap
        R_list.append(R)
    return cfg, np.array(R_list), az_true, el_true


# =============================================================================
# Load recorded data
# =============================================================================

def _load_data(path: str) -> tuple:
    """Returns (cfg, R_list, rec_az, rec_el) where R_list is list of (5,5) complex."""
    d = np.load(path, allow_pickle=False)
    cfg = UcaConfig(
        n_ant=5, radius_lambda=C.RADIUS_LAMBDA,
        n_az=C.N_AZ, n_el=C.N_EL,
        el_min_deg=C.EL_MIN_DEG, el_max_deg=float(getattr(C, "EL_MAX_DEG", 65.0)),
        num_expected_signals=C.NUM_SIGNALS,
        ant0_offset_deg=float(C.ANT0_OFFSET_DEG), ant_ccw=bool(C.ANT_CCW),
    )
    R_real = d["R_real"].astype(np.float64)   # (N, 5, 5)
    R_imag = d["R_imag"].astype(np.float64)
    R_list = R_real + 1j * R_imag             # (N, 5, 5) complex
    rec_az = d["az_deg"].astype(np.float64)
    rec_el = d["el_deg"].astype(np.float64)
    rec_papr = d.get("papr_db", np.zeros(len(rec_az))).astype(np.float64)
    return cfg, R_list, rec_az, rec_el, rec_papr


# =============================================================================
# Run one algorithm on a batch of covariance matrices
# =============================================================================

def _run_algo(key: str, R_list: np.ndarray, cfg: UcaConfig,
              fb: str | None) -> np.ndarray:
    """Returns (N, 2) array of [az_est, el_est] per burst. el=NaN for az-only algos."""
    N = len(R_list)
    out = np.empty((N, 2), dtype=np.float64)
    out[:, 1] = np.nan  # default el = NaN for az-only

    for i, R in enumerate(R_list):
        try:
            if key in ("music_fb", "music_ss"):
                spec2d = doa_music_uca_2d(None, cfg, R_in=R, decorr=fb)
                az, el, _ = find_peak_uca_2d(spec2d, cfg)
                out[i] = [az, el]
            elif key == "capon":
                spec2d = doa_capon_uca_2d(None, cfg, R_in=R)
                az, el, _ = find_peak_uca_2d(spec2d, cfg)
                out[i] = [az, el]
            elif key == "bartlett":
                spec2d = doa_bartlett_uca_2d(None, cfg, R_in=R)
                az, el, _ = find_peak_uca_2d(spec2d, cfg)
                out[i] = [az, el]
            elif key == "root_music":
                spec2d, _ = doa_root_music_uca_2d(R, cfg)
                az, el, _ = find_peak_uca_2d(spec2d, cfg)
                out[i] = [az, el]
            elif key == "esprit":
                spec2d, _ = doa_unitary_esprit_uca_2d(R, cfg)
                az, el, _ = find_peak_uca_2d(spec2d, cfg)
                out[i] = [az, el]
        except Exception:
            out[i] = [np.nan, np.nan]
    return out


def _circ_diff(a: np.ndarray, b: float) -> np.ndarray:
    """Circular difference a − b wrapped to (−180, 180]."""
    return ((a - b + 180) % 360) - 180


# =============================================================================
# Main comparison logic
# =============================================================================

def compare(cfg: UcaConfig, R_list: np.ndarray,
            rec_az: np.ndarray | None = None,
            rec_el: np.ndarray | None = None,
            rec_papr: np.ndarray | None = None,
            out_path: str = "./algo_compare.pdf",
            gt_az: float | None = None, gt_el: float | None = None) -> None:

    N = len(R_list)
    print(f"\nRunning comparison on {N} covariance matrices ...")

    results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    for key, label, color, supports_2d, fb in ALGO_DEFS:
        t0 = time.perf_counter()
        est = _run_algo(key, R_list, cfg, fb)
        t1 = time.perf_counter()
        timings[key] = (t1 - t0) / N * 1000   # ms per burst

        az_e  = est[:, 0]
        el_e  = est[:, 1]
        valid = ~np.isnan(az_e)

        # Az statistics (circular)
        if gt_az is not None:
            az_err = _circ_diff(az_e[valid], gt_az)
            az_rmse = float(np.sqrt(np.mean(az_err ** 2)))
            az_bias = float(np.mean(az_err))
        else:
            az_err  = None
            az_rmse = float("nan")
            az_bias = float("nan")

        # El statistics (linear; skip for az-only)
        if gt_el is not None and supports_2d:
            el_err  = el_e[valid] - gt_el
            el_rmse = float(np.sqrt(np.mean(el_err ** 2)))
            el_bias = float(np.mean(el_err))
        else:
            el_err  = None
            el_rmse = float("nan")
            el_bias = float("nan")

        results[key] = dict(
            label=label, color=color,
            az=az_e, el=el_e,
            az_rmse=az_rmse, az_bias=az_bias,
            el_rmse=el_rmse, el_bias=el_bias,
            ms_per_burst=timings[key],
            supports_2d=supports_2d,
            valid_pct=100.0 * np.sum(valid) / N,
        )
        print(f"  {label:20s}  az_rmse={az_rmse:7.2f}°  el_rmse={el_rmse:7.2f}°  "
              f"{timings[key]:.2f} ms/burst  valid={results[key]['valid_pct']:.0f}%")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'Algorithm':<22} {'Az RMSE':>8} {'Az bias':>8} {'El RMSE':>8} "
          f"{'El bias':>8} {'ms/burst':>9} {'valid%':>7}")
    print("-" * 75)
    for key, info in results.items():
        print(f"{info['label']:<22} {info['az_rmse']:>8.2f} {info['az_bias']:>8.2f} "
              f"{info['el_rmse']:>8.2f} {info['el_bias']:>8.2f} "
              f"{info['ms_per_burst']:>9.3f} {info['valid_pct']:>7.1f}%")

    # ── Multi-panel figure ─────────────────────────────────────────────────────
    n_algos = len(ALGO_DEFS)
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        f"DoA Algorithm Comparison — {N} bursts  "
        f"{'GT az=%.1f° el=%.1f°  ' % (gt_az, gt_el) if gt_az is not None else '(recorded data)'}",
        color=C_TEXT, fontsize=11, y=0.99,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           left=0.07, right=0.97, top=0.95, bottom=0.06,
                           hspace=0.50, wspace=0.35)

    # ── [0,0]  Az estimates time series ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0:2], facecolor=BG2)
    ax0.set_facecolor(BG2)
    ax0.set_title("Azimuth estimate — all algorithms", color=C_TEXT, fontsize=9)
    ax0.set_ylabel("Az [°]", color=C_MUT, fontsize=8)
    ax0.set_xlim(0, N); ax0.set_ylim(0, 360)
    ax0.set_xticks(np.linspace(0, N, 6, dtype=int))
    ax0.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax0.spines.values(): sp.set_edgecolor(C_BDR)
    ax0.grid(color=C_BDR, lw=0.4, alpha=0.4)
    if gt_az is not None:
        ax0.axhline(gt_az, color="white", lw=1.5, ls="--", alpha=0.6, label="GT")
    if rec_az is not None:
        ax0.step(np.arange(N), rec_az, where="mid", color=C_MUT,
                 lw=0.6, alpha=0.25, label="recorded")
    for key, label, color, *_ in ALGO_DEFS:
        ax0.plot(results[key]["az"], "-", color=color, lw=0.9, alpha=0.75, label=label)
    ax0.legend(loc="upper right", fontsize=6, ncol=2,
               facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # ── [0,2]  Az RMSE / El RMSE bar chart ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 2], facecolor=BG2)
    ax1.set_facecolor(BG2)
    ax1.set_title("RMSE summary", color=C_TEXT, fontsize=9)
    ax1.set_ylabel("[°]", color=C_MUT, fontsize=8)
    labels = [r["label"] for r in results.values()]
    cols   = [r["color"] for r in results.values()]
    az_rmse_vals = [r["az_rmse"] for r in results.values()]
    el_rmse_vals = [r["el_rmse"] for r in results.values()]
    x = np.arange(len(labels))
    w = 0.38
    ax1.bar(x - w/2, az_rmse_vals, w, color=cols, alpha=0.85, label="Az RMSE")
    ax1.bar(x + w/2, el_rmse_vals, w, color=cols, alpha=0.45, hatch="//",
            edgecolor=C_BDR, label="El RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=6, color=C_MUT)
    ax1.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax1.spines.values(): sp.set_edgecolor(C_BDR)
    ax1.grid(axis="y", color=C_BDR, lw=0.4, alpha=0.35)
    ax1.legend(fontsize=6, facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # ── [1,0]  El estimates time series (2D algos only) ───────────────────────
    ax2 = fig.add_subplot(gs[1, 0:2], facecolor=BG2)
    ax2.set_facecolor(BG2)
    ax2.set_title("Elevation estimate — 2D algorithms", color=C_TEXT, fontsize=9)
    ax2.set_ylabel("El [°]", color=C_MUT, fontsize=8)
    el_min = cfg.el_min_deg; el_max = cfg.el_max_deg
    ax2.set_xlim(0, N); ax2.set_ylim(el_min, el_max)
    ax2.set_xticks(np.linspace(0, N, 6, dtype=int))
    ax2.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax2.spines.values(): sp.set_edgecolor(C_BDR)
    ax2.grid(color=C_BDR, lw=0.4, alpha=0.4)
    if gt_el is not None:
        ax2.axhline(gt_el, color="white", lw=1.5, ls="--", alpha=0.6, label="GT")
    if rec_el is not None:
        ax2.step(np.arange(N), rec_el, where="mid", color=C_MUT,
                 lw=0.6, alpha=0.25, label="recorded")
    for key, label, color, supports_2d, *_ in ALGO_DEFS:
        if supports_2d:
            ax2.plot(results[key]["el"], "-", color=color, lw=0.9, alpha=0.75, label=label)
    ax2.legend(loc="upper right", fontsize=6, ncol=2,
               facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # ── [1,2]  Compute time ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2], facecolor=BG2)
    ax3.set_facecolor(BG2)
    ax3.set_title("Computation time (ms/burst)", color=C_TEXT, fontsize=9)
    ax3.set_ylabel("ms per burst", color=C_MUT, fontsize=8)
    ms_vals = [r["ms_per_burst"] for r in results.values()]
    bars = ax3.bar(x, ms_vals, color=cols, alpha=0.85, edgecolor=BG2)
    ax3.set_xticks(x)
    ax3.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=6, color=C_MUT)
    ax3.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax3.spines.values(): sp.set_edgecolor(C_BDR)
    ax3.grid(axis="y", color=C_BDR, lw=0.4, alpha=0.35)

    # ── [2,0-2]  Az error distributions (one violin/hist per algo) ───────────
    ax4 = fig.add_subplot(gs[2, :], facecolor=BG2)
    ax4.set_facecolor(BG2)
    if gt_az is not None:
        ax4.set_title("Azimuth error distribution  (relative to ground truth)",
                      color=C_TEXT, fontsize=9)
        ax4.set_xlabel("Az error [°]", color=C_MUT, fontsize=8)
        ax4.set_ylabel("Density", color=C_MUT, fontsize=8)
        ax4.axvline(0, color="white", lw=1.0, ls="--", alpha=0.5)
        for i, (key, label, color, *_) in enumerate(ALGO_DEFS):
            az_e = results[key]["az"]
            valid = ~np.isnan(az_e)
            if valid.sum() < 5:
                continue
            err = _circ_diff(az_e[valid], gt_az)
            # KDE via histogram
            bins = np.linspace(-90, 90, 60)
            h, b = np.histogram(err, bins=bins, density=True)
            bc = 0.5 * (b[:-1] + b[1:])
            ax4.plot(bc, h + i * 0.08, "-", color=color, lw=1.4, alpha=0.85, label=label)
            ax4.fill_between(bc, i * 0.08, h + i * 0.08, color=color, alpha=0.15)
        ax4.legend(loc="upper right", fontsize=7, ncol=3,
                   facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)
        ax4.tick_params(colors=C_MUT, labelsize=7)
        for sp in ax4.spines.values(): sp.set_edgecolor(C_BDR)
    else:
        # No GT → scatter az vs el for each 2D algorithm
        ax4.set_title("DoA scatter  (az vs el, 2D algorithms, recency→colour)",
                      color=C_TEXT, fontsize=9)
        ax4.set_xlabel("Az [°]", color=C_MUT, fontsize=8)
        ax4.set_ylabel("El [°]", color=C_MUT, fontsize=8)
        ax4.set_xlim(0, 360); ax4.set_ylim(el_min, el_max)
        ax4.tick_params(colors=C_MUT, labelsize=7)
        for sp in ax4.spines.values(): sp.set_edgecolor(C_BDR)
        ax4.grid(color=C_BDR, lw=0.4, alpha=0.35)
        offsets = [-0.5, -0.25, 0, 0.25, 0.5, 0.75]
        for i, (key, label, color, supports_2d, *_) in enumerate(ALGO_DEFS):
            if not supports_2d:
                continue
            az_e = results[key]["az"]; el_e = results[key]["el"]
            valid = ~np.isnan(az_e) & ~np.isnan(el_e)
            if valid.sum() == 0:
                continue
            cols_pts = np.linspace(0.1, 1.0, valid.sum())
            cmap_i = LinearSegmentedColormap.from_list("", ["#1a1d27", color])
            ax4.scatter(az_e[valid], el_e[valid] + offsets[i] * 0.5,
                        c=cols_pts, cmap=cmap_i, s=8, alpha=0.55, label=label, zorder=3+i)
        ax4.legend(loc="upper right", fontsize=7, ncol=3,
                   facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # also axes
    for ax in [ax0, ax2]:
        ax.set_xlabel("Burst index", color=C_MUT, fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n[SAVED] {out_path}")
    print("  Open the file with your image viewer or a PDF reader.")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline DoA algorithm comparison on recorded burst data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file",  default=None,
                   help="Path to burst_data_*.npz file to analyse. "
                        "If omitted, uses the most recent file in krakenSDR/data/doa_868/.")
    p.add_argument("--demo",  action="store_true",
                   help="Use synthetic bursts instead of recorded data (needs no file).")
    p.add_argument("--gt-az", type=float, default=None, metavar="DEG",
                   help="Ground-truth azimuth [°] for error metrics.")
    p.add_argument("--gt-el", type=float, default=None, metavar="DEG",
                   help="Ground-truth elevation [°] for error metrics.")
    p.add_argument("--out",   default="./algo_compare.png",
                   help="Output file path (.png or .pdf).")
    p.add_argument("--max-bursts", type=int, default=500, metavar="N",
                   help="Limit bursts to process (speed vs accuracy trade-off).")
    args = p.parse_args()

    if args.demo:
        print("Generating synthetic benchmark dataset  (az=54°, el=44°, SNR=20 dB)...")
        cfg, R_list, gt_az, gt_el = _make_synthetic(
            n_bursts=min(args.max_bursts, 400), snr_db=20.0,
            az_true=54.0, el_true=44.0,
        )
        compare(cfg, R_list, gt_az=gt_az, gt_el=gt_el, out_path=args.out)
        return

    # Find file
    if args.file:
        fpath = args.file
    else:
        data_dir = os.path.normpath(os.path.join(_SRC, "..", "data", "doa_868"))
        candidates = sorted(glob.glob(os.path.join(data_dir, "burst_data_*.npz")))
        # Prefer files without "_iq" suffix
        candidates = [f for f in candidates if "_iq.npz" not in f]
        if not candidates:
            print(f"[ERROR] No burst_data_*.npz files found in {data_dir}")
            print("  Run the main script first to record data, or use --demo.")
            sys.exit(1)
        fpath = candidates[-1]

    print(f"Loading: {fpath}")
    cfg, R_list, rec_az, rec_el, rec_papr = _load_data(fpath)

    if args.max_bursts and len(R_list) > args.max_bursts:
        R_list  = R_list[-args.max_bursts:]
        rec_az  = rec_az[-args.max_bursts:]
        rec_el  = rec_el[-args.max_bursts:]
        rec_papr = rec_papr[-args.max_bursts:]
        print(f"  Truncated to last {args.max_bursts} bursts.")

    print(f"  Bursts: {len(R_list)}  "
          f"az=[{rec_az.min():.1f}°…{rec_az.max():.1f}°]  "
          f"el=[{rec_el.min():.1f}°…{rec_el.max():.1f}°]  "
          f"papr=[{rec_papr.min():.1f}…{rec_papr.max():.1f} dB]")

    compare(cfg, R_list,
            rec_az=rec_az, rec_el=rec_el, rec_papr=rec_papr,
            gt_az=args.gt_az, gt_el=args.gt_el,
            out_path=args.out)


if __name__ == "__main__":
    main()
