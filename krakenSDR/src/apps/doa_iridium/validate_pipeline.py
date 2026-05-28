#!/usr/bin/env python3
"""
validate_pipeline.py — End-to-end pipeline validation with LibreSDR TX.

Loads a dataset collected by collect_iridium_burst_dataset.py and runs a
suite of diagnostic tests to locate errors in the DoA pipeline:

  1. Burst quality  — SNR, PAPR, CFO distribution
  2. Phase diff     — measured Δφ vs. theoretical (known TX geometry)
  3. DoA estimation — runs offline DoA, computes RMSE / bias vs. ground truth
  4. CRB ratio      — empirical σ / Cramér-Rao lower-bound
  5. Eigenvalue     — checks subspace rank per burst
  6. Channel health — per-antenna power and phase contribution

Usage:
    python3 validate_pipeline.py dataset.npz --az 0 --el 25
    python3 validate_pipeline.py dataset.npz --az 90 --el 25 --algo capon
    python3 validate_pipeline.py dataset.npz               # gt_az/gt_el from .npz if present
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.normpath(os.path.join(_HERE, "..", ".."))
_REPO = os.path.normpath(os.path.join(_SRC, "..", ".."))
for _p in (_SRC, _HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Local config & core imports ──────────────────────────────────────────────
from core.iridium_dataset_utils import load_local_config, build_uca_cfg_from_config

C = load_local_config(_HERE)

from core.doa_uca_2d import doa_music_uca_2d, doa_capon_uca_2d, doa_bartlett_uca_2d
from core.doa_uca_2d import pick_doa_peak_uca_2d, UcaConfig

# Build UcaConfig from loaded config (needed for DoA calls)
_doa_cfg, _az_grid, _el_grid = build_uca_cfg_from_config(C, UcaConfig)

# ── Matplotlib (non-interactive backend if no display) ───────────────────────
import matplotlib
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# Helpers
# =============================================================================

def _wrap180(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-180, +180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _uca_phase_theory(az_deg: float, el_deg: float, cfg) -> np.ndarray:
    """
    Return theoretical inter-element phase differences [rad] for a UCA.

    Phase of antenna k relative to antenna 0:
        Δφ_k = (2π R/λ) · [cos(φ_k)·sin(θ)·cos(φ_src) + sin(φ_k)·sin(θ)·sin(φ_src)]
    where θ = elevation, φ_src = azimuth, φ_k = physical angle of antenna k.

    Returns an array of length N_ANTENNAS with Δφ[0] = 0.
    """
    n  = int(cfg.N_ANTENNAS)
    R  = float(cfg.RADIUS_LAMBDA)          # radius in wavelengths
    ccw = bool(getattr(cfg, "ANT_CCW", False))
    off = float(getattr(cfg, "ANT0_OFFSET_DEG", 0.0))

    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    # Physical angle of antenna k around the circle [rad]
    sign = -1.0 if ccw else 1.0
    phi_k = np.deg2rad(off) + sign * 2 * np.pi * np.arange(n) / n

    dphi = 2 * np.pi * R * np.sin(el_rad) * (
        np.cos(phi_k) * np.cos(az_rad) + np.sin(phi_k) * np.sin(az_rad)
    )
    return dphi - dphi[0]   # relative to antenna 0


def _run_doa(X: np.ndarray, cfg, algo: str = "music") -> np.ndarray:
    """
    Run DoA on a multi-burst IQ window matrix X (shape: n_ant, n_samp).
    Returns spec2d (n_el, n_az).
    """
    algo = algo.lower()
    if algo == "capon":
        return doa_capon_uca_2d(X, cfg)
    if algo == "bartlett":
        return doa_bartlett_uca_2d(X, cfg)
    return doa_music_uca_2d(X, cfg)


def _compute_crb_azimuth(snr_linear: float, n_snapshots: int,
                          radius_lambda: float, el_deg: float) -> float:
    """
    Approximate Cramér-Rao Bound on azimuth estimate [deg] for a UCA.

    CRB_az ≈ (1 / (2·π·R·cos(el))²) · (1 / (SNR · N)) in rad²
    Salama 2025 §8.2.1 simplified expression.
    """
    el_rad = np.deg2rad(el_deg)
    if snr_linear <= 0 or n_snapshots <= 0:
        return float("inf")
    factor = 2 * np.pi * radius_lambda * np.cos(el_rad)
    if abs(factor) < 1e-12:
        return float("inf")
    crb_rad2 = 1.0 / (factor**2 * snr_linear * n_snapshots)
    return np.degrees(np.sqrt(crb_rad2))


# =============================================================================
# Main analysis
# =============================================================================

def analyse(dataset_path: str, gt_az: float | None, gt_el: float | None,
            algo: str, n_multi: int, save_pdf: str | None) -> None:

    print(f"\n{'='*70}")
    print(f"  LARK — Pipeline validation report")
    print(f"  Dataset : {os.path.basename(dataset_path)}")
    print(f"  Algorithm: {algo.upper()}")
    print(f"{'='*70}\n")

    # ── Load dataset ─────────────────────────────────────────────────────────
    data = np.load(dataset_path, allow_pickle=False)

    # Accept both the runner format (--save-iq) and the collector format.
    # Runner keys:    bursts, t (Unix s), snr_db, sat_cfo_hz
    # Collector keys: bursts, timestamps_ms (ms), tone_snr_db, cfo_hz
    if "bursts" not in data:
        raise SystemExit(
            "Dataset has no 'bursts' key.\n"
            "Re-collect with the runner using --save-iq, or use the collector."
        )
    bursts        = data["bursts"]                                          # (N, n_ant, n_samp)
    cfo_hz        = data["cfo_hz"] if "cfo_hz" in data else data["sat_cfo_hz"]   # (N,)
    tone_snr_db   = data["tone_snr_db"] if "tone_snr_db" in data else data["snr_db"]  # (N,)
    # timestamps: collector stores ms, runner stores Unix seconds — normalise to ms
    if "timestamps_ms" in data:
        timestamps_ms = data["timestamps_ms"]
    else:
        t_s = data["t"]
        timestamps_ms = (t_s - t_s[0]) * 1e3   # relative ms from first burst
    n_bursts, n_ant, n_samp = bursts.shape

    # Ground truth from dataset if not provided on CLI
    has_gt = False
    if gt_az is None and "gt_az_deg" in data:
        gt_az_arr  = data["gt_az_deg"]
        gt_el_arr  = data["gt_el_deg"]
        has_gt = True
        print(f"[INFO] Ground truth loaded from dataset ({data.get('gt_az_deg', np.array([])).shape[0]} labels)")
    elif gt_az is not None and gt_el is not None:
        gt_az_arr  = np.full(n_bursts, float(gt_az))
        gt_el_arr  = np.full(n_bursts, float(gt_el))
        has_gt = True
        print(f"[INFO] Ground truth from CLI: AZ={gt_az:.1f}° EL={gt_el:.1f}°")
    else:
        gt_az_arr = gt_el_arr = None
        print("[WARN] No ground truth available — skipping DoA error analysis.")

    print(f"[INFO] {n_bursts} bursts, {n_ant} antennas, {n_samp} samples/burst\n")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 1 — Burst quality
    # ──────────────────────────────────────────────────────────────────────────
    print("── TEST 1: Burst quality ──────────────────────────────────────────")
    snr_mean = float(np.mean(tone_snr_db))
    snr_std  = float(np.std(tone_snr_db))
    cfo_mean = float(np.mean(cfo_hz))
    cfo_std  = float(np.std(cfo_hz))
    low_snr  = int(np.sum(tone_snr_db < 6.0))

    print(f"  Tone SNR  : mean={snr_mean:+6.1f} dB  std={snr_std:.1f} dB")
    print(f"  CFO       : mean={cfo_mean:+8.1f} Hz  std={cfo_std:.1f} Hz")
    print(f"  Low-SNR (<6 dB): {low_snr}/{n_bursts} ({100*low_snr/n_bursts:.1f}%)")
    if snr_mean < 8.0:
        print("  [WARN] Low mean SNR — consider increasing TX gain by 5 dB")
    if cfo_std > 5000.0:
        print(f"  [WARN] High CFO spread (std={cfo_std:.0f} Hz) — possible multiple satellites or TX drift")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 2 — Phase difference: measured vs. theoretical
    # ──────────────────────────────────────────────────────────────────────────
    print("── TEST 2: Phase difference — measured vs theoretical ────────────")
    # Compute per-burst phase difference of each antenna relative to antenna 0
    # using the preamble IQ (last quarter of burst window, dominated by preamble tone)
    q = n_samp // 4
    # Cross-correlate: phase(ant_k) - phase(ant_0) = angle(X_k · X_0*)
    dphi_meas = np.zeros((n_bursts, n_ant))
    for b in range(n_bursts):
        ref = bursts[b, 0, -q:]
        for k in range(n_ant):
            dphi_meas[b, k] = np.angle(np.dot(bursts[b, k, -q:], ref.conj()))
    dphi_meas -= dphi_meas[:, 0:1]   # relative to ant 0

    # Calibration offsets applied by the runner
    cal_offsets = np.deg2rad(np.array(
        getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * n_ant), dtype=float
    ))

    dphi_corrected = dphi_meas - cal_offsets[np.newaxis, :]

    if has_gt:
        # Use per-burst GT if available, else the constant passed on CLI
        gt_az_val = float(np.mean(gt_az_arr)) if has_gt else float(gt_az)
        gt_el_val = float(np.mean(gt_el_arr)) if has_gt else float(gt_el)
        dphi_theory = _uca_phase_theory(gt_az_val, gt_el_val, C)
        print(f"  Theory (AZ={gt_az_val:.1f}°, EL={gt_el_val:.1f}°):")
        print(f"  {'Ant':>4}  {'Theory [°]':>12}  {'Measured [°]':>13}  {'Residual [°]':>13}  {'std [°]':>8}")
        for k in range(n_ant):
            m   = np.degrees(np.mean(_wrap180(dphi_corrected[:, k] - dphi_theory[k])))
            s   = np.degrees(np.std(dphi_corrected[:, k]))
            th  = np.degrees(dphi_theory[k])
            mm  = np.degrees(np.mean(dphi_corrected[:, k]))
            print(f"  {k:>4}  {th:>+12.1f}  {mm:>+13.1f}  {m:>+13.1f}  {s:>8.1f}")
        print()
        residual_rms = np.degrees(np.sqrt(np.mean(
            _wrap180(dphi_corrected[:, 1:] - dphi_theory[np.newaxis, 1:])**2
        )))
        print(f"  Phase residual RMS: {residual_rms:.1f}°")
        if residual_rms > 15.0:
            print("  [WARN] Large phase residual — calibration may be inaccurate or multipath present.")
        elif residual_rms < 5.0:
            print("  [OK]   Phase residual within calibration tolerance.")
    else:
        print("  (No GT — showing measured mean phase per antenna)")
        for k in range(n_ant):
            mm = np.degrees(np.mean(dphi_corrected[:, k]))
            ss = np.degrees(np.std(dphi_corrected[:, k]))
            print(f"    Ant {k}: mean={mm:+.1f}°  std={ss:.1f}°")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 3 — DoA estimation: RMSE and bias
    # ──────────────────────────────────────────────────────────────────────────
    print("── TEST 3: Offline DoA estimation ────────────────────────────────")
    az_est   = []
    el_est   = []
    papr_est = []

    for i in range(n_bursts):
        X = bursts[i]                              # (n_ant, n_samp)
        if i < n_multi - 1:
            continue                               # not enough bursts yet
        try:
            spec2d = _run_doa(X, _doa_cfg, algo)
            az_i, el_i, papr_i = pick_doa_peak_uca_2d(spec2d, _doa_cfg)
            az_est.append(az_i)
            el_est.append(el_i)
            papr_est.append(papr_i)
        except Exception as exc:
            az_est.append(float("nan"))
            el_est.append(float("nan"))
            papr_est.append(float("nan"))
            print(f"  [WARN] burst {i}: DoA failed — {exc}")

    az_est   = np.array(az_est)
    el_est   = np.array(el_est)
    papr_est = np.array(papr_est)

    valid = np.isfinite(az_est) & np.isfinite(el_est)
    print(f"  Valid estimates: {int(np.sum(valid))}/{len(az_est)}")

    if has_gt and np.any(valid):
        # Align GT to estimation range (after n_multi warm-up)
        gt_az_slice = gt_az_arr[n_multi - 1:] if len(gt_az_arr) > 1 else np.full(len(az_est), gt_az_arr[0])
        gt_el_slice = gt_el_arr[n_multi - 1:] if len(gt_el_arr) > 1 else np.full(len(el_est), gt_el_arr[0])
        # Pad or truncate to same length
        minlen = min(len(az_est), len(gt_az_slice))
        az_err = _wrap180(az_est[:minlen][valid[:minlen]] - gt_az_slice[:minlen][valid[:minlen]])
        el_err = el_est[:minlen][valid[:minlen]] - gt_el_slice[:minlen][valid[:minlen]]

        az_bias = float(np.mean(az_err))
        el_bias = float(np.mean(el_err))
        az_rmse = float(np.sqrt(np.mean(az_err**2)))
        el_rmse = float(np.sqrt(np.mean(el_err**2)))
        az_std  = float(np.std(az_err))
        el_std  = float(np.std(el_err))

        print(f"\n  AZ  bias={az_bias:+6.1f}°  RMSE={az_rmse:.1f}°  std={az_std:.1f}°")
        print(f"  EL  bias={el_bias:+6.1f}°  RMSE={el_rmse:.1f}°  std={el_std:.1f}°")

        # Interpret
        if abs(az_bias) > 10.0:
            print(f"\n  [WARN] Large AZ bias ({az_bias:+.1f}°) — check ANT0_OFFSET_DEG or DOA_AZ_OFFSET_DEG in config.py")
        if abs(el_bias) > 10.0:
            print(f"  [WARN] Large EL bias ({el_bias:+.1f}°) — check DOA_EL_OFFSET_DEG in config.py")
        if az_rmse < 5.0:
            print("  [OK]   AZ RMSE within target (< 5°)")
        elif az_rmse < 10.0:
            print("  [INFO] AZ RMSE 5–10° — acceptable for indoor multipath conditions")
        else:
            print(f"  [WARN] AZ RMSE={az_rmse:.1f}° is high — check calibration / SNR / multipath")

        # ── TEST 4 — CRB ratio ───────────────────────────────────────────────
        print("\n── TEST 4: Cramér-Rao Bound ratio ────────────────────────────────")
        snr_lin  = 10 ** (snr_mean / 10.0)
        crb_az   = _compute_crb_azimuth(snr_lin, n_samp, C.RADIUS_LAMBDA,
                                         float(np.mean(gt_el_arr)))
        crb_ratio = az_std / (crb_az + 1e-12)
        crb_computed = True
        print(f"  CRB (AZ)  : {crb_az:.2f}°")
        print(f"  Empirical σ(AZ): {az_std:.2f}°")
        print(f"  CRB ratio  = σ/CRB = {crb_ratio:.1f}x")
        if crb_ratio < 2.0:
            print("  [OK]   Close to CRB — algorithm is near-optimal.")
        elif crb_ratio < 5.0:
            print("  [INFO] Moderate gap to CRB — likely indoor multipath or mild calibration error.")
        else:
            print(f"  [WARN] Large gap to CRB ({crb_ratio:.1f}x) — investigate: calibration, multipath, or SNR.")
    else:
        az_bias = el_bias = az_rmse = el_rmse = az_std = el_std = float("nan")
        crb_computed = False
        print("  (GT not available — RMSE/bias not computed)")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 5 — Eigenvalue profile (subspace rank check)
    # ──────────────────────────────────────────────────────────────────────────
    print("── TEST 5: Eigenvalue profile (subspace rank) ────────────────────")
    n_check = min(200, n_bursts)
    rank1_count = 0
    rank2_count = 0
    rankN_count = 0
    eigval_matrix = []
    for i in range(n_check):
        R = bursts[i] @ bursts[i].conj().T / n_samp
        eigs = np.sort(np.linalg.eigvalsh(R).real)[::-1]
        eigval_matrix.append(eigs / eigs[0])   # normalised
        # MDL-like rank: count eigenvalues > 10% of max
        rank = int(np.sum(eigs > 0.1 * eigs[0]))
        if rank == 1:
            rank1_count += 1
        elif rank == 2:
            rank2_count += 1
        else:
            rankN_count += 1

    eigval_matrix = np.array(eigval_matrix)
    print(f"  (Checked first {n_check} bursts)")
    print(f"  Rank=1 (expected): {rank1_count}/{n_check} ({100*rank1_count/n_check:.0f}%)")
    print(f"  Rank=2           : {rank2_count}/{n_check} ({100*rank2_count/n_check:.0f}%)")
    print(f"  Rank≥3           : {rankN_count}/{n_check} ({100*rankN_count/n_check:.0f}%)")
    if rank1_count / n_check < 0.5:
        print("  [WARN] Less than 50% rank-1 — likely multipath or a second signal source.")
    else:
        print("  [OK]   Dominant rank-1 subspace — good coherence.")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 6 — Per-channel power balance (health check)
    # ──────────────────────────────────────────────────────────────────────────
    print("── TEST 6: Per-channel power balance ─────────────────────────────")
    pwr = np.mean(np.abs(bursts)**2, axis=(0, 2))   # (n_ant,) mean power
    pwr_db = 10 * np.log10(pwr / pwr[0] + 1e-30)
    print(f"  Power relative to channel 0 [dB]:")
    for k in range(n_ant):
        bar = "█" * int(max(0, 20 + pwr_db[k]))
        flag = " [WARN] large imbalance — check cable/connector" if abs(pwr_db[k]) > 6.0 else ""
        print(f"    CH{k}: {pwr_db[k]:+5.1f} dB  {bar}{flag}")
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#1a1a2e")
    C_TEXT = "#e0e0e0"
    C_GRID = "#333355"
    fig.suptitle(
        f"LARK — Pipeline Validation  |  {os.path.basename(dataset_path)}  |  {algo.upper()}",
        color=C_TEXT, fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    def _ax(row, col, rowspan=1, colspan=1):
        ax = fig.add_subplot(gs[row:row + rowspan, col:col + colspan])
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values():
            sp.set_color(C_GRID)
        ax.tick_params(colors=C_TEXT, labelsize=8)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        return ax

    # 1. Tone SNR histogram
    ax = _ax(0, 0)
    ax.hist(tone_snr_db, bins=30, color="#4fc3f7", edgecolor="none", alpha=0.85)
    ax.axvline(6.0, color="#ff7043", lw=1.5, linestyle="--", label="min gate 6 dB")
    ax.set_xlabel("Tone SNR [dB]"); ax.set_ylabel("Count")
    ax.set_title("Burst Tone SNR", color=C_TEXT, fontsize=9)
    ax.legend(fontsize=7, labelcolor=C_TEXT, framealpha=0.2)

    # 2. CFO distribution
    ax = _ax(0, 1)
    ax.hist(cfo_hz / 1e3, bins=30, color="#80cbc4", edgecolor="none", alpha=0.85)
    ax.set_xlabel("CFO [kHz]"); ax.set_ylabel("Count")
    ax.set_title("CFO Distribution", color=C_TEXT, fontsize=9)

    # 3. Phase diff per antenna (corrected), overlay theory
    ax = _ax(0, 2)
    colors = ["#ef5350", "#ab47bc", "#42a5f5", "#66bb6a"]
    for k in range(1, n_ant):
        ax.plot(np.degrees(dphi_corrected[:, k]), color=colors[k - 1],
                alpha=0.4, lw=0.6, label=f"CH{k}")
        if has_gt:
            ax.axhline(np.degrees(dphi_theory[k]), color=colors[k - 1],
                       lw=1.5, linestyle="--")
    ax.set_xlabel("Burst index"); ax.set_ylabel("Δφ [°]")
    ax.set_title("Phase diff (solid=meas / dashed=theory)", color=C_TEXT, fontsize=9)
    ax.legend(fontsize=7, labelcolor=C_TEXT, framealpha=0.2)

    # 4. AZ estimate over time
    ax = _ax(1, 0, colspan=2)
    t_s = (timestamps_ms[n_multi - 1:] - timestamps_ms[0]) / 1e3
    minlen2 = min(len(t_s), len(az_est))
    ax.plot(t_s[:minlen2], az_est[:minlen2], color="#4fc3f7", lw=1.0, label="AZ est")
    if has_gt and np.any(valid):
        ax.plot(t_s[:minlen2], gt_az_slice[:minlen2][valid[:minlen2]], color="#ff7043",
                lw=1.0, linestyle="--", label="AZ GT")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Azimuth [°]")
    ax.set_title("Azimuth estimate vs time", color=C_TEXT, fontsize=9)
    ax.legend(fontsize=7, labelcolor=C_TEXT, framealpha=0.2)

    # 5. EL estimate over time
    ax = _ax(1, 2)
    ax.plot(t_s[:minlen2], el_est[:minlen2], color="#80cbc4", lw=1.0)
    if has_gt and np.any(valid):
        ax.axhline(float(np.mean(gt_el_slice[:minlen2])), color="#ff7043", lw=1.5,
                   linestyle="--", label="EL GT")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Elevation [°]")
    ax.set_title("Elevation estimate vs time", color=C_TEXT, fontsize=9)

    # 6. AZ error histogram
    ax = _ax(2, 0)
    if has_gt and np.any(valid):
        ax.hist(az_err, bins=20, color="#ffca28", edgecolor="none", alpha=0.85)
        ax.axvline(az_bias, color="#ff7043", lw=1.5, linestyle="--",
                   label=f"bias={az_bias:+.1f}°")
        ax.set_title(f"AZ error  RMSE={az_rmse:.1f}°", color=C_TEXT, fontsize=9)
        ax.legend(fontsize=7, labelcolor=C_TEXT, framealpha=0.2)
    else:
        ax.text(0.5, 0.5, "No GT", transform=ax.transAxes,
                ha="center", va="center", color=C_TEXT)
        ax.set_title("AZ error", color=C_TEXT, fontsize=9)
    ax.set_xlabel("Error [°]"); ax.set_ylabel("Count")

    # 7. Eigenvalue profiles (box)
    ax = _ax(2, 1)
    ax.boxplot(eigval_matrix, notch=False, patch_artist=True,
               boxprops=dict(facecolor="#1e3a5f", color=C_GRID),
               medianprops=dict(color="#4fc3f7"),
               whiskerprops=dict(color=C_GRID),
               capprops=dict(color=C_GRID),
               flierprops=dict(marker=".", color=C_GRID, markersize=2))
    ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("Normalised value")
    ax.set_title("Eigenvalue profile (normalised)", color=C_TEXT, fontsize=9)

    # 8. Per-channel power bar
    ax = _ax(2, 2)
    ax.bar(range(n_ant), pwr_db, color="#66bb6a", edgecolor="none", alpha=0.85)
    ax.axhline(0, color=C_TEXT, lw=0.8, linestyle="--")
    ax.axhline(6,  color="#ff7043", lw=0.8, linestyle=":")
    ax.axhline(-6, color="#ff7043", lw=0.8, linestyle=":")
    ax.set_xlabel("Channel"); ax.set_ylabel("Relative power [dB]")
    ax.set_title("Per-channel power balance", color=C_TEXT, fontsize=9)
    ax.set_xticks(range(n_ant))

    if save_pdf:
        fig.savefig(save_pdf, dpi=150, bbox_inches="tight")
        print(f"[INFO] Report saved to {save_pdf}")
    else:
        plt.show()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("── SUMMARY ───────────────────────────────────────────────────────")
    print(f"  Bursts collected   : {n_bursts}")
    print(f"  Mean tone SNR      : {snr_mean:.1f} dB")
    print(f"  CFO std            : {cfo_std:.0f} Hz")
    if has_gt and np.any(valid):
        print(f"  AZ RMSE / bias     : {az_rmse:.1f}° / {az_bias:+.1f}°")
        print(f"  EL RMSE / bias     : {el_rmse:.1f}° / {el_bias:+.1f}°")
        if crb_computed:
            print(f"  CRB ratio (σ/CRB)  : {crb_ratio:.1f}x")
    print(f"  Rank-1 subspace    : {100*rank1_count/n_check:.0f}%")
    print(f"  CH power balance   : max imbalance {float(np.max(np.abs(pwr_db))):.1f} dB")
    print("═" * 70)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="LARK pipeline validation tool — analyses a burst dataset "
                    "against a known TX position.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dataset", help="Path to .npz dataset from collect_iridium_burst_dataset.py")
    p.add_argument("--az",   type=float, default=None,
                   help="Known TX azimuth [deg, geographic North = 0]")
    p.add_argument("--el",   type=float, default=None,
                   help="Known TX elevation [deg]")
    p.add_argument("--algo", default="music",
                   choices=["music", "capon", "bartlett"],
                   help="DoA algorithm to use for offline estimation")
    p.add_argument("--multi", type=int, default=4,
                   help="Burst averaging window size (same as --multi in runner)")
    p.add_argument("--save-pdf", metavar="FILE", default=None,
                   help="Save report plot to this PDF/PNG file instead of showing it")
    args = p.parse_args()

    if not os.path.isfile(args.dataset):
        p.error(f"Dataset not found: {args.dataset}")

    analyse(args.dataset, args.az, args.el, args.algo, args.multi, args.save_pdf)


if __name__ == "__main__":
    main()
