#!/usr/bin/env python3
"""
diagnose_npz.py — Offline diagnostic for Iridium DoA recordings.

Run:
    python3 diagnose_npz.py path/to/doa_iridium_*.npz

Checks:
  1. Per-burst rank-1 quality (eig_db from R_inst)
  2. Multi-burst stack corruption (eigenvalues of saved R_avg)
  3. Phase stability vs burst index
  4. MDL source-count histogram
  5. PAPR / SNR statistics
  6. Doppler drift rate

Exit codes:
  0 = healthy   1 = suspicious   2 = corrupted stack
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _wrap180(a: np.ndarray) -> np.ndarray:
    return ((a + 180.0) % 360.0) - 180.0


def diagnose(path: str) -> int:
    data = np.load(path)
    n = len(data["az_deg"])

    # ── 1. Per-burst rank-1 quality (from eig_db = eigenvalue_spread_uca_db(R_inst))
    rank1_db = data["eig_db"][:, 0]          # λ₁ vs noise floor [dB]
    rank1_ok = np.sum(rank1_db > 100.0)     # R_inst is rank-1 if λ₁ ≫ noise

    # ── 2. Multi-burst stack corruption (eigenvalues of saved R_avg)
    ev_flat = 0
    ev_spread_ok = 0
    for i in range(min(n, 20)):
        R = data["R"][i]
        ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
        ratio = ev[0] / (ev[1] + 1e-30)
        if ratio < 2.0:
            ev_flat += 1
        if ratio > 3.0:
            ev_spread_ok += 1

    # ── 3. Phase stability (saved phase_diffs from R_avg)
    phase_std = [float(data["phase_diff"][:, ch].std()) for ch in range(4)]

    # ── 4. MDL histogram
    mdl_vals, mdl_counts = np.unique(data["mdl_k"], return_counts=True)

    # ── 5. PAPR / SNR
    papr_mean = float(data["papr_db"].mean())
    snr_mean = float(data["snr_db"].mean())

    # ── 6. CFO drift
    cfo = data["sat_cfo_hz"]
    cfo_range = float(cfo.max() - cfo.min())
    cfo_drift_per_burst = cfo_range / max(n - 1, 1)

    print(f"File: {path}")
    print(f"Bursts: {n}")
    print()
    print("━" * 50)
    print("1. SINGLE-BURST R_inst  (from eig_db)")
    print(f"   rank-1 quality λ₁>100 dB: {rank1_ok}/{n} ({100*rank1_ok/n:.0f}%)")
    print(f"   → If high: raw bursts are clean. Problem is in the stack.")
    print()
    print("2. MULTI-BURST R_avg    (saved R matrix eigenvalues)")
    print(f"   Flat λ₁/λ₂ < 2.0 (first 20): {ev_flat}/20")
    print(f"   Spread λ₁/λ₂ > 3.0 (first 20): {ev_spread_ok}/20")
    if ev_flat > 10:
        print("   ⚠  CORRUPTED STACK: even short stacks lose rank-1 structure.")
        print("      → Set MULTI_BURST_N = 1 for indoor multipath.")
    elif ev_spread_ok > 15:
        print("   ✓  Stack preserves dominant direction.")
    else:
        print("   ⚠  Marginal: some stacks are corrupted.")
    print()
    print("3. PHASE DIFF STABILITY (saved phase_diff from R_avg)")
    for ch, std in enumerate(phase_std):
        status = "✓" if std < 15 else "⚠" if std < 40 else "✗"
        print(f"   CH{ch+1}: std = {std:5.1f}°  {status}")
    if max(phase_std) > 40:
        print("   → High phase jitter confirms stack corruption or strong multipath.")
    print()
    print("4. MDL SOURCE COUNT")
    for k, c in zip(mdl_vals, mdl_counts):
        pct = 100.0 * c / n
        bar = "█" * int(pct / 5)
        print(f"   K={k}: {c:4d}/{n} ({pct:5.1f}%) {bar}")
    if np.sum(mdl_counts[mdl_vals > 1]) > 0.5 * n:
        print("   ⚠  MDL sees >1 source: multipath or stack corruption.")
    print()
    print("5. PAPR / SNR")
    print(f"   PAPR mean: {papr_mean:.1f} dB")
    print(f"   SNR  mean: {snr_mean:.1f} dB")
    if papr_mean < 5.0:
        print("   ⚠  Low PAPR → MUSIC peak is weak (flat spectrum).")
    print()
    print("6. DOPPLER DRIFT")
    print(f"   CFO range: {cfo_range:.0f} Hz over {n} bursts")
    print(f"   Drift rate: {cfo_drift_per_burst:.2f} Hz/burst")
    if cfo_range > 1000:
        print("   ⚠  Large CFO range → Doppler alignment IS needed (outdoor/pass mode).")
    else:
        print("   ✓  Small CFO drift → Doppler alignment barely triggers (normal for indoor IRA).")
    print()

    # Overall verdict
    if ev_flat > 10 or max(phase_std) > 60:
        print("VERDICT: 2 (CORRUPTED STACK)")
        print("  Fix: set MULTI_BURST_N = 1 in config.py for indoor_ira.")
        return 2
    elif ev_flat > 5 or max(phase_std) > 30:
        print("VERDICT: 1 (SUSPICIOUS)")
        print("  Consider reducing MULTI_BURST_N or checking calibration.")
        return 1
    else:
        print("VERDICT: 0 (HEALTHY)")
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Offline DoA npz diagnostic")
    p.add_argument("npz", nargs="+", help="Path to one or more .npz recordings")
    args = p.parse_args()

    worst = 0
    for path in args.npz:
        rc = diagnose(path)
        worst = max(worst, rc)
        print()
    sys.exit(worst)


if __name__ == "__main__":
    main()
