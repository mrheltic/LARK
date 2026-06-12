#!/usr/bin/env python3
"""
tilt_study.py — Does tilting the UCA improve low-elevation accuracy?

A flat UCA senses elevation only through cos(el) in the steering phase, so
its elevation sensitivity goes as sin(el) — weakest exactly near the horizon.
Tilting the array plane adds vertical baselines (r·sin tilt) whose elevation
sensitivity goes as cos(el) — strongest near the horizon.

This Monte Carlo quantifies the trade-off before any field work: synthetic
rank-1 bursts (same model as the real pipeline: steering vector + white
noise, MF covariance, 2D MUSIC + parabolic peak interpolation) at several
true elevations and tilt angles, reporting median and RMS angular errors.

Result with the defaults (5 ch, r = 0.4253λ, 22 dB SNR, June 2026): tilting
improves the *median* elevation error only at el ≈ 5° (5.0° → ~3.2° at 20°
tilt) but the broken circular symmetry creates near-ambiguous (az, el)
pairs, so outliers inflate the RMS at every elevation (e.g. 4.9° → 10.7°
at el = 5°).  For this 5-element array the tilt is not worth the field
effort — track smoothing (scripts/smooth_tracks.py) buys more at no risk.

Usage:
    python3 scripts/tilt_study.py                       # defaults, saves PNG
    python3 scripts/tilt_study.py --snr-db 15 --trials 300
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for p in (_SRC,):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.doa_uca_2d import (  # noqa: E402
    UcaConfig,
    doa_music_uca_2d,
    find_peak_uca_2d,
)

# Same geometry as the real array (doa_config.toml)
RADIUS_LAMBDA = 0.4253

# Dark theme (matches plot_track_vs_tle.py)
BG = "#1a1d27"
BG2 = "#21253a"
C_BDR = "#3b4263"
C_MUT = "#8891b0"
C_TEXT = "#d8dae8"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UCA tilt vs low-elevation accuracy")
    p.add_argument("--tilts", type=float, nargs="+",
                   default=[0.0, 5.0, 10.0, 15.0, 20.0],
                   help="Array tilt angles to test [deg]")
    p.add_argument("--els", type=float, nargs="+",
                   default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
                   help="True elevations to test [deg]")
    p.add_argument("--snr-db", type=float, default=22.0,
                   help="Per-channel SNR of the synthetic burst [dB] "
                        "(default matches typical real bursts)")
    p.add_argument("--trials", type=int, default=150,
                   help="Monte Carlo trials per (tilt, elevation) point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="tilt_study.png", help="Output PNG path")
    p.add_argument("--show", action="store_true", help="Open interactive window")
    return p.parse_args(argv)


def steering_vector(cfg: UcaConfig, az_deg: float, el_deg: float) -> np.ndarray:
    """Steering vector at an arbitrary (az, el), off the scan grid."""
    pos = cfg.positions
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    tau = 2.0 * np.pi * (
        pos[:, 0] * np.cos(el) * np.sin(az)
        + pos[:, 1] * np.cos(el) * np.cos(az)
        + pos[:, 2] * np.sin(el)
    )
    return np.exp(1j * tau)


def run_point(cfg: UcaConfig, el_true: float, snr_db: float, trials: int,
              rng: np.random.Generator) -> dict[str, float]:
    """Error stats [deg] over `trials` bursts at random azimuths."""
    amp = np.sqrt(10.0 ** (snr_db / 10.0))
    az_err = np.empty(trials)
    el_err = np.empty(trials)
    for i in range(trials):
        az_true = float(rng.uniform(0.0, 360.0))
        a = steering_vector(cfg, az_true, el_true)
        n = (rng.standard_normal(cfg.n_ant)
             + 1j * rng.standard_normal(cfg.n_ant)) / np.sqrt(2.0)
        y = amp * a + n                       # rank-1 snapshot (post-MF model)
        R = np.outer(y, y.conj())
        spec = doa_music_uca_2d(np.empty((cfg.n_ant, 0)), cfg, R_in=R)
        az_est, el_est, _papr = find_peak_uca_2d(spec, cfg)
        az_err[i] = (az_est - az_true + 180.0) % 360.0 - 180.0
        el_err[i] = el_est - el_true
    return {
        "el_med": float(np.median(np.abs(el_err))),
        "el_rms": float(np.sqrt(np.mean(el_err ** 2))),
        "az_rms": float(np.sqrt(np.mean(az_err ** 2))),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # 1° elevation grid + 3° azimuth grid: fine enough that grid quantisation
    # (~0.3° RMS after parabolic interpolation) does not mask the comparison.
    # The scan floor is 0° (not the operational 5°) so low-elevation errors
    # are measured instead of being clipped at the grid edge.
    base = dict(n_ant=5, radius_lambda=RADIUS_LAMBDA,
                n_az=120, n_el=91, el_min_deg=0.0,
                num_expected_signals=1)

    el_med = np.zeros((len(args.tilts), len(args.els)))
    el_rms = np.zeros_like(el_med)
    az_rms = np.zeros_like(el_med)
    for ti, tilt in enumerate(args.tilts):
        cfg = UcaConfig(**base, tilt_deg=tilt, tilt_az_deg=0.0)
        for ei, el_true in enumerate(args.els):
            st = run_point(cfg, el_true, args.snr_db, args.trials, rng)
            el_med[ti, ei] = st["el_med"]
            el_rms[ti, ei] = st["el_rms"]
            az_rms[ti, ei] = st["az_rms"]
        print(f"tilt {tilt:4.1f}°  el med: " +
              "  ".join(f"{v:4.1f}" for v in el_med[ti]) +
              "   el RMS: " +
              "  ".join(f"{v:4.1f}" for v in el_rms[ti]) +
              "   az RMS: " +
              "  ".join(f"{v:4.1f}" for v in az_rms[ti]))

    print(f"\n(true elevations: {args.els}, SNR {args.snr_db} dB/ch, "
          f"{args.trials} trials/point)")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=BG)
    cmap = plt.get_cmap("plasma")
    for ax, data, label in ((axes[0], el_med, "elevation median |error| [°]"),
                            (axes[1], el_rms, "elevation RMS error [°]"),
                            (axes[2], az_rms, "azimuth RMS error [°]")):
        ax.set_facecolor(BG2)
        for s in ax.spines.values():
            s.set_color(C_BDR)
        ax.tick_params(colors=C_MUT, labelsize=8)
        ax.grid(color=C_BDR, alpha=0.5, lw=0.5)
        for ti, tilt in enumerate(args.tilts):
            ax.plot(args.els, data[ti], "o-", ms=4, lw=1.6,
                    color=cmap(ti / max(len(args.tilts) - 1, 1) * 0.85),
                    label=f"tilt {tilt:.0f}°")
        ax.set_xlabel("true elevation [°]", color=C_MUT, fontsize=9)
        ax.set_ylabel(label, color=C_MUT, fontsize=9)
    axes[0].legend(fontsize=8, facecolor=BG2, edgecolor=C_BDR,
                   labelcolor=C_TEXT)
    fig.suptitle(
        f"UCA tilt study — 5 ch, r = {RADIUS_LAMBDA}λ, "
        f"SNR {args.snr_db:.0f} dB/ch, {args.trials} trials/point, 2D MUSIC",
        color=C_TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=140, facecolor=BG)
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
