#!/usr/bin/env python3
"""
run_doa_offline.py — Offline Iridium burst DoA from captured data.

Processes pre-recorded .npz or .cf32 files through the LARK DOA pipeline
without any GNU Radio dependency.

Input formats:
  .npz: file with arrays 'ch0'..'ch4' (or 'ant0'..'ant4'), each 1D complex64
  .cf32: 5 separate files, raw interleaved I/Q (complex64) binary

Usage:
    python3 run_doa_offline.py capture.npz [options]
    python3 run_doa_offline.py --files ch0.cf32 ch1.cf32 ch2.cf32 ch3.cf32 ch4.cf32 [options]
    python3 run_doa_offline.py capture.npz --gui
    python3 run_doa_offline.py capture.npz --save-spectrum spec_output.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.doa_uca_2d import (
    UcaConfig, doa_music_uca_2d, doa_bartlett_uca_2d, doa_capon_uca_2d,
    doa_phase_fit_uca_2d, find_peak_uca_2d, amplitude_normalize_channels,
    eigenvalue_spread_uca_db, snr_uca_db,
)
from core.doa_algorithms import apply_phase_correction
from apps.doa_iridium.burst_processing import (
    detect_energy_bursts, scan_preamble_tones,
    compute_mf_covariance, apply_bpf_and_normalize,
)

_ALGO_MAP = {
    "MUSIC": doa_music_uca_2d,
    "BARTLETT": doa_bartlett_uca_2d,
    "CAPON": doa_capon_uca_2d,
    "PHASE-FIT": None,
}

_MODE_PROFILES = {
    "indoor": dict(
        tone_nom_hz=3125.0, scan_bw_hz=3_000.0, bpf_bw_hz=8_000.0,
        dc_guard_hz=200.0, min_snr_db=2.0, min_sep_hz=1_000.0,
        energy_threshold=3.0, prefer_nom=False,
    ),
    "outdoor": dict(
        tone_nom_hz=3125.0, scan_bw_hz=45_000.0, bpf_bw_hz=15_000.0,
        dc_guard_hz=500.0, min_snr_db=3.0, min_sep_hz=5_000.0,
        energy_threshold=2.0, prefer_nom=False,
    ),
    "indoor_tx": dict(
        tone_nom_hz=0.0, scan_bw_hz=5_000.0, bpf_bw_hz=4_000.0,
        dc_guard_hz=0.0, min_snr_db=2.0, min_sep_hz=500.0,
        energy_threshold=3.0, prefer_nom=True,
    ),
}


def load_npz(path, n_ant=5):
    data = np.load(path)
    arrays = []
    for i in range(n_ant):
        for key in [f"ch{i}", f"ant{i}"]:
            if key in data:
                arr = data[key]
                if arr.dtype != np.complex64:
                    arr = arr.astype(np.complex64)
                arrays.append(arr.flatten())
                break
        else:
            raise ValueError(f"Missing channel {i} in {path} (tried ch{i}, ant{i})")
    return np.stack(arrays)


def load_cf32_files(paths, n_ant=5):
    arrays = []
    for p in paths:
        arr = np.fromfile(p, dtype=np.complex64)
        arrays.append(arr)
    min_len = min(len(a) for a in arrays)
    return np.stack([a[:min_len] for a in arrays[:n_ant]])


def process_frame(X, cfg, phase_offs, profile, fs, pre_samples, bpf_guard,
                  window_samples, cov_alpha, az_alpha, el_alpha,
                  snr_min, papr_min, algo, state):
    n_ant = X.shape[0]

    ch0 = X[0, :]
    burst_starts = detect_energy_bursts(
        ch0, fs, energy_window=256,
        threshold_factor=profile["energy_threshold"],
        min_gap_samples=int(0.045 * fs),
    )
    if not burst_starts:
        return None

    burst_start = burst_starts[0]
    burst_end = min(burst_start + window_samples, X.shape[1])
    if burst_end - burst_start < pre_samples + bpf_guard:
        return None

    tones = scan_preamble_tones(
        ch0[burst_start:burst_end], fs, profile["tone_nom_hz"],
        scan_bw_hz=profile["scan_bw_hz"], n_peaks=3,
        min_sep_hz=profile["min_sep_hz"],
        min_snr_db=profile["min_snr_db"],
        dc_guard_hz=profile["dc_guard_hz"],
        prefer_nom=profile.get("prefer_nom", False),
    )
    if not tones:
        return None

    tone_hz = tones[0][0]
    X_win = X[:, burst_start:burst_end]
    if X_win.shape[1] < window_samples:
        return None

    try:
        X_bpf = apply_bpf_and_normalize(
            X_win, window_samples, fs, tone_hz, profile["bpf_bw_hz"]
        )
    except ValueError:
        return None

    X_cal = apply_phase_correction(X_bpf, phase_offs)

    try:
        R_mf, y_mf, snr_db = compute_mf_covariance(
            X_cal, tone_hz, fs, pre_samples, bpf_guard
        )
    except ValueError:
        return None

    if snr_db < snr_min:
        return None

    if state["R_ema"] is None:
        state["R_ema"] = R_mf.copy()
    else:
        state["R_ema"] = cov_alpha * state["R_ema"] + (1 - cov_alpha) * R_mf

    algo_fn = _ALGO_MAP.get(algo)
    if algo == "PHASE-FIT":
        az_est, el_est, _ = doa_phase_fit_uca_2d(
            state["R_ema"], cfg,
            az_hint_deg=state.get("az_ema"),
            el_hint_deg=state.get("el_ema"),
        )
        papr = 0.0
        spec = None
    elif algo_fn is not None:
        if algo == "CAPON":
            spec = algo_fn(X_cal, cfg, R_in=state["R_ema"], decorr="none")
        else:
            spec = algo_fn(X_cal, cfg, R_in=state["R_ema"])
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        if papr < papr_min:
            return None
    else:
        spec = doa_music_uca_2d(X_cal, cfg, R_in=state["R_ema"])
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        if papr < papr_min:
            return None

    if state.get("az_ema") is None:
        state["az_ema"] = az_est
        state["el_ema"] = el_est
    else:
        d_az = ((az_est - state["az_ema"] + 180) % 360) - 180
        state["az_ema"] += az_alpha * d_az
        state["az_ema"] %= 360.0
        state["el_ema"] += el_alpha * (el_est - state["el_ema"])

    state["n_bursts"] += 1

    return dict(
        az=state["az_ema"], el=state["el_ema"],
        snr=snr_db, papr=papr,
        tone=tone_hz, doppler=tone_hz - profile["tone_nom_hz"],
        spec=spec, R=state["R_ema"],
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline Iridium burst DoA from captured data")
    p.add_argument("input", nargs="?", help="Input .npz file")
    p.add_argument("--files", nargs=5, metavar="CF32",
                   help="5 separate .cf32 channel files")
    p.add_argument("--freq", type=float, default=1626.27e6,
                   help="Center frequency [Hz]")
    p.add_argument("--fs", type=float, default=1_024_000.0,
                   help="Effective sample rate [Hz]")
    p.add_argument("--n-ant", type=int, default=5)
    p.add_argument("--radius", type=float, default=0.4253)
    p.add_argument("--ant0-offset", type=float, default=0.0)
    p.add_argument("--ccw", action="store_true")
    p.add_argument("--algo", type=str, default="MUSIC",
                   choices=["MUSIC", "CAPON", "BARTLETT", "PHASE-FIT"])
    p.add_argument("--mode", type=str, default="indoor",
                   choices=["indoor", "outdoor", "indoor_tx"])
    p.add_argument("--cpi-size", type=int, default=131072,
                   help="Frame size to process [samples]")
    p.add_argument("--n-frames", type=int, default=0,
                   help="Max frames to process (0=all)")
    p.add_argument("--n-az", type=int, default=360)
    p.add_argument("--n-el", type=int, default=86)
    p.add_argument("--el-min", type=float, default=5.0)
    p.add_argument("--el-max", type=float, default=90.0)
    p.add_argument("--phase-offs", type=str,
                   default="0.0,54.95,137.24,133.58,48.31")
    p.add_argument("--pre-samples", type=int, default=2621)
    p.add_argument("--bpf-guard", type=int, default=128)
    p.add_argument("--window-samples", type=int, default=3000)
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--snr-min", type=float, default=-3.0)
    p.add_argument("--papr-min", type=float, default=3.0)
    p.add_argument("--cov-alpha", type=float, default=0.93)
    p.add_argument("--az-alpha", type=float, default=0.88)
    p.add_argument("--el-alpha", type=float, default=0.65)
    p.add_argument("--tone-nom", type=float, default=None)
    p.add_argument("--scan-bw", type=float, default=None)
    p.add_argument("--bpf-bw", type=float, default=None)
    p.add_argument("--gui", action="store_true", help="Show matplotlib display")
    p.add_argument("--save-spectrum", type=str, default=None,
                   help="Save DOA spectra to .npz")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.input and not args.files:
        print("Error: provide input .npz file or --files ch0.cf32 ... ch4.cf32")
        sys.exit(1)

    profile = dict(_MODE_PROFILES[args.mode])
    if args.tone_nom is not None:
        profile["tone_nom_hz"] = args.tone_nom
    if args.scan_bw is not None:
        profile["scan_bw_hz"] = args.scan_bw
    if args.bpf_bw is not None:
        profile["bpf_bw_hz"] = args.bpf_bw

    phase_offs = [float(x) for x in args.phase_offs.split(",")]

    cfg = UcaConfig(
        n_ant=args.n_ant, radius_lambda=args.radius,
        n_az=args.n_az, n_el=args.n_el,
        el_min_deg=args.el_min, el_max_deg=args.el_max,
        num_expected_signals=1,
        ant0_offset_deg=args.ant0_offset, ant_ccw=args.ccw,
    )

    print(f"\n{'='*60}")
    print(f"  Iridium DoA Offline — {args.algo} on {args.n_ant}-element UCA")
    print(f"  Freq: {args.freq/1e6:.3f} MHz | fs: {args.fs/1e0:.0f} Hz")
    print(f"  Mode: {args.mode} | CPI: {args.cpi_size} | Window: {args.window_samples}")
    print(f"  Phase offsets: {phase_offs}")
    if args.input:
        print(f"  Input: {args.input}")
    else:
        print(f"  Input: {args.files}")
    print(f"{'='*60}\n")

    if args.input:
        print(f"Loading {args.input}...")
        X_full = load_npz(args.input, args.n_ant)
    else:
        print(f"Loading {args.files}...")
        X_full = load_cf32_files(args.files, args.n_ant)

    print(f"  Shape: {X_full.shape} ({X_full.shape[1]} samples, {X_full.shape[1]/args.fs:.2f}s)")

    n_total_frames = X_full.shape[1] // args.cpi_size
    if args.n_frames > 0:
        n_total_frames = min(n_total_frames, args.n_frames)

    print(f"  Processing {n_total_frames} frames of {args.cpi_size} samples each\n")

    state = dict(R_ema=None, az_ema=None, el_ema=None, n_bursts=0)
    results = []
    spectra = []

    if args.gui:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        az_grid = cfg.az_range_deg()
        el_grid = cfg.el_range_deg()
        az_hist, el_hist = [], []
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Iridium DoA Offline — {args.algo}", fontsize=14)

    for fi in range(n_total_frames):
        start = fi * args.cpi_size
        end = start + args.cpi_size
        X = X_full[:, start:end]

        result = process_frame(
            X, cfg, phase_offs, profile, args.fs,
            args.pre_samples, args.bpf_guard, args.window_samples,
            args.cov_alpha, args.az_alpha, args.el_alpha,
            args.snr_min, args.papr_min, args.algo, state,
        )

        if result is None:
            if args.verbose:
                print(f"  [{fi:4d}] no burst detected")
            continue

        n = state["n_bursts"]
        print(f"[{n:4d}] az={result['az']:6.1f}deg  el={result['el']:5.1f}deg  "
              f"snr={result['snr']:5.1f}dB  papr={result['papr']:5.1f}dB  "
              f"tone={result['tone']:.0f}Hz  doppler={result['doppler']:+.0f}Hz")

        results.append(result)
        if result.get("spec") is not None:
            spectra.append(result["spec"])

        if args.gui and result.get("spec") is not None:
            spec = result["spec"]
            az = result["az"]
            el = result["el"]

            ax_spec, ax_az, ax_track, ax_snr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

            ax_spec.clear()
            ax_spec.set_title(f"DOA Spectrum — az={az:.1f}° el={el:.1f}°")
            ax_spec.imshow(spec, aspect="auto", origin="lower",
                           extent=[az_grid[0], az_grid[-1], el_grid[0], el_grid[-1]],
                           cmap="hot", vmin=-40, vmax=0)
            ax_spec.plot(az, el, "g+", markersize=15, markeredgewidth=2)
            ax_spec.set_xlabel("Azimuth [°]")
            ax_spec.set_ylabel("Elevation [°]")

            ax_az.clear()
            el_idx = np.argmin(np.abs(el_grid - el))
            ax_az.plot(az_grid, spec[el_idx, :], "b-")
            ax_az.set_xlabel("Azimuth [°]")
            ax_az.set_ylabel("Spectrum [dB]")
            ax_az.set_ylim(-40, 0)
            ax_az.set_title("Azimuth Cut")

            az_hist.append(az)
            el_hist.append(el)
            ax_track.clear()
            ax_track.plot(az_hist, "b-", label="Azimuth")
            ax_track.plot(el_hist, "r-", label="Elevation")
            ax_track.legend()
            ax_track.set_ylabel("Degrees")
            ax_track.set_title("Tracking History")

            snrs = [r["snr"] for r in results]
            ax_snr.clear()
            ax_snr.plot(snrs, "g-")
            ax_snr.set_xlabel("Burst #")
            ax_snr.set_ylabel("SNR [dB]")
            ax_snr.set_title("SNR History")

            plt.tight_layout()
            plt.pause(0.01)

    if args.gui:
        plt.show()

    if args.save_spectrum and spectra:
        np.savez(args.save_spectrum,
                 spectra=np.stack(spectra),
                 az=np.array([r["az"] for r in results]),
                 el=np.array([r["el"] for r in results]),
                 snr=np.array([r["snr"] for r in results]))
        print(f"\nSaved {len(spectra)} spectra to {args.save_spectrum}")

    print(f"\nDone. {state['n_bursts']} bursts detected in {n_total_frames} frames.")


if __name__ == "__main__":
    main()