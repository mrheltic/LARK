#!/usr/bin/env python3
"""
run_doa.py — Real-time Iridium burst DoA with KrakenSDR.

Supports two modes:
  1. Standalone (default): uses KrakenIQSource (pure TCP/Python)
  2. GNU Radio (--use-gr): uses krakensdr_source from gr-krakensdr

Both modes reuse the LARK core DOA library (UCA 2D-MUSIC/Capon/Bartlett/Phase-Fit).

Usage
-----
    python3 run_doa.py [options]

    Options:
        --freq 1626.27e6   Center frequency [Hz]
        --gain 40.2         IF gain [dB] per channel
        --algo MUSIC        DOA algorithm: MUSIC, CAPON, BARTLETT, PHASE-FIT
        --mode indoor       indoor: narrow tone scan ±3 kHz; outdoor: wide ±45 kHz
        --autocal           Auto-calibrate phase offsets from first good burst
        --cal-az / --cal-el Known TX direction for autocal
        --use-gr            Use krakensdr_source (GNU Radio) instead of KrakenIQSource
        --gui               Show live matplotlib display
        --verbose            Print per-frame info
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_GR_KRAKEN_PATH = os.path.abspath(
    os.path.join(_SRC, "..", "..", "external", "gr-krakensdr", "python")
)
if _GR_KRAKEN_PATH not in sys.path:
    sys.path.insert(0, _GR_KRAKEN_PATH)

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
from hardware.kraken_iq_source import KrakenIQSource

_ALGO_MAP = {
    "MUSIC": doa_music_uca_2d,
    "BARTLETT": doa_bartlett_uca_2d,
    "CAPON": doa_capon_uca_2d,
    "PHASE-FIT": None,
}

_MODE_PROFILES = {
    "indoor": dict(
        tone_nom_hz=3125.0,
        scan_bw_hz=3_000.0,
        bpf_bw_hz=8_000.0,
        dc_guard_hz=200.0,
        min_snr_db=2.0,
        min_sep_hz=1_000.0,
        energy_threshold=3.0,
        prefer_nom=False,
    ),
    "outdoor": dict(
        tone_nom_hz=3125.0,
        scan_bw_hz=45_000.0,
        bpf_bw_hz=15_000.0,
        dc_guard_hz=500.0,
        min_snr_db=3.0,
        min_sep_hz=5_000.0,
        energy_threshold=2.0,
        prefer_nom=False,
    ),
    "indoor_tx": dict(
        tone_nom_hz=0.0,
        scan_bw_hz=5_000.0,
        bpf_bw_hz=4_000.0,
        dc_guard_hz=0.0,
        min_snr_db=2.0,
        min_sep_hz=500.0,
        energy_threshold=3.0,
        prefer_nom=True,
    ),
}


def estimate_phase_residuals(R, cfg, az_deg, el_deg):
    p = cfg.positions
    az_r = np.deg2rad(az_deg)
    el_r = np.deg2rad(el_deg)
    tau_theory = 2.0 * np.pi * (
        p[:, 0] * np.cos(el_r) * np.sin(az_r)
        + p[:, 1] * np.cos(el_r) * np.cos(az_r)
    )
    a_theory = np.exp(1j * tau_theory)
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    v1 = eigenvectors[:, -1]
    v1 = v1 / v1[0]
    a_theory = a_theory / a_theory[0]
    phase_err = np.angle(v1 * np.conj(a_theory))
    return np.degrees(phase_err)


def process_frame(X, cfg, phase_offs, profile, args, state):
    fs = 1_024_000.0
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
    burst_end = min(burst_start + args.window_samples, X.shape[1])
    if burst_end - burst_start < args.pre_samples + args.bpf_guard:
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

    if args.verbose:
        for i, (f, s) in enumerate(tones[:5]):
            tag = " <-- LOCK" if i == 0 else ""
            print(f"    tone[{i}]: {f:+.0f} Hz  SNR={s:.1f} dB{tag}")

    X_win = X[:, burst_start:burst_end]
    if X_win.shape[1] < args.window_samples:
        return None

    try:
        X_bpf = apply_bpf_and_normalize(
            X_win, args.window_samples, fs, tone_hz, profile["bpf_bw_hz"]
        )
    except ValueError:
        return None

    X_cal = apply_phase_correction(X_bpf, phase_offs)

    try:
        R_mf, y_mf, snr_db = compute_mf_covariance(
            X_cal, tone_hz, fs, args.pre_samples, args.bpf_guard
        )
    except ValueError:
        return None

    if snr_db < args.snr_min:
        return None

    if state["R_ema"] is None:
        state["R_ema"] = R_mf.copy()
    else:
        state["R_ema"] = args.cov_alpha * state["R_ema"] + (1 - args.cov_alpha) * R_mf

    algo_fn = _ALGO_MAP.get(args.algo)
    if args.algo == "PHASE-FIT":
        az_est, el_est, ph_err = doa_phase_fit_uca_2d(
            state["R_ema"], cfg,
            az_hint_deg=state.get("az_ema"),
            el_hint_deg=state.get("el_ema"),
        )
        papr = 0.0
        spec = None
    elif algo_fn is not None:
        if args.algo == "CAPON":
            spec = algo_fn(X_cal, cfg, R_in=state["R_ema"], decorr="none")
        elif args.algo == "BARTLETT":
            spec = algo_fn(X_cal, cfg, R_in=state["R_ema"])
        else:
            spec = algo_fn(X_cal, cfg, R_in=state["R_ema"])
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        if papr < args.papr_min:
            return None
    else:
        spec = doa_music_uca_2d(X_cal, cfg, R_in=state["R_ema"])
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        if papr < args.papr_min:
            return None

    if state.get("az_ema") is None:
        state["az_ema"] = az_est
        state["el_ema"] = el_est
    else:
        d_az = ((az_est - state["az_ema"] + 180) % 360) - 180
        state["az_ema"] += args.az_alpha * d_az
        state["az_ema"] %= 360.0
        state["el_ema"] += args.el_alpha * (el_est - state["el_ema"])

    state["n_bursts"] += 1

    return dict(
        az=state["az_ema"], el=state["el_ema"],
        snr=snr_db, papr=papr,
        tone=tone_hz, doppler=tone_hz - profile["tone_nom_hz"],
        spec=spec, R=state["R_ema"],
    )


def parse_args():
    p = argparse.ArgumentParser(description="Real-time Iridium burst DoA with KrakenSDR")
    p.add_argument("--freq", type=float, default=1626.27e6)
    p.add_argument("--gain", type=float, default=40.2)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--ctrl", type=int, default=5001)
    p.add_argument("--n-ant", type=int, default=5)
    p.add_argument("--radius", type=float, default=0.4253)
    p.add_argument("--ant0-offset", type=float, default=0.0)
    p.add_argument("--ccw", action="store_true")
    p.add_argument("--algo", type=str, default="MUSIC",
                   choices=["MUSIC", "CAPON", "BARTLETT", "PHASE-FIT"])
    p.add_argument("--mode", type=str, default="indoor",
                   choices=["indoor", "outdoor", "indoor_tx"],
                   help="indoor=narrow IRA scan, outdoor=wide, indoor_tx=TX near DC")
    p.add_argument("--tone-nom", type=float, default=None,
                   help="Override profile tone_nom_hz [Hz]")
    p.add_argument("--scan-bw", type=float, default=None,
                   help="Override profile scan_bw_hz [Hz]")
    p.add_argument("--bpf-bw", type=float, default=None,
                   help="Override profile bpf_bw_hz [Hz]")
    p.add_argument("--n-az", type=int, default=360)
    p.add_argument("--n-el", type=int, default=86)
    p.add_argument("--el-min", type=float, default=5.0)
    p.add_argument("--el-max", type=float, default=90.0)
    p.add_argument("--phase-offs", type=str, default="0.0,54.95,137.24,133.58,48.31")
    p.add_argument("--cpi-size", type=int, default=131072)
    p.add_argument("--threshold", type=float, default=3.0,
                   help="Energy burst detection threshold factor")
    p.add_argument("--snr-min", type=float, default=-3.0)
    p.add_argument("--papr-min", type=float, default=3.0)
    p.add_argument("--cov-alpha", type=float, default=0.93)
    p.add_argument("--az-alpha", type=float, default=0.88)
    p.add_argument("--el-alpha", type=float, default=0.65)
    p.add_argument("--pre-samples", type=int, default=2621)
    p.add_argument("--bpf-guard", type=int, default=128)
    p.add_argument("--window-samples", type=int, default=3000,
                   help="BPF window length (>= pre_samples + bpf_guard + margin)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--gui", action="store_true", help="Show live matplotlib display")
    p.add_argument("--use-gr", action="store_true",
                   help="Use krakensdr_source (GNU Radio) instead of KrakenIQSource")
    p.add_argument("--autocal", action="store_true",
                   help="Auto-calibrate phase offsets from first good burst")
    p.add_argument("--cal-az", type=float, default=None)
    p.add_argument("--cal-el", type=float, default=None)
    p.add_argument("--cal-bursts", type=int, default=10)
    p.add_argument("--cal-ema", type=float, default=0.7)
    return p.parse_args()


def run_standalone(args):
    profile = _MODE_PROFILES[args.mode]
    if args.tone_nom is not None:
        profile = {**profile, "tone_nom_hz": args.tone_nom}
    if args.scan_bw is not None:
        profile = {**profile, "scan_bw_hz": args.scan_bw}
    if args.bpf_bw is not None:
        profile = {**profile, "bpf_bw_hz": args.bpf_bw}
    phase_offs = [float(x) for x in args.phase_offs.split(",")]
    fs = 1_024_000.0

    cfg = UcaConfig(
        n_ant=args.n_ant, radius_lambda=args.radius,
        n_az=args.n_az, n_el=args.n_el,
        el_min_deg=args.el_min, el_max_deg=args.el_max,
        num_expected_signals=1,
        ant0_offset_deg=args.ant0_offset,
        ant_ccw=args.ccw,
    )

    print(f"\n{'='*60}")
    print(f"  Iridium DoA — {args.algo} on {args.n_ant}-element UCA")
    print(f"  Freq: {args.freq/1e6:.3f} MHz | Gain: {args.gain:.1f} dB")
    print(f"  Mode: {args.mode} | Scan BW: ±{profile['scan_bw_hz']/1e3:.0f} kHz | Tone nom: {profile['tone_nom_hz']:.0f} Hz")
    print(f"  UCA radius: {args.radius:.4f} λ | Grid: {args.n_az}×{args.n_el}")
    print(f"  Phase offsets: {phase_offs}")
    print(f"  Source: KrakenIQSource ({args.host}:{args.port})")
    if args.autocal:
        print(f"  Autocal: az={args.cal_az}° el={args.cal_el}° ({args.cal_bursts} bursts)")
    print(f"  Ctrl+C to stop")
    print(f"{'='*60}\n")

    src = KrakenIQSource(
        host=args.host, port=args.port, ctrl_port=args.ctrl,
        num_channels=args.n_ant, freq_hz=args.freq,
        gain_db=args.gain, queue_size=4,
        verbose=3 if args.verbose else 0,
    )

    state = dict(R_ema=None, az_ema=None, el_ema=None, n_bursts=0)
    cal_state = dict(accumulated=0, phase_sums=np.zeros(args.n_ant, dtype=complex))

    stop = __import__("threading").Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set() or print("\nStopping..."))

    if args.gui:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        az_grid = cfg.az_range_deg()
        el_grid = cfg.el_range_deg()
        az_hist, el_hist, eig_hist = [], [], []
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Iridium DoA — {args.algo}", fontsize=14)

    src.start()

    while not stop.is_set():
        frame = src.get_frame(timeout=2.0)
        if frame is None:
            continue

        X = frame[:args.n_ant, :]

        result = process_frame(X, cfg, phase_offs, profile, args, state)
        if result is None:
            continue

        az = result["az"]
        el = result["el"]

        if args.autocal and args.cal_az is not None:
            if cal_state["accumulated"] < args.cal_bursts:
                res = estimate_phase_residuals(
                    state["R_ema"], cfg, args.cal_az, args.cal_el
                )
                cal_state["phase_sums"] += np.exp(1j * np.deg2rad(res))
                cal_state["accumulated"] += 1
                if cal_state["accumulated"] == args.cal_bursts:
                    mean_err = np.degrees(np.angle(
                        cal_state["phase_sums"] / cal_state["accumulated"]
                    ))
                    phase_offs = [phase_offs[k] + mean_err[k] for k in range(args.n_ant)]
                    phase_offs[0] = 0.0
                    print(f"\n  ★ Autocal complete: {[f'{p:.2f}' for p in phase_offs]}\n")

        n = state["n_bursts"]
        print(f"[{n:4d}] az={az:6.1f}°  el={el:5.1f}°  "
              f"snr={result['snr']:5.1f}dB  papr={result['papr']:5.1f}dB  "
              f"tone={result['tone']:.0f}Hz  doppler={result['doppler']:+.0f}Hz")

        if args.gui and result.get("spec") is not None:
            spec = result["spec"]
            ax_spec, ax_az, ax_track, ax_eig = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

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

            az_hist.append(az)
            el_hist.append(el)
            ax_track.clear()
            ax_track.plot(az_hist, "b-", label="Azimuth")
            ax_track.plot(el_hist, "r-", label="Elevation")
            ax_track.legend()
            ax_track.set_ylabel("Degrees")

            plt.tight_layout()
            plt.pause(0.01)

    src.stop()
    if args.gui:
        plt.close("all")
    print(f"\nStopped after {state['n_bursts']} bursts.")


def run_gnuradio(args):
    from gnuradio import gr, qtgui
    from krakensdr import krakensdr_source

    profile = _MODE_PROFILES[args.mode]
    phase_offs = [float(x) for x in args.phase_offs.split(",")]
    fs = 1_024_000.0

    cfg = UcaConfig(
        n_ant=args.n_ant, radius_lambda=args.radius,
        n_az=args.n_az, n_el=args.n_el,
        el_min_deg=args.el_min, el_max_deg=args.el_max,
        num_expected_signals=1,
        ant0_offset_deg=args.ant0_offset,
        ant_ccw=args.ccw,
    )

    print(f"\n{'='*60}")
    print(f"  Iridium DoA — {args.algo} on {args.n_ant}-element UCA (GNU Radio)")
    print(f"  Freq: {args.freq/1e6:.3f} MHz | Gain: {args.gain:.1f} dB")
    print(f"  Mode: {args.mode} | Scan BW: ±{profile['scan_bw_hz']/1e3:.0f} kHz | Tone nom: {profile['tone_nom_hz']:.0f} Hz")
    print(f"  Source: krakensdr_source ({args.host}:{args.port})")
    print(f"  Ctrl+C to stop")
    print(f"{'='*60}\n")

    from iridium_doa_processor import iridium_doa_processor

    tb = gr.top_block("Iridium DoA")

    src = krakensdr_source(
        ipAddr=args.host, port=args.port, ctrlPort=args.ctrl,
        numChannels=args.n_ant, freq=args.freq / 1e6,
        gain=[args.gain] * args.n_ant, debug=bool(args.verbose),
    )

    s2v = []
    for ch in range(args.n_ant):
        s2v.append(gr.stream_to_vector(gr.sizeof_gr_complex, args.cpi_size))

    doa = iridium_doa_processor(
        cpi_size=args.cpi_size, fs=fs, freq_hz=args.freq,
        n_ant=args.n_ant, radius_lambda=args.radius,
        ant0_offset_deg=args.ant0_offset, ant_ccw=args.ccw,
        n_az=args.n_az, n_el=args.n_el,
        el_min_deg=args.el_min, el_max_deg=args.el_max,
        algorithm=args.algo, num_signals=1,
        phase_offsets_deg=phase_offs,
        energy_threshold=profile["energy_threshold"],
        tone_nom_hz=profile["tone_nom_hz"],
        tone_scan_bw_hz=profile["scan_bw_hz"],
        bpf_bw_hz=profile["bpf_bw_hz"],
        snr_min_db=args.snr_min,
        papr_min_db=args.papr_min,
        cov_alpha=args.cov_alpha,
        az_ema_alpha=args.az_alpha,
        el_ema_alpha=args.el_alpha,
        pre_samples=args.pre_samples,
        bpf_guard=args.bpf_guard,
        window_samples=args.window_samples,
    )

    for ch in range(args.n_ant):
        tb.connect((src, ch), (s2v[ch], 0))
        tb.connect((s2v[ch], 0), (doa, ch))

    msg_dbg = gr.message_debug()

    tb.msg_connect(doa, "azimuth", msg_dbg, "print")
    tb.msg_connect(doa, "elevation", msg_dbg, "print")
    tb.msg_connect(doa, "snr", msg_dbg, "print")
    tb.msg_connect(doa, "burst_detected", msg_dbg, "print")

    tb.start()
    try:
        tb.wait()
    except KeyboardInterrupt:
        pass
    finally:
        tb.stop()
        tb.wait()
        print("\nStopped.")


def main():
    args = parse_args()
    if args.use_gr:
        run_gnuradio(args)
    else:
        run_standalone(args)


if __name__ == "__main__":
    main()