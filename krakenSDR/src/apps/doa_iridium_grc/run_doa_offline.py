#!/usr/bin/env python3
"""
run_doa_offline.py — Offline Iridium DOA from recorded data.

Supported inputs
──────────────────
  session_.../           Incremental raw/frame_*.npy  (memory-safe streaming)
  session_.../doa/       Live estimates est_*.npz (--plot-only --from-doa)
  offline_doa.jsonl      Previous run_doa_offline output (--plot-only)
  doa_music.npz          Consolidated spectra (--plot-only)
  *_iq.npz               Pre-segmented burst windows  (N, 5, 2621)
  *.wav / *.cf32         Single-channel — burst/tone detection only

Usage:
    # Process raw IQ + show plot
    python3 run_doa_offline.py data/doa_iridium/session_.../ --mode outdoor --gui

    # Plot existing live estimates (no IQ reload)
    python3 run_doa_offline.py data/doa_iridium/session_.../ --plot-only --from-doa --gui

    # Interactive replay with timeline controls
    python3 run_doa_offline.py data/doa_iridium/session_.../ --replay --from-doa

    # Save figure without opening window
    python3 run_doa_offline.py session_.../ --plot-only --from-doa --save-fig session_.../plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from apps.doa_iridium_grc.run_doa import (  # noqa: E402
    _PROFILES,
    _apply_cli,
    _load_cal,
    load_config,
    parse_args as _live_parse_args,
)
from apps.doa_iridium_grc.lark.burst_processing import (
    apply_bpf_and_normalize,
    compute_mf_covariance,
    detect_energy_bursts,
    scan_preamble_tones,
)
from apps.doa_iridium_grc.lark.pipeline_debug import PipelineDebugSaver
from core.doa_algorithms import apply_phase_correction
from core.doa_uca_2d import (
    UcaConfig,
    doa_bartlett_uca_2d,
    doa_capon_uca_2d,
    doa_music_uca_2d,
    find_peak_uca_2d,
)


FS = 1_024_000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline Iridium DOA from recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to .npz, .wav, .cf32, or .iq file")
    p.add_argument("--config", default=os.path.join(_HERE, "doa_config.toml"))
    p.add_argument("--mode", choices=["indoor", "outdoor"])
    p.add_argument("--algo", choices=["music", "capon", "bartlett"])
    p.add_argument("--cal-file", metavar="PATH")
    p.add_argument("--phase-offs", metavar="DEG_LIST",
                   help="Comma-separated phase offsets (overrides cal-file)")
    p.add_argument("--snr-min", type=float, default=-5.0,
                   help="Minimum SINR gate [dB] (default -5 for pre-segmented recordings)")
    p.add_argument("--papr-min", type=float, default=None,
                   help="Minimum PAPR gate [dB] (default: from config)")
    p.add_argument("--max-bursts", type=int, default=0,
                   help="Limit CPI frames processed (0 = all; incremental sessions)")
    p.add_argument("--frame-start", type=int, default=0,
                   help="First frame index for incremental session replay")
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Process every Nth CPI frame (1 = all)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Max CPI frames to process after start/stride (0 = all)")
    p.add_argument("--out", metavar="FILE", help="Write results JSONL to file")
    p.add_argument("--debug-dir", metavar="DIR")
    p.add_argument("--save-doa", metavar="FILE",
                   help="Write doa_music.npz (spec2d + Kraken-style doa_az)")
    p.add_argument("--gui", action="store_true",
                   help="Show matplotlib summary after processing (or with --plot-only)")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip IQ processing — plot existing JSONL or doa/ estimates")
    p.add_argument("--from-doa", action="store_true",
                   help="With --plot-only: load session doa/est_*.npz (not offline JSONL)")
    p.add_argument("--save-fig", metavar="DIR",
                   help="Save summary PNG to directory")
    p.add_argument("--plot-stride", type=int, default=1,
                   help="Plot every Nth estimate (default 1)")
    p.add_argument("--max-plot", type=int, default=5000,
                   help="Max estimates to load for plotting (0 = unlimited)")
    p.add_argument("--replay", action="store_true",
                   help="Interactive timeline replay (slider, play/pause, speed, reverse)")
    p.add_argument("--reprocess-multi", action="store_true",
                   help="Reprocess session raw/ → doa_multi/ + tracks.json")
    p.add_argument("--compare", action="store_true",
                   help="Static comparison of MUSIC/Capon/Bartlett reprocess outputs")
    p.add_argument("--out-subdir", metavar="NAME",
                   help="Output subdir for --reprocess-multi (default: doa_multi or doa_multi_<algo>)")
    p.add_argument("--el-min", type=float, default=10.0,
                   help="Minimum elevation [°] for --reprocess-multi (filters horizon noise)")
    p.add_argument("--k-peaks", type=int, default=3,
                   help="Max DOA peaks per burst (multi reprocess)")
    p.add_argument("--track-gap-s", type=float, default=30.0,
                   help="Max time gap [s] before closing a satellite track")
    p.add_argument("--min-track-len", type=int, default=5,
                   help="Min peaks per track (shorter → outlier)")
    p.add_argument("--save-spec", action="store_true",
                   help="With --reprocess-multi: store spec2d in burst npz files")
    p.add_argument("--max-az-deg", type=float, default=25.0,
                   help="Max azimuth separation [°] for track association")
    p.add_argument("--max-el-deg", type=float, default=15.0,
                   help="Max elevation separation [°] for track association")
    p.add_argument("--max-cfo-hz", type=float, default=12000.0,
                   help="Max CFO separation [Hz] for track association")
    p.add_argument("--recluster-only", action="store_true",
                   help="Re-run clustering on existing JSONL without reprocessing IQ")
    p.add_argument("--no-tracks", action="store_true",
                   help="Disable track coloring — show all peaks with flat color")
    p.add_argument("--phase-cal", action="store_true",
                   help="Enable hardware phase calibration")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _build_uca(cfg: dict) -> UcaConfig:
    arr = cfg["array"]
    alg = cfg["algorithm"]
    return UcaConfig(
        n_ant=arr["n_ant"], radius_lambda=arr["radius_lambda"],
        n_az=alg["n_az"], n_el=alg["n_el"],
        el_min_deg=5.0, el_max_deg=90.0,
        ant0_offset_deg=arr["ant0_offset_deg"],
        ant_ccw=arr["ant_ccw"],
        num_expected_signals=1,
    )


def _phase_offsets(cfg: dict, args: argparse.Namespace) -> list[float]:
    n = cfg["array"]["n_ant"]
    if not cfg["array"].get("use_phase_cal", False) and not args.phase_cal:
        return [0.0] * n
    if args.phase_offs:
        offs = [float(x) for x in args.phase_offs.split(",")]
    else:
        offs = _load_cal(cfg["array"].get("cal_file", ""), n)
    if all(o == 0.0 for o in offs) and not cfg["array"].get("cal_file"):
        legacy = os.path.join(_SRC, "apps", "doa_iridium", "config.py")
        if os.path.isfile(legacy):
            import importlib.util
            spec = importlib.util.spec_from_file_location("doa_iridium_cfg", legacy)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            offs = list(getattr(mod, "CHANNEL_PHASE_OFFSETS_DEG", offs))
    offs = (offs + [0.0] * n)[:n]
    return [float(x) for x in offs]


def _run_doa_on_window(
    X_win: np.ndarray,
    *,
    cfg: dict,
    profile: dict,
    uca: UcaConfig,
    phase_offs: list[float],
    algo: str,
    thr: dict,
    state: dict,
    burst_idx: int,
    debug_dir: str = "",
    spec_out: list | None = None,
) -> dict | None:
    """Process one burst window (n_ant, N) through stages 2–7."""
    hw = cfg["hardware"]
    pre_samples = hw["pre_samples"]
    window_samples = min(hw["window_samples"], X_win.shape[1])
    bpf_guard = hw["bpf_guard"]
    has_cal = any(o != 0.0 for o in phase_offs)

    if X_win.shape[1] < pre_samples + bpf_guard:
        # Pre-segmented _iq.npz windows are often exactly pre_samples long
        if X_win.shape[1] < bpf_guard + 512:
            return None
        pre_samples = X_win.shape[1] - bpf_guard
        window_samples = X_win.shape[1]

    dbg = PipelineDebugSaver(debug_dir, burst_idx) if debug_dir else None
    if dbg and dbg.enabled:
        dbg.save("raw_iq", X_win)

    tones = scan_preamble_tones(
        X_win[0], FS,
        nom_tone_hz=profile["tone_nom_hz"],
        scan_bw_hz=profile["scan_bw_hz"],
        min_snr_db=profile["min_snr_db"],
        dc_guard_hz=profile["dc_guard_hz"],
    )
    if not tones:
        return None
    tone_hz, _ = tones[0]
    if dbg and dbg.enabled:
        dbg.save("tones", [{"tone_hz": float(t), "snr_db": float(s)} for t, s in tones])

    try:
        X_bpf = apply_bpf_and_normalize(
            X_win[:, :window_samples], window_samples, FS, tone_hz, profile["bpf_bw_hz"]
        )
    except ValueError:
        return None

    n_pre_eff = min(pre_samples, X_bpf.shape[1] - bpf_guard)
    if n_pre_eff < 512:
        return None

    X_cal = apply_phase_correction(X_bpf, phase_offs) if has_cal else X_bpf
    if dbg and dbg.enabled:
        dbg.save("bpf", X_bpf)
        dbg.save("phase_corrected", X_cal)

    try:
        R_mf, _, snr_db = compute_mf_covariance(
            X_cal, tone_hz, FS, n_pre_eff, bpf_guard
        )
    except ValueError:
        return None

    if snr_db < thr["snr_min_db"]:
        return None

    cov_alpha = thr["cov_alpha"]
    R_ema = state.get("R_ema")
    R_ema = R_mf.copy() if R_ema is None else (
        cov_alpha * R_ema + (1.0 - cov_alpha) * R_mf
    )
    state["R_ema"] = R_ema

    if dbg and dbg.enabled:
        dbg.save("R_mf", R_mf)
        dbg.save("R_ema", R_ema)

    algo_u = algo.upper()
    if algo_u == "CAPON":
        spec = doa_capon_uca_2d(X_cal, uca, R_in=R_ema, decorr="none")
    elif algo_u == "BARTLETT":
        spec = doa_bartlett_uca_2d(X_cal, uca, R_in=R_ema)
    else:
        spec = doa_music_uca_2d(X_cal, uca, R_in=R_ema)

    az_raw, el_raw, papr = find_peak_uca_2d(spec, uca)
    if papr < thr["papr_min_db"]:
        return None

    az_alpha = thr["az_ema_alpha"]
    el_alpha = thr["el_ema_alpha"]
    az_ema = state.get("az_ema")
    el_ema = state.get("el_ema")
    if az_ema is None:
        az_ema, el_ema = az_raw, el_raw
    else:
        d_az = ((az_raw - az_ema + 180.0) % 360.0) - 180.0
        az_ema = (az_ema + az_alpha * d_az) % 360.0
        el_ema += el_alpha * (el_raw - el_ema)
    state["az_ema"] = az_ema
    state["el_ema"] = el_ema
    state["n"] = state.get("n", 0) + 1

    result = {
        "n": state["n"],
        "burst_idx": burst_idx,
        "az": round(az_ema, 1),
        "el": round(el_ema, 1),
        "az_raw": round(az_raw, 1),
        "el_raw": round(el_raw, 1),
        "snr_db": round(snr_db, 1),
        "papr_db": round(float(papr), 1),
        "cfo_hz": round(tone_hz - profile["tone_nom_hz"], 0),
        "tone_hz": round(tone_hz, 1),
        "algo": algo_u,
    }
    if dbg and dbg.enabled:
        dbg.save("spec2d", spec)
        dbg.save("doa_result", result)
    if spec_out is not None:
        spec_out.append(np.asarray(spec, dtype=np.float32))
    return result


def _save_doa_music(path: str, specs: list[np.ndarray], results: list[dict], cfg: dict) -> None:
    """Write Kraken-style doa_music.npz from offline processing."""
    alg = cfg["algorithm"]
    hw = cfg["hardware"]
    az_grid = np.linspace(0.0, 360.0, alg["n_az"], endpoint=False, dtype=np.float32)
    el_grid = np.linspace(5.0, 90.0, alg["n_el"], dtype=np.float32)
    spec2d = np.stack(specs, axis=0)
    doa_az = np.max(spec2d, axis=1).astype(np.float32)
    np.savez_compressed(
        path,
        spec2d=spec2d,
        doa_az=doa_az,
        last_spec2d=spec2d[-1],
        last_doa_az=doa_az[-1],
        az_deg=np.array([r["az"] for r in results], dtype=np.float32),
        el_deg=np.array([r["el"] for r in results], dtype=np.float32),
        papr_db=np.array([r["papr_db"] for r in results], dtype=np.float32),
        snr_db=np.array([r["snr_db"] for r in results], dtype=np.float32),
        az_grid_deg=az_grid,
        el_grid_deg=el_grid,
        freq_hz=np.int64(hw["freq_mhz"] * 1e6),
    )
    print(f"Saved {len(specs)} DOA spectra → {path}")


def process_burst_npz(path: str, cfg: dict, args: argparse.Namespace) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    if "X" in data:
        X_all = data["X"]
    elif "bursts" in data:
        X_all = data["bursts"]
    else:
        raise ValueError(f"{path}: expected key 'X' or 'bursts', got {list(data.keys())}")

    X_all = np.asarray(X_all)
    if X_all.ndim != 3:
        raise ValueError(f"Expected (N, n_ant, N_samp), got shape {X_all.shape}")

    n_total = X_all.shape[0]
    n_proc = n_total if args.max_bursts <= 0 else min(n_total, args.max_bursts)
    print(f"Loaded {n_total} burst windows from {os.path.basename(path)} "
          f"(processing {n_proc})")

    alg = cfg["algorithm"]
    profile = _PROFILES[alg["mode"]]
    uca = _build_uca(cfg)
    phase_offs = _phase_offsets(cfg, args)
    thr = cfg["thresholds"]
    if args.snr_min is not None:
        thr = {**thr, "snr_min_db": args.snr_min}
    if args.papr_min is not None:
        thr = {**thr, "papr_min_db": args.papr_min}
    debug_dir = args.debug_dir or cfg["output"].get("debug_dir", "")

    state: dict = {}
    results: list[dict] = []
    specs: list[np.ndarray] = []
    spec_out = specs if args.save_doa else None
    t0 = time.time()

    for i in range(n_proc):
        Xw = np.asarray(X_all[i], dtype=np.complex64)
        if Xw.shape[0] < cfg["array"]["n_ant"]:
            continue
        Xw = Xw[:cfg["array"]["n_ant"], :]

        rec = _run_doa_on_window(
            Xw, cfg=cfg, profile=profile, uca=uca, phase_offs=phase_offs,
            algo=alg["algo"], thr=thr, state=state, burst_idx=i + 1,
            debug_dir=debug_dir, spec_out=spec_out,
        )
        if rec is None:
            continue
        results.append(rec)
        line = json.dumps(rec)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if args.verbose and len(results) % 10 == 0:
            print(f"  … {len(results)} estimates in {time.time()-t0:.1f}s", file=sys.stderr)

    if args.save_doa and specs:
        _save_doa_music(args.save_doa, specs, results, cfg)
    return results


def _process_cpi_frames(
    frame_iter,
    *,
    cfg: dict,
    args: argparse.Namespace,
    label: str,
    total_hint: int = 0,
) -> list[dict]:
    """Run burst detection + DOA on CPI frames from an iterator (idx, array)."""
    alg = cfg["algorithm"]
    profile = _PROFILES[alg["mode"]]
    uca = _build_uca(cfg)
    phase_offs = _phase_offsets(cfg, args)
    thr = cfg["thresholds"]
    if args.snr_min is not None:
        thr = {**thr, "snr_min_db": args.snr_min}
    if args.papr_min is not None:
        thr = {**thr, "papr_min_db": args.papr_min}
    debug_dir = args.debug_dir or cfg["output"].get("debug_dir", "")

    state: dict = {}
    results: list[dict] = []
    specs: list[np.ndarray] = []
    spec_out = specs if args.save_doa else None
    window_samples = cfg["hardware"]["window_samples"]
    t0 = time.time()
    n_seen = 0

    hint = f" / ~{total_hint}" if total_hint else ""
    print(f"Processing CPI frames from {label}{hint}")

    for fi, X in frame_iter:
        n_seen += 1
        starts = detect_energy_bursts(
            X[0], FS, threshold_factor=profile["energy_threshold"]
        )
        if not starts:
            continue
        b0 = starts[0]
        bend = min(b0 + window_samples, X.shape[1])
        X_win = X[:, b0:bend]
        rec = _run_doa_on_window(
            X_win, cfg=cfg, profile=profile, uca=uca, phase_offs=phase_offs,
            algo=alg["algo"], thr=thr, state=state, burst_idx=fi + 1,
            debug_dir=debug_dir, spec_out=spec_out,
        )
        if rec is None:
            continue
        rec["frame"] = fi
        results.append(rec)
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        if not args.out:
            print(json.dumps(rec), flush=True)
        if args.verbose and len(results) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  … {len(results)} DOA estimates from {n_seen} CPI "
                  f"({elapsed:.0f}s)", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"Processed {n_seen} CPI in {elapsed:.0f}s → {len(results)} DOA estimates",
          file=sys.stderr)

    if args.save_doa and specs:
        _save_doa_music(args.save_doa, specs, results, cfg)
    return results


def process_raw_iq_npz(path: str, cfg: dict, args: argparse.Namespace) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    if "X" not in data:
        raise ValueError(f"{path}: expected key 'X' with shape (N, n_ant, cpi_size)")

    X_all = np.asarray(data["X"], dtype=np.complex64)
    if X_all.ndim != 3:
        raise ValueError(f"Expected (N, n_ant, cpi_size), got {X_all.shape}")

    n_ant = cfg["array"]["n_ant"]
    cpi_size = int(data["cpi_size"]) if "cpi_size" in data else X_all.shape[2]
    n_total = X_all.shape[0]
    n_proc = n_total if args.max_bursts <= 0 else min(n_total, args.max_bursts)

    if "freq_hz" in data:
        cfg = {**cfg, "hardware": {**cfg["hardware"],
                                   "freq_mhz": float(data["freq_hz"]) / 1e6}}

    print(f"Loaded {n_total} raw CPI frames, cpi_size={cpi_size} "
          f"(processing {n_proc})")

    frames = [X_all[i, :n_ant, :cpi_size] for i in range(n_proc)]
    return _process_cpi_frames(
        ((i, frames[i]) for i in range(len(frames))),
        cfg=cfg, args=args, label=os.path.basename(path), total_hint=n_proc,
    )


def process_multichan_npz(path: str, cfg: dict, args: argparse.Namespace) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    n_ant = cfg["array"]["n_ant"]
    channels = []
    for i in range(n_ant):
        for key in (f"ch{i}", f"ant{i}"):
            if key in data:
                channels.append(np.asarray(data[key]).flatten().astype(np.complex64))
                break
        else:
            raise ValueError(f"Missing channel {i} in {path}")
    min_len = min(len(c) for c in channels)
    X_full = np.stack([c[:min_len] for c in channels])
    cpi = cfg["hardware"].get("cpi_size", 131072)
    n_frames = min_len // cpi
    if args.max_bursts > 0:
        n_frames = min(n_frames, args.max_bursts)

    print(f"Loaded {n_ant}-ch recording, {min_len/cpi:.0f} CPI frames "
          f"(processing {n_frames})")

    frames = [X_full[:, fi * cpi:(fi + 1) * cpi] for fi in range(n_frames)]
    return _process_cpi_frames(
        ((fi, frames[fi]) for fi in range(n_frames)),
        cfg=cfg, args=args, label=os.path.basename(path), total_hint=n_frames,
    )


def process_single_channel(path: str, cfg: dict, args: argparse.Namespace) -> None:
    from hardware.file_iq_source import FileIQSource

    profile = _PROFILES[cfg["algorithm"]["mode"]]
    src = FileIQSource(path, sample_rate=FS, center_freq_hz=cfg["hardware"]["freq_mhz"] * 1e6)
    src.start()

    print(f"\nSingle-channel file: DOA skipped (need {cfg['array']['n_ant']} antennas)")
    print(f"Running burst + tone detection on {os.path.basename(path)} …\n")

    n_frames = n_bursts = 0
    cfo_list: list[float] = []

    while True:
        frame = src.get_frame()
        if frame is None:
            break
        n_frames += 1
        if args.max_bursts > 0 and n_bursts >= args.max_bursts:
            break

        ch0 = frame[0] if frame.ndim == 2 else frame
        starts = detect_energy_bursts(ch0, FS, threshold_factor=profile["energy_threshold"])
        for b0 in starts:
            bend = min(b0 + cfg["hardware"]["window_samples"], len(ch0))
            tones = scan_preamble_tones(
                ch0[b0:bend], FS,
                nom_tone_hz=profile["tone_nom_hz"],
                scan_bw_hz=profile["scan_bw_hz"],
                min_snr_db=profile["min_snr_db"],
                dc_guard_hz=profile["dc_guard_hz"],
            )
            if not tones:
                continue
            tone_hz, snr = tones[0]
            cfo = tone_hz - profile["tone_nom_hz"]
            cfo_list.append(cfo)
            n_bursts += 1
            if args.verbose or n_bursts <= 20:
                t_s = (src.current_sample - len(ch0) + b0) / FS
                print(f"  burst {n_bursts:4d}  t={t_s:8.1f}s  "
                      f"cfo={cfo:+7.0f}Hz  snr={snr:.1f}dB")

    src.stop()
    print(f"\nSummary: {n_frames} frames, {n_bursts} bursts detected")
    if cfo_list:
        cfo_a = np.array(cfo_list)
        print(f"  CFO range: {cfo_a.min():+.0f} … {cfo_a.max():+.0f} Hz  "
              f"(median {np.median(cfo_a):+.0f} Hz)")


def process_incremental_session(session_dir: str, cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Replay CPI frames from session_.../raw/frame_*.npy (memory-safe streaming)."""
    from apps.doa_iridium_grc.lark.recording import count_session_raw_frames, iter_session_raw_frames

    meta_path = os.path.join(session_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cfg = {**cfg, "hardware": {**cfg["hardware"],
                                   "freq_mhz": float(meta.get("freq_hz", cfg["hardware"]["freq_mhz"] * 1e6)) / 1e6}}
        if args.mode is None and meta.get("mode"):
            cfg = {**cfg, "algorithm": {**cfg["algorithm"], "mode": meta["mode"]}}
        if args.algo is None and meta.get("algo"):
            cfg = {**cfg, "algorithm": {**cfg["algorithm"], "algo": meta["algo"]}}

    total = count_session_raw_frames(session_dir)
    start = max(0, args.frame_start)
    stride = max(1, args.frame_stride)
    max_frames = args.max_frames if args.max_frames > 0 else (
        args.max_bursts if args.max_bursts > 0 else 0
    )
    if max_frames <= 0:
        max_frames = max(0, (total - start + stride - 1) // stride)

    print(f"Session: {total} CPI in raw/  start={start} stride={stride} "
          f"max_frames={max_frames}")

    frame_iter = iter_session_raw_frames(
        session_dir, start=start, stride=stride, max_frames=max_frames,
    )
    return _process_cpi_frames(
        frame_iter, cfg=cfg, args=args,
        label=os.path.basename(session_dir), total_hint=max_frames,
    )


def _resolve_input(path: str) -> tuple[str, str]:
    """Return (path, format_id). Accepts session dirs with raw/ or raw_iq.npz."""
    if os.path.isdir(path):
        raw_dir = os.path.join(path, "raw")
        if os.path.isdir(raw_dir) and os.path.isfile(os.path.join(raw_dir, "frame_000000.npy")):
            return path, "incremental_session"
        for name in ("raw_iq.npz", "raw_iq_checkpoint.npz"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate, "raw_iq_npz"
        raise ValueError(f"No raw/ frames or raw_iq.npz found in directory {path!r}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = set(data.keys())
        if "X" in keys:
            X = np.asarray(data["X"])
            if X.ndim == 3 and (X.shape[2] > 8192 or "cpi_size" in keys):
                return path, "raw_iq_npz"
            return path, "burst_npz"
        if "bursts" in keys:
            return path, "burst_npz"
        if any(k.startswith("ch") or k.startswith("ant") for k in keys):
            return path, "multichan_npz"
        raise ValueError(f"Unknown npz layout: {sorted(keys)}")
    if ext in (".wav", ".cf32", ".iq", ".raw"):
        return path, "single_channel"
    raise ValueError(f"Unsupported file type: {ext}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.recluster_only or args.reprocess_multi:
        from apps.doa_iridium_grc.reprocess_session import reprocess_session

        if not os.path.isdir(args.input):
            print(f"Not a session directory: {args.input}", file=sys.stderr)
            sys.exit(1)
        summary = reprocess_session(args.input, cfg=cfg, args=args)
        if summary.get("recluster_only"):
            return
        if summary["n_bursts"] == 0:
            sys.exit(1)
        return

    if args.compare:
        from apps.doa_iridium_grc.lark.algo_compare import show_algo_comparison

        if not os.path.isdir(args.input):
            print(f"Not a session directory: {args.input}", file=sys.stderr)
            sys.exit(1)
        show_algo_comparison(
            args.input,
            save_dir=args.save_fig or "",
            show=args.gui or not bool(args.save_fig),
            no_tracks=getattr(args, "no_tracks", False),
        )
        return

    if args.replay:
        from apps.doa_iridium_grc.lark.offline_replay import show_replay

        max_rows = args.max_plot if args.max_plot > 0 else 0
        show_replay(
            args.input,
            stride=max(1, args.plot_stride),
            max_rows=max_rows,
            from_doa=args.from_doa or not args.input.endswith(".jsonl"),
            algo=args.algo,
            out_subdir=getattr(args, "out_subdir", None),
            no_tracks=getattr(args, "no_tracks", False),
        )
        return

    if args.plot_only:
        from apps.doa_iridium_grc.lark.offline_viz import load_results_for_plot, show_results

        max_rows = args.max_plot if args.max_plot > 0 else 0
        rows, meta, title = load_results_for_plot(
            args.input,
            stride=max(1, args.plot_stride),
            max_rows=max_rows,
            from_doa=args.from_doa,
        )
        print(f"Plotting {len(rows)} estimates from {args.input}")
        show_results(
            rows, title=title, meta=meta,
            save_dir=args.save_fig or "", show=args.gui or bool(args.save_fig),
        )
        if not rows:
            sys.exit(1)
        return

    live_ns = argparse.Namespace(
        config=args.config, gui=False, host=None, freq=None, gain=None,
        n_ant=None, cal_file=args.cal_file, ant0_offset=None, ccw=False,
        algo=args.algo, mode=args.mode, out=args.out, debug_dir=args.debug_dir,
        record=None, no_record_raw=False, record_checkpoint=None,
        phase_cal=args.phase_cal, verbose=args.verbose,
    )
    cfg = _apply_cli(cfg, live_ns)
    if args.phase_cal:
        cfg["array"]["use_phase_cal"] = True

    if args.out and os.path.exists(args.out):
        os.remove(args.out)

    input_path, fmt = _resolve_input(args.input)
    print(f"Input: {args.input}  format={fmt}  mode={cfg['algorithm']['mode']}  "
          f"algo={cfg['algorithm']['algo']}")

    if fmt == "burst_npz":
        results = process_burst_npz(input_path, cfg, args)
    elif fmt == "incremental_session":
        results = process_incremental_session(input_path, cfg, args)
    elif fmt == "raw_iq_npz":
        results = process_raw_iq_npz(input_path, cfg, args)
    elif fmt == "multichan_npz":
        results = process_multichan_npz(input_path, cfg, args)
    else:
        process_single_channel(input_path, cfg, args)
        return

    if not results:
        print("\nNo DOA estimates produced. Try --mode outdoor or lower thresholds in doa_config.toml",
              file=sys.stderr)
        sys.exit(1)

    az = [r["az"] for r in results]
    el = [r["el"] for r in results]
    print(f"\nDone: {len(results)} estimates")
    print(f"  Az median {np.median(az):.1f}°  (std {np.std(az):.1f}°)")
    print(f"  El median {np.median(el):.1f}°  (std {np.std(el):.1f}°)")

    if args.gui or args.save_fig:
        from apps.doa_iridium_grc.lark.offline_viz import show_results

        meta = {}
        if fmt == "incremental_session" and os.path.isfile(
            os.path.join(input_path, "meta.json")
        ):
            with open(os.path.join(input_path, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
        show_results(
            results,
            title=os.path.basename(input_path.rstrip("/")),
            meta=meta,
            save_dir=args.save_fig or "",
            show=args.gui,
        )


if __name__ == "__main__":
    main()
