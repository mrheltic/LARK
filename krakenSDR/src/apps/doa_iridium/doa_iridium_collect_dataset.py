#!/usr/bin/env python3
"""
Collect 5-channel Iridium burst measurements for offline processing.

This collector intentionally reuses internal functions from
`doa_iridium_burst.py` so online and offline pipelines share the same
burst detection and tone/CFO estimation behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
_REPO_ROOT = os.path.normpath(os.path.join(_SRC, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_CFG_PATH = os.path.join(_HERE, "config.py")
_cfg_spec = importlib.util.spec_from_file_location("doa_iridium_config", _CFG_PATH)
C = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec and _cfg_spec.loader
_cfg_spec.loader.exec_module(C)

import doa_iridium_burst as dib
from hardware.kraken_iq_source import KrakenIQSource
from shared.observer import get_observer
from shared.satellite_tracker import match_bursts_to_satellites


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect 5-channel Iridium burst dataset with TLE ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq", type=float, default=C.FREQ_HZ / 1e6,
                   help="RF center frequency [MHz]")
    p.add_argument("--gain", type=float, default=C.GAIN_DB,
                   help="RX gain [dB]")
    p.add_argument("--max-bursts", type=int, default=1500,
                   help="Maximum number of collected bursts")
    p.add_argument("--max-time-s", type=float, default=0.0,
                   help="Maximum collection time in seconds (0 = unlimited)")
    p.add_argument("--out-dir", default=os.path.normpath(os.path.join(dib._SRC, "..", "data", "doa_iridium")),
                   help="Output directory")
    p.add_argument("--fd-max", type=float, default=float(getattr(C, "DOPPLER_GATE_HZ", 0.0)),
                   help="Doppler gate [Hz], 0 = disabled")
    p.add_argument("--tone-snr-min", type=float, default=6.0,
                   help="Minimum SNR for tone peak scan [dB]")
    p.add_argument("--observer-lat", type=float, default=None,
                   help="Observer latitude override [deg]")
    p.add_argument("--observer-lon", type=float, default=None,
                   help="Observer longitude override [deg]")
    p.add_argument("--observer-alt", type=float, default=0.0,
                   help="Observer altitude override [m]")
    p.add_argument("--no-gt", action="store_true",
                   help="Skip TLE ground-truth annotation")
    p.add_argument("--verbose", action="store_true",
                   help="Print every accepted burst")
    return p


def _save_dataset(payload: dict, meta: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"doa_iridium_dataset_{ts}")
    npz_path = base + ".npz"
    json_path = base + ".json"
    np.savez_compressed(npz_path, **payload)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return npz_path, json_path


def main() -> None:
    args = _build_argparser().parse_args()

    freq_hz = int(args.freq * 1e6)
    host = C.HEIMDALL_HOST
    port = C.HEIMDALL_PORT

    if not dib._check_heimdall(host, port):
        raise SystemExit(f"Heimdall unreachable at {host}:{port}")

    print("[COLLECT] Starting 5-channel burst collector")
    print(f"[COLLECT] RF={freq_hz/1e6:.3f} MHz gain={args.gain:.1f} dB fd_max={args.fd_max:.0f} Hz")

    src = KrakenIQSource(
        host=host,
        port=port,
        num_channels=C.N_ANTENNAS,
        freq_hz=freq_hz,
        gain_db=args.gain,
    )
    src.start()

    fs = float(getattr(C, "SAMPLE_RATE_HZ", dib._FS))
    window_samples = dib._WINDOW_SAMPLES
    pre_samples = dib._PRE_SAMPLES

    t0_wall = time.time()
    t0_utc = datetime.now(timezone.utc)

    bursts: list[np.ndarray] = []
    timestamps_ms: list[float] = []
    cfo_hz_list: list[float] = []
    tone_snr_db_list: list[float] = []

    # Buffer frames to keep the exact same burst detector behavior as online DoA.
    frame_buf: list[np.ndarray] = []
    frame_buf_len = 0
    min_buf = dib._SF_SAMPLES + dib._WINDOW_SAMPLES + 2048

    running = True
    lock = threading.Lock()

    def _stop_if_needed() -> bool:
        if args.max_bursts > 0 and len(bursts) >= args.max_bursts:
            return True
        if args.max_time_s > 0 and (time.time() - t0_wall) >= args.max_time_s:
            return True
        return False

    try:
        while running:
            frame = src.get_frame(timeout=2.0)
            if frame is None or frame.shape[1] < 512:
                continue

            frame_buf.append(frame)
            frame_buf_len += frame.shape[1]
            if frame_buf_len < min_buf:
                continue

            X_stream = np.hstack(frame_buf)
            frame_buf.clear()
            frame_buf_len = 0
            n_total = X_stream.shape[1]

            burst_starts = dib._detect_bursts(X_stream[0], threshold_factor=3.0)
            for b_start in burst_starts:
                b_end = min(b_start + window_samples, n_total)
                if b_end - b_start < pre_samples:
                    continue

                X_win = X_stream[:, b_start:b_end]
                if X_win.shape[1] < pre_samples:
                    continue

                # Reuse same preamble onset/tone estimator as online pipeline.
                onset_ref, tone_ref = dib._find_preamble_onset(
                    X_win[0], 0, X_win.shape[1], known_hz=float(dib._PREAMBLE_TONE_HZ),
                    win=4096, freq_lock_bw=max(1200.0, float(args.fd_max) if args.fd_max > 0 else 2000.0),
                )
                if onset_ref > dib._ENERGY_WIN and onset_ref < X_win.shape[1] // 2:
                    b_start_adj = b_start + onset_ref - dib._ENERGY_WIN
                    b_end_adj = min(b_start_adj + window_samples, n_total)
                    X_win = X_stream[:, b_start_adj:b_end_adj]
                    if X_win.shape[1] < pre_samples:
                        continue

                X0_pre = X_win[0, :pre_samples]
                peaks = dib._scan_doppler_peaks(
                    X0_pre, fs,
                    nom_tone_hz=float(dib._PREAMBLE_TONE_HZ),
                    scan_bw_hz=float(getattr(C, "DOPPLER_SCAN_BW_HZ", 45_000)),
                    n_peaks=1,
                    min_sep_hz=float(getattr(C, "SAT_MIN_SEP_HZ", 5_000)),
                    min_snr_db=float(args.tone_snr_min),
                )
                if not peaks:
                    continue

                tone_hz, tone_snr_db = peaks[0]
                cfo_hz = float(tone_hz - dib._PREAMBLE_TONE_HZ)
                if args.fd_max > 0 and abs(cfo_hz) > args.fd_max:
                    continue

                t_ms = (time.time() - t0_wall) * 1000.0
                with lock:
                    bursts.append(X_win.astype(np.complex64, copy=False))
                    timestamps_ms.append(t_ms)
                    cfo_hz_list.append(cfo_hz)
                    tone_snr_db_list.append(float(tone_snr_db))
                    n = len(bursts)

                if args.verbose:
                    print(f"[COLLECT] #{n:4d} t={t_ms/1000:7.2f}s cfo={cfo_hz:+8.1f}Hz toneSNR={tone_snr_db:5.1f}dB")

                if _stop_if_needed():
                    running = False
                    break

            if _stop_if_needed():
                running = False

    except KeyboardInterrupt:
        print("[COLLECT] Interrupted by user")
    finally:
        src.stop()

    if not bursts:
        raise SystemExit("No bursts collected")

    bursts_arr = np.stack(bursts, axis=0)
    ts_arr = np.array(timestamps_ms, dtype=np.float64)
    cfo_arr = np.array(cfo_hz_list, dtype=np.float64)
    tone_snr_arr = np.array(tone_snr_db_list, dtype=np.float32)

    payload = {
        "bursts": bursts_arr,
        "timestamps_ms": ts_arr,
        "cfo_hz": cfo_arr,
        "tone_snr_db": tone_snr_arr,
        "freq_hz": np.array([freq_hz], dtype=np.int64),
        "sample_rate_hz": np.array([fs], dtype=np.float64),
        "gain_db": np.array([args.gain], dtype=np.float32),
        "window_samples": np.array([window_samples], dtype=np.int32),
        "pre_samples": np.array([pre_samples], dtype=np.int32),
    }

    meta = {
        "tool": "doa_iridium_collect_dataset",
        "timestamp_utc": t0_utc.isoformat(),
        "n_bursts": int(len(bursts)),
        "duration_s": float(time.time() - t0_wall),
        "freq_hz": int(freq_hz),
        "sample_rate_hz": float(fs),
        "gain_db": float(args.gain),
        "n_antennas": int(C.N_ANTENNAS),
        "host": host,
        "port": int(port),
    }

    if not args.no_gt:
        try:
            if args.observer_lat is not None and args.observer_lon is not None:
                lat, lon, alt = float(args.observer_lat), float(args.observer_lon), float(args.observer_alt)
            else:
                lat, lon, alt = get_observer()

            gt = match_bursts_to_satellites(
                ts_arr,
                cfo_arr,
                t0_utc,
                lat,
                lon,
                alt,
                el_min_deg=0.0,
                # Keep robust fallback in case CFO has residual offset.
                use_elevation_heuristic=True,
                verbose=True,
            )
            payload.update({
                "gt_az_deg": gt["az_deg"],
                "gt_el_deg": gt["el_deg"],
                "gt_sat_name": gt["sat_name"],
                "gt_norad_id": gt["norad_id"],
                "gt_doppler_hz": gt["doppler_hz"],
            })
            meta.update({
                "observer_lat": round(float(lat), 7),
                "observer_lon": round(float(lon), 7),
                "observer_alt_m": round(float(alt), 1),
                "ground_truth_source": gt["source"],
                "ground_truth_n_matched": int(gt["n_matched"]),
            })
        except Exception as exc:
            print(f"[COLLECT] Ground truth annotation skipped: {exc}")
            meta["ground_truth_error"] = str(exc)

    npz_path, json_path = _save_dataset(payload, meta, args.out_dir)
    print(f"[COLLECT] Saved {npz_path}")
    print(f"[COLLECT] Saved {json_path}")


if __name__ == "__main__":
    main()
