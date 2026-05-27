#!/usr/bin/env python3
"""Offline comparison of DoA algorithms on saved iridium_burst_doa_runner NPZ files."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.iridium_dataset_utils import load_local_config

C = load_local_config(_HERE)
import iridium_burst_doa_runner as dib
from core.doa_uca_2d import (
    UcaConfig,
    pick_doa_peak_uca_2d,
    find_peak_uca_2d,
    doa_phase_fit_uca_2d,
)


def _circular_std_deg(az_deg: np.ndarray) -> float:
    ph = np.exp(1j * np.deg2rad(az_deg))
    m = np.abs(np.mean(ph))
    if m < 1e-9:
        return 180.0
    return float(np.degrees(np.sqrt(-2.0 * np.log(m))))


def _segment_stats(az: np.ndarray, el: np.ndarray, t: np.ndarray) -> list[dict]:
    out = []
    n = len(az)
    for k in range(4):
        i0 = k * n // 4
        i1 = (k + 1) * n // 4
        seg_az = az[i0:i1]
        seg_el = el[i0:i1]
        out.append({
            "quarter": k + 1,
            "t_start_s": float(t[i0] - t[0]),
            "t_end_s": float(t[i1 - 1] - t[0]),
            "az_median_deg": float(np.median(seg_az)),
            "az_std_deg": float(_circular_std_deg(seg_az)),
            "el_median_deg": float(np.median(seg_el)),
            "el_std_deg": float(np.std(seg_el)),
        })
    return out


def _estimate_series(
    R_all: np.ndarray,
    cfg: UcaConfig,
    algo: str,
    multi: int,
    *,
    indoor: bool,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az_out, el_out, papr_out = [], [], []
    for i in range(multi, len(R_all), step):
        R = np.mean(R_all[i - multi:i], axis=0)
        ph = np.degrees(np.angle(R[1:, 0]))
        try:
            if algo == "phase-fit":
                az, el, err = doa_phase_fit_uca_2d(R, cfg)
                papr = max(0.0, 12.0 - err)
            else:
                spec = dib._run_doa_algo(
                    np.zeros((cfg.n_ant, 1), dtype=np.complex128),
                    R, cfg, algo, n_snapshots=multi * 500,
                )
                if spec.ndim != 2:
                    az, el, papr = find_peak_uca_2d(spec, cfg)
                else:
                    az, el, papr = pick_doa_peak_uca_2d(
                        spec, cfg, indoor=indoor, phase_diffs=ph,
                        az_hint_deg=None, el_hint_deg=None,
                        az_hint_score_weight=0.0, el_hint_score_weight=0.0,
                    )
            az_off = float(getattr(C, "DOA_AZ_OFFSET_DEG", 0.0))
            el_off = float(getattr(C, "DOA_EL_OFFSET_DEG", 0.0))
            az = (float(az) + az_off) % 360.0
            el = float(np.clip(float(el) + el_off, cfg.el_min_deg, cfg.el_max_deg))
            az_out.append(az)
            el_out.append(el)
            papr_out.append(float(papr))
        except Exception:
            az_out.append(np.nan)
            el_out.append(np.nan)
            papr_out.append(np.nan)
    return np.array(az_out), np.array(el_out), np.array(papr_out)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare DoA algorithms on runner NPZ")
    p.add_argument("--input", required=True, help="Path to doa_iridium_*.npz")
    p.add_argument("--multi", type=int, default=int(getattr(C, "MULTI_BURST_N", 32)))
    p.add_argument("--step", type=int, default=4, help="Subsampling step between estimates")
    p.add_argument("--out", default="", help="Output JSON path")
    args = p.parse_args()

    d = np.load(args.input, allow_pickle=True)
    if "R" not in d:
        raise SystemExit("NPZ missing 'R' — need recording from iridium_burst_doa_runner")

    el_max = min(float(C.EL_MAX_DEG), float(getattr(C, "INDOOR_EL_MAX_DEG", C.EL_MAX_DEG)))
    cfg = UcaConfig(
        n_ant=C.N_ANTENNAS, radius_lambda=C.RADIUS_LAMBDA,
        n_az=C.N_AZ, n_el=C.N_EL,
        el_min_deg=C.EL_MIN_DEG, el_max_deg=el_max,
        num_expected_signals=C.NUM_SIGNALS,
        ant0_offset_deg=C.ANT0_OFFSET_DEG, ant_ccw=C.ANT_CCW,
    )

    R_all = d["R"]
    t_rec = d["t"] if "t" in d else np.arange(len(R_all), dtype=float)
    live_az = d["az_deg"] if "az_deg" in d else None

    algos = ["music", "capon", "bartlett", "phase-fit", "root-music", "unitary-esprit", "mfba-music"]
    report = {
        "input": os.path.abspath(args.input),
        "n_R": int(len(R_all)),
        "multi_burst_n": int(args.multi),
        "music_decorr": str(getattr(C, "MUSIC_DECORR", "none")),
        "live_recorded": {},
        "algorithms": {},
    }

    if live_az is not None:
        report["live_recorded"] = {
            "az_median_deg": float(np.median(live_az)),
            "az_std_deg": float(_circular_std_deg(live_az)),
            "el_median_deg": float(np.median(d["el_deg"])),
            "el_std_deg": float(np.std(d["el_deg"])),
            "segments": _segment_stats(live_az, d["el_deg"], t_rec),
        }

    best_responsive = ("", -1.0)
    for algo in algos:
        az, el, papr = _estimate_series(
            R_all, cfg, algo, args.multi, indoor=True, step=args.step,
        )
        valid = np.isfinite(az)
        entry = {"n_valid": int(valid.sum()), "status": "ok"}
        if valid.sum() < 5:
            entry["status"] = "failed"
            report["algorithms"][algo] = entry
            continue
        azv, elv = az[valid], el[valid]
        seg_t = np.linspace(0, 1, len(azv))
        t_proxy = seg_t * (float(t_rec[-1] - t_rec[0]) if len(t_rec) > 1 else 1.0)
        # responsiveness: combined az+el variation across quarters
        segs = _segment_stats(azv, elv, t_proxy)
        az_spread = max(s["az_median_deg"] for s in segs) - min(s["az_median_deg"] for s in segs)
        az_spread = min(az_spread, 360.0 - az_spread)
        el_spread = max(s["el_median_deg"] for s in segs) - min(s["el_median_deg"] for s in segs)
        responsiveness = float(az_spread + el_spread)
        entry.update({
            "az_median_deg": float(np.median(azv)),
            "az_std_deg": float(_circular_std_deg(azv)),
            "az_range_deg": [float(np.min(azv)), float(np.max(azv))],
            "el_median_deg": float(np.median(elv)),
            "el_std_deg": float(np.std(elv)),
            "el_range_deg": [float(np.min(elv)), float(np.max(elv))],
            "papr_median_db": float(np.median(papr[valid])),
            "segment_drift": segs,
            "responsiveness_score": responsiveness,
        })
        report["algorithms"][algo] = entry
        if responsiveness > best_responsive[1]:
            best_responsive = (algo, responsiveness)

    report["recommendation"] = {
        "best_responsive_algo": best_responsive[0],
        "note": (
            "music/capon/bartlett are nearly identical on UCA. "
            "phase-fit follows motion faster when calibrated. "
            "root-music/unitary-esprit/mfba-music are experimental/broken on this HW."
        ),
    }

    out_path = args.out or args.input.replace(".npz", "_algo_compare.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["recommendation"], indent=2))
    print(f"\nFull report: {out_path}")
    for algo, e in report["algorithms"].items():
        if e.get("status") != "ok":
            print(f"  {algo:16s} FAILED")
            continue
        print(
            f"  {algo:16s} az={e['az_median_deg']:6.1f}±{e['az_std_deg']:5.1f}  "
            f"el={e['el_median_deg']:5.1f}±{e['el_std_deg']:4.1f}  "
            f"resp={e['responsiveness_score']:5.1f}"
        )


if __name__ == "__main__":
    main()
