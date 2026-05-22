#!/usr/bin/env python3
"""
Offline DoA processor for datasets produced by collect_iridium_burst_dataset.py.

The processing path intentionally reuses internal functions from
`iridium_burst_doa_runner.py` so results are aligned with the online burst engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core.iridium_dataset_utils import (
    bootstrap_paths,
    build_uca_cfg_from_config,
    compute_phase_coherence,
    default_offline_report_path,
    enrich_summary_with_gt_errors,
    estimate_papr_db,
    extract_tone_and_cfo,
    json_default,
    load_local_config,
    summarize_rows,
)

_HERE, _SRC, _ = bootstrap_paths(__file__, include_repo_root=False)
C = load_local_config(_HERE)

import iridium_burst_doa_runner as dib
from core.doa_uca_2d import (
    UcaConfig,
    pick_doa_peak_uca_2d,
    amplitude_normalize_channels,
    extract_pilot_tone,
)
from burst_processing import compute_mf_covariance as _compute_mf_covariance_api


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline DoA processing for 5-channel Iridium burst datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input .npz dataset path")
    p.add_argument("--algo", default=str(getattr(C, "DOA_ALGORITHM", "MUSIC")).lower(),
                   choices=["music", "capon", "bartlett", "root-music", "unitary-esprit", "mfba-music"],
                   help="DoA algorithm")
    p.add_argument("--out", default="", help="Output JSON path (default near input)")
    p.add_argument("--fd-max", type=float, default=float(getattr(C, "DOPPLER_GATE_HZ", 0.0)),
                   help="Doppler absolute gate [Hz], 0 = disabled")
    p.add_argument("--phase-coh-min", type=float, default=float(getattr(C, "PHASE_COH_MIN", 0.0)),
                   help="Phase coherence gate [0..1], 0 = disabled")
    p.add_argument("--diag-every", type=int, default=50,
                   help="Diagnostic print interval in bursts")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "bursts" not in data:
        raise SystemExit("Input file missing 'bursts' array")

    bursts = data["bursts"]
    ts_ms = data["timestamps_ms"] if "timestamps_ms" in data else np.arange(len(bursts), dtype=np.float64)
    fs = float(data["sample_rate_hz"][0]) if "sample_rate_hz" in data else float(getattr(C, "SAMPLE_RATE_HZ", dib._FS))
    freq_hz = int(data["freq_hz"][0]) if "freq_hz" in data else int(C.FREQ_HZ)

    n_bursts = int(bursts.shape[0])
    if n_bursts == 0:
        raise SystemExit("Empty dataset")

    cfg, az_grid, el_grid = build_uca_cfg_from_config(C, UcaConfig)

    tone_hz_nom = float(dib._PREAMBLE_TONE_HZ)
    pre_samples = int(data["pre_samples"][0]) if "pre_samples" in data else int(dib._PRE_SAMPLES)

    R_acc = None
    n_acc = 0

    out_rows = []
    n_accept = 0

    for i in range(n_bursts):
        Xw = bursts[i]
        if Xw.ndim != 2 or Xw.shape[0] != int(getattr(C, "N_ANTENNAS", 5)):
            continue

        if Xw.shape[1] < pre_samples:
            continue

        Xpre = Xw[:, :pre_samples]
        X0_pre = Xpre[0]

        tone_info = extract_tone_and_cfo(
            dib=dib,
            C=C,
            x_pre=X0_pre,
            fs=fs,
            tone_hz_nom=tone_hz_nom,
            min_snr_db=float(getattr(C, "DOPPLER_SNR_MIN_DB", 0.0)),
        )
        if tone_info is None:
            continue

        tone_hz = tone_info.tone_hz
        tone_snr_db = tone_info.tone_snr_db
        cfo_hz = tone_info.cfo_hz

        if args.fd_max > 0 and abs(cfo_hz) > args.fd_max:
            continue

        Xp = extract_pilot_tone(Xpre, fs, tone_hz)
        Xp = amplitude_normalize_channels(Xp)

        coh = compute_phase_coherence(Xp)
        if args.phase_coh_min > 0 and coh < args.phase_coh_min:
            continue

        if args.algo.lower() in {"mfba-music", "mfba_music", "mfba"}:
            R_inst = _compute_mf_covariance_api(
                Xw,
                fs_hz=fs,
                f0_hz=tone_hz,
                f_ref_hz=tone_hz_nom,
                nfft=int(getattr(C, "DOPPLER_NFFT", 16384)),
                energy_win=int(getattr(C, "ENERGY_WIN_SAMPLES", 256)),
                pre_len=int(getattr(C, "PRE_SAMPLES", pre_samples)),
                tap_ratio=float(getattr(C, "PILOT_TAP_RATIO", 0.6)),
                enforce_equal_taps=bool(getattr(C, "PILOT_EQUAL_TAPS", True)),
                eps=float(getattr(C, "PILOT_EPS", 1e-3)),
                papr_max_db=float(getattr(C, "PAPR_MAX_DB", 4.0)),
                w_hz=float(getattr(C, "PILOT_BW_HZ", 150.0)),
            )
        else:
            R_inst = (Xp @ Xp.conj().T) / max(1, Xp.shape[1])

        if R_acc is None:
            R_acc = R_inst
            n_acc = 1
        else:
            R_acc = (R_acc * n_acc + R_inst) / (n_acc + 1)
            n_acc += 1

        if n_acc < int(getattr(C, "MULTI_BURST_N", 3)):
            continue

        R_use = R_acc.copy()
        P = dib._run_doa_algo(
            Xp,
            R_use,
            cfg,
            args.algo,
            n_snapshots=Xp.shape[1],
        )

        if P.ndim == 2:
            az_pk, el_pk, p_norm = pick_doa_peak_uca_2d(P, az_grid, el_grid)
            az_est, el_est = float(az_pk), float(el_pk)
            peak_metric = float(np.nanmax(p_norm))
        else:
            peak_metric = float(np.nanmax(P))

        row = {
            "idx": int(i),
            "t_ms": float(ts_ms[i]),
            "cfo_hz": float(cfo_hz),
            "tone_snr_db": float(tone_snr_db),
            "phase_coherence": float(coh),
            "az_deg": float(az_est),
            "el_deg": float(el_est),
            "peak": peak_metric,
            "papr_db": float(estimate_papr_db(Xpre[0])),
        }

        if "gt_az_deg" in data and "gt_el_deg" in data:
            gt_az = data["gt_az_deg"][i]
            gt_el = data["gt_el_deg"][i]
            row["gt_az_deg"] = None if np.isnan(gt_az) else float(gt_az)
            row["gt_el_deg"] = None if np.isnan(gt_el) else float(gt_el)

        out_rows.append(row)
        n_accept += 1

        if args.diag_every > 0 and (n_accept % args.diag_every) == 0:
            print(
                f"[OFFLINE] accepted={n_accept:4d}/{n_bursts} "
                f"az={az_est:6.1f} el={el_est:5.1f} cfo={cfo_hz:+8.1f}Hz coh={coh:.2f}"
            )

    if not out_rows:
        raise SystemExit("No bursts accepted by offline pipeline")

    out_path = args.out if args.out else default_offline_report_path(args.input, args.algo)
    report = {
        "tool": "process_iridium_burst_dataset_offline",
        "input": os.path.abspath(args.input),
        "algo": args.algo,
        "freq_hz": int(freq_hz),
        "sample_rate_hz": float(fs),
        "n_input_bursts": int(n_bursts),
        "n_output_rows": int(len(out_rows)),
        "rows": out_rows,
        "summary": summarize_rows(out_rows),
    }

    if any(("gt_az_deg" in r and r["gt_az_deg"] is not None) for r in out_rows):
        report["summary"] = enrich_summary_with_gt_errors(report["summary"], out_rows)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_default)

    print(f"[OFFLINE] Saved {out_path}")
    print(f"[OFFLINE] rows={len(out_rows)} / input={n_bursts}")


if __name__ == "__main__":
    main()
