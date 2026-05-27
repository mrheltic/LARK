#!/usr/bin/env python3
"""
Reprocess raw IQ windows from an Iridium recording with improved parameters
for outdoor satellite tracking.

Usage:
    python3 reprocess_iq.py data/doa_iridium/doa_iridium_20260526_155627_iq.npz
"""
import sys, os, argparse, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

import config as C
from core.doa_uca_2d import (
    UcaConfig, doa_music_uca_2d, doa_capon_uca_2d, doa_bartlett_uca_2d,
    pick_doa_peak_uca_2d,
    extract_pilot_tone, amplitude_normalize_channels,
)
from core.doa_algorithms import apply_phase_correction as _apply_phase_correction
from burst_processing import compute_mf_covariance as _compute_mf_covariance_api
from core.tone_extraction import find_preamble_onset as _fpo_core


def main():
    p = argparse.ArgumentParser(description="Reprocess raw IQ windows with better params")
    p.add_argument("input", help="Path to _iq.npz file")
    p.add_argument("--algo", default="music", choices=["music","capon","bartlett"],
                   help="DoA algorithm")
    p.add_argument("--multi", type=int, default=40, help="Multi-burst accumulation")
    p.add_argument("--snr-min", type=float, default=-5, help="SINR gate dB")
    p.add_argument("--papr-min", type=float, default=3, help="PAPR gate dB")
    p.add_argument("--out", default="", help="Output .npz path (auto if empty)")
    args = p.parse_args()

    d = np.load(args.input, allow_pickle=True)
    # Accept both the old _iq.npz key ('X') and the merged runner key ('bursts')
    X_all = d['bursts'] if 'bursts' in d else d['X']  # (N, n_ant, pre_samples)
    n_total, n_ant, pre_samples = X_all.shape
    print(f"Loaded {n_total} IQ windows ({n_ant} antennas, {pre_samples} samples)")

    # ── Config ────────────────────────────────────────────────────────────
    fs = float(getattr(C, "SAMPLE_RATE_HZ", 1_024_000))
    freq_hz = int(getattr(C, "FREQ_HZ", 1_626_270_000))
    tone_nom = 3125.0
    bpf_bw = float(getattr(C, "PREAMBLE_BPF_BW_HZ", 8_000))
    bpf_guard = max(64, int(np.ceil(fs / bpf_bw)))
    n_pre = pre_samples - bpf_guard
    phase_offs = list(getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0]*n_ant))
    phase_offs = (phase_offs + [0]*n_ant)[:n_ant]
    has_cal = any(o != 0.0 for o in phase_offs)

    cfg = UcaConfig(
        n_ant=n_ant,
        radius_lambda=float(getattr(C, "RADIUS_LAMBDA", 0.4253)),
        n_az=int(getattr(C, "N_AZ", 360)),
        n_el=int(getattr(C, "N_EL", 86)),
        el_min_deg=float(getattr(C, "EL_MIN_DEG", 5)),
        el_max_deg=float(getattr(C, "EL_MAX_DEG", 90)),
        num_expected_signals=int(getattr(C, "NUM_SIGNALS", 0)),
        ant0_offset_deg=float(getattr(C, "ANT0_OFFSET_DEG", 0)),
        ant_ccw=bool(getattr(C, "ANT_CCW", False)),
    )

# ── Process ─────────────────────────────────────────────────────────────
    X_batch = []
    R_batch = []
    doppler_gate = float(getattr(C, "DOPPLER_GATE_HZ", 0))

    # Use CFO from the same file if available (merged runner format), else look for companion
    orig_cfo = None
    if 'sat_cfo_hz' in d and len(d['sat_cfo_hz']) == n_total:
        orig_cfo = d['sat_cfo_hz']
        print("Using CFO from dataset file (sat_cfo_hz)")
    else:
        orig_path = args.input.replace('_iq.npz', '.npz')
        if os.path.exists(orig_path) and orig_path != args.input:
            orig = np.load(orig_path, allow_pickle=True)
            if 'sat_cfo_hz' in orig and len(orig['sat_cfo_hz']) == n_total:
                orig_cfo = orig['sat_cfo_hz']
                print(f"Using original CFO from {os.path.basename(orig_path)}")

    out_az, out_el, out_papr, out_snr, out_cfo = [], [], [], [], []
    out_t, out_phase, out_R = [], [], []

    print(f"Processing {n_total} windows (multi={args.multi}, algo={args.algo})...")
    t0 = time.time()

    for i in range(n_total):
        Xw = X_all[i]  # (5, pre_samples)
        X0_pre = Xw[0]

        # ── Tone detection ────────────────────────────────────────────────
        if orig_cfo is not None:
            cfo_hz = float(orig_cfo[i])
            tone_hz = tone_nom + cfo_hz
        else:
            # Fallback: preamble onset refinement
            onset_ref, tone_hz = _fpo_core(
                X0_pre, 0, pre_samples, fs=fs, win=4096, known_hz=None,
                burst_samples=pre_samples, preamble_tone_hz=tone_nom,
                freq_lock_bw=45000,
            )

        cfo_hz = tone_hz - tone_nom
        if doppler_gate > 0 and abs(cfo_hz) > doppler_gate:
            continue

        # ── BPF + amplitude normalisation ─────────────────────────────────
        X_bpf = extract_pilot_tone(Xw[:pre_samples][np.newaxis,:,:].squeeze() if Xw.ndim==2 else Xw, fs, tone_hz, bpf_bw)
        if X_bpf.ndim == 1: X_bpf = X_bpf.reshape(1, -1)
        X_bpf = X_bpf[:n_ant, :]
        X_bpf = amplitude_normalize_channels(X_bpf)

        if has_cal:
            X_cal = _apply_phase_correction(X_bpf, phase_offs)
        else:
            X_cal = X_bpf

        X_pre = X_cal[:, bpf_guard:bpf_guard+n_pre]

        # ── MF covariance ─────────────────────────────────────────────────
        R_inst, y_mf, snr_db = _compute_mf_covariance_api(
            X_cal, tone_hz, fs, n_pre, bpf_guard,
        )

        if snr_db < args.snr_min:
            continue

        # ── Accumulate (no phase-align, CFO is known) ─────────────────────
        X_batch.append(X_pre)
        R_batch.append(R_inst)

        if len(X_batch) < args.multi:
            continue

        # ── DoA estimate ──────────────────────────────────────────────────
        X_big = np.hstack(X_batch)
        R_avg = (X_big @ X_big.conj().T) / X_big.shape[1]
        phase_diffs = np.degrees(np.angle(R_avg[1:, 0]))

        if args.algo == "capon":
            spec2d = doa_capon_uca_2d(X_cal, cfg, R_in=R_avg)
        elif args.algo == "bartlett":
            spec2d = doa_bartlett_uca_2d(X_cal, cfg, R_in=R_avg)
        else:
            spec2d = doa_music_uca_2d(X_cal, cfg, R_in=R_avg, n_snapshots=X_big.shape[1])

        az_doa, el_doa, papr_doa = pick_doa_peak_uca_2d(
            spec2d, cfg, indoor=False,
            el_pref_hi=35, el_pref_lo=5,
        )

        if papr_doa < args.papr_min:
            # keep batch but skip this estimate
            X_batch.pop(0); R_batch.pop(0)
            continue

        out_t.append(float(i))
        out_az.append(float(az_doa)); out_el.append(float(el_doa))
        out_papr.append(float(papr_doa)); out_snr.append(float(snr_db))
        out_cfo.append(float(tone_hz - tone_nom))
        out_phase.append(phase_diffs)
        out_R.append(R_avg)

        # Slide window
        X_batch.pop(0); R_batch.pop(0)

        if len(out_az) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{len(out_az):4d}] az={az_doa:.0f}° el={el_doa:.0f}° "
                  f"papr={papr_doa:.0f}dB sinr={snr_db:.1f}dB cfo={tone_hz-tone_nom:.0f}Hz "
                  f"({elapsed:.0f}s)")

    # ── Save ────────────────────────────────────────────────────────────────
    n_est = len(out_az)
    if n_est == 0:
        print("ERROR: No estimates produced. Try --snr-min -10 or --papr-min 1")
        sys.exit(1)

    _default_out = args.input.replace('_iq.npz', '_reprocessed.npz').replace('.npz', '_reprocessed.npz') if '_reprocessed' not in args.input else args.input
    out_path = args.out if args.out else _default_out
    np.savez_compressed(
        out_path,
        az_deg=np.array(out_az), el_deg=np.array(out_el),
        papr_db=np.array(out_papr), snr_db=np.array(out_snr),
        sat_cfo_hz=np.array(out_cfo), t=np.array(out_t, dtype=np.float64),
        phase_diff=np.array(out_phase), R=np.array(out_R),
        freq_hz=np.array([freq_hz]),
    )
    elapsed = time.time() - t0
    print(f"\nSaved {n_est} estimates to {out_path}")
    print(f"Duration: {elapsed:.0f}s  "
          f"Az: {np.median(out_az):.0f}±{np.std(out_az):.0f}°  "
          f"El: {np.median(out_el):.0f}±{np.std(out_el):.0f}°  "
          f"PAPR: {np.median(out_papr):.0f}dB")

if __name__ == "__main__":
    main()
