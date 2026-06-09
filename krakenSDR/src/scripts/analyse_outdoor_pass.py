#!/usr/bin/env python3
"""
Analyse an outdoor Iridium satellite pass recording.

Usage:
    python3 analyse_outdoor_pass.py data/doa_iridium/doa_iridium_20260526_155627.npz
"""
import sys, os, argparse
import numpy as np

def main():
    p = argparse.ArgumentParser(description="Analyse outdoor Iridium satellite pass")
    p.add_argument("input", help="Path to .npz recording")
    p.add_argument("--out", default="", help="Output plot path (default: auto)")
    args = p.parse_args()

    d = np.load(args.input, allow_pickle=True)
    az = d['az_deg']; el = d['el_deg']; papr = d['papr_db']; snr = d['snr_db']
    cfo = d['sat_cfo_hz']; t = d['t']; phase = d['phase_diff']
    n = len(az); dur = t[-1] - t[0] if n > 1 else 0
    freq = d['freq_hz'][0] / 1e6 if 'freq_hz' in d else 1626.27

    print(f"=== Satellite Pass Analysis ===")
    print(f"File:     {os.path.basename(args.input)}")
    print(f"Bursts:   {n} in {dur:.0f}s ({n/(dur/0.09)*100:.0f}% acceptance)")
    print(f"Freq:     {freq:.3f} MHz")
    print(f"Azimuth:  med={np.median(az):.1f}°  std={np.std(az):.1f}°  range=[{np.min(az):.0f},{np.max(az):.0f}]°")
    print(f"Elevation: med={np.median(el):.1f}°  std={np.std(el):.1f}°  range=[{np.min(el):.0f},{np.max(el):.0f}]°")
    print(f"PAPR:     med={np.median(papr):.1f} dB  max={np.max(papr):.0f} dB")
    print(f"SINR:     med={np.median(snr):.2f} dB")
    print(f"CFO:      med={np.median(cfo):.0f} Hz  std={np.std(cfo):.0f} Hz  range=[{np.min(cfo):.0f},{np.max(cfo):.0f}]")
    print(f"Phase std: {np.std(phase, axis=0).round(1)}°")

    # Find max elevation (peak of pass)
    peak_idx = np.argmax(el)
    print(f"\nPeak elevation: {el[peak_idx]:.1f}° at t={t[peak_idx]-t[0]:.0f}s  az={az[peak_idx]:.0f}°")
    print(f"CFO at peak: {cfo[peak_idx]:.0f} Hz (should be near 0 for LEO pass)")

    # CFO zero-crossing (closest approach)
    signs = np.sign(cfo)
    zero_cross = np.where(np.diff(signs) != 0)[0]
    if len(zero_cross) > 0:
        zc = zero_cross[0]
        print(f"CFO zero-crossing at t={t[zc]-t[0]:.0f}s: el={el[zc]:.1f}° az={az[zc]:.0f}°")
        print(f"  → Maximum elevation (closest approach)")

    # Elevation distribution
    print(f"\n=== Elevation Distribution ===")
    for lo, hi, lab in [(0,15,'0-15°'),(15,30,'15-30°'),(30,45,'30-45°'),(45,60,'45-60°'),(60,90,'60-90°')]:
        mask = (el >= lo) & (el < hi)
        if np.sum(mask) > 0:
            print(f"  {lab}: {np.sum(mask):4d} ({100*np.sum(mask)/n:.0f}%)  az={np.median(az[mask]):.0f}±{np.std(az[mask]):.0f}°")

    # Time evolution (10 segments)
    seg = n // 10
    print(f"\n=== Pass Evolution ===")
    for i in range(10):
        s = i*seg; e = min((i+1)*seg, n)
        if e > s:
            a = az[s:e]; el_s = el[s:e]; c = cfo[s:e]; p = papr[s:e]
            print(f"  {t[s]-t[0]:5.0f}s: az={np.median(a):.0f}±{np.std(a):.0f}°  "
                  f"el={np.median(el_s):.0f}°  CFO={np.median(c)/1000:+.1f}k  "
                  f"PAPR={np.median(p):.0f}dB  n={e-s}")

    # Save summary
    import json
    out_path = args.out if args.out else args.input.replace('.npz', '_analysis.json')
    summary = {
        "file": os.path.basename(args.input),
        "bursts": int(n), "duration_s": float(dur),
        "az_median": float(np.median(az)), "az_std": float(np.std(az)),
        "el_median": float(np.median(el)), "el_std": float(np.std(el)),
        "el_peak": float(el[peak_idx]), "t_peak_s": float(t[peak_idx] - t[0]),
        "cfo_range_hz": [float(np.min(cfo)), float(np.max(cfo))],
        "papr_median": float(np.median(papr)),
        "phase_stability": [float(x) for x in np.std(phase, axis=0)],
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {out_path}")

if __name__ == "__main__":
    main()
