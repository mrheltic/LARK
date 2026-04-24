#!/usr/bin/env python3
"""
_test_gt_matching.py  — one-shot CLI test for the GT Doppler matching pipeline.

Connects to Heimdall, collects 60 s of bursts (or N_LIMIT), runs
compensate_doppler() on each accepted burst, then compares the measured
Doppler against TLE-predicted Doppler for every currently-visible Iridium
satellite.  Prints a summary so you can verify the threshold (8 kHz) works.

Usage
-----
    python3 apps/space/_test_gt_matching.py
    python3 apps/space/_test_gt_matching.py --duration 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import collections

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(os.path.dirname(_SRC))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

import numpy as np
import config as C
from hardware.kraken_iq_source import KrakenIQSource
from core.iridium_doa_burst import detect_and_extract_all_bursts, compensate_doppler

# ── TLE / observer ─────────────────────────────────────────────────────────
def _load_tle_cat():
    try:
        from shared.iridium_tle import load_catalogue
        cat = load_catalogue()
        print(f"[TLE] Loaded {len(cat.satellites)} Iridium satellites")
        return cat
    except Exception as e:
        print(f"[TLE] FAILED to load catalogue: {e}")
        return None

def _get_observer():
    lat = float(os.environ.get("LARK_LAT", "43.5"))
    lon = float(os.environ.get("LARK_LON", "7.1"))
    alt = float(os.environ.get("LARK_ALT", "0.0"))
    return lat, lon, alt

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0, help="Test duration [s]")
    ap.add_argument("--freq", type=float, default=C.FREQ_HZ, help="Centre freq [Hz]")
    ap.add_argument("--gain", type=float, default=float(C.GAIN_DB))
    ap.add_argument("--threshold", type=float, default=8.0, help="GT Doppler threshold [kHz]")
    args = ap.parse_args()

    GT_THRESH = args.threshold * 1000.0  # → Hz

    print(f"\n{'='*65}")
    print(f"  GT matching test  |  freq={args.freq/1e6:.3f} MHz  gain={args.gain:.0f} dB")
    print(f"  duration={args.duration:.0f} s  GT_threshold={args.threshold:.1f} kHz")
    print(f"{'='*65}\n")

    # Load TLE once
    cat = _load_tle_cat()
    lat, lon, alt = _get_observer()
    print(f"[OBS] Observer: lat={lat}° lon={lon}° alt={alt} m\n")

    # Initial visible satellites
    if cat:
        vis = cat.visible_now(lat, lon, alt, el_min_deg=5.0)
        if vis:
            print(f"[SATS] {len(vis)} visible at start:")
            for sv in vis:
                print(f"       {sv['name']:20s}  az={sv['az_deg']:5.1f}°  el={sv['el_deg']:4.1f}°  "
                      f"dop={sv['doppler_hz']/1e3:+6.2f} kHz  range={sv['range_km']:.0f} km")
        else:
            print("[SATS] No satellites visible!")
        print()

    # Connect KrakenIQ
    kraken = KrakenIQSource(
        host=C.HEIMDALL_HOST,
        port=C.HEIMDALL_PORT,
        num_channels=C.N_ANTENNAS,
        freq_hz=int(args.freq),
        gain_db=args.gain,
    )
    kraken.start()
    time.sleep(2.0)
    print(f"[HW] Heimdall connected: {kraken.is_connected}")
    print()

    # Tracking
    bursts_raw      = 0
    bursts_accepted = 0
    bursts_ring     = 0   # bursts on the ring-alert FDMA channel (|CFO| < 50 kHz)
    bursts_fdma_adj = 0   # bursts on adjacent FDMA channels
    gt_matched   = 0
    by_sat: dict[str, int] = collections.defaultdict(int)
    mismatch_samples: list[float] = []   # |measured - best_tle| Hz on non-matches

    t_end      = time.time() + args.duration
    t_last_tle = 0.0
    vis_sats   = vis if cat else []

    WARMUP = 3
    warmup = 0

    try:
        while time.time() < t_end:
            frame = kraken.get_frame(timeout=0.3)
            if frame is None:
                continue

            if warmup < WARMUP:
                warmup += 1
                continue

            X = frame.astype(np.complex128)
            if hasattr(C, "HW_NUM_SAMPLES") and C.HW_NUM_SAMPLES > 0:
                X = X[:, :C.HW_NUM_SAMPLES]

            # Refresh TLE Doppler every 15 s
            if cat and (time.time() - t_last_tle) > 15.0:
                try:
                    vis_sats = cat.visible_now(lat, lon, alt, el_min_deg=5.0)
                    t_last_tle = time.time()
                except Exception:
                    pass

            # Burst detection on Channel 0
            bursts_found = detect_and_extract_all_bursts(
                X, threshold_db=float(C.THRESHOLD_DB if hasattr(C, "THRESHOLD_DB") else 8.0),
                sample_rate=int(C.SAMPLE_RATE_HZ)
            )

            for bi in bursts_found:
                bursts_raw += 1
                try:
                    comp, f_dop = compensate_doppler(bi, sample_rate=int(C.SAMPLE_RATE_HZ))
                except Exception:
                    continue
                bursts_accepted += 1

                # Ring-alert guard: compensate_doppler returns total CFO =
                # f_FDMA_channel + f_Doppler.  For 1626.270 MHz (ring-alert) bursts
                # f_FDMA ≈ 0 so |f_dop| < 50 kHz.  Bursts at ±185/±313 kHz are on
                # adjacent Iridium FDMA channels — skip GT matching for those.
                RING_BW = 50_000  # Hz
                if abs(f_dop) >= RING_BW:
                    bursts_fdma_adj += 1
                    print(f"  [FDMA] f_dop={f_dop/1e3:+7.2f} kHz  → adjacent channel, skip GT")
                    continue

                bursts_ring += 1

                # GT matching (ring-alert bursts only)
                if vis_sats:
                    bs = min(vis_sats, key=lambda s: abs(s["doppler_hz"] - f_dop))
                    diff = abs(bs["doppler_hz"] - f_dop)
                    if diff < GT_THRESH:
                        gt_matched += 1
                        by_sat[bs["name"]] += 1
                        print(f"  [GT ✓] f_dop={f_dop/1e3:+7.2f} kHz  TLE={bs['doppler_hz']/1e3:+7.2f} kHz"
                              f"  Δ={diff/1e3:.2f} kHz  → {bs['name']}")
                    else:
                        mismatch_samples.append(diff / 1e3)
                        print(f"  [GT ✗] f_dop={f_dop/1e3:+7.2f} kHz  closest TLE={bs['doppler_hz']/1e3:+7.2f} kHz"
                              f"  Δ={diff/1e3:.2f} kHz  (>{args.threshold:.0f} kHz threshold)")
                else:
                    print(f"  [GT ?] f_dop={f_dop/1e3:+7.2f} kHz  (no visible sats)")

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        kraken.stop()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = args.duration if time.time() >= t_end else (args.duration - (t_end - time.time()))
    print(f"\n{'='*65}")
    print(f"  SUMMARY  ({elapsed:.0f} s)")
    print(f"{'='*65}")
    print(f"  Bursts detected  : {bursts_raw}")
    print(f"  Bursts accepted  : {bursts_accepted}  (after compensate_doppler)")
    print(f"  Ring-alert bursts: {bursts_ring}  (|CFO| < 50 kHz — GT-eligible)")
    print(f"  Adjacent FDMA    : {bursts_fdma_adj}  (|CFO| ≥ 50 kHz — DoA ok, GT skipped)")
    print(f"  GT matched       : {gt_matched}/{bursts_ring}"
          f"  ({100*gt_matched//bursts_ring if bursts_ring else 0}%)")
    if by_sat:
        print(f"  By satellite     :")
        for name, cnt in sorted(by_sat.items(), key=lambda x: -x[1]):
            print(f"    {name:22s} : {cnt:4d} bursts")
    if mismatch_samples:
        print(f"  Mismatch samples : n={len(mismatch_samples)}"
              f"  median Δ={np.median(mismatch_samples):.2f} kHz"
              f"  max Δ={max(mismatch_samples):.2f} kHz")
    if not vis_sats:
        print("  WARNING: No satellites were visible — GT cannot match!")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
