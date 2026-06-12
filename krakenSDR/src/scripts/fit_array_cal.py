#!/usr/bin/env python3
"""
fit_array_cal.py — Array self-calibration using Iridium satellites as beacons.

Estimates the per-channel phase offsets (cable/receiver mismatch) and the
mechanical rotation of the array (ant0_offset_deg) directly from a recorded
session, using TLE-predicted satellite directions as ground truth — no
reference transmitter needed, and the calibration is at the operating
frequency (1626 MHz) by construction.

Method
------
1. For every raw CPI frame, detect bursts and extract the matched-filter
   array response y_mf per preamble tone (NO phase calibration applied).
2. Assign each burst tone to a satellite by Doppler: the tone CFO must match
   the SGP4-predicted Doppler of exactly one visible satellite (after
   removing the receiver LO offset, estimated as the median residual).
3. The expected steering vector a(az, el) at the TLE direction is compared
   with y_mf: the per-channel phase residual (relative to antenna 0) should
   be constant across bursts if the array is calibrated.
4. Grid search over the array rotation; at each candidate the per-channel
   offsets are the circular means of the residuals, and the cost is the
   residual circular spread.  Needs bursts spread over several azimuths
   (≥ 2-3 different passes) to separate rotation from per-channel offsets.

Output: <session>/cal_tle.npz with "phase_offsets_deg" (compatible with
use_phase_cal/cal_file in doa_config.toml) + fitted ant0_offset_deg report.

Usage:
    python3 scripts/fit_array_cal.py session_20260605_110716/
    python3 scripts/fit_array_cal.py session_dir/ --snr-min 8 --dopp-tol 2000
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)                       # krakenSDR/src/
_ROOT = os.path.dirname(os.path.dirname(_SRC))      # LARK project root
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.burst_processing import (  # noqa: E402
    apply_bpf_and_normalize,
    compute_mf_covariance,
    detect_energy_bursts,
    scan_preamble_tones,
)
from core.recording import (  # noqa: E402
    count_session_raw_frames,
    iter_session_raw_frames,
    session_frame_times,
)
from scripts.iridium_groundtruth import (  # noqa: E402
    OBSERVER_ALT,
    OBSERVER_LAT,
    OBSERVER_LON,
    load_session_window,
    parse_isotime,
)
from shared.iridium_tle import use_session_tle  # noqa: E402

FS = 1_024_000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit array phase offsets + rotation from Iridium TLE ground truth",
    )
    p.add_argument("session_dir", help="Path to session_YYYYMMDD_HHMMSS/")
    p.add_argument("--config", default=os.path.join(
        _SRC, "apps", "doa_iridium", "doa_config.toml"))
    p.add_argument("--mode", choices=["indoor", "outdoor"], default="outdoor")
    p.add_argument("--lat", type=float, default=OBSERVER_LAT)
    p.add_argument("--lon", type=float, default=OBSERVER_LON)
    p.add_argument("--alt", type=float, default=OBSERVER_ALT)
    p.add_argument("--snr-min", type=float, default=8.0,
                   help="Min burst SINR [dB] — keep only clean calibration bursts")
    p.add_argument("--el-min", type=float, default=10.0,
                   help="Min satellite elevation [deg] for calibration bursts")
    p.add_argument("--dopp-tol", type=float, default=2_000.0,
                   help="Doppler match tolerance [Hz] for satellite assignment")
    p.add_argument("--lo-offset", type=float, default=None,
                   help="Receiver LO offset [Hz] (default: auto-estimate)")
    p.add_argument("--rot-step", type=float, default=1.0,
                   help="Rotation grid step [deg]")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--out", metavar="PATH",
                   help="Output .npz (default: <session>/cal_tle.npz)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Burst measurement extraction
# ─────────────────────────────────────────────────────────────────────────────

def collect_burst_measurements(
    session_dir: str,
    cfg: dict,
    profile: dict,
    args: argparse.Namespace,
) -> list[dict]:
    """Extract (t_rel, cfo_hz, y_mf, snr_db) per burst tone, uncalibrated."""
    hw = cfg["hardware"]
    pre_samples = hw["pre_samples"]
    window_samples = hw["window_samples"]
    bpf_guard = hw["bpf_guard"]
    cpi_size = hw.get("cpi_size", 131072)

    total = count_session_raw_frames(session_dir)
    max_frames = args.max_frames if args.max_frames > 0 else 0
    # Wall-clock per frame (live recording drops CPIs → index-based timing
    # drifts by minutes and would break the TLE alignment).
    frame_ts = session_frame_times(session_dir)
    print(f"Scanning {total} raw CPI frames for calibration bursts…")

    out: list[dict] = []
    for fi, X in iter_session_raw_frames(
        session_dir, start=args.frame_start,
        stride=max(1, args.frame_stride), max_frames=max_frames,
    ):
        starts = detect_energy_bursts(
            X, FS, threshold_factor=profile["energy_threshold"],
        )
        for b0 in starts:
            bend = min(b0 + window_samples, X.shape[1])
            if bend - b0 < pre_samples + bpf_guard:
                continue
            tones = scan_preamble_tones(
                X[:, b0:bend], FS,
                nom_tone_hz=profile["tone_nom_hz"],
                scan_bw_hz=profile["scan_bw_hz"],
                min_snr_db=profile["min_snr_db"],
                dc_guard_hz=profile["dc_guard_hz"],
            )
            X_win = X[:, b0:bend]
            for tone_hz, _snr in tones:
                try:
                    X_bpf = apply_bpf_and_normalize(
                        X_win, min(window_samples, X_win.shape[1]),
                        FS, tone_hz, profile["bpf_bw_hz"],
                    )
                    n_pre_eff = min(pre_samples, X_bpf.shape[1] - bpf_guard)
                    if n_pre_eff < 512:
                        continue
                    _, y_mf, snr_db = compute_mf_covariance(
                        X_bpf, tone_hz, FS, n_pre_eff, bpf_guard,
                    )
                except ValueError:
                    continue
                if snr_db < args.snr_min:
                    continue
                if frame_ts is not None and fi < len(frame_ts):
                    t_rel = float(frame_ts[fi] - frame_ts[0]) + b0 / FS
                else:
                    t_rel = (fi * cpi_size + b0) / FS
                out.append({
                    "t_rel": t_rel,
                    "cfo_hz": float(tone_hz - profile["tone_nom_hz"]),
                    "y_mf": y_mf,
                    "snr_db": float(snr_db),
                })
        if args.verbose and (fi + 1) % 500 == 0:
            print(f"  … frame {fi + 1}/{total}, {len(out)} bursts", file=sys.stderr)

    print(f"Collected {len(out)} burst tones with SINR ≥ {args.snr_min:.0f} dB")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TLE ground truth + satellite assignment
# ─────────────────────────────────────────────────────────────────────────────

def build_sat_tracks(t0, t1, args) -> list[dict]:
    """High-res (1 s) az/el/Doppler tracks for all passes in the window."""
    from shared.iridium_tle import load_catalogue

    cat = load_catalogue()
    passes = cat.predict_passes(
        lat=args.lat, lon=args.lon, alt=args.alt,
        t0_utc=t0, t1_utc=t1, el_min_deg=args.el_min, step_s=20.0,
    )
    print(f"Predicted passes (el ≥ {args.el_min:.0f}°): {len(passes)}")

    tracks = []
    for s in passes:
        rise, sset = parse_isotime(s["rise_utc"]), parse_isotime(s["set_utc"])
        trk = cat.doppler_track(
            s["name"], args.lat, args.lon, args.alt,
            rise, sset, freq_hz=1_626_270_000.0, step_s=1.0,
        )
        n = len(trk["doppler_hz"])
        dur = (sset - rise).total_seconds()
        tracks.append({
            "name": s["name"],
            "t_rel": (rise - t0).total_seconds() + np.linspace(0.0, dur, n),
            "doppler_hz": trk["doppler_hz"],
            "az_deg": trk["az_deg"],
            "el_deg": trk["el_deg"],
        })
    return tracks


def _interp_track(trk: dict, t: float) -> tuple[float, float, float] | None:
    """(doppler, az, el) of one satellite track at relative time t, or None."""
    tr = trk["t_rel"]
    if t < tr[0] or t > tr[-1]:
        return None
    dop = float(np.interp(t, tr, trk["doppler_hz"]))
    el = float(np.interp(t, tr, trk["el_deg"]))
    # Azimuth needs circular interpolation
    az_u = np.unwrap(np.deg2rad(trk["az_deg"]))
    az = float(np.rad2deg(np.interp(t, tr, az_u))) % 360.0
    return dop, az, el


def assign_satellites(
    bursts: list[dict],
    tracks: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    """Attach (az_true, el_true) to bursts whose CFO uniquely matches one satellite."""
    # LO offset: median residual to the nearest predicted Doppler (loose gate).
    if args.lo_offset is not None:
        lo = float(args.lo_offset)
    else:
        resid = []
        for b in bursts:
            cands = [_interp_track(trk, b["t_rel"]) for trk in tracks]
            diffs = [b["cfo_hz"] - c[0] for c in cands if c is not None]
            if diffs:
                d = min(diffs, key=abs)
                if abs(d) < 5_000.0:
                    resid.append(d)
        lo = float(np.median(resid)) if len(resid) >= 10 else 0.0
        print(f"LO offset estimate: {lo:+.0f} Hz "
              f"({lo / 1626.27:+.2f} ppm, from {len(resid)} bursts)")

    assigned = []
    for b in bursts:
        cfo = b["cfo_hz"] - lo
        matches = []
        for trk in tracks:
            c = _interp_track(trk, b["t_rel"])
            if c is None:
                continue
            dop, az, el = c
            if abs(cfo - dop) < args.dopp_tol:
                matches.append((abs(cfo - dop), trk["name"], az, el))
        if len(matches) != 1:        # require unambiguous assignment
            continue
        _, name, az, el = matches[0]
        assigned.append({**b, "sat": name, "az_true": az, "el_true": el})

    n_sats = len({b["sat"] for b in assigned})
    print(f"Assigned {len(assigned)} bursts to {n_sats} satellites "
          f"(unique Doppler match within ±{args.dopp_tol:.0f} Hz)")
    return assigned


# ─────────────────────────────────────────────────────────────────────────────
# Phase-offset + rotation fit
# ─────────────────────────────────────────────────────────────────────────────

def _steering(az_deg, el_deg, *, n_ant, radius_lambda, ant0_offset_deg, ant_ccw):
    """Steering vectors (B, n_ant) — same convention as UcaConfig.positions."""
    k = np.arange(n_ant, dtype=np.float64)
    sign = -1.0 if ant_ccw else 1.0
    phi = np.deg2rad(ant0_offset_deg) + sign * 2.0 * np.pi * k / n_ant
    p_e = radius_lambda * np.sin(phi)            # (n_ant,)
    p_n = radius_lambda * np.cos(phi)
    az = np.deg2rad(np.atleast_1d(az_deg))[:, None]   # (B, 1)
    el = np.deg2rad(np.atleast_1d(el_deg))[:, None]
    tau = 2.0 * np.pi * (p_e[None, :] * np.cos(el) * np.sin(az)
                         + p_n[None, :] * np.cos(el) * np.cos(az))
    return np.exp(1j * tau)                       # (B, n_ant)


def fit_offsets_and_rotation(
    assigned: list[dict],
    *,
    n_ant: int,
    radius_lambda: float,
    ant_ccw: bool,
    rot_step_deg: float = 1.0,
) -> dict:
    """
    Grid search over array rotation; per-channel offsets by weighted circular
    mean of the residual phasors relative to antenna 0.
    """
    Y = np.stack([b["y_mf"] for b in assigned])               # (B, n_ant)
    az = np.array([b["az_true"] for b in assigned])
    el = np.array([b["el_true"] for b in assigned])
    w = np.minimum(10.0 ** (np.array([b["snr_db"] for b in assigned]) / 10.0), 100.0)
    w = w[:, None] / np.sum(w)

    best = None
    for rot in np.arange(-180.0, 180.0, rot_step_deg):
        A = _steering(az, el, n_ant=n_ant, radius_lambda=radius_lambda,
                      ant0_offset_deg=rot, ant_ccw=ant_ccw)
        Z = Y * np.conj(A)                                    # residual phasors
        Z = Z * np.conj(Z[:, :1])                             # relative to ant 0
        mag = np.abs(Z)
        Z = np.where(mag > 1e-20, Z / np.maximum(mag, 1e-20), 0.0)
        mean_ph = np.sum(w * Z, axis=0)                       # (n_ant,)
        conc = np.abs(mean_ph)                                # 1 = perfectly consistent
        cost = float(np.sum(1.0 - conc[1:]))
        if best is None or cost < best["cost"]:
            # Circular std per channel [deg] (Mardia): sqrt(-2 ln R)
            circ_std = np.degrees(np.sqrt(-2.0 * np.log(np.clip(conc, 1e-6, 1.0))))
            best = {
                "cost": cost,
                "rot_deg": float(rot),
                "offsets_deg": np.degrees(np.angle(mean_ph)),
                "concentration": conc,
                "circ_std_deg": circ_std,
            }
    best["offsets_deg"][0] = 0.0
    return best


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session_dir = os.path.abspath(args.session_dir.rstrip("/"))

    # Config (re-uses the live pipeline loader and profiles)
    from apps.doa_iridium.run_doa import _PROFILES, load_config
    cfg = load_config(args.config)
    profile = _PROFILES[args.mode]
    arr = cfg["array"]

    use_session_tle(session_dir)   # freeze ground-truth elements per session
    t0, t1, meta = load_session_window(session_dir)
    print(f"Session window: {t0.isoformat()} → {t1.isoformat()}")

    bursts = collect_burst_measurements(session_dir, cfg, profile, args)
    if len(bursts) < 20:
        sys.exit(f"Too few bursts ({len(bursts)}) — need ≥ 20 for a stable fit")

    tracks = build_sat_tracks(t0, t1 + timedelta(seconds=30), args)
    assigned = assign_satellites(bursts, tracks, args)
    if len(assigned) < 20:
        sys.exit(f"Too few assigned bursts ({len(assigned)}) — "
                 "lower --snr-min or check the session/TLE window")

    n_sats = len({b["sat"] for b in assigned})
    if n_sats < 2:
        print("WARNING: all bursts from one satellite — rotation and phase "
              "offsets are degenerate; ant0_offset_deg estimate is unreliable.")

    fit = fit_offsets_and_rotation(
        assigned,
        n_ant=arr["n_ant"], radius_lambda=arr["radius_lambda"],
        ant_ccw=arr["ant_ccw"], rot_step_deg=args.rot_step,
    )

    offs = fit["offsets_deg"]
    print("\n── Fit result ──────────────────────────────────────────")
    print(f"ant0_offset_deg (array rotation): {fit['rot_deg']:+.1f}°")
    for k in range(len(offs)):
        print(f"  ch{k}: phase offset {offs[k]:+7.1f}°   "
              f"residual spread ±{fit['circ_std_deg'][k]:.1f}°")
    print(f"cost = {fit['cost']:.4f}  ({len(assigned)} bursts, {n_sats} satellites)")

    out_path = args.out or os.path.join(session_dir, "cal_tle.npz")
    np.savez(
        out_path,
        phase_offsets_deg=offs.astype(np.float64),
        ant0_offset_deg=np.float64(fit["rot_deg"]),
        circ_std_deg=fit["circ_std_deg"].astype(np.float64),
        n_bursts=np.int32(len(assigned)),
        n_sats=np.int32(n_sats),
    )
    print(f"\nSaved calibration to {out_path}")
    print("Apply it in doa_config.toml:")
    print("  [array]")
    print(f"  ant0_offset_deg = {fit['rot_deg']:.1f}")
    print("  use_phase_cal   = true")
    print(f'  cal_file        = "{out_path}"')


if __name__ == "__main__":
    main()
