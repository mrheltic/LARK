#!/usr/bin/env python3
"""
playback_analysis.py — post-hoc MUSIC analysis on recorded NPZ burst files
============================================================================

Applies the full fixed pipeline (no FBA, n_sig=1) to every burst stored in
one or more raw recording NPZ files produced by space_doa_realtime.py or
earlier capture scripts, then shows:

  • Per-burst az/el vs TLE GT (when available)
  • Per-burst inter-channel phase errors against the GT steering vector
  • Per-FDMA-channel summary table (GT%, az/el bias ± σ, median spread/SNR)
  • Optional cross-check of stored doppler_hz (old format) against TLE Doppler
  • NPZ playback result saved to krakenSDR/data/playback_<timestamp>.npz

Two recording format variants are handled transparently:
  New (2026-04-23+): arrays bursts, timestamps  + ISO timestamp_utc in JSON
  Old (2026-04-15) : +doppler_hz, snr_db arrays + "YYYYMMDD_HHMMSS" ts string

Usage
-----
    python3 src/scripts/playback_analysis.py                          # all *.npz in recordings/
    python3 src/scripts/playback_analysis.py FILE1.npz FILE2.npz
    python3 src/scripts/playback_analysis.py --recordings-dir path/to/dir
    python3 src/scripts/playback_analysis.py recordings/*.npz --verbose
    python3 src/scripts/playback_analysis.py recordings/*.npz --no-save
    python3 src/scripts/playback_analysis.py recordings/*.npz --d-lambda 0.45
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src
_ROOT = os.path.dirname(os.path.dirname(_SRC))    # LARK root
for _p in (_HERE, _SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, _HERE)

import numpy as np

import config as C
from core.iridium_doa_burst import (
    compensate_doppler,
    compute_single_shot_covariance,
    validate_burst_uw,
    narrowband_filter_burst,
)
from core.doa_algorithms_3d import (
    CROSS_ARRAY_CANONICAL_ORDER,
    CrossArrayConfig,
    doa_music_2d,
    find_peak_2d,
    eigenvalue_spread_db,
    snr_from_covariance,
    reorder_cross_array_channels,
    normalize_cross_array_order,
)

# ── Constants ─────────────────────────────────────────────────────────────────
FDMA_CLUSTER_THRESH_HZ = 25_000   # Hz  — new cluster if |CFO − nearest| > this
FDMA_PRUNE_MIN_BURSTS  = 3        # drop transient clusters (< 3 bursts total)
GT_THRESH_HZ           = 8_000    # Hz  — max Doppler residual for GT match
UW_SCORE_MIN           = 0.35     # UW correlation gate (impostare 0.0 per test con Arduino/FSK @ 868 MHz)
EIG_SPREAD_MIN_DB      = 4.0      # dB  — eigenvalue-spread gate
TLE_REFRESH_MS         = 15_000   # ms  — refresh visible satellites every 15 s
                                  #        of recording time


# =============================================================================
# FDMA channel tracker (identical logic to deep_multichannel_analysis.py)
# =============================================================================

@dataclass
class FdmaChannel:
    center_hz:      float
    n_bursts:       int   = 0
    _cfo_sum:       float = 0.0
    cfos:           List[float] = field(default_factory=list)
    pure_dopplers:  List[float] = field(default_factory=list)
    az_doas:        List[float] = field(default_factory=list)
    el_doas:        List[float] = field(default_factory=list)
    gt_sat_names:   List[str]   = field(default_factory=list)
    gt_az_deg:      List[float] = field(default_factory=list)
    gt_el_deg:      List[float] = field(default_factory=list)
    eig_spreads:    List[float] = field(default_factory=list)
    snr_dbs:        List[float] = field(default_factory=list)
    papr_dbs:       List[float] = field(default_factory=list)
    timestamps_ms:  List[float] = field(default_factory=list)
    covs:           List        = field(default_factory=list)
    sat_name:       str   = ""
    sat_match_count: int  = 0

    def update_center(self, new_cfo: float):
        self._cfo_sum += new_cfo
        self.center_hz = self._cfo_sum / (self.n_bursts + 1)

    def freq_mhz(self) -> float:
        return (C.FREQ_HZ + self.center_hz) / 1e6

    def gt_match_rate(self) -> float:
        matched = sum(1 for n in self.gt_sat_names if n)
        return matched / len(self.gt_sat_names) if self.gt_sat_names else 0.0

    def az_residual_stats(self):
        diffs = [az - gaz for az, gaz, nm in
                 zip(self.az_doas, self.gt_az_deg, self.gt_sat_names) if nm]
        if not diffs:
            return float("nan"), float("nan")
        a = ((np.array(diffs, dtype=np.float64) + 180.0) % 360.0) - 180.0
        return float(np.mean(a)), float(np.std(a))

    def el_residual_stats(self):
        diffs = [el - gel for el, gel, nm in
                 zip(self.el_doas, self.gt_el_deg, self.gt_sat_names) if nm]
        if not diffs:
            return float("nan"), float("nan")
        a = np.array(diffs, dtype=np.float64)
        return float(np.mean(a)), float(np.std(a))


class FdmaChannelMap:
    def __init__(self):
        self._channels: List[FdmaChannel] = []

    def assign(self, cfo_hz: float) -> FdmaChannel:
        if not self._channels:
            ch = FdmaChannel(center_hz=cfo_hz)
            self._channels.append(ch)
            return ch
        distances = [abs(cfo_hz - ch.center_hz) for ch in self._channels]
        idx = int(np.argmin(distances))
        if distances[idx] < FDMA_CLUSTER_THRESH_HZ:
            return self._channels[idx]
        ch = FdmaChannel(center_hz=cfo_hz)
        self._channels.append(ch)
        return ch

    def prune(self):
        self._channels = [
            ch for ch in self._channels if ch.n_bursts >= FDMA_PRUNE_MIN_BURSTS
        ]

    def channels_sorted(self) -> List[FdmaChannel]:
        return sorted(self._channels, key=lambda c: c.center_hz)

    def __len__(self):
        return len(self._channels)


# =============================================================================
# Helpers
# =============================================================================

def _bar(value: float, total: float, width: int = 10) -> str:
    filled = int(round(value / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _fmt_sat(name: str) -> str:
    return name.replace("IRIDIUM ", "#") if name else "—"


def _parse_timestamp_utc(ts_str: str) -> Optional[datetime]:
    """Parse both ISO-8601 and legacy 'YYYYMMDD_HHMMSS' timestamp strings."""
    if not ts_str or ts_str == "?":
        return None
    # Try ISO-8601 first
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(ts_str, fmt)
        except (ValueError, TypeError):
            pass
    # ISO without timezone
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    # Legacy format: "20260415_111637"
    try:
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        pass
    return None


def _load_tle():
    try:
        from shared.iridium_tle import load_catalogue
        cat = load_catalogue()
        print(f"[TLE] Loaded {len(cat.satellites)} Iridium satellites")
        return cat
    except Exception as e:
        print(f"[TLE] FAILED: {e}")
        return None


def _get_observer():
    try:
        from shared.observer import get_observer
        lat, lon, alt = get_observer(interactive=False)
        return lat, lon, alt
    except Exception:
        lat = float(os.environ.get("LARK_LAT", "43.5"))
        lon = float(os.environ.get("LARK_LON",  "7.1"))
        alt = float(os.environ.get("LARK_ALT",  "0.0"))
        return lat, lon, alt


def _load_phase_cal() -> np.ndarray:
    cal_file = Path(_ROOT, "krakenSDR", "calibration", "phase_offsets_latest.json")
    if cal_file.is_file():
        try:
            data = json.load(open(cal_file))
            offsets_deg = data.get("phase_offsets_deg_input_order")
            if offsets_deg and len(offsets_deg) == 5:
                offsets = np.deg2rad(np.array(offsets_deg, dtype=np.float64))
                print(f"[CAL] Phase offsets loaded from {cal_file.name}  "
                      f"Az MAE {data.get('az_mae_before_deg','?')}° → "
                      f"{data.get('az_mae_after_deg','?')}°")
                return offsets
        except Exception as e:
            print(f"[CAL] Warning: {e}")
    print("[CAL] Using zero phase offsets (no calibration file or file missing)")
    return np.zeros(5, dtype=np.float64)


# =============================================================================
# Per-recording analysis
# =============================================================================

def _analyse_recording(
    npz_path: Path,
    cfg: CrossArrayConfig,
    ph_offsets: np.ndarray,
    cat,
    lat: float,
    lon: float,
    alt: float,
    verbose: bool,
    show_doppler_crosscheck: bool,
) -> Optional[FdmaChannelMap]:
    """
    Process all bursts in one NPZ recording file.

    Returns a populated FdmaChannelMap (or None if file is corrupt/unusable).
    """
    print(f"\n{'─'*72}")
    print(f"  FILE: {npz_path.name}")

    # ── Load NPZ ──────────────────────────────────────────────────────────
    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception as e:
        print(f"  [SKIP] Cannot load NPZ: {e}")
        return None

    bursts = data.get("bursts")
    if bursts is None or bursts.ndim != 3 or bursts.shape[1] != 5:
        print(f"  [SKIP] 'bursts' array missing or wrong shape: "
              f"{None if bursts is None else bursts.shape}")
        return None

    timestamps_ms = data.get("timestamps")  # relative [ms] from recording start
    stored_dop    = data.get("doppler_hz")  # old format: pre-computed CFO [Hz]

    n_total = bursts.shape[0]
    print(f"  bursts: {n_total}  shape: {bursts.shape}")

    # ── Load companion JSON ───────────────────────────────────────────────
    meta: dict = {}
    json_path = npz_path.with_suffix(".json")
    if json_path.is_file():
        try:
            meta = json.load(open(json_path))
        except Exception:
            pass

    t0_utc   = _parse_timestamp_utc(meta.get("timestamp_utc", ""))
    fs       = float(meta.get("sample_rate_hz", C.SAMPLE_RATE_HZ))
    freq_hz  = float(meta.get("freq_hz",        C.FREQ_HZ))
    d_lambda = float(meta.get("d_lambda",        getattr(C, "D_LAMBDA", 0.5)))

    input_order = normalize_cross_array_order(
        meta.get("antenna_input_order", CROSS_ARRAY_CANONICAL_ORDER)
    )

    if t0_utc:
        print(f"  t0_utc: {t0_utc.isoformat()}")
    else:
        print("  [WARN] No parseable timestamp — TLE GT will be frozen at 'now'")

    print(f"  fs={fs/1e6:.3f} MHz  freq={freq_hz/1e6:.4f} MHz  "
          f"d/λ={d_lambda}  input_order={input_order}")

    # ── Cross-check: override cfg d_lambda from file if not CLI-overridden ─
    # (cfg is passed in already with correct d_lambda from CLI or config)

    # ── TLE initial vis satellites ────────────────────────────────────────
    vis_sats: list = []
    t_tle_last_recording_ms: float = -TLE_REFRESH_MS  # force refresh on first burst

    def _refresh_tle(burst_ms: float):
        nonlocal vis_sats, t_tle_last_recording_ms
        if cat is None:
            return
        if (burst_ms - t_tle_last_recording_ms) < TLE_REFRESH_MS:
            return
        # Compute absolute UTC for TLE propagation
        if t0_utc is not None:
            t_abs = t0_utc + timedelta(milliseconds=burst_ms)
        else:
            t_abs = datetime.now(timezone.utc)
        vis_sats = cat.visible_now(lat, lon, alt, el_min_deg=5.0,
                                   t_utc=t_abs)
        t_tle_last_recording_ms = burst_ms

    # ── Phase cal applier ──────────────────────────────────────────────────
    def _apply_cal(X: np.ndarray) -> np.ndarray:
        return X * np.exp(1j * ph_offsets[:, np.newaxis])

    # ── Process each burst ────────────────────────────────────────────────
    fdma_map = FdmaChannelMap()
    n_skip_uw = 0
    n_skip_spread = 0
    n_music_err = 0
    n_accepted = 0

    for i in range(n_total):
        burst_raw = bursts[i]                           # (5, 10690) complex64
        ts_ms     = float(timestamps_ms[i]) if timestamps_ms is not None else float(i * 128)

        # TLE refresh based on recording time
        _refresh_tle(ts_ms)

        # ── Phase calibration + canonical reorder ─────────────────────────
        b_cal = _apply_cal(burst_raw.astype(np.complex128))
        b_ord = reorder_cross_array_channels(b_cal, input_order)

        # ── Doppler compensation ──────────────────────────────────────────
        try:
            comp, f_cfo = compensate_doppler(b_ord, sample_rate=int(fs))
        except Exception:
            continue

        # ── Narrowband filter ─────────────────────────────────────────────
        try:
            comp = narrowband_filter_burst(comp, sample_rate=int(fs))
        except Exception:
            pass

        # ── UW gate ───────────────────────────────────────────────────────
        try:
            pilot_snr, uw_score = validate_burst_uw(comp, sample_rate=int(fs))
        except Exception:
            pilot_snr, uw_score = 0.0, 0.0

        if uw_score < UW_SCORE_MIN:
            n_skip_uw += 1
            continue

        # ── Covariance (no FBA) ────────────────────────────────────────────
        R = compute_single_shot_covariance(comp)

        # ── Eigenvalue-spread gate ─────────────────────────────────────────
        ev     = eigenvalue_spread_db(R)
        spread = float(ev[0] - ev[-1]) if len(ev) > 1 else 0.0
        if spread < EIG_SPREAD_MIN_DB:
            n_skip_spread += 1
            continue

        snr  = snr_from_covariance(R)

        # ── 2-D MUSIC (n_sig=1, no FBA) ────────────────────────────────────
        try:
            spec = doa_music_2d(comp, cfg, R_in=R)
            az_doa, el_doa, papr = find_peak_2d(spec, cfg)
        except Exception:
            n_music_err += 1
            continue

        n_accepted += 1

        # ── FDMA channel assignment ────────────────────────────────────────
        ch = fdma_map.assign(f_cfo)
        ch.update_center(f_cfo)
        ch.n_bursts += 1
        ch.cfos.append(f_cfo)
        ch.eig_spreads.append(spread)
        ch.snr_dbs.append(snr)
        ch.papr_dbs.append(papr)
        ch.az_doas.append(az_doa)
        ch.el_doas.append(el_doa)
        ch.timestamps_ms.append(ts_ms)
        ch.covs.append(R.copy())

        # ── GT matching ────────────────────────────────────────────────────
        pure_dop = f_cfo - ch.center_hz
        ch.pure_dopplers.append(pure_dop)

        matched_sat   = ""
        matched_az_gt = float("nan")
        matched_el_gt = float("nan")
        if vis_sats:
            best_sv = min(vis_sats, key=lambda s: abs(s["doppler_hz"] - pure_dop))
            dt = abs(best_sv["doppler_hz"] - pure_dop)
            if dt < GT_THRESH_HZ:
                matched_sat   = best_sv["name"]
                matched_az_gt = best_sv["az_deg"]
                matched_el_gt = best_sv["el_deg"]
                ch.sat_name        = matched_sat
                ch.sat_match_count += 1

        ch.gt_sat_names.append(matched_sat)
        ch.gt_az_deg.append(matched_az_gt)
        ch.gt_el_deg.append(matched_el_gt)

        # ── Cross-check stored doppler_hz (old format) ─────────────────────
        dop_stored_str = ""
        if stored_dop is not None and show_doppler_crosscheck:
            sd = float(stored_dop[i])
            dop_stored_str = f"  stored_dop={sd/1e3:+6.1f}kHz  dop_err={abs(sd-f_cfo)/1e3:5.2f}kHz"

        # ── Verbose per-burst output ────────────────────────────────────────
        if verbose:
            sat_str   = f"→ {_fmt_sat(matched_sat)}" if matched_sat else "  no-GT"
            t_str     = f"t={ts_ms/1e3:6.1f}s"
            cfo_str   = f"CFO={f_cfo/1e3:+7.1f}kHz"
            dop_str   = f"pureDop={pure_dop/1e3:+6.1f}kHz"
            az_str    = f"az={az_doa:5.1f}°(gt={matched_az_gt:.1f}°)"
            el_str    = f"el={el_doa:4.1f}°(gt={matched_el_gt:.1f}°)"
            qual_str  = f"snr={snr:4.1f}dB  spd={spread:4.1f}dB  uw={uw_score:.2f}"
            print(f"  [{ch.freq_mhz():.3f}MHz]  {t_str}  {cfo_str}  {dop_str}  "
                  f"{az_str}  {el_str}  {qual_str}  {sat_str}{dop_stored_str}")
            if matched_sat:
                v          = np.linalg.eigh(R)[1][:, -1]
                phases_deg = np.angle(v * np.conj(v[0]), deg=True)
                az_r       = np.deg2rad(matched_az_gt)
                el_r       = np.deg2rad(matched_el_gt)
                d_lam      = cfg.d_lambda
                pos        = np.array([[0,0],[d_lam,0],[0,d_lam],[-d_lam,0],[0,-d_lam]])
                tau        = 2*np.pi*(pos[:,0]*np.cos(el_r)*np.sin(az_r)
                                      + pos[:,1]*np.cos(el_r)*np.cos(az_r))
                a_gt       = np.exp(1j*tau)
                exp_ph     = np.angle(a_gt * np.conj(a_gt[0]), deg=True)
                ph_err     = ((phases_deg - exp_ph + 180) % 360) - 180
                print(f"    ph_err[C,E,N,W,S]=[{ph_err[0]:+5.1f},{ph_err[1]:+5.1f},"
                      f"{ph_err[2]:+5.1f},{ph_err[3]:+5.1f},{ph_err[4]:+5.1f}]°")

    fdma_map.prune()

    rej_total = n_skip_uw + n_skip_spread + n_music_err
    print(f"\n  Accepted: {n_accepted}/{n_total}"
          f"  (skip_uw={n_skip_uw}  skip_spread={n_skip_spread}"
          f"  music_err={n_music_err})")

    return fdma_map


# =============================================================================
# Report printer
# =============================================================================

def _print_report(fdma_map: FdmaChannelMap, file_label: str) -> None:
    channels = fdma_map.channels_sorted()
    if not channels:
        print("  No FDMA channels discovered in this file.")
        return

    print(f"\n  {'Freq [MHz]':>10}  {'CFO [kHz]':>10}  "
          f"{'#burst':>6}  {'sat':^14}  {'GT%':>5}  "
          f"{'az_err±std':>13}  {'el_err±std':>13}  "
          f"{'med_spread':>10}  {'med_snr':>8}")
    print("  " + "─" * 105)

    for ch in channels:
        if ch.n_bursts < 1:
            continue
        gt_pct   = ch.gt_match_rate() * 100
        az_m, az_s = ch.az_residual_stats()
        el_m, el_s = ch.el_residual_stats()
        med_spread = float(np.median(ch.eig_spreads)) if ch.eig_spreads else float("nan")
        med_snr    = float(np.median(ch.snr_dbs))     if ch.snr_dbs     else float("nan")
        az_str = f"{az_m:+5.1f}°±{az_s:4.1f}°" if not math.isnan(az_m) else "        —"
        el_str = f"{el_m:+5.1f}°±{el_s:4.1f}°" if not math.isnan(el_m) else "        —"
        sat_str = _fmt_sat(ch.sat_name)[:14]

        print(f"  {ch.freq_mhz():>10.3f}  "
              f"{ch.center_hz/1e3:>+10.1f}  "
              f"{ch.n_bursts:>6}  "
              f"{sat_str:^14}  "
              f"{gt_pct:>5.1f}%  "
              f"{az_str:>13}  "
              f"{el_str:>13}  "
              f"{med_spread:>10.1f}  "
              f"{med_snr:>8.1f}")

    all_spreads = [s for ch in channels for s in ch.eig_spreads]
    if all_spreads:
        a = np.array(all_spreads)
        print(f"\n  Eig-spread histogram (dB):")
        bins   = [(0, 6), (6, 10), (10, 15), (15, 20), (20, 100)]
        labels = ["< 6", "6–10", "10–15", "15–20", "≥20"]
        total = len(a)
        for (lo, hi), lbl in zip(bins, labels):
            cnt = int(np.sum((a >= lo) & (a < hi)))
            print(f"    {lbl:>6} dB: {_bar(cnt, total)} {cnt:4d} ({100*cnt/total:.0f}%)")


# =============================================================================
# NPZ save
# =============================================================================

def _save_npz(
    all_maps:    list[tuple[str, FdmaChannelMap]],
    out_tag:     str,
) -> None:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(_ROOT) / "krakenSDR" / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"playback_{ts}.npz"

    save_dict: dict = {}
    global_burst_count = 0

    for file_idx, (fname, fdma_map) in enumerate(all_maps):
        file_pfx = f"f{file_idx:02d}_"
        save_dict[file_pfx + "filename"] = fname
        channels = fdma_map.channels_sorted()
        save_dict[file_pfx + "n_channels"] = np.int32(len(channels))

        for ci, ch in enumerate(channels):
            pfx = f"{file_pfx}ch{ci:02d}_"

            def _arr(lst, dtype=np.float64):
                return np.array(lst, dtype=dtype) if lst else np.empty(0, dtype=dtype)

            save_dict[pfx + "center_hz"]     = np.float64(ch.center_hz)
            save_dict[pfx + "freq_mhz"]      = np.float64(ch.freq_mhz())
            save_dict[pfx + "n_bursts"]      = np.int32(ch.n_bursts)
            save_dict[pfx + "sat_name"]      = ch.sat_name
            save_dict[pfx + "cfos_hz"]       = _arr(ch.cfos)
            save_dict[pfx + "pure_dop_hz"]   = _arr(ch.pure_dopplers)
            save_dict[pfx + "az_doa_deg"]    = _arr(ch.az_doas)
            save_dict[pfx + "el_doa_deg"]    = _arr(ch.el_doas)
            save_dict[pfx + "gt_az_deg"]     = _arr(ch.gt_az_deg)
            save_dict[pfx + "gt_el_deg"]     = _arr(ch.gt_el_deg)
            save_dict[pfx + "eig_spread_db"] = _arr(ch.eig_spreads)
            save_dict[pfx + "snr_db"]        = _arr(ch.snr_dbs)
            save_dict[pfx + "papr_db"]       = _arr(ch.papr_dbs)
            save_dict[pfx + "timestamps_ms"] = _arr(ch.timestamps_ms)
            save_dict[pfx + "gt_names"]      = np.array(ch.gt_sat_names, dtype=object)
            if ch.covs:
                save_dict[pfx + "covs"]      = np.array(ch.covs, dtype=np.complex128)
            global_burst_count += ch.n_bursts

    np.savez_compressed(str(out), **save_dict)
    print(f"\n[SAVE] {out.name}  ({out.stat().st_size // 1024} KB)  "
          f"{len(all_maps)} file(s)  {global_burst_count} total bursts")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Post-hoc MUSIC analysis on recorded KrakenSDR burst NPZ files"
    )
    ap.add_argument("files", nargs="*",
                    help="NPZ recording files to analyse (glob-expanded by shell)")
    ap.add_argument("--recordings-dir", type=str, default=None,
                    help="Scan this directory for *.npz files instead of listing them")
    ap.add_argument("--include-old", action="store_true", default=True,
                    help="Include files in <dir>/old/ sub-folder (default: on)")
    ap.add_argument("--no-old", action="store_true",
                    help="Skip the old/ sub-folder")
    ap.add_argument("--d-lambda", type=float, default=None,
                    help="Override arm length d/λ (default: from JSON or config)")
    ap.add_argument("--n-az", type=int, default=180,
                    help="Azimuth scan points (default 180 → 2° resolution)")
    ap.add_argument("--n-el", type=int, default=43,
                    help="Elevation scan points 5..90° (default 43 → ~2° resolution)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print every accepted burst with phase error diagnostics")
    ap.add_argument("--doppler-check", action="store_true",
                    help="For old-format files, print stored vs recomputed Doppler")
    ap.add_argument("--no-save", action="store_true",
                    help="Skip saving the NPZ output")
    args = ap.parse_args()

    # ── Build file list ────────────────────────────────────────────────────
    file_paths: List[Path] = []
    if args.recordings_dir:
        rec_dir = Path(args.recordings_dir)
        file_paths += sorted(rec_dir.glob("*.npz"))
        if not args.no_old:
            file_paths += sorted((rec_dir / "old").glob("*.npz"))
    elif args.files:
        file_paths = [Path(f) for f in args.files]
    else:
        # Default: LARK/recordings/ directory
        rec_dir = Path(_ROOT) / "recordings"
        if not rec_dir.is_dir():
            ap.error(
                "No files specified and default recordings/ directory not found. "
                "Pass NPZ paths as arguments or use --recordings-dir."
            )
        file_paths += sorted(rec_dir.glob("*.npz"))
        if not args.no_old:
            file_paths += sorted((rec_dir / "old").glob("*.npz"))

    if not file_paths:
        ap.error("No NPZ files found. Pass files as arguments or use --recordings-dir.")

    print(f"\n{'='*72}")
    print(f"  PLAYBACK ANALYSIS  ({len(file_paths)} file(s))")
    print(f"{'='*72}")

    # ── Load calibration + TLE + observer ─────────────────────────────────
    ph_offsets = _load_phase_cal()
    cat        = _load_tle()
    lat, lon, alt = _get_observer()
    print(f"[OBS] lat={lat:.4f}°  lon={lon:.4f}°  alt={alt:.0f} m")

    # ── CrossArrayConfig (d_lambda from CLI or file-level fallback later) ─
    d_lam = args.d_lambda if args.d_lambda is not None else float(getattr(C, "D_LAMBDA", 0.5))
    cfg = CrossArrayConfig(
        d_lambda             = d_lam,
        n_az                 = args.n_az,
        n_el                 = args.n_el,
        el_min_deg           = 5.0,
        num_expected_signals = 1,   # single Iridium satellite, FBA disabled
    )
    print(f"[CFG] d/λ={cfg.d_lambda}  n_az={cfg.n_az}  n_el={cfg.n_el}  "
          f"n_sig=1  FBA=off\n")

    # ── Process each file ──────────────────────────────────────────────────
    all_maps: list[tuple[str, FdmaChannelMap]] = []
    for npz_path in file_paths:
        fdma_map = _analyse_recording(
            npz_path,
            cfg               = cfg,
            ph_offsets        = ph_offsets,
            cat               = cat,
            lat               = lat,
            lon               = lon,
            alt               = alt,
            verbose           = args.verbose,
            show_doppler_crosscheck = args.doppler_check,
        )
        if fdma_map is None:
            continue
        _print_report(fdma_map, npz_path.name)
        all_maps.append((npz_path.name, fdma_map))

    if not all_maps:
        print("\n[DONE] No usable recordings found.")
        return

    # ── Global summary ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  GLOBAL SUMMARY  ({len(all_maps)} file(s))")
    print(f"{'='*72}")
    total_bursts  = 0
    total_matched = 0
    all_az_errs   = []
    all_el_errs   = []
    for fname, fdma_map in all_maps:
        for ch in fdma_map.channels_sorted():
            total_bursts  += ch.n_bursts
            total_matched += ch.sat_match_count
            for az, gaz, nm in zip(ch.az_doas, ch.gt_az_deg, ch.gt_sat_names):
                if nm:
                    all_az_errs.append(((az - gaz + 180) % 360) - 180)
            for el, gel, nm in zip(ch.el_doas, ch.gt_el_deg, ch.gt_sat_names):
                if nm:
                    all_el_errs.append(el - gel)

    gt_rate = total_matched / total_bursts * 100 if total_bursts else 0.0
    print(f"  Total accepted bursts : {total_bursts}")
    print(f"  GT matched bursts     : {total_matched}  ({gt_rate:.1f}%)")
    if all_az_errs:
        az_arr = np.array(all_az_errs)
        el_arr = np.array(all_el_errs)
        print(f"  Az error (bias ± σ)   : {np.mean(az_arr):+.2f}° ± {np.std(az_arr):.2f}°  "
              f"MAE={np.mean(np.abs(az_arr)):.2f}°")
        print(f"  El error (bias ± σ)   : {np.mean(el_arr):+.2f}° ± {np.std(el_arr):.2f}°  "
              f"MAE={np.mean(np.abs(el_arr)):.2f}°")
    else:
        print("  No GT-matched bursts — az/el statistics unavailable.")

    # ── Save NPZ ───────────────────────────────────────────────────────────
    if not args.no_save:
        _save_npz(all_maps, out_tag="")
    else:
        print("\n[SAVE] Skipped (--no-save)")


if __name__ == "__main__":
    main()
