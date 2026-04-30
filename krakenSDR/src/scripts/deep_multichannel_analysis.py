#!/usr/bin/env python3
"""
deep_multichannel_analysis.py — comprehensive multi-FDMA-channel MUSIC analysis
=============================================================================

Runs a deep-dive analysis of ALL Iridium FDMA channels visible in the
KrakenSDR's ±512 kHz bandwidth around 1626.270 MHz.

For each Heimdall frame the pipeline:
  1. Detects ALL Iridium TDMA bursts (multi-burst, up to 6 per 128 ms frame)
  2. Computes total CFO per burst  (= FDMA_channel_offset + pure_Doppler)
  3. Maintains an online FDMA channel map via adaptive clustering
  4. Identifies each burst's satellite by:
       pure_Doppler = CFO − channel_center
       closest TLE satellite with |TLE_Doppler − pure_Doppler| < GT_THRESH
  5. Applies 2D-MUSIC to every accepted burst (any FDMA channel)
  6. Computes DoA (az, el) and compares against TLE ground-truth

Live stats are printed every REPORT_INTERVAL seconds:
  • Active channels with satellite ID, burst rate, median CFO, pure Doppler
  • Per-channel GT match rate and DoA residual (az_err, el_err) statistics
  • Aggregate eigenvalue-spread histogram (signal-quality indicator)

A comprehensive NPZ is saved at the end for post-processing.

Theory note — inter-channel phase independence from FDMA offset
---------------------------------------------------------------
compensate_doppler() applies the SAME complex phasor exp(-j·2π·CFO·t/fs) to
all 5 channels at once, removing both the FDMA offset and the Doppler.
The inter-channel phase differences Δφ_k = φ_k − φ_0 encode the spatial
wavefront delay and are unchanged by this operation:

    Δφ_k_after  =  (Δφ_k_before + CFO_k) − (Δφ_k_before + CFO_0)
                =  Δφ_k_before      ← since CFO is the same for all k

So MUSIC is equally valid for ring-alert bursts AND off-channel bursts.

Theory note — frequency-dependent phase calibration correction
-------------------------------------------------------------
Cable-induced inter-channel phase difference at freq f:
    Δφ_k(f) = 2π·f·Δl_k/c   (proportional to f)
Ratio across our ±512 kHz bandwidth:
    Δf/f₀  ≤  512 kHz / 1626 MHz  ≈  3.15×10⁻⁴
For a 90° differential offset: correction ≤ 0.028° → negligible.
Stored calibration from phase_offsets_latest.json is valid for ALL channels.

Usage
-----
    python3 src/scripts/deep_multichannel_analysis.py
    python3 src/scripts/deep_multichannel_analysis.py --duration 600
    python3 src/scripts/deep_multichannel_analysis.py --duration 300 --verbose
    python3 src/scripts/deep_multichannel_analysis.py --threshold 6 --gain 40 --duration 120
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src
_ROOT = os.path.dirname(os.path.dirname(_SRC))    # LARK root
for _p in (_HERE, _SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Always keep _HERE at index 0 to shadow src-level config.py
sys.path.insert(0, _HERE)

import numpy as np

import config as C
from hardware.kraken_iq_source import KrakenIQSource
from core.iridium_doa_burst import (
    detect_and_extract_all_bursts,
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
    estimate_signal_count,
    reorder_cross_array_channels,
    normalize_cross_array_order,
)

# ── constants ─────────────────────────────────────────────────────────────────
FDMA_CLUSTER_THRESH_HZ = 25_000   # Hz — new cluster if |CFO - nearest| > this
FDMA_PRUNE_MIN_BURSTS  = 3        # drop clusters with fewer confirmed bursts
GT_THRESH_HZ           = 8_000    # Hz — Doppler residual for GT satellite match
UW_SCORE_MIN           = 0.35     # 4/12 UW dibits (lowered vs realtime to catch more)
EIG_SPREAD_MIN_DB      = 4.0      # dB — lower than realtime to capture weak sats
REPORT_INTERVAL_S      = 20.0     # seconds between progress printouts
TLE_REFRESH_S          = 15.0     # seconds between TLE visible_now() refresh
N_BURST_FBA            = False    # FBA disabled: cross array does not support fliplr J (creates spurious eigenvalue → MDL says n_sig=2 → wrong MUSIC)


# =============================================================================
# FDMA channel tracker — online adaptive clustering
# =============================================================================

@dataclass
class FdmaChannel:
    """Tracks statistics for one FDMA cluster (one Iridium channel sub-band)."""
    center_hz:      float             # cluster center = mean of all CFOs in cluster
    n_bursts:       int   = 0         # total bursts confirmed
    _cfo_sum:       float = 0.0       # running sum for mean update
    # Per-burst records (appended on every accepted burst)
    cfos:           List[float] = field(default_factory=list)
    pure_dopplers:  List[float] = field(default_factory=list)   # CFO − center
    az_doas:        List[float] = field(default_factory=list)
    el_doas:        List[float] = field(default_factory=list)
    gt_sat_names:   List[str]   = field(default_factory=list)
    gt_az_deg:      List[float] = field(default_factory=list)
    gt_el_deg:      List[float] = field(default_factory=list)
    eig_spreads:    List[float] = field(default_factory=list)
    snr_dbs:        List[float] = field(default_factory=list)
    papr_dbs:       List[float] = field(default_factory=list)
    timestamps:     List[float] = field(default_factory=list)
    covs:           List        = field(default_factory=list)   # raw (5,5) R per burst
    # Satellite ID (most recently matched)
    sat_name:       str   = ""
    sat_match_count: int  = 0

    def update_center(self, new_cfo: float):
        """Update the cluster center with an online mean."""
        self._cfo_sum += new_cfo
        self.center_hz = self._cfo_sum / (self.n_bursts + 1)

    def freq_mhz(self) -> float:
        """Approximate FDMA channel frequency [MHz]."""
        return (C.FREQ_HZ + self.center_hz) / 1e6

    def gt_match_rate(self) -> float:
        matched = sum(1 for n in self.gt_sat_names if n)
        return matched / len(self.gt_sat_names) if self.gt_sat_names else 0.0

    def az_residual_stats(self):
        """(mean, std) of az DoA − GT az for matched bursts [deg]."""
        diffs = [
            az - gaz
            for az, gaz, nm in zip(self.az_doas, self.gt_az_deg, self.gt_sat_names)
            if nm
        ]
        if not diffs:
            return float("nan"), float("nan")
        a = np.array(diffs, dtype=np.float64)
        # Wrap to ±180°
        a = ((a + 180.0) % 360.0) - 180.0
        return float(np.mean(a)), float(np.std(a))

    def el_residual_stats(self):
        """(mean, std) of el DoA − GT el for matched bursts [deg]."""
        diffs = [
            el - gel
            for el, gel, nm in zip(self.el_doas, self.gt_el_deg, self.gt_sat_names)
            if nm
        ]
        if not diffs:
            return float("nan"), float("nan")
        a = np.array(diffs, dtype=np.float64)
        return float(np.mean(a)), float(np.std(a))


class FdmaChannelMap:
    """
    Maintains the set of discovered FDMA channels via online clustering.

    A new CFO is assigned to the nearest existing cluster if the distance
    is within FDMA_CLUSTER_THRESH_HZ; otherwise a new cluster is created.
    Clusters are pruned if they have fewer than FDMA_PRUNE_MIN_BURSTS after
    the warm-up phase (first 30 s).
    """

    def __init__(self):
        self._channels: List[FdmaChannel] = []
        self._t_start = time.time()

    def assign(self, cfo_hz: float) -> FdmaChannel:
        """Find or create the cluster for this CFO; return the FdmaChannel."""
        if not self._channels:
            # _cfo_sum starts at 0 — update_center() handles the first sample
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
        """Remove noise clusters (low burst count) after warm-up."""
        if time.time() - self._t_start < 30.0:
            return
        before = len(self._channels)
        self._channels = [
            ch for ch in self._channels if ch.n_bursts >= FDMA_PRUNE_MIN_BURSTS
        ]
        removed = before - len(self._channels)
        if removed:
            print(f"[FDMA] Pruned {removed} noise cluster(s)  "
                  f"({len(self._channels)} channels remain)")

    def channels_sorted(self) -> List[FdmaChannel]:
        """Return channels sorted by centre frequency (ascending)."""
        return sorted(self._channels, key=lambda c: c.center_hz)

    def __len__(self):
        return len(self._channels)


# =============================================================================
# TLE helper
# =============================================================================

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
        # Fallback to env vars if observer module unavailable
        lat = float(os.environ.get("LARK_LAT", "43.5"))
        lon = float(os.environ.get("LARK_LON",  "7.1"))
        alt = float(os.environ.get("LARK_ALT",  "0.0"))
        return lat, lon, alt


# =============================================================================
# Phase calibration loader
# =============================================================================

def _load_phase_cal() -> np.ndarray:
    """Load phase offsets from calibration file → shape (5,) radians."""
    cal_file = Path(_ROOT, "krakenSDR", "calibration", "phase_offsets_latest.json")
    if cal_file.is_file():
        try:
            with open(cal_file) as f:
                data = json.load(f)
            offsets_deg = data.get("phase_offsets_deg_input_order")
            if offsets_deg and len(offsets_deg) == 5:
                offsets = np.deg2rad(np.array(offsets_deg, dtype=np.float64))
                print(f"[CAL] Phase offsets loaded from {cal_file.name}  "
                      f"Az MAE {data.get('az_mae_before_deg','?')}° → "
                      f"{data.get('az_mae_after_deg','?')}°")
                return offsets
        except Exception as e:
            print(f"[CAL] Warning: {e}")
    print("[CAL] Using zero phase offsets (no calibration file)")
    return np.zeros(5, dtype=np.float64)


# =============================================================================
# Forward-backward averaging
# =============================================================================

def _fba(R: np.ndarray) -> np.ndarray:
    """Forward-backward averaging: R_fba = (R + J·R*·J) / 2."""
    M = R.shape[0]
    J = np.fliplr(np.eye(M, dtype=complex))
    return (R + J @ R.conj() @ J) / 2.0


# =============================================================================
# Pretty-print helpers
# =============================================================================

def _bar(value: float, total: float, width: int = 10) -> str:
    filled = int(round(value / total * width)) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _fmt_sat(name: str) -> str:
    return name.replace("IRIDIUM ", "#") if name else "—"


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Deep multi-FDMA-channel MUSIC analysis for KrakenSDR + Iridium"
    )
    ap.add_argument("--duration",  type=float, default=300.0,
                    help="Test duration [s]  (default 300 = 5 min)")
    ap.add_argument("--freq",      type=float, default=C.FREQ_HZ,
                    help="Centre frequency [Hz]")
    ap.add_argument("--gain",      type=float, default=float(C.GAIN_DB))
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="Burst detection threshold [dB above noise floor]")
    ap.add_argument("--d-lambda",  type=float, default=None,
                    help="Override arm length [fraction of λ]  (default: config D_LAMBDA=0.5)")
    ap.add_argument("--n_az",      type=int,   default=180,
                    help="Az scan points (default 180 → 2° resolution)")
    ap.add_argument("--n_el",      type=int,   default=43,
                    help="El scan points 5..90° (default 43 → ~2° resolution)")
    ap.add_argument("--verbose",   action="store_true",
                    help="Print every burst with per-channel phase error vs GT")
    ap.add_argument("--no-save",   action="store_true",
                    help="Skip saving the NPZ output file")
    args = ap.parse_args()

    FS     = float(C.SAMPLE_RATE_HZ)
    N_ANT  = int(C.N_ANTENNAS)
    _INPUT_ORDER = normalize_cross_array_order(
        getattr(C, "ANTENNA_INPUT_ORDER", CROSS_ARRAY_CANONICAL_ORDER)
    )

    print(f"\n{'='*72}")
    print(f"  DEEP MULTI-CHANNEL MUSIC TEST")
    print(f"  freq={args.freq/1e6:.3f} MHz  gain={args.gain:.0f} dB  "
          f"duration={args.duration:.0f} s  thr={args.threshold:.1f} dB")
    print(f"  FDMA cluster thresh={FDMA_CLUSTER_THRESH_HZ/1e3:.0f} kHz  "
          f"GT thresh={GT_THRESH_HZ/1e3:.0f} kHz")
    print(f"{'='*72}\n")

    # ── Phase calibration ──────────────────────────────────────────────────
    ph_offsets = _load_phase_cal()

    # ── CrossArrayConfig ──────────────────────────────────────────────────
    d_lam = args.d_lambda if args.d_lambda is not None else float(getattr(C, "D_LAMBDA", 0.5))
    cfg = CrossArrayConfig(
        d_lambda             = d_lam,
        n_az                 = args.n_az,
        n_el                 = args.n_el,
        el_min_deg           = 5.0,
        num_expected_signals = 1,    # 1 = single Iridium satellite (MDL with fliplr-J FBA inflates to 2)
    )
    print(f"[CFG] CrossArray d/λ={cfg.d_lambda}  az_pts={cfg.n_az}  el_pts={cfg.n_el}  "
          f"n_sig=1 (fixed, FBA disabled)")

    # ── TLE ───────────────────────────────────────────────────────────────
    cat = _load_tle()
    lat, lon, alt = _get_observer()
    print(f"[OBS] lat={lat:.4f}°  lon={lon:.4f}°  alt={alt:.0f} m\n")

    vis_sats: list = []
    t_tle_last = 0.0

    if cat:
        vis_sats = cat.visible_now(lat, lon, alt, el_min_deg=5.0)
        t_tle_last = time.time()
        if vis_sats:
            print(f"[SATS] {len(vis_sats)} visible at start:")
            for sv in vis_sats:
                print(f"         {sv['name']:20s}  "
                      f"az={sv['az_deg']:5.1f}°  el={sv['el_deg']:4.1f}°  "
                      f"dop={sv['doppler_hz']/1e3:+7.2f} kHz  "
                      f"rng={sv['range_km']:.0f} km")
        else:
            print("[SATS] No satellites visible — GT matching will be unavailable.")
        print()

    # ── Kraken connection ─────────────────────────────────────────────────
    kraken = KrakenIQSource(
        host         = C.HEIMDALL_HOST,
        port         = C.HEIMDALL_PORT,
        ctrl_port    = C.HEIMDALL_CTRL,
        num_channels = N_ANT,
        freq_hz      = int(args.freq),
        gain_db      = args.gain,
    )
    kraken.start()
    time.sleep(2.0)
    print(f"[HW] Heimdall connected: {kraken.is_connected}  "
          f"({C.HEIMDALL_HOST}:{C.HEIMDALL_PORT})\n")

    # ── State ─────────────────────────────────────────────────────────────
    fdma_map     = FdmaChannelMap()
    frames_total = 0
    frames_with_burst = 0
    bursts_raw   = 0
    bursts_accepted = 0   # passed UW + eig_spread gates
    t_start      = time.time()
    t_last_report = t_start
    t_last_prune  = t_start

    # ── Warm-up ───────────────────────────────────────────────────────────
    WARMUP_FRAMES = 4
    warmup = 0

    def _apply_cal(X: np.ndarray) -> np.ndarray:
        return X * np.exp(1j * ph_offsets[:, np.newaxis])

    # ── Main acquisition loop ─────────────────────────────────────────────
    t_end = t_start + args.duration
    try:
        while time.time() < t_end:
            # ── periodic TLE refresh ──────────────────────────────────────
            if cat and (time.time() - t_tle_last) > TLE_REFRESH_S:
                try:
                    vis_sats = cat.visible_now(lat, lon, alt, el_min_deg=5.0)
                    t_tle_last = time.time()
                except Exception:
                    pass

            # ── periodic FDMA prune ───────────────────────────────────────
            if (time.time() - t_last_prune) > 60.0:
                fdma_map.prune()
                t_last_prune = time.time()

            # ── get frame ────────────────────────────────────────────────
            frame = kraken.get_frame(timeout=0.3)
            if frame is None:
                continue
            frames_total += 1

            if warmup < WARMUP_FRAMES:
                warmup += 1
                continue

            # Phase-calibrate and reorder to canonical cross-array order
            X_cal = _apply_cal(frame[:N_ANT].astype(np.complex128))
            X_ord = reorder_cross_array_channels(X_cal, _INPUT_ORDER)

            # ── Burst detection (multi-burst, up to 6 per frame)  ────────
            bursts_found = detect_and_extract_all_bursts(
                X_cal,
                threshold_db = args.threshold,
                sample_rate  = int(FS),
                max_bursts   = 6,
            )
            if not bursts_found:
                continue
            frames_with_burst += 1

            for bi in bursts_found:
                bursts_raw += 1
                t_burst = time.time()

                # ── Doppler compensation: one phasor for all 5 channels ──
                try:
                    b_ord       = reorder_cross_array_channels(bi, _INPUT_ORDER)
                    comp, f_cfo = compensate_doppler(b_ord, sample_rate=int(FS))
                except Exception:
                    continue

                # ── Narrowband filter (isolates one FDMA channel at DC) ──
                try:
                    comp = narrowband_filter_burst(comp, sample_rate=int(FS))
                except Exception:
                    pass

                # ── UW gate ───────────────────────────────────────────────
                try:
                    pilot_snr, uw_score = validate_burst_uw(comp, sample_rate=int(FS))
                except Exception:
                    pilot_snr, uw_score = 0.0, 0.0

                if uw_score < UW_SCORE_MIN:
                    continue

                # ── Covariance (no FBA — cross array doesn't support fliplr FBA)
                R = compute_single_shot_covariance(comp)
                # NOTE: FBA is disabled. Using J=fliplr(eye(5)) on a cross array
                # creates a spurious second eigenvalue that makes MDL estimate n_sig=2
                # instead of 1, reducing the noise subspace from 4 to 3 vectors and
                # degrading MUSIC to essentially random peaks.

                # ── Eigenvalue spread gate ────────────────────────────────
                ev   = eigenvalue_spread_db(R)
                spread = float(ev[0] - ev[-1]) if len(ev) > 1 else 0.0
                if spread < EIG_SPREAD_MIN_DB:
                    continue

                # ── SNR ────────────────────────────────────────────────────
                snr = snr_from_covariance(R)

                # ── MUSIC (auto-MDL source count) ───────────────────────
                try:
                    spec = doa_music_2d(comp, cfg, R_in=R)
                    az_doa, el_doa, papr = find_peak_2d(spec, cfg)
                except Exception:
                    continue

                bursts_accepted += 1

                # ── FDMA channel assignment ───────────────────────────────
                ch = fdma_map.assign(f_cfo)
                ch.update_center(f_cfo)
                ch.n_bursts += 1
                ch.cfos.append(f_cfo)
                ch.eig_spreads.append(spread)
                ch.snr_dbs.append(snr)
                ch.papr_dbs.append(papr)
                ch.timestamps.append(t_burst)
                ch.az_doas.append(az_doa)
                ch.el_doas.append(el_doa)
                ch.covs.append(R.copy())

                # ── GT matching: pure Doppler = CFO − channel center ─────
                pure_dop = f_cfo - ch.center_hz
                ch.pure_dopplers.append(pure_dop)

                matched_sat   = ""
                matched_az_gt = float("nan")
                matched_el_gt = float("nan")
                if vis_sats:
                    best_sv = min(vis_sats,
                                  key=lambda s: abs(s["doppler_hz"] - pure_dop))
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

                # ── Verbose per-burst line ────────────────────────────────
                if args.verbose:
                    sat_str = (f"→ {_fmt_sat(matched_sat)}"
                               if matched_sat else "  no-GT")
                    # Raw inter-channel phases (canonical order C,E,N,W,S)
                    v = np.linalg.eigh(R)[1][:, -1]  # dominant eigenvector
                    phases_deg = np.angle(v * np.conj(v[0]), deg=True)
                    if matched_sat:
                        # Expected steering vector at GT direction
                        az_r = np.deg2rad(matched_az_gt)
                        el_r = np.deg2rad(matched_el_gt)
                        d_lam = cfg.d_lambda
                        pos = np.array([[0,0],[d_lam,0],[0,d_lam],[-d_lam,0],[0,-d_lam]])
                        tau = 2*np.pi*(pos[:,0]*np.cos(el_r)*np.sin(az_r)
                                       + pos[:,1]*np.cos(el_r)*np.cos(az_r))
                        a_gt = np.exp(1j*tau)
                        exp_phases = np.angle(a_gt * np.conj(a_gt[0]), deg=True)
                        phase_err = phases_deg - exp_phases
                        # wrap to ±180
                        phase_err = ((phase_err + 180) % 360) - 180
                        phase_str = (f"ph_err[C,E,N,W,S]="
                                     f"[{phase_err[0]:+5.1f},{phase_err[1]:+5.1f},"
                                     f"{phase_err[2]:+5.1f},{phase_err[3]:+5.1f},"
                                     f"{phase_err[4]:+5.1f}]°")
                    else:
                        ph_str = ",".join(f"{p:+5.1f}" for p in phases_deg)
                        phase_str = f"raw_ph=[{ph_str}]°"
                    print(
                        f"  [{ch.freq_mhz():.3f}MHz]  "
                        f"CFO={f_cfo/1e3:+7.1f}kHz  "
                        f"pureDop={pure_dop/1e3:+6.1f}kHz  "
                        f"az={az_doa:5.1f}°(gt={matched_az_gt:.1f}°)  "
                        f"el={el_doa:4.1f}°(gt={matched_el_gt:.1f}°)  "
                        f"snr={snr:4.1f}dB  spread={spread:4.1f}dB  "
                        f"uw={uw_score:.2f}  {sat_str}\n"
                        f"    {phase_str}"
                    )

            # ── Periodic report ───────────────────────────────────────────
            if (time.time() - t_last_report) >= REPORT_INTERVAL_S:
                _print_report(fdma_map, frames_total, frames_with_burst,
                              bursts_raw, bursts_accepted, t_start, vis_sats)
                t_last_report = time.time()

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    finally:
        kraken.stop()

    # ── Final report ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  FINAL REPORT  ({elapsed:.0f} s)")
    print(f"{'='*72}")
    _print_report(fdma_map, frames_total, frames_with_burst,
                  bursts_raw, bursts_accepted, t_start, vis_sats, final=True)

    # ── Save NPZ ──────────────────────────────────────────────────────────
    if not args.no_save:
        _save_npz(fdma_map, t_start, elapsed, args)


# =============================================================================
# Report printer
# =============================================================================

def _print_report(
    fdma_map: FdmaChannelMap,
    frames_total: int,
    frames_with_burst: int,
    bursts_raw: int,
    bursts_accepted: int,
    t_start: float,
    vis_sats: list,
    final: bool = False,
) -> None:
    elapsed = time.time() - t_start
    burst_rate = bursts_accepted / elapsed if elapsed > 0 else 0.0
    frame_rate = frames_total / elapsed   if elapsed > 0 else 0.0

    tag = "FINAL" if final else f"t={elapsed:5.0f}s"
    print(f"\n─── [{tag}]  frames={frames_total}  "
          f"frame_rate={frame_rate:.1f}fps  "
          f"bursts_raw={bursts_raw}  "
          f"accepted={bursts_accepted}  "
          f"({burst_rate:.1f}/s) ───")

    channels = fdma_map.channels_sorted()
    if not channels:
        print("  No FDMA channels discovered yet.")
        return

    print(f"\n  {'Freq [MHz]':>10}  {'CFO [kHz]':>10}  "
          f"{'#burst':>6}  {'sat':^14}  {'GT%':>5}  "
          f"{'az_err±std':>12}  {'el_err±std':>12}  "
          f"{'med_spread':>10}  {'med_snr':>8}")
    print("  " + "─" * 100)

    for ch in channels:
        if ch.n_bursts < 1:
            continue
        gt_pct   = ch.gt_match_rate() * 100
        az_m, az_s = ch.az_residual_stats()
        el_m, el_s = ch.el_residual_stats()
        med_spread = float(np.median(ch.eig_spreads)) if ch.eig_spreads else float("nan")
        med_snr    = float(np.median(ch.snr_dbs))     if ch.snr_dbs     else float("nan")

        az_str = f"{az_m:+5.1f}°±{az_s:4.1f}°" if not math.isnan(az_m) else "    —"
        el_str = f"{el_m:+5.1f}°±{el_s:4.1f}°" if not math.isnan(el_m) else "    —"
        sat_str = _fmt_sat(ch.sat_name)[:14]

        print(f"  {ch.freq_mhz():>10.3f}  "
              f"{ch.center_hz/1e3:>+10.1f}  "
              f"{ch.n_bursts:>6}  "
              f"{sat_str:^14}  "
              f"{gt_pct:>5.1f}%  "
              f"{az_str:>12}  "
              f"{el_str:>12}  "
              f"{med_spread:>10.1f}  "
              f"{med_snr:>8.1f}")

    # Eigenvalue spread histogram across all channels
    all_spreads = []
    for ch in channels:
        all_spreads.extend(ch.eig_spreads)
    if all_spreads:
        a = np.array(all_spreads)
        print(f"\n  Eig-spread histogram (dB):")
        bins = [(0, 6), (6, 10), (10, 15), (15, 20), (20, 100)]
        labels = ["< 6", "6–10", "10–15", "15–20", "≥20"]
        total_sp = len(a)
        for (lo, hi), lbl in zip(bins, labels):
            cnt = int(np.sum((a >= lo) & (a < hi)))
            print(f"    {lbl:>6} dB: {_bar(cnt, total_sp)} {cnt:4d} ({100*cnt/total_sp:.0f}%)")

    # Currently visible satellites
    if vis_sats:
        print(f"\n  Currently visible ({len(vis_sats)}):")
        for sv in vis_sats:
            print(f"    {sv['name']:20s}  "
                  f"az={sv['az_deg']:5.1f}°  el={sv['el_deg']:4.1f}°  "
                  f"dop={sv['doppler_hz']/1e3:+7.2f} kHz")


# =============================================================================
# NPZ save
# =============================================================================

def _save_npz(
    fdma_map:  FdmaChannelMap,
    t_start:   float,
    elapsed:   float,
    args,
) -> None:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(_ROOT) / "krakenSDR" / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"deep_multichannel_{ts}.npz"

    channels = fdma_map.channels_sorted()
    if not channels:
        print("[SAVE] No data to save.")
        return

    # Build arrays per channel and store as dict entries
    save_dict: dict = {
        "t_start":   np.float64(t_start),
        "duration_s": np.float64(elapsed),
        "center_freq_hz": np.float64(args.freq),
        "n_channels": np.int32(len(channels)),
    }

    for i, ch in enumerate(channels):
        pfx = f"ch{i:02d}_"
        def _arr(lst, dtype=np.float64):
            return np.array(lst, dtype=dtype) if lst else np.empty(0, dtype=dtype)

        save_dict[pfx + "center_hz"]      = np.float64(ch.center_hz)
        save_dict[pfx + "freq_mhz"]       = np.float64(ch.freq_mhz())
        save_dict[pfx + "n_bursts"]       = np.int32(ch.n_bursts)
        save_dict[pfx + "sat_name"]       = ch.sat_name
        save_dict[pfx + "cfos_hz"]        = _arr(ch.cfos)
        save_dict[pfx + "pure_dop_hz"]    = _arr(ch.pure_dopplers)
        save_dict[pfx + "az_doa_deg"]     = _arr(ch.az_doas)
        save_dict[pfx + "el_doa_deg"]     = _arr(ch.el_doas)
        save_dict[pfx + "gt_az_deg"]      = _arr(ch.gt_az_deg)
        save_dict[pfx + "gt_el_deg"]      = _arr(ch.gt_el_deg)
        save_dict[pfx + "eig_spread_db"]  = _arr(ch.eig_spreads)
        save_dict[pfx + "snr_db"]         = _arr(ch.snr_dbs)
        save_dict[pfx + "papr_db"]        = _arr(ch.papr_dbs)
        save_dict[pfx + "timestamps"]     = _arr(ch.timestamps)
        save_dict[pfx + "gt_names"]       = np.array(ch.gt_sat_names, dtype=object)
        if ch.covs:
            save_dict[pfx + "covs"]       = np.array(ch.covs, dtype=np.complex128)

    np.savez_compressed(str(out), **save_dict)
    print(f"\n[SAVE] Wrote {out.name}  ({out.stat().st_size // 1024} KB)")
    print(f"       {len(channels)} FDMA channels  "
          f"total bursts={sum(ch.n_bursts for ch in channels)}")

    # Print per-channel summary in NPZ
    print("\n  Channel summary in NPZ file:")
    for i, ch in enumerate(channels):
        gt_pct = ch.gt_match_rate() * 100
        az_m, az_s = ch.az_residual_stats()
        el_m, el_s = ch.el_residual_stats()
        print(f"    ch{i:02d}: {ch.freq_mhz():.3f} MHz  "
              f"n={ch.n_bursts:4d}  "
              f"sat={_fmt_sat(ch.sat_name):12s}  "
              f"GT={gt_pct:4.1f}%  "
              + (f"az_err={az_m:+5.1f}°±{az_s:.1f}°  el_err={el_m:+5.1f}°±{el_s:.1f}°"
                 if not math.isnan(az_m) else "  no-GT"))


if __name__ == "__main__":
    main()
