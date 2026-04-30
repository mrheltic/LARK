#!/usr/bin/env python3
"""
iridium_pass_predict.py — Iridium constellation visibility vs DoA comparison
=============================================================================
Compares KrakenSDR burst DoA estimates (from .npz recordings) with predicted
Iridium satellite positions computed from NORAD TLE data.

The tool answers the fundamental question: **"Was the DoA I measured consistent
with an actual Iridium satellite in that direction?"**

Features
--------
* Auto-downloads Iridium NEXT TLE catalogue from CelesTrak (24h cache)
* Auto-detects observer GPS position (gpsd → GeoIP → cache → prompt)
* Loads any .npz recording produced by space_collector / space_doa_realtime
* Runs the full DoA pipeline on each burst (Doppler → BPF → UW → R → MUSIC)
* For each accepted burst, matches against the nearest visible Iridium satellite
* Produces a combined sky plot: observed DoA (scatter) + TLE tracks (arcs)
* Exports a JSON report with per-burst matching results
* Output format compatible with space_doa_playback.py

Display — 4 panels
-------------------
  [0] Sky plot:  observed DoA (dots) + TLE satellite tracks (arcs)
  [1] Az error histogram:  DoA_az − TLE_az  [°]
  [2] El error histogram:  DoA_el − TLE_el  [°]
  [3] Doppler observed vs predicted scatter

Usage
-----
    python3 iridium_pass_predict.py                           # file picker
    python3 iridium_pass_predict.py recording.npz             # explicit file
    python3 iridium_pass_predict.py recording.npz --lat 45.07 --lon 7.69
    python3 iridium_pass_predict.py --predict-now             # show visible now
    python3 iridium_pass_predict.py --predict-window 2        # predict next 2h
    python3 iridium_pass_predict.py recording.npz --export out.json

Environment
-----------
    LARK_OBSERVER_LAT / LARK_OBSERVER_LON / LARK_OBSERVER_ALT
        Override observer GPS coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

_SRC_BOOT = str(Path(__file__).resolve().parents[2])  # krakenSDR/src
if _SRC_BOOT not in sys.path:
    sys.path.insert(0, _SRC_BOOT)

from runtime_paths import setup_paths
_HERE, _SRC, _ROOT, _APP = setup_paths(__file__, include_root=True)

import numpy as np

# ── Shared modules ────────────────────────────────────────────────────────────
from shared.observer import get_observer, save_to_meta
from shared.iridium_tle import load_catalogue, IridiumCatalogue
from shared.iridium import (
    SIMPLEX_RING_CH_HZ, CHANNEL_SPACING_HZ, C_LIGHT,
    MAX_DOPPLER_HZ, ORBIT_ALTITUDE_M,
)
from shared.geo_utils import angular_distance_deg as _angular_distance_deg

# ── KrakenSDR core ────────────────────────────────────────────────────────────
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
    skyplot_coords,
    eigenvalue_spread_db,
    normalize_cross_array_order,
    reorder_cross_array_channels,
)

# ── Gate thresholds ───────────────────────────────────────────────────────────
_UW_SCORE_MIN      = 0.4
_EIG_SPREAD_MIN_DB = 6.0
_PAPR_MIN_DB       = 1.0
_FDMA_SPACING_HZ   = 41_667.0
_DOPPLER_JUMP_HZ   = 20_000.0
_PASS_GAP_S        = 600.0

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
C_BORDER = "#3b4263"; C_DIM = "#4e5680"
C_BLUE   = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIOLET = "#a78bfa"; C_ROSE  = "#f16b6f"; C_LIME  = "#6dd97d"
C_TEXT   = "#d8dae8"; C_MUTED = "#8891b0"
C_WHITE  = "#e8eaf6"


# =============================================================================
# Helpers
# =============================================================================

# Backward-compatible alias: existing code and tests import _angular_distance
# from this module; the canonical implementation now lives in shared.geo_utils.
_angular_distance = _angular_distance_deg


def _pick_file() -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Open KrakenSDR space capture",
        filetypes=[("NumPy archives", "*.npz"), ("All files", "*.*")],
        initialdir=os.path.normpath(os.path.join(_ROOT, "recordings")),
    )
    root.destroy()
    if not path:
        raise SystemExit(0)
    return path


def _load(path: str) -> tuple:
    """Load .npz recording. Returns (frames, timestamps, meta)."""
    data = np.load(path, allow_pickle=False)
    if "bursts" in data:
        frames = data["bursts"]
    elif "frames" in data:
        frames = data["frames"]
    else:
        for k in data.files:
            v = data[k]
            if v.ndim == 3 and np.iscomplexobj(v):
                frames = v; break
        else:
            raise ValueError(f"Cannot find IQ data in {path}. Keys: {data.files}")
    ts = data["timestamps"] if "timestamps" in data else np.arange(len(frames), dtype=float)
    json_path = os.path.splitext(path)[0] + ".json"
    meta: dict = {}
    if os.path.isfile(json_path):
        with open(json_path) as f:
            meta = json.load(f)
    return frames.astype(np.complex64), ts.astype(np.float64), meta


def _fba(R: np.ndarray) -> np.ndarray:
    M = R.shape[0]; J = np.fliplr(np.eye(M))
    return 0.5 * (R + J @ R.conj() @ J)


# =============================================================================
# DoA pipeline (runs offline on all bursts)
# =============================================================================

def process_recording(frames, timestamps, meta) -> dict:
    """Run full DoA pipeline on a recording and return per-burst results."""
    N, N_ANT, N_SAMP = frames.shape
    FS = float(meta.get("sample_rate_hz", 1_024_000))
    D_LAMBDA = float(meta.get("d_lambda", 0.5))
    N_AZ = int(meta.get("n_az", 72))
    N_EL = int(meta.get("n_el", 18))
    EL_MIN = float(meta.get("el_min_deg", 5.0))
    N_SIG = int(meta.get("n_signals", 1))
    INPUT_ORDER = normalize_cross_array_order(
        meta.get("antenna_input_order", list(CROSS_ARRAY_CANONICAL_ORDER))
    )

    cfg = CrossArrayConfig(
        d_lambda=D_LAMBDA, n_az=N_AZ, n_el=N_EL,
        el_min_deg=EL_MIN, num_expected_signals=N_SIG,
    )

    # Arrays
    all_az       = np.zeros(N, dtype=np.float32)
    all_el       = np.zeros(N, dtype=np.float32)
    all_papr     = np.full(N, -999.0, dtype=np.float32)
    all_doppler  = np.zeros(N, dtype=np.float64)
    all_fdma_ch  = np.zeros(N, dtype=np.int32)
    all_accepted = np.zeros(N, dtype=bool)

    print(f"[PP] Processing {N} bursts …", end=" ", flush=True)
    t0 = time.time()

    for i in range(N):
        X = reorder_cross_array_channels(frames[i].astype(np.complex128), INPUT_ORDER)

        # Doppler
        try:
            X, dop = compensate_doppler(X, sample_rate=int(FS))
        except Exception:
            dop = 0.0
        all_doppler[i] = dop
        all_fdma_ch[i] = int(round(dop / _FDMA_SPACING_HZ))

        # BPF
        X_filt = narrowband_filter_burst(X, sample_rate=int(FS))

        # UW gate
        _, uw_score = validate_burst_uw(X_filt, sample_rate=int(FS))
        if uw_score < _UW_SCORE_MIN:
            continue

        # Covariance + FBA
        R = _fba(compute_single_shot_covariance(X_filt))

        # Eigenvalue spread gate
        ev = eigenvalue_spread_db(R)
        if float(ev[0] - ev[-1]) < _EIG_SPREAD_MIN_DB:
            continue

        # 2D-MUSIC
        spec = doa_music_2d(X_filt, cfg, R_in=R)
        az, el, papr = find_peak_2d(spec, cfg)
        if papr < _PAPR_MIN_DB:
            continue

        all_az[i] = az
        all_el[i] = el
        all_papr[i] = papr
        all_accepted[i] = True

    n_acc = int(all_accepted.sum())
    print(f"done ({time.time()-t0:.1f}s, {n_acc}/{N} accepted)")

    # Pass boundaries (Doppler-aware)
    boundaries = set()
    for j in range(1, N):
        if abs(all_doppler[j] - all_doppler[j-1]) > _DOPPLER_JUMP_HZ:
            boundaries.add(j)
        elif timestamps[j] - timestamps[j-1] > _PASS_GAP_S:
            boundaries.add(j)

    return {
        "N": N, "n_accepted": n_acc,
        "all_az": all_az, "all_el": all_el, "all_papr": all_papr,
        "all_doppler": all_doppler, "all_fdma_ch": all_fdma_ch,
        "all_accepted": all_accepted, "timestamps": timestamps,
        "pass_boundaries": boundaries, "cfg": cfg,
    }


# =============================================================================
# TLE matching
# =============================================================================

def match_doa_to_tle(
    results: dict,
    catalogue: IridiumCatalogue,
    lat: float, lon: float, alt: float,
    t0_utc: datetime,
    freq_hz: float = SIMPLEX_RING_CH_HZ,
) -> list[dict]:
    """For each accepted burst, find the nearest visible Iridium satellite.

    Parameters
    ----------
    results : dict from process_recording()
    catalogue : IridiumCatalogue
    lat, lon, alt : observer position
    t0_utc : datetime — UTC time of first burst (timestamps[0])
    freq_hz : float — carrier frequency for Doppler calculation

    Returns list of per-burst match dicts:
        burst_idx, doa_az, doa_el, doa_doppler_hz,
        tle_name, tle_norad_id, tle_az, tle_el, tle_doppler_hz,
        az_error, el_error, angular_sep_deg, doppler_error_hz
    """
    all_az = results["all_az"]
    all_el = results["all_el"]
    all_doppler = results["all_doppler"]
    all_accepted = results["all_accepted"]
    timestamps = results["timestamps"]
    N = results["N"]

    matches = []
    _cached_visible = {}  # (time_key) → visible list, avoid redundant computation

    for i in range(N):
        if not all_accepted[i]:
            continue

        # Burst UTC time
        burst_dt = t0_utc + timedelta(milliseconds=float(timestamps[i]))

        # Round to 5s for caching (satellite positions don't change significantly)
        cache_key = int(burst_dt.timestamp() // 5)
        if cache_key not in _cached_visible:
            _cached_visible[cache_key] = catalogue.visible_now(
                lat, lon, alt, burst_dt, el_min_deg=0.0
            )
        visible = _cached_visible[cache_key]

        doa_az = float(all_az[i])
        doa_el = float(all_el[i])
        doa_dop = float(all_doppler[i])

        if not visible:
            matches.append({
                "burst_idx": i, "matched": False,
                "doa_az": round(doa_az, 1), "doa_el": round(doa_el, 1),
                "doa_doppler_hz": round(doa_dop, 1),
            })
            continue

        # Find closest match by angular distance (weighted by Doppler similarity)
        best = None
        best_sep = 999.0
        for sat in visible:
            sep = _angular_distance(doa_az, doa_el, sat["az_deg"], sat["el_deg"])
            # Also consider Doppler consistency: penalize if sign disagrees
            dop_diff = abs(doa_dop - sat["doppler_hz"])
            combined = sep + dop_diff / 5000.0  # 5 kHz ≈ 1° penalty
            if combined < best_sep:
                best_sep = combined
                best = sat
                best_angular = sep

        az_err = doa_az - best["az_deg"]
        # Wrap az error to [-180, +180]
        if az_err > 180: az_err -= 360
        elif az_err < -180: az_err += 360
        el_err = doa_el - best["el_deg"]

        matches.append({
            "burst_idx":       i,
            "matched":         True,
            "doa_az":          round(doa_az, 1),
            "doa_el":          round(doa_el, 1),
            "doa_doppler_hz":  round(doa_dop, 1),
            "tle_name":        best["name"],
            "tle_norad_id":    best["norad_id"],
            "tle_az":          best["az_deg"],
            "tle_el":          best["el_deg"],
            "tle_doppler_hz":  best["doppler_hz"],
            "az_error":        round(az_err, 1),
            "el_error":        round(el_err, 1),
            "angular_sep_deg": round(best_angular, 1),
            "doppler_error_hz":round(doa_dop - best["doppler_hz"], 1),
        })

    return matches


# =============================================================================
# Predict-only modes (no recording needed)
# =============================================================================

def predict_visible_now(
    catalogue: IridiumCatalogue, lat: float, lon: float, alt: float,
) -> list[dict]:
    """Print currently visible Iridium satellites."""
    now = datetime.now(timezone.utc)
    visible = catalogue.visible_now(lat, lon, alt, now, el_min_deg=5.0)
    print(f"\n{'='*72}")
    print(f"  Iridium NEXT visible from ({lat:.4f}°N, {lon:.4f}°E)  "
          f"at {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*72}")
    if not visible:
        print("  No Iridium satellites above 5° elevation.")
    else:
        print(f"  {'Name':25s} {'Az [°]':>8s} {'El [°]':>8s} "
              f"{'Range [km]':>10s} {'Doppler [Hz]':>12s}")
        print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10} {'─'*12}")
        for s in visible:
            print(f"  {s['name']:25s} {s['az_deg']:8.1f} {s['el_deg']:8.1f} "
                  f"{s['range_km']:10.0f} {s['doppler_hz']:+12.0f}")
    print()
    return visible


def predict_window(
    catalogue: IridiumCatalogue, lat: float, lon: float, alt: float,
    hours: float = 2.0,
) -> list[dict]:
    """Predict Iridium passes in the next *hours*."""
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=hours)
    passes = catalogue.predict_passes(lat, lon, alt, now, end, el_min_deg=10.0)
    print(f"\n{'='*80}")
    print(f"  Iridium NEXT passes from ({lat:.4f}°N, {lon:.4f}°E)  "
          f"next {hours:.0f}h  ({now.strftime('%H:%M')}—{end.strftime('%H:%M')} UTC)")
    print(f"{'='*80}")
    if not passes:
        print("  No passes above 10° elevation.")
    else:
        print(f"  {'Name':22s} {'Rise':>8s} {'Culm':>8s} {'Set':>8s}  "
              f"{'Max El':>6s} {'Rise Az':>7s} {'Set Az':>7s} {'Dur':>5s}")
        print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8}  {'─'*6} {'─'*7} {'─'*7} {'─'*5}")
        for p in passes:
            r = p["rise_utc"][11:19]
            c = p["culmination_utc"][11:19]
            s = p["set_utc"][11:19]
            print(f"  {p['name']:22s} {r:>8s} {c:>8s} {s:>8s}  "
                  f"{p['max_el_deg']:5.1f}° {p['rise_az_deg']:6.1f}° "
                  f"{p['set_az_deg']:6.1f}° {p['duration_s']:4.0f}s")
    print()
    return passes


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(
    results: dict,
    matches: list[dict],
    catalogue: IridiumCatalogue,
    lat: float, lon: float, alt: float,
    t0_utc: datetime,
    recording_name: str = "",
) -> None:
    """4-panel comparison plot: sky, Az error, El error, Doppler scatter."""
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    all_az = results["all_az"]
    all_el = results["all_el"]
    all_papr = results["all_papr"]
    all_fdma_ch = results["all_fdma_ch"]
    all_accepted = results["all_accepted"]
    timestamps = results["timestamps"]
    N = results["N"]

    matched = [m for m in matches if m.get("matched")]
    n_matched = len(matched)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": C_BORDER, "axes.grid": True,
        "grid.color": C_BORDER, "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "text.color": C_TEXT,
    })

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.97, top=0.92, bottom=0.08,
                           hspace=0.40, wspace=0.35)

    # ── Panel 0: Sky plot (polar) ─────────────────────────────────────────────
    ax_sky = fig.add_subplot(gs[0, 0], polar=True)
    ax_sky.set_facecolor(BG2)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90)
    ax_sky.set_rlabel_position(112.5)
    ax_sky.set_rticks([0, 30, 60, 90])
    ax_sky.set_yticklabels(["90°", "60°", "30°", "0°"], color=C_MUTED, fontsize=6)
    ax_sky.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax_sky.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                            color=C_MUTED, fontsize=6.5)
    ax_sky.grid(color=C_BORDER, linewidth=0.5, alpha=0.6)
    ax_sky.spines["polar"].set_color(C_BORDER)

    # TLE tracks: for each matched satellite, plot the track during recording window
    _tle_plotted_sats = set()
    if matched:
        t_end = t0_utc + timedelta(milliseconds=float(timestamps[-1]))
        t_margin = timedelta(minutes=2)
        for m in matched:
            name = m.get("tle_name", "")
            if name in _tle_plotted_sats:
                continue
            _tle_plotted_sats.add(name)
            try:
                track = catalogue.sat_track(
                    name, lat, lon, alt,
                    t0_utc - t_margin, t_end + t_margin, step_s=5.0,
                )
                above = track["el_deg"] >= 0
                if not np.any(above):
                    continue
                th = np.deg2rad(track["az_deg"][above])
                r  = 90.0 - track["el_deg"][above]
                ax_sky.plot(th, r, "-", linewidth=1.5, alpha=0.6,
                            color=C_WHITE, zorder=1)
                # Label at culmination
                pk = np.argmax(track["el_deg"][above])
                ax_sky.annotate(
                    name.replace("IRIDIUM ", "IR"),
                    xy=(th[pk], r[pk]),
                    fontsize=5.5, color=C_WHITE, alpha=0.8,
                    ha="center", va="bottom",
                )
            except Exception:
                pass

    # Observed burst DoA scatter (same FDMA channel coloring as playback)
    _papr_max = float(all_papr[all_accepted].max()) if results["n_accepted"] > 0 else 1.0
    _seen_chs = sorted(set(int(all_fdma_ch[i]) for i in range(N) if all_accepted[i]))
    _ch_palette = [C_AMBER, C_TEAL, C_BLUE, C_LIME, C_VIOLET, C_ROSE, C_MUTED]
    _ch_color_map = {ch: _ch_palette[k % len(_ch_palette)] for k, ch in enumerate(_seen_chs)}
    _ch_shown = set()
    for i in range(N):
        if not all_accepted[i]:
            continue
        t_r, r_r = skyplot_coords(float(all_az[i]), float(all_el[i]))
        alpha_v = float(np.clip((all_papr[i] - 1.0) / (_papr_max - 1.0 + 1e-6), 0.1, 0.9))
        ch = int(all_fdma_ch[i])
        col = _ch_color_map.get(ch, C_DIM)
        lbl = f"DoA ch{ch:+d}" if ch not in _ch_shown else None
        if lbl:
            _ch_shown.add(ch)
        ax_sky.plot(t_r, r_r, "o", color=col, markersize=4, alpha=alpha_v,
                    zorder=3, markeredgecolor="none", label=lbl)

    # TLE positions as X markers for matched bursts
    for m in matched:
        t_r, r_r = skyplot_coords(m["tle_az"], m["tle_el"])
        ax_sky.plot(t_r, r_r, "x", color=C_WHITE, markersize=5, alpha=0.4,
                    markeredgewidth=0.8, zorder=2)

    ax_sky.legend(loc="lower left", fontsize=5.5, framealpha=0.5,
                  facecolor=BG3, edgecolor=C_BORDER)
    ax_sky.set_title(f"Sky: DoA vs TLE  ({len(_tle_plotted_sats)} sats)",
                     color=C_TEXT, fontsize=9, pad=10, fontweight="semibold")

    # ── Panel 1: Az error histogram ───────────────────────────────────────────
    ax_az = fig.add_subplot(gs[0, 1])
    az_errors = [m["az_error"] for m in matched]
    if az_errors:
        ax_az.hist(az_errors, bins=36, range=(-180, 180), color=C_AMBER,
                   alpha=0.8, edgecolor=BG2, linewidth=0.5)
        med_az = np.median(az_errors)
        ax_az.axvline(med_az, color=C_LIME, linewidth=1.5, linestyle="--",
                      label=f"median={med_az:+.1f}°")
        ax_az.legend(fontsize=7, facecolor=BG3, edgecolor=C_BORDER)
    ax_az.set_title(f"Azimuth Error  (n={n_matched})", color=C_TEXT,
                    fontsize=9, fontweight="semibold")
    ax_az.set_xlabel("DoA_az − TLE_az  [°]", color=C_MUTED)
    ax_az.set_ylabel("count", color=C_MUTED)

    # ── Panel 2: El error histogram ───────────────────────────────────────────
    ax_el = fig.add_subplot(gs[0, 2])
    el_errors = [m["el_error"] for m in matched]
    if el_errors:
        ax_el.hist(el_errors, bins=30, range=(-60, 60), color=C_VIOLET,
                   alpha=0.8, edgecolor=BG2, linewidth=0.5)
        med_el = np.median(el_errors)
        ax_el.axvline(med_el, color=C_LIME, linewidth=1.5, linestyle="--",
                      label=f"median={med_el:+.1f}°")
        ax_el.legend(fontsize=7, facecolor=BG3, edgecolor=C_BORDER)
    ax_el.set_title(f"Elevation Error  (n={n_matched})", color=C_TEXT,
                    fontsize=9, fontweight="semibold")
    ax_el.set_xlabel("DoA_el − TLE_el  [°]", color=C_MUTED)
    ax_el.set_ylabel("count", color=C_MUTED)

    # ── Panel 3: Doppler scatter ──────────────────────────────────────────────
    ax_dop = fig.add_subplot(gs[1, 0])
    if matched:
        doa_dop  = [m["doa_doppler_hz"] / 1e3 for m in matched]
        tle_dop  = [m["tle_doppler_hz"] / 1e3 for m in matched]
        ax_dop.scatter(tle_dop, doa_dop, s=12, color=C_TEAL, alpha=0.6,
                       edgecolors="none")
        # Perfect-match diagonal
        dop_range = [min(tle_dop + doa_dop) - 5, max(tle_dop + doa_dop) + 5]
        ax_dop.plot(dop_range, dop_range, "--", color=C_LIME, linewidth=0.8,
                    alpha=0.6, label="ideal")
        ax_dop.legend(fontsize=7, facecolor=BG3, edgecolor=C_BORDER)
    ax_dop.set_title("Doppler: Observed vs Predicted", color=C_TEXT,
                     fontsize=9, fontweight="semibold")
    ax_dop.set_xlabel("TLE predicted Doppler [kHz]", color=C_MUTED)
    ax_dop.set_ylabel("Measured Doppler [kHz]", color=C_MUTED)
    ax_dop.set_aspect("equal")

    # ── Panel 4: Angular separation histogram ─────────────────────────────────
    ax_sep = fig.add_subplot(gs[1, 1])
    sep_vals = [m["angular_sep_deg"] for m in matched]
    if sep_vals:
        ax_sep.hist(sep_vals, bins=30, range=(0, 90), color=C_BLUE,
                    alpha=0.8, edgecolor=BG2, linewidth=0.5)
        med_sep = np.median(sep_vals)
        ax_sep.axvline(med_sep, color=C_LIME, linewidth=1.5, linestyle="--",
                       label=f"median={med_sep:.1f}°")
        ax_sep.legend(fontsize=7, facecolor=BG3, edgecolor=C_BORDER)
    ax_sep.set_title(f"Angular Separation  (n={n_matched})", color=C_TEXT,
                     fontsize=9, fontweight="semibold")
    ax_sep.set_xlabel("DoA ↔ TLE  [°]", color=C_MUTED)
    ax_sep.set_ylabel("count", color=C_MUTED)

    # ── Panel 5: Summary text ─────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.set_facecolor(BG2)
    ax_txt.axis("off")
    summary_lines = [
        f"Recording: {recording_name}" if recording_name else "",
        f"Observer: ({lat:.4f}°N, {lon:.4f}°E, {alt:.0f}m)",
        f"Time: {t0_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"",
        f"Total bursts: {results['N']}",
        f"Accepted (DoA): {results['n_accepted']}",
        f"Matched to TLE: {n_matched}",
        f"Unique TLE sats: {len(_tle_plotted_sats)}",
        f"",
    ]
    if matched:
        summary_lines += [
            f"Az error median: {np.median(az_errors):+.1f}°  "
            f"(MAE: {np.mean(np.abs(az_errors)):.1f}°)",
            f"El error median: {np.median(el_errors):+.1f}°  "
            f"(MAE: {np.mean(np.abs(el_errors)):.1f}°)",
            f"Angular sep median: {np.median(sep_vals):.1f}°",
            f"Doppler MAE: {np.mean(np.abs([m['doppler_error_hz'] for m in matched])):.0f} Hz",
        ]
    ax_txt.text(0.05, 0.95, "\n".join(summary_lines),
                transform=ax_txt.transAxes, fontsize=8.5, color=C_TEXT,
                va="top", family="monospace",
                bbox=dict(facecolor=BG3, edgecolor=C_BORDER, alpha=0.8, pad=8))
    ax_txt.set_title("Summary", color=C_TEXT, fontsize=9, fontweight="semibold")

    fig.suptitle(
        "LARK — Iridium DoA vs TLE Constellation Comparison",
        color=C_TEXT, fontsize=11, fontweight="bold", y=0.97,
    )
    plt.show()


# =============================================================================
# Export results
# =============================================================================

def export_json(
    matches: list[dict],
    results: dict,
    lat: float, lon: float, alt: float,
    t0_utc: datetime,
    recording_path: str,
    output_path: str,
) -> None:
    """Export comparison results to JSON."""
    report = {
        "tool": "iridium_pass_predict",
        "recording": os.path.basename(recording_path),
        "observer": {"lat": lat, "lon": lon, "alt_m": alt},
        "t0_utc": t0_utc.isoformat(),
        "total_bursts": results["N"],
        "accepted_bursts": results["n_accepted"],
        "matched_bursts": sum(1 for m in matches if m.get("matched")),
        "matches": matches,
    }
    # Add statistics
    matched = [m for m in matches if m.get("matched")]
    if matched:
        az_err = [m["az_error"] for m in matched]
        el_err = [m["el_error"] for m in matched]
        sep    = [m["angular_sep_deg"] for m in matched]
        report["statistics"] = {
            "az_error_median_deg":  round(float(np.median(az_err)), 1),
            "az_error_mae_deg":     round(float(np.mean(np.abs(az_err))), 1),
            "el_error_median_deg":  round(float(np.median(el_err)), 1),
            "el_error_mae_deg":     round(float(np.mean(np.abs(el_err))), 1),
            "angular_sep_median_deg": round(float(np.median(sep)), 1),
        }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[PP] Exported: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iridium DoA vs TLE constellation comparison tool")
    parser.add_argument("file", nargs="?", help=".npz capture file")
    parser.add_argument("--lat", type=float, default=None, help="Observer latitude [°N]")
    parser.add_argument("--lon", type=float, default=None, help="Observer longitude [°E]")
    parser.add_argument("--alt", type=float, default=0.0, help="Observer altitude [m AMSL]")
    parser.add_argument("--tle", type=str, default=None, help="Path to TLE file")
    parser.add_argument("--export", type=str, default=None, help="Export JSON report path")
    parser.add_argument("--predict-now", action="store_true",
                        help="Show currently visible Iridium satellites (no recording needed)")
    parser.add_argument("--predict-window", type=float, default=None, metavar="HOURS",
                        help="Predict passes in the next N hours (no recording needed)")
    parser.add_argument("--force-tle", action="store_true",
                        help="Force TLE re-download regardless of cache age")
    args = parser.parse_args()

    # ── Env overrides ─────────────────────────────────────────────────────────
    if args.lat is None:
        env_lat = os.environ.get("LARK_OBSERVER_LAT")
        if env_lat:
            args.lat = float(env_lat)
    if args.lon is None:
        env_lon = os.environ.get("LARK_OBSERVER_LON")
        if env_lon:
            args.lon = float(env_lon)
    if args.alt == 0.0:
        env_alt = os.environ.get("LARK_OBSERVER_ALT")
        if env_alt:
            args.alt = float(env_alt)

    # ── Predict-only modes ────────────────────────────────────────────────────
    if args.predict_now or args.predict_window is not None:
        lat, lon, alt = get_observer(args.lat, args.lon, args.alt)
        catalogue = load_catalogue(args.tle, force_download=args.force_tle)
        if args.predict_now:
            predict_visible_now(catalogue, lat, lon, alt)
        if args.predict_window is not None:
            predict_window(catalogue, lat, lon, alt, hours=args.predict_window)
        return

    # ── Recording comparison mode ─────────────────────────────────────────────
    npz_path = args.file or _pick_file()
    if not os.path.isfile(npz_path):
        print(f"[PP] File not found: {npz_path}")
        raise SystemExit(1)

    frames, timestamps, meta = _load(npz_path)
    lat, lon, alt = get_observer(args.lat, args.lon, args.alt, meta=meta)

    # ── Determine recording UTC start time ────────────────────────────────────
    ts_str = meta.get("timestamp_utc", "")
    if ts_str:
        try:
            t0_utc = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            t0_utc = datetime.now(timezone.utc)
    else:
        # Try to infer from filename (kraken_space_rt_YYYYMMDD_HHMMSS.npz)
        base = os.path.basename(npz_path).replace(".npz", "")
        parts = base.split("_")
        try:
            date_str = [p for p in parts if len(p) == 8 and p.isdigit()][0]
            time_str = [p for p in parts if len(p) == 6 and p.isdigit()][-1]
            t0_utc = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except (IndexError, ValueError):
            t0_utc = datetime.now(timezone.utc)
            print(f"[PP] WARNING: Cannot determine recording UTC time. Using now().")

    print(f"[PP] Observer: ({lat:.4f}°N, {lon:.4f}°E, {alt:.0f}m)")
    print(f"[PP] Recording: {os.path.basename(npz_path)}  "
          f"t0={t0_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Load TLE catalogue
    catalogue = load_catalogue(args.tle, force_download=args.force_tle)

    # Process recording
    results = process_recording(frames, timestamps, meta)

    # Match DoA to TLE
    matches = match_doa_to_tle(results, catalogue, lat, lon, alt, t0_utc)

    # Summary
    matched = [m for m in matches if m.get("matched")]
    print(f"\n[PP] {'='*60}")
    print(f"[PP] RESULTS: {len(matched)}/{results['n_accepted']} accepted bursts matched to TLE")
    if matched:
        az_err = [m["az_error"] for m in matched]
        el_err = [m["el_error"] for m in matched]
        sep = [m["angular_sep_deg"] for m in matched]
        print(f"[PP]   Az error  — median: {np.median(az_err):+.1f}°   "
              f"MAE: {np.mean(np.abs(az_err)):.1f}°")
        print(f"[PP]   El error  — median: {np.median(el_err):+.1f}°   "
              f"MAE: {np.mean(np.abs(el_err)):.1f}°")
        print(f"[PP]   Angular sep — median: {np.median(sep):.1f}°")
        # Unique satellites matched
        unique_sats = set(m["tle_name"] for m in matched)
        print(f"[PP]   Unique sats: {len(unique_sats)}: "
              + ", ".join(sorted(unique_sats)))
    print(f"[PP] {'='*60}\n")

    # Export JSON
    if args.export:
        export_json(matches, results, lat, lon, alt, t0_utc, npz_path, args.export)

    # Save observer location back to sidecar (if not already present)
    json_path = os.path.splitext(npz_path)[0] + ".json"
    if os.path.isfile(json_path):
        with open(json_path) as f:
            sidecar = json.load(f)
        if "observer_lat" not in sidecar:
            save_to_meta(sidecar, lat, lon, alt)
            with open(json_path, "w") as f:
                json.dump(sidecar, f, indent=2)
            print(f"[PP] Updated sidecar with observer location: {json_path}")

    # Plot
    plot_comparison(results, matches, catalogue, lat, lon, alt, t0_utc,
                    recording_name=os.path.basename(npz_path))


if __name__ == "__main__":
    main()
