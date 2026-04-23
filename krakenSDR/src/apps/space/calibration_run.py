#!/usr/bin/env python3
"""
calibration_run.py — Phase-difference → (az, el) neural calibration pipeline
==============================================================================
Builds a calibration dataset from .npz recordings + TLE ground truth, trains
a lightweight MLP, validates accuracy, and exports the model for online use.

The entire pipeline uses **no external ML framework** — only numpy/scipy.
Model weights are saved as plain .npz (no pickle).

Workflow
--------
1. **collect**  — Process .npz recordings: extract features + match with TLE
2. **train**    — Train MLP on collected (features, TLE-az, TLE-el) pairs
3. **validate** — Evaluate on held-out data, display error distributions
4. **export**   — Save model .npz + calibration report .json

Data flow
---------
    .npz recordings
         │
         ▼
    DSP pipeline (Doppler → BPF → FBA → covariance)
         │
         ▼
    Feature extraction (14 features per burst)
         │
         ├──→  TLE matching (ground truth az/el)
         │
         ▼
    Calibration dataset (.npz)
         │
         ▼
    MLP training (14 → 64 → 64 → 32 → 3)
         │
         ▼
    Model .npz + validation report

Usage
-----
    # Build dataset from a recording:
    python3 calibration_run.py collect recording.npz

    # Build dataset from all .npz files in a directory:
    python3 calibration_run.py collect recordings/

    # Train on collected dataset:
    python3 calibration_run.py train calibration_dataset.npz

    # Full pipeline (collect + train + validate):
    python3 calibration_run.py auto recording.npz
    python3 calibration_run.py auto recordings/ --epochs 300

    # Validate existing model on new data:
    python3 calibration_run.py validate model.npz --data new_recording.npz

    # Compare NN vs MUSIC on a recording:
    python3 calibration_run.py compare model.npz recording.npz

Environment
-----------
    LARK_OBSERVER_LAT / LARK_OBSERVER_LON / LARK_OBSERVER_ALT
        Override observer GPS coordinates (also detected automatically).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
_ROOT = os.path.dirname(os.path.dirname(_SRC))     # LARK/
for _p in [_HERE, _SRC, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# ── Core imports ──────────────────────────────────────────────────────────────
from core.calibration_features import (
    extract_features_from_recording,
    encode_target,
    decode_target,
)
from core.calibration_model import (
    CalibrationMLP,
    TrainConfig,
    train as train_model,
    evaluate,
)
from shared.observer import get_observer
from shared.iridium_tle import load_catalogue

# ── Palette (consistent with space apps) ──────────────────────────────────────
BG       = "#1a1d27"
C_BLUE   = "#5ea4e0"
C_TEAL   = "#4ecdc4"
C_AMBER  = "#f4a431"
C_RED    = "#e74c3c"
C_GREEN  = "#43b581"
C_DIM    = "#4e5680"

# ── Default output directory ──────────────────────────────────────────────────
_CALIB_DIR = Path(_ROOT) / "krakenSDR" / "calibration"


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse a recording timestamp string in any of the supported formats.

    Handles:
    * ISO 8601 with timezone (``2026-04-15T16:35:12+00:00``)
    * ISO 8601 without timezone (assumed UTC)
    * Legacy compact format from strftime (``20260415_163512``)
    """
    # Try ISO formats first (fromisoformat handles many variants)
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # Legacy: strftime "%Y%m%d_%H%M%S"
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M%S.%f"):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse recording timestamp: {ts_str!r}")


def collect_dataset(
    npz_paths: list[Path],
    lat: float,
    lon: float,
    alt: float,
    *,
    max_angular_sep_deg: float = 15.0,
    verbose: bool = True,
) -> dict:
    """Extract features and TLE labels from .npz recordings.

    Parameters
    ----------
    npz_paths : list of Path — .npz recording files
    lat, lon, alt : observer WGS-84 position
    max_angular_sep_deg : discard matches with angular separation > this

    Returns
    -------
    dict with:
        features     (M, 14) float32
        az_tle       (M,)    float32    — TLE ground-truth azimuth [deg]
        el_tle       (M,)    float32    — TLE ground-truth elevation [deg]
        az_music     (M,)    float32    — MUSIC estimate for comparison
        el_music     (M,)    float32    — MUSIC estimate for comparison
        doppler_hz   (M,)    float64
        papr_db      (M,)    float32
        timestamps   (M,)    float64
        sat_names    list[str]          — matched satellite name per sample
        recording    list[str]          — source recording per sample
    """
    _catalogue = [None]   # lazy-load TLE catalogue only if needed by fallback

    def _get_catalogue():
        if _catalogue[0] is None:
            _catalogue[0] = load_catalogue()
        return _catalogue[0]

    all_features = []
    all_az_tle = []
    all_el_tle = []
    all_az_music = []
    all_el_music = []
    all_doppler = []
    all_papr = []
    all_timestamps = []
    all_sat_names = []
    all_recordings = []

    for npz_path in npz_paths:
        if verbose:
            print(f"\n[COLLECT] {npz_path.name}")

        # Load recording
        try:
            data = np.load(str(npz_path))
        except (EOFError, Exception) as exc:
            print(f"  [SKIP] Cannot load {npz_path.name}: {exc}")
            continue
        frames = data.get("bursts", data.get("frames"))
        if frames is None:
            print(f"  [SKIP] No 'bursts' or 'frames' array in {npz_path.name}")
            continue
        timestamps = data["timestamps"]

        # Load sidecar metadata
        json_path = npz_path.with_suffix(".json")
        meta = {}
        if json_path.is_file():
            with open(json_path) as f:
                meta = json.load(f)

        # Check whether this recording has embedded ground truth
        has_embedded_gt = (
            "gt_az_deg" in data
            and len(data["gt_az_deg"]) == len(frames)
            and np.any(np.isfinite(data["gt_az_deg"].astype(float)))
        )

        # Get recording UTC start time (for TLE fallback)
        t0_utc = None
        ts_str = meta.get("timestamp_utc")
        if ts_str:
            try:
                t0_utc = _parse_timestamp(str(ts_str))
            except ValueError as exc:
                if verbose:
                    print(f"  [WARN] Timestamp parse error: {exc}")
        if t0_utc is None:
            mtime = npz_path.stat().st_mtime
            t0_utc = datetime.fromtimestamp(mtime, tz=timezone.utc)
            if verbose:
                print("  [WARN] No valid timestamp_utc in metadata, using file mtime")

        # Extract features
        result = extract_features_from_recording(frames, timestamps, meta)
        M = result["features"].shape[0]
        if M == 0:
            print(f"  [SKIP] No accepted bursts")
            continue

        if verbose:
            src = "embedded GT" if has_embedded_gt else "TLE match"
            print(f"  {M} accepted bursts — ground truth via {src} …")

        n_matched = 0

        # ── Path A: embedded ground truth ──────────────────────────────────────
        if has_embedded_gt:
            gt_az_raw   = data["gt_az_deg"].astype(float)
            gt_el_raw   = data["gt_el_deg"].astype(float)
            gt_name_raw = data["gt_sat_name"] if "gt_sat_name" in data else None
            gt_nor_raw  = data["gt_norad_id"] if "gt_norad_id" in data else None

            for j in range(M):
                burst_idx = int(result["burst_indices"][j])
                if burst_idx >= len(gt_az_raw):
                    continue
                az_gt = gt_az_raw[burst_idx]
                el_gt = gt_el_raw[burst_idx]
                if not (np.isfinite(az_gt) and np.isfinite(el_gt)):
                    continue
                sat_name = (
                    str(gt_name_raw[burst_idx])
                    if gt_name_raw is not None and len(gt_name_raw) > burst_idx
                    else "embedded"
                )
                all_features.append(result["features"][j])
                all_az_tle.append(float(az_gt))
                all_el_tle.append(float(el_gt))
                all_az_music.append(float(result["az_music"][j]))
                all_el_music.append(float(result["el_music"][j]))
                all_doppler.append(float(result["doppler_hz"][j]))
                all_papr.append(float(result["papr_db"][j]))
                all_timestamps.append(float(result["timestamps"][j]))
                all_sat_names.append(sat_name)
                all_recordings.append(npz_path.name)
                n_matched += 1

            if verbose:
                print(f"  → {n_matched} embedded-GT samples used")

        # ── Path B: TLE re-matching (fallback for legacy recordings) ───────────
        else:
            _vis_cache = {}
            catalogue = _get_catalogue()
            for j in range(M):
                burst_ts_ms = float(result["timestamps"][j])
                burst_dt = t0_utc + timedelta(milliseconds=burst_ts_ms)

                # Cache visible satellites (5 s resolution)
                cache_key = int(burst_dt.timestamp() // 5)
                if cache_key not in _vis_cache:
                    _vis_cache[cache_key] = catalogue.visible_now(
                        lat, lon, alt, burst_dt, el_min_deg=0.0,
                    )
                visible = _vis_cache[cache_key]

                if not visible:
                    continue

                # Find nearest satellite by angular distance + Doppler
                doa_az = float(result["az_music"][j])
                doa_el = float(result["el_music"][j])
                doa_dop = float(result["doppler_hz"][j])

                best_sat = None
                best_score = 999.0

                for sat in visible:
                    az_err = abs((doa_az - sat["az_deg"] + 180) % 360 - 180)
                    el_err = abs(doa_el - sat["el_deg"])
                    # Pure angular distance — no Doppler term.
                    # The CFO returned by compensate_doppler includes the FDMA
                    # channel offset and cannot be directly compared to the
                    # satellite's predicted Doppler shift.
                    sep = np.sqrt(az_err**2 * np.cos(np.deg2rad(doa_el))**2 + el_err**2)

                    if sep < best_score:
                        best_score = sep
                        best_sat = sat

                if best_sat is not None:
                    az_e = abs((doa_az - best_sat["az_deg"] + 180) % 360 - 180)
                    el_e = abs(doa_el - best_sat["el_deg"])
                    true_sep = np.sqrt(
                        az_e**2 * np.cos(np.deg2rad(doa_el))**2 + el_e**2
                    )

                    if true_sep <= max_angular_sep_deg:
                        all_features.append(result["features"][j])
                        all_az_tle.append(best_sat["az_deg"])
                        all_el_tle.append(best_sat["el_deg"])
                        all_az_music.append(doa_az)
                        all_el_music.append(doa_el)
                        all_doppler.append(doa_dop)
                        all_papr.append(float(result["papr_db"][j]))
                        all_timestamps.append(burst_ts_ms)
                        all_sat_names.append(best_sat["name"])
                        all_recordings.append(npz_path.name)
                        n_matched += 1



    N_total = len(all_features)
    if verbose:
        print(f"\n[COLLECT] Total: {N_total} calibration samples "
              f"from {len(npz_paths)} recording(s)")

    return {
        "features":    np.array(all_features, dtype=np.float32) if N_total else np.empty((0, 14), dtype=np.float32),
        "az_tle":      np.array(all_az_tle, dtype=np.float32),
        "el_tle":      np.array(all_el_tle, dtype=np.float32),
        "az_music":    np.array(all_az_music, dtype=np.float32),
        "el_music":    np.array(all_el_music, dtype=np.float32),
        "doppler_hz":  np.array(all_doppler, dtype=np.float64),
        "papr_db":     np.array(all_papr, dtype=np.float32),
        "timestamps":  np.array(all_timestamps, dtype=np.float64),
        "sat_names":   all_sat_names,
        "recordings":  all_recordings,
    }


def save_dataset(dataset: dict, path: Path) -> None:
    """Save calibration dataset to .npz."""
    np.savez_compressed(
        str(path),
        features=dataset["features"],
        az_tle=dataset["az_tle"],
        el_tle=dataset["el_tle"],
        az_music=dataset["az_music"],
        el_music=dataset["el_music"],
        doppler_hz=dataset["doppler_hz"],
        papr_db=dataset["papr_db"],
        timestamps=dataset["timestamps"],
        sat_names=np.array(dataset["sat_names"], dtype="U"),
        recordings=np.array(dataset["recordings"], dtype="U"),
    )


def load_dataset(path: Path) -> dict:
    """Load calibration dataset from .npz."""
    data = np.load(str(path), allow_pickle=False)
    return {
        "features":   data["features"],
        "az_tle":     data["az_tle"],
        "el_tle":     data["el_tle"],
        "az_music":   data["az_music"],
        "el_music":   data["el_music"],
        "doppler_hz": data["doppler_hz"],
        "papr_db":    data["papr_db"],
        "timestamps": data["timestamps"],
        "sat_names":  list(data["sat_names"]),
        "recordings": list(data["recordings"]),
    }


# =============================================================================
# Step 2: Train
# =============================================================================

def train_pipeline(
    dataset: dict,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    verbose: bool = True,
) -> CalibrationMLP:
    """Train a CalibrationMLP on the dataset.

    Returns the trained model.
    """
    X = dataset["features"]
    N = X.shape[0]
    if N < 10:
        raise ValueError(f"Need at least 10 samples, got {N}")

    # Encode targets
    Y = np.array(
        [encode_target(az, el)
         for az, el in zip(dataset["az_tle"], dataset["el_tle"])],
        dtype=np.float32,
    )

    if verbose:
        print(f"\n[TRAIN] {N} samples, 14 features → 3 targets")
        print(f"        Az range: {dataset['az_tle'].min():.0f}° – "
              f"{dataset['az_tle'].max():.0f}°")
        print(f"        El range: {dataset['el_tle'].min():.0f}° – "
              f"{dataset['el_tle'].max():.0f}°")

    model = CalibrationMLP()
    cfg = TrainConfig(epochs=epochs, lr=lr, verbose=verbose)
    train_model(model, X, Y, cfg)

    return model


# =============================================================================
# Step 3: Validate
# =============================================================================

def validate_pipeline(
    model: CalibrationMLP,
    dataset: dict,
    *,
    verbose: bool = True,
) -> dict:
    """Validate model and compare NN vs MUSIC accuracy.

    Returns dict with NN metrics, MUSIC metrics, and comparison.
    """
    X = dataset["features"]
    az_tle = dataset["az_tle"]
    el_tle = dataset["el_tle"]
    az_music = dataset["az_music"]
    el_music = dataset["el_music"]

    # NN evaluation
    nn_metrics = evaluate(model, X, az_tle, el_tle)

    # MUSIC evaluation (direct comparison)
    az_err_music = np.abs((az_music - az_tle + 180) % 360 - 180)
    el_err_music = np.abs(el_music - el_tle)

    music_metrics = {
        "az_mae":    float(np.mean(az_err_music)),
        "az_median": float(np.median(az_err_music)),
        "az_p90":    float(np.percentile(az_err_music, 90)),
        "el_mae":    float(np.mean(el_err_music)),
        "el_median": float(np.median(el_err_music)),
        "el_p90":    float(np.percentile(el_err_music, 90)),
    }

    if verbose:
        print("\n[VALIDATE] NN vs MUSIC accuracy against TLE ground truth:")
        print(f"  {'Metric':<25s} {'NN':>8s}  {'MUSIC':>8s}  {'Δ':>8s}")
        print(f"  {'─'*55}")
        for key in ["az_mae", "az_median", "az_p90", "el_mae", "el_median", "el_p90"]:
            nn_v = nn_metrics[key]
            mu_v = music_metrics[key]
            delta = nn_v - mu_v
            better = "✓" if delta < 0 else " "
            print(f"  {key:<25s} {nn_v:7.2f}°  {mu_v:7.2f}°  {delta:+7.2f}° {better}")
        print(f"  {'─'*55}")
        print(f"  {'angular_sep_mean':<25s} {nn_metrics['angular_sep_mean']:7.2f}°")
        print(f"  {'angular_sep_p90':<25s} {nn_metrics['angular_sep_p90']:7.2f}°")

    return {
        "nn": nn_metrics,
        "music": music_metrics,
        "n_samples": int(len(az_tle)),
    }


# =============================================================================
# Step 4: Visualization
# =============================================================================

def plot_results(
    model: CalibrationMLP,
    dataset: dict,
    save_path: Optional[Path] = None,
) -> None:
    """4-panel diagnostic plot: sky plot, error distributions, validation."""
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    X = dataset["features"]
    az_tle = dataset["az_tle"]
    el_tle = dataset["el_tle"]
    az_music = dataset["az_music"]
    el_music = dataset["el_music"]

    az_nn, el_nn = model.predict(X)

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ── Panel 1: Sky plot — TLE vs NN vs MUSIC ───────────────────────────────
    ax_sky = fig.add_subplot(gs[0, 0], projection="polar")
    ax_sky.set_facecolor(BG)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90)
    ax_sky.set_yticks([0, 30, 60, 90])
    ax_sky.set_yticklabels(["90°", "60°", "30°", "0°"], fontsize=7, color="#aaa")

    r_tle = 90.0 - el_tle
    r_nn = 90.0 - el_nn
    r_music = 90.0 - el_music

    ax_sky.scatter(np.deg2rad(az_tle), r_tle, s=8, c=C_GREEN, alpha=0.6,
                   label="TLE truth", zorder=3)
    ax_sky.scatter(np.deg2rad(az_nn), r_nn, s=8, c=C_BLUE, alpha=0.5,
                   marker="^", label="NN pred", zorder=2)
    ax_sky.scatter(np.deg2rad(az_music), r_music, s=8, c=C_AMBER, alpha=0.3,
                   marker="x", label="MUSIC", zorder=1)
    ax_sky.legend(loc="lower left", fontsize=7, facecolor=BG, edgecolor="#666",
                  labelcolor="#ccc")
    ax_sky.set_title("Sky plot", color="#ccc", fontsize=10, pad=15)

    # ── Panel 2: Azimuth error histogram ─────────────────────────────────────
    ax_az = fig.add_subplot(gs[0, 1], facecolor=BG)
    az_err_nn = (az_nn - az_tle + 180) % 360 - 180
    az_err_music = (az_music - az_tle + 180) % 360 - 180
    bins = np.linspace(-30, 30, 61)
    ax_az.hist(az_err_music, bins, alpha=0.5, color=C_AMBER, label="MUSIC")
    ax_az.hist(az_err_nn, bins, alpha=0.5, color=C_BLUE, label="NN")
    ax_az.axvline(0, color="#666", ls="--", lw=0.5)
    ax_az.set_xlabel("Az error [°]", color="#aaa", fontsize=9)
    ax_az.set_title("Azimuth error", color="#ccc", fontsize=10)
    ax_az.legend(fontsize=7, facecolor=BG, edgecolor="#666", labelcolor="#ccc")
    ax_az.tick_params(colors="#888")

    # ── Panel 3: Elevation error histogram ───────────────────────────────────
    ax_el = fig.add_subplot(gs[0, 2], facecolor=BG)
    el_err_nn = el_nn - el_tle
    el_err_music = el_music - el_tle
    ax_el.hist(el_err_music, bins, alpha=0.5, color=C_AMBER, label="MUSIC")
    ax_el.hist(el_err_nn, bins, alpha=0.5, color=C_BLUE, label="NN")
    ax_el.axvline(0, color="#666", ls="--", lw=0.5)
    ax_el.set_xlabel("El error [°]", color="#aaa", fontsize=9)
    ax_el.set_title("Elevation error", color="#ccc", fontsize=10)
    ax_el.legend(fontsize=7, facecolor=BG, edgecolor="#666", labelcolor="#ccc")
    ax_el.tick_params(colors="#888")

    # ── Panel 4: Per-satellite error boxplot ─────────────────────────────────
    ax_sat = fig.add_subplot(gs[1, 0:2], facecolor=BG)
    sat_names = dataset["sat_names"]
    unique_sats = sorted(set(sat_names))
    if len(unique_sats) > 1:
        sat_errors = []
        sat_labels = []
        for s in unique_sats[:15]:  # top 15
            mask = [n == s for n in sat_names]
            err = np.abs((az_nn[mask] - az_tle[mask] + 180) % 360 - 180)
            if len(err) >= 3:
                sat_errors.append(err)
                sat_labels.append(s.replace("IRIDIUM ", "IR-"))
        if sat_errors:
            bp = ax_sat.boxplot(sat_errors, labels=sat_labels, patch_artist=True)
            for box in bp["boxes"]:
                box.set_facecolor(C_BLUE)
                box.set_alpha(0.6)
            for med in bp["medians"]:
                med.set_color(C_AMBER)
            ax_sat.tick_params(axis="x", rotation=45, colors="#888")
            ax_sat.tick_params(axis="y", colors="#888")
    ax_sat.set_ylabel("Az error [°]", color="#aaa", fontsize=9)
    ax_sat.set_title("Per-satellite azimuth error (NN)", color="#ccc", fontsize=10)

    # ── Panel 5: Training history ────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 2], facecolor=BG)
    hist = model.train_history
    if "train_loss" in hist:
        ax_hist.semilogy(hist["train_loss"], color=C_BLUE, alpha=0.7, label="train")
        ax_hist.semilogy(hist["val_loss"], color=C_AMBER, alpha=0.7, label="val")
        if "best_epoch" in hist:
            ax_hist.axvline(hist["best_epoch"], color=C_GREEN, ls="--",
                            lw=0.8, label=f"best={hist['best_epoch']}")
        ax_hist.set_xlabel("Epoch", color="#aaa", fontsize=9)
        ax_hist.set_ylabel("Loss", color="#aaa", fontsize=9)
        ax_hist.legend(fontsize=7, facecolor=BG, edgecolor="#666", labelcolor="#ccc")
    ax_hist.set_title("Training history", color="#ccc", fontsize=10)
    ax_hist.tick_params(colors="#888")

    fig.suptitle("LARK — Phase-diff NN Calibration", color="#eee",
                 fontsize=14, y=0.98)

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=BG)
        print(f"[PLOT] Saved to {save_path}")
    else:
        plt.show()


# =============================================================================
# Step 5: Long-term validation tracker
# =============================================================================

def append_validation_log(
    log_path: Path,
    model_path: Path,
    metrics: dict,
    n_samples: int,
) -> None:
    """Append a validation entry to the calibration log (JSON lines).

    This builds a long-term accuracy history for drift detection.
    Each line is an independent JSON object with timestamp, model hash,
    metrics, and sample count.
    """
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_file":    model_path.name,
        "n_samples":     n_samples,
        "nn_az_mae":     metrics["nn"]["az_mae"],
        "nn_el_mae":     metrics["nn"]["el_mae"],
        "nn_sep_mean":   metrics["nn"]["angular_sep_mean"],
        "music_az_mae":  metrics["music"]["az_mae"],
        "music_el_mae":  metrics["music"]["el_mae"],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =============================================================================
# CLI
# =============================================================================

def _find_npz_files(path: Path) -> list[Path]:
    """Resolve a file path or directory to a list of recording .npz files.

    Tries ``path`` as-is first, then relative to the LARK repository root,
    so callers can pass either an absolute path, a CWD-relative path, or a
    path relative to the project root (e.g. ``recordings/session.npz``).
    """
    # Build candidate list: given path + LARK-root-relative path
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path(_ROOT) / path)

    for p in candidates:
        if p.is_file() and p.suffix == ".npz":
            return [p]
        if p.is_dir():
            files = sorted(p.glob("*.npz"))
            # Exclude pipeline artefacts
            files = [f for f in files
                     if "calibration_dataset" not in f.name
                     and "calib_model" not in f.name]
            if files:
                return files
    return []


def main():
    parser = argparse.ArgumentParser(
        description="LARK — Phase-difference NN calibration pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── collect ───────────────────────────────────────────────────────────────
    p_col = sub.add_parser("collect", help="Extract features + ground-truth labels from recordings")
    p_col.add_argument("inputs", type=Path, nargs="+",
                       help="One or more .npz recording files or directories")
    p_col.add_argument("-o", "--out", type=Path, default=None,
                       help="Output dataset .npz (default: krakenSDR/calibration/calibration_dataset.npz)")
    p_col.add_argument("--lat", type=float, default=None)
    p_col.add_argument("--lon", type=float, default=None)
    p_col.add_argument("--alt", type=float, default=None)
    p_col.add_argument("--max-sep", type=float, default=50.0,
                       help="Max angular separation for TLE fallback matching [deg] (default: 50)")

    # ── train ─────────────────────────────────────────────────────────────────
    p_tr = sub.add_parser("train", help="Train MLP on calibration dataset")
    p_tr.add_argument("dataset", type=Path, help="Calibration dataset .npz")
    p_tr.add_argument("-o", "--out", type=Path, default=None,
                      help="Output model .npz")
    p_tr.add_argument("--epochs", type=int, default=200)
    p_tr.add_argument("--lr", type=float, default=1e-3)

    # ── validate ──────────────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="Evaluate model on data")
    p_val.add_argument("model", type=Path, help="Model .npz")
    p_val.add_argument("--data", type=Path, required=True,
                       help="Dataset or recording .npz")
    p_val.add_argument("--lat", type=float, default=None)
    p_val.add_argument("--lon", type=float, default=None)
    p_val.add_argument("--alt", type=float, default=None)

    # ── auto ──────────────────────────────────────────────────────────────────
    p_auto = sub.add_parser("auto", help="Full pipeline: collect → train → validate → plot")
    p_auto.add_argument("inputs", type=Path, nargs="+",
                        help="One or more .npz recording files or directories")
    p_auto.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="Output directory (default: krakenSDR/calibration/)")
    p_auto.add_argument("--epochs", type=int, default=200)
    p_auto.add_argument("--lr", type=float, default=1e-3)
    p_auto.add_argument("--lat", type=float, default=None)
    p_auto.add_argument("--lon", type=float, default=None)
    p_auto.add_argument("--alt", type=float, default=None)
    p_auto.add_argument("--max-sep", type=float, default=50.0)
    p_auto.add_argument("--no-plot", action="store_true")

    # ── compare ───────────────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Compare NN vs MUSIC on a recording")
    p_cmp.add_argument("model", type=Path, help="Trained model .npz")
    p_cmp.add_argument("input", type=Path, help=".npz recording")
    p_cmp.add_argument("--lat", type=float, default=None)
    p_cmp.add_argument("--lon", type=float, default=None)
    p_cmp.add_argument("--alt", type=float, default=None)

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────────

    if args.command == "collect":
        lat, lon, alt = get_observer(args.lat, args.lon, args.alt)
        npz_files: list[Path] = []
        for inp in args.inputs:
            found = _find_npz_files(inp)
            if not found:
                print(f"[WARN] No .npz files found at {inp}")
            npz_files.extend(found)
        if not npz_files:
            print("[ERROR] No .npz recording files found")
            sys.exit(1)
        ds = collect_dataset(npz_files, lat, lon, alt, max_angular_sep_deg=args.max_sep)
        out = args.out or (_CALIB_DIR / "calibration_dataset.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(ds, out)
        print(f"\n[COLLECT] Saved {ds['features'].shape[0]} samples → {out}")

    elif args.command == "train":
        ds = load_dataset(args.dataset)
        model = train_pipeline(ds, epochs=args.epochs, lr=args.lr)
        out = args.out or args.dataset.parent / "calib_model.npz"
        model.save(out)
        print(f"\n[TRAIN] Model saved → {out}")
        metrics = validate_pipeline(model, ds)
        print(json.dumps(metrics, indent=2))

    elif args.command == "validate":
        model = CalibrationMLP.load(args.model)
        if args.data.name.startswith("calibration_dataset"):
            ds = load_dataset(args.data)
        else:
            lat, lon, alt = get_observer(args.lat, args.lon, args.alt)
            npz_files = _find_npz_files(args.data)
            ds = collect_dataset(npz_files, lat, lon, alt)
        metrics = validate_pipeline(model, ds)
        # Append to long-term log
        log_path = args.model.parent / "calibration_log.jsonl"
        append_validation_log(log_path, args.model, metrics, metrics["n_samples"])
        print(f"[LOG] Appended to {log_path}")

    elif args.command == "auto":
        out_dir = args.out_dir or _CALIB_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Collect
        lat, lon, alt = get_observer(args.lat, args.lon, args.alt)
        npz_files: list[Path] = []
        for inp in args.inputs:
            found = _find_npz_files(inp)
            if not found:
                print(f"[WARN] No .npz files found at {inp}")
            npz_files.extend(found)
        if not npz_files:
            print("[ERROR] No .npz recording files found")
            sys.exit(1)
        print(f"[AUTO] Processing {len(npz_files)} recording(s)")
        ds = collect_dataset(npz_files, lat, lon, alt, max_angular_sep_deg=args.max_sep)
        ds_path = out_dir / "calibration_dataset.npz"
        save_dataset(ds, ds_path)

        if ds["features"].shape[0] < 10:
            print(f"\n[ABORT] Only {ds['features'].shape[0]} matched samples. "
                  "Need more recordings for training.")
            sys.exit(1)

        # 2. Train
        model = train_pipeline(ds, epochs=args.epochs, lr=args.lr)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = out_dir / f"calib_model_{ts}.npz"
        model.save(model_path)

        # Also save a "latest" symlink/copy
        latest = out_dir / "calib_model_latest.npz"
        model.save(latest)

        # 3. Validate
        metrics = validate_pipeline(model, ds)

        # 4. Log
        log_path = out_dir / "calibration_log.jsonl"
        append_validation_log(log_path, model_path, metrics, metrics["n_samples"])

        # 5. Report
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(ds_path),
            "model": str(model_path),
            "n_samples": int(ds["features"].shape[0]),
            "n_satellites": len(set(ds["sat_names"])),
            "metrics": metrics,
            "train_history_summary": {
                "best_epoch": model.train_history.get("best_epoch"),
                "best_val_loss": model.train_history.get("best_val_loss"),
                "n_params": model.n_params(),
            },
        }
        report_path = out_dir / f"calibration_report_{ts}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[REPORT] {report_path}")

        # 6. Plot
        if not args.no_plot:
            plot_path = out_dir / f"calibration_plot_{ts}.png"
            plot_results(model, ds, save_path=plot_path)

        print(f"\n{'═'*60}")
        print(f"  Calibration complete!")
        print(f"  Model:   {model_path}")
        print(f"  Samples: {ds['features'].shape[0]}")
        print(f"  NN  Az MAE: {metrics['nn']['az_mae']:.1f}°  "
              f"El MAE: {metrics['nn']['el_mae']:.1f}°")
        print(f"  MUSIC Az MAE: {metrics['music']['az_mae']:.1f}°  "
              f"El MAE: {metrics['music']['el_mae']:.1f}°")
        print(f"{'═'*60}")

    elif args.command == "compare":
        lat, lon, alt = get_observer(args.lat, args.lon, args.alt)
        model = CalibrationMLP.load(args.model)
        npz_files = _find_npz_files(args.input)
        ds = collect_dataset(npz_files, lat, lon, alt)
        metrics = validate_pipeline(model, ds)
        plot_results(model, ds)


if __name__ == "__main__":
    main()
