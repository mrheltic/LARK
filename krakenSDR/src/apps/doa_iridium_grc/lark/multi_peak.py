"""
multi_peak.py — Multi-peak DOA extraction for offline Iridium reprocessing.

Each valid burst yields up to K (az, el, power, papr_local) peaks from the 2D
MUSIC spectrum plus CFO from the preamble tone scan.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.doa_algorithms import apply_phase_correction
from core.doa_uca_2d import (
    UcaConfig,
    doa_bartlett_uca_2d,
    doa_capon_uca_2d,
    doa_music_uca_2d,
    find_peak_uca_2d,
    find_top_k_peaks_uca_2d,
)

from .burst_processing import (
    apply_bpf_and_normalize,
    compute_mf_covariance,
    detect_energy_bursts,
    scan_preamble_tones,
)

__all__ = [
    "peaks_to_array",
    "peaks_from_list",
    "process_cpi_for_multi",
]

FS = 1_024_000.0


def peaks_to_array(
    peaks: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Pack peaks as (K, 4) float32: az, el, power_db, papr_local_db."""
    if not peaks:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(peaks, dtype=np.float32)


def peaks_from_list(peaks_arr: np.ndarray) -> list[tuple[float, float, float, float]]:
    rows = []
    for row in np.asarray(peaks_arr):
        rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return rows


def process_cpi_for_multi(
    X: np.ndarray,
    *,
    frame_idx: int,
    cfg: dict,
    profile: dict,
    uca: UcaConfig,
    phase_offs: list[float],
    algo: str,
    thr: dict,
    k_peaks: int = 3,
    min_sep_az_deg: float = 15.0,
    min_sep_el_deg: float = 8.0,
    timestamp: float | None = None,
) -> dict[str, Any] | None:
    """
    Detect first burst in a CPI, compute DOA spectrum, extract top-K peaks.

    Uses per-burst R_mf (no EMA). Returns None if burst fails quality gates.
    """
    hw = cfg["hardware"]
    pre_samples = hw["pre_samples"]
    window_samples = min(hw["window_samples"], X.shape[1])
    bpf_guard = hw["bpf_guard"]
    has_cal = any(o != 0.0 for o in phase_offs)

    starts = detect_energy_bursts(
        X[0], FS, threshold_factor=profile["energy_threshold"],
    )
    if not starts:
        return None

    b0 = starts[0]
    bend = min(b0 + window_samples, X.shape[1])
    if bend - b0 < pre_samples + bpf_guard:
        return None

    tones = scan_preamble_tones(
        X[0, b0:bend], FS,
        nom_tone_hz=profile["tone_nom_hz"],
        scan_bw_hz=profile["scan_bw_hz"],
        min_snr_db=profile["min_snr_db"],
        dc_guard_hz=profile["dc_guard_hz"],
    )
    if not tones:
        return None
    tone_hz, _tone_snr = tones[0]

    X_win = X[:, b0:bend]
    try:
        X_bpf = apply_bpf_and_normalize(
            X_win, window_samples, FS, tone_hz, profile["bpf_bw_hz"],
        )
    except ValueError:
        return None

    n_pre_eff = min(pre_samples, X_bpf.shape[1] - bpf_guard)
    if n_pre_eff < 512:
        return None

    X_cal = apply_phase_correction(X_bpf, phase_offs) if has_cal else X_bpf
    try:
        R_mf, _, snr_db = compute_mf_covariance(
            X_cal, tone_hz, FS, n_pre_eff, bpf_guard,
        )
    except ValueError:
        return None

    if snr_db < thr["snr_min_db"]:
        return None

    algo_u = algo.upper()
    if algo_u == "CAPON":
        spec = doa_capon_uca_2d(X_cal, uca, R_in=R_mf, decorr="none")
    elif algo_u == "BARTLETT":
        spec = doa_bartlett_uca_2d(X_cal, uca, R_in=R_mf)
    else:
        spec = doa_music_uca_2d(X_cal, uca, R_in=R_mf)

    _, _, papr_global = find_peak_uca_2d(spec, uca)
    if papr_global < thr["papr_min_db"]:
        return None

    peak_list = find_top_k_peaks_uca_2d(
        spec, uca, k_peaks,
        min_sep_az_deg=min_sep_az_deg,
        min_sep_el_deg=min_sep_el_deg,
        min_papr_db=thr["papr_min_db"],
    )
    if not peak_list:
        return None

    cfo_hz = tone_hz - profile["tone_nom_hz"]
    return {
        "frame": frame_idx,
        "t": float(timestamp if timestamp is not None else 0.0),
        "cfo_hz": float(cfo_hz),
        "tone_hz": float(tone_hz),
        "snr_db": float(snr_db),
        "papr_db_global": float(papr_global),
        "peaks": peaks_to_array(peak_list),
        "spec2d": np.asarray(spec, dtype=np.float32),
    }
