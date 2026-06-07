"""
multi_peak.py — Multi-peak DOA extraction for offline Iridium reprocessing.

Architecture: each preamble tone corresponds to ONE satellite.  For a CPI that
contains K simultaneous Iridium transmissions, ``scan_preamble_tones`` returns
up to K tones (each at a different Doppler-shifted frequency).  We BPF and
compute a separate covariance for each tone, then run DOA on each covariance
to obtain one (az, el, cfo) triplet per satellite.

This avoids the ghost-peak problem that arises when K peaks are extracted from
a single covariance: secondary MUSIC peaks in that case are elevation harmonics
of the same satellite, not real satellites at different azimuths.
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
    """Pack peaks as (K, 4) float32: az, el, power_db, papr_local_db (=papr_global for per-tone)."""
    if not peaks:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(peaks, dtype=np.float32)


def peaks_from_list(peaks_arr: np.ndarray) -> list[tuple[float, float, float, float]]:
    rows = []
    for row in np.asarray(peaks_arr):
        rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return rows


def _doa_from_tone(
    X_win: np.ndarray,
    tone_hz: float,
    *,
    window_samples: int,
    pre_samples: int,
    bpf_guard: int,
    bpf_bw_hz: float,
    phase_offs: list[float],
    has_cal: bool,
    uca: UcaConfig,
    algo_u: str,
    snr_min_db: float,
    papr_min_db: float,
    el_min_deg: float = 10.0,
) -> tuple[float, float, float, float, float] | None:
    """
    BPF + MF covariance + DOA for one preamble tone.

    Returns (az_deg, el_deg, power_db, papr_db, snr_db) or None if quality gate fails.
    """
    try:
        X_bpf = apply_bpf_and_normalize(
            X_win, window_samples, FS, tone_hz, bpf_bw_hz,
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

    if snr_db < snr_min_db:
        return None

    if algo_u == "CAPON":
        spec = doa_capon_uca_2d(X_cal, uca, R_in=R_mf, decorr="none")
    elif algo_u == "BARTLETT":
        spec = doa_bartlett_uca_2d(X_cal, uca, R_in=R_mf)
    else:
        spec = doa_music_uca_2d(X_cal, uca, R_in=R_mf)

    az_deg, el_deg, papr = find_peak_uca_2d(spec, uca)
    if papr < papr_min_db:
        return None
    if el_deg < el_min_deg:
        return None

    # Return argmax power in dB (peak value of the spectrum = 0 dB by construction)
    return float(az_deg), float(el_deg), 0.0, float(papr), float(snr_db)


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
    el_min_deg: float = 10.0,
    timestamp: float | None = None,
) -> dict[str, Any] | None:
    """
    Detect burst in a CPI and return one DOA estimate per detected satellite tone.

    Each satellite has a distinct Doppler-shifted preamble tone.  ``scan_preamble_tones``
    finds up to ``k_peaks`` such tones.  For each tone a separate BPF+covariance+DOA
    is computed, yielding one (az, el) per real satellite.  Each peak entry gets its own
    ``cfo_per_peak`` (Doppler = tone_hz − nom_tone_hz).

    ``peaks`` shape: (n_sat_tones, 4) — az, el, power_db=0, papr_db
    ``cfo_hz`` scalar: CFO of the strongest tone (for backward compat)
    ``cfo_per_peak``: list of CFO per row of peaks (the real per-satellite discriminator)
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

    # Scan for up to k_peaks satellite tones in one shot
    tones = scan_preamble_tones(
        X[0, b0:bend], FS,
        nom_tone_hz=profile["tone_nom_hz"],
        scan_bw_hz=profile["scan_bw_hz"],
        n_peaks=k_peaks,
        min_snr_db=profile["min_snr_db"],
        dc_guard_hz=profile["dc_guard_hz"],
    )
    if not tones:
        return None

    X_win = X[:, b0:bend]
    algo_u = algo.upper()
    peak_list: list[tuple[float, float, float, float]] = []
    cfo_per_peak: list[float] = []
    snr_per_peak: list[float] = []
    best_snr_db = -999.0
    best_tone_hz = tones[0][0]
    best_spec: np.ndarray | None = None

    for tone_hz, _tone_snr in tones:
        result = _doa_from_tone(
            X_win, tone_hz,
            window_samples=window_samples,
            pre_samples=pre_samples,
            bpf_guard=bpf_guard,
            bpf_bw_hz=profile["bpf_bw_hz"],
            phase_offs=phase_offs,
            has_cal=has_cal,
            uca=uca,
            algo_u=algo_u,
            snr_min_db=thr["snr_min_db"],
            papr_min_db=thr["papr_min_db"],
            el_min_deg=el_min_deg,
        )
        if result is None:
            continue
        az_deg, el_deg, power_db, papr_db, snr_db = result
        peak_list.append((az_deg, el_deg, power_db, papr_db))
        cfo_per_peak.append(float(tone_hz - profile["tone_nom_hz"]))
        snr_per_peak.append(snr_db)
        if snr_db > best_snr_db:
            best_snr_db = snr_db
            best_tone_hz = tone_hz
            # Re-compute spec for the strongest tone for visualisation
            try:
                X_bpf = apply_bpf_and_normalize(X_win, window_samples, FS, tone_hz, profile["bpf_bw_hz"])
                X_cal = apply_phase_correction(X_bpf, phase_offs) if has_cal else X_bpf
                R_mf, _, _ = compute_mf_covariance(X_cal, tone_hz, FS,
                                                   min(pre_samples, X_bpf.shape[1] - bpf_guard), bpf_guard)
                if algo_u == "CAPON":
                    best_spec = doa_capon_uca_2d(X_cal, uca, R_in=R_mf, decorr="none")
                elif algo_u == "BARTLETT":
                    best_spec = doa_bartlett_uca_2d(X_cal, uca, R_in=R_mf)
                else:
                    best_spec = doa_music_uca_2d(X_cal, uca, R_in=R_mf)
            except (ValueError, Exception):
                pass

    if not peak_list:
        return None

    return {
        "frame": frame_idx,
        "t": float(timestamp if timestamp is not None else 0.0),
        "cfo_hz": float(best_tone_hz - profile["tone_nom_hz"]),
        "tone_hz": float(best_tone_hz),
        "snr_db": float(best_snr_db),
        "papr_db_global": float(peak_list[0][3]),
        "peaks": peaks_to_array(peak_list),
        "cfo_per_peak": cfo_per_peak,
        "snr_per_peak": snr_per_peak,
        "spec2d": np.asarray(best_spec, dtype=np.float32) if best_spec is not None else np.zeros((1, 1), dtype=np.float32),
    }
