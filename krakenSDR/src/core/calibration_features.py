"""
core.calibration_features
=========================
Feature extraction from 5-channel IQ bursts for the phase-difference → (az, el)
neural-network calibration pipeline.

Feature vector (per burst)
--------------------------
From the 5×5 spatial covariance matrix R (after Doppler comp + BPF + FBA):

  * 4 inter-antenna phase differences   Δφ_k = ∠R_{0,k}  for k=1..4
    (center vs east, north, west, south — canonical order)
    Encoded as (cos Δφ_k, sin Δφ_k) to avoid 2π wrapping → 8 features

  * 4 coherence magnitudes   |ρ_{0,k}| = |R_{0,k}| / √(R_{0,0}·R_{k,k})
    High = clean signal; low = noise/multipath → 4 features

  * Eigenvalue spread  λ_max/λ_min  [dB] → 1 feature (signal quality proxy)

  * Doppler offset [kHz, normalized] → 1 feature
    (Doppler shifts array pattern slightly at L-band; learnable)

Total: 14 features per burst.

Target encoding
---------------
Azimuth uses (cos az, sin az) to avoid 0°/360° discontinuity → 2 values.
Elevation uses raw degrees (naturally bounded 0–90°) → 1 value.
Total target: 3 values.

All arrays are float32 for compact storage and fast training.
"""

from __future__ import annotations

import numpy as np


def phase_diff_features(R: np.ndarray) -> np.ndarray:
    """Extract (cos Δφ, sin Δφ) for 4 cross-array arms from covariance R.

    Parameters
    ----------
    R : (5, 5) complex128 — spatial covariance (canonical channel order).

    Returns
    -------
    (8,) float32 — [cos Δφ₁, sin Δφ₁, cos Δφ₂, sin Δφ₂,
                     cos Δφ₃, sin Δφ₃, cos Δφ₄, sin Δφ₄]
    """
    phases = np.angle(R[0, 1:5])  # Δφ_k = ∠R_{0,k}  shape (4,)
    features = np.empty(8, dtype=np.float32)
    features[0::2] = np.cos(phases).astype(np.float32)
    features[1::2] = np.sin(phases).astype(np.float32)
    return features


def coherence_features(R: np.ndarray) -> np.ndarray:
    """Extract 4 coherence magnitudes |ρ_{0,k}| from covariance R.

    Parameters
    ----------
    R : (5, 5) complex128

    Returns
    -------
    (4,) float32 — coherence magnitudes in [0, 1]
    """
    diag = np.real(np.diag(R))
    denom = np.sqrt(np.maximum(diag[0] * diag[1:5], 1e-30))
    rho = np.abs(R[0, 1:5]) / denom
    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def eigenvalue_spread_feature(R: np.ndarray) -> float:
    """Eigenvalue spread λ_max/λ_min [dB] — signal quality proxy.

    Parameters
    ----------
    R : (5, 5) complex128

    Returns
    -------
    float32 — spread in dB (positive; typically 6–30 for Iridium)
    """
    ev = np.linalg.eigvalsh(R)
    ev = np.sort(ev)[::-1]  # descending
    ev_pos = np.maximum(ev, 1e-30)
    spread_db = 10.0 * np.log10(ev_pos[0] / ev_pos[-1])
    return np.float32(spread_db)


def extract_feature_vector(
    R: np.ndarray,
    doppler_hz: float = 0.0,
    doppler_scale: float = 40_000.0,
) -> np.ndarray:
    """Full 14-feature vector from one burst covariance.

    Parameters
    ----------
    R : (5, 5) complex128 — covariance (canonical channel order, after FBA)
    doppler_hz : float — estimated Doppler offset [Hz]
    doppler_scale : float — normalization (max expected Doppler ≈ ±40 kHz)

    Returns
    -------
    (14,) float32 — feature vector
    """
    feat = np.empty(14, dtype=np.float32)
    feat[0:8] = phase_diff_features(R)
    feat[8:12] = coherence_features(R)
    feat[12] = eigenvalue_spread_feature(R)
    feat[13] = np.float32(doppler_hz / doppler_scale)
    return feat


def encode_target(az_deg: float, el_deg: float) -> np.ndarray:
    """Encode (az, el) as (cos az, sin az, el_deg/90) for regression.

    Azimuth in cos/sin avoids wrap-around discontinuity.
    Elevation normalized to [0, 1] for balanced gradients.

    Parameters
    ----------
    az_deg : float — azimuth [0, 360)
    el_deg : float — elevation [0, 90]

    Returns
    -------
    (3,) float32 — [cos(az_rad), sin(az_rad), el_deg / 90]
    """
    az_rad = np.deg2rad(az_deg)
    return np.array([np.cos(az_rad), np.sin(az_rad), el_deg / 90.0],
                    dtype=np.float32)


def decode_target(y: np.ndarray) -> tuple[float, float]:
    """Decode (cos az, sin az, el_norm) → (az_deg, el_deg).

    Parameters
    ----------
    y : (3,) or (N, 3) float — encoded targets

    Returns
    -------
    (az_deg, el_deg) or ((N,), (N,)) arrays
    """
    if y.ndim == 1:
        az_deg = float(np.rad2deg(np.arctan2(y[1], y[0])) % 360.0)
        el_deg = float(np.clip(y[2] * 90.0, 0.0, 90.0))
        return az_deg, el_deg
    az_deg = np.rad2deg(np.arctan2(y[:, 1], y[:, 0])) % 360.0
    el_deg = np.clip(y[:, 2] * 90.0, 0.0, 90.0)
    return az_deg, el_deg


# ═══════════════════════════════════════════════════════════════════════════════
# Batch extraction from recording
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_from_recording(
    frames: np.ndarray,
    timestamps: np.ndarray,
    meta: dict,
    *,
    uw_score_min: float = 0.4,
    eig_spread_min_db: float = 6.0,
    papr_min_db: float = 1.0,
) -> dict:
    """Run the full DSP pipeline on a recording and extract features.

    Parameters
    ----------
    frames : (N, 5, N_burst) complex64 — raw burst IQ
    timestamps : (N,) float64 — burst onset [ms from session start]
    meta : dict — sidecar metadata

    Returns
    -------
    dict with keys:
        features     : (M, 14)  float32 — feature vectors (accepted bursts only)
        doppler_hz   : (M,)     float64 — per-burst Doppler
        papr_db      : (M,)     float32 — per-burst PAPR
        burst_indices: (M,)     int32   — original burst index
        timestamps   : (M,)     float64 — burst timestamps [ms]
        az_music     : (M,)     float32 — MUSIC azimuth estimate [deg]
        el_music     : (M,)     float32 — MUSIC elevation estimate [deg]
    """
    # Lazy imports to keep module lightweight
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
        eigenvalue_spread_db as eig_spread,
        reorder_cross_array_channels,
    )

    N = frames.shape[0]
    FS = float(meta.get("sample_rate_hz", 1_024_000))
    D_LAMBDA = float(meta.get("d_lambda", 0.5))
    N_AZ = int(meta.get("n_az", 72))
    N_EL = int(meta.get("n_el", 18))
    EL_MIN = float(meta.get("el_min_deg", 5.0))
    INPUT_ORDER = tuple(meta.get("antenna_input_order",
                                 ["center", "north", "east", "south", "west"]))

    cfg = CrossArrayConfig(
        d_lambda=D_LAMBDA, n_az=N_AZ, n_el=N_EL,
        el_min_deg=EL_MIN, num_expected_signals=1,
    )

    # FBA helper
    def _fba(R):
        M = R.shape[0]
        J = np.fliplr(np.eye(M))
        return 0.5 * (R + J @ R.conj() @ J)

    # Pre-allocate (worst case = all accepted)
    feat_list = []
    dop_list = []
    papr_list = []
    idx_list = []
    ts_list = []
    az_list = []
    el_list = []
    cov_list = []

    for i in range(N):
        X = reorder_cross_array_channels(
            frames[i].astype(np.complex128), INPUT_ORDER
        )

        # Doppler compensation
        try:
            X, dop = compensate_doppler(X, sample_rate=int(FS))
        except Exception:
            continue
        dop_hz = float(dop)

        # Narrowband filter
        X_filt = narrowband_filter_burst(X, sample_rate=int(FS))

        # UW validation
        _, uw_score = validate_burst_uw(X_filt, sample_rate=int(FS))
        if uw_score < uw_score_min:
            continue

        # Covariance + FBA
        R = _fba(compute_single_shot_covariance(X_filt))

        # Eigenvalue spread gate
        ev = eig_spread(R)
        spread = float(ev[0] - ev[-1])
        if spread < eig_spread_min_db:
            continue

        # 2D-MUSIC for reference estimate
        spec = doa_music_2d(X_filt, cfg, R_in=R)
        az, el, papr = find_peak_2d(spec, cfg)
        if papr < papr_min_db:
            continue

        # Extract features
        feat = extract_feature_vector(R, doppler_hz=dop_hz)

        feat_list.append(feat)
        dop_list.append(dop_hz)
        papr_list.append(papr)
        idx_list.append(i)
        ts_list.append(float(timestamps[i]))
        az_list.append(az)
        el_list.append(el)
        cov_list.append(R.copy())

    M = len(feat_list)
    return {
        "features":      np.array(feat_list, dtype=np.float32) if M else np.empty((0, 14), dtype=np.float32),
        "doppler_hz":    np.array(dop_list, dtype=np.float64) if M else np.empty(0, dtype=np.float64),
        "papr_db":       np.array(papr_list, dtype=np.float32) if M else np.empty(0, dtype=np.float32),
        "burst_indices": np.array(idx_list, dtype=np.int32) if M else np.empty(0, dtype=np.int32),
        "timestamps":    np.array(ts_list, dtype=np.float64) if M else np.empty(0, dtype=np.float64),
        "az_music":      np.array(az_list, dtype=np.float32) if M else np.empty(0, dtype=np.float32),
        "el_music":      np.array(el_list, dtype=np.float32) if M else np.empty(0, dtype=np.float32),
        "cov_matrices":  np.array(cov_list, dtype=np.complex128) if M else np.empty((0, 5, 5), dtype=np.complex128),
    }
