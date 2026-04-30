"""
core — KrakenSDR signal processing library
==========================================

Refactored modular API for Direction-of-Arrival estimation.

New recommended imports (clean, modular):
    from core.array_geometry import UniformCircularArray, CrossArray
    from core.covariance import covariance, apply_decorrelation, CovarianceAccumulator
    from core.doa_estimators import music, capon, root_music, esprit
    from core.burst import BurstDetector, BurstResult

Legacy compatibility (still supported):
    from core import ArrayConfig, doa_music, ...  # old monolithic API
"""

from __future__ import annotations

# =============================================================================
# New modular API (recommended)
# =============================================================================

# Array geometry
from .array_geometry import (
    ArrayGeometryBase,
    UniformLinearArray,
    UniformCircularArray,
    CrossArray,
    GeometryType,
    compute_steering_matrix_1d,
    compute_steering_matrix_2d,
    reorder_channels,
    CROSS_ARRAY_CANONICAL_ORDER,
)

# Covariance operations
from .covariance import (
    covariance,
    sample_covariance,
    forward_backward_avg,
    toeplitzify,
    fb_toeplitz,
    apply_decorrelation,
    spatial_smoothing,
    CovarianceAccumulator,
)

# DoA estimators
from .doa_estimators import (
    music,
    capon,
    bartlett,
    root_music,
    esprit,
    music_2d,
    capon_2d,
    subspace_decomposition,
    estimate_signal_count,
    peak_interpolation_1d,
    peak_interpolation_2d,
    normalize_spectrum_db,
    compute_papr,
)

# Burst processing
from .burst import (
    BurstDetector,
    BurstResult,
    PassTracker,
    IRD_CHANS,
    TDMA_FRAME_S,
    TDMA_SLOT_S,
    MAX_DOP_HZ,
    PILOT_TONE_OFFSET_HZ,
)

# Iridium-specific burst processing
from .iridium_doa_burst import (
    detect_and_extract_burst,
    detect_and_extract_all_bursts,
    compensate_doppler,
    compute_single_shot_covariance,
    validate_burst_uw,
    narrowband_filter_burst,
)

# Signal quality metrics
from .signal_quality import (
    channel_power_balance,
    ChannelPowerReport,
    check_recording_health,
    WEAK_CHANNEL_THRESHOLD,
    CRITICAL_CHANNEL_THRESHOLD,
)

# =============================================================================
# Legacy compatibility exports
# =============================================================================

# These maintain backward compatibility with existing code
from .doa_algorithms import (
    ArrayConfig as _ArrayConfig,
    Geometry as _Geometry,
    steering as _steering,
    doa_music as _doa_music,
    doa_capon as _doa_capon,
    doa_ml as _doa_ml,
    doa_root_music as _doa_root_music,
    doa_esprit as _doa_esprit,
    apply_phase_correction as _apply_phase_correction,
    snr_from_covariance as _snr_from_covariance,
    papr_db as _papr_db_legacy,
    eigenvalue_spread_db as _eigenvalue_spread_db,
    CovarianceAccumulator as _CovarianceAccumulatorLegacy,
    coherence_matrix as _coherence_matrix,
)

from .doa_algorithms_3d import (
    CrossArrayConfig as _CrossArrayConfig,
    doa_music_2d as _doa_music_2d_legacy,
    doa_capon_2d as _doa_capon_2d_legacy,
    doa_bartlett_2d as _doa_bartlett_2d,
    find_peak_2d as _find_peak_2d,
    SatellitePassAccumulator as _SatellitePassAccumulator,
)

# Re-export with original names for compatibility
ArrayConfig = _ArrayConfig
Geometry = _Geometry
doa_music = _doa_music
doa_capon = _doa_capon
doa_ml = _doa_ml
doa_root_music = _doa_root_music
doa_esprit = _doa_esprit
apply_phase_correction = _apply_phase_correction
snr_from_covariance = _snr_from_covariance
eigenvalue_spread_db = _eigenvalue_spread_db
coherence_matrix = _coherence_matrix

# Aliases for naming consistency
papr_db = compute_papr

__all__ = [
    # New modular API
    "ArrayGeometryBase",
    "UniformLinearArray",
    "UniformCircularArray",
    "CrossArray",
    "GeometryType",
    "compute_steering_matrix_1d",
    "compute_steering_matrix_2d",
    "reorder_channels",
    "covariance",
    "apply_decorrelation",
    "CovarianceAccumulator",
    "music",
    "capon",
    "bartlett",
    "root_music",
    "esprit",
    "music_2d",
    "capon_2d",
    "estimate_signal_count",
    "BurstDetector",
    "BurstResult",
    "PassTracker",
    "detect_and_extract_burst",
    "compensate_doppler",
    "compute_single_shot_covariance",
    "channel_power_balance",
    "ChannelPowerReport",
    # Legacy compatibility
    "ArrayConfig",
    "Geometry",
    "doa_music",
    "doa_capon",
    "doa_ml",
    "doa_root_music",
    "doa_esprit",
    "apply_phase_correction",
    "snr_from_covariance",
    "papr_db",
    "eigenvalue_spread_db",
    "coherence_matrix",
]
