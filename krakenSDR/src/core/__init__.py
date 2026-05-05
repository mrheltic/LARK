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

# Calibration features
from .calibration_features import (
    phase_diff_features,
    coherence_features,
    eigenvalue_spread_feature,
    extract_feature_vector,
    encode_target,
    decode_target,
)

# Calibration model
from .calibration_model import (
    CalibrationMLP,
    TrainConfig,
    angular_loss,
    angular_loss_grad,
    train,
)

# UCA 2D DoA
from .doa_uca_2d import (
    UcaConfig,
    find_peak_uca_2d,
    extract_pilot_tone,
    amplitude_normalize_channels,
    doa_music_uca_2d,
    doa_bartlett_uca_2d,
    doa_capon_uca_2d,
    eigenvalue_spread_uca_db,
    snr_uca_db,
    CovarianceAccumulatorUca,
    doa_root_music_uca_2d,
    doa_unitary_esprit_uca_2d,
    doa_mfba_music_uca_2d,
    enhanced_preprocessing,
)

# Burst pipeline
from .burst_pipeline import (
    PipelineResult,
    BurstPipeline,
)

# ── New LEGO modular API ──────────────────────────────────────────────────────

# Tracking filters (Kalman, EMA)
from .tracking import (
    CircularEMA,
    ScalarEMA,
    KalmanScalar,
    KalmanAngular,
    circular_ema_batch,
    scalar_ema_batch,
)

# Tone extraction and narrowband BPF
from .tone_extraction import (
    find_preamble_onset,
    find_tone_onset,
    extract_pilot_tone as extract_pilot_tone_core,
    narrowband_filter_fft,
)

# Acceptance gates
from .gates import (
    GateVerdict,
    circ_median_deg,
    PAGGate,
    EigenGate,
    BoundaryGate,
    OutlierGate,
    GatePipeline,
)

# Typed configuration hierarchy
from .config_base import (
    HardwareConfig,
    ArrayConfig as HardwareArrayConfig,
    DoAConfig,
    BurstConfig,
    UIConfig,
    DisplayConfig,
    load_config,
)

# Iridium demodulator
from .iridium_demod import (
    IridiumDemod,
    DemodDebug,
    SYMBOLS_PER_SECOND,
    UW_LENGTH,
    DOWNLINK,
    UPLINK,
    UW_DOWNLINK,
    UW_UPLINK,
    PREAMBLE_LENGTH,
)

# Aliases for naming consistency
papr_db = compute_papr

__all__ = [
    # === New Modular API (Recommended) ===
    # Array geometry
    "ArrayGeometryBase",
    "UniformLinearArray",
    "UniformCircularArray",
    "CrossArray",
    "GeometryType",
    "compute_steering_matrix_1d",
    "compute_steering_matrix_2d",
    "reorder_channels",
    "CROSS_ARRAY_CANONICAL_ORDER",
    # Covariance operations
    "covariance",
    "sample_covariance",
    "forward_backward_avg",
    "toeplitzify",
    "fb_toeplitz",
    "apply_decorrelation",
    "spatial_smoothing",
    "CovarianceAccumulator",
    # DoA estimators
    "music",
    "capon",
    "bartlett",
    "root_music",
    "esprit",
    "music_2d",
    "capon_2d",
    "subspace_decomposition",
    "estimate_signal_count",
    "peak_interpolation_1d",
    "peak_interpolation_2d",
    "normalize_spectrum_db",
    "compute_papr",
    # Burst processing
    "BurstDetector",
    "BurstResult",
    "PassTracker",
    "IRD_CHANS",
    "TDMA_FRAME_S",
    "TDMA_SLOT_S",
    "MAX_DOP_HZ",
    "PILOT_TONE_OFFSET_HZ",
    # Iridium-specific burst processing
    "detect_and_extract_burst",
    "detect_and_extract_all_bursts",
    "compensate_doppler",
    "compute_single_shot_covariance",
    "validate_burst_uw",
    "narrowband_filter_burst",
    # Signal quality
    "channel_power_balance",
    "ChannelPowerReport",
    "check_recording_health",
    "WEAK_CHANNEL_THRESHOLD",
    "CRITICAL_CHANNEL_THRESHOLD",
    # Calibration features
    "phase_diff_features",
    "coherence_features",
    "eigenvalue_spread_feature",
    "extract_feature_vector",
    "encode_target",
    "decode_target",
    # Calibration model
    "CalibrationMLP",
    "TrainConfig",
    "angular_loss",
    "angular_loss_grad",
    "train",
    # UCA 2D DoA
    "UcaConfig",
    "find_peak_uca_2d",
    "extract_pilot_tone",
    "amplitude_normalize_channels",
    "doa_music_uca_2d",
    "doa_bartlett_uca_2d",
    "doa_capon_uca_2d",
    "eigenvalue_spread_uca_db",
    "snr_uca_db",
    "CovarianceAccumulatorUca",
    "doa_root_music_uca_2d",
    "doa_unitary_esprit_uca_2d",
    "doa_mfba_music_uca_2d",
    "enhanced_preprocessing",
    # Burst pipeline
    "PipelineResult",
    "BurstPipeline",
    # Tracking filters
    "CircularEMA",
    "ScalarEMA",
    "KalmanScalar",
    "KalmanAngular",
    "circular_ema_batch",
    "scalar_ema_batch",
    # Tone extraction
    "find_preamble_onset",
    "find_tone_onset",
    "extract_pilot_tone_core",
    "narrowband_filter_fft",
    # Acceptance gates
    "GateVerdict",
    "circ_median_deg",
    "PAGGate",
    "EigenGate",
    "BoundaryGate",
    "OutlierGate",
    "GatePipeline",
    # Typed configuration
    "HardwareConfig",
    "HardwareArrayConfig",
    "DoAConfig",
    "BurstConfig",
    "UIConfig",
    "DisplayConfig",
    "load_config",
    # Iridium demodulator
    "IridiumDemod",
    "DemodDebug",
    "SYMBOLS_PER_SECOND",
    "UW_LENGTH",
    "DOWNLINK",
    "UPLINK",
    "UW_DOWNLINK",
    "UW_UPLINK",
    "PREAMBLE_LENGTH",
]
