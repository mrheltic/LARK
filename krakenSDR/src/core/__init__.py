"""
core — KrakenSDR signal processing library
==========================================

Direction-of-Arrival estimation, burst processing, and session recording.
"""

from __future__ import annotations

# ── 2D UCA DoA (primary engine) ────────────────────────────────────────────
from .doa_uca_2d import (
    UcaConfig,
    CovarianceAccumulatorUca,
    doa_music_uca_2d,
    doa_capon_uca_2d,
    doa_bartlett_uca_2d,
    find_peak_uca_2d,
    find_peaks_uca_2d,
    pick_doa_peak_uca_2d,
    extract_pilot_tone,
    amplitude_normalize_channels,
    eigenvalue_spread_uca_db,
    snr_uca_db,
    crb_azimuth_deg,
    doa_root_music_uca_2d,
    doa_unitary_esprit_uca_2d,
    doa_mfba_music_uca_2d,
    enhanced_preprocessing,
)

# ── Legacy DoA algorithms (ULA/UCA) + phase calibration ───────────────────
from .doa_algorithms import (
    Geometry,
    ArrayConfig,
    covariance,
    forward_backward_avg,
    toeplitzify,
    fb_toeplitz,
    apply_decorrelation,
    CovarianceAccumulator,
    steering,
    uca_to_vula,
    compute_doa_covariance,
    doa_music,
    doa_capon,
    doa_ml,
    doa_root_music,
    doa_esprit,
    apply_phase_correction,
    papr_db,
    condition_number,
)

# ── Cross-array 3D DoA (space/satellite) ──────────────────────────────────
from .doa_algorithms_3d import (
    CrossArrayConfig,
    doa_music_2d,
    doa_bartlett_2d,
    doa_capon_2d,
    doa_iaa_2d,
    find_peak_2d,
    reorder_cross_array_channels,
    estimate_signal_count,
    CROSS_ARRAY_CANONICAL_ORDER,
    normalize_cross_array_order,
)

# ── Burst DSP pipeline (energy → tone scan → BPF → MF covariance) ────────
from .burst_processing import (
    detect_energy_bursts,
    scan_preamble_tones,
    apply_bpf_and_normalize,
    compute_mf_covariance,
)

# ── Multi-peak DOA (per-burst multi-tone covariance) ──────────────────────
from .multi_peak import process_cpi_for_multi

# ── Session recording (crash-safe incremental frame + spectra writer) ─────
from .recording import (
    SessionRecorder,
    consolidate_session,
    default_record_dir,
    list_session_raw_frames,
    count_session_raw_frames,
    iter_session_raw_frames,
)

# ── Pipeline debug (per-stage intermediate data dumps) ────────────────────
from .pipeline_debug import (
    PipelineDebugSaver,
    STAGE_FILES,
    save_pipeline_stage,
)

# ── Track clustering (online burst-to-satellite association) ──────────────
from .track_clusterer import (
    cluster_from_jsonl,
    load_tracks_json,
)

# ── Tracking filters (EMA, Kalman) ────────────────────────────────────────
from .tracking import (
    CircularEMA,
    ScalarEMA,
    KalmanScalar,
    KalmanAngular,
    circular_ema_batch,
    scalar_ema_batch,
)

# ── Acceptance gates ──────────────────────────────────────────────────────
from .gates import (
    GateVerdict,
    circ_median_deg,
    PAGGate,
    EigenGate,
    BoundaryGate,
    OutlierGate,
    GatePipeline,
)

# ── Tone extraction & narrowband BPF ──────────────────────────────────────
from .tone_extraction import (
    find_preamble_onset,
    find_tone_onset,
    extract_pilot_tone as extract_pilot_tone_core,
    narrowband_filter_fft,
)

# ── Low-level covariance (used by doa_algorithms re-exports) ──────────────
from .covariance import (
    sample_covariance,
    spatial_smoothing,
    DecorrelationMethod,
)
