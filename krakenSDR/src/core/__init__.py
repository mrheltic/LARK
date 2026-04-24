"""
pysdr_doa.core
==============
Re-exports all public symbols from doa_algorithms for clean import paths:

    from core import ArrayConfig, Geometry, doa_music, ...
    from core.burst import BurstDetector, BurstResult, PassTracker
"""

from __future__ import annotations

# Re-export everything from the existing monolithic module so callers that
# already do `from doa_algorithms import ...` keep working, while new code can
# use the sub-package path `from core import ...`.
from .doa_algorithms import (   # noqa: F401
    ArrayConfig,
    Geometry,
    steering,
    covariance,
    forward_backward_avg,
    toeplitzify,
    fb_toeplitz,
    apply_decorrelation,
    uca_to_vula,
    compute_doa_covariance,
    doa_music,
    doa_capon,
    doa_ml,
    doa_root_music,
    doa_esprit,
    apply_phase_correction,
    measure_power_db,
    snr_from_covariance,
    papr_db,
    condition_number,
    eigenvalue_spread_db,
    CovarianceAccumulator,
    coherence_matrix,
)

from .burst import BurstDetector, BurstResult, PassTracker  # noqa: F401

from .iridium_doa_burst import (   # noqa: F401
    detect_and_extract_burst,
    compensate_doppler,
    compute_single_shot_covariance,
)

from .doa_algorithms_3d import (  # noqa: F401
    estimate_signal_count,
    SatellitePassAccumulator,
    doa_music_2d,
    doa_bartlett_2d,
    doa_capon_2d,
    doa_iaa_2d,
    find_peak_2d,
    CrossArrayConfig,
    CROSS_ARRAY_CANONICAL_ORDER,
)

from .signal_quality import (   # noqa: F401
    channel_power_balance,
    check_recording_health,
    ChannelPowerReport,
    papr_db_from_spectrum,
    WEAK_CHANNEL_THRESHOLD,
    CRITICAL_CHANNEL_THRESHOLD,
)
