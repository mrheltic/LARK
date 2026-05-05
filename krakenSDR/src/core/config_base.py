"""
core.config_base
================
Dataclass-based typed configuration hierarchy shared across all krakenSDR apps.

Motivation
----------
Legacy apps use flat ``config.py`` modules with bare module-level names
(``FREQ_HZ``, ``GAIN_DB``, …).  This works, but:

* No IDE auto-complete / type checking across modules.
* No default-override pattern  — every app re-declares every constant.
* Impossible to instantiate multiple configs at runtime (e.g. scan multiple
  frequencies in parallel).

This module provides:

1. Typed dataclasses with SaneDefaults for the 868 MHz / KrakenSDR setup.
2. Each dataclass has a ``from_module(module)`` classmethod that reads the
   corresponding legacy attributes with safe ``getattr()`` fallbacks.
3. ``load_config(module_path)`` loads a flat module by dotted path and returns
   a full config tuple.

Backward compatibility
----------------------
All existing code that reads ``config.module.CONSTANT`` continues to work
without changes.  The new API is additive.

Usage
-----

    # Load from a flat legacy module:
    from core.config_base import load_config
    hw, arr, doa, burst, ui = load_config("apps.doa_test_868.config")

    # Or build from scratch:
    from core.config_base import HardwareConfig, ArrayConfig, DoAConfig
    hw  = HardwareConfig(freq_hz=868_100_000, gain_db=40)
    arr = ArrayConfig(radius_lambda=0.4253)
    doa = DoAConfig(algorithm="MUSIC", num_signals=2)
"""

from __future__ import annotations

import importlib
import types
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "HardwareConfig",
    "ArrayConfig",
    "DoAConfig",
    "BurstConfig",
    "UIConfig",
    "load_config",
]


# =============================================================================
# Internal helpers
# =============================================================================

def _g(module: Any, name: str, default: Any) -> Any:
    """getattr with default — thin convenience wrapper."""
    return getattr(module, name, default)


# =============================================================================
# HardwareConfig
# =============================================================================

@dataclass
class HardwareConfig:
    """
    Hardware-level constants: network addresses, ADC sample rate, RF parameters.

    Maps to: config_hw.py + FREQ_HZ / GAIN_DB / SQUELCH_* from app config.
    """
    freq_hz:          int   = 868_100_000    # RF centre frequency [Hz]
    sample_rate_hz:   int   = 1_024_000      # ADC sample rate [Hz]
    gain_db:          float = 40.0           # IF gain [dB]
    squelch_enabled:  bool  = True
    squelch_db:       float = -55.0          # minimum RX power [dBW]
    heimdall_host:    str   = "127.0.0.1"
    heimdall_port:    int   = 5000
    heimdall_ctrl:    int   = 5001
    n_antennas:       int   = 5

    @classmethod
    def from_module(cls, m: types.ModuleType) -> "HardwareConfig":
        return cls(
            freq_hz         = _g(m, "FREQ_HZ",             cls.freq_hz),
            sample_rate_hz  = int(_g(m, "SAMPLE_RATE_HZ",  cls.sample_rate_hz)),
            gain_db         = _g(m, "GAIN_DB",             cls.gain_db),
            squelch_enabled = _g(m, "SQUELCH_ENABLED",     cls.squelch_enabled),
            squelch_db      = _g(m, "SQUELCH_THRESHOLD_DB", cls.squelch_db),
            heimdall_host   = _g(m, "HEIMDALL_HOST",        cls.heimdall_host),
            heimdall_port   = _g(m, "HEIMDALL_PORT",        cls.heimdall_port),
            heimdall_ctrl   = _g(m, "HEIMDALL_CTRL",        cls.heimdall_ctrl),
            n_antennas      = _g(m, "N_ANTENNAS",           cls.n_antennas),
        )


# =============================================================================
# ArrayConfig
# =============================================================================

@dataclass
class ArrayConfig:
    """
    Physical antenna array geometry and per-channel calibration.

    Maps to: GEOMETRY / RADIUS_LAMBDA / ANT_CCW / ANT0_OFFSET_DEG /
             CHANNEL_PHASE_OFFSETS_DEG / AMPLITUDE_NORMALIZE in app config.
    """
    n_antennas:           int   = 5
    geometry:             Literal["UCA", "ULA", "CROSS"] = "UCA"
    radius_lambda:        float = 0.4253     # UCA radius in wavelengths
    ant_ccw:              bool  = False      # True → antennas CCW viewed from above
    ant0_offset_deg:      float = 0.0        # antenna-0 offset from North [deg]
    phase_offsets_deg:    list  = field(default_factory=lambda: [0.0] * 5)
    amplitude_normalize:  bool  = True

    @classmethod
    def from_module(cls, m: types.ModuleType) -> "ArrayConfig":
        n = int(_g(m, "N_ANTENNAS", cls.n_antennas))
        return cls(
            n_antennas          = n,
            geometry            = _g(m, "GEOMETRY",                   cls.geometry),
            radius_lambda       = _g(m, "RADIUS_LAMBDA",               cls.radius_lambda),
            ant_ccw             = _g(m, "ANT_CCW",                     cls.ant_ccw),
            ant0_offset_deg     = _g(m, "ANT0_OFFSET_DEG",             cls.ant0_offset_deg),
            phase_offsets_deg   = list(_g(m, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * n)),
            amplitude_normalize = _g(m, "AMPLITUDE_NORMALIZE",         cls.amplitude_normalize),
        )


# =============================================================================
# DoAConfig
# =============================================================================

@dataclass
class DoAConfig:
    """
    Direction-of-Arrival estimation parameters.

    Maps to: DOA_ALGORITHM / NUM_SIGNALS / MUSIC_DECORR / N_AZ / N_EL /
             EL_MIN_DEG / EL_MAX_DEG / COV_ALPHA / MULTI_BURST_N in app config.
    """
    algorithm:         Literal["MUSIC", "CAPON", "BARTLETT",
                               "ROOT-MUSIC", "UNITARY-ESPRIT",
                               "MFBA-MUSIC"] = "MUSIC"
    num_signals:       int   = 2
    music_decorr:      Literal["none", "fb", "circulant", "both"] = "fb"
    capon_decorr:      Literal["none", "fb"] = "none"
    n_az:              int   = 360
    n_el:              int   = 72
    el_min_deg:        float = 5.0
    el_max_deg:        float = 65.0
    cov_alpha:         float = 0.95      # EMA smoothing across frames
    multi_burst_n:     int   = 1         # average this many R matrices before DoA
    snr_adaptive:      bool  = True
    snr_high_db:       float = 10.0
    snr_low_db:        float = 4.0
    high_el_threshold: float = 50.0      # switch algorithm above this elevation
    high_el_algo:      str   = "BARTLETT"

    @classmethod
    def from_module(cls, m: types.ModuleType) -> "DoAConfig":
        return cls(
            algorithm         = _g(m, "DOA_ALGORITHM",       cls.algorithm),
            num_signals       = _g(m, "NUM_SIGNALS",          cls.num_signals),
            music_decorr      = _g(m, "MUSIC_DECORR",         cls.music_decorr),
            capon_decorr      = _g(m, "CAPNT_DECORR",         cls.capon_decorr),
            n_az              = _g(m, "N_AZ",                 cls.n_az),
            n_el              = _g(m, "N_EL",                 cls.n_el),
            el_min_deg        = _g(m, "EL_MIN_DEG",           cls.el_min_deg),
            el_max_deg        = _g(m, "EL_MAX_DEG",           cls.el_max_deg),
            cov_alpha         = _g(m, "COV_ALPHA",            cls.cov_alpha),
            multi_burst_n     = _g(m, "MULTI_BURST_N",        cls.multi_burst_n),
            snr_adaptive      = _g(m, "SNR_ADAPTIVE_ENABLED", cls.snr_adaptive),
            snr_high_db       = _g(m, "SNR_HIGH_DB",          cls.snr_high_db),
            snr_low_db        = _g(m, "SNR_LOW_DB",           cls.snr_low_db),
            high_el_threshold = _g(m, "HIGH_EL_THRESHOLD_DEG", cls.high_el_threshold),
            high_el_algo      = _g(m, "HIGH_EL_ALGO",          cls.high_el_algo),
        )


# =============================================================================
# BurstConfig
# =============================================================================

@dataclass
class BurstConfig:
    """
    Burst detection and preamble extraction parameters.

    Maps to: PAPR_INST_MIN_DB / EIG_SPREAD_MIN_DB / PREAMBLE_BPF_BW_HZ /
             AZ_SMOOTH_ALPHA / EL_SMOOTH_ALPHA / MULTI_BURST_N / etc.
    """
    # IRA physical layer constants
    symbol_rate_hz:        int   = 25_000
    burst_syms:            int   = 245   # total IRA burst symbols
    preamble_syms:         int   = 64    # IRA preamble symbols (all-zero → pure tone)
    superframe_s:          float = 0.090  # TDMA superframe period [s]

    # Detection windows
    energy_win:            int   = 256   # short-window RMS energy detector
    tone_scan_win:         int   = 512   # FFT window for preamble-onset search

    # Bandpass filter
    preamble_bpf_bw_hz:    float = 10_000.0  # BPF extraction bandwidth [Hz]
    tone_search_bw_hz:     float = 100_000.0  # initial tone frequency search range

    # Acceptance gate thresholds
    papr_min_db:           float = 12.0  # minimum instantaneous MUSIC PAPR
    eig_spread_min_db:     float = 0.5   # minimum λ_max − λ_noise spread [dB]
    eig_inst_max_db:       float = 50.0  # maximum λ₁ (ADC saturation advisory)
    eig_sn_gap_min_db:     float = 0.0   # minimum λ₂ − λ₃ gap

    # Tracking smoothing
    az_smooth_alpha:       float = 0.60  # circular EMA alpha for azimuth
    el_smooth_alpha:       float = 0.50  # linear EMA alpha for elevation

    # Outlier gate
    az_outlier_enabled:    bool  = True
    az_outlier_max_dev:    float = 45.0  # maximum deviation from circular median [deg]
    az_outlier_min_hist:   int   = 5     # minimum history before outlier gate activates
    az_outlier_reset_after: int  = 3     # force-relock after N consecutive rejects

    # Phase coherence gate
    phase_coh_enabled:     bool  = True
    phase_coh_max_jump:    float = 60.0  # max per-channel phase jump [deg]

    @classmethod
    def from_module(cls, m: types.ModuleType) -> "BurstConfig":
        return cls(
            preamble_bpf_bw_hz     = _g(m, "PREAMBLE_BPF_BW_HZ",    cls.preamble_bpf_bw_hz),
            tone_search_bw_hz      = _g(m, "TONE_SEARCH_BW_HZ",      cls.tone_search_bw_hz),
            papr_min_db            = _g(m, "PAPR_INST_MIN_DB",        cls.papr_min_db),
            eig_spread_min_db      = _g(m, "EIG_SPREAD_MIN_DB",       cls.eig_spread_min_db),
            eig_inst_max_db        = _g(m, "EIG_INST_MAX_DB",         cls.eig_inst_max_db),
            eig_sn_gap_min_db      = _g(m, "EIG_SN_GAP_MIN_DB",       cls.eig_sn_gap_min_db),
            az_smooth_alpha        = _g(m, "AZ_SMOOTH_ALPHA",          cls.az_smooth_alpha),
            el_smooth_alpha        = _g(m, "EL_SMOOTH_ALPHA",          cls.el_smooth_alpha),
            az_outlier_enabled     = _g(m, "AZ_OUTLIER_ENABLED",       cls.az_outlier_enabled),
            az_outlier_max_dev     = _g(m, "AZ_OUTLIER_MAX_DEV_DEG",   cls.az_outlier_max_dev),
            az_outlier_min_hist    = _g(m, "AZ_OUTLIER_MIN_HISTORY",   cls.az_outlier_min_hist),
            az_outlier_reset_after = _g(m, "AZ_OUTLIER_RESET_AFTER",   cls.az_outlier_reset_after),
            phase_coh_enabled      = _g(m, "PHASE_COHERENCE_ENABLED",  cls.phase_coh_enabled),
            phase_coh_max_jump     = _g(m, "PHASE_COHERENCE_MAX_JUMP_DEG", cls.phase_coh_max_jump),
        )


# =============================================================================
# UIConfig
# =============================================================================

@dataclass
class UIConfig:
    """UI / animation parameters."""
    update_interval_ms: int   = 200
    history_len:        int   = 60
    spec_ema:           float = 0.20

    @classmethod
    def from_module(cls, m: types.ModuleType) -> "UIConfig":
        return cls(
            update_interval_ms = _g(m, "UPDATE_INTERVAL_MS", cls.update_interval_ms),
            history_len        = _g(m, "HISTORY_LEN",         cls.history_len),
        )


# keeping the old name as alias for backward compat
DisplayConfig = UIConfig


# =============================================================================
# load_config
# =============================================================================

def load_config(
    module_path: str,
) -> tuple[HardwareConfig, ArrayConfig, DoAConfig, BurstConfig, UIConfig]:
    """
    Load a flat legacy ``config.py`` module and return typed dataclass instances.

    Parameters
    ----------
    module_path : str
        Dotted Python import path, e.g. ``"apps.doa_test_868.config"``.

    Returns
    -------
    (HardwareConfig, ArrayConfig, DoAConfig, BurstConfig, UIConfig)

    Example
    -------
        hw, arr, doa, burst, ui = load_config("apps.doa_test_868.config")
        print(f"RX at {hw.freq_hz / 1e6:.3f} MHz, gain={hw.gain_db:.0f} dB")
        print(f"Array: {arr.n_antennas}-ant {arr.geometry}, r={arr.radius_lambda:.4f}λ")
    """
    m = importlib.import_module(module_path)
    return (
        HardwareConfig.from_module(m),
        ArrayConfig.from_module(m),
        DoAConfig.from_module(m),
        BurstConfig.from_module(m),
        UIConfig.from_module(m),
    )
