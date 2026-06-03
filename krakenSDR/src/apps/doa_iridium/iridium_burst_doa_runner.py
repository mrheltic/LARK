#!/usr/bin/env python3
"""
iridium_burst_doa_runner.py — Multi-satellite burst-gated 2D DoA on Iridium IRA
=========================================================================
2D DoA (azimuth 0-360°, elevation 5-90°) on Iridium IRA bursts; supports
simultaneous reception of up to 3 satellites (discriminated by preamble
Doppler offset), each with an independent tracker and distinct colour.

Hardware setup
--------------
  Indoor  (1626 MHz, near-field / short cable):
    TX → LibreSDR AD9363 @ 1626.270 MHz  (tx/indoor_1626.py)
    RX → KrakenSDR 5-ch UCA RHCP @ 1626.270 MHz  (Heimdall DAQ)
  Outdoor (real Iridium satellites):
    RX → KrakenSDR @ 1626.270 MHz  (no TX needed)

How multi-satellite detection works
-----------------------------------
Every IRA burst begins with 64 preamble symbols (all-zero dibits) producing
a pure tone at fc + Rs/8 = fc + 3125 Hz.  Two satellites in view have
different Doppler offsets (sat at 60° el ≈ ±10 kHz; sat at horizon ≈ ±40 kHz
at 1626 MHz), so their preamble tones land at different frequencies in the
receiver reference frame.

`_scan_doppler_peaks()` runs an FFT scan over ±DOPPLER_SCAN_BW_HZ
(default ±45 kHz) around the nominal tone at 3125 Hz for every burst window,
returning up to MAX_SATELLITES peaks.  A narrow BPF is applied per peak, then
2D-MUSIC DoA is run.  The result is associated with the nearest-CFO satellite
in the active tracker registry.

Usage
-----
    python3 iridium_burst_doa_runner.py              # real hardware (Heimdall active)
    python3 iridium_burst_doa_runner.py --demo       # synthetic 2-satellite simulation
    python3 iridium_burst_doa_runner.py --demo --n-demo-sats 3
    python3 iridium_burst_doa_runner.py --freq 1626.270
    python3 iridium_burst_doa_runner.py --calibrate 45.0 --calibrate-el 25.0
    python3 iridium_burst_doa_runner.py --out-dir /tmp/doa_iridium
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
_DATA_DIR = os.path.normpath(os.path.join(_SRC, "..", "data", "doa_iridium"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

import socket as _socket
import numpy as np
import matplotlib
from burst_processing import (
    detect_energy_bursts,   # replaces removed _detect_bursts() duplicate
    scan_preamble_tones,    # replaces removed _scan_doppler_peaks() duplicate
    compute_mf_covariance,  # ex _compute_mf_covariance_api
)
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

import config as C
from hardware.kraken_iq_source import KrakenIQSource
from core.doa_uca_2d import (
    UcaConfig, CovarianceAccumulatorUca,
    doa_music_uca_2d, doa_capon_uca_2d, doa_bartlett_uca_2d,
    doa_root_music_uca_2d, doa_unitary_esprit_uca_2d, doa_mfba_music_uca_2d,
    find_peak_uca_2d, find_peaks_uca_2d, pick_doa_peak_uca_2d,
    phase_residual_deg,
    doa_phase_fit_uca_2d, uca_synthetic_peak_spectrum,
    eigenvalue_spread_uca_db, snr_uca_db,
    extract_pilot_tone, amplitude_normalize_channels,
    crb_azimuth_deg, estimate_signal_count_mdl,
)
from core.doa_algorithms import apply_phase_correction as _apply_phase_correction
from core.tracking import KalmanAngular, KalmanScalar
from core.gates import circ_median_deg, OutlierGate
from core.tone_extraction import (
    find_preamble_onset as _fpo_core,
    find_tone_onset    as _fto_core,
)

# ── Colour palette ───────────────────────────────────────────────────────────
BG    = "#1a1d27"; BG2   = "#21253a"; BG3  = "#2a2f47"
C_BDR = "#3b4263"; C_MUT = "#8891b0"; C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIO  = "#a78bfa"; C_ROSE = "#f16b6f"; C_LIME  = "#6dd97d"

# Three distinct colors for the three satellite trackers (override via C.SAT_COLORS)
_SAT_COLORS = list(getattr(C, "SAT_COLORS", ["#f4a431", "#4ecdc4", "#a78bfa"]))

# ── Iridium IRA parameters (from gr-iridium / iridium-toolkit) ──────────────
_SYMBOL_RATE   = 25_000           # sps
_SPS_BASE      = 10               # samples/symbol @ 250 kHz
_IRA_SAMPLE_RATE = _SYMBOL_RATE * _SPS_BASE   # 250 000 Hz
_TX_SAMPLE_RATE  = 1_000_000      # AD9363 TX rate
_IRA_UPS         = _TX_SAMPLE_RATE // _IRA_SAMPLE_RATE   # = 4

_PREAMBLE_SYMS = 64
_BURST_SYMS    = 245              # 64 pream + 12 UW + 167 data + 2 tail
_SUPERFRAME_S  = 0.090            # 90 ms between consecutive IRA bursts from the same satellite

# Preamble tone: dibit (0,0) → +π/4 per symbol → pure CW tone at fc + Rs/8
_PREAMBLE_TONE_HZ = _SYMBOL_RATE // 8   # = 3125 Hz

_FS = float(getattr(C, "SAMPLE_RATE_HZ", 1_024_000))

_BURST_SAMPLES = int(round(_BURST_SYMS    * _SPS_BASE * _IRA_UPS * _FS / _TX_SAMPLE_RATE))
_PRE_SAMPLES   = int(round(_PREAMBLE_SYMS * _SPS_BASE * _IRA_UPS * _FS / _TX_SAMPLE_RATE))
_SF_SAMPLES    = int(round(_SUPERFRAME_S  * _FS))

_WINDOW_SAMPLES = min(_BURST_SAMPLES + 512, _SF_SAMPLES - 256)

_ENERGY_WIN    = 256
_TONE_SCAN_WIN = 512
_PAPR_INST_MIN_DB = 8.0
_SPEC_EMA         = 0.25
_CAL_DURATION_S   = 30.0
_CAL_MIN_UPDATES  = 20


def _load_scenario_profile(scenario: str, cli_overrides: dict) -> None:
    """Apply a scenario profile from C.SCENARIO_PROFILES to module C.

    Priority: 1) CLI argument  2) explicit variable in config.py  3) scenario default.
    """
    profiles = getattr(C, "SCENARIO_PROFILES", {})
    profile  = profiles.get(scenario)
    if profile is None:
        print(f"[WARN] Unknown scenario {scenario!r}; falling back to 'indoor_ira'.")
        profile = profiles.get("indoor_ira", {})

    for key, default in profile.items():
        if key.upper() in cli_overrides:
            setattr(C, key.upper(), cli_overrides[key.upper()])
            continue
        if hasattr(C, key.upper()):
            continue
        setattr(C, key.upper(), default)

    # Absolute fallbacks — values that must always exist
    _ensure_defaults = {
        "DOPPLER_SCAN_BW_HZ"      : 45_000,
        "HISTORY_LEN"             : 100,
        "UPDATE_INTERVAL_MS"      : 300,
        "N_AZ"                    : 360,
        "N_EL"                    : 86,
        "EL_MIN_DEG"              : 5.0,
        "EL_MAX_DEG"              : 90.0,
        "AZ_FREEZE_EL_DEG"        : 75.0,
        "SAT_COLORS"              : ["#f4a431", "#4ecdc4", "#a78bfa"],
        "PREAMBLE_BPF_BW_HZ"      : 8_000,
        "EIG_SPREAD_MIN_DB"       : 0.5,
        "PHASE_DISPLAY_EMA_ALPHA" : 0.88,
    }
    for key, val in _ensure_defaults.items():
        if not hasattr(C, key):
            setattr(C, key, val)


# ── Demo parameters (default 2 satellites) ──────────────────────────────────
_DEMO_SATS = [
    dict(az=45.0,   el=30.0, doppler=+8_000,  snr_db=15.0),  # Sat-0 amber
    dict(az=200.0,  el=55.0, doppler=-14_000, snr_db=12.0),  # Sat-1 teal
    dict(az=310.0,  el=20.0, doppler=+28_000, snr_db=10.0),  # Sat-2 violet
]

# =============================================================================
# DoA algorithm dispatcher
# =============================================================================


def _check_narrowband(freq_hz: float, cfg: UcaConfig) -> None:
    """Print a one-time validation of the narrowband array assumption.

    The narrowband assumption (Salama 2025 §2.3) requires that the signal
    bandwidth B satisfies:

        B × τ_max  ≪  1

    where τ_max = 2r / c is the maximum propagation delay across the array
    aperture (diameter / speed of light).

    For Iridium IRA the relevant bandwidth is the preamble symbol rate
    (25 kSps); the 15 kHz BPF used in the pipeline makes it even narrower.
    At 1626 MHz with the 5-element KrakenSDR UCA (r ≈ 7.8 cm):

        τ_max ≈ 2 × 0.078 / 3e8 ≈ 0.52 ns
        B × τ_max = 25 000 × 0.52e-9 ≈ 1.3e-5  ≪  1  ✓

    The narrowband assumption is more than satisfied; wideband ISSM processing
    (Salama 2025 §7.2) is not needed for this setup.
    """
    lam_m   = 3e8 / freq_hz                   # wavelength [m]
    r_m     = cfg.radius_lambda * lam_m        # UCA physical radius [m]
    tau_max = 2.0 * r_m / 3e8                  # max delay across diameter [s]
    B       = float(_SYMBOL_RATE)              # Iridium preamble BW ≈ symbol rate
    ratio   = B * tau_max
    ok      = ratio < 0.01
    status  = "✓  narrowband" if ok else "⚠  wideband effects possible"
    print(
        f"[NARROWBAND CHECK] B×τmax = {ratio:.2e}  {status}\n"
        f"  diameter = {2*r_m*100:.1f} cm, τmax = {tau_max*1e9:.2f} ns, "
        f"BW = {B/1e3:.0f} kHz @ {freq_hz/1e6:.3f} MHz"
    )


def _run_doa_algo(X_cal: np.ndarray, R: np.ndarray,
                  cfg: UcaConfig, algo: str,
                  n_snapshots: int | None = None) -> np.ndarray:
    decorr = getattr(C, "MUSIC_DECORR", "none")
    if algo == "capon":
        return doa_capon_uca_2d(X_cal, cfg, R_in=R, decorr=decorr)
    if algo == "bartlett":
        return doa_bartlett_uca_2d(X_cal, cfg, R_in=R)
    if algo == "root-music":
        return doa_root_music_uca_2d(R, cfg)[0]
    if algo == "unitary-esprit":
        return doa_unitary_esprit_uca_2d(R, cfg)[0]
    if algo == "mfba-music":
        return doa_mfba_music_uca_2d(R, cfg)[0]
    return doa_music_uca_2d(X_cal, cfg, R_in=R, decorr=decorr,
                            n_snapshots=n_snapshots)


def _estimate_doa_burst(
    X_cal: np.ndarray,
    R_avg: np.ndarray,
    cfg: UcaConfig,
    algo: str,
    *,
    indoor: bool,
    phase_diffs: np.ndarray,
    az_hint: float | None,
    el_hint: float | None,
    phase_w: float,
    az_hint_w: float,
    el_hint_w: float,
    mirror_m: float,
    mirror_phase_m: float,
    el_pref_hi: float,
    el_pref_lo: float,
    n_snapshots: int | None,
) -> tuple[float, float, float, np.ndarray]:
    """Run selected DoA algorithm and return (az, el, papr, spec2d)."""
    if algo == "phase-fit":
        az_doa, el_doa, ph_err = doa_phase_fit_uca_2d(
            R_avg, cfg, az_hint_deg=az_hint, el_hint_deg=el_hint,
        )
        papr_doa = float(max(0.0, 12.0 - ph_err))
        spec2d = uca_synthetic_peak_spectrum(cfg, az_doa, el_doa)
        return az_doa, el_doa, papr_doa, spec2d

    spec2d = _run_doa_algo(X_cal, R_avg, cfg, algo, n_snapshots=n_snapshots)
    if spec2d.ndim != 2:
        az_doa, el_doa, papr_doa = find_peak_uca_2d(spec2d, cfg)
    else:
        az_doa, el_doa, papr_doa = pick_doa_peak_uca_2d(
            spec2d, cfg, indoor=indoor,
            el_pref_hi=el_pref_hi, el_pref_lo=el_pref_lo,
            phase_diffs=phase_diffs,
            az_hint_deg=az_hint,
            el_hint_deg=el_hint,
            phase_score_weight=phase_w,
            az_hint_score_weight=az_hint_w,
            el_hint_score_weight=el_hint_w,
            mirror_margin_deg=mirror_m,
            mirror_phase_min_margin_deg=mirror_phase_m,
        )
    return az_doa, el_doa, papr_doa, spec2d


def _apply_doa_frame_offset(
    az_doa: float, el_doa: float, cfg: UcaConfig, has_cal: bool,
) -> tuple[float, float]:
    """Map raw MUSIC (az, el) to the calibrated reference frame."""
    if not has_cal:
        return az_doa, el_doa
    az_off = float(getattr(C, "DOA_AZ_OFFSET_DEG", 0.0))
    el_off = float(getattr(C, "DOA_EL_OFFSET_DEG", 0.0))
    if az_off == 0.0 and el_off == 0.0:
        return az_doa, el_doa
    az = (float(az_doa) + az_off) % 360.0
    el = float(np.clip(float(el_doa) + el_off, cfg.el_min_deg, cfg.el_max_deg))
    return az, el


def _pick_doa_hints(
    trk: SimpleNamespace, has_cal: bool, az_pick_hint_min: int,
) -> tuple[float | None, float | None]:
    """Tracker EMA hints, or cal-reference hints before the tracker has locked."""
    _tx_mode = str(getattr(C, "INDOOR_TX_MODE", "pass")).lower()
    _track = _tx_mode == "track"
    if (trk.burst_count >= az_pick_hint_min
            and not trk.no_doa and not trk.az_init):
        return float(trk.az_ema), float(trk.el_ema)
    if has_cal and not _track:
        cal_az = getattr(C, "CAL_REFERENCE_AZ_DEG", None)
        if cal_az is not None:
            cal_el = getattr(C, "CAL_REFERENCE_EL_DEG", None)
            el_h = float(cal_el) if cal_el is not None else float(trk.el_ema)
            return float(cal_az), el_h
    return None, None


def _phase_model_angles(
    trk: SimpleNamespace, has_cal: bool,
) -> tuple[float, float]:
    """Az/el used for phase-residual display (cal reference when IRA TX is static)."""
    az = float(trk.az_deg)
    el = float(trk.el_deg)
    _ira_static = (
        str(getattr(C, "INDOOR_TX_MODE", "pass")).lower() == "ira"
        and bool(getattr(C, "INDOOR_IRA_STATIC_PHASE_MODEL", True))
    )
    if _ira_static and has_cal:
        _cal_az = getattr(C, "CAL_REFERENCE_AZ_DEG", None)
        _cal_el = getattr(C, "CAL_REFERENCE_EL_DEG", None)
        if _cal_az is not None:
            az = float(_cal_az)
        if _cal_el is not None:
            el = float(_cal_el)
    return az, el


# =============================================================================
# Per-satellite tracker
# =============================================================================

def _make_sat_tracker(
    sat_id: int, color: str, cfo_hz: float,
    n_el: int, n_az: int, hist_len: int, multi_n: int,
    el_mid: float,
    cal_az: float | None = None,
    cal_el: float | None = None,
) -> "SimpleNamespace":
    az0 = float(cal_az) if cal_az is not None else 0.0
    el0 = float(cal_el) if cal_el is not None else el_mid
    return SimpleNamespace(
        sat_id      = sat_id,
        color       = color,
        cfo_hz      = cfo_hz,
        az_kf       = KalmanAngular(q=5.0, r=20.0),
        el_kf       = KalmanScalar(q=2.0,  r=8.0),
        az_ema      = az0,
        el_ema      = el0,
        az_phasor   = np.exp(1j * np.deg2rad(az0)),
        az_hist     = collections.deque(maxlen=hist_len),
        el_hist     = collections.deque(maxlen=hist_len),
        snr_hist    = collections.deque(maxlen=hist_len),
        cfo_hist    = collections.deque(maxlen=hist_len),
        spec2d      = np.full((n_el, n_az), -40.0),
        R_batch     = collections.deque(maxlen=multi_n),
        X_batch     = collections.deque(maxlen=multi_n),
        cfo_batch   = collections.deque(maxlen=multi_n),
        Y_batch     = collections.deque(maxlen=multi_n),
        burst_count = 0,
        last_seen   = time.monotonic(),
        az_init     = True,
        papr_db     = 0.0,
        snr_db      = 0.0,
        az_deg      = az0,
        el_deg      = el0,
        no_doa      = True,
        az_other    = 0.0, el_other = el_mid, papr_other = 0.0, has_other = False,
        trail       = collections.deque(maxlen=30),
        az_raw_hist = collections.deque(maxlen=hist_len),
        el_raw_hist = collections.deque(maxlen=hist_len),
        az_kf_hist  = collections.deque(maxlen=hist_len),
        el_kf_hist  = collections.deque(maxlen=hist_len),
        cfo_prev_sign = 0,
        el_at_zero_crossing = None,
        t_zero_crossing     = None,
        last_phase_diffs    = None,
        az_reject_streak    = 0,
        ph_reject_streak    = 0,
        gate_bypass_left    = 0,
        az_gate_hist        = collections.deque(maxlen=hist_len),
    )


def _update_tracker(
    trk: "SimpleNamespace",
    az_doa: float, el_doa: float,
    papr_db: float, snr_db: float,
    cfo_hz: float, spec2d: np.ndarray,
    az_alpha: float, el_alpha: float,
    el_min: float, el_max: float,
    az_freeze_el: float = 75.0,
) -> None:
    el_step = max(0.1, (el_max - el_min) / 90.0)
    on_floor   = el_doa < el_min + el_step
    on_ceiling = el_doa > el_max - el_step
    az_frozen  = el_doa >= az_freeze_el
    phasor_new = np.exp(1j * np.deg2rad(az_doa))

    if trk.az_init:
        if not az_frozen and not on_ceiling and not on_floor:
            trk.az_phasor = phasor_new
            trk.az_kf.update(az_doa)
            trk.az_init   = False
    elif not az_frozen:
        trk.az_phasor = az_alpha * trk.az_phasor + (1.0 - az_alpha) * phasor_new

    trk.az_ema = float(np.degrees(np.angle(trk.az_phasor)) % 360.0)
    if not az_frozen:
        trk.az_kf.update(az_doa)
    if not on_floor and not on_ceiling:
        trk.el_ema = el_alpha * trk.el_ema + (1.0 - el_alpha) * el_doa
    trk.el_kf.update(el_doa)

    # Always append display deques so their lengths stay in sync (GUI plots
    # az_hist and el_hist on the same time-axis; mismatched lengths crash
    # matplotlib).  The outlier gate uses az_gate_hist which is only seeded
    # with real (initialised, non-frozen) estimates so that 0 ° init values
    # never bias the circular-median reference.
    trk.az_hist.append(trk.az_ema)
    trk.az_raw_hist.append(az_doa)
    trk.el_hist.append(trk.el_ema)
    trk.el_raw_hist.append(el_doa)
    if not trk.az_init:
        trk.az_gate_hist.append(trk.az_ema)
    _az_kf = trk.az_kf.state_deg if trk.az_kf.state_deg is not None else trk.az_ema
    _el_kf = trk.el_kf.state
    trk.az_kf_hist.append(float(_az_kf))
    trk.el_kf_hist.append(float(_el_kf))
    trk.snr_hist.append(snr_db)
    trk.cfo_hist.append(cfo_hz)

    _indoor_mode = float(getattr(C, "DOPPLER_GATE_HZ", 0.0)) > 0.0
    _xz_en = bool(getattr(C, "DOPPLER_XZ_ENABLED", False)) and not _indoor_mode
    _xz_min_abs_hz = float(getattr(C, "DOPPLER_XZ_MIN_ABS_HZ", 600.0))
    _xz_min_jump_hz = float(getattr(C, "DOPPLER_XZ_MIN_JUMP_HZ", 1200.0))
    _xz_deadtime_s = float(getattr(C, "DOPPLER_XZ_DEADTIME_S", 2.0))

    new_sign = int(np.sign(cfo_hz)) if abs(cfo_hz) > _xz_min_abs_hz else 0
    _jump_ok = abs(cfo_hz - trk.cfo_hz) >= _xz_min_jump_hz
    _deadtime_ok = (
        trk.t_zero_crossing is None
        or (time.monotonic() - trk.t_zero_crossing) >= _xz_deadtime_s
    )
    if (_xz_en
            and trk.cfo_prev_sign != 0 and new_sign != 0
            and new_sign != trk.cfo_prev_sign
            and _jump_ok and _deadtime_ok):
        trk.el_at_zero_crossing = el_doa
        trk.t_zero_crossing     = time.monotonic()
        print(
            f"[DOPPLER-XZ] Sat {trk.sat_id}: Doppler zero-crossing detected. "
            f"fd: {trk.cfo_hz:+.0f} → {cfo_hz:+.0f} Hz  "
            f"DoA el at crossing = {el_doa:.1f}°  (≈ true peak elevation)"
        )
    if new_sign != 0:
        trk.cfo_prev_sign = new_sign

    trk.cfo_hz  = 0.85 * trk.cfo_hz + 0.15 * cfo_hz
    trk.spec2d  = spec2d
    trk.papr_db = papr_db
    trk.snr_db  = snr_db
    trk.az_deg  = trk.az_ema
    trk.el_deg  = trk.el_ema
    trk.no_doa  = False
    trk.burst_count += 1
    trk.last_seen    = time.monotonic()
    trk.trail.append((trk.az_deg, trk.el_deg))


def _find_or_create_tracker(
    satellites: dict, cfo_hz: float,
    sat_colors: list, min_sep_hz: float, max_sats: int,
    n_el: int, n_az: int, hist_len: int, multi_n: int, el_mid: float,
) -> "SimpleNamespace | None":
    """Find the active tracker closest to cfo_hz, or create a new one."""
    best_id, best_dist = None, float("inf")
    for sid, trk in satellites.items():
        d = abs(trk.cfo_hz - cfo_hz)
        if d < best_dist:
            best_dist, best_id = d, sid
    # Association window = 0.5 × min_sep_hz (≈ 4 kHz for SAT_MIN_SEP=8 kHz).
    # This must stay close to CFO_TRACK_MAX_JUMP_HZ (2.5 kHz) to avoid a
    # "dead zone" where noise peaks are assigned to a tracker but then
    # immediately fd_rej'd by the jump gate — causing inflated fd_rej counts
    # (e.g. 51/69 detected) and wasting CPU.  The old value (2×min_sep=16 kHz)
    # created a 13.5 kHz dead zone that captured most random scan peaks.
    if best_id is not None and best_dist < min_sep_hz * 0.5:
        return satellites[best_id]
    if len(satellites) >= max_sats:
        return None
    new_id = (max(satellites.keys()) + 1) if satellites else 0
    color  = sat_colors[new_id % len(sat_colors)]
    _cal_az = getattr(C, "CAL_REFERENCE_AZ_DEG", None)
    _cal_el = getattr(C, "CAL_REFERENCE_EL_DEG", None)
    _has_cal = any(o != 0.0 for o in getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * 5))
    trk    = _make_sat_tracker(
        new_id, color, cfo_hz, n_el, n_az, hist_len, multi_n, el_mid,
        cal_az=float(_cal_az) if _has_cal and _cal_az is not None else None,
        cal_el=float(_cal_el) if _has_cal and _cal_el is not None else None,
    )
    satellites[new_id] = trk
    return trk


def _prune_trackers(satellites: dict, timeout_s: float) -> None:
    """Remove trackers that haven't been seen for timeout_s seconds."""
    now  = time.monotonic()
    dead = [sid for sid, trk in satellites.items() if now - trk.last_seen > timeout_s]
    for sid in dead:
        del satellites[sid]


def _check_heimdall(host: str, port: int) -> bool:
    try:
        s = _socket.create_connection((host, port), timeout=2.0)
        s.close()
        return True
    except OSError:
        return False

_circ_median = circ_median_deg


# =============================================================================
# Global shared state (single lock for the acquisition/UI threads)
# =============================================================================

def _make_state(n_az: int, n_el: int) -> SimpleNamespace:
    return SimpleNamespace(
        # Global azimuth/spectrum view (dominant satellite projection or max-composite)
        az_spec     = np.full(n_az, -40.0),
        spec2d      = np.full((n_el, n_az), -40.0),
        # Kept for backward compatibility with calibration pipeline
        eig_db      = np.zeros(5),
        phase_diffs = np.zeros(4),
        phase_hist  = [collections.deque(maxlen=C.HISTORY_LEN) for _ in range(4)],
        # Circular EMA state for smooth phase display (unit-phasor domain).
        # EMA is applied before appending to phase_hist so the plot is stable.
        _phase_phasors = np.ones(4, dtype=complex),
        tone_hz_ema    = None,   # locked preamble tone [Hz] for static IRA TX
        energy_hist = collections.deque(maxlen=C.HISTORY_LEN),
        snr_hist    = collections.deque(maxlen=C.HISTORY_LEN),
        burst_n     = 0,
        frame_n     = 0,
        no_signal   = True,
        lock        = threading.Lock(),
        running     = True,
        # Latest per-burst metrics (updated by acquisition thread)
        crb_az_deg  = float("inf"),  # CRB azimuth standard deviation [°]
        mdl_k       = 1,             # MDL-estimated number of sources
        # ── Satellite tracker registry (dict {sat_id: namespace}) ──────────────
        satellites  = {},     # updated by the acquisition thread, read by the UI thread
        sat_colors  = list(_SAT_COLORS),
        # ── Calibration (uses sat_id=0 only) ───────────────────────────────────
        R_cal       = None,
        n_cal_bursts= 0,
        # ── Recording ────────────────────────────────────────────────────────────
        rec_enabled  = True,
        rec_iq_enabled = False,
        rec_t        = [], rec_az    = [], rec_el    = [],
        rec_papr     = [], rec_snr   = [], rec_eig   = [],
        rec_phase    = [], rec_phase_residual = [], rec_R     = [], rec_has_sig= [],
        rec_sat_cfo  = [], rec_X     = [],
        # Cramér-Rao Bound (Salama 2025 §8.2.1) and MDL source count per burst
        rec_crb      = [], rec_mdl_k = [],
        # Tracker identity + raw MUSIC spectrum (unweighted) per estimate
        rec_sat_id   = [],  # int tracker ID per estimate → use for per-sat grouping
        rec_spec2d   = [],  # (n_el, n_az) float32 MUSIC power per estimate
        # Live diagnostics panels
        cfo_diag_hist = collections.deque(maxlen=300),
        burst_env = np.array([], dtype=float),
        burst_pre_idx = 0,
        burst_data_idx = int(_PRE_SAMPLES),
    )


# =============================================================================
# Preamble-onset wrappers
# =============================================================================

def _find_preamble_onset(iq, b_start, n_total,
                          fs=_FS, win=_TONE_SCAN_WIN, known_hz=None,
                          freq_lock_bw: float = 2_000.0):
    return _fpo_core(iq, b_start, n_total, fs=fs, win=win, known_hz=known_hz,
                     burst_samples=_BURST_SAMPLES,
                     preamble_tone_hz=float(_PREAMBLE_TONE_HZ),
                     freq_lock_bw=freq_lock_bw)


def _find_tone_onset(iq, b_start, n_total,
                     tone_hz=float(_PREAMBLE_TONE_HZ),
                     fs=_FS, win=_TONE_SCAN_WIN):
    return _fto_core(iq, b_start, n_total, tone_hz=tone_hz, fs=fs, win=win,
                     burst_samples=_BURST_SAMPLES, preamble_samples=_PRE_SAMPLES)


# =============================================================================
# Demo: generate synthetic frame with N_sats satellites at different az/el/Doppler
# =============================================================================

def _demo_frame(rng: np.random.Generator, cfg: UcaConfig,
                demo_params: list[dict]) -> np.ndarray:
    """Generate a synthetic multi-satellite frame (no hardware required)."""
    N = _SF_SAMPLES
    X = np.zeros((cfg.n_ant, N), dtype=np.complex128)
    pos = cfg.positions   # (n_ant, 2)  [Est, Nord]

    for sat in demo_params:
        az_r   = np.deg2rad(sat["az"])
        el_r   = np.deg2rad(sat["el"])
        fd     = sat["doppler"]          # Doppler Hz
        snr_lin = 10 ** (sat["snr_db"] / 10.0)

        # Steering vector (planar, 2D)
        tau = 2 * np.pi * (pos[:, 0] * np.cos(el_r) * np.sin(az_r)
                           + pos[:, 1] * np.cos(el_r) * np.cos(az_r))

        t = np.arange(N, dtype=np.float64)
        tone = np.exp(2j * np.pi * (_PREAMBLE_TONE_HZ + fd) / _FS * t)

        b0     = 512    # burst offset within the frame
        pre_len = _PRE_SAMPLES
        b_end  = min(b0 + pre_len, N)

        for k in range(cfg.n_ant):
            X[k, b0:b_end] += (
                np.sqrt(snr_lin)
                * np.exp(1j * tau[k])
                * tone[b0:b_end]
            )

    # AWGN
    X += (rng.standard_normal((cfg.n_ant, N)) + 1j*rng.standard_normal((cfg.n_ant, N))) / np.sqrt(2)
    return X


# =============================================================================
# Acquisition + DoA thread
# =============================================================================

def _acq_loop(
    src, cfg: UcaConfig, algo: str,
    acc: CovarianceAccumulatorUca, S: SimpleNamespace,
    demo: bool = False,
    demo_params: list[dict] | None = None,
) -> None:
    rng = np.random.default_rng(42)
    _fs = _FS

    _phase_offs = list(getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * cfg.n_ant))
    _phase_offs = (_phase_offs + [0.0] * cfg.n_ant)[:cfg.n_ant]
    _has_cal    = any(o != 0.0 for o in _phase_offs)
    if demo:
        _has_cal = False  # synthetic signals have no hardware phase errors
    _az_alpha   = float(getattr(C, "AZ_SMOOTH_ALPHA",   0.50))
    _el_alpha   = float(getattr(C, "EL_SMOOTH_ALPHA",   0.50))
    el_mid      = (cfg.el_min_deg + cfg.el_max_deg) / 2.0

    _papr_min        = float(getattr(C, "PAPR_INST_MIN_DB", _PAPR_INST_MIN_DB))
    _snr_inst_min    = float(getattr(C, "SNR_INST_MIN_DB",   5.0))
    _eig_min         = float(getattr(C, "EIG_SPREAD_MIN_DB", 0.5))
    _multi_n         = max(1, int(getattr(C, "MULTI_BURST_N",       3)))
    _scan_bw         = float(getattr(C, "DOPPLER_SCAN_BW_HZ",   45_000))
    _min_sep         = float(getattr(C, "SAT_MIN_SEP_HZ",        5_000))
    _sat_timeout     = float(getattr(C, "SAT_TIMEOUT_S",          8.0))
    _max_sats        = int(getattr(C, "MAX_SATELLITES",            3))
    # Multi-channel IRA scan offsets [Hz] relative to FREQ_HZ.
    # Default: IRA-5 only (legacy, single channel at 0 offset).
    # Set in config to scan all 7 IRA channels across the ±512 kHz capture.
    _raw_offsets = getattr(C, "IRA_SCAN_OFFSETS_HZ", [0])
    _ira_offsets: list[float] = [float(o) for o in _raw_offsets]
    if not _ira_offsets:
        _ira_offsets = [0.0]
    # Doppler gate: reject FFT peaks whose |cfo_hz| > _doppler_gate_hz.
    # 0 = disabled (accept all Doppler, outdoor satellite mode).
    # 5000 = indoor TX: pass TX at fd≈0 Hz, block satellites at ±24 kHz.
    _doppler_gate_hz = float(getattr(C, "DOPPLER_GATE_HZ", 0.0))
    if demo:
        _doppler_gate_hz = 0.0
    # Max allowed CFO jump for an already-locked tracker.
    # Helps reject spurious FFT peaks inside the gate window.
    # 0 = disabled.
    _cfo_track_jump_hz = float(
        getattr(C, "CFO_TRACK_MAX_JUMP_HZ", 2000.0 if _doppler_gate_hz > 0 else 0.0)
    )
    # Narrow BPF centred on the detected (Doppler-corrected) tone.
    # tone_hz from _scan_doppler_peaks already tracks the per-satellite CFO
    # to ~62 Hz resolution (nfft=8192 at 1.024 MSPS), so a 8 kHz window
    # comfortably captures the tone while rejecting out-of-band DQPSK energy.
    # SNR gain vs. full band: 10·log10(1 024 000 / 8 000) ≈ +21.1 dB.
    _bpf_bw          = float(getattr(C, "PREAMBLE_BPF_BW_HZ",   8_000))
    # Guard samples to skip the FFT rectangular-window ringing at the
    # start of the IFFT output (sinc impulse-response width ≈ Fs / BW).
    _bpf_guard       = max(64, int(np.ceil(_FS / _bpf_bw)))
    _hist_len        = int(getattr(C, "HISTORY_LEN",              100))

    _buf: list[np.ndarray] = []
    _buf_len  = 0
    _min_buf  = _SF_SAMPLES + _WINDOW_SAMPLES + 2048

    _no_burst_streak = 0
    _diag_t0   = time.monotonic()
    _cnt_det = _cnt_eig = _cnt_snr = _cnt_papr = _cnt_acc = 0
    _cnt_no_tone = _cnt_fd_rej = 0
    _cnt_az_rej = _cnt_ph_rej = _cnt_relock = 0
    _papr_sum = 0.0;  _papr_n = 0
    # Pre-load gate config (constant for the lifetime of _acq_loop).
    # AZ outlier gate is only valid on calibrated hardware: on uncalibrated
    # arrays MUSIC azimuths can span ±180°, so the circular median of the
    # first few estimates is an unreliable reference and the gate rejects
    # almost everything.  Disable automatically when _has_cal is False.
    _az_outlier_en  = bool(getattr(C,  "AZ_OUTLIER_ENABLED",         True)) and _has_cal
    _az_outlier_min = int(getattr(C,   "AZ_OUTLIER_MIN_HISTORY",        5))
    _az_outlier_dev = float(getattr(C, "AZ_OUTLIER_MAX_DEV_DEG",     45.0))
    _az_relock_streak = int(getattr(C, "AZ_OUTLIER_RELOCK_STREAK",      8))
    _ph_coh_en      = bool(getattr(C,  "PHASE_COHERENCE_ENABLED",     True)) and _has_cal
    _ph_coh_dev     = float(getattr(C, "PHASE_COHERENCE_MAX_JUMP_DEG", 60.0))
    _ph_relock_streak = int(getattr(C, "PHASE_COHERENCE_RELOCK_STREAK", 8))
    _relock_bypass_bursts = int(getattr(C, "GATE_RELOCK_BYPASS_BURSTS", 12))
    _indoor_lab     = str(getattr(C, "INDOOR_TX_MODE", "pass")).lower() in ("ira", "pass", "track")
    _ira_static     = str(getattr(C, "INDOOR_TX_MODE", "pass")).lower() == "ira"
    _indoor_tx      = _doppler_gate_hz > 0
    _ira_tone_ema_a = float(getattr(C, "INDOOR_IRA_TONE_EMA_ALPHA", 0.92))
    _ira_tone_lock_bw = float(getattr(C, "INDOOR_IRA_TONE_LOCK_BW_HZ", 150.0))
    _indoor_single_src = _indoor_lab and _max_sats <= 1
    _az_pick_hint_min = int(getattr(C, "AZ_PICK_HINT_MIN_HISTORY", 4))
    _indoor_phase_w   = float(getattr(C, "INDOOR_PHASE_SCORE_WEIGHT", 0.35))
    _indoor_hint_w    = float(getattr(C, "INDOOR_AZ_HINT_SCORE_WEIGHT", 0.25))
    _indoor_el_hint_w = float(getattr(C, "INDOOR_EL_HINT_SCORE_WEIGHT", 0.20))
    _indoor_mirror_m  = float(getattr(C, "INDOOR_UCA_MIRROR_MARGIN_DEG", 5.0))
    _mirror_phase_m   = float(getattr(C, "INDOOR_MIRROR_PHASE_MIN_MARGIN_DEG", 10.0))
    _phase_snr_min    = float(getattr(C, "PHASE_SCORE_MIN_SNR_DB", 2.0))
    _ph_display_alpha = float(getattr(C, "PHASE_DISPLAY_EMA_ALPHA", 0.88))
    _energy_thr       = float(getattr(C, "ENERGY_DETECT_THRESHOLD", 3.0))
    _papr_single_min  = float(getattr(C, "PAPR_SINGLE_BURST_MIN_DB", 0.5))
    _single_burst_fb  = bool(getattr(C, "INDOOR_SINGLE_BURST_FALLBACK", True))
    if _indoor_single_src and bool(getattr(C, "INDOOR_SINGLE_SOURCE_RELAX_GATES", True)):
        # Indoor near-field + single source is strongly affected by multipath.
        # Hard AZ/phase continuity gates can reject most valid bursts and stall tracking.
        _az_outlier_en = False
        _ph_coh_en = False

    _el_pref_hi     = float(getattr(C, "INDOOR_EL_PREF_MAX_DEG", 28.0))
    _el_pref_lo     = float(getattr(C, "INDOOR_EL_PREF_MIN_DEG", 10.0))

    _tone_known: dict[int, float] = {}   # sat_id → last known tone_hz

    demo_params = demo_params or _DEMO_SATS[:2]

    while S.running:
        # ── Acquire frame ─────────────────────────────────────────────────────
        if demo:
            frame = _demo_frame(rng, cfg, demo_params)
            time.sleep(_SUPERFRAME_S)
        else:
            try:
                frame = src.get_frame(timeout=2.0)
            except Exception:
                continue
            if frame is None or frame.shape[1] < 512:
                continue

        with S.lock:
            S.frame_n += 1

        _buf.append(frame)
        _buf_len += frame.shape[1]
        if _buf_len < _min_buf:
            continue

        X_stream = np.hstack(_buf)
        _buf.clear(); _buf_len = 0
        n_total  = X_stream.shape[1]

        # ── Energy detection on ch0 ──────────────────────────────────────────
        bursts = detect_energy_bursts(X_stream[0], _fs, threshold_factor=_energy_thr)
        if not bursts:
            pwr_db = float(10 * np.log10(np.mean(np.abs(X_stream[0])**2) + 1e-20))
            with S.lock:
                S.energy_hist.append(pwr_db)
                S.no_signal = True
            _no_burst_streak += 1
            if _no_burst_streak == 10:
                print(
                    "[WARN] 10 consecutive frames without an IRA burst detected.\n"
                    "  Indoor:  check that LibreSDR TX is active\n"
                    "           (python3 tx/indoor_1626.py --mode ira --gain -50 --cyclic)\n"
                    "           or pass simulation: tx/indoor_1626.py --mode pass --gain -50 --elev 45 --dur 90 --cyclic\n"
                    "  Outdoor: wait for an Iridium pass\n"
                    "  Testing: add --demo to simulate the signal."
                )
            # Still emit periodic DIAG even with no bursts so the user can see
            # the runner is alive and receiving frames.
            now = time.monotonic()
            if now - _diag_t0 >= 10.0:
                print(f"[DIAG] no_burst (streak={_no_burst_streak}) "
                      f"ch0_pwr={pwr_db:.1f}dB — waiting for Iridium burst")
                _diag_t0 = now
            continue
        _no_burst_streak = 0
        _cnt_det += len(bursts)

        # ── Periodic diagnostics ──────────────────────────────────────────────
        now = time.monotonic()
        if now - _diag_t0 >= 10.0:
            sats_str = ", ".join(
                f"S{sid}:az={trk.az_deg:.0f}°/el={trk.el_deg:.0f}°/fd={trk.cfo_hz:+.0f}Hz"
                for sid, trk in S.satellites.items()
            )
            _papr_mean_str = f"{_papr_sum/_papr_n:.1f}" if _papr_n > 0 else "--"
            _diag_line = (
                f"[DIAG] det={_cnt_det} no_tone={_cnt_no_tone} fd_rej={_cnt_fd_rej} "
                f"az_rej={_cnt_az_rej} ph_rej={_cnt_ph_rej} relock={_cnt_relock} "
                f"eig_rej={_cnt_eig} snr_rej={_cnt_snr} "
                f"papr_rej={_cnt_papr} acc={_cnt_acc} papr_mean={_papr_mean_str}dB | "
                f"sats={len(S.satellites)}: {sats_str}"
            )
            print(_diag_line)
            if _ira_static and S.satellites:
                _t0 = next(iter(S.satellites.values()))
                # Static IRA TX can still show ~±1 kHz CFO due to LO ppm offsets.
                # Warn only when offset is clearly outside the expected range.
                if abs(float(_t0.cfo_hz)) > 1500.0:
                    print(
                        f"[WARN] IRA static TX but CFO={_t0.cfo_hz:+.0f} Hz "
                        f"(expected near 0, typically within ±1.5 kHz). "
                        f"Check TX: indoor_1626.py --mode ira "
                        f"(NOT pass).  tone_ema="
                        f"{S.tone_hz_ema if S.tone_hz_ema is not None else 'unset'}"
                    )
            if (_papr_n > 0) and ((_papr_sum / _papr_n) < _papr_min):
                _mode = str(getattr(C, "INDOOR_TX_MODE", "pass"))
                print(
                    f"[WARN] PAPR gate ({_papr_min:.1f} dB) rejects all estimates "
                    f"(mean={_papr_sum/_papr_n:.1f} dB).  "
                    f"INDOOR_TX_MODE={_mode!r}: pass needs DOPPLER_GATE_HZ=0, "
                    f"ira needs DOPPLER_GATE_HZ=3000.  "
                    f"Try TX gain −50 dB or --papr-min 0.5."
                )
            _diag_t0 = now
            _cnt_det = _cnt_eig = _cnt_snr = _cnt_papr = _cnt_acc = 0
            _cnt_no_tone = _cnt_fd_rej = 0
            _cnt_az_rej = _cnt_ph_rej = _cnt_relock = 0
            _papr_sum = 0.0;  _papr_n = 0

        pwr_db = float(10 * np.log10(np.mean(np.abs(X_stream[0])**2) + 1e-20))

        for b_start in bursts:
            b_end = min(b_start + _WINDOW_SAMPLES, n_total)
            if b_end - b_start < _PRE_SAMPLES:
                continue

            X_win = X_stream[:, b_start:b_end]   # (n_ant, window)

            # ── Preamble onset refinement (joint time×frequency search) ──────
            # Default: anchor always to nominal preamble tone so that
            # find_preamble_onset uses ±lock_bw instead of a full-band search
            # (full-band returns the strongest carrier in the 1 MHz band which
            # is NOT the Iridium tone in outdoor L-band environments).
            _known_tone: float = float(_PREAMBLE_TONE_HZ)
            # Outdoor: lock_bw = full Doppler scan (±45 kHz) to capture
            #   satellites at ±17–40 kHz Doppler.
            # Indoor: narrow lock around known TX CFO (few kHz).
            _lock_bw = float(_doppler_gate_hz) if _doppler_gate_hz > 0 else float(_scan_bw)
            if _ira_static and S.tone_hz_ema is not None and S.burst_n > 5:
                _known_tone = float(S.tone_hz_ema)
                _lock_bw = _ira_tone_lock_bw
            elif _indoor_tx:
                with S.lock:
                    if S.satellites:
                        _sid = next(iter(S.satellites))
                        _cfo = S.satellites[_sid].cfo_hz
                        _kt = _tone_known.get(_sid)
                        if _kt is not None and S.satellites[_sid].burst_count > 5:
                            _known_tone = float(_kt)
                            _lock_bw = 1200.0  # narrow lock after history
                        else:
                            _known_tone = float(_PREAMBLE_TONE_HZ + _cfo)
            else:
                # Outdoor: refine anchor from established tracker when available
                with S.lock:
                    if S.satellites:
                        _best_sid = min(S.satellites,
                                        key=lambda sid: (S.satellites[sid].no_doa,
                                                         -S.satellites[sid].burst_count))
                        _trk0 = S.satellites[_best_sid]
                        if _trk0.burst_count > 3 and not _trk0.no_doa:
                            _kt = _tone_known.get(_best_sid)
                            if _kt is not None:
                                _known_tone = float(_kt)
                                _lock_bw = float(_min_sep) * 2.0  # ±10 kHz post-lock
                # No established tracker: widen the anchor search to cover all
                # IRA channels.  _find_preamble_onset returns the strongest CW
                # tone in [_known_tone ± _lock_bw]; setting _lock_bw to the
                # half-span of IRA channel offsets (max |offset| + scan_bw)
                # ensures that IRA-1 at −166 kHz is reachable in the first burst.
                if _known_tone == float(_PREAMBLE_TONE_HZ) and len(_ira_offsets) > 1:
                    _max_offset = max(abs(o) for o in _ira_offsets)
                    _lock_bw = _max_offset + _scan_bw

            onset_ref, tone_ref = _find_preamble_onset(
                X_win[0], 0, X_win.shape[1],
                known_hz=_known_tone, win=4096, freq_lock_bw=_lock_bw,
            )
            # Validate tone_ref: accept tones that fall within DOPPLER_SCAN_BW
            # of ANY configured IRA channel offset, not just IRA-5 (offset=0).
            # Indoor: gate to DOPPLER_GATE_HZ around the reference tone.
            _tone_valid_bw = _doppler_gate_hz if _doppler_gate_hz > 0 else _scan_bw
            if _doppler_gate_hz > 0:
                tone_ref_valid = abs(tone_ref - _PREAMBLE_TONE_HZ) <= _tone_valid_bw
            else:
                tone_ref_valid = any(
                    abs(tone_ref - (_PREAMBLE_TONE_HZ + off)) <= _tone_valid_bw
                    for off in _ira_offsets
                )

            if tone_ref_valid:
                pre_start_stream = b_start + max(0, int(onset_ref - _ENERGY_WIN))
                if onset_ref > _ENERGY_WIN and onset_ref < X_win.shape[1] // 2:
                    b_start_adj = b_start + onset_ref - _ENERGY_WIN
                    b_end_adj = min(b_start_adj + _WINDOW_SAMPLES, n_total)
                    X_win = X_stream[:, b_start_adj:b_end_adj]
                    pre_start_stream = b_start_adj + _ENERGY_WIN
                    if X_win.shape[1] < _PRE_SAMPLES:
                        X_win = X_stream[:, b_start:b_end]
                        pre_start_stream = b_start
                if _ira_static:
                    if S.tone_hz_ema is None:
                        S.tone_hz_ema = float(tone_ref)
                    else:
                        S.tone_hz_ema = (
                            _ira_tone_ema_a * float(S.tone_hz_ema)
                            + (1.0 - _ira_tone_ema_a) * float(tone_ref)
                        )
                    tone_ref = float(S.tone_hz_ema)
                peaks: list[tuple[float, float]] = [(float(tone_ref), 10.0)]
            else:
                pre_start_stream = b_start
                # Refinement failed — fall back to multi-channel Doppler FFT scan.
                # Scan around every configured IRA channel offset so that LARK
                # can lock onto whichever satellite/channel is currently strongest,
                # not just IRA-5 (offset=0).  With 1.024 MSPS bandwidth, channels
                # IRA-1…7 all lie within ±170 kHz ⊂ ±512 kHz capture window.
                X0_pre = X_win[0, :_PRE_SAMPLES] if X_win.shape[1] >= _PRE_SAMPLES else X_win[0]
                X0_wide = X_win[0]
                _PSCAN_SNR = 6.0
                _n_peaks_scan = min(_max_sats + 3, 6)
                _scan_bw_used = _doppler_gate_hz if _doppler_gate_hz > 0 else _scan_bw
                _offsets_to_scan = [0.0] if _doppler_gate_hz > 0 else _ira_offsets
                all_peaks: list[tuple[float, float]] = []
                for _ch_off in _offsets_to_scan:
                    _nom = float(_PREAMBLE_TONE_HZ + _ch_off)
                    _ch_peaks = scan_preamble_tones(
                        X0_pre, _fs, nom_tone_hz=_nom,
                        scan_bw_hz=_scan_bw_used, n_peaks=_n_peaks_scan,
                        min_sep_hz=_min_sep, min_snr_db=_PSCAN_SNR,
                    )
                    all_peaks.extend(p for p in _ch_peaks if p[1] >= _PSCAN_SNR)
                    # Also try with the full window if the preamble-only window
                    # produced no result for this channel.
                    if not all_peaks:
                        _ch_wide = scan_preamble_tones(
                            X0_wide, _fs, nom_tone_hz=_nom,
                            scan_bw_hz=_scan_bw_used, n_peaks=_n_peaks_scan,
                            min_sep_hz=_min_sep, min_snr_db=_PSCAN_SNR,
                        )
                        all_peaks.extend(p for p in _ch_wide if p[1] >= _PSCAN_SNR)
                # Deduplicate: drop peaks closer than _min_sep to a higher-SNR peak
                all_peaks.sort(key=lambda p: -p[1])
                dedup: list[tuple[float, float]] = []
                for pk in all_peaks:
                    if not any(abs(pk[0] - q[0]) < _min_sep for q in dedup):
                        dedup.append(pk)
                peaks = dedup[:_n_peaks_scan] if dedup else [(float(_PREAMBLE_TONE_HZ), 0.0)]
                if _ira_static and peaks:
                    _tr = float(peaks[0][0])
                    if S.tone_hz_ema is None:
                        S.tone_hz_ema = _tr
                    else:
                        S.tone_hz_ema = (
                            _ira_tone_ema_a * float(S.tone_hz_ema)
                            + (1.0 - _ira_tone_ema_a) * _tr
                        )
                    peaks = [(float(S.tone_hz_ema), float(peaks[0][1]))]

            if not peaks:
                _cnt_no_tone += 1

            # Prioritise peaks close to current tracker CFO to avoid hopping
            # between unrelated local maxima inside the same burst window.
            if peaks:
                with S.lock:
                    _trk_cfos = [float(tk.cfo_hz) for tk in S.satellites.values()]
                if _trk_cfos:
                    peaks = sorted(
                        peaks,
                        key=lambda p: (
                            min(abs((p[0] - _PREAMBLE_TONE_HZ) - c) for c in _trk_cfos),
                            -p[1],
                        ),
                    )

            # At most one accepted peak per tracker for this burst.
            _used_tracker_ids: set[int] = set()

            for tone_hz, peak_snr_db in peaks:
                cfo_hz = tone_hz - _PREAMBLE_TONE_HZ

                # ── Doppler gate ──────────────────────────────────────────────
                # Reject peaks outside the expected Doppler window.
                # Indoor TX: _doppler_gate_hz=5000 → cfo≈0 passes, satellites
                # at cfo≈−24kHz are discarded. Gate 0 = disabled (outdoor).
                if _doppler_gate_hz > 0 and abs(cfo_hz) > _doppler_gate_hz:
                    _cnt_fd_rej += 1
                    continue

                # ── BPF on the preamble region only ────────────────────────────
                # Apply the narrowband filter only over the first PRE_SAMPLES of
                # X_win. The energy detector fires within ±_ENERGY_WIN/2 = ±128
                # samples of the actual burst onset, so X_win[:,0:PRE_SAMPLES]
                # always contains the full preamble CW region.  Limiting the
                # input to PRE_SAMPLES avoids IFFT circular-convolution from the
                # DQPSK data region bleeding back into the preamble filter output.
                if X_win.shape[1] < _PRE_SAMPLES:
                    continue
                X_bpf = extract_pilot_tone(X_win[:, :_PRE_SAMPLES], _fs,
                                           tone_hz=tone_hz, bw_hz=_bpf_bw)
                X_bpf = amplitude_normalize_channels(X_bpf)

                if _has_cal:
                    X_cal = _apply_phase_correction(X_bpf, _phase_offs)
                else:
                    X_cal = X_bpf

                # ── Preamble-only sample covariance + matched-filter ───────────
                # Skip _bpf_guard samples to avoid sinc-ringing at the start of
                # the IFFT output (impulse-response width ≈ Fs/BW = 128 samples).
                # The remaining N_pre samples are pure CW → high λ1/σ² gap.
                _n_pre = _PRE_SAMPLES - _bpf_guard
                if _n_pre < 128:
                    continue

                # Matched-filter covariance via modular API.
                # NOTE: t_vec must start at _bpf_guard (not 0) so the reference
                # phase matches the actual sample positions in X_cal.
                # Bug fixed 2026-05-20: using np.arange(_n_pre) caused a phase
                # offset of _bpf_guard × 2π × f_tone / fs ≈ 8 rad → y_mf ≈ 0
                # → fallback to sample covariance → low eigenvalue ratio → 95%
                # of real preamble bursts rejected by SNR gate.
                X_pre = X_cal[:, _bpf_guard : _bpf_guard + _n_pre]
                R_inst, y_mf, snr = compute_mf_covariance(
                    X_cal, tone_hz, _fs, _n_pre, _bpf_guard
                )
                try:
                    eig = eigenvalue_spread_uca_db(R_inst)
                except Exception:
                    continue
                # snr already returned by _compute_mf_covariance_api

                if eig[0] < _eig_min:
                    _cnt_eig += 1
                    continue

                # ── Instant SNR gate ──────────────────────────────────────────
                # Use the already-computed per-element SNR as the instant quality
                # gate instead of MUSIC PAPR.  The MUSIC-PAPR gate requires the
                # signal to lie on the UCA steering manifold, which fails on
                # uncalibrated hardware: inter-channel phase offsets push y_mf
                # off the manifold, collapsing MUSIC PAPR to < 8 dB even when a
                # strong coherent signal is present.
                # snr_uca_db() is calibration-agnostic: it only looks at the
                # eigenvalue ratio of R_inst.  For rank-1 R_mf from a real
                # preamble, SNR >> 20 dB; for noise falling through the energy
                # detector, SNR < 0 dB.
                if snr < _snr_inst_min:
                    _cnt_snr += 1
                    continue

                # ── Associate to tracker ───────────────────────────────────────
                with S.lock:
                    trk = _find_or_create_tracker(
                        S.satellites, cfo_hz,
                        S.sat_colors, _min_sep, _max_sats,
                        cfg.n_el, cfg.n_az, _hist_len, _multi_n, el_mid,
                    )
                if trk is None:
                    continue

                if trk.sat_id in _used_tracker_ids:
                    continue

                if (_cfo_track_jump_hz > 0.0
                        and trk.burst_count >= 3
                        and abs(cfo_hz - trk.cfo_hz) > _cfo_track_jump_hz):
                    _cnt_fd_rej += 1
                    continue

                _used_tracker_ids.add(trk.sat_id)

                trk.R_batch.append(R_inst)
                trk.cfo_batch.append(float(cfo_hz))
                trk.X_batch.append(X_pre)
                trk.last_seen = time.monotonic()
                if len(trk.R_batch) < _multi_n:
                    continue

                # ── Covariance for DoA ──────────────────────────────────────────
                # When MULTI_BURST_N == 1 there is nothing to stack; using the
                # matched-filter covariance R_inst (rank-1) avoids destroying
                # the clean CW structure with sample-covariance noise / multipath.
                # For N > 1 we Doppler-align the stack and build a sample cov.
                if _multi_n == 1:
                    R_avg = R_inst
                    X_big = X_pre
                    phase_diffs = np.degrees(np.angle(R_inst[1:, 0]))
                else:
                    _t_align = np.arange(_bpf_guard, _bpf_guard + _n_pre,
                                         dtype=np.float64)
                    X_parts: list[np.ndarray] = []
                    for x_hist, cfo_hist in zip(trk.X_batch, trk.cfo_batch):
                        d_cfo = float(cfo_hz) - float(cfo_hist)
                        if abs(d_cfo) > 30.0:
                            # rot = exp(+j·2π·Δf·t)  rotates the historical burst
                            # from its old CFO to the current CFO so they stack
                            # coherently.
                            rot = np.exp(2j * np.pi * d_cfo / _fs * _t_align)
                            X_parts.append(x_hist * rot)
                        else:
                            X_parts.append(x_hist)
                    X_big = np.hstack(X_parts)
                    R_avg = (X_big @ X_big.conj().T) / X_big.shape[1]
                    phase_diffs = np.degrees(np.angle(R_avg[1:, 0]))
                try:
                    _az_hint, _el_hint = _pick_doa_hints(trk, _has_cal, _az_pick_hint_min)
                    _ph_w = _indoor_phase_w if snr >= _phase_snr_min else 0.0
                    if algo == "phase-fit":
                        az_doa, el_doa, papr_doa, spec2d = _estimate_doa_burst(
                            X_cal, R_avg, cfg, algo,
                            indoor=_indoor_lab,
                            phase_diffs=phase_diffs,
                            az_hint=_az_hint,
                            el_hint=_el_hint,
                            phase_w=_ph_w,
                            az_hint_w=_indoor_hint_w,
                            el_hint_w=_indoor_el_hint_w,
                            mirror_m=_indoor_mirror_m,
                            mirror_phase_m=_mirror_phase_m,
                            el_pref_hi=_el_pref_hi,
                            el_pref_lo=_el_pref_lo,
                            n_snapshots=X_big.shape[1],
                        )
                    else:
                        spec2d = _run_doa_algo(X_cal, R_avg, cfg, algo,
                                               n_snapshots=X_big.shape[1])
                        az_doa, el_doa, papr_doa = pick_doa_peak_uca_2d(
                            spec2d, cfg, indoor=_indoor_lab,
                            el_pref_hi=_el_pref_hi, el_pref_lo=_el_pref_lo,
                            phase_diffs=phase_diffs,
                            az_hint_deg=_az_hint,
                            el_hint_deg=_el_hint,
                            phase_score_weight=_ph_w,
                            az_hint_score_weight=_indoor_hint_w,
                            el_hint_score_weight=_indoor_el_hint_w,
                            mirror_margin_deg=_indoor_mirror_m,
                            mirror_phase_min_margin_deg=_mirror_phase_m,
                        )
                    az_doa, el_doa = _apply_doa_frame_offset(az_doa, el_doa, cfg, _has_cal)
                    # Find secondary peak for GUI multi-peak display
                    _all_peaks = find_peaks_uca_2d(spec2d, cfg, n_peaks=3, min_sep_deg=8.0)
                    _other = None
                    for _az, _el, _p in _all_peaks:
                        d = ((abs(_az - az_doa) + 180) % 360 - 180)
                        if abs(d) > 10 or abs(_el - el_doa) > 8:
                            _other = (_az, _el, _p); break
                    if _other is not None:
                        trk.az_other, trk.el_other, trk.papr_other = _other
                        trk.has_other = True
                except Exception as _doa_exc:
                    continue

                # Multi-burst PAPR gate: reject estimates where the MUSIC spectrum
                # is essentially flat (no spatial null from the noise subspace).
                # At PAPR < _papr_min the peak position is dominated by noise and
                # multipath; accepting it corrupts the tracker az/el EMA.
                # The gate is intentionally enabled even on calibrated hardware:
                # PAPR < 2 dB means the steering manifold match failed (multipath
                # too strong indoor, or wrong calibration) and the DoA is useless.
                _papr_sum += papr_doa;  _papr_n += 1
                if papr_doa < _papr_min:
                    # Multi-burst average can flatten MUSIC when Doppler drifts
                    # within the window.  Fall back to the current burst alone.
                    if _single_burst_fb and _indoor_lab:
                        try:
                            _az_hint_sb, _el_hint_sb = _pick_doa_hints(
                                trk, _has_cal, _az_pick_hint_min,
                            )
                            _ph_w_sb = _indoor_phase_w if snr >= _phase_snr_min else 0.0
                            if algo == "phase-fit":
                                az_sb, el_sb, papr_sb, spec2d_sb = _estimate_doa_burst(
                                    X_cal, R_inst, cfg, algo,
                                    indoor=_indoor_lab,
                                    phase_diffs=np.degrees(np.angle(R_inst[1:, 0])),
                                    az_hint=_az_hint_sb, el_hint=_el_hint_sb,
                                    phase_w=_ph_w_sb,
                                    az_hint_w=_indoor_hint_w,
                                    el_hint_w=_indoor_el_hint_w,
                                    mirror_m=_indoor_mirror_m,
                                    mirror_phase_m=_mirror_phase_m,
                                    el_pref_hi=_el_pref_hi, el_pref_lo=_el_pref_lo,
                                    n_snapshots=_n_pre,
                                )
                            else:
                                spec2d_sb = _run_doa_algo(
                                    X_cal, R_inst, cfg, algo, n_snapshots=_n_pre,
                                )
                                az_sb, el_sb, papr_sb = pick_doa_peak_uca_2d(
                                    spec2d_sb, cfg, indoor=_indoor_lab,
                                    el_pref_hi=_el_pref_hi, el_pref_lo=_el_pref_lo,
                                    phase_diffs=np.degrees(np.angle(R_inst[1:, 0])),
                                    az_hint_deg=_az_hint_sb,
                                    el_hint_deg=_el_hint_sb,
                                    phase_score_weight=_ph_w_sb,
                                    az_hint_score_weight=_indoor_hint_w,
                                    el_hint_score_weight=_indoor_el_hint_w,
                                    mirror_margin_deg=_indoor_mirror_m,
                                    mirror_phase_min_margin_deg=_mirror_phase_m,
                                )
                            az_sb, el_sb = _apply_doa_frame_offset(az_sb, el_sb, cfg, _has_cal)
                            if papr_sb >= _papr_single_min:
                                az_doa, el_doa, papr_doa = az_sb, el_sb, papr_sb
                                spec2d = spec2d_sb
                                R_avg = R_inst
                                phase_diffs = np.degrees(np.angle(R_inst[1:, 0]))
                            else:
                                _cnt_papr += 1
                                continue
                        except Exception:
                            _cnt_papr += 1
                            continue
                    else:
                        _cnt_papr += 1
                        continue

                # ── AZ outlier gate ───────────────────────────────────────────
                # Reject bursts that deviate from the tracker's recent circular
                # median by more than AZ_OUTLIER_MAX_DEV_DEG.  Inactive until
                # at least AZ_OUTLIER_MIN_HISTORY DoA estimates have been
                # accumulated — before that every estimate seeds the history.
                # This suppresses multi-modal MUSIC scatter on uncalibrated HW.
                _gate_bypass_active = trk.gate_bypass_left > 0
                if (_az_outlier_en and not trk.no_doa
                        and len(trk.az_gate_hist) >= _az_outlier_min
                        and not _gate_bypass_active):
                    med = _circ_median(list(trk.az_gate_hist))
                    d   = abs(az_doa - med) % 360.0
                    dev_direct = min(d, 360.0 - d)
                    # UCA has inherent 180° ambiguity: az and az+180° are the
                    # same physical source.  Test the mirror bearing too so that
                    # the gate does not reject valid estimates solely because the
                    # MUSIC peak flipped to the opposite ambiguity side.
                    d_mirror   = abs((az_doa + 180.0) % 360.0 - med) % 360.0
                    dev_mirror = min(d_mirror, 360.0 - d_mirror)
                    dev = min(dev_direct, dev_mirror)
                    if dev > _az_outlier_dev:
                        trk.az_reject_streak += 1
                        _cnt_az_rej += 1
                        if trk.az_reject_streak >= _az_relock_streak:
                            # Soft re-lock for abrupt antenna/TX rotation.
                            # Keep visual history intact, reset only gating state.
                            trk.az_init = True
                            trk.R_batch.clear(); trk.X_batch.clear(); trk.cfo_batch.clear(); trk.Y_batch.clear()
                            trk.last_phase_diffs = None
                            trk.gate_bypass_left = max(trk.gate_bypass_left, _relock_bypass_bursts)
                            trk.az_reject_streak = 0
                            trk.ph_reject_streak = 0
                            _cnt_relock += 1
                        continue   # outlier azimuth — discard
                    trk.az_reject_streak = 0

                # ── Phase coherence gate ──────────────────────────────────────
                # Reject bursts where inter-channel phase diffs jump more than
                # PHASE_COHERENCE_MAX_JUMP_DEG from the last accepted burst.
                # A real signal has slowly varying phase (Doppler ramp between
                # 90 ms superframes is typically < 10°); a phantom peak from a
                # spurious multipath or DoA ambiguity produces a large jump.
                if (_ph_coh_en and not trk.no_doa
                        and trk.last_phase_diffs is not None
                        and not _gate_bypass_active):
                    ph_delta = np.abs(((phase_diffs - trk.last_phase_diffs + 180.0)
                                       % 360.0) - 180.0)
                    if float(np.max(ph_delta)) > _ph_coh_dev:
                        trk.ph_reject_streak += 1
                        _cnt_ph_rej += 1
                        if trk.ph_reject_streak >= _ph_relock_streak:
                            # Reset phase baseline and burst accumulator on discontinuity.
                            trk.R_batch.clear(); trk.X_batch.clear(); trk.cfo_batch.clear(); trk.Y_batch.clear()
                            trk.last_phase_diffs = None
                            trk.az_init = True
                            trk.gate_bypass_left = max(trk.gate_bypass_left, _relock_bypass_bursts)
                            trk.az_reject_streak = 0
                            trk.ph_reject_streak = 0
                            _cnt_relock += 1
                        continue   # phase discontinuity — discard
                    trk.ph_reject_streak = 0

                _cnt_acc += 1
                trk.az_reject_streak = 0
                trk.ph_reject_streak = 0
                if trk.gate_bypass_left > 0:
                    trk.gate_bypass_left -= 1
                acc.update(R_avg)   # used by the calibration routine

                # CRB for azimuth (Salama 2025 §8.2.1; Stoica & Nehorai 1990).
                # n_snaps is the preamble-only snapshot count used for R_inst;
                # this is the effective N that determines the Fisher information.
                # crb_azimuth_deg expects per-element SNR; snr_uca_db returns
                # the array-gain SNR = M × per-element, so subtract 10·log10(M).
                n_snaps  = X_pre.shape[1]
                snr_per_element_db = snr - 10.0 * np.log10(float(cfg.n_ant))
                crb_deg_ = crb_azimuth_deg(snr_per_element_db, n_snaps, cfg,
                                           el_deg=el_doa)

                # MDL source-count estimate on the averaged covariance.
                # For Iridium IRA we expect K=1; deviations indicate multipath
                # or a second satellite within the Doppler search window.
                mdl_k = estimate_signal_count_mdl(R_avg, n_snaps,
                                                  max_signals=cfg.n_ant - 1)

                with S.lock:
                    _update_tracker(
                        trk, az_doa, el_doa, papr_doa, snr,
                        cfo_hz, spec2d, _az_alpha, _el_alpha,
                        cfg.el_min_deg, cfg.el_max_deg,
                        az_freeze_el=float(getattr(C, "AZ_FREEZE_EL_DEG", 75.0)),
                    )
                    # Remember tone frequency for next preamble refinement
                    _tone_known[trk.sat_id] = float(tone_hz)
                    # Store phase baseline for the coherence gate (only updated
                    # after a burst passes every quality check)
                    trk.last_phase_diffs = phase_diffs.copy()
                    S.no_signal = False
                    S.burst_n  += 1
                    S.energy_hist.append(pwr_db)
                    S.snr_hist.append(float(snr))
                    S.eig_db    = eig
                    S.crb_az_deg = float(crb_deg_)
                    S.mdl_k      = int(mdl_k)
                    S.cfo_diag_hist.append(float(cfo_hz))
                    dbg_len = max(1, min(_BURST_SAMPLES, n_total - int(pre_start_stream)))
                    dbg_env = np.abs(
                        X_stream[0, int(pre_start_stream): int(pre_start_stream) + dbg_len]
                    ).astype(float, copy=False)
                    S.burst_env = dbg_env
                    S.burst_pre_idx = 0
                    S.burst_data_idx = max(0, min(int(_PRE_SAMPLES), dbg_len - 1))
                    S.phase_diffs = phase_diffs
                    _ph_az, _ph_el = _phase_model_angles(trk, _has_cal)
                    _phase_residual = phase_residual_deg(
                        phase_diffs, _ph_az, _ph_el, cfg,
                    )
                    raw_phasors = np.exp(1j * np.radians(_phase_residual))
                    S._phase_phasors = (
                        _ph_display_alpha * S._phase_phasors
                        + (1.0 - _ph_display_alpha) * raw_phasors
                    )
                    ph_smooth = np.degrees(np.angle(S._phase_phasors))
                    for i in range(4):
                        S.phase_hist[i].append(float(ph_smooth[i]))
                    # Global spec2d: max over all active satellite trackers
                    if S.satellites:
                        S.spec2d = np.max([t.spec2d for t in S.satellites.values()], axis=0)
                        S.az_spec = S.spec2d.max(axis=0)

                    if S.rec_enabled:
                        S.rec_t.append(time.time())
                        S.rec_az.append(float(trk.az_deg))
                        S.rec_el.append(float(trk.el_deg))
                        S.rec_papr.append(float(papr_doa))
                        S.rec_snr.append(float(snr))
                        S.rec_eig.append(eig.copy())
                        S.rec_phase.append(phase_diffs.copy())
                        _rec_ph_az, _rec_ph_el = _phase_model_angles(trk, _has_cal)
                        S.rec_phase_residual.append(
                            phase_residual_deg(phase_diffs, _rec_ph_az, _rec_ph_el, cfg).copy()
                        )
                        S.rec_R.append(R_avg.copy())
                        S.rec_has_sig.append(True)
                        S.rec_sat_cfo.append(float(cfo_hz))
                        S.rec_crb.append(float(crb_deg_))
                        S.rec_mdl_k.append(int(mdl_k))
                        S.rec_sat_id.append(int(trk.sat_id))
                        S.rec_spec2d.append(spec2d.astype(np.float32))
                        if S.rec_iq_enabled:
                            S.rec_X.append(X_cal[:, :_PRE_SAMPLES].copy())

                # Prune stale trackers
                with S.lock:
                    _prune_trackers(S.satellites, _sat_timeout)


# =============================================================================
# Build UI — multi-satellite
# =============================================================================

def _build_ui(S: SimpleNamespace, cfg: UcaConfig,
              algo: str, freq_hz: int) -> None:
    n_ant  = cfg.n_ant
    el_min = cfg.el_min_deg
    el_max = cfg.el_max_deg
    n_az   = cfg.n_az
    H      = C.HISTORY_LEN
    az_rad = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    MAX_SATS = int(getattr(C, "MAX_SATELLITES", 3))
    sat_colors = list(getattr(C, "SAT_COLORS", _SAT_COLORS))

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG, dpi=100)
    fig.patch.set_facecolor(BG)

    # ── Master grid: skyplot left, monitoring panels right ──────────────────
    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[0.85, 1.35],
        left=0.02, right=0.99, top=0.97, bottom=0.025, wspace=0.04,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # LEFT — Skyplot (polar)
    # ═════════════════════════════════════════════════════════════════════════
    ax_sky = fig.add_subplot(gs_outer[0, 0], projection="polar", facecolor=BG2)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90)
    ax_sky.set_rticks([15, 30, 60, 90])
    ax_sky.set_yticklabels(["75°", "60°", "30°", "0°"], fontsize=7, color=C_MUT)
    ax_sky.tick_params(colors=C_MUT, labelsize=7)
    ax_sky.set_facecolor(BG2)
    for sp in ax_sky.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_sky.grid(color=C_BDR, lw=0.5, alpha=0.4)
    ax_sky.set_title("DoA Skyplot", color=C_TEXT, fontsize=11, pad=15)

    sky_dots, sky_trails, sky_labels = [], [], []
    for i in range(MAX_SATS):
        col = sat_colors[i % len(sat_colors)]
        dot, = ax_sky.plot([], [], "o", color=col, ms=14, zorder=8, mec="white", mew=1.5)
        trail, = ax_sky.plot([], [], "o", color=col, ms=4, alpha=0.35, zorder=6)
        lbl = ax_sky.text(0, 0, "", ha="left", va="bottom",
                           color=col, fontsize=7.5, fontweight="bold", zorder=9, visible=False)
        sky_dots.append(dot); sky_trails.append(trail)
        sky_labels.append(lbl)

    lbl_nosig = ax_sky.text(np.pi/2, 45, "waiting\nfor bursts…",
                             ha="center", va="center", color=C_MUT, fontsize=11, fontstyle="italic")

    # ═════════════════════════════════════════════════════════════════════════
    # RIGHT — Monitoring panels (4 rows)
    # ═════════════════════════════════════════════════════════════════════════
    gs_right = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=gs_outer[0, 1],
        height_ratios=[2.2, 2.2, 1.8, 1.5], hspace=0.38,
    )

    # Row 0 — Az/El timeseries (EMA + raw + Kalman overlay)
    ax_hist = fig.add_subplot(gs_right[0, 0], facecolor=BG2)
    ax_hist.set_facecolor(BG2)
    ax_hist.set_title("Azimuth / Elevation — per satellite  (Az left axis, El right axis)", color=C_TEXT, fontsize=9)
    ax_hist.set_xlim(0, H); ax_hist.set_ylim(0, 360)
    ax_hist.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_hist.spines.values(): sp.set_edgecolor(C_BDR)
    ax_hist.grid(color=C_BDR, lw=0.4, alpha=0.4)
    ax_hist.set_ylabel("Azimuth [°]", color=C_MUT, fontsize=7)

    ax_hist_el = ax_hist.twinx()
    ax_hist_el.set_ylim(el_min, max(90.0, el_max))
    ax_hist_el.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_hist_el.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_hist_el.set_ylabel("Elevation [°]", color=C_MUT, fontsize=7)

    hist_az_lines, hist_el_lines = [], []
    hist_az_raw, hist_el_raw = [], []
    hist_az_kf, hist_el_kf = [], []
    for i in range(MAX_SATS):
        col = sat_colors[i % len(sat_colors)]
        la, = ax_hist.plot([], [], "-", color=col, lw=2.0, label=f"S{i} Az")
        le, = ax_hist_el.plot([], [], "--", color=col, lw=1.2, alpha=0.9, label=f"S{i} El")
        lkfa, = ax_hist.plot([], [], "-", color=col, lw=1.3, alpha=0.8, label=f"S{i} Az KF")
        lkfe, = ax_hist_el.plot([], [], ":", color=col, lw=1.3, alpha=0.9, label=f"S{i} El KF")
        lra, = ax_hist.plot([], [], ".", color=col, ms=2.5, alpha=0.4)
        lre, = ax_hist_el.plot([], [], marker="s", color=col, ms=2.5, alpha=0.35, ls="")
        hist_az_lines.append(la); hist_el_lines.append(le)
        hist_az_raw.append(lra); hist_el_raw.append(lre)
        hist_az_kf.append(lkfa); hist_el_kf.append(lkfe)
    _h1, _l1 = ax_hist.get_legend_handles_labels()
    _h2, _l2 = ax_hist_el.get_legend_handles_labels()
    ax_hist.legend(_h1 + _h2, _l1 + _l2, loc="upper left", fontsize=6, ncol=2,
                   facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # Row 1 — 2D MUSIC heatmap + multi-peak
    ax_2d = fig.add_subplot(gs_right[1, 0], facecolor=BG2)
    ax_2d.set_facecolor(BG2)
    ax_2d.set_title("2D MUSIC  az–el  (+ secondary peaks)", color=C_TEXT, fontsize=9)
    ax_2d.set_xlabel("Azimuth [°]", color=C_MUT, fontsize=7)
    ax_2d.set_ylabel("Elevation [°]", color=C_MUT, fontsize=7)
    ax_2d.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_2d.spines.values(): sp.set_edgecolor(C_BDR)
    hm_img = ax_2d.imshow(
        np.full((cfg.n_el, cfg.n_az), -40.0), aspect="auto", origin="lower",
        extent=[0, 360, el_min, el_max], vmin=-40, vmax=0, cmap="plasma",
    )
    hm_peaks = [ax_2d.plot([], [], c=sat_colors[i%len(sat_colors)], marker="D", ms=10, mew=1.5, mec="white", ls="", zorder=5)[0] for i in range(MAX_SATS)]
    hm_peaks2 = [ax_2d.plot([], [], c=sat_colors[i%len(sat_colors)], marker="s", ms=8, mew=1, mec="white", ls="", alpha=0.55, zorder=4)[0] for i in range(MAX_SATS)]
    plt.colorbar(hm_img, ax=ax_2d, fraction=0.04, pad=0.03, label="dB", location="right")

    # Row 2 — Spectrum projection + Eigenvalue profile + Phase monitor
    gs_r2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[2, 0], wspace=0.3)
    ax_spec = fig.add_subplot(gs_r2[0, 0], facecolor=BG2)
    ax_spec.set_facecolor(BG2)
    ax_spec.set_title("Azimuth spectrum", color=C_TEXT, fontsize=8)
    ax_spec.set_xlim(0, 360); ax_spec.set_ylim(-40, 0)
    ax_spec.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_spec.spines.values(): sp.set_edgecolor(C_BDR)
    ax_spec.grid(color=C_BDR, lw=0.3, alpha=0.4)
    az_spec_line, = ax_spec.plot(np.linspace(0, 360, n_az), np.full(n_az, -40), color=C_BLUE, lw=1.2)
    az_spec_markers = [ax_spec.plot([], [], "D", color=sat_colors[i%len(sat_colors)], ms=8, zorder=5)[0] for i in range(MAX_SATS)]

    ax_eig = fig.add_subplot(gs_r2[0, 1], facecolor=BG2)
    ax_eig.set_facecolor(BG2)
    ax_eig.set_title("Eigenvalue profile", color=C_TEXT, fontsize=8)
    ax_eig.set_xlim(-0.5, n_ant - 0.5)
    ax_eig.set_ylim(-2, 40)
    ax_eig.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_eig.spines.values(): sp.set_edgecolor(C_BDR)
    ax_eig.grid(color=C_BDR, lw=0.3, alpha=0.4)
    eig_bars = ax_eig.bar(np.arange(n_ant), np.zeros(n_ant), color=C_TEAL, alpha=0.8)
    eig_idx_highlight = ax_eig.axvline(1.0, color=C_ROSE, alpha=0.75, lw=1.8, ls="--")
    txt_eig_rank = ax_eig.text(0.03, 0.92, "D=1", transform=ax_eig.transAxes,
                               color=C_ROSE, fontsize=7, fontweight="bold")

    ax_ph = fig.add_subplot(gs_r2[0, 2], facecolor=BG2)
    ax_ph.set_facecolor(BG2)
    ax_ph.set_title("ΔΦ residual CH1..4 – CH0  (vs DoA model)", color=C_TEXT, fontsize=8)
    ax_ph.set_xlim(0, H); ax_ph.set_ylim(-90, 90)
    ax_ph.axhline(0, color=C_BDR, lw=0.6)
    ax_ph.axhspan(-30, 30, alpha=0.06, color=C_LIME)
    ax_ph.axhspan(-60, -30, alpha=0.04, color=C_AMBER)
    ax_ph.axhspan(30, 60, alpha=0.04, color=C_AMBER)
    ax_ph.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_ph.spines.values(): sp.set_edgecolor(C_BDR)
    ax_ph.grid(color=C_BDR, lw=0.3, alpha=0.4)
    _ph_colors = [C_BLUE, C_TEAL, C_AMBER, C_VIO]
    ph_lines = [ax_ph.plot([], [], "-", color=_ph_colors[i], lw=1.2, label=f"CH{i+1}")[0] for i in range(4)]
    ax_ph.legend(loc="upper left", fontsize=5.5, ncol=2, facecolor=BG3, edgecolor=C_BDR, labelcolor=C_TEXT)

    # Row 3 — CFO trend + Burst envelope + Status
    gs_r3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[3, 0], wspace=0.3)
    ax_cfo = fig.add_subplot(gs_r3[0, 0], facecolor=BG2)
    ax_cfo.set_facecolor(BG2)
    ax_cfo.set_title("Doppler / CFO trend", color=C_TEXT, fontsize=8)
    ax_cfo.set_xlim(0, 300)
    ax_cfo.set_ylim(-3000, 3000)
    ax_cfo.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_cfo.spines.values(): sp.set_edgecolor(C_BDR)
    ax_cfo.grid(color=C_BDR, lw=0.3, alpha=0.4)
    ax_cfo.axhline(0, color=C_BDR, lw=0.8)
    cfo_line, = ax_cfo.plot([], [], color=C_AMBER, lw=1.4)

    ax_env = fig.add_subplot(gs_r3[0, 1], facecolor=BG2)
    ax_env.set_facecolor(BG2)
    ax_env.set_title("Last burst envelope |IQ|", color=C_TEXT, fontsize=8)
    ax_env.set_xlim(0, _PRE_SAMPLES)
    ax_env.set_ylim(0, 1.2)
    ax_env.tick_params(colors=C_MUT, labelsize=6.5)
    for sp in ax_env.spines.values(): sp.set_edgecolor(C_BDR)
    ax_env.grid(color=C_BDR, lw=0.3, alpha=0.4)
    env_line, = ax_env.plot([], [], color=C_LIME, lw=1.2)
    env_vpre = ax_env.axvline(0, color=C_TEAL, ls="--", lw=1.0, alpha=0.9)
    env_vdata = ax_env.axvline(_PRE_SAMPLES, color=C_AMBER, ls="--", lw=1.0, alpha=0.9)

    ax_stat = fig.add_subplot(gs_r3[0, 2], facecolor=BG2)
    ax_stat.set_facecolor(BG2); ax_stat.set_xlim(0, 10); ax_stat.set_ylim(0, 5); ax_stat.axis("off")
    ax_stat.set_title("Status", color=C_TEXT, fontsize=8)
    txt_stat_rec   = ax_stat.text(0.5, 4.5, "", ha="left", va="top", color=C_LIME, fontsize=7, family="monospace")
    txt_stat_time  = ax_stat.text(0.5, 3.8, "", ha="left", va="top", color=C_TEXT, fontsize=7, family="monospace")
    txt_stat_burst = ax_stat.text(0.5, 3.1, "", ha="left", va="top", color=C_TEXT, fontsize=7, family="monospace")
    txt_stat_cfo   = ax_stat.text(0.5, 2.4, "", ha="left", va="top", color=C_TEXT, fontsize=7, family="monospace")
    txt_stat_cal   = ax_stat.text(0.5, 1.7, "", ha="left", va="top", color=C_AMBER, fontsize=7, family="monospace")
    txt_stat_msg   = ax_stat.text(0.5, 0.5, "", ha="left", va="bottom", color=C_MUT, fontsize=6.5, family="monospace")

    fig.suptitle(
        f"LARK  ·  {algo.upper()}  ·  {freq_hz/1e6:.3f} MHz  ·  "
        f"UCA {n_ant}-ant  r={cfg.radius_lambda:.4f}λ  ·  "
        f"IRA +{_PREAMBLE_TONE_HZ} Hz  ·  el=[{el_min:.0f}°,{el_max:.0f}°]  ·  "
        f"max {MAX_SATS} sat",
        color=C_TEXT, fontsize=7.5, y=0.995,
    )

    def _update(_):
        with S.lock:
            sats_snap = {sid: SimpleNamespace(
                az_deg=t.az_deg, el_deg=t.el_deg,
                az_hist=list(t.az_hist), el_hist=list(t.el_hist),
                az_raw_hist=list(t.az_raw_hist), el_raw_hist=list(t.el_raw_hist),
                az_kf_hist=list(t.az_kf_hist), el_kf_hist=list(t.el_kf_hist),
                cfo_hist=list(t.cfo_hist), cfo_hz=t.cfo_hz,
                papr_db=t.papr_db, snr_db=t.snr_db,
                no_doa=t.no_doa, color=t.color,
                trail=list(t.trail), az_other=t.az_other,
                el_other=t.el_other, has_other=t.has_other,
            ) for sid, t in S.satellites.items()}
            spec2d   = S.spec2d.copy()
            az_spec  = S.az_spec.copy()
            eig      = S.eig_db.copy()
            ph_hist  = [list(q) for q in S.phase_hist]
            n_burst  = S.burst_n
            mdl_k_snap = int(S.mdl_k)
            crb_snap = float(S.crb_az_deg)
            rec_on   = S.rec_enabled
            rec_n    = len(S.rec_t)
            rec_t_snap = list(S.rec_t)
            cfo_diag = list(S.cfo_diag_hist)
            burst_env = S.burst_env.copy()
            burst_pre_idx = int(S.burst_pre_idx)
            burst_data_idx = int(S.burst_data_idx)

        sats_sorted = sorted(sats_snap.values(), key=lambda t: 0 if not t.no_doa else 1)

        # ── Skyplot dots + trails + PAPR rings ──────────────────────────────
        has_any = False
        for i in range(MAX_SATS):
            dot, trail, lbl = (sky_dots[i], sky_trails[i], sky_labels[i])
            if i < len(sats_sorted) and not sats_sorted[i].no_doa:
                t = sats_sorted[i]
                r = float(np.clip(90.0 - t.el_deg, 0, 90))
                th = np.deg2rad(t.az_deg)
                dot.set_data([th], [r]); dot.set_visible(True)
                if t.trail and len(t.trail) > 1:
                    tr = np.array(t.trail)
                    tr_r = 90.0 - tr[:, 1]; tr_th = np.deg2rad(tr[:, 0])
                    trail.set_data(tr_th, tr_r); trail.set_visible(True)
                else:
                    trail.set_visible(False)

                lbl.set_text(f"S{i}  {t.az_deg:.0f}°/{t.el_deg:.0f}°")
                lbl.set_position((th + 0.12, r + 4)); lbl.set_visible(True)
                has_any = True
            else:
                dot.set_visible(False); trail.set_visible(False); lbl.set_visible(False)
        lbl_nosig.set_visible(not has_any)

        # ── 2D MUSIC heatmap + multi-peak ────────────────────────────────────
        hm_img.set_data(spec2d); hm_img.set_clim(-40, 0)
        for i in range(MAX_SATS):
            pk, pk2 = hm_peaks[i], hm_peaks2[i]
            if i < len(sats_sorted) and not sats_sorted[i].no_doa:
                t = sats_sorted[i]
                pk.set_data([t.az_deg], [t.el_deg]); pk.set_visible(True)
                pk2.set_data([t.az_other], [t.el_other]) if t.has_other else None
                pk2.set_visible(t.has_other)
            else:
                pk.set_visible(False); pk2.set_visible(False)

        # ── Az/El history (EMA lines + raw scatter) ──────────────────────────
        for i, (la, le, lkfa, lkfe, lra, lre) in enumerate(
            zip(hist_az_lines, hist_el_lines, hist_az_kf, hist_el_kf, hist_az_raw, hist_el_raw)
        ):
            if i < len(sats_sorted) and not sats_sorted[i].no_doa:
                t = sats_sorted[i]
                xa = np.arange(len(t.az_hist))
                la.set_data(xa, t.az_hist); le.set_data(xa, t.el_hist)
                lkfa.set_data(np.arange(len(t.az_kf_hist)), t.az_kf_hist)
                lkfe.set_data(np.arange(len(t.el_kf_hist)), t.el_kf_hist)
                if t.az_raw_hist:
                    lra.set_data(np.arange(len(t.az_raw_hist)), t.az_raw_hist)
                    lre.set_data(np.arange(len(t.el_raw_hist)), t.el_raw_hist)
            else:
                la.set_data([], []); le.set_data([], [])
                lkfa.set_data([], []); lkfe.set_data([], [])
                lra.set_data([], []); lre.set_data([], [])

        # ── Azimuth spectrum ─────────────────────────────────────────────────
        if az_spec is not None and len(az_spec) > 0:
            az_spec_line.set_ydata(az_spec)
        for i, mk in enumerate(az_spec_markers):
            if i < len(sats_sorted) and not sats_sorted[i].no_doa and az_spec is not None and len(az_spec) > 0:
                _az = float(sats_sorted[i].az_deg % 360.0)
                _idx = int(round((_az / 360.0) * (len(az_spec) - 1)))
                _idx = int(np.clip(_idx, 0, len(az_spec) - 1))
                _y = float(az_spec[_idx])
                mk.set_data([_az], [_y])
                mk.set_visible(True)
            else:
                mk.set_data([], [])
                mk.set_visible(False)

        # ── Phase monitor ────────────────────────────────────────────────────
        for i, (line, q) in enumerate(zip(ph_lines, ph_hist)):
            line.set_data(np.arange(len(q)), q)

        # ── Eigen profile ────────────────────────────────────────────────────
        # eigenvalue_spread_uca_db already normalises noise floor to 0 dB.
        # eig[0] = signal spread (dB), eig[-1] ≈ 0 dB (noise).  No extra
        # normalisation needed — the signal bar must be the tallest, visible bar.
        if len(eig) == n_ant:
            for b, v in zip(eig_bars, eig):
                b.set_height(float(v))
            d_hat = max(1, min(n_ant - 1, mdl_k_snap))
            eig_idx_highlight.set_xdata([float(d_hat), float(d_hat)])
            txt_eig_rank.set_text(f"D={d_hat}")

        # ── CFO trend ───────────────────────────────────────────────────────
        if cfo_diag:
            xc = np.arange(len(cfo_diag))
            cfo_line.set_data(xc, cfo_diag)
            ymax = max(1500.0, 1.1 * float(np.max(np.abs(cfo_diag))))
            ax_cfo.set_ylim(-ymax, ymax)

        # ── Burst envelope ───────────────────────────────────────────────────
        if burst_env.size:
            env = burst_env.astype(float)
            xe = np.arange(len(env))
            env_line.set_data(xe, env)
            ax_env.set_xlim(0, max(len(env), 16))
            y_max = max(1e-6, 1.10 * float(np.max(env)))
            ax_env.set_ylim(0.0, y_max)
            env_vpre.set_xdata([float(burst_pre_idx), float(burst_pre_idx)])
            env_vdata.set_xdata([float(burst_data_idx), float(burst_data_idx)])

        # ── Status dashboard ─────────────────────────────────────────────────
        n_active = len([t for t in sats_snap.values() if not t.no_doa])
        papr0 = sats_sorted[0].papr_db if sats_sorted else 0.0
        snr0  = sats_sorted[0].snr_db if sats_sorted else 0.0

        # ── Status panel ─────────────────────────────────────────────────────
        txt_stat_rec.set_text(f"{'●' if rec_on else '○'} REC  saved: {rec_n}")
        txt_stat_time.set_text(f"elapsed: ~{n_burst * 0.09:.0f}s  (burst × 90ms)")
        txt_stat_burst.set_text("data/doa_iridium/doa_iridium_*.npz")
        cfo0 = sats_sorted[0].cfo_hz if sats_sorted else 0.0
        txt_stat_cfo.set_text(f"CFO: {cfo0/1e3:+.2f} kHz")
        _has_cal = any(o != 0.0 for o in getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", []))
        txt_stat_cal.set_text(f"cal: {'YES' if _has_cal else 'NO'}  gain={getattr(C,'GAIN_DB','?')}dB  multi={getattr(C,'MULTI_BURST_N','?')}")
        crb_str = f"{crb_snap:.1f}°" if crb_snap < 90.0 else "—"
        txt_stat_msg.set_text(
            f"CFO={cfo0/1e3:+.2f}k  PAPR={papr0:.0f}dB  SINR={snr0:.1f}dB  "
            f"K̂={mdl_k_snap}  CRB={crb_str}  acc={n_burst}"
        )

        return (*sky_dots, *sky_trails, *sky_labels, lbl_nosig,
                hm_img, *hm_peaks, *hm_peaks2,
                *hist_az_lines, *hist_el_lines, *hist_az_kf, *hist_el_kf,
                *hist_az_raw, *hist_el_raw,
                az_spec_line, *az_spec_markers, *ph_lines,
            cfo_line, env_line, env_vpre, env_vdata,
                txt_stat_rec, txt_stat_time, txt_stat_burst,
                txt_stat_cfo, txt_stat_cal, txt_stat_msg,
                txt_eig_rank)

    ani = animation.FuncAnimation(fig, _update, interval=C.UPDATE_INTERVAL_MS,
                                   blit=False, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        S.running = False


def _run_calibration(
    acc: CovarianceAccumulatorUca, S: SimpleNamespace,
    cfg: UcaConfig, known_az_deg: float,
    known_el_deg: float = 0.0,
) -> None:
    import datetime, re
    # Show existing offsets so the user knows whether calibration is cumulative.
    _existing_now = list(getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * cfg.n_ant))
    _existing_now = (_existing_now + [0.0] * cfg.n_ant)[:cfg.n_ant]
    _has_existing = any(o != 0.0 for o in _existing_now)
    if _has_existing:
        print(f"\n[CAL] NOTE: existing offsets will be COMPOSED with the new residual:")
        print(f"[CAL]   existing = {np.round(_existing_now, 2).tolist()}")
        print(f"[CAL]   total = existing + residual  (correct from raw data)")
    print(f"\n[CAL] Collecting bursts for {_CAL_DURATION_S:.0f} s  "
          f"(TX az={known_az_deg:.1f}°  el={known_el_deg:.1f}°)")
    print( "[CAL] Do NOT move the TX or the array during calibration.")
    t0 = time.time()
    # Simple running sum of EMA snapshots — equal weight to every second
    # regardless of COV_ALPHA.  Averaging N snapshots reduces eigenvector
    # variance by √N even when each snapshot has the same EMA memory.
    R_sum: np.ndarray | None = None
    n_sum   = 0
    prev_n  = 0
    while time.time() - t0 < _CAL_DURATION_S:
        time.sleep(1.0)
        if not S.running:
            break
        cur_n = acc.n_updates
        if cur_n > prev_n and acc.R is not None:
            if R_sum is None:
                R_sum = acc.R.copy()
            else:
                R_sum = R_sum + acc.R
            n_sum  += 1
            prev_n  = cur_n
        elapsed = time.time() - t0
        pct = min(100, int(elapsed / _CAL_DURATION_S * 100))
        print(f"\r  [{elapsed:5.1f}s / {_CAL_DURATION_S:.0f}s]  "
              f"{acc.n_updates} burst-avgd R matrices  [{pct:3d}%]",
              end="", flush=True)
    print()
    if R_sum is None or acc.n_updates < _CAL_MIN_UPDATES:
        print(f"[CAL] Only {acc.n_updates} valid bursts (need {_CAL_MIN_UPDATES}). "
              f"Check TX / Heimdall."); return

    R_cal = R_sum / n_sum   # time-averaged covariance
    ev, V = np.linalg.eigh(R_cal)
    v = V[:, -1]
    v = v * np.exp(-1j * np.angle(v[0]))

    az_rad = np.deg2rad(known_az_deg)
    el_rad = np.deg2rad(known_el_deg)
    pos    = cfg.positions
    cos_el = np.cos(el_rad)
    tau    = 2*np.pi*(pos[:,0]*cos_el*np.sin(az_rad) + pos[:,1]*cos_el*np.cos(az_rad))
    tau   -= tau[0]

    hw_offsets = np.degrees(np.angle(v) - tau)
    hw_offsets = (hw_offsets + 180) % 360 - 180
    hw_offsets[0] = 0.0

    # ── Compose with the offsets that were active during data collection ──────
    # If the program was already running with a previous calibration, X_cal was
    # pre-corrected with those offsets before being fed to acc.update().  The
    # eigenvector therefore encodes only the RESIDUAL hardware error on top of
    # the existing correction.  We must ADD the existing offsets to obtain the
    # total correction that works from raw (uncorrected) X_bpf.
    #
    #   total[k] = existing[k] + hw_residual[k]
    #
    # When calibrating from scratch (existing = [0,0,...,0]) this is a no-op.
    existing = list(getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0] * cfg.n_ant))
    existing = (existing + [0.0] * cfg.n_ant)[:cfg.n_ant]
    total_offsets = hw_offsets + np.array(existing, dtype=float)
    total_offsets = (total_offsets + 180) % 360 - 180
    total_offsets[0] = 0.0

    # ── MUSIC frame alignment (same decorr as runtime) ───────────────────────
    _decorr = str(getattr(C, "MUSIC_DECORR", "none")).lower()
    _spec_cal = doa_music_uca_2d(
        np.zeros((cfg.n_ant, 1), dtype=np.complex128),
        cfg, R_in=R_cal, n_snapshots=acc.n_updates, decorr=_decorr,
    )
    _peak_az, _peak_el, _peak_papr = find_peak_uca_2d(_spec_cal, cfg)
    _doa_az_off = ((known_az_deg - _peak_az + 180.0) % 360.0) - 180.0
    _doa_el_off = float(known_el_deg - _peak_el)

    print(f"\n[CAL] Hardware phase offsets from {acc.n_updates} burst-avgd matrices "
          f"(simple avg over {n_sum} snapshots):")
    print(f"  ┌─ Written to config.py automatically ──────────────────────────")
    print(f"  │  residual  = {hw_offsets.round(2).tolist()}")
    print(f"  │  existing  = {np.array(existing).round(2).tolist()}")
    print(f"  │  CHANNEL_PHASE_OFFSETS_DEG = {total_offsets.round(2).tolist()}")
    print(f"  │  CAL_REFERENCE_AZ_DEG = {known_az_deg:.1f}")
    print(f"  │  CAL_REFERENCE_EL_DEG = {known_el_deg:.1f}")
    print(f"  │  MUSIC peak (decor={_decorr}): az={_peak_az:.1f}° el={_peak_el:.1f}° "
          f"papr={_peak_papr:.1f} dB")
    print(f"  │  DOA_AZ_OFFSET_DEG = {_doa_az_off:.2f}")
    print(f"  │  DOA_EL_OFFSET_DEG = {_doa_el_off:.2f}")
    print(f"  └───────────────────────────────────────────────────────────────")
    print(f"[CAL] These offsets are PERMANENT — they survive restarts.")
    print(f"[CAL] Re-run calibration only if you change cables or the AD9363.")

    cfg_path = os.path.join(_HERE, "config.py")
    try:
        with open(cfg_path) as f: txt = f.read()
        _cal_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        _cal_cmt = (f"  # auto-cal {_cal_ts} from {acc.n_updates} bursts "
                    f"({n_sum} snapshots) az={known_az_deg:.1f}° el={known_el_deg:.1f}°")
        if "CHANNEL_PHASE_OFFSETS_DEG" in txt:
            new_val = f"CHANNEL_PHASE_OFFSETS_DEG = {total_offsets.round(2).tolist()}"
            txt = re.sub(
                r"CHANNEL_PHASE_OFFSETS_DEG = \[.*?\].*",
                new_val + _cal_cmt, txt,
            )
        _scalar_updates = {
            "CAL_REFERENCE_AZ_DEG": f"{known_az_deg:.2f}",
            "CAL_REFERENCE_EL_DEG": f"{known_el_deg:.2f}",
            "DOA_AZ_OFFSET_DEG":    f"{_doa_az_off:.2f}",
            "DOA_EL_OFFSET_DEG":    f"{_doa_el_off:.2f}",
        }
        for _key, _val in _scalar_updates.items():
            _line = f"{_key} = {_val}"
            if re.search(rf"^{re.escape(_key)} = ", txt, re.M):
                txt = re.sub(rf"^{re.escape(_key)} = .*$", _line, txt, count=1, flags=re.M)
            else:
                txt = txt.replace(
                    _cal_cmt.strip(),
                    _cal_cmt.strip() + f"\n{_line}",
                    1,
                )
        with open(cfg_path, "w") as f: f.write(txt)
        print("[CAL] config.py updated. Restart DoA to apply.")
    except Exception as e:
        print(f"[CAL] Update failed ({e}) — edit config.py manually.")


# =============================================================================
# Recording
# =============================================================================

def _save_recording(S: SimpleNamespace, out_dir: str, tag: str = "") -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"doa_iridium_{ts}{tag}")
    payload = dict(
        t        = np.array(S.rec_t),
        az_deg   = np.array(S.rec_az),
        el_deg   = np.array(S.rec_el),
        papr_db  = np.array(S.rec_papr),
        snr_db   = np.array(S.rec_snr),
        eig_db   = np.array(S.rec_eig),
        phase_diff = np.array(S.rec_phase),
        phase_residual = np.array(S.rec_phase_residual) if S.rec_phase_residual else np.array([]),
        R        = np.array(S.rec_R),
        has_sig  = np.array(S.rec_has_sig),
        sat_cfo_hz = np.array(S.rec_sat_cfo),
        # Cramér-Rao Bound and MDL source count per burst (Salama 2025 §8.2.1, §4.2.4)
        crb_az_deg = np.array(S.rec_crb),
        mdl_k      = np.array(S.rec_mdl_k),
        # Per-estimate tracker id + raw MUSIC spectrum
        sat_id     = np.array(S.rec_sat_id, dtype=np.int32),
        spec2d     = np.array(S.rec_spec2d, dtype=np.float32),  # (N, n_el, n_az)
        freq_hz  = np.array([C.FREQ_HZ]),
    )
    if S.rec_iq_enabled and S.rec_X:
        payload["bursts"] = np.array(S.rec_X, dtype=np.complex64)  # (N, n_ant, n_samp)
    np.savez_compressed(base + ".npz", **payload)
    iq_note = f" + {len(S.rec_X)} IQ windows" if S.rec_iq_enabled and S.rec_X else ""
    print(f"[REC] Saved {base}.npz  ({len(S.rec_t)} bursts{iq_note})")


# =============================================================================
# Indoor TX mode profiles (match tx/indoor_1626.py --mode)
# =============================================================================
# (rimossa: _apply_indoor_tx_mode_profile — sostituita da _load_scenario_profile)
# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            f"2D multi-satellite DoA on Iridium IRA bursts "
            f"(preamble tone +{_PREAMBLE_TONE_HZ} Hz) — KrakenSDR UCA RHCP"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--freq",    type=float, default=C.FREQ_HZ / 1e6,
                   help="Central RF frequency [MHz]")
    p.add_argument("--gain",    type=float, default=C.GAIN_DB)
    p.add_argument("--radius",  type=float, default=C.RADIUS_LAMBDA,
                   help="UCA radius in wavelengths")
    p.add_argument("--offset",  type=float, default=C.ANT0_OFFSET_DEG,
                   help="Antenna-0 offset from North [°]")
    p.add_argument("--tx-mode", choices=["ira", "pass", "track"], default=None,
                   help="Indoor TX mode profile. Overrides config INDOOR_TX_MODE for this run.")
    p.add_argument("--algo",
                   choices=["music","capon","bartlett","phase-fit",
                             "root-music","unitary-esprit","mfba-music"],
                   default=C.DOA_ALGORITHM.lower())
    p.add_argument("--nsig",    type=int,   default=C.NUM_SIGNALS,
                   help="Expected sources per satellite (MUSIC subspace D)")
    p.add_argument("--alpha",   type=float, default=C.COV_ALPHA,
                   help="Covariance EMA weight per burst (0=no memory)")
    p.add_argument("--demo",    action="store_true",
                   help="Synthetic simulation (no hardware required)")
    p.add_argument("--n-demo-sats", type=int, default=2, choices=[1,2,3],
                   help="Number of synthetic satellites in --demo mode")
    p.add_argument("--max-sats",type=int,   default=int(getattr(C,"MAX_SATELLITES",3)),
                   help="Max tracked satellites (overrides config)")
    p.add_argument("--fd-max",  type=float, default=None, metavar="HZ",
                   help="Doppler gate: discard peaks with |cfo_hz| > HZ. "
                        "0 = no gate (outdoor). 5000 = indoor TX-only mode.")
    p.add_argument("--papr-min",type=float, default=None,
                   help="Minimum MUSIC PAPR [dB] for multi-burst DoA acceptance "
                        "(default: from INDOOR_TX_MODE profile or config)")
    p.add_argument("--snr-min", type=float, default=None,
                   help="Minimum per-element SNR [dB] to accept a burst (calibration-agnostic)")
    p.add_argument("--multi",   type=int, default=None,
                   help="Bursts to average for DoA (default: from INDOOR_TX_MODE profile or config)")
    p.add_argument("--calibrate", type=float, default=None, metavar="AZ_DEG",
                   help="Auto-calibrate HW phase offsets with TX at known az=AZ_DEG")
    p.add_argument("--calibrate-el", type=float, default=0.0, metavar="EL_DEG",
                   help="Known TX elevation [°] for HW phase calibration (3D UCA model)")
    p.add_argument("--out-dir", default=_DATA_DIR, metavar="DIR")
    p.add_argument("--no-rec",  action="store_true")
    p.add_argument("--save-iq", action="store_true")
    p.add_argument("--max-time", type=float, default=0.0, metavar="SECONDS",
                   help="Stop automatically after SECONDS of wall time (0 = no limit). "
                        "Recording is saved on exit regardless.")
    p.add_argument("--no-plot", action="store_true",
                   help="Headless mode: run acquisition loop without opening the Qt GUI. "
                        "Useful for SSH sessions or automated test runs. "
                        "DIAG lines are printed to stdout every 10 s.")
    p.add_argument("--scenario",
                   choices=["indoor_ira", "indoor_pass", "outdoor"],
                   default=getattr(C, "SCENARIO", "outdoor"),
                   help="Operational scenario: loads the appropriate default profile. "
                        "Overridable with --tx-mode, --fd-max, --multi, --papr-min.")
    args = p.parse_args()

    # ── Apply scenario profile ────────────────────────────────────────────────
    # Collect explicit CLI overrides (only those actually provided by the user).
    _cli_overrides: dict[str, object] = {}
    if args.tx_mode is not None:
        _cli_overrides["INDOOR_TX_MODE"] = args.tx_mode
    if args.fd_max is not None:
        _cli_overrides["DOPPLER_GATE_HZ"] = args.fd_max
    if args.multi is not None:
        _cli_overrides["MULTI_BURST_N"] = args.multi
    if args.papr_min is not None:
        _cli_overrides["PAPR_INST_MIN_DB"] = args.papr_min
    if args.snr_min is not None:
        _cli_overrides["SNR_INST_MIN_DB"] = args.snr_min
    _cli_overrides["MAX_SATELLITES"] = args.max_sats

    _load_scenario_profile(args.scenario, _cli_overrides)

    freq_hz = int(args.freq * 1e6)
    C.FREQ_HZ = freq_hz

    _tx_mode = str(getattr(C, "INDOOR_TX_MODE", "pass")).lower()
    if _tx_mode in ("ira", "pass") and int(getattr(C, "MULTI_BURST_N", 8)) > 64:
        print(f"[WARN] MULTI_BURST_N={C.MULTI_BURST_N} is very high for indoor lab — "
              f"use 32 (ira, ≈4 s) or 4 (pass).  Very long windows smear covariance "
              f"across multipath drift.")

    _fd_gate_for_mode = float(getattr(C, "DOPPLER_GATE_HZ", 0.0))
    _el_max_cfg = float(getattr(C, "EL_MAX_DEG", 90.0))
    if _tx_mode in ("ira", "pass"):
        _el_max_cfg = min(_el_max_cfg, float(getattr(C, "INDOOR_EL_MAX_DEG", _el_max_cfg)))
    elif _fd_gate_for_mode > 0.0:
        _el_max_cfg = min(_el_max_cfg, float(getattr(C, "INDOOR_EL_MAX_DEG", _el_max_cfg)))

    cfg = UcaConfig(
        n_ant=C.N_ANTENNAS, radius_lambda=args.radius,
        n_az=C.N_AZ, n_el=C.N_EL,
        el_min_deg=C.EL_MIN_DEG,
        el_max_deg=_el_max_cfg,
        num_expected_signals=args.nsig,
        ant0_offset_deg=args.offset,
        ant_ccw=C.ANT_CCW,   # False = CW (clockwise)
    )
    acc = CovarianceAccumulatorUca(alpha=args.alpha)
    S   = _make_state(cfg.n_az, cfg.n_el)
    S.rec_enabled    = not args.no_rec
    S.rec_iq_enabled = args.save_iq

    ccw_str = "CCW" if C.ANT_CCW else "CW (clockwise)"
    print("=" * 66)
    print(f"  DoA IRIDIUM MULTI-SATELLITE  —  {args.algo.upper()}  @  {freq_hz/1e6:.3f} MHz")
    if freq_hz < 1_000_000_000:
        print(f"  [MODE] Indoor lab 868 MHz  — LibreSDR TX required")
    else:
        print(f"  [MODE] Indoor/Outdoor 1626 MHz  — LibreSDR TX (indoor) or real Iridium (outdoor)")
    print(f"  UCA: {cfg.n_ant} ant  {ccw_str}  r={args.radius:.4f}λ  offset={args.offset:.1f}°  RHCP")
    print(f"  Heimdall: {C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}")
    print(f"  Preamble tone: +{_PREAMBLE_TONE_HZ} Hz  |  burst: {_BURST_SYMS} sym  |  SF: {int(_SUPERFRAME_S*1000)} ms")
    print(f"  Max satellites: {args.max_sats}  "
          f"multi={getattr(C, 'MULTI_BURST_N', 4)}  "
          f"snr_min={getattr(C, 'SNR_INST_MIN_DB', 0):.0f} dB  "
          f"papr_min={getattr(C, 'PAPR_INST_MIN_DB', 0):.1f} dB")
    _tx_mode = str(getattr(C, "INDOOR_TX_MODE", "pass")).lower()
    if _tx_mode in ("ira", "pass", "track"):
        print(f"  Indoor TX mode: {_tx_mode}  (match tx/indoor_1626.py --mode {_tx_mode})")
        if _tx_mode == "ira":
            print("  TX cmd: python3 tx/indoor_1626.py --mode ira --gain -50 --cyclic")
            print("         (CFO≈0 Hz — NOT --mode pass which chirps Doppler)")
        elif _tx_mode == "pass":
            print("  TX cmd: python3 tx/indoor_1626.py --mode pass --gain -50 --cyclic")
    scan_bw = int(getattr(C, "DOPPLER_SCAN_BW_HZ", 45_000))
    sep_hz  = int(getattr(C, "SAT_MIN_SEP_HZ", 5_000))
    fd_gate = float(getattr(C, "DOPPLER_GATE_HZ", 0.0))
    fd_gate_str = f"  fd-gate: ±{fd_gate/1e3:.1f} kHz  (indoor TX mode)" if fd_gate > 0 else "  fd-gate: OFF  (outdoor/satellite mode)"
    print(f"  Doppler scan: ±{scan_bw/1e3:.0f} kHz  min separation: {sep_hz/1e3:.0f} kHz  |{fd_gate_str}")
    phase_offs = getattr(C, "CHANNEL_PHASE_OFFSETS_DEG", [0.0]*cfg.n_ant)
    if any(o != 0.0 for o in phase_offs):
        print(f"  HW phase cal: {[f'{o:.1f}' for o in phase_offs]} °")
    else:
        print("  HW phase cal: not calibrated — run --calibrate <az_deg> [--calibrate-el <el_deg>]")
    print("=" * 66)
    _check_narrowband(freq_hz, cfg)

    if not args.demo:
        if not _check_heimdall(C.HEIMDALL_HOST, C.HEIMDALL_PORT):
            print(f"\n[ERROR] Heimdall DAQ unreachable at "
                  f"{C.HEIMDALL_HOST}:{C.HEIMDALL_PORT}.")
            print("  Start Heimdall first (task 'Heimdall: Start'), then retry.")
            print("  For offline testing: add --demo")
            sys.exit(1)
        src = KrakenIQSource(
            host=C.HEIMDALL_HOST,
            port=C.HEIMDALL_PORT,
            num_channels=C.N_ANTENNAS,
            freq_hz=freq_hz,
            gain_db=args.gain,
        )
        src.start()
    else:
        src        = None
        demo_params = _DEMO_SATS[:args.n_demo_sats]
        print(f"[DEMO] {args.n_demo_sats} synthetic satellite(s) active:")
        for i, d in enumerate(demo_params):
            print(f"  S{i}: az={d['az']:.0f}°  el={d['el']:.0f}°  "
                  f"fd={d['doppler']:+.0f} Hz  SNR={d['snr_db']:.0f} dB  "
                  f"({_SAT_COLORS[i]})")
        print()

    if args.calibrate is not None:
        acq_thread = threading.Thread(
            target=_acq_loop,
            args=(src, cfg, args.algo, acc, S, args.demo,
                  _DEMO_SATS[:1] if args.demo else None),
            daemon=True,
        )
        acq_thread.start()
        try:
            _run_calibration(acc, S, cfg, args.calibrate, args.calibrate_el)
        finally:
            S.running = False
        return

    def _auto_save_loop() -> None:
        last_n = 0
        while S.running:
            time.sleep(5.0)
            with S.lock:
                n = len(S.rec_t)
            if n - last_n >= 200:
                with S.lock:
                    snap = SimpleNamespace(**vars(S))
                _save_recording(snap, args.out_dir, tag="_autosave")
                last_n = n

    demo_params_arg = _DEMO_SATS[:args.n_demo_sats] if args.demo else None
    acq_thread = threading.Thread(
        target=_acq_loop,
        args=(src, cfg, args.algo, acc, S, args.demo, demo_params_arg),
        daemon=True,
    )
    acq_thread.start()
    if S.rec_enabled and args.out_dir:
        threading.Thread(target=_auto_save_loop, daemon=True).start()

    try:
        if args.no_plot:
            print("[HEADLESS] Acquisition running. Press Ctrl+C to stop.")
            if args.max_time > 0:
                print(f"[HEADLESS] Auto-stop in {args.max_time:.0f} s.")
                _t0_run = time.monotonic()
                while acq_thread.is_alive():
                    time.sleep(1.0)
                    if time.monotonic() - _t0_run >= args.max_time:
                        print(f"[HEADLESS] Max time {args.max_time:.0f}s reached — stopping.")
                        S.running = False
                        break
            else:
                acq_thread.join()
        else:
            _build_ui(S, cfg, args.algo, freq_hz)
    except KeyboardInterrupt:
        pass
    finally:
        S.running = False
        if src is not None and not args.demo:
            src.stop()
        if S.rec_enabled and S.rec_t and args.out_dir:
            with S.lock:
                snap = SimpleNamespace(**vars(S))
            _save_recording(snap, args.out_dir)


if __name__ == "__main__":
    main()
