"""
core.iridium_doa_pipeline
=========================
Unified burst-gated DoA estimator for Iridium IRA.

Shared by iridium_burst_doa_runner (real-time) and validate_pipeline (offline)
so both produce identical results from identical input data.

One estimate every ``n_bursts`` × 90 ms (default n_bursts=50 → 4.5 s).

Pipeline per burst
------------------
    X_win (n_ant, N_pre)  raw preamble IQ
        │
        ▼  extract_pilot_tone        BPF around preamble tone (Hann soft window)
        ▼  amplitude_normalize_channels
        ▼  apply_phase_correction    hardware calibration offsets
        ▼  skip BPF guard samples
        │
        ▼  rolling buffer (n_bursts deep)
        │
        ▼  Doppler-align windows     re-phase each window to current CFO
        ▼  stack → R_avg             sample covariance over all windows
        │
        ▼  doa_music_uca_2d          2D MUSIC (el × az)
        ▼  pick_doa_peak_uca_2d      indoor scorer with hint from previous estimate
        ▼  DOA frame offsets         DOA_AZ_OFFSET_DEG / DOA_EL_OFFSET_DEG
        │
        ▼  DoaResult (az, el, papr, snr, spec2d)

References
----------
* Schmidt 1986  — MUSIC
* Van Trees 2002 §8.3 — UCA steering, CRB
* PySDR textbook (pysdr.org) — covariance formula R=XX^H/N, pinv for Capon
* Iridium IRA frame: 64 preamble symbols @ Rs/8 = 3125 Hz above carrier
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .doa_uca_2d import (
    UcaConfig,
    doa_music_uca_2d,
    pick_doa_peak_uca_2d,
    extract_pilot_tone,
    amplitude_normalize_channels,
    snr_uca_db,
    crb_azimuth_deg,
)
from .doa_algorithms import apply_phase_correction

__all__ = ["DoaEstimatorConfig", "DoaEstimator", "DoaResult"]

_IRA_FS          = 1_024_000.0   # KrakenSDR default sample rate [Hz]
_PREAMBLE_TON_HZ = 3_125.0       # Iridium preamble tone = Rs/8 [Hz above LO]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DoaResult:
    """Single DoA estimate from DoaEstimator.push()."""
    az_deg:    float          # calibrated azimuth  [°, 0–360, CW from North]
    el_deg:    float          # calibrated elevation [°, from horizon]
    papr_db:   float          # MUSIC peak-to-average power ratio [dB]
    snr_db:    float          # eigenvalue SINR [dB]
    crb_deg:   float          # Cramér-Rao bound on azimuth [°]
    n_bursts:  int            # burst windows in this estimate
    spec2d:    np.ndarray     # (n_el, n_az) MUSIC spectrum [dB, peak=0]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DoaEstimatorConfig:
    """
    All parameters for DoaEstimator.

    Defaults match the indoor_ira scenario.
    For outdoor use: n_bursts=4, indoor=False, phase/hint weights=0.
    """
    cfg:              UcaConfig
    n_bursts:         int   = 50         # window size; 50×90ms = 4.5 s per estimate
    phase_offsets:    list  = field(default_factory=list)   # CHANNEL_PHASE_OFFSETS_DEG
    az_offset_deg:    float = 0.0        # DOA_AZ_OFFSET_DEG   (from --calibrate)
    el_offset_deg:    float = 0.0        # DOA_EL_OFFSET_DEG
    bpf_bw_hz:        float = 8_000.0   # BPF half-bandwidth around preamble tone
    bpf_guard:        int   = 128        # samples to skip after BPF (ringing)
    decorr:           str   = "none"      # UCA: "none" only; "fb" shifts peak ~180°
    indoor:           bool  = True       # True → elevation-preference peak scorer
    el_pref_lo:       float = 8.0        # preferred el range lower bound [°]
    el_pref_hi:       float = 35.0       # preferred el range upper bound [°]
    phase_score_w:    float = 0.35       # weight of phase-match score in peak picker
    az_hint_w:        float = 0.25       # weight of az-hint score
    el_hint_w:        float = 0.15       # weight of el-hint score
    mirror_margin:    float = 10.0       # 180°-ambiguity resolution margin [°]
    fs:               float = _IRA_FS    # sample rate [Hz]
    num_signals:      int   = 1          # expected sources (MUSIC subspace D)

    def __post_init__(self) -> None:
        if not self.phase_offsets:
            self.phase_offsets = [0.0] * self.cfg.n_ant

    @property
    def has_cal(self) -> bool:
        return any(o != 0.0 for o in self.phase_offsets)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class DoaEstimator:
    """
    Burst-gated 2D DoA estimator — shared by runner and offline validator.

    Usage
    -----
        cfg_est = DoaEstimatorConfig(cfg=uca_cfg, n_bursts=50, ...)
        est = DoaEstimator(cfg_est)

        for X_win, cfo_hz in burst_stream:   # X_win: (n_ant, N_pre)
            result = est.push(X_win, cfo_hz)
            if result is not None:
                print(f"az={result.az_deg:.1f}°  el={result.el_deg:.1f}°")

    The estimator maintains a sliding window of the last ``n_bursts`` burst
    covariances.  Every time a new burst is pushed it produces a fresh
    estimate (sliding, not block) so the output rate equals the burst rate
    after the initial warm-up of ``n_bursts`` bursts.
    """

    def __init__(self, config: DoaEstimatorConfig) -> None:
        self._c = config
        # Pre-compute time vector for Doppler alignment (reused per push)
        # Length = pre_samples - bpf_guard; set lazily on first push.
        self._t_vec:  Optional[np.ndarray] = None
        self._R_buf:   deque = deque(maxlen=config.n_bursts)
        self._X_buf:   deque = deque(maxlen=config.n_bursts)
        self._cfo_buf: deque = deque(maxlen=config.n_bursts)
        self._prev_az: Optional[float] = None
        self._prev_el: Optional[float] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all internal state (call when array orientation changes)."""
        self._R_buf.clear()
        self._X_buf.clear()
        self._cfo_buf.clear()
        self._prev_az = None
        self._prev_el = None

    # ------------------------------------------------------------------
    def push(
        self,
        X_win:  np.ndarray,   # (n_ant, N_pre) raw preamble IQ, BEFORE BPF
        cfo_hz: float,        # Doppler offset of the detected preamble tone [Hz]
    ) -> Optional[DoaResult]:
        """
        Push one burst preamble window.

        Parameters
        ----------
        X_win   : (n_ant, N_pre) complex — raw IQ window.
                  Must span at least the preamble region (typically 2621
                  samples at 1.024 MSPS).  Raw (un-BPF'd, un-calibrated).
        cfo_hz  : Doppler CFO of the detected preamble tone [Hz].
                  cfo_hz = detected_tone_hz - 3125 Hz.

        Returns
        -------
        DoaResult or None during the first ``n_bursts - 1`` calls.
        """
        c = self._c

        # ── 1. BPF around preamble tone (Hann soft window, see PySDR filters) ──
        tone_hz = _PREAMBLE_TON_HZ + cfo_hz
        Xp = extract_pilot_tone(X_win, c.fs, tone_hz, bw_hz=c.bpf_bw_hz)

        # ── 2. Amplitude normalisation (cancels per-channel gain imbalance) ────
        Xp = amplitude_normalize_channels(Xp)

        # ── 3. Hardware phase calibration ──────────────────────────────────────
        if c.has_cal:
            Xp = apply_phase_correction(Xp, c.phase_offsets)

        # ── 4. Skip BPF ringing (impulse response width ≈ fs / bpf_bw) ────────
        Xp = Xp[:, c.bpf_guard:]          # (n_ant, N_pre - bpf_guard)
        N  = Xp.shape[1]

        # Lazily build the time index vector for Doppler re-alignment.
        # The index starts at bpf_guard because that is the sample offset
        # relative to the start of the original window.
        if self._t_vec is None or len(self._t_vec) != N:
            self._t_vec = np.arange(c.bpf_guard, c.bpf_guard + N, dtype=np.float64)

        # ── 5. Per-burst sample covariance ─────────────────────────────────────
        R_inst = (Xp @ Xp.conj().T) / N
        self._R_buf.append(R_inst)
        self._X_buf.append(Xp)
        self._cfo_buf.append(float(cfo_hz))

        if len(self._R_buf) < c.n_bursts:
            return None   # still warming up

        # ── 6. Doppler-align all windows to current CFO, then stack ────────────
        # Each window was acquired at a slightly different CFO; re-phase them
        # all to the current CFO before stacking so the CW preamble tone adds
        # coherently across the whole window (critical for LEO Doppler drift).
        # When n_bursts == 1 there is nothing to align; R_avg = R_inst directly.
        if c.n_bursts == 1:
            R_avg = self._R_buf[0]
            X_big = self._X_buf[0]
        else:
            X_parts = []
            for x_h, cfo_h in zip(self._X_buf, self._cfo_buf):
                d_cfo = float(cfo_hz) - cfo_h
                if abs(d_cfo) > 30.0:
                    # rot = exp(+j·2π·Δf·t)  rotates the historical burst
                    # from its old CFO to the current CFO so they stack coherently.
                    rot = np.exp(2j * np.pi * d_cfo / c.fs * self._t_vec)
                    X_parts.append(x_h * rot)
                else:
                    X_parts.append(x_h)
            X_big = np.hstack(X_parts)                       # (n_ant, N * n_bursts)
            R_avg = (X_big @ X_big.conj().T) / X_big.shape[1]

        # ── 7. 2D-MUSIC ────────────────────────────────────────────────────────
        cfg_use = c.cfg
        if c.num_signals != cfg_use.num_expected_signals:
            from dataclasses import replace
            cfg_use = replace(cfg_use, num_expected_signals=c.num_signals)

        spec2d = doa_music_uca_2d(
            X_big, cfg_use,
            R_in=R_avg,
            decorr=c.decorr,
            n_snapshots=X_big.shape[1],
        )

        # ── 8. Peak picking with hint from previous estimate ───────────────────
        # pick_doa_peak_uca_2d (indoor=True) considers elevation preference,
        # phase-match score, and a directional hint from the previous estimate
        # to resolve the 180° UCA ambiguity.  This is the same function used
        # by the runner; using it here makes offline = online.
        phase_diffs = np.degrees(np.angle(R_avg[1:, 0]))
        az_raw, el_raw, papr = pick_doa_peak_uca_2d(
            spec2d, cfg_use,
            indoor=c.indoor,
            el_pref_lo=c.el_pref_lo,
            el_pref_hi=c.el_pref_hi,
            phase_diffs=phase_diffs,
            az_hint_deg=self._prev_az,
            el_hint_deg=self._prev_el,
            phase_score_weight=c.phase_score_w,
            az_hint_score_weight=c.az_hint_w,
            el_hint_score_weight=c.el_hint_w,
            mirror_margin_deg=c.mirror_margin,
        )

        # ── 9. Quality metrics ──────────────────────────────────────────────────
        snr   = float(snr_uca_db(R_avg))
        n_ant = cfg_use.n_ant
        snr_per_el = snr - 10.0 * np.log10(float(n_ant))
        crb   = float(crb_azimuth_deg(snr_per_el, X_big.shape[1], cfg_use, el_raw))

        # ── 10. Optional DOA frame offset (default 0 — caller applies if needed) ─
        # By default az_offset_deg=0 and el_offset_deg=0 so push() returns the
        # RAW MUSIC direction.  The runner applies DOA_AZ_OFFSET_DEG /
        # DOA_EL_OFFSET_DEG after the call; validate_pipeline compares raw.
        if c.az_offset_deg != 0.0 or c.el_offset_deg != 0.0:
            az_out = (az_raw + c.az_offset_deg) % 360.0
            el_out = float(np.clip(
                el_raw + c.el_offset_deg,
                cfg_use.el_min_deg, cfg_use.el_max_deg,
            ))
        else:
            az_out, el_out = az_raw, el_raw

        # Update hint with the OUTPUT az/el so subsequent calls use the same frame
        self._prev_az = az_out
        self._prev_el = el_out

        return DoaResult(
            az_deg   = az_out,
            el_deg   = el_out,
            papr_db  = float(papr),
            snr_db   = snr,
            crb_deg  = crb,
            n_bursts = len(self._R_buf),
            spec2d   = spec2d,
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def estimator_from_config(C, UcaConfigClass=None) -> "DoaEstimator":
    """
    Build a DoaEstimator from a loaded config module ``C``.

    Reads the same attributes that iridium_burst_doa_runner.py uses so that
    runner and validator are guaranteed to use identical parameters.

    Example
    -------
        import config as C
        from core.doa_uca_2d import UcaConfig
        from core.iridium_doa_pipeline import estimator_from_config

        est = estimator_from_config(C)
        result = est.push(X_preamble, cfo_hz)
    """
    if UcaConfigClass is None:
        from .doa_uca_2d import UcaConfig as UcaConfigClass  # type: ignore

    uca_cfg = UcaConfigClass(
        n_ant              = int(getattr(C, "N_ANTENNAS",       5)),
        radius_lambda      = float(getattr(C, "RADIUS_LAMBDA",  0.4253)),
        n_az               = int(getattr(C, "N_AZ",             360)),
        n_el               = int(getattr(C, "N_EL",             86)),
        el_min_deg         = float(getattr(C, "EL_MIN_DEG",     5.0)),
        el_max_deg         = float(getattr(C, "EL_MAX_DEG",     90.0)),
        ant0_offset_deg    = float(getattr(C, "ANT0_OFFSET_DEG", 0.0)),
        ant_ccw            = bool(getattr(C, "ANT_CCW",         False)),
        num_expected_signals = int(getattr(C, "NUM_SIGNALS",    1)),
    )

    _fd_gate   = float(getattr(C, "DOPPLER_GATE_HZ",     0.0))
    _tx_mode   = str(getattr(C, "INDOOR_TX_MODE",        "pass")).lower()
    _indoor    = (_fd_gate > 0.0) or (_tx_mode in ("ira", "pass"))
    _el_max    = float(getattr(C, "INDOOR_EL_MAX_DEG",   90.0)) if _indoor else 90.0
    uca_cfg.el_max_deg = min(uca_cfg.el_max_deg, _el_max)

    cfg_est = DoaEstimatorConfig(
        cfg             = uca_cfg,
        n_bursts        = int(getattr(C, "MULTI_BURST_N",                50)),
        phase_offsets   = list(getattr(C, "CHANNEL_PHASE_OFFSETS_DEG",   [0.0]*5)),
        az_offset_deg   = float(getattr(C, "DOA_AZ_OFFSET_DEG",          0.0)),
        el_offset_deg   = float(getattr(C, "DOA_EL_OFFSET_DEG",          0.0)),
        bpf_bw_hz       = float(getattr(C, "PREAMBLE_BPF_BW_HZ",         8_000.0)),
        bpf_guard       = int(getattr(C, "BPF_GUARD",                    128)),
        decorr          = str(getattr(C, "MUSIC_DECORR",                  "fb")),
        indoor          = _indoor,
        el_pref_lo      = float(getattr(C, "INDOOR_EL_PREF_MIN_DEG",     8.0)),
        el_pref_hi      = float(getattr(C, "INDOOR_EL_PREF_MAX_DEG",     35.0)),
        phase_score_w   = float(getattr(C, "INDOOR_PHASE_SCORE_WEIGHT",   0.35)),
        az_hint_w       = float(getattr(C, "INDOOR_AZ_HINT_SCORE_WEIGHT", 0.25)),
        el_hint_w       = float(getattr(C, "INDOOR_EL_HINT_SCORE_WEIGHT", 0.15)),
        mirror_margin   = float(getattr(C, "INDOOR_UCA_MIRROR_MARGIN_DEG", 10.0)),
        fs              = float(getattr(C, "SAMPLE_RATE_HZ",              _IRA_FS)),
        num_signals     = int(getattr(C, "NUM_SIGNALS",                   1)),
    )
    return DoaEstimator(cfg_est)
