"""
core.signal_quality
===================
Signal-quality and array-health metrics for the 5-element KrakenSDR cross array.

All functions accept either raw IQ frames or the 5×5 spatial covariance matrix.
They are purely computational (no I/O, no side effects) and are safe to call
from any thread.

Public API
----------
channel_power_balance(X)         → ChannelPowerReport
eigenvalue_spread_db(R)          → (5,) ndarray [dB]
snr_from_covariance(R)           → float [dB]
coherence_matrix(R)              → (5, 5) float [0..1]
papr_db_from_spectrum(spec)      → float [dB]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# Re-export quality metrics defined in doa_algorithms_3d to provide a single
# import path for callers that want *all* quality functions together.
from .doa_algorithms_3d import (           # noqa: F401
    eigenvalue_spread_db,
    snr_from_covariance,
    coherence_matrix,
    CROSS_ARRAY_CANONICAL_ORDER,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fraction of peak power below which a channel is flagged as weak.
#: 11 % matches real hardware observations (≈ −11 dB below strongest channel).
WEAK_CHANNEL_THRESHOLD = 0.30

#: Fraction below which a channel is flagged as critically degraded.
CRITICAL_CHANNEL_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Channel-power health report
# ---------------------------------------------------------------------------

@dataclass
class ChannelPowerReport:
    """
    Per-channel power balance report for a 5-element cross array.

    Attributes
    ----------
    powers_linear : (5,) float64 — mean |x|² per channel (un-normalised)
    powers_norm   : (5,) float64 — channel powers normalised to peak [0..1]
    channel_order : tuple[str,...] — label for each channel index
    weak_mask     : (5,) bool — True where power < WEAK_CHANNEL_THRESHOLD
    critical_mask : (5,) bool — True where power < CRITICAL_CHANNEL_THRESHOLD
    imbalance_db  : float — peak-to-min ratio [dB]; < 6 dB is well balanced

    Properties
    ----------
    is_balanced       : True when all channels are above WEAK_CHANNEL_THRESHOLD
    has_critical      : True when any channel is CRITICAL
    weak_labels       : list of channel label strings that are weak
    critical_labels   : list of channel label strings that are critical
    """
    powers_linear: np.ndarray
    powers_norm:   np.ndarray
    channel_order: tuple
    weak_mask:     np.ndarray
    critical_mask: np.ndarray
    imbalance_db:  float

    @property
    def is_balanced(self) -> bool:
        return bool(np.all(~self.weak_mask))

    @property
    def has_critical(self) -> bool:
        return bool(np.any(self.critical_mask))

    @property
    def weak_labels(self) -> list[str]:
        return [self.channel_order[i] for i in range(len(self.channel_order))
                if self.weak_mask[i]]

    @property
    def critical_labels(self) -> list[str]:
        return [self.channel_order[i] for i in range(len(self.channel_order))
                if self.critical_mask[i]]

    def summary_line(self) -> str:
        """Single-line human-readable summary for logging / UI."""
        parts = []
        for i, (lbl, p) in enumerate(zip(self.channel_order, self.powers_norm)):
            flag = ""
            if self.critical_mask[i]:
                flag = "⚠CRIT"
            elif self.weak_mask[i]:
                flag = "⚠"
            parts.append(f"{lbl}:{p*100:.0f}%{flag}")
        status = "OK" if self.is_balanced else ("CRITICAL" if self.has_critical else "WEAK")
        return f"[{status}] " + "  ".join(parts) + f"  imbal={self.imbalance_db:.1f}dB"

    def console_log(self) -> None:
        """Print a colour-annotated power bar to stdout."""
        print("── Channel power balance ─────────────────────────────────")
        for i, (lbl, p) in enumerate(zip(self.channel_order, self.powers_norm)):
            bar_len = int(p * 20)
            bar = "#" * bar_len + "·" * (20 - bar_len)
            pct = p * 100.0
            tag = ""
            if self.critical_mask[i]:
                tag = "  ⚠ CRITICAL — check SMA connector"
            elif self.weak_mask[i]:
                tag = "  ⚠ weak"
            print(f"  {lbl:6s} [{bar}] {pct:5.1f}%{tag}")
        print(f"  imbalance: {self.imbalance_db:.1f} dB"
              f"  ({'OK' if self.is_balanced else 'DEGRADED'})")
        print("──────────────────────────────────────────────────────────")


def channel_power_balance(
    X:             np.ndarray,
    channel_order: Sequence[str] | None = None,
) -> ChannelPowerReport:
    """
    Compute per-channel mean power and flag weak / critical channels.

    Parameters
    ----------
    X             : (N_ch, N_samples) complex  — multi-channel IQ data.
                    Accepts a single frame, a burst window, or concatenated
                    burst data (all shapes (N_ch, …) treated uniformly).
    channel_order : optional list of channel labels, length N_ch.
                    Defaults to CROSS_ARRAY_CANONICAL_ORDER when N_ch == 5.

    Returns
    -------
    ChannelPowerReport

    Notes
    -----
    The IQ matrix can be in any channel ordering (physical or canonical) —
    only relative powers matter here.  The caller is responsible for passing
    the correct ``channel_order`` to get meaningful labels.

    Examples
    --------
    >>> rpt = channel_power_balance(bursts.reshape(5, -1))
    >>> rpt.console_log()
    >>> if rpt.has_critical:
    ...     raise RuntimeError(f"Critical channels: {rpt.critical_labels}")
    """
    X = np.asarray(X)
    n_ch = X.shape[0]

    if channel_order is None:
        if n_ch == 5:
            channel_order = CROSS_ARRAY_CANONICAL_ORDER
        else:
            channel_order = tuple(f"CH{i}" for i in range(n_ch))
    else:
        channel_order = tuple(channel_order)

    if len(channel_order) != n_ch:
        raise ValueError(
            f"channel_order has {len(channel_order)} labels but X has {n_ch} channels"
        )

    # Mean power per channel over all remaining dimensions
    flat = X.reshape(n_ch, -1)
    powers_linear = np.mean(np.abs(flat) ** 2, axis=1).astype(np.float64)

    peak = float(np.max(powers_linear)) + 1e-30
    powers_norm = powers_linear / peak

    weak_mask     = powers_norm < WEAK_CHANNEL_THRESHOLD
    critical_mask = powers_norm < CRITICAL_CHANNEL_THRESHOLD

    pmin = float(np.min(powers_linear)) + 1e-30
    imbalance_db = 10.0 * np.log10(peak / pmin)

    return ChannelPowerReport(
        powers_linear=powers_linear,
        powers_norm=powers_norm,
        channel_order=channel_order,
        weak_mask=weak_mask,
        critical_mask=critical_mask,
        imbalance_db=imbalance_db,
    )


# ---------------------------------------------------------------------------
# Spectrum quality
# ---------------------------------------------------------------------------

def papr_db_from_spectrum(spec: np.ndarray) -> float:
    """
    Peak-to-average power ratio of a 2D (or 1D) MUSIC/Capon spectrum.

    The spectrum is expected in dB with peak = 0 (standard output of
    doa_music_2d / doa_capon_2d / doa_iaa_2d).  Convert back to linear
    for the power ratio, so that a narrow sharp peak gives a high PAPR.

    Parameters
    ----------
    spec : (...) float ndarray — spectrum in dB

    Returns
    -------
    papr_db : float — PAPR in dB; 0 = flat (no peak); > 10 dB = clear peak
    """
    s_lin  = 10.0 ** (np.clip(spec, -200.0, 0.0) / 10.0)
    mean_v = float(np.mean(s_lin))
    if mean_v < 1e-15:
        return 0.0
    return float(10.0 * np.log10(float(np.max(s_lin)) / mean_v))


# ---------------------------------------------------------------------------
# Convenience: check recording health before analysis
# ---------------------------------------------------------------------------

def check_recording_health(
    bursts: np.ndarray,
    input_order: Sequence[str] | None = None,
    *,
    warn_only: bool = True,
) -> ChannelPowerReport:
    """
    Compute power balance across all bursts in a recording and optionally warn.

    Parameters
    ----------
    bursts      : (N_bursts, N_ch, N_samples) complex
    input_order : channel label sequence (length N_ch)
    warn_only   : if False, raise RuntimeError when critical channels found

    Returns
    -------
    ChannelPowerReport — aggregated over all bursts
    """
    n_bursts, n_ch = bursts.shape[0], bursts.shape[1]
    flat = bursts.reshape(n_bursts, n_ch, -1)
    # Stack all burst samples per channel
    combined = flat.transpose(1, 0, 2).reshape(n_ch, -1)
    rpt = channel_power_balance(combined, channel_order=input_order)
    rpt.console_log()
    if rpt.has_critical and not warn_only:
        raise RuntimeError(
            f"Critical hardware degradation detected on channels: "
            f"{rpt.critical_labels}. "
            "Reseat the SMA connectors before running DoA analysis."
        )
    return rpt
