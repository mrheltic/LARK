"""
core.gates
==========
Composable burst-acceptance gates for preamble-gated DoA estimation.

STATUS: not used by the Iridium pipeline — kept only for the older 868 MHz
experiments in ``apps/legacy/doa_test_868``.  Remove together with them.

Each gate is a lightweight callable that checks one quality criterion and
returns a ``GateVerdict`` dataclass.  Gates are stateless (PAGGate, EigenGate,
BoundaryGate) or carry minimal state (OutlierGate maintains its own az history).
Compose them with ``GatePipeline``.

Public API
----------
GateVerdict          — frozen dataclass: (accepted, el_clamped, reason)
PAGGate(threshold_db)          — PAPR minimum gate (no state)
EigenGate(min_spread, max_dom) — eigenvalue gates (no state)
BoundaryGate(el_min, el_max, el_step) — floor/ceiling elevation clamp (no state)
OutlierGate(max_dev_deg, ...)  — circular-median az outlier gate (stateful)
GatePipeline([gates])          — sequential pipeline, stops at first rejection

circ_median_deg(angles_deg)    — circular median of azimuth values in [0, 360°)

Gate protocol
-------------
    gate.check(**kwargs) → GateVerdict

All built-in gates use **kwargs so GatePipeline can pass a superset of keyword
arguments to every gate without each gate needing to know the other gates' fields:

    verdict = pipeline.check(papr_db=9.3, az_inst=120.0, el_inst=45.0, eig=eig_arr)

The pipeline stops on the first rejection; otherwise returns the last verdict
(which propagates ``el_clamped=True`` from any BoundaryGate that was triggered).

Quick usage
-----------
    from core.gates import GatePipeline, PAGGate, BoundaryGate, OutlierGate

    pipeline = GatePipeline([
        PAGGate(threshold_db=8.0),
        BoundaryGate(el_min=5.0, el_max=65.0, el_step=1.0),
        OutlierGate(max_dev_deg=45.0, min_history=5),
    ])

    v = pipeline.check(papr_db=9.3, az_inst=120.0, el_inst=45.0, eig=eig_arr)
    if v.accepted:
        if not v.el_clamped:
            el_ema.update(el_inst)
        az_kf.update(az_inst)
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "GateVerdict",
    "circ_median_deg",
    "PAGGate",
    "EigenGate",
    "BoundaryGate",
    "OutlierGate",
    "GatePipeline",
]


# =============================================================================
# GateVerdict
# =============================================================================

@dataclass(frozen=True)
class GateVerdict:
    """
    Immutable result returned by every gate.

    Attributes
    ----------
    accepted   : bool   True when the burst passes the gate criterion.
    el_clamped : bool   True when elevation hit the floor/ceiling boundary.
                        Az estimate is still considered reliable; only el update
                        should be skipped.  Always False for non-BoundaryGates.
    reason     : str    Human-readable description on rejection.
    """
    accepted:   bool
    el_clamped: bool = False
    reason:     str  = ""


# =============================================================================
# Gate protocol
# =============================================================================

@runtime_checkable
class Gate(Protocol):
    """Structural Protocol for gate objects."""
    def check(self, **kwargs) -> GateVerdict: ...  # noqa: E704


# =============================================================================
# Circular median helper
# =============================================================================

def circ_median_deg(angles_deg: np.ndarray) -> float:
    """
    Circular median of azimuth values in degrees.

    Computes the circular mean first to robustly handle the 0°/360° wrap,
    then takes the median of the centred deviations.

    Parameters
    ----------
    angles_deg : (N,) float array.  Values can be outside [0, 360).

    Returns
    -------
    median_deg : float in [0, 360).  Returns 0.0 for empty input.
    """
    if len(angles_deg) == 0:
        return 0.0
    a        = np.deg2rad(np.asarray(angles_deg, dtype=float))
    mean_ang = np.angle(np.mean(np.exp(1j * a)))
    centred  = np.degrees(np.angle(np.exp(1j * (a - mean_ang))))
    return float((np.median(centred) + np.degrees(mean_ang)) % 360)


# =============================================================================
# PAGGate — PAPR threshold
# =============================================================================

class PAGGate:
    """
    PAPR threshold gate: reject bursts where the MUSIC PAPR is too low.

    A genuine preamble window (rank-1 after pilot BPF) yields a high PAPR even
    with hardware phase errors.  Wideband data / noise windows give near-flat
    MUSIC spectra → low PAPR → rejected.

    Parameters
    ----------
    threshold_db : float
        Minimum PAPR [dB] to accept.  Typical calibrated array: ≥ 20 dB.
        With ±20° hardware phase errors: lower to ≥ 12 dB.
    """

    def __init__(self, threshold_db: float = 8.0) -> None:
        self.threshold_db = threshold_db

    def check(self, *, papr_db: float, **_) -> GateVerdict:
        if papr_db >= self.threshold_db:
            return GateVerdict(accepted=True)
        return GateVerdict(
            accepted=False,
            reason=f"PAPR {papr_db:.1f} dB < threshold {self.threshold_db:.1f} dB",
        )


# =============================================================================
# EigenGate — eigenvalue spread + ADC saturation advisory
# =============================================================================

class EigenGate:
    """
    Eigenvalue gates: minimum spread (noise rejection) + maximum dominant
    eigenvalue (ADC saturation advisory).

    Parameters
    ----------
    min_spread_db   : float  Minimum eigenvalue spread λ₁ − λ_N [dB].
                             Burst rejected when spread is too low (pure noise).
    max_dominant_db : float  Warning threshold for λ₁ [dB].
                             If exceeded the gate returns a rejection with
                             an "ADC saturation" reason.  Set to ``inf`` to
                             disable.
    min_sn_gap_db   : float  Minimum λ₂ − λ₃ gap [dB].  Enforces that the
                             second eigenvalue is clearly distinguishable from
                             the noise floor.  Ignored when set to 0.0.
    """

    def __init__(
        self,
        min_spread_db:   float = 0.5,
        max_dominant_db: float = float("inf"),
        min_sn_gap_db:   float = 0.0,
    ) -> None:
        self.min_spread_db   = min_spread_db
        self.max_dominant_db = max_dominant_db
        self.min_sn_gap_db   = min_sn_gap_db

    def check(self, *, eig: np.ndarray, **_) -> GateVerdict:
        if len(eig) < 2:
            return GateVerdict(accepted=True)

        spread = float(eig[0] - eig[-1])
        if spread < self.min_spread_db:
            return GateVerdict(
                accepted=False,
                reason=f"eigenvalue spread {spread:.1f} dB < {self.min_spread_db:.1f} dB",
            )

        dominant = float(eig[0])
        if dominant > self.max_dominant_db:
            return GateVerdict(
                accepted=False,
                reason=f"λ₁={dominant:.1f} dB > {self.max_dominant_db:.1f} dB (ADC saturation)",
            )

        if self.min_sn_gap_db > 0.0 and len(eig) >= 3:
            sn_gap = float(eig[1] - eig[2])
            if sn_gap < self.min_sn_gap_db:
                return GateVerdict(
                    accepted=False,
                    reason=f"S/N gap {sn_gap:.1f} dB < {self.min_sn_gap_db:.1f} dB",
                )

        return GateVerdict(accepted=True)


# =============================================================================
# BoundaryGate — floor / ceiling elevation clamp
# =============================================================================

class BoundaryGate:
    """
    Elevation boundary clamp gate.

    When ``el_inst`` lands at the floor or ceiling of the scan grid, the 2D
    joint peak may slide along the coupling ridge to a boundary cell.  In this
    case the az estimate from the 1D marginal is still reliable, but el is
    not.

    The gate therefore ACCEPTS the burst but sets ``el_clamped=True`` so the
    caller can skip the elevation EMA / Kalman update.

    This matches the behaviour for both floor and ceiling:
    - Floor: low-elevation direct path at SNR > 0 dB DOES land at grid boundary.
      The pure PAPR gate already confirmed the preamble is genuine.
    - Ceiling: el_inst ≥ el_max − step/2 → ridge artefact, skip el update.

    Parameters
    ----------
    el_min          : float  Lower elevation grid limit [degrees].
    el_max          : float  Upper elevation grid limit [degrees].
    el_step         : float  Elevation grid step = (el_max − el_min) / (n_el − 1).
    margin_fraction : float  Fraction of el_step used as boundary margin.
                             Default 0.5 → triggers within ±step/2 of limits.
    """

    def __init__(
        self,
        el_min:          float,
        el_max:          float,
        el_step:         float,
        margin_fraction: float = 0.5,
    ) -> None:
        self.el_min  = el_min
        self.el_max  = el_max
        self._lo_lim = el_min + el_step * margin_fraction
        self._hi_lim = el_max - el_step * margin_fraction

    def check(self, *, el_inst: float, **_) -> GateVerdict:
        if el_inst <= self._lo_lim or el_inst >= self._hi_lim:
            return GateVerdict(accepted=True, el_clamped=True)
        return GateVerdict(accepted=True)


# =============================================================================
# OutlierGate — circular-median azimuth outlier rejection
# =============================================================================

class OutlierGate:
    """
    Circular-median outlier rejection for azimuth estimates.

    Maintains an internal deque of recently accepted az values.  When a new
    az estimate deviates more than ``max_dev_deg`` from the running circular
    median, the burst is rejected.

    After ``reset_after`` consecutive rejections the gate force-resets,
    assuming the array has physically rotated to a new bearing.  The
    ``force_reseed`` property returns True (once) when this happens so the
    caller can cold-start its tracking filters.

    Parameters
    ----------
    max_dev_deg  : float  Maximum allowed deviation from circular median [deg].
    min_history  : int    Minimum history length before the gate activates.
                          Short histories are unreliable for median estimation.
    reset_after  : int    Consecutive rejections before force-reset.
    history_len  : int    Maximum deque length.
    """

    def __init__(
        self,
        max_dev_deg:  float = 45.0,
        min_history:  int   = 5,
        reset_after:  int   = 3,
        history_len:  int   = 60,
    ) -> None:
        self.max_dev_deg  = max_dev_deg
        self.min_history  = min_history
        self.reset_after  = reset_after
        self._history: collections.deque[float] = collections.deque(maxlen=history_len)
        self._streak      = 0
        self._force_reseed_flag = False

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def force_reseed(self) -> bool:
        """
        True once when the gate force-resets after ``reset_after`` consecutive
        rejections.  Reading this property clears the flag (one-shot).
        """
        v = self._force_reseed_flag
        self._force_reseed_flag = False
        return v

    # ------------------------------------------------------------------
    # Accept / check
    # ------------------------------------------------------------------

    def accept(self, az_deg: float) -> None:
        """Record an accepted azimuth estimate into the internal history."""
        self._history.append(float(az_deg))
        self._streak = 0

    def check(self, *, az_inst: float, **_) -> GateVerdict:
        if len(self._history) < self.min_history:
            return GateVerdict(accepted=True)

        med  = circ_median_deg(np.array(self._history))
        dev  = float(abs(((az_inst - med + 180.0) % 360.0) - 180.0))
        if dev <= self.max_dev_deg:
            return GateVerdict(accepted=True)

        self._streak += 1
        if self._streak >= self.reset_after:
            self._streak           = 0
            self._force_reseed_flag = True
        return GateVerdict(
            accepted=False,
            reason=f"az outlier: deviation {dev:.1f}° > {self.max_dev_deg:.1f}°",
        )

    def reset(self) -> None:
        """Clear all state."""
        self._history.clear()
        self._streak            = 0
        self._force_reseed_flag = False


# =============================================================================
# GatePipeline
# =============================================================================

class GatePipeline:
    """
    Compose multiple gates into a sequential evaluation pipeline.

    Gates are evaluated in declaration order.  The pipeline stops on the first
    rejection and returns that verdict.  If all gates pass, the last verdict is
    returned — which may carry ``el_clamped=True`` if any BoundaryGate triggered.

    Parameters
    ----------
    gates : list of Gate
        Any objects implementing the Gate protocol
        (i.e. ``gate.check(**kwargs) → GateVerdict``).

    Usage
    -----
        pipeline = GatePipeline([
            PAGGate(8.0),
            BoundaryGate(el_min=5.0, el_max=65.0, el_step=1.0),
            OutlierGate(45.0, min_history=5),
        ])
        v = pipeline.check(papr_db=9.3, az_inst=120.0, el_inst=45.0, eig=eig_arr)
        if v.accepted:
            if not v.el_clamped:
                el_filter.update(el_inst)
    """

    def __init__(self, gates: list) -> None:
        self._gates = list(gates)

    @property
    def gates(self) -> list:
        return self._gates

    def check(self, **kwargs) -> GateVerdict:
        last = GateVerdict(accepted=True)
        for gate in self._gates:
            v = gate.check(**kwargs)
            if not v.accepted:
                return v
            # Propagate el_clamped: if ANY gate triggers it, the pipeline
            # returns it even though subsequent gates pass.
            if v.el_clamped:
                last = v
        return last
