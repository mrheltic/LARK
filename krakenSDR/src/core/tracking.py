"""
core.tracking
=============
Angular and scalar tracking filters for real-time Direction-of-Arrival estimation.

STATUS: not wired into the Iridium pipeline yet (run_doa.py uses a plain
inline EMA).  Used by ``apps/legacy/doa_test_868`` and covered by unit
tests; ``KalmanAngular`` is the natural upgrade for SNR-weighted smoothing
of the live az/el output if ever needed.

All classes are lightweight, stateful, and thread-unsafe by design — the caller
is responsible for locking when state is shared across threads.

Public API
----------
CircularEMA(alpha)           — phasor EMA for angular quantities (handles 0°/360° wrap)
ScalarEMA(alpha, init)       — standard IIR EMA for linear quantities
KalmanScalar(q, r, ...)      — 1D constant-model Kalman for linear quantities
KalmanAngular(q, r, ...)     — 1D Kalman on SO(2) phasor (handles angular wrap)

Typical usage
-------------
    from core.tracking import KalmanAngular, KalmanScalar, CircularEMA, ScalarEMA

    az_kf  = KalmanAngular(q=5.0, r=20.0)
    el_kf  = KalmanScalar(q=2.0,  r=8.0,  init=30.0)
    az_ema = CircularEMA(alpha=0.60)
    cfo    = ScalarEMA(alpha=0.05)

    for measurement_az, measurement_el, tone_hz in stream:
        smoothed_az  = az_kf.update(measurement_az)
        smoothed_el  = el_kf.update(measurement_el)
        display_az   = az_ema.update(measurement_az)
        cfo_estimate = cfo.update(tone_hz)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = [
    "CircularEMA",
    "ScalarEMA",
    "KalmanScalar",
    "KalmanAngular",
]


# =============================================================================
# CircularEMA
# =============================================================================

class CircularEMA:
    """
    Circular / phasor exponential moving average for angular quantities [degrees].

    Avoids the 0°/360° discontinuity by operating on unit-complex phasors
    (elements of the SO(2) Lie group).

    Parameters
    ----------
    alpha : float in (0, 1]
        Smoothing weight for the newest sample.
        τ (bursts) ≈ 1 / alpha.  alpha=0.15 → τ ≈ 6.7 bursts.
    """

    def __init__(self, alpha: float = 0.15) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        self._alpha = float(alpha)
        self._phasor: Optional[complex] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def is_initialised(self) -> bool:
        return self._phasor is not None

    @property
    def value_deg(self) -> Optional[float]:
        """Smoothed angle in [0, 360) degrees, or None before first update."""
        if self._phasor is None:
            return None
        return float(math.degrees(math.atan2(self._phasor.imag, self._phasor.real)) % 360)

    @property
    def value_rad(self) -> Optional[float]:
        """Smoothed angle in [-π, π) radians, or None before first update."""
        if self._phasor is None:
            return None
        return float(math.atan2(self._phasor.imag, self._phasor.real))

    @property
    def phasor(self) -> Optional[complex]:
        """Internal unit-complex phasor, or None before first update."""
        return self._phasor

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, angle_deg: float) -> float:
        """
        Incorporate a new angular measurement and return the smoothed value [deg].

        On the first call the state is seeded directly (no smoothing), avoiding
        the β-iteration burn-in from the zero-initialised phasor.

        Parameters
        ----------
        angle_deg : float
            New measurement in degrees (any range, not restricted to [0, 360)).

        Returns
        -------
        smoothed : float
            Current estimate in [0, 360) degrees.
        """
        ph = complex(
            math.cos(math.radians(angle_deg)),
            math.sin(math.radians(angle_deg)),
        )
        if self._phasor is None:
            self._phasor = ph
        else:
            self._phasor = (1.0 - self._alpha) * self._phasor + self._alpha * ph
        return self.value_deg  # type: ignore[return-value]

    def reset(self) -> None:
        """Clear state (next update will seed directly)."""
        self._phasor = None


# =============================================================================
# ScalarEMA
# =============================================================================

class ScalarEMA:
    """
    Standard exponential moving average (IIR lowpass) for scalar quantities.

    Parameters
    ----------
    alpha : float in (0, 1]
        Weight of the newest sample.  τ ≈ 1 / alpha.
    init : float | None
        Seed value.  None → cold-start (first update seeds directly).
    """

    def __init__(self, alpha: float = 0.30, init: Optional[float] = None) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        self._alpha = float(alpha)
        self._state: Optional[float] = None if init is None else float(init)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def is_initialised(self) -> bool:
        return self._state is not None

    @property
    def value(self) -> Optional[float]:
        return self._state

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, x: float) -> float:
        """
        Update with a new observation and return the smoothed value.

        Parameters
        ----------
        x : float  New measurement.

        Returns
        -------
        smoothed : float  Current EMA state.
        """
        if self._state is None:
            self._state = float(x)
        else:
            self._state = (1.0 - self._alpha) * self._state + self._alpha * float(x)
        return self._state  # type: ignore[return-value]

    def reset(self, init: Optional[float] = None) -> None:
        """Reset state to *init* (None → cold-start on next update)."""
        self._state = None if init is None else float(init)


# =============================================================================
# KalmanScalar
# =============================================================================

class KalmanScalar:
    """
    Scalar (1D) constant-model Kalman filter for linear quantities.

    State model  : x_{k+1} = x_k + w_k     (process noise w_k ~ N(0, Q))
    Measurement  : z_k     = x_k + v_k     (measurement noise v_k ~ N(0, R))

    Parameters
    ----------
    q        : float  Process noise variance.  Higher → faster tracking, more noise.
    r        : float  Measurement noise variance.  Higher → more smoothing.
    init     : float  Initial state estimate.  Overridden by first measurement.
    init_var : float  Initial error variance (high → trust first measurement quickly).
    """

    def __init__(
        self,
        q:        float = 2.0,
        r:        float = 8.0,
        init:     float = 0.0,
        init_var: float = 200.0,
    ) -> None:
        self.q = float(q)
        self.r = float(r)
        self._x = float(init)
        self._p = float(init_var)
        self._initialised = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> float:
        return self._x

    @property
    def variance(self) -> float:
        return self._p

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, z: float) -> float:
        """
        Process one measurement and return the updated state estimate.

        The first call seeds the state directly from *z* (no smoothing) to
        avoid the initial bias from the arbitrary prior.

        Returns
        -------
        state : float  Updated state estimate.
        """
        if not self._initialised:
            self._x = float(z)
            self._initialised = True
            return self._x

        # Predict
        p_pred = self._p + self.q

        # Update (Kalman gain)
        k      = p_pred / (p_pred + self.r)
        self._x = self._x + k * (float(z) - self._x)
        self._p = (1.0 - k) * p_pred
        return self._x

    def reset(self, init: float = 0.0, init_var: float = 200.0) -> None:
        """Reset to prior state."""
        self._x = float(init)
        self._p = float(init_var)
        self._initialised = False


# =============================================================================
# KalmanAngular
# =============================================================================

class KalmanAngular:
    """
    1D Kalman filter for angular (circular) quantities in degrees.

    Operates on SO(2) phasors to handle the 0°/360° wrap-around.
    The innovation is computed as the shortest-path angular difference so the
    filter does not diverge across the wrap boundary.

    Same state/measurement model as KalmanScalar, but the innovation uses:
        innov = ((z - x̂ + 180) % 360) - 180   [degrees]

    Parameters
    ----------
    q        : float  Process noise variance [deg²].
    r        : float  Measurement noise variance [deg²].
    init_var : float  Initial error variance [deg²].
    """

    def __init__(
        self,
        q:        float = 5.0,
        r:        float = 20.0,
        init_var: float = 200.0,
    ) -> None:
        self.q = float(q)
        self.r = float(r)
        self._x: Optional[float] = None   # state [deg], None → uninitialised
        self._p: float = float(init_var)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialised(self) -> bool:
        return self._x is not None

    @property
    def state_deg(self) -> Optional[float]:
        """Current state estimate [degrees in 0..360), or None before first update."""
        return self._x

    @property
    def variance(self) -> float:
        return self._p

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, z_deg: float) -> float:
        """
        Process one angular measurement [degrees] and return updated estimate.

        Returns
        -------
        state_deg : float  Current estimate in [0, 360).
        """
        z = float(z_deg)
        if self._x is None:
            self._x = z % 360.0
            return self._x

        # Predict
        p_pred = self._p + self.q

        # Innovation (shortest-path angular distance)
        innov  = ((z - self._x + 180.0) % 360.0) - 180.0

        # Update
        k      = p_pred / (p_pred + self.r)
        self._x = (self._x + k * innov) % 360.0
        self._p = (1.0 - k) * p_pred
        return self._x

    def reset(self, init_var: float = 200.0) -> None:
        """Clear state (next update will seed directly from measurement)."""
        self._x    = None
        self._p    = float(init_var)


# =============================================================================
# Vectorised batch helpers
# =============================================================================

def circular_ema_batch(
    angles_deg: np.ndarray,
    alpha:      float = 0.15,
) -> np.ndarray:
    """
    Compute CircularEMA over a sequence of angles without instantiating the class.

    Useful for offline analysis.

    Parameters
    ----------
    angles_deg : (N,) float array
    alpha      : smoothing weight

    Returns
    -------
    smoothed : (N,) float array  in [0, 360)
    """
    phasor = complex(
        math.cos(math.radians(float(angles_deg[0]))),
        math.sin(math.radians(float(angles_deg[0]))),
    )
    out = np.empty(len(angles_deg))
    out[0] = math.degrees(math.atan2(phasor.imag, phasor.real)) % 360.0
    for i in range(1, len(angles_deg)):
        ph = complex(
            math.cos(math.radians(float(angles_deg[i]))),
            math.sin(math.radians(float(angles_deg[i]))),
        )
        phasor = (1.0 - alpha) * phasor + alpha * ph
        out[i] = math.degrees(math.atan2(phasor.imag, phasor.real)) % 360.0
    return out


def scalar_ema_batch(
    values: np.ndarray,
    alpha:  float = 0.30,
) -> np.ndarray:
    """
    Compute ScalarEMA over a sequence of values without instantiating the class.

    Parameters
    ----------
    values : (N,) float array
    alpha  : smoothing weight

    Returns
    -------
    smoothed : (N,) float array
    """
    out      = np.empty(len(values))
    out[0]   = float(values[0])
    for i in range(1, len(values)):
        out[i] = (1.0 - alpha) * out[i - 1] + alpha * float(values[i])
    return out
