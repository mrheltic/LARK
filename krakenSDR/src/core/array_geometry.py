"""
core.array_geometry — Base array geometry definitions and steering vectors
===========================================================================

Unified array geometry abstractions for KrakenSDR DoA processing.
Provides base classes and steering vector computations shared across
ULA, UCA 2D, and cross array configurations.

Design Principles
-----------------
- Immutable geometry configurations (dataclasses)
- Cached steering matrices for repeated DoA scans
- Coordinate system: East-North-Up (ENU) with wavelengths as unit
- Phase convention: exp(j·τ) where τ = 2π/λ · (p · û)

Classes
-------
ArrayGeometryBase     : Abstract base for all array types
UniformLinearArray    : ULA with d/λ spacing
UniformCircularArray  : UCA with radius/λ in East-North plane
CrossArray            : 5-element "+" array (center + 4 arms)

Functions
---------
steering_vector_1d    : 1D azimuth-only steering (ULA/UCA 1D)
steering_vector_2d    : 2D azimuth-elevation steering
compute_steering_matrix : Batch steering matrix for grid search
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class GeometryType(Enum):
    """Enumeration of supported array geometries."""
    ULA = "ULA"       # Uniform Linear Array
    UCA = "UCA"       # Uniform Circular Array
    CROSS = "CROSS"   # 5-element cross array


@dataclass(frozen=True)
class ArrayGeometryBase(ABC):
    """
    Abstract base class for array geometries.
    
    All geometries define:
    - n_ant: number of antenna elements
    - positions: (n_ant, 3) array of element positions in wavelengths [E, N, U]
    """
    n_ant: int
    
    # Cache for expensive computations
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    
    @property
    @abstractmethod
    def positions(self) -> np.ndarray:
        """Element positions in wavelengths (n_ant, 3) [East, North, Up]."""
        ...
    
    @property
    def positions_2d(self) -> np.ndarray:
        """Element positions in East-North plane (n_ant, 2)."""
        return self.positions[:, :2]
    
    def clear_cache(self) -> None:
        """Invalidate cached computations."""
        self._cache.clear()


@dataclass(frozen=True)
class UniformLinearArray(ArrayGeometryBase):
    """
    Uniform Linear Array (ULA) geometry.
    
    Elements placed along x-axis (East direction) with uniform spacing.
    """
    d_lambda: float = 0.5  # Inter-element spacing in wavelengths
    
    def __post_init__(self):
        object.__setattr__(self, 'n_ant', self.n_ant)
    
    @property
    def positions(self) -> np.ndarray:
        """ULA positions: equally spaced along East axis."""
        key = ('ula_pos', self.n_ant, round(self.d_lambda, 9))
        if key not in self._cache:
            pos = np.zeros((self.n_ant, 3))
            pos[:, 0] = np.arange(self.n_ant) * self.d_lambda  # East
            self._cache[key] = pos
        return self._cache[key]
    
    def steering_1d(self, theta_rad: float) -> np.ndarray:
        """
        1D steering vector for azimuth angle theta.
        
        Parameters
        ----------
        theta_rad : float
            Azimuth angle in radians (0 = broadside/perpendicular to array)
        
        Returns
        -------
        a : np.ndarray
            Steering vector (n_ant,) complex
        """
        k = np.arange(self.n_ant)
        return np.exp(2j * np.pi * self.d_lambda * k * np.sin(theta_rad))


@dataclass(frozen=True)
class UniformCircularArray(ArrayGeometryBase):
    """
    Uniform Circular Array (UCA) geometry in East-North plane.
    
    Elements placed on a circle of radius r/λ in the East-North plane.
    Antenna 0 is at angle φ=0 (North by default, configurable via offset).
    """
    radius_lambda: float = 0.5    # Circle radius in wavelengths
    ant0_offset_deg: float = 0.0  # Rotation of ant0 from North
    ant_ccw: bool = False         # True if antennas are CCW (vs default CW)
    
    def __post_init__(self):
        object.__setattr__(self, 'n_ant', self.n_ant)
    
    @property
    def positions(self) -> np.ndarray:
        """UCA positions: on circle in East-North plane."""
        key = ('uca_pos', self.n_ant, round(self.radius_lambda, 9),
               round(self.ant0_offset_deg, 4), self.ant_ccw)
        if key not in self._cache:
            k = np.arange(self.n_ant, dtype=np.float64)
            sign = -1.0 if self.ant_ccw else 1.0
            phi_k = np.deg2rad(self.ant0_offset_deg) + sign * 2.0 * np.pi * k / self.n_ant
            
            pos = np.zeros((self.n_ant, 3))
            pos[:, 0] = self.radius_lambda * np.sin(phi_k)  # East
            pos[:, 1] = self.radius_lambda * np.cos(phi_k)  # North
            self._cache[key] = pos
        return self._cache[key]
    
    def element_angle(self, k: int) -> float:
        """Angular position of element k in radians from North."""
        sign = -1.0 if self.ant_ccw else 1.0
        return np.deg2rad(self.ant0_offset_deg) + sign * 2.0 * np.pi * k / self.n_ant
    
    def steering_1d(self, theta_rad: float) -> np.ndarray:
        """
        1D steering vector for azimuth angle theta (elevation = 0).
        
        Parameters
        ----------
        theta_rad : float
            Azimuth angle in radians from North, clockwise
        
        Returns
        -------
        a : np.ndarray
            Steering vector (n_ant,) complex
        """
        k = np.arange(self.n_ant)
        phi_k = self.element_angle(k)
        return np.exp(2j * np.pi * self.radius_lambda * np.cos(theta_rad - phi_k))
    
    def steering_2d(self, az_rad: float, el_rad: float) -> np.ndarray:
        """
        2D steering vector for (azimuth, elevation).
        
        Parameters
        ----------
        az_rad : float
            Azimuth from North, clockwise [rad]
        el_rad : float
            Elevation above horizon [rad]
        
        Returns
        -------
        a : np.ndarray
            Steering vector (n_ant,) complex
        """
        # Direction cosines in East-North plane
        u_east = np.cos(el_rad) * np.sin(az_rad)
        u_north = np.cos(el_rad) * np.cos(az_rad)
        
        # Phase delays: 2π · (p_E · u_E + p_N · u_N)
        pos = self.positions_2d
        tau = 2.0 * np.pi * (pos[:, 0] * u_east + pos[:, 1] * u_north)
        return np.exp(1j * tau)


@dataclass(frozen=True)
class CrossArray(ArrayGeometryBase):
    """
    5-element cross ("+") array geometry.
    
    Canonical order: [center, east, north, west, south]
    Physical wiring may differ; use reordering utilities.
    """
    d_lambda: float = 0.5  # Arm length from center to each element
    
    def __post_init__(self):
        object.__setattr__(self, 'n_ant', 5)
    
    @property
    def positions(self) -> np.ndarray:
        """Cross array positions: center + 4 cardinal directions."""
        key = ('cross_pos', round(self.d_lambda, 9))
        if key not in self._cache:
            pos = np.array([
                [0.0, 0.0, 0.0],           # center
                [self.d_lambda, 0.0, 0.0], # east
                [0.0, self.d_lambda, 0.0], # north
                [-self.d_lambda, 0.0, 0.0], # west
                [0.0, -self.d_lambda, 0.0], # south
            ])
            self._cache[key] = pos
        return self._cache[key]
    
    def steering_2d(self, az_rad: float, el_rad: float) -> np.ndarray:
        """
        2D steering vector for (azimuth, elevation).
        
        Parameters
        ----------
        az_rad : float
            Azimuth from North, clockwise [rad]
        el_rad : float
            Elevation above horizon [rad]
        
        Returns
        -------
        a : np.ndarray
            Steering vector (5,) complex in canonical order
        """
        # Direction cosines
        u_east = np.cos(el_rad) * np.sin(az_rad)
        u_north = np.cos(el_rad) * np.cos(az_rad)
        
        # Phase delays
        pos = self.positions_2d
        tau = 2.0 * np.pi * (pos[:, 0] * u_east + pos[:, 1] * u_north)
        return np.exp(1j * tau)


# =============================================================================
# Unified steering matrix computation
# =============================================================================

def compute_steering_matrix_2d(
    geometry: ArrayGeometryBase,
    az_range: np.ndarray,
    el_range: np.ndarray,
) -> np.ndarray:
    """
    Compute full 2D steering matrix for grid search.
    
    Parameters
    ----------
    geometry : ArrayGeometryBase
        Array geometry (UCA or CrossArray)
    az_range : np.ndarray
        Azimuth grid points [rad]
    el_range : np.ndarray
        Elevation grid points [rad]
    
    Returns
    -------
    A : np.ndarray
        Steering matrix (n_ant, n_el * n_az) complex
    """
    AZ, EL = np.meshgrid(az_range, el_range)
    n_grid = AZ.size
    
    if isinstance(geometry, UniformCircularArray):
        # Vectorized UCA steering
        u_east = np.cos(EL).ravel() * np.sin(AZ).ravel()
        u_north = np.cos(EL).ravel() * np.cos(AZ).ravel()
        pos = geometry.positions_2d
        tau = 2.0 * np.pi * (
            pos[:, 0:1] * u_east[np.newaxis, :]
            + pos[:, 1:2] * u_north[np.newaxis, :]
        )
        return np.exp(1j * tau).astype(np.complex128)
    
    elif isinstance(geometry, CrossArray):
        # Vectorized cross array steering
        u_east = np.cos(EL).ravel() * np.sin(AZ).ravel()
        u_north = np.cos(EL).ravel() * np.cos(AZ).ravel()
        pos = geometry.positions_2d
        tau = 2.0 * np.pi * (
            pos[:, 0:1] * u_east[np.newaxis, :]
            + pos[:, 1:2] * u_north[np.newaxis, :]
        )
        return np.exp(1j * tau).astype(np.complex128)
    
    else:
        raise TypeError(f"2D steering not implemented for {type(geometry)}")


def compute_steering_matrix_1d(
    geometry: ArrayGeometryBase,
    theta_range: np.ndarray,
) -> np.ndarray:
    """
    Compute 1D steering matrix for azimuth-only search.
    
    Parameters
    ----------
    geometry : ArrayGeometryBase
        Array geometry (ULA or UCA)
    theta_range : np.ndarray
        Azimuth angles [rad]
    
    Returns
    -------
    A : np.ndarray
        Steering matrix (n_ant, n_points) complex
    """
    if isinstance(geometry, UniformLinearArray):
        k = np.arange(geometry.n_ant)[:, np.newaxis]
        return np.exp(
            2j * np.pi * geometry.d_lambda 
            * k * np.sin(theta_range[np.newaxis, :])
        )
    
    elif isinstance(geometry, UniformCircularArray):
        k = np.arange(geometry.n_ant)[:, np.newaxis]
        phi_k = np.array([geometry.element_angle(i) for i in range(geometry.n_ant)])
        return np.exp(
            2j * np.pi * geometry.radius_lambda 
            * np.cos(theta_range[np.newaxis, :] - phi_k[:, np.newaxis])
        )
    
    else:
        raise TypeError(f"1D steering not implemented for {type(geometry)}")


# =============================================================================
# Channel reordering utilities
# =============================================================================

CROSS_ARRAY_CANONICAL_ORDER: tuple[str, ...] = (
    "center", "east", "north", "west", "south"
)

_CROSS_ALIASES = {
    "c": "center", "ctr": "center", "center": "center", "centre": "center",
    "e": "east", "east": "east",
    "n": "north", "north": "north",
    "s": "south", "south": "south",
    "w": "west", "west": "west",
}


def normalize_channel_order(
    order: list[str] | tuple[str, ...],
    valid_labels: tuple[str, ...] = CROSS_ARRAY_CANONICAL_ORDER,
) -> tuple[str, ...]:
    """
    Validate and normalize channel order description.
    
    Parameters
    ----------
    order : sequence of str
        Channel labels (may be aliases)
    valid_labels : tuple[str, ...]
        Valid canonical labels
    
    Returns
    -------
    normalized : tuple[str, ...]
        Canonical labels
    """
    if len(order) != len(valid_labels):
        raise ValueError(
            f"Order must contain {len(valid_labels)} labels, got {len(order)}"
        )
    
    normalized = []
    for lbl in order:
        canonical = _CROSS_ALIASES.get(lbl.lower(), lbl.lower())
        if canonical not in valid_labels:
            raise ValueError(f"Unknown channel label: {lbl!r}")
        normalized.append(canonical)
    
    # Check all required labels present
    if set(normalized) != set(valid_labels):
        missing = set(valid_labels) - set(normalized)
        raise ValueError(f"Missing channels: {missing}")
    
    return tuple(normalized)


def reorder_channels(
    x: np.ndarray,
    input_order: tuple[str, ...],
    output_order: tuple[str, ...] = CROSS_ARRAY_CANONICAL_ORDER,
) -> np.ndarray:
    """
    Reorder array channels from input wiring to canonical order.
    
    Parameters
    ----------
    x : np.ndarray
        Array data with channels in first dimension (n_ant, ...)
    input_order : tuple[str, ...]
        Current channel order
    output_order : tuple[str, ...]
        Desired channel order (default: canonical)
    
    Returns
    -------
    x_reordered : np.ndarray
        Data with channels in output_order
    """
    input_order = normalize_channel_order(input_order)
    output_order = normalize_channel_order(output_order)
    
    perm = [input_order.index(ch) for ch in output_order]
    return x[perm, ...]
