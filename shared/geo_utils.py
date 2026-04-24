"""
shared.geo_utils
================
Geometric utility functions for sky-coordinate calculations.

These functions are shared across multiple apps and tests.  Keeping them
here avoids copy-paste duplication and ensures a single authoritative
implementation.

Public API
----------
angular_distance_deg(az1, el1, az2, el2) → float
    Great-circle angular separation between two sky positions.
az_el_to_unit_vec(az_deg, el_deg) → (3,) ndarray
    Convert (az, el) to a Cartesian unit vector.
unit_vec_to_az_el(v) → (az_deg, el_deg)
    Inverse of az_el_to_unit_vec.
"""

from __future__ import annotations

import numpy as np


def az_el_to_unit_vec(az_deg: float, el_deg: float) -> np.ndarray:
    """
    Convert sky coordinates to a Cartesian unit vector.

    Convention (matched throughout the LARK codebase):
        azimuth  : degrees clockwise from North (0° = N, 90° = E)
        elevation: degrees above horizon (0° = horizon, 90° = zenith)

    Returns
    -------
    (3,) float64 — [x_east, y_north, z_up]
    """
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    cos_el = np.cos(el)
    return np.array([
        cos_el * np.sin(az),    # East component
        cos_el * np.cos(az),    # North component
        np.sin(el),             # Up component
    ], dtype=np.float64)


def unit_vec_to_az_el(v: np.ndarray) -> tuple[float, float]:
    """
    Convert a Cartesian unit vector to sky coordinates.

    Parameters
    ----------
    v : (3,) array — [x_east, y_north, z_up] (need not be unit length)

    Returns
    -------
    az_deg : float [0, 360)
    el_deg : float [−90, +90]
    """
    v = np.asarray(v, dtype=np.float64)
    v = v / (np.linalg.norm(v) + 1e-30)
    el_deg = float(np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    az_deg = float(np.rad2deg(np.arctan2(v[0], v[1])) % 360.0)
    return az_deg, el_deg


def angular_distance_deg(
    az1: float, el1: float,
    az2: float, el2: float,
) -> float:
    """
    Great-circle angular distance between two sky positions [degrees].

    Computes the angle between the two corresponding unit vectors on the
    unit sphere.  Result is in [0°, 180°].

    Parameters
    ----------
    az1, el1 : azimuth & elevation of position 1 [degrees]
    az2, el2 : azimuth & elevation of position 2 [degrees]

    Returns
    -------
    dist_deg : float — separation [degrees]

    Examples
    --------
    >>> angular_distance_deg(0, 90, 180, 90)   # both at zenith → 0°
    0.0
    >>> angular_distance_deg(0, 0, 90, 0)      # horizon, 90° apart → 90°
    90.0
    """
    v1 = az_el_to_unit_vec(az1, el1)
    v2 = az_el_to_unit_vec(az2, el2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))
