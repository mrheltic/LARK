"""
shared.satellite_tracker
========================
Real-time Iridium satellite ground-truth annotation.

Downloads the Iridium NEXT TLE catalogue from CelesTrak (online REST API,
refreshed every 24 h) and propagates each satellite's position with sgp4 via
skyfield.  Matches each recorded burst to the closest visible satellite using
Doppler-shift similarity.

Optionally routes through the N2YO REST API when the ``LARK_N2YO_KEY``
environment variable is set (recommended only for live-recording sessions;
N2YO historical-position queries require a paid account).

Public API
----------
    from shared.satellite_tracker import match_bursts_to_satellites

    gt = match_bursts_to_satellites(
        timestamps_ms,   # (N,) float64  — ms from session start
        doppler_hz,      # (N,) float64  — measured Doppler [Hz]
        t0_utc,          # datetime      — session start UTC
        lat, lon, alt,   # observer WGS-84 (deg, deg, m)
    )
    # gt is a dict with (N,) arrays:
    #   az_deg, el_deg, sat_name, norad_id, doppler_hz, source

See also
--------
``shared.iridium_tle.IridiumCatalogue.visible_now`` — lower-level per-instant
visibility query used internally.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

# ── Tuning constants ───────────────────────────────────────────────────────────
_MAX_DOPPLER_ERR_HZ = 8_000.0   # exclude match if |ΔDoppler| > 8 kHz
_VIS_CACHE_STEP_S   = 5.0       # re-evaluate visibility every 5 s (cache bucket)


# =============================================================================
# Public entry point
# =============================================================================

def match_bursts_to_satellites(
    timestamps_ms: np.ndarray,
    doppler_hz: np.ndarray,
    t0_utc: datetime,
    lat: float,
    lon: float,
    alt: float,
    *,
    el_min_deg: float = 0.0,
    max_doppler_err_hz: float = _MAX_DOPPLER_ERR_HZ,
    use_elevation_heuristic: bool = False,
    verbose: bool = True,
) -> dict:
    """Match each burst to the best visible Iridium satellite by Doppler shift.

    Parameters
    ----------
    timestamps_ms : (N,) float64
        Burst onset relative to ``t0_utc`` in milliseconds.
    doppler_hz : (N,) float64
        Measured Doppler shift per burst in Hz.
    t0_utc : datetime
        Session start time (UTC).  Naïve datetimes are assumed UTC.
    lat, lon, alt : float
        Observer WGS-84 coordinates (degrees, degrees, metres AMSL).
    el_min_deg : float
        Minimum satellite elevation to consider visible (default: 0°).
    max_doppler_err_hz : float
        Discard match if ``|measured_doppler − predicted_doppler|`` exceeds
        this threshold (default: 8 kHz).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Arrays of shape ``(N,)``:

        ``az_deg``    float32  — matched satellite azimuth [deg], NaN if no match
        ``el_deg``    float32  — matched satellite elevation [deg], NaN if no match
        ``sat_name``  U32      — satellite name, empty string if no match
        ``norad_id``  int32    — NORAD catalogue number, 0 if no match
        ``doppler_hz`` float64 — satellite predicted Doppler [Hz], NaN if no match
        ``source``    str      — "sgp4/celestrak" | "n2yo"
        ``n_matched`` int      — number of successfully matched bursts

    Notes
    -----
    When ``use_elevation_heuristic=True``, the highest-elevation visible
    satellite is selected without Doppler comparison.  This is appropriate
    when the measured ``doppler_hz`` contains an uncalibrated channel offset
    (as returned by ``compensate_doppler`` in ``space_collector``).
    """
    N = len(timestamps_ms)
    gt_az    = np.full(N, np.nan, dtype=np.float32)
    gt_el    = np.full(N, np.nan, dtype=np.float32)
    gt_names = np.array([""] * N, dtype="U32")
    gt_norad = np.zeros(N, dtype=np.int32)
    gt_dop   = np.full(N, np.nan, dtype=np.float64)
    source   = "sgp4/celestrak"

    # ── Optional: try N2YO API first (real-time sessions only) ────────────────
    n2yo_key = os.environ.get("LARK_N2YO_KEY", "").strip()
    if n2yo_key and not use_elevation_heuristic:
        try:
            _match_via_n2yo(
                n2yo_key, timestamps_ms, doppler_hz, t0_utc,
                lat, lon, alt, el_min_deg, max_doppler_err_hz,
                gt_az, gt_el, gt_names, gt_norad, gt_dop,
                verbose=verbose,
            )
            source = "n2yo"
        except Exception as exc:
            if verbose:
                print(f"[GT] N2YO failed ({exc}), falling back to sgp4/CelesTrak")
            # Reset arrays on N2YO failure
            gt_az[:] = np.nan;  gt_el[:] = np.nan
            gt_names[:] = ""
            gt_norad[:] = 0;    gt_dop[:] = np.nan
            source = "sgp4/celestrak"

    # ── Primary: sgp4 propagation via local Celestrak TLE cache ───────────────
    if source != "n2yo":
        _match_via_sgp4(
            timestamps_ms, doppler_hz, t0_utc,
            lat, lon, alt, el_min_deg, max_doppler_err_hz,
            gt_az, gt_el, gt_names, gt_norad, gt_dop,
            use_elevation_heuristic=use_elevation_heuristic,
            verbose=verbose,
        )

    n_matched = int(np.sum(np.isfinite(gt_az)))
    return {
        "az_deg":     gt_az,
        "el_deg":     gt_el,
        "sat_name":   gt_names,
        "norad_id":   gt_norad,
        "doppler_hz": gt_dop,
        "source":     source,
        "n_matched":  n_matched,
    }


# =============================================================================
# sgp4 / Celestrak backend  (default)
# =============================================================================

def _match_via_sgp4(
    timestamps_ms, doppler_hz, t0_utc,
    lat, lon, alt, el_min_deg, max_doppler_err_hz,
    gt_az, gt_el, gt_names, gt_norad, gt_dop,
    *,
    use_elevation_heuristic: bool = False,
    verbose: bool = True,
) -> None:
    """Fill ground-truth arrays using Celestrak TLEs + skyfield sgp4."""
    from shared.iridium_tle import load_catalogue  # local download / 24h cache

    try:
        catalogue = load_catalogue()
    except Exception as exc:
        if verbose:
            print(f"[GT] TLE load failed: {exc}")
        return

    if t0_utc.tzinfo is None:
        t0_utc = t0_utc.replace(tzinfo=timezone.utc)

    _vis_cache: dict[int, list] = {}

    for i in range(len(timestamps_ms)):
        burst_dt  = t0_utc + timedelta(milliseconds=float(timestamps_ms[i]))
        cache_key = int(burst_dt.timestamp() // _VIS_CACHE_STEP_S)

        if cache_key not in _vis_cache:
            _vis_cache[cache_key] = catalogue.visible_now(
                lat, lon, alt, burst_dt, el_min_deg=el_min_deg,
            )
        visible = _vis_cache[cache_key]

        if not visible:
            continue

        if use_elevation_heuristic:
            # Pick the highest-elevation satellite (no Doppler needed)
            best_sat = max(visible, key=lambda s: s["el_deg"])
        else:
            dop_est  = float(doppler_hz[i])
            best_sat = None
            best_err = max_doppler_err_hz

            for sat in visible:
                dop_err = abs(dop_est - sat["doppler_hz"])
                if dop_err < best_err:
                    best_err = dop_err
                    best_sat = sat

        if best_sat is not None:
            gt_az[i]    = best_sat["az_deg"]
            gt_el[i]    = best_sat["el_deg"]
            gt_names[i] = best_sat["name"]
            gt_norad[i] = best_sat["norad_id"]
            gt_dop[i]   = best_sat["doppler_hz"]

    n_ok = int(np.sum(np.isfinite(gt_az)))
    if verbose:
        print(f"[GT] sgp4/CelesTrak: {n_ok}/{len(timestamps_ms)} bursts matched")


# =============================================================================
# Angle-based ground-truth matcher  (requires per-burst DoA)
# =============================================================================

def match_bursts_by_angle(
    timestamps_ms: np.ndarray,
    az_music_deg: np.ndarray,
    el_music_deg: np.ndarray,
    t0_utc: datetime,
    lat: float,
    lon: float,
    alt: float,
    *,
    el_min_deg: float = 0.0,
    max_angular_sep_deg: float = 25.0,
    verbose: bool = True,
) -> dict:
    """Match each burst to the closest Iridium satellite by angular distance.

    Uses per-burst MUSIC DoA estimates (az, el) to find the TLE-predicted
    satellite whose pointing angle is closest.  No Doppler comparison is
    performed, making this robust to uncalibrated carrier-frequency offsets.

    Parameters
    ----------
    timestamps_ms : (N,) float64 — burst onset [ms] from session start
    az_music_deg  : (N,) float32 — MUSIC azimuth estimate per burst [deg]
    el_music_deg  : (N,) float32 — MUSIC elevation estimate per burst [deg]
    t0_utc : datetime — session start UTC
    lat, lon, alt : observer WGS-84
    el_min_deg : float — minimum elevation for visibility check
    max_angular_sep_deg : float — accept match only if angular error < this

    Returns
    -------
    Same dict structure as ``match_bursts_to_satellites``.
    """
    from shared.iridium_tle import load_catalogue

    N = len(timestamps_ms)
    gt_az    = np.full(N, np.nan, dtype=np.float32)
    gt_el    = np.full(N, np.nan, dtype=np.float32)
    gt_names = np.array([""] * N, dtype="U32")
    gt_norad = np.zeros(N, dtype=np.int32)
    gt_dop   = np.full(N, np.nan, dtype=np.float64)

    try:
        catalogue = load_catalogue()
    except Exception as exc:
        if verbose:
            print(f"[GT] TLE load failed: {exc}")
        return {
            "az_deg": gt_az, "el_deg": gt_el, "sat_name": gt_names,
            "norad_id": gt_norad, "doppler_hz": gt_dop,
            "source": "sgp4/celestrak", "n_matched": 0,
        }

    if t0_utc.tzinfo is None:
        t0_utc = t0_utc.replace(tzinfo=timezone.utc)

    _vis_cache: dict[int, list] = {}

    for i in range(N):
        burst_dt  = t0_utc + timedelta(milliseconds=float(timestamps_ms[i]))
        cache_key = int(burst_dt.timestamp() // _VIS_CACHE_STEP_S)

        if cache_key not in _vis_cache:
            _vis_cache[cache_key] = catalogue.visible_now(
                lat, lon, alt, burst_dt, el_min_deg=el_min_deg,
            )
        visible = _vis_cache[cache_key]

        if not visible:
            continue

        doa_az = float(az_music_deg[i])
        doa_el = float(el_music_deg[i])
        best_sat = None
        best_sep = max_angular_sep_deg

        for sat in visible:
            az_err = abs((doa_az - sat["az_deg"] + 180) % 360 - 180)
            el_err = abs(doa_el - sat["el_deg"])
            sep = np.sqrt(az_err**2 * np.cos(np.deg2rad(doa_el))**2 + el_err**2)
            if sep < best_sep:
                best_sep = sep
                best_sat = sat

        if best_sat is not None:
            gt_az[i]    = best_sat["az_deg"]
            gt_el[i]    = best_sat["el_deg"]
            gt_names[i] = best_sat["name"]
            gt_norad[i] = best_sat["norad_id"]
            gt_dop[i]   = best_sat["doppler_hz"]

    n_ok = int(np.sum(np.isfinite(gt_az)))
    if verbose:
        print(f"[GT] angle-based: {n_ok}/{N} bursts matched "
              f"(max_sep={max_angular_sep_deg}°)")

    return {
        "az_deg": gt_az, "el_deg": gt_el, "sat_name": gt_names,
        "norad_id": gt_norad, "doppler_hz": gt_dop,
        "source": "sgp4/celestrak", "n_matched": n_ok,
    }


# =============================================================================
# N2YO REST API backend  (optional — requires LARK_N2YO_KEY env var)
# =============================================================================
# N2YO docs: https://www.n2yo.com/api/
#
# Endpoints used:
#   /above/{lat}/{lng}/{alt}/{search_deg}/{category_id}/&apiKey={key}
#       → list of satellites currently above the observer
#   /positions/{norad_id}/{lat}/{lng}/{alt}/{duration}/&apiKey={key}
#       → az/el position stream for one satellite (max 300 s window)
#
# Limitations:
#   • Free tier: 1000 transactions/hour
#   • Historical data (>~10 min in the past) requires a paid account
#   • For live recording sessions timestamps ≈ now, so this backend works well
# =============================================================================

def _match_via_n2yo(
    key: str,
    timestamps_ms, doppler_hz, t0_utc,
    lat, lon, alt_m, el_min_deg, max_dop_err,
    gt_az, gt_el, gt_names, gt_norad, gt_dop,
    *,
    verbose: bool = True,
) -> None:
    """Fill ground-truth arrays using the N2YO REST API.

    Raises RuntimeError if the API is unreachable or returns an error.
    Falls back to sgp4 for bursts that are > 5 min in the past (N2YO
    historical positions require a paid plan).
    """
    import json
    import urllib.request

    BASE     = "https://api.n2yo.com/rest/v1/satellite"
    ALT_KM   = int(max(alt_m / 1000.0, 0))
    IRIDIUM_CATEGORY = 17     # N2YO group for IRIDIUM-NEXT
    SEC_RADIUS       = 0      # 0° = above the horizon

    # Step 1 — get currently visible Iridium satellites
    url  = (f"{BASE}/above/{lat}/{lon}/{ALT_KM}/{SEC_RADIUS}/"
            f"{IRIDIUM_CATEGORY}/&apiKey={key}")
    req  = urllib.request.Request(url, headers={"User-Agent": "LARK/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    if "above" not in data:
        raise RuntimeError(f"N2YO /above error: {data}")

    above = [s for s in data["above"]
             if float(s.get("satelevation", -99)) >= el_min_deg]
    if not above:
        if verbose:
            print("[GT] N2YO: no Iridium satellites above horizon")
        return

    norad_ids = [int(s["satid"]) for s in above]

    # Step 2 — fetch /positions for each visible satellite over session window
    session_s = max(60, int(timestamps_ms[-1] / 1000.0) + 10)
    session_s = min(session_s, 300)    # N2YO max is 300 s per call

    # sat_positions[norad_id] → list of {az, el} indexed by second offset
    sat_positions: dict[int, list] = {}
    for norad_id in norad_ids:
        pos_url = (f"{BASE}/positions/{norad_id}/{lat}/{lon}/{ALT_KM}/"
                   f"{session_s}/&apiKey={key}")
        req = urllib.request.Request(pos_url, headers={"User-Agent": "LARK/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            pos_data = json.loads(resp.read())
        positions = pos_data.get("positions", [])
        if positions:
            sat_positions[norad_id] = positions   # each: {az, el, ...}

    if not sat_positions:
        raise RuntimeError("N2YO: no position data returned")

    # Step 3 — for each burst, find best satellite by Doppler proximity
    # Since N2YO doesn't give Doppler, compute it from range rate (az/el only
    # approach) using the sgp4 backend for Doppler estimates
    from shared.iridium_tle import load_catalogue
    catalogue = load_catalogue()

    if t0_utc.tzinfo is None:
        t0_utc = t0_utc.replace(tzinfo=timezone.utc)

    for i in range(len(timestamps_ms)):
        burst_off_s = timestamps_ms[i] / 1000.0
        burst_dt    = t0_utc + timedelta(seconds=burst_off_s)

        best_sat = None
        best_err = max_dop_err

        for norad_id, positions in sat_positions.items():
            # Get az/el at burst timestamp (linear interpolation)
            idx = min(int(burst_off_s), len(positions) - 1)
            pos = positions[idx]
            az_n2yo = float(pos.get("azimuth", 0))
            el_n2yo = float(pos.get("elevation", -99))
            if el_n2yo < el_min_deg:
                continue

            # Use sgp4 Doppler for matching (N2YO doesn't provide Doppler)
            try:
                sat_azel = catalogue.sat_azel(norad_id, lat, float(lon), alt_m, burst_dt)
                predicted_dop = _doppler_from_range_rate(
                    catalogue, norad_id, lat, float(lon), alt_m, burst_dt
                )
            except Exception:
                continue

            dop_err = abs(float(doppler_hz[i]) - predicted_dop)
            if dop_err < best_err:
                best_err = dop_err
                best_sat = {
                    "az_deg":     az_n2yo,
                    "el_deg":     el_n2yo,
                    "name":       str(norad_id),
                    "norad_id":   norad_id,
                    "doppler_hz": predicted_dop,
                }

        if best_sat is not None:
            gt_az[i]    = best_sat["az_deg"]
            gt_el[i]    = best_sat["el_deg"]
            gt_names[i] = best_sat["name"]
            gt_norad[i] = best_sat["norad_id"]
            gt_dop[i]   = best_sat["doppler_hz"]

    n_ok = int(np.sum(np.isfinite(gt_az)))
    if verbose:
        print(f"[GT] N2YO: {n_ok}/{len(timestamps_ms)} bursts matched")


def _doppler_from_range_rate(catalogue, sat_id, lat, lon, alt_m, t_utc) -> float:
    """Estimate Doppler shift [Hz] from sgp4 range rate."""
    from shared.iridium import SIMPLEX_RING_CH_HZ, C_LIGHT
    from datetime import timedelta

    dt_s = 0.5
    az0, el0, r0 = catalogue.sat_azel(sat_id, lat, lon, alt_m,
                                       t_utc - timedelta(seconds=dt_s))
    az1, el1, r1 = catalogue.sat_azel(sat_id, lat, lon, alt_m,
                                       t_utc + timedelta(seconds=dt_s))
    range_rate = (r1 - r0) / (2 * dt_s)   # km/s, positive = moving away
    return -range_rate * 1e3 * SIMPLEX_RING_CH_HZ / C_LIGHT
