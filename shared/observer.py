"""
shared.observer
===============
Observer (ground station) location auto-detection and caching.

Resolution order
-----------------
1. Explicit ``(lat, lon, alt)`` passed by caller
2. JSON sidecar ``observer_lat`` / ``observer_lon`` / ``observer_alt_m``
3. Cached location in ``~/.config/lark/observer.json``
4. ``gpsd`` daemon (``gpsd`` Python package, then ``gpsd -n`` scan)
5. GeoIP fallback (coarse: city-level, ~10–50 km accuracy)
6. Interactive prompt

The resolved location is cached to ``~/.config/lark/observer.json`` so
subsequent runs do not require network or GPS.

Usage
-----
    from shared.observer import get_observer
    lat, lon, alt = get_observer()                   # auto-detect
    lat, lon, alt = get_observer(lat=45.0, lon=7.5)  # explicit
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

_CACHE_DIR  = Path.home() / ".config" / "lark"
_CACHE_FILE = _CACHE_DIR / "observer.json"

Location = tuple[float, float, float]   # (lat_deg, lon_deg, alt_m)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_observer(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    alt: Optional[float] = None,
    meta: Optional[dict] = None,
    interactive: bool = True,
) -> Location:
    """Return ``(lat_deg, lon_deg, alt_m)`` via the resolution chain.

    Parameters
    ----------
    lat, lon, alt : float | None
        Explicit coordinates.  If all three are given, returned immediately.
    meta : dict | None
        JSON sidecar metadata (from .npz companion file).
    interactive : bool
        If *True* and all automatic methods fail, prompt the user.
    """
    # 1 — explicit
    if lat is not None and lon is not None:
        return _validated(lat, lon, alt or 0.0)

    # 2 — sidecar metadata
    loc = _from_meta(meta)
    if loc:
        return loc

    # 3 — cache
    loc = _from_cache()
    if loc:
        return loc

    # 4 — gpsd
    loc = _from_gpsd()
    if loc:
        _save_cache(*loc)
        return loc

    # 5 — GeoIP
    loc = _from_geoip()
    if loc:
        _save_cache(*loc)
        return loc

    # 6 — interactive prompt
    if interactive and sys.stdin.isatty():
        loc = _from_prompt()
        _save_cache(*loc)
        return loc

    raise RuntimeError(
        "Cannot determine observer location. "
        "Pass --lat/--lon, set ~/.config/lark/observer.json, or start gpsd."
    )


def save_to_meta(meta: dict, lat: float, lon: float, alt: float) -> dict:
    """Inject observer location into a JSON-sidecar metadata dict (in-place)."""
    meta["observer_lat"] = round(lat, 7)
    meta["observer_lon"] = round(lon, 7)
    meta["observer_alt_m"] = round(alt, 1)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Resolution backends
# ─────────────────────────────────────────────────────────────────────────────

def _validated(lat: float, lon: float, alt: float) -> Location:
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of range: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude out of range: {lon}")
    return (float(lat), float(lon), float(alt))


def _from_meta(meta: Optional[dict]) -> Optional[Location]:
    if meta is None:
        return None
    lat = meta.get("observer_lat")
    lon = meta.get("observer_lon")
    if lat is not None and lon is not None:
        return _validated(lat, lon, meta.get("observer_alt_m", 0.0))
    return None


def _from_cache() -> Optional[Location]:
    if not _CACHE_FILE.is_file():
        return None
    try:
        with open(_CACHE_FILE) as f:
            d = json.load(f)
        return _validated(d["lat"], d["lon"], d.get("alt", 0.0))
    except Exception:
        return None


def _save_cache(lat: float, lon: float, alt: float) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump({"lat": round(lat, 7), "lon": round(lon, 7),
                    "alt": round(alt, 1)}, f, indent=2)


def _from_gpsd() -> Optional[Location]:
    """Try gpsd Python bindings (``gpsd`` or ``gps3`` packages)."""
    # Attempt 1: python-gps (gpsd official)
    try:
        import gps                               # type: ignore
        session = gps.gps(mode=gps.WATCH_ENABLE)
        for _ in range(50):    # up to ~5 s of NMEA sentences
            report = session.next()
            if report["class"] == "TPV":
                lat = report.get("lat")
                lon = report.get("lon")
                alt = report.get("altMSL", report.get("alt", 0.0))
                if lat is not None and lon is not None:
                    return _validated(lat, lon, alt or 0.0)
    except Exception:
        pass

    # Attempt 2: gpsd-py3 package
    try:
        import gpsd as _gpsd                     # type: ignore
        _gpsd.connect()
        pkt = _gpsd.get_current()
        return _validated(pkt.lat, pkt.lon, getattr(pkt, "alt", 0.0))
    except Exception:
        pass

    return None


def _from_geoip() -> Optional[Location]:
    """Coarse city-level from ip-api.com (no key required, HTTP only)."""
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            "http://ip-api.com/json/?fields=lat,lon,city,country",
            timeout=4,
        )
        d = json.loads(resp.read().decode())
        lat, lon = d.get("lat"), d.get("lon")
        if lat is not None and lon is not None:
            city = d.get("city", "?")
            country = d.get("country", "?")
            print(f"[OBS] GeoIP location: {city}, {country}  "
                  f"({lat:.3f}°, {lon:.3f}°)  — coarse, ~10–50 km")
            return _validated(lat, lon, 0.0)
    except Exception:
        pass
    return None


def _from_prompt() -> Location:
    print("\n[OBS] Enter observer location (WGS-84):")
    while True:
        try:
            lat = float(input("  Latitude  [°N]: "))
            lon = float(input("  Longitude [°E]: "))
            alt_s = input("  Altitude  [m AMSL, default=0]: ").strip()
            alt = float(alt_s) if alt_s else 0.0
            return _validated(lat, lon, alt)
        except (ValueError, KeyError) as e:
            print(f"  Invalid input: {e}. Try again.")
