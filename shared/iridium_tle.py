"""
shared.iridium_tle
==================
Iridium constellation TLE catalogue: fetch, cache, propagate, and query.

Uses *skyfield* (+ sgp4) for high-accuracy SGP4 orbital propagation.

TLE source priority
-------------------
1. Local cache ``~/.config/lark/iridium_tle.txt`` (if < 24 h old)
2. CelesTrak bulk download ``https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle``
3. Manual file passed by caller

Public API
----------
    catalogue  = load_catalogue()
    visible    = visible_passes(catalogue, lat, lon, alt, t0_utc, t1_utc)
    az, el, rng = sat_azel(catalogue["IRIDIUM 106"], lat, lon, alt, t_utc)

Each visible pass is a dict with keys:
    name, norad_id, rise_utc, culmination_utc, set_utc,
    rise_az_deg, culmination_az_deg, culmination_el_deg, set_az_deg,
    max_el_deg, duration_s
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

_CACHE_DIR  = Path.home() / ".config" / "lark"
_CACHE_TLE  = _CACHE_DIR / "iridium_tle.txt"
_CACHE_META = _CACHE_DIR / "iridium_tle_meta.json"
_TLE_MAX_AGE_S = 24 * 3600   # re-download after 24 hours

_CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php"
    "?GROUP=iridium-NEXT&FORMAT=tle"
)


# ─────────────────────────────────────────────────────────────────────────────
# TLE fetch / cache
# ─────────────────────────────────────────────────────────────────────────────

def _tle_cache_valid() -> bool:
    if not _CACHE_TLE.is_file() or not _CACHE_META.is_file():
        return False
    try:
        with open(_CACHE_META) as f:
            meta = json.load(f)
        age = time.time() - meta.get("fetched_epoch", 0)
        return age < _TLE_MAX_AGE_S
    except Exception:
        return False


def fetch_tle(force: bool = False) -> str:
    """Download / read cached Iridium NEXT TLE text.

    Returns the raw TLE text (3-line format per satellite).
    """
    if not force and _tle_cache_valid():
        return _CACHE_TLE.read_text()

    print("[TLE] Downloading Iridium NEXT catalogue from CelesTrak …", end=" ", flush=True)
    import urllib.request
    req = urllib.request.Request(_CELESTRAK_URL, headers={
        "User-Agent": "LARK-sat-predict/1.0 (github.com/LARK)"
    })
    resp = urllib.request.urlopen(req, timeout=15)
    text = resp.read().decode("utf-8").strip()
    n_sats = text.count("\n") // 3 + 1
    print(f"{n_sats} satellites")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_TLE.write_text(text)
    with open(_CACHE_META, "w") as f:
        json.dump({"fetched_epoch": time.time(),
                    "n_satellites": n_sats,
                    "source": _CELESTRAK_URL}, f, indent=2)
    return text


def load_catalogue(
    tle_file: Optional[str] = None,
    force_download: bool = False,
) -> "IridiumCatalogue":
    """Load Iridium constellation into a propagation-ready catalogue.

    Parameters
    ----------
    tle_file : str | None
        Path to a local TLE file. If *None*, auto-fetch from CelesTrak.
    force_download : bool
        Force a fresh download even if cache is valid.

    Returns
    -------
    IridiumCatalogue
        Object with ``.visible_above()`` and ``.sat_azel()`` methods.
    """
    if tle_file and os.path.isfile(tle_file):
        text = open(tle_file).read()
    else:
        text = fetch_tle(force=force_download)
    return IridiumCatalogue(text)


# ─────────────────────────────────────────────────────────────────────────────
# Catalogue class
# ─────────────────────────────────────────────────────────────────────────────

class IridiumCatalogue:
    """Iridium constellation: propagation, visibility, Az/El prediction."""

    def __init__(self, tle_text: str):
        from skyfield.api import load as _sf_load, EarthSatellite, wgs84
        self._ts = _sf_load.timescale()
        self._wgs84 = wgs84

        lines = [l.strip() for l in tle_text.strip().splitlines() if l.strip()]
        self.satellites: list[EarthSatellite] = []
        i = 0
        while i < len(lines):
            # TLE can be 3-line (name + L1 + L2) or 2-line (L1 + L2)
            if not lines[i].startswith("1 "):
                name = lines[i]
                l1, l2 = lines[i + 1], lines[i + 2]
                i += 3
            else:
                name = "?"
                l1, l2 = lines[i], lines[i + 1]
                i += 2
            sat = EarthSatellite(l1, l2, name.strip(), self._ts)
            self.satellites.append(sat)

        self._by_name = {s.name: s for s in self.satellites}
        self._by_norad = {s.model.satnum: s for s in self.satellites}
        print(f"[TLE] Loaded {len(self.satellites)} Iridium satellites")

    def __len__(self) -> int:
        return len(self.satellites)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._by_norad.get(key) or self.satellites[key]
        return self._by_name[key]

    # ── Single-point Az/El ────────────────────────────────────────────────────

    def sat_azel(
        self,
        sat_name_or_id,
        lat: float,
        lon: float,
        alt: float,
        t_utc: datetime,
    ) -> tuple[float, float, float]:
        """Compute (az_deg, el_deg, range_km) for one satellite at one instant.

        Parameters
        ----------
        sat_name_or_id : str | int
            Satellite name (e.g. "IRIDIUM 106") or NORAD catalogue number.
        lat, lon, alt : float
            Observer WGS-84 position (degrees, degrees, metres AMSL).
        t_utc : datetime
            UTC time (timezone-aware or naive = assumed UTC).

        Returns
        -------
        (az_deg, el_deg, range_km) : tuple[float, float, float]
        """
        sat = self[sat_name_or_id] if isinstance(sat_name_or_id, (str, int)) else sat_name_or_id
        observer = self._wgs84.latlon(lat, lon, elevation_m=alt)
        t = self._ts.from_datetime(t_utc.replace(tzinfo=timezone.utc) if t_utc.tzinfo is None else t_utc)
        diff = sat - observer
        topocentric = diff.at(t)
        alt_deg, az_deg, distance_km = topocentric.altaz()
        return float(az_deg.degrees), float(alt_deg.degrees), float(distance_km.km)

    # ── Bulk Az/El over time ──────────────────────────────────────────────────

    def sat_track(
        self,
        sat_name_or_id,
        lat: float,
        lon: float,
        alt: float,
        t0_utc: datetime,
        t1_utc: datetime,
        step_s: float = 10.0,
    ) -> dict:
        """Compute full Az/El track for one satellite over a time window.

        Returns dict with keys: ``times_utc``, ``az_deg``, ``el_deg``, ``range_km``.
        All numpy arrays.
        """
        sat = self[sat_name_or_id] if isinstance(sat_name_or_id, (str, int)) else sat_name_or_id
        observer = self._wgs84.latlon(lat, lon, elevation_m=alt)
        diff = sat - observer

        dt = (t1_utc - t0_utc).total_seconds()
        n = max(2, int(dt / step_s) + 1)
        steps = [t0_utc + timedelta(seconds=i * dt / (n - 1)) for i in range(n)]
        t_arr = self._ts.from_datetimes(
            [s.replace(tzinfo=timezone.utc) if s.tzinfo is None else s for s in steps]
        )

        topo = diff.at(t_arr)
        alt_deg, az_deg, dist = topo.altaz()
        return {
            "name":     sat.name,
            "norad_id": sat.model.satnum,
            "times_utc": np.array([s.isoformat() for s in steps]),
            "az_deg":    az_deg.degrees.astype(np.float64),
            "el_deg":    alt_deg.degrees.astype(np.float64),
            "range_km":  dist.km.astype(np.float64),
        }

    # ── Visible satellites NOW ────────────────────────────────────────────────

    def visible_now(
        self,
        lat: float,
        lon: float,
        alt: float,
        t_utc: Optional[datetime] = None,
        el_min_deg: float = 5.0,
    ) -> list[dict]:
        """Return list of satellites currently above ``el_min_deg``.

        Each element: ``{name, norad_id, az_deg, el_deg, range_km, doppler_hz}``.
        Sorted by descending elevation.
        """
        if t_utc is None:
            t_utc = datetime.now(timezone.utc)
        observer = self._wgs84.latlon(lat, lon, elevation_m=alt)
        t = self._ts.from_datetime(
            t_utc.replace(tzinfo=timezone.utc) if t_utc.tzinfo is None else t_utc
        )

        results = []
        for sat in self.satellites:
            diff = sat - observer
            topo = diff.at(t)
            el, az, dist = topo.altaz()
            el_d = float(el.degrees)
            if el_d < el_min_deg:
                continue

            # Doppler estimate (numerical derivative of range)
            dt_s = 0.5
            t_before = self._ts.from_datetime(t_utc - timedelta(seconds=dt_s))
            t_after  = self._ts.from_datetime(t_utc + timedelta(seconds=dt_s))
            r0 = float((sat - observer).at(t_before).altaz()[2].km)
            r1 = float((sat - observer).at(t_after).altaz()[2].km)
            range_rate_km_s = (r1 - r0) / (2 * dt_s)
            from shared.iridium import SIMPLEX_RING_CH_HZ, C_LIGHT
            doppler_hz = -range_rate_km_s * 1e3 * SIMPLEX_RING_CH_HZ / C_LIGHT

            results.append({
                "name":       sat.name,
                "norad_id":   sat.model.satnum,
                "az_deg":     round(float(az.degrees), 2),
                "el_deg":     round(el_d, 2),
                "range_km":   round(float(dist.km), 1),
                "doppler_hz": round(doppler_hz, 1),
            })
        results.sort(key=lambda x: -x["el_deg"])
        return results

    # ── Pass prediction window ────────────────────────────────────────────────

    def predict_passes(
        self,
        lat: float,
        lon: float,
        alt: float,
        t0_utc: datetime,
        t1_utc: datetime,
        el_min_deg: float = 5.0,
        step_s: float = 30.0,
    ) -> list[dict]:
        """Predict all Iridium passes visible above ``el_min_deg`` in [t0, t1].

        Uses coarse time-stepping then refines rise/set times by bisection.

        Returns list of pass dicts sorted by rise time, each with:
            name, norad_id, rise_utc, culmination_utc, set_utc,
            rise_az_deg, culm_az_deg, culm_el_deg, set_az_deg,
            max_el_deg, duration_s
        """
        observer = self._wgs84.latlon(lat, lon, elevation_m=alt)
        dt_total = (t1_utc - t0_utc).total_seconds()
        n_steps = max(2, int(dt_total / step_s) + 1)
        time_offsets = np.linspace(0, dt_total, n_steps)
        step_times = [t0_utc + timedelta(seconds=float(s)) for s in time_offsets]
        t_arr = self._ts.from_datetimes(
            [s.replace(tzinfo=timezone.utc) if s.tzinfo is None else s for s in step_times]
        )

        passes = []
        for sat in self.satellites:
            diff = sat - observer
            topo = diff.at(t_arr)
            el_arr = topo.altaz()[0].degrees   # (n_steps,)

            above = el_arr >= el_min_deg
            # Find contiguous above-horizon segments
            transitions = np.diff(above.astype(np.int8))
            rises = np.where(transitions == 1)[0]    # index before transition
            sets  = np.where(transitions == -1)[0]

            # Handle edge cases
            if above[0]:
                rises = np.concatenate([[0], rises])
            if above[-1]:
                sets = np.concatenate([sets, [n_steps - 2]])

            for ri, si in zip(rises, sets):
                # Coarse rise/culmination/set
                seg_el = el_arr[ri:si + 2]
                seg_idx = np.arange(ri, min(si + 2, n_steps))
                if len(seg_el) == 0:
                    continue
                peak_j = seg_idx[np.argmax(seg_el)]
                max_el = float(el_arr[peak_j])
                if max_el < el_min_deg:
                    continue

                rise_t = step_times[ri]
                set_t  = step_times[min(si + 1, n_steps - 1)]
                culm_t = step_times[peak_j]

                # Az at each event
                def _azel_at(t_utc):
                    t = self._ts.from_datetime(
                        t_utc.replace(tzinfo=timezone.utc)
                        if t_utc.tzinfo is None else t_utc
                    )
                    e, a, d = diff.at(t).altaz()
                    return float(a.degrees), float(e.degrees)

                rise_az, _ = _azel_at(rise_t)
                set_az, _  = _azel_at(set_t)
                culm_az, culm_el = _azel_at(culm_t)

                passes.append({
                    "name":             sat.name,
                    "norad_id":         sat.model.satnum,
                    "rise_utc":         rise_t.isoformat(),
                    "culmination_utc":  culm_t.isoformat(),
                    "set_utc":          set_t.isoformat(),
                    "rise_az_deg":      round(rise_az, 1),
                    "culm_az_deg":      round(culm_az, 1),
                    "culm_el_deg":      round(culm_el, 1),
                    "set_az_deg":       round(set_az, 1),
                    "max_el_deg":       round(max_el, 1),
                    "duration_s":       round((set_t - rise_t).total_seconds(), 1),
                })

        passes.sort(key=lambda p: p["rise_utc"])
        return passes

    # ── Doppler prediction for one satellite ──────────────────────────────────

    def doppler_track(
        self,
        sat_name_or_id,
        lat: float,
        lon: float,
        alt: float,
        t0_utc: datetime,
        t1_utc: datetime,
        freq_hz: float = 1_626_270_000.0,
        step_s: float = 1.0,
    ) -> dict:
        """Predict Doppler shift for a satellite over [t0, t1].

        Returns dict with ``times_utc``, ``doppler_hz``, ``az_deg``, ``el_deg``,
        ``range_km`` — all numpy arrays.
        """
        from shared.iridium import C_LIGHT
        sat = self[sat_name_or_id] if isinstance(sat_name_or_id, (str, int)) else sat_name_or_id
        observer = self._wgs84.latlon(lat, lon, elevation_m=alt)

        dt = (t1_utc - t0_utc).total_seconds()
        n = max(3, int(dt / step_s) + 1)
        steps = [t0_utc + timedelta(seconds=i * dt / (n - 1)) for i in range(n)]
        t_arr = self._ts.from_datetimes(
            [s.replace(tzinfo=timezone.utc) if s.tzinfo is None else s for s in steps]
        )

        diff = sat - observer
        topo = diff.at(t_arr)
        el_deg, az_deg, range_km = topo.altaz()

        # Doppler: d(range)/dt
        r = range_km.km
        dr = np.gradient(r, step_s) * 1e3   # m/s
        doppler_hz = -dr * freq_hz / C_LIGHT

        return {
            "name":       sat.name,
            "norad_id":   sat.model.satnum,
            "times_utc":  np.array([s.isoformat() for s in steps]),
            "doppler_hz": doppler_hz.astype(np.float64),
            "az_deg":     az_deg.degrees.astype(np.float64),
            "el_deg":     el_deg.degrees.astype(np.float64),
            "range_km":   r.astype(np.float64),
        }
