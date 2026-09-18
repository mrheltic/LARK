"""
broadcast_ephemeris.py — Ephemeris and time recovered from the Iridium downlink.

The DOA pipeline treats a burst as a tone and discards the payload.  The payload,
however, is exactly what a self-contained PNT solution needs, broadcast in the
clear by every satellite:

  * **IRA** (Ring Alert) carries ``sat_id``, ``beam`` and the transmitting
    satellite's own geocentric **ECEF position**, quantised to 4 km per axis.
  * **IBC** (Broadcast) carries **Iridium system time**, derived from the
    satellite's clock rather than from the receiving host's.

Together they remove the two external dependencies of the positioning solver:
the TLE catalogue (no network, no cached almanac) and the host clock.

This module turns ``iridium-parser.py -o line`` output back into a usable
ephemeris.  It deliberately does no I/O beyond reading those text files and no
orbital propagation: pure geometry over decoded numbers.

Three things make the raw decodes unusable as-is, and each has a stage here:

1. **Roughly half the position fields are wrong even when the BCH check passes.**
   They are rejected by orbital radius: an Iridium NEXT satellite sits at
   ~7158 km geocentric, so anything outside [7100, 7250] km is a corrupt decode.
   ``iridium-toolkit``'s own tooling applies the same test (``q.ra_alt>7100``).
2. **Timestamps are chunk-relative and the chunk time axis is compressed**,
   because dropped CPIs are concatenated during export.  ``remap_epochs``
   inverts that with the ``index.json`` written by ``scripts/export_iq.py``.
3. **4 km quantisation makes finite differences useless for velocity** — two
   fixes 90 ms apart would imply 44 km/s.  ``ShortArc`` fits a low-order
   polynomial through the whole pass instead: the quantisation error is
   independent between fixes so it averages down, and the analytic derivative
   gives a clean velocity.  Degree 4 tracks a 600 s arc to ~20 m and degree 5 to
   ~1 m, three orders of magnitude below the quantisation floor, so the fit
   order is never the limiting term.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# Iridium NEXT orbits at ~780 km altitude -> ~7158 km geocentric radius.
IRA_RADIUS_MIN_KM = 7100.0
IRA_RADIUS_MAX_KM = 7250.0
IRA_POS_LSB_KM = 4.0

# A new pass of the same satellite starts after a gap this long [s].
DEFAULT_ARC_GAP_S = 120.0
DEFAULT_ARC_DEG = 5

_RE_NAME = re.compile(r"u-(\S+?)-e\d+")
_RE_XYZ = re.compile(r"xyz=\(([+-]\d+),([+-]\d+),([+-]\d+)\)")
_RE_SAT = re.compile(r"sat:(\d+)")
_RE_BEAM = re.compile(r"beam:(\d+)")
_RE_TIME = re.compile(r"time:(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z")


@dataclass
class IraFix:
    """One decoded satellite position."""

    t: float                  # unix epoch [s] (chunk-relative until remapped)
    sat_id: int
    beam: int
    pos_km: np.ndarray        # (3,) ECEF
    chunk: str = ""
    confidence: float = 0.0

    @property
    def radius_km(self) -> float:
        return float(np.linalg.norm(self.pos_km))


@dataclass
class IbcTime:
    """One Iridium system-time stamp, paired with our own receive time."""

    t_rx: float               # our clock (chunk-relative until remapped)
    t_iridium: float          # unix epoch decoded from the satellite
    sat_id: int
    chunk: str = ""


def _parse_common(line: str) -> tuple[str, float, float] | None:
    """(chunk, t_chunk_s, confidence) from a parsed line, or None."""
    parts = line.split()
    if len(parts) < 5:
        return None
    m = _RE_NAME.match(parts[1])
    if m is None:
        return None
    try:
        t_ms = float(parts[2])
        conf = float(parts[4].rstrip("%"))
    except ValueError:
        return None
    return m.group(1), t_ms / 1000.0, conf


def parse_decodes(parsed_path: str, *, require_ok: bool = True,
                  ) -> tuple[list[IraFix], list[IbcTime]]:
    """Read ``iridium-parser.py -o line`` output into IRA fixes and IBC times.

    ``require_ok`` keeps only frames whose error correction reported ``{OK}``.
    That filter is cheap and roughly halves the corrupt-position rate before the
    radius test does the rest.
    """
    fixes: list[IraFix] = []
    times: list[IbcTime] = []
    with open(parsed_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("IRA:"):
                if require_ok and "{OK}" not in line:
                    continue
                common = _parse_common(line)
                xyz = _RE_XYZ.search(line)
                sat = _RE_SAT.search(line)
                if common is None or xyz is None or sat is None:
                    continue
                chunk, t, conf = common
                beam = _RE_BEAM.search(line)
                pos = np.array([int(g) * IRA_POS_LSB_KM for g in xyz.groups()],
                               dtype=np.float64)
                fixes.append(IraFix(t=t, sat_id=int(sat.group(1)),
                                    beam=int(beam.group(1)) if beam else -1,
                                    pos_km=pos, chunk=chunk, confidence=conf))
            elif line.startswith("IBC:"):
                common = _parse_common(line)
                tm = _RE_TIME.search(line)
                sat = _RE_SAT.search(line)
                if common is None or tm is None:
                    continue
                chunk, t, _ = common
                dt = datetime.fromisoformat(tm.group(1)).replace(tzinfo=timezone.utc)
                times.append(IbcTime(t_rx=t, t_iridium=dt.timestamp(),
                                     sat_id=int(sat.group(1)) if sat else -1,
                                     chunk=chunk))
    return fixes, times


def load_export_index(index_path: str) -> dict:
    """The ``index.json`` written next to the exported .cf32 chunks."""
    with open(index_path, encoding="utf-8") as fh:
        return json.load(fh)


def remap_epochs(records, index: dict) -> list:
    """Chunk-relative times -> true unix epochs, dropping unmappable records.

    Export concatenates CPIs that are not adjacent in time (the recorder drops
    CPIs when live processing lags), so a chunk's internal clock runs slow by the
    recording duty cycle — about 30% on the reference session.  Every CPI
    contributes exactly ``cpi_size`` samples, so the mapping back is exact:

        k = offset // cpi_size ;  t = epochs[k] + (offset - k*cpi_size) / fs

    Records landing on a CPI with an unknown epoch (NaN) are dropped.
    """
    fs = float(index["fs_hz"])
    cpi = int(index["cpi_size"])
    by_chunk = {os.path.splitext(c["file"])[0]: c for c in index["chunks"]}

    out = []
    for rec in records:
        chunk = by_chunk.get(rec.chunk)
        if chunk is None:
            continue
        t_field = "t_rx" if isinstance(rec, IbcTime) else "t"
        offset = getattr(rec, t_field) * fs
        k = int(offset // cpi)
        epochs = chunk["epochs"]
        if k < 0 or k >= len(epochs):
            continue
        ep = epochs[k]
        if ep is None or not np.isfinite(ep):
            continue
        setattr(rec, t_field, float(ep) + (offset - k * cpi) / fs)
        out.append(rec)
    return out


def filter_plausible(fixes: list[IraFix], *, r_min: float = IRA_RADIUS_MIN_KM,
                     r_max: float = IRA_RADIUS_MAX_KM) -> list[IraFix]:
    """Drop fixes whose geocentric radius is not a plausible Iridium orbit.

    About half of BCH-clean IRA frames still carry a corrupt position field; this
    single test removes essentially all of them, because a corrupted 12-bit word
    almost never lands back on the orbital sphere.
    """
    return [f for f in fixes if r_min < f.radius_km < r_max]


def estimate_clock_offset(times: list[IbcTime]) -> tuple[float, float, int]:
    """(offset, MAD, n) [s] between the host clock and Iridium system time.

    Positive offset means the host timestamps are *late*.  Signal propagation
    (~2.6 ms at most) is far below the resolution that matters here and is
    ignored.  Use the median: a handful of IBC frames decode with a corrupt
    time field and would drag a mean.
    """
    if not times:
        return float("nan"), float("nan"), 0
    d = np.array([t.t_rx - t.t_iridium for t in times], dtype=np.float64)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    return med, mad, int(d.size)


@dataclass
class ShortArc:
    """Polynomial fit of one pass in ECEF, with analytic velocity.

    Time is normalised to [-1, 1] before fitting; without that a degree-5 fit
    over epoch-sized numbers is numerically hopeless.
    """

    t0: float
    scale: float
    coeffs: np.ndarray            # (3, deg+1)
    n_used: int
    residual_km: float
    t_start: float
    t_end: float

    def _norm(self, t):
        return (np.asarray(t, dtype=np.float64) - self.t0) / self.scale

    def position(self, t) -> np.ndarray:
        tn = self._norm(t)
        return np.stack([np.polyval(c, tn) for c in self.coeffs], axis=-1)

    def velocity(self, t) -> np.ndarray:
        """[km/s] — derivative of the fit, chain rule through the normalisation."""
        tn = self._norm(t)
        return np.stack([np.polyval(np.polyder(c), tn) / self.scale
                         for c in self.coeffs], axis=-1)

    def state(self, t) -> tuple[np.ndarray, np.ndarray]:
        return self.position(t), self.velocity(t)


def fit_short_arc(t: np.ndarray, pos_km: np.ndarray, *, deg: int = DEFAULT_ARC_DEG,
                  reject_km: float = 12.0, max_iter: int = 5) -> ShortArc | None:
    """Robust polynomial fit of one pass; None if there is not enough data.

    The rejection threshold has to be adaptive.  A corrupt 12-bit position word
    lands hundreds of km away, so the first least-squares fit is dragged far off
    the true arc and *every* point — good ones included — sits beyond any fixed
    tolerance.  Thresholding on the residual's own median and MAD instead (the
    same robust scheme as ``core/track_filter.py``) peels the gross outliers off
    first and tightens as the fit recovers; ``reject_km`` only sets the floor, so
    a clean arc is never over-pruned down into the 4 km quantisation noise.
    """
    t = np.asarray(t, dtype=np.float64)
    pos_km = np.asarray(pos_km, dtype=np.float64)
    if t.size < 2 * (deg + 1):
        deg = max(2, t.size // 2 - 1)
    if t.size < deg + 1 or t.size < 4:
        return None

    t0 = float(t.mean())
    scale = float(max(np.ptp(t), 1e-6) / 2.0)
    tn_all = (t - t0) / scale
    keep = np.ones(t.size, dtype=bool)

    for _ in range(max_iter):
        coeffs = np.stack([np.polyfit(tn_all[keep], pos_km[keep, k], deg)
                           for k in range(3)])
        pred = np.stack([np.polyval(c, tn_all) for c in coeffs], axis=-1)
        err = np.linalg.norm(pred - pos_km, axis=1)

        med = float(np.median(err[keep]))
        mad = float(np.median(np.abs(err[keep] - med)))
        thr = max(reject_km, med + 4.0 * 1.4826 * mad)

        new_keep = err < thr
        if new_keep.sum() < deg + 1 or np.array_equal(new_keep, keep):
            break
        keep = new_keep

    tn = (t[keep] - t0) / scale
    coeffs = np.stack([np.polyfit(tn, pos_km[keep, k], deg) for k in range(3)])
    pred = np.stack([np.polyval(c, tn) for c in coeffs], axis=-1)
    resid = float(np.sqrt(np.mean(np.sum((pred - pos_km[keep]) ** 2, axis=1))))

    return ShortArc(t0=t0, scale=scale, coeffs=coeffs, n_used=int(keep.sum()),
                    residual_km=resid, t_start=float(t[keep].min()),
                    t_end=float(t[keep].max()))


def split_arcs(t: np.ndarray, gap_s: float = DEFAULT_ARC_GAP_S) -> list[slice]:
    """Index slices of a time-sorted array, cut wherever a gap exceeds gap_s."""
    if t.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) > gap_s) + 1
    edges = [0, *cuts.tolist(), t.size]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > 0]


@dataclass
class IraEphemeris:
    """Ephemeris source backed purely by decoded IRA frames.

    Deliberately mirrors what a TLE-backed source would expose, so the PNT
    solver never learns where its ephemeris came from: swapping this for SGP4
    (or the reverse) is a constructor change.
    """

    arcs: dict[int, list[ShortArc]] = field(default_factory=dict)

    @classmethod
    def from_fixes(cls, fixes: list[IraFix], *, deg: int = DEFAULT_ARC_DEG,
                   gap_s: float = DEFAULT_ARC_GAP_S,
                   reject_km: float = 12.0) -> IraEphemeris:
        arcs: dict[int, list[ShortArc]] = {}
        by_sat: dict[int, list[IraFix]] = {}
        for f in fixes:
            by_sat.setdefault(f.sat_id, []).append(f)
        for sat_id, group in by_sat.items():
            group.sort(key=lambda f: f.t)
            t = np.array([f.t for f in group])
            pos = np.stack([f.pos_km for f in group])
            fitted = []
            for sl in split_arcs(t, gap_s):
                arc = fit_short_arc(t[sl], pos[sl], deg=deg, reject_km=reject_km)
                if arc is not None:
                    fitted.append(arc)
            if fitted:
                arcs[sat_id] = fitted
        return cls(arcs=arcs)

    @property
    def satellites(self) -> list[int]:
        return sorted(self.arcs)

    def _arc_for(self, sat_id: int, t: float) -> ShortArc | None:
        """The arc covering t, else the nearest one (extrapolation is the
        caller's risk — check ``t_start``/``t_end`` if that matters)."""
        cand = self.arcs.get(sat_id)
        if not cand:
            return None
        for arc in cand:
            if arc.t_start <= t <= arc.t_end:
                return arc
        return min(cand, key=lambda a: min(abs(t - a.t_start), abs(t - a.t_end)))

    def states(self, sat_id: int, epochs) -> tuple[np.ndarray, np.ndarray]:
        """(pos, vel) in ECEF [km, km/s], shapes (N, 3).

        The satellite state does not depend on the observer, so the solver
        evaluates this once per epoch and then iterates over candidate positions
        with pure vector algebra.
        """
        epochs = np.atleast_1d(np.asarray(epochs, dtype=np.float64))
        pos = np.full((epochs.size, 3), np.nan)
        vel = np.full((epochs.size, 3), np.nan)
        for i, t in enumerate(epochs):
            arc = self._arc_for(sat_id, float(t))
            if arc is None:
                continue
            pos[i], vel[i] = arc.position(t), arc.velocity(t)
        return pos, vel
