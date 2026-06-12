"""
track_clusterer.py — Online association of multi-peak DOA bursts into satellite tracks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "TrackClusterer",
    "assign_tracks",
    "load_tracks_json",
    "save_tracks_json",
    "cluster_from_jsonl",
]


def _circ_az_sep(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _predict_cfo(t_hist: list[float], cfo_hist: list[float], t: float) -> float:
    """
    Robust linear extrapolation of the Doppler trend.  Iridium CFO is locally
    linear (slope ≤ ~400 Hz/s), so predicting it makes a far tighter gate
    than the raw |Δcfo| bound — the key to not mixing concurrent satellites.

    Theil–Sen slope (median of pairwise slopes): real tracks contain
    occasional kHz-level CFO outliers that would corrupt a least-squares fit
    and then split the track on the next prediction.
    """
    if len(t_hist) >= 3:
        ts = np.asarray(t_hist[-8:])
        cs = np.asarray(cfo_hist[-8:])
        if float(ts[-1] - ts[0]) > 1e-3:
            slopes = []
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    if ts[j] - ts[i] > 1e-3:
                        slopes.append((cs[j] - cs[i]) / (ts[j] - ts[i]))
            slope = float(np.median(slopes))
            icpt = float(np.median(cs - slope * ts))
            return icpt + slope * t
    return cfo_hist[-1]


def _cfo_gate_hz(dt: float) -> float:
    """
    Allowed |cfo − predicted| [Hz] after a gap of dt seconds.

    Base term covers measurement jitter (p99 ≈ 600 Hz on real tracks);
    quadratic term covers Doppler curvature (≈ 8 Hz/s² near the
    zero-crossing) that the linear prediction cannot follow.
    """
    return 800.0 + 4.0 * dt * dt


@dataclass
class _ActiveTrack:
    track_id: int
    az_last: float
    el_last: float
    cfo_last: float
    t_last: float
    n_peaks: int = 0
    az_min: float = 0.0
    az_max: float = 0.0
    el_min: float = 0.0
    el_max: float = 0.0
    cfo_min: float = 0.0
    cfo_max: float = 0.0
    t_start: float = 0.0
    az_first: float = 0.0
    el_first: float = 0.0
    cfo_first: float = 0.0
    peak_indices: list[list[int]] = field(default_factory=list)
    # Recent (t, cfo) tail for the Doppler-trend prediction
    t_hist: list[float] = field(default_factory=list)
    cfo_hist: list[float] = field(default_factory=list)
    # Full history — used by the merge pass to recognise tracks of the same
    # satellite (a satellite transmits on several frequency accesses, each
    # of which becomes its own track during online association).
    t_all: list[float] = field(default_factory=list)
    cfo_all: list[float] = field(default_factory=list)
    az_all: list[float] = field(default_factory=list)
    el_all: list[float] = field(default_factory=list)

    def append(
        self,
        burst_idx: int,
        peak_idx: int,
        az: float,
        el: float,
        cfo_hz: float,
        t: float,
    ) -> None:
        if self.n_peaks == 0:
            self.t_start = t
            self.az_first, self.el_first, self.cfo_first = az, el, cfo_hz
            self.az_min = self.az_max = az
            self.el_min = self.el_max = el
            self.cfo_min = self.cfo_max = cfo_hz
        else:
            self.az_min = min(self.az_min, az)
            self.az_max = max(self.az_max, az)
            self.el_min = min(self.el_min, el)
            self.el_max = max(self.el_max, el)
            self.cfo_min = min(self.cfo_min, cfo_hz)
            self.cfo_max = max(self.cfo_max, cfo_hz)
        self.az_last = az
        self.el_last = el
        self.cfo_last = cfo_hz
        self.t_last = t
        self.n_peaks += 1
        self.peak_indices.append([burst_idx, peak_idx])
        self.t_hist.append(t)
        self.cfo_hist.append(cfo_hz)
        if len(self.t_hist) > 8:
            self.t_hist = self.t_hist[-8:]
            self.cfo_hist = self.cfo_hist[-8:]
        self.t_all.append(t)
        self.cfo_all.append(cfo_hz)
        self.az_all.append(az)
        self.el_all.append(el)

    def absorb(self, other: "_ActiveTrack") -> None:
        """Merge another track of the same pass into this one."""
        self.az_min = min(self.az_min, other.az_min)
        self.az_max = max(self.az_max, other.az_max)
        self.el_min = min(self.el_min, other.el_min)
        self.el_max = max(self.el_max, other.el_max)
        self.cfo_min = min(self.cfo_min, other.cfo_min)
        self.cfo_max = max(self.cfo_max, other.cfo_max)
        if other.t_last >= self.t_last:
            self.az_last, self.el_last = other.az_last, other.el_last
            self.cfo_last, self.t_last = other.cfo_last, other.t_last
            self.t_hist = other.t_hist
            self.cfo_hist = other.cfo_hist
        self.t_start = min(self.t_start, other.t_start)
        self.n_peaks += other.n_peaks
        self.peak_indices.extend(other.peak_indices)
        merged = sorted(zip(self.t_all + other.t_all,
                            self.cfo_all + other.cfo_all,
                            self.az_all + other.az_all,
                            self.el_all + other.el_all))
        self.t_all = [m[0] for m in merged]
        self.cfo_all = [m[1] for m in merged]
        self.az_all = [m[2] for m in merged]
        self.el_all = [m[3] for m in merged]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.track_id,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_last, 3),
            "n_peaks": self.n_peaks,
            "az_range": [round(self.az_min, 1), round(self.az_max, 1)],
            "el_range": [round(self.el_min, 1), round(self.el_max, 1)],
            "cfo_range_hz": [round(self.cfo_min, 0), round(self.cfo_max, 0)],
            "peak_indices": self.peak_indices,
        }


class TrackClusterer:
    """Greedy online peak-to-track association."""

    def __init__(
        self,
        *,
        max_gap_s: float = 30.0,
        max_az_deg: float = 25.0,
        max_el_deg: float = 15.0,
        max_cfo_hz: float = 12000.0,
        min_track_len: int = 5,
        merge_gap_s: float = 60.0,
    ) -> None:
        self.max_gap_s = max_gap_s
        self.max_az_deg = max_az_deg
        self.max_el_deg = max_el_deg
        self.max_cfo_hz = max_cfo_hz
        self.min_track_len = min_track_len
        self.merge_gap_s = merge_gap_s  # 0 disables fragment merging
        self.merge_max_sep_deg = 12.0   # same-sat trajectory agreement gate
        # Merging time-OVERLAPPING tracks is off by default: parallel tracks
        # of the same satellite are usually different frequency accesses or
        # multipath ghosts (same Doppler!), and absorbing the ghost-laden
        # ones measurably degrades the per-track statistics (validated on
        # the reference session: smoothed az bias 0° → −3°).
        self.merge_overlapping = False
        self._next_id = 1
        self._active: list[_ActiveTrack] = []
        self._archive: list[_ActiveTrack] = []

    def _retire_stale(self, t_now: float) -> None:
        still: list[_ActiveTrack] = []
        for tr in self._active:
            if t_now - tr.t_last > self.max_gap_s:
                self._archive.append(tr)
            else:
                still.append(tr)
        self._active = still

    def _match_cost(
        self,
        tr: _ActiveTrack,
        az: float,
        el: float,
        cfo_hz: float,
        t: float,
    ) -> float | None:
        dt = t - tr.t_last
        if dt > self.max_gap_s:
            return None
        d_az = _circ_az_sep(az, tr.az_last)
        d_el = abs(el - tr.el_last)
        # Gate on the *predicted* Doppler when the trend is established —
        # much tighter than |Δcfo| and the real discriminator between
        # concurrent satellites.
        d_cfo = abs(cfo_hz - _predict_cfo(tr.t_hist, tr.cfo_hist, t))
        cfo_gate = min(_cfo_gate_hz(dt), self.max_cfo_hz) \
            if len(tr.t_hist) >= 3 else self.max_cfo_hz
        if d_az > self.max_az_deg or d_el > self.max_el_deg or d_cfo > cfo_gate:
            return None
        return (d_az / 8.0) ** 2 + (d_el / 5.0) ** 2 + (d_cfo / 2000.0) ** 2

    def assign_burst(
        self,
        burst_idx: int,
        t: float,
        cfo_hz: float,
        peaks: np.ndarray,
        cfo_per_peak: list[float] | None = None,
    ) -> list[int]:
        """
        Assign track_id per peak row. Returns list of track ids (same order as peaks).

        ``cfo_per_peak``: if provided (per-tone architecture), use the per-peak CFO
        for matching instead of the shared burst CFO.
        """
        self._retire_stale(t)
        track_ids: list[int] = []
        peaks = np.asarray(peaks)
        if peaks.ndim != 2 or peaks.shape[0] == 0:
            return track_ids

        for pi in range(peaks.shape[0]):
            az, el = float(peaks[pi, 0]), float(peaks[pi, 1])
            peak_cfo = float(cfo_per_peak[pi]) if cfo_per_peak and pi < len(cfo_per_peak) else cfo_hz
            best_tr: _ActiveTrack | None = None
            best_cost = float("inf")
            for tr in self._active:
                cost = self._match_cost(tr, az, el, peak_cfo, t)
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_tr = tr
            if best_tr is not None:
                best_tr.append(burst_idx, pi, az, el, peak_cfo, t)
                track_ids.append(best_tr.track_id)
            else:
                tr = _ActiveTrack(
                    track_id=self._next_id,
                    az_last=az,
                    el_last=el,
                    cfo_last=peak_cfo,
                    t_last=t,
                )
                tr.append(burst_idx, pi, az, el, peak_cfo, t)
                self._active.append(tr)
                track_ids.append(self._next_id)
                self._next_id += 1
        return track_ids

    def _same_satellite(self, a: _ActiveTrack, b: _ActiveTrack) -> bool:
        """
        Same-pass test for the merge step (b.t_start >= a.t_start assumed).

        A satellite transmits on several frequency accesses (CFO offsets of
        a few kHz), and the online Doppler gate correctly keeps each access
        in its own track — so a pass shows up as parallel tracks.  Two
        time-overlapping tracks are the same satellite iff, at near-in-time
        points, (1) their az/el trajectories coincide (median great-circle
        separation ≈ error level) AND (2) their cfo(t) curves are *parallel*
        (the access offset is constant, so the spread of the difference is
        small even when the offset itself is kHz).  Requiring both keeps out
        ghost tracks that share rough direction but have incoherent Doppler.
        Sequential tracks (gap) are fragments of the same pass iff the
        junction continues the Doppler trend and az/el position.
        """
        gap = b.t_start - a.t_last
        if gap > self.merge_gap_s:
            return False
        if gap <= 0.0:                                  # time overlap
            if not self.merge_overlapping:
                return False
            ta = np.asarray(a.t_all)
            tb = np.asarray(b.t_all)
            idx = np.searchsorted(ta, tb).clip(1, len(ta) - 1)
            near = np.minimum(np.abs(ta[idx] - tb), np.abs(ta[idx - 1] - tb))
            sel = near <= 5.0
            if sel.sum() >= 5:
                az_a = np.interp(tb[sel],
                                 ta, np.unwrap(np.deg2rad(a.az_all)))
                el_a = np.deg2rad(np.interp(tb[sel], ta, a.el_all))
                az_b = np.deg2rad(np.asarray(b.az_all)[sel])
                el_b = np.deg2rad(np.asarray(b.el_all)[sel])
                cos_sep = (np.sin(el_a) * np.sin(el_b)
                           + np.cos(el_a) * np.cos(el_b)
                           * np.cos(az_a - az_b))
                sep = np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
                if float(np.median(sep)) >= self.merge_max_sep_deg:
                    return False
                dcfo = (np.interp(tb[sel], ta, a.cfo_all)
                        - np.asarray(b.cfo_all)[sel])
                mad = float(np.median(np.abs(dcfo - np.median(dcfo))))
                return mad < 1_200.0
            gap = 0.0                                   # tiny overlap: junction test
        d_cfo = abs(b.cfo_first
                    - _predict_cfo(a.t_hist, a.cfo_hist, b.t_start))
        if d_cfo > _cfo_gate_hz(gap):
            return False
        if _circ_az_sep(b.az_first, a.az_last) > self.max_az_deg + 1.0 * gap:
            return False
        return abs(b.el_first - a.el_last) <= self.max_el_deg + 0.5 * gap

    def _merge_fragments(self, tracks: list[_ActiveTrack]) -> list[_ActiveTrack]:
        """
        Merge tracks that belong to the same satellite pass: parallel
        duplicates (spawned by DOA outliers) and sequential fragments
        (detection dropouts).  Before this, the reference session had 78
        tracks for ~12 passes, several of them time-overlapping copies.
        """
        tracks = sorted(tracks, key=lambda tr: tr.t_start)
        merged: list[_ActiveTrack] = []
        for tr in tracks:
            host = None
            for cand in merged:
                if self._same_satellite(cand, tr):
                    host = cand
                    break
            if host is not None:
                host.absorb(tr)
            else:
                merged.append(tr)
        return merged

    def finalize(self) -> tuple[list[dict], dict[tuple[int, int], int]]:
        """
        Close all active tracks, merge pass fragments, prune short ones to
        outlier id -1.

        Returns (tracks_list, peak_key -> track_id) where peak_key = (burst_idx, peak_idx).
        """
        all_tracks = self._archive + self._active
        self._active = []
        if self.merge_gap_s > 0.0:
            all_tracks = self._merge_fragments(all_tracks)

        valid: list[_ActiveTrack] = []
        outlier_keys: set[tuple[int, int]] = set()
        for tr in all_tracks:
            if tr.n_peaks < self.min_track_len:
                for bi, pi in tr.peak_indices:
                    outlier_keys.add((bi, pi))
            else:
                valid.append(tr)

        peak_map: dict[tuple[int, int], int] = {}
        tracks_out: list[dict] = []
        for tr in valid:
            tracks_out.append(tr.to_dict())
            for bi, pi in tr.peak_indices:
                peak_map[(bi, pi)] = tr.track_id
        for bi, pi in outlier_keys:
            peak_map[(bi, pi)] = -1

        return tracks_out, peak_map


def assign_tracks(
    bursts: list[dict],
    *,
    max_gap_s: float = 30.0,
    min_track_len: int = 5,
    max_az_deg: float = 25.0,
    max_el_deg: float = 15.0,
    max_cfo_hz: float = 12000.0,
    merge_gap_s: float = 60.0,
) -> tuple[list[dict], list[dict]]:
    """
    Cluster burst records (each with peaks array and t, cfo_hz, burst_idx).

    Returns (tracks_json, bursts_with_track_ids) where each burst gets
    ``track_ids`` list parallel to peaks rows.
    """
    clusterer = TrackClusterer(
        max_gap_s=max_gap_s, min_track_len=min_track_len,
        max_az_deg=max_az_deg, max_el_deg=max_el_deg, max_cfo_hz=max_cfo_hz,
        merge_gap_s=merge_gap_s,
    )
    out_bursts: list[dict] = []
    for b in sorted(bursts, key=lambda x: float(x.get("t", 0))):
        burst_idx = int(b["burst_idx"])
        peaks = np.asarray(b["peaks"])
        cfo_per_peak = b.get("cfo_per_peak") or None
        tids = clusterer.assign_burst(
            burst_idx, float(b["t"]), float(b["cfo_hz"]), peaks,
            cfo_per_peak=cfo_per_peak,
        )
        row = dict(b)
        row["track_ids"] = tids
        out_bursts.append(row)

    tracks, peak_map = clusterer.finalize()
    for b in out_bursts:
        bi = int(b["burst_idx"])
        peaks = np.asarray(b["peaks"])
        b["track_ids"] = [
            peak_map.get((bi, pi), -1)
            for pi in range(peaks.shape[0])
        ]
    return tracks, out_bursts


def save_tracks_json(path: str, tracks: list[dict], meta: dict | None = None) -> None:
    payload = {"tracks": tracks}
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_tracks_json(path: str) -> tuple[list[dict], dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("tracks", [])), dict(data.get("meta", {}))


def cluster_from_jsonl(
    jsonl_path: str,
    out_tracks_path: str,
    *,
    max_gap_s: float = 30.0,
    min_track_len: int = 5,
    max_az_deg: float = 25.0,
    max_el_deg: float = 15.0,
    max_cfo_hz: float = 12000.0,
    merge_gap_s: float = 60.0,
    meta: dict | None = None,
) -> list[dict]:
    """Load doa_multi.jsonl, assign tracks, write tracks.json, update jsonl rows."""
    bursts: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                bursts.append(json.loads(line))

    tracks, annotated = assign_tracks(
        bursts, max_gap_s=max_gap_s, min_track_len=min_track_len,
        max_az_deg=max_az_deg, max_el_deg=max_el_deg, max_cfo_hz=max_cfo_hz,
        merge_gap_s=merge_gap_s,
    )
    save_tracks_json(out_tracks_path, tracks, meta=meta)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in annotated:
            f.write(json.dumps(row) + "\n")
    return tracks
