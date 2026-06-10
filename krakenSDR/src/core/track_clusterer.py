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
    peak_indices: list[list[int]] = field(default_factory=list)

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
    ) -> None:
        self.max_gap_s = max_gap_s
        self.max_az_deg = max_az_deg
        self.max_el_deg = max_el_deg
        self.max_cfo_hz = max_cfo_hz
        self.min_track_len = min_track_len
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
        if t - tr.t_last > self.max_gap_s:
            return None
        d_az = _circ_az_sep(az, tr.az_last)
        d_el = abs(el - tr.el_last)
        d_cfo = abs(cfo_hz - tr.cfo_last)
        if d_az > self.max_az_deg or d_el > self.max_el_deg or d_cfo > self.max_cfo_hz:
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

    def finalize(self) -> tuple[list[dict], dict[tuple[int, int], int]]:
        """
        Close all active tracks, prune short ones to outlier id -1.

        Returns (tracks_list, peak_key -> track_id) where peak_key = (burst_idx, peak_idx).
        """
        all_tracks = self._archive + self._active
        self._active = []

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
) -> tuple[list[dict], list[dict]]:
    """
    Cluster burst records (each with peaks array and t, cfo_hz, burst_idx).

    Returns (tracks_json, bursts_with_track_ids) where each burst gets
    ``track_ids`` list parallel to peaks rows.
    """
    clusterer = TrackClusterer(
        max_gap_s=max_gap_s, min_track_len=min_track_len,
        max_az_deg=max_az_deg, max_el_deg=max_el_deg, max_cfo_hz=max_cfo_hz,
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
    )
    save_tracks_json(out_tracks_path, tracks, meta=meta)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in annotated:
            f.write(json.dumps(row) + "\n")
    return tracks
