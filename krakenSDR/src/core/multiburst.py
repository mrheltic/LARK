"""
multiburst.py — Multi-burst covariance: beyond the rank-1 single-burst limit.

A single Iridium burst yields a rank-1 matched-filter covariance, for which
MUSIC, Capon and Bartlett share the same argmax.  Consecutive bursts of the
*same satellite* carry independent noise and — as the satellite moves —
decorrelating ground multipath, so averaging normalised MF vectors

    R = (1/B) Σ ŷ_j · ŷ_jᴴ        (ŷ = y / ‖y‖)

makes the noise subspace estimable.  Validated on the reference session
(B = 16, span ≤ 20 s): per-burst elevation MAD 5.5° → 4.6°, bias
−4.7° → −4.1°, azimuth unchanged.

Offline: :func:`collect_track_peaks` + :func:`reestimate` re-run DOA on
sliding-window covariances (used by ``reprocess_session.py --cov-bursts``
and ``scripts/multiburst_doa.py``).
Live: :class:`TrackCovarianceEma` keeps one covariance EMA per active
satellite (keyed by Doppler), so concurrent satellites no longer reset
each other's accumulation.
"""

from __future__ import annotations

import sys

import numpy as np

from core.doa_uca_2d import (
    UcaConfig,
    doa_bartlett_uca_2d,
    doa_capon_uca_2d,
    doa_music_uca_2d,
    find_peak_uca_2d,
)

__all__ = [
    "WEAK_Y_NORM2",
    "collect_track_peaks",
    "reestimate",
    "TrackCovarianceEma",
]

# compute_mf_covariance falls back to the sample covariance when the MF
# output is weak (‖y‖² ≤ 1% of the sample power, which is exactly n_ant for
# unit-RMS channels).  Those peaks are not reproducible from y alone and a
# weak y is noise-dominated, so they are excluded from window averages.
WEAK_Y_NORM2 = 0.01 * 5.0


# =============================================================================
# Offline: sliding-window covariance re-estimation
# =============================================================================

def collect_track_peaks(rows: list[dict]) -> dict[int, list[dict]]:
    """
    Group JSONL rows into {track_id: [peak dicts sorted by time]} with the
    MF vector attached.  Peaks without a track (id < 0) and weak-MF peaks
    are skipped (the latter keep their original single-burst estimate).
    """
    by_tid: dict[int, list[dict]] = {}
    n_missing_y = 0
    n_weak = 0
    for ri, row in enumerate(rows):
        tids = row.get("track_ids") or []
        cfo_pp = row.get("cfo_per_peak") or []
        y_pp = row.get("y_per_peak") or []
        for pi in range(len(row["peaks"])):
            tid = int(tids[pi]) if pi < len(tids) else -1
            if tid < 0:
                continue
            if pi >= len(y_pp):
                n_missing_y += 1
                continue
            y = np.array([complex(re, im) for re, im in y_pp[pi]])
            nrm2 = float(np.real(np.vdot(y, y)))
            if nrm2 <= WEAK_Y_NORM2:
                n_weak += 1
                continue
            by_tid.setdefault(tid, []).append({
                "row_idx": ri,
                "peak_idx": pi,
                "t": float(row["t"]),
                "cfo_hz": float(cfo_pp[pi]) if pi < len(cfo_pp)
                else float(row["cfo_hz"]),
                "y_hat": y / np.sqrt(nrm2),
            })
    if n_missing_y:
        sys.exit(f"{n_missing_y} peaks have no y_per_peak — re-run "
                 "reprocess_session.py on this session first.")
    if n_weak:
        print(f"Skipped {n_weak} weak-MF peaks (keep their original estimate)")
    for pts in by_tid.values():
        pts.sort(key=lambda p: p["t"])
    return by_tid


def reestimate(by_tid: dict[int, list[dict]], *, window: int, max_span_s: float,
               uca: UcaConfig, algo: str, mdl: bool = False) -> list[dict]:
    """
    One re-estimated peak per input peak: covariance averaged over the
    nearest `window` bursts of the same track (within max_span_s of the
    centre burst), then a fresh spectrum scan + peak pick.
    """
    uca.num_expected_signals = 0 if mdl else 1
    doa_fn = {"music": doa_music_uca_2d, "capon": doa_capon_uca_2d,
              "bartlett": doa_bartlett_uca_2d}[algo]
    X_dummy = np.empty((uca.n_ant, 0))
    out: list[dict] = []
    for pts in by_tid.values():
        n = len(pts)
        ts = np.array([p["t"] for p in pts])
        for i, p in enumerate(pts):
            # Nearest-by-index window centred on i, clamped at the ends.
            lo = max(0, min(i - window // 2, n - window))
            sel = range(lo, min(lo + window, n))
            ys = [pts[j]["y_hat"] for j in sel
                  if abs(ts[j] - ts[i]) <= max_span_s]
            Y = np.asarray(ys)                       # (B_eff, n_ant)
            R = Y.T @ np.conj(Y) / len(ys)           # (1/B) Σ y_j y_jᴴ
            spec = doa_fn(X_dummy, uca, R_in=R)
            az, el, papr = find_peak_uca_2d(spec, uca)
            out.append({**p, "az": az, "el": el, "papr_db": papr,
                        "n_window": len(ys)})
    return out


# =============================================================================
# Live: per-track covariance EMA
# =============================================================================

class _EmaTrack:
    __slots__ = ("track_id", "R", "n", "t_hist", "cfo_hist")

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.R: np.ndarray | None = None
        self.n = 0
        self.t_hist: list[float] = []
        self.cfo_hist: list[float] = []

    def predict_cfo(self, t: float) -> float:
        """Linear extrapolation of the Doppler trend (last few bursts)."""
        if len(self.t_hist) >= 3:
            ts = np.asarray(self.t_hist[-8:])
            cs = np.asarray(self.cfo_hist[-8:])
            # Degenerate timestamps (bursts in the same CPI) break lstsq.
            if float(ts[-1] - ts[0]) > 1e-3:
                slope, icpt = np.polyfit(ts - ts[0], cs, 1)
                return float(icpt + slope * (t - ts[0]))
        return self.cfo_hist[-1]


class TrackCovarianceEma:
    """
    One covariance EMA per active satellite, keyed by Doppler continuity.

    The live pipeline used to keep a single global R EMA, reset on any CFO
    jump > 3 kHz — so two satellites alternating in the same CPIs reset each
    other constantly and the EMA never integrated.  This registry matches
    each burst to an active track by *predicted* CFO (linear Doppler trend,
    Iridium slope ≤ ~400 Hz/s) and accumulates per track.

    alpha ≈ 0.94 gives an effective window of ~16 bursts — the optimum
    found offline on the reference session.
    """

    def __init__(self, *, alpha: float = 0.94, max_cfo_resid_hz: float = 1_500.0,
                 max_gap_s: float = 10.0, max_tracks: int = 8):
        self.alpha = alpha
        self.max_cfo_resid_hz = max_cfo_resid_hz
        self.max_gap_s = max_gap_s
        self.max_tracks = max_tracks
        self._tracks: list[_EmaTrack] = []
        self._next_id = 1

    def update(self, y: np.ndarray, cfo_hz: float, t: float,
               ) -> tuple[np.ndarray, int, int]:
        """
        Fold one burst into its track's EMA.

        Parameters: MF output vector ``y`` (n_ant complex, any scale),
        burst Doppler [Hz], wall-clock time [s].

        Returns (R_ema, track_id, n_bursts_in_track).  Weak y (sample-
        covariance fallback in compute_mf_covariance) is not blended:
        the rank-1 outer product is returned as-is with track_id −1.
        """
        y = np.asarray(y)
        nrm2 = float(np.real(np.vdot(y, y)))
        if nrm2 <= WEAK_Y_NORM2:
            return np.outer(y, y.conj()), -1, 1
        y_hat = y / np.sqrt(nrm2)
        R1 = np.outer(y_hat, y_hat.conj())

        # Retire stale tracks.
        self._tracks = [tr for tr in self._tracks
                        if t - tr.t_hist[-1] <= self.max_gap_s]

        best: _EmaTrack | None = None
        best_resid = self.max_cfo_resid_hz
        for tr in self._tracks:
            resid = abs(cfo_hz - tr.predict_cfo(t))
            if resid < best_resid:
                best_resid = resid
                best = tr

        if best is None:
            best = _EmaTrack(self._next_id)
            self._next_id += 1
            self._tracks.append(best)
            if len(self._tracks) > self.max_tracks:
                self._tracks.pop(0)

        best.R = R1 if best.R is None else (
            self.alpha * best.R + (1.0 - self.alpha) * R1)
        best.n += 1
        best.t_hist.append(t)
        best.cfo_hist.append(cfo_hz)
        if len(best.t_hist) > 16:
            best.t_hist = best.t_hist[-16:]
            best.cfo_hist = best.cfo_hist[-16:]
        return best.R, best.track_id, best.n
