#!/usr/bin/env python3
"""
Tests for core/track_clusterer.py (Doppler-trend gating + fragment merge)
and core/multiburst.py (TrackCovarianceEma).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from core.multiburst import TrackCovarianceEma
from core.track_clusterer import assign_tracks


def _pass_bursts(*, t0=0.0, duration=600.0, dt=1.0, az0=320.0, az1=40.0,
                 el_peak=55.0, cfo0=35_000.0, cfo1=-35_000.0,
                 burst_idx0=0, drop_windows=()):
    """Synthetic single-satellite pass as burst records (1 peak per burst)."""
    out = []
    bi = burst_idx0
    n = int(duration / dt)
    for k in range(n + 1):
        t = t0 + k * dt
        if any(a <= t - t0 <= b for a, b in drop_windows):
            continue
        frac = k / n
        az = (az0 + frac * ((az1 - az0) % 360.0)) % 360.0
        el = 5.0 + (el_peak - 5.0) * np.sin(np.pi * frac)
        cfo = cfo0 + frac * (cfo1 - cfo0)
        out.append({
            "burst_idx": bi,
            "t": t,
            "cfo_hz": cfo,
            "peaks": [[az, el, 0.0, 8.0]],
            "cfo_per_peak": [cfo],
        })
        bi += 1
    return out


class TestFragmentMerge:
    def test_gaps_merge_to_one_track(self):
        """Dropout gaps longer than max_gap_s must not split the pass."""
        bursts = _pass_bursts(drop_windows=[(200, 240), (400, 436)])
        tracks, _ = assign_tracks(bursts)
        long_tracks = [t for t in tracks if t["n_peaks"] >= 30]
        assert len(long_tracks) == 1, [t["n_peaks"] for t in tracks]
        assert long_tracks[0]["n_peaks"] == len(bursts)

    def test_merge_disabled(self):
        bursts = _pass_bursts(drop_windows=[(200, 240), (400, 436)])
        tracks, _ = assign_tracks(bursts, merge_gap_s=0.0)
        long_tracks = [t for t in tracks if t["n_peaks"] >= 30]
        assert len(long_tracks) == 3

    def test_concurrent_satellites_stay_separate(self):
        """Two interleaved satellites must not merge or steal peaks."""
        sat_a = _pass_bursts(az0=80.0, az1=140.0, cfo0=30_000.0,
                             cfo1=-30_000.0, burst_idx0=0)
        sat_b = _pass_bursts(az0=250.0, az1=300.0, cfo0=10_000.0,
                             cfo1=-38_000.0, burst_idx0=10_000)
        bursts = sorted(sat_a + sat_b, key=lambda b: (b["t"], b["burst_idx"]))
        tracks, _ = assign_tracks(bursts)
        long_tracks = [t for t in tracks if t["n_peaks"] >= 30]
        assert len(long_tracks) == 2
        sizes = sorted(t["n_peaks"] for t in long_tracks)
        assert sizes == [len(sat_a), len(sat_b)]

    def test_short_tracks_pruned(self):
        bursts = _pass_bursts(duration=3.0)      # only 4 bursts
        tracks, annotated = assign_tracks(bursts, min_track_len=5)
        assert tracks == []
        assert all(tid == -1 for b in annotated for tid in b["track_ids"])


class TestTrackCovarianceEma:
    def _y(self, rng, phase_seed: float, snr_amp=10.0):
        a = np.exp(1j * (phase_seed + np.arange(5)))
        return snr_amp * a + 0.1 * (rng.standard_normal(5)
                                    + 1j * rng.standard_normal(5))

    def test_interleaved_satellites_keep_separate_emas(self):
        rng = np.random.default_rng(0)
        ema = TrackCovarianceEma()
        ids_a, ids_b = set(), set()
        for k in range(40):
            t = float(k)
            # sat A: cfo 20k − 300·t, sat B: cfo −5k + 250·t
            _, tid_a, n_a = ema.update(self._y(rng, 0.3), 20_000.0 - 300 * t, t)
            _, tid_b, n_b = ema.update(self._y(rng, 2.1), -5_000.0 + 250 * t,
                                       t + 0.4)
            ids_a.add(tid_a)
            ids_b.add(tid_b)
        assert len(ids_a) == 1 and len(ids_b) == 1
        assert ids_a != ids_b
        assert n_a == 40 and n_b == 40

    def test_gap_starts_new_track(self):
        rng = np.random.default_rng(1)
        ema = TrackCovarianceEma(max_gap_s=10.0)
        _, tid1, _ = ema.update(self._y(rng, 0.5), 1_000.0, 0.0)
        _, tid2, _ = ema.update(self._y(rng, 0.5), 1_000.0, 30.0)
        assert tid1 != tid2

    def test_weak_y_not_blended(self):
        ema = TrackCovarianceEma()
        y_weak = 0.01 * np.ones(5, dtype=complex)
        R, tid, n = ema.update(y_weak, 0.0, 0.0)
        assert tid == -1 and n == 1
        np.testing.assert_allclose(R, np.outer(y_weak, y_weak.conj()))

    def test_ema_converges_to_signal_direction(self):
        rng = np.random.default_rng(3)
        ema = TrackCovarianceEma(alpha=0.9)
        a = np.exp(1j * np.arange(5))
        for k in range(50):
            R, _, _ = ema.update(self._y(rng, 1.0), 0.0, float(k))
        w, v = np.linalg.eigh(R)
        u1 = v[:, -1]
        corr = abs(np.vdot(u1, a / np.sqrt(5)))
        assert corr > 0.99