#!/usr/bin/env python3
"""
Tests for core/track_filter.py — Kalman + RTS smoothing of DOA tracks.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from core.track_filter import (
    azel_to_unit,
    kalman_smooth_track,
    kalman_smooth_track_robust,
    unit_to_azel,
)

RNG = np.random.default_rng(42)


def _angular_error_deg(az_a, el_a, az_b, el_b) -> np.ndarray:
    """Great-circle separation [deg] between two az/el series."""
    ua = azel_to_unit(az_a, el_a)
    ub = azel_to_unit(az_b, el_b)
    dots = np.clip(np.sum(ua * ub, axis=-1), -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def _synthetic_pass(n=300, az0=320.0, az1=40.0, el_peak=55.0, duration=300.0):
    """Smooth pass crossing North: az 320°→40°, el rising to el_peak and back."""
    t = np.linspace(0.0, duration, n)
    frac = t / duration
    az_true = (az0 + frac * ((az1 - az0) % 360.0)) % 360.0
    el_true = 5.0 + (el_peak - 5.0) * np.sin(np.pi * frac)
    return t, az_true, el_true


def _add_noise(az, el, s_az=4.0, s_el=6.0):
    az_n = (az + RNG.normal(0.0, s_az, len(az))) % 360.0
    el_n = np.clip(el + RNG.normal(0.0, s_el, len(el)), 0.0, 90.0)
    return az_n, el_n


class TestConversions:
    def test_roundtrip(self):
        az = np.array([0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
        el = np.array([0.0, 10.0, 30.0, 60.0, 85.0, 5.0])
        az2, el2 = unit_to_azel(azel_to_unit(az, el))
        np.testing.assert_allclose(az2, az, atol=1e-9)
        np.testing.assert_allclose(el2, el, atol=1e-9)

    def test_compass_convention(self):
        # az 90° = East, el 0° → unit vector (1, 0, 0) in ENU
        u = azel_to_unit(90.0, 0.0)
        np.testing.assert_allclose(u, [1.0, 0.0, 0.0], atol=1e-12)


class TestKalmanSmoothTrack:
    def test_reduces_noise(self):
        t, az_true, el_true = _synthetic_pass()
        az_n, el_n = _add_noise(az_true, el_true)
        # sigma_acc matched to the synthetic dynamics (peak ~1e-4 rad/s²)
        sm = kalman_smooth_track(t, az_n, el_n,
                                 sigma_az_deg=4.0, sigma_el_deg=6.0,
                                 sigma_acc=1e-4)
        err_raw = _angular_error_deg(az_n, el_n, az_true, el_true)
        err_sm = _angular_error_deg(sm.az_deg, sm.el_deg, az_true, el_true)
        rms_raw = np.sqrt(np.mean(err_raw**2))
        rms_sm = np.sqrt(np.mean(err_sm**2))
        assert rms_sm < rms_raw / 2.5, (rms_raw, rms_sm)

    def test_north_wrap_continuous(self):
        """Smoothed track must cross 0°/360° without artifacts."""
        t, az_true, el_true = _synthetic_pass()
        az_n, el_n = _add_noise(az_true, el_true)
        sm = kalman_smooth_track(t, az_n, el_n,
                                 sigma_az_deg=4.0, sigma_el_deg=6.0,
                                 sigma_acc=1e-4)
        err = _angular_error_deg(sm.az_deg, sm.el_deg, az_true, el_true)
        near_north = (az_true < 15.0) | (az_true > 345.0)
        assert near_north.any()
        assert np.max(err[near_north]) < 5.0

    def test_near_zenith_no_nan(self):
        t, az_true, el_true = _synthetic_pass(el_peak=88.0)
        az_n, el_n = _add_noise(az_true, el_true, s_az=3.0, s_el=3.0)
        sm = kalman_smooth_track(t, az_n, el_n)
        for arr in (sm.az_deg, sm.el_deg, sm.sigma_az_deg, sm.sigma_el_deg):
            assert np.all(np.isfinite(arr))
        assert np.max(sm.el_deg) <= 90.0

    def test_gap_in_track(self):
        """A long measurement gap must not destabilise the smoother."""
        t, az_true, el_true = _synthetic_pass(n=300)
        keep = (t < 100.0) | (t > 200.0)
        t, az_true, el_true = t[keep], az_true[keep], el_true[keep]
        az_n, el_n = _add_noise(az_true, el_true)
        sm = kalman_smooth_track(t, az_n, el_n,
                                 sigma_az_deg=4.0, sigma_el_deg=6.0,
                                 sigma_acc=1e-4)
        err = _angular_error_deg(sm.az_deg, sm.el_deg, az_true, el_true)
        assert np.all(np.isfinite(err))
        assert np.sqrt(np.mean(err**2)) < 4.0

    def test_uncertainty_shrinks_vs_measurement(self):
        t, az_true, el_true = _synthetic_pass()
        az_n, el_n = _add_noise(az_true, el_true)
        sm = kalman_smooth_track(t, az_n, el_n,
                                 sigma_az_deg=4.0, sigma_el_deg=6.0)
        # Posterior 1σ in mid-track must be well below measurement noise.
        mid = slice(50, 250)
        assert np.median(sm.sigma_el_deg[mid]) < 6.0 / 2.0
        assert np.median(sm.sigma_az_deg[mid]) < 4.0 / np.cos(
            np.deg2rad(np.max(el_true))) / 2.0

    def test_short_track_passthrough(self):
        t = np.array([0.0, 1.0])
        sm = kalman_smooth_track(t, np.array([10.0, 12.0]),
                                 np.array([20.0, 21.0]))
        np.testing.assert_allclose(sm.az_deg, [10.0, 12.0], atol=1e-9)
        np.testing.assert_allclose(sm.el_deg, [20.0, 21.0], atol=1e-9)

    def test_outliers_rejected(self):
        """5% gross outliers must not pull the robust smoothed track."""
        t, az_true, el_true = _synthetic_pass()
        az_n, el_n = _add_noise(az_true, el_true)
        idx = RNG.choice(len(t), size=len(t) // 20, replace=False)
        az_out, el_out = az_n.copy(), el_n.copy()
        el_out[idx] = np.clip(el_out[idx] + RNG.choice([-40, 40], len(idx)),
                              0.0, 90.0)
        kw = dict(sigma_az_deg=4.0, sigma_el_deg=6.0, sigma_acc=1e-4)
        clean = kalman_smooth_track_robust(t, az_n, el_n, **kw)
        dirty = kalman_smooth_track_robust(t, az_out, el_out, **kw)
        # The contaminated solution stays close to the clean one...
        drift = _angular_error_deg(dirty.az_deg, dirty.el_deg,
                                   clean.az_deg, clean.el_deg)
        assert np.median(drift) < 1.0
        # ...and most injected outliers are flagged.
        assert dirty.outlier is not None
        assert dirty.outlier[idx].mean() > 0.7

    def test_smoothness(self):
        """Second differences of the smoothed track ≪ those of raw data."""
        t, az_true, el_true = _synthetic_pass()
        az_n, el_n = _add_noise(az_true, el_true)
        sm = kalman_smooth_track_robust(t, az_n, el_n,
                                        sigma_az_deg=4.0, sigma_el_deg=6.0)
        rough_raw = np.sqrt(np.mean(np.diff(el_n, 2) ** 2))
        rough_sm = np.sqrt(np.mean(np.diff(sm.el_deg, 2) ** 2))
        assert rough_sm < rough_raw / 10.0, (rough_raw, rough_sm)

    def test_weights_downweight_bad_points(self):
        """Low-weight noisy points must influence the fit less."""
        t, az_true, el_true = _synthetic_pass()
        az_n, el_n = _add_noise(az_true, el_true)
        idx = np.arange(100, 150)
        el_w = el_n.copy()
        el_w[idx] = np.clip(el_w[idx] + RNG.normal(0, 15, len(idx)), 0, 90)
        w = np.ones(len(t))
        w[idx] = 0.05
        kw = dict(sigma_az_deg=4.0, sigma_el_deg=6.0,
                  sigma_acc=1e-4, gate_chi2=None)
        sm_w = kalman_smooth_track(t, az_n, el_w, weights=w, **kw)
        sm_nw = kalman_smooth_track(t, az_n, el_w, **kw)
        err_w = _angular_error_deg(sm_w.az_deg, sm_w.el_deg,
                                   az_true, el_true)[idx]
        err_nw = _angular_error_deg(sm_nw.az_deg, sm_nw.el_deg,
                                    az_true, el_true)[idx]
        assert np.mean(err_w) < np.mean(err_nw)

    def test_non_increasing_time_raises(self):
        t = np.array([0.0, 1.0, 1.0, 2.0])
        try:
            kalman_smooth_track(t, np.zeros(4) + 10.0, np.zeros(4) + 20.0)
        except ValueError:
            return
        raise AssertionError("expected ValueError for non-increasing t")
