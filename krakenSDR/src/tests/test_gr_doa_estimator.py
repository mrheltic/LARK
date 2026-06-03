#!/usr/bin/env python3
"""
test_gr_doa_estimator.py — Unit tests for UCA 2D DOA estimation (core Python, no GR).

Tests doa_music_uca_2d, doa_bartlett_uca_2d, doa_capon_uca_2d,
find_peak_uca_2d, doa_phase_fit_uca_2d from core.doa_uca_2d.

Run:
    python3 -m pytest krakenSDR/src/tests/test_gr_doa_estimator.py -v
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pytest

from core.doa_uca_2d import (
    UcaConfig,
    doa_music_uca_2d,
    doa_bartlett_uca_2d,
    doa_capon_uca_2d,
    doa_phase_fit_uca_2d,
    find_peak_uca_2d,
    amplitude_normalize_channels,
    eigenvalue_spread_uca_db,
    snr_uca_db,
)

N_ANT = 5
R_LAMBDA = 0.4253
FS = 1_024_000.0
TONE_HZ = 3_125.0


def _uca_config(n_az=360, n_el=86, el_min=5.0, el_max=90.0):
    return UcaConfig(
        n_ant=N_ANT, radius_lambda=R_LAMBDA,
        n_az=n_az, n_el=n_el,
        el_min_deg=el_min, el_max_deg=el_max,
        num_expected_signals=1,
        ant0_offset_deg=0.0, ant_ccw=False,
    )


def _synth_uca_signal(cfg, az_deg, el_deg, snr_db=20.0, n_snap=2621,
                       tone_hz=TONE_HZ, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    k = np.arange(cfg.n_ant)
    phi = 2 * np.pi * k / cfg.n_ant
    az_r = np.deg2rad(az_deg)
    el_r = np.deg2rad(el_deg)
    tau = 2 * np.pi * cfg.radius_lambda * np.cos(el_r) * np.cos(az_r - phi)
    a = np.exp(1j * tau)

    amp = 10 ** (snr_db / 20.0)
    t = np.arange(n_snap, dtype=np.float64)
    cw = amp * np.exp(2j * np.pi * tone_hz / FS * t)
    nse = (rng.standard_normal((cfg.n_ant, n_snap)) + 1j * rng.standard_normal((cfg.n_ant, n_snap))) / np.sqrt(2)
    return (np.outer(a, cw) + nse).astype(np.complex64)


def _angular_error(est, true):
    d = abs(est - true) % 360
    return min(d, 360.0 - d)


class TestDOAMUSIC2D:

    @pytest.mark.parametrize("az,el", [(0, 20), (45, 30), (90, 45), (180, 25), (270, 60), (355, 15)])
    def test_azimuth_accuracy(self, az, el):
        cfg = _uca_config()
        X = _synth_uca_signal(cfg, az, el, snr_db=25.0)
        X_bpf = amplitude_normalize_channels(X)
        R = (X_bpf @ X_bpf.conj().T) / X_bpf.shape[1]
        spec = doa_music_uca_2d(X_bpf, cfg, R_in=R)
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        az_err = _angular_error(az_est, az)
        assert az_err < 10.0, f"az={az}°: est={az_est:.1f}°, err={az_err:.1f}°"

    def test_elevation_accuracy(self):
        cfg = _uca_config()
        for el in [15, 30, 45, 60]:
            X = _synth_uca_signal(cfg, 90.0, el, snr_db=25.0)
            X_bpf = amplitude_normalize_channels(X)
            R = (X_bpf @ X_bpf.conj().T) / X_bpf.shape[1]
            spec = doa_music_uca_2d(X_bpf, cfg, R_in=R)
            _, el_est, _ = find_peak_uca_2d(spec, cfg)
            assert abs(el_est - el) < 15.0, f"el={el}°: est={el_est:.1f}°"


class TestDOABartlett2D:

    def test_bartlett_finds_peak(self):
        cfg = _uca_config(n_az=72, n_el=18)
        X = _synth_uca_signal(cfg, 45.0, 30.0, snr_db=25.0)
        X_bpf = amplitude_normalize_channels(X)
        R = (X_bpf @ X_bpf.conj().T) / X_bpf.shape[1]
        spec = doa_bartlett_uca_2d(X_bpf, cfg, R_in=R)
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        assert _angular_error(az_est, 45.0) < 15.0


class TestDOACapon2D:

    def test_capon_finds_peak(self):
        cfg = _uca_config(n_az=72, n_el=18)
        X = _synth_uca_signal(cfg, 120.0, 40.0, snr_db=25.0)
        X_bpf = amplitude_normalize_channels(X)
        R = (X_bpf @ X_bpf.conj().T) / X_bpf.shape[1]
        spec = doa_capon_uca_2d(X_bpf, cfg, R_in=R, decorr="none")
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        assert _angular_error(az_est, 120.0) < 15.0


class TestDOAPhaseFit2D:

    def test_phase_fit_with_hint(self):
        cfg = _uca_config(n_az=72, n_el=18)
        X = _synth_uca_signal(cfg, 90.0, 30.0, snr_db=20.0)
        X_bpf = amplitude_normalize_channels(X)
        R = (X_bpf @ X_bpf.conj().T) / X_bpf.shape[1]
        az_est, el_est, ph_err = doa_phase_fit_uca_2d(R, cfg, az_hint_deg=90.0, el_hint_deg=30.0)
        assert _angular_error(az_est, 90.0) < 15.0


class TestDOAQualityMetrics:

    def test_eigenvalue_spread(self):
        cfg = _uca_config()
        X = _synth_uca_signal(cfg, 90.0, 30.0, snr_db=25.0)
        X_bpf = amplitude_normalize_channels(X)
        R = (X_bpf @ X_bpf.conj().T) / X_bpf.shape[1]
        ev_spread = eigenvalue_spread_uca_db(R)
        assert np.max(ev_spread) > 5.0, f"Eigenvalue spread too low: {np.max(ev_spread):.1f} dB"

    def test_snr_estimate(self):
        cfg = _uca_config()
        X = _synth_uca_signal(cfg, 90.0, 30.0, snr_db=20.0)
        X_bpf = amplitude_normalize_channels(X)
        R = (X_bpf @ X_bpf.conj().T) / X_bpf.shape[1]
        snr = snr_uca_db(R, n_sources=1)
        assert snr > 0.0, f"SNR should be positive, got {snr:.1f}"


class TestDOASweep:

    def test_snr_sweep(self):
        cfg = _uca_config(n_az=72, n_el=18)
        for snr_db in [30, 20, 10]:
            X = _synth_uca_signal(cfg, 135.0, 30.0, snr_db=snr_db)
            X_bpf = amplitude_normalize_channels(X)
            R = (X_bpf @ X_bpf.conj().T) / X_bpf.shape[1]
            spec = doa_music_uca_2d(X_bpf, cfg, R_in=R)
            az_est, _, _ = find_peak_uca_2d(spec, cfg)
            err = _angular_error(az_est, 135.0)
            assert err < 20.0, f"SNR={snr_db}dB: err={err:.1f}°"