#!/usr/bin/env python3
"""
test_burst_bpf.py — Unit tests for BPF extraction + normalization.

Tests apply_bpf_and_normalize() from core.burst_processing.

Run:
    python3 -m pytest krakenSDR/src/tests/test_burst_bpf.py -v
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

from core.burst_processing import apply_bpf_and_normalize

FS = 1_024_000.0
TONE_HZ = 3_125.0
N_ANT = 5


def _steering_vector(az_rad, el_rad, n_ant=N_ANT, radius_lambda=0.4253):
    k = np.arange(n_ant)
    phi = 2 * np.pi * k / n_ant
    tau = 2 * np.pi * radius_lambda * np.cos(el_rad) * np.cos(az_rad - phi)
    return np.exp(1j * tau)


def _make_multichannel_cw(az_deg, el_deg, n_samples=3000, snr_db=20.0,
                           tone_hz=TONE_HZ, radius_lambda=0.4253):
    rng = np.random.default_rng(42)
    a = _steering_vector(np.deg2rad(az_deg), np.deg2rad(el_deg),
                         radius_lambda=radius_lambda)
    amp = 10 ** (snr_db / 20.0)
    t = np.arange(n_samples, dtype=np.float64)
    cw = amp * np.exp(2j * np.pi * tone_hz / FS * t)
    nse = (rng.standard_normal((N_ANT, n_samples)) + 1j * rng.standard_normal((N_ANT, n_samples))) / np.sqrt(2)
    return (np.outer(a, cw) + nse).astype(np.complex64)


class TestBPFNormalizerShape:

    def test_output_shape(self):
        X = np.random.randn(5, 4000).astype(np.complex64)
        out = apply_bpf_and_normalize(X, pre_samples=2621, fs=FS, tone_hz=TONE_HZ)
        assert out.shape == (5, 2621)

    def test_output_shape_matches_pre_samples(self):
        X = np.random.randn(5, 5000).astype(np.complex64)
        for ps in [1000, 2000, 2621]:
            out = apply_bpf_and_normalize(X, pre_samples=ps, fs=FS, tone_hz=TONE_HZ)
            assert out.shape == (5, ps)

    def test_short_input_raises(self):
        X = np.ones((5, 100), dtype=np.complex64)
        with pytest.raises(ValueError, match="need"):
            apply_bpf_and_normalize(X, pre_samples=200, fs=FS, tone_hz=TONE_HZ)


class TestBPFNormalizerRMS:

    def test_unit_rms_per_channel(self):
        rng = np.random.default_rng(30)
        amps = np.array([1.0, 5.0, 0.2, 3.0, 0.8])
        X = (amps[:, None] *
             (rng.standard_normal((5, 3000)) + 1j * rng.standard_normal((5, 3000))))
        out = apply_bpf_and_normalize(X.astype(np.complex64), 3000, FS, TONE_HZ)
        rms = np.sqrt(np.mean(np.abs(out) ** 2, axis=1))
        assert np.allclose(rms, 1.0, atol=0.01), f"RMS not unit: {rms}"


class TestBPFNormalizerFiltering:

    def test_rejects_out_of_band(self):
        rng = np.random.default_rng(31)
        n = 4096
        t = np.arange(n, dtype=np.float64)
        cw_in = np.exp(2j * np.pi * TONE_HZ / FS * t)
        cw_out = 100.0 * np.exp(2j * np.pi * (TONE_HZ + 100_000) / FS * t)
        X = np.tile((cw_in + cw_out)[None, :].astype(np.complex64), (5, 1))
        out = apply_bpf_and_normalize(X, n, FS, TONE_HZ, bpf_bw_hz=15_000)
        fft_out = np.fft.fft(out[0])
        freqs = np.fft.fftfreq(n, d=1.0 / FS)
        in_band = np.sum(np.abs(fft_out[np.abs(freqs - TONE_HZ) < 7_500]) ** 2)
        oob_power = np.sum(np.abs(fft_out[np.abs(freqs - (TONE_HZ + 100_000)) < 7_500]) ** 2)
        assert in_band > 1000 * oob_power

    def test_preserves_inter_antenna_phase(self):
        rng = np.random.default_rng(32)
        n = 4096
        t = np.arange(n, dtype=np.float64)
        phases = np.array([0.0, 0.5, 1.2, 2.1, 3.0])
        cw = np.exp(2j * np.pi * TONE_HZ / FS * t)
        X = np.outer(np.exp(1j * phases), cw).astype(np.complex64)
        out = apply_bpf_and_normalize(X, n, FS, TONE_HZ)
        ref = np.exp(-2j * np.pi * TONE_HZ / FS * t)
        y = np.array([float(np.angle((out[k] @ ref) / n)) for k in range(5)])
        delta = (y - y[0] + np.pi) % (2 * np.pi) - np.pi
        expected = (phases - phases[0] + np.pi) % (2 * np.pi) - np.pi
        assert np.allclose(delta, expected, atol=0.05)


class TestBPFNormalizerIridium:

    def test_satellite_steering_vector_preserved(self):
        az, el = 355.0, 25.0
        X = _make_multichannel_cw(az, el, snr_db=25.0)
        out = apply_bpf_and_normalize(X, pre_samples=2621, fs=FS, tone_hz=TONE_HZ)
        a_true = _steering_vector(np.deg2rad(az), np.deg2rad(el))
        ref = np.exp(-2j * np.pi * TONE_HZ / FS * np.arange(2621, dtype=np.float64))
        y = (out @ ref) / 2621
        y_norm = y / (np.abs(y[0]) + 1e-30)
        a_norm = a_true / (np.abs(a_true[0]) + 1e-30)
        cos2 = abs(np.dot(a_norm.conj(), y_norm)) ** 2 / (
            np.linalg.norm(a_norm) * np.linalg.norm(y_norm)
        ) ** 2
        assert cos2 > 0.95, f"Steering vector correlation too low: cos²={cos2:.4f}"

    def test_doppler_shifted_tone(self):
        fd = -22_000.0
        tone = TONE_HZ + fd
        X = _make_multichannel_cw(90.0, 30.0, snr_db=20.0, tone_hz=tone)
        out = apply_bpf_and_normalize(X, pre_samples=2621, fs=FS, tone_hz=tone)
        assert out.shape == (5, 2621)
        rms = np.sqrt(np.mean(np.abs(out) ** 2, axis=1))
        assert np.allclose(rms, 1.0, atol=0.05)