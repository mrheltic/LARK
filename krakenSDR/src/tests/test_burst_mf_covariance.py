#!/usr/bin/env python3
"""
test_burst_mf_covariance.py — Unit tests for matched-filter covariance.

Tests compute_mf_covariance() from apps.doa_iridium_grc.lark.burst_processing.

Run:
    python3 -m pytest krakenSDR/src/tests/test_burst_mf_covariance.py -v
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

from apps.doa_iridium_grc.lark.burst_processing import compute_mf_covariance

FS = 1_024_000.0
TONE_HZ = 3_125.0
N_ANT = 5


def _steering_vector(az_rad, el_rad, n_ant=N_ANT, radius_lambda=0.4253):
    k = np.arange(n_ant)
    phi = 2 * np.pi * k / n_ant
    tau = 2 * np.pi * radius_lambda * np.cos(el_rad) * np.cos(az_rad - phi)
    return np.exp(1j * tau)


def _make_preamble(az_deg, el_deg, snr_db=20.0, n_samples=2621,
                    tone_hz=TONE_HZ, radius_lambda=0.4253):
    rng = np.random.default_rng(20)
    a = _steering_vector(np.deg2rad(az_deg), np.deg2rad(el_deg), radius_lambda=radius_lambda)
    amp = 10 ** (snr_db / 20.0)
    t = np.arange(n_samples, dtype=np.float64)
    cw = amp * np.exp(2j * np.pi * tone_hz / FS * t)
    nse = (rng.standard_normal((N_ANT, n_samples)) + 1j * rng.standard_normal((N_ANT, n_samples))) / np.sqrt(2)
    return (np.outer(a, cw) + nse).astype(np.complex64)


class TestMFCovarianceStructure:

    def test_rank1_structure(self):
        X = _make_preamble(0.0, 20.0, snr_db=20.0)
        R, y, snr = compute_mf_covariance(X, TONE_HZ, FS, n_pre=2621, bpf_guard=0)
        ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
        ratio_db = 10.0 * np.log10(ev[0] / (ev[1] + 1e-30))
        assert ratio_db > 20.0, f"R_mf should be rank-1, λ1/λ2 = {ratio_db:.1f} dB"

    def test_hermitian(self):
        X = _make_preamble(0.2, np.deg2rad(40.0))
        R, _, _ = compute_mf_covariance(X, TONE_HZ, FS, 2621, 0)
        assert np.allclose(R, R.conj().T, atol=1e-10), "R_mf is not Hermitian"

    def test_positive_semidefinite(self):
        X = _make_preamble(1.0, np.deg2rad(25.0))
        R, _, _ = compute_mf_covariance(X, TONE_HZ, FS, 2621, 0)
        ev = np.real(np.linalg.eigvalsh(R))
        assert np.all(ev >= -1e-10), f"R_mf has negative eigenvalues: {ev}"


class TestMFCovarianceSteeringVector:

    def test_steering_vector_recovered(self):
        az, el = 0.0, np.deg2rad(20.0)
        k = np.arange(5)
        phi = 2 * np.pi * k / 5
        tau = 2 * np.pi * 0.4253 * np.cos(el) * np.cos(az - phi)
        a_true = np.exp(1j * tau)
        a_true /= np.abs(a_true[0])

        X = _make_preamble(0.0, 20.0, snr_db=30.0)
        R, _, _ = compute_mf_covariance(X, TONE_HZ, FS, 2621, 0)

        ev, V = np.linalg.eigh(R)
        v_est = V[:, -1]
        v_est = v_est / (v_est[0] / abs(v_est[0]))

        cos2 = abs(np.dot(a_true.conj(), v_est)) ** 2 / (
            np.linalg.norm(a_true) * np.linalg.norm(v_est)
        ) ** 2
        assert cos2 > 0.95, f"Steering vector recovery: cos²={cos2:.4f}"


class TestMFCovarianceSNR:

    def test_snr_positive_for_strong_signal(self):
        X = _make_preamble(0.0, 30.0, snr_db=20.0)
        _, _, snr = compute_mf_covariance(X, TONE_HZ, FS, 2621, 0)
        assert snr > 0.0, f"SNR should be positive for 20 dB input, got {snr:.1f}"

    def test_snr_low_for_noise_only(self):
        rng = np.random.default_rng(21)
        X = (rng.standard_normal((5, 3000)) + 1j * rng.standard_normal((5, 3000))).astype(np.complex64)
        _, _, snr = compute_mf_covariance(X, TONE_HZ, FS, 2621, 0)
        assert snr < 3.0, f"Noise-only SNR should be low, got {snr:.1f}"


class TestMFCovarianceEdgeCases:

    def test_short_input_raises(self):
        X = np.ones((5, 100), dtype=np.complex64)
        with pytest.raises(ValueError, match="too short"):
            compute_mf_covariance(X, TONE_HZ, FS, n_pre=200, bpf_guard=0)

    def test_bpf_guard_skips_leading(self):
        X = _make_preamble(0.0, 20.0, n_samples=3000)
        R0, _, _ = compute_mf_covariance(X, TONE_HZ, FS, n_pre=2500, bpf_guard=0)
        R1, _, _ = compute_mf_covariance(X, TONE_HZ, FS, n_pre=2431, bpf_guard=69)
        for R in (R0, R1):
            ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
            assert 10.0 * np.log10(ev[0] / (ev[1] + 1e-30)) > 20.0


class TestMFCovarianceIridiumScenarios:

    @pytest.mark.parametrize("az,el", [(0, 20), (90, 45), (180, 10), (270, 60), (355, 25)])
    def test_various_satellite_directions(self, az, el):
        X = _make_preamble(az, el, snr_db=20.0)
        R, y, snr = compute_mf_covariance(X, TONE_HZ, FS, 2621, 0)
        ev = np.sort(np.real(np.linalg.eigvalsh(R)))[::-1]
        assert ev[0] > ev[1] * 10, f"Steering vector not dominant at az={az}, el={el}"

    def test_doppler_shifted_tone(self):
        fd = -22_000.0
        tone = TONE_HZ + fd
        X = _make_preamble(90.0, 30.0, snr_db=20.0, tone_hz=tone)
        R, y, snr = compute_mf_covariance(X, tone, FS, 2621, 0)
        assert snr > 0.0, f"SNR should be positive for Doppler-shifted tone"