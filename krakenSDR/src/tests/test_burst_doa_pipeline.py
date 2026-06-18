#!/usr/bin/env python3
"""
test_burst_doa_pipeline.py — Integration test for full burst → DOA pipeline.

End-to-end: burst detection → tone scan → BPF → MF covariance → DOA estimation.

Run:
    python3 -m pytest krakenSDR/src/tests/test_burst_doa_pipeline.py -v
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

from core.burst_processing import (
    detect_energy_bursts,
    scan_preamble_tones,
    apply_bpf_and_normalize,
    compute_mf_covariance,
)
from core.doa_uca_2d import (
    UcaConfig,
    doa_music_uca_2d,
    find_peak_uca_2d,
    amplitude_normalize_channels,
)
from core.doa_algorithms import apply_phase_correction

FS = 1_024_000.0
TONE_HZ = 3_125.0
N_ANT = 5
R_LAMBDA = 0.4253


def _make_iridium_frame(az_deg, el_deg, snr_db=20.0, doppler_hz=0.0,
                          burst_start=50_000, burst_len=2_621,
                          frame_len=131_072, rng_seed=None):
    rng = np.random.default_rng(rng_seed or 42)
    k = np.arange(N_ANT)
    phi = 2 * np.pi * k / N_ANT
    az_r = np.deg2rad(az_deg)
    el_r = np.deg2rad(el_deg)
    tau = 2 * np.pi * R_LAMBDA * np.cos(el_r) * np.cos(az_r - phi)
    a = np.exp(1j * tau)

    amp = 10 ** (snr_db / 20.0)
    tone_hz = TONE_HZ + doppler_hz
    t = np.arange(burst_len, dtype=np.float64)
    cw = np.exp(2j * np.pi * tone_hz / FS * t)
    burst_iq = np.outer(a, amp * cw)

    frame = (rng.standard_normal((N_ANT, frame_len)) + 1j * rng.standard_normal((N_ANT, frame_len))) / np.sqrt(2)
    frame.astype(np.complex64)
    end = min(burst_start + burst_len, frame_len)
    actual_len = end - burst_start
    frame[:, burst_start:end] += burst_iq[:, :actual_len]

    return frame.astype(np.complex64), burst_start, min(burst_start + burst_len, frame_len), tone_hz


def _uca_cfg():
    return UcaConfig(
        n_ant=N_ANT, radius_lambda=R_LAMBDA,
        n_az=360, n_el=86,
        el_min_deg=5.0, el_max_deg=90.0,
        num_expected_signals=1,
        ant0_offset_deg=0.0, ant_ccw=False,
    )


def _angular_error(est, true):
    d = abs(est - true) % 360
    return min(d, 360.0 - d)


class TestFullPipeline:

    def test_pipeline_single_burst_az0_el20(self):
        cfg = _uca_cfg()
        X, bs, be, tone = _make_iridium_frame(0.0, 20.0, snr_db=25.0)

        starts = detect_energy_bursts(X[0, :], FS, threshold_factor=3.0)
        assert len(starts) >= 1

        burst_start = starts[0]
        burst_end = min(burst_start + 10_547, X.shape[1])

        tones = scan_preamble_tones(X[0, burst_start:burst_end], FS, TONE_HZ,
                                    scan_bw_hz=45_000, min_snr_db=2.0)
        assert len(tones) >= 1

        X_win = X[:, burst_start:burst_end]
        X_bpf = apply_bpf_and_normalize(X_win, 2621, FS, tones[0][0])
        X_cal = apply_phase_correction(X_bpf, [0.0] * 5)

        R, y, snr = compute_mf_covariance(X_cal, tones[0][0], FS, 2621, 0)
        assert snr > 5.0

        spec = doa_music_uca_2d(X_cal, cfg, R_in=R)
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        assert papr > 3.0, f"PAPR too low: {papr:.1f} dB"
        assert _angular_error(az_est, 0.0) < 10.0, f"az={az_est:.1f}°, expected ~0°"
        assert abs(el_est - 20.0) < 10.0, f"el={el_est:.1f}°, expected ~20°"

    @pytest.mark.parametrize("az,el", [(45, 30), (90, 45), (180, 25), (270, 60), (355, 15)])
    def test_pipeline_various_directions(self, az, el):
        cfg = _uca_cfg()
        X, _, _, _ = _make_iridium_frame(az, el, snr_db=25.0, rng_seed=100 + int(az))

        starts = detect_energy_bursts(X[0, :], FS, threshold_factor=3.0)
        assert len(starts) >= 1

        burst_start = starts[0]
        burst_end = min(burst_start + 10_547, X.shape[1])

        tones = scan_preamble_tones(X[0, burst_start:burst_end], FS, TONE_HZ,
                                    scan_bw_hz=45_000, min_snr_db=2.0)
        assert len(tones) >= 1

        X_win = X[:, burst_start:burst_end]
        if X_win.shape[1] < 2621:
            pytest.skip(f"Window too short: {X_win.shape[1]}")

        X_bpf = apply_bpf_and_normalize(X_win, 2621, FS, tones[0][0])
        X_cal = apply_phase_correction(X_bpf, [0.0] * 5)

        R, y, snr = compute_mf_covariance(X_cal, tones[0][0], FS, 2621, 0)
        spec = doa_music_uca_2d(X_cal, cfg, R_in=R)
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)

        az_err = _angular_error(az_est, az)
        assert az_err < 15.0, f"az={az}°: est={az_est:.1f}°, err={az_err:.1f}°"
        assert abs(el_est - el) < 15.0, f"el={el}°: est={el_est:.1f}°, err={abs(el_est - el):.1f}°"

    def test_pipeline_with_doppler(self):
        cfg = _uca_cfg()
        X, _, _, tone = _make_iridium_frame(90.0, 30.0, snr_db=20.0, doppler_hz=-22_000.0)

        starts = detect_energy_bursts(X[0, :], FS, threshold_factor=3.0)
        assert len(starts) >= 1

        burst_start = starts[0]
        burst_end = min(burst_start + 10_547, X.shape[1])

        tones = scan_preamble_tones(X[0, burst_start:burst_end], FS, TONE_HZ,
                                    scan_bw_hz=45_000, min_snr_db=2.0)
        assert len(tones) >= 1

        detected_tone = tones[0][0]
        assert abs(detected_tone - (TONE_HZ - 22_000)) < 1000.0

        X_win = X[:, burst_start:burst_end]
        X_bpf = apply_bpf_and_normalize(X_win, 2621, FS, detected_tone)
        X_cal = apply_phase_correction(X_bpf, [0.0] * 5)

        R, y, snr = compute_mf_covariance(X_cal, detected_tone, FS, 2621, 0)
        spec = doa_music_uca_2d(X_cal, cfg, R_in=R)
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)

        assert _angular_error(az_est, 90.0) < 15.0

    def test_pipeline_low_snr(self):
        cfg = _uca_cfg()
        X, _, _, _ = _make_iridium_frame(90.0, 30.0, snr_db=5.0)

        starts = detect_energy_bursts(X[0, :], FS, threshold_factor=2.0)
        if len(starts) == 0:
            pytest.skip("Burst not detected at 5 dB SNR")

        burst_start = starts[0]
        burst_end = min(burst_start + 10_547, X.shape[1])

        tones = scan_preamble_tones(X[0, burst_start:burst_end], FS, TONE_HZ,
                                    scan_bw_hz=45_000, min_snr_db=1.0)
        if len(tones) == 0:
            pytest.skip("Tone not detected at 5 dB SNR")

        X_win = X[:, burst_start:burst_end]
        if X_win.shape[1] < 2621:
            pytest.skip("Window too short")

        X_bpf = apply_bpf_and_normalize(X_win, 2621, FS, tones[0][0])
        X_cal = apply_phase_correction(X_bpf, [0.0] * 5)

        R, y, snr = compute_mf_covariance(X_cal, tones[0][0], FS, 2621, 0)

        spec = doa_music_uca_2d(X_cal, cfg, R_in=R)
        az_est, el_est, papr = find_peak_uca_2d(spec, cfg)
        az_err = _angular_error(az_est, 90.0)
        assert az_err < 30.0, f"Low-SNR DOA error too large: {az_err:.1f}°"

    def test_pipeline_papr_rejection(self):
        cfg = _uca_cfg()
        rng = np.random.default_rng(99)
        frame_len = 131_072
        noise = (rng.standard_normal((N_ANT, frame_len)) + 1j * rng.standard_normal((N_ANT, frame_len))) / np.sqrt(2)
        noise = noise.astype(np.complex64)

        starts = detect_energy_bursts(noise[0, :], FS, threshold_factor=5.0)
        assert len(starts) == 0, "Pure noise should not produce burst detection"


def _make_burst_window_with_data_tail(
    az_deg: float,
    el_deg: float,
    *,
    pre_samples: int = 2621,
    window_samples: int = 3000,
    snr_db: float = 25.0,
    rng_seed: int = 77,
) -> np.ndarray:
    """Preamble CW followed by a wideband DQPSK-like data section (production window)."""
    rng = np.random.default_rng(rng_seed)
    k = np.arange(N_ANT)
    phi = 2 * np.pi * k / N_ANT
    az_r = np.deg2rad(az_deg)
    el_r = np.deg2rad(el_deg)
    tau = 2 * np.pi * R_LAMBDA * np.cos(el_r) * np.cos(az_r - phi)
    a = np.exp(1j * tau)
    amp = 10 ** (snr_db / 20.0)

    t_pre = np.arange(pre_samples, dtype=np.float64)
    cw = amp * np.exp(2j * np.pi * TONE_HZ / FS * t_pre)
    pre = np.outer(a, cw)

    tail_len = window_samples - pre_samples
    sym_rate = 25_000.0
    t_tail = np.arange(tail_len, dtype=np.float64)
    # Strong in-band modulation after the CW preamble (mimics IRA data section).
    data = amp * np.exp(1j * (2 * np.pi * TONE_HZ / FS * t_tail
                              + np.cumsum(rng.choice([-1, 1], tail_len) * 0.8)))
    data = np.outer(a, data)

    X_win = np.zeros((N_ANT, window_samples), dtype=np.complex64)
    X_win[:, :pre_samples] = pre.astype(np.complex64)
    X_win[:, pre_samples:] = data.astype(np.complex64)
    return X_win


class TestBPFWindowLength:
    """BPF must use preamble length only — not the wider burst window."""

    def test_pre_samples_differs_from_window_samples_with_data_tail(self):
        """Longer BPF FFT includes post-preamble data and alters preamble IQ."""
        X_win = _make_burst_window_with_data_tail(90.0, 35.0)

        X_bpf_pre = apply_bpf_and_normalize(X_win, 2621, FS, TONE_HZ)
        X_bpf_win = apply_bpf_and_normalize(X_win, 3000, FS, TONE_HZ)

        n = min(2621, X_bpf_pre.shape[1], X_bpf_win.shape[1])
        rel_diff = float(
            np.linalg.norm(X_bpf_pre[:, :n] - X_bpf_win[:, :n])
            / (np.linalg.norm(X_bpf_pre[:, :n]) + 1e-20)
        )
        assert rel_diff > 0.01, (
            f"post-preamble data should change BPF output: rel_diff={rel_diff:.4f}"
        )


class TestPipelineEMA:

    def test_ema_convergence(self):
        from core.tracking import CircularEMA, ScalarEMA

        cfg = _uca_cfg()
        az_ema = CircularEMA(alpha=0.88)
        el_ema = ScalarEMA(alpha=0.65)

        true_az = 90.0
        true_el = 30.0
        errors = []

        for i in range(10):
            X, _, _, tone = _make_iridium_frame(true_az, true_el, snr_db=25.0,
                                                   rng_seed=200 + i)
            starts = detect_energy_bursts(X[0, :], FS, threshold_factor=3.0)
            if not starts:
                continue

            bs = starts[0]
            be = min(bs + 10_547, X.shape[1])
            tones = scan_preamble_tones(X[0, bs:be], FS, TONE_HZ, scan_bw_hz=45_000)
            if not tones:
                continue

            X_win = X[:, bs:be]
            if X_win.shape[1] < 2621:
                continue

            X_bpf = apply_bpf_and_normalize(X_win, 2621, FS, tones[0][0])
            X_cal = apply_phase_correction(X_bpf, [0.0] * 5)
            R, y, snr = compute_mf_covariance(X_cal, tones[0][0], FS, 2621, 0)

            spec = doa_music_uca_2d(X_cal, cfg, R_in=R)
            az_est, el_est, _ = find_peak_uca_2d(spec, cfg)

            az_smooth = az_ema.update(az_est)
            el_smooth = el_ema.update(el_est)
            errors.append(_angular_error(az_smooth, true_az))

        if len(errors) >= 3:
            assert np.mean(errors[-3:]) < 15.0, f"EMA did not converge: errors={errors}"