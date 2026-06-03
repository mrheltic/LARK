#!/usr/bin/env python3
"""
test_gr_burst_energy.py — Unit tests for burst energy detection (core Python, no GR).

Tests detect_energy_bursts() from apps.doa_iridium.burst_processing with
Iridium L-band parameters (1.024 MSPS, CW tone at 3125 Hz).

Run:
    python3 -m pytest krakenSDR/src/tests/test_gr_burst_energy.py -v
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

from apps.doa_iridium.burst_processing import detect_energy_bursts

FS = 1_024_000.0
TONE_HZ = 3_125.0


def _make_burst(n_total, burst_start, burst_len, snr_db=20.0, tone_hz=TONE_HZ):
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(n_total) + 1j * rng.standard_normal(n_total)) / np.sqrt(2)
    amp = 10 ** (snr_db / 20.0)
    t = np.arange(burst_len, dtype=np.float64)
    burst = amp * np.exp(2j * np.pi * tone_hz / FS * t)
    noise[burst_start:burst_start + burst_len] += burst
    return noise.astype(np.complex64)


class TestBurstEnergyDetection:

    def test_detects_single_burst_at_iridium_freq(self):
        iq = _make_burst(200_000, 50_000, 2_560, snr_db=20.0)
        starts = detect_energy_bursts(iq, FS, energy_window=256, threshold_factor=3.0)
        assert len(starts) >= 1
        assert abs(starts[0] - 50_000) <= 256

    def test_detects_burst_with_doppler_shift(self):
        fd = -22_000.0
        tone = TONE_HZ + fd
        iq = _make_burst(200_000, 40_000, 2_560, snr_db=15.0, tone_hz=tone)
        starts = detect_energy_bursts(iq, FS, energy_window=256, threshold_factor=3.0)
        assert len(starts) >= 1

    def test_no_burst_in_pure_noise(self):
        rng = np.random.default_rng(7)
        iq = (rng.standard_normal(100_000) + 1j * rng.standard_normal(100_000)) / np.sqrt(2)
        starts = detect_energy_bursts(iq.astype(np.complex64), FS, threshold_factor=5.0)
        assert starts == []

    def test_two_bursts_separated_by_superframe(self):
        n = 400_000
        rng = np.random.default_rng(9)
        iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.5
        amp = 10.0
        burst_len = 2_560
        for pos in [20_000, 150_000]:
            t = np.arange(burst_len, dtype=np.float64)
            iq[pos:pos + burst_len] += amp * np.exp(2j * np.pi * TONE_HZ / FS * t)
        starts = detect_energy_bursts(iq.astype(np.complex64), FS, threshold_factor=3.0)
        assert len(starts) == 2

    def test_min_gap_enforced(self):
        n = 50_000
        rng = np.random.default_rng(10)
        iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.1
        iq[5_000:5_512] += 10.0
        iq[8_000:8_512] += 10.0
        starts = detect_energy_bursts(iq.astype(np.complex64), FS, energy_window=64, min_gap_samples=10_000)
        assert len(starts) == 1

    def test_empty_input(self):
        assert detect_energy_bursts(np.array([], dtype=np.complex64), FS) == []

    def test_low_snr_high_threshold_no_detect(self):
        rng = np.random.default_rng(15)
        n = 50_000
        iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        iq[10_000:12_560] += 0.3 * np.exp(2j * np.pi * TONE_HZ / FS * np.arange(2_560))
        starts = detect_energy_bursts(iq.astype(np.complex64), FS, threshold_factor=10.0)
        assert starts == []


class TestBurstEnergyOffsets:

    def test_returns_sample_offsets(self):
        iq = _make_burst(50_000, 20_000, 2_560)
        starts = detect_energy_bursts(iq, FS, energy_window=256)
        for s in starts:
            assert s % 256 == 0

    def test_burst_at_frame_boundary(self):
        iq = _make_burst(200_000, 131_072 - 1_280, 2_560, snr_db=25.0)
        starts = detect_energy_bursts(iq, FS, energy_window=256)
        assert len(starts) >= 1

    def test_multiple_snr_levels(self):
        for snr in [5, 10, 15, 20, 30]:
            iq = _make_burst(100_000, 30_000, 2_560, snr_db=snr)
            starts = detect_energy_bursts(iq, FS, threshold_factor=3.0)
            assert len(starts) >= 1, f"Failed at SNR={snr} dB"