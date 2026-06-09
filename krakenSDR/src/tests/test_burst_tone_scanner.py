#!/usr/bin/env python3
"""
test_burst_tone_scanner.py — Unit tests for preamble tone scanning.

Tests scan_preamble_tones() from core.burst_processing.

Run:
    python3 -m pytest krakenSDR/src/tests/test_burst_tone_scanner.py -v
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

from core.burst_processing import scan_preamble_tones

FS = 1_024_000.0
TONE_HZ = 3_125.0


def _cw_burst(tone_hz, n=8192, snr_db=20.0):
    rng = np.random.default_rng(10)
    amp = 10 ** (snr_db / 20.0)
    t = np.arange(n, dtype=np.float64)
    sig = amp * np.exp(2j * np.pi * tone_hz / FS * t)
    nse = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return (sig + nse).astype(np.complex64)


class TestToneScannerNominal:

    def test_detects_preamble_tone_at_3125hz(self):
        iq = _cw_burst(TONE_HZ)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000)
        assert len(res) >= 1
        tone, snr = res[0]
        assert abs(tone - TONE_HZ) < 200.0
        assert snr > 10.0

    def test_doppler_offset_detected(self):
        fd = 12_000.0
        tone = TONE_HZ + fd
        iq = _cw_burst(tone, n=16_384)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000)
        assert len(res) >= 1
        assert abs(res[0][0] - tone) < 500.0

    def test_negative_doppler(self):
        fd = -35_000.0
        tone = TONE_HZ + fd
        iq = _cw_burst(tone, n=16_384, snr_db=25.0)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000)
        assert len(res) >= 1
        assert abs(res[0][0] - tone) < 500.0


class TestToneScannerMultiPeak:

    def test_two_tones_separated(self):
        rng = np.random.default_rng(11)
        n = 16_384
        amp = 30.0
        t = np.arange(n, dtype=np.float64)
        fd1, fd2 = 8_000.0, -15_000.0
        sig = amp * (np.exp(2j * np.pi * (TONE_HZ + fd1) / FS * t)
                     + np.exp(2j * np.pi * (TONE_HZ + fd2) / FS * t))
        nse = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        iq = (sig + nse).astype(np.complex64)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ,
                                  scan_bw_hz=45_000, n_peaks=2, min_sep_hz=5_000)
        assert len(res) == 2


class TestToneScannerEdgeCases:

    def test_dc_guard_suppresses_lo_leakage(self):
        rng = np.random.default_rng(12)
        n = 8_192
        t = np.arange(n, dtype=np.float64)
        dc_leakage = 1000.0 * np.ones(n, dtype=np.complex64)
        cw_tone = 50.0 * np.exp(2j * np.pi * TONE_HZ / FS * t)
        nse = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        iq = (dc_leakage + cw_tone + nse).astype(np.complex64)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, dc_guard_hz=500.0)
        for tone, _ in res:
            assert abs(tone) > 500.0

    def test_pure_noise_returns_fallback(self):
        rng = np.random.default_rng(13)
        iq = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)) / np.sqrt(2)
        res = scan_preamble_tones(iq.astype(np.complex64), FS,
                                   nom_tone_hz=TONE_HZ, min_snr_db=100.0)
        assert len(res) == 1
        assert res[0][0] == TONE_HZ

    def test_short_input_returns_fallback(self):
        iq = np.ones(64, dtype=np.complex64)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ)
        assert res == [(TONE_HZ, 0.0)]

    def test_prefer_nom_mode(self):
        iq = _cw_burst(TONE_HZ, n=16_384, snr_db=30.0)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ,
                                  scan_bw_hz=45_000, prefer_nom=True)
        assert len(res) >= 1
        assert abs(res[0][0] - TONE_HZ) < 200.0


class TestToneScannerSNR:

    def test_high_snr(self):
        iq = _cw_burst(TONE_HZ, snr_db=30.0)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000)
        assert res[0][1] > 20.0

    def test_low_snr(self):
        iq = _cw_burst(TONE_HZ, snr_db=3.0)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000,
                                  min_snr_db=1.0)
        assert len(res) >= 1

    def test_indoor_narrow_scan(self):
        iq = _cw_burst(TONE_HZ + 500.0, n=8_192, snr_db=10.0)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=3_000)
        assert len(res) >= 1
        assert abs(res[0][0] - (TONE_HZ + 500.0)) < 500.0