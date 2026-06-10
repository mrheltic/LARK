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

class TestSubBinInterpolation:
    """Parabolic interpolation: tone frequency error well below the FFT bin."""

    def test_offgrid_tone_short_window(self):
        # 3000-sample window → nfft 4096 → 250 Hz bins; an off-grid tone
        # should still come back within a fraction of a bin.
        true_hz = 3_125.0 + 7_387.0   # deliberately off the 250 Hz grid
        iq = _cw_burst(true_hz, n=3_000, snr_db=20.0)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000)
        assert len(res) >= 1
        assert abs(res[0][0] - true_hz) < 25.0

    def test_offgrid_negative_doppler(self):
        true_hz = 3_125.0 - 21_111.0
        iq = _cw_burst(true_hz, n=3_000, snr_db=20.0)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000)
        assert len(res) >= 1
        assert abs(res[0][0] - true_hz) < 25.0

    def test_two_offgrid_tones(self):
        f1 = 3_125.0 + 10_087.0
        f2 = 3_125.0 - 15_213.0
        iq = _cw_burst(f1, n=3_000, snr_db=20.0) + _cw_burst(f2, n=3_000, snr_db=17.0)
        res = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ,
                                  scan_bw_hz=45_000, n_peaks=2)
        assert len(res) == 2
        found = sorted(r[0] for r in res)
        assert abs(found[0] - f2) < 50.0
        assert abs(found[1] - f1) < 50.0


class TestMultiChannelToneScan:
    """scan_preamble_tones with (n_ant, N) input: power spectra summed."""

    def test_2d_matches_1d(self):
        iq = _cw_burst(TONE_HZ + 8_000.0, n=3_000, snr_db=20.0)
        X = np.tile(iq, (5, 1))
        res1 = scan_preamble_tones(iq, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000)
        res5 = scan_preamble_tones(X, FS, nom_tone_hz=TONE_HZ, scan_bw_hz=45_000)
        assert abs(res5[0][0] - res1[0][0]) < 5.0

    def test_tone_faded_on_channel0(self):
        # Tone present only on channels 1-4: ch0-only scan misses it,
        # combined scan finds it.
        rng = np.random.default_rng(21)
        n = 3_000
        true_hz = TONE_HZ - 12_000.0
        X = ((rng.standard_normal((5, n)) + 1j * rng.standard_normal((5, n)))
             / np.sqrt(2)).astype(np.complex64)
        t = np.arange(n, dtype=np.float64)
        amp = 10 ** (10.0 / 20.0)
        X[1:] += (amp * np.exp(2j * np.pi * true_hz / FS * t)).astype(np.complex64)
        res0 = scan_preamble_tones(X[0], FS, nom_tone_hz=TONE_HZ,
                                   scan_bw_hz=45_000, min_snr_db=8.0)
        res = scan_preamble_tones(X, FS, nom_tone_hz=TONE_HZ,
                                  scan_bw_hz=45_000, min_snr_db=8.0)
        assert all(abs(f - true_hz) > 500.0 for f, _ in res0)
        assert any(abs(f - true_hz) < 100.0 for f, _ in res)
