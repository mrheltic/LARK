#!/usr/bin/env python3
"""
test_multi_peak.py — Tests for process_cpi_for_multi (core.multi_peak).

Covers the multi-burst-per-CPI path: a 128 ms CPI can contain more than one
90 ms Iridium frame, so every detected burst must be processed.

Run:
    python3 -m pytest krakenSDR/src/tests/test_multi_peak.py -v
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from core.doa_uca_2d import UcaConfig
from core.multi_peak import process_cpi_for_multi

FS = 1_024_000.0
TONE_HZ = 3_125.0
N_ANT = 5
R_LAMBDA = 0.4253

PROFILE = dict(
    tone_nom_hz=3125.0, scan_bw_hz=45_000.0, bpf_bw_hz=15_000.0,
    dc_guard_hz=500.0, min_snr_db=4.0, energy_threshold=2.5,
)
CFG = {
    "hardware": {"pre_samples": 2621, "window_samples": 3000, "bpf_guard": 128},
}
THR = {"snr_min_db": 4.0, "papr_min_db": 2.5}


def _uca_cfg():
    return UcaConfig(
        n_ant=N_ANT, radius_lambda=R_LAMBDA,
        n_az=360, n_el=86,
        el_min_deg=5.0, el_max_deg=90.0,
        num_expected_signals=1,
        ant0_offset_deg=0.0, ant_ccw=False,
    )


def _add_burst(frame, az_deg, el_deg, doppler_hz, burst_start,
               snr_db=25.0, burst_len=2_900):
    k = np.arange(N_ANT)
    phi = 2 * np.pi * k / N_ANT
    az_r, el_r = np.deg2rad(az_deg), np.deg2rad(el_deg)
    tau = 2 * np.pi * R_LAMBDA * np.cos(el_r) * np.cos(az_r - phi)
    a = np.exp(1j * tau)
    amp = 10 ** (snr_db / 20.0)
    t = np.arange(burst_len, dtype=np.float64)
    cw = np.exp(2j * np.pi * (TONE_HZ + doppler_hz) / FS * t)
    end = min(burst_start + burst_len, frame.shape[1])
    frame[:, burst_start:end] += np.outer(a, amp * cw)[:, : end - burst_start]


def _make_frame(frame_len=131_072, seed=11):
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal((N_ANT, frame_len))
             + 1j * rng.standard_normal((N_ANT, frame_len))) / np.sqrt(2))


def _angular_error(est, true):
    d = abs(est - true) % 360
    return min(d, 360.0 - d)


class TestMultiBurstPerCpi:

    def test_single_burst(self):
        X = _make_frame()
        _add_burst(X, 60.0, 40.0, +10_000.0, 30_000)
        rec = process_cpi_for_multi(
            X.astype(np.complex64), frame_idx=0, cfg=CFG, profile=PROFILE,
            uca=_uca_cfg(), phase_offs=[0.0] * N_ANT, algo="music",
            thr=THR, k_peaks=3, el_min_deg=5.0,
        )
        assert rec is not None
        assert rec["peaks"].shape[0] >= 1
        assert _angular_error(float(rec["peaks"][0][0]), 60.0) < 10.0
        assert abs(rec["cfo_per_peak"][0] - 10_000.0) < 300.0

    def test_two_bursts_same_cpi_both_processed(self):
        # Two bursts > 45 ms apart (the detector min gap) at different
        # azimuths and Doppler shifts: both must produce a DOA peak.
        X = _make_frame()
        _add_burst(X, 60.0, 40.0, +10_000.0, 20_000)
        _add_burst(X, 200.0, 25.0, -15_000.0, 90_000)
        rec = process_cpi_for_multi(
            X.astype(np.complex64), frame_idx=0, cfg=CFG, profile=PROFILE,
            uca=_uca_cfg(), phase_offs=[0.0] * N_ANT, algo="music",
            thr=THR, k_peaks=3, el_min_deg=5.0,
        )
        assert rec is not None
        assert rec["peaks"].shape[0] >= 2

        cfos = np.asarray(rec["cfo_per_peak"])
        i1 = int(np.argmin(np.abs(cfos - 10_000.0)))
        i2 = int(np.argmin(np.abs(cfos + 15_000.0)))
        assert abs(cfos[i1] - 10_000.0) < 300.0
        assert abs(cfos[i2] + 15_000.0) < 300.0
        assert _angular_error(float(rec["peaks"][i1][0]), 60.0) < 10.0
        assert _angular_error(float(rec["peaks"][i2][0]), 200.0) < 10.0

    def test_noise_only_returns_none_or_empty(self):
        X = _make_frame(seed=3)
        rec = process_cpi_for_multi(
            X.astype(np.complex64), frame_idx=0, cfg=CFG, profile=PROFILE,
            uca=_uca_cfg(), phase_offs=[0.0] * N_ANT, algo="music",
            thr=THR, k_peaks=3, el_min_deg=5.0,
        )
        assert rec is None or rec["peaks"].shape[0] == 0
