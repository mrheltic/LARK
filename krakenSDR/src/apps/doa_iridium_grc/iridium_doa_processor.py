#!/usr/bin/env python3
"""
iridium_doa_processor.py — GNU Radio embedded block for Iridium burst DoA.

Reuses the LARK core DOA library (UCA 2D-MUSIC/Capon/Bartlett/Phase-Fit)
inside a GNU Radio scheduler block.

Inputs: 5 vector streams of CPI length (from krakensdr_source via stream_to_vector)
Outputs: message ports: azimuth, elevation, snr, spectrum, burst_detected
"""

from __future__ import annotations

import sys
import os
import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gnuradio import gr
import pmt

from core.doa_uca_2d import (
    UcaConfig, doa_music_uca_2d, doa_bartlett_uca_2d,
    doa_capon_uca_2d, doa_phase_fit_uca_2d, find_peak_uca_2d,
    amplitude_normalize_channels, eigenvalue_spread_uca_db, snr_uca_db,
)
from core.doa_algorithms import apply_phase_correction
from apps.doa_iridium.burst_processing import (
    detect_energy_bursts, scan_preamble_tones,
    compute_mf_covariance, apply_bpf_and_normalize,
)

_ALGO_MAP = {
    "MUSIC": doa_music_uca_2d,
    "BARTLETT": doa_bartlett_uca_2d,
    "CAPON": doa_capon_uca_2d,
    "PHASE-FIT": None,
}


class iridium_doa_processor(gr.sync_block):
    def __init__(
        self,
        cpi_size: int = 131072,
        fs: float = 1_024_000.0,
        freq_hz: float = 1_626_270_000.0,
        n_ant: int = 5,
        radius_lambda: float = 0.4253,
        ant0_offset_deg: float = 0.0,
        ant_ccw: bool = False,
        n_az: int = 360,
        n_el: int = 86,
        el_min_deg: float = 5.0,
        el_max_deg: float = 90.0,
        algorithm: str = "MUSIC",
        num_signals: int = 1,
        phase_offsets_deg: list | None = None,
        energy_threshold: float = 3.0,
        tone_nom_hz: float = 3125.0,
        tone_scan_bw_hz: float = 3000.0,
        tone_min_snr_db: float = 2.0,
        bpf_bw_hz: float = 8000.0,
        pre_samples: int = 2621,
        bpf_guard: int = 128,
        window_samples: int = 3000,
        cov_alpha: float = 0.93,
        az_ema_alpha: float = 0.88,
        el_ema_alpha: float = 0.65,
        snr_min_db: float = -3.0,
        papr_min_db: float = 3.0,
    ):
        gr.sync_block.__init__(
            self,
            name="Iridium DoA Processor",
            in_sig=[(np.complex64, cpi_size)] * n_ant,
            out_sig=None,
        )

        self.cpi_size = cpi_size
        self.fs = fs
        self.n_ant = n_ant
        self.algorithm = algorithm.upper()
        self.pre_samples = min(pre_samples, cpi_size)
        self.bpf_guard = bpf_guard
        self.window_samples = min(window_samples, cpi_size)
        self.bpf_bw_hz = bpf_bw_hz
        self.energy_threshold = energy_threshold
        self.tone_nom_hz = tone_nom_hz
        self.tone_scan_bw_hz = tone_scan_bw_hz
        self.tone_min_snr_db = tone_min_snr_db
        self.cov_alpha = cov_alpha
        self.az_ema_alpha = az_ema_alpha
        self.el_ema_alpha = el_ema_alpha
        self.snr_min_db = snr_min_db
        self.papr_min_db = papr_min_db
        if phase_offsets_deg is None:
            self.phase_offsets_deg = [0.0] * n_ant
        elif isinstance(phase_offsets_deg, str):
            self.phase_offsets_deg = [float(x) for x in phase_offsets_deg.split(",")]
        else:
            self.phase_offsets_deg = list(phase_offsets_deg)

        self.cfg = UcaConfig(
            n_ant=n_ant, radius_lambda=radius_lambda,
            n_az=n_az, n_el=n_el,
            el_min_deg=el_min_deg, el_max_deg=el_max_deg,
            num_expected_signals=num_signals,
            ant0_offset_deg=ant0_offset_deg, ant_ccw=ant_ccw,
        )

        self._R_ema = None
        self._az_ema = None
        self._el_ema = None
        self._n_bursts = 0

        self.message_port_register_out(pmt.intern("azimuth"))
        self.message_port_register_out(pmt.intern("elevation"))
        self.message_port_register_out(pmt.intern("spectrum"))
        self.message_port_register_out(pmt.intern("snr"))
        self.message_port_register_out(pmt.intern("burst_detected"))

    def _publish(self, port, value):
        self.message_port_pub(pmt.intern(port), pmt.from_double(float(value)))

    def _publish_bool(self, port, value):
        self.message_port_pub(pmt.intern(port), pmt.from_bool(value))

    def _publish_spectrum(self, spec):
        flat = spec.ravel().astype(np.float32)
        v = pmt.init_f32vector(flat.size, flat)
        self.message_port_pub(pmt.intern("spectrum"), v)

    def work(self, input_items, output_items):
        n_ant = self.n_ant
        n_frames = len(input_items[0])
        if n_frames == 0:
            return 0

        for fi in range(n_frames):
            X = np.empty((n_ant, self.cpi_size), dtype=np.complex64)
            for ch in range(n_ant):
                X[ch, :] = input_items[ch][fi]

            burst_starts = detect_energy_bursts(
                X[0, :], self.fs,
                energy_window=256,
                threshold_factor=self.energy_threshold,
                min_gap_samples=int(0.045 * self.fs),
            )
            if not burst_starts:
                self._publish_bool("burst_detected", False)
                continue

            burst_start = burst_starts[0]
            burst_end = min(burst_start + self.window_samples, X.shape[1])
            if burst_end - burst_start < self.pre_samples + self.bpf_guard:
                self._publish_bool("burst_detected", False)
                continue

            tones = scan_preamble_tones(
                X[0, burst_start:burst_end], self.fs, self.tone_nom_hz,
                scan_bw_hz=self.tone_scan_bw_hz, n_peaks=1,
                min_sep_hz=1000.0, min_snr_db=self.tone_min_snr_db, dc_guard_hz=200.0,
            )
            if not tones:
                self._publish_bool("burst_detected", False)
                continue

            tone_hz = tones[0][0]
            X_win = X[:, burst_start:burst_end]
            if X_win.shape[1] < self.window_samples:
                self._publish_bool("burst_detected", False)
                continue

            try:
                X_bpf = apply_bpf_and_normalize(
                    X_win, self.window_samples, self.fs, tone_hz, self.bpf_bw_hz
                )
            except ValueError:
                self._publish_bool("burst_detected", False)
                continue

            X_cal = apply_phase_correction(X_bpf, self.phase_offsets_deg)

            try:
                R_mf, y_mf, snr_db = compute_mf_covariance(
                    X_cal, tone_hz, self.fs, self.pre_samples, self.bpf_guard
                )
            except ValueError:
                self._publish_bool("burst_detected", False)
                continue

            if snr_db < self.snr_min_db:
                self._publish_bool("burst_detected", False)
                continue

            if self._R_ema is None:
                self._R_ema = R_mf.copy()
            else:
                self._R_ema = self.cov_alpha * self._R_ema + (1 - self.cov_alpha) * R_mf

            algo_fn = _ALGO_MAP.get(self.algorithm)
            spec = None
            if self.algorithm == "PHASE-FIT":
                az_est, el_est, _ = doa_phase_fit_uca_2d(
                    self._R_ema, self.cfg,
                    az_hint_deg=self._az_ema, el_hint_deg=self._el_ema,
                )
            elif algo_fn is not None:
                if self.algorithm == "CAPON":
                    spec = algo_fn(X_cal, self.cfg, R_in=self._R_ema, decorr="none")
                elif self.algorithm == "BARTLETT":
                    spec = algo_fn(X_cal, self.cfg, R_in=self._R_ema)
                else:
                    spec = algo_fn(X_cal, self.cfg, R_in=self._R_ema)
                az_est, el_est, papr = find_peak_uca_2d(spec, self.cfg)
                if papr < self.papr_min_db:
                    self._publish_bool("burst_detected", False)
                    continue
            else:
                spec = doa_music_uca_2d(X_cal, self.cfg, R_in=self._R_ema)
                az_est, el_est, papr = find_peak_uca_2d(spec, self.cfg)
                if papr < self.papr_min_db:
                    self._publish_bool("burst_detected", False)
                    continue

            if self._az_ema is None:
                self._az_ema = az_est
                self._el_ema = el_est
            else:
                d_az = ((az_est - self._az_ema + 180) % 360) - 180
                self._az_ema += self.az_ema_alpha * d_az
                self._az_ema %= 360.0
                self._el_ema += self.el_ema_alpha * (el_est - self._el_ema)

            self._n_bursts += 1
            self._publish("azimuth", self._az_ema)
            self._publish("elevation", self._el_ema)
            self._publish("snr", snr_db)
            self._publish_bool("burst_detected", True)
            if spec is not None:
                self._publish_spectrum(spec)

        return n_frames