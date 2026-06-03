#!/usr/bin/env python3
"""
iridium_doa_estimator — GNU Radio block for 2D UCA DOA estimation.

Input : PDU covariance (from iridium_mf_covariance)
Output: 4 PDU message ports:
            "azimuth"   → float (degrees)
            "elevation" → float (degrees)
            "snr"       → float (dB)
            "spectrum"  → f32vector (n_el × n_az, row-major dB)

Delegates to core.doa_uca_2d (MUSIC/Capon/Bartlett/Phase-Fit)
            + core.tracking (CircularEMA, ScalarEMA)
"""

from __future__ import annotations

import os
import sys
import numpy as np

_LARK_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "krakenSDR", "src"))
if _LARK_SRC not in sys.path:
    sys.path.insert(0, _LARK_SRC)

from gnuradio import gr
import pmt

from core.doa_uca_2d import (
    UcaConfig,
    doa_music_uca_2d,
    doa_bartlett_uca_2d,
    doa_capon_uca_2d,
    doa_phase_fit_uca_2d,
    find_peak_uca_2d,
    amplitude_normalize_channels,
    eigenvalue_spread_uca_db,
)
from core.tracking import CircularEMA, ScalarEMA


_ALGO_MAP = {
    "MUSIC": doa_music_uca_2d,
    "BARTLETT": doa_bartlett_uca_2d,
    "CAPON": doa_capon_uca_2d,
}


class iridium_doa_estimator(gr.basic_block):
    def __init__(
        self,
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
        cov_alpha: float = 0.93,
        az_ema_alpha: float = 0.88,
        el_ema_alpha: float = 0.65,
        snr_min_db: float = -3.0,
        papr_min_db: float = 3.0,
        music_decorr: str = "fb",
    ):
        gr.basic_block.__init__(
            self,
            name="Iridium DOA Estimator",
            in_sig=None,
            out_sig=None,
        )

        self.algorithm = algorithm.upper()
        self.snr_min_db = snr_min_db
        self.papr_min_db = papr_min_db
        self.cov_alpha = cov_alpha
        self.music_decorr = music_decorr

        self.cfg = UcaConfig(
            n_ant=n_ant, radius_lambda=radius_lambda,
            n_az=n_az, n_el=n_el,
            el_min_deg=el_min_deg, el_max_deg=el_max_deg,
            num_expected_signals=num_signals,
            ant0_offset_deg=ant0_offset_deg, ant_ccw=ant_ccw,
        )

        self._R_ema = None
        self._az_ema = CircularEMA(alpha=az_ema_alpha)
        self._el_ema = ScalarEMA(alpha=el_ema_alpha)
        self._n_bursts = 0

        self.message_port_register_out(pmt.intern("azimuth"))
        self.message_port_register_out(pmt.intern("elevation"))
        self.message_port_register_out(pmt.intern("snr"))
        self.message_port_register_out(pmt.intern("spectrum"))
        self.message_port_register_out(pmt.intern("burst_detected"))

        self.message_port_register_in(pmt.intern("covariance"))
        self.set_msg_handler(pmt.intern("covariance"), self._handle_msg)

    def _handle_msg(self, msg):
        meta = pmt.car(msg)
        vec_data = pmt.cdr(msg)

        n_ant = int(pmt.to_long(pmt.dict_ref(meta, pmt.intern("R_mf_size"), pmt.from_long(5))))
        snr_db = float(pmt.to_double(pmt.dict_ref(meta, pmt.intern("snr_db"), pmt.from_double(-999.0))))
        tone_hz = float(pmt.to_double(pmt.dict_ref(meta, pmt.intern("tone_hz"), pmt.from_double(3125.0))))

        if snr_db < self.snr_min_db:
            self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
            return

        f32_vec = pmt.f32vector_elements(vec_data)
        R_mf = np.array(f32_vec, dtype=np.float32).view(np.complex64).reshape(n_ant, n_ant)

        if self._R_ema is None:
            self._R_ema = R_mf.copy()
        else:
            self._R_ema = self.cov_alpha * self._R_ema + (1 - self.cov_alpha) * R_mf

        X_dummy = np.eye(n_ant, dtype=np.complex64)

        az_est = None
        el_est = None
        spec = None

        if self.algorithm == "PHASE-FIT":
            result = doa_phase_fit_uca_2d(
                self._R_ema, self.cfg,
                az_hint_deg=self._az_ema.value if self._n_bursts > 0 else None,
                el_hint_deg=self._el_ema.value if self._n_bursts > 0 else None,
            )
            az_est, el_est, _ = result
            papr = 0.0
        else:
            algo_fn = _ALGO_MAP.get(self.algorithm, doa_music_uca_2d)
            if self.algorithm == "CAPON":
                spec = algo_fn(X_dummy, self.cfg, R_in=self._R_ema, decorr=self.music_decorr)
            elif self.algorithm == "BARTLETT":
                spec = algo_fn(X_dummy, self.cfg, R_in=self._R_ema)
            else:
                spec = algo_fn(X_dummy, self.cfg, R_in=self._R_ema)
            az_est, el_est, papr = find_peak_uca_2d(spec, self.cfg)

            if papr < self.papr_min_db:
                self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                return

        if az_est is not None:
            az_smooth = self._az_ema.update(az_est)
            el_smooth = self._el_ema.update(el_est)
        else:
            az_smooth = az_est
            el_smooth = el_est

        self._n_bursts += 1

        self.message_port_pub(pmt.intern("azimuth"), pmt.from_double(float(az_smooth)))
        self.message_port_pub(pmt.intern("elevation"), pmt.from_double(float(el_smooth)))
        self.message_port_pub(pmt.intern("snr"), pmt.from_double(float(snr_db)))
        self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(True))

        if spec is not None:
            flat = spec.ravel().astype(np.float32)
            v = pmt.init_f32vector(flat.size, flat)
            self.message_port_pub(pmt.intern("spectrum"), v)