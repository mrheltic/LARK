#!/usr/bin/env python3
"""
iridium_mf_covariance — GNU Radio block for matched-filter covariance estimation.

Input : PDU bpf_output (from iridium_bpf_normalizer)
Output: PDU message on port "covariance" with dict:
            R_mf (as f32 vector, n_ant × n_ant row-major),
            snr_db, tone_hz, burst_start

Delegates to apps.doa_iridium.burst_processing.compute_mf_covariance
            + core.doa_algorithms.apply_phase_correction.
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

from apps.doa_iridium.burst_processing import compute_mf_covariance
from core.doa_algorithms import apply_phase_correction


class iridium_mf_covariance(gr.sync_block):
    def __init__(
        self,
        n_ant: int = 5,
        pre_samples: int = 2621,
        bpf_guard: int = 128,
        fs: float = 1_024_000.0,
        phase_offsets_deg: list | None = None,
    ):
        gr.sync_block.__init__(
            self,
            name="Iridium MF Covariance",
            in_sig=None,
            out_sig=None,
        )
        self.n_ant = n_ant
        self.pre_samples = pre_samples
        self.bpf_guard = bpf_guard
        self.fs = fs
        self.phase_offsets_deg = phase_offsets_deg or [0.0] * n_ant

        self.message_port_register_out(pmt.intern("covariance"))

        self.message_port_register_in(pmt.intern("bpf_output"))
        self.set_msg_handler(pmt.intern("bpf_output"), self._handle_msg)

    def _handle_msg(self, msg):
        meta = pmt.car(msg)
        vec_data = pmt.cdr(msg)

        n_ant = int(pmt.to_long(pmt.dict_ref(meta, pmt.intern("n_ant"), pmt.from_long(self.n_ant))))
        pre_samples = int(pmt.to_long(pmt.dict_ref(meta, pmt.intern("pre_samples"), pmt.from_long(self.pre_samples))))
        tone_hz = float(pmt.to_double(pmt.dict_ref(meta, pmt.intern("tone_hz"), pmt.from_double(3125.0))))
        burst_start = int(pmt.to_long(pmt.dict_ref(meta, pmt.intern("burst_start"), pmt.from_long(0))))

        f32_vec = pmt.f32vector_elements(vec_data)
        X_bpf = np.array(f32_vec, dtype=np.float32).view(np.complex64).reshape(n_ant, pre_samples)

        X_cal = apply_phase_correction(X_bpf, self.phase_offsets_deg)

        try:
            R_mf, y_mf, snr_db = compute_mf_covariance(
                X_cal, tone_hz, self.fs, self.pre_samples, self.bpf_guard
            )
        except ValueError:
            return

        R_flat = R_mf.astype(np.complex64).view(np.float32)
        R_vec = pmt.init_f32vector(R_flat.size, R_flat)

        msg = pmt.make_dict()
        msg = pmt.dict_add(msg, pmt.intern("R_mf_size"), pmt.from_long(n_ant))
        msg = pmt.dict_add(msg, pmt.intern("snr_db"), pmt.from_double(float(snr_db)))
        msg = pmt.dict_add(msg, pmt.intern("tone_hz"), pmt.from_double(float(tone_hz)))
        msg = pmt.dict_add(msg, pmt.intern("burst_start"), pmt.from_long(int(burst_start)))

        self.message_port_pub(pmt.intern("covariance"), pmt.cons(msg, R_vec))