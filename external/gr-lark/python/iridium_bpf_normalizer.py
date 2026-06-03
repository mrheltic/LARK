#!/usr/bin/env python3
"""
iridium_bpf_normalizer — GNU Radio block for BPF extraction + amplitude normalization.

Input : 5 vector streams of CPI length + PDU tone_info
Output: PDU message on port "bpf_output" with dict:
            X_bpf (as f32 vector, row-major n_ant × pre_samples),
            tone_hz, bpf_bw_hz, snr_db, burst_start

Delegates to apps.doa_iridium.burst_processing.apply_bpf_and_normalize.
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

from apps.doa_iridium.burst_processing import apply_bpf_and_normalize


class iridium_bpf_normalizer(gr.sync_block):
    def __init__(
        self,
        cpi_size: int = 131072,
        fs: float = 1_024_000.0,
        n_ant: int = 5,
        pre_samples: int = 2621,
        bpf_bw_hz: float = 8_000.0,
    ):
        gr.sync_block.__init__(
            self,
            name="Iridium BPF Normalizer",
            in_sig=[(np.complex64, cpi_size)] * n_ant,
            out_sig=None,
        )
        self.cpi_size = cpi_size
        self.fs = fs
        self.n_ant = n_ant
        self.pre_samples = min(pre_samples, cpi_size)
        self.bpf_bw_hz = bpf_bw_hz

        self.message_port_register_out(pmt.intern("bpf_output"))

        self._tone_info = None
        self.message_port_register_in(pmt.intern("tone_info"))
        self.set_msg_handler(pmt.intern("tone_info"), self._handle_tone_info)

    def _handle_tone_info(self, msg):
        meta = pmt.car(msg)
        self._tone_info = {}
        for key in ["tone_hz", "snr_db", "doppler_hz", "burst_start", "burst_end"]:
            sym = pmt.intern(key)
            if pmt.dict_has_key(meta, sym):
                val = pmt.dict_ref(meta, sym, pmt.PMT_NIL)
                if pmt.is_real(val):
                    self._tone_info[key] = pmt.to_double(val)
                elif pmt.is_integer(val):
                    self._tone_info[key] = pmt.to_long(val)

    def work(self, input_items, output_items):
        n_frames = len(input_items[0])
        if n_frames == 0:
            return 0

        for fi in range(n_frames):
            X = np.empty((self.n_ant, self.cpi_size), dtype=np.complex64)
            for ch in range(self.n_ant):
                X[ch, :] = input_items[ch][fi]

            if self._tone_info is None:
                continue

            tone_hz = self._tone_info.get("tone_hz", 3125.0)
            bs = int(self._tone_info.get("burst_start", 0))
            be = int(self._tone_info.get("burst_end", min(bs + 3000, self.cpi_size)))
            snr_db = self._tone_info.get("snr_db", -999.0)

            X_win = X[:, bs:be]
            if X_win.shape[1] < self.pre_samples:
                self._tone_info = None
                continue

            try:
                X_bpf = apply_bpf_and_normalize(
                    X_win, self.pre_samples, self.fs, tone_hz, self.bpf_bw_hz
                )
            except ValueError:
                self._tone_info = None
                continue

            flat = X_bpf.astype(np.complex64).view(np.float32)
            vec = pmt.init_f32vector(flat.size, flat)
            msg = pmt.make_dict()
            msg = pmt.dict_add(msg, pmt.intern("X_bpf"), vec)
            msg = pmt.dict_add(msg, pmt.intern("n_ant"), pmt.from_long(self.n_ant))
            msg = pmt.dict_add(msg, pmt.intern("pre_samples"), pmt.from_long(self.pre_samples))
            msg = pmt.dict_add(msg, pmt.intern("tone_hz"), pmt.from_double(float(tone_hz)))
            msg = pmt.dict_add(msg, pmt.intern("bpf_bw_hz"), pmt.from_double(float(self.bpf_bw_hz)))
            msg = pmt.dict_add(msg, pmt.intern("snr_db"), pmt.from_double(float(snr_db)))
            msg = pmt.dict_add(msg, pmt.intern("burst_start"), pmt.from_long(int(bs)))

            self.message_port_pub(pmt.intern("bpf_output"), pmt.cons(msg, vec))
            self._tone_info = None

        return n_frames