#!/usr/bin/env python3
"""
iridium_burst_energy — GNU Radio block for Iridium burst energy detection.

Input : 5 vector streams of CPI length (complex64 vector)
Output: PDU message on port "burst_info" with dict:
            burst_start, burst_end, energy_db, frame_index

Delegates 100% of DSP to apps.doa_iridium.burst_processing.detect_energy_bursts.
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

from apps.doa_iridium.burst_processing import detect_energy_bursts


class iridium_burst_energy(gr.sync_block):
    def __init__(
        self,
        cpi_size: int = 131072,
        fs: float = 1_024_000.0,
        n_ant: int = 5,
        energy_window: int = 256,
        threshold_factor: float = 3.0,
        min_gap_s: float = 0.045,
    ):
        gr.sync_block.__init__(
            self,
            name="Iridium Burst Energy",
            in_sig=[(np.complex64, cpi_size)] * n_ant,
            out_sig=None,
        )
        self.cpi_size = cpi_size
        self.fs = fs
        self.n_ant = n_ant
        self.energy_window = energy_window
        self.threshold_factor = threshold_factor
        self.min_gap_samples = max(energy_window, int(min_gap_s * fs))
        self._frame_idx = 0

        self.message_port_register_out(pmt.intern("burst_info"))

    def work(self, input_items, output_items):
        n_frames = len(input_items[0])
        if n_frames == 0:
            return 0

        for fi in range(n_frames):
            X = np.empty((self.n_ant, self.cpi_size), dtype=np.complex64)
            for ch in range(self.n_ant):
                X[ch, :] = input_items[ch][fi]

            starts = detect_energy_bursts(
                X[0, :],
                self.fs,
                energy_window=self.energy_window,
                threshold_factor=self.threshold_factor,
                min_gap_samples=self.min_gap_samples,
            )

            for bs in starts:
                be = min(bs + int(0.01044 * self.fs), self.cpi_size)
                ch0_seg = X[0, bs:be]
                pwr = float(np.mean(np.abs(ch0_seg) ** 2))
                energy_db = 10.0 * np.log10(pwr + 1e-20)

                msg = pmt.make_dict()
                msg = pmt.dict_add(msg, pmt.intern("burst_start"), pmt.from_long(bs))
                msg = pmt.dict_add(msg, pmt.intern("burst_end"), pmt.from_long(be))
                msg = pmt.dict_add(msg, pmt.intern("energy_db"), pmt.from_double(energy_db))
                msg = pmt.dict_add(msg, pmt.intern("frame_index"), pmt.from_long(self._frame_idx))

                self.message_port_pub(pmt.intern("burst_info"), pmt.cons(msg, pmt.PMT_NIL))

            self._frame_idx += 1

        return n_frames