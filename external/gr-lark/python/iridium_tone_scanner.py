#!/usr/bin/env python3
"""
iridium_tone_scanner — GNU Radio block for Iridium preamble tone scanning.

Input : 5 vector streams of CPI length + PDU burst_info
Output: PDU message on port "tone_info" with dict:
            tone_hz, snr_db, doppler_hz, burst_start, burst_end

Delegates to apps.doa_iridium.burst_processing.scan_preamble_tones.
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

from apps.doa_iridium.burst_processing import scan_preamble_tones


class iridium_tone_scanner(gr.sync_block):
    def __init__(
        self,
        cpi_size: int = 131072,
        fs: float = 1_024_000.0,
        n_ant: int = 5,
        tone_nom_hz: float = 3125.0,
        scan_bw_hz: float = 3_000.0,
        n_peaks: int = 3,
        min_sep_hz: float = 1_000.0,
        min_snr_db: float = 2.0,
        dc_guard_hz: float = 200.0,
    ):
        gr.sync_block.__init__(
            self,
            name="Iridium Tone Scanner",
            in_sig=[(np.complex64, cpi_size)] * n_ant,
            out_sig=None,
        )
        self.cpi_size = cpi_size
        self.fs = fs
        self.n_ant = n_ant
        self.tone_nom_hz = tone_nom_hz
        self.scan_bw_hz = scan_bw_hz
        self.n_peaks = n_peaks
        self.min_sep_hz = min_sep_hz
        self.min_snr_db = min_snr_db
        self.dc_guard_hz = dc_guard_hz

        self.message_port_register_out(pmt.intern("tone_info"))

        self._burst_info = None
        self.message_port_register_in(pmt.intern("burst_info"))
        self.set_msg_handler(pmt.intern("burst_info"), self._handle_burst_info)

    def _handle_burst_info(self, msg):
        meta = pmt.car(msg)
        self._burst_info = {}
        for key in ["burst_start", "burst_end", "energy_db", "frame_index"]:
            sym = pmt.intern(key)
            if pmt.dict_has_key(meta, sym):
                val = pmt.dict_ref(meta, sym, pmt.PMT_NIL)
                if pmt.is_integer(val):
                    self._burst_info[key] = pmt.to_long(val)
                elif pmt.is_real(val):
                    self._burst_info[key] = pmt.to_double(val)

    def work(self, input_items, output_items):
        n_frames = len(input_items[0])
        if n_frames == 0:
            return 0

        for fi in range(n_frames):
            X = np.empty((self.n_ant, self.cpi_size), dtype=np.complex64)
            for ch in range(self.n_ant):
                X[ch, :] = input_items[ch][fi]

            if self._burst_info is None:
                continue

            bs = self._burst_info.get("burst_start", 0)
            be = self._burst_info.get("burst_end", min(bs + 3000, self.cpi_size))
            if be > self.cpi_size:
                be = self.cpi_size

            segment = X[0, bs:be]
            if len(segment) < 128:
                self._burst_info = None
                continue

            tones = scan_preamble_tones(
                segment, self.fs,
                nom_tone_hz=self.tone_nom_hz,
                scan_bw_hz=self.scan_bw_hz,
                n_peaks=self.n_peaks,
                min_sep_hz=self.min_sep_hz,
                min_snr_db=self.min_snr_db,
                dc_guard_hz=self.dc_guard_hz,
            )

            if tones:
                tone_hz, snr_db = tones[0]
                doppler_hz = tone_hz - self.tone_nom_hz
            else:
                tone_hz = self.tone_nom_hz
                snr_db = -999.0
                doppler_hz = 0.0

            msg = pmt.make_dict()
            msg = pmt.dict_add(msg, pmt.intern("tone_hz"), pmt.from_double(float(tone_hz)))
            msg = pmt.dict_add(msg, pmt.intern("snr_db"), pmt.from_double(float(snr_db)))
            msg = pmt.dict_add(msg, pmt.intern("doppler_hz"), pmt.from_double(float(doppler_hz)))
            msg = pmt.dict_add(msg, pmt.intern("burst_start"), pmt.from_long(int(bs)))
            msg = pmt.dict_add(msg, pmt.intern("burst_end"), pmt.from_long(int(be)))

            self.message_port_pub(pmt.intern("tone_info"), pmt.cons(msg, pmt.PMT_NIL))
            self._burst_info = None

        return n_frames