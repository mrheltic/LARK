#!/usr/bin/env python3
"""Generate GRC flowgraph files with properly escaped epy_block source code.

NOTE: The epy_block approach for GRC has proven fragile in practice.
The recommended way to run the Iridium DOA pipeline is:

  Live:    python3 run_doa.py (standalone) or python3 run_doa.py --use-gr (GNU Radio)
  Offline: python3 run_doa_offline.py capture.npz

This script is kept for reference if you want to attempt GRC integration.
Uses yaml.dump() to ensure proper YAML escaping of the epy_block source field.
"""

import os
import yaml

EPY_BLOCK_SOURCE = """\
import numpy as np
from gnuradio import gr
import pmt

class iridium_doa_processor(gr.sync_block):
    def __init__(self, cpi_size=131072, fs=1024000.0, freq_hz=1626270000.0, n_ant=5,
                 radius_lambda=0.4253, ant0_offset_deg=0.0, ant_ccw=False,
                 n_az=360, n_el=86, el_min_deg=5.0, el_max_deg=90.0,
                 algorithm="MUSIC", num_signals=1,
                 phase_offsets_deg="0.0,54.95,137.24,133.58,48.31",
                 energy_threshold=3.0, tone_nom_hz=3125.0, tone_scan_bw_hz=3000.0,
                 tone_min_snr_db=2.0, bpf_bw_hz=8000.0, pre_samples=2621,
                 bpf_guard=128, window_samples=3000, cov_alpha=0.93,
                 az_ema_alpha=0.88, el_ema_alpha=0.65,
                 snr_min_db=-3.0, papr_min_db=3.0):
        gr.sync_block.__init__(self, name="Iridium DoA",
            in_sig=[(np.complex64, cpi_size)] * n_ant, out_sig=None)
        self.cpi_size = cpi_size
        self.fs = fs
        self.freq_hz = freq_hz
        self.n_ant = n_ant
        self.algorithm = algorithm.upper() if isinstance(algorithm, str) else "MUSIC"
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
        self.radius_lambda = radius_lambda
        self.ant0_offset_deg = ant0_offset_deg
        self.ant_ccw = ant_ccw
        self.n_az = n_az
        self.n_el = n_el
        self.el_min_deg = el_min_deg
        self.el_max_deg = el_max_deg
        self.num_signals = num_signals
        self._phase_offsets_deg_str = phase_offsets_deg
        self._R_ema = None
        self._az_ema = None
        self._el_ema = None
        self._n_bursts = 0
        self._initialized = False
        self.message_port_register_out(pmt.intern("azimuth"))
        self.message_port_register_out(pmt.intern("elevation"))
        self.message_port_register_out(pmt.intern("spectrum"))
        self.message_port_register_out(pmt.intern("snr"))
        self.message_port_register_out(pmt.intern("burst_detected"))

    def _lazy_init(self):
        if self._initialized:
            return
        import sys, os
        _here = os.path.dirname(os.path.realpath(__file__))
        _paths = [
            os.path.abspath(os.path.join(_here, "..", "src")),
            os.path.abspath(os.path.join(_here, "..", "krakenSDR", "src")),
            os.path.abspath(os.path.join(_here, "..", "..", "krakenSDR", "src")),
            os.path.abspath(os.path.join(_here, "..", "..", "..", "krakenSDR", "src")),
            os.path.abspath(os.path.join(_here, "..", "doa_iridium")),
            os.path.abspath(os.path.join(_here, "..", "..", "krakenSDR", "src", "apps", "doa_iridium")),
        ]
        for p in _paths:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        from core.doa_uca_2d import (
            UcaConfig, doa_music_uca_2d, doa_bartlett_uca_2d,
            doa_capon_uca_2d, doa_phase_fit_uca_2d, find_peak_uca_2d)
        from core.doa_algorithms import apply_phase_correction
        from apps.doa_iridium.burst_processing import (
            detect_energy_bursts, scan_preamble_tones,
            compute_mf_covariance, apply_bpf_and_normalize)
        self._phase_offsets_deg = (
            [float(x) for x in self._phase_offsets_deg_str.split(",")]
            if isinstance(self._phase_offsets_deg_str, str)
            else list(self._phase_offsets_deg_str))
        self._cfg = UcaConfig(
            n_ant=self.n_ant, radius_lambda=self.radius_lambda,
            n_az=self.n_az, n_el=self.n_el,
            el_min_deg=self.el_min_deg, el_max_deg=self.el_max_deg,
            num_expected_signals=self.num_signals,
            ant0_offset_deg=self.ant0_offset_deg, ant_ccw=self.ant_ccw)
        self._algo_map = {
            "MUSIC": doa_music_uca_2d, "BARTLETT": doa_bartlett_uca_2d,
            "CAPON": doa_capon_uca_2d, "PHASE-FIT": None}
        self._detect_energy_bursts = detect_energy_bursts
        self._scan_preamble_tones = scan_preamble_tones
        self._compute_mf_covariance = compute_mf_covariance
        self._apply_bpf_and_normalize = apply_bpf_and_normalize
        self._apply_phase_correction = apply_phase_correction
        self._find_peak_uca_2d = find_peak_uca_2d
        self._doa_music_uca_2d = doa_music_uca_2d
        self._doa_bartlett_uca_2d = doa_bartlett_uca_2d
        self._doa_capon_uca_2d = doa_capon_uca_2d
        self._doa_phase_fit_uca_2d = doa_phase_fit_uca_2d
        self._initialized = True

    def work(self, input_items, output_items):
        if not self._initialized:
            self._lazy_init()
        n_ant = self.n_ant
        n_frames = len(input_items[0])
        if n_frames == 0:
            return 0
        for fi in range(n_frames):
            X = np.empty((n_ant, self.cpi_size), dtype=np.complex64)
            for ch in range(n_ant):
                X[ch, :] = input_items[ch][fi]
            burst_starts = self._detect_energy_bursts(
                X[0, :], self.fs, energy_window=256,
                threshold_factor=self.energy_threshold,
                min_gap_samples=int(0.045 * self.fs))
            if not burst_starts:
                self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                continue
            burst_start = burst_starts[0]
            burst_end = min(burst_start + self.window_samples, X.shape[1])
            if burst_end - burst_start < self.pre_samples + self.bpf_guard:
                self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                continue
            tones = self._scan_preamble_tones(
                X[0, burst_start:burst_end], self.fs, self.tone_nom_hz,
                scan_bw_hz=self.tone_scan_bw_hz, n_peaks=1,
                min_sep_hz=1000.0, min_snr_db=self.tone_min_snr_db, dc_guard_hz=200.0)
            if not tones:
                self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                continue
            tone_hz = tones[0][0]
            X_win = X[:, burst_start:burst_end]
            if X_win.shape[1] < self.window_samples:
                self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                continue
            try:
                X_bpf = self._apply_bpf_and_normalize(
                    X_win, self.window_samples, self.fs, tone_hz, self.bpf_bw_hz)
            except ValueError:
                self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                continue
            X_cal = self._apply_phase_correction(X_bpf, self._phase_offsets_deg)
            try:
                R_mf, y_mf, snr_db = self._compute_mf_covariance(
                    X_cal, tone_hz, self.fs, self.pre_samples, self.bpf_guard)
            except ValueError:
                self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                continue
            if snr_db < self.snr_min_db:
                self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                continue
            if self._R_ema is None:
                self._R_ema = R_mf.copy()
            else:
                self._R_ema = self.cov_alpha * self._R_ema + (1 - self.cov_alpha) * R_mf
            algo_fn = self._algo_map.get(self.algorithm)
            spec = None
            if self.algorithm == "PHASE-FIT":
                az_est, el_est, _ = self._doa_phase_fit_uca_2d(
                    self._R_ema, self._cfg,
                    az_hint_deg=self._az_ema, el_hint_deg=self._el_ema)
            elif algo_fn is not None:
                if self.algorithm == "CAPON":
                    spec = algo_fn(X_cal, self._cfg, R_in=self._R_ema, decorr="none")
                else:
                    spec = algo_fn(X_cal, self._cfg, R_in=self._R_ema)
                az_est, el_est, papr = self._find_peak_uca_2d(spec, self._cfg)
                if papr < self.papr_min_db:
                    self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
                    continue
            else:
                spec = self._doa_music_uca_2d(X_cal, self._cfg, R_in=self._R_ema)
                az_est, el_est, papr = self._find_peak_uca_2d(spec, self._cfg)
                if papr < self.papr_min_db:
                    self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(False))
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
            self.message_port_pub(pmt.intern("azimuth"), pmt.from_double(float(self._az_ema)))
            self.message_port_pub(pmt.intern("elevation"), pmt.from_double(float(self._el_ema)))
            self.message_port_pub(pmt.intern("snr"), pmt.from_double(float(snr_db)))
            self.message_port_pub(pmt.intern("burst_detected"), pmt.from_bool(True))
            if spec is not None:
                flat = spec.ravel().astype(np.float32)
                v = pmt.init_f32vector(flat.size, flat)
                self.message_port_pub(pmt.intern("spectrum"), v)
        return n_frames"""


def _st(x, y, w=80, h=False):
    return dict(bus_sink=False, bus_source=False, bus_structure=None,
                coordinate=[x, y], rotation=0, state='enabled' if h else 'enabled')


def _blk(name, blk_id, params, states_extra=None, enabled=True):
    blk = dict(name=name, id=blk_id, parameters=params,
               states=_st(0, 0))
    if states_extra:
        blk['states'].update(states_extra)
    return blk


def generate_lband_grc():
    grc = {
        'options': {
            'parameters': {
                'author': 'LARK',
                'catch_exceptions': 'True',
                'category': '[LARK]',
                'cmake_opt': '',
                'comment': '',
                'gen_cmake': 'On',
                'gen_linking': 'dynamic',
                'generate_options': 'qt_gui',
                'hier_block_src_path': '.:',
                'id': 'iridium_doa_lband',
                'max_nouts': '0',
                'output_language': 'python',
                'placement': '(0,0)',
                'qt_qss_theme': '',
                'realtime_scheduling': '',
                'run': 'True',
                'run_command': '{python} -u {filename}',
                'run_options': 'prompt',
                'sizing_mode': 'fixed',
                'thread_safe_setters': '',
                'title': 'Iridium DOA L-band (Live)',
                'window_size': '(1920,1080)',
            },
            'states': _st(8, 8),
        },
        'blocks': [
            _blk('cpi_size', 'variable', {'comment': '', 'value': '131072'}, {'coordinate': [184, 12]}),
            _blk('decimation', 'variable', {'comment': '', 'value': '2'}, {'coordinate': [296, 12]}),
            _blk('freq', 'variable', {'comment': '', 'value': '1626.27'}, {'coordinate': [504, 12]}),
            _blk('samp_rate', 'variable', {'comment': '', 'value': '2400000'}, {'coordinate': [400, 12]}),
            _blk('fir_filter_xxx_0', 'fir_filter_xxx',
                 {'affinity': '', 'alias': '', 'comment': '', 'decim': 'decimation',
                  'maxoutbuf': '0', 'minoutbuf': '0', 'samp_delay': '0',
                  'taps': 'decimation*5', 'type': 'ccc'}, {'coordinate': [344, 196]}),
            _blk('fir_filter_xxx_0_0', 'fir_filter_xxx',
                 {'affinity': '', 'alias': '', 'comment': '', 'decim': 'decimation',
                  'maxoutbuf': '0', 'minoutbuf': '0', 'samp_delay': '0',
                  'taps': 'decimation*5', 'type': 'ccc'}, {'coordinate': [344, 260]}),
            _blk('fir_filter_xxx_0_0_0', 'fir_filter_xxx',
                 {'affinity': '', 'alias': '', 'comment': '', 'decim': 'decimation',
                  'maxoutbuf': '0', 'minoutbuf': '0', 'samp_delay': '0',
                  'taps': 'decimation*5', 'type': 'ccc'}, {'coordinate': [344, 324]}),
            _blk('fir_filter_xxx_0_0_1', 'fir_filter_xxx',
                 {'affinity': '', 'alias': '', 'comment': '', 'decim': 'decimation',
                  'maxoutbuf': '0', 'minoutbuf': '0', 'samp_delay': '0',
                  'taps': 'decimation*5', 'type': 'ccc'}, {'coordinate': [344, 388]}),
            _blk('fir_filter_xxx_0_0_2', 'fir_filter_xxx',
                 {'affinity': '', 'alias': '', 'comment': '', 'decim': 'decimation',
                  'maxoutbuf': '0', 'minoutbuf': '0', 'samp_delay': '0',
                  'taps': 'decimation*5', 'type': 'ccc'}, {'coordinate': [344, 452]}),
            _blk('blocks_stream_to_vector_0', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size//decimation',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [592, 272]}),
            _blk('blocks_stream_to_vector_0_0', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size//decimation',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [592, 304]}),
            _blk('blocks_stream_to_vector_0_0_0', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size//decimation',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [592, 336]}),
            _blk('blocks_stream_to_vector_0_0_1', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size//decimation',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [592, 368]}),
            _blk('blocks_stream_to_vector_0_0_2', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size//decimation',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [592, 400]}),
            _blk('iridium_doa_processor_0', 'epy_block', {
                'affinity': '', 'alias': '', 'comment': '',
                '_io_count': '0',
                'algorithm': 'MUSIC',
                'ant0_offset_deg': '0.0',
                'ant_ccw': 'False',
                'az_ema_alpha': '0.88',
                'bpf_bw_hz': '8000.0',
                'bpf_guard': '128',
                'cpi_size': 'cpi_size//decimation',
                'cov_alpha': '0.93',
                'el_ema_alpha': '0.65',
                'el_max_deg': '90.0',
                'el_min_deg': '5.0',
                'energy_threshold': '3.0',
                'fs': 'samp_rate/decimation',
                'freq_hz': 'freq*1e6',
                'n_ant': '5',
                'n_az': '360',
                'n_el': '86',
                'num_signals': '1',
                'papr_min_db': '3.0',
                'phase_offsets_deg': '0.0,54.95,137.24,133.58,48.31',
                'pre_samples': '2621',
                'radius_lambda': '0.4253',
                'snr_min_db': '-3.0',
                'tone_min_snr_db': '2.0',
                'tone_nom_hz': '3125.0',
                'tone_scan_bw_hz': '3000.0',
                'window_samples': '3000',
                'maxoutbuf': '0',
                'minoutbuf': '0',
            }, {'coordinate': [816, 272], 'source': EPY_BLOCK_SOURCE}),
            _blk('krakensdr_krakensdr_source_0', 'krakensdr_krakensdr_source', {
                'affinity': '', 'alias': '', 'comment': '',
                'ctrlPort': '5001', 'debug': 'False',
                'freq': 'freq',
                'gain': '[40.2, 40.2, 40.2, 40.2, 40.2]',
                'ipAddr': '127.0.0.1',
                'maxoutbuf': '0', 'minoutbuf': '0',
                'numChannels': '5', 'port': '5000',
            }, {'coordinate': [16, 320], 'state': True}),
            _blk('blocks_message_debug_0', 'blocks_message_debug', {
                'affinity': '', 'alias': '', 'comment': '',
            }, {'coordinate': [1100, 272], 'state': True}),
        ],
        'connections': [
            ['krakensdr_krakensdr_source_0', '0', 'fir_filter_xxx_0', '0'],
            ['krakensdr_krakensdr_source_0', '1', 'fir_filter_xxx_0_0', '0'],
            ['krakensdr_krakensdr_source_0', '2', 'fir_filter_xxx_0_0_0', '0'],
            ['krakensdr_krakensdr_source_0', '3', 'fir_filter_xxx_0_0_1', '0'],
            ['krakensdr_krakensdr_source_0', '4', 'fir_filter_xxx_0_0_2', '0'],
            ['fir_filter_xxx_0', '0', 'blocks_stream_to_vector_0', '0'],
            ['fir_filter_xxx_0_0', '0', 'blocks_stream_to_vector_0_0', '0'],
            ['fir_filter_xxx_0_0_0', '0', 'blocks_stream_to_vector_0_0_0', '0'],
            ['fir_filter_xxx_0_0_1', '0', 'blocks_stream_to_vector_0_0_1', '0'],
            ['fir_filter_xxx_0_0_2', '0', 'blocks_stream_to_vector_0_0_2', '0'],
            ['blocks_stream_to_vector_0', '0', 'iridium_doa_processor_0', '0'],
            ['blocks_stream_to_vector_0_0', '0', 'iridium_doa_processor_0', '1'],
            ['blocks_stream_to_vector_0_0_0', '0', 'iridium_doa_processor_0', '2'],
            ['blocks_stream_to_vector_0_0_1', '0', 'iridium_doa_processor_0', '3'],
            ['blocks_stream_to_vector_0_0_2', '0', 'iridium_doa_processor_0', '4'],
            ['iridium_doa_processor_0', 'azimuth', 'blocks_message_debug_0', 'print'],
            ['iridium_doa_processor_0', 'elevation', 'blocks_message_debug_0', 'print'],
            ['iridium_doa_processor_0', 'snr', 'blocks_message_debug_0', 'print'],
            ['iridium_doa_processor_0', 'burst_detected', 'blocks_message_debug_0', 'print'],
        ],
        'metadata': {'file_format': 1, 'grc_version': '3.10.9.2'},
    }
    return yaml.dump(grc, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate_offline_grc():
    grc = {
        'options': {
            'parameters': {
                'author': 'LARK',
                'catch_exceptions': 'True',
                'category': '[LARK]',
                'cmake_opt': '',
                'comment': ('Offline DoA processing from captured complex float32 IQ files.\n'
                            'Set file paths to 5 channel capture files (one per antenna).\n'
                            'Each file should be raw complex64 (interleaved I/Q).\n'
                            'Set cpi_size and fs to match capture parameters.'),
                'gen_cmake': 'On',
                'gen_linking': 'dynamic',
                'generate_options': 'no_gui',
                'hier_block_src_path': '.:',
                'id': 'iridium_doa_offline',
                'max_nouts': '0',
                'output_language': 'python',
                'placement': '(0,0)',
                'qt_qss_theme': '',
                'realtime_scheduling': '',
                'run': 'True',
                'run_command': '{python} -u {filename}',
                'run_options': 'prompt',
                'sizing_mode': 'fixed',
                'thread_safe_setters': '',
                'title': 'Iridium DOA L-band (Offline)',
                'window_size': '(1000,1000)',
            },
            'states': _st(8, 8),
        },
        'blocks': [
            _blk('cpi_size', 'variable', {'comment': '', 'value': '131072'}, {'coordinate': [184, 12]}),
            _blk('fs', 'variable', {'comment': 'Effective sample rate (Hz)', 'value': '1024000.0'}, {'coordinate': [296, 12]}),
            _blk('file_source_ch0', 'blocks_file_source',
                 {'affinity': '', 'alias': '', 'comment': 'Channel 0 IQ data (complex float32)',
                  'file': '/tmp/ch0.cf32', 'length': '0',
                  'maxoutbuf': '0', 'minoutbuf': '0',
                  'repeat': 'True', 'type': 'complex', 'vlen': '1'}, {'coordinate': [16, 200]}),
            _blk('file_source_ch1', 'blocks_file_source',
                 {'affinity': '', 'alias': '', 'comment': 'Channel 1 IQ data (complex float32)',
                  'file': '/tmp/ch1.cf32', 'length': '0',
                  'maxoutbuf': '0', 'minoutbuf': '0',
                  'repeat': 'True', 'type': 'complex', 'vlen': '1'}, {'coordinate': [16, 260]}),
            _blk('file_source_ch2', 'blocks_file_source',
                 {'affinity': '', 'alias': '', 'comment': 'Channel 2 IQ data (complex float32)',
                  'file': '/tmp/ch2.cf32', 'length': '0',
                  'maxoutbuf': '0', 'minoutbuf': '0',
                  'repeat': 'True', 'type': 'complex', 'vlen': '1'}, {'coordinate': [16, 320]}),
            _blk('file_source_ch3', 'blocks_file_source',
                 {'affinity': '', 'alias': '', 'comment': 'Channel 3 IQ data (complex float32)',
                  'file': '/tmp/ch3.cf32', 'length': '0',
                  'maxoutbuf': '0', 'minoutbuf': '0',
                  'repeat': 'True', 'type': 'complex', 'vlen': '1'}, {'coordinate': [16, 380]}),
            _blk('file_source_ch4', 'blocks_file_source',
                 {'affinity': '', 'alias': '', 'comment': 'Channel 4 IQ data (complex float32)',
                  'file': '/tmp/ch4.cf32', 'length': '0',
                  'maxoutbuf': '0', 'minoutbuf': '0',
                  'repeat': 'True', 'type': 'complex', 'vlen': '1'}, {'coordinate': [16, 440]}),
            _blk('blocks_stream_to_vector_0', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [248, 200]}),
            _blk('blocks_stream_to_vector_0_0', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [248, 260]}),
            _blk('blocks_stream_to_vector_0_0_0', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [248, 320]}),
            _blk('blocks_stream_to_vector_0_0_1', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [248, 380]}),
            _blk('blocks_stream_to_vector_0_0_2', 'blocks_stream_to_vector',
                 {'affinity': '', 'alias': '', 'comment': '', 'maxoutbuf': '0',
                  'minoutbuf': '0', 'num_items': 'cpi_size',
                  'type': 'complex', 'vlen': '1'}, {'coordinate': [248, 440]}),
            _blk('iridium_doa_processor_0', 'epy_block', {
                'affinity': '', 'alias': '', 'comment': '',
                '_io_count': '0',
                'algorithm': 'MUSIC',
                'ant0_offset_deg': '0.0',
                'ant_ccw': 'False',
                'az_ema_alpha': '0.88',
                'bpf_bw_hz': '8000.0',
                'bpf_guard': '0',
                'cpi_size': 'cpi_size',
                'cov_alpha': '0.93',
                'el_ema_alpha': '0.65',
                'el_max_deg': '90.0',
                'el_min_deg': '5.0',
                'energy_threshold': '3.0',
                'fs': 'fs',
                'freq_hz': '1626270000.0',
                'n_ant': '5',
                'n_az': '360',
                'n_el': '86',
                'num_signals': '1',
                'papr_min_db': '3.0',
                'phase_offsets_deg': '0.0,54.95,137.24,133.58,48.31',
                'pre_samples': '2621',
                'radius_lambda': '0.4253',
                'snr_min_db': '-3.0',
                'tone_min_snr_db': '2.0',
                'tone_nom_hz': '3125.0',
                'tone_scan_bw_hz': '3000.0',
                'window_samples': '3000',
                'maxoutbuf': '0',
                'minoutbuf': '0',
            }, {'coordinate': [456, 272], 'source': EPY_BLOCK_SOURCE}),
            _blk('blocks_message_debug_0', 'blocks_message_debug', {
                'affinity': '', 'alias': '', 'comment': '',
            }, {'coordinate': [672, 200], 'state': True}),
        ],
        'connections': [
            ['file_source_ch0', '0', 'blocks_stream_to_vector_0', '0'],
            ['file_source_ch1', '0', 'blocks_stream_to_vector_0_0', '0'],
            ['file_source_ch2', '0', 'blocks_stream_to_vector_0_0_0', '0'],
            ['file_source_ch3', '0', 'blocks_stream_to_vector_0_0_1', '0'],
            ['file_source_ch4', '0', 'blocks_stream_to_vector_0_0_2', '0'],
            ['blocks_stream_to_vector_0', '0', 'iridium_doa_processor_0', '0'],
            ['blocks_stream_to_vector_0_0', '0', 'iridium_doa_processor_0', '1'],
            ['blocks_stream_to_vector_0_0_0', '0', 'iridium_doa_processor_0', '2'],
            ['blocks_stream_to_vector_0_0_1', '0', 'iridium_doa_processor_0', '3'],
            ['blocks_stream_to_vector_0_0_2', '0', 'iridium_doa_processor_0', '4'],
            ['iridium_doa_processor_0', 'azimuth', 'blocks_message_debug_0', 'print'],
            ['iridium_doa_processor_0', 'elevation', 'blocks_message_debug_0', 'print'],
            ['iridium_doa_processor_0', 'snr', 'blocks_message_debug_0', 'print'],
            ['iridium_doa_processor_0', 'burst_detected', 'blocks_message_debug_0', 'print'],
        ],
        'metadata': {'file_format': 1, 'grc_version': '3.10.9.2'},
    }
    return yaml.dump(grc, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    lband_path = os.path.join(out_dir, "iridium_doa_lband.grc")
    with open(lband_path, "w") as f:
        f.write(generate_lband_grc())
    print(f"Written: {lband_path}")

    offline_path = os.path.join(out_dir, "iridium_doa_offline.grc")
    with open(offline_path, "w") as f:
        f.write(generate_offline_grc())
    print(f"Written: {offline_path}")


if __name__ == "__main__":
    main()