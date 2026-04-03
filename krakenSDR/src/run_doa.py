#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KrakenSDR DoA N-antenna -- standalone application with display widgets
======================================================================
Supports 2-5 antenna channels with ULA or UCA array geometry.
Standalone file: does not depend on any GRC-generated script.

Configuration (environment variables or defaults)
--------------------------------------------------
  NUM_CHANNELS  number of antennas/channels  (default: 2)
  ARRAY_TYPE    'ULA' or 'UCA'               (default: ULA)
  CENTER_FREQ   centre frequency in MHz       (default: 868.0)
  GAIN_DB       RF gain in dB                 (default: 40.2)
  ARRAY_DIST    antenna spacing in metres     (default: 0.17)

Launch (inside the container):
    python3 /workspace/run_doa.py

UI Layout
---------
  Row 0:   freq slider | gain | array_dist | estimated range
  Row 1-2: CH0 FFT (col 0-2) | Polar MUSIC spectrum widget (col 3-4)
  Row 3:   Compass (col 0-1) | 2D Map (col 2-3) | quality gauge (col 4)
  Row 4:   Bearing history (col 0-2) | Calibration (col 3-4)

Phase calibration
-----------------
Calibration corrects the hardware phase offset between the RF branches.
The offset is saved to /workspace/.doa_calibration.json and reloaded
automatically on the next start.

Phase consistency (2-element ULA only)
--------------------------------------
For 2-antenna ULA: deviation between MUSIC bearing and the bearing
expected from the cross-correlation phase.  Values <= 5 deg indicate
a well-calibrated array with a dominant source.
"""

import sys
import os
import math
import json
import signal
import statistics
from collections import deque

import numpy as np

# -- add workspace to Python path
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from packaging.version import Version as StrictVersion

import ctypes
if sys.platform.startswith('linux'):
    try:
        ctypes.cdll.LoadLibrary('libX11.so').XInitThreads()
    except Exception:
        pass

from PyQt5 import Qt, QtCore
from gnuradio import blocks, filter, gr, qtgui
from gnuradio.fft import window
from gnuradio.filter import firdes
from gnuradio.qtgui import Range, RangeWidget
from gnuradio import eng_notation
from gnuradio.eng_arg import eng_float, intx
import sip
from gnuradio import krakensdr

from doa_display_widgets import (
    PolarSpectrumWidget,
    CompassWidget, DoAMapWidget, SignalQualityWidget,
    BearingHistoryWidget, CalibrationWidget,
)

_CAL_FILE = os.path.join(_WORKSPACE, ".doa_calibration.json")

# -- configuration from environment variables
NUM_CHANNELS = int(os.environ.get("NUM_CHANNELS", "2"))
ARRAY_TYPE   = os.environ.get("ARRAY_TYPE", "ULA").upper()
CENTER_FREQ  = float(os.environ.get("CENTER_FREQ", "868.0"))
GAIN_DB      = float(os.environ.get("GAIN_DB", "40.2"))
ARRAY_DIST   = float(os.environ.get("ARRAY_DIST", "0.17"))


# --- Flowgraph + Widget -------------------------------------------------------

class KrakenDoA(gr.top_block, Qt.QWidget):
    """
    KrakenSDR N-antenna MUSIC DoA with display widgets.
    Supports 2-5 channels, ULA or UCA array.
    """

    def __init__(self):
        gr.top_block.__init__(self, "KrakenSDR DoA", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle(
            f"KrakenSDR DoA {NUM_CHANNELS}-Ant  |  {ARRAY_TYPE}  |  {CENTER_FREQ} MHz")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except Exception:
            pass

        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "kraken_doa_n_ant")
        try:
            self.restoreGeometry(self.settings.value("geometry"))
        except Exception:
            pass

        # dark stylesheet
        self.setStyleSheet("background-color: #0a0c14; color: #b0b8d8;")

        # -- flowgraph variables
        self.num_channels = NUM_CHANNELS
        self.array_type   = ARRAY_TYPE
        self.samp_rate    = 1024000
        self.gain         = GAIN_DB
        self.freq         = CENTER_FREQ
        self.fft_cut      = 512
        self.decimation   = 8
        self.cpi_size     = 131072
        self.array_dist   = ARRAY_DIST
        self.est_range_m  = 100.0

        # calibration
        self._cal_offset  = 0
        self._bear_buffer: deque[float] = deque(maxlen=15)   # ~3 s @ 5 Hz
        self._load_calibration()

        # convenience
        self._use_correlator = (self.num_channels == 2)

        ##################################################
        # Row 0: sliders
        ##################################################
        self._freq_range = Range(80.0, 1700.0, 0.001, self.freq, 200)
        self._freq_win   = RangeWidget(self._freq_range, self.set_freq,
                                       "Frequency (MHz)", "counter_slider",
                                       float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._freq_win, 0, 0, 1, 1)

        self._gain_range = Range(0, 49.6, 0.1, self.gain, 200)
        self._gain_win   = RangeWidget(self._gain_range, self.set_gain,
                                       "Gain (dB)", "counter_slider",
                                       float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._gain_win, 0, 1, 1, 1)

        self._array_dist_range = Range(0.05, 1.00, 0.005, self.array_dist, 200)
        self._array_dist_win   = RangeWidget(self._array_dist_range, self.set_array_dist,
                                             "Antenna spacing (m)", "counter_slider",
                                             float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._array_dist_win, 0, 2, 1, 1)

        self._range_range = Range(10.0, 5000.0, 10.0, self.est_range_m, 200)
        self._range_win   = RangeWidget(self._range_range, self.set_est_range_m,
                                        "Estimated TX range (m)", "counter_slider",
                                        float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._range_win, 0, 3, 1, 2)

        for c in range(5):
            self.top_grid_layout.setColumnStretch(c, 1)

        ##################################################
        # Row 1-2: decimated FFT (col 0-2) + Polar MUSIC spectrum (col 3-4)
        ##################################################
        self.qtgui_fft_ch0 = qtgui.freq_sink_c(
            2048, window.WIN_BLACKMAN_hARRIS,
            self.freq * 1e6, self.samp_rate / self.decimation,
            "CH0 Decimated Spectrum", 1, None)
        self.qtgui_fft_ch0.set_update_time(0.10)
        self.qtgui_fft_ch0.set_y_axis(-80, 10)
        self.qtgui_fft_ch0.set_y_label("Relative gain", "dB")
        self.qtgui_fft_ch0.enable_autoscale(True)
        self.qtgui_fft_ch0.enable_grid(True)
        self.qtgui_fft_ch0.set_fft_average(1.0)
        self.qtgui_fft_ch0.enable_axis_labels(True)
        self.qtgui_fft_ch0.enable_control_panel(False)
        self.qtgui_fft_ch0.set_fft_window_normalized(False)
        self.qtgui_fft_ch0.set_line_label(0, "CH0")
        self.qtgui_fft_ch0.set_line_width(0, 1)
        self.qtgui_fft_ch0.set_line_color(0, "blue")
        self.qtgui_fft_ch0.set_line_alpha(0, 1.0)
        self._qtgui_fft_ch0_win = sip.wrapinstance(
            self.qtgui_fft_ch0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_fft_ch0_win, 1, 0, 2, 3)

        self._polar_widget = PolarSpectrumWidget()
        self._polar_widget.setMinimumSize(200, 200)
        self.top_grid_layout.addWidget(self._polar_widget, 1, 3, 2, 2)

        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)

        ##################################################
        # KrakenSDR source
        ##################################################
        gain_list = [self.gain] * self.num_channels
        self.krakensdr_src = krakensdr.krakensdr_source(
            '127.0.0.1', 5000, 5001,
            self.num_channels, self.freq, gain_list, False)

        ##################################################
        # N FIR decimation filters + stream-to-vector for MUSIC
        ##################################################
        vec_len = self.cpi_size // self.decimation
        self.fir_filters = []
        self.s2v_doa = []
        for ch in range(self.num_channels):
            fir = filter.fir_filter_ccc(self.decimation,
                                        [self.decimation * self.num_channels])
            fir.declare_sample_delay(0)
            self.fir_filters.append(fir)

            s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, vec_len)
            self.s2v_doa.append(s2v)

        ##################################################
        # MUSIC DoA block
        ##################################################
        self.doa_music = krakensdr.doa_music(
            vec_len, self.freq, self.array_dist,
            self.num_channels, self.array_type)

        ##################################################
        # Probe block for Python polling of MUSIC vector
        ##################################################
        self.probe_music = blocks.probe_signal_vf(360)

        ##################################################
        # Correlator + phase probe (2-antenna only)
        ##################################################
        if self._use_correlator:
            self.s2v_corr = []
            for ch in range(2):
                s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, self.cpi_size)
                self.s2v_corr.append(s2v)
            self.corr = krakensdr.krakensdr_correlator(self.cpi_size, self.fft_cut)
            self.probe_phase = blocks.probe_signal_f()

        ##################################################
        # Row 3: Compass + Map + Quality gauge
        ##################################################
        self._compass_widget = CompassWidget(
            num_elements=self.num_channels, array_type=self.array_type)
        self._compass_widget.setMinimumSize(200, 200)
        self.top_grid_layout.addWidget(self._compass_widget, 3, 0, 1, 2)

        self._map_widget = DoAMapWidget(
            num_elements=self.num_channels, array_type=self.array_type)
        self._map_widget.set_range(self.est_range_m)
        self._map_widget.setMinimumSize(240, 200)
        self.top_grid_layout.addWidget(self._map_widget, 3, 2, 1, 2)

        self._quality_widget = SignalQualityWidget()
        self._quality_widget.setMinimumSize(100, 100)
        self.top_grid_layout.addWidget(self._quality_widget, 3, 4, 1, 1)

        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)

        ##################################################
        # Row 4: Bearing history + Calibration
        ##################################################
        self._history_widget = BearingHistoryWidget()
        self._history_widget.setMinimumSize(200, 100)
        self.top_grid_layout.addWidget(self._history_widget, 4, 0, 1, 3)

        self._cal_widget = CalibrationWidget()
        self._cal_widget.setMinimumSize(180, 140)
        self._cal_widget.on_calibrate_requested = self._do_calibrate
        self._cal_widget.on_reset_requested     = self._reset_calibration
        self.top_grid_layout.addWidget(self._cal_widget, 4, 3, 1, 2)

        for r in range(4, 5):
            self.top_grid_layout.setRowStretch(r, 1)

        # restore calibration status
        self._cal_widget.set_calibration_status(
            self._cal_offset if self._cal_offset != 0 else None)

        ##################################################
        # Timer for widget polling (200 ms = 5 Hz)
        ##################################################
        self._doa_timer = QtCore.QTimer()
        self._doa_timer.setInterval(200)
        self._doa_timer.timeout.connect(self._update_widgets)
        self._doa_timer.start()

        ##################################################
        # GNU Radio Connections
        ##################################################
        # source -> FIR -> s2v -> MUSIC
        for ch in range(self.num_channels):
            self.connect((self.krakensdr_src, ch), (self.fir_filters[ch], 0))
            self.connect((self.fir_filters[ch], 0), (self.s2v_doa[ch], 0))
            self.connect((self.s2v_doa[ch], 0), (self.doa_music, ch))

        # CH0 decimated -> FFT display
        self.connect((self.fir_filters[0], 0), (self.qtgui_fft_ch0, 0))

        # MUSIC output -> probe
        self.connect((self.doa_music, 0), (self.probe_music, 0))

        # Correlator connections (2-antenna only)
        if self._use_correlator:
            for ch in range(2):
                self.connect((self.krakensdr_src, ch), (self.s2v_corr[ch], 0))
            self.connect((self.s2v_corr[0], 0), (self.corr, 0))
            self.connect((self.s2v_corr[1], 0), (self.corr, 1))
            self.connect((self.corr, 1), (self.probe_phase, 0))

    # -- widget update callback (called by timer) --------------------------------

    def _update_widgets(self):
        raw = self.probe_music.level()
        if raw is None or len(raw) < 360:
            return
        music_vec = np.array(raw, dtype=np.float32)

        # apply calibration offset (vector rotation)
        if self._cal_offset != 0:
            music_vec = np.roll(music_vec, self._cal_offset)

        bearing_idx = int(np.argmax(music_vec))
        quality_dbfs = float(music_vec[bearing_idx])

        self._bear_buffer.append(float(bearing_idx))

        # update all widgets
        self._polar_widget.set_spectrum(music_vec)
        self._polar_widget.set_bearing(bearing_idx, quality_dbfs)
        self._compass_widget.set_bearing(bearing_idx, quality_dbfs)
        self._map_widget.set_bearing(bearing_idx, quality_dbfs)
        self._quality_widget.set_bearing(bearing_idx, quality_dbfs)
        self._history_widget.set_bearing(bearing_idx, quality_dbfs)

        # phase consistency (2-antenna ULA only)
        consistency_deg = None
        if self._use_correlator:
            consistency_deg = self._phase_consistency(bearing_idx)
        if len(self._bear_buffer) >= 3:
            mean_b = statistics.mean(self._bear_buffer)
            std_b  = statistics.pstdev(self._bear_buffer)
            self._cal_widget.set_stats(mean_b, std_b, consistency_deg)

    def _phase_consistency(self, bearing_deg: float) -> float | None:
        """
        Estimates consistency between MUSIC peak and cross-correlation phase.
        Returns the deviation in degrees (None if unavailable).
        Only valid for 2-element ULA.
        """
        try:
            phase_deg = float(self.probe_phase.level())
        except Exception:
            return None
        lam = 300.0 / self.freq          # wavelength in m
        # expected phase from ULA formula: phase_diff = 2*pi*d*sin(theta)/lambda
        expected_phase = math.degrees(
            2 * math.pi * self.array_dist
            * math.sin(math.radians(bearing_deg)) / lam)
        diff = ((expected_phase - phase_deg + 180) % 360) - 180
        return min(180.0, abs(diff))

    # -- calibration ---------------------------------------------------------------

    def _do_calibrate(self, known_angle_deg: float):
        """Called by CalibrationWidget when the user presses 'Calibrate'."""
        if len(self._bear_buffer) < 3:
            return
        avg = statistics.mean(self._bear_buffer)
        self._cal_offset = int(round(known_angle_deg - avg)) % 360
        self._cal_widget.set_calibration_status(self._cal_offset)
        self._save_calibration()

    def _reset_calibration(self):
        self._cal_offset = 0
        self._cal_widget.set_calibration_status(None)
        try:
            os.remove(_CAL_FILE)
        except FileNotFoundError:
            pass

    def _save_calibration(self):
        try:
            with open(_CAL_FILE, "w") as f:
                json.dump({"cal_offset_deg": self._cal_offset,
                           "freq_mhz": self.freq,
                           "array_dist_m": self.array_dist,
                           "num_channels": self.num_channels,
                           "array_type": self.array_type}, f)
        except OSError:
            pass

    def _load_calibration(self):
        try:
            with open(_CAL_FILE) as f:
                data = json.load(f)
            self._cal_offset = int(data.get("cal_offset_deg", 0))
        except (FileNotFoundError, ValueError, KeyError):
            self._cal_offset = 0

    # -- teardown ------------------------------------------------------------------

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self._doa_timer.stop()
        self.stop()
        self.wait()
        event.accept()

    # -- GNU Radio setters (called by sliders) -------------------------------------

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_fft_ch0.set_frequency_range(
            self.freq * 1e6, self.samp_rate / self.decimation)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.krakensdr_src.set_gain([self.gain] * self.num_channels)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.krakensdr_src.set_freq(self.freq)
        self.doa_music.set_freq(self.freq)
        self.qtgui_fft_ch0.set_frequency_range(
            self.freq * 1e6, self.samp_rate / self.decimation)

    def get_fft_cut(self):
        return self.fft_cut

    def set_fft_cut(self, fft_cut):
        self.fft_cut = fft_cut

    def get_decimation(self):
        return self.decimation

    def set_decimation(self, decimation):
        self.decimation = decimation
        for fir in self.fir_filters:
            fir.set_taps([self.decimation * self.num_channels])
        self.qtgui_fft_ch0.set_frequency_range(
            self.freq * 1e6, self.samp_rate / self.decimation)

    def get_cpi_size(self):
        return self.cpi_size

    def set_cpi_size(self, cpi_size):
        self.cpi_size = cpi_size

    def get_array_dist(self):
        return self.array_dist

    def set_array_dist(self, array_dist):
        self.array_dist = array_dist
        self.doa_music.set_array_dist(self.array_dist)

    def get_est_range_m(self):
        return self.est_range_m

    def set_est_range_m(self, est_range_m):
        self.est_range_m = est_range_m
        self._map_widget.set_range(self.est_range_m)


# --- main ---------------------------------------------------------------------

def main():
    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)

    qapp = Qt.QApplication(sys.argv)
    tb = KrakenDoA()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # dummy timer tick to allow SIGINT to reach Python
    _tick = Qt.QTimer()
    _tick.start(500)
    _tick.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == '__main__':
    main()
