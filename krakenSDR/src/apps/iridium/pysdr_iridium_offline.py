#!/usr/bin/env python3
"""
pysdr_iridium_offline.py – Offline Iridium burst detector for recorded IQ files
================================================================================

Reads a baseband IQ recording (SDR++ WAV, CF32 or U8) and runs the same
burst-detection pipeline as pysdr_iridium_detect.py.  No live hardware or
Heimdall connection required.

Typical workflow
----------------
1. Go outdoors with a laptop running SDR++ (RTL-SDR).
2. Tune to 1 626.270 MHz, recorder: Baseband → WAV → Int16.
3. Record for a few minutes while a satellite is in view.
4. Back at your desk, run:

       python3 pysdr_iridium_offline.py pass_20250601_1626MHz.wav

5. Same 5-panel display as pysdr_iridium_detect.py.  Use the scrub slider
   at the bottom to jump around the recording.  Detected bursts can be
   written to a JSON file with --json-out for later DoA correlation.

Usage
-----
    python3 pysdr_iridium_offline.py pass.wav
    python3 pysdr_iridium_offline.py pass.wav --loop
    python3 pysdr_iridium_offline.py pass.wav --demod --raw-out bursts.txt
    python3 pysdr_iridium_offline.py pass.wav --json-out detections.json
    python3 pysdr_iridium_offline.py pass.cf32 --fs 2048000 --freq 1626270000
    python3 pysdr_iridium_offline.py pass.bin  --fmt u8 --fs 2016000
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time
import argparse
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

import config as C
from hardware.file_iq_source import FileIQSource
from core.burst import (
    BurstDetector, PassTracker,
    IRD_CHANS, MAX_DOP_HZ, TDMA_FRAME_S,
)
from core.burst_pipeline import BurstPipeline
from ui.theme import (
    apply_mpl_style,
    BG, BG2, BG3, BORDER, DIM,
    BLUE, TEAL, AMBER, VIOLET, ROSE, LIME,
    TEXT, MUTED,
)

_FFT_N     = 512
_HIST      = 120
_HIST_D    = 200
_SPEC_HIST = 100
_INTERVAL_MS = max(40, int(TDMA_FRAME_S * 1000))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file",       type=str, nargs="?", default=None, metavar="FILE",
                    help="IQ recording (WAV / CF32 / U8).")
    ap.add_argument("--file",     type=str, default=None, dest="file_opt", metavar="FILE")
    ap.add_argument("--fmt",      type=str, default="auto",
                    choices=["auto", "wav", "cf32", "u8"],
                    help="Force IQ format (default: auto)")
    ap.add_argument("--fs",       type=float, default=2_048_000.0, metavar="HZ",
                    help="Sample rate for CF32/U8 (default: 2048000)")
    ap.add_argument("--freq",     type=float, default=1_626_270_000.0, metavar="HZ",
                    help="Centre frequency Hz (default: 1626270000)")
    ap.add_argument("--snr",      type=float, default=8.0,   metavar="DB")
    ap.add_argument("--papr",     type=float, default=5.0,   metavar="DB")
    ap.add_argument("--power",    type=float, default=-90.0, metavar="DBW")
    ap.add_argument("--frame-size", type=int, default=131_072, metavar="N",
                    help="Samples per analysis frame (default: 131072)")
    ap.add_argument("--loop",     action="store_true", help="Loop file at EOF")
    ap.add_argument("--demod",    action="store_true", help="Enable DQPSK demod")
    ap.add_argument("--raw-out",  type=str, default=None, metavar="FILE",
                    help="Append RAW: lines to FILE")
    ap.add_argument("--json-out", type=str, default=None, metavar="FILE",
                    help="Write burst detections to JSON at close")

    ARGS = ap.parse_args()

    IQ_FILE = ARGS.file or ARGS.file_opt
    if IQ_FILE is None:
        ap.error("IQ recording file is required (positional argument or --file FILE)")

    FREQ_HZ    = ARGS.freq
    BURST_SNR  = ARGS.snr
    BURST_PAPR = ARGS.papr
    BURST_PWR  = ARGS.power
    DEMOD_EN   = ARGS.demod
    LOOP_MODE  = ARGS.loop
    RAW_OUT    = ARGS.raw_out
    JSON_OUT   = ARGS.json_out
    FRAME_SIZE = ARGS.frame_size

    # ── Open file source ──────────────────────────────────────────────────────────
    _file_src = FileIQSource(
        path           = IQ_FILE,
        fmt            = ARGS.fmt,
        sample_rate    = ARGS.fs,
        center_freq_hz = FREQ_HZ,
        frame_size     = FRAME_SIZE,
    )
    print(f"[OFF] Loading {IQ_FILE} ...", end=" ", flush=True)
    _file_src.start()
    _FS = _file_src.sample_rate
    print(f"OK -- {_file_src.total_samples:,} samples @ {_FS/1e6:.3f} MS/s "
          f"({_file_src.duration_s:.1f} s)")

    # ── Build pipeline ────────────────────────────────────────────────────────────
    _pipeline = BurstPipeline(
        input_fs       = int(_FS),
        center_freq_hz = FREQ_HZ,
        fft_n          = _FFT_N,
        burst_snr      = BURST_SNR,
        burst_papr     = BURST_PAPR,
        burst_pwr      = BURST_PWR,
        demod_enabled  = DEMOD_EN,
    )
    _detector = _pipeline.detector
    _tracker  = _pipeline.tracker

    _fft_freqs_kHz = _detector.fft_freqs_kHz
    _spec_freq_kHz = _detector.spec_freq_kHz
    _N_SPEC_COLS   = _detector.n_spec_cols

    _raw_file   = open(RAW_OUT, "a") if RAW_OUT else None
    _burst_log: list[dict] = []

    if _raw_file:
        import atexit
        atexit.register(_raw_file.close)
        print(f"[OFF] RAW: lines -> {RAW_OUT}")

    print(f"[OFF] Burst thresholds: SNR>={BURST_SNR} dB  PAPR>={BURST_PAPR} dB  "
          f"power>={BURST_PWR} dBW")
    print(f"[OFF] frame_size={FRAME_SIZE:,}  input_fs={int(_FS):,} Hz  "
          f"BURST_N={_detector.burst_n}")
    if _FS < 200_000:
        print(f"[OFF] *** WARNING: sample rate {_FS/1e3:.0f} kHz is too LOW for "
              f"reliable Iridium burst detection (Iridium bursts are ~27 kHz wide).")
        print(f"[OFF] ***   Need >= 200 kHz; ideally 1-2 MS/s baseband IQ.")
        print(f"[OFF] ***   In SDR++: Recorder tab -> Baseband (NOT Radio), "
              f"WAV, Int16, sample rate 2.048 MS/s.")
        print(f"[OFF] ***   Noise floor estimate uses inner-band; expect no detections.")
    if LOOP_MODE:
        print("[OFF] Loop mode ON")

    # ── IQ fetch helper ───────────────────────────────────────────────────────────

    def _get_x():
        frame = _file_src.get_frame(timeout=0.0)
        if frame is None:
            if LOOP_MODE:
                _file_src.rewind()
                frame = _file_src.get_frame(timeout=0.0)
                if frame is None:
                    return None
            else:
                return None
        return frame[0].astype(np.complex128)

    # ── Display state ─────────────────────────────────────────────────────────────
    _S = SimpleNamespace(
        fps          = 0.0,
        t_last       = time.time(),
        done         = False,
        last_doppler = 0.0,
        h_snr    = collections.deque([0.0] * _HIST,   maxlen=_HIST),
        h_papr   = collections.deque([0.0] * _HIST,   maxlen=_HIST),
        h_burst  = collections.deque([0]   * _HIST,   maxlen=_HIST),
        h_dop    = collections.deque([0.0] * _HIST_D, maxlen=_HIST_D),
        h_dop_q  = collections.deque([0]   * _HIST_D, maxlen=_HIST_D),
        h_snr_d  = collections.deque([0.0] * _HIST_D, maxlen=_HIST_D),
        spec_data = np.full((_SPEC_HIST, _N_SPEC_COLS), -60.0, dtype=np.float32),
        last_raw      = "",
        decoded_count = 0,
        scrubbing     = False,
    )

    # ── Figure ────────────────────────────────────────────────────────────────────
    apply_mpl_style()
    _ZOOM_LIM_KHZ = float(np.abs(_spec_freq_kHz).max())
    _fig = plt.figure(figsize=(18, 10.5), facecolor=BG)
    _fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.10,
                         hspace=0.60, wspace=0.35)

    _fname_short = os.path.basename(IQ_FILE)
    _fig.suptitle(
        f"Iridium Offline Burst Detector  .  {_fname_short}"
        f"  .  {FREQ_HZ/1e6:.4f} MHz  .  {_FS/1e6:.3f} MS/s"
        f"  .  SNR>={BURST_SNR} dB  PAPR>={BURST_PAPR} dB",
        fontsize=11, fontweight="bold", color=TEXT,
    )

    _gs = gridspec.GridSpec(3, 2, figure=_fig,
                            height_ratios=[1.0, 1.15, 0.80],
                            width_ratios=[1.7, 1.0])

    _x_hist   = np.arange(_HIST)
    _x_hist_d = np.arange(_HIST_D)

    # Panel A: IQ Spectrum
    _ax_spec = _fig.add_subplot(_gs[0, 0])
    _ax_spec.set_xlim(_fft_freqs_kHz[0], _fft_freqs_kHz[-1])
    _ax_spec.set_ylim(-48, 4)
    _ax_spec.grid(True, color=BORDER, alpha=0.4)
    _ax_spec.set_title(f"IQ Spectrum  --  full +/-{_FS/2/1e3:.0f} kHz",
                       color=MUTED, fontsize=8)
    _ax_spec.set_xlabel("df from centre (kHz)", color=MUTED, fontsize=7)
    _ax_spec.set_ylabel("Norm. power (dB)",     color=MUTED, fontsize=7)
    _ax_spec.axvspan(-MAX_DOP_HZ/1e3, MAX_DOP_HZ/1e3, color=AMBER, alpha=0.07)
    _ax_spec.axvline(-MAX_DOP_HZ/1e3, color=AMBER, lw=0.8, alpha=0.55, ls=":")
    _ax_spec.axvline( MAX_DOP_HZ/1e3, color=AMBER, lw=0.8, alpha=0.55, ls=":")
    _ax_spec.text(MAX_DOP_HZ/1e3 + 2, -45,
                  f"+/-{MAX_DOP_HZ/1e3:.0f} kHz\nIridium Doppler",
                  color=AMBER, fontsize=6, va="bottom")
    _line_spec,   = _ax_spec.plot(_fft_freqs_kHz, np.full(_FFT_N, -40.0),
                                   color=BLUE, lw=0.9, zorder=2)
    _line_dop_mk, = _ax_spec.plot([0, 0], [-48, 4],
                                   color=LIME, lw=1.6, alpha=0.8, ls="--", zorder=3)
    _txt_spec_pk  = _ax_spec.text(0.98, 0.89, "Doppler: ---",
                                   transform=_ax_spec.transAxes, ha="right",
                                   fontsize=8, color=DIM)

    # Panel B: Doppler S-curve
    _ax_scurve = _fig.add_subplot(_gs[0, 1])
    _ax_scurve.set_xlim(0, _HIST_D - 1)
    _ax_scurve.set_ylim(-MAX_DOP_HZ/1e3, MAX_DOP_HZ/1e3)
    _ax_scurve.axhline(0, color=BORDER, lw=1.2, ls="--", alpha=0.85)
    _ax_scurve.grid(True, color=BORDER, alpha=0.3)
    _ax_scurve.set_title(
        "Doppler S-curve  (+->approach  0=TCA  -->recede  dot=SNR [dB])",
        color=MUTED, fontsize=8)
    _ax_scurve.set_xlabel("Frames (recent ->)", color=MUTED, fontsize=7)
    _ax_scurve.set_ylabel("kHz",                color=MUTED, fontsize=7)
    _line_dop_sc, = _ax_scurve.plot(_x_hist_d, list(_S.h_dop),
                                     color=MUTED, lw=1.0, alpha=0.7, zorder=1)
    _scat_dop     = _ax_scurve.scatter([], [], c=[], cmap="RdYlGn", s=24,
                                        vmin=0, vmax=22, zorder=3, alpha=0.90)
    _txt_tca    = _ax_scurve.text(0.98, 0.91, "TCA: ---",
                                  transform=_ax_scurve.transAxes, ha="right",
                                  fontsize=7, color=MUTED)
    _txt_pass_n = _ax_scurve.text(0.03, 0.91, "Pass #0",
                                  transform=_ax_scurve.transAxes, ha="left",
                                  fontsize=8, color=TEAL, fontweight="bold")
    _txt_dop_now= _ax_scurve.text(0.03, 0.06, "df --- kHz",
                                  transform=_ax_scurve.transAxes, ha="left",
                                  fontsize=7, color=AMBER)

    # Panel C: Spectrogram waterfall
    _ax_sg = _fig.add_subplot(_gs[1, :])
    _ax_sg.set_facecolor(BG)
    _ax_sg.set_title(
        f"Spectrogram -- zoomed +/-{_ZOOM_LIM_KHZ:.0f} kHz  "
        f"(newest = top  time span ~{_SPEC_HIST*_INTERVAL_MS/1000:.1f} s)",
        color=MUTED, fontsize=8)
    _ax_sg.set_xlabel("df from centre (kHz)", color=MUTED, fontsize=7)
    _ax_sg.set_ylabel("Time",                 color=MUTED, fontsize=7)
    _sg_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ird_sg", [BG, "#0c1a3a", "#1a3472", VIOLET, BLUE, TEAL, LIME, AMBER])
    _im_sg = _ax_sg.imshow(
        _S.spec_data, aspect="auto", origin="upper",
        extent=(_spec_freq_kHz[0], _spec_freq_kHz[-1], _SPEC_HIST, 0),
        vmin=-45, vmax=0, cmap=_sg_cmap, interpolation="nearest")
    _ax_sg.axvline(-MAX_DOP_HZ/1e3, color=AMBER,  lw=0.9, alpha=0.5, ls=":")
    _ax_sg.axvline( MAX_DOP_HZ/1e3, color=AMBER,  lw=0.9, alpha=0.5, ls=":")
    _ax_sg.axvline(0,                color=BORDER, lw=0.8, alpha=0.5, ls="--")
    _t_lbl = [
        "now",
        f"-{_INTERVAL_MS * _SPEC_HIST//4  / 1000:.1f} s",
        f"-{_INTERVAL_MS * _SPEC_HIST//2  / 1000:.1f} s",
        f"-{_INTERVAL_MS * 3*_SPEC_HIST//4/ 1000:.1f} s",
        f"-{_INTERVAL_MS * _SPEC_HIST      / 1000:.1f} s",
    ]
    _ax_sg.set_yticks([0, _SPEC_HIST//4, _SPEC_HIST//2, 3*_SPEC_HIST//4, _SPEC_HIST])
    _ax_sg.set_yticklabels(_t_lbl, color=MUTED, fontsize=6)
    _txt_sg_burst = _ax_sg.text(0.998, 0.96, "",
                                transform=_ax_sg.transAxes, ha="right",
                                fontsize=9, color=LIME, fontweight="bold")

    # Panel D: SNR + PAPR history
    _ax_snr  = _fig.add_subplot(_gs[2, 0])
    _ax_snr.set_title("SNR + PAPR  [dB/frame]  (zero = no burst)",
                      color=MUTED, fontsize=8)
    _ax_snr.set_xlim(0, _HIST - 1)
    _ax_snr.set_ylim(0, 30)
    _ax_snr.grid(True, color=BORDER, alpha=0.3)
    _ax_snr.set_xlabel("Frames", color=MUTED, fontsize=7)
    _ax_snr.axhline(BURST_SNR,  color=VIOLET, lw=0.8, ls=":", alpha=0.75)
    _ax_snr.axhline(BURST_PAPR, color=AMBER,  lw=0.8, ls=":", alpha=0.75)
    _ax_snr.text(2, BURST_SNR  + 0.5, f"SNR thresh {BURST_SNR}",   color=VIOLET, fontsize=6)
    _ax_snr.text(2, BURST_PAPR + 0.5, f"PAPR thresh {BURST_PAPR}", color=AMBER,  fontsize=6)
    _line_snr,  = _ax_snr.plot(_x_hist, list(_S.h_snr),  color=VIOLET, lw=1.4, label="SNR")
    _line_papr, = _ax_snr.plot(_x_hist, list(_S.h_papr), color=AMBER,  lw=1.4, label="PAPR")
    _ax_snr.legend(fontsize=7, facecolor=BG3, edgecolor=BORDER,
                   labelcolor=TEXT, loc="upper right")
    _txt_snr_val = _ax_snr.text(0.03, 0.87, "SNR --- dB  PAPR --- dB",
                                transform=_ax_snr.transAxes,
                                ha="left", fontsize=8, color=MUTED)
    _txt_pwr_val = _ax_snr.text(0.03, 0.06, "pwr --- dBW",
                                transform=_ax_snr.transAxes,
                                ha="left", fontsize=7, color=DIM)

    # Panel E: Burst timeline
    _ax_tl = _fig.add_subplot(_gs[2, 1])
    _ax_tl.set_title("Burst timeline  (amber fill = BURST)", color=MUTED, fontsize=8)
    _ax_tl.set_xlim(0, _HIST - 1)
    _ax_tl.set_ylim(-0.08, 1.28)
    _ax_tl.set_xlabel("Frames", color=MUTED, fontsize=7)
    _ax_tl.set_yticks([0, 1])
    _ax_tl.set_yticklabels(["scan", "BURST"], color=MUTED, fontsize=7)
    _ax_tl.grid(True, axis="x", color=BORDER, alpha=0.3)
    _line_tl, = _ax_tl.step(_x_hist, [0] * _HIST, where="post",
                             color=AMBER, lw=1.4, zorder=2)
    _txt_tl_stats = _ax_tl.text(0.97, 0.88,
                                "Bursts: 0  |  Passes: 0\nBurst rate: 0 %",
                                transform=_ax_tl.transAxes, ha="right",
                                fontsize=7, color=AMBER)

    # Status / FPS
    _txt_status = _fig.text(0.005, 0.961, "O INIT", fontsize=9,
                            fontweight="bold", color=DIM, va="top")
    _txt_fps    = _fig.text(0.880, 0.961, "--- fps", fontsize=8,
                            color=MUTED, va="top")
    _txt_time   = _fig.text(0.005, 0.001, "", fontsize=7, color=MUTED, va="bottom")

    # Progress slider
    _ax_slider = _fig.add_axes((0.05, 0.035, 0.85, 0.020), facecolor=BG3)
    _slider = Slider(
        ax      = _ax_slider,
        label   = "",
        valmin  = 0.0,
        valmax  = max(_file_src.duration_s, 1.0),
        valinit = 0.0,
        color   = AMBER,
    )
    _slider.label.set_color(MUTED)
    _slider.valtext.set_visible(False)

    def _on_slider(val: float) -> None:
        _S.scrubbing = True
        _file_src.seek_time(val)
        for q in (_S.h_snr, _S.h_papr, _S.h_burst,
                  _S.h_dop, _S.h_dop_q, _S.h_snr_d):
            q.clear()
            for _ in range(q.maxlen):
                q.append(0 if q is _S.h_burst or q is _S.h_dop_q else 0.0)
        _S.spec_data[:] = -60.0
        _S.done = False
        _S.scrubbing = False

    _slider.on_changed(_on_slider)

    # ── Update helpers ────────────────────────────────────────────────────────────

    def _update_scurve() -> None:
        dop_arr = np.array(list(_S.h_dop))
        q_arr   = np.array(list(_S.h_dop_q))
        snr_arr = np.array(list(_S.h_snr_d))
        _line_dop_sc.set_ydata(dop_arr)
        b_idx = np.where(q_arr == 1)[0]
        if len(b_idx):
            _scat_dop.set_offsets(np.column_stack([b_idx, dop_arr[b_idx]]))
            _scat_dop.set_array(np.clip(snr_arr[b_idx], 0.0, 22.0))
        else:
            _scat_dop.set_offsets(np.empty((0, 2)))
        recent = dop_arr[max(0, len(dop_arr) - 60):]
        if len(recent) >= 4:
            signs = np.sign(recent)
            cross = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
            if len(cross):
                ago = len(recent) - 1 - cross[-1]
                _txt_tca.set_text(f"TCA: {ago} fr ago")
                _txt_tca.set_color(LIME if ago < 5 else AMBER)
            elif recent[-1] > 300:
                _txt_tca.set_text("TCA: approaching")
                _txt_tca.set_color(BLUE)
            elif recent[-1] < -300:
                _txt_tca.set_text("TCA: receded")
                _txt_tca.set_color(DIM)
            else:
                _txt_tca.set_text("TCA: ---")
                _txt_tca.set_color(DIM)
        sign = "+" if _S.last_doppler >= 0 else ""
        _txt_dop_now.set_text(f"df {sign}{_S.last_doppler/1e3:.2f} kHz")
        _txt_pass_n.set_text(f"Pass #{_tracker.pass_count}")

    def _update_timeline() -> None:
        bdata = list(_S.h_burst)
        _line_tl.set_ydata(bdata)
        for coll in _ax_tl.collections[:]:
            coll.remove()
        if any(bdata):
            _ax_tl.fill_between(_x_hist, 0, bdata, step="post",
                                color=AMBER, alpha=0.28, linewidth=0, zorder=1)
        br = 100 * sum(bdata) / max(len(bdata), 1)
        _txt_tl_stats.set_text(
            f"Bursts: {_tracker.burst_count}  |  Passes: {_tracker.pass_count}\n"
            f"Burst rate: {br:.0f} %")

    # ── Main animation callback ───────────────────────────────────────────────────

    def _update(_frame) -> None:
        now = time.time()
        _S.fps    = 0.9 * _S.fps + 0.1 / max(now - _S.t_last, 1e-6)
        _S.t_last = now
        _txt_fps.set_text(f"{_S.fps:.1f} fps")

        if not _S.scrubbing and plt.fignum_exists(_fig.number):
            _slider.eventson = False
            _slider.set_val(_file_src.elapsed_s)
            _slider.eventson = True

        _txt_time.set_text(
            f"{_file_src.elapsed_s:.1f} s  /  {_file_src.duration_s:.1f} s"
            + ("  [LOOP]" if LOOP_MODE else ""))

        if _S.done:
            _txt_status.set_text("[ End of file ]")
            _txt_status.set_color(DIM)
            return

        x = _get_x()
        if x is None:
            _S.done = True
            _txt_status.set_text(
                f"[ End of file ]  {_tracker.burst_count} bursts  "
                f"{_tracker.pass_count} passes")
            _txt_status.set_color(MUTED)
            return

        pr       = _pipeline.process(x, timestamp=_file_src.elapsed_s)
        r        = pr.burst
        is_burst = r.is_burst
        dop_hz   = r.doppler_hz
        snr_db   = r.burst_snr_db
        papr_db  = r.burst_papr_db
        abs_pwr  = r.abs_pwr_db
        sign     = "+" if dop_hz >= 0 else ""

        if pr.raw_line is not None:
            _S.last_raw      = pr.raw_line
            _S.decoded_count += 1
            print(pr.raw_line, flush=True)
            if _raw_file:
                _raw_file.write(pr.raw_line + "\n")
                _raw_file.flush()

        if is_burst:
            entry: dict = {
                "timestamp_s": round(_file_src.elapsed_s, 4),
                "doppler_hz":  round(float(dop_hz),  2),
                "snr_db":      round(float(snr_db),  2),
                "papr_db":     round(float(papr_db), 2),
                "abs_pwr_db":  round(float(abs_pwr), 2),
            }
            if pr.raw_line is not None:
                entry["raw_line"] = pr.raw_line
            _burst_log.append(entry)

        _S.last_doppler = dop_hz
        _S.h_dop.append(dop_hz / 1e3)
        _S.h_dop_q.append(1    if is_burst else 0)
        _S.h_snr_d.append(snr_db  if is_burst else 0.0)
        _S.h_burst.append(1    if is_burst else 0)
        _S.h_snr.append(  snr_db  if is_burst else 0.0)
        _S.h_papr.append( papr_db if is_burst else 0.0)

        _S.spec_data[1:, :] = _S.spec_data[:-1, :]
        _S.spec_data[0,  :] = r.zoom_db
        p5  = float(np.percentile(_S.spec_data,  5))
        p99 = float(np.percentile(_S.spec_data, 99))
        _im_sg.set_data(_S.spec_data)
        _im_sg.set_clim(vmin=max(p5, -55.0), vmax=min(p99 + 3.0, 2.0))

        if is_burst:
            _txt_sg_burst.set_text(
                f"BURST  {sign}{dop_hz/1e3:.1f} kHz  "
                f"SNR {snr_db:.1f} dB  PAPR {papr_db:.1f} dB")
        else:
            _txt_sg_burst.set_text("")

        _line_spec.set_ydata(r.spec_db)
        _line_dop_mk.set_xdata([dop_hz/1e3, dop_hz/1e3])
        _line_dop_mk.set_color(LIME if is_burst else ROSE)
        _txt_spec_pk.set_text(
            f"{'BURST  ' if is_burst else ''}{sign}{dop_hz/1e3:.2f} kHz  "
            f"SNR {snr_db:.1f} dB")
        _txt_spec_pk.set_color(LIME if is_burst else DIM)

        _update_scurve()

        _line_snr.set_ydata(list(_S.h_snr))
        _line_papr.set_ydata(list(_S.h_papr))
        _ax_snr.set_ylim(0, max(30.0, max(_S.h_snr) + 3.0, max(_S.h_papr) + 3.0))
        col = (LIME  if (is_burst and snr_db >= BURST_SNR * 1.5)
               else AMBER if is_burst else DIM)
        _txt_snr_val.set_text(
            f"SNR {snr_db:.1f} dB  PAPR {papr_db:.1f} dB"
            if is_burst else "SNR --- dB  PAPR --- dB  (scanning)")
        _txt_snr_val.set_color(col)
        _txt_pwr_val.set_text(f"pwr {abs_pwr:.1f} dBW  (floor >={BURST_PWR} dBW)")

        _update_timeline()

        if is_burst:
            dop_lbl = ("approach" if dop_hz > 200 else "receding" if dop_hz < -200 else "TCA")
            demod_sfx = ""
            if DEMOD_EN:
                if pr.raw_line is not None:
                    raw_s = pr.raw_line.split()[-1][:40] if pr.raw_line else ""
                    demod_sfx = f"  .  decoded ({_S.decoded_count})  {raw_s}"
                else:
                    demod_sfx = "  .  demod..."
            _txt_status.set_text(
                f"BURST #{_tracker.burst_count}"
                f"  .  {sign}{dop_hz/1e3:.2f} kHz  ({dop_lbl})"
                f"  .  SNR {snr_db:.1f} dB  PAPR {papr_db:.1f} dB"
                f"  .  pwr {abs_pwr:.1f} dBW"
                f"  .  Pass #{_tracker.pass_count}"
                + demod_sfx)
            _txt_status.set_color(LIME if pr.raw_line is None else TEAL)
        else:
            elapsed_b = _tracker.time_since_last_burst
            if elapsed_b > 0:
                _txt_status.set_text(
                    f"scanning  .  {sign}{dop_hz/1e3:.1f} kHz"
                    f"  .  last burst {elapsed_b:.0f} s ago"
                    f"  .  {_tracker.burst_count} total"
                    + (f"  .  decoded {_S.decoded_count}" if DEMOD_EN else ""))
            else:
                _txt_status.set_text(
                    f"scanning for Iridium TDMA burst ...  "
                    f"df {sign}{dop_hz/1e3:.1f} kHz  SNR {snr_db:.1f} dB")
            _txt_status.set_color(DIM)

    # ── Close handler ─────────────────────────────────────────────────────────────
    def _on_close(_evt) -> None:
        _S.done = True                      # stop _update() callbacks immediately
        try:
            _ani.event_source.stop()        # cancel the animation timer
        except Exception:
            pass
        _file_src.stop()
        if _raw_file:
            _raw_file.close()
        print(f"\n[OFF] Closed.  Bursts: {_tracker.burst_count}"
              f"  Passes: {_tracker.pass_count}"
              + (f"  Decoded: {_S.decoded_count}" if DEMOD_EN else ""))
        if JSON_OUT and _burst_log:
            with open(JSON_OUT, "w") as fh:
                json.dump(_burst_log, fh, indent=2)
            print(f"[OFF] Burst log -> {JSON_OUT}  ({len(_burst_log)} entries)")
        elif JSON_OUT:
            print(f"[OFF] No bursts detected -- {JSON_OUT} not written")

    _fig.canvas.mpl_connect("close_event", _on_close)

    # ── Launch ────────────────────────────────────────────────────────────────────
    _ani = animation.FuncAnimation(  # type: ignore[arg-type]
        _fig, _update, interval=_INTERVAL_MS, blit=False, cache_frame_data=False)

    print(f"[OFF] Animation interval = {_INTERVAL_MS} ms  "
          f"({1000 // _INTERVAL_MS} fps target)")
    print("[OFF] Use the slider at the bottom to scrub through the recording.")
    print("[OFF] Close the plot window to stop.")

    plt.show()


if __name__ == "__main__":
    main()
