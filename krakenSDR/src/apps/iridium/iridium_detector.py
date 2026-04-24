#!/usr/bin/env python3
"""
iridium_detector.py – KrakenSDR single-antenna Iridium burst detector
==========================================================================

Connects to Heimdall and detects Iridium L-band TDMA bursts from a single
antenna channel.  No DoA is performed (≥2 antennas needed for bearings).
Goal: verify the hardware chain can see Iridium bursts before deploying the
full multi-antenna DoA pipeline.

Algorithm (identical to pysdr_doa_iridium.grc):
  1. BURST_N=4096 FFT + Hann window  → freq resolution ≈ 250 Hz @ 1.024 MSPS
  2. Non-coherent |FFT(x)|² in signal band  |f| ≤ 40 kHz (max Doppler at LEO)
  3. Peak / out-of-band noise  → burst_snr_db
  4. In-band peak / in-band mean → burst_papr_db  (DQPSK burst: narrow PAPR spike)
  5. Absolute frame power → squelch gate
  6. Pass detector: |Δf_Doppler| > 12 kHz OR gap > 6 s → new satellite

Layout (3 rows):

  ┌──────────────────────────────────┬───────────────────────────┐
  │  A: IQ spectrum ─ full BW        │  B: Doppler S-curve        │
  │     amber band = Iridium window  │     scatter: SNR-coloured   │
  │     rose dash  = Doppler marker  │     TCA auto-detected       │
  ├──────────────────────────────────┴───────────────────────────┤
  │  C: Spectrogram waterfall — zoomed to ±64 kHz (newest = top) │
  │     time flows downward · bursts glow · coloured by power     │
  ├──────────────────────────────────┬───────────────────────────┤
  │  D: SNR + PAPR per burst         │  E: Burst binary timeline  │
  │     threshold lines drawn        │     + pass / burst counters │
  └──────────────────────────────────┴───────────────────────────┘

Signal processing  →  core.burst.BurstDetector / PassTracker
Startup dialog     →  ui.dialogs.run_iridium_dialog
Colour theme       →  ui.theme
Hardware I/O       →  hardware.KrakenIQSource
System config      →  config

Usage:
    python3 apps/iridium/iridium_detector.py
    python3 apps/iridium/iridium_detector.py --no-dialog
    python3 apps/iridium/iridium_detector.py --freq 1626270000
    python3 apps/iridium/iridium_detector.py --snr 5 --papr 3.5
    python3 apps/iridium/iridium_detector.py --channel 1

References:
    ITU-R M.1031   – Iridium radio interface
    ETSI EN 300 461 – Iridium TDMA frame structure
    Iridium SIS-ICD – simplex channel list
"""

from __future__ import annotations

import collections
import os
import sys
import time
import argparse
from types import SimpleNamespace

# ── Resolve script directory and add to import path ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# Always re-insert _HERE at 0: guarantees app-local config.py priority.
sys.path.insert(0, _HERE)

import numpy as np

import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import config as C
from hardware.kraken_iq_source import KrakenIQSource
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
from ui.dialogs import run_iridium_dialog, IridiumConfig

# ══════════════════════════════════════════════════════════════════════════════
# DSP PARAMETERS  (fixed, independent of config)
# ══════════════════════════════════════════════════════════════════════════════
_FFT_N     = 512    # spectrum panel: bins across full ±FS/2
_HIST      = 120    # rolling history depth for panels D and E
_HIST_D    = 200    # Doppler S-curve history depth
_SPEC_HIST = 100    # spectrogram rows (newest = row 0)
_WARMUP    = 3      # Heimdall settle-frames before burst search

_FS = float(C.SAMPLE_RATE_HZ)

# Animation interval aligned to one Iridium TDMA frame
_INTERVAL_MS = max(40, int(TDMA_FRAME_S * 1000))   # ≈ 90 ms

def main() -> None:
    # ══════════════════════════════════════════════════════════════════════════════
    # ARGUMENT PARSER
    # ══════════════════════════════════════════════════════════════════════════════
    _ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ap.add_argument("--freq",      type=float, default=None,   metavar="HZ",
                     help="Centre frequency [Hz]  (default: 1626270000)")
    _ap.add_argument("--gain",      type=float, default=None,   metavar="DB",
                     help="IF gain [dB]  (default: from config.py)")
    _ap.add_argument("--snr",       type=float, default=8.0,    metavar="DB",
                     help="Burst in-band SNR threshold [dB]  (default: 8)")
    _ap.add_argument("--papr",      type=float, default=5.0,    metavar="DB",
                     help="Burst in-band PAPR threshold [dB]  (default: 5)")
    _ap.add_argument("--power",     type=float, default=-90.0,  metavar="DBW",
                     help="Absolute power floor [dBW]  (default: -90)")
    _ap.add_argument("--channel",   type=int,   default=0,      metavar="N",
                     help="Heimdall IQ channel index  (default: 0)")
    _ap.add_argument("--no-dialog", action="store_true",
                     help="Skip startup dialog, use CLI args / config.py defaults")
    _ap.add_argument("--demod", action="store_true",
                     help="Enable DQPSK demodulation and display decoded frame type")
    _ap.add_argument("--raw-out", type=str, default=None, metavar="FILE",
                     help="Append RAW: lines to FILE (compatible with iridium-parser.py)")
    _ap.add_argument("--profile", type=str, default=None, metavar="NAME",
                     help="Named config profile (e.g. iridium_1626). Overrides LARK_PROFILE env var.")
    _ARGS = _ap.parse_args()

    # ── Profile (must run before any C.* read) ────────────────────────────────
    if _ARGS.profile:
        import sys as _sys_mod
        from profiles import apply_profile
        apply_profile(_ARGS.profile, _sys_mod.modules["config"])

    # ══════════════════════════════════════════════════════════════════════════════
    # RESOLVE CONFIGURATION  (dialog → CLI args → config.py defaults)
    # ══════════════════════════════════════════════════════════════════════════════
    _def_freq = _ARGS.freq or 1626270000.0
    _def_gain = _ARGS.gain or float(C.GAIN_DB)
    _def_snr  = _ARGS.snr
    _def_papr = _ARGS.papr
    _def_pwr  = _ARGS.power

    if _ARGS.no_dialog:
        _cfg = IridiumConfig(
            freq_hz    = _def_freq,
            gain_db    = _def_gain,
            burst_snr  = _def_snr,
            burst_papr = _def_papr,
            burst_pwr  = _def_pwr,
        )
    else:
        _cfg = run_iridium_dialog(
            _def_freq, _def_gain, _def_snr, _def_papr, _def_pwr,
            iridium_chans=IRD_CHANS,
        )
        if _cfg is None:
            raise SystemExit(0)

    FREQ_HZ    = _cfg.freq_hz
    GAIN_DB    = _cfg.gain_db
    BURST_SNR  = _cfg.burst_snr
    BURST_PAPR = _cfg.burst_papr
    BURST_PWR  = _cfg.burst_pwr
    CH_IDX     = _ARGS.channel
    DEMOD_EN   = _ARGS.demod
    RAW_OUT    = _ARGS.raw_out

    # ══════════════════════════════════════════════════════════════════════════════
    # BUILD DETECTOR / PIPELINE AND TRACKER
    # ══════════════════════════════════════════════════════════════════════════════
    _pipeline = BurstPipeline(
        input_fs       = int(_FS),
        center_freq_hz = FREQ_HZ,
        fft_n          = _FFT_N,
        burst_snr      = BURST_SNR,
        burst_papr     = BURST_PAPR,
        burst_pwr      = BURST_PWR,
        demod_enabled  = DEMOD_EN,
    )
    _detector = _pipeline.detector   # reuse the same BurstDetector instance
    _tracker  = _pipeline.tracker    # reuse the same PassTracker

    # Optional RAW: output file
    _raw_file = open(RAW_OUT, "a") if RAW_OUT else None
    if _raw_file:
        import atexit
        atexit.register(_raw_file.close)
        print(f"[IRD] RAW: lines → {RAW_OUT}")

    # Convenience aliases for figure layout
    _fft_freqs_kHz = _detector.fft_freqs_kHz
    _spec_freq_kHz = _detector.spec_freq_kHz
    _N_SPEC_COLS   = _detector.n_spec_cols

    # ══════════════════════════════════════════════════════════════════════════════
    # HEIMDALL CONNECTION
    # ══════════════════════════════════════════════════════════════════════════════
    _kraken = KrakenIQSource(
        host=C.HEIMDALL_HOST, port=C.HEIMDALL_PORT, ctrl_port=C.HEIMDALL_CTRL,
        num_channels=C.N_ANTENNAS, freq_hz=FREQ_HZ, gain_db=GAIN_DB,
        verbose=C.VERBOSE_FRAMES,
    )
    _kraken.start()
    print(f"[IRD] Connecting → {FREQ_HZ/1e6:.4f} MHz  gain={GAIN_DB} dB  ch={CH_IDX}  ...",
          end=" ", flush=True)
    time.sleep(2.0)
    print("OK" if _kraken.is_connected else "not reachable — will keep retrying")
    print(f"[IRD] Burst thresholds:  SNR≥{BURST_SNR} dB  PAPR≥{BURST_PAPR} dB  power≥{BURST_PWR} dBW")
    if DEMOD_EN:
        print(f"[IRD] Demod: ENABLED — DQPSK decode active (RAW: lines emitted)")
    print(f"[IRD] DSP:  BURST_N={_detector.burst_n}  freq_res≈{_FS/_detector.burst_n:.0f} Hz  "
          f"±MAX_DOP={MAX_DOP_HZ/1e3:.0f} kHz  interval={_INTERVAL_MS} ms")

    # ══════════════════════════════════════════════════════════════════════════════
    # IQ DATA FETCH
    # ══════════════════════════════════════════════════════════════════════════════

    def _get_x() -> np.ndarray | None:
        """
        Fetch one IQ frame from Heimdall and return channel CH_IDX as 1-D complex128.
        Returns None while Heimdall is not reachable.
        """
        frame = _kraken.get_frame(timeout=0.05)
        if frame is None:
            return None
        X = frame.astype(np.complex128)
        if C.HW_NUM_SAMPLES > 0 and X.shape[1] > C.HW_NUM_SAMPLES:
            X = X[:, :C.HW_NUM_SAMPLES]
        return X[min(CH_IDX, X.shape[0] - 1), :]

    # ══════════════════════════════════════════════════════════════════════════════
    # DISPLAY STATE
    # ══════════════════════════════════════════════════════════════════════════════
    _S = SimpleNamespace(
        # Timing / warmup
        fps          = 0.0,
        t_last       = time.time(),
        warming      = True,
        warmup_count = 0,
        # Last Doppler value (for panel B overlay label)
        last_doppler = 0.0,
        # Rolling histories
        h_snr    = collections.deque([0.0] * _HIST,   maxlen=_HIST),
        h_papr   = collections.deque([0.0] * _HIST,   maxlen=_HIST),
        h_burst  = collections.deque([0]   * _HIST,   maxlen=_HIST),
        h_dop    = collections.deque([0.0] * _HIST_D, maxlen=_HIST_D),
        h_dop_q  = collections.deque([0]   * _HIST_D, maxlen=_HIST_D),
        h_snr_d  = collections.deque([0.0] * _HIST_D, maxlen=_HIST_D),
        # Spectrogram waterfall (row 0 = newest, fftshift cols)
        spec_data = np.full((_SPEC_HIST, _N_SPEC_COLS), -60.0, dtype=np.float32),
        # Demodulation state
        last_raw      = "",           # last RAW: line decoded
        decoded_count = 0,            # total successfully decoded bursts
    )

    # ══════════════════════════════════════════════════════════════════════════════
    # FIGURE LAYOUT  (3 rows × 2 cols)
    # ══════════════════════════════════════════════════════════════════════════════
    apply_mpl_style()

    _ZOOM_LIM_KHZ = float(np.abs(_spec_freq_kHz).max())

    _fig = plt.figure(figsize=(18, 10), facecolor=BG)
    _fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.06,
                         hspace=0.60, wspace=0.35)

    _fig.suptitle(
        f"KrakenSDR  ·  Iridium L-Band Burst Detector  ·  {FREQ_HZ/1e6:.4f} MHz"
        f"  ·  ch-{CH_IDX}  ·  SNR≥{BURST_SNR} dB  PAPR≥{BURST_PAPR} dB",
        fontsize=12, fontweight="bold", color=TEXT,
    )

    _gs = gridspec.GridSpec(3, 2, figure=_fig,
                            height_ratios=[1.0, 1.15, 0.80],
                            width_ratios=[1.7, 1.0])

    _x_hist   = np.arange(_HIST)
    _x_hist_d = np.arange(_HIST_D)

    # ── A: IQ Spectrum ────────────────────────────────────────────────────────────
    _ax_spec = _fig.add_subplot(_gs[0, 0])
    _ax_spec.set_xlim(_fft_freqs_kHz[0], _fft_freqs_kHz[-1])
    _ax_spec.set_ylim(-48, 4)
    _ax_spec.grid(True, color=BORDER, alpha=0.4)
    _ax_spec.set_title(f"IQ Spectrum  —  full ±{_FS/2/1e3:.0f} kHz view  (ch-{CH_IDX})",
                       color=MUTED, fontsize=8)
    _ax_spec.set_xlabel("\u0394f from centre (kHz)", color=MUTED, fontsize=7)
    _ax_spec.set_ylabel("Norm. power (dB)", color=MUTED, fontsize=7)

    _ax_spec.axvspan(-MAX_DOP_HZ/1e3, MAX_DOP_HZ/1e3, color=AMBER, alpha=0.07)
    _ax_spec.axvline(-MAX_DOP_HZ/1e3, color=AMBER, lw=0.8, alpha=0.55, ls=":")
    _ax_spec.axvline( MAX_DOP_HZ/1e3, color=AMBER, lw=0.8, alpha=0.55, ls=":")
    _ax_spec.text(MAX_DOP_HZ/1e3 + 2, -45,
                  f"±{MAX_DOP_HZ/1e3:.0f} kHz\nIridium Doppler",
                  color=AMBER, fontsize=6, va="bottom")

    _line_spec,   = _ax_spec.plot(_fft_freqs_kHz, np.full(_FFT_N, -40.0),
                                   color=BLUE, lw=0.9, zorder=2)
    _line_dop_mk, = _ax_spec.plot([0, 0], [-48, 4],
                                   color=LIME, lw=1.6, alpha=0.8, ls="--", zorder=3)
    _txt_spec_pk  = _ax_spec.text(0.98, 0.89, "Doppler: ---",
                                   transform=_ax_spec.transAxes, ha="right",
                                   fontsize=8, color=DIM)

    # ── B: Doppler S-curve ────────────────────────────────────────────────────────
    _ax_scurve = _fig.add_subplot(_gs[0, 1])
    _ax_scurve.set_xlim(0, _HIST_D - 1)
    _ax_scurve.set_ylim(-MAX_DOP_HZ/1e3, MAX_DOP_HZ/1e3)
    _ax_scurve.axhline(0, color=BORDER, lw=1.2, ls="--", alpha=0.85)
    _ax_scurve.grid(True, color=BORDER, alpha=0.3)
    _ax_scurve.set_title(
        "Doppler S-curve  (+\u2192 approach · 0=TCA · \u2212\u2192 recede · dot colour = SNR [dB])",
        color=MUTED, fontsize=8)
    _ax_scurve.set_xlabel("Frames (recent \u2192)", color=MUTED, fontsize=7)
    _ax_scurve.set_ylabel("kHz", color=MUTED, fontsize=7)

    _line_dop_sc, = _ax_scurve.plot(_x_hist_d, list(_S.h_dop),
                                     color=MUTED, lw=1.0, alpha=0.7, zorder=1)
    _scat_dop     = _ax_scurve.scatter([], [], c=[], cmap="RdYlGn", s=24,
                                        vmin=0, vmax=22, zorder=3, alpha=0.90)
    _txt_tca      = _ax_scurve.text(0.98, 0.91, "TCA: ---",
                                    transform=_ax_scurve.transAxes, ha="right",
                                    fontsize=7, color=MUTED)
    _txt_pass_n   = _ax_scurve.text(0.03, 0.91, "Pass #0",
                                    transform=_ax_scurve.transAxes, ha="left",
                                    fontsize=8, color=TEAL, fontweight="bold")
    _txt_dop_now  = _ax_scurve.text(0.03, 0.06, "\u0394f --- kHz",
                                    transform=_ax_scurve.transAxes, ha="left",
                                    fontsize=7, color=AMBER)

    # ── C: Spectrogram waterfall — full width ─────────────────────────────────────
    _ax_sg = _fig.add_subplot(_gs[1, :])
    _ax_sg.set_facecolor(BG)
    _ax_sg.set_title(
        f"Spectrogram — zoomed \u00b1{_ZOOM_LIM_KHZ:.0f} kHz  "
        f"(newest frame = top · colour = normalised power · "
        f"time span \u2248 {_SPEC_HIST * _INTERVAL_MS / 1000:.1f} s)",
        color=MUTED, fontsize=8)
    _ax_sg.set_xlabel("\u0394f from centre (kHz)", color=MUTED, fontsize=7)
    _ax_sg.set_ylabel("Time", color=MUTED, fontsize=7)

    _sg_cmap = mcolors.LinearSegmentedColormap.from_list(
        "ird_sg", [BG, "#0c1a3a", "#1a3472", VIOLET, BLUE, TEAL, LIME, AMBER])

    _im_sg = _ax_sg.imshow(
        _S.spec_data, aspect="auto", origin="upper",
        extent=(_spec_freq_kHz[0], _spec_freq_kHz[-1], _SPEC_HIST, 0),
        vmin=-45, vmax=0, cmap=_sg_cmap, interpolation="nearest")

    _ax_sg.axvline(-MAX_DOP_HZ/1e3, color=AMBER,  lw=0.9, alpha=0.5, ls=":")
    _ax_sg.axvline( MAX_DOP_HZ/1e3, color=AMBER,  lw=0.9, alpha=0.5, ls=":")
    _ax_sg.axvline(0,                color=BORDER, lw=0.8, alpha=0.5, ls="--")

    _t_labels = [
        "now",
        f"\u2212{_INTERVAL_MS *   _SPEC_HIST // 4  / 1000:.1f} s",
        f"\u2212{_INTERVAL_MS *   _SPEC_HIST // 2  / 1000:.1f} s",
        f"\u2212{_INTERVAL_MS * 3*_SPEC_HIST // 4  / 1000:.1f} s",
        f"\u2212{_INTERVAL_MS *   _SPEC_HIST        / 1000:.1f} s",
    ]
    _ax_sg.set_yticks([0, _SPEC_HIST//4, _SPEC_HIST//2, 3*_SPEC_HIST//4, _SPEC_HIST])
    _ax_sg.set_yticklabels(_t_labels, color=MUTED, fontsize=6)

    _txt_sg_burst = _ax_sg.text(0.998, 0.96, "",
                                 transform=_ax_sg.transAxes, ha="right",
                                 fontsize=9, color=LIME, fontweight="bold")

    # ── D: SNR + PAPR history ─────────────────────────────────────────────────────
    _ax_snr = _fig.add_subplot(_gs[2, 0])
    _ax_snr.set_title("SNR + PAPR  [dB/frame]  (zero = no burst)", color=MUTED, fontsize=8)
    _ax_snr.set_xlim(0, _HIST - 1)
    _ax_snr.set_ylim(0, 30)
    _ax_snr.grid(True, color=BORDER, alpha=0.3)
    _ax_snr.set_xlabel("Frames", color=MUTED, fontsize=7)

    _ax_snr.axhline(BURST_SNR,  color=VIOLET, lw=0.8, ls=":", alpha=0.75)
    _ax_snr.axhline(BURST_PAPR, color=AMBER,  lw=0.8, ls=":", alpha=0.75)
    _ax_snr.text(2, BURST_SNR  + 0.5, f"SNR thresh {BURST_SNR}",  color=VIOLET, fontsize=6)
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

    # ── E: Burst binary timeline ──────────────────────────────────────────────────
    _ax_tl = _fig.add_subplot(_gs[2, 1])
    _ax_tl.set_title("Burst timeline  (amber fill = BURST detected)", color=MUTED, fontsize=8)
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

    # ── Status / FPS bar ──────────────────────────────────────────────────────────
    _txt_status = _fig.text(0.005, 0.961, "\u25cb INIT", fontsize=9,
                            fontweight="bold", color=DIM, va="top")
    _txt_fps    = _fig.text(0.880, 0.961, "--- fps", fontsize=8,
                            color=MUTED, va="top")

    # ══════════════════════════════════════════════════════════════════════════════
    # UPDATE HELPERS
    # ══════════════════════════════════════════════════════════════════════════════

    def _update_scurve() -> None:
        """Refresh Doppler S-curve panel (B) with TCA detection."""
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
                _txt_tca.set_text("TCA: approaching \u2197")
                _txt_tca.set_color(BLUE)
            elif recent[-1] < -300:
                _txt_tca.set_text("TCA: \u2198 receded")
                _txt_tca.set_color(DIM)
            else:
                _txt_tca.set_text("TCA: ---")
                _txt_tca.set_color(DIM)

        sign = "+" if _S.last_doppler >= 0 else ""
        _txt_dop_now.set_text(f"\u0394f {sign}{_S.last_doppler/1e3:.2f} kHz")
        _txt_pass_n.set_text(f"Pass #{_tracker.pass_count}")


    def _update_timeline() -> None:
        """Redraw burst timeline (panel E) using step + fill_between."""
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


    # ══════════════════════════════════════════════════════════════════════════════
    # MAIN ANIMATION CALLBACK
    # ══════════════════════════════════════════════════════════════════════════════

    def _update(_frame) -> None:
        now = time.time()
        _S.fps    = 0.9 * _S.fps + 0.1 / max(now - _S.t_last, 1e-6)
        _S.t_last = now
        _txt_fps.set_text(f"{_S.fps:.1f} fps")

        x = _get_x()
        if x is None:
            _txt_status.set_text("\u25cb waiting for Heimdall  \u2026")
            _txt_status.set_color(ROSE)
            return

        # Warmup: show spectrum while Heimdall settles
        if _S.warming:
            _S.warmup_count += 1
            r = _detector.process(x)
            _line_spec.set_ydata(r.spec_db)
            if _S.warmup_count >= _WARMUP:
                _S.warming = False
            _txt_status.set_text(
                f"WARMUP {_S.warmup_count}/{_WARMUP}  —  settling Heimdall \u2026")
            _txt_status.set_color(AMBER)
            return

        # ── Detect (+ optional demod) ─────────────────────────────────────────────
        pr       = _pipeline.process(x, timestamp=now)
        r        = pr.burst
        is_burst = r.is_burst
        dop_hz   = r.doppler_hz
        snr_db   = r.burst_snr_db
        papr_db  = r.burst_papr_db
        abs_pwr  = r.abs_pwr_db
        sign     = "+" if dop_hz >= 0 else ""

        # Process decoded RAW: line (if demod is enabled and succeeded)
        if pr.raw_line is not None:
            _S.last_raw      = pr.raw_line
            _S.decoded_count += 1
            print(pr.raw_line, flush=True)   # stdout → pipe to iridium-parser.py
            if _raw_file:
                _raw_file.write(pr.raw_line + "\n")
                _raw_file.flush()

        _S.last_doppler = dop_hz

        # ── Rolling histories ─────────────────────────────────────────────────────
        _S.h_dop.append(dop_hz / 1e3)
        _S.h_dop_q.append(1    if is_burst else 0)
        _S.h_snr_d.append(snr_db  if is_burst else 0.0)
        _S.h_burst.append(1    if is_burst else 0)
        _S.h_snr.append(  snr_db  if is_burst else 0.0)
        _S.h_papr.append( papr_db if is_burst else 0.0)

        # ── Spectrogram ───────────────────────────────────────────────────────────
        _S.spec_data[1:, :] = _S.spec_data[:-1, :]
        _S.spec_data[0,  :] = r.zoom_db

        p5  = float(np.percentile(_S.spec_data,  5))
        p99 = float(np.percentile(_S.spec_data, 99))
        _im_sg.set_data(_S.spec_data)
        _im_sg.set_clim(vmin=max(p5, -55.0), vmax=min(p99 + 3.0, 2.0))

        if is_burst:
            _txt_sg_burst.set_text(
                f"\u25cf BURST  {sign}{dop_hz/1e3:.1f} kHz  "
                f"SNR {snr_db:.1f} dB  PAPR {papr_db:.1f} dB")
        else:
            _txt_sg_burst.set_text("")

        # ── Panel A ───────────────────────────────────────────────────────────────
        _line_spec.set_ydata(r.spec_db)
        _line_dop_mk.set_xdata([dop_hz/1e3, dop_hz/1e3])
        _line_dop_mk.set_color(LIME if is_burst else ROSE)
        _txt_spec_pk.set_text(
            f"{'● BURST  ' if is_burst else ''}{sign}{dop_hz/1e3:.2f} kHz  SNR {snr_db:.1f} dB")
        _txt_spec_pk.set_color(LIME if is_burst else DIM)

        # ── Panel B ───────────────────────────────────────────────────────────────
        _update_scurve()

        # ── Panel D ───────────────────────────────────────────────────────────────
        _line_snr.set_ydata(list(_S.h_snr))
        _line_papr.set_ydata(list(_S.h_papr))
        _ax_snr.set_ylim(0, max(30.0, max(_S.h_snr) + 3.0, max(_S.h_papr) + 3.0))
        col = (LIME  if (is_burst and snr_db >= BURST_SNR * 1.5)
               else AMBER if is_burst
               else DIM)
        _txt_snr_val.set_text(
            f"SNR {snr_db:.1f} dB  PAPR {papr_db:.1f} dB"
            if is_burst else "SNR --- dB  PAPR --- dB  (scanning)")
        _txt_snr_val.set_color(col)
        _txt_pwr_val.set_text(f"pwr {abs_pwr:.1f} dBW  (floor \u2265{BURST_PWR} dBW)")

        # ── Panel E ───────────────────────────────────────────────────────────────
        _update_timeline()

        # ── Status bar ────────────────────────────────────────────────────────────
        if is_burst:
            dop_lbl = ("approach \u2197" if dop_hz >  200
                       else "\u2198 receding"  if dop_hz < -200
                       else "\u2022 TCA")
            demod_suffix = ""
            if DEMOD_EN:
                if pr.raw_line is not None:
                    # Show first 40 chars of the bit field
                    _raw_short = pr.raw_line.split()[-1][:40] if pr.raw_line else ""
                    demod_suffix = f"  \u00b7  \u2713 decoded ({_S.decoded_count})  {_raw_short}"
                else:
                    demod_suffix = "  \u00b7  \u231b demod…"
            _txt_status.set_text(
                f"\u25cf BURST #{_tracker.burst_count}"
                f"  \u00b7  {sign}{dop_hz/1e3:.2f} kHz  ({dop_lbl})"
                f"  \u00b7  SNR {snr_db:.1f} dB"
                f"  \u00b7  PAPR {papr_db:.1f} dB"
                f"  \u00b7  pwr {abs_pwr:.1f} dBW"
                f"  \u00b7  Pass #{_tracker.pass_count}"
                + demod_suffix)
            _txt_status.set_color(LIME if pr.raw_line is None else TEAL)
        else:
            elapsed = _tracker.time_since_last_burst
            if elapsed > 0:
                _txt_status.set_text(
                    f"\u25cb scanning  \u00b7  {sign}{dop_hz/1e3:.1f} kHz"
                    f"  \u00b7  last burst {elapsed:.0f} s ago"
                    f"  \u00b7  {_tracker.burst_count} total"
                    + (f"  \u00b7  decoded {_S.decoded_count}" if DEMOD_EN else ""))
            else:
                _txt_status.set_text(
                    f"\u25cb scanning for Iridium TDMA burst \u2026  "
                    f"\u0394f {sign}{dop_hz/1e3:.1f} kHz  SNR {snr_db:.1f} dB")
            _txt_status.set_color(DIM)


    # ══════════════════════════════════════════════════════════════════════════════
    # LAUNCH
    # ══════════════════════════════════════════════════════════════════════════════
    _ani = animation.FuncAnimation(
        _fig, _update, interval=_INTERVAL_MS, blit=False, cache_frame_data=False)


    def _on_close(_evt) -> None:
        _kraken.stop()
        if _raw_file:
            _raw_file.close()
        print(f"\n[IRD] Closed.  Bursts detected: {_tracker.burst_count}"
              f"  Passes: {_tracker.pass_count}"
              + (f"  Decoded: {_S.decoded_count}" if DEMOD_EN else ""))


    _fig.canvas.mpl_connect("close_event", _on_close)

    print(f"[IRD] Display running — animation interval = {_INTERVAL_MS} ms"
          f"  ({1000 // _INTERVAL_MS} fps target)")
    print(f"[IRD] Spectrogram: {_SPEC_HIST} rows × {_N_SPEC_COLS} cols"
          f"  (\u00b1{_ZOOM_LIM_KHZ:.0f} kHz  ·  "
          f"time span \u2248 {_SPEC_HIST*_INTERVAL_MS/1000:.1f} s)")
    print("[IRD] Close the plot window to stop.")

    plt.show()


if __name__ == "__main__":
    main()
