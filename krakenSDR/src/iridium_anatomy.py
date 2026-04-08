#!/usr/bin/env python3
"""
iridium_anatomy.py — Iridium burst packet anatomy viewer
=========================================================

Scans a baseband IQ recording, demodulates every detected burst and renders
an interactive dissection of each packet:

  ┌───────────────────────────────────────────────────────────┐
  │  TIMELINE — all bursts on a time axis (A:OK=teal / no=dim)│
  ├──────────────────────────┬────────────────────────────────│
  │  IQ ENVELOPE             │  DQPSK CONSTELLATION          │
  │  with preamble/UW/data   │  symbol decisions in IQ plane  │
  │  regions highlighted     │                                │
  ├──────────────────────────┴────────────────────────────────│
  │  BIT ANATOMY — colored strip: [PREAM][   UW  ][ DATA ···] │
  ├───────────────────────────────────────────────────────────│
  │  metadata: freq / SNR / doppler / A:OK / conf   ◀  idx  ▶ │
  └───────────────────────────────────────────────────────────┘

Navigation
----------
  ← / →   previous / next burst
  click on timeline   jump to burst

Usage
-----
    # Auto-discover the only WAV in krakenSDR/recordings/
    python3 iridium_anatomy.py

    # Explicit file
    python3 iridium_anatomy.py path/to/recording.wav

    # Custom thresholds
    python3 iridium_anatomy.py --snr 10 --papr 6

    # Limit scan to first N seconds (faster startup)
    python3 iridium_anatomy.py --limit 120
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

from hardware.file_iq_source import FileIQSource
from core.burst_pipeline import BurstPipeline
from core.burst import TDMA_FRAME_S
from core.iridium_demod import (
    DemodDebug, IridiumDemod, DOWNLINK, UPLINK,
    UW_LENGTH, UW_DOWNLINK, UW_UPLINK, PREAMBLE_LENGTH,
)
from ui.theme import (
    apply_mpl_style,
    BG, BG2, BG3, BORDER, DIM,
    BLUE, TEAL, AMBER, VIOLET, ROSE, LIME, TEXT, MUTED,
)

# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------
_C_PREAM = BLUE
_C_UW_OK = AMBER
_C_UW_NO = ROSE
_C_DATA  = TEAL
_C_TRAIL = MUTED

_ALPHA_REGION = 0.18     # background shading alpha


def _rgb(css: str) -> tuple:
    return mcolors.to_rgb(css)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _auto_find_wav() -> Path:
    """Return the first WAV file in krakenSDR/recordings/ relative to _HERE."""
    candidates = [
        Path(_HERE).parent.parent / "krakenSDR" / "recordings",  # repo root
        Path(_HERE).parent / "recordings",
        Path(_HERE) / "recordings",
    ]
    for d in candidates:
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            return wavs[0]
    raise FileNotFoundError(
        "No WAV file found. Pass the path as a positional argument."
    )


def scan_recording(
    wav_path: Path,
    snr:      float = 12.0,
    papr:     float = 8.0,
    pwr:      float = -95.0,
    limit_s:  float = float("inf"),
    frame_sz: int   = 131_072,
) -> list[dict]:
    """
    Scan *wav_path* and return a list of burst dicts.

    Each dict has keys:
        t_s          - burst time in seconds from recording start
        snr_db       - burst SNR (dB)
        doppler_hz   - carrier offset (Hz)
        center_freq  - SDR base frequency (Hz)
        debug        - DemodDebug | None  (None if demod buffer too short)
    """
    src = FileIQSource(wav_path, frame_size=frame_sz)
    src.start()
    fs = src.sample_rate

    demodulator = IridiumDemod(input_fs=int(fs))

    pipe = BurstPipeline(
        input_fs       = int(fs),
        center_freq_hz = 1_626_270_000.0,
        burst_snr      = snr,
        burst_papr     = papr,
        burst_pwr      = pwr,
        demod_enabled  = False,   # we call demod_full manually below
        filename       = wav_path.stem,
    )
    T0    = pipe._t0
    bursts: list[dict] = []
    total  = src.total_samples
    done   = 0

    print(f"[ANATOMY] Scanning {wav_path.name}  "
          f"({src.duration_s:.0f} s, {fs/1e6:.3f} MS/s) …")

    while True:
        frame = src.get_frame(timeout=0)
        if frame is None:
            break
        t_s = src.elapsed_s - frame_sz / fs
        if t_s > limit_s:
            break

        x   = frame[0].astype(np.complex128)
        pr  = pipe.process(x, timestamp=T0 + t_s)
        done += frame_sz

        # Progress bar --------------------------------------------------
        pct = min(done / total, 1.0)
        bar = int(pct * 40)
        print(f"\r  [{('#' * bar).ljust(40)}] {pct*100:5.1f}%  "
              f"{len(bursts)} bursts", end="", flush=True)

        if not pr.burst.is_burst:
            continue

        # Demodulate with full debug output ----------------------------
        ts_ms  = t_s * 1000.0
        debug  = demodulator.demod_full(
            x              = x,
            doppler_hz     = pr.burst.doppler_hz,
            timestamp_ms   = ts_ms,
            center_freq_hz = 1_626_270_000.0,
            filename       = wav_path.stem,
            snr_db         = pr.burst.burst_snr_db,
        )

        bursts.append({
            "t_s":        t_s,
            "snr_db":     pr.burst.burst_snr_db,
            "doppler_hz": pr.burst.doppler_hz,
            "center_freq": 1_626_270_000.0,
            "debug":      debug,
        })

    src.stop()
    n_ok = sum(1 for b in bursts if b["debug"] and b["debug"].access_ok)
    print(f"\n[ANATOMY] Done — {len(bursts)} bursts  "
          f"{n_ok} A:OK  ({100*n_ok//max(len(bursts),1)}%)")
    return bursts


# ---------------------------------------------------------------------------
# Anatomy viewer
# ---------------------------------------------------------------------------

class AnatomyViewer:
    """Interactive matplotlib figure showing Iridium burst anatomy."""

    _SPS = 40   # samples per symbol at 1 Msps

    def __init__(self, bursts: list[dict], wav_name: str) -> None:
        self._bursts   = bursts
        self._wav_name = wav_name
        self._idx      = 0
        self._n        = len(bursts)

        apply_mpl_style()
        self._fig = plt.figure(
            figsize     = (15, 9),
            facecolor   = BG,
            constrained_layout = False,
        )
        self._fig.canvas.manager.set_window_title(
            f"Iridium Burst Anatomy — {wav_name}"
        )

        self._build_layout()
        self._draw_timeline()
        self._draw_burst()

        # Keyboard / mouse bindings
        self._fig.canvas.mpl_connect("key_press_event",     self._on_key)
        self._fig.canvas.mpl_connect("button_press_event",  self._on_click)

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        gs = gridspec.GridSpec(
            4, 2,
            figure        = self._fig,
            height_ratios = [0.90, 2.20, 2.00, 0.55],
            width_ratios  = [2.0, 1.0],
            hspace        = 0.55,
            wspace        = 0.30,
            left=0.07, right=0.97, top=0.93, bottom=0.07,
        )

        self._ax_tl   = self._fig.add_subplot(gs[0, :])   # timeline
        self._ax_env  = self._fig.add_subplot(gs[1, 0])   # IQ envelope
        self._ax_con  = self._fig.add_subplot(gs[1, 1])   # constellation
        self._ax_bits = self._fig.add_subplot(gs[2, :])   # bit anatomy
        self._ax_meta = self._fig.add_subplot(gs[3, :])   # metadata + nav

        for ax in (self._ax_tl, self._ax_env, self._ax_con,
                   self._ax_bits, self._ax_meta):
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        self._ax_meta.axis("off")

        # Navigation buttons
        btn_kw = dict(color=BG3, hovercolor="#3d4675")
        ax_prev = self._fig.add_axes([0.09, 0.015, 0.06, 0.035])
        ax_next = self._fig.add_axes([0.86, 0.015, 0.06, 0.035])
        self._btn_prev = Button(ax_prev, "◀  Prev", **btn_kw)
        self._btn_next = Button(ax_next, "Next  ▶", **btn_kw)
        self._btn_prev.label.set_color(TEXT)
        self._btn_next.label.set_color(TEXT)
        self._btn_prev.on_clicked(lambda _: self._navigate(-1))
        self._btn_next.on_clicked(lambda _: self._navigate(+1))

    # ------------------------------------------------------------------
    def _draw_timeline(self) -> None:
        ax = self._ax_tl
        ax.clear()

        ts   = [b["t_s"]    for b in self._bursts]
        dops = [b["doppler_hz"] / 1000 for b in self._bursts]
        snrs = [b["snr_db"] for b in self._bursts]
        oks  = [bool(b["debug"] and b["debug"].access_ok) for b in self._bursts]

        colors = [TEAL if ok else DIM for ok in oks]
        sizes  = [max(30, s * 4) for s in snrs]

        ax.scatter(ts, dops, c=colors, s=sizes, alpha=0.85,
                   linewidths=0, zorder=3)

        # Highlight current burst
        if self._bursts:
            b = self._bursts[self._idx]
            ax.scatter(
                [b["t_s"]],
                [b["doppler_hz"] / 1000],
                s=220, c=AMBER, marker="D", zorder=5, linewidths=1.5,
                edgecolors=TEXT,
            )

        ax.set_facecolor(BG2)
        ax.set_ylabel("Doppler  (kHz)", color=MUTED, fontsize=8)
        ax.set_xlabel("Time  (s)", color=MUTED, fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(True, alpha=0.25, color=BORDER)
        ax.set_title(
            f"{self._wav_name}  —  {self._n} bursts  "
            f"( ● A:OK  ·  ● no-sync )",
            color=TEXT, fontsize=9, pad=4,
        )

        # Legend
        leg = ax.legend(
            handles=[
                mpatches.Patch(color=TEAL, label="A:OK (UW found)"),
                mpatches.Patch(color=DIM,  label="A:no"),
                mpatches.Patch(color=AMBER, label="selected"),
            ],
            loc="upper right", fontsize=7,
            framealpha=0.3, facecolor=BG3, edgecolor=BORDER,
            labelcolor=TEXT,
        )

    # ------------------------------------------------------------------
    def _draw_burst(self) -> None:
        b = self._bursts[self._idx]
        d = b["debug"]

        self._draw_envelope(b, d)
        self._draw_constellation(d)
        self._draw_bits(d)
        self._draw_metadata(b, d)
        self._draw_timeline()     # refresh timeline to move diamond marker
        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _draw_envelope(self, b: dict, d: Optional[DemodDebug]) -> None:
        ax = self._ax_env
        ax.clear()
        ax.set_facecolor(BG2)

        if d is None:
            ax.text(0.5, 0.5, "demod unavailable", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED, fontsize=11)
            ax.set_title("IQ Envelope", color=MUTED, fontsize=9)
            return

        sps         = d.sps
        sync        = d.sync_start
        pream_start = max(0, sync - PREAMBLE_LENGTH * sps)
        uw_end      = sync + UW_LENGTH * sps
        data_end    = sync + d.nsymbols * sps

        iq   = d.iq_1m
        env  = np.abs(iq)
        t_us = np.arange(len(env)) / 1e6 * 1e3   # ms

        # Zoom into burst region (preamble start … data end + margin)
        margin_samp = sps * 8
        view_start  = max(0, pream_start - margin_samp)
        view_end    = min(len(t_us) - 1, data_end + margin_samp)
        ax.plot(t_us[view_start:view_end], env[view_start:view_end],
                color=BLUE, linewidth=0.8, alpha=0.85)

        def shade(start_idx, end_idx, color, label):
            s = max(view_start, start_idx)
            e = min(view_end,   end_idx)
            if e <= s or s >= len(t_us):
                return
            ax.axvspan(t_us[s], t_us[min(e, len(t_us)-1)],
                       alpha=_ALPHA_REGION, color=color, lw=0)
            mid = (t_us[s] + t_us[min(e, len(t_us)-1)]) / 2
            ax.text(mid, 0.91, label, ha="center", va="top",
                    color=color, fontsize=7.5, fontweight="bold",
                    transform=ax.get_xaxis_transform())

        shade(pream_start, sync,              _C_PREAM, "PREAMBLE")
        shade(sync,        uw_end,            _C_UW_OK if d.access_ok else _C_UW_NO, "UW")
        shade(uw_end,      data_end + sps*2,  _C_DATA,  "DATA")

        # Sync marker
        ax.axvline(t_us[min(sync, len(t_us)-1)], color=AMBER,
                   linewidth=1.2, linestyle="--", alpha=0.75)
        ax.set_xlim(t_us[view_start], t_us[view_end])

        ax.set_xlabel("ms (relative to burst frame)", color=MUTED, fontsize=8)
        ax.set_ylabel("|IQ|", color=MUTED, fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(True, alpha=0.25, color=BORDER)
        direction = "DL" if d.direction == DOWNLINK else "UL"
        ax.set_title(
            f"IQ Envelope  ({direction})  "
            f"SNR={b['snr_db']:.1f} dB  "
            f"Δf={b['doppler_hz']/1000:+.1f} kHz",
            color=TEXT, fontsize=9, pad=4,
        )

    # ------------------------------------------------------------------
    def _draw_constellation(self, d: Optional[DemodDebug]) -> None:
        ax = self._ax_con
        ax.clear()
        ax.set_facecolor(BG2)
        ax.set_aspect("equal")

        if d is None or len(d.symbol_samps) == 0:
            ax.text(0.5, 0.5, "no symbols", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_title("Constellation", color=MUTED, fontsize=9)
            return

        uw_n   = min(UW_LENGTH, len(d.symbol_samps))
        data_n = len(d.symbol_samps)

        # UW symbols
        uw_s = d.symbol_samps[:uw_n]
        ax.scatter(uw_s.real, uw_s.imag,
                   c=_C_UW_OK if d.access_ok else _C_UW_NO,
                   s=30, alpha=0.90, linewidths=0, zorder=4, label="UW")

        # DATA symbols
        if data_n > uw_n:
            dt_s = d.symbol_samps[uw_n:]
            ax.scatter(dt_s.real, dt_s.imag,
                       c=_C_DATA, s=18, alpha=0.65, linewidths=0, zorder=3,
                       label="DATA")

        # Ideal QPSK references
        ref = np.array([1+1j, -1+1j, -1-1j, 1-1j]) * d.level
        ax.scatter(ref.real, ref.imag,
                   marker="+", s=120, c=MUTED, linewidths=1.5, zorder=5)

        lim = max(float(np.max(np.abs(d.symbol_samps))) * 1.25, d.level * 2.0, 0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color=BORDER, linewidth=0.6, alpha=0.5)
        ax.axvline(0, color=BORDER, linewidth=0.6, alpha=0.5)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(True, alpha=0.20, color=BORDER)
        ax.set_xlabel("I", color=MUTED, fontsize=8)
        ax.set_ylabel("Q", color=MUTED, fontsize=8)
        ax.legend(fontsize=7, framealpha=0.3, facecolor=BG3,
                  edgecolor=BORDER, labelcolor=TEXT, loc="upper right")
        ax.set_title(
            f"DQPSK Constellation  ({data_n} symbols  conf={d.confidence:.0f}%)",
            color=TEXT, fontsize=9, pad=4,
        )

    # ------------------------------------------------------------------
    def _draw_bits(self, d: Optional[DemodDebug]) -> None:
        ax = self._ax_bits
        ax.clear()
        ax.set_facecolor(BG)

        if d is None:
            ax.text(0.5, 0.5, "no bit data", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_title("Bit Anatomy", color=MUTED, fontsize=9)
            ax.axis("off")
            return

        bits     = d.dataarray
        nbits    = len(bits)
        uw_bits  = UW_LENGTH * 2          # 24 bits
        uw_color = _C_UW_OK if d.access_ok else _C_UW_NO

        if nbits == 0:
            ax.text(0.5, 0.5, "no bits decoded", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.axis("off")
            return

        # Build pixel-row: shape (8, nbits) for a thick strip
        rows  = 8
        img   = np.zeros((rows, nbits, 3), dtype=float)

        for i, bit in enumerate(bits):
            if i < uw_bits:
                rgb = _rgb(uw_color)
                fade = 0.55 + 0.45 * bit
            else:
                rgb  = _rgb(_C_DATA)
                fade = 0.40 + 0.60 * bit

            img[:, i, :] = [c * fade for c in rgb]

        ax.imshow(
            img, aspect="auto", interpolation="nearest",
            extent=[0, nbits, 0, rows],
        )

        # Section labels above the strip
        def label_section(x_start, x_end, text, color):
            mid = (x_start + x_end) / 2
            ax.annotate(
                text,
                xy     = (mid, rows),
                xytext = (mid, rows + 0.4),
                ha     = "center", va     = "bottom",
                fontsize = 7.5, fontweight = "bold", color = color,
                annotation_clip = False,
            )
            ax.axvline(x_start, ymin=0, ymax=1.3, color=color,
                       linewidth=0.8, alpha=0.55, clip_on=False)

        label_section(0,       uw_bits,  f"UW  ({uw_bits}b)", uw_color)
        label_section(uw_bits, nbits,    f"DATA  ({nbits-uw_bits}b)", _C_DATA)

        # Bit value text (every 8th bit to avoid clutter)
        for i in range(0, nbits, 8):
            byte_val = 0
            for j in range(8):
                if i + j < nbits:
                    byte_val = (byte_val << 1) | bits[i + j]
            ax.text(i + 4, -0.6, f"{byte_val:02X}", ha="center", va="top",
                    fontsize=5.5, color=MUTED, family="monospace")

        # UW string overlay
        uw_sym_str = "".join(str(s) for s in d.symbols[:UW_LENGTH])
        expected   = UW_DOWNLINK if d.direction == DOWNLINK else UW_UPLINK
        match_str  = "✓ UW match" if d.access_ok else f"✗ UW mismatch  (got {uw_sym_str}  exp {expected})"

        ax.set_xlim(0, nbits)
        ax.set_ylim(-1.5, rows)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6.5, colors=MUTED)
        ax.set_xlabel("Bit index", color=MUTED, fontsize=8)
        ax.grid(axis="x", alpha=0.15, color=BORDER)

        status_color = LIME if d.access_ok else ROSE
        ax.set_title(
            f"Bit Anatomy  {nbits} bits / {d.nsymbols} symbols  —  {match_str}",
            color=status_color, fontsize=9, pad=10,
        )

    # ------------------------------------------------------------------
    def _draw_metadata(self, b: dict, d: Optional[DemodDebug]) -> None:
        ax = self._ax_meta
        ax.clear()
        ax.axis("off")

        freq_mhz = (b["center_freq"] + b["doppler_hz"]) / 1e6
        dop_khz  = b["doppler_hz"] / 1e3
        direction = "— "
        conf_str  = "—"
        nsym_str  = "—"
        status_col = MUTED

        if d is not None:
            direction  = "DL ↓" if d.direction == DOWNLINK else "UL ↑"
            conf_str   = f"{d.confidence:.0f}%"
            nsym_str   = str(d.nsymbols)
            status_col = LIME if d.access_ok else ROSE

        access_str = "A:OK ✓" if (d and d.access_ok) else "A:no ✗"
        lo_str     = "lead-out ✓" if (d and d.lead_out_ok) else ""

        parts = [
            f"Burst {self._idx + 1} / {self._n}",
            f"  t = {b['t_s']:.2f} s",
            f"  freq = {freq_mhz:.3f} MHz",
            f"  Δf = {dop_khz:+.2f} kHz",
            f"  SNR = {b['snr_db']:.1f} dB",
            f"  {direction}",
            f"  {access_str}",
            f"  conf = {conf_str}",
            f"  sym = {nsym_str}",
        ]
        if lo_str:
            parts.append(f"  {lo_str}")

        text = "    ".join(parts)
        ax.text(
            0.5, 0.5, text,
            transform = ax.transAxes,
            ha="center", va="center",
            fontsize=9, color=TEXT, family="monospace",
        )

        # Keyboard hint
        ax.text(
            0.98, 0.05, "← → arrows or buttons to navigate",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=MUTED, style="italic",
        )

    # ------------------------------------------------------------------
    def _navigate(self, delta: int) -> None:
        self._idx = (self._idx + delta) % self._n
        self._draw_burst()

    def _on_key(self, event) -> None:
        if event.key in ("left",  "a"): self._navigate(-1)
        if event.key in ("right", "d"): self._navigate(+1)

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax_tl:
            return
        if event.xdata is None:
            return
        # Find nearest burst by time
        ts_arr = np.array([b["t_s"] for b in self._bursts])
        self._idx = int(np.argmin(np.abs(ts_arr - event.xdata)))
        self._draw_burst()

    # ------------------------------------------------------------------
    def show(self) -> None:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file", nargs="?", default=None, metavar="FILE",
                    help="IQ WAV recording (auto-detected if omitted)")
    ap.add_argument("--snr",   type=float, default=12.0, metavar="DB",
                    help="Burst detection SNR threshold (default: 12)")
    ap.add_argument("--papr",  type=float, default=8.0,  metavar="DB",
                    help="PAPR threshold (default: 8)")
    ap.add_argument("--limit", type=float, default=float("inf"), metavar="S",
                    help="Scan only first N seconds of recording")
    args = ap.parse_args()

    if args.file:
        wav = Path(args.file)
    else:
        wav = _auto_find_wav()

    if not wav.exists():
        sys.exit(f"[ERROR] File not found: {wav}")

    bursts = scan_recording(
        wav_path = wav,
        snr      = args.snr,
        papr     = args.papr,
        limit_s  = args.limit,
    )

    if not bursts:
        sys.exit("[ANATOMY] No bursts detected — try lowering --snr or --papr.")

    viewer = AnatomyViewer(bursts, wav.name)
    viewer.show()


if __name__ == "__main__":
    main()
