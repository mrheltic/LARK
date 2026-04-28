#!/usr/bin/env python3
"""
tx_868_gui.py — LibreSDR 868 MHz TX + loopback GUI (AD9363 / Zynq7020)

Interactive real-time GUI for transmitting at 868 MHz and monitoring the
TX→RX loopback signal on the same device.

Modes
-----
  CW      — continuous-wave tone at centre frequency (DC baseband)
  BURST   — π/4-DQPSK bursts with RRC pulse shaping (Iridium-like framing)

Display panels (3 × 2 grid)
-----------------------------
  [0,0]  Spectrum (FFT magnitude) — RX loopback
  [0,1]  Waterfall (time × frequency) scrolling spectrogram
  [0,2]  IQ constellation — RX loopback
  [1,0]  Time-domain amplitude — RX loopback
  [1,1]  TX waveform — last transmitted buffer
  [1,2]  Status + metrics (power, SNR, frame counter)

Controls (toolbar buttons)
--------------------------
  [CW / BURST] toggle mode
  [TX ON / TX OFF] start / stop transmission
  [Gain ▲ / Gain ▼] ± 3 dB TX attenuation step
  [Freq ▲ / Freq ▼] ± 100 kHz frequency step (TX + RX retuned together)

Usage
-----
    python3 tx_868_gui.py
    python3 tx_868_gui.py --uri ip:192.168.2.1 --freq 868100000 --gain -40
    python3 tx_868_gui.py --demo          # no hardware, synthetic loopback

Requirements
------------
    pip install pyadi-iio matplotlib scipy numpy
"""

from __future__ import annotations

import argparse
import collections
import os
import queue
import sys
import threading
import time
import warnings
from types import SimpleNamespace

# Suppress spurious Axes3D import warning (system vs venv matplotlib conflict)
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import numpy as np
import scipy.signal as sp_signal
import matplotlib
matplotlib.use("TkAgg")           # works headless on Linux with $DISPLAY
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ── path setup so we can import from the libreSDR package ────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Real Iridium parameters (from iridium/realistic_sim.py) ──────────────────
# realistic_sim.py now defers its matplotlib import, so importing here is safe.
from iridium.realistic_sim import (
    SYMBOL_RATE,                        # 25 000 sps
    SPS,                                # 10   (samples/symbol at base rate)
    SAMPLE_RATE       as IRA_SAMPLE_RATE,  # 250 000 Hz (= SYMBOL_RATE × SPS)
    RRC_BETA,                           # 0.4  (confirmed by gr-iridium)
    RRC_NUM_TAPS,                       # 111
    IRA_BURST_SYMS,                     # 245  (preamble+UW+data+tail)
    IRA_PREAMBLE_SYMS,                  # 64
    IRA_GUARD_SYMS,                     # 8
    SLOT_SYMS,                          # 281
    SUPERFRAME_S,                       # 0.090 s  (90 ms)
    generate_rrc_filter,
    generate_ira_burst,
)

# Upsampling ratio: IRA_SAMPLE_RATE (250 kHz) → TX_SAMPLE_RATE (1 MHz)
_IRA_UPS  = 4   # TX_SAMPLE_RATE // IRA_SAMPLE_RATE

# Preamble tone offset (64 × π/4-DQPSK all-zero dibits → tone at +Rs/8 above carrier)
_PREAMBLE_TONE_HZ = SYMBOL_RATE // 8  # = 3125 Hz

# ── Defaults (overridden by CLI args) ────────────────────────────────────────
DEFAULT_URI       = "ip:192.168.1.10"
DEFAULT_FREQ_HZ   = 868_100_000   # 868.1 MHz ISM
TX_SAMPLE_RATE    = 1_000_000     # 1 MSPS (reliable over Ethernet/USB)
TX_RF_BW          = 200_000       # 200 kHz
DEFAULT_TX_GAIN   = -20.0         # dB attenuation (start at -20; -89.75 = min power)
# NOTE (data-driven, 2026-04-28): at -40 dB RX SNR ≈ 1.5 dB → only 1% of bursts
# valid.  At -20 dB (+20 dB), expected SNR ≈ 21 dB → ~100% valid.
# Reduce back toward -40 if ADC saturates (check KrakenSDR eigenvalue spread > 20 dB).
RX_GAIN_DB        = 30.0
RX_GAIN_MODE      = "slow_attack"

# Loopback RX buffer: grab this many samples per refresh
RX_BUF_SIZE       = 2048
WATERFALL_ROWS    = 60            # scrolling history lines
FFT_SIZE          = 512
WATERFALL_VMIN    = -80           # dB
WATERFALL_VMAX    = -10
_DISP_SAMPLES     = 512           # samples shown in time-domain plots (subset)

BURST_PERIOD_S    = 0.10          # transmit one burst every N seconds (burst mode)
FREQ_STEP_HZ      = 100_000       # ± step for frequency buttons
GAIN_STEP_DB      = 3.0           # ± step for gain buttons

# ── Palette ──────────────────────────────────────────────────────────────────
BG     = "#1a1d27"
BG2    = "#21253a"
BG3    = "#2a2f47"
C_BDR  = "#3b4263"
C_MUT  = "#8891b0"
C_TEXT = "#d8dae8"
C_BLUE = "#5ea4e0"
C_TEAL = "#4ecdc4"
C_AMB  = "#f4a431"
C_GRN  = "#6dd97d"
C_ROSE = "#f16b6f"
C_VIO  = "#a78bfa"


# =============================================================================
# Signal generators
# =============================================================================

def _make_cw_buf(n: int = TX_SAMPLE_RATE // 10) -> np.ndarray:
    """DC IQ buffer → carrier at LO frequency. Amplitude = 0.9 × full-scale."""
    return (np.ones(n, dtype=np.complex64) * 0.9 * (2 ** 14)).astype(np.complex64)


def _make_ira_buf(rrc: np.ndarray,
                  frame_count: int = 0,
                  sat_id: int = 47,
                  beam_id: int = 3) -> np.ndarray:
    """
    Generate one complete IRA TDMA slot using the faithful Iridium model.

    Structure (281 symbols @ 25 ksps = 11.24 ms per slot):
      [Guard 8] [Preamble 64] [UW 12] [Data 167+2 tail] [Guard 8] [Silence 20]

    Preamble properties:
      All-zero dibits → constant Δφ = +π/4 per symbol → single-frequency
      tone at carrier + Rs/8 = carrier + 3 125 Hz.  This is the burst
      detection signature used by gr-iridium.

    RRC pulse shaping: β = 0.4, 111-tap filter (10 samples/symbol @ 250 kHz)
    Convolutional encoding: rate 1/2, K=7, G0=0x79, G1=0x5B (NASA/CCSDS)
    UW: absolute BPSK symbols from iridium.h UW_DL[]

    The slot is resampled 250 kHz → 1 MHz (×4) for the AD9363.
    """
    # Generate slot at 250 kHz (10 SPS)
    slot_iq, _ = generate_ira_burst(rrc, sat_id=sat_id, beam_id=beam_id,
                                     frame_count=frame_count)
    # Resample to TX_SAMPLE_RATE = 1 MHz (ratio 4:1, exact integer)
    upsampled = sp_signal.resample_poly(slot_iq, _IRA_UPS, 1)
    # Scale to 80 % DAC full-scale
    peak = float(np.max(np.abs(upsampled))) + 1e-12
    return (upsampled / peak * 0.8 * (2 ** 14)).astype(np.complex64)


# =============================================================================
# Hardware helpers
# =============================================================================

def _connect_sdr(uri: str):
    try:
        import adi
    except ImportError:
        print("[ERROR] pyadi-iio not installed:  pip install pyadi-iio")
        sys.exit(1)
    print(f"[SDR] Connecting to {uri} ...", end=" ", flush=True)
    for attempt in range(3):
        try:
            sdr = adi.Pluto(uri)
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            print("OK")
            return sdr
        except OSError as exc:
            if exc.errno == 16 and attempt < 2:
                print(f"busy ({attempt+1}/3), retry in 3 s ...", end=" ", flush=True)
                time.sleep(3)
            else:
                print(f"FAILED\n[ERROR] {exc}")
                sys.exit(1)
        except Exception as exc:
            print(f"FAILED\n[ERROR] {exc}")
            sys.exit(1)


def _configure_hw(sdr, freq_hz: int, tx_gain: float) -> None:
    sdr.sample_rate           = int(TX_SAMPLE_RATE)
    sdr.tx_rf_bandwidth       = int(TX_RF_BW)
    sdr.rx_rf_bandwidth       = int(TX_RF_BW)
    sdr.tx_lo                 = int(freq_hz)
    sdr.rx_lo                 = int(freq_hz)
    sdr.tx_hardwaregain_chan0 = float(tx_gain)
    sdr.gain_control_mode_chan0 = RX_GAIN_MODE
    if RX_GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = float(RX_GAIN_DB)
    sdr.rx_buffer_size        = int(RX_BUF_SIZE)
    try:
        sdr.tx_cyclic_buffer  = False
    except Exception:
        pass


# =============================================================================
# TX thread
# =============================================================================

class TxThread(threading.Thread):
    """
    Background thread that continuously pushes IQ samples to the SDR
    (or generates synthetic data in demo mode).
    """

    def __init__(self, sdr, state: SimpleNamespace, demo: bool = False):
        super().__init__(daemon=True, name="TxThread")
        self._sdr   = sdr
        self._state = state
        self._demo  = demo
        self._rng   = np.random.default_rng(42)
        # Pre-build RRC filter once (at construction time, not per-burst)
        self._rrc          = generate_rrc_filter(RRC_BETA, SPS, RRC_NUM_TAPS)
        self._frame_count  = 0
        self._sat_id       = 47   # typical Iridium satellite ID
        self._beam_id      = 3    # typical Iridium beam ID

    def run(self) -> None:
        S = self._state
        last_burst = 0.0

        while S.running:
            if not S.tx_on:
                time.sleep(0.05)
                continue

            mode = S.mode   # "cw" | "ira"

            if mode == "cw":
                buf = _make_cw_buf()
            else:
                # IRA: one TDMA slot every SUPERFRAME_S (90 ms)
                # Slot = 281 symbols @ 25 ksps = 11.24 ms of data;
                # the remaining ~78.8 ms we sleep to honour the superframe period.
                now = time.monotonic()
                if now - last_burst < SUPERFRAME_S:
                    time.sleep(0.005)
                    continue
                buf = _make_ira_buf(self._rrc,
                                    frame_count=self._frame_count,
                                    sat_id=self._sat_id,
                                    beam_id=self._beam_id)
                self._frame_count += 1
                S.ira_frame_count  = self._frame_count
                last_burst = time.monotonic()

            with S.tx_buf_lock:
                S.tx_buf = buf.copy()

            if self._demo:
                time.sleep(0.02)
                continue

            try:
                self._sdr.tx_destroy_buffer()
            except Exception:
                pass
            try:
                self._sdr.tx(buf)
            except Exception as exc:
                print(f"[TX] {exc}")
                time.sleep(0.1)


# =============================================================================
# RX thread
# =============================================================================

class RxThread(threading.Thread):
    """Grabs RX loopback samples and fills a shared ring buffer."""

    def __init__(self, sdr, state: SimpleNamespace, demo: bool = False):
        super().__init__(daemon=True, name="RxThread")
        self._sdr   = sdr
        self._state = state
        self._demo  = demo
        self._rng   = np.random.default_rng(7)
        # Position within tx_buf for cyclic loopback (advances per _demo_samples call)
        self._loopback_pos = 0

    def _demo_samples(self) -> np.ndarray:
        """
        Realistic loopback demo.

        When TX is off: AWGN noise floor at −50 dBFS (always visible in waterfall).

        When TX is on:
          CW mode — CW tone at +50 kHz offset (clearly off-centre in spectrum).
          IRA mode — cyclic loopback of the last transmitted IRA slot.
                     The RX window (2048 samples = 2.048 ms) scrolls through the
                     11.24 ms slot, so the waterfall shows alternating sections:
                       • Guard (silence)          0.32 ms
                       • Preamble (tone at +3125 Hz) 2.56 ms  ← visible spike!
                       • UW + Data (broadband)    7.32 ms
                       • Guard + Silence           1.12 ms

        All samples are scaled to DAC range (±2^14) to match _compute_fft.
        """
        S = self._state
        n = RX_BUF_SIZE

        # Baseline AWGN noise — always visible
        noise_sigma = 10 ** (-52.0 / 20.0)   # −52 dBFS per-sample RMS
        noise = (noise_sigma * (2 ** 14) * (
            self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)
        )).astype(np.complex64)

        if not S.tx_on:
            return noise

        # ── TX ON ──────────────────────────────────────────────────────────────
        with S.tx_buf_lock:
            tx = S.tx_buf.copy()

        if len(tx) < n:
            # Buffer not ready yet (first frame still being generated)
            return noise

        if S.mode == "ira":
            # Cyclic scroll through the IRA slot to expose all sections
            pos = self._loopback_pos % len(tx)
            end = pos + n
            if end <= len(tx):
                loopback = tx[pos:end].copy()
            else:
                loopback = np.concatenate([tx[pos:], tx[:end - len(tx)]])
            self._loopback_pos = end % len(tx)
            # 20 dB path loss (cable loopback)
            loopback = loopback * 0.1
        else:
            # CW: static tone at +50 kHz offset
            t = np.arange(n) / TX_SAMPLE_RATE
            tone_amp = 10 ** (-10.0 / 20.0) * (2 ** 14)
            loopback = (tone_amp * np.exp(1j * 2 * np.pi * 50e3 * t)
                        ).astype(np.complex64)

        return (loopback + noise).astype(np.complex64)

    def run(self) -> None:
        S = self._state

        while S.running:
            try:
                if self._demo:
                    samples = self._demo_samples()
                    time.sleep(RX_BUF_SIZE / TX_SAMPLE_RATE)
                else:
                    raw     = self._sdr.rx()
                    samples = np.asarray(raw, dtype=np.complex64)

                with S.rx_lock:
                    S.rx_buf = samples
                    S.rx_ready = True
            except Exception as exc:
                print(f"[RX] {exc}")
                time.sleep(0.1)


# =============================================================================
# Shared state
# =============================================================================

def _make_state(freq_hz: int, tx_gain: float) -> SimpleNamespace:
    wfall = np.full((WATERFALL_ROWS, FFT_SIZE), WATERFALL_VMIN, dtype=np.float32)
    return SimpleNamespace(
        running     = True,
        tx_on       = False,
        mode        = "cw",        # "cw" | "ira"
        freq_hz     = freq_hz,
        tx_gain     = tx_gain,
        # TX waveform
        tx_buf      = np.zeros(512, dtype=np.complex64),
        tx_buf_lock = threading.Lock(),
        # RX loopback
        rx_buf      = np.zeros(RX_BUF_SIZE, dtype=np.complex64),
        rx_ready    = False,
        rx_lock     = threading.Lock(),
        # Spectrum + waterfall
        fft_db      = np.full(FFT_SIZE, WATERFALL_VMIN, dtype=np.float32),
        waterfall   = wfall,
        # Metrics
        rx_power_db = -999.0,
        snr_db      = -999.0,
        frame_count = 0,
        ira_frame_count = 0,
        # Reconfig request from UI
        reconfig    = False,
    )


# =============================================================================
# Metrics helpers
# =============================================================================

def _compute_fft(samples: np.ndarray) -> np.ndarray:
    win = np.blackman(FFT_SIZE).astype(np.float32)
    n   = min(len(samples), FFT_SIZE)
    x   = samples[-n:] * (2 ** -14)   # normalise to ±1
    X   = np.fft.fftshift(np.fft.fft(x[:FFT_SIZE] * win, FFT_SIZE))
    psd = 20 * np.log10(np.abs(X) / FFT_SIZE + 1e-12)
    return psd.astype(np.float32)


def _compute_power_snr(fft_db: np.ndarray) -> tuple[float, float]:
    sig_bins = FFT_SIZE // 4
    centre   = FFT_SIZE // 2
    sig_pow  = float(np.max(fft_db[centre - sig_bins:centre + sig_bins]))
    noise_fl = float(np.median(
        np.concatenate([fft_db[:FFT_SIZE // 8], fft_db[7 * FFT_SIZE // 8:]])))
    return sig_pow, sig_pow - noise_fl


# =============================================================================
# GUI
# =============================================================================

def _build_gui(S: SimpleNamespace, sdr, demo: bool) -> None:  # noqa: C901
    """
    Layout (figure coordinates, bottom=0 top=1):
    ┌─────────────────────────────────────────────────────────┐  ← y=1.00
    │  title bar                                               │  ← y=0.96
    ├──────────────┬──────────────┬──────────────────────────┤  ← y=0.93
    │ RX Spectrum  │  Waterfall   │   IQ Constellation        │  (top row)
    │  [0,0]       │  [0,1]       │   [0,2]                   │
    ├──────────────┼──────────────┼──────────────────────────┤  ← y=0.50
    │ RX Amplitude │ TX Waveform  │   Status & Metrics        │  (bottom row)
    │  [1,0]       │  [1,1]       │   [1,2]                   │
    ├──────────────┴──────────────┴──────────────────────────┤  ← y=0.21
    │  [TX START/STOP]  [CW|BURST]  [Freq−]  [Freq+]  [Gain−]  [Gain+]  │  ← buttons
    └─────────────────────────────────────────────────────────┘  ← y=0.00

    Performance notes:
    - fill_between is NOT recreated each frame (uses Polygon.set_xy instead)
    - Time-domain plots show _DISP_SAMPLES points, not the full RX buffer
    - Waterfall is updated via imshow.set_data (no new Artist)
    - Animation interval = 350 ms (~3 fps) — sufficient for visual feedback
    """
    from matplotlib.patches import Polygon as MplPolygon

    freq_axis = (np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / TX_SAMPLE_RATE)) / 1e3)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8.5), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title(  # type: ignore[union-attr]
            "LibreSDR 868 MHz  —  TX + RX loopback" + ("  [DEMO]" if demo else ""))
    except Exception:
        pass

    # ── Plot grid (top 75%, from y=0.21 to y=0.93) ───────────────────────────
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.15, 1.0],
        left=0.07, right=0.97,
        top=0.93, bottom=0.22,
        hspace=0.55, wspace=0.35,
    )

    # helper: style an axes consistently
    def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_facecolor(BG2)
        ax.set_title(title, color=C_TEXT, fontsize=9, pad=4)
        ax.set_xlabel(xlabel, color=C_MUT, fontsize=7)
        ax.set_ylabel(ylabel, color=C_MUT, fontsize=7)
        ax.tick_params(colors=C_MUT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BDR)
        ax.grid(color=C_BDR, lw=0.4, alpha=0.4)

    # ── [0,0]  RX Spectrum ────────────────────────────────────────────────────
    ax_spec = fig.add_subplot(gs[0, 0])
    _style(ax_spec, "RX Spectrum", "Freq offset [kHz]", "Power [dBFS]")
    ax_spec.set_xlim(freq_axis[0], freq_axis[-1])
    ax_spec.set_ylim(WATERFALL_VMIN, 5)
    spec_line, = ax_spec.plot(freq_axis, np.full(FFT_SIZE, WATERFALL_VMIN),
                               "-", color=C_TEAL, lw=1.1, zorder=3)
    # Filled area under spectrum — updated cheaply by mutating Polygon vertices
    _fill_xs = np.r_[freq_axis, freq_axis[::-1]]
    _fill_ys = np.r_[np.full(FFT_SIZE, WATERFALL_VMIN),
                     np.full(FFT_SIZE, WATERFALL_VMIN)]
    spec_poly = MplPolygon(
        np.column_stack([_fill_xs, _fill_ys]),
        closed=True, color=C_TEAL, alpha=0.13, zorder=2,
    )
    ax_spec.add_patch(spec_poly)
    # Pre-allocate polygon vertex array — reused every frame (zero heap alloc in hot path)
    _poly_verts = np.empty((2 * FFT_SIZE, 2), dtype=np.float64)
    _poly_verts[:FFT_SIZE, 0] = freq_axis
    _poly_verts[FFT_SIZE:, 0] = freq_axis[::-1]
    _poly_verts[FFT_SIZE:, 1] = WATERFALL_VMIN   # bottom edge is constant
    _poly_verts[:FFT_SIZE, 1] = WATERFALL_VMIN   # initialise top edge too

    # ── [0,1]  Waterfall ──────────────────────────────────────────────────────
    ax_wfall = fig.add_subplot(gs[0, 1])
    _style(ax_wfall, "Waterfall  (newest → top)", "Freq offset [kHz]", "← older")
    ax_wfall.set_yticks([])
    im_wfall = ax_wfall.imshow(
        S.waterfall,
        origin="upper", aspect="auto",
        extent=[freq_axis[0], freq_axis[-1], WATERFALL_ROWS, 0],
        cmap="inferno",
        vmin=WATERFALL_VMIN, vmax=WATERFALL_VMAX,
        interpolation="nearest",
    )

    # ── [0,2]  IQ Constellation ───────────────────────────────────────────────
    ax_iq = fig.add_subplot(gs[0, 2])
    _style(ax_iq, "IQ Constellation  (RX loopback)", "I", "Q")
    _lim = 1.15
    ax_iq.set_xlim(-_lim, _lim)
    ax_iq.set_ylim(-_lim, _lim)
    ax_iq.set_aspect("equal")
    ax_iq.axhline(0, color=C_BDR, lw=0.7)
    ax_iq.axvline(0, color=C_BDR, lw=0.7)
    # Unit circle reference
    _theta = np.linspace(0, 2 * np.pi, 128)
    ax_iq.plot(np.cos(_theta), np.sin(_theta), "--", color=C_BDR, lw=0.6, alpha=0.5)
    iq_dots, = ax_iq.plot([], [], ".", color=C_VIO, ms=1.8, alpha=0.35)

    # ── [1,0]  RX time domain ─────────────────────────────────────────────────
    ax_rxtime = fig.add_subplot(gs[1, 0])
    _style(ax_rxtime, "RX Amplitude  (time domain)", "Sample index", "Normalised |IQ|")
    ax_rxtime.set_xlim(0, _DISP_SAMPLES)
    ax_rxtime.set_ylim(-0.05, 1.15)
    rxtime_line, = ax_rxtime.plot(
        np.arange(_DISP_SAMPLES), np.zeros(_DISP_SAMPLES),
        "-", color=C_BLUE, lw=0.9)

    # ── [1,1]  TX waveform preview ────────────────────────────────────────────
    ax_txtime = fig.add_subplot(gs[1, 1])
    _style(ax_txtime, "TX Waveform Preview  (last buffer)", "Sample index", "Normalised |IQ|")
    ax_txtime.set_xlim(0, _DISP_SAMPLES)
    ax_txtime.set_ylim(-0.05, 1.15)
    txtime_line, = ax_txtime.plot(
        np.arange(_DISP_SAMPLES), np.zeros(_DISP_SAMPLES),
        "-", color=C_AMB, lw=0.9)

    # ── [1,2]  Status panel ───────────────────────────────────────────────────
    ax_stat = fig.add_subplot(gs[1, 2])
    ax_stat.set_facecolor(BG2)
    ax_stat.set_title("Status", color=C_TEXT, fontsize=9, pad=4)
    ax_stat.axis("off")
    for sp in ax_stat.spines.values():
        sp.set_edgecolor(C_BDR)
    _stat_defs = [
        # (y_axes_fraction, color)
        (0.92, C_TEXT),   # Frequency
        (0.78, C_TEXT),   # TX Gain
        (0.64, C_AMB),    # Mode
        (0.50, C_GRN),    # TX status
        (0.36, C_TEAL),   # RX Power
        (0.22, C_TEAL),   # SNR
        (0.08, C_MUT),    # Frames / sample rate
    ]
    stat_texts = []
    for y, col in _stat_defs:
        t = ax_stat.text(
            0.06, y, "—",
            transform=ax_stat.transAxes,
            color=col, fontsize=8.5, va="top",
            fontfamily="monospace",
        )
        stat_texts.append(t)

    # ── Figure title (fixed text, not suptitle that may drift) ───────────────
    title_txt = fig.text(
        0.5, 0.966,
        "LibreSDR  AD9363  |  868 MHz  TX + RX Loopback"
        + ("  [DEMO MODE]" if demo else ""),
        ha="center", va="top",
        color=C_TEXT, fontsize=10, fontweight="bold",
    )

    # =========================================================================
    # NOTE: blit=False is used intentionally. blit=True with TkAgg has known
    # issues with Patch (spec_poly bbox) and AxesImage (im_wfall) — the blit
    # background is captured before data arrives, leaving plots invisible.
    # Performance is adequate because: Polygon.set_xy + in-place waterfall
    # shift eliminate the main allocation hotspots; draw_idle() is coalesced
    # by the event loop; and the 400 ms interval only fires 2.5 times/sec.
    # Button strip  (figure coords y=0.04..0.17, clear of the plot area)
    # Layout: [  TX START/STOP  ]  [  Mode: CW  ]  [ ◄ Freq ]  [ Freq ► ]  [ Gain ▼ ]  [ Gain ▲ ]
    # =========================================================================
    # ── Button layout (no overlaps, verified positions) ──────────────────────
    # Figure x: 0.04 … 0.96 (0.92 available)
    # TX:      0.04  w=0.22  → ends 0.26
    # [gap 0.01]
    # Mode:    0.27  w=0.13  → ends 0.40
    # [gap 0.01]
    # Freq-:   0.41  w=0.115 → ends 0.525
    # [gap 0.01]
    # Freq+:   0.535 w=0.115 → ends 0.65
    # [gap 0.01]
    # Gain-:   0.66  w=0.115 → ends 0.775
    # [gap 0.01]
    # Gain+:   0.785 w=0.115 → ends 0.90
    # (right margin 0.10) ────────────────────────────────────────────────────

    # Group labels above the button row
    fig.text(0.150, 0.193, "TRANSMIT",          ha="center", va="bottom",
             color=C_MUT, fontsize=7)
    fig.text(0.335, 0.193, "SIGNAL MODE",        ha="center", va="bottom",
             color=C_MUT, fontsize=7)
    fig.text(0.530, 0.193, "FREQUENCY  ±100 kHz", ha="center", va="bottom",
             color=C_MUT, fontsize=7)
    fig.text(0.780, 0.193, "TX GAIN  ±3 dB",    ha="center", va="bottom",
             color=C_MUT, fontsize=7)
    # Thin separator characters (pure text, no axes → no click interception)
    fig.text(0.405, 0.110, "│", ha="center", va="center",
             color=C_BDR, fontsize=18, alpha=0.5)
    fig.text(0.655, 0.110, "│", ha="center", va="center",
             color=C_BDR, fontsize=18, alpha=0.5)

    # Button definitions: (key, label, face_color, text_color, x, width)
    _BTN_Y = 0.05
    _BTN_H = 0.11
    _btn_defs = [
        ("tx",       "▶  START TX",   "#1e3d1e", C_GRN,  0.04,  0.22),
        ("mode",     "Mode:  CW",     BG3,       C_AMB,  0.27,  0.13),
        ("freq_dn",  "◄  −100 kHz",  BG3,       C_TEXT, 0.41,  0.115),
        ("freq_up",  "+100 kHz  ►",  BG3,       C_TEXT, 0.535, 0.115),
        ("gain_dn",  "Gain  −3 dB",  BG3,       C_TEXT, 0.66,  0.115),
        ("gain_up",  "Gain  +3 dB",  BG3,       C_TEXT, 0.785, 0.115),
    ]
    btns: dict[str, Button] = {}
    for key, label, bg, fg, x0, w in _btn_defs:
        bax = fig.add_axes([x0, _BTN_Y, w, _BTN_H])
        b = Button(bax, label, color=bg, hovercolor=BG3)
        b.label.set_color(fg)
        b.label.set_fontsize(9)
        b.label.set_fontweight("bold")
        btns[key] = b

    # ── Button callbacks ──────────────────────────────────────────────────────
    def _on_tx(event):
        S.tx_on = not S.tx_on
        if S.tx_on:
            btns["tx"].label.set_text("■  STOP TX")
            btns["tx"].label.set_color(C_ROSE)
            btns["tx"].ax.set_facecolor("#3d1e1e")
        else:
            btns["tx"].label.set_text("▶  START TX")
            btns["tx"].label.set_color(C_GRN)
            btns["tx"].ax.set_facecolor("#1e3d1e")
            if not demo:
                try:
                    sdr.tx_destroy_buffer()
                except Exception:
                    pass
        fig.canvas.draw_idle()

    def _on_mode(event):
        S.mode = "ira" if S.mode == "cw" else "cw"
        btns["mode"].label.set_text(f"Mode:  {S.mode.upper()}")
        fig.canvas.draw_idle()

    def _retune(freq_hz: int, tx_gain: float) -> None:
        S.freq_hz = freq_hz
        S.tx_gain = tx_gain
        if not demo:
            try:
                sdr.tx_lo = int(freq_hz)
                sdr.rx_lo = int(freq_hz)
                sdr.tx_hardwaregain_chan0 = float(tx_gain)
            except Exception as exc:
                print(f"[RETUNE] {exc}")

    def _on_freq_dn(event):
        _retune(max(1_000_000, S.freq_hz - FREQ_STEP_HZ), S.tx_gain)

    def _on_freq_up(event):
        _retune(S.freq_hz + FREQ_STEP_HZ, S.tx_gain)

    def _on_gain_dn(event):
        # More attenuation = less TX power
        _retune(S.freq_hz, max(-89.75, S.tx_gain - GAIN_STEP_DB))

    def _on_gain_up(event):
        # Less attenuation = more TX power
        _retune(S.freq_hz, min(0.0, S.tx_gain + GAIN_STEP_DB))

    btns["tx"].on_clicked(_on_tx)
    btns["mode"].on_clicked(_on_mode)
    btns["freq_dn"].on_clicked(_on_freq_dn)
    btns["freq_up"].on_clicked(_on_freq_up)
    btns["gain_dn"].on_clicked(_on_gain_dn)
    btns["gain_up"].on_clicked(_on_gain_up)

    # ── Animation update ──────────────────────────────────────────────────────
    def _update(_):
        with S.rx_lock:
            if not S.rx_ready:
                return
            rx = S.rx_buf.copy()
            S.rx_ready = False

        with S.tx_buf_lock:
            tx = S.tx_buf.copy()

        # ── FFT + metrics ──
        fft_db       = _compute_fft(rx)
        pwr, snr     = _compute_power_snr(fft_db)
        S.fft_db     = fft_db
        S.rx_power_db = pwr
        S.snr_db     = snr
        # Scroll waterfall in-place (no allocation — np.roll would create a new array)
        S.waterfall[1:] = S.waterfall[:-1]
        S.waterfall[0]  = fft_db
        if S.tx_on:
            S.frame_count += 1

        # ── Spectrum line + fill polygon (zero alloc: mutate pre-allocated verts) ──
        spec_line.set_ydata(fft_db)
        _poly_verts[:FFT_SIZE, 1] = fft_db
        spec_poly.set_xy(_poly_verts)

        # ── Waterfall ──
        im_wfall.set_data(S.waterfall)

        # ── IQ constellation (subsample to 256 pts for speed) ──
        norm  = float(np.max(np.abs(rx)) + 1e-12)
        iq_n  = rx / norm
        step  = max(1, len(iq_n) // 256)
        iq_dots.set_data(iq_n[::step].real, iq_n[::step].imag)

        # ── RX time domain (first _DISP_SAMPLES) ──
        rx_amp = np.abs(rx[:_DISP_SAMPLES]) / (norm + 1e-12)
        n_rx   = len(rx_amp)
        if n_rx < _DISP_SAMPLES:
            rx_amp = np.pad(rx_amp, (0, _DISP_SAMPLES - n_rx))
        rxtime_line.set_ydata(rx_amp)

        # ── TX waveform preview (first _DISP_SAMPLES) ──
        tx_peak = float(np.max(np.abs(tx)) + 1e-12)
        tx_amp  = np.abs(tx[:_DISP_SAMPLES]) / tx_peak
        n_tx    = len(tx_amp)
        if n_tx < _DISP_SAMPLES:
            tx_amp = np.pad(tx_amp, (0, _DISP_SAMPLES - n_tx))
        txtime_line.set_ydata(tx_amp)

        # ── Status panel text ──
        _vals = [
            f"Freq:  {S.freq_hz / 1e6:.4f} MHz",
            f"Gain:  {S.tx_gain:+.1f} dB  (TX attenuation)",
            f"Mode:  {S.mode.upper()}",
            f"TX:    {'● ACTIVE' if S.tx_on else '○ OFF'}",
            f"RX pwr: {pwr:.1f} dBFS",
            f"SNR est: {snr:.1f} dB",
            (f"IRA frame: {S.ira_frame_count}  |  sat#47 beam#3"
             if S.mode == "ira" else
             f"Frames: {S.frame_count}  |  {TX_SAMPLE_RATE/1e6:.1f} MSPS"),
        ]
        _colors = [
            C_TEXT, C_TEXT,
            C_AMB,
            C_GRN if S.tx_on else C_ROSE,
            C_TEAL, C_TEAL,
            C_MUT,
        ]
        for t, v, c in zip(stat_texts, _vals, _colors):
            t.set_text(v)
            t.set_color(c)

    ani = FuncAnimation(   # noqa: F841
        fig, _update,
        interval=400,          # 400 ms = 2.5 fps — sufficient for monitoring
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        S.running = False
        if not demo:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="LibreSDR 868 MHz TX + RX loopback GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uri",   default=DEFAULT_URI,
                   help="IIO URI of the LibreSDR (ip:x.x.x.x | usb: | local:)")
    p.add_argument("--freq",  type=int,   default=DEFAULT_FREQ_HZ,
                   help="TX/RX centre frequency [Hz]")
    p.add_argument("--gain",  type=float, default=DEFAULT_TX_GAIN,
                   help="TX attenuation [dB]  0=max power, -89.75=min. Start at -40!")
    p.add_argument("--mode",  choices=["cw", "ira"], default="cw",
                   help="Initial TX mode: cw (continuous wave) | ira (Iridium IRA burst, pi/4-DQPSK + conv K=7)")
    p.add_argument("--demo",  action="store_true",
                   help="Run without hardware — synthetic loopback data")
    args = p.parse_args()

    print("=" * 60)
    print("  LibreSDR 868 MHz TX + RX loopback GUI")
    print("=" * 60)

    if args.demo:
        print("  [DEMO] Synthetic loopback — no hardware required.")
        sdr = None
    else:
        sdr = _connect_sdr(args.uri)
        _configure_hw(sdr, args.freq, args.gain)
        print(f"  Frequency  : {args.freq / 1e6:.3f} MHz")
        print(f"  TX Gain    : {args.gain:+.0f} dB")
        print(f"  Mode       : {args.mode.upper()}")

    S       = _make_state(args.freq, args.gain)
    S.mode  = args.mode

    tx_thread = TxThread(sdr, S, demo=args.demo)
    rx_thread = RxThread(sdr, S, demo=args.demo)
    tx_thread.start()
    rx_thread.start()

    _build_gui(S, sdr, demo=args.demo)

    S.running = False
    print("[INFO] GUI closed. Stopping threads …")
    tx_thread.join(timeout=2.0)
    rx_thread.join(timeout=2.0)
    print("Done.")


if __name__ == "__main__":
    main()
