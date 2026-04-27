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
from types import SimpleNamespace

import numpy as np
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

from iridium.burst_gen import (
    SYMBOL_RATE, SAMPLES_PER_SYMBOL, SAMPLE_RATE as BURST_SAMPLE_RATE,
    RRC_BETA, RRC_NUM_TAPS,
    generate_rrc_filter, generate_burst, apply_pulse_shaping,
)

# ── Defaults (overridden by CLI args) ────────────────────────────────────────
DEFAULT_URI       = "ip:192.168.2.1"
DEFAULT_FREQ_HZ   = 868_100_000   # 868.1 MHz ISM
TX_SAMPLE_RATE    = 1_000_000     # 1 MSPS (reliable over Ethernet/USB)
TX_RF_BW          = 200_000       # 200 kHz
DEFAULT_TX_GAIN   = -40.0         # dB attenuation (start at -40; -89.75 = min power)
RX_GAIN_DB        = 30.0
RX_GAIN_MODE      = "slow_attack"

# Loopback RX buffer: grab this many samples per refresh
RX_BUF_SIZE       = 4096
WATERFALL_ROWS    = 80            # scrolling history lines
FFT_SIZE          = 512
WATERFALL_VMIN    = -80           # dB
WATERFALL_VMAX    = -10

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


def _make_burst_buf(rrc: np.ndarray) -> np.ndarray:
    """One Iridium-like π/4-DQPSK burst, resampled to TX_SAMPLE_RATE."""
    symbols, _ = generate_burst()
    bb = apply_pulse_shaping(symbols, rrc, SAMPLES_PER_SYMBOL)  # @ BURST_SAMPLE_RATE
    # Resample to TX_SAMPLE_RATE (integer ratio: both match at 1 MSPS when
    # BURST_SAMPLE_RATE == 200 kHz → need 5× upsample)
    ratio = TX_SAMPLE_RATE // BURST_SAMPLE_RATE   # e.g. 5
    if ratio != 1:
        bb = np.repeat(bb, ratio)
    # Normalise and scale to 80 % full-scale DAC
    peak = float(np.max(np.abs(bb))) + 1e-12
    bb = (bb / peak * 0.8 * (2 ** 14)).astype(np.complex64)
    return bb


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
        # Pre-build RRC filter once
        self._rrc   = generate_rrc_filter(RRC_BETA, SAMPLES_PER_SYMBOL, RRC_NUM_TAPS)

    def run(self) -> None:
        S = self._state
        last_burst = 0.0

        while S.running:
            if not S.tx_on:
                time.sleep(0.05)
                continue

            mode = S.mode   # "cw" | "burst"

            if mode == "cw":
                buf = _make_cw_buf()
            else:
                # Burst: transmit one burst every BURST_PERIOD_S, silence otherwise
                now = time.monotonic()
                if now - last_burst < BURST_PERIOD_S:
                    time.sleep(0.005)
                    continue
                buf = _make_burst_buf(self._rrc)
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

    def _demo_samples(self) -> np.ndarray:
        """Synthetic loopback: CW tone + AWGN (+ burst envelope if in burst mode)."""
        S = self._state
        n  = RX_BUF_SIZE
        t  = np.arange(n) / TX_SAMPLE_RATE
        if S.tx_on:
            noise_floor = -60.0  # dBFS
            tone_pow    = 0.3
        else:
            noise_floor = -80.0
            tone_pow    = 0.0
        snr_lin  = 10 ** ((30 - noise_floor) * 0.1)
        sigma    = 1.0 / np.sqrt(2.0 * snr_lin)
        noise    = sigma * (self._rng.standard_normal(n)
                            + 1j * self._rng.standard_normal(n)).astype(np.complex64)
        if S.mode == "burst" and S.tx_on:
            # Synthetic burst envelope: 1 ms on, 9 ms off within 10 ms window
            env   = np.zeros(n, dtype=np.float32)
            on_n  = int(TX_SAMPLE_RATE * 0.001)
            env[:min(on_n, n)] = 1.0
            tone = (env * tone_pow * np.exp(1j * 2 * np.pi * 5e3 * t)).astype(np.complex64)
        else:
            tone = (tone_pow * np.exp(1j * 2 * np.pi * 1e3 * t)).astype(np.complex64)
        return (2 ** 14) * (tone + noise)

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
        mode        = "cw",        # "cw" | "burst"
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
    freq_axis = (np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / TX_SAMPLE_RATE)) / 1e3)

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.canvas.manager.set_window_title(  # type: ignore[union-attr]
        f"LibreSDR 868 MHz TX/RX  —  {'DEMO' if demo else sdr}")

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[1.0, 1.0, 0.08],
        left=0.06, right=0.97,
        top=0.93, bottom=0.05,
        hspace=0.45, wspace=0.35,
    )

    # ── [0,0]  RX Spectrum ────────────────────────────────────────────────────
    ax_spec = fig.add_subplot(gs[0, 0], facecolor=BG2)
    ax_spec.set_facecolor(BG2)
    ax_spec.set_title("RX Spectrum", color=C_TEXT, fontsize=9)
    ax_spec.set_xlabel("Frequency offset [kHz]", color=C_MUT, fontsize=8)
    ax_spec.set_ylabel("Power [dBFS]",            color=C_MUT, fontsize=8)
    ax_spec.set_xlim(freq_axis[0], freq_axis[-1])
    ax_spec.set_ylim(WATERFALL_VMIN, 5)
    ax_spec.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_spec.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_spec.grid(color=C_BDR, lw=0.4, alpha=0.5)
    spec_line, = ax_spec.plot(freq_axis, S.fft_db, "-", color=C_TEAL, lw=1.0)
    spec_fill  = ax_spec.fill_between(freq_axis, WATERFALL_VMIN, S.fft_db,
                                       color=C_TEAL, alpha=0.12)
    ax_spec.axhline(WATERFALL_VMIN + 5, color=C_BDR, lw=0.6, ls="--")

    # ── [0,1]  Waterfall ──────────────────────────────────────────────────────
    ax_wfall = fig.add_subplot(gs[0, 1], facecolor=BG2)
    ax_wfall.set_facecolor(BG2)
    ax_wfall.set_title("Waterfall (time ↓)", color=C_TEXT, fontsize=9)
    ax_wfall.set_xlabel("Frequency offset [kHz]", color=C_MUT, fontsize=8)
    ax_wfall.set_ylabel("Time (older ↓)",          color=C_MUT, fontsize=8)
    ax_wfall.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_wfall.spines.values():
        sp.set_edgecolor(C_BDR)
    im_wfall = ax_wfall.imshow(
        S.waterfall,
        origin="upper",
        aspect="auto",
        extent=[freq_axis[0], freq_axis[-1], WATERFALL_ROWS, 0],
        cmap="inferno",
        vmin=WATERFALL_VMIN, vmax=WATERFALL_VMAX,
        interpolation="bilinear",
    )

    # ── [0,2]  IQ Constellation ───────────────────────────────────────────────
    ax_iq = fig.add_subplot(gs[0, 2], facecolor=BG2)
    ax_iq.set_facecolor(BG2)
    ax_iq.set_title("IQ Constellation (RX)", color=C_TEXT, fontsize=9)
    ax_iq.set_xlabel("I", color=C_MUT, fontsize=8)
    ax_iq.set_ylabel("Q", color=C_MUT, fontsize=8)
    _iq_lim = 1.1
    ax_iq.set_xlim(-_iq_lim, _iq_lim)
    ax_iq.set_ylim(-_iq_lim, _iq_lim)
    ax_iq.set_aspect("equal")
    ax_iq.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_iq.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_iq.grid(color=C_BDR, lw=0.4, alpha=0.5)
    ax_iq.axhline(0, color=C_BDR, lw=0.6)
    ax_iq.axvline(0, color=C_BDR, lw=0.6)
    iq_dots, = ax_iq.plot([], [], ".", color=C_VIO, ms=1.5, alpha=0.4)

    # ── [1,0]  RX Time domain ─────────────────────────────────────────────────
    ax_rxtime = fig.add_subplot(gs[1, 0], facecolor=BG2)
    ax_rxtime.set_facecolor(BG2)
    ax_rxtime.set_title("RX amplitude (time)", color=C_TEXT, fontsize=9)
    ax_rxtime.set_xlabel("Sample", color=C_MUT, fontsize=8)
    ax_rxtime.set_ylabel("|IQ| (normalised)",  color=C_MUT, fontsize=8)
    ax_rxtime.set_ylim(-0.05, 1.2)
    ax_rxtime.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_rxtime.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_rxtime.grid(color=C_BDR, lw=0.4, alpha=0.5)
    _n_disp    = RX_BUF_SIZE
    rx_xs      = np.arange(_n_disp)
    rxtime_line, = ax_rxtime.plot(rx_xs, np.zeros(_n_disp), "-",
                                   color=C_BLUE, lw=0.8)
    ax_rxtime.set_xlim(0, _n_disp)

    # ── [1,1]  TX waveform ────────────────────────────────────────────────────
    ax_txtime = fig.add_subplot(gs[1, 1], facecolor=BG2)
    ax_txtime.set_facecolor(BG2)
    ax_txtime.set_title("TX waveform (last buffer)", color=C_TEXT, fontsize=9)
    ax_txtime.set_xlabel("Sample", color=C_MUT, fontsize=8)
    ax_txtime.set_ylabel("|IQ| (normalised)",       color=C_MUT, fontsize=8)
    ax_txtime.set_ylim(-0.05, 1.2)
    ax_txtime.tick_params(colors=C_MUT, labelsize=7)
    for sp in ax_txtime.spines.values():
        sp.set_edgecolor(C_BDR)
    ax_txtime.grid(color=C_BDR, lw=0.4, alpha=0.5)
    _tx_disp  = 512
    txtime_xs = np.arange(_tx_disp)
    txtime_line, = ax_txtime.plot(txtime_xs, np.zeros(_tx_disp), "-",
                                   color=C_AMB, lw=0.8)
    ax_txtime.set_xlim(0, _tx_disp)

    # ── [1,2]  Status panel ───────────────────────────────────────────────────
    ax_stat = fig.add_subplot(gs[1, 2], facecolor=BG2)
    ax_stat.set_facecolor(BG2)
    ax_stat.set_title("Status", color=C_TEXT, fontsize=9)
    ax_stat.axis("off")
    _stat_items = [
        ("Frequency",   "",  0.90, C_TEXT),
        ("TX Gain",     "",  0.78, C_TEXT),
        ("Mode",        "",  0.66, C_TEXT),
        ("TX",          "",  0.54, C_GRN),
        ("RX Power",    "",  0.42, C_TEAL),
        ("SNR",         "",  0.30, C_TEAL),
        ("Frames TX",   "",  0.18, C_MUT),
        ("Sample rate", "",  0.06, C_MUT),
    ]
    stat_texts: list = []
    for label, _, y, col in _stat_items:
        t = ax_stat.text(0.08, y, f"{label}: —", transform=ax_stat.transAxes,
                          color=col, fontsize=9, va="top",
                          fontfamily="monospace")
        stat_texts.append(t)

    # ── [2,*]  Control buttons ─────────────────────────────────────────────────
    btn_axs = [fig.add_subplot(gs[2, i]) for i in range(3)]
    # Put 6 buttons across row 2 by splitting each cell in 2
    _btn_y, _btn_h = 0.015, 0.04
    _btn_rows = [(0.06, 0.15), (0.21, 0.15), (0.37, 0.15),
                 (0.53, 0.15), (0.69, 0.15), (0.85, 0.10)]
    btns: dict[str, Button] = {}
    _btn_defs = [
        ("tx",       "TX OFF",   C_ROSE,  C_TEXT),
        ("mode",     "CW",       C_BDR,   C_TEXT),
        ("freq_up",  "Freq ▲",   BG3,     C_TEXT),
        ("freq_dn",  "Freq ▼",   BG3,     C_TEXT),
        ("gain_up",  "Gain ▲",   BG3,     C_TEXT),
        ("gain_dn",  "Gain ▼",   BG3,     C_TEXT),
    ]
    for (key, label, bg, fg), (x0, w) in zip(_btn_defs, _btn_rows):
        bax = fig.add_axes([x0, _btn_y, w, _btn_h])
        bax.set_facecolor(BG3)
        b = Button(bax, label, color=bg, hovercolor=BG2)
        b.label.set_color(fg)
        b.label.set_fontsize(8)
        btns[key] = b

    # ── Suptitle ──────────────────────────────────────────────────────────────
    title_txt = fig.suptitle(
        f"LibreSDR  AD9363  —  TX 868 MHz loopback  |  "
        f"{'DEMO MODE' if demo else f'URI: ip:…'}",
        color=C_TEXT, fontsize=9, y=0.975,
    )

    # ── Button callbacks ──────────────────────────────────────────────────────
    _btn_clicked = [False]   # lock to avoid re-entrancy from FuncAnimation

    def _on_tx(event):
        S.tx_on = not S.tx_on
        if S.tx_on:
            btns["tx"].label.set_text("TX ON")
            btns["tx"].ax.set_facecolor("#2a4a2a")
        else:
            btns["tx"].label.set_text("TX OFF")
            btns["tx"].ax.set_facecolor(C_ROSE)
            try:
                if not demo:
                    sdr.tx_destroy_buffer()
            except Exception:
                pass
        fig.canvas.draw_idle()

    def _on_mode(event):
        S.mode = "burst" if S.mode == "cw" else "cw"
        btns["mode"].label.set_text(S.mode.upper())
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

    def _on_freq_up(event):
        _retune(S.freq_hz + FREQ_STEP_HZ, S.tx_gain)

    def _on_freq_dn(event):
        _retune(max(1_000_000, S.freq_hz - FREQ_STEP_HZ), S.tx_gain)

    def _on_gain_up(event):
        # Gain = attenuation in dB: lower value = more power
        _retune(S.freq_hz, min(0.0, S.tx_gain + GAIN_STEP_DB))

    def _on_gain_dn(event):
        _retune(S.freq_hz, max(-89.75, S.tx_gain - GAIN_STEP_DB))

    btns["tx"].on_clicked(_on_tx)
    btns["mode"].on_clicked(_on_mode)
    btns["freq_up"].on_clicked(_on_freq_up)
    btns["freq_dn"].on_clicked(_on_freq_dn)
    btns["gain_up"].on_clicked(_on_gain_up)
    btns["gain_dn"].on_clicked(_on_gain_dn)

    # ── Animation update ──────────────────────────────────────────────────────
    # Keep track of a resizable polyCollection for fill_between (re-create each frame)
    _fill_state = [spec_fill]

    def _update(_):
        with S.rx_lock:
            if not S.rx_ready:
                return
            rx  = S.rx_buf.copy()
            S.rx_ready = False

        with S.tx_buf_lock:
            tx = S.tx_buf.copy()

        # FFT
        fft_db  = _compute_fft(rx)
        pwr, snr = _compute_power_snr(fft_db)

        S.fft_db      = fft_db
        S.rx_power_db = pwr
        S.snr_db      = snr
        S.waterfall   = np.roll(S.waterfall, 1, axis=0)
        S.waterfall[0, :] = fft_db
        if S.tx_on:
            S.frame_count += 1

        # ── Spectrum ──
        spec_line.set_ydata(fft_db)
        _fill_state[0].remove()
        _fill_state[0] = ax_spec.fill_between(freq_axis, WATERFALL_VMIN, fft_db,
                                               color=C_TEAL, alpha=0.12)

        # ── Waterfall ──
        im_wfall.set_data(S.waterfall)

        # ── IQ constellation ──
        norm = float(np.max(np.abs(rx)) + 1e-12)
        iq_n = rx / norm
        sub  = iq_n[::max(1, len(iq_n) // 512)]
        iq_dots.set_data(sub.real, sub.imag)

        # ── RX time domain ──
        rx_amp = np.abs(rx) / (norm + 1e-12)
        n_rx   = min(len(rx_amp), _n_disp)
        rxtime_line.set_ydata(np.pad(rx_amp[:n_rx], (0, _n_disp - n_rx)))

        # ── TX waveform ──
        tx_amp = np.abs(tx) / (float(np.max(np.abs(tx)) + 1e-12))
        n_tx   = min(len(tx_amp), _tx_disp)
        txtime_line.set_ydata(np.pad(tx_amp[:n_tx], (0, _tx_disp - n_tx)))
        txtime_xs[:] = np.arange(_tx_disp)

        # ── Status text ──
        tx_str  = "ACTIVE" if S.tx_on else "OFF"
        _vals = [
            f"Frequency:   {S.freq_hz / 1e6:.3f} MHz",
            f"TX Gain:     {S.tx_gain:+.1f} dB",
            f"Mode:        {S.mode.upper()}",
            f"TX:          {tx_str}",
            f"RX Power:    {pwr:.1f} dBFS",
            f"SNR est.:    {snr:.1f} dB",
            f"Frames TX:   {S.frame_count}",
            f"Sample rate: {TX_SAMPLE_RATE/1e6:.1f} MSPS",
        ]
        _colors = [C_TEXT, C_TEXT, C_TEXT,
                   C_GRN if S.tx_on else C_ROSE,
                   C_TEAL, C_TEAL, C_MUT, C_MUT]
        for t, v, c in zip(stat_texts, _vals, _colors):
            t.set_text(v)
            t.set_color(c)

        # ── Title ──
        title_txt.set_text(
            f"LibreSDR  AD9363  |  {S.mode.upper()}  |  "
            f"{S.freq_hz/1e6:.3f} MHz  |  TX Gain {S.tx_gain:+.0f} dB  |  "
            + ("DEMO" if demo else "LIVE")
        )

    ani = FuncAnimation(   # noqa: F841
        fig, _update,
        interval=200,
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
    p.add_argument("--mode",  choices=["cw", "burst"], default="cw",
                   help="Initial TX mode: cw (continuous wave) | burst (pi/4-DQPSK)")
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
