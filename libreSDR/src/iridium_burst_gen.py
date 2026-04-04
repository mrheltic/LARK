#!/usr/bin/env python3
"""
Iridium-like burst generator using PySDR techniques.

Iridium system parameters (from public specifications):
  - Band: 1616–1626.5 MHz (L-band)
  - Channel width: ~41.667 kHz
  - Symbol rate: 25 ksps
  - Modulation: DQPSK (Differential QPSK)
  - TDMA burst structure: preamble + unique word + payload + guard

This script generates baseband DQPSK bursts with RRC pulse shaping,
plots them, and saves them as IQ files (complex64).

Usage:
  python3 iridium_burst_gen.py                  # generate and plot
  python3 iridium_burst_gen.py --save burst.iq  # save IQ file
  python3 iridium_burst_gen.py --num-bursts 5   # generate 5 bursts
  python3 iridium_burst_gen.py --no-plot         # no GUI
"""

import argparse
import numpy as np
from scipy import signal as sp_signal
import matplotlib
matplotlib.use("Agg")  # non-interactive backend by default
import matplotlib.pyplot as plt


# ── Iridium-like parameters ──────────────────────────────────────────────
SYMBOL_RATE = 25_000          # 25 ksps
SAMPLES_PER_SYMBOL = 8        # oversampling
SAMPLE_RATE = SYMBOL_RATE * SAMPLES_PER_SYMBOL  # 200 kHz
RRC_BETA = 0.35               # RRC roll-off
RRC_NUM_TAPS = 101            # RRC filter length

# Burst structure (in symbols)
PREAMBLE_LEN = 64             # preamble symbols (alternating pattern)
UNIQUE_WORD_LEN = 12          # unique word symbols (frame synchronization)
HEADER_LEN = 12               # burst header
PAYLOAD_LEN = 156             # data payload
GUARD_SYMBOLS = 20            # guard time between bursts (silence)

# Unique Word noto (12 dibit → 24 bit, pattern fisso Iridium-like)
# Pattern chosen for good autocorrelation properties
UNIQUE_WORD_BITS = np.array([
    0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
    1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0,
], dtype=np.int8)


def generate_rrc_filter(beta, sps, num_taps):
    """Generate Root Raised Cosine (RRC) filter."""
    t = np.arange(num_taps) - (num_taps - 1) // 2
    t = t / sps  # normalize to symbol period

    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + \
                  4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den

    h /= np.sqrt(np.sum(h**2))  # normalize energy
    return h


def dqpsk_modulate(bits):
    """
    DQPSK modulation: maps bit pairs into differential phase rotations.

    Dibit → phase rotation mapping:
      00 → +π/4
      01 → +3π/4
      10 → -π/4
      11 → -3π/4
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    dibits = bits.reshape(-1, 2)
    num_symbols = len(dibits)

    # Map dibit → phase rotation (Gray coding)
    phase_map = {
        (0, 0): np.pi / 4,
        (0, 1): 3 * np.pi / 4,
        (1, 0): -np.pi / 4,
        (1, 1): -3 * np.pi / 4,
    }

    # Initial reference phase
    phase = 0.0
    symbols = np.zeros(num_symbols, dtype=np.complex128)

    for i in range(num_symbols):
        dibit_key = (int(dibits[i, 0]), int(dibits[i, 1]))
        delta_phase = phase_map[dibit_key]
        phase += delta_phase
        symbols[i] = np.exp(1j * phase)

    return symbols


def generate_preamble(length):
    """
    Generate Iridium-like preamble: alternating pattern 01 01 01...
    that produces a constant phase rotation (single tone in DQPSK).
    """
    bits = np.tile([0, 1], length)  # 2 bit per simbolo
    return bits


def generate_burst(payload_bits=None, burst_type="data"):
    """
    Generate a single Iridium-like burst.

    Structure:
      [Preamble | Unique Word | Header | Payload]

    Args:
        payload_bits: payload bits (if None, generated randomly)
        burst_type: "data" (simplex data) or "ring_alert" (shorter burst)

    Returns:
        symbols: array of complex DQPSK symbols
        sections: dict with indices of each burst section
    """
    if burst_type == "ring_alert":
        preamble_len = 32   # Ring Alert has a shorter preamble
        payload_len = 48
    else:
        preamble_len = PREAMBLE_LEN
        payload_len = PAYLOAD_LEN

    # Preamble
    preamble_bits = generate_preamble(preamble_len)

    # Unique Word
    uw_bits = UNIQUE_WORD_BITS.copy()

    # Header (simplified: burst type + channel)
    header_bits = np.random.randint(0, 2, HEADER_LEN * 2).astype(np.int8)

    # Payload
    if payload_bits is None:
        payload_bits = np.random.randint(0, 2, payload_len * 2).astype(np.int8)
    else:
        payload_bits = np.array(payload_bits, dtype=np.int8)

    # Concatenate all bits
    all_bits = np.concatenate([preamble_bits, uw_bits, header_bits, payload_bits])

    # DQPSK modulate
    symbols = dqpsk_modulate(all_bits)

    sections = {
        "preamble": (0, preamble_len),
        "unique_word": (preamble_len, preamble_len + UNIQUE_WORD_LEN),
        "header": (preamble_len + UNIQUE_WORD_LEN,
                   preamble_len + UNIQUE_WORD_LEN + HEADER_LEN),
        "payload": (preamble_len + UNIQUE_WORD_LEN + HEADER_LEN,
                    len(symbols)),
    }

    return symbols, sections


def apply_pulse_shaping(symbols, rrc_filter, sps):
    """
    Apply RRC pulse shaping: upsampling + filtering.

    1. Insert zeros between symbols (upsampling)
    2. Convolve with RRC filter
    """
    # Upsampling: insert sps-1 zeros between each symbol
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols

    # Filter with RRC
    shaped = np.convolve(upsampled, rrc_filter, mode="same")
    return shaped


def add_frequency_offset(signal, freq_offset_hz, sample_rate):
    """Add a frequency offset (simulates Doppler shift or channel offset)."""
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(1j * 2 * np.pi * freq_offset_hz * t)


def generate_tdma_frame(num_bursts=4, burst_types=None, freq_offsets=None):
    """
    Generate a TDMA frame with multiple bursts, Iridium-like.

    An Iridium frame is ~90 ms and contains up to 4 timeslots.

    Args:
        num_bursts: number of bursts in the frame
        burst_types: list of burst types (default: all "data")
        freq_offsets: per-burst frequency offsets (Hz), to simulate FDMA

    Returns:
        frame_signal: complex IQ signal of the entire frame
        burst_info: list of dicts with info for each burst
    """
    if burst_types is None:
        burst_types = ["data"] * num_bursts
    if freq_offsets is None:
        freq_offsets = [0.0] * num_bursts

    rrc = generate_rrc_filter(RRC_BETA, SAMPLES_PER_SYMBOL, RRC_NUM_TAPS)

    frame_signal = np.array([], dtype=np.complex128)
    burst_info = []

    for i in range(num_bursts):
        # Generate burst
        symbols, sections = generate_burst(burst_type=burst_types[i])

        # Pulse shaping
        shaped = apply_pulse_shaping(symbols, rrc, SAMPLES_PER_SYMBOL)

        # Frequency offset (FDMA)
        if freq_offsets[i] != 0.0:
            shaped = add_frequency_offset(shaped, freq_offsets[i], SAMPLE_RATE)

        # Guard time (silence between bursts)
        guard = np.zeros(GUARD_SYMBOLS * SAMPLES_PER_SYMBOL, dtype=np.complex128)

        # Save burst info
        start_sample = len(frame_signal)
        frame_signal = np.concatenate([frame_signal, shaped, guard])

        burst_info.append({
            "index": i,
            "type": burst_types[i],
            "start_sample": start_sample,
            "end_sample": start_sample + len(shaped),
            "num_symbols": len(symbols),
            "freq_offset": freq_offsets[i],
            "sections": sections,
        })

    return frame_signal, burst_info


def add_channel_effects(signal, snr_db=20.0, phase_offset=0.0,
                        freq_drift_hz=0.0, sample_rate=SAMPLE_RATE):
    """
    Add realistic channel effects.

    Args:
        signal: IQ signal
        snr_db: signal-to-noise ratio in dB
        phase_offset: constant phase rotation (rad)
        freq_drift_hz: linear frequency drift (Hz)
    """
    out = signal.copy()

    # Phase rotation
    if phase_offset != 0.0:
        out *= np.exp(1j * phase_offset)

    # Frequency drift
    if freq_drift_hz != 0.0:
        t = np.arange(len(out)) / sample_rate
        out *= np.exp(1j * 2 * np.pi * freq_drift_hz * t)

    # AWGN
    sig_power = np.mean(np.abs(out) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(out)) + 1j * np.random.randn(len(out))
    )
    out += noise

    return out


def plot_burst(frame_signal, burst_info, snr_db=None, save_path=None):
    """Generate burst analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Iridium-like DQPSK Burst Analysis", fontsize=14, fontweight="bold")

    t_ms = np.arange(len(frame_signal)) / SAMPLE_RATE * 1000  # tempo in ms

    # 1) Amplitude over time
    ax = axes[0, 0]
    ax.plot(t_ms, np.abs(frame_signal), linewidth=0.5, color="steelblue")
    for info in burst_info:
        t_start = info["start_sample"] / SAMPLE_RATE * 1000
        t_end = info["end_sample"] / SAMPLE_RATE * 1000
        ax.axvspan(t_start, t_end, alpha=0.15, color="orange")
        ax.text((t_start + t_end) / 2, ax.get_ylim()[0],
                f"Burst {info['index']}\n({info['type']})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal envelope")
    ax.grid(True, alpha=0.3)

    # 2) Spectrum
    ax = axes[0, 1]
    nfft = 1024
    f_axis = np.linspace(-SAMPLE_RATE / 2, SAMPLE_RATE / 2, nfft) / 1000
    # Use only the first burst for the spectrum
    if len(burst_info) > 0:
        b = burst_info[0]
        burst_samples = frame_signal[b["start_sample"]:b["end_sample"]]
        if len(burst_samples) > 0:
            spectrum = np.fft.fftshift(
                np.fft.fft(burst_samples, n=nfft)
            )
            psd = 20 * np.log10(np.abs(spectrum) + 1e-12)
            psd -= np.max(psd)  # normalizza
            ax.plot(f_axis, psd, linewidth=0.8, color="darkgreen")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"Burst spectrum (BW ≈ {SYMBOL_RATE * (1 + RRC_BETA) / 1000:.1f} kHz)")
    ax.set_ylim([-60, 5])
    ax.grid(True, alpha=0.3)

    # 3) DQPSK constellation
    ax = axes[1, 0]
    # Sample at decision points (every sps samples)
    if len(burst_info) > 0:
        b = burst_info[0]
        burst_samples = frame_signal[b["start_sample"]:b["end_sample"]]
        decision_points = burst_samples[SAMPLES_PER_SYMBOL // 2::SAMPLES_PER_SYMBOL]
        # Normalize for better display
        if len(decision_points) > 0:
            decision_points /= np.max(np.abs(decision_points))
            ax.scatter(np.real(decision_points), np.imag(decision_points),
                       s=8, alpha=0.6, c="crimson", edgecolors="none")
    ax.set_xlabel("I (In-Phase)")
    ax.set_ylabel("Q (Quadrature)")
    ax.set_title("DQPSK Constellation")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2, linewidth=0.5)

    # 4) Phase over time (differential)
    ax = axes[1, 1]
    if len(burst_info) > 0:
        b = burst_info[0]
        burst_samples = frame_signal[b["start_sample"]:b["end_sample"]]
        decision_points = burst_samples[SAMPLES_PER_SYMBOL // 2::SAMPLES_PER_SYMBOL]
        if len(decision_points) > 1:
            # Differential phase
            diff_phase = np.angle(decision_points[1:] * np.conj(decision_points[:-1]))
            ax.plot(np.degrees(diff_phase), ".", markersize=3, color="purple")
            # Reference lines for the 4 DQPSK levels
            for level in [45, 135, -45, -135]:
                ax.axhline(y=level, color="gray", linestyle="--",
                           alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Symbol index")
    ax.set_ylabel("ΔΦ (degrees)")
    ax.set_title("Differential phase (DQPSK levels)")
    ax.set_ylim([-180, 180])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig("iridium_burst_plot.png", dpi=150, bbox_inches="tight")
        print("Plot saved to: iridium_burst_plot.png")


def save_iq_file(signal, filepath):
    """Save IQ signal as a complex64 file (standard SDR format)."""
    signal_32 = signal.astype(np.complex64)
    signal_32.tofile(filepath)
    duration_ms = len(signal) / SAMPLE_RATE * 1000
    print(f"IQ file saved: {filepath}")
    print(f"  Samples: {len(signal)}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Duration: {duration_ms:.1f} ms")
    print(f"  Size: {signal_32.nbytes} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Iridium-like burst generator with DQPSK modulation"
    )
    parser.add_argument("--num-bursts", type=int, default=4,
                        help="Number of bursts in the TDMA frame (default: 4)")
    parser.add_argument("--snr", type=float, default=25.0,
                        help="Channel SNR in dB (default: 25)")
    parser.add_argument("--save", type=str, default=None,
                        help="Output IQ file path (complex64)")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="PNG file path for the plot")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plot generation")
    parser.add_argument("--fdma", action="store_true",
                        help="Simulate FDMA channels (different frequency offsets)")
    parser.add_argument("--doppler", type=float, default=0.0,
                        help="Doppler drift in Hz (default: 0)")
    parser.add_argument("--ring-alert", action="store_true",
                        help="Generate Ring Alert burst type (shorter)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Iridium-like Burst Generator (DQPSK)")
    print("=" * 60)
    print(f"  Symbol rate:      {SYMBOL_RATE / 1000:.0f} ksps")
    print(f"  Sample rate:      {SAMPLE_RATE / 1000:.0f} kHz")
    print(f"  Samples/symbol:   {SAMPLES_PER_SYMBOL}")
    print(f"  RRC filter:       β={RRC_BETA}, {RRC_NUM_TAPS} taps")
    print(f"  Bursts requested: {args.num_bursts}")
    print(f"  Channel SNR:      {args.snr} dB")
    print()

    # Burst types
    if args.ring_alert:
        burst_types = ["ring_alert"] * args.num_bursts
    else:
        burst_types = ["data"] * args.num_bursts

    # FDMA offsets (adjacent channels spaced ~41.667 kHz)
    if args.fdma:
        channel_spacing = 41667  # Hz
        freq_offsets = [
            (i - args.num_bursts // 2) * channel_spacing
            for i in range(args.num_bursts)
        ]
        print(f"  FDMA mode: offset = {freq_offsets} Hz")
    else:
        freq_offsets = [0.0] * args.num_bursts

    # Generate TDMA frame
    frame_signal, burst_info = generate_tdma_frame(
        num_bursts=args.num_bursts,
        burst_types=burst_types,
        freq_offsets=freq_offsets,
    )

    # Channel effects
    frame_signal = add_channel_effects(
        frame_signal,
        snr_db=args.snr,
        phase_offset=np.random.uniform(0, 2 * np.pi),
        freq_drift_hz=args.doppler,
    )

    # Print burst info
    for info in burst_info:
        t_start = info["start_sample"] / SAMPLE_RATE * 1000
        t_end = info["end_sample"] / SAMPLE_RATE * 1000
        print(f"  Burst {info['index']}: type={info['type']}, "
              f"{info['num_symbols']} symbols, "
              f"t=[{t_start:.1f}–{t_end:.1f}] ms"
              + (f", Δf={info['freq_offset']:.0f} Hz"
                 if info['freq_offset'] != 0 else ""))
    print()

    # Plot
    if not args.no_plot:
        plot_burst(frame_signal, burst_info,
                   snr_db=args.snr, save_path=args.save_plot)

    # Save IQ file
    if args.save:
        save_iq_file(frame_signal, args.save)

    print("Done!")


if __name__ == "__main__":
    main()
