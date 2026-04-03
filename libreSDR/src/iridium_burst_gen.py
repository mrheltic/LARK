#!/usr/bin/env python3
"""
Generatore di burst simili a Iridium usando tecniche PySDR.

Parametri del sistema Iridium (da specifiche pubbliche):
  - Banda: 1616–1626.5 MHz (L-band)
  - Larghezza canale: ~41.667 kHz
  - Symbol rate: 25 ksps
  - Modulazione: DQPSK (Differential QPSK)
  - Struttura burst TDMA: preamble + unique word + payload + guard

Questo script genera burst DQPSK in banda base con pulse shaping RRC,
li visualizza e salva come file IQ (complex64).

Uso:
  python3 iridium_burst_gen.py                  # genera e plotta
  python3 iridium_burst_gen.py --save burst.iq  # salva file IQ
  python3 iridium_burst_gen.py --num-bursts 5   # genera 5 burst
  python3 iridium_burst_gen.py --no-plot         # niente GUI
"""

import argparse
import numpy as np
from scipy import signal as sp_signal
import matplotlib
matplotlib.use("Agg")  # backend non-interattivo di default
import matplotlib.pyplot as plt


# ── Parametri Iridium-like ──────────────────────────────────────────────
SYMBOL_RATE = 25_000          # 25 ksps
SAMPLES_PER_SYMBOL = 8        # oversampling
SAMPLE_RATE = SYMBOL_RATE * SAMPLES_PER_SYMBOL  # 200 kHz
RRC_BETA = 0.35               # roll-off RRC
RRC_NUM_TAPS = 101            # lunghezza filtro RRC

# Struttura burst (in simboli)
PREAMBLE_LEN = 64             # simboli di preambolo (pattern alternante)
UNIQUE_WORD_LEN = 12          # simboli di unique word (sincronizzazione frame)
HEADER_LEN = 12               # header burst
PAYLOAD_LEN = 156             # payload dati
GUARD_SYMBOLS = 20            # guard time tra burst (silenzio)

# Unique Word noto (12 dibit → 24 bit, pattern fisso Iridium-like)
# Pattern scelto per buone proprietà di autocorrelazione
UNIQUE_WORD_BITS = np.array([
    0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
    1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0,
], dtype=np.int8)


def generate_rrc_filter(beta, sps, num_taps):
    """Genera filtro Root Raised Cosine (RRC)."""
    t = np.arange(num_taps) - (num_taps - 1) // 2
    t = t / sps  # normalizza al periodo di simbolo

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

    h /= np.sqrt(np.sum(h**2))  # normalizza energia
    return h


def dqpsk_modulate(bits):
    """
    Modulazione DQPSK: mappa coppie di bit in rotazioni di fase differenziali.

    Mapping dibit → rotazione di fase:
      00 → +π/4
      01 → +3π/4
      10 → -π/4
      11 → -3π/4
    """
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)

    dibits = bits.reshape(-1, 2)
    num_symbols = len(dibits)

    # Mappa dibit → rotazione di fase (Gray coding)
    phase_map = {
        (0, 0): np.pi / 4,
        (0, 1): 3 * np.pi / 4,
        (1, 0): -np.pi / 4,
        (1, 1): -3 * np.pi / 4,
    }

    # Fase iniziale di riferimento
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
    Genera preambolo Iridium-like: pattern alternante 01 01 01...
    che produce una rotazione di fase costante (tono singolo in DQPSK).
    """
    bits = np.tile([0, 1], length)  # 2 bit per simbolo
    return bits


def generate_burst(payload_bits=None, burst_type="data"):
    """
    Genera un singolo burst Iridium-like.

    Struttura:
      [Preamble | Unique Word | Header | Payload]

    Args:
        payload_bits: bit del payload (se None, generati random)
        burst_type: "data" (simplex data) o "ring_alert" (burst corto)

    Returns:
        symbols: array di simboli DQPSK complessi
        sections: dict con indici delle sezioni del burst
    """
    if burst_type == "ring_alert":
        preamble_len = 32   # Ring Alert ha preambolo più corto
        payload_len = 48
    else:
        preamble_len = PREAMBLE_LEN
        payload_len = PAYLOAD_LEN

    # Preambolo
    preamble_bits = generate_preamble(preamble_len)

    # Unique Word
    uw_bits = UNIQUE_WORD_BITS.copy()

    # Header (semplificato: tipo burst + canale)
    header_bits = np.random.randint(0, 2, HEADER_LEN * 2).astype(np.int8)

    # Payload
    if payload_bits is None:
        payload_bits = np.random.randint(0, 2, payload_len * 2).astype(np.int8)
    else:
        payload_bits = np.array(payload_bits, dtype=np.int8)

    # Concatena tutti i bit
    all_bits = np.concatenate([preamble_bits, uw_bits, header_bits, payload_bits])

    # Modula DQPSK
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
    Applica pulse shaping RRC: upsampling + filtro.

    1. Inserisce zeri tra i simboli (upsampling)
    2. Convolve con filtro RRC
    """
    # Upsampling: inserisci sps-1 zeri tra ogni simbolo
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols

    # Filtra con RRC
    shaped = np.convolve(upsampled, rrc_filter, mode="same")
    return shaped


def add_frequency_offset(signal, freq_offset_hz, sample_rate):
    """Aggiunge un offset di frequenza (simula spostamento Doppler o canale)."""
    t = np.arange(len(signal)) / sample_rate
    return signal * np.exp(1j * 2 * np.pi * freq_offset_hz * t)


def generate_tdma_frame(num_bursts=4, burst_types=None, freq_offsets=None):
    """
    Genera un frame TDMA con più burst, simile a Iridium.

    Un frame Iridium è ~90 ms e contiene fino a 4 timeslot.

    Args:
        num_bursts: numero di burst nel frame
        burst_types: lista di tipi burst (default: tutti "data")
        freq_offsets: offset di frequenza per burst (Hz), per simulare FDMA

    Returns:
        frame_signal: segnale IQ complesso del frame intero
        burst_info: lista di dizionari con info per ogni burst
    """
    if burst_types is None:
        burst_types = ["data"] * num_bursts
    if freq_offsets is None:
        freq_offsets = [0.0] * num_bursts

    rrc = generate_rrc_filter(RRC_BETA, SAMPLES_PER_SYMBOL, RRC_NUM_TAPS)

    frame_signal = np.array([], dtype=np.complex128)
    burst_info = []

    for i in range(num_bursts):
        # Genera burst
        symbols, sections = generate_burst(burst_type=burst_types[i])

        # Pulse shaping
        shaped = apply_pulse_shaping(symbols, rrc, SAMPLES_PER_SYMBOL)

        # Offset di frequenza (FDMA)
        if freq_offsets[i] != 0.0:
            shaped = add_frequency_offset(shaped, freq_offsets[i], SAMPLE_RATE)

        # Guard time (silenzio tra burst)
        guard = np.zeros(GUARD_SYMBOLS * SAMPLES_PER_SYMBOL, dtype=np.complex128)

        # Salva info burst
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
    Aggiunge effetti di canale realistici.

    Args:
        signal: segnale IQ
        snr_db: rapporto segnale-rumore in dB
        phase_offset: rotazione di fase costante (rad)
        freq_drift_hz: drift di frequenza lineare (Hz)
    """
    out = signal.copy()

    # Rotazione di fase
    if phase_offset != 0.0:
        out *= np.exp(1j * phase_offset)

    # Drift di frequenza
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
    """Genera plot di analisi del burst."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Iridium-like DQPSK Burst Analysis", fontsize=14, fontweight="bold")

    t_ms = np.arange(len(frame_signal)) / SAMPLE_RATE * 1000  # tempo in ms

    # 1) Ampiezza nel tempo
    ax = axes[0, 0]
    ax.plot(t_ms, np.abs(frame_signal), linewidth=0.5, color="steelblue")
    for info in burst_info:
        t_start = info["start_sample"] / SAMPLE_RATE * 1000
        t_end = info["end_sample"] / SAMPLE_RATE * 1000
        ax.axvspan(t_start, t_end, alpha=0.15, color="orange")
        ax.text((t_start + t_end) / 2, ax.get_ylim()[0],
                f"Burst {info['index']}\n({info['type']})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Tempo (ms)")
    ax.set_ylabel("Ampiezza")
    ax.set_title("Inviluppo del segnale")
    ax.grid(True, alpha=0.3)

    # 2) Spettro
    ax = axes[0, 1]
    nfft = 1024
    f_axis = np.linspace(-SAMPLE_RATE / 2, SAMPLE_RATE / 2, nfft) / 1000
    # Usa solo il primo burst per lo spettro
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
    ax.set_xlabel("Frequenza (kHz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"Spettro burst (BW ≈ {SYMBOL_RATE * (1 + RRC_BETA) / 1000:.1f} kHz)")
    ax.set_ylim([-60, 5])
    ax.grid(True, alpha=0.3)

    # 3) Costellazione DQPSK
    ax = axes[1, 0]
    # Campiona ai punti di decisione (ogni sps campioni)
    if len(burst_info) > 0:
        b = burst_info[0]
        burst_samples = frame_signal[b["start_sample"]:b["end_sample"]]
        decision_points = burst_samples[SAMPLES_PER_SYMBOL // 2::SAMPLES_PER_SYMBOL]
        # Normalizza per visualizzare meglio
        if len(decision_points) > 0:
            decision_points /= np.max(np.abs(decision_points))
            ax.scatter(np.real(decision_points), np.imag(decision_points),
                       s=8, alpha=0.6, c="crimson", edgecolors="none")
    ax.set_xlabel("I (In-Phase)")
    ax.set_ylabel("Q (Quadrature)")
    ax.set_title("Costellazione DQPSK")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # Cerchio unitario
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2, linewidth=0.5)

    # 4) Fase nel tempo (differenziale)
    ax = axes[1, 1]
    if len(burst_info) > 0:
        b = burst_info[0]
        burst_samples = frame_signal[b["start_sample"]:b["end_sample"]]
        decision_points = burst_samples[SAMPLES_PER_SYMBOL // 2::SAMPLES_PER_SYMBOL]
        if len(decision_points) > 1:
            # Fase differenziale
            diff_phase = np.angle(decision_points[1:] * np.conj(decision_points[:-1]))
            ax.plot(np.degrees(diff_phase), ".", markersize=3, color="purple")
            # Linee di riferimento per i 4 livelli DQPSK
            for level in [45, 135, -45, -135]:
                ax.axhline(y=level, color="gray", linestyle="--",
                           alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Indice simbolo")
    ax.set_ylabel("ΔΦ (gradi)")
    ax.set_title("Fase differenziale (livelli DQPSK)")
    ax.set_ylim([-180, 180])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot salvato in: {save_path}")
    else:
        plt.savefig("iridium_burst_plot.png", dpi=150, bbox_inches="tight")
        print("Plot salvato in: iridium_burst_plot.png")


def save_iq_file(signal, filepath):
    """Salva segnale IQ come file complex64 (formato SDR standard)."""
    signal_32 = signal.astype(np.complex64)
    signal_32.tofile(filepath)
    duration_ms = len(signal) / SAMPLE_RATE * 1000
    print(f"File IQ salvato: {filepath}")
    print(f"  Campioni: {len(signal)}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Durata: {duration_ms:.1f} ms")
    print(f"  Dimensione: {signal_32.nbytes} byte")


def main():
    parser = argparse.ArgumentParser(
        description="Generatore di burst Iridium-like con modulazione DQPSK"
    )
    parser.add_argument("--num-bursts", type=int, default=4,
                        help="Numero di burst nel frame TDMA (default: 4)")
    parser.add_argument("--snr", type=float, default=25.0,
                        help="SNR del canale in dB (default: 25)")
    parser.add_argument("--save", type=str, default=None,
                        help="Percorso file IQ output (complex64)")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Percorso file PNG per il plot")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disabilita generazione plot")
    parser.add_argument("--fdma", action="store_true",
                        help="Simula canali FDMA (offset di frequenza diversi)")
    parser.add_argument("--doppler", type=float, default=0.0,
                        help="Drift Doppler in Hz (default: 0)")
    parser.add_argument("--ring-alert", action="store_true",
                        help="Genera burst di tipo Ring Alert (più corti)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Generatore Burst Iridium-like (DQPSK)")
    print("=" * 60)
    print(f"  Symbol rate:     {SYMBOL_RATE / 1000:.0f} ksps")
    print(f"  Sample rate:     {SAMPLE_RATE / 1000:.0f} kHz")
    print(f"  Campioni/simbolo: {SAMPLES_PER_SYMBOL}")
    print(f"  Filtro RRC:      β={RRC_BETA}, {RRC_NUM_TAPS} tap")
    print(f"  Burst richiesti: {args.num_bursts}")
    print(f"  SNR canale:      {args.snr} dB")
    print()

    # Tipi di burst
    if args.ring_alert:
        burst_types = ["ring_alert"] * args.num_bursts
    else:
        burst_types = ["data"] * args.num_bursts

    # Offset FDMA (canali adiacenti spaziati ~41.667 kHz)
    if args.fdma:
        channel_spacing = 41667  # Hz
        freq_offsets = [
            (i - args.num_bursts // 2) * channel_spacing
            for i in range(args.num_bursts)
        ]
        print(f"  Modalità FDMA: offset = {freq_offsets} Hz")
    else:
        freq_offsets = [0.0] * args.num_bursts

    # Genera frame TDMA
    frame_signal, burst_info = generate_tdma_frame(
        num_bursts=args.num_bursts,
        burst_types=burst_types,
        freq_offsets=freq_offsets,
    )

    # Effetti di canale
    frame_signal = add_channel_effects(
        frame_signal,
        snr_db=args.snr,
        phase_offset=np.random.uniform(0, 2 * np.pi),
        freq_drift_hz=args.doppler,
    )

    # Stampa info burst
    for info in burst_info:
        t_start = info["start_sample"] / SAMPLE_RATE * 1000
        t_end = info["end_sample"] / SAMPLE_RATE * 1000
        print(f"  Burst {info['index']}: tipo={info['type']}, "
              f"{info['num_symbols']} simboli, "
              f"t=[{t_start:.1f}–{t_end:.1f}] ms"
              + (f", Δf={info['freq_offset']:.0f} Hz"
                 if info['freq_offset'] != 0 else ""))
    print()

    # Plot
    if not args.no_plot:
        plot_burst(frame_signal, burst_info,
                   snr_db=args.snr, save_path=args.save_plot)

    # Salva file IQ
    if args.save:
        save_iq_file(frame_signal, args.save)

    print("Fatto!")


if __name__ == "__main__":
    main()
