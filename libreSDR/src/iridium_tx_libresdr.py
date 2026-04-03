#!/usr/bin/env python3
"""
iridium_tx_libresdr.py — Trasmette burst Iridium-like via LibreSDR (Zynq7020 + AD9363)

Usa pyadi-iio (stessa API del PlutoSDR PySDR) per configurare l'AD9363 e
trasmettere i burst DQPSK generati da iridium_burst_gen.py.

Hardware: LibreSDR (Zynq7020 + AD9363) @ ip:192.168.1.10

Dipendenze:
  pip install pyadi-iio           # oppure: pip install adi
  sudo apt install python3-libiio  # libiio backend

Uso:
  python3 scripts/iridium_tx_libresdr.py
  python3 scripts/iridium_tx_libresdr.py --freq 433920000 --gain -40
  python3 scripts/iridium_tx_libresdr.py --uri ip:192.168.1.10 --cyclic
  python3 scripts/iridium_tx_libresdr.py --num-bursts 8 --no-channel-effects

ATTENZIONE: trasmettere a 1616-1626 MHz (banda Iridium) senza licenza è illegale.
            Usare sempre una frequenza consentita (es. banda ISM, banco di prova RF)
            o collegare TX → RX con un cavo + attenuatore.
"""

import argparse
import sys
import os
import time
import numpy as np
from scipy import signal as sp_signal

# Importa il generatore di burst dallo stesso package scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iridium_burst_gen import (
    generate_tdma_frame,
    add_channel_effects,
    SAMPLE_RATE as BURST_SAMPLE_RATE,   # 200 kHz
    SYMBOL_RATE,
    SAMPLES_PER_SYMBOL,
    RRC_BETA,
)

# ── Parametri hardware ────────────────────────────────────────────────────────
DEFAULT_URI = "ip:192.168.1.10"
DEFAULT_CENTER_FREQ = 1_615_937_500    # canale IRA Iridium: 1615604164 + 8×41667 Hz
TX_SAMPLE_RATE = 1_000_000             # 1 MSPS — minimo pratico per AD9363 su Ethernet
TX_RF_BANDWIDTH = 200_000              # 200 kHz — uguale alla banda del segnale
DEFAULT_TX_GAIN_DB = -60               # Attenuazione TX (range: -90 … 0 dB)
                                        # Iniziare basso e aumentare se necessario!
# Fattore di upsampling: da BURST_SAMPLE_RATE (200 kHz) a TX_SAMPLE_RATE (1 MHz)
UPSAMPLE_NUM = TX_SAMPLE_RATE          # numeratore
UPSAMPLE_DEN = BURST_SAMPLE_RATE       # denominatore
# scipy.resample_poly vuole interi ridotti ai minimi termini
from math import gcd as _gcd
_g = _gcd(UPSAMPLE_NUM, UPSAMPLE_DEN)
_UP = UPSAMPLE_NUM // _g               # 5
_DOWN = UPSAMPLE_DEN // _g             # 1


def resample_to_hw(baseband_iq: np.ndarray) -> np.ndarray:
    """
    Ricampiona il segnale da BURST_SAMPLE_RATE (200 kHz) a TX_SAMPLE_RATE (1 MHz).
    Usa resample_poly che preserva la fase e non introduce aliasing.
    """
    return sp_signal.resample_poly(baseband_iq, _UP, _DOWN).astype(np.complex64)


def scale_for_dac(iq: np.ndarray) -> np.ndarray:
    """
    Scala i campioni IQ nel range atteso dall'AD9363 via pyadi-iio.
    PyaDI-IIO si aspetta |sample| ≤ 2^14 (non ±1 come i nostri campioni generati).
    """
    # Normalizza a ±0.9 poi scala a 2^14
    max_amp = np.max(np.abs(iq))
    if max_amp > 0:
        iq = iq / max_amp * 0.9
    iq *= 2 ** 14
    return iq


def connect_libresdr(uri: str):
    """
    Connette al LibreSDR via pyadi-iio e restituisce l'oggetto sdr.
    Il LibreSDR usa firmware Pluto-like, quindi usa adi.Pluto (o adi.ad9364).
    """
    try:
        import adi
    except ImportError:
        print("[ERRORE] pyadi-iio non installato.")
        print("  Installa con:  pip install pyadi-iio")
        print("  Oppure:        pip install adi")
        sys.exit(1)

    print(f"  Connessione al LibreSDR: {uri} ...", end=" ", flush=True)
    import time as _time
    for attempt in range(3):
        try:
            # Il firmware LibreSDR/AntSDR E200 è compatibile con l'API Pluto
            sdr = adi.Pluto(uri)
            # Cleanup preventivo: distruggi eventuale buffer residuo da sessioni precedenti
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            print("OK")
            return sdr
        except OSError as e:
            if e.errno == 16 and attempt < 2:  # EBUSY: device or resource busy
                print(f"occupato (tentativo {attempt+1}/3), attendo 3s ...",
                      end=" ", flush=True)
                _time.sleep(3)
            else:
                print(f"FALLITA\n[ERRORE] {e}")
                print("\nVerifica:")
                print("  1. LibreSDR alimentato e raggiungibile:")
                host = uri.split(":", 1)[-1] if ":" in uri else uri
                print(f"     ping {host}")
                print("  2. Nessun altro processo usa il device:")
                print(f"     pkill -9 -f iridium")
                print("  3. iio_info -u " + uri + "  (mostra dispositivi IIO)")
                sys.exit(1)
        except Exception as e:
            print(f"FALLITA\n[ERRORE] {e}")
            sys.exit(1)


def configure_tx(sdr, center_freq_hz: int, gain_db: float):
    """
    Configura il canale TX dell'AD9363.

    Args:
        sdr: oggetto pyadi-iio
        center_freq_hz: frequenza portante TX in Hz
        gain_db: attenuazione TX in dB (range: -90 … 0 dB; valori più bassi = meno potenza)
    """
    sdr.sample_rate = int(TX_SAMPLE_RATE)
    sdr.tx_rf_bandwidth = int(TX_RF_BANDWIDTH)
    sdr.tx_lo = int(center_freq_hz)
    sdr.tx_hardwaregain_chan0 = float(gain_db)

    print(f"  TX configurato:")
    print(f"    Centro:       {center_freq_hz / 1e6:.3f} MHz")
    print(f"    Sample rate:  {TX_SAMPLE_RATE / 1e6:.1f} MSPS")
    print(f"    RF bandwidth: {TX_RF_BANDWIDTH / 1e3:.0f} kHz")
    print(f"    Gain TX:      {gain_db:+.0f} dB")


def transmit_once(sdr, samples: np.ndarray, num_repetitions: int = 1):
    """
    Trasmette una sequenza di campioni N volte (non ciclico).
    Utile per inviare un burst o una serie limitata di frame.
    """
    for i in range(num_repetitions):
        sdr.tx(samples)
        if num_repetitions > 1:
            print(f"  Trasmesso frame {i + 1}/{num_repetitions}")


def transmit_cyclic(sdr, samples: np.ndarray):
    """
    Trasmette in loop continuo (cyclic buffer) fino a Ctrl+C.
    L'hardware ripete autonomamente il buffer senza overhead Python.
    """
    # Cleanup preventivo: se era rimasto un buffer aperto da sessione precedente
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx_cyclic_buffer = True
    sdr.tx(samples)
    print("  Trasmissione ciclica avviata. Premi Ctrl+C per fermare.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  Stop richiesto.")
    finally:
        sdr.tx_destroy_buffer()
        print("  Buffer TX distrutto.")


def main():
    parser = argparse.ArgumentParser(
        description="Trasmette burst Iridium-like via LibreSDR (AD9363) con pyadi-iio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", default=DEFAULT_URI,
                        help="URI IIO del LibreSDR")
    parser.add_argument("--freq", type=int, default=DEFAULT_CENTER_FREQ,
                        help="Frequenza portante TX in Hz")
    parser.add_argument("--gain", type=float, default=DEFAULT_TX_GAIN_DB,
                        help="Gain TX in dB (range -90…0). Iniziare basso!")
    parser.add_argument("--num-bursts", type=int, default=4,
                        help="Numero di burst nel frame TDMA")
    parser.add_argument("--cyclic", action="store_true",
                        help="Trasmissione continua (ciclica) fino a Ctrl+C")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Ripetizioni del frame (senza --cyclic)")
    parser.add_argument("--fdma", action="store_true",
                        help="Simula canali FDMA (offset di freq per ogni burst)")
    parser.add_argument("--ring-alert", action="store_true",
                        help="Genera burst Ring Alert (più corti)")
    parser.add_argument("--no-channel-effects", action="store_true",
                        help="Disabilita rumore/offset AWGN (segnale ideale)")
    parser.add_argument("--snr", type=float, default=40.0,
                        help="SNR simulato in dB (ignorato con --no-channel-effects)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Genera e mostra info, ma NON trasmette")
    args = parser.parse_args()

    print("=" * 60)
    print("  Iridium-like burst TX — LibreSDR (AD9363)")
    print("=" * 60)

    # ── Genera il frame IQ ──────────────────────────────────────────────────
    burst_types = ["ring_alert" if args.ring_alert else "data"] * args.num_bursts

    freq_offsets = None
    if args.fdma:
        channel_spacing = 41667
        freq_offsets = [
            (i - args.num_bursts // 2) * channel_spacing
            for i in range(args.num_bursts)
        ]
        print(f"  FDMA: offset = {freq_offsets} Hz")

    print(f"  Generazione {args.num_bursts} burst ({burst_types[0]})...")
    frame_iq, burst_info = generate_tdma_frame(
        num_bursts=args.num_bursts,
        burst_types=burst_types,
        freq_offsets=freq_offsets,
    )
    print(f"  Frame generato: {len(frame_iq)} campioni @ {BURST_SAMPLE_RATE / 1e3:.0f} kHz "
          f"({len(frame_iq) / BURST_SAMPLE_RATE * 1e3:.1f} ms)")

    # ── Effetti di canale (opzionale) ───────────────────────────────────────
    if not args.no_channel_effects:
        frame_iq = add_channel_effects(
            frame_iq,
            snr_db=args.snr,
            phase_offset=0.0,  # nessun offset di fase artificiale in TX
            freq_drift_hz=0.0,
        )

    # ── Ricampionamento a TX_SAMPLE_RATE ────────────────────────────────────
    print(f"  Ricampionamento: {BURST_SAMPLE_RATE / 1e3:.0f} kHz → "
          f"{TX_SAMPLE_RATE / 1e6:.1f} MHz (×{_UP}/{_DOWN})...")
    tx_iq = resample_to_hw(frame_iq)

    # ── Scala per il DAC ────────────────────────────────────────────────────
    tx_iq = scale_for_dac(tx_iq)
    peak = np.max(np.abs(tx_iq))
    print(f"  Campioni TX pronti: {len(tx_iq)} @ {TX_SAMPLE_RATE / 1e6:.1f} MSPS "
          f"({len(tx_iq) / TX_SAMPLE_RATE * 1e3:.1f} ms), picco DAC={peak:.0f}/16384")

    burst_summary = []
    for info in burst_info:
        t_start = info["start_sample"] / BURST_SAMPLE_RATE * 1000
        t_end = info["end_sample"] / BURST_SAMPLE_RATE * 1000
        burst_summary.append(f"    Burst {info['index']}: {info['num_symbols']} simboli, "
                              f"{t_start:.1f}–{t_end:.1f} ms")
    print("\n".join(burst_summary))
    print()

    if args.dry_run:
        print("[DRY RUN] Nessuna trasmissione effettuata.")
        return

    # ── Connessione e TX ────────────────────────────────────────────────────
    sdr = connect_libresdr(args.uri)
    configure_tx(sdr, args.freq, args.gain)
    print()

    if args.cyclic:
        transmit_cyclic(sdr, tx_iq)
    else:
        print(f"  Trasmissione di {args.repeat} frame...")
        t0 = time.time()
        transmit_once(sdr, tx_iq, num_repetitions=args.repeat)
        elapsed = time.time() - t0
        print(f"  Completato in {elapsed:.2f} s")

    print("Fatto!")


if __name__ == "__main__":
    main()
