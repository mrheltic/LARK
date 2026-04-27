#!/usr/bin/env python3
"""
tx_868_libresdr.py — Beacon CW 868 MHz via LibreSDR (AD9363).

Trasmette un tono CW puro a 868.1 MHz come riferimento coerente
per testare gli algoritmi DoA sull'UCA KrakenSDR a 5 antenne.

AVVISO: Usa sempre TX → RX via cavo coassiale + attenuatore in laboratorio.

Utilizzo:
    python3 tx_868_libresdr.py                       # CW continuo finché Ctrl+C
    python3 tx_868_libresdr.py --freq 868200000      # frequenza diversa
    python3 tx_868_libresdr.py --gain -30            # -30 dB attenuazione TX
    python3 tx_868_libresdr.py --dry-run             # solo info, nessuna TX
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# ── Parametri default ────────────────────────────────────────────────────────
DEFAULT_URI       = "ip:192.168.1.10"
DEFAULT_FREQ_HZ   = 868_100_000   # [Hz] 868.1 MHz ISM
TX_SAMPLE_RATE    = 1_000_000     # 1 MSPS
TX_RF_BANDWIDTH   = 200_000       # 200 kHz
DEFAULT_GAIN_DB   = -60.0         # [dB] attenuazione TX — parti basso!

# Durata del buffer CW (viene ripetuto in loop hardware)
_BUF_DURATION_S = 0.10            # 100 ms → 100 000 campioni a 1 MSPS


# =============================================================================
# Funzioni di supporto
# =============================================================================

def _make_cw(duration_s: float = _BUF_DURATION_S) -> np.ndarray:
    """Buffer CW: portante pura a frequenza zero (DC), ampiezza costante."""
    n = int(round(TX_SAMPLE_RATE * duration_s))
    # DC IQ: I=1, Q=0 → portante alla LO frequency senza offset
    iq = np.ones(n, dtype=np.complex64) * 0.9 * (2 ** 14)
    return iq


def _connect(uri: str):
    """Connette al LibreSDR via pyadi-iio."""
    try:
        import adi
    except ImportError:
        print("[ERRORE] pyadi-iio non installato. Installa con:")
        print("  pip install pyadi-iio")
        sys.exit(1)

    print(f"  Connessione a {uri} ...", end=" ", flush=True)
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
            if exc.errno == 16 and attempt < 2:   # Device busy
                print(f"occupato ({attempt+1}/3), attesa 3s ...", end=" ", flush=True)
                time.sleep(3)
            else:
                print(f"FALLITO\n[ERRORE] {exc}")
                host = uri.split(":", 1)[-1] if ":" in uri else uri
                print(f"  Verifica connessione:  ping {host}")
                print(f"  Diagnostica IIO:       iio_info -u {uri}")
                sys.exit(1)
        except Exception as exc:
            print(f"FALLITO\n[ERRORE] {exc}")
            sys.exit(1)


def _configure_tx(sdr, freq_hz: int, gain_db: float) -> None:
    """Configura il canale TX dell'AD9363."""
    sdr.sample_rate           = int(TX_SAMPLE_RATE)
    sdr.tx_rf_bandwidth       = int(TX_RF_BANDWIDTH)
    sdr.tx_lo                 = int(freq_hz)
    sdr.tx_hardwaregain_chan0 = float(gain_db)
    print(f"  Frequenza : {freq_hz / 1e6:.3f} MHz")
    print(f"  MSPS      : {TX_SAMPLE_RATE / 1e6:.1f}")
    print(f"  Guadagno  : {gain_db:+.0f} dB")


def _try_dds_cw(uri: str, sdr=None, scale: float = 0.9):
    """
    Configura il DDS hardware dell'AD9363 per CW puro.

    Il core DDS FPGA genera il segnale internamente: nessun buffer software,
    nessun push periodico. Il segnale resta attivo finché la connessione IIO
    è aperta.

    IMPORTANTE: il contesto iio.Context restituito DEVE essere mantenuto vivo
    dal chiamante (variabile in scope), altrimenti il GC può chiudere la
    connessione e resettare la configurazione DDS.

    Restituisce
    -----------
    iio.Context attivo se DDS configurato, None se non disponibile.
    """
    try:
        import iio
    except ImportError:
        return None

    # Preferisce riusare il contesto interno di adi.Pluto (stessa connessione)
    ctx = getattr(sdr, "_ctx", None)
    _ctx_owned = False                  # True = creato qui, va tenuto vivo
    if ctx is None:
        try:
            ctx = iio.Context(uri)
            _ctx_owned = True
        except Exception:
            return None

    _DDS_NAMES = [
        "cf-ad9361-dds-core-lpc",
        "cf-ad9361-dds-core-hpc",
        "axi-ad9361-dds-lpc",
        "axi-ad9361-dds-hpc",
    ]
    dds = None
    for name in _DDS_NAMES:
        dds = ctx.find_device(name)
        if dds is not None:
            break
    if dds is None:
        return None

    # TX1 DDS: altvoltage0/1 = I_F1/F2, altvoltage2/3 = Q_F1/F2
    # CW DC (freq=0): I=scale, Q=scale*cos(90°)=0 → portante alla LO
    cfgs = [
        ("altvoltage0", 0, 0,     scale),   # TX1_I tono1
        ("altvoltage1", 0, 0,     0.0  ),   # TX1_I tono2 off
        ("altvoltage2", 0, 90000, scale),   # TX1_Q tono1 (+90° → Q=0 per DC)
        ("altvoltage3", 0, 0,     0.0  ),   # TX1_Q tono2 off
    ]
    try:
        for ch_name, freq, phase_mdeg, sc in cfgs:
            ch = dds.find_channel(ch_name, is_output=True)
            if ch is None:
                return None
            try: ch.attrs["frequency"].value = str(freq)
            except Exception: pass
            try: ch.attrs["phase"].value = str(phase_mdeg)
            except Exception: pass
            try: ch.attrs["scale"].value = f"{sc:.6f}"
            except Exception: pass
            try: ch.attrs["raw"].value = "1" if sc > 0 else "0"
            except Exception: pass
        return ctx if _ctx_owned else True    # tieni vivo solo se creato qui
    except Exception:
        return None


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Beacon CW 868 MHz via LibreSDR per test DoA KrakenSDR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uri",     default=DEFAULT_URI,     help="URI IIO del LibreSDR")
    p.add_argument("--freq",    type=int, default=DEFAULT_FREQ_HZ, help="Frequenza TX [Hz]")
    p.add_argument("--gain",    type=float, default=DEFAULT_GAIN_DB,
                   help="Guadagno TX [dB]  (range -90…0). Parti basso!")
    p.add_argument("--dry-run", action="store_true",
                   help="Genera il segnale e mostra info senza trasmettere")
    args = p.parse_args()

    print("=" * 50)
    print("  LibreSDR — Beacon CW 868 MHz")
    print("=" * 50)

    tx_buf = _make_cw()
    print(f"  Buffer CW: {len(tx_buf)} campioni  "
          f"({len(tx_buf) / TX_SAMPLE_RATE * 1e3:.0f} ms, ripetuto in loop)")

    if args.dry_run:
        print("  [DRY RUN] Nessuna trasmissione.")
        return

    sdr = _connect(args.uri)
    _configure_tx(sdr, args.freq, args.gain)

    # ── Strategia TX ──────────────────────────────────────────────────────────
    # Il cyclic buffer hardware NON è affidabile su iiod via Ethernet:
    # il DMA trasmette il buffer una volta e si ferma.
    # Tentativo 1: DDS hardware (CW puro, nessun buffer, autonomo).
    # Tentativo 2: software push loop (re-push continuo, non-ciclico).
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

    # _dds_ctx DEVE restare in scope (non rinominare o del!) per tutta la durata
    # del loop: se viene garbage-collected il DDS potrebbe resettarsi.
    _dds_ctx = _try_dds_cw(args.uri, sdr=sdr, scale=0.9)
    dds_ok   = _dds_ctx is not None

    print()
    if dds_ok:
        print("  [DDS]  CW hardware attivo \u2014 nessun buffer software.")
    else:
        # Non-cyclic: sdr.tx() è rate-limited da DMA/TCP flow control.
        # push() si blocca finché il DMA non ha spazio → loop naturalmente cadenzato.
        sdr.tx_cyclic_buffer = False
        sdr.tx(tx_buf)
        print("  [SW]   CW attivo via software push loop.")

    print("  Premi Ctrl+C per fermare.")
    try:
        while True:
            if dds_ok:
                # Keepalive: impedisce all'iiod di chiudere la connessione
                time.sleep(2.0)
                try:
                    _ = sdr.tx_lo
                except Exception:
                    pass
            else:
                # Re-push continuo: push() si blocca finché il DMA non consuma
                # il buffer precedente → il loop è cadenzato automaticamente.
                sdr.tx(tx_buf)
    except KeyboardInterrupt:
        print("\n  Stop.")
    finally:
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
