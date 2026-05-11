#!/usr/bin/env python3
"""
tx_iridium_realsim.py — Simulazione realistica di satellite Iridium via LibreSDR
==================================================================================
Genera e trasmette una passata LEO completa simulata al AD9363 (LibreSDR).

Il segnale TX è identico al segnale reale Iridium per:
    • Modulazione π/4-DQPSK con filtro RRC β=0.4 (gr-iridium compatible)
    • Struttura burst IRA: 64 pream + 12 UW + 167 payload + 2 tail = 245 sym
    • TDMA 90 ms, frequenza portante configurabile
    • Modello Doppler LEO orbitale con parabola realistica:
          fd(t) = fd_max · sin(el(t)) · sign(t_closest − t)
      dove fd_max = ±40 kHz @ 1626 MHz oppure ±21 kHz @ 868 MHz
    • Envelope di potenza che segue il diagramma RHCP patch (cos²(θ) elevazione)
    • Ampiezza/fase stazionaria per canale singolo (TX monoantennas)

Frequenze supportate
--------------------
  868.1 MHz   → indoor, ISM, nessuna licenza    [DEFAULT — SICURO]
  1626.270 MHz → reale Iridium                 [RICHIEDE LICENZA / uso in laboratorio
                                                  schermato SOLO con cavo coassiale]

Uso tipico
----------
    # Indoor loop (passata simulata 45° max el, ripetuta in loop):
    python3 tx_iridium_realsim.py --freq 868.1 --elev 45 --cyclic

    # Passata singola di 60 secondi:
    python3 tx_iridium_realsim.py --freq 868.1 --elev 60 --pass-dur 60

    # Dry-run: genera IQ ma non trasmette:
    python3 tx_iridium_realsim.py --dry-run --out iridium_pass.iq

    # Riproduce il file pre-generato:
    python3 tx_iridium_realsim.py --play iridium_pass.iq

Integrazione con doa_iridium_burst.py
--------------------------------------
    Terminal 1:  python3 tx_iridium_realsim.py --freq 868.1 --cyclic
    Terminal 2:  python3 krakenSDR/src/apps/doa_iridium/doa_iridium_burst.py --freq 868.1

Il ricevitore vede:
  • IRA burst ogni 90 ms (come da satellite reale)
  • Tono di preambolo a fo + 3125 Hz (identico al satellite)
  • Deriva Doppler parabolica sull'arco di cielo (≤ ± fd_max)
  • Potenza che cresce/decresce con l'elevazione (pattern RHCP)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Local imports (resolve path relative to repo root)
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
_LARK    = os.path.normpath(os.path.join(_HERE, "..", ".."))   # repo root
_LIBRESDR = os.path.join(_LARK, "libreSDR", "src")
if _LIBRESDR not in sys.path:
    sys.path.insert(0, _LIBRESDR)

# Core iridium signal generation from realistic_sim.py
try:
    from iridium.realistic_sim import (
        generate_rrc_filter,
        generate_ira_burst,
        IridiumLEODoppler,
        simulate_iridium_pass,
        transmit_via_libresdr,
        SYMBOL_RATE   as _SYMBOL_RATE,
        SPS           as _SPS,
        SAMPLE_RATE   as _IRA_BASE_RATE,
        IRA_BURST_SYMS as _BURST_SYMS,
        IRA_PREAMBLE_SYMS as _PREAMBLE_SYMS,
        SUPERFRAME_S  as _SUPERFRAME_S,
    )
    _HAS_SIM = True
except Exception as exc:
    print(f"[WARN] Could not import iridium.realistic_sim: {exc}")
    print("       Falling back to built-in signal generation.")
    _HAS_SIM = False

# ---------------------------------------------------------------------------
# Constants (identical to the real Iridium IRA specification)
# ---------------------------------------------------------------------------
_TX_SAMPLE_RATE  = 1_000_000        # AD9363 TX rate [sps]
_IRA_UPS         = 4                # upsample factor: 250 kHz → 1 MHz
_PREAMBLE_TONE_HZ= 3_125            # tone offset for 64-sym IRA preamble (Rs/8)

if not _HAS_SIM:
    _SYMBOL_RATE    = 25_000
    _SPS            = 10
    _IRA_BASE_RATE  = 250_000
    _BURST_SYMS     = 245
    _PREAMBLE_SYMS  = 64
    _SUPERFRAME_S   = 0.090

_BURST_SAMPLES_1M = int(_BURST_SYMS * _SPS * _IRA_UPS)   # @ 1 MSPS


# ===========================================================================
# Fallback: built-in signal generator (used when realistic_sim is unavailable)
# ===========================================================================

def _rrc_taps(beta: float = 0.4, sps: int = 10, n_taps: int = 101) -> np.ndarray:
    """Root-raised cosine, per-sample windowed FIR matching gr-iridium."""
    t = np.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=np.float64) / sps
    eps = 1e-9
    h = np.zeros(n_taps, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < eps:
            h[i] = 1.0 + beta * (4 / np.pi - 1)
        elif abs(abs(4 * beta * ti) - 1.0) < eps:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            h[i] = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    h /= np.sqrt(np.sum(h ** 2))
    return h.astype(np.complex128)


def _pi4_dqpsk_symbols(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random π/4-DQPSK symbol sequence (complex, unit magnitude)."""
    delta_map = np.array([np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4])
    dibs = rng.integers(0, 4, size=n)
    phase = 0.0
    sym = np.empty(n, dtype=complex)
    for i, d in enumerate(dibs):
        phase += delta_map[d]
        sym[i] = np.exp(1j * phase)
    return sym


def _preamble_symbols() -> np.ndarray:
    """IRA preamble: 64 dibits (0,0) → all +π/4 → pure tone @ +3125 Hz."""
    phase = 0.0
    p = np.empty(_PREAMBLE_SYMS, dtype=complex)
    for i in range(_PREAMBLE_SYMS):
        phase += np.pi / 4
        p[i] = np.exp(1j * phase)
    return p


def _generate_ira_burst_local(rng: np.random.Generator, rrc: np.ndarray) -> np.ndarray:
    """Generate one IRA burst: preamble + UW + data convolved with RRC."""
    preamble = _preamble_symbols()
    uw       = _pi4_dqpsk_symbols(12, rng)          # random UW (simplified)
    data     = _pi4_dqpsk_symbols(167 + 2, rng)     # payload + tail
    symbols  = np.concatenate([preamble, uw, data])  # 245 symbols

    # Upsample (zero-insertion)
    up = np.zeros(len(symbols) * _SPS, dtype=complex)
    up[::_SPS] = symbols

    # RRC pulse shaping
    shaped = np.convolve(up, rrc, mode="same")

    # Upsample to TX rate (1 MSPS)
    shaped_us = np.zeros(len(shaped) * _IRA_UPS, dtype=complex)
    shaped_us[::_IRA_UPS] = shaped
    shaped_us = np.convolve(shaped_us,
                             np.ones(_IRA_UPS, dtype=complex) / _IRA_UPS, mode="same")

    # Guard band: pad to one full slot @ 1 MSPS
    slot_samples = int(_SUPERFRAME_S * _TX_SAMPLE_RATE)
    guard = slot_samples - len(shaped_us)
    if guard < 0:
        shaped_us = shaped_us[:slot_samples]
    else:
        shaped_us = np.pad(shaped_us, (0, guard))
    return shaped_us.astype(np.complex64)


# ===========================================================================
# Doppler application to a burst array
# ===========================================================================

def _apply_doppler_to_burst(
    burst_iq: np.ndarray,
    doppler_hz_at_start: float,
    doppler_hz_at_end: float,
    sample_rate: float = float(_TX_SAMPLE_RATE),
) -> np.ndarray:
    """
    Apply linear Doppler shift (frequency ramp) across one burst.

    The phase accumulation is:
        φ(n) = 2π·(fd_start + (fd_end - fd_start)/2 · n/N) · n / fs
    which matches a linear chirp between fd_start and fd_end.
    """
    N  = len(burst_iq)
    n  = np.arange(N, dtype=np.float64)
    fd = doppler_hz_at_start + (doppler_hz_at_end - doppler_hz_at_start) * n / N
    phase = 2 * np.pi * np.cumsum(fd) / sample_rate
    return (burst_iq * np.exp(1j * phase)).astype(burst_iq.dtype)


# ===========================================================================
# RHCP patch antenna gain envelope (elevation-dependent)
# ===========================================================================

def _rhcp_gain_linear(el_deg: float, peak_gain_lin: float = 1.0) -> float:
    """
    Approximate RHCP patch gain vs elevation angle.
    Pattern: G(el) = G0 · max(cos^1.5(90° - el), 0.05)
    This is a plausible broad cosine taper for a hemispherical patch.
    """
    if el_deg <= 0:
        return 0.05 * peak_gain_lin
    el_rad = np.deg2rad(float(el_deg))
    return float(max(np.sin(el_rad) ** 1.5, 0.05)) * peak_gain_lin


# ===========================================================================
# Main pass generator
# ===========================================================================

def generate_pass_iq(
    freq_hz: float,
    max_elev_deg: float = 45.0,
    pass_dur_s: float | None = None,
    snr_db: float = 20.0,
    sat_id: int = 47,
    beam_id: int = 0,
    verbose: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate the IQ stream for a complete simulated Iridium satellite pass.

    Returns
    -------
    iq      : np.ndarray (complex64)  — ready to feed to the AD9363 TX
    log     : list[dict]  — per-burst log with fields:
                  t_s, az_deg, el_deg, doppler_hz, papr_db, burst_idx
    """
    if _HAS_SIM:
        # Use the full-fidelity simulation from realistic_sim.py
        if pass_dur_s is None:
            # Compute visible window using doppler model
            doppler = IridiumLEODoppler(
                carrier_hz=freq_hz,
                max_elev_deg=max_elev_deg,
                t_closest_approach=0.0,
            )
            half_dur = doppler.pass_half_duration_s(min_elev_deg=5.0)
            pass_dur_s = float(2 * half_dur)

        iq, log, _, _ = simulate_iridium_pass(
            duration_s   = pass_dur_s,
            carrier_hz   = freq_hz,
            max_elev_deg = max_elev_deg,
            snr_db       = snr_db,
            sat_id       = sat_id,
            beam_id      = beam_id,
        )
        return iq.astype(np.complex64), log

    # ── Fallback: built-in generator ─────────────────────────────────────────
    rng = np.random.default_rng(seed)
    rrc = _rrc_taps()

    # Duration from geometry
    if pass_dur_s is None:
        # Approximate visible window for LEO at max_elev_deg
        r_earth = 6_371_000.0
        r_orbit = r_earth + 780_000.0
        sin_rho  = r_earth / r_orbit
        el_rad   = np.deg2rad(max_elev_deg)
        half_d   = np.arcsin(np.cos(el_rad) * r_orbit / r_earth) - (np.pi/2 - el_rad) - np.arcsin(sin_rho)
        v_sat    = 7464.0
        pass_dur_s = float(2 * half_d * r_orbit / v_sat) if half_d > 0 else 60.0

    n_bursts     = int(pass_dur_s / _SUPERFRAME_S)
    slot_samples = int(_SUPERFRAME_S * _TX_SAMPLE_RATE)
    total_iq     = np.zeros(n_bursts * slot_samples, dtype=np.complex64)

    # Doppler model: fd(t) = fd_max * cos(el(t)) * sign(t_closest - t),
    # where el(t) = el_max * sinc(t / half_pass) approximated numerically.
    t_closest  = pass_dur_s / 2.0
    v_sat      = 7464.0
    fd_max     = float(v_sat * freq_hz / 3e8)
    # Elevation arc
    # el(t) = el_max * sin(π * t / pass_dur_s)  (simple tent model)

    log: list[dict] = []
    n_sig    = 10 ** (snr_db / 20.0)
    n_noise  = 1.0

    if verbose:
        print(f"\n  Pass: {pass_dur_s:.1f}s  elev_max={max_elev_deg:.0f}°  "
              f"fd_max=±{fd_max/1e3:.1f} kHz  n_bursts={n_bursts}")

    for idx in range(n_bursts):
        t  = idx * _SUPERFRAME_S
        el = max_elev_deg * np.sin(np.pi * t / pass_dur_s)

        # Doppler: max at t=0 and t=T (horizon), 0 near t_closest
        t_c  = t_closest
        doppler = fd_max * np.cos(np.deg2rad(el)) * np.sign(t - t_c)

        # RHCP envelope
        gain = _rhcp_gain_linear(float(el)) * n_sig

        # Generate burst
        burst = _generate_ira_burst_local(rng, rrc)
        burst_len = min(len(burst), slot_samples)

        # Apply Doppler ramp across the burst duration
        burst_dur_s = len(burst) / _TX_SAMPLE_RATE
        dop_end = fd_max * np.cos(np.deg2rad(el)) * np.sign(
            (t + burst_dur_s) - t_c
        )
        burst = _apply_doppler_to_burst(burst, doppler, dop_end)

        # Scale: SNR + RHCP envelope
        burst = (burst / (np.std(burst) + 1e-30) * gain).astype(np.complex64)

        # Add AWGN noise
        noise = (rng.standard_normal(burst_len) + 1j * rng.standard_normal(burst_len))
        noise = (noise / np.sqrt(2) * n_noise).astype(np.complex64)

        start = idx * slot_samples
        total_iq[start: start + burst_len] += burst[:burst_len]
        total_iq[start: start + burst_len] += noise

        log.append(dict(
            t_s=float(t), el_deg=float(el), az_deg=0.0,
            doppler_hz=float(doppler), burst_idx=idx,
        ))
        if verbose and idx % max(1, n_bursts // 20) == 0:
            print(f"  [{t:5.1f}s] el={el:5.1f}°  fd={doppler:+7.0f} Hz  "
                  f"gain={20*np.log10(gain+1e-9):.1f} dB", flush=True)

    return total_iq, log


# ===========================================================================
# Multi-satellite pass generator
# ===========================================================================

# Parametri predefiniti per N satelliti simultanei (plausibili per Iridium):
# - elevazioni diverse → gain RHCP diverso
# - posizioni orbitali diverse → Doppler diverso
# - SNR decrescente (il satellite più lontano è più debole)
_MULTI_SAT_DEFAULTS = [
    dict(max_elev=45.0, snr_db_offset= 0,  sat_id=47, beam_id=0,
         pass_phase=0.0,   label="Sat-A (domninante)"),
    dict(max_elev=25.0, snr_db_offset=-4,  sat_id=52, beam_id=3,
         pass_phase=0.35,  label="Sat-B (vicino horizon)"),
    dict(max_elev=60.0, snr_db_offset=-7,  sat_id=61, beam_id=7,
         pass_phase=0.70,  label="Sat-C (ad alta elevazione)"),
]
# pass_phase [0,1): offset di fase nel profilo Doppler.  A pass_phase=0 il
# satellite è all'inizio del pass (Doppler massimo positivo); a 0.5 è al
# picco d'elevazione (Doppler ≈ 0); a 1.0 è all'uscita verso l'orizzonte.
# Con satelliti a fasi diverse si ottengono Doppler distinti nello stesso
# instante, che il ricevitore può discriminare via FFT.


def generate_multisat_pass_iq(
    n_sats: int,
    freq_hz: float,
    base_snr_db: float = 20.0,
    pass_dur_s: float | None = None,
    verbose: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict]]:
    """
    Genera la sovrapposizione di n_sats passate satellitari indipendenti.

    Ogni satellite ha un profilo Doppler diverso (offset di fase nel pass)
    in modo che il ricevitore veda TONI DI PREAMBOLO A FREQUENZE DIVERSE.
    Questo è il meccanismo reale con cui Iridium distribuisce i burst TDMA
    provenienti da satelliti in posizioni orbitali differenti.

    Nota sulla DoA indoor
    ---------------------
    Con TX a singola antenna, tutti i "satelliti" arrivano dalla stessa
    direzione fisica.  La scansione Doppler del ricevitore identifica
    correttamente N sorgenti distinte, ma le stime di azimuth saranno
    identiche (coincidono con la posizione del TX).
    Per testare la SEPARAZIONE angolare di più sorgenti:
      • usare due TX fisici a angoli diversi (due LibreSDR, o due antenne)
      • oppure fare il test outdoor con il vero Iridium
    """
    defaults = _MULTI_SAT_DEFAULTS[:n_sats]
    total_iq: np.ndarray | None = None
    all_logs: list[dict] = []

    for i, sat_params in enumerate(defaults):
        label   = sat_params["label"]
        snr_db  = base_snr_db + sat_params["snr_db_offset"]
        max_el  = sat_params["max_elev"]
        sat_id  = sat_params["sat_id"]
        beam_id = sat_params["beam_id"]
        phase   = sat_params["pass_phase"]

        if verbose:
            fd_max = 7464.0 * freq_hz / 3e8
            print(f"\n  [{label}] SNR={snr_db:.0f}dB  el_max={max_el:.0f}°  "
                  f"pass_phase={phase:.2f}  fd_peak≈{fd_max*np.cos(np.deg2rad(max_el))/1e3:+.1f}kHz")

        iq_sat, log_sat = generate_pass_iq(
            freq_hz      = freq_hz,
            max_elev_deg = max_el,
            pass_dur_s   = pass_dur_s,
            snr_db       = snr_db,
            sat_id       = sat_id,
            beam_id      = beam_id,
            verbose      = verbose,
            seed         = seed + i * 7,
        )

        # Applico l'offset di fase nel profilo Doppler: slino circolarmente
        # il segnale di "phase × len" campioni per simulare un satellite
        # che è a una fase diversa del suo pass.
        if phase > 0.0:
            shift = int(phase * len(iq_sat))
            iq_sat = np.roll(iq_sat, shift)

        # Aggiorna log con sat_id
        for entry in log_sat:
            entry["sat_id"]  = sat_id
            entry["sat_label"] = label
        all_logs.extend(log_sat)

        n_min = min(len(total_iq), len(iq_sat)) if total_iq is not None else len(iq_sat)
        if total_iq is None:
            total_iq = iq_sat[:n_min].copy()
        else:
            L = min(len(total_iq), len(iq_sat))
            total_iq = total_iq[:L] + iq_sat[:L]

    if total_iq is None:
        total_iq = np.zeros(1, dtype=np.complex64)

    if verbose:
        print(f"\n  Totale IQ: {len(total_iq)/1e6:.2f} M campioni  "
              f"({len(total_iq)/_TX_SAMPLE_RATE:.1f}s)  da {n_sats} satelliti sovrapposti")

    return total_iq.astype(np.complex64), all_logs


# ===========================================================================
# Safety / informational check
# ===========================================================================

def _check_frequency_safety(freq_hz: float) -> None:
    if freq_hz >= 1_000_000_000:
        print("\n" + "=" * 60)
        print(" *** ATTENZIONE / WARNING ***")
        print(f" Frequenza TX = {freq_hz/1e6:.3f} MHz")
        print(" TRASMETTERE SULLA FREQUENZA IRIDIUM REALE RICHIEDE")
        print(" LICENZA AMATORIALE + PERMESSO SPECIALE. USARE SOLO")
        print(" IN LABORATORIO SCHERMATO / CAVO COAXIALE DIRETTO.")
        print("=" * 60 + "\n")
    elif 868_000_000 <= freq_hz <= 868_600_000:
        print(f"[INFO] Frequenza: {freq_hz/1e6:.3f} MHz (ISM 868 MHz — indoor, libera)")
    else:
        print(f"[INFO] Frequenza TX: {freq_hz/1e6:.3f} MHz — verificare normativa locale")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Simulatore realistica di satellite Iridium via LibreSDR AD9363.\n"
            f"Burst IRA π/4-DQPSK con modulazione + Doppler LEO orbitale (fd_max "
            f"± {7464*1626.27e6/3e8/1e3:.0f} kHz a 1626 MHz,  "
            f"±{7464*868.1e6/3e8/1e3:.0f} kHz a 868 MHz)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--freq",     type=float, default=868.1,
                   help="Frequenza RF [MHz]. Default: 868.1 (ISM, sicuro indoor)")
    p.add_argument("--elev",     type=float, default=45.0,
                   help="Elevazione massima del pass simulato [gradi]")
    p.add_argument("--pass-dur", type=float, default=None,
                   help="Durata del pass [s] (default: calcolato dalla geometria LEO)")
    p.add_argument("--snr",      type=float, default=20.0,
                   help="SNR simulato del burst [dB] (preamble tone vs rumore)")
    p.add_argument("--sat-id",   type=int,   default=47, help="Satellite ID Iridium (1-66)")
    p.add_argument("--beam-id",  type=int,   default=0,  help="Beam ID del satellite (0-47)")
    p.add_argument("--gain",     type=float, default=-20.0,
                   help="Guadagno TX AD9363 [dB] (tipico: -30 indoor, -20 outdoor/cavo)")
    p.add_argument("--uri",      default="ip:192.168.1.10",
                   help="URI del LibreSDR (pyadi-iio)")
    p.add_argument("--cyclic",   action="store_true",
                   help="Loop continuo: ripete il pass in DMA ciclico")
    p.add_argument("--one-shot", action="store_true",
                   help="Trasmette il pass una volta sola poi esce")
    p.add_argument("--dry-run",  action="store_true",
                   help="Genera il segnale IQ ma non trasmette")
    p.add_argument("--out",      default=None, metavar="FILE.IQ",
                   help="Salva il segnale IQ in binary float32 I,Q interleaved")
    p.add_argument("--play",     default=None, metavar="FILE.IQ",
                   help="Carica e trasmette un file IQ precedentemente salvato")
    p.add_argument("--n-sats",   type=int,   default=1, choices=[1, 2, 3],
                   help=(
                       "Numero di satelliti simulati simultaneamente (1-3). "
                       "Ogni satellite ha Doppler diverso → toni preambolo "
                       "distinti rilevabili dal ricevitore KrakenSDR. "
                       "NB: per separazione angolare indoor servono più TX fisici."
                   ))
    p.add_argument("--seed",     type=int,   default=42, help="Seme random (riproducibilità)")
    p.add_argument("--verbose",  action="store_true", help="Stampa info per burst")
    args = p.parse_args()

    freq_hz = args.freq * 1e6
    _check_frequency_safety(freq_hz)

    print("=" * 64)
    print(f"  TX IRIDIUM PASS SIMULATOR  @  {args.freq:.3f} MHz  ({args.n_sats} satellite/i)")
    print(f"  LEO orbit 780 km  max_el={args.elev:.0f}°  SNR={args.snr:.0f} dB")
    if args.n_sats > 1:
        print(f"  Multi-satellite: {args.n_sats} sorgenti a Doppler diversi sovrapposti")
        print(f"  (indoor: direction separazione richiede {args.n_sats} TX fisici distinti)")
    if _HAS_SIM:
        print("  Signal engine: iridium.realistic_sim  [full fidelity]")
    else:
        print("  Signal engine: built-in fallback  [simplified]")
    print(f"  Symbol rate: {_SYMBOL_RATE/1e3:.0f} ksps  "
          f"TX rate: {_TX_SAMPLE_RATE/1e6:.0f} MSPS  "
          f"IRA burst: {_BURST_SYMS} sym = {_BURST_SYMS*_SPS*_IRA_UPS/_TX_SAMPLE_RATE*1e3:.1f} ms")
    print(f"  Preamble tone offset: +{_PREAMBLE_TONE_HZ} Hz  (identical to real Iridium IRA)")
    print("=" * 64)

    # ── Load from file ────────────────────────────────────────────────────────
    if args.play:
        if not os.path.isfile(args.play):
            print(f"[ERROR] File non trovato: {args.play}"); sys.exit(1)
        raw = np.fromfile(args.play, dtype=np.float32)
        iq_stream = raw[0::2] + 1j * raw[1::2]
        iq_stream = iq_stream.astype(np.complex64)
        log: list[dict] = []
        print(f"[INFO] Caricato {args.play}: {len(iq_stream)/1e6:.2f} M campioni "
              f"({len(iq_stream)/_TX_SAMPLE_RATE:.1f} s)")
    else:
        # ── Generate pass ─────────────────────────────────────────────────────
        print("[GEN] Simulazione oracle del pass in corso…")
        t0 = time.monotonic()
        if args.n_sats > 1:
            iq_stream, log = generate_multisat_pass_iq(
                n_sats      = args.n_sats,
                freq_hz     = freq_hz,
                base_snr_db = args.snr,
                pass_dur_s  = args.pass_dur,
                verbose     = args.verbose,
                seed        = args.seed,
            )
        else:
            iq_stream, log = generate_pass_iq(
                freq_hz      = freq_hz,
                max_elev_deg = args.elev,
                pass_dur_s   = args.pass_dur,
                snr_db       = args.snr,
                sat_id       = args.sat_id,
                beam_id      = args.beam_id,
                verbose      = args.verbose,
                seed         = args.seed,
            )
        elapsed = time.monotonic() - t0
        dur_s   = len(iq_stream) / _TX_SAMPLE_RATE
        print(f"[GEN] Generato {dur_s:.1f}s di segnale in {elapsed:.2f}s  "
              f"({len(log)} burst IRA)")

        if log:
            dops = [e["doppler_hz"] for e in log]
            els  = [e["el_deg"]     for e in log]
            print(f"  Doppler: min={min(dops):+.0f} Hz  max={max(dops):+.0f} Hz  "
                  f"(range: {max(dops)-min(dops):.0f} Hz)")
            print(f"  Elevation: {min(els):.1f}° … {max(els):.1f}° … {min(els):.1f}°")
            print(f"  Peak gain at el_max={max(els):.1f}°: "
                  f"{20*np.log10(_rhcp_gain_linear(max(els))+1e-30):.1f} dBi (relative)")

        # ── Optional file save ────────────────────────────────────────────────
        if args.out:
            interleaved = np.empty(2 * len(iq_stream), dtype=np.float32)
            interleaved[0::2] = iq_stream.real
            interleaved[1::2] = iq_stream.imag
            interleaved.tofile(args.out)
            size_mb = os.path.getsize(args.out) / 1e6
            print(f"[SAVE] Salvato: {args.out}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("[DRY-RUN] Segnale generato.  Nessuna trasmissione (--dry-run).")
        return

    # ── Transmit ──────────────────────────────────────────────────────────────
    if _HAS_SIM:
        print(f"\n[TX] Trasmissione via LibreSDR  {args.uri}  …")
        try:
            transmit_via_libresdr(
                iq_stream,
                uri            = args.uri,
                center_freq_hz = freq_hz,
                tx_gain_db     = args.gain,
                cyclic         = args.cyclic,
            )
        except KeyboardInterrupt:
            print("\n[TX] Interrotto dall'utente.")
    else:
        # Fallback: manual pyadi-iio transmit
        try:
            import adi  # type: ignore
        except ImportError:
            print("[ERROR] pyadi-iio non installato.  Installa con: pip install pyadi-iio")
            sys.exit(1)

        try:
            sdr = adi.ad9364(args.uri) if "9364" in args.uri else adi.ad9363(args.uri)
        except Exception:
            sdr = adi.Pluto(args.uri)

        sdr.sample_rate       = int(_TX_SAMPLE_RATE)
        sdr.tx_lo             = int(freq_hz)
        sdr.tx_rf_bandwidth   = int(_TX_SAMPLE_RATE * 0.8)
        sdr.tx_hardwaregain_chan0 = int(args.gain)
        sdr.tx_enabled_channels   = [0]

        # Scale to int16
        scale = 0.9 * 2**15 / (np.abs(iq_stream).max() + 1e-12)
        iq_s  = (iq_stream * scale).astype(np.complex64)

        if args.cyclic:
            sdr.tx_cyclic_buffer = True
            sdr.tx(iq_s)
            dur_s = len(iq_stream) / _TX_SAMPLE_RATE
            print(f"[TX] Ciclico iniziato.  Press Ctrl+C per fermare.  "
                  f"Durata passata: {dur_s:.1f}s")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("[TX] Fermato.")
        else:
            # Stream in chunks matching one pass
            chunk_size = len(iq_s)
            sdr.tx(iq_s[:chunk_size])
            dur_s = len(iq_stream) / _TX_SAMPLE_RATE
            print(f"[TX] TX completato: {dur_s:.1f}s di segnale trasmesso.")
            if not args.one_shot:
                print("[TX] Usa --cyclic per ripetere, --one-shot per uscire subito.")


if __name__ == "__main__":
    main()
