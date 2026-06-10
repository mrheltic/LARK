#!/usr/bin/env python3
"""
minimal_phase_check.py — Standalone KrakenSDR phase sanity checker
===========================================================

A *minimal* script that connects directly to Heimdall (like the external
gr-krakensdr examples) and prints the **relative phases** between antenna-0
and the other 4 channels in real time.

Why this exists
---------------
The full DoA pipeline (burst detection → tone scan → MUSIC → tracking) is
heavy.  If you only want to know:

  1. Is the KrakenSDR sending valid IQ frames?
  2. Are the inter-channel phases stable?
  3. Is the calibration roughly correct?

…then this script is enough.  It uses the exact same TCP protocol as
`krakensdr_source.py` (external/gr-krakensdr) but with no GNU Radio deps.

Usage
-----
    python3 minimal_phase_check.py                    # indoor 1626 MHz, 5 s
    python3 minimal_phase_check.py --duration 30      # 30 seconds
    python3 minimal_phase_check.py --freq 433 --gain 30
    python3 minimal_phase_check.py --out phases.csv   # save to CSV

Exit codes
----------
    0  OK  (phases stable, std < 15°)
    1  Suspicious (std 15–40°)
    2  Bad (std > 40° or no frames received)
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import struct
import sys
import time

import numpy as np

# ── Minimal IQ header decoder (same as external/gr-krakensdr) ─────────────────

IQ_SYNC_WORD = 0x2BF7B95A
IQ_HEADER_SIZE = 1024
RESERVED_BYTES = 192


def decode_iq_header(raw: bytes):
    fmt = ("II16sIIIQQQIQIIQIII" + "I" * 32 + "IIII" + "I" * RESERVED_BYTES + "I")
    lst = struct.unpack(fmt, raw)
    return {
        "sync_word": lst[0],
        "frame_type": lst[1],
        "active_ant_chs": lst[4],
        "rf_center_freq": lst[6],
        "sampling_freq": lst[8],
        "cpi_length": lst[9],
        "sample_bit_depth": lst[15],
        "delay_sync_flag": lst[49],
        "iq_sync_flag": lst[50],
        "noise_source_state": lst[52],
    }


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    received = 0
    while received < n:
        chunk = sock.recv_into(view[received:], n - received)
        if chunk == 0:
            raise ConnectionError(f"Socket closed after {received}/{n} bytes")
        received += chunk
    return bytes(buf)


# ── Phase extraction helpers ────────────────────────────────────────────────


def frame_phase_and_coherence(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute phase difference and coherence magnitude between CH0 and CH1..CH4.

    Returns
    -------
    phases : (4,) array of angles in degrees
    coh    : (4,) array of |mean(ch_i * conj(ch_0))| / (rms_i * rms_0)
             In noise this is near 0; in strong signal near 1.
    """
    ch0 = frame[0]
    rms0 = np.sqrt(np.mean(np.abs(ch0) ** 2))
    phases = np.empty(4, dtype=np.float64)
    coh = np.empty(4, dtype=np.float64)
    for i in range(1, frame.shape[0]):
        chi = frame[i]
        corr = np.mean(chi * np.conj(ch0))
        rmsi = np.sqrt(np.mean(np.abs(chi) ** 2))
        phases[i - 1] = np.degrees(np.angle(corr))
        denom = rmsi * rms0
        coh[i - 1] = np.abs(corr) / denom if denom > 0 else 0.0
    return phases, coh


def phase_std_across_time(phase_history: list[np.ndarray]) -> np.ndarray:
    """Per-channel standard deviation across the collected history."""
    arr = np.stack(phase_history, axis=0)  # (N, 4)
    # Wrap to [-180, 180] before computing std
    arr = ((arr + 180.0) % 360.0) - 180.0
    return np.std(arr, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Minimal KrakenSDR phase checker")
    p.add_argument("--host", default="127.0.0.1", help="Heimdall host")
    p.add_argument("--port", type=int, default=5000, help="IQ data port")
    p.add_argument("--ctrl-port", type=int, default=5001, help="Control port")
    p.add_argument("--freq", type=float, default=1626.270, help="RF freq [MHz]")
    p.add_argument("--gain", type=float, default=40.0, help="RX gain [dB]")
    p.add_argument("--duration", type=float, default=5.0, help="Capture duration [s]")
    p.add_argument("--out", default=None, help="Optional CSV output path")
    args = p.parse_args()

    freq_hz = int(args.freq * 1e6)
    valid_gains = [0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
                   19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
                   38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]
    gain = min(valid_gains, key=lambda x: abs(x - args.gain))

    # ── TCP connect ─────────────────────────────────────────────────────────
    print(f"[PHASE-CHECK] Connecting to {args.host}:{args.port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30.0)
    sock.connect((args.host, args.port))
    sock.sendall(b"streaming")

    # Read bootstrap frame (Heimdall sends it automatically after "streaming")
    raw_hdr = recv_exact(sock, IQ_HEADER_SIZE)
    hdr = decode_iq_header(raw_hdr)
    n_ch = hdr["active_ant_chs"]
    cpi_len = hdr["cpi_length"]
    bit_depth = hdr["sample_bit_depth"]
    payload_bytes = n_ch * cpi_len * 2 * (bit_depth // 8)
    _ = recv_exact(sock, payload_bytes)  # discard bootstrap payload

    print(f"[PHASE-CHECK] Bootstrap OK  ch={n_ch}  cpi={cpi_len}  "
          f"fs={hdr['sampling_freq']/1e6:.3f} MHz")

    # Control socket
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl_sock.settimeout(10.0)
    ctrl_sock.connect((args.host, args.ctrl_port))
    ctrl_sock.sendall(b"INIT" + bytes(124))

    # Set frequency
    ctrl_sock.sendall(b"FREQ" + struct.pack("Q", freq_hz) + bytes(116))

    # Set gain ( snapped to valid RTL-SDR steps )
    gain_int = int(gain * 10)
    gain_payload = struct.pack("I" * n_ch, *[gain_int] * n_ch)
    gain_padding = bytes(128 - (n_ch + 1) * 4)
    ctrl_sock.sendall(b"GAIN" + gain_payload + gain_padding)

    print(f"[PHASE-CHECK] Set freq={freq_hz/1e6:.3f} MHz  gain={gain:.1f} dB")
    print(f"[PHASE-CHECK] Collecting for {args.duration:.0f} s …")
    print()

    # ── Acquisition loop ────────────────────────────────────────────────────
    t0 = time.time()
    frames = 0
    good_frames = 0
    phase_history: list[np.ndarray] = []
    coh_history: list[np.ndarray] = []
    csv_rows: list[dict] = []

    try:
        while time.time() - t0 < args.duration:
            sock.sendall(b"IQDownload")
            raw_hdr = recv_exact(sock, IQ_HEADER_SIZE)
            hdr = decode_iq_header(raw_hdr)
            payload_bytes = (hdr["active_ant_chs"] * hdr["cpi_length"] * 2
                             * (hdr["sample_bit_depth"] // 8))
            if payload_bytes == 0:
                continue

            raw_payload = recv_exact(sock, payload_bytes)
            frames += 1

            # Skip non-data frames
            if hdr["frame_type"] != 0:
                continue

            # Skip frames while delay_sync / iq_sync not ready
            if hdr["delay_sync_flag"] == 0 or hdr["iq_sync_flag"] == 0:
                continue

            # Skip noise-source frames
            if hdr["noise_source_state"] == 1:
                continue

            n_ch = hdr["active_ant_chs"]
            cpi_len = hdr["cpi_length"]
            count = n_ch * cpi_len
            frame = (np.frombuffer(raw_payload, dtype=np.complex64, count=count)
                     .reshape(n_ch, cpi_len)
                     .copy())

            good_frames += 1
            phases, coh = frame_phase_and_coherence(frame)
            phase_history.append(phases)
            coh_history.append(coh)

            # Rolling print every 10 good frames
            if good_frames % 10 == 0:
                _rolling = np.stack(phase_history[-10:], axis=0)
                _rolling = ((_rolling + 180.0) % 360.0) - 180.0
                _mean = _rolling.mean(axis=0)
                _std = _rolling.std(axis=0)
                _coh = np.stack(coh_history[-10:], axis=0).mean(axis=0)
                ts = time.time() - t0
                line = (
                    f"t={ts:5.1f}s  "
                    f"CH1={_mean[0]:+7.2f}±{_std[0]:5.2f}°|c={_coh[0]:.2f}  "
                    f"CH2={_mean[1]:+7.2f}±{_std[1]:5.2f}°|c={_coh[1]:.2f}  "
                    f"CH3={_mean[2]:+7.2f}±{_std[2]:5.2f}°|c={_coh[2]:.2f}  "
                    f"CH4={_mean[3]:+7.2f}±{_std[3]:5.2f}°|c={_coh[3]:.2f}"
                )
                print(f"\r{line}", end="", flush=True)

            if args.out:
                csv_rows.append({
                    "time_s": round(time.time() - t0, 3),
                    "ch1_deg": round(float(phases[0]), 3),
                    "ch2_deg": round(float(phases[1]), 3),
                    "ch3_deg": round(float(phases[2]), 3),
                    "ch4_deg": round(float(phases[3]), 3),
                    "ch1_coh": round(float(coh[0]), 4),
                    "ch2_coh": round(float(coh[1]), 4),
                    "ch3_coh": round(float(coh[2]), 4),
                    "ch4_coh": round(float(coh[3]), 4),
                })

    except KeyboardInterrupt:
        print("\n[PHASE-CHECK] Interrupted by user")
    finally:
        print()  # newline after \r
        try:
            sock.sendall(b"q")
            ctrl_sock.sendall(b"EXIT" + bytes(124))
        except Exception:
            pass
        sock.close()
        ctrl_sock.close()

    # ── Summary ─────────────────────────────────────────────────────────────
    total = time.time() - t0
    print(f"[PHASE-CHECK] Finished  duration={total:.1f}s  frames={frames}  "
          f"good={good_frames}  dropped={frames - good_frames}")

    if not phase_history:
        print("[PHASE-CHECK] ERROR: no valid frames received")
        return 2

    all_phases = np.stack(phase_history, axis=0)
    all_phases = ((all_phases + 180.0) % 360.0) - 180.0
    mean_phases = all_phases.mean(axis=0)
    std_phases = all_phases.std(axis=0)

    all_coh = np.stack(coh_history, axis=0)
    mean_coh = all_coh.mean(axis=0)

    print()
    print("━" * 55)
    print("SUMMARY  (mean ± std over all good frames)")
    for i in range(4):
        status = "✓" if std_phases[i] < 15 else "⚠" if std_phases[i] < 40 else "✗"
        print(f"  CH{i+1} vs CH0:  mean={mean_phases[i]:+8.2f}°  "
              f"std={std_phases[i]:6.2f}°  |coh|={mean_coh[i]:.3f}  {status}")

    # Compare to the calibration the DOA pipeline actually uses
    # (doa_config.toml → cal_file, typically cal_tle.npz from fit_array_cal.py)
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _src = os.path.dirname(_here)
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from apps.doa_iridium.run_doa import _load_cal, load_config
        cfg = load_config(os.path.join(_src, "apps", "doa_iridium",
                                       "doa_config.toml"))
        cal = _load_cal(cfg["array"].get("cal_file", ""), 5)
        if any(c != 0.0 for c in cal):
            print()
            print("CALIBRATION check (doa_config.toml cal_file)")
            for i in range(4):
                expected = cal[i + 1] - cal[0]
                expected = ((expected + 180.0) % 360.0) - 180.0
                delta = abs(((mean_phases[i] - expected) + 180.0) % 360.0 - 180.0)
                ok = "✓" if delta < 15 else "⚠" if delta < 40 else "✗"
                print(f"  CH{i+1}: measured={mean_phases[i]:+7.2f}°  "
                      f"cal={expected:+7.2f}°  Δ={delta:5.1f}°  {ok}")
    except Exception as exc:
        print(f"[PHASE-CHECK] Could not load calibration for comparison: {exc}")

    print("━" * 55)

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["time_s", "ch1_deg", "ch2_deg", "ch3_deg", "ch4_deg",
                            "ch1_coh", "ch2_coh", "ch3_coh", "ch4_coh"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[PHASE-CHECK] Saved CSV → {args.out}")

    # ── Verdict ────────────────────────────────────────────────────────────
    # If coherence is very low, we are looking at noise → phases are meaningless
    if mean_coh.mean() < 0.15:
        print()
        print("VERDICT: 2 (NO SIGNAL)")
        print("  Mean coherence < 0.15 → only noise detected.")
        print("  Phases are random by definition; std ~103° is expected for noise.")
        print("  → Turn on the TX (indoor) or wait for a satellite pass (outdoor).")
        return 2

    if max(std_phases) > 40:
        print()
        print("VERDICT: 2 (BAD) — phases are unstable despite signal present.")
        print("  Check cabling / sync / calibration offsets.")
        return 2
    if max(std_phases) > 15:
        print()
        print("VERDICT: 1 (SUSPICIOUS) — moderate phase jitter.")
        return 1
    print()
    print("VERDICT: 0 (OK) — phases are stable and signal is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
