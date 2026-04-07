#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest_loopback.py — Encode→Decode loopback test (no hardware needed)

Generates a GPS L1 C/A signal with gps_engine (TX encoder), writes it
to a temporary file, then runs gnss-sdr (C++ receiver) to decode it.

Verifies:
  1. TX signal generation (Gold codes + nav message)
  2. gnss-sdr acquisition (correct PRN detected)
  3. gnss-sdr tracking (C/N₀ and lock via UDP monitoring)

Usage:
  python3 scripts/selftest_loopback.py
  python3 scripts/selftest_loopback.py --prn 7 --duration 5.0 --snr 30
"""

import sys
import os
import argparse
import time
import tempfile
import subprocess
import signal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss.gps_engine import (
    generate_gps_baseband, generate_ca_code,
    CA_CODE_RATE, CA_CODE_LEN, CODE_PERIOD_S,
    NAV_BIT_RATE, encode_nav_frame, nav_bits_to_bipolar,
)

try:
    from gnss.gnss_decoder import find_gnss_sdr, generate_temp_config
    from gnss.gnss_monitor import GnssSdrMonitor
    HAS_GNSS_SDR = True
except ImportError:
    HAS_GNSS_SDR = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CONF_DIR = os.path.join(PROJECT_DIR, 'conf')


def add_noise(sig: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN noise to achieve desired SNR."""
    sig_power = np.mean(np.abs(sig) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
    )
    return (sig + noise).astype(np.complex64)


def test_tx_encoding(prn, samp_rate, duration_s, snr, doppler, code_phase):
    """Test 1: TX signal generation (pure Python, no hardware)."""
    print(f"\n[TX] Generating GPS L1 C/A signal:")
    print(f"     PRN={prn}, Doppler={doppler:+.0f} Hz, "
          f"code_phase={code_phase:.1f} chips")
    print(f"     SNR={snr:.0f} dB, duration={duration_s:.1f} s, "
          f"samp_rate={samp_rate/1e6:.3f} MSPS")

    t0 = time.time()
    sig = generate_gps_baseband(
        prn=prn, samp_rate=samp_rate, duration_s=duration_s,
        doppler_hz=doppler, code_phase_chips=code_phase, amplitude=1.0)
    sig = add_noise(sig, snr)
    gen_time = time.time() - t0

    print(f"     Generated {len(sig)} samples in {gen_time:.2f}s")

    # Basic verification: signal has expected power and duration
    n_expected = int(samp_rate * duration_s)
    ok_len = abs(len(sig) - n_expected) <= samp_rate * 0.001
    ok_power = 0.01 < np.mean(np.abs(sig) ** 2) < 100.0

    ca_code = generate_ca_code(prn)
    ok_code = len(ca_code) == CA_CODE_LEN and ca_code.dtype == np.float32

    return sig, ok_len, ok_power, ok_code


def test_gnss_sdr_decode(sig, prn, samp_rate, timeout_s=30):
    """Test 2: gnss-sdr acquisition + tracking via File_Signal_Source."""
    gnss_sdr_bin = find_gnss_sdr()
    if gnss_sdr_bin is None:
        print("[SKIP] gnss-sdr binary not found")
        return None, None, None

    # Write IQ to temp file
    tmp_fd, tmp_iq = tempfile.mkstemp(suffix='.dat', prefix='selftest_iq_')
    os.close(tmp_fd)
    sig.tofile(tmp_iq)
    file_size = os.path.getsize(tmp_iq)
    print(f"\n[gnss-sdr] IQ file: {tmp_iq} ({file_size / 1e6:.1f} MB)")

    # Create config with File_Signal_Source pointing to our temp IQ
    base_conf = os.path.join(CONF_DIR, 'gnss_sdr_file_source.conf')
    if not os.path.isfile(base_conf):
        print(f"[SKIP] Config not found: {base_conf}")
        os.unlink(tmp_iq)
        return None, None, None

    overrides = {
        'SignalSource.filename': tmp_iq,
        'SignalSource.sampling_frequency': str(int(samp_rate)),
        'GNSS-SDR.internal_fs_sps': str(int(samp_rate)),
        'SignalSource.samples': '0',
        'Acquisition_1C.threshold': '2.0',
        'Acquisition_1C.doppler_max': '15000',
    }
    tmp_conf = generate_temp_config(base_conf, overrides)

    # Start UDP monitor
    monitor = GnssSdrMonitor()
    monitor.start(monitors=['acquisition', 'tracking'])

    # Launch gnss-sdr
    cmd = [gnss_sdr_bin, f'--config_file={tmp_conf}']
    print(f"[gnss-sdr] Launching: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=PROJECT_DIR)

    # Wait for gnss-sdr to finish or timeout
    acquired_prns = set()
    tracked_prns = {}
    t_start = time.time()

    try:
        while time.time() - t_start < timeout_s:
            if proc.poll() is not None:
                break
            time.sleep(0.5)

            acq = monitor.get_acquisition_results()
            for key, msg in acq.items():
                acquired_prns.add(msg.prn)

            trk = monitor.get_tracking_state()
            for key, msg in trk.items():
                tracked_prns[msg.prn] = msg.cn0_db_hz

        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    finally:
        monitor.stop()
        os.unlink(tmp_iq)
        if os.path.exists(tmp_conf):
            os.unlink(tmp_conf)

    rc = proc.returncode
    print(f"[gnss-sdr] Exit code: {rc}")

    ok_acq = prn in acquired_prns
    ok_trk = prn in tracked_prns
    cn0 = tracked_prns.get(prn, 0.0)

    if acquired_prns:
        print(f"[gnss-sdr] Acquired PRNs: {sorted(acquired_prns)}")
    else:
        print(f"[gnss-sdr] No satellites acquired")

    if tracked_prns:
        for p, c in sorted(tracked_prns.items()):
            print(f"[gnss-sdr] Tracking PRN {p}: C/N₀={c:.1f} dB-Hz")
    else:
        print(f"[gnss-sdr] No channels tracking")

    return ok_acq, ok_trk, cn0


def main():
    parser = argparse.ArgumentParser(
        description='GPS L1 C/A Encode→Decode loopback self-test '
                    '(TX: gps_engine, RX: gnss-sdr)')
    parser.add_argument('--prn', type=int, default=7,
                        help='PRN satellite to encode (default: 7)')
    parser.add_argument('--samp-rate', type=float, default=4092000,
                        help='Sample rate Hz (default: 4092000)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Signal duration seconds (default: 5.0)')
    parser.add_argument('--snr', type=float, default=30.0,
                        help='Signal-to-noise ratio dB (default: 30)')
    parser.add_argument('--doppler', type=float, default=1500.0,
                        help='Doppler shift Hz (default: 1500)')
    parser.add_argument('--code-phase', type=float, default=300.0,
                        help='Code phase offset chips (default: 300)')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='gnss-sdr timeout seconds (default: 30)')
    args = parser.parse_args()

    print("=" * 70)
    print("  GPS L1 C/A — Encode → Decode Loopback Self-Test")
    print("  TX: gps_engine.py | RX: gnss-sdr")
    print("=" * 70)

    # ── TEST 1: TX Encoding ──
    print(f"\n{'─' * 70}")
    print(f"[TEST 1] TX Signal Generation")
    print(f"{'─' * 70}")

    sig, ok_len, ok_power, ok_code = test_tx_encoding(
        args.prn, args.samp_rate, args.duration,
        args.snr, args.doppler, args.code_phase)

    # ── TEST 2: gnss-sdr Decode ──
    print(f"\n{'─' * 70}")
    print(f"[TEST 2] gnss-sdr Decode (acquisition + tracking)")
    print(f"{'─' * 70}")

    if HAS_GNSS_SDR:
        ok_acq, ok_trk, cn0 = test_gnss_sdr_decode(
            sig, args.prn, args.samp_rate, timeout_s=args.timeout)
    else:
        print("[SKIP] gnss_decoder/gnss_monitor not available")
        ok_acq = ok_trk = cn0 = None

    # ── SUMMARY ──
    print(f"\n{'═' * 70}")
    print("  SELF-TEST RESULTS")
    print(f"{'═' * 70}")

    tests = [
        ("TX signal length", ok_len, "correct sample count"),
        ("TX signal power", ok_power, "valid power level"),
        ("C/A Gold code", ok_code, "1023 chips, float32"),
    ]

    if ok_acq is not None:
        tests.append(("gnss-sdr acquisition", ok_acq,
                       f"PRN {args.prn} {'detected' if ok_acq else 'NOT found'}"))
    if ok_trk is not None:
        tests.append(("gnss-sdr tracking", ok_trk,
                       f"C/N₀={cn0:.1f} dB-Hz" if cn0 else "no lock"))

    all_pass = True
    for name, passed, detail in tests:
        if passed is None:
            mark = "SKIP"
        elif passed:
            mark = "PASS ✓"
        else:
            mark = "FAIL ✗"
            all_pass = False
        print(f"  {mark:8s} {name:25s} {detail}")

    if ok_acq is None and ok_trk is None:
        print(f"\n  TX tests passed. gnss-sdr tests skipped "
              f"(binary not found).")
        print(f"  Build gnss-sdr with: ./setup.sh\n")
    elif all_pass:
        print(f"\n  All tests PASSED — TX encoder + gnss-sdr decoder "
              f"pipeline working.\n")
    else:
        print(f"\n  Some tests FAILED — see details above.\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
