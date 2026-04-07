#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnss_decoder.py — GNSS Decoder using gnss-sdr

Wrapper around the gnss-sdr binary for signal decoding. Uses gnss-sdr's
full C++ signal processing pipeline:
  Signal Source → Signal Conditioner → Acquisition → Tracking →
  Telemetry Decoder → Observables → PVT

Reads IQ samples from:
  - File (recorded from gnss_lab.py)    → gnss-sdr File_Signal_Source
  - Real-time from AD9363               → gnss-sdr Fmcomms2_Signal_Source

Monitors gnss-sdr's status via UDP protobuf on ports 1231-1234.

Usage:
  python3 gnss_decoder.py --file data/capture.dat
  python3 gnss_decoder.py --live
  python3 gnss_decoder.py --live --multi

Hardware target: LibreSDR (Zynq7020 + AD9363)
"""

import sys
import os
import argparse
import subprocess
import signal
import time
import shutil
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CONF_DIR = os.path.join(PROJECT_DIR, 'conf')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Import the monitor client for real-time status
sys.path.insert(0, SCRIPT_DIR)
try:
    from gnss_monitor import GnssSdrMonitor, format_pvt, pvt_to_dict
    HAS_MONITOR = True
except ImportError:
    HAS_MONITOR = False


def find_gnss_sdr():
    """Locate the gnss-sdr binary."""
    candidates = [
        shutil.which('gnss-sdr'),
        os.path.join(PROJECT_DIR, 'gnss-sdr', 'build', 'src', 'main',
                     'gnss-sdr'),
        '/usr/local/bin/gnss-sdr',
        '/usr/bin/gnss-sdr',
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def generate_temp_config(base_conf, overrides):
    """Create a temporary config file with overrides applied."""
    with open(base_conf, 'r') as f:
        lines = f.readlines()

    for key, value in overrides.items():
        found = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(';') or '=' not in stripped:
                continue
            conf_key = stripped.split('=', 1)[0].strip()
            if conf_key == key:
                lines[i] = f"{key}={value}\n"
                found = True
                break
        if not found:
            lines.append(f"{key}={value}\n")

    fd, tmp_path = tempfile.mkstemp(suffix='.conf', prefix='gnss_sdr_')
    with os.fdopen(fd, 'w') as f:
        f.writelines(lines)
    return tmp_path


def run_gnss_sdr(config_file, extra_args=None, monitor=True,
                 verbose=True):
    """Launch gnss-sdr and optionally monitor its progress.

    Args:
        config_file: Path to gnss-sdr .conf file
        extra_args: Additional --Key=Value command-line overrides
        monitor: If True, start UDP monitor for real-time status
        verbose: Print gnss-sdr stdout/stderr

    Returns:
        dict with results summary
    """
    gnss_sdr_bin = find_gnss_sdr()
    if gnss_sdr_bin is None:
        print("ERROR: gnss-sdr binary not found.", file=sys.stderr)
        print("Build it with: ./setup.sh (includes gnss-sdr build)",
              file=sys.stderr)
        print("Or install: sudo apt install gnss-sdr", file=sys.stderr)
        sys.exit(1)

    cmd = [gnss_sdr_bin, f'--config_file={config_file}']
    if extra_args:
        cmd.extend(extra_args)

    print(f"[gnss-sdr] Binary: {gnss_sdr_bin}")
    print(f"[gnss-sdr] Config: {config_file}")
    print(f"[gnss-sdr] Command: {' '.join(cmd)}")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Start monitoring if available
    mon = None
    if monitor and HAS_MONITOR:
        mon = GnssSdrMonitor()
        mon.start()
        print("[Monitor] Listening on UDP ports 1231-1234")

    results = {
        'config': config_file,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pvt_solutions': [],
        'satellites_tracked': [],
        'return_code': None,
    }

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.STDOUT if not verbose else None,
        cwd=PROJECT_DIR,
    )

    def signal_handler(sig, frame):
        print("\n[gnss-sdr] Sending SIGINT...")
        proc.send_signal(signal.SIGINT)

    prev_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        last_pvt_count = 0
        while proc.poll() is None:
            time.sleep(1.0)

            if mon:
                status = mon.get_status_summary()
                pvt = mon.get_pvt()

                pvt_count = status['msg_counts'].get('pvt', 0)
                if pvt and pvt_count > last_pvt_count:
                    last_pvt_count = pvt_count
                    results['pvt_solutions'].append(pvt_to_dict(pvt))

                trk = mon.get_tracking_state()
                n_trk = len(trk)
                if n_trk > 0:
                    sats = [f"{m.system}{m.signal} PRN{m.prn}"
                            for m in trk.values()]
                    results['satellites_tracked'] = list(set(sats))

        results['return_code'] = proc.returncode

    finally:
        signal.signal(signal.SIGINT, prev_handler)
        if mon:
            pvt = mon.get_pvt()
            if pvt:
                print(f"\n{format_pvt(pvt)}")
            mon.stop()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='GNSS Decoder — gnss-sdr wrapper for LibreSDR '
                    '(Zynq7020 + AD9363)')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--file', type=str,
                        help='Input IQ file (gr_complex format)')
    source.add_argument('--live', action='store_true',
                        help='Real-time decoding from AD9363')

    parser.add_argument('--multi', action='store_true',
                        help='Multi-constellation (GPS L1 + Galileo E1)')
    parser.add_argument('--config', type=str, default=None,
                        help='Custom gnss-sdr config file (overrides defaults)')
    parser.add_argument('--samp-rate', type=float, default=None,
                        help='Override sampling rate (Hz)')
    parser.add_argument('--gain', type=float, default=None,
                        help='Override RX gain (dB, 0-73)')
    parser.add_argument('--uri', type=str, default='192.168.1.10',
                        help='IIO device address (default: 192.168.1.10)')
    parser.add_argument('--no-monitor', action='store_true',
                        help='Disable UDP monitoring')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress gnss-sdr stdout')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    print("=" * 70)
    print("  GNSS Decoder — gnss-sdr — LibreSDR (Zynq7020 + AD9363)")
    print("=" * 70)

    # Select config file
    if args.config:
        conf_path = args.config
    elif args.file:
        conf_path = os.path.join(CONF_DIR, 'gnss_sdr_file_source.conf')
    elif args.multi:
        conf_path = os.path.join(CONF_DIR, 'gnss_sdr_fmcomms2_multi.conf')
    else:
        conf_path = os.path.join(CONF_DIR, 'gnss_sdr_fmcomms2_gps_l1.conf')

    if not os.path.isfile(conf_path):
        print(f"ERROR: Config not found: {conf_path}", file=sys.stderr)
        sys.exit(1)

    # Build config overrides
    overrides = {}
    extra_args = []

    if args.file:
        abs_file = os.path.abspath(args.file)
        if not os.path.isfile(abs_file):
            print(f"ERROR: IQ file not found: {abs_file}", file=sys.stderr)
            sys.exit(1)
        overrides['SignalSource.filename'] = abs_file
        file_size = os.path.getsize(abs_file)
        print(f"\n[SRC] File: {abs_file}")
        print(f"      Size: {file_size / 1e6:.1f} MB")
    elif args.live:
        overrides['SignalSource.device_address'] = args.uri
        print(f"\n[SRC] Live AD9363 at {args.uri}")

    if args.samp_rate:
        overrides['SignalSource.sampling_frequency'] = str(int(args.samp_rate))
        overrides['GNSS-SDR.internal_fs_sps'] = str(int(args.samp_rate))

    if args.gain is not None:
        overrides['SignalSource.gain_rx1'] = str(args.gain)

    # Generate temp config if overrides needed
    if overrides:
        tmp_conf = generate_temp_config(conf_path, overrides)
        print(f"[CFG] Base: {conf_path}")
        print(f"[CFG] Overrides: {overrides}")
    else:
        tmp_conf = conf_path

    try:
        results = run_gnss_sdr(
            tmp_conf,
            extra_args=extra_args,
            monitor=not args.no_monitor,
            verbose=not args.quiet,
        )
    finally:
        if overrides and os.path.exists(tmp_conf):
            os.unlink(tmp_conf)

    # Summary
    print(f"\n{'═' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Return code: {results['return_code']}")
    print(f"  PVT solutions: {len(results['pvt_solutions'])}")
    if results['satellites_tracked']:
        print(f"  Satellites tracked: "
              f"{', '.join(results['satellites_tracked'])}")
    if results['pvt_solutions']:
        last = results['pvt_solutions'][-1]
        print(f"  Last position: {last['latitude']:.7f}°N "
              f"{last['longitude']:.7f}°E  H={last['height']:.2f}m")
        print(f"  Satellites: {last['valid_sats']}  "
              f"HDOP={last['hdop']:.2f}")

    # Check for output files
    nmea_files = [f for f in os.listdir(DATA_DIR)
                  if f.endswith('.nmea')] if os.path.isdir(DATA_DIR) else []
    rinex_files = [f for f in os.listdir(PROJECT_DIR)
                   if f.endswith(('.obs', '.nav', '.22o', '.22n'))]
    if nmea_files:
        print(f"\n  NMEA output: {', '.join(nmea_files)}")
    if rinex_files:
        print(f"  RINEX output: {', '.join(rinex_files[:5])}")

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.output}")

    print()


if __name__ == '__main__':
    main()
