#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnss_monitor.py — Real-time monitor client for gnss-sdr

Reads protobuf-encoded UDP streams from gnss-sdr's 4 monitoring ports:
  - Acquisition  (port 1231) : GnssSynchro messages
  - Tracking     (port 1232) : GnssSynchro messages
  - Observables  (port 1233) : GnssSynchro messages
  - PVT          (port 1234) : MonitorPvt messages

Requires gnss-sdr config with monitors enabled (see conf/ directory).

Usage:
  python3 scripts/gnss_monitor.py                    # all monitors
  python3 scripts/gnss_monitor.py --pvt-only         # PVT monitor only
  python3 scripts/gnss_monitor.py --port-acq 1231    # custom port

Hardware target: LibreSDR (Zynq7020 + AD9363)
"""

import sys
import os
import socket
import struct
import argparse
import threading
import time
import json
from collections import defaultdict

# Add proto directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'proto'))

try:
    import gnss_synchro_pb2
    import monitor_pvt_pb2
    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False
    print("[WARN] Protobuf modules not found. Run: "
          "protoc --python_out=scripts/proto "
          "--proto_path=gnss-sdr/docs/protobuf "
          "gnss-sdr/docs/protobuf/gnss_synchro.proto "
          "gnss-sdr/docs/protobuf/monitor_pvt.proto",
          file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# UDP Listener
# ─────────────────────────────────────────────────────────────────────────────

class UDPMonitorListener:
    """Listens on a UDP port and parses gnss-sdr protobuf monitoring messages."""

    def __init__(self, port, msg_type='gnss_synchro', bind_addr='127.0.0.1'):
        self.port = port
        self.msg_type = msg_type
        self.bind_addr = bind_addr
        self.sock = None
        self._running = False
        self._thread = None
        self.messages = []
        self.last_message = None
        self.msg_count = 0
        self._callback = None

    def set_callback(self, callback):
        self._callback = callback

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(1.0)
        self.sock.bind((self.bind_addr, self.port))
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.sock:
            self.sock.close()
            self.sock = None

    def _listen_loop(self):
        while self._running:
            try:
                data, addr = self.sock.recvfrom(8192)
                msg = self._parse(data)
                if msg is not None:
                    self.last_message = msg
                    self.msg_count += 1
                    if self._callback:
                        self._callback(self.msg_type, msg)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[{self.msg_type}:{self.port}] Error: {e}",
                          file=sys.stderr)

    def _parse(self, data):
        if not HAS_PROTOBUF:
            return None
        if self.msg_type == 'pvt':
            msg = monitor_pvt_pb2.MonitorPvt()
        else:
            msg = gnss_synchro_pb2.GnssSynchro()
        try:
            msg.ParseFromString(data)
            return msg
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Monitor Manager
# ─────────────────────────────────────────────────────────────────────────────

class GnssSdrMonitor:
    """Manages all 4 gnss-sdr UDP monitoring streams."""

    DEFAULT_PORTS = {
        'acquisition': 1231,
        'tracking': 1232,
        'observables': 1233,
        'pvt': 1234,
    }

    def __init__(self, ports=None, bind_addr='127.0.0.1'):
        self.ports = ports or dict(self.DEFAULT_PORTS)
        self.bind_addr = bind_addr
        self.listeners = {}
        self._acq_results = {}
        self._trk_state = {}
        self._pvt_solution = None
        self._obs_state = {}
        self._lock = threading.Lock()

    def start(self, monitors=None):
        if monitors is None:
            monitors = list(self.ports.keys())
        for name in monitors:
            port = self.ports[name]
            msg_type = 'pvt' if name == 'pvt' else 'gnss_synchro'
            listener = UDPMonitorListener(port, msg_type, self.bind_addr)
            listener.set_callback(self._on_message)
            listener.start()
            self.listeners[name] = listener

    def stop(self):
        for listener in self.listeners.values():
            listener.stop()
        self.listeners.clear()

    def _on_message(self, msg_type, msg):
        with self._lock:
            if msg_type == 'pvt':
                self._pvt_solution = msg
            elif msg_type == 'gnss_synchro':
                key = (msg.system, msg.signal, msg.prn)
                if hasattr(msg, 'flag_valid_acquisition') and msg.flag_valid_acquisition:
                    self._acq_results[key] = msg
                if hasattr(msg, 'flag_valid_symbol_output') and msg.flag_valid_symbol_output:
                    self._trk_state[key] = msg
                if hasattr(msg, 'flag_valid_pseudorange') and msg.flag_valid_pseudorange:
                    self._obs_state[key] = msg

    def get_acquisition_results(self):
        with self._lock:
            return dict(self._acq_results)

    def get_tracking_state(self):
        with self._lock:
            return dict(self._trk_state)

    def get_observables(self):
        with self._lock:
            return dict(self._obs_state)

    def get_pvt(self):
        with self._lock:
            return self._pvt_solution

    def get_status_summary(self):
        with self._lock:
            return {
                'acquired': len(self._acq_results),
                'tracking': len(self._trk_state),
                'observables': len(self._obs_state),
                'pvt_valid': self._pvt_solution is not None,
                'msg_counts': {
                    name: l.msg_count
                    for name, l in self.listeners.items()
                },
            }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_synchro(msg):
    """Format a GnssSynchro message for display."""
    parts = [f"{msg.system}{msg.signal} PRN{msg.prn:02d} ch{msg.channel_id}"]
    if msg.flag_valid_acquisition:
        parts.append(f"ACQ: delay={msg.acq_delay_samples:.0f}samp "
                     f"Doppler={msg.acq_doppler_hz:+.0f}Hz")
    if msg.flag_valid_symbol_output:
        parts.append(f"TRK: C/N₀={msg.cn0_db_hz:.1f}dB-Hz "
                     f"Doppler={msg.carrier_doppler_hz:+.1f}Hz "
                     f"I={msg.prompt_i:.0f} Q={msg.prompt_q:.0f}")
    if msg.flag_valid_word:
        parts.append(f"NAV: TOW={msg.tow_at_current_symbol_ms}ms")
    if msg.flag_valid_pseudorange:
        parts.append(f"OBS: PR={msg.pseudorange_m:.3f}m "
                     f"rx_t={msg.rx_time:.6f}s")
    return " | ".join(parts)


def format_pvt(msg):
    """Format a MonitorPvt message for display."""
    lines = [
        f"PVT Solution — {msg.utc_time}",
        f"  Position: {msg.latitude:.7f}°N  {msg.longitude:.7f}°E  "
        f"{msg.height:.2f}m",
        f"  ECEF: X={msg.pos_x:.3f}  Y={msg.pos_y:.3f}  Z={msg.pos_z:.3f} m",
        f"  Velocity: E={msg.vel_e:.3f}  N={msg.vel_n:.3f}  U={msg.vel_u:.3f} m/s",
        f"  DOP: GDOP={msg.gdop:.2f}  PDOP={msg.pdop:.2f}  "
        f"HDOP={msg.hdop:.2f}  VDOP={msg.vdop:.2f}",
        f"  Satellites: {msg.valid_sats}  "
        f"Clock offset: {msg.user_clk_offset*1e9:.1f}ns  "
        f"Drift: {msg.user_clk_drift_ppm:.3f}ppm",
    ]
    return "\n".join(lines)


def pvt_to_dict(msg):
    """Convert MonitorPvt to a plain dict (for JSON serialization)."""
    return {
        'utc_time': msg.utc_time,
        'latitude': msg.latitude,
        'longitude': msg.longitude,
        'height': msg.height,
        'pos_x': msg.pos_x, 'pos_y': msg.pos_y, 'pos_z': msg.pos_z,
        'vel_x': msg.vel_x, 'vel_y': msg.vel_y, 'vel_z': msg.vel_z,
        'vel_e': msg.vel_e, 'vel_n': msg.vel_n, 'vel_u': msg.vel_u,
        'gdop': msg.gdop, 'pdop': msg.pdop, 'hdop': msg.hdop, 'vdop': msg.vdop,
        'valid_sats': msg.valid_sats,
        'user_clk_offset': msg.user_clk_offset,
        'user_clk_drift_ppm': msg.user_clk_drift_ppm,
        'week': msg.week,
        'tow_ms': msg.tow_at_current_symbol_ms,
        'geohash': msg.geohash,
    }


def synchro_to_dict(msg):
    """Convert GnssSynchro to a plain dict."""
    return {
        'system': msg.system,
        'signal': msg.signal,
        'prn': msg.prn,
        'channel_id': msg.channel_id,
        'acq_delay_samples': msg.acq_delay_samples,
        'acq_doppler_hz': msg.acq_doppler_hz,
        'cn0_db_hz': msg.cn0_db_hz,
        'carrier_doppler_hz': msg.carrier_doppler_hz,
        'carrier_phase_rads': msg.carrier_phase_rads,
        'code_phase_samples': msg.code_phase_samples,
        'prompt_i': msg.prompt_i,
        'prompt_q': msg.prompt_q,
        'tow_at_current_symbol_ms': msg.tow_at_current_symbol_ms,
        'pseudorange_m': msg.pseudorange_m,
        'flag_valid_acquisition': msg.flag_valid_acquisition,
        'flag_valid_symbol_output': msg.flag_valid_symbol_output,
        'flag_valid_word': msg.flag_valid_word,
        'flag_valid_pseudorange': msg.flag_valid_pseudorange,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI — Live Monitor Display
# ─────────────────────────────────────────────────────────────────────────────

def run_live_display(monitor, refresh_s=1.0):
    """Real-time terminal display of gnss-sdr status."""
    CLEAR = '\033[2J\033[H'
    BOLD = '\033[1m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    RESET = '\033[0m'

    try:
        while True:
            time.sleep(refresh_s)
            status = monitor.get_status_summary()
            acq = monitor.get_acquisition_results()
            trk = monitor.get_tracking_state()
            pvt = monitor.get_pvt()

            out = [CLEAR]
            out.append(f"{BOLD}═══ gnss-sdr Monitor "
                       f"({time.strftime('%H:%M:%S')}) ═══{RESET}\n")

            # Message counts
            counts = status['msg_counts']
            out.append(f"  Messages: ACQ={counts.get('acquisition', 0)}  "
                       f"TRK={counts.get('tracking', 0)}  "
                       f"OBS={counts.get('observables', 0)}  "
                       f"PVT={counts.get('pvt', 0)}\n")

            # Acquisition
            if acq:
                out.append(f"\n{CYAN}── Acquisition ({len(acq)} satellites) "
                           f"──{RESET}")
                for key, msg in sorted(acq.items()):
                    out.append(f"  {msg.system}{msg.signal} PRN{msg.prn:02d}: "
                               f"Doppler={msg.acq_doppler_hz:+8.0f} Hz  "
                               f"delay={msg.acq_delay_samples:.0f} samp")

            # Tracking
            if trk:
                out.append(f"\n{YELLOW}── Tracking ({len(trk)} channels) "
                           f"──{RESET}")
                for key, msg in sorted(trk.items()):
                    pll_lock = "PLL180" if msg.flag_PLL_180_deg_phase_locked else ""
                    nav_ok = "NAV✓" if msg.flag_valid_word else ""
                    out.append(
                        f"  {msg.system}{msg.signal} PRN{msg.prn:02d} "
                        f"ch{msg.channel_id}: "
                        f"C/N₀={msg.cn0_db_hz:5.1f} dB-Hz  "
                        f"Doppler={msg.carrier_doppler_hz:+8.1f} Hz  "
                        f"{pll_lock} {nav_ok}")

            # PVT
            if pvt:
                out.append(f"\n{GREEN}── PVT Solution ──{RESET}")
                out.append(f"  {pvt.utc_time}")
                out.append(f"  Lat={pvt.latitude:.7f}°  "
                           f"Lon={pvt.longitude:.7f}°  "
                           f"H={pvt.height:.2f}m")
                out.append(f"  Sats={pvt.valid_sats}  "
                           f"HDOP={pvt.hdop:.2f}  VDOP={pvt.vdop:.2f}")
            else:
                out.append(f"\n{RED}── PVT: waiting for fix... ──{RESET}")

            print("\n".join(out))

    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='gnss-sdr real-time UDP monitor client (protobuf)')
    parser.add_argument('--port-acq', type=int, default=1231,
                        help='Acquisition monitor UDP port (default: 1231)')
    parser.add_argument('--port-trk', type=int, default=1232,
                        help='Tracking monitor UDP port (default: 1232)')
    parser.add_argument('--port-obs', type=int, default=1233,
                        help='Observables monitor UDP port (default: 1233)')
    parser.add_argument('--port-pvt', type=int, default=1234,
                        help='PVT monitor UDP port (default: 1234)')
    parser.add_argument('--bind', type=str, default='127.0.0.1',
                        help='Bind address (default: 127.0.0.1)')
    parser.add_argument('--pvt-only', action='store_true',
                        help='Monitor PVT only')
    parser.add_argument('--json', type=str, default=None,
                        help='Log PVT solutions to JSON file')
    parser.add_argument('--refresh', type=float, default=1.0,
                        help='Display refresh rate in seconds')
    args = parser.parse_args()

    if not HAS_PROTOBUF:
        print("ERROR: protobuf Python modules required.", file=sys.stderr)
        print("Install: pip install protobuf", file=sys.stderr)
        print("Compile: protoc --python_out=scripts/proto "
              "--proto_path=gnss-sdr/docs/protobuf "
              "gnss-sdr/docs/protobuf/*.proto", file=sys.stderr)
        sys.exit(1)

    ports = {
        'acquisition': args.port_acq,
        'tracking': args.port_trk,
        'observables': args.port_obs,
        'pvt': args.port_pvt,
    }

    monitors = ['pvt'] if args.pvt_only else None

    monitor = GnssSdrMonitor(ports=ports, bind_addr=args.bind)
    monitor.start(monitors=monitors)

    print(f"gnss-sdr Monitor — listening on UDP ports "
          f"{', '.join(f'{n}:{p}' for n, p in ports.items())}")
    print("Press Ctrl+C to exit.\n")

    json_file = None
    if args.json:
        json_file = open(args.json, 'w')
        json_file.write('[\n')

    try:
        pvt_count = 0
        run_live_display(monitor, refresh_s=args.refresh)
    finally:
        if json_file:
            # Write accumulated PVT solutions
            pvt = monitor.get_pvt()
            if pvt:
                json_file.write(json.dumps(pvt_to_dict(pvt), indent=2))
            json_file.write('\n]\n')
            json_file.close()
            print(f"\nPVT log saved to {args.json}")

        monitor.stop()
        print("\nMonitor stopped.")


if __name__ == '__main__':
    main()
