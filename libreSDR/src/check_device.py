#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_device.py -- AD9363 connection diagnostic for LibreSDR (Zynq7020)

Verifies the IIO connection to the AD9363/AD9361 device and prints useful
information: devices, channels, attributes, LO frequency, TX gain.

Usage:
    python3 check_device.py [uri]

    URI examples:
      ip:192.168.2.1   Ethernet (default -- ADALM-Pluto / LibreSDR)
      ip:192.168.1.100 Custom Ethernet address
      usb:1.3.5        USB (find URI with: iio_info -s)
      local:           Run directly on the Zynq ARM core

Requires:
    python3-libiio   (sudo apt install python3-libiio)
    OR, for the shell fallback only:
    libiio-utils     (sudo apt install libiio-utils)
"""

import sys
import subprocess


# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_URI = "ip:192.168.2.1"

# Known PHY device names (depends on the board device tree)
PHY_DEVICE_NAMES = [
    "ad9363-phy",
    "ad9361-phy",
    "ad9364-phy",
]

# Known DDS FPGA core device names
DDS_DEVICE_NAMES = [
    "cf-ad9361-dds-core-lpc",
    "cf-ad9361-dds-core-hpc",
    "axi-ad9361-dds-hpc",
    "axi-ad9361-dds-lpc",
]

# Known ADC/DAC streaming device names
DAC_DEVICE_NAMES = [
    "cf-ad9361-lpc",
    "cf-ad9361-hpc",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_via_shell(uri: str) -> None:
    """Fallback: use iio_info from libiio-utils (no Python bindings required)."""
    print(f"\n[SHELL] Trying iio_info -u {uri} ...\n")
    try:
        result = subprocess.run(
            ["iio_info", "-u", uri],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(result.stdout[:4000])
        else:
            print(f"[ERROR] iio_info returned: {result.returncode}")
            print(result.stderr)
    except FileNotFoundError:
        print("[ERROR] iio_info not found.")
        print("  Install with:  sudo apt install libiio-utils")
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Connection timeout to {uri}")
        print("  Check:")
        print("  1. Is the device powered on?")
        print("  2. Is the URI correct?")
        print("  3. For USB: run  iio_info -s  to find the exact URI")


def _try_scan_usb() -> None:
    """Attempt to find IIO devices via USB scan."""
    print("\n[SCAN] Scanning for IIO devices on USB ...")
    try:
        result = subprocess.run(
            ["iio_info", "-s"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("  No devices found.")
            print(result.stderr)
    except FileNotFoundError:
        print("  iio_info not available.")
    except subprocess.TimeoutExpired:
        print("  Scan timeout.")


# ── Main class ────────────────────────────────────────────────────────────────

class AD9363Checker:
    """Connects to an AD9363/AD9361 device via libiio and reports its state."""

    def __init__(self, uri: str):
        self.uri = uri
        self.ctx = None
        self.phy = None
        self.dds = None
        self.dac = None

    def connect(self) -> bool:
        try:
            import iio
        except ImportError:
            print("[!] python3-libiio not installed.")
            print("    Install with:  sudo apt install python3-libiio")
            print("    Falling back to iio_info ...\n")
            _check_via_shell(self.uri)
            return False

        print(f"  Connecting to: {self.uri}")
        try:
            self.ctx = iio.Context(self.uri)
        except Exception as e:
            print(f"[ERROR] Cannot connect: {e}")
            print("\nTroubleshooting:")
            host = self.uri.split(":", 1)[-1] if ":" in self.uri else self.uri
            print(f"  - Ethernet: ping {host}")
            print("  - USB: run  iio_info -s  to list available URIs")
            print("  - Verify the device is powered on and reachable")
            _try_scan_usb()
            return False

        print(f"[OK] IIO context: {self.ctx.description}")
        return True

    def list_devices(self) -> None:
        if not self.ctx:
            return
        devices = list(self.ctx.devices)
        print(f"\n{'─'*60}")
        print(f"  IIO devices found: {len(devices)}")
        print(f"{'─'*60}")
        for dev in devices:
            marker = ""
            if dev.name in PHY_DEVICE_NAMES:
                marker = "  <- AD936x PHY ✓"
                self.phy = dev
            elif dev.name in DDS_DEVICE_NAMES:
                marker = "  <- DDS Core ✓"
                self.dds = dev
            elif dev.name in DAC_DEVICE_NAMES:
                marker = "  <- DAC/ADC Stream ✓"
                self.dac = dev
            print(f"  [{dev.id}] {dev.name or '(unnamed)'}{marker}")

    def check_phy(self) -> None:
        if not self.phy:
            print("\n[WARNING] AD936x PHY not found with standard device names.")
            print("  Expected names:", PHY_DEVICE_NAMES)
            print("  Check your board device tree.")
            return

        print(f"\n{'─'*60}")
        print(f"  AD9363 PHY: {self.phy.name}")
        print(f"{'─'*60}")

        # TX LO
        try:
            tx_lo = self.phy.find_channel("altvoltage1", output=True)
            if tx_lo:
                freq_hz = int(tx_lo.attrs["frequency"].value)
                print(f"  TX LO frequency:   {freq_hz / 1e6:.3f} MHz")
        except Exception as e:
            print(f"  [WARN] TX LO not readable: {e}")

        # RX LO
        try:
            rx_lo = self.phy.find_channel("altvoltage0", output=True)
            if rx_lo:
                freq_hz = int(rx_lo.attrs["frequency"].value)
                print(f"  RX LO frequency:   {freq_hz / 1e6:.3f} MHz")
        except Exception as e:
            print(f"  [WARN] RX LO not readable: {e}")

        # TX channel attributes (attenuation, sample rate, bandwidth)
        try:
            tx_ch = self.phy.find_channel("voltage0", output=True)
            if tx_ch:
                atten = tx_ch.attrs.get("hardwaregain")
                srate = tx_ch.attrs.get("sampling_frequency")
                bw    = tx_ch.attrs.get("rf_bandwidth")
                if atten:
                    print(f"  TX hardware gain:  {atten.value} dB")
                if srate:
                    print(f"  TX sample rate:    {int(srate.value) / 1e6:.3f} MSPS")
                if bw:
                    print(f"  TX RF bandwidth:   {int(bw.value) / 1e3:.0f} kHz")
        except Exception as e:
            print(f"  [WARN] TX voltage0 not readable: {e}")

    def check_dds(self) -> None:
        if not self.dds:
            print("\n[WARNING] DDS Core not found with standard device names.")
            print("  Expected names:", DDS_DEVICE_NAMES)
            return

        print(f"\n{'─'*60}")
        print(f"  DDS Core: {self.dds.name}")
        print(f"{'─'*60}")
        for ch_name in ["TX1_I_F1", "TX1_I_F2", "TX1_Q_F1", "TX1_Q_F2"]:
            ch = self.dds.find_channel(ch_name, output=True)
            if ch:
                try:
                    freq  = ch.attrs.get("frequency")
                    scale = ch.attrs.get("scale")
                    phase = ch.attrs.get("phase")
                    raw   = ch.attrs.get("raw")
                    print(f"  {ch_name}: raw={raw.value if raw else '?'}, "
                          f"freq={freq.value if freq else '?'} Hz, "
                          f"scale={scale.value if scale else '?'}, "
                          f"phase={phase.value if phase else '?'} deg")
                except Exception as e:
                    print(f"  {ch_name}: [read error: {e}]")

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"  PHY found:  {'YES -- ' + self.phy.name if self.phy else 'NO'}")
        print(f"  DDS found:  {'YES -- ' + self.dds.name if self.dds else 'NO'}")
        print(f"  DAC found:  {'YES -- ' + self.dac.name if self.dac else 'NO'}")
        if self.phy and self.dds:
            print("\n  OK  Device ready for CW transmission!")
        elif self.phy:
            print("\n  WARNING  PHY found but DDS not available.")
            print("    Use the GNU Radio flowgraph (fmcomms2_sink_fc32).")
        else:
            print("\n  ERROR  Device not properly detected.")
        print(f"{'='*60}\n")

    def run(self) -> None:
        print(f"\n{'='*60}")
        print("  LibreSDR AD9363 -- Connection Check")
        print(f"{'='*60}\n")
        if not self.connect():
            return
        self.list_devices()
        self.check_phy()
        self.check_dds()
        self.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URI
    checker = AD9363Checker(uri)
    checker.run()


if __name__ == "__main__":
    main()
