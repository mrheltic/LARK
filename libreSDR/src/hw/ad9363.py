#!/usr/bin/env python3
"""
hw/ad9363.py — LibreSDR / AD9363 hardware abstraction layer.

Single access point for all pyadi-iio operations.
All TX scripts in tx/ import from here — no direct adi.Pluto calls elsewhere.

Typical usage
-------------
    from hw.ad9363 import Ad9363

    with Ad9363.connect("ip:192.168.1.10") as sdr:
        sdr.configure_tx(freq_hz=868_100_000, gain_db=-30)
        sdr.transmit_cyclic(iq_samples)

    # One-liner for IRA bursts:
    with Ad9363.connect() as sdr:
        sdr.configure_tx(1_626_270_000, -60)
        sdr.transmit_once(iq_buf, n=4)
"""

from __future__ import annotations

import time
import numpy as np

# -- Hardware limits ---------------------------------------------------------
# Ethernet IIO DMA pipeline limit; larger buffers cause BrokenPipe on tx push.
HW_BUF_MAX: int = 2**20          # 1,048,576 samples ≈ 1.05 s @ 1 MSPS

# -- Defaults ----------------------------------------------------------------
DEFAULT_URI         = "ip:192.168.1.10"
DEFAULT_SAMPLE_RATE = 1_000_000   # 1 MSPS — minimum stable rate over Ethernet
DEFAULT_RF_BW       = 200_000     # 200 kHz — covers Iridium signal bandwidth

# AD9363 DDS core device names (board-dependent; tried in order)
_DDS_NAMES = [
    "cf-ad9361-dds-core-lpc",
    "cf-ad9361-dds-core-hpc",
    "axi-ad9361-dds-lpc",
    "axi-ad9361-dds-hpc",
]


class Ad9363:
    """
    Context-manager wrapper for a LibreSDR / AD9363 (pyadi-iio).

    Create via Ad9363.connect(), not directly.

    The AD9363 on LibreSDR uses Pluto-compatible firmware (adi.Pluto).
    """

    def __init__(self, sdr, uri: str) -> None:
        self._sdr = sdr
        self._uri = uri
        self._dds_ctx = None   # keeps the iio.Context alive during DDS mode

    # -- Construction --------------------------------------------------------

    @classmethod
    def connect(cls, uri: str = DEFAULT_URI, retries: int = 3) -> "Ad9363":
        """
        Connect to LibreSDR and return an Ad9363 instance.

        Retries up to `retries` times on EBUSY (device occupied by a stale
        session from a previous crash).

        Raises SystemExit on unrecoverable connection failure.
        """
        try:
            import adi
        except ImportError:
            raise ImportError(
                "pyadi-iio is required.\n"
                "  Install: pip install pyadi-iio"
            )

        print(f"  Connecting to {uri} ...", end=" ", flush=True)
        for attempt in range(retries):
            try:
                sdr = adi.Pluto(uri)
                # Destroy any stale buffer left by a previous session
                try:
                    sdr.tx_destroy_buffer()
                except Exception:
                    pass
                print("OK")
                return cls(sdr, uri)
            except OSError as exc:
                if exc.errno == 16 and attempt < retries - 1:  # EBUSY
                    print(f"busy ({attempt + 1}/{retries}), waiting 3 s ...",
                          end=" ", flush=True)
                    time.sleep(3)
                else:
                    print(f"FAILED\n[ERROR] {exc}")
                    host = uri.split(":", 1)[-1] if ":" in uri else uri
                    print(f"  ping {host}")
                    print(f"  iio_info -u {uri}")
                    raise SystemExit(1) from exc
            except Exception as exc:
                print(f"FAILED\n[ERROR] {exc}")
                raise SystemExit(1) from exc
        raise SystemExit(1)   # unreachable, silences type checkers

    # -- TX configuration ----------------------------------------------------

    def configure_tx(
        self,
        freq_hz: int,
        gain_db: float,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        rf_bw: int = DEFAULT_RF_BW,
    ) -> None:
        """
        Configure the AD9363 TX channel.

        Args:
            freq_hz:     TX LO frequency [Hz]
            gain_db:     TX attenuation [dB].  Range: 0 (max power) … −89.75 (min).
                         Start at −60 dB for first connection; increase cautiously.
            sample_rate: DAC sample rate [Hz] (default 1 MSPS)
            rf_bw:       TX RF bandwidth [Hz] (default 200 kHz)
        """
        self._sdr.sample_rate           = int(sample_rate)
        self._sdr.tx_rf_bandwidth       = int(rf_bw)
        self._sdr.tx_lo                 = int(freq_hz)
        self._sdr.tx_hardwaregain_chan0 = float(gain_db)

        print(f"  TX:  {freq_hz / 1e6:.3f} MHz  |  {sample_rate / 1e6:.1f} MSPS  "
              f"|  gain {gain_db:+.1f} dB  |  RF BW {rf_bw / 1e3:.0f} kHz")

    # -- RX configuration ----------------------------------------------------

    def configure_rx(
        self,
        freq_hz: int,
        gain_db: float = 30.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        rf_bw: int = DEFAULT_RF_BW,
        gain_mode: str = "slow_attack",
    ) -> None:
        """
        Configure the AD9363 RX channel for loopback / monitoring.

        Args:
            freq_hz:    RX LO frequency [Hz]
            gain_db:    RX gain [dB] (used only when gain_mode='manual')
            sample_rate:ADC sample rate [Hz]
            rf_bw:      RX RF bandwidth [Hz]
            gain_mode:  'manual', 'slow_attack', or 'fast_attack'
        """
        self._sdr.rx_lo                   = int(freq_hz)
        self._sdr.sample_rate             = int(sample_rate)
        self._sdr.rx_rf_bandwidth         = int(rf_bw)
        self._sdr.gain_control_mode_chan0  = gain_mode
        if gain_mode == "manual":
            self._sdr.rx_hardwaregain_chan0 = float(gain_db)

        print(f"  RX:  {freq_hz / 1e6:.3f} MHz  |  {sample_rate / 1e6:.1f} MSPS  "
              f"|  gain {gain_db:+.1f} dB ({gain_mode})")

    # -- CW / DDS mode (FPGA hardware tone, no buffer needed) ----------------

    def configure_dds_cw(
        self,
        offset_hz: int = 0,
        scale: float = 0.9,
    ) -> bool:
        """
        Configure the on-chip FPGA DDS to generate a CW tone at `offset_hz`
        above the TX LO.  No software buffer is required: the FPGA generates
        the tone autonomously.

        Args:
            offset_hz: Tone offset from LO [Hz].
                        0 = plain carrier at DC.
                        100_000 = pilot tone at LO + 100 kHz (KrakenSDR default).
            scale:     DDS full-scale [0 … 1.0].

        Returns:
            True if DDS was configured; False if the DDS core is unavailable.
            When False, fall back to transmit_cyclic() with a software CW buffer.

        Note: the returned value must not be discarded — this method stores the
        internal iio.Context in self._dds_ctx to keep the DDS alive.
        """
        try:
            import iio
        except ImportError:
            return False

        ctx = getattr(self._sdr, "_ctx", None)
        ctx_owned = False
        if ctx is None:
            try:
                ctx = iio.Context(self._uri)
                ctx_owned = True
            except Exception:
                return False

        dds = None
        for name in _DDS_NAMES:
            dds = ctx.find_device(name)
            if dds is not None:
                break
        if dds is None:
            return False

        # Quadrature CW:  altvoltage0 (I, 0°)  +  altvoltage2 (Q, +90°)
        cfgs = [
            ("altvoltage0", int(offset_hz), 0,      scale),   # I tone-1
            ("altvoltage1", 0,              0,      0.0  ),   # I tone-2 off
            ("altvoltage2", int(offset_hz), 90000,  scale),   # Q tone-1 (+90°)
            ("altvoltage3", 0,              0,      0.0  ),   # Q tone-2 off
        ]
        try:
            for ch_name, freq, phase_mdeg, sc in cfgs:
                ch = dds.find_channel(ch_name, is_output=True)
                if ch is None:
                    return False
                for attr, val in [("frequency", str(freq)),
                                   ("phase",     str(phase_mdeg)),
                                   ("scale",     f"{sc:.6f}"),
                                   ("raw",       "1" if sc > 0 else "0")]:
                    try:
                        ch.attrs[attr].value = val
                    except Exception:
                        pass
            # Keep context alive — DDS resets if the context is garbage-collected
            self._dds_ctx = ctx if ctx_owned else True
            return True
        except Exception:
            return False

    # -- Transmission --------------------------------------------------------

    def transmit_once(self, samples: np.ndarray, n: int = 1) -> None:
        """
        Push `samples` to the DAC exactly `n` times.

        For long sessions use transmit_cyclic() instead.
        """
        sdr = self._sdr
        sdr.tx_cyclic_buffer = False
        for i in range(n):
            sdr.tx(samples)
            if n > 1:
                print(f"  TX  {i + 1}/{n}")

    def transmit_cyclic(self, samples: np.ndarray) -> None:
        """
        Transmit `samples` in a continuous hardware loop until Ctrl+C.

        The AD9363 DMA repeats the buffer autonomously; Python just waits.
        Buffer size is capped at HW_BUF_MAX (≈ 1 s @ 1 MSPS) to stay within
        the Ethernet IIO DMA pipeline limit.

        samples is automatically truncated to HW_BUF_MAX if larger.
        """
        sdr = self._sdr
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass

        if len(samples) > HW_BUF_MAX:
            print(f"  [WARN] Buffer truncated {len(samples)} → {HW_BUF_MAX} samples "
                  f"(HW limit). Use n_slots to fit within 1 s.")
            samples = samples[:HW_BUF_MAX]

        sdr.tx_cyclic_buffer = True
        # Retry on EBUSY: a previous crash may have left the IIO buffer locked.
        for _attempt in range(4):
            try:
                sdr.tx(samples)
                break
            except OSError as exc:
                if exc.errno == 16 and _attempt < 3:  # EBUSY
                    print(f"  [WARN] TX buffer busy, retrying in 3 s ... ({_attempt + 1}/3)")
                    try:
                        sdr.tx_destroy_buffer()
                    except Exception:
                        pass
                    time.sleep(3)
                else:
                    raise
        print("  Cyclic TX active. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n  Stopping ...")
        finally:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            print("  TX buffer destroyed.")

    # -- Helpers -------------------------------------------------------------

    def scale_for_dac(
        self,
        samples: np.ndarray,
        full_scale: float = 0.8,
    ) -> np.ndarray:
        """
        Scale complex IQ samples to the AD9363 DAC range (±2^14 = ±16384).

        Args:
            samples:    Complex IQ array (arbitrary amplitude).
            full_scale: Fraction of DAC full-scale to use (0 < x ≤ 1.0).
                        0.8 leaves 2 dB headroom above the RMS target.

        Returns:
            complex64 array, DAC peak ≤ full_scale × 16384.
        """
        peak = float(np.max(np.abs(samples))) + 1e-12
        return (samples / peak * full_scale * 2**14).astype(np.complex64)

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "Ad9363":
        return self

    def __exit__(self, *_) -> None:
        try:
            self._sdr.tx_destroy_buffer()
        except Exception:
            pass
