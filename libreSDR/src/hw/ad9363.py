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
        # Store for potential reconnect inside transmit_cyclic()
        self._last_tx_cfg = dict(
            freq_hz=int(freq_hz), gain_db=float(gain_db),
            sample_rate=int(sample_rate), rf_bw=int(rf_bw),
        )

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

    @staticmethod
    def _try_restart_iiod(uri: str, password: str = "analog") -> bool:
        """
        SSH to the LibreSDR and restart iiod to release a stuck DMA buffer.

        Called automatically when EBUSY persists after reconnect attempts.
        Only works for IP URIs with a reachable host and known SSH credentials.
        Default PlutoSDR/LibreSDR password is 'analog'.

        Returns True if iiod was successfully restarted.
        """
        if not uri.startswith("ip:"):
            return False
        host = uri.split(":", 1)[1]
        try:
            import pexpect
            cmd = (f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "
                   f"root@{host} "
                   f"\"killall iiod 2>/dev/null; sleep 1; iiod -D; echo IIOD_OK\"")
            child = pexpect.spawn(cmd, timeout=18)
            i = child.expect([r"password:", r"IIOD_OK", pexpect.TIMEOUT, pexpect.EOF])
            if i == 0:
                child.sendline(password)
                j = child.expect([r"IIOD_OK", pexpect.TIMEOUT, pexpect.EOF], timeout=14)
                ok = (j == 0)
            else:
                ok = (i == 1)
            if ok:
                print(f"  [RECOVER] iiod restarted on {host} via SSH.")
            return ok
        except Exception:
            return False

    def transmit_cyclic(self, samples: np.ndarray) -> None:
        """
        Transmit `samples` in a continuous hardware loop until Ctrl+C.

        The AD9363 DMA repeats the buffer autonomously; Python just waits.
        Buffer size is capped at HW_BUF_MAX (≈ 1 s @ 1 MSPS) to stay within
        the Ethernet IIO DMA pipeline limit.

        samples is automatically truncated to HW_BUF_MAX if larger.

        EBUSY recovery
        --------------
        A previous crash may have left the IIO DMA buffer locked in the iiod
        server.  The recovery sequence is:
          1. tx_destroy_buffer() + reconnect (new IIO context + 3 s wait)   × 2
          2. SSH restart of iiod on the device (releases the kernel DMA lock) × 1
          3. Reconnect + tx() → should succeed after iiod restart
        """
        import gc
        try:
            import adi as _adi
        except ImportError:
            _adi = None

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
        for _attempt in range(4):
            try:
                sdr.tx(samples)
                self._sdr = sdr   # commit new sdr if we reconnected
                break
            except OSError as exc:
                if exc.errno != 16 or _attempt >= 3:   # not EBUSY or all retries done
                    raise
                # On the second failed attempt, try the nuclear option: SSH iiod restart
                if _attempt == 1:
                    print("  [WARN] TX DMA buffer stuck — attempting iiod restart via SSH ...")
                    restarted = self._try_restart_iiod(self._uri)
                    if restarted:
                        time.sleep(3)   # let iiod initialise
                else:
                    print(f"  [WARN] TX buffer busy — reconnecting IIO ({_attempt}/2) ...")
                try:
                    sdr.tx_destroy_buffer()
                except Exception:
                    pass
                del sdr
                gc.collect()
                time.sleep(2)
                if _adi is None:
                    raise
                try:
                    sdr = _adi.Pluto(self._uri)
                    try:
                        sdr.tx_destroy_buffer()
                    except Exception:
                        pass
                    if hasattr(self, '_last_tx_cfg'):
                        c = self._last_tx_cfg
                        sdr.sample_rate            = c['sample_rate']
                        sdr.tx_rf_bandwidth        = c['rf_bw']
                        sdr.tx_lo                  = c['freq_hz']
                        sdr.tx_hardwaregain_chan0   = c['gain_db']
                    sdr.tx_cyclic_buffer = True
                except Exception as _e:
                    print(f"  [WARN] Reconnect failed: {_e}")
                    sdr = self._sdr
        print("  Cyclic TX active. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n  Stopping ...")
        finally:
            try:
                self._sdr.tx_destroy_buffer()
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

    def transmit_streaming(
        self,
        samples: np.ndarray,
        chunk_size: int = HW_BUF_MAX,
        loop: bool = False,
    ) -> None:
        """
        Stream a large IQ buffer to the DAC in sequential non-cyclic chunks.

        Unlike transmit_cyclic(), which requires the whole buffer to fit in
        the IIO DMA window (≤ HW_BUF_MAX = 1 048 576 samples), this method
        handles arbitrarily large buffers by splitting them into DMA-safe
        chunks and sending them back-to-back via the non-cyclic DAC FIFO.

        Typical use: simulated satellite passes (60 s @ 1 MSPS = 60 M samples).

        When loop=True, the full buffer is repeated until Ctrl+C is pressed.
        Each loop iteration re-transmits all chunks in order, so the full
        Doppler/amplitude envelope of a simulated satellite pass plays back
        from the beginning on every cycle.

        Args:
            samples:    Complex64 IQ array (arbitrary length).
            chunk_size: DMA chunk size in samples (default HW_BUF_MAX ≈ 1 s).
            loop:       If True, repeat the whole buffer until Ctrl+C.
        """
        sdr = self._sdr
        sdr.tx_cyclic_buffer = False
        n_total   = len(samples)
        dur_s     = n_total / float(DEFAULT_SAMPLE_RATE)
        chunks    = [samples[i : i + chunk_size]
                     for i in range(0, n_total, chunk_size)]
        n_chunks  = len(chunks)

        if loop:
            print(f"  Streaming {n_total} samples ({dur_s:.1f} s) "
                  f"in {n_chunks} chunk(s), looping until Ctrl+C ...")
        else:
            print(f"  Streaming {n_total} samples ({dur_s:.1f} s) "
                  f"in {n_chunks} chunk(s) ...")

        pass_n = 0
        try:
            while True:
                pass_n += 1
                if loop and pass_n > 1:
                    print(f"\n  [Pass {pass_n}]", end=" ", flush=True)
                for i, chunk in enumerate(chunks):
                    if n_chunks > 1:
                        pct = int(i / n_chunks * 100)
                        print(f"\r  [{pct:3d}%] chunk {i+1}/{n_chunks}",
                              end="", flush=True)
                    sdr.tx(chunk)
                if not loop:
                    break
        except KeyboardInterrupt:
            print("\n  Stop requested.")
        finally:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
        if n_chunks > 1:
            print()
        print("  TX streaming complete.")

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "Ad9363":
        return self

    def __exit__(self, *_) -> None:
        try:
            self._sdr.tx_destroy_buffer()
        except Exception:
            pass
