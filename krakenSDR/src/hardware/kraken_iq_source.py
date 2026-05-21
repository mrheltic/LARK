#!/usr/bin/env python3
"""
KrakenIQSource – Standalone IQ reader for Heimdall DAQ
=======================================================
Receives IQ frames directly from the Heimdall server via TCP with no
dependency on GNU Radio. Implements the same protocol as krakensdr_source.py.

Protocol:
    Data port (default 5000):
        1. client sends  b"streaming"
        2. client sends  b"IQDownload"  each time it wants a frame
        3. server responds: 1024-byte header + complex64 payload
    Control port (default 5001):
        - INIT  (128 bytes)
        - FREQ  (cmd 4B + freq uint64 8B + padding)  -> set frequency
        - GAIN  (cmd 4B + gains uint32*Nr 4B*Nr + padding) -> set gain
        - EXIT  (128 bytes) -> close

Usage:
    src = KrakenIQSource(host="127.0.0.1", port=5000, ctrl_port=5001,
                         num_channels=3, freq_hz=868e6, gain_db=30.0)
    src.start()
    frame = src.get_frame(timeout=2.0)  # ndarray (3, N) complex64 or None
    src.stop()

    # Or as a context manager:
    with KrakenIQSource(...) as src:
        frame = src.get_frame()
"""

import socket
import queue
import threading
import time
import numpy as np
from struct import pack, unpack


# =============================================================================
# IQ Frame Header (1024 bytes) – same decoder as krakensdr_source.py
# =============================================================================

class IQHeader:
    FRAME_TYPE_DATA  = 0
    FRAME_TYPE_DUMMY = 1
    FRAME_TYPE_RAMP  = 2
    FRAME_TYPE_CAL   = 3
    FRAME_TYPE_TRIGW = 4

    SYNC_WORD = 0x2bf7b95a
    HEADER_SIZE = 1024
    RESERVED_BYTES = 192

    def __init__(self):
        self.sync_word           = self.SYNC_WORD
        self.frame_type          = 0
        self.hardware_id         = ""
        self.unit_id             = 0
        self.active_ant_chs      = 0
        self.ioo_type            = 0
        self.rf_center_freq      = 0
        self.adc_sampling_freq   = 0
        self.sampling_freq       = 0
        self.cpi_length          = 0
        self.time_stamp          = 0
        self.daq_block_index     = 0
        self.cpi_index           = 0
        self.ext_integration_cntr = 0
        self.data_type           = 0
        self.sample_bit_depth    = 0
        self.adc_overdrive_flags = 0
        self.if_gains            = [0] * 32
        self.delay_sync_flag     = 0
        self.iq_sync_flag        = 0
        self.sync_state          = 0
        self.noise_source_state  = 0
        self.reserved            = [0] * self.RESERVED_BYTES
        self.header_version      = 0

    def decode(self, raw: bytes):
        fmt = "II16sIIIQQQIQIIQIII" + "I"*32 + "IIII" + "I"*self.RESERVED_BYTES + "I"
        lst = unpack(fmt, raw)
        self.sync_word            = lst[0]
        self.frame_type           = lst[1]
        self.hardware_id          = lst[2].decode(errors="replace").strip("\x00")
        self.unit_id              = lst[3]
        self.active_ant_chs       = lst[4]
        self.ioo_type             = lst[5]
        self.rf_center_freq       = lst[6]
        self.adc_sampling_freq    = lst[7]
        self.sampling_freq        = lst[8]
        self.cpi_length           = lst[9]
        self.time_stamp           = lst[10]
        self.daq_block_index      = lst[11]
        self.cpi_index            = lst[12]
        self.ext_integration_cntr = lst[13]
        self.data_type            = lst[14]
        self.sample_bit_depth     = lst[15]
        self.adc_overdrive_flags  = lst[16]
        self.if_gains             = list(lst[17:49])
        self.delay_sync_flag      = lst[49]
        self.iq_sync_flag         = lst[50]
        self.sync_state           = lst[51]
        self.noise_source_state   = lst[52]
        self.header_version       = lst[52 + self.RESERVED_BYTES + 1]

    @property
    def payload_bytes(self) -> int:
        return (self.active_ant_chs * self.cpi_length * 2
                * (self.sample_bit_depth // 8))

    @property
    def is_data(self) -> bool:
        return self.frame_type == self.FRAME_TYPE_DATA

    def __repr__(self):
        types = {0:"DATA", 1:"DUMMY", 2:"RAMP", 3:"CAL", 4:"TRIGW"}
        return (f"IQHeader(type={types.get(self.frame_type, self.frame_type)}, "
                f"chs={self.active_ant_chs}, cpi={self.cpi_length}, "
                f"sync_state={self.sync_state}, "
                f"freq={self.rf_center_freq/1e6:.3f}MHz, "
                f"fs={self.sampling_freq/1e6:.3f}MHz)")


# =============================================================================
# Helper: robust TCP receive
# =============================================================================

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receives exactly n bytes; raises ConnectionError if the socket closes."""
    buf = bytearray(n)
    view = memoryview(buf)
    received = 0
    while received < n:
        chunk = sock.recv_into(view[received:], n - received)
        if chunk == 0:
            raise ConnectionError(f"Socket closed after {received}/{n} bytes")
        received += chunk
    return bytes(buf)


# =============================================================================
# KrakenIQSource
# =============================================================================

class KrakenIQSource:
    """
    TCP connection to Heimdall with background buffer and thread-safe access.

    Parameters
    ----------
    host         : Heimdall server IP (default localhost)
    port         : IQ data port        (default 5000)
    ctrl_port    : CTRL control port   (default 5001)
    num_channels : number of antennas  (3 for 3-ant configuration)
    freq_hz      : carrier frequency in Hz  (default 868 MHz)
    gain_db      : IF gain in dB (scalar or per-channel list)
    queue_size   : max frames in queue (oldest are dropped)
    verbose      : print diagnostic headers every N frames
    """

    def __init__(
        self,
        host:         str   = "127.0.0.1",
        port:         int   = 5000,
        ctrl_port:    int   = 5001,
        num_channels: int   = 3,
        freq_hz:      float = 868e6,
        gain_db:      float = 30.0,
        queue_size:   int   = 4,
        verbose:      int   = 0,          # 0 = silent, N = print every N frames
        recv_timeout_s: float = 45.0,     # per-frame TCP recv timeout (large CPI)
    ):
        self.host         = host
        self.port         = port
        self.ctrl_port    = ctrl_port
        self.num_channels = num_channels
        self.freq_hz      = int(freq_hz)
        self.gain_db      = gain_db if isinstance(gain_db, list) else [gain_db] * num_channels
        self.verbose      = verbose
        self._recv_timeout = float(recv_timeout_s)

        self._sock      = None
        self._ctrl_sock = None
        self._connected = False
        self._queue     = queue.Queue(maxsize=queue_size)
        self._stop_evt  = threading.Event()
        self._thread    = None
        self._ctrl_lock = threading.Lock()

        self._frame_ctr   = 0
        self._retry_ctr   = 0   # reconnection attempt counter
        self.last_header  = IQHeader()

        # Valid RTL-SDR gain values
        self._valid_gains = [0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4,
                             15.7, 16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7,
                             32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
                             43.9, 44.5, 48.0, 49.6]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Opens data and control sockets, performs handshake. Returns True on success."""
        try:
            # Data socket
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Large CPI frames (~4 MB) need a generous recv timeout when the
            # DAQ is busy, calibrating delay_sync, or recovering from a restart.
            self._sock.settimeout(self._recv_timeout)
            self._sock.connect((self.host, self.port))
            self._sock.sendall(b"streaming")

            # Read bootstrap frame: Heimdall sends it automatically
            # after "streaming" — no "IQDownload" needed here.
            _boot = None
            for _try in range(3):
                try:
                    _boot = self._recv_frame(request=False)
                    break
                except socket.timeout:
                    if _try < 2:
                        time.sleep(2.0)
                    else:
                        raise
            if _boot is None:
                raise ConnectionError("bootstrap frame empty")

            # Control socket
            self._ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._ctrl_sock.settimeout(15.0)
            self._ctrl_sock.connect((self.host, self.ctrl_port))

            self._connected = True

            # INIT
            self._send_ctrl(b"INIT" + bytes(124))

            # Set frequency and gain
            self.set_frequency(self.freq_hz)
            self.set_gain(self.gain_db)

            print(f"[KrakenIQ] Connected to {self.host}:{self.port}  "
                  f"ctrl:{self.ctrl_port}  ch={self.num_channels}  "
                  f"freq={self.freq_hz/1e6:.3f}MHz")
            self._retry_ctr = 0   # reset counter on successful connection
            return True

        except Exception as exc:
            self._retry_ctr += 1
            # Log only the first failure and then every 10 attempts (~20s)
            if self._retry_ctr == 1 or self._retry_ctr % 10 == 0:
                print(f"[KrakenIQ] Connection failed (attempt {self._retry_ctr}): {exc}")
            self._connected = False
            return False

    def _disconnect(self):
        try:
            if self._connected and self._sock:
                self._sock.sendall(b"q")
        except Exception:
            pass
        try:
            # EXIT on control channel: only if socket exists and seems reachable.
            # Do not wait for reply (socket may be broken).
            if self._ctrl_sock:
                self._ctrl_sock.send(b"EXIT" + bytes(124))
        except Exception:
            pass
        for s in (self._sock, self._ctrl_sock):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        self._sock = None
        self._ctrl_sock = None
        self._connected = False

    # ------------------------------------------------------------------
    # Frame receive
    # ------------------------------------------------------------------

    def _recv_frame(self, request: bool = True) -> np.ndarray | None:
        """
        Reads a complete frame: 1024 B header + payload.
        `request=True`  -> sends "IQDownload" before reading (normal frames)
        `request=False` -> reads directly (bootstrap: Heimdall sends it
                           automatically after the "streaming" string)
        Returns array (active_ant_chs, cpi_length) complex64 or None.
        """
        if request:
            self._sock.sendall(b"IQDownload")

        raw_hdr = _recv_exact(self._sock, IQHeader.HEADER_SIZE)
        hdr = IQHeader()
        hdr.decode(raw_hdr)
        self.last_header = hdr

        if hdr.payload_bytes == 0:
            return None

        raw_payload = _recv_exact(self._sock, hdr.payload_bytes)

        count = hdr.active_ant_chs * hdr.cpi_length
        samples = (np.frombuffer(raw_payload, dtype=np.complex64, count=count)
                   .reshape(hdr.active_ant_chs, hdr.cpi_length)
                   .copy())
        return samples

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _worker(self):
        while not self._stop_evt.is_set():
            if not self._connected:
                if not self._connect():
                    time.sleep(3.0)
                    continue

            try:
                frame = self._recv_frame()
            except socket.timeout:
                if not self._stop_evt.is_set():
                    print("[KrakenIQ] Frame recv timed out — retrying once")
                try:
                    frame = self._recv_frame()
                except Exception as exc2:
                    if not self._stop_evt.is_set():
                        print(f"[KrakenIQ] Receive error: {exc2}")
                    self._connected = False
                    for s in (self._sock, self._ctrl_sock):
                        try:
                            if s:
                                s.close()
                        except Exception:
                            pass
                    self._sock = None
                    self._ctrl_sock = None
                    time.sleep(3.0)
                    continue
            except Exception as exc:
                if not self._stop_evt.is_set():
                    print(f"[KrakenIQ] Receive error: {exc}")
                self._connected = False
                # Close existing sockets; _connect() will create new ones
                for s in (self._sock, self._ctrl_sock):
                    try:
                        if s: s.close()
                    except Exception:
                        pass
                self._sock = None
                self._ctrl_sock = None
                time.sleep(3.0)
                continue

            if frame is None:
                continue

            # Filter out non-DATA frames (calibration, dummy, etc.)
            if not self.last_header.is_data:
                continue

            self._frame_ctr += 1
            if self.verbose > 0 and self._frame_ctr % self.verbose == 0:
                print(f"[KrakenIQ] frame #{self._frame_ctr}: {self.last_header}")

            # Drop oldest frame if queue is full (drop oldest)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

            self._queue.put_nowait(frame)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Starts the background receive thread."""
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="KrakenIQ-rx")
        self._thread.start()

    def stop(self):
        """Stops the thread and closes sockets."""
        self._stop_evt.set()
        self._disconnect()
        if self._thread:
            self._thread.join(timeout=6.0)

    def get_frame(self, timeout: float = 3.0) -> np.ndarray | None:
        """
        Returns the latest IQ frame (Nr × N) complex64.
        Returns None if no frame is available within timeout seconds.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_frequency(self, freq_hz: float):
        """Sets the carrier frequency via the control channel."""
        self.freq_hz = int(freq_hz)
        if self._connected:
            msg = b"FREQ" + pack("Q", self.freq_hz) + bytes(116)
            self._send_ctrl(msg)

    def set_gain(self, gain_db):
        """Sets the IF gain (scalar or per-channel list)."""
        if not isinstance(gain_db, list):
            gain_db = [gain_db] * self.num_channels
        self.gain_db = gain_db
        if self._connected:
            snapped = [min(self._valid_gains, key=lambda x: abs(x - g)) for g in gain_db]
            gain_int = [int(g * 10) for g in snapped]
            payload = pack("I" * self.num_channels, *gain_int)
            padding = bytes(128 - (self.num_channels + 1) * 4)
            self._send_ctrl(b"GAIN" + payload + padding)

    def _send_ctrl(self, msg: bytes):
        """
        Sends a message on the control channel and waits for the FNSD reply
        in a separate daemon thread (as the original krakensdr_source.py does).
        Does not block the caller.
        """
        if not self._ctrl_sock:
            return

        def _comm(sock, lock, data):
            with lock:
                try:
                    sock.send(data)
                    reply = sock.recv(128)
                    status = reply[0:4].decode(errors="replace")
                    if status == "FNSD":
                        pass  # OK silently
                    else:
                        print(f"[KrakenIQ] ctrl reply: {status!r}")
                except Exception as exc:
                    print(f"[KrakenIQ] ctrl err: {exc}")

        t = threading.Thread(
            target=_comm,
            args=(self._ctrl_sock, self._ctrl_lock, msg),
            daemon=True,
        )
        t.start()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()
