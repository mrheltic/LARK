"""
hardware.file_iq_source
=======================

Drop-in IQ source that reads a recorded baseband file instead of live
hardware.  Presents the same get_frame() interface as KrakenIQSource.

Supported formats
-----------------
wav   (auto from .wav)
    SDR++ baseband WAV: signed 16-bit SC16, 2-channel (L=I, R=Q).
    Sample rate is read from the WAV header.

cf32  (auto from .iq / .cf32 / .raw)
    Raw interleaved complex float-32: I0 Q0 I1 Q1 ...
    sample_rate must be supplied.

u8    (auto from .bin / .u8)
    Raw interleaved unsigned-8, 127.5-biased (rtl_sdr format).
    sample_rate must be supplied.

Usage
-----
    src = FileIQSource("iridium_pass.wav", center_freq_hz=1_626_270_000.0)
    src.start()
    while True:
        frame = src.get_frame()   # ndarray (1, frame_size) complex64 or None
        if frame is None:
            break
        # ...
    src.stop()

    # Context-manager form
    with FileIQSource("rec.cf32", sample_rate=2_048_000.0) as src:
        for frame in iter(lambda: src.get_frame(), None):
            ...
"""

from __future__ import annotations

from pathlib import Path
from typing  import Optional

import numpy as np

_EXT_FORMAT: dict[str, str] = {
    ".wav":   "wav",
    ".iq":    "cf32",
    ".cf32":  "cf32",
    ".raw":   "cf32",
    ".cs8":   "u8",
    ".u8":    "u8",
    ".bin":   "u8",
    ".8bit":  "u8",
}

_WAV_SCALE = np.float32(1.0 / 32_768.0)
_U8_OFFSET = np.float32(127.5)
_U8_SCALE  = np.float32(1.0 / 127.5)


class FileIQSource:
    """
    File-backed IQ source with the same get_frame() interface as KrakenIQSource.

    Parameters
    ----------
    path           : IQ recording to open.
    fmt            : 'wav', 'cf32', 'u8', or 'auto' (default).
    sample_rate    : Sample rate Hz.  Required for cf32/u8; ignored for wav.
    center_freq_hz : Nominal centre frequency Hz.  Informational only.
    frame_size     : Samples per frame.  Default 131072 (~64 ms @ 2.048 MS/s).

    Properties
    ----------
    sample_rate, center_freq_hz, num_channels, total_samples,
    current_sample, progress, duration_s, elapsed_s, at_eof, is_connected
    """

    def __init__(
        self,
        path:           str | Path,
        fmt:            str             = "auto",
        sample_rate:    Optional[float] = None,
        center_freq_hz: float           = 0.0,
        frame_size:     int             = 131_072,
    ) -> None:
        self._path          = Path(path)
        self._fmt           = self._resolve_fmt(fmt, self._path)
        self._frame_size    = int(frame_size)
        self.center_freq_hz = float(center_freq_hz)
        self._user_rate     = sample_rate
        self._samples: Optional[np.ndarray] = None
        self._pos     = 0
        self._sr      = 0.0
        self._started = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FileIQSource":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "FileIQSource":
        if self._started:
            return self
        self._samples, self._sr = self._load()
        self._pos     = 0
        self._started = True
        return self

    def stop(self) -> None:
        self._started = False
        self._samples = None

    # ------------------------------------------------------------------
    # Frame delivery  (KrakenIQSource-compatible)
    # ------------------------------------------------------------------

    def get_frame(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """
        Return the next (1, frame_size) complex64 frame, or None at EOF.
        'timeout' is accepted for API compatibility but unused.
        """
        if not self._started or self._samples is None:
            raise RuntimeError("FileIQSource.start() must be called before get_frame()")
        if self._pos >= len(self._samples):
            return None

        chunk = self._samples[self._pos : self._pos + self._frame_size]
        self._pos += self._frame_size

        if len(chunk) == 0:
            return None
        if len(chunk) < self._frame_size:
            pad   = np.zeros(self._frame_size - len(chunk), dtype=np.complex64)
            chunk = np.concatenate([chunk, pad])

        return chunk.reshape(1, self._frame_size)

    # ------------------------------------------------------------------
    # Seeking
    # ------------------------------------------------------------------

    def seek(self, sample_idx: int) -> None:
        """Jump to sample_idx (clamped to valid range)."""
        self._pos = max(0, min(int(sample_idx), self.total_samples))

    def seek_time(self, seconds: float) -> None:
        self.seek(int(seconds * self._sr))

    def rewind(self) -> None:
        self._pos = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> float:
        return self._sr

    @property
    def num_channels(self) -> int:
        return 1

    @property
    def total_samples(self) -> int:
        return len(self._samples) if self._samples is not None else 0

    @property
    def current_sample(self) -> int:
        return self._pos

    @property
    def progress(self) -> float:
        total = self.total_samples
        return self._pos / total if total > 0 else 0.0

    @property
    def duration_s(self) -> float:
        return self.total_samples / self._sr if self._sr > 0 else 0.0

    @property
    def elapsed_s(self) -> float:
        return self._pos / self._sr if self._sr > 0 else 0.0

    @property
    def at_eof(self) -> bool:
        return self._pos >= self.total_samples

    @property
    def is_connected(self) -> bool:
        """Compatibility shim: mirrors KrakenIQSource.is_connected."""
        return self._started

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_fmt(fmt: str, path: Path) -> str:
        if fmt != "auto":
            return fmt.lower()
        return _EXT_FORMAT.get(path.suffix.lower(), "cf32")

    def _load(self) -> tuple[np.ndarray, float]:
        handlers = {"wav": self._load_wav, "cf32": self._load_cf32, "u8": self._load_u8}
        handler  = handlers.get(self._fmt)
        if handler is None:
            raise ValueError(
                f"Unknown IQ format {self._fmt!r}.  Choose: {list(handlers)}")
        return handler()

    def _load_wav(self) -> tuple[np.ndarray, float]:
        try:
            from scipy.io import wavfile as _wf
        except ImportError as exc:
            raise ImportError(
                "scipy is required for WAV files.  pip install scipy") from exc
        rate, data = _wf.read(str(self._path), mmap=True)
        if data.ndim == 1:
            raise ValueError(
                f"{self._path.name}: WAV has 1 channel; expected 2-ch SC16 (I/Q)")
        if data.shape[1] < 2:
            raise ValueError(f"{self._path.name}: WAV shape {data.shape!r} invalid")
        I = data[:, 0].astype(np.float32) * _WAV_SCALE
        Q = data[:, 1].astype(np.float32) * _WAV_SCALE
        return (I + 1j * Q).astype(np.complex64), float(rate)

    def _load_cf32(self) -> tuple[np.ndarray, float]:
        if self._user_rate is None:
            raise ValueError(
                f"{self._path.name}: sample_rate required for CF32 files")
        raw = np.fromfile(str(self._path), dtype=np.float32)
        if raw.size % 2:
            raw = raw[:-1]
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64), float(self._user_rate)

    def _load_u8(self) -> tuple[np.ndarray, float]:
        if self._user_rate is None:
            raise ValueError(
                f"{self._path.name}: sample_rate required for U8 files")
        raw = np.fromfile(str(self._path), dtype=np.uint8)
        if raw.size % 2:
            raw = raw[:-1]
        I = (raw[0::2].astype(np.float32) - _U8_OFFSET) * _U8_SCALE
        Q = (raw[1::2].astype(np.float32) - _U8_OFFSET) * _U8_SCALE
        return (I + 1j * Q).astype(np.complex64), float(self._user_rate)
