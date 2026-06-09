"""
recording.py — Live session recorder for KrakenSDR Iridium DOA.

Each CPI frame and DOA estimate is flushed to disk immediately (crash-safe).
Shutdown is fast by default: data stays in raw/ and doa/ subdirs.
Optional offline consolidation builds raw_iq.npz (slow for long sessions).

Output layout (one folder per session):

    session_YYYYMMDD_HHMMSS/
        meta.json
        state.json
        raw/frame_NNNNNN.npy     one CPI per file (~5 MB), written live
        doa/est_NNNNNN.npz       one DOA spectrum per valid burst
        doa_music.npz            small consolidated file (written on exit)
        raw_iq.npz               optional, built offline via consolidate_session.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["SessionRecorder", "default_record_dir", "consolidate_session", "list_session_raw_frames"]

_STATE_WRITE_INTERVAL_S = 5.0


def default_record_dir() -> str:
    """Default: krakenSDR/data/doa_iridium relative to repo root."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "data", "doa_iridium"))


def _atomic_write(path: str, write_fn: Callable[[str], None]) -> None:
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _atomic_write_json(path: str, obj: dict) -> None:
    def _write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    _atomic_write(path, _write)


def _atomic_save_npy_simple(path: str, array: np.ndarray) -> None:
    assert path.endswith(".npy")
    directory = os.path.dirname(path) or "."
    tmp_base = os.path.join(directory, f".{os.path.basename(path[:-4])}.{os.getpid()}.tmp")
    try:
        np.save(tmp_base, array)
        os.replace(tmp_base + ".npy", path)
    finally:
        for leftover in (tmp_base, tmp_base + ".npy"):
            if os.path.exists(leftover) and leftover != path:
                try:
                    os.remove(leftover)
                except OSError:
                    pass


def _atomic_savez_compressed(path: str, **arrays: Any) -> None:
    directory = os.path.dirname(path) or "."
    stem = os.path.basename(path)[:-4] if path.endswith(".npz") else os.path.basename(path)
    tmp_base = os.path.join(directory, f".{stem}.{os.getpid()}.tmp")
    tmp_npz = tmp_base + ".npz"
    try:
        np.savez_compressed(tmp_base, **arrays)
        os.replace(tmp_npz, path)
    finally:
        if os.path.exists(tmp_npz) and tmp_npz != path:
            try:
                os.remove(tmp_npz)
            except OSError:
                pass


def _count_raw_frames(raw_dir: str) -> int:
    n = 0
    while os.path.isfile(os.path.join(raw_dir, f"frame_{n:06d}.npy")):
        n += 1
    return n


def _count_doa_estimates(doa_dir: str) -> int:
    n = 0
    while os.path.isfile(os.path.join(doa_dir, f"est_{n:06d}.npz")):
        n += 1
    return n


@dataclass
class SessionRecorder:
    """Incrementally record Kraken CPI frames and DOA spectra."""

    out_dir: str
    freq_hz: float
    fs: float
    gain_db: float
    n_ant: int
    cpi_size: int
    n_az: int
    n_el: int
    el_min_deg: float = 5.0
    el_max_deg: float = 90.0
    record_raw: bool = True
    checkpoint_s: float = 60.0
    consolidate_on_exit: bool = False
    mode: str = ""
    algo: str = ""

    session_dir: str = field(init=False)
    raw_dir: str = field(init=False)
    doa_dir: str = field(init=False)
    _raw_count: int = field(default=0, init=False)
    _doa_count: int = field(default=0, init=False)
    _raw_t: list[float] = field(default_factory=list, init=False)
    _last_checkpoint: float = field(default=0.0, init=False)
    _last_state_write: float = field(default=0.0, init=False)
    _start_t: float = field(default_factory=time.time, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _save_done: threading.Event = field(default_factory=threading.Event, init=False)
    _saving: bool = field(default=False, init=False)
    _finalized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.out_dir, f"session_{ts}")
        self.raw_dir = os.path.join(self.session_dir, "raw")
        self.doa_dir = os.path.join(self.session_dir, "doa")
        os.makedirs(self.session_dir, exist_ok=True)
        if self.record_raw:
            os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.doa_dir, exist_ok=True)
        self._write_meta()
        self._write_state(status="recording")

    @property
    def enabled(self) -> bool:
        return True

    @property
    def save_done(self) -> threading.Event:
        return self._save_done

    def _write_meta(self) -> None:
        meta = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "freq_hz": self.freq_hz,
            "fs_hz": self.fs,
            "gain_db": self.gain_db,
            "n_ant": self.n_ant,
            "cpi_size": self.cpi_size,
            "n_az": self.n_az,
            "n_el": self.n_el,
            "el_min_deg": self.el_min_deg,
            "el_max_deg": self.el_max_deg,
            "record_raw": self.record_raw,
            "consolidate_on_exit": self.consolidate_on_exit,
            "mode": self.mode,
            "algo": self.algo,
        }
        _atomic_write_json(os.path.join(self.session_dir, "meta.json"), meta)

    def _write_state(self, *, status: str = "recording") -> None:
        state = {
            "status": status,
            "raw_frames": self._raw_count,
            "doa_estimates": self._doa_count,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.time() - self._start_t, 1),
        }
        _atomic_write_json(os.path.join(self.session_dir, "state.json"), state)
        self._last_state_write = time.time()

    def _maybe_write_state(self, *, force: bool = False) -> None:
        if force or time.time() - self._last_state_write >= _STATE_WRITE_INTERVAL_S:
            self._write_state()

    def add_raw_frame(self, X: np.ndarray, timestamp: float | None = None) -> None:
        if not self.record_raw:
            return
        ts = float(timestamp if timestamp is not None else time.time())
        with self._lock:
            idx = self._raw_count
            path = os.path.join(self.raw_dir, f"frame_{idx:06d}.npy")
            _atomic_save_npy_simple(path, np.asarray(X, dtype=np.complex64))
            self._raw_t.append(ts)
            self._raw_count += 1
            self._maybe_write_state()
        self._maybe_checkpoint()

    def add_doa(
        self,
        spec2d: np.ndarray,
        az_deg: float,
        el_deg: float,
        *,
        papr_db: float = 0.0,
        snr_db: float = 0.0,
        timestamp: float | None = None,
    ) -> None:
        spec = np.asarray(spec2d, dtype=np.float32)
        az_spec = np.max(spec, axis=0).astype(np.float32)
        ts = float(timestamp if timestamp is not None else time.time())

        with self._lock:
            idx = self._doa_count
            path = os.path.join(self.doa_dir, f"est_{idx:06d}.npz")
            _atomic_savez_compressed(
                path,
                spec2d=spec,
                doa_az=az_spec,
                az_deg=np.float32(az_deg),
                el_deg=np.float32(el_deg),
                papr_db=np.float32(papr_db),
                snr_db=np.float32(snr_db),
                t=np.float64(ts),
            )
            self._doa_count += 1
            self._maybe_write_state()
        self._maybe_checkpoint()

    def _maybe_checkpoint(self) -> None:
        if self.checkpoint_s <= 0:
            return
        now = time.time()
        if now - self._last_checkpoint >= self.checkpoint_s:
            with self._lock:
                self._write_state()
            print(f"[REC] checkpoint: {self._raw_count} CPI frames, "
                  f"{self._doa_count} DOA estimates", flush=True)
            self._last_checkpoint = now

    def _load_doa_stack(self) -> dict[str, Any]:
        specs: list[np.ndarray] = []
        doa_az: list[np.ndarray] = []
        az_deg: list[float] = []
        el_deg: list[float] = []
        papr_db: list[float] = []
        snr_db: list[float] = []
        times: list[float] = []

        for i in range(self._doa_count):
            path = os.path.join(self.doa_dir, f"est_{i:06d}.npz")
            d = np.load(path)
            specs.append(d["spec2d"])
            doa_az.append(d["doa_az"])
            az_deg.append(float(d["az_deg"]))
            el_deg.append(float(d["el_deg"]))
            papr_db.append(float(d["papr_db"]))
            snr_db.append(float(d["snr_db"]))
            times.append(float(d["t"]))

        az_grid = np.linspace(0.0, 360.0, self.n_az, endpoint=False, dtype=np.float32)
        el_grid = np.linspace(self.el_min_deg, self.el_max_deg, self.n_el, dtype=np.float32)
        spec2d = np.stack(specs, axis=0)
        doa_az_arr = np.stack(doa_az, axis=0)
        return dict(
            spec2d=spec2d,
            doa_az=doa_az_arr,
            last_spec2d=spec2d[-1],
            last_doa_az=doa_az_arr[-1],
            az_deg=np.array(az_deg, dtype=np.float32),
            el_deg=np.array(el_deg, dtype=np.float32),
            papr_db=np.array(papr_db, dtype=np.float32),
            snr_db=np.array(snr_db, dtype=np.float32),
            t=np.array(times, dtype=np.float64),
            az_grid_deg=az_grid,
            el_grid_deg=el_grid,
            freq_hz=np.int64(self.freq_hz),
        )

    def save(self, tag: str = "", *, consolidate_raw: bool | None = None) -> dict[str, str]:
        """Fast shutdown by default. Raw consolidation is optional (slow)."""
        if self._saving:
            self._save_done.wait()
            return {}
        self._saving = True
        self._save_done.clear()
        written: dict[str, str] = {}
        suffix = tag
        do_consolidate_raw = (
            self.consolidate_on_exit if consolidate_raw is None else consolidate_raw
        )

        try:
            print(f"[REC] Finalizing session ({self._raw_count} CPI, "
                  f"{self._doa_count} DOA)…", flush=True)

            with self._lock:
                self._write_state(status="finalizing")

                if self._doa_count > 0:
                    path = os.path.join(self.session_dir, f"doa_music{suffix}.npz")
                    print("[REC] Writing doa_music.npz…", flush=True)
                    payload = self._load_doa_stack()
                    _atomic_savez_compressed(path, **payload)
                    written["doa_music"] = path
                    print(f"[REC] {self._doa_count} DOA spectra → {path}", flush=True)

                if do_consolidate_raw and self.record_raw and self._raw_count > 0:
                    path = os.path.join(self.session_dir, f"raw_iq{suffix}.npz")
                    print(f"[REC] Consolidating {self._raw_count} CPI frames into "
                          f"raw_iq.npz (may take several minutes)…", flush=True)
                    consolidate_session(
                        self.session_dir,
                        out_path=path,
                        raw=True,
                        doa=False,
                        timestamps=self._raw_t,
                        freq_hz=self.freq_hz,
                        fs_hz=self.fs,
                        gain_db=self.gain_db,
                        n_ant=self.n_ant,
                        cpi_size=self.cpi_size,
                    )
                    written["raw_iq"] = path
                elif self.record_raw and self._raw_count > 0:
                    print(f"[REC] Raw IQ safe in {self.raw_dir}/ "
                          f"({self._raw_count} frames) — skipped raw_iq.npz", flush=True)
                    print("[REC] To build raw_iq.npz later: "
                          "python3 consolidate_session.py "
                          f"{self.session_dir}", flush=True)

                self._write_state(status="complete")
                self._finalized = True

            elapsed = time.time() - self._start_t
            print(f"[REC] Done — {self.session_dir}  ({elapsed:.0f}s elapsed)", flush=True)
        finally:
            self._saving = False
            self._save_done.set()

        return written


def count_session_raw_frames(session_dir: str) -> int:
    """Count frame_*.npy files without loading them."""
    raw_dir = os.path.join(session_dir, "raw")
    n = 0
    while os.path.isfile(os.path.join(raw_dir, f"frame_{n:06d}.npy")):
        n += 1
    return n


def iter_session_raw_frames(
    session_dir: str,
    *,
    start: int = 0,
    stride: int = 1,
    max_frames: int = 0,
):
    """Yield (frame_index, array) from raw/ one file at a time (memory-safe)."""
    raw_dir = os.path.join(session_dir, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"No raw/ directory in {session_dir}")
    total = count_session_raw_frames(session_dir)
    if start >= total:
        return
    yielded = 0
    idx = start
    while idx < total:
        path = os.path.join(raw_dir, f"frame_{idx:06d}.npy")
        if not os.path.isfile(path):
            break
        yield idx, np.load(path)
        yielded += 1
        if max_frames > 0 and yielded >= max_frames:
            break
        idx += max(1, stride)


def list_session_raw_frames(session_dir: str) -> list[np.ndarray]:
    raw_dir = os.path.join(session_dir, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"No raw/ directory in {session_dir}")
    frames: list[np.ndarray] = []
    i = 0
    while True:
        path = os.path.join(raw_dir, f"frame_{i:06d}.npy")
        if not os.path.isfile(path):
            break
        frames.append(np.load(path))
        i += 1
    if not frames:
        raise ValueError(f"No frame_*.npy files in {raw_dir}")
    return frames


def consolidate_session(
    session_dir: str,
    *,
    out_path: str | None = None,
    raw: bool = True,
    doa: bool = True,
    timestamps: list[float] | None = None,
    freq_hz: float | None = None,
    fs_hz: float | None = None,
    gain_db: float | None = None,
    n_ant: int | None = None,
    cpi_size: int | None = None,
    progress_every: int = 50,
) -> dict[str, str]:
    """Build raw_iq.npz / doa_music.npz from incremental session files (offline)."""
    session_dir = os.path.abspath(session_dir)
    meta_path = os.path.join(session_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    freq_hz = freq_hz if freq_hz is not None else float(meta.get("freq_hz", 0))
    fs_hz = fs_hz if fs_hz is not None else float(meta.get("fs_hz", 1_024_000))
    gain_db = gain_db if gain_db is not None else float(meta.get("gain_db", 0))
    n_ant = n_ant if n_ant is not None else int(meta.get("n_ant", 5))
    cpi_size = cpi_size if cpi_size is not None else int(meta.get("cpi_size", 131072))
    n_az = int(meta.get("n_az", 360))
    n_el = int(meta.get("n_el", 86))

    written: dict[str, str] = {}
    raw_dir = os.path.join(session_dir, "raw")
    doa_dir = os.path.join(session_dir, "doa")

    if raw:
        n_frames = _count_raw_frames(raw_dir)
        if n_frames == 0:
            raise ValueError(f"No frames in {raw_dir}")
        out = out_path or os.path.join(session_dir, "raw_iq.npz")
        print(f"[REC] Loading {n_frames} frames…", flush=True)

        frames: list[np.ndarray] = []
        for i in range(n_frames):
            frames.append(np.load(os.path.join(raw_dir, f"frame_{i:06d}.npy")))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[REC]   … {i + 1}/{n_frames} frames loaded", flush=True)

        X = np.stack(frames, axis=0)
        del frames

        if timestamps is None:
            timestamps_arr = np.arange(n_frames, dtype=np.float64)
        else:
            timestamps_arr = np.array(timestamps[:n_frames], dtype=np.float64)

        print(f"[REC] Compressing → {out} …", flush=True)
        _atomic_savez_compressed(
            out,
            X=X,
            timestamps=timestamps_arr,
            freq_hz=np.int64(freq_hz),
            fs_hz=np.float64(fs_hz),
            gain_db=np.float32(gain_db),
            cpi_size=np.int32(cpi_size),
            n_ant=np.int32(n_ant),
        )
        written["raw_iq"] = out
        print(f"[REC] raw_iq.npz written ({n_frames} frames)", flush=True)

    if doa:
        n_est = _count_doa_estimates(doa_dir)
        if n_est > 0:
            out = os.path.join(session_dir, "doa_music.npz")
            specs, doa_az_l, az, el, papr, snr, times = [], [], [], [], [], [], []
            for i in range(n_est):
                d = np.load(os.path.join(doa_dir, f"est_{i:06d}.npz"))
                specs.append(d["spec2d"])
                doa_az_l.append(d["doa_az"])
                az.append(float(d["az_deg"]))
                el.append(float(d["el_deg"]))
                papr.append(float(d["papr_db"]))
                snr.append(float(d["snr_db"]))
                times.append(float(d["t"]))
            spec2d = np.stack(specs, axis=0)
            doa_az_arr = np.stack(doa_az_l, axis=0)
            az_grid = np.linspace(0, 360, n_az, endpoint=False, dtype=np.float32)
            el_grid = np.linspace(
                float(meta.get("el_min_deg", 5)), float(meta.get("el_max_deg", 90)),
                n_el, dtype=np.float32,
            )
            _atomic_savez_compressed(
                out,
                spec2d=spec2d,
                doa_az=doa_az_arr,
                last_spec2d=spec2d[-1],
                last_doa_az=doa_az_arr[-1],
                az_deg=np.array(az, dtype=np.float32),
                el_deg=np.array(el, dtype=np.float32),
                papr_db=np.array(papr, dtype=np.float32),
                snr_db=np.array(snr, dtype=np.float32),
                t=np.array(times, dtype=np.float64),
                az_grid_deg=az_grid,
                el_grid_deg=el_grid,
                freq_hz=np.int64(freq_hz),
            )
            written["doa_music"] = out
            print(f"[REC] doa_music.npz written ({n_est} estimates)", flush=True)

    return written
