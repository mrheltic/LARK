"""Tests for lark.recording.SessionRecorder."""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import pytest

from core.recording import (
    SessionRecorder,
    list_session_raw_frames,
)


def test_session_recorder_writes_raw_and_doa(tmp_path):
    rec = SessionRecorder(
        out_dir=str(tmp_path),
        freq_hz=1_626_270_000,
        fs=1_024_000.0,
        gain_db=30.0,
        n_ant=5,
        cpi_size=1024,
        n_az=360,
        n_el=86,
        record_raw=True,
        checkpoint_s=0,
        mode="indoor",
        algo="music",
    )

    X = np.zeros((5, 1024), dtype=np.complex64)
    rec.add_raw_frame(X, timestamp=1000.0)

    spec = np.random.randn(86, 360).astype(np.float32)
    spec -= spec.max()
    rec.add_doa(spec, 45.0, 30.0, papr_db=5.0, snr_db=8.0, timestamp=1001.0)

    # Incremental files exist before consolidate
    assert os.path.isfile(os.path.join(rec.raw_dir, "frame_000000.npy"))
    assert os.path.isfile(os.path.join(rec.doa_dir, "est_000000.npz"))

    paths = rec.save()
    assert "doa_music" in paths
    assert "raw_iq" not in paths  # skipped by default (fast exit)

    paths_full = rec.save(consolidate_raw=True)
    assert "raw_iq" in paths_full
    assert zipfile.is_zipfile(paths_full["raw_iq"])
    assert zipfile.is_zipfile(paths["doa_music"])

    raw = np.load(paths_full["raw_iq"])
    assert raw["X"].shape == (1, 5, 1024)
    assert raw["freq_hz"] == 1_626_270_000

    doa = np.load(paths["doa_music"])
    assert doa["spec2d"].shape == (1, 86, 360)
    assert doa["doa_az"].shape == (1, 360)
    assert doa["last_doa_az"].shape == (360,)
    assert float(doa["az_deg"][0]) == pytest.approx(45.0)

    with open(os.path.join(rec.session_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["cpi_size"] == 1024
    assert meta["mode"] == "indoor"


def test_session_recorder_doa_only(tmp_path):
    rec = SessionRecorder(
        out_dir=str(tmp_path),
        freq_hz=1_626_270_000,
        fs=1_024_000.0,
        gain_db=30.0,
        n_ant=5,
        cpi_size=131072,
        n_az=360,
        n_el=86,
        record_raw=False,
    )
    spec = np.zeros((86, 360), dtype=np.float32)
    rec.add_doa(spec, 0.0, 45.0)
    paths = rec.save()
    assert "raw_iq" not in paths
    assert "doa_music" in paths


def test_incremental_frames_survive_without_consolidate(tmp_path):
    rec = SessionRecorder(
        out_dir=str(tmp_path),
        freq_hz=1_626_270_000,
        fs=1_024_000.0,
        gain_db=30.0,
        n_ant=5,
        cpi_size=512,
        n_az=360,
        n_el=86,
        record_raw=True,
        checkpoint_s=0,
    )
    for i in range(3):
        rec.add_raw_frame(np.full((5, 512), i, dtype=np.complex64))

    frames = list_session_raw_frames(rec.session_dir)
    assert len(frames) == 3
    assert frames[2][0, 0] == pytest.approx(2 + 0j)

    rec.save()
    with open(os.path.join(rec.session_dir, "state.json"), encoding="utf-8") as f:
        state = json.load(f)
    assert state["raw_frames"] == 3
    assert state["status"] == "complete"
