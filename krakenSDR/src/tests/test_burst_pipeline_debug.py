#!/usr/bin/env python3
"""
test_burst_pipeline_debug.py — Tests for pipeline stage debug saving.

Run:
    python3 -m pytest krakenSDR/src/tests/test_burst_pipeline_debug.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pytest

from core.pipeline_debug import (
    PipelineDebugSaver,
    STAGE_FILES,
    save_pipeline_stage,
)


class TestPipelineDebugSaver:

    def test_disabled_when_no_dir(self):
        dbg = PipelineDebugSaver(None, 1)
        assert not dbg.enabled
        assert dbg.save("raw_iq", np.zeros(5)) is None

    def test_writes_all_stage_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dbg = PipelineDebugSaver(tmp, 42)
            assert dbg.enabled
            assert dbg.frame_dir.endswith("frame_000042")

            X = np.ones((5, 100), dtype=np.complex64)
            dbg.save("raw_iq", X)
            dbg.save("burst_starts", np.array([100, 200]))
            dbg.save("tones", [{"tone_hz": 3125.0, "snr_db": 12.0}])
            dbg.save("bpf", X[:, :50])
            dbg.save("phase_corrected", X[:, :50])
            dbg.save("R_mf", np.eye(5, dtype=np.complex64))
            dbg.save("R_ema", np.eye(5, dtype=np.complex64))
            dbg.save("spec2d", np.zeros((10, 20), dtype=np.float32))
            dbg.save("doa_result", {"az": 90.0, "el": 30.0})

            for stage, fname in STAGE_FILES.items():
                path = os.path.join(dbg.frame_dir, fname)
                assert os.path.isfile(path), f"missing {stage} → {fname}"

            with open(os.path.join(dbg.frame_dir, "07_doa_result.json")) as f:
                result = json.load(f)
            assert result["az"] == 90.0

    def test_save_pipeline_stage_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_pipeline_stage(tmp, 7, "spec2d", np.zeros((4, 8)))
            assert path is not None
            assert path.endswith("06_spec2d.npy")
            assert os.path.isfile(path)

    def test_unknown_stage_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            dbg = PipelineDebugSaver(tmp, 0)
            with pytest.raises(KeyError):
                dbg.save("not_a_stage", [])
