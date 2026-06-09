"""
pipeline_debug.py — Save intermediate data from each DOA pipeline stage.

Output layout (per frame, under debug_dir/frame_NNNNNN/):

    00_raw_iq.npy           (n_ant, N) complex IQ frame
    01_burst_starts.npy     sample indices from energy detector
    02_tones.json           preamble tone scan results
    03_bpf.npy              BPF + normalised narrowband IQ
    04_phase_corrected.npy  after hardware phase calibration
    05_R_mf.npy             matched-filter covariance (instant)
    05b_R_ema.npy           EMA-smoothed covariance
    06_spec2d.npy           2D DOA spectrum (n_el, n_az) [dB]
    07_doa_result.json      az/el, SNR, PAPR, CFO, etc.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

STAGE_FILES: dict[str, str] = {
    "raw_iq":          "00_raw_iq.npy",
    "burst_starts":    "01_burst_starts.npy",
    "tones":           "02_tones.json",
    "bpf":             "03_bpf.npy",
    "phase_corrected": "04_phase_corrected.npy",
    "R_mf":            "05_R_mf.npy",
    "R_ema":           "05b_R_ema.npy",
    "spec2d":          "06_spec2d.npy",
    "doa_result":      "07_doa_result.json",
}

__all__ = ["STAGE_FILES", "PipelineDebugSaver", "save_pipeline_stage"]


@dataclass
class PipelineDebugSaver:
    """Write numbered stage files under ``base_dir/frame_NNNNNN/``."""

    base_dir: str | None
    frame_idx: int
    _frame_dir: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.base_dir:
            self._frame_dir = os.path.join(self.base_dir, f"frame_{self.frame_idx:06d}")
            os.makedirs(self._frame_dir, exist_ok=True)
        else:
            self._frame_dir = None

    @property
    def enabled(self) -> bool:
        return self._frame_dir is not None

    @property
    def frame_dir(self) -> str | None:
        return self._frame_dir

    def save(self, stage: str, data: Any) -> str | None:
        """
        Save one pipeline stage artifact.

        Parameters
        ----------
        stage : key in STAGE_FILES (e.g. ``"bpf"``, ``"doa_result"``)
        data  : ndarray for .npy stages, dict/list for .json stages

        Returns
        -------
        str or None — path written, or None when debug is disabled
        """
        if not self.enabled:
            return None

        fname = STAGE_FILES.get(stage)
        if fname is None:
            raise KeyError(f"Unknown stage {stage!r}; known: {list(STAGE_FILES)}")

        path = os.path.join(self._frame_dir, fname)
        if fname.endswith(".npy"):
            np.save(path, np.asarray(data))
        elif fname.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported stage file type: {fname}")

        return path

    def save_all(self, **stages: Any) -> dict[str, str]:
        """Save multiple stages; skips ``None`` values."""
        written: dict[str, str] = {}
        for name, val in stages.items():
            if val is not None:
                path = self.save(name, val)
                if path:
                    written[name] = path
        return written


def save_pipeline_stage(
    out_dir: str | None,
    frame_idx: int,
    stage: str,
    data: Any,
) -> str | None:
    """
    Convenience wrapper: save a single stage for one frame.

    Example::

        save_pipeline_stage("/tmp/dbg", 42, "bpf", X_bpf)
    """
    return PipelineDebugSaver(out_dir, frame_idx).save(stage, data)
