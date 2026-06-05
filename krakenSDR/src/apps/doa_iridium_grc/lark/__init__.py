"""
lark — Iridium DOA signal processing (pure Python, no GNU Radio).

    burst_processing.py   Energy detection, tone scan, BPF, MF covariance
    pipeline_debug.py     Save intermediate data at each pipeline stage
    recording.py          Raw Kraken IQ + doa_music.npz session recorder
"""

from __future__ import annotations

from .burst_processing import (
    apply_bpf_and_normalize,
    compute_mf_covariance,
    detect_energy_bursts,
    scan_preamble_tones,
)
from .pipeline_debug import PipelineDebugSaver, STAGE_FILES, save_pipeline_stage
from .recording import SessionRecorder, consolidate_session, default_record_dir, list_session_raw_frames

__all__ = [
    "detect_energy_bursts",
    "scan_preamble_tones",
    "apply_bpf_and_normalize",
    "compute_mf_covariance",
    "PipelineDebugSaver",
    "STAGE_FILES",
    "save_pipeline_stage",
    "SessionRecorder",
    "consolidate_session",
    "default_record_dir",
    "list_session_raw_frames",
]
