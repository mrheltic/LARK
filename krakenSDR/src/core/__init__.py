"""
core — KrakenSDR Iridium DOA signal-processing library
=======================================================

Everything needed to estimate the direction of arrival (DOA) of Iridium
satellite bursts received with a KrakenSDR (5 coherent channels) and a
uniform circular array (UCA) of 5 antennas.

Processing pipeline (one CPI frame = one block of coherent IQ samples,
shape ``(n_ant, n_samples)``):

    1. burst detection      detect_energy_bursts()      where in the frame?
    2. tone scan            scan_preamble_tones()       which frequency?
                                                        (3125 Hz + Doppler)
    3. band-pass filter     apply_bpf_and_normalize()   isolate one satellite
    4. phase calibration    apply_phase_correction()    align the channels
    5. MF covariance        compute_mf_covariance()     array response y·yᴴ
    6. DOA spectrum         doa_music_uca_2d() & co.    scan (az, el) grid
    7. peak picking         find_peak_uca_2d()          → azimuth, elevation

`multi_peak.process_cpi_for_multi()` runs the whole chain for every burst
and every satellite tone found in a CPI.  `recording.SessionRecorder`
persists raw CPI frames so sessions can be reprocessed offline, and
`track_clusterer` groups the per-burst estimates into satellite tracks.

Notes for newcomers
-------------------
* The matched-filter covariance is rank-1, so MUSIC, Capon and Bartlett
  all peak at the same (az, el) — algorithm choice is not the lever here,
  phase calibration is (see ``scripts/fit_array_cal.py``).
* Angles follow compass convention: azimuth in degrees clockwise from
  geographic North, elevation in degrees above the horizon.

The modules ``gates``, ``tone_extraction`` and ``tracking`` are kept only
for the older 868 MHz experiments in ``apps/legacy`` and are not part of
the Iridium pipeline (they are intentionally not re-exported here).
"""

from __future__ import annotations

# ── DOA on the uniform circular array (the estimation engine) ──────────────
from .doa_uca_2d import (
    UcaConfig,
    doa_music_uca_2d,
    doa_capon_uca_2d,
    doa_bartlett_uca_2d,
    find_peak_uca_2d,
    snr_uca_db,
    crb_azimuth_deg,
)

# ── Burst DSP (energy detection → tone scan → BPF → MF covariance) ─────────
from .burst_processing import (
    detect_energy_bursts,
    scan_preamble_tones,
    apply_bpf_and_normalize,
    compute_mf_covariance,
)

# ── Per-channel phase calibration + spectrum quality metric ────────────────
from .doa_algorithms import (
    apply_phase_correction,
    papr_db,
)

# ── Whole-CPI processing (all bursts, all satellite tones) ─────────────────
from .multi_peak import process_cpi_for_multi

# ── Session recording / offline replay ─────────────────────────────────────
from .recording import (
    SessionRecorder,
    consolidate_session,
    default_record_dir,
    count_session_raw_frames,
    iter_session_raw_frames,
    list_session_raw_frames,
    session_frame_times,
)

# ── Per-stage debug dumps (--debug-dir) ────────────────────────────────────
from .pipeline_debug import PipelineDebugSaver

# ── Burst-to-satellite track clustering ────────────────────────────────────
from .track_clusterer import cluster_from_jsonl, load_tracks_json

__all__ = [
    # DOA engine
    "UcaConfig",
    "doa_music_uca_2d",
    "doa_capon_uca_2d",
    "doa_bartlett_uca_2d",
    "find_peak_uca_2d",
    "snr_uca_db",
    "crb_azimuth_deg",
    # burst DSP
    "detect_energy_bursts",
    "scan_preamble_tones",
    "apply_bpf_and_normalize",
    "compute_mf_covariance",
    # calibration + metrics
    "apply_phase_correction",
    "papr_db",
    # CPI processing
    "process_cpi_for_multi",
    # recording
    "SessionRecorder",
    "consolidate_session",
    "default_record_dir",
    "count_session_raw_frames",
    "iter_session_raw_frames",
    "list_session_raw_frames",
    "session_frame_times",
    # debug + tracks
    "PipelineDebugSaver",
    "cluster_from_jsonl",
    "load_tracks_json",
]
