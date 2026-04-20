#!/usr/bin/env python3
"""
space_doa_playback.py — KrakenSDR 3D Space DoA (offline playback)
==================================================================
Loads a .npz file captured by space_collector.py or the recorder in
space_doa_realtime.py and replays the full 2D DoA pipeline frame by frame.

Compatible formats
------------------
  key ``bursts``  : (N, 5, N_burst)  complex64  — burst-gated windows (collector)
  key ``frames``  : (N, 5, N_frame)  complex64  — raw IQ frames (legacy)
  key ``timestamps`` : (N,) float64 — ms from session start
  .json sidecar with metadata (auto-loaded if present)

Display — 8 panels  (same layout as space_doa_realtime.py)
-----------------------------------------------------------
  Row 0: [Sky plot (polar)]  [2D heatmap (rect)]  [Az + El history]  [Eigenvalues]
  Row 1: [Coherence matrix]  [PAPR + SNR history]  [IQ FFT ch0]  [Phase stability]

Playback controls (bottom toolbar)
------------------------------------
  ◀◀ Rewind   ▶ Play / ‖ Pause   frame slider   ×0.25 ×0.5 ×1 ×2 ×4 speed

Usage
-----
    python3 space_doa_playback.py
    python3 space_doa_playback.py /path/to/capture.npz
    python3 space_doa_playback.py capture.npz --algo capon --n_az 90 --n_el 27
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(os.path.dirname(_HERE))   # krakenSDR/src/
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# Always re-insert _HERE at 0: Python may have already added it further down
# the list when the script launched, which would let krakenSDR/src/config.py
# shadow the app-local apps/space/config.py.
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider

import config as C
from core.iridium_doa_burst import (
    compensate_doppler,
    compute_single_shot_covariance,
    validate_burst_uw,
    narrowband_filter_burst,
)
from core.doa_algorithms_3d import (
    CROSS_ARRAY_CANONICAL_ORDER,
    CrossArrayConfig,
    SatellitePassAccumulator,
    doa_music_2d,
    doa_capon_2d,
    find_peak_2d,
    make_sky_heatmap_edges,
    skyplot_coords,
    eigenvalue_spread_db,
    snr_from_covariance,
    coherence_matrix,
    estimate_signal_count,
    normalize_cross_array_order,
    reorder_cross_array_channels,
    short_cross_array_labels,
)

# ── Gate thresholds (mirrored from space_doa_realtime.py) ────────────────────
_UW_SCORE_MIN      = 0.4    # minimum UW correlation score
_EIG_SPREAD_MIN_DB = 6.0    # minimum eigenvalue spread [dB]
_PAPR_MIN_DB_ACC   = 1.0    # minimum PAPR to accumulate into pass_acc / density map
_PASS_GAP_S        = 600.0  # inter-burst gap [s] that marks a new satellite pass
_FDMA_SPACING_HZ   = 41_667.0  # Iridium FDMA channel spacing [Hz]
_DOPPLER_JUMP_HZ   = 20_000.0  # Doppler jump that marks a new satellite (< half FDMA spacing)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#1a1d27"; BG2 = "#21253a"; BG3 = "#2a2f47"
C_BORDER = "#3b4263"; C_DIM = "#4e5680"
C_BLUE   = "#5ea4e0"; C_TEAL = "#4ecdc4"; C_AMBER = "#f4a431"
C_VIOLET = "#a78bfa"; C_ROSE  = "#f16b6f"; C_LIME  = "#6dd97d"
C_TEXT   = "#d8dae8"; C_MUTED = "#8891b0"
_ANT_COLORS = [C_BLUE, C_TEAL, C_AMBER, C_VIOLET, C_ROSE]

# ── Per-satellite visual catalogue ────────────────────────────────────────────
# Each detected FDMA channel (= satellite) gets a unique colour + marker style.
_SAT_PALETTE = [C_AMBER, C_TEAL, C_BLUE, C_LIME, C_VIOLET, C_ROSE, "#e879f9", "#fb923c"]
_SAT_MARKERS = ["o", "s", "D", "^", "v", "P", "*", "X"]   # ●■◆▲▼✚★✕


from dataclasses import dataclass, field as _field


@dataclass
class PassInfo:
    """Metadata and accumulation state for one satellite pass."""
    pass_id:    int
    fdma_ch:    int                             # FDMA channel index
    color:      str                             # hex color
    marker:     str                             # matplotlib marker char
    start_idx:  int                             # first frame index in this pass
    end_idx:    int         = -1                # last frame index (inclusive)
    acc:        SatellitePassAccumulator | None = _field(default=None, repr=False)
    indices:    list        = _field(default_factory=list, repr=False)   # accepted frame indices
    az_list:    list        = _field(default_factory=list, repr=False)
    el_list:    list        = _field(default_factory=list, repr=False)
    papr_list:  list        = _field(default_factory=list, repr=False)
    dop_list:   list        = _field(default_factory=list, repr=False)
    t_list:     list        = _field(default_factory=list, repr=False)   # timestamps [ms]

    @property
    def label(self) -> str:
        kHz = self.fdma_ch * _FDMA_SPACING_HZ / 1e3
        return f"Pass {self.pass_id+1} · ch{self.fdma_ch:+d} ({kHz:+.0f} kHz)"

    @property
    def n_accepted(self) -> int:
        return len(self.indices)


# =============================================================================
# File picker dialog (if no path given on CLI)
# =============================================================================

def _pick_file() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Open KrakenSDR space capture",
        filetypes=[("NumPy archives", "*.npz"), ("All files", "*.*")],
        initialdir=os.path.normpath(os.path.join(_SRC, "..", "..", "recordings")),
    )
    root.destroy()
    if not path:
        raise SystemExit(0)
    return path


# =============================================================================
# Load capture
# =============================================================================

def _load(path: str) -> tuple:
    """
    Returns (frames, timestamps, meta_dict).

    frames : (N, 5, N_samples) complex64
    """
    data = np.load(path, allow_pickle=False)
    if "bursts" in data:
        frames = data["bursts"]
    elif "frames" in data:
        frames = data["frames"]
    else:
        # Try to use any key that has a 3-D complex array
        for k in data.files:
            v = data[k]
            if v.ndim == 3 and np.iscomplexobj(v):
                frames = v
                break
        else:
            raise ValueError(f"Cannot find IQ data in {path}. Keys: {data.files}")

    timestamps = data["timestamps"] if "timestamps" in data else np.arange(len(frames), dtype=float)

    # Sidecar JSON
    json_path = os.path.splitext(path)[0] + ".json"
    meta: dict = {}
    if os.path.isfile(json_path):
        with open(json_path) as f:
            meta = json.load(f)
    return frames.astype(np.complex64), timestamps.astype(np.float64), meta


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KrakenSDR 3D space DoA — offline playback from .npz")
    parser.add_argument("file",        nargs="?", help=".npz capture file")
    parser.add_argument("--algo",      type=str,  default=None,
                        choices=["music", "capon"])
    parser.add_argument("--n_az",      type=int,  default=None)
    parser.add_argument("--n_el",      type=int,  default=None)
    parser.add_argument("--d_lambda",  type=float, default=None)
    parser.add_argument("--el_min",    type=float, default=None)
    parser.add_argument("--n_signals", type=int,  default=None)
    parser.add_argument("--no_doppler", action="store_true",
                        help="Skip Doppler compensation (faster, less accurate)")
    args = parser.parse_args()

    npz_path = args.file or _pick_file()
    if not os.path.isfile(npz_path):
        print(f"[PB] File not found: {npz_path}"); raise SystemExit(1)

    frames, timestamps, meta = _load(npz_path)
    N, N_ANT, N_SAMP = frames.shape
    FS = float(meta.get("sample_rate_hz", C.SAMPLE_RATE_HZ))

    # DoA config — CLI overrides sidecar, sidecar overrides defaults
    D_LAMBDA  = args.d_lambda  or float(meta.get("d_lambda",  0.5))
    N_AZ      = args.n_az      or int(meta.get("n_az",  72))
    N_EL      = args.n_el      or int(meta.get("n_el",  18))
    EL_MIN    = args.el_min    or float(meta.get("el_min_deg",  5.0))
    N_SIG     = args.n_signals or int(meta.get("n_signals", 1))
    ALGO      = (args.algo or meta.get("algo", "2D-MUSIC")).upper().replace("-", "_")
    FREQ_HZ   = float(meta.get("freq_hz", C.FREQ_HZ))
    GAIN_DB   = float(meta.get("gain_db", C.GAIN_DB))
    INPUT_ORDER = normalize_cross_array_order(
        meta.get("antenna_input_order", list(CROSS_ARRAY_CANONICAL_ORDER))
    )
    SOLVER_SHORT = short_cross_array_labels(CROSS_ARRAY_CANONICAL_ORDER)

    cfg = CrossArrayConfig(
        d_lambda             = D_LAMBDA,
        n_az                 = N_AZ,
        n_el                 = N_EL,
        el_min_deg           = EL_MIN,
        num_expected_signals = N_SIG,
    )

    _doa_fn   = doa_music_2d if "MUSIC" in ALGO else doa_capon_2d
    SKIP_DOP  = args.no_doppler

    FFT_N = 512
    HIST  = min(N, 120)

    # ── Pre-compute all DoA results ────────────────────────────────────────────
    print(f"[PB] {N} frames  ·  {N_ANT} antennas  ·  {N_SAMP} samples/frame")
    t0 = time.time()

    def _fba(R: np.ndarray) -> np.ndarray:
        M = R.shape[0]; J = np.fliplr(np.eye(M))
        return 0.5 * (R + J @ R.conj() @ J)

    all_spec     = np.zeros((N, N_EL, N_AZ), dtype=np.float32)
    all_az       = np.zeros(N, dtype=np.float32)
    all_el       = np.zeros(N, dtype=np.float32)
    all_papr     = np.full(N, -999.0, dtype=np.float32)   # -999 = rejected
    all_snr      = np.zeros(N, dtype=np.float32)
    all_ev       = np.zeros((N, N_ANT), dtype=np.float32)
    all_coh      = np.zeros((N, N_ANT, N_ANT), dtype=np.float32)
    all_fft      = np.zeros((N, FFT_N), dtype=np.float32)
    all_accepted = np.zeros(N, dtype=bool)
    all_doppler  = np.zeros(N, dtype=np.float64)  # carrier freq offset per burst [Hz]
    all_fdma_ch  = np.zeros(N, dtype=np.int32)    # FDMA channel index (integer)
    all_pass_id  = np.full(N, -1, dtype=np.int32)  # pass index per frame
    all_n_sig    = np.zeros(N, dtype=np.int32)     # MDL-estimated signal count per burst

    # ── Pre-pass: estimate Doppler for every frame (fast — used for pass boundary detection)
    # Different Iridium satellites use different FDMA channels (spacing = 41.667 kHz).
    # A Doppler jump > 20 kHz between consecutive bursts means a channel switch → new satellite.
    print(f"[PB] Pre-pass Doppler estimation...", end=" ", flush=True)
    for _pi in range(N):
        _Xp = reorder_cross_array_channels(frames[_pi].astype(np.complex128), INPUT_ORDER)
        try:
            _, _d = compensate_doppler(_Xp, sample_rate=int(FS))
        except Exception:
            _d = 0.0
        all_doppler[_pi] = _d
        all_fdma_ch[_pi] = int(round(_d / _FDMA_SPACING_HZ))
    print(f"done ({N} frames)")

    # ── Build Doppler-aware pass boundaries
    _n_dop_jumps = 0; _n_time_gaps = 0
    _pass_boundary_set: set = set()
    for _j in range(1, N):
        dop_jump = abs(all_doppler[_j] - all_doppler[_j - 1])
        time_gap = timestamps[_j] - timestamps[_j - 1]
        if dop_jump > _DOPPLER_JUMP_HZ:
            _pass_boundary_set.add(_j); _n_dop_jumps += 1
        elif time_gap > _PASS_GAP_S:
            _pass_boundary_set.add(_j); _n_time_gaps += 1
    n_detected_passes = 1 + len(_pass_boundary_set)
    print(f"[PB] {n_detected_passes} passes detected  "
          f"(Doppler jumps: {_n_dop_jumps}, time gaps: {_n_time_gaps})")

    # ── Build per-pass PassInfo objects ────────────────────────────────────────
    _pass_start_indices = sorted([0] + list(_pass_boundary_set))
    _pass_ends = _pass_start_indices[1:] + [N]
    passes: list[PassInfo] = []
    for _p, (_ps, _pe) in enumerate(zip(_pass_start_indices, _pass_ends)):
        _med_dop = float(np.median(all_doppler[_ps:_pe]))
        _fch = int(round(_med_dop / _FDMA_SPACING_HZ))
        passes.append(PassInfo(
            pass_id   = _p,
            fdma_ch   = _fch,
            color     = _SAT_PALETTE[_p % len(_SAT_PALETTE)],
            marker    = _SAT_MARKERS[_p % len(_SAT_MARKERS)],
            start_idx = _ps,
            end_idx   = _pe - 1,
            acc       = SatellitePassAccumulator(cfg),
        ))
        for _fi in range(_ps, _pe):
            all_pass_id[_fi] = _p

    print(f"[PB] Running {ALGO.replace('_','-')}  {N_AZ}az × {N_EL}el …", end=" ", flush=True)

    pass_acc = SatellitePassAccumulator(cfg)
    win      = np.hanning(FFT_N)

    n_rejected_uw = 0; n_rejected_spread = 0

    for i in range(N):
        X_input = frames[i].astype(np.complex128)
        X = reorder_cross_array_channels(X_input, INPUT_ORDER)

        # ── Doppler compensation (re-run to actually apply the phase correction to X)
        if not SKIP_DOP:
            try:
                X, _ = compensate_doppler(X, sample_rate=int(FS))
            except Exception:
                pass

        # ── Narrowband BPF (28 kHz, Butterworth IIR order 8)
        X_filt = narrowband_filter_burst(X, sample_rate=int(FS))

        # ── UW correlation gate
        _, uw_score = validate_burst_uw(X_filt, sample_rate=int(FS))
        if uw_score < _UW_SCORE_MIN:
            n_rejected_uw += 1
            # Still fill FFT and ev for display (use unfiltered X)
            R_raw = compute_single_shot_covariance(X)
            all_ev[i]  = eigenvalue_spread_db(R_raw).astype(np.float32)
            all_coh[i] = coherence_matrix(R_raw).astype(np.float32)
            seg = X[0, :FFT_N] if X.shape[1] >= FFT_N else np.pad(X[0], (0, FFT_N - X.shape[1]))
            all_fft[i] = np.clip(
                20.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(seg * win))) + 1e-12), -80.0, 0.0
            ).astype(np.float32)
            # Carry forward previous valid spectrum if available
            if i > 0:
                all_spec[i] = all_spec[i - 1]
                all_az[i]   = all_az[i - 1]
                all_el[i]   = all_el[i - 1]
            continue

        # ── Covariance + FBA
        R = compute_single_shot_covariance(X_filt)
        R = _fba(R)

        # ── Eigenvalue spread gate
        ev = eigenvalue_spread_db(R)
        spread = float(ev[0] - ev[-1])
        if spread < _EIG_SPREAD_MIN_DB:
            n_rejected_spread += 1
            all_ev[i]  = ev.astype(np.float32)
            all_coh[i] = coherence_matrix(R).astype(np.float32)
            seg = X[0, :FFT_N] if X.shape[1] >= FFT_N else np.pad(X[0], (0, FFT_N - X.shape[1]))
            all_fft[i] = np.clip(
                20.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(seg * win))) + 1e-12), -80.0, 0.0
            ).astype(np.float32)
            if i > 0:
                all_spec[i] = all_spec[i - 1]
                all_az[i]   = all_az[i - 1]
                all_el[i]   = all_el[i - 1]
            continue

        # ── MDL signal count estimation (diagnostic — does NOT override D for MUSIC)
        #
        # MDL frequently over-estimates D (3–4) after narrowband filtering because:
        #   (a) residual FDMA carriers leak through the 28 kHz BPF
        #   (b) FBA introduces eigenvalue structure
        #   (c) N_samples=10690 >> M²=25, making MDL overly aggressive
        # For MUSIC resolution: D=1 (→ 4 noise eigenvectors) is always optimal
        # for single-satellite bursts.  MDL=0 means "pure noise" → flag it.
        n_sig_mdl = estimate_signal_count(R, N_SAMP, method="mdl", max_signals=3)
        all_n_sig[i] = n_sig_mdl

        # Use D from CLI/config (default 1).  Only override if MDL says 0 (noise).
        cfg_i = cfg
        if n_sig_mdl == 0:
            # MDL says no signal — still process but flag for user
            pass

        # ── 2D DoA
        spec = _doa_fn(X_filt, cfg_i, R_in=R)
        az, el, papr = find_peak_2d(spec, cfg)
        snr  = snr_from_covariance(R)
        coh  = coherence_matrix(R)

        # ── Pass accumulator (global + per-pass)
        _is_new_pass = i in _pass_boundary_set
        pass_acc.update(spec, papr_db=papr,
                        new_pass=_is_new_pass, papr_min_db=_PAPR_MIN_DB_ACC)

        # Per-pass accumulation
        _pid = int(all_pass_id[i])
        if 0 <= _pid < len(passes):
            pi = passes[_pid]
            pi.acc.update(spec, papr_db=papr, papr_min_db=_PAPR_MIN_DB_ACC)
            pi.indices.append(i)
            pi.az_list.append(float(az))
            pi.el_list.append(float(el))
            pi.papr_list.append(float(papr))
            pi.dop_list.append(float(all_doppler[i]))
            pi.t_list.append(float(timestamps[i]))

        all_spec[i]     = spec.astype(np.float32)
        all_az[i]       = az;    all_el[i]   = el
        all_papr[i]     = papr;  all_snr[i]  = snr
        all_ev[i]       = ev.astype(np.float32)
        all_coh[i]      = coh.astype(np.float32)
        all_accepted[i] = True

        # ── FFT ch0 (post-filter)
        seg = X_filt[0, :FFT_N] if X_filt.shape[1] >= FFT_N else np.pad(X_filt[0], (0, FFT_N - X_filt.shape[1]))
        fft_db = np.clip(20.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(seg * win))) + 1e-12), -80.0, 0.0)
        all_fft[i] = fft_db.astype(np.float32)

        if (i + 1) % 20 == 0 or i == N - 1:
            print(f"\r[PB] {i+1}/{N} frames  accepted={all_accepted.sum()}  "
                  f"rej_uw={n_rejected_uw}  rej_spread={n_rejected_spread}  "
                  f"({time.time()-t0:.1f}s) …", end="", flush=True)

    n_accepted = int(all_accepted.sum())
    print(f"\n[PB] Done in {time.time()-t0:.1f}s  —  "
          f"{n_accepted}/{N} accepted  "
          f"({n_rejected_uw} rej UW, {n_rejected_spread} rej spread)  "
          f"— {n_detected_passes} passes (gap>{_PASS_GAP_S:.0f}s)")

    # ── MDL statistics
    _mdl_vals = all_n_sig[all_accepted]
    if len(_mdl_vals):
        print(f"[PB] MDL D estimates: D=0:{np.sum(_mdl_vals==0)} "
              f"D=1:{np.sum(_mdl_vals==1)} D=2:{np.sum(_mdl_vals==2)} "
              f"D≥3:{np.sum(_mdl_vals>=3)}")

    # ── Sky density map — PAPR²-weighted 2D histogram of per-burst Az/El peaks
    from scipy.ndimage import gaussian_filter as _gf
    _az_grid = cfg.az_range_deg()   # (N_AZ,)
    _el_grid = cfg.el_range_deg()   # (N_EL,)
    _density = np.zeros((N_EL, N_AZ), dtype=np.float64)
    for _i in range(N):
        if not all_accepted[_i] or all_papr[_i] < _PAPR_MIN_DB_ACC:
            continue
        _i_az = int(np.argmin(np.abs(_az_grid - all_az[_i])))
        _i_el = int(np.argmin(np.abs(_el_grid - all_el[_i])))
        _density[_i_el, _i_az] += float(all_papr[_i]) ** 2  # PAPR² weight
    _density_sm = _gf(_density, sigma=1.5)
    _dens_max = float(_density_sm.max())
    acc_norm = (_density_sm / (_dens_max + 1e-30)).astype(np.float32)

    # Per-pass density maps
    pass_density = {}
    for pi in passes:
        if not pi.indices:
            continue
        _pd = np.zeros((N_EL, N_AZ), dtype=np.float64)
        for _i in pi.indices:
            if all_papr[_i] < _PAPR_MIN_DB_ACC:
                continue
            _i_az = int(np.argmin(np.abs(_az_grid - all_az[_i])))
            _i_el = int(np.argmin(np.abs(_el_grid - all_el[_i])))
            _pd[_i_el, _i_az] += float(all_papr[_i]) ** 2
        _pd_sm = _gf(_pd, sigma=1.5)
        _pd_mx = float(_pd_sm.max())
        pass_density[pi.pass_id] = (_pd_sm / (_pd_mx + 1e-30)).astype(np.float32)

    # Density peak → best overall sky position
    _dp_idx = np.unravel_index(np.argmax(_density_sm), _density_sm.shape)
    best_az  = float(_az_grid[_dp_idx[1]])
    best_el  = float(_el_grid[_dp_idx[0]])
    best_papr = float(_dens_max ** 0.5)   # sqrt of summed PAPR² ≈ effective PAPR

    # Per-pass console summary
    print(f"[PB] Per-pass summary  (Doppler threshold: {_DOPPLER_JUMP_HZ/1e3:.0f} kHz, "
          f"time gap: {_PASS_GAP_S:.0f} s):")
    for pi in passes:
        if not pi.indices:
            continue
        n_acc = pi.n_accepted
        _p_t    = pi.t_list[0] if pi.t_list else 0.0
        # Per-pass best estimate from accumulator
        _best_az, _best_el, _best_papr = pi.acc.get_best_estimate() if pi.acc else (0, 0, 0)
        if n_acc == 1:
            print(f"  {pi.label:40s}  t={_p_t:.0f}s  1 burst   "
                  f"Az={pi.az_list[0]:.1f}°  El={pi.el_list[0]:.1f}°  PAPR={pi.papr_list[0]:.1f} dB")
        else:
            _az_std = float(np.std(pi.az_list))
            print(f"  {pi.label:40s}  t={_p_t:.0f}s  {n_acc} bursts  "
                  f"Az: med={np.median(pi.az_list):.1f}° ±{_az_std:.1f}°  "
                  f"El: med={np.median(pi.el_list):.1f}°  "
                  f"PAPR: med={np.median(pi.papr_list):.1f} dB  "
                  f"Best▸ Az={_best_az:.1f}° El={_best_el:.1f}°")
    print(f"[PB] Sky density peak: Az={best_az:.1f}°  El={best_el:.1f}°  "
          f"(N_acc={n_accepted}, N_in_density={pass_acc.n_bursts})")

    # ── GUI ───────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": C_BORDER, "axes.grid": True,
        "grid.color": C_BORDER, "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "text.color": C_TEXT,
    })

    fig = plt.figure(figsize=(22, 10.5), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        left=0.04, right=0.98, top=0.93, bottom=0.14,
        hspace=0.48, wspace=0.38,
    )

    ax_sky  = fig.add_subplot(gs[0, 0], polar=True)
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_eig  = fig.add_subplot(gs[0, 3])
    ax_coh  = fig.add_subplot(gs[1, 0])
    ax_pq   = fig.add_subplot(gs[1, 1])
    ax_fft  = fig.add_subplot(gs[1, 2])
    ax_dop  = fig.add_subplot(gs[1, 3])    # replaces "Phase stability"

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_color(C_BORDER); sp.set_linewidth(0.8)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        if title:  ax.set_title(title, color=C_TEXT, fontsize=8.5, pad=5, fontweight="semibold")
        if xlabel: ax.set_xlabel(xlabel, color=C_MUTED, fontsize=7)
        if ylabel: ax.set_ylabel(ylabel, color=C_MUTED, fontsize=7)

    # Panel 0: Sky plot ────────────────────────────────────────────────────────
    ax_sky.set_facecolor(BG2)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90)
    ax_sky.set_rlabel_position(112.5)
    ax_sky.tick_params(colors=C_MUTED, labelsize=6.5)
    ax_sky.set_rticks([0, 30, 60, 90])
    ax_sky.set_yticklabels(["90°", "60°", "30°", "0°"], color=C_MUTED, fontsize=6)
    ax_sky.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax_sky.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                            color=C_MUTED, fontsize=6.5)
    ax_sky.set_title("Sky Plot — Per-satellite tracks", color=C_TEXT, fontsize=8.5, pad=10, fontweight="semibold")
    ax_sky.grid(color=C_BORDER, linewidth=0.5, alpha=0.6)
    ax_sky.spines["polar"].set_color(C_BORDER)

    # Per-pass trajectory lines + scatter with distinct markers/colors
    _papr_max = float(all_papr[all_accepted].max()) if n_accepted > 0 else 1.0
    for pi in passes:
        if not pi.indices:
            continue
        # ── Trajectory line (connected burst positions, chronological)
        _t_pts = [skyplot_coords(az, el) for az, el in zip(pi.az_list, pi.el_list)]
        if len(_t_pts) >= 2:
            _th_arr = [p[0] for p in _t_pts]
            _r_arr  = [p[1] for p in _t_pts]
            ax_sky.plot(_th_arr, _r_arr, color=pi.color, linewidth=1.2,
                        alpha=0.35, zorder=1, linestyle="-")
        # ── Scatter points with per-pass marker
        for _k, _idx in enumerate(pi.indices):
            _t_i, _r_i = skyplot_coords(float(all_az[_idx]), float(all_el[_idx]))
            _alpha = float(np.clip((all_papr[_idx] - 1.0) / (_papr_max - 1.0 + 1e-6), 0.15, 0.9))
            _sz = 4.0 + 4.0 * float(np.clip(all_papr[_idx] / (_papr_max + 1e-6), 0, 1))
            _lbl = pi.label if _k == 0 else None
            ax_sky.plot(_t_i, _r_i, pi.marker, color=pi.color, markersize=_sz,
                        alpha=_alpha, zorder=2, markeredgecolor="none", label=_lbl)
        # ── Per-pass accumulated best estimate (large marker)
        _best_az_p, _best_el_p, _best_papr_p = pi.acc.get_best_estimate() if pi.acc else (0, 0, 0)
        if _best_papr_p > 0:
            _tb, _rb = skyplot_coords(_best_az_p, _best_el_p)
            ax_sky.plot(_tb, _rb, pi.marker, color=pi.color, markersize=12,
                        markeredgecolor=BG, markeredgewidth=1.5, zorder=5)
    # Compact multi-column legend (only if multiple passes)
    if n_detected_passes > 1:
        ax_sky.legend(loc="lower left", fontsize=5.0, framealpha=0.55,
                      facecolor=BG3, edgecolor=C_BORDER, markerscale=1.4,
                      ncol=max(1, n_detected_passes // 4))
    # Mark accumulated best-estimate on sky plot
    _t_best, _r_best = skyplot_coords(best_az, best_el)
    ax_sky.plot(_t_best, _r_best, "*", color=C_LIME, markersize=14,
                markeredgecolor=BG, markeredgewidth=1.0, zorder=6)

    th_edges, r_edges = make_sky_heatmap_edges(cfg)
    T_e, R_e = np.meshgrid(th_edges, r_edges)
    sky_mesh = ax_sky.pcolormesh(T_e, R_e, np.flipud(all_spec[0]),
                                  cmap="inferno", vmin=-40, vmax=0, shading="flat")

    sky_peak,       = ax_sky.plot([], [], "o", color=C_LIME, markersize=8,
                                   markeredgecolor=BG, markeredgewidth=1.5, zorder=5)
    sky_peak_outer, = ax_sky.plot([], [], "o", color="none", markersize=14,
                                   markeredgecolor=C_LIME, markeredgewidth=1.0, zorder=4)
    sky_az_line,    = ax_sky.plot([0, 0], [0, 90], color=C_LIME,
                                   linewidth=0.6, alpha=0.4, zorder=3)

    txt_sky = ax_sky.text(
        0.5, -0.07, "", transform=ax_sky.transAxes,
        ha="center", fontsize=7.5, color=C_TEXT, fontweight="semibold",
    )

    # Panel 1: Hot zone accumulated map ───────────────────────────────────────
    _style(ax_heat, f"Sky Density  ({n_accepted}/{N} acc  ·  {n_detected_passes} passes)", "Azimuth [°]", "Elevation [°]")
    az_c, el_c = cfg.az_range_deg(), cfg.el_range_deg()
    heat_img   = ax_heat.imshow(
        acc_norm, aspect="auto", origin="lower",
        extent=[az_c[0], az_c[-1], el_c[0], el_c[-1]],
        cmap="hot", vmin=0, vmax=1,
    )
    # Per-pass contours at 50% of each pass's density peak
    for pi in passes:
        if pi.pass_id not in pass_density:
            continue
        _pd = pass_density[pi.pass_id]
        try:
            ax_heat.contour(az_c, el_c, _pd,
                            levels=[0.5], colors=[pi.color],
                            linewidths=[1.0], alpha=0.85)
        except Exception:
            pass
    # Global contour at 80%
    try:
        ax_heat.contour(az_c, el_c, acc_norm,
                        levels=[0.8], colors=[C_LIME],
                        linewidths=[1.2], alpha=0.9)
    except Exception:
        pass
    # Mark accumulated peak (global)
    ax_heat.plot(best_az, best_el, "*", color=C_LIME, markersize=12,
                 markeredgecolor=BG, markeredgewidth=1.0, zorder=6)
    # Mark per-pass best positions
    for pi in passes:
        if not pi.indices:
            continue
        _baz, _bel, _bp = pi.acc.get_best_estimate() if pi.acc else (0, 0, 0)
        if _bp > 0:
            ax_heat.plot(_baz, _bel, pi.marker, color=pi.color, markersize=8,
                         markeredgecolor=BG, markeredgewidth=0.8, zorder=5)
    heat_vline = ax_heat.axvline(best_az, color=C_AMBER, linewidth=1.0, alpha=0.6, linestyle="--")
    heat_hline = ax_heat.axhline(best_el, color=C_AMBER, linewidth=1.0, alpha=0.6, linestyle="--")
    txt_heat   = ax_heat.text(0.02, 0.97, f"Best: Az={best_az:.1f}°  El={best_el:.1f}°",
                               transform=ax_heat.transAxes,
                               fontsize=7, color=C_TEXT, va="top",
                               bbox=dict(facecolor=BG3, edgecolor="none", alpha=0.7))

    # Panel 2: Az + El history ─────────────────────────────────────────────────
    _style(ax_hist, "Az + El  (last frames)", "frame", "")
    ax_hist_el = ax_hist.twinx()
    ax_hist_el.set_facecolor(BG2)
    ax_hist_el.tick_params(colors=C_VIOLET, labelsize=7)
    ax_hist_el.set_ylabel("Elevation [°]", color=C_VIOLET, fontsize=7)
    ax_hist.set_ylabel("Azimuth [°]", color=C_AMBER, fontsize=7)
    ax_hist.set_ylim(0, 360); ax_hist_el.set_ylim(0, 90)
    x_h = np.arange(HIST)
    ax_hist.set_xlim(0, HIST - 1); ax_hist_el.set_xlim(0, HIST - 1)
    line_az, = ax_hist.plot(x_h, np.zeros(HIST), color=C_AMBER, linewidth=1.4)
    line_el, = ax_hist_el.plot(x_h, np.full(HIST, 45.0), color=C_VIOLET, linewidth=1.4)
    ax_hist.legend(handles=[line_az, line_el], labels=["Az", "El"],
                   loc="upper right", fontsize=6.5, framealpha=0.4,
                   facecolor=BG3, edgecolor=C_BORDER)

    # Panel 3: Eigenvalues ─────────────────────────────────────────────────────
    _style(ax_eig, "Eigenvalues  (noise floor = 0 dB)", "", "dB")
    ax_eig.set_xlim(-0.5, N_ANT - 0.5)
    ax_eig.set_xticks(range(N_ANT))
    ax_eig.set_xticklabels([f"λ{k}" for k in range(N_ANT)], color=C_MUTED, fontsize=7)
    ax_eig.set_ylim(-2, 40)
    bars_eig = ax_eig.bar(range(N_ANT), [0.0] * N_ANT,
                          color=[C_BLUE, C_TEAL, C_TEAL, C_TEAL, C_BORDER],
                          edgecolor="none", alpha=0.9)
    ax_eig.axhline(0, color=C_BORDER, linewidth=0.6)
    txt_eig = ax_eig.text(0.98, 0.97, "", transform=ax_eig.transAxes,
                          ha="right", fontsize=7, color=C_MUTED, va="top")

    # Panel 4: Coherence ───────────────────────────────────────────────────────
    _style(ax_coh, "Coherence  |ρ_{ij}|", "", "")
    coh_img = ax_coh.imshow(np.eye(N_ANT), cmap="viridis", vmin=0, vmax=1,
                             aspect="equal", origin="lower")
    ax_coh.set_xticks(range(N_ANT)); ax_coh.set_yticks(range(N_ANT))
    ax_coh.set_xticklabels(SOLVER_SHORT, fontsize=6, color=C_MUTED)
    ax_coh.set_yticklabels(SOLVER_SHORT, fontsize=6, color=C_MUTED)
    fig.colorbar(coh_img, ax=ax_coh, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # Panel 5: PAPR + SNR ──────────────────────────────────────────────────────
    _style(ax_pq, "PAPR + SNR History", "frame", "dB")
    ax_pq.set_xlim(0, HIST - 1); ax_pq.set_ylim(-2, 45)
    ax_pq_snr = ax_pq.twinx()
    ax_pq_snr.set_facecolor(BG2); ax_pq_snr.set_ylim(-2, 35)
    ax_pq_snr.tick_params(colors=C_ROSE, labelsize=7)
    ax_pq_snr.set_ylabel("SNR [dB]", color=C_ROSE, fontsize=7)
    ax_pq.set_ylabel("PAPR [dB]", color=C_TEAL, fontsize=7)
    line_papr, = ax_pq.plot(x_h, np.zeros(HIST), color=C_TEAL, linewidth=1.4)
    line_snr,  = ax_pq_snr.plot(x_h, np.zeros(HIST), color=C_ROSE, linewidth=1.4)
    ax_pq.legend(handles=[line_papr, line_snr], labels=["PAPR", "SNR"],
                 loc="upper right", fontsize=6.5, framealpha=0.4,
                 facecolor=BG3, edgecolor=C_BORDER)

    # Panel 6: IQ FFT ──────────────────────────────────────────────────────────
    _style(ax_fft, "IQ Spectrum  ch0", "offset [kHz]", "dB")
    freq_axis = np.fft.fftshift(np.fft.fftfreq(FFT_N)) * FS / 1e3
    line_fft, = ax_fft.plot(freq_axis, all_fft[0], color=C_BLUE, linewidth=0.9)
    ax_fft.set_xlim(freq_axis[0], freq_axis[-1]); ax_fft.set_ylim(-80, 5)
    ax_fft.axvline(0, color=C_ROSE, linewidth=0.7, linestyle="--", alpha=0.5)

    # Panel 7: Doppler per satellite (replaces Phase stability) ──────────────
    _style(ax_dop, "Doppler per satellite", "time [s]", "Doppler [kHz]")
    _t_axis_s = timestamps / 1000.0
    for pi in passes:
        if not pi.indices:
            continue
        _dt = np.array(pi.t_list) / 1000.0
        _dd = np.array(pi.dop_list) / 1e3  # kHz
        ax_dop.plot(_dd, color=pi.color, linewidth=1.2, alpha=0.85,
                    marker=pi.marker, markersize=3, markeredgecolor="none",
                    label=pi.label)
    ax_dop.legend(loc="best", fontsize=5.0, framealpha=0.5,
                  facecolor=BG3, edgecolor=C_BORDER,
                  ncol=max(1, n_detected_passes // 3))
    dop_vline = ax_dop.axvline(0, color=C_AMBER, linewidth=1.0, alpha=0.6, linestyle="--")

    fig.suptitle(
        f"Space DoA — Playback  │  {os.path.basename(npz_path)}  │  "
        f"{ALGO.replace('_','-')}  │  {N} frames  │  {FREQ_HZ/1e6:.4f} MHz",
        color=C_TEXT, fontsize=9.5, fontweight="semibold", y=0.980,
    )

    # ── Playback controls ─────────────────────────────────────────────────────
    CTL_Y = 0.020; CTL_H = 0.042
    ax_rew   = fig.add_axes((0.040, CTL_Y, 0.044, CTL_H))
    ax_play  = fig.add_axes((0.088, CTL_Y, 0.044, CTL_H))
    ax_sld   = fig.add_axes((0.145, CTL_Y, 0.350, CTL_H))
    ax_sp025 = fig.add_axes((0.508, CTL_Y, 0.035, CTL_H))
    ax_sp05  = fig.add_axes((0.546, CTL_Y, 0.035, CTL_H))
    ax_sp1   = fig.add_axes((0.584, CTL_Y, 0.035, CTL_H))
    ax_sp2   = fig.add_axes((0.622, CTL_Y, 0.035, CTL_H))
    ax_sp4   = fig.add_axes((0.660, CTL_Y, 0.035, CTL_H))

    btn_rew   = Button(ax_rew,   "◀◀",   color=BG3, hovercolor="#3a4060")
    btn_play  = Button(ax_play,  "▶",    color=BG3, hovercolor="#3a4060")
    sld_frame = Slider(ax_sld, "", 0, N - 1, valinit=0, valstep=1,
                       color=C_BLUE, track_color=BG3)

    # Speed buttons
    for ax_s in (ax_sp025, ax_sp05, ax_sp1, ax_sp2, ax_sp4):
        ax_s.set_facecolor(BG3)

    _speed_btns = {}
    for ax_s, lbl in zip(
            (ax_sp025, ax_sp05, ax_sp1, ax_sp2, ax_sp4),
            ("×¼",     "×½",    "×1",   "×2",   "×4")):
        b = Button(ax_s, lbl, color=BG3, hovercolor="#3a4060")
        b.label.set_color(C_TEXT); b.label.set_fontsize(7.5)
        _speed_btns[lbl] = b

    for b in (btn_rew, btn_play):
        b.label.set_color(C_TEXT); b.label.set_fontsize(9.5)

    sld_frame.label.set_color(C_MUTED)
    sld_frame.valtext.set_color(C_TEXT)

    # Frame label next to slider
    txt_frame = fig.text(
        0.498, CTL_Y + CTL_H / 2 + 0.005, f"0/{N-1}",
        color=C_TEXT, fontsize=7.5, ha="right", va="center",
    )

    # ── Playback state ────────────────────────────────────────────────────────
    class PB:
        playing   = False
        cur_frame = 0
        speed     = 1.0
        t_last    = time.time()

    # ── Render a single frame ─────────────────────────────────────────────────
    def _render(i: int):
        i = int(np.clip(i, 0, N - 1))
        spec = all_spec[i];  az = float(all_az[i]);  el = float(all_el[i])
        papr = float(all_papr[i]); snr = float(all_snr[i])
        ev   = all_ev[i];    coh = all_coh[i]
        fft_d = all_fft[i]
        accepted_i = bool(all_accepted[i])

        # History window centred on current frame
        i0 = max(0, i - HIST + 1);  i1 = i + 1
        az_w  = np.pad(all_az[i0:i1],   (HIST - (i1-i0), 0), constant_values=0.0)
        el_w  = np.pad(all_el[i0:i1],   (HIST - (i1-i0), 0), constant_values=45.0)
        # Clamp -999 (rejected) to 0 for PAPR history display
        pq_raw = all_papr[i0:i1].copy(); pq_raw[pq_raw < 0] = 0.0
        pq_w  = np.pad(pq_raw,           (HIST - (i1-i0), 0), constant_values=0.0)
        snr_w = np.pad(all_snr[i0:i1],  (HIST - (i1-i0), 0), constant_values=0.0)

        # Sky plot
        sky_mesh.set_array(np.flipud(spec).ravel())
        t_pk, r_pk = skyplot_coords(az, el)
        sky_peak.set_data([t_pk], [r_pk])
        sky_peak_outer.set_data([t_pk], [r_pk])
        sky_az_line.set_data([t_pk, t_pk], [0, 90])
        txt_sky.set_text(
            f"Az: {az:6.1f}°   El: {el:5.1f}°   PAPR: {papr:.1f} dB   D={int(all_n_sig[i])}"
            if accepted_i else
            f"Az: —   El: —   (rejected)"
        )

        # Hot zone (static accumulated map) — only update crosshair for current frame
        heat_vline.set_xdata([az, az])
        heat_hline.set_ydata([el, el])
        txt_heat.set_text(
            f"Best(accum): Az={best_az:.1f}°  El={best_el:.1f}°\n"
            f"Frame {i}: "
            + (f"Az={az:.1f}°  El={el:.1f}°  PAPR={papr:.1f}dB  "
               f"Dop={all_doppler[i]/1e3:+.1f}kHz  D={int(all_n_sig[i])}"
               if accepted_i else "(rejected)")
        )

        # History
        line_az.set_ydata(az_w); line_el.set_ydata(el_w)

        # Eigenvalues
        for bar, val in zip(bars_eig, ev):
            bar.set_height(max(float(val), 0.0))
        spread = float(ev[0]) - float(ev[-1]) if len(ev) > 1 else 0.0
        txt_eig.set_text(f"spread={spread:.0f}dB")
        bars_eig[0].set_color(C_BLUE if spread > 5.0 else C_DIM)

        # Coherence
        coh_img.set_data(coh)

        # PAPR + SNR
        line_papr.set_ydata(pq_w); line_snr.set_ydata(snr_w)

        # FFT
        line_fft.set_ydata(fft_d)

        # Doppler panel: update vertical cursor to current frame's burst index
        # Find which burst-within-pass is current frame
        _pid = int(all_pass_id[i])
        if 0 <= _pid < len(passes) and i in passes[_pid].indices:
            _bidx = passes[_pid].indices.index(i)
            dop_vline.set_xdata([_bidx, _bidx])
        else:
            dop_vline.set_xdata([-1, -1])  # hide

        # Slider + frame label (eventson=False prevents re-entrant _on_slider call)
        sld_frame.eventson = False
        sld_frame.set_val(i)
        sld_frame.eventson = True
        ts_s = float(timestamps[i]) / 1000.0
        txt_frame.set_text(f"#{i}  t={ts_s:.1f}s")

    # ── Animation ─────────────────────────────────────────────────────────────
    def _animate(_):
        if PB.playing:
            now = time.time()
            dt  = now - PB.t_last
            if dt >= C.INTERVAL_MS / 1000.0 / PB.speed:
                PB.t_last = now
                PB.cur_frame = (PB.cur_frame + 1) % N
                _render(PB.cur_frame)

    ani = animation.FuncAnimation(fig, _animate, interval=30, cache_frame_data=False)

    # ── Button callbacks ───────────────────────────────────────────────────────
    def _on_rew(_):
        PB.playing = False; PB.cur_frame = 0
        btn_play.label.set_text("▶")
        _render(0)

    def _on_play(_):
        PB.playing = not PB.playing
        btn_play.label.set_text("‖" if PB.playing else "▶")
        PB.t_last = time.time()

    def _on_slider(val):
        PB.cur_frame = int(val)
        _render(PB.cur_frame)

    def _make_speed(factor):
        def _cb(_):
            PB.speed = factor
            for lbl, b in _speed_btns.items():
                f_map = {"×¼": 0.25, "×½": 0.5, "×1": 1.0, "×2": 2.0, "×4": 4.0}
                b.color = C_BORDER if abs(f_map[lbl] - factor) < 0.01 else BG3
            fig.canvas.draw_idle()
        return _cb

    btn_rew.on_clicked(_on_rew)
    btn_play.on_clicked(_on_play)
    sld_frame.on_changed(_on_slider)

    for lbl, factor in [("×¼", 0.25), ("×½", 0.5), ("×1", 1.0), ("×2", 2.0), ("×4", 4.0)]:
        _speed_btns[lbl].on_clicked(_make_speed(factor))

    # Keyboard shortcuts
    def _on_key(event):
        if event.key == " ":       _on_play(None)
        elif event.key == "left":
            PB.cur_frame = max(0, PB.cur_frame - 1);          _render(PB.cur_frame)
        elif event.key == "right":
            PB.cur_frame = min(N-1, PB.cur_frame + 1);        _render(PB.cur_frame)
        elif event.key == "home":  _on_rew(None)
        elif event.key == "end":   PB.cur_frame = N-1;         _render(PB.cur_frame)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # Initial render
    _render(0)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
