# Iridium DOA — Operations Handbook

Direction-of-Arrival (DOA) estimation of **Iridium NEXT satellite bursts** at
1626 MHz, using a **KrakenSDR** (5 coherent RTL-SDR channels) and a uniform
circular array (UCA) of 5 antennas. Pure Python — no GNU Radio.

This document is the practical guide: how the pipeline works, every command
with its use case, field procedures for outdoor measurements, and when/how to
calibrate. It is written for someone picking up the project from scratch.

Validated performance — reference session 2026-06-05 (36 min outdoor),
processed with `--el-min 9`, evaluated against **epoch-matched** SGP4
elements (Space-Track, median epoch distance 2 h — see §4.4), 1599 bursts
uniquely Doppler-assigned to 10 satellites:

| Per-burst error | bias | MedAE | RMSE |
|---|---|---|---|
| azimuth, single-burst | +1.7° | 4.1° | 9.4° |
| azimuth, multi-burst B=16 (§4.5) | +1.5° | **3.7°** | 9.0° |
| elevation, single-burst | −2.1° | 3.9° | 8.0° |
| elevation, multi-burst B=16 (§4.5) | −1.9° | **3.3°** | 6.9° |

RMSE ≫ MedAE flags the heavy outlier tail (sidelobes/reflections) — the
robust track smoothing (§4.5) exists to reject it. Receiver LO offset,
estimated automatically: ≈ +3.5 kHz (+2.1 ppm). Numbers quoted elsewhere
in this document (e.g. calibration history, smoothing) were measured on the
earlier `--el-min 5` processing of the same session; absolute values shift
with the elevation gate and the TLE epoch, comparisons within one dataset
do not.

---

## Table of contents

1. [How it works — the signal path](#1-how-it-works--the-signal-path)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Workflows (commands + use cases)](#4-workflows)
   - 4.1 [Live DOA](#41-live-doa)
   - 4.2 [Record a session](#42-record-a-session)
   - 4.3 [Offline reprocessing](#43-offline-reprocessing)
   - 4.4 [Ground truth & accuracy evaluation](#44-ground-truth--accuracy-evaluation)
   - 4.5 [Track smoothing (Kalman/RTS)](#45-track-smoothing-kalmanrts)
   - 4.6 [Array calibration](#46-array-calibration)
   - 4.7 [Visualization](#47-visualization)
   - 4.8 [Utilities & diagnostics](#48-utilities--diagnostics)
   - 4.9 [Tests](#49-tests)
5. [Data formats](#5-data-formats)
6. [Field guide — outdoor measurements](#6-field-guide--outdoor-measurements)
7. [Calibration policy — when and how](#7-calibration-policy--when-and-how)
8. [Troubleshooting](#8-troubleshooting)
9. [Known limitations](#9-known-limitations)
10. [Glossary](#10-glossary)

---

## 1. How it works — the signal path

Iridium satellites transmit downlink bursts in the 1616–1626.5 MHz band. Each
**IRA (Iridium Ring Alert)** burst starts with an unmodulated preamble tone at
`f_c + 3125 Hz`, shifted by up to ±40 kHz of Doppler. That tone is a gift: a
pure CW carrier, perfect for measuring the relative phase between antennas —
which is exactly what DOA needs.

One **CPI** (Coherent Processing Interval) is a block of coherent IQ samples
from all 5 channels, shape `(5, n_samples)` at 1.024 MS/s. For each CPI:

```
 (5, N) IQ frame
      │
 1. detect_energy_bursts()     block power vs median noise floor, summed
      │                        non-coherently over all 5 channels
      ▼  burst offsets
 2. scan_preamble_tones()      FFT scan ±45 kHz around the nominal tone;
      │                        power spectra summed over channels; sub-bin
      │                        peak frequency via parabolic interpolation
      ▼  one tone per satellite (tone − 3125 Hz = Doppler)
 3. apply_bpf_and_normalize()  Hann-tapered band-pass around the tone,
      │                        per-channel unit-RMS normalization
      ▼
 4. apply_phase_correction()   subtract the per-channel calibration phases
      │                        (cal_tle.npz — see §7)
      ▼
 5. compute_mf_covariance()    matched-filter projection onto the tone:
      │                        y_k = Σ x_k[n]·e^{−j2πf·n/fs};  R = y·yᴴ
      ▼  rank-1 array response
 6. doa_music_uca_2d()         scan the (azimuth, elevation) grid with the
      │                        UCA steering model
      ▼
 7. find_peak_uca_2d()         spectrum peak + parabolic refinement
      ▼
 (az°, el°, PAPR, SNR, CFO)    one estimate per satellite per burst
```

All of this lives in `core/` (`burst_processing.py`, `multi_peak.py`,
`doa_uca_2d.py`) and is orchestrated per-CPI by
`core.multi_peak.process_cpi_for_multi()`.

**Three facts worth internalizing before touching anything:**

1. **The MF covariance is rank-1**, so MUSIC, Capon and Bartlett all peak at
   the same (az, el) — choosing the "best algorithm" is not the lever here.
   **Phase calibration is** (it took azimuth bias from +98° to +1.7° on real
   data). There is no single-burst super-resolution; PAPR measures how well
   the array response matches the steering model, not source separation.
2. **Every preamble tone is one satellite.** Doppler (CFO) is the cleanest
   discriminator between simultaneous satellites and the key to matching
   bursts against TLE predictions.
3. **Wall-clock time matters.** Live recording drops CPIs when processing
   lags, so `frame_index × CPI_duration` is NOT wall time (it drifted 10+
   minutes over a 36-min session before this was fixed). Frame timestamps now
   come from `raw/timestamps.csv` (or file mtimes for old sessions) — see §5.

**Conventions:** azimuth in degrees clockwise from geographic North
(0°=N, 90°=E), elevation in degrees above the horizon. UCA: antenna *k* at
angle `ant0_offset_deg + k·72°` (clockwise; `ant_ccw=true` flips the
direction), radius 0.4253 λ → adjacent spacing exactly λ/2 at 1626 MHz.

---

## 2. Repository layout

```
krakenSDR/src/
├── core/                    DSP library (hardware-independent, unit-tested)
│   ├── burst_processing.py  detection, tone scan, BPF, MF covariance
│   ├── multi_peak.py        whole-CPI processing (all bursts, all tones)
│   ├── doa_uca_2d.py        UCA steering model + MUSIC/Capon/Bartlett
│   ├── doa_algorithms.py    phase correction + 1D reference algorithms
│   ├── covariance.py        covariance estimators / decorrelation
│   ├── recording.py         SessionRecorder + frame iterators + timing
│   ├── track_clusterer.py   burst → satellite-track association
│   ├── pipeline_debug.py    per-stage npy dumps (--debug-dir)
│   └── gates / tone_extraction / tracking — legacy-only (see docstrings)
├── apps/doa_iridium/        the application
│   ├── run_doa.py           live pipeline (Heimdall TCP → DOA)
│   ├── run_doa_offline.py   offline pipeline (recordings → DOA)
│   ├── reprocess_session.py multi-peak offline reprocess (the main one)
│   ├── batch_reprocess.py   reprocess with all 3 algorithms at once
│   ├── offline_viz.py / offline_replay.py / algo_compare.py   plotting
│   ├── consolidate_session.py  raw/ frames → single raw_iq.npz
│   ├── doa_config.toml      THE config file (defaults are calibrated)
│   └── cal_tle.npz          current phase calibration (see §7)
├── scripts/                 analysis & ground-truth tools
│   ├── fit_array_cal.py     TLE self-calibration (satellites as beacons)
│   ├── eval_doa_accuracy.py per-burst az/el error vs SGP4
│   ├── plot_track_vs_tle.py measured trajectory vs real satellite track
│   ├── iridium_groundtruth.py  pass prediction + track-level matching
│   ├── fetch_session_tle.py epoch-matched TLEs from Space-Track (see §4.4)
│   ├── predict_passes.py    quick pass forecast (when to measure)
│   ├── minimal_phase_check.py  live inter-channel phase sanity check
│   ├── diag_raw_iq.py       live capture diagnostic (power/tone/coherence)
│   └── legacy/              tools for the old single-npz recording format
├── hardware/                kraken_iq_source (TCP), file_iq_source
└── tests/                   pytest suite (156 tests, synthetic signals)

krakenSDR/data/doa_iridium/  recorded sessions (gitignored)
shared/iridium_tle.py        TLE download + SGP4 (passes, Doppler, az/el)
```

Deeper theory (Heimdall protocol internals, MUSIC math, array geometry) is in
`krakenSDR/README.md` and `krakenSDR/src/README.md` (Italian).

---

## 3. Setup

```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib, skyfield, pytest
```

The live pipeline needs the **Heimdall DAQ** running on the KrakenSDR host
(IQ server on port 5000, control on 5001) — see `external/gr-krakensdr`.
Offline work needs nothing but a recorded session directory.

All knobs live in **`apps/doa_iridium/doa_config.toml`**; every CLI flag
overrides it. The three you will actually touch:

```toml
[hardware]
freq_mhz = 1626.27      # Iridium ring-alert channel (most active)
gain_db  = 40.0         # 40 outdoor; do not exceed ~49 (ADC saturation)

[array]
ant0_offset_deg = -4.0  # array rotation vs North — from fit_array_cal.py
use_phase_cal   = true  # NEVER turn this off for real measurements
cal_file        = "cal_tle.npz"   # relative to the config file's directory
```

The observer location (Biot/Sophia Antipolis) is hard-coded as the default in
`scripts/iridium_groundtruth.py` — pass `--lat/--lon/--alt` everywhere if you
measure somewhere else.

---

## 4. Workflows

All commands below are run from `krakenSDR/src/`.

### 4.1 Live DOA

**Use case: point at the sky, see bearings now.**

```bash
python3 apps/doa_iridium/run_doa.py                 # headless, JSON to stdout
python3 apps/doa_iridium/run_doa.py --gui           # live matplotlib UI
python3 apps/doa_iridium/run_doa.py --host 10.0.0.5 --gain 40
python3 apps/doa_iridium/run_doa.py --out /tmp/doa.jsonl   # also log to file
```

One JSON line per burst: `t, az_deg, el_deg, papr_db, snr_db, cfo_hz, …`.
The live loop applies EMA smoothing across bursts and resets it when the CFO
jumps > 3 kHz (= a different satellite became dominant).

### 4.2 Record a session

**Use case: capture raw IQ in the field, analyze at home. This is the
recommended way to work** — every result becomes reproducible.

```bash
python3 apps/doa_iridium/run_doa.py --record ../data/doa_iridium/
```

Creates `session_YYYYMMDD_HHMMSS/` with one `raw/frame_NNNNNN.npy` per CPI
plus `raw/timestamps.csv`, `meta.json`, `state.json`. Recording survives
Ctrl-C; the `raw/` directory is canonical. **Mind the disk**: 5 channels of
complex64 ≈ 2.6 MB per CPI ≈ **100 GB per hour** (the 36-min reference
session is 58 GB). Aim for **≥ 30 minutes** so the session contains several
satellite passes (needed for calibration and ground-truth statistics).

Optional afterwards: `python3 apps/doa_iridium/consolidate_session.py
<session_dir>` packs `raw/` into a single `raw_iq.npz` (slow; only useful for
archiving or moving the data).

### 4.3 Offline reprocessing

**Use case: turn a recorded session into DOA estimates + satellite tracks.
This is the workhorse command:**

```bash
python3 apps/doa_iridium/reprocess_session.py ../data/doa_iridium/session_20260605_110716 \
        --algo music --el-min 5 --cov-bursts 16
```

Writes `<session>/doa_multi_music/` containing `doa_multi.jsonl` (one line
per CPI with all peaks + per-peak CFO/SNR/MF vector), `burst_*.npz`,
`waterfall.npz`, and `tracks.json` (peaks clustered into per-satellite
tracks by az/el/CFO continuity, with Doppler-trend gating and merging of
sequential pass fragments).

Useful options:

| Flag | Use case |
|---|---|
| `--cov-bursts 16` | **recommended** — average the covariance over a 16-burst window per track before the final DOA (validated: el MAD 5.5° → 4.5°, see §4.5); 0 = single-burst rank-1 |
| `--cov-span-s 20` | max window time span (steering smear limit) |
| `--max-frames 2000 --frame-start 5000` | quick look at a segment |
| `--k-peaks 3` | max simultaneous satellites per burst (default 3) |
| `--el-min 5` | accept low-elevation peaks (early/late pass tracking) |
| `--recluster-only` | re-run track clustering (and `--cov-bursts`, if given) without re-reading IQ |
| `--out-subdir NAME` | parallel experiments without overwriting |
| `--phase-cal --cal-file PATH` | override the calibration |

Compare all three algorithms in one go (mostly didactic — see §1 fact 1):

```bash
python3 apps/doa_iridium/batch_reprocess.py <session_dir> --algos music capon bartlett
```

`run_doa_offline.py` is the single-peak variant that also reads plain IQ
files (`.npz`, `.wav`, `.cf32`, `.iq`) and has `--replay` / `--compare`
plotting modes; for sessions, prefer `reprocess_session.py`.

### 4.4 Ground truth & accuracy evaluation

**Use case: "how good are my estimates, really?"** Satellites are beacons
with exactly known positions — use them.

Step 1 — predict passes and match the clustered tracks (track-level view):

```bash
python3 scripts/iridium_groundtruth.py <session_dir>            # predict + save groundtruth.json
cp <session_dir>/doa_multi_music/tracks.json <session_dir>/tracks.json
python3 scripts/iridium_groundtruth.py <session_dir> --match    # → groundtruth_matches.json
```

Step 2 — per-burst error statistics (the honest metric; use this to compare
pipeline/parameter changes):

```bash
python3 scripts/eval_doa_accuracy.py <session_dir> --subdir doa_multi_music
```

Each DOA peak is assigned to a satellite only if its Doppler matches exactly
one predicted track within ±2 kHz (after automatic LO-offset removal), then
az/el errors vs the SGP4 position at burst time are reported: signed bias,
MedAE (median |error| — the robust headline), MAE, MAD and RMSE
(RMSE ≫ MedAE flags heavy outlier tails), plus per-satellite counts. Use
`--lo-offset` to pin the LO estimate when comparing runs of the same
session.

**TLE freshness — read this once.** A TLE is a set of orbital elements at
an *epoch*, and SGP4 error grows ~1–3 km/day away from it (mostly
along-track → a Doppler timing error that silently weakens the matching).
CelesTrak only serves the latest elements, so re-running an evaluation
weeks later degrades the ground truth. Two safeguards are built in:

1. **Per-session snapshot** — the first ground-truth run saves the elements
   to `<session>/iridium_tle.txt` and every script uses that file from then
   on (results frozen and reproducible).
2. **Epoch-matched elements** — for the best ground truth, fill the
   snapshot with elements whose epoch matches the session date
   (free Space-Track account required):

```bash
# credentials: copy scripts/spacetrack.json.example → ~/.config/lark/spacetrack.json
# (chmod 600) and fill in, or export SPACETRACK_USER / SPACETRACK_PASS
python3 scripts/fetch_session_tle.py <session_dir>
# → <session>/iridium_tle.txt with the closest-epoch element set per satellite
#   (reference session: median epoch distance 0.08 d vs ~6 d from CelesTrak)

# then re-run the evaluations/plots — they pick up the snapshot automatically:
python3 scripts/eval_doa_accuracy.py <session_dir> --subdir doa_multi_music
python3 scripts/plot_track_vs_tle.py <session_dir>
```

### 4.5 Track smoothing (Kalman/RTS)

**Use case: trajectory-level estimates.** Per-burst errors are nearly white
while a satellite pass is smooth (peak angular rate ≈ 0.6°/s), so smoothing
along each clustered track removes most of the random error. A 6-state
Kalman filter on the direction *unit vector* (no azimuth-wrap or zenith
issues) plus a Rauch–Tung–Striebel backward pass — every point conditioned
on the whole track. Real DOA errors are heavy-tailed, so the filter runs
robustly by default: a χ² innovation gate plus a reject-and-resmooth pass
(points > 4×MAD from the first solution are excluded and flagged).

```bash
python3 scripts/smooth_tracks.py <session_dir> --subdir doa_multi_music
# → <subdir>/tracks_smoothed.json + raw-vs-smoothed accuracy stats vs TLE
```

On the reference session this roughly halves the per-point spread on the
long tracks (az MAD 3.7° → 2.1°, el MAD 5.2° → 4.5°; az RMS 5.7° → 3.9°),
rejecting ~8% of points as outliers. The remaining error is systematic
(manifold/calibration), which no amount of smoothing removes. Tune
`--sigma-acc` if passes look over- or under-smoothed (default 2e-5 rad/s²,
fitted on the reference session with the robust pass enabled);
`--no-robust` gives the plain least-squares behaviour. Weighting
measurements by SNR makes the stats slightly *worse* (the error is
manifold-dominated, not noise-dominated) — that is why there is no such
option. The smoothed trajectory is automatically overlaid by
`plot_track_vs_tle.py` (§4.7), with rejected points marked ×.

#### Beyond rank-1 — multi-burst covariance

A single burst gives a rank-1 covariance (MUSIC = Capon = Bartlett). But
consecutive bursts of the same track carry independent noise and, as the
satellite moves, decorrelating ground multipath — so averaging normalised
MF vectors over a sliding window of B bursts makes the noise subspace
estimable. The JSONL stores the per-peak MF vector (`y_per_peak`, sessions
reprocessed after June 2026), so this is a pure post-processing step.

**The normal interface is the reprocess flag** (`--cov-bursts 16`, §4.3),
which re-estimates the JSONL peaks in place after clustering. The research
tool behind it also sweeps window sizes and writes derived result sets:

```bash
# Sweep window sizes, report accuracy vs TLE per window:
python3 scripts/multiburst_doa.py <session_dir> --windows 1 2 4 8 16

# Write a derived result set; every existing tool works on it unchanged:
python3 scripts/multiburst_doa.py <session_dir> --window 16 --max-span-s 20 --write
python3 scripts/smooth_tracks.py  <session_dir> --subdir doa_multi_music_covB16
```

Measured on the reference session: per-burst el MAD 5.5° → 4.5° and el bias
−4.6° → −4.1° at B=16 (azimuth ~flat; B=32 adds little) — consistent with
the multipath-decorrelation picture. After Kalman smoothing the two
pipelines converge to similar numbers: the covariance window and the
smoother average over the same time axis, and the systematic manifold error
is the common floor. The live pipeline does the equivalent with a
per-satellite covariance EMA (`cov_alpha`, effective window ≈ 14 bursts).

### 4.6 Array calibration

**Use case: new deployment, or accuracy suddenly degraded.** See §7 for the
policy. The trick: Iridium satellites themselves are the calibration source —
no reference transmitter needed, and the calibration is at the operating
frequency by construction.

```bash
# 1. Record ≥ 30 min outdoors (several passes — MUST contain ≥ 2 satellites)
# 2. Fit per-channel phase offsets + array rotation against SGP4 directions:
python3 scripts/fit_array_cal.py <session_dir>
# → <session_dir>/cal_tle.npz  + a printed TOML snippet

# 3. Apply: copy cal_tle.npz next to doa_config.toml and set in [array]:
#    ant0_offset_deg = <fitted value>, use_phase_cal = true, cal_file = "cal_tle.npz"

# 4. Verify: reprocess + evaluate (expect az/el MAD of a few degrees)
python3 apps/doa_iridium/reprocess_session.py <session_dir> --algo music --el-min 5
python3 scripts/eval_doa_accuracy.py <session_dir>
```

`fit_array_cal.py` reports a residual spread per channel: ±15–25° is normal
with linear antennas on a circularly-polarized signal; if a single channel is
much worse than the others, suspect its cable/connector. It warns if all
bursts come from one satellite (rotation and offsets become degenerate —
record longer).

### 4.7 Visualization

**Measured trajectory vs the real satellite track** (the thesis figure):

```bash
python3 scripts/plot_track_vs_tle.py <session_dir>                       # all satellites
python3 scripts/plot_track_vs_tle.py <session_dir> --sat "IRIDIUM 125" --show
python3 scripts/plot_track_vs_tle.py <session_dir> --subdir doa_multi_capon
```

Produces, per satellite pass: a polar sky plot (TLE trajectory as a
time-colored line, DOA peaks as same-colormap scatter) plus az(t), el(t) and
Doppler(t) panels with residual statistics. PNGs land in `<session>/plots/`.
If `tracks_smoothed.json` exists (§4.5) the Kalman/RTS trajectory is drawn
as a solid teal line with its own residual stats (`--no-smooth` to disable).

Other plotting modes (via the offline pipeline):

```bash
python3 apps/doa_iridium/run_doa_offline.py <session_dir> --replay    # replay results UI
python3 apps/doa_iridium/run_doa_offline.py <session_dir> --compare   # MUSIC/Capon/Bartlett overlay
```

### 4.8 Utilities & diagnostics

```bash
# When should I go outside? Pass forecast for the next N hours:
python3 scripts/predict_passes.py --hours 12 --min-el 15

# Is the hardware sane? Live inter-channel phase + coherence check
# (connects directly to Heimdall, compares against the active calibration):
python3 scripts/minimal_phase_check.py

# Quick live capture: power, tone at +3125 Hz, channel coherence:
python3 scripts/diag_raw_iq.py

# Would tilting the array improve low-elevation accuracy? (synthetic Monte
# Carlo; spoiler: no — median gain at el≈5° only, ambiguity outliers everywhere)
python3 scripts/tilt_study.py
```

`scripts/legacy/` holds diagnostics for the pre-June-2026 single-file
recording format — not for session directories.

### 4.9 Tests

```bash
python3 -m pytest tests/ -q        # 156 tests, all synthetic, ~17 s
```

Every `core/` DSP stage has unit tests with known-truth synthetic signals
(burst detection, tone scan incl. sub-bin interpolation and multichannel
combining, BPF, MF covariance, full-pipeline DOA, multi-burst per CPI,
recording). Run them after any change to `core/`.

---

## 5. Data formats

**Session directory** (produced by `--record`, consumed by everything else):

```
session_YYYYMMDD_HHMMSS/
├── meta.json            freq_hz, fs_hz, gain_db, n_ant, cpi_size, created (UTC)
├── state.json           frame counts, elapsed_s, status
├── raw/
│   ├── frame_000000.npy   one CPI: complex64 (n_ant, cpi_size)
│   ├── …
│   └── timestamps.csv     "frame_index,unix_epoch" — the wall clock
├── doa_multi_<algo>/    reprocess output (jsonl + npz + tracks.json)
├── groundtruth.json     predicted passes (iridium_groundtruth.py)
├── groundtruth_matches.json   track ↔ satellite matches + lo_offset_hz
├── cal_tle.npz          calibration fitted from this session (if run)
└── plots/               PNGs from plot_track_vs_tle.py
```

**Frame timing — important.** The recorder drops CPIs when the live pipeline
can't keep up, so frame index × CPI duration is *not* wall time. Always get
per-frame epochs via `core.recording.session_frame_times(session_dir)`
(reads `timestamps.csv`, falls back to the `.npy` mtimes for old sessions).
All shipped tools already do this. Also note: `meta.json["created"]` of
sessions recorded before 2026-06-10 is **local time**, not UTC — the frame
timestamps are the reliable clock there too.

**`doa_multi.jsonl` row** (one per CPI with detections):

```json
{"burst_idx": 12, "frame": 480, "t": 43.3,
 "tone_hz": 13125.0, "cfo_hz": 10000.0, "snr_db": 14.2, "papr_db_global": 6.1,
 "peaks": [[az, el, power_db, papr_db], ...],
 "cfo_per_peak": [10000.0, -22500.0],     // per-satellite Doppler
 "snr_per_peak": [14.2, 9.8],
 "y_per_peak": [[[re, im], ... x n_ant]], // MF array-response vector
 "track_ids": []}
```

`t` is seconds since session start (wall clock). `cfo_per_peak` is the
per-satellite discriminator — use it, not the scalar `cfo_hz`, when peaks
from different satellites share a CPI. `y_per_peak` (added 2026-06-11) is
the calibrated matched-filter output the rank-1 covariance is built from;
`scripts/multiburst_doa.py` uses it to average covariances across bursts.

**`cal_tle.npz`**: `phase_offsets_deg (n_ant,)` (ch0 = 0 reference),
`ant0_offset_deg`, `circ_std_deg` (per-channel residual spread), `n_bursts`,
`n_sats`.

---

## 6. Field guide — outdoor measurements

What actually limits accuracy in the field, in order of impact:

**Phase calibration** — see §7. Without it the azimuth can be rotated by
tens of degrees and elevation is meaningless. With it, expect az MAD ≈ 3–4°.

**Site selection.** Open sky in all directions; ≥ 10 m from metal surfaces,
walls and vehicles (multipath biases both az and el); mast or tripod, never
directly on the ground or a car roof.

**Leveling.** Array tilt translates *directly* into elevation bias (1° tilt
≈ 1° el error in the tilt direction). Use a bubble level / inclinometer on
the array plane, every time.

**North reference.** A phone/handheld compass is good to maybe ±5–10° (worse
near metal). Better options: solar azimuth at a known time, an RTK/GNSS
baseline, or simply **let the satellites tell you** — `fit_array_cal.py`
fits `ant0_offset_deg` from TLE directions, turning surveying error into a
fitted parameter. Then physically note the array orientation so you can
reproduce it.

**Polarization.** Iridium downlink is **RHCP**; the current monopoles are
linear → ~3 dB loss and direction-dependent phase (this is the main source
of the ±15–25° calibration residual spread). If you upgrade: 5 *identical*
RHCP patches (GPS/Iridium L-band type) on a common circular ground plane.
With linear elements, at least orient all five identically.

**Gain and ADC.** 40 dB is the validated outdoor setting; ~49 dB saturates
the ADC. If strong out-of-band signals are nearby, prefer lower gain — a
saturated frame corrupts the phase on all channels at once.

**Thermal stability.** Channel phase drifts ≈ 1°/°C. Keep the Kraken shaded
and ventilated; let it warm up ~10 min before relying on the data; avoid
sessions spanning a large temperature swing without re-checking calibration.

**Timing.** Make sure the recording laptop runs NTP (`timedatectl`): the
TLE matching needs ~1 s absolute accuracy. The recorder stamps every frame.

**Session planning.** Check `predict_passes.py` first; a useful session has
≥ 2–3 passes (~30–45 min at this latitude there is almost always something
up). For calibration sessions specifically, more sky diversity = better fit.

**Pre-flight checklist:**

```
□ NTP synchronized                  □ array leveled (bubble)
□ antenna orientation noted         □ Heimdall running, 5 channels locked
□ minimal_phase_check.py sane       □ gain 40 dB, freq 1626.27 MHz
□ predict_passes.py shows passes    □ free disk ≥ 100 GB per hour planned
□ run_doa.py --record started       □ note site, weather, temperature
```

---

## 7. Calibration policy — when and how

Two distinct things are calibrated, both by `fit_array_cal.py`:

1. **Per-channel phase offsets** (`cal_tle.npz`) — cable length differences,
   receiver chains, antenna placement. These are *electrical* and frequency
   dependent: a calibration done at 868 MHz is useless at 1626 MHz.
2. **Array rotation** (`ant0_offset_deg`) — *mechanical* orientation vs
   geographic North.

**Recalibrate (full procedure of §4.6) whenever:**

- antennas, cables or connectors are touched, swapped or re-routed;
- the array is moved to a new site or re-assembled;
- `eval_doa_accuracy.py` on a fresh session shows azimuth bias or MAD
  drifting (e.g. MAD > ~6–8° when it used to be ~4°);
- operating temperature differs strongly from the calibration session.

**Re-fit only the rotation** (mechanical re-orientation, same cabling): run
`fit_array_cal.py` and take just `ant0_offset_deg`, or equivalently estimate
the constant azimuth bias from `eval_doa_accuracy.py` and subtract it.

**Sanity numbers** from the reference calibration (2026-06-05): offsets
ch1..4 = +149.4°, −36.9°, −45.8°, −69.0°; rotation −4.0°; residual spread
±17–26°/channel from 1699 bursts and 9 satellites. The LO frequency offset
(≈ +2 ppm) is *not* part of the calibration file — it is re-estimated
automatically from Doppler residuals by every analysis tool.

A cheap health check between calibrations: `minimal_phase_check.py` during a
strong pass — measured inter-channel phases should sit near the calibration
values it prints.

---

## 8. Troubleshooting

| Symptom | Likely cause → what to do |
|---|---|
| No bursts detected | Wrong frequency/gain, antenna issue. `diag_raw_iq.py`: is there power and a tone? Check passes are actually overhead (`predict_passes.py`). |
| Bursts but low PAPR (< 3 dB), peaks rejected | Calibration off or disabled — check `use_phase_cal`, re-run §4.6 verification. Indoors this is expected (multipath). |
| Azimuth tracks the satellite but with a constant offset | `ant0_offset_deg` wrong (array rotated). Re-fit rotation (§7). |
| Elevation systematically high/low | Array tilt (re-level) or stale phase calibration. |
| Estimates jump between two directions | Two satellites alternating in the same CPI — normal; look at `cfo_per_peak`, the track clusterer separates them downstream. |
| `eval_doa_accuracy` assigns almost nothing | Session timing broken (pre-fix session without `timestamps.csv`? frame mtimes destroyed by copying?) or TLE stale — `iridium_groundtruth.py --force-download`. |
| Many unmatched tracks at low elevation | Multipath ghosts — raise `--el-min` to 10 for clean statistics. |
| Live pipeline laggy / frames dropped | Normal on small CPUs; recorded timestamps keep offline analysis correct. Reduce GUI refresh, or record headless. |

When something looks wrong inside the DSP, run with `--debug-dir /tmp/dbg`:
every stage dumps its intermediate arrays (`00_raw_iq.npy` … `07_doa_result`)
for one-frame inspection.

---

## 9. Known limitations

- **No single-burst super-resolution**: the rank-1 MF covariance makes all
  spectral estimators equivalent to beamforming on the matched-filter output.
  Multi-burst accumulation (EMA live, track clustering offline) is what
  builds confidence.
- **Elevation is the weak axis** (MAD ≈ 5.5° vs 3.8° az), degrading further
  below ~10° elevation (multipath, geometry). The UCA also has an inherent
  el-ambiguity softness near the zenith.
- **Linear antennas on an RHCP signal**: ~3 dB loss + direction-dependent
  phase residuals. RHCP patches are the highest-impact hardware upgrade.
- **Calibration is deployment-specific** and drifts with temperature
  (≈ 1°/°C); treat `cal_tle.npz` as a perishable.
- Tracking smoothing is a simple EMA; `core.tracking.KalmanAngular` exists
  and would give principled SNR-weighted smoothing if needed.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **CPI** | Coherent Processing Interval — one block of multi-channel IQ (65536 or 131072 samples at 1.024 MS/s) |
| **IRA** | Iridium Ring Alert — broadcast burst type with the CW preamble used here |
| **CFO** | Carrier Frequency Offset = measured tone − 3125 Hz ≈ satellite Doppler + receiver LO offset |
| **LO offset** | Receiver TCXO error; shifts *all* CFOs equally (1 ppm = 1626 Hz); auto-estimated as median Doppler residual |
| **MF covariance** | Rank-1 matrix y·yᴴ from projecting each channel onto the preamble tone |
| **PAPR** | Peak-to-average ratio of the DOA spectrum [dB]; how well the data matches the steering model |
| **UCA** | Uniform Circular Array; here 5 elements, radius 0.4253 λ |
| **TLE / SGP4** | Two-Line Elements + propagator: predicted satellite az/el/Doppler (ground truth) |
| **Track** | Cluster of consecutive burst peaks consistent in az/el/CFO ≈ one satellite pass |
| **MAD** | Median Absolute Deviation — robust spread of the error distribution |
