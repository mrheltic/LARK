# LARK architecture

High-level structure of the repository and how components interact.

## Design goals

1. **Separate DSP from I/O** — `core/` is hardware-independent and pytest-tested.
2. **Thin application layer** — `apps/doa_iridium/` wires config, CLI, recording, viz.
3. **Crash-safe recording** — incremental `raw/frame_*.npy` + optional consolidation.
4. **Offline parity** — same CPI processing live and from disk (`multi_peak.py`).

## Layer diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Hardware / files                                               │
│  KrakenIQSource (TCP) · raw/frame_*.npy · legacy npz/wav        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  apps/doa_iridium/                                              │
│  run_doa.py          live loop + optional SessionRecorder       │
│  run_doa_offline.py  replay, compare, plot-only                 │
│  reprocess_session.py  offline multi-peak → doa_multi_*/        │
│  batch_reprocess.py  MUSIC + Capon + Bartlett batch             │
│  offline_replay.py / algo_compare.py / offline_viz.py           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  core/                                                          │
│  burst_processing   detect · tone scan · BPF · MF covariance    │
│  multi_peak         per-CPI multi-satellite DOA                 │
│  doa_uca_2d         UCA steering · MUSIC · Capon · Bartlett   │
│  doa_algorithms     phase correction                          │
│  track_clusterer    peak → satellite track association          │
│  recording          SessionRecorder · frame iterators           │
│  pipeline_debug     optional stage dumps                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  scripts/ · shared/                                             │
│  fit_array_cal · eval_doa_accuracy · iridium_groundtruth         │
│  shared/iridium_tle.py — SGP4 pass prediction                   │
└─────────────────────────────────────────────────────────────────┘
```

## Module responsibilities

### `core/burst_processing.py`

Stateless signal-processing primitives:

- `detect_energy_bursts` — median noise floor, block power threshold
- `scan_preamble_tones` — FFT peak search around 3125 Hz ± Doppler band
- `apply_bpf_and_normalize` — Hann-tapered spectral mask, unit RMS per channel
- `compute_mf_covariance` — rank-1 MF covariance + SINR estimate

### `core/multi_peak.py`

Orchestrates one CPI:

1. Detect burst window in channel 0.
2. Find up to `k_peaks` tones (one per visible satellite).
3. For **each tone**: BPF → phase cal → MF cov → 2D spectrum → peak + gates.
4. Returns `peaks` array `(K, 4)` = az, el, power_db, papr_db.

See ADR [001-multi-peak-per-tone](./decisions/001-multi-peak-per-tone.md).

### `core/doa_uca_2d.py`

- `UcaConfig` — geometry, grid size, `ant0_offset_deg`, steering matrix cache
- `doa_music_uca_2d` / `doa_capon_uca_2d` / `doa_bartlett_uca_2d`
- `find_peak_uca_2d` — argmax + parabolic sub-grid refinement + PAPR

### `core/track_clusterer.py`

Greedy online association of peaks across time into tracks using az, el, CFO, and
gap timeout. Short tracks become outlier id `-1`.

### `core/recording.py`

- `SessionRecorder` — live write of `raw/` and `doa/` estimates
- `iter_session_raw_frames` — memory-safe offline iteration
- `consolidate_session` — optional `raw_iq.npz` builder

## Configuration

Single source of truth: `apps/doa_iridium/doa_config.toml`

Profiles in code (`indoor` / `outdoor`) override tone scan bandwidth and energy
threshold. CLI flags override TOML for experiments.

Critical array parameters:

| Parameter | Typical value | Meaning |
|-----------|---------------|---------|
| `radius_lambda` | 0.4253 | UCA radius; chord spacing = λ/2 for N=5 |
| `ant0_offset_deg` | field-specific | Mechanical North alignment |
| `use_phase_cal` | true | Apply `cal_tle.npz` phase offsets |
| `n_az` / `n_el` | 360 / 86 | DOA search grid |

## Data artifacts

| Artifact | Producer | Consumer |
|----------|----------|----------|
| `raw/frame_*.npy` | live record | reprocess_session |
| `doa/est_*.npz` | live record | legacy plot path |
| `doa_multi.jsonl` | reprocess | clusterer, replay |
| `burst_*.npz` | reprocess | offline_replay (spectra) |
| `tracks.json` | clusterer | replay track colors |
| `groundtruth.json` | scripts | accuracy eval |

Session directories under `krakenSDR/data/doa_iridium/` are **gitignored**.

## Test strategy

- Unit tests in `krakenSDR/src/tests/` use synthetic IQ (no hardware).
- Burst stages: `test_burst_*.py`, `test_multi_peak.py`
- DOA geometry: `test_doa.py`
- Run from `krakenSDR/src/`: `pytest tests/ -q`

## Legacy code

- `apps/legacy/` — older 868 MHz and monolithic DOA experiments; do not extend.
- `core/gates`, `tone_extraction`, `tracking` — legacy-only per docstrings.

## External dependencies

- **Heimdall DAQ** — IQ stream for live capture (`external/gr-krakensdr`)
- **skyfield** — TLE / SGP4 for ground truth scripts
- Submodules under `external/` — treat as vendored; avoid drive-by edits

## Related docs

- [doa-pipeline.md](./doa-pipeline.md) — signal-processing detail
- [apps/doa_iridium/README.md](../krakenSDR/src/apps/doa_iridium/README.md) — operations
