# LARK — Agent Guide

Model-agnostic entry point for AI coding agents working in this repository.
Keep this file short; deep detail lives in `docs/` and the operations handbook.

## What this project is

**LARK** is a KrakenSDR-based Direction-of-Arrival (DOA) system for **Iridium NEXT**
satellite bursts at **~1626 MHz**. A 5-element uniform circular array (UCA) estimates
azimuth and elevation from the IRA preamble CW tone. Pure Python — no GNU Radio in the
main pipeline.

Validated outdoor accuracy (session 2026-06-05): azimuth MAD **~3.8°**, elevation MAD
**~5.5°** vs SGP4 ground truth (with phase calibration enabled).

## Repository map

```
LARK/
├── AGENTS.md                          ← you are here
├── docs/                              ← architecture, pipeline, ADRs
├── krakenSDR/src/
│   ├── core/                          ← reusable DSP (unit-tested)
│   ├── apps/doa_iridium/              ← live + offline application
│   ├── scripts/                       ← calibration, ground truth, analysis
│   ├── hardware/                      ← Kraken TCP IQ source
│   └── tests/                         ← pytest
├── krakenSDR/data/doa_iridium/        ← recorded sessions (gitignored)
├── shared/                            ← TLE / SGP4 helpers
└── external/                          ← submodules (Heimdall, gr-krakensdr, …)
```

Run Python commands from **`krakenSDR/src/`** (that directory is on `sys.path`).

## Key files

| Purpose | Path |
|---------|------|
| Operations handbook (commands, workflows) | `krakenSDR/src/apps/doa_iridium/README.md` |
| Live pipeline | `krakenSDR/src/apps/doa_iridium/run_doa.py` |
| Offline / replay / compare | `krakenSDR/src/apps/doa_iridium/run_doa_offline.py` |
| Multi-peak reprocess | `krakenSDR/src/apps/doa_iridium/reprocess_session.py` |
| Batch MUSIC/Capon/Bartlett | `krakenSDR/src/apps/doa_iridium/batch_reprocess.py` |
| Config (defaults are calibrated) | `krakenSDR/src/apps/doa_iridium/doa_config.toml` |
| UCA steering + DOA algorithms | `krakenSDR/src/core/doa_uca_2d.py` |
| Burst DSP stages | `krakenSDR/src/core/burst_processing.py` |
| Multi-sat CPI processing | `krakenSDR/src/core/multi_peak.py` |
| Track clustering | `krakenSDR/src/core/track_clusterer.py` |

## Physical conventions (do not invert)

- **Azimuth**: degrees clockwise from geographic **North** (0°=N, 90°=E).
- **Elevation**: degrees above the horizon (0°=horizon, 90°=zenith).
- **UCA**: 5 antennas; radius **0.4253λ** → adjacent chord spacing **λ/2**.
- **Antenna 0** at North when `ant0_offset_deg = 0`; set offset from field calibration.
- **CFO / Doppler**: `tone_hz − 3125 Hz` — primary multi-satellite discriminator.

Steering phase delay: `τ_k = 2π · r · cos(el) · cos(φ_k − az)`.

## Architecture (one paragraph)

Each CPI `(5, N)` at 1.024 MS/s → energy burst detect → FFT tone scan (up to K
satellites) → per-tone BPF + phase cal → matched-filter rank-1 covariance → 2D
MUSIC/Capon/Bartlett on UCA grid → quality gates (SNR, PAPR, optional `el_min`).
Offline reprocess writes `doa_multi_*/doa_multi.jsonl`, then clusters peaks into
satellite tracks. See **`docs/doa-pipeline.md`**.

## Common commands

From `krakenSDR/src/`:

```bash
# Live DOA + record session
python3 apps/doa_iridium/run_doa.py --gui --record ../data/doa_iridium

# Offline multi-peak reprocess (one algorithm)
python3 apps/doa_iridium/reprocess_session.py ../data/doa_iridium/session_.../

# Recluster only (fast parameter tuning)
python3 apps/doa_iridium/reprocess_session.py ../data/doa_iridium/session_.../ \
  --recluster-only --min-track-len 20

# All three algorithms + compare plot
python3 apps/doa_iridium/batch_reprocess.py ../data/doa_iridium/session_.../
python3 apps/doa_iridium/run_doa_offline.py ../data/doa_iridium/session_.../ --compare --gui

# Interactive replay
python3 apps/doa_iridium/run_doa_offline.py ../data/doa_iridium/session_.../ \
  --replay --no-tracks

# Tests
pytest tests/ -q
```

## Coding rules for agents

1. **Minimize scope** — smallest correct diff; do not refactor unrelated code.
2. **Match existing style** — read surrounding code before adding abstractions.
3. **Core vs app** — reusable DSP in `core/`; CLI and viz in `apps/doa_iridium/`.
4. **Do not edit `external/` submodules** unless explicitly asked.
5. **Do not commit** unless the user explicitly requests it.
6. **Do not commit secrets**, large IQ files, or generated session data.
7. **Phase calibration** — never disable for real measurement workflows in docs/examples.
8. **Tests** — run relevant pytest modules after substantive DSP changes.

## Where to read next

| Topic | Document |
|-------|----------|
| Repo structure & layers | `docs/architecture.md` |
| Signal path & algorithms | `docs/doa-pipeline.md` |
| Why multi-peak per tone | `docs/decisions/001-multi-peak-per-tone.md` |
| Why elevation gate | `docs/decisions/002-elevation-gate.md` |
| Commands & field procedures | `krakenSDR/src/apps/doa_iridium/README.md` |
| Cursor-specific rules | `.cursor/rules/*.mdc` |

## Session data layout

```
session_YYYYMMDD_HHMMSS/
├── meta.json
├── raw/frame_NNNNNN.npy
├── doa_multi_music/          # or doa_multi_capon, doa_multi_bartlett
│   ├── doa_multi.jsonl
│   ├── burst_NNNNNN.npz
│   ├── waterfall.npz
│   └── tracks.json
└── groundtruth.json          # optional, from scripts/
```

Do not store tuning experiments or chat state in permanent repo docs — use issues,
PR descriptions, or local notes.
