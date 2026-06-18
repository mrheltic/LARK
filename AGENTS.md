# LARK — Agent Guide

Model-agnostic entry point for AI coding agents working in this repository.
Keep this file short; deep detail lives in `docs/` and the operations handbook.

> **This is the one and only agent-instructions file — every agent reads it.**
> Claude Code, Cursor, Codex and other tools all load `AGENTS.md` from the repo root.
> `CLAUDE.md` is a **symlink** to this file (Claude Code's historical entry point), so
> there is a single set of bytes to read and write — no per-tool folders, no copies to
> keep in sync. Edit here and every agent sees it. The human-facing project tour is the
> repo-root [`README.md`](./README.md).

## What this project is

**LARK** is a KrakenSDR-based Direction-of-Arrival (DOA) system for **Iridium NEXT**
satellite bursts at **~1626 MHz**. A 5-element uniform circular array (UCA) estimates
azimuth and elevation from the IRA preamble CW tone. Pure Python — no GNU Radio in the
main pipeline.

Validated outdoor accuracy (reference session 2026-06-05, phase calibration on, vs SGP4
ground truth). Headline metric is **MedAE** (median absolute error):

| Axis | Single-burst | Multi-burst (B=16) |
|------|--------------|--------------------|
| Azimuth   | 4.1° | **3.7°** |
| Elevation | 3.9° | **3.3°** |

Elevation is the weaker axis (per-burst MAD ≈ 5.5° vs ≈ 3.8° az); multi-burst covariance
averaging (`--cov-bursts 16`) is what closes the gap.

## Repository map

```
LARK/
├── AGENTS.md                          ← you are here (all agent rules, single file)
├── CLAUDE.md → AGENTS.md              ← symlink (Claude Code entry point)
├── README.md                          ← human-facing project tour
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

## Layer-specific notes

### `core/` — DSP library (`krakenSDR/src/core/`)

Hardware-independent, unit-tested signal processing.

| Module | Responsibility |
|--------|----------------|
| `burst_processing.py` | detect, tone scan, BPF, MF covariance |
| `multi_peak.py` | full CPI multi-satellite processing |
| `doa_uca_2d.py` | UCA steering, MUSIC/Capon/Bartlett, peak find |
| `track_clusterer.py` | peak → track association |
| `recording.py` | session record + frame iteration |

- Pure functions where possible; explicit parameters over hidden globals.
- NumPy shapes: IQ `(n_ant, n_samples)`, spectrum `(n_el, n_az)`.
- **Do not** add GNU Radio dependencies to `core/`.
- **Do not** extend legacy modules (`gates`, `tone_extraction`, `tracking`) without an
  explicit request.
- After substantive changes: `pytest tests/test_burst*.py tests/test_multi_peak.py tests/test_doa.py -q`.

### `apps/doa_iridium/` — application layer

CLI entry points, config, offline viz/replay. Import DSP from `core/`; do not duplicate it.

- Defaults live in `doa_config.toml`; CLI flags override for experiments.
- `use_phase_cal = true` for any real-measurement workflow or doc example.
- Outdoor profile: wide Doppler scan (±45 kHz); indoor: narrow band.
- Reprocess writes `doa_multi_<algo>/` (`doa_multi.jsonl`, `tracks.json`); use
  `--recluster-only` to tune clustering without re-reading IQ.
- Viz flags: `--replay` (timeline), `--compare` (algorithm overlay), `--no-tracks`
  (flat peak coloring, disable track-filter hotkeys).

## Where to read next

| Topic | Document |
|-------|----------|
| Project overview (humans) | `README.md` |
| Repo structure & layers | `docs/architecture.md` |
| Signal path & algorithms | `docs/doa-pipeline.md` |
| Why multi-peak per tone | `docs/decisions/001-multi-peak-per-tone.md` |
| Why elevation gate | `docs/decisions/002-elevation-gate.md` |
| Commands & field procedures | `krakenSDR/src/apps/doa_iridium/README.md` |

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
