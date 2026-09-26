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

**The DOA is the method; the application is the inverse problem.** Given where the
satellites are, the same bursts fix where the *receiver* is — GNSS-independent PNT. The
satellites broadcast what that needs: **IRA** frames carry the transmitter's own ECEF
position, **IBC** frames carry Iridium system time. Decoding them (via `gr-iridium` +
`iridium-toolkit`) makes the fix self-contained: **0.15 km, blind, no TLE, no network, no
host clock** — read it as a few hundred metres: bootstrap 0.12–0.63 km, 0.61 km at
the true altitude, 0.66 km with SGP4 in place of the broadcast ephemeris. Use one
receiver offset for all satellites (the default); `--per-sat-df` gave 0.47 km on
the corrected epochs. Doppler carries the
information (~14 Hz per km of observer displacement, ~35 Hz MAD on clean bursts); angles
supply the initial guess, the side of the ground track, and **+31% decoded frames** from
steering the array. See `docs/decisions/004-broadcast-ephemeris-pnt.md`.

**Timing conventions (easy to break, costly when broken).** A burst's epoch is its CPI's
epoch plus its start inside the CPI (`b0_per_peak` / `remap_epochs()`); sessions without
`raw/timestamps.csv` get CPI epochs rebuilt on the sample grid by `session_frame_times()`.
Getting either wrong moved the reference fix from 0.15 km to 0.97 km, all along track.

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
| Ephemeris + time from decoded frames | `krakenSDR/src/core/broadcast_ephemeris.py` |
| Observer position solver | `krakenSDR/src/core/pnt_solver.py` |
| IQ export for the demodulator | `krakenSDR/src/scripts/export_iq.py` |
| PNT fix (CLI) | `krakenSDR/src/scripts/pnt_solve.py` |
| PNT interactive replay | `krakenSDR/src/apps/doa_iridium/pnt_replay.py` |

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

# PNT: decode the payload, then solve for the receiver's own position
python3 scripts/export_iq.py ../data/doa_iridium/session_.../ --ant 0 --out /tmp/cpi
# ... iridium-extractor + iridium-parser.py -o line -> /tmp/frames.parsed (see README)
python3 scripts/decode_ephemeris.py ../data/doa_iridium/session_.../ \
  --parsed /tmp/frames.parsed --index /tmp/cpi/index.json --validate-vs-tle
python3 scripts/pnt_solve.py ../data/doa_iridium/session_.../ \
  --ephemeris ira --mode doppler --per-satellite
# interactive: timeline = listening time, re-solves at each step
python3 scripts/pnt_solve.py ../data/doa_iridium/session_.../ \
  --ephemeris ira --mode doppler --gui

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
| `broadcast_ephemeris.py` | IRA/IBC parsing, short-arc fit, clock offset |
| `pnt_solver.py` | WGS-84 geometry, Doppler model, observer solve |

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
- `pnt_replay.py` reuses the same theme constants and `_style_button` from
  `offline_viz` / `offline_replay` — extend those rather than defining new colours.
  Its timeline is *listening time*, and every step is a real re-solve, so the fixes
  are precomputed once at startup (`--gui-steps`) to keep scrubbing instant.

### PNT chain — traps that cost real debugging time

- **`iridium-parser.py` needs `-o line`.** Without it nothing prints and it looks like a
  decode failure.
- **`gr-iridium` must be built against the *system* pybind11 2.11.1**, not the pip 3.x:
  GNU Radio's ABI is `__pybind11_internals_v5` and the mismatch surfaces as
  `unknown base type "gr::block"`. Build to a local prefix; never modify `external/`.
- **gr-iridium rejects sample rates that are not a multiple of 100 kHz.** `export_iq.py`
  resamples 1.024 → 1.0 MS/s; 125/128 is exact and maps a 65536-sample CPI to exactly
  64000, so sample-offset timing stays integer. Resample per CPI, never per chunk.
- **Exported chunks have a compressed time axis.** Dropped CPIs are concatenated, so a
  chunk's internal clock runs ~23% slow. Always map back through `index.json`.
- **About half of BCH-clean IRA position fields are corrupt** — filter on orbital radius
  (`filter_plausible`). And never finite-difference IRA positions for velocity: 4 km
  quantisation over 90 ms implies 44 km/s. Fit a short arc.
- **Do not use DOA tracks for PNT.** The clusterer mixes satellites; associate per burst
  by Doppler instead. Use rank-0 peaks only — rank-1 and rank-2 are ~80% ghosts.

## Where to read next

| Topic | Document |
|-------|----------|
| Project overview (humans) | `README.md` |
| Repo structure & layers | `docs/architecture.md` |
| Signal path & algorithms | `docs/doa-pipeline.md` |
| Why multi-peak per tone | `docs/decisions/001-multi-peak-per-tone.md` |
| Why elevation gate | `docs/decisions/002-elevation-gate.md` |
| Why decode the payload; PNT error budget | `docs/decisions/004-broadcast-ephemeris-pnt.md` |
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
├── groundtruth.json          # optional, from scripts/
├── broadcast_ephemeris.json  # optional, from scripts/decode_ephemeris.py
└── pnt_solution.json         # optional, from scripts/pnt_solve.py
```

Do not store tuning experiments or chat state in permanent repo docs — use issues,
PR descriptions, or local notes.
