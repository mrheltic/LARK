# Iridium DOA — Outdoor Quick Start

Short guide for live measurements with the KrakenSDR + Heimdall DAQ.  
All commands assume you are in the repo root (`LARK/`) unless noted otherwise.

---

## What the pipeline does (7 stages)

Each Kraken CPI frame is `(5 antennas × 131072 samples)` at **1.024 MS/s**.

```
Kraken IQ (TCP)
    → 1. Burst detection      find IRA energy spikes in antenna 0
    → 2. Tone scan            locate the 3125 Hz preamble tone (+ Doppler)
    → 3. Band-pass filter     narrow to the burst, normalise amplitude
    → 4. Phase correction     optional (off by default)
    → 5. Covariance           matched-filter R matrix + EMA smoothing
    → 6. 2D DOA (MUSIC)       azimuth × elevation spectrum (spec2d)
    → 7. Peak + EMA           az_deg, el_deg printed as JSON lines
```

**Main output you care about:** the DOA spectrum — a 2D heatmap `(elevation × azimuth)` in dB, plus a 1D azimuth cut `(360,)` similar to Kraken’s `doa_music` block. Saved in `doa_music.npz` when recording.

Phase calibration is **disabled by default** (`use_phase_cal = false`). You get the spectrum shape without a separate cal workflow.

---

## Before you go outside

### 1. Start Heimdall DAQ

Kraken must be streaming IQ on `localhost:5000`.

```bash
LARK_ROOT=$PWD bash krakenSDR/src/scripts/start_heimdall.sh
```

Wait until you see **“Heimdall ready”** (or equivalent) in the terminal.

### 2. Check hardware settings (optional edit)

Open `krakenSDR/src/apps/doa_iridium_grc/doa_config.toml`:

| Setting | Typical outdoor value | Notes |
|---------|----------------------|-------|
| `freq_mhz` | `1626.27` | Iridium ring-alert uplink |
| `gain_db` | `30`–`38` | Raise if weak; lower if clipping |
| `mode` | `outdoor` | Wider Doppler search (±45 kHz) |
| `ant0_offset_deg` | compass bearing of ant 0 | Set once per deployment |

For a one-off outdoor run you can skip editing the file and pass flags instead (see below).

### 3. Go to the app directory

```bash
cd krakenSDR/src/apps/doa_iridium_grc
```

---

## Outdoor measurement — recommended commands

### A. Live with GUI + recording (best for field work)

Records raw Kraken IQ **and** DOA spectra. Press `Ctrl+C` **once** and wait for the save to finish.

```bash
python3 run_doa.py \
  --mode outdoor \
  --gain 34 \
  --gui \
  --record ../../../data/doa_iridium
```

- JSON estimates stream to the terminal (one line per valid burst).
- GUI panels: skyplot, az/el history, 2D MUSIC heatmap, eigenvalues, status.
- Session folder example:
  ```
  krakenSDR/data/doa_iridium/session_20260603_143000/
    meta.json
    state.json
    raw/
      frame_000000.npy   # written immediately, one CPI each (~5 MB)
      frame_000001.npy
    doa/
      est_000000.npz     # one DOA spectrum per valid burst
    raw_iq.npz           # optional — build offline with consolidate_session.py
    doa_music.npz
  ```

Each CPI is saved to `raw/` **as it arrives**. Shutdown is **fast** (seconds): it writes `doa_music.npz` and leaves IQ in `raw/`. To pack IQ into one file later (slow for long sessions):

```bash
python3 consolidate_session.py ../../../data/doa_iridium/session_20260603_143000/ --raw-only
```

Replay without consolidating:

```bash
python3 run_doa_offline.py ../../../data/doa_iridium/session_20260603_143000/
```

### B. Headless + recording (no display, e.g. over SSH)

```bash
python3 run_doa.py \
  --mode outdoor \
  --gain 34 \
  --record ../../../data/doa_iridium \
  --out /tmp/doa_live.jsonl
```

### C. Spectra only (smaller files, no raw IQ)

Useful if disk space is limited; you still get `doa_music.npz`.

```bash
python3 run_doa.py \
  --mode outdoor \
  --record ../../../data/doa_iridium \
  --no-record-raw
```

### D. Quick test without saving

```bash
python3 run_doa.py --mode outdoor --verbose
```

You should see JSON lines like:

```json
{"t": 1717420800.1, "n": 1, "az": 164.2, "el": 49.1, "snr_db": 5.3, "papr_db": 4.1, ...}
```

If nothing appears for a few minutes, try `--gain 36` or confirm Iridium bursts are visible on a spectrum/waterfall.

---

## After the session

### Inspect the latest DOA spectrum (Python)

```bash
python3 - <<'EOF'
import numpy as np, glob, os
sessions = sorted(glob.glob("../../../data/doa_iridium/session_*"))
d = np.load(os.path.join(sessions[-1], "doa_music.npz"))
print("estimates:", d["spec2d"].shape[0])
print("last az/el:", float(d["az_deg"][-1]), float(d["el_deg"][-1]))
print("doa_az shape (Kraken-style):", d["last_doa_az"].shape)
EOF
```

Run from `krakenSDR/src/apps/doa_iridium_grc/`.

### Re-process a recording offline

```bash
python3 run_doa_offline.py \
  ../../../data/doa_iridium/session_YYYYMMDD_HHMMSS/raw_iq.npz \
  --mode outdoor \
  --max-bursts 200 \
  --save-doa /tmp/reprocessed_doa_music.npz
```

You can also pass the session **directory** instead of the `.npz` path.

---

## Useful flags (cheat sheet)

| Flag | Purpose |
|------|---------|
| `--mode outdoor` | Wide Doppler, outdoor thresholds |
| `--mode indoor` | Narrow Doppler (lab / fixed TX) |
| `--gain DB` | IF gain override |
| `--freq HZ` | Centre frequency override |
| `--algo music\|capon\|bartlett` | DOA algorithm |
| `--record DIR` | Enable session recording |
| `--no-record-raw` | Skip raw CPI in recording |
| `--gui` | Matplotlib live UI |
| `--phase-cal` | Enable phase correction (needs `--cal-file`) |
| `--debug-dir /tmp/doa_dbg` | Save per-stage `.npy` files for debugging |

Config file (persistent defaults): `doa_config.toml` in this directory.

---

## Troubleshooting

| Symptom | Try |
|---------|-----|
| No frames / connection error | Restart Heimdall; check `daq_ip` / ports in config |
| Bursts detected, no DOA lines | Lower `--mode outdoor` thresholds or reduce `snr_min_db` in config |
| Wild az/el jumps | Normal with low SNR; use GUI to watch `papr_db` |
| Huge `raw_iq.npz` | Use `--no-record-raw` or shorter sessions (~5 MB per CPI frame) |

---

## File map

```
doa_iridium_grc/
  run_doa.py           ← live pipeline (this guide)
  run_doa_offline.py   ← replay recordings
  doa_config.toml      ← defaults
  lark/                ← DSP + recording library
krakenSDR/data/doa_iridium/   ← default recording output
```
