# LARK — Iridium satellite direction finding with a KrakenSDR

LARK is a master's-thesis project that **passively locates Iridium satellites** from the
ground. It listens to the satellites' L-band downlink bursts with a 5-channel
[KrakenSDR](https://www.krakenrf.com/) **uniform circular array (UCA)**, estimates the
**azimuth and elevation** of each satellite from the phase differences across the five
antennas, and checks the result against orbital predictions (TLEs).

Everything is pure Python. The signal-processing "core" lives in
[`krakenSDR/src/core/`](krakenSDR/src/core/) and the application that drives it is in
[`krakenSDR/src/apps/doa_iridium/`](krakenSDR/src/apps/doa_iridium/).

- **Frequency:** 1626.27 MHz (Iridium ring-alert simplex channel — the most active one)
- **Sample rate:** 1.024 MS/s (KrakenSDR / Heimdall recording rate)
- **Array:** 5-element UCA, radius **0.4253 λ** (≈ λ/2 chord spacing at 1626 MHz)
- **Angle convention:** azimuth **0° = North, clockwise** (90° = East); elevation = degrees
  above the horizon.

If you are working on something similar, this README is the fast tour. The deep
operations handbook is [`krakenSDR/src/apps/doa_iridium/README.md`](krakenSDR/src/apps/doa_iridium/README.md);
the architecture write-up is [`docs/architecture.md`](docs/architecture.md).

---

## Repository layout

| Path | What's in it |
|------|--------------|
| `krakenSDR/src/core/` | The DSP library: burst detection, beamforming, the UCA DOA estimators. |
| `krakenSDR/src/apps/doa_iridium/` | The DOA app: `run_doa.py` (live) and `reprocess_session.py` (offline), plus config. |
| `krakenSDR/src/scripts/` | TLE fetch, ground-truth prediction, array calibration, plotting. |
| `krakenSDR/src/tests/` | Synthetic-signal unit tests for the DOA pipeline. |
| `libreSDR/` | The **satellite simulator / transmitter** (AD9363 "LibreSDR") — see below. |
| `shared/` | Iridium constants, the TLE catalogue, and burst↔satellite matching. |
| `external/` | Vendored upstream tools: `gr-iridium`, `iridium-toolkit`, `gr-krakensdr`. |
| `docs/`, `paper/` | Architecture notes and the thesis paper. |

---

## Background: the Iridium signal (why this works)

Iridium satellites transmit short TDMA bursts. Each downlink burst starts with a
**preamble run-in**: 64 symbols of constant rotation that, at 25 ksps, appear as a single
**continuous-wave tone at Rs/8 = 3125 Hz** above the channel centre. The rest of the burst
is π/4-DQPSK data, which we deliberately *ignore* — we never demodulate the message.

Two facts make direction finding tractable:

1. **One preamble tone = one satellite.** We don't need to decode anything; the clean CW
   tone is a perfect, narrowband target for beamforming.
2. **Doppler separates concurrent satellites.** A LEO satellite at ~780 km produces a
   Doppler shift (carrier-frequency offset, CFO) up to **±40 kHz**. When two satellites are
   up at once, their tones sit at different frequencies, so we isolate each one with a
   band-pass filter and process them independently.

All Iridium RF/modulation constants live in one place: [`shared/iridium.py`](shared/iridium.py).

---

## The DOA core (receive chain)

Each KrakenSDR **CPI** (coherent block of ~128 ms × 5 channels) goes through a 7-stage
pipeline. The stages are small, named functions you can read and test in isolation:

```
 5-channel IQ (one CPI)
   │
   ├─1. detect_energy_bursts()      where in the block is a burst?      core/burst_processing.py
   ├─2. scan_preamble_tones()       which tone(s)? (3125 Hz + Doppler)  core/burst_processing.py
   ├─3. apply_bpf_and_normalize()   isolate one satellite, unit-RMS     core/burst_processing.py
   ├─4. apply_phase_correction()    align the 5 channels (calibration)  core/doa_algorithms.py
   ├─5. compute_mf_covariance()     matched filter → R = y·yᴴ (rank-1)  core/burst_processing.py
   ├─6. doa_{music,capon,bartlett}_uca_2d()   scan (az,el) grid         core/doa_uca_2d.py
   └─7. find_peak_uca_2d()          peak + parabolic refine → az, el    core/doa_uca_2d.py
```

When several satellites share a CPI, [`core/multi_peak.py`](krakenSDR/src/core/multi_peak.py)
(`process_cpi_for_multi`) runs stages 3–7 once **per Doppler tone**, yielding one
`(az, el, CFO)` triplet per satellite. This is cleaner than looking for multiple peaks in a
single MUSIC spectrum, where secondary peaks are usually elevation harmonics of the *same*
satellite (ghosts), not new ones.

### Two things worth internalizing

- **The matched-filter covariance is rank-1**, so MUSIC, Capon and Bartlett all peak at the
  *same* `(az, el)`. Picking "the best algorithm" is not the lever. **Phase calibration is** —
  on real data it moved the azimuth bias from **+98° to +1.7°**. Use `--algo` to compare,
  but spend your effort on calibration.
- **The UCA geometry is the model.** Steering vectors come from the antenna positions
  (radius 0.4253 λ, az 0° = N clockwise, el above horizon) in
  [`core/doa_uca_2d.py`](krakenSDR/src/core/doa_uca_2d.py) (`UcaConfig`). Get the geometry and
  the per-antenna phase offsets right and the rest follows.

---

## Running the DOA

### Live (radio attached, Heimdall/DAQ running)

```bash
cd krakenSDR/src/apps/doa_iridium
python3 run_doa.py --mode outdoor --gain 40 --record ../../../data/session --gui
```

It prints one JSON line per accepted burst (`az`, `el`, `snr_db`, `papr_db`, CFO…) to
stdout, and with `--record` it saves every CPI to a session directory. Useful flags:
`--phase-cal` / `--cal-file` (calibration), `--algo`, `--out FILE`, `--debug-dir`.

### Offline (reprocess a recorded session)

```bash
python3 reprocess_session.py <session_dir> --algo music --cov-bursts 16
```

This re-runs the pipeline on `raw/frame_*.npy`, extracts the top-K peaks per burst
(`--k-peaks`), clusters bursts into per-satellite **tracks**, and (with `--cov-bursts B`)
averages the covariance over B bursts to cut elevation noise. Output lands in a
`doa_multi_<algo>/` subfolder (`doa_multi.jsonl`, `tracks.json`, per-burst `.npz`).

A recording session looks like:

```
session_YYYYMMDD_HHMMSS/
├── meta.json                 freq, fs, gain, n_ant, cpi_size
├── raw/frame_000000.npy      complex64 (5, 131072) — one CPI
├── raw/timestamps.csv        frame_index → unix epoch (wall-clock matters!)
└── doa_multi_music/          written by reprocess_session.py
```

Full operational detail (modes, thresholds, every config key) is in
[`apps/doa_iridium/README.md`](krakenSDR/src/apps/doa_iridium/README.md) and
[`doa_config.toml`](krakenSDR/src/apps/doa_iridium/doa_config.toml).

---

## "Simulating" the satellite (the LibreSDR transmitter)

You can't schedule a real satellite for a bench test, so LARK builds a **fake satellite** and
transmits it over the air from a *known* direction. This is the `libreSDR/` side of the repo.

**1. Generate a realistic pass** — [`libreSDR/src/iridium/realistic_sim.py`](libreSDR/src/iridium/realistic_sim.py)
synthesizes a full Iridium downlink pass from one satellite, faithful to the real physical
layer:

- authentic TDMA burst timing (one IRA burst every 90 ms);
- π/4-DQPSK with RRC pulse shaping (roll-off β = 0.4) and a rate-1/2, K=7 convolutional code;
- a **geometric LEO Doppler model** (780 km altitude, 86.4° inclination): the carrier sweeps
  through the ±40 kHz Doppler curve with the right chirp rate as the satellite rises and sets;
- AWGN at a chosen SNR, plus a preamble correlator for self-testing detection.

**2. Transmit it** — [`libreSDR/src/tx/pass_sim.py`](libreSDR/src/tx/pass_sim.py) plays that IQ
out of a LibreSDR (AD9363) so the KrakenSDR receives a *real* over-the-air signal from a known
bearing — effectively "a satellite on a bench". Transmit on a safe band and at low power:

```bash
# Dry run: build the pass and save IQ, transmit nothing
python3 libreSDR/src/tx/pass_sim.py --dry-run --save pass.iq

# Loop a 45°-elevation pass over the air at low gain (note: --gain, not --tx-gain;
# --freq is an integer in Hz)
python3 libreSDR/src/tx/pass_sim.py --freq 100000000 --gain -55 --elev 45 --cyclic
```

(The TX flags are `--freq`, `--gain` [dB, default −20], `--elev`, `--dur`, `--snr`,
`--sat-id`, `--cyclic`, `--save`, `--dry-run`.)

**3. Pure-software synthetic bursts** — for unit-level checks that don't need any hardware,
[`krakenSDR/src/tests/test_burst_doa_pipeline.py`](krakenSDR/src/tests/test_burst_doa_pipeline.py)
(`_make_iridium_frame`) fabricates 5-channel bursts with a known `(az, el)`, Doppler and SNR by
applying the UCA steering phases directly, then asserts the pipeline recovers the angles.
Run them with `pytest krakenSDR/src/tests/`.

---

## Ground truth & validation (TLE)

To know whether an estimate is *correct*, we compare it to where the satellite actually was,
computed from orbital elements (TLEs) propagated with SGP4 (via `skyfield`).

- [`shared/iridium_tle.py`](shared/iridium_tle.py) — fetches the Iridium NEXT catalogue
  (CelesTrak), caches it, and predicts each satellite's Az/El/Doppler from your location.
- [`shared/satellite_tracker.py`](shared/satellite_tracker.py) — matches each detected burst to
  a satellite, either by **Doppler** similarity or by **angular** distance.
- [`krakenSDR/src/scripts/fetch_session_tle.py`](krakenSDR/src/scripts/fetch_session_tle.py) —
  downloads **epoch-matched** TLEs from Space-Track and freezes them into the session. This
  matters: SGP4 drifts ~1–3 km/day from the TLE epoch (mostly along-track → a Doppler *timing*
  error), so evaluating an old recording with today's elements silently weakens the ground truth.
- [`krakenSDR/src/scripts/iridium_groundtruth.py`](krakenSDR/src/scripts/iridium_groundtruth.py) —
  predicts all passes in a session and matches them to the measured DOA tracks.
- [`krakenSDR/src/scripts/plot_track_vs_tle.py`](krakenSDR/src/scripts/plot_track_vs_tle.py) —
  overlays measured tracks on the TLE prediction (polar sky plot + Az/El/Doppler time series)
  and reports residual statistics.

On the reference outdoor session this pipeline reaches roughly **3.7° median azimuth error**
and **3.3° median elevation error** (with multi-burst covariance averaging).

---

## Calibration (don't skip this)

Phase calibration is the single biggest factor in accuracy. For any real measurement set
`use_phase_cal = true` in [`doa_config.toml`](krakenSDR/src/apps/doa_iridium/doa_config.toml).
The calibration file `cal_tle.npz` (per-antenna phase offsets + array rotation from North) is
fitted from Iridium beacons of known direction with
[`krakenSDR/src/scripts/fit_array_cal.py`](krakenSDR/src/scripts/fit_array_cal.py). **Re-fit it
whenever you physically move or rotate the array.**

---

## Further reading

- [`AGENTS.md`](AGENTS.md) — full project guide / conventions.
- [`docs/architecture.md`](docs/architecture.md) — system architecture.
- [`krakenSDR/src/apps/doa_iridium/README.md`](krakenSDR/src/apps/doa_iridium/README.md) — the
  detailed operations handbook (theory + field procedure + every config knob).
- Upstream tools in `external/`: [`gr-iridium`](external/gr-iridium),
  [`iridium-toolkit`](external/iridium-toolkit) (used as references for the signal model).
