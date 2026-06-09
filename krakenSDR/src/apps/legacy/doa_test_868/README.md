# 868 MHz DoA Pipeline (KrakenSDR UCA)

This module provides a robust **2D Direction-of-Arrival (DoA)** pipeline for a
5-element KrakenSDR Uniform Circular Array (UCA) in the 868 MHz ISM band.

It supports two acquisition modes:

- **CW realtime mode** (`doa_test_868_realtime.py`) for continuous beacon tones
- **Burst-gated mode** (`doa_test_868_burst.py`) for IRA-like bursts (preamble tone at +3125 Hz)

The processing chain is tuned for challenging indoor conditions (low SNR,
multipath, occasional channel instability), while preserving high resolution
when SNR is good.

---

## 1) High-Level Architecture

### CW Realtime pipeline

1. IQ acquisition from Heimdall (`KrakenIQSource`)
2. Optional pilot-tone extraction (`extract_pilot_tone`)
3. Optional amplitude normalization (`amplitude_normalize_channels`)
4. Optional enhanced preprocessing (disabled by default)
5. Covariance update with EMA (`CovarianceAccumulatorUca`)
6. Quality metrics: eigenspread + SNR + inter-channel phase differences
7. **Phase coherence gate** (reject unstable phase jumps)
8. **SNR-adaptive algorithm selection** (Bartlett / Capon / user-selected)
9. 2D spectrum + peak extraction (azimuth, elevation, PAPR)
10. **Circular azimuth outlier rejection**
11. UI update + optional recording to `.npz`

### Burst-gated pipeline

1. Streaming IQ buffering
2. Energy-based burst detection (`_detect_bursts`)
3. Preamble tone onset localization (`_find_tone_onset`)
4. Preamble extraction
5. Optional pilot extraction / normalization / preprocessing
6. Optional hardware phase correction
7. Stage-1 gate: eigenspread threshold
8. **Multi-burst covariance accumulation** (`MULTI_BURST_N`)
9. Stage-2 DoA with SNR-adaptive algorithm + PAPR threshold
10. EMA covariance update for stable phase/DoA tracking
11. Phase coherence + azimuth outlier rejection
12. UI update + optional recording to `.npz`

---

## 2) Why This Pipeline Is Robust

### SNR-adaptive algorithm switching

Configured in `config.py`:

- `SNR_LOW_DB`: below this, use **Bartlett** (most robust)
- `SNR_HIGH_DB`: above this, use user-selected algorithm (typically MUSIC)
- between low/high: use **Capon** as compromise

This avoids using high-resolution methods in regimes where they become unstable.

### Phase coherence gating

The pipeline tracks circular median of inter-channel phase differences
(`angle(R[k,0])`) and rejects snapshots with abrupt phase jumps.

Useful against:

- hardware glitches
- sudden multipath flips
- transient non-stationary interference

### Circular azimuth outlier rejection

DoA estimates are compared against running circular median and discarded if the
angular deviation exceeds a configured threshold.

This prevents large sporadic jumps from contaminating displayed direction and
recorded statistics.

### Multi-burst accumulation (burst mode)

`MULTI_BURST_N` accumulates several burst covariances before DoA estimation.
This increases effective SNR approximately by `10*log10(N)` dB.

Examples:

- `N=3` -> ~4.8 dB gain
- `N=5` -> ~7 dB gain

---

## 3) Core Files

- `config.py`: full runtime tuning (RF, geometry, gates, smoothing, thresholds)
- `doa_test_868_realtime.py`: CW realtime acquisition + UI + recording
- `doa_test_868_burst.py`: burst detection + preamble-gated DoA + UI + recording
- `../../core/doa_uca_2d.py`: UCA steering, MUSIC/Capon/Bartlett, metrics, EMA covariance

---

## 4) Typical Usage

Run from `krakenSDR/src/apps/doa_test_868`.

### Realtime CW mode

```bash
python3 doa_test_868_realtime.py --algo music
python3 doa_test_868_realtime.py --algo capon
python3 doa_test_868_realtime.py --demo
```

### Burst mode

```bash
python3 doa_test_868_burst.py --algo music
python3 doa_test_868_burst.py --algo bartlett
python3 doa_test_868_burst.py --demo
```

### Calibration

```bash
python3 doa_test_868_burst.py --calibrate 20.0
```

Where `20.0` is known transmitter azimuth in degrees.

---

## 5) Important Configuration Knobs

In `config.py`:

- `DOA_ALGORITHM`: default algorithm (`MUSIC`, `CAPON`, `BARTLETT`)
- `EIG_SPREAD_MIN_DB`: signal presence threshold
- `COV_ALPHA`: temporal smoothing for covariance
- `AZ_SMOOTH_ALPHA`: smoothing on burst-wise azimuth estimates
- `SNR_ADAPTIVE_ENABLED`, `SNR_LOW_DB`, `SNR_HIGH_DB`
- `PHASE_COHERENCE_ENABLED`, `PHASE_COHERENCE_MAX_JUMP_DEG`
- `AZ_OUTLIER_ENABLED`, `AZ_OUTLIER_MAX_DEV_DEG`, `AZ_OUTLIER_MIN_HISTORY`
- `MULTI_BURST_N`, `USE_EMA_FOR_DOA_BELOW_SNR`
- `CHANNEL_PHASE_OFFSETS_DEG`: hardware phase calibration vector

---

## 6) Data Output

Both scripts can record compressed `.npz` files in:

- default: `krakenSDR/data/doa_868`
- or custom `--out-dir`

Stored arrays include timestamps, az/el estimates, PAPR, SNR, eigenspread,
phase differences, covariance matrices (`R_real`, `R_imag`), and validity flags.

These files are intended for offline tuning and benchmarking.

---

## 7) Hardware Setup (Recommended)

- **TX beacon**: LibreSDR (CW or IRA burst profile)
- **RX array**: KrakenSDR 5-channel UCA
- **DAQ**: Heimdall reachable on configured TCP host/port
- **Geometry**: verify `RADIUS_LAMBDA`, `ANT_CCW`, `ANT0_OFFSET_DEG`
- **Calibration**: set `CHANNEL_PHASE_OFFSETS_DEG` after calibration run

---

## 8) Troubleshooting

### No detections / frequent `NO SIGNAL`

- check Heimdall connectivity and center frequency
- lower `SQUELCH_THRESHOLD_DB` if needed
- verify TX is active and mode is correct (CW vs BURST)

### Direction jumps wildly

- increase `COV_ALPHA` (stronger temporal decorrelation)
- enable/tighten phase coherence and az outlier gates
- increase `MULTI_BURST_N` in burst mode
- run hardware phase calibration

### PAPR too low in burst mode

- ensure algorithm-specific PAPR threshold is used
- verify burst alignment and preamble tone extraction
- avoid incompatible preprocessing settings in low-SNR sessions