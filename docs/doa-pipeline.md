# DOA pipeline — signal processing reference

Technical reference for presentations, onboarding, and agent context. For
command-line workflows see
`krakenSDR/src/apps/doa_iridium/README.md`.

## Physical signal

Iridium **IRA (Ring Alert)** bursts include an unmodulated preamble tone at:

```
f_tone = f_carrier + 3125 Hz + f_Doppler
```

At 1626.27 MHz, Doppler from LEO satellites can shift the tone by roughly ±40 kHz.
The tone is a pure CW — ideal for measuring **inter-antenna phase** and thus DOA.

## Array geometry

Five antennas on a circle of radius **r = 0.4253λ**:

- Chord spacing between adjacent antennas: **λ/2** (from `2r·sin(π/5)`).
- Antenna k at angle `φ_k = ant0_offset + k·(360°/5)` clockwise from North.
- Positions: `(p_E, p_N) = (r·sin φ_k, r·cos φ_k)`.

### Angular conventions

| Quantity | Convention |
|----------|------------|
| Azimuth φ | 0° = North, increases **clockwise** (compass) |
| Elevation θ | 0° = horizon, 90° = zenith |
| CFO | `tone_hz − 3125 Hz` (Doppler discriminator) |

### Steering vector

For direction (φ, θ), the phase at antenna k is:

```
τ_k = 2π · r · cos(θ) · cos(φ_k − φ)
a_k = exp(j · τ_k)
```

Elevation enters through **cos(θ)** — the projection of the wave vector onto the
horizontal plane. At low elevation, many grid points have similar steering vectors;
MUSIC becomes unstable near the horizon (see ADR 002).

## Processing stages (per CPI)

```
(5, N) IQ  @ 1.024 MS/s, N ≈ 131072 (128 ms CPI)
    │
    ├─ 1. Energy burst detection (ch0)
    │      median noise floor, threshold × block power
    │
    ├─ 2. Preamble tone scan (FFT, up to K peaks)
    │      each peak → one satellite candidate
    │
    └─ for each tone:
           ├─ 3. BPF + per-channel RMS normalize (Hann mask ~15 kHz)
           ├─ 4. Phase calibration (optional, required in field)
           ├─ 5. Matched-filter covariance  R = y·yᴴ  (rank-1)
           ├─ 6. 2D spectrum on 360×86 grid (MUSIC / Capon / Bartlett)
           └─ 7. Peak pick + gates (SNR, PAPR, el_min)
```

Live pipeline adds **EMA smoothing** on az/el and covariance; offline reprocess
does not (raw per-burst estimates for analysis).

## Stage details

### Matched-filter covariance

Optimal for a single CW tone in AWGN:

```
y_k = (1/N) Σ x_k[n] · exp(−j2πf_tone·n/fs)
R_mf = y · yᴴ   (rank-1)
```

SINR is estimated from eigenvalue spread of the sample covariance (scale-invariant
after RMS normalization).

**Important:** with rank-1 R, MUSIC/Capon/Bartlett often peak at the **same**
(az, el) for a given tone. Algorithm choice mainly affects **accept/reject**
behavior on marginal bursts and multi-path edge cases — not a magic resolution
switch. **Phase calibration** is the dominant accuracy lever.

### 2D algorithms (summary)

| Algorithm | Spectrum | Strength | Weakness |
|-----------|----------|----------|----------|
| **MUSIC** | 1 / ‖E_n^H a‖² | Sharp peaks when SNR good | Unstable on weak / low-el |
| **Capon** | 1 / (a^H R⁻¹ a) | Adaptive nulling | Sensitive to R conditioning |
| **Bartlett** | a^H R a | Always stable, broad lobe | More false accepts, lower selectivity |

Bartlett tends to produce **more bursts** (passes gates on weaker CPIs) with
**higher mean PAPR** (broad lobe raises peak-to-average ratio).

### Quality gates

| Gate | Default | Purpose |
|------|---------|---------|
| `snr_min_db` | 3.0 | Reject incoherent bursts |
| `papr_min_db` | 2.0 | Reject flat / noisy spectra |
| `el_min_deg` | 10.0 (reprocess) | Reject horizon artifacts |

PAPR = `10·log10(max(spec_linear) / mean(spec_linear))` over the 2D grid.

## Multi-satellite strategy

**Correct approach (implemented):** scan K tones → **separate** BPF + covariance +
DOA per tone. Each tone's Doppler identifies a different satellite.

**Rejected approach:** extract K peaks from one spectrum on one covariance.
Secondary MUSIC peaks are often **elevation harmonics** of the same source, not
separate satellites.

## Offline reprocess output

`reprocess_session.py` writes per algorithm subdirectory:

```
doa_multi_music/   (or capon, bartlett)
├── doa_multi.jsonl    one JSON object per burst
├── burst_NNNNNN.npz   optional spec2d / az_slice
├── waterfall.npz      stacked azimuth cuts vs time
└── tracks.json        clustered satellite passages
```

Each JSONL row contains `peaks[]`, `cfo_hz`, `track_ids[]`, timing metadata.

### Track clustering

Greedy association using proximity in (az, el, CFO) and max time gap. Parameters:
`max_az_deg`, `max_el_deg`, `max_cfo_hz`, `min_track_len`, `max_gap_s`.

Use `--recluster-only` to re-run clustering on existing JSONL without reprocessing IQ.

## Visualization tools

| Tool | Mode | Purpose |
|------|------|---------|
| `run_doa.py --gui` | Live | Skyplot, spectrum, phase diagnostics |
| `run_doa_offline.py --replay` | Interactive | Timeline, multi-algo switch, hist skyplot |
| `run_doa_offline.py --compare` | Static | Overlay MUSIC/Capon/Bartlett |
| `--no-tracks` | Flag | Flat peak coloring, disable track filter keys |

## Known limitations

1. **Planar UCA** — poor elevation resolution near horizon; use `el_min` gate.
2. **Rank-1 covariance** — no multi-source super-resolution within one tone.
3. **5 elements** — limited spatial diversity; calibration critical.
4. **Live timing** — use session timestamps, not `frame_index × CPI_duration`.

## Glossary (short)

| Term | Meaning |
|------|---------|
| CPI | Coherent Processing Interval — one IQ frame from Kraken |
| IRA | Iridium Ring Alert burst type |
| CFO | Carrier frequency offset ≈ satellite Doppler |
| PAPR | Peak-to-average power ratio of DOA spectrum |
| UCA | Uniform Circular Array |

## Related

- [architecture.md](./architecture.md)
- [decisions/001-multi-peak-per-tone.md](./decisions/001-multi-peak-per-tone.md)
- [decisions/002-elevation-gate.md](./decisions/002-elevation-gate.md)
