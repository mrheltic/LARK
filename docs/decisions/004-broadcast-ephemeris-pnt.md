# ADR 004: Broadcast ephemeris and self-positioning (PNT)

**Status:** Accepted
**Date:** 2026-08
**Context:** Giving the DOA system a purpose beyond "where are the satellites"

## Problem

LARK measured directions to Iridium satellites and scored them against TLEs. It
answered *how well* it worked but never *what it was for*. The natural inverse —
take the satellites as known and solve for the **observer** — is a GNSS-denied
positioning application, but it appeared to need exactly what such an
application cannot assume: a TLE catalogue, a network to fetch it, and a
disciplined clock.

An earlier angles-only attempt (`scripts/localize_from_doa.py`) fixed the
observer to **150 km**, with the east axis not moving at all.

## Decision

Decode the payload we were discarding, and solve from Doppler rather than angles.

1. **Ephemeris from the air.** IRA (Ring Alert) frames carry the transmitting
   satellite's own ECEF position; IBC frames carry Iridium system time. Both are
   demodulated with `gr-iridium` + `iridium-toolkit` from exported single-channel
   IQ (`scripts/export_iq.py`), then turned into a usable ephemeris by
   `core/broadcast_ephemeris.py`.
2. **Doppler as the primary observable**, angles only for the initial guess and
   as an independent check (`core/pnt_solver.py`, `scripts/pnt_solve.py`).
3. **Per-burst satellite assignment by Doppler**, not the DOA track clustering.

## Rationale

**Angles cannot do this.** At ~4° of DOA error and ~800 km slant range the
cross-range error is ~55 km per burst. Doppler is far stronger: displacing the
observer 1 km changes the pass Doppler by ~14 Hz north / 6.5 Hz east, and the
measured residual against the broadcast ephemeris has a **MAD of 46 Hz** once a
single LO offset is removed. The cost surface has one clean minimum over ±300 km.

**Three stages are mandatory, each for a measured reason.**

- *Radius filter.* About **51%** of IRA position fields are corrupt even with a
  valid BCH check. Rejecting anything outside a 7100–7250 km geocentric radius
  removes essentially all of them; a corrupted 12-bit word almost never lands
  back on the orbital sphere. (`iridium-toolkit` applies the same test.)
- *Time remapping.* Export concatenates CPIs that are not adjacent in time, so a
  chunk's internal clock runs slow by the recording duty cycle (~77%). Left
  uncorrected the decoded positions imply 9.86 km/s instead of 7.47.
- *Short-arc fit.* 4 km quantisation makes finite differences useless for
  velocity — two fixes 90 ms apart would imply 44 km/s. A degree-5 polynomial
  through the pass tracks a 600 s arc to ~1 m, three orders of magnitude below
  the quantisation floor, and its derivative gives a clean velocity.

**Tracks are the wrong unit for PNT.** The clusterer mixes satellites: roughly a
fifth of the peaks in the longest track of the reference session belong to a
different satellite. Assigning each burst independently by Doppler residual is
close to unambiguous, since satellites sit tens of kHz apart.

**Time is worth as much as ephemeris.** With the same Doppler data and the same
solver, SGP4 gives 7.31 km on the uncorrected host clock and 2.68 km once the
IBC-measured offset (+1.29 s) is applied. The broadcast path wins not because it
is a better ephemeris — it is not — but because it is the only source of the
time.

## Results (session_20260605_110716, single antenna, blind)

No TLE, no network, no known start, no satellite labels:

| Observable | Error | 1σ ellipse |
|---|---|---|
| Angles only | 27.6 km | 3.6 × 1.8 km |
| Doppler, all peaks, one global Δf | 2.29 km | 0.83 × 0.31 km |
| **Doppler, rank-0 peaks + per-satellite Δf** | **0.87 km** | 0.58 × 0.21 km |

Two refinements got it from 2.29 km to 0.87 km, both justified by measurement
rather than tuning:

- **Rank-0 peaks only.** Against the broadcast ephemeris, 24% of strongest-tone
  peaks are Doppler outliers versus 83% of rank-1 and 75% of rank-2 — the
  secondary peaks are the ghosts ADR 001 describes. Dropping them: 2.26 → 1.18 km.
- **One Δf per satellite.** The four transmitters differ by ~50 Hz (0.03 ppm),
  and the *same pattern appears under two independent ephemerides* (correlation
  0.957 between broadcast and SGP4), so it is a real transmitter effect being
  absorbed into position, not overfitting. 1.18 → 0.87 km.

Single satellite, single pass: 1.35 km (IRIDIUM 100), 1.37 (125), 2.41 (133),
3.53 (136).

Error saturates at ~2 km after **~10 minutes** of listening — about one pass.
More data past that does not help, which independently confirms the floor is
systematic, not statistical: the formal statistical floor for 1541 bursts at
62 Hz is 0.27 km, eight times below what is achieved.

Ephemeris validation: the four decoded `sat_id`s match TLE satellites to
2.1 km — the quantisation floor — with the runner-up 1300–1450× further away.

### The array earns its keep at the demodulator too

Same recording, same CPIs, only the spatial combining differs. Restricted to the
1807 CPIs that carry a DOA estimate (and therefore beamforming weights):

| Combining | IRA frames | vs one element | IBC frames |
|---|---|---|---|
| One element | 683 | — | 15 |
| Flat sum | 472 | −31% | 5 |
| **Steered (DOA weights)** | **893** | **+31%** | 32 |

All four satellites gain (+20% to +46%); the spatial selectivity of steering at
the strongest peak does not cost the others.

The flat sum is the instructive control: it is *worse* than a single element,
because summing a UCA without steering is a beam pointed at zenith and a
satellite at 30–60° elevation lands in the array factor's roll-off. Array gain
requires pointing, and pointing requires knowing the direction — which is what
the rest of this repo computes.

Scope matters here and is easy to get wrong: over *all* CPIs beamforming scores
−9%, because 92% of them have no DOA and fall back to the (worse) sum.

## Consequences

- The system now produces a position fix, so "why" has a one-sentence answer:
  GNSS-independent PNT from signals of opportunity, with an array.
- A new external dependency for the decode stage only: `gr-iridium` must be
  built (system **pybind11 2.11.1**, not the pip 3.x — GNU Radio's ABI is
  `__pybind11_internals_v5`). The solver itself has no such dependency.
- `scripts/localize_from_doa.py` remains as the angles-only baseline.
- The formal 1σ does not cover every single-satellite fix: systematic terms —
  2 km ephemeris quantisation, per-track elevation bias, LO drift — are outside
  the measurement covariance. Reported as-is.
- **Single-pass fixes scatter by a few km for reasons we could not pin down.**
  The spread (1.4–3.5 km) is not explained by geometry — IRIDIUM 125 has the
  worst Doppler sensitivity and one of the best fixes — nor by the freedom in Δf
  (fixing it to a known value changes little), nor by the ephemeris source (with
  SGP4 the spread persists but reorders: IRIDIUM 100 improves to 0.12 km while
  136 degrades to 4.22 km). Use several satellites; treat any one pass as
  carrying an unpredictable few-km bias. This is an open question, not a
  solved one.

## Alternatives considered

- **Cached TLE almanac.** Removes the network but not the dependency; also
  supplies no time. Rejected once IBC decoding proved reliable.
- **Pseudorange/TDOA** as `iridium-toolkit`'s `locator.py` does. Needs
  sample-accurate burst timing, the most fragile part of this recorder. Doppler
  needs only a stable frequency reference.
- **Writing our own demodulator** instead of building gr-iridium. Deferred: with
  no reference decoder it is debugging blind. `libreSDR`'s simulator cannot serve
  as that reference — it generates an invented frame format (rate-1/2 K=7
  convolutional) while real IRA uses BCH(31,21).
