# ADR 001: Multi-peak DOA per preamble tone

**Status:** Accepted  
**Date:** 2026-06  
**Context:** Offline multi-satellite reprocessing

## Problem

A single CPI can contain bursts from multiple Iridium satellites simultaneously.
Each satellite has a distinct Doppler-shifted preamble tone. We need one (az, el)
estimate per satellite, not one blended estimate per CPI.

## Decision

For each detected burst window:

1. `scan_preamble_tones()` returns up to **K** tones (default K=3).
2. For **each tone**, run an independent chain: BPF → MF covariance → 2D DOA.
3. Use **per-tone CFO** (`tone_hz − 3125 Hz`) as the satellite discriminator.

Do **not** extract multiple peaks from a single 2D spectrum on a single covariance.

## Rationale

Secondary peaks in one MUSIC spectrum on rank-1 covariance are typically
**elevation sidelobes / harmonics** of the same source, not separate satellites.
Physical separation is in **frequency** (Doppler), not in angular spectrum peaks.

## Consequences

- `core/multi_peak.py` is the canonical CPI processor for offline reprocess.
- JSONL rows may contain multiple peaks; clustering uses per-peak CFO when available.
- Live `run_doa.py` still uses single-tone path (first/strongest tone) unless extended.

## References

- `core/multi_peak.py` module docstring
- `docs/doa-pipeline.md`
