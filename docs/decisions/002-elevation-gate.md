# ADR 002: Minimum elevation gate (`el_min_deg`)

**Status:** Accepted  
**Date:** 2026-06  
**Context:** Offline reprocess quality filtering

## Problem

MUSIC (and to a lesser extent Capon) produces many **spurious low-elevation**
estimates (roughly 5°–15°) that do not correspond to real satellite geometry.
Users observed excessive short tracks and skyplot clutter near the horizon.

## Decision

Apply `el_min_deg` (default **10°**) in `multi_peak._doa_from_tone()` after peak
pick: reject peaks with `el_deg < el_min_deg`.

Configurable via `--el-min` on `reprocess_session.py` / `batch_reprocess.py`.

## Rationale

1. **UCA geometry:** steering vectors at low elevation are densely spaced; small
   eigenvalue noise causes the MUSIC peak to jump unpredictably.
2. **Physical prior:** useful Iridium passes for a fixed ground station spend
   most visible time above ~15°–20°; sub-10° estimates are rarely actionable.
3. **Horizon symmetry:** a planar array cannot reliably distinguish near-horizon
   directions.

This is a **pragmatic filter**, not a CRB-optimal estimator improvement.

## Consequences

- Reduces burst/peak count and track fragmentation in noisy sessions.
- May discard valid edge-of-pass samples — tune `el_min` per deployment if needed.
- Bartlett is less affected but still benefits from the same gate for consistency.

## Alternatives considered

- Raise `papr_min_db` globally — rejects weak bursts everywhere, not just horizon.
- Post-filter in clustering only — spurious peaks still pollute JSONL and viz.
- Elevation CRB weighting — more complex, not implemented.

## References

- `core/multi_peak.py` — `el_min_deg` parameter
- `docs/doa-pipeline.md` — steering vector and low-el instability
