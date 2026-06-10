# LARK — always-on agent context

Full guide: **[AGENTS.md](./AGENTS.md)** · Architecture: **[docs/architecture.md](./docs/architecture.md)**

## Quick facts

- Iridium DOA at **1626 MHz**, KrakenSDR **5-channel UCA**, pure Python.
- Work from **`krakenSDR/src/`**; app in **`apps/doa_iridium/`**, DSP in **`core/`**.
- Az **0°=N clockwise**; el above horizon; UCA radius **0.4253λ** (λ/2 chord spacing).
- **Phase calibration required** for real measurements (`use_phase_cal = true`).
- One preamble **tone = one satellite**; Doppler (CFO) separates concurrent sats.

## Agent rules

- Minimal diffs; match existing conventions; no commits unless asked.
- Do not modify `external/` submodules without explicit request.
- Operations/commands: `krakenSDR/src/apps/doa_iridium/README.md`.
