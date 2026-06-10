# scripts/legacy — diagnostics for the old single-file recording format

These tools operate on the pre-session `doa_iridium_YYYYMMDD_HHMMSS_iq.npz`
recordings (single consolidated file with `R_inst` / `R_avg` stacks), a format
replaced in June 2026 by session directories (`session_*/raw/frame_*.npy`).

| Script | Purpose |
|---|---|
| `analyse_outdoor_pass.py` | Summary stats (CFO/SNR/burst count) of one old-format pass recording |
| `diagnose_npz.py` | Rank-1 quality, phase stability and MDL histograms from saved covariances |

For current sessions use instead:

- `scripts/eval_doa_accuracy.py` — per-burst az/el error vs TLE ground truth
- `scripts/plot_track_vs_tle.py` — measured DOA trajectory vs SGP4 track
- `apps/doa_iridium/reprocess_session.py` — full offline reprocess
