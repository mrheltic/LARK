"""Timing conventions: CPI epochs rebuilt on the sample grid, burst offsets."""

import json
import os

import numpy as np

from core.recording import rebuild_epochs_on_grid


def test_grid_rebuild_removes_bursty_write_jitter():
    rng = np.random.default_rng(0)
    cpi = 0.064
    # 400 CPIs on the grid, 30% dropped, mtime = end of CPI + bursty latency
    slots = np.sort(rng.choice(np.arange(600), 400, replace=False))
    end = 1000.0 + (slots + 1) * cpi
    lat = 0.02 + rng.exponential(0.01, slots.size)
    mt = end + lat
    # bursty writes: some frames flushed together with the next one
    for k in range(1, slots.size - 1, 17):
        mt[k] = mt[k + 1] - 0.001
    out = rebuild_epochs_on_grid(mt, cpi)
    err = (out - np.median(out - end)) - end        # compare up to the constant lag
    assert np.all(np.diff(out) > cpi * 0.99)         # strictly one CPI or more apart
    assert np.median(np.abs(err)) < 1e-9             # most frames back on their slot exactly
    assert np.mean(np.abs(err) < 1e-6) > 0.9


def test_grid_rebuild_keeps_mtime_convention():
    cpi = 0.064
    end = 50.0 + (np.arange(100) + 1) * cpi
    mt = end + 0.021
    out = rebuild_epochs_on_grid(mt, cpi)
    np.testing.assert_allclose(out, mt, atol=1e-9)   # clean mtimes: nothing to change


def test_load_peaks_adds_burst_offset(tmp_path):
    import scripts.pnt_solve as ps
    session = tmp_path / "session_x"
    (session / "raw").mkdir(parents=True)
    (session / "doa_multi_music").mkdir()
    (session / "meta.json").write_text(json.dumps({"fs_hz": 1_000_000.0}))
    np.save(session / "raw" / "frame_000000.npy", np.zeros((5, 64000), np.complex64))
    np.save(session / "raw" / "frame_000001.npy", np.zeros((5, 64000), np.complex64))
    with open(session / "raw" / "timestamps.csv", "w") as fh:
        fh.write("0,100.000\n1,100.064\n")
    rows = [{"frame": 0, "peaks": [[10.0, 40.0, 0.0, 5.0]], "cfo_per_peak": [100.0],
             "snr_per_peak": [9.0], "b0_per_peak": [20000]},
            {"frame": 1, "peaks": [[11.0, 41.0, 0.0, 5.0]], "cfo_per_peak": [90.0],
             "snr_per_peak": [9.0]}]
    with open(session / "doa_multi_music" / "doa_multi.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    t, cfo, *_ = ps.load_peaks(str(session), "doa_multi_music", 1.0, -90.0, 0)
    np.testing.assert_allclose(t, [100.0 - 1.0 + 0.020, 100.064 - 1.0])
