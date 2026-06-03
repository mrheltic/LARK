"""
doa_iridium_grc — Iridium burst DOA pipeline (PySDR, no GNU Radio required).

Files
─────
run_doa.py         Live pipeline: KrakenIQSource → burst detection → 2D DOA.
                   Usage: python3 run_doa.py [--gui] [--record DIR]

run_doa_offline.py Offline replay from raw_iq.npz, *_iq.npz, or .wav.
                   Usage: python3 run_doa_offline.py path/to/raw_iq.npz --mode outdoor

doa_config.toml  All tunable parameters (hardware, array, algorithm, recording, UI).

lark/            Signal-processing library (numpy only).
                 burst_processing.py — DSP stages
                 pipeline_debug.py — per-stage debug dumps
                 recording.py        — raw IQ + doa_music.npz session recorder
"""
