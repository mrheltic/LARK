# =============================================================================
#  apps/space — 3-D satellite DoA (cross array) configuration
#
#  Covers Iridium L-band direction-finding using a 5-element cross ("+") array
#  for simultaneous azimuth + elevation estimation.
#
#  Physical Kraken / Heimdall input order used by this setup:
#      ch0 = center   ch1 = north   ch2 = east   ch3 = south   ch4 = west
#
#  The 3-D solver internally reorders channels to its canonical geometry:
#      [center, east, north, west, south]
#
#  This means you can wire the array as above for field collection and still
#  obtain correct absolute azimuth/elevation estimates, provided the array is
#  physically aligned to geographic north.
#
#  Hardware constants (Heimdall address, N_ANTENNAS, SAMPLE_RATE_HZ, …) are
#  re-exported from config_hw.py so scripts can import a single module:
#
#    import config as C    →   C.FREQ_HZ, C.D_LAMBDA, C.HEIMDALL_HOST …
#
#  Apply a named profile at launch to adjust parameters without editing
#  this file:
#
#    LARK_PROFILE=iridium_1626      python3 apps/space/space_doa_realtime.py
#    python3 apps/space/space_collector.py --profile iridium_1626
#
#  See krakenSDR/src/profiles.py for all available profiles and their values.
# =============================================================================
from __future__ import annotations

import os as _os
import sys as _sys

# ── Hardware base ─────────────────────────────────────────────────────────────
#  Re-exports HEIMDALL_HOST, HEIMDALL_PORT, HEIMDALL_CTRL,
#  N_ANTENNAS, SAMPLE_RATE_HZ, HW_NUM_SAMPLES
from config_hw import *   # noqa: F401, F403

# ── RF / Radio ────────────────────────────────────────────────────────────────
FREQ_HZ        = 1_626_270_000  # carrier frequency [Hz] — Iridium simplex ring-alert channel
GAIN_DB        = 15             # IF gain [dB] applied to all channels
#                                #  Same guidance as apps/iridium/config.py.

# ── Cross array geometry ──────────────────────────────────────────────────────
D_LAMBDA       = 0.5            # arm length [fraction of λ]
#                                #  λ @ 1626 MHz ≈ 18.44 cm  →  D_LAMBDA=0.5 → arm ≈ 9.22 cm
#                                #  Adjust to your physical build. Values above 0.5 λ
#                                #  start introducing grating-lobe ambiguities.
#                                #  Measure the physical arm length from center to element tip
#                                #  and divide by (c / FREQ_HZ).
ANTENNA_INPUT_ORDER = ["center", "north", "east", "south", "west"]
#                                #  Physical input order delivered by Heimdall/Kraken.
#                                #  Keep this aligned with the real coax wiring used in the field.

# ── Processing ────────────────────────────────────────────────────────────────
COV_ALPHA      = 0.88           # EMA covariance weight for continuous (CW) mode
#                                #  Lower than DoA default: Iridium bursts are short (~20 ms),
#                                #  so each burst provides a near-independent snapshot.

# ── Display / animation ───────────────────────────────────────────────────────
INTERVAL_MS    = 80             # milliseconds between Matplotlib animation frames
VERBOSE_FRAMES = 0              # print Heimdall diagnostics every N frames (0 = silent)

# =============================================================================
# Profile override (LARK_PROFILE env var or --profile CLI flag)
# =============================================================================
#  The LARK_PROFILE env var is applied at import time so module-level reads see
#  correct values. The --profile CLI flag calls apply_profile(name, C) after
#  argparse, overriding these module attributes directly.
_lark_profile = _os.environ.get("LARK_PROFILE", "").strip()
if _lark_profile:
    from profiles import apply_profile as _ap
    _ap(_lark_profile, _sys.modules[__name__])
