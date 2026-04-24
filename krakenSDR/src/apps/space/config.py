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
#
# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY NOTES — cross array arm length (D_LAMBDA)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Standard cross array:  D_LAMBDA = 0.5 λ  (conservative, grating-lobe free)
#    azimuth 3 dB beamwidth ≈ 57° at el=30°  (MUSIC super-resolution)
#
#  Sparse cross:          D_LAMBDA = 1.0 λ  (recommended for satellite work)
#    azimuth 3 dB beamwidth ≈ 29° at el=30°  — 2× better angular resolution
#    Grating lobes: only below el=0° (horizon) — invisible to satellite DoA.
#    Safe for all targets with el > 0°.
#
#  Set D_LAMBDA = 1.0 and rebuild the array arms to ~18.44 cm for the best
#  achievable single-satellite grating-lobe-free resolution at 1626 MHz.
#
# ─────────────────────────────────────────────────────────────────────────────
#  ALGORITHM NOTES
# ─────────────────────────────────────────────────────────────────────────────
#
#  Available 2D DoA algorithms (selectable at runtime via the config dialog):
#
#    2D-MUSIC  (default) — sub-space method, sharp peaks, needs good D estimate
#    2D-Capon  (MVDR)    — data-adaptive, more robust when D is uncertain
#    2D-IAA              — iterative, sparser solution, best sidelobe suppression
#                          No D estimate needed. ~3× slower than Capon.
#
# =============================================================================
from __future__ import annotations

import os as _os
import sys as _sys

# \u2500\u2500 Hardware base \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#  Re-exports HEIMDALL_HOST, HEIMDALL_PORT, HEIMDALL_CTRL,
#  N_ANTENNAS, SAMPLE_RATE_HZ, HW_NUM_SAMPLES
from config_hw import *   # noqa: F401, F403

# \u2500\u2500 RF / Radio \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
FREQ_HZ        = 1_626_270_000  # carrier frequency [Hz] \u2014 Iridium simplex ring-alert channel
GAIN_DB        = 49.6             # IF gain [dB] applied to all channels

# \u2500\u2500 Cross array geometry \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
D_LAMBDA       = 0.5            # arm length [fraction of \u03bb]
#                                #  See GEOMETRY NOTES above.  Standard = 0.5, sparse = 1.0
ANTENNA_INPUT_ORDER = ["center", "north", "east", "south", "west"]
#                                #  Physical input order delivered by Heimdall/Kraken.
#                                #  Keep aligned with real coax wiring in the field.

# \u2500\u2500 Processing \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
COV_ALPHA      = 0.88           # EMA covariance weight for continuous (CW) mode

# \u2500\u2500 Display / animation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
INTERVAL_MS    = 80
VERBOSE_FRAMES = 0

# =============================================================================
# Profile override (LARK_PROFILE env var or --profile CLI flag)
# =============================================================================
_lark_profile = _os.environ.get("LARK_PROFILE", "").strip()
if _lark_profile:
    from profiles import apply_profile as _ap
    _ap(_lark_profile, _sys.modules[__name__])
