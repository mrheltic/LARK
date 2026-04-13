# =============================================================================
#  apps/iridium — Iridium L-band burst receive & decode configuration
#
#  Covers Iridium TDMA downlink at 1626.270 MHz, received on a single antenna
#  or on channel 0 of the KrakenSDR.
#
#  Hardware constants (Heimdall address, N_ANTENNAS, SAMPLE_RATE_HZ, …) are
#  re-exported from config_hw.py so scripts can import a single module:
#
#    import config as C    →   C.FREQ_HZ, C.HEIMDALL_HOST, C.GAIN_DB …
#
#  Apply a named profile at launch to adjust parameters without editing
#  this file:
#
#    LARK_PROFILE=iridium_1626   python3 apps/iridium/iridium_live.py
#    python3 apps/iridium/iridium_detector.py --profile iridium_1626
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
#                                #  All Iridium TDMA downlink simplex bursts on this channel.
#                                #  Alternative channels: 1621.25 / 1623.0 / 1625.5 MHz.

GAIN_DB        = 15             # IF gain [dB] applied to all channels
#                                #  Iridium signals at ground level: −70 … −50 dBm typical.
#                                #  15 dB is a safe starting point; raise to 20–25 if bursts
#                                #  are missed; lower if ADC clips (PAPR drops below 3 dB).

# ── Display / diagnostics ─────────────────────────────────────────────────────
VERBOSE_FRAMES = 0              # print Heimdall frame diagnostics every N frames (0 = silent)
#                                #  Set to 1 to trace every frame; useful during HW bring-up.

# =============================================================================
# Profile override (LARK_PROFILE env var or --profile CLI flag)
# =============================================================================
#  The LARK_PROFILE env var is applied at import time so module-level reads
#  (e.g. _FS = float(C.SAMPLE_RATE_HZ)) see the correct values.
#  The --profile CLI flag in individual scripts calls apply_profile(name, C)
#  AFTER argparse, overriding these module attributes directly.
_lark_profile = _os.environ.get("LARK_PROFILE", "").strip()
if _lark_profile:
    from profiles import apply_profile as _ap
    _ap(_lark_profile, _sys.modules[__name__])
