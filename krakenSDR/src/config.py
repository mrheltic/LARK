# =============================================================================
#  KrakenSDR — top-level hardware skeleton
#
#  This module re-exports the hardware constants from config_hw.py so that
#  legacy code doing "import config as C" still resolves Heimdall addresses,
#  sample rate, and channel count without modification.
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  Application-specific configuration lives INSIDE each app folder:       │
#  │    krakenSDR/src/apps/doa/config.py      — ISM band DoA (868 / 433 MHz) │
#  │    krakenSDR/src/apps/iridium/config.py  — Iridium L-band receive       │
#  │    krakenSDR/src/apps/space/config.py    — 3-D satellite DoA (cross arr)│
#  │                                                                         │
#  │  Each app's config.py re-exports hardware constants from config_hw plus │
#  │  its own frequency, gain, geometry, and algorithm defaults.             │
#  │  Scripts in apps/<group>/ put their own folder first on sys.path so     │
#  │  "import config as C" resolves to the app-local config.py.              │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  Profile system (LARK_PROFILE):
#    Each app config.py checks the LARK_PROFILE env var at import time.
#    You can also pass --profile NAME on the CLI of individual scripts.
#    See krakenSDR/src/profiles.py for available profiles.
# =============================================================================

from config_hw import *   # noqa: F401, F403  — hardware constants only
