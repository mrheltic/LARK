# =============================================================================
#  KrakenSDR — compatibility shim
#  Configuration is now split into two dedicated files:
#    config_hw.py   — hardware, RF, antenna array
#    config_doa.py  — DoA algorithm, processing, calibration, display
#
#  This file re-exports everything from both so existing code that does
#  "import config as C" continues to work unchanged.
#  Edit config_hw.py or config_doa.py directly instead of this file.
# =============================================================================
from config_hw  import *   # noqa: F401, F403
from config_doa import *   # noqa: F401, F403

# ── Profile override via LARK_PROFILE env var ─────────────────────────────────
# Set the environment variable before launching any script to select a
# pre-defined parameter set, e.g.:
#
#   LARK_PROFILE=ism_868       python3 apps/doa/doa_runner.py
#   LARK_PROFILE=iridium_1626  python3 apps/iridium/iridium_live.py
#
# A --profile CLI flag on individual scripts overrides this at runtime.
# =============================================================================
import os as _os
_lark_profile = _os.environ.get("LARK_PROFILE", "").strip()
if _lark_profile:
    from profiles import apply_profile as _apply_lark_profile
    _apply_lark_profile(_lark_profile)
