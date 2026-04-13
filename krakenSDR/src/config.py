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
