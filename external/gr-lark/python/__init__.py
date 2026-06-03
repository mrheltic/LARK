"""
gr-lark — GNU Radio OOT module for LARK Iridium DOA processing.

Provides message-based (PDU) blocks for burst energy detection, tone scanning,
BPF normalization, matched-filter covariance, and 2D DOA estimation on UCA arrays.

All DSP logic is delegated to the LARK core library:
    core.doa_uca_2d, core.doa_algorithms, core.tracking, core.covariance
    apps.doa_iridium.burst_processing

These blocks are thin GNU Radio wrappers that:
  1. Accept vector streams or PDUs from upstream
  2. Call the corresponding core function
  3. Emit PDUs or messages downstream
"""

import os
import sys

_LARK_SRC = os.environ.get("LARK_SRC")
if _LARK_SRC:
    _src = os.path.abspath(_LARK_SRC)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from .iridium_burst_energy import iridium_burst_energy
from .iridium_tone_scanner import iridium_tone_scanner
from .iridium_bpf_normalizer import iridium_bpf_normalizer
from .iridium_mf_covariance import iridium_mf_covariance
from .iridium_doa_estimator import iridium_doa_estimator