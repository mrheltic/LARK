"""
burst_processing.py — backward-compat shim.

Canonical module: apps.doa_iridium_grc.lark.burst_processing
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from apps.doa_iridium_grc.lark.burst_processing import (  # noqa: F401
    apply_bpf_and_normalize,
    compute_mf_covariance,
    detect_energy_bursts,
    scan_preamble_tones,
)
from apps.doa_iridium_grc.lark.burst_processing import __all__  # noqa: F401
