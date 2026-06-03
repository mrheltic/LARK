"""Shared sys.path setup for lark modules."""

from __future__ import annotations

import os
import sys


def ensure_src_path() -> str:
    """Add krakenSDR/src to sys.path so core.* imports resolve."""
    src = os.environ.get(
        "LARK_SRC",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    )
    src = os.path.abspath(src)
    if src not in sys.path:
        sys.path.insert(0, src)
    return src
