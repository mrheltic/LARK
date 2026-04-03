"""
pysdr_doa.io
============
Hardware I/O layer.

Re-exports :class:`KrakenIQSource` and :class:`IQHeader` from the existing
``kraken_iq_source`` module so callers can use either import style::

    from kraken_iq_source import KrakenIQSource   # legacy
    from io import KrakenIQSource                 # new-style
"""

from __future__ import annotations

from kraken_iq_source import IQHeader, KrakenIQSource  # noqa: F401

__all__ = ["IQHeader", "KrakenIQSource"]
