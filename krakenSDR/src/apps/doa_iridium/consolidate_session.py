#!/usr/bin/env python3
"""Build raw_iq.npz / doa_music.npz from a session's incremental raw/ and doa/ files."""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.recording import consolidate_session  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Consolidate session raw/ + doa/ into .npz files")
    p.add_argument("session_dir", help="Path to session_YYYYMMDD_HHMMSS/")
    p.add_argument("--raw-only", action="store_true", help="Build raw_iq.npz only")
    p.add_argument("--doa-only", action="store_true", help="Build doa_music.npz only")
    args = p.parse_args()

    do_raw = not args.doa_only
    do_doa = not args.raw_only
    consolidate_session(args.session_dir, raw=do_raw, doa=do_doa)


if __name__ == "__main__":
    main()
