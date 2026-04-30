"""
Shared import-path bootstrap utilities for KrakenSDR entrypoint scripts.

Goal
----
Keep a single coherent logic for path setup across apps and scripts so that:
- app-local ``config.py`` is resolved consistently,
- ``core`` / ``hardware`` modules import without launcher-dependent hacks,
- ``shared`` package is available when needed.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple


def _prepend(path: Path) -> None:
    p = str(path.resolve())
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def _find_src(start: Path) -> Path:
    for p in [start, *start.parents]:
        if p.name == "src":
            return p
    raise RuntimeError(f"Unable to locate 'src' parent from {start}")


def setup_paths(
    file_path: str,
    app_relative: Optional[str] = None,
    include_root: bool = False,
) -> Tuple[str, str, str, Optional[str]]:
    """
    Configure sys.path for KrakenSDR entrypoints.

    Returns
    -------
    tuple[str, str, str, Optional[str]]
        (here_dir, src_dir, repo_root_dir, app_dir_or_none)
    """
    here = Path(file_path).resolve().parent
    src = _find_src(here)
    root = src.parent

    app_dir: Optional[Path] = None
    if app_relative:
        app_dir = (src / app_relative).resolve()
        if not app_dir.exists():
            raise RuntimeError(f"Configured app path does not exist: {app_dir}")
    elif here.parent.name == "apps":
        app_dir = here

    # Desired priority: app_dir (if any) > script dir > src > repo root (optional)
    if include_root:
        _prepend(root)
    _prepend(src)
    _prepend(here)
    if app_dir is not None:
        _prepend(app_dir)

    return str(here), str(src), str(root), (str(app_dir) if app_dir else None)
