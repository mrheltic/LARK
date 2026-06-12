#!/usr/bin/env python3
"""
fetch_session_tle.py — Epoch-matched TLEs for a session, from Space-Track.

CelesTrak only serves the *latest* orbital elements, and SGP4 accuracy
degrades ~1–3 km/day away from the TLE epoch (mostly along-track, which
turns into a Doppler timing error and silently weakens the ground-truth
matching).  This script downloads, for every Iridium NEXT satellite, the
historical element set whose epoch is closest to the session midpoint, and
writes it to ``<session>/iridium_tle.txt`` — the per-session snapshot that
all evaluation scripts use automatically from then on.

Requires a free Space-Track account (https://www.space-track.org).
Credentials, in order of precedence:
  1. env vars  SPACETRACK_USER / SPACETRACK_PASS
  2. JSON file ~/.config/lark/spacetrack.json
     {"identity": "you@mail.com", "password": "..."}

Usage:
    python3 scripts/fetch_session_tle.py <session_dir>
    python3 scripts/fetch_session_tle.py <session_dir> --margin-days 2
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_SRC))
for p in (_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.iridium_groundtruth import load_session_window  # noqa: E402
from shared.iridium_tle import load_catalogue  # noqa: E402

_BASE = "https://www.space-track.org"
_CRED_FILE = Path.home() / ".config" / "lark" / "spacetrack.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Epoch-matched session TLEs from Space-Track")
    p.add_argument("session_dir")
    p.add_argument("--margin-days", type=float, default=2.0,
                   help="Search elements with epoch within ± this many days "
                        "of the session midpoint (default: 2)")
    p.add_argument("--out", default="",
                   help="Output path (default: <session>/iridium_tle.txt)")
    return p.parse_args(argv)


def credentials() -> tuple[str, str]:
    user = os.environ.get("SPACETRACK_USER")
    pw = os.environ.get("SPACETRACK_PASS")
    if user and pw:
        return user, pw
    if _CRED_FILE.is_file():
        with open(_CRED_FILE, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("identity") and d.get("password"):
            return d["identity"], d["password"]
    sys.exit(
        "No Space-Track credentials.\n"
        "Set SPACETRACK_USER / SPACETRACK_PASS, or create "
        f"{_CRED_FILE} with {{\"identity\": …, \"password\": …}}"
    )


def tle_epoch(line1: str) -> datetime:
    """Epoch of a TLE line 1 (UTC)."""
    yy = int(line1[18:20])
    doy = float(line1[20:32])
    return (datetime(2000 + yy, 1, 1, tzinfo=timezone.utc)
            + timedelta(days=doy - 1.0))


def fetch_historical(norad_ids: list[int], t_from: datetime, t_to: datetime,
                     user: str, pw: str) -> str:
    """3LE text with all element sets of the given satellites in the window."""
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    login = urllib.parse.urlencode(
        {"identity": user, "password": pw}).encode()
    resp = opener.open(f"{_BASE}/ajaxauth/login", data=login, timeout=30)
    if resp.status != 200:
        sys.exit(f"Space-Track login failed (HTTP {resp.status})")

    ids = ",".join(str(i) for i in sorted(norad_ids))
    window = f"{t_from:%Y-%m-%d}--{t_to:%Y-%m-%d}"
    url = (f"{_BASE}/basicspacedata/query/class/gp_history"
           f"/NORAD_CAT_ID/{ids}/EPOCH/{window}"
           f"/orderby/NORAD_CAT_ID,EPOCH/format/3le")
    print(f"Querying gp_history for {len(norad_ids)} satellites, epoch {window} …")
    resp = opener.open(url, timeout=120)
    text = resp.read().decode("utf-8", errors="replace").strip()
    if not text or text.startswith("{"):
        sys.exit(f"Space-Track returned no TLE data:\n{text[:300]}")
    return text


def closest_per_satellite(tle_text: str, t_mid: datetime) -> dict[int, tuple]:
    """{norad_id: (name, line1, line2)} with epoch closest to t_mid."""
    lines = [ln.strip() for ln in tle_text.splitlines() if ln.strip()]
    best: dict[int, tuple] = {}
    i = 0
    while i + 1 < len(lines):
        if lines[i].startswith("1 "):
            name, l1, l2 = "?", lines[i], lines[i + 1]
            i += 2
        else:
            # Space-Track 3LE name lines start with "0 " — strip it so the
            # catalogue names match CelesTrak/plot labels ("IRIDIUM 100").
            name = lines[i][2:] if lines[i].startswith("0 ") else lines[i]
            l1, l2 = lines[i + 1], lines[i + 2]
            i += 3
        norad = int(l1[2:7])
        d = abs((tle_epoch(l1) - t_mid).total_seconds())
        if norad not in best or d < best[norad][0]:
            best[norad] = (d, name.strip(), l1, l2)
    return {k: v[1:] for k, v in best.items()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session_dir = os.path.abspath(args.session_dir.rstrip("/"))
    t0, t1, _meta = load_session_window(session_dir)
    t_mid = t0 + (t1 - t0) / 2
    print(f"Session window: {t0.isoformat()} → {t1.isoformat()}")

    # NORAD ids of the current constellation (from cache/CelesTrak)
    cat = load_catalogue()
    norad_ids = [s.model.satnum for s in cat.satellites]

    user, pw = credentials()
    text = fetch_historical(
        norad_ids,
        t_mid - timedelta(days=args.margin_days),
        t_mid + timedelta(days=args.margin_days),
        user, pw,
    )
    best = closest_per_satellite(text, t_mid)
    if len(best) < len(norad_ids) * 0.8:
        print(f"WARNING: elements found for only {len(best)}/{len(norad_ids)} "
              "satellites — consider a larger --margin-days", file=sys.stderr)

    out_path = args.out or os.path.join(session_dir, "iridium_tle.txt")
    ages = []
    with open(out_path, "w", encoding="utf-8") as f:
        for norad in sorted(best):
            name, l1, l2 = best[norad]
            f.write(f"{name}\n{l1}\n{l2}\n")
            ages.append(abs((tle_epoch(l1) - t_mid).total_seconds()) / 86400.0)
    print(f"Wrote {len(best)} satellites → {out_path}")
    print(f"Epoch distance from session midpoint: "
          f"median {sorted(ages)[len(ages) // 2]:.2f} d, max {max(ages):.2f} d")
    print("All evaluation scripts will now use this snapshot automatically.")


if __name__ == "__main__":
    main()
