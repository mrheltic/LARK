#!/usr/bin/env python3
"""
KrakenSDR DoA – Configuration Profiles
========================================
Pre-defined parameter sets for common physical test scenarios.

Usage in config.py:
    from profiles import PROFILES, apply_profile
    apply_profile("outdoor_5ant")   # overwrites config.py module-level vars

Usage from CLI:
    python3 pysdr_doa/pysdr_doa_realtime.py --profile outdoor_5ant

Each profile is a dict of config.py attribute names → values.
Only the keys present in a profile are overwritten; everything else
keeps its config.py default.
"""

from __future__ import annotations

# =============================================================================
# Profile definitions
# =============================================================================

PROFILES: dict[str, dict] = {

    # ── Default outdoor 5-antenna UCA ─────────────────────────────────────────
    "outdoor_5ant": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.358,
        "FREQ_HZ":             865.21e6,
        "GAIN_DB":             15,
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.95,
        "ANGLE_SMOOTH_ALPHA":  0.80,
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -60.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },

    # ── Outdoor 5-ant, fast-moving source ─────────────────────────────────────
    "outdoor_5ant_fast": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.358,
        "FREQ_HZ":             865.21e6,
        "GAIN_DB":             15,
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.70,           # less memory → faster response
        "ANGLE_SMOOTH_ALPHA":  0.40,           # more reactive
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -60.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },

    # ── Outdoor 5-ant, high-precision stationary ──────────────────────────────
    "outdoor_5ant_precision": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.358,
        "FREQ_HZ":             865.21e6,
        "GAIN_DB":             15,
        "DOA_ALGORITHM":       "ROOT-MUSIC",   # sub-grid accuracy
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.96,           # heavy smoothing for stationary
        "ANGLE_SMOOTH_ALPHA":  0.90,
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -60.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },

    # ── Indoor 5-ant (heavy multipath) ────────────────────────────────────────
    "indoor_5ant": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.358,
        "FREQ_HZ":             865.21e6,
        "GAIN_DB":             40,             # higher gain indoor
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.92,
        "ANGLE_SMOOTH_ALPHA":  0.75,
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -50.0,         # tighter threshold
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },

    # ── 3-antenna Kerberos UCA ────────────────────────────────────────────────
    "outdoor_3ant": {
        "N_ANTENNAS":          3,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.289,          # 3-ant @ 868 MHz
        "FREQ_HZ":             868.0e6,
        "GAIN_DB":             20,
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.90,
        "ANGLE_SMOOTH_ALPHA":  0.70,
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -60.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0],
    },

    # ── 433 MHz ISM ───────────────────────────────────────────────────────────
    "outdoor_5ant_433": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.178,          # ~12.35 cm / 69.2 cm
        "FREQ_HZ":             433.92e6,
        "GAIN_DB":             20,
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.92,
        "ANGLE_SMOOTH_ALPHA":  0.75,
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -60.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },

    # ── Wi-Fi 2.4 GHz ────────────────────────────────────────────────────────
    "outdoor_5ant_wifi": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.988,          # ~12.35 cm / 12.5 cm
        "FREQ_HZ":             2.437e9,        # ch 6
        "GAIN_DB":             30,
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.90,
        "ANGLE_SMOOTH_ALPHA":  0.70,
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -55.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },

    # ── Benchmark (minimal smoothing for raw performance testing) ─────────────
    "benchmark": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.358,
        "FREQ_HZ":             865.21e6,
        "GAIN_DB":             15,
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.0,            # no EMA → single-frame R
        "ANGLE_SMOOTH_ALPHA":  0.0,            # no angle smoothing
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     False,
        "SQUELCH_THRESHOLD_DB": -80.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },

    # ── Iridium L-band — 5-element CROSS array (apps/space/ only) ──────────────
    # Use with: space_doa_realtime.py, space_collector.py, space_doa_playback.py
    # λ ≈ 18.44 cm @ 1626 MHz.  D_LAMBDA = 0.5 → arm length ≈ 9.2 cm.
    # Cross layout: ant0=center, ant1=E, ant2=N, ant3=W, ant4=S (see doa_algorithms_3d.py).
    # Do NOT apply this profile to doa_runner.py (UCA/ULA only).
    "iridium_1626": {
        "N_ANTENNAS":          5,
        "FREQ_HZ":             1_626_270_000,
        "GAIN_DB":             49,
        "D_LAMBDA":            0.5,            # cross array arm length [λ] ≈ 9.2 cm
        "DOA_ALGORITHM":       "2D-MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.88,           # lighter smoothing — burst TDMA signals
        "ANGLE_SMOOTH_ALPHA":  0.75,
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -60.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },

    # ── ISM 868 MHz – explicit alias for the standard outdoor-5ant profile ─────
    "ism_868": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.358,          # 12.5 cm @ 865 MHz (≈ λ/2.79)
        "FREQ_HZ":             865.21e6,
        "GAIN_DB":             15,
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.95,
        "ANGLE_SMOOTH_ALPHA":  0.80,
        "AMPLITUDE_NORMALIZE": True,
        "SQUELCH_ENABLED":     True,
        "SQUELCH_THRESHOLD_DB": -60.0,
        "PHASE_OFFSETS_DEG":   [0.0, 0.0, 0.0, 0.0, 0.0],
    },
}


# =============================================================================
# Apply a profile to a config module
# =============================================================================

def apply_profile(name: str, target=None) -> None:
    """
    Overwrite config module attributes with values from the named profile.

    Parameters
    ----------
    name : str
        Profile name as defined in PROFILES.
    target : module, optional
        The config module to update.  When omitted the function uses the
        ``config`` module already present in ``sys.modules`` (which, thanks
        to each app's sys.path ordering, is the app-local config.py).

    Notes
    -----
    * Profile keys that do not exist in *target* are silently skipped.
      This lets a generic profile be applied to an app-specific config that
      only defines a subset of the profile's keys.
    * Raises ``KeyError`` for unknown profile names.

    Examples
    --------
    From a script that has already done ``import config as C``::

        from profiles import apply_profile
        apply_profile("ism_868")           # targets sys.modules['config']
        apply_profile("iridium_1626", C)   # explicit target — identical result

    From an app config at import time (LARK_PROFILE env var)::

        import sys as _sys
        from profiles import apply_profile as _ap
        _ap("iridium_1626", _sys.modules[__name__])
    """
    import sys as _sys

    if target is None:
        target = _sys.modules.get("config")
        if target is None:
            import config as _c
            target = _c

    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")

    profile = PROFILES[name]
    applied = 0
    for key, val in profile.items():
        if hasattr(target, key):
            setattr(target, key, val)
            applied += 1

    print(f"[Profile] Applied '{name}' ({applied}/{len(profile)} params)",
          file=_sys.stderr)


def list_profiles() -> None:
    """Print all available profiles."""
    print("Available profiles:")
    for name in sorted(PROFILES.keys()):
        p = PROFILES[name]
        algo = p.get("DOA_ALGORITHM", "?")
        decorr = p.get("DECORRELATION", "?")
        freq = p.get("FREQ_HZ", 0) / 1e6
        n = p.get("N_ANTENNAS", "?")
        print(f"  {name:<28s}  {n}-ant  {algo}/{decorr}  {freq:.2f} MHz")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    list_profiles()
