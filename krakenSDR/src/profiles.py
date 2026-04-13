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

    # ── Iridium L-band (KrakenSDR 5-ant, 1626.270 MHz) ────────────────────────
    # λ ≈ 18.44 cm at 1626 MHz.  If using the standard KrakenSDR 12.5 cm-radius
    # ring the spacing/λ ≈ 0.68 — grating lobes possible; tune RADIUS_LAMBDA to
    # your actual array radius in wavelengths.
    "iridium_1626": {
        "N_ANTENNAS":          5,
        "GEOMETRY":            "UCA",
        "RADIUS_LAMBDA":       0.679,          # 12.5 cm @ 1626 MHz (≈ λ/1.47)
        "FREQ_HZ":             1_626_270_000,
        "GAIN_DB":             15,
        "DOA_ALGORITHM":       "MUSIC",
        "DECORRELATION":       "FBA",
        "COV_ALPHA":           0.88,           # lighter smoothing — burst signals
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
# Apply a profile to the config module
# =============================================================================

def apply_profile(name: str) -> None:
    """
    Overwrite config.py module attributes with values from the named profile.

    Raises KeyError if profile name is not found.
    """
    import config as C

    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")

    profile = PROFILES[name]
    for key, val in profile.items():
        if not hasattr(C, key):
            raise AttributeError(
                f"Profile '{name}' sets '{key}' but config.py has no such attribute"
            )
        setattr(C, key, val)

    print(f"[Profile] Applied '{name}' ({len(profile)} params)")


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
