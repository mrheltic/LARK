"""Tests for core/broadcast_ephemeris.py — ephemeris decoded from the downlink.

The properties worth pinning down are the three that make raw IRA decodes
unusable: the corrupt-position rate, the compressed chunk time axis, and the
4 km position quantisation that rules out finite-difference velocity.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.broadcast_ephemeris import (
    IRA_POS_LSB_KM,
    IbcTime,
    IraEphemeris,
    IraFix,
    estimate_clock_offset,
    filter_plausible,
    fit_short_arc,
    parse_decodes,
    remap_epochs,
    split_arcs,
)

R_ORBIT_KM = 7158.0
MU_EARTH = 398600.4418


def _circular_orbit(t: np.ndarray, incl_deg: float = 86.4) -> np.ndarray:
    """Textbook circular orbit sampled at t [s] -> (N, 3) km."""
    n = np.sqrt(MU_EARTH / R_ORBIT_KM**3)          # mean motion [rad/s]
    th = n * t
    i = np.deg2rad(incl_deg)
    x = R_ORBIT_KM * np.cos(th)
    y = R_ORBIT_KM * np.sin(th) * np.cos(i)
    z = R_ORBIT_KM * np.sin(th) * np.sin(i)
    return np.stack([x, y, z], axis=-1)


def _quantise(pos: np.ndarray) -> np.ndarray:
    """The 4 km-per-axis truncation the IRA position field applies."""
    return np.round(pos / IRA_POS_LSB_KM) * IRA_POS_LSB_KM


# ── short-arc fitting ────────────────────────────────────────────────────────

def test_short_arc_beats_quantisation():
    """Averaging many quantised fixes must land well inside one 4 km LSB."""
    t = np.arange(0.0, 400.0, 1.0)
    truth = _circular_orbit(t)
    arc = fit_short_arc(t, _quantise(truth))
    assert arc is not None
    err = np.linalg.norm(arc.position(t) - truth, axis=1)
    assert np.median(err) < 1.0, f"median position error {np.median(err):.2f} km"


def test_short_arc_velocity_is_usable():
    """Finite differences on quantised fixes are hopeless; the fit is not.

    Two fixes 90 ms apart differ by up to 4 km of quantisation noise alone,
    i.e. ~44 km/s against a true 7.5 km/s. The analytic derivative of the arc
    has to do far better than that.
    """
    t = np.arange(0.0, 400.0, 1.0)
    truth = _circular_orbit(t)
    v_true = np.gradient(truth, t, axis=0)

    arc = fit_short_arc(t, _quantise(truth))
    assert arc is not None
    v_err = np.linalg.norm(arc.velocity(t) - v_true, axis=1)
    assert np.median(v_err) < 0.05, f"median velocity error {np.median(v_err):.3f} km/s"

    # ... and confirm the naive alternative really is unusable.
    naive = np.diff(_quantise(truth), axis=0) / np.diff(t)[:, None]
    assert np.median(np.linalg.norm(naive - v_true[:-1], axis=1)) > 1.0


def test_short_arc_rejects_outliers():
    """A few corrupt positions must not drag the fit."""
    t = np.arange(0.0, 300.0, 2.0)
    pos = _quantise(_circular_orbit(t))
    corrupted = pos.copy()
    corrupted[::11] += 400.0                     # a bad 12-bit word looks like this
    arc = fit_short_arc(t, corrupted)
    assert arc is not None
    assert arc.n_used < t.size                   # something was actually rejected
    err = np.linalg.norm(arc.position(t) - _circular_orbit(t), axis=1)
    assert np.median(err) < 2.0


def test_short_arc_needs_enough_points():
    assert fit_short_arc(np.array([0.0, 1.0]), np.zeros((2, 3))) is None


def test_split_arcs_cuts_on_gaps():
    t = np.array([0.0, 1.0, 2.0, 500.0, 501.0])
    assert split_arcs(t, gap_s=120.0) == [slice(0, 3), slice(3, 5)]


# ── plausibility filter ──────────────────────────────────────────────────────

def test_filter_plausible_drops_corrupt_positions():
    good = IraFix(t=0.0, sat_id=73, beam=1, pos_km=np.array([R_ORBIT_KM, 0.0, 0.0]))
    below = IraFix(t=1.0, sat_id=73, beam=1, pos_km=np.array([6368.0, 0.0, 0.0]))
    above = IraFix(t=2.0, sat_id=73, beam=1, pos_km=np.array([9000.0, 0.0, 0.0]))
    assert filter_plausible([good, below, above]) == [good]


# ── time remapping ───────────────────────────────────────────────────────────

def _index(cpi=64000, fs=1e6, epochs=(1000.0, 1000.5, 1002.0)):
    return {"fs_hz": fs, "cpi_size": cpi,
            "chunks": [{"file": "ant0_000000.cf32", "first_frame": 0,
                        "frames": list(range(len(epochs))),
                        "epochs": list(epochs)}]}


def test_remap_epochs_undoes_time_compression():
    """CPIs are contiguous in the file but not in time; the map must be exact."""
    idx = _index()
    cpi, fs = idx["cpi_size"], idx["fs_hz"]
    # A record 0.25 CPI into the third CPI: chunk time says 2.16 s, truth 1002.016.
    t_chunk = (2 * cpi + 0.25 * cpi) / fs
    rec = IraFix(t=t_chunk, sat_id=73, beam=0, pos_km=np.zeros(3),
                 chunk="ant0_000000")
    (out,) = remap_epochs([rec], idx)
    assert out.t == pytest.approx(1002.0 + 0.25 * cpi / fs)


def test_remap_epochs_drops_unknown_epochs():
    idx = _index(epochs=(1000.0, float("nan")))
    rec = IraFix(t=1.5 * 64000 / 1e6, sat_id=73, beam=0, pos_km=np.zeros(3),
                 chunk="ant0_000000")
    assert remap_epochs([rec], idx) == []


def test_remap_epochs_drops_unknown_chunk():
    rec = IraFix(t=0.0, sat_id=73, beam=0, pos_km=np.zeros(3), chunk="nope")
    assert remap_epochs([rec], _index()) == []


# ── clock offset ─────────────────────────────────────────────────────────────

def test_estimate_clock_offset_is_robust():
    """A late host clock must be recovered despite corrupt IBC time fields."""
    rng = np.random.default_rng(0)
    good = [IbcTime(t_rx=1000.0 + i + 1.15 + rng.normal(0, 0.01),
                    t_iridium=1000.0 + i, sat_id=73) for i in range(20)]
    bad = [IbcTime(t_rx=1000.0, t_iridium=5000.0, sat_id=73)]
    off, mad, n = estimate_clock_offset(good + bad)
    assert off == pytest.approx(1.15, abs=0.05)
    assert mad < 0.05
    assert n == 21


def test_estimate_clock_offset_empty():
    off, mad, n = estimate_clock_offset([])
    assert n == 0 and np.isnan(off) and np.isnan(mad)


# ── parsing ──────────────────────────────────────────────────────────────────

_IRA_LINE = (
    "IRA: u-ant0_002000-e000 000000739.2731 1626282571  99% "
    "-23.65|-082.32|26.55 432 DL sat:073 beam:45 xyz=(+1319,+0047,+1206) "
    "pos=(+42.42/+002.04) alt=796 RAI:48 ?00 bc_sb:20 P01: "
    "PAGE(tmsi:c437cd14 msc_id:17) {OK} FILL=10\n"
)
_IRA_BAD = _IRA_LINE.replace("{OK}", "{ERR}")
_IBC_LINE = (
    "IBC: u-ant0_002000-e000 000006964.7959 1625863333  86% "
    "-32.38|-085.60|20.06 131 DL bc:0 sat:073 cell:33 0 slot:0 sv_blkn:0 "
    "aq_cl:1111111111111111 aq_sb:30 aq_ch:2 00 0000 "
    "time:2026-06-05T09:10:22.91Z [7 1000101] \n"
)


def test_parse_decodes(tmp_path):
    p = tmp_path / "frames.parsed"
    p.write_text(_IRA_LINE + _IRA_BAD + _IBC_LINE + "RAW: junk\n", encoding="utf-8")

    fixes, times = parse_decodes(str(p))
    assert len(fixes) == 1, "the {ERR} frame must be dropped by default"
    fix = fixes[0]
    assert (fix.sat_id, fix.beam, fix.chunk) == (73, 45, "ant0_002000")
    assert fix.t == pytest.approx(0.7392731)
    # xyz is in units of 4 km.
    assert np.allclose(fix.pos_km, [1319 * 4.0, 47 * 4.0, 1206 * 4.0])
    assert 7100 < fix.radius_km < 7250

    assert len(times) == 1
    assert times[0].t_rx == pytest.approx(6.9647959)
    assert times[0].sat_id == 73
    # 2026-06-05T09:10:22.91Z
    assert times[0].t_iridium == pytest.approx(1780650622.91, abs=1e-2)

    assert len(parse_decodes(str(p), require_ok=False)[0]) == 2


# ── the ephemeris source as the solver sees it ───────────────────────────────

def test_ira_ephemeris_states():
    t = np.arange(0.0, 400.0, 1.0) + 1_780_000_000.0
    truth = _circular_orbit(t - t[0])
    fixes = [IraFix(t=float(ti), sat_id=73, beam=0, pos_km=p)
             for ti, p in zip(t, _quantise(truth))]

    eph = IraEphemeris.from_fixes(fixes)
    assert eph.satellites == [73]

    pos, vel = eph.states(73, t)
    assert pos.shape == (t.size, 3) and vel.shape == (t.size, 3)
    assert np.median(np.linalg.norm(pos - truth, axis=1)) < 1.0
    # A circular orbit has constant speed; check we recover it.
    assert np.median(np.linalg.norm(vel, axis=1)) == pytest.approx(
        np.sqrt(MU_EARTH / R_ORBIT_KM), rel=0.02)


def test_ira_ephemeris_separates_passes():
    """Two passes of one satellite must become two arcs, not one bad fit."""
    t1 = np.arange(0.0, 200.0, 1.0)
    t2 = t1 + 6000.0
    fixes = []
    for t in (t1, t2):
        for ti, p in zip(t, _quantise(_circular_orbit(t))):
            fixes.append(IraFix(t=float(ti), sat_id=73, beam=0, pos_km=p))
    eph = IraEphemeris.from_fixes(fixes)
    assert len(eph.arcs[73]) == 2


def test_ira_ephemeris_unknown_satellite():
    eph = IraEphemeris.from_fixes([])
    pos, vel = eph.states(99, [0.0])
    assert np.isnan(pos).all() and np.isnan(vel).all()
