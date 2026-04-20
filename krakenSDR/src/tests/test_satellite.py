"""Tests for shared.observer and shared.iridium_tle modules."""
import json, os, tempfile, pytest
from datetime import datetime, timezone


# ══════════════════════════════════════════════════════════════════════════════
# shared.observer
# ══════════════════════════════════════════════════════════════════════════════

class TestObserver:
    def test_explicit_coords(self):
        from shared.observer import get_observer
        lat, lon, alt = get_observer(lat=45.0, lon=7.5, alt=150.0)
        assert lat == 45.0
        assert lon == 7.5
        assert alt == 150.0

    def test_from_meta(self):
        from shared.observer import get_observer
        meta = {"observer_lat": 48.8566, "observer_lon": 2.3522, "observer_alt_m": 35.0}
        lat, lon, alt = get_observer(meta=meta, interactive=False)
        assert lat == pytest.approx(48.8566)
        assert lon == pytest.approx(2.3522)
        assert alt == pytest.approx(35.0)

    def test_invalid_lat_raises(self):
        from shared.observer import get_observer
        with pytest.raises(ValueError):
            get_observer(lat=91.0, lon=0.0)

    def test_invalid_lon_raises(self):
        from shared.observer import get_observer
        with pytest.raises(ValueError):
            get_observer(lat=0.0, lon=181.0)

    def test_save_to_meta(self):
        from shared.observer import save_to_meta
        meta = {}
        save_to_meta(meta, 45.0, 7.5, 100.0)
        assert meta["observer_lat"] == 45.0
        assert meta["observer_lon"] == 7.5
        assert meta["observer_alt_m"] == 100.0

    def test_cache_roundtrip(self):
        from shared.observer import _save_cache, _from_cache
        _save_cache(43.5, 7.1, 50.0)
        loc = _from_cache()
        assert loc is not None
        assert loc[0] == pytest.approx(43.5, abs=0.01)
        assert loc[1] == pytest.approx(7.1, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# shared.iridium_tle
# ══════════════════════════════════════════════════════════════════════════════

# Sample 3-line TLE for ISS (not Iridium, but tests parsing)
_SAMPLE_TLE = """\
IRIDIUM 106
1 41917U 17003A   26110.50000000  .00000084  00000-0  24447-4 0  9991
2 41917  86.3921 115.2000 0002100  90.0000 270.0000 14.34217000000010
IRIDIUM 112
1 41920U 17003D   26110.50000000  .00000089  00000-0  25800-4 0  9992
2 41920  86.3920 115.2100 0002200  91.0000 269.0000 14.34217500000012
"""


class TestIridiumTLE:
    def test_parse_tle(self):
        from shared.iridium_tle import IridiumCatalogue
        cat = IridiumCatalogue(_SAMPLE_TLE)
        assert len(cat) == 2
        assert cat.satellites[0].name == "IRIDIUM 106"
        assert cat.satellites[1].name == "IRIDIUM 112"

    def test_sat_azel(self):
        from shared.iridium_tle import IridiumCatalogue
        cat = IridiumCatalogue(_SAMPLE_TLE)
        t = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        az, el, rng = cat.sat_azel("IRIDIUM 106", 45.0, 7.0, 0.0, t)
        assert -180.0 <= az <= 360.0
        assert -90.0 <= el <= 90.0
        assert rng > 0.0

    def test_visible_now(self):
        from shared.iridium_tle import IridiumCatalogue
        cat = IridiumCatalogue(_SAMPLE_TLE)
        t = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        vis = cat.visible_now(45.0, 7.0, 0.0, t, el_min_deg=-90)
        # With el_min=-90, both should always appear (below horizon is OK)
        assert isinstance(vis, list)
        for s in vis:
            assert "name" in s
            assert "az_deg" in s
            assert "doppler_hz" in s

    def test_sat_track(self):
        from shared.iridium_tle import IridiumCatalogue
        cat = IridiumCatalogue(_SAMPLE_TLE)
        t0 = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 4, 20, 12, 10, 0, tzinfo=timezone.utc)
        track = cat.sat_track("IRIDIUM 106", 45.0, 7.0, 0.0, t0, t1, step_s=60)
        assert "az_deg" in track
        assert "el_deg" in track
        assert len(track["az_deg"]) >= 2

    def test_predict_passes(self):
        from shared.iridium_tle import IridiumCatalogue
        cat = IridiumCatalogue(_SAMPLE_TLE)
        t0 = datetime(2026, 4, 20, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 4, 20, 23, 59, 59, tzinfo=timezone.utc)
        passes = cat.predict_passes(45.0, 7.0, 0.0, t0, t1, el_min_deg=5.0)
        assert isinstance(passes, list)
        for p in passes:
            assert "name" in p
            assert "max_el_deg" in p
            assert "duration_s" in p

    def test_doppler_track(self):
        from shared.iridium_tle import IridiumCatalogue
        cat = IridiumCatalogue(_SAMPLE_TLE)
        t0 = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 4, 20, 12, 10, 0, tzinfo=timezone.utc)
        dop = cat.doppler_track("IRIDIUM 106", 45.0, 7.0, 0.0, t0, t1, step_s=60)
        assert "doppler_hz" in dop
        assert len(dop["doppler_hz"]) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# iridium_pass_predict helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestPassPredict:
    @staticmethod
    def _import():
        import sys, os
        _p = os.path.join(os.path.dirname(__file__), "..", "apps", "space")
        if _p not in sys.path:
            sys.path.insert(0, _p)
        from iridium_pass_predict import _angular_distance
        return _angular_distance

    def test_angular_distance_zero(self):
        _angular_distance = self._import()
        assert _angular_distance(0, 90, 180, 90) == pytest.approx(0.0, abs=0.1)  # zenith

    def test_angular_distance_90(self):
        _angular_distance = self._import()
        d = _angular_distance(0, 0, 90, 0)
        assert d == pytest.approx(90.0, abs=0.5)

    def test_angular_distance_opposite(self):
        _angular_distance = self._import()
        d = _angular_distance(0, 0, 180, 0)
        assert d == pytest.approx(180.0, abs=0.5)
