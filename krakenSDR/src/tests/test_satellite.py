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


# ══════════════════════════════════════════════════════════════════════════════
# shared.geo_utils
# ══════════════════════════════════════════════════════════════════════════════

class TestGeoUtils:
    def test_angular_distance_canonical_import(self):
        from shared.geo_utils import angular_distance_deg
        assert angular_distance_deg(0, 90, 180, 90) == pytest.approx(0.0, abs=0.1)
        assert angular_distance_deg(0, 0, 90, 0) == pytest.approx(90.0, abs=0.5)

    def test_unit_vec_roundtrip(self):
        from shared.geo_utils import az_el_to_unit_vec, unit_vec_to_az_el
        import numpy as np
        for az, el in [(0, 30), (90, 45), (270, 10), (180, 80)]:
            v = az_el_to_unit_vec(az, el)
            az2, el2 = unit_vec_to_az_el(v)
            assert abs(az2 - az) < 0.1 or abs(abs(az2 - az) - 360) < 0.1
            assert abs(el2 - el) < 0.1


# ══════════════════════════════════════════════════════════════════════════════
# core.signal_quality — channel power balance
# ══════════════════════════════════════════════════════════════════════════════

class TestSignalQuality:
    def test_balanced_array(self):
        import numpy as np
        from core.signal_quality import channel_power_balance
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 1024)) + 1j * rng.standard_normal((5, 1024))
        rpt = channel_power_balance(X)
        assert rpt.is_balanced
        assert not rpt.has_critical
        assert rpt.imbalance_db < 6.0    # white noise ≈ equal power

    def test_weak_channel_detected(self):
        import numpy as np
        from core.signal_quality import channel_power_balance
        rng = np.random.default_rng(1)
        X = rng.standard_normal((5, 1024)) + 1j * rng.standard_normal((5, 1024))
        X[2] *= 0.1    # channel 2 at 1% power — should be critical
        rpt = channel_power_balance(X)
        assert not rpt.is_balanced
        assert rpt.has_critical
        assert rpt.weak_mask[2]

    def test_channel_order_labels(self):
        import numpy as np
        from core.signal_quality import channel_power_balance
        rng = np.random.default_rng(2)
        X = rng.standard_normal((5, 512)) + 1j * rng.standard_normal((5, 512))
        X[1] *= 0.05   # north arm critically weak
        order = ("center", "north", "east", "south", "west")
        rpt = channel_power_balance(X, channel_order=order)
        assert "north" in rpt.critical_labels


# ══════════════════════════════════════════════════════════════════════════════
# core.doa_algorithms_3d — IAA-2D
# ══════════════════════════════════════════════════════════════════════════════

class TestIAA2D:
    def _make_signal(self, az_deg: float, el_deg: float, snr_db: float = 15.0):
        import numpy as np
        cfg = __import__("core.doa_algorithms_3d", fromlist=["CrossArrayConfig",
                         "doa_iaa_2d", "find_peak_2d"])
        CrossArrayConfig = cfg.CrossArrayConfig
        doa_iaa_2d = cfg.doa_iaa_2d
        find_peak_2d = cfg.find_peak_2d
        return CrossArrayConfig, doa_iaa_2d, find_peak_2d

    def test_iaa_single_source_peak(self):
        import numpy as np
        from core.doa_algorithms_3d import CrossArrayConfig, doa_iaa_2d, find_peak_2d
        cfg = CrossArrayConfig(d_lambda=0.5, n_az=72, n_el=18, el_min_deg=5.0,
                               num_expected_signals=1)
        target_az, target_el = 60.0, 40.0

        az_vals = cfg.az_range_deg()
        el_vals = cfg.el_range_deg()
        i_az = int(np.argmin(np.abs(az_vals - target_az)))
        i_el = int(np.argmin(np.abs(el_vals - target_el)))
        a = cfg.get_steering_matrix()[:, i_el * cfg.n_az + i_az]

        rng = np.random.default_rng(42)
        N = 4096
        sig = np.exp(1j * rng.uniform(0, 2 * np.pi, N))
        noise = 0.05 * (rng.standard_normal((5, N)) + 1j * rng.standard_normal((5, N)))
        X = a[:, None] * sig[None, :] + noise

        spec = doa_iaa_2d(X, cfg)
        az_hat, el_hat, papr = find_peak_2d(spec, cfg)

        def circ_err(a, b): return abs((a - b + 180) % 360 - 180)
        assert circ_err(az_hat, az_vals[i_az]) <= 10.0
        assert abs(el_hat - el_vals[i_el]) <= 10.0
        assert papr > 3.0  # clear peak

