import numpy as np
import pytest

from core.doa_algorithms_3d import (
    CROSS_ARRAY_CANONICAL_ORDER,
    CrossArrayConfig,
    doa_music_2d,
    doa_bartlett_2d,
    doa_capon_2d,
    doa_iaa_2d,
    find_peak_2d,
    reorder_cross_array_channels,
    estimate_signal_count,
)


def _make_single_source(az_deg: float, el_deg: float, n_samples: int = 4096,
                        noise_std: float = 0.05, seed: int = 0):
    """Synthesize a rank-1 cross-array signal from a point source at (az, el)."""
    cfg = CrossArrayConfig(d_lambda=0.5, n_az=181, n_el=51, el_min_deg=5.0)
    az_v = cfg.az_range_deg()
    el_v = cfg.el_range_deg()
    i_az = int(np.argmin(np.abs(az_v - az_deg)))
    i_el = int(np.argmin(np.abs(el_v - el_deg)))
    a = cfg.get_steering_matrix()[:, i_el * cfg.n_az + i_az]
    rng = np.random.default_rng(seed)
    sig = np.exp(1j * rng.uniform(0, 2 * np.pi, n_samples))
    noise = noise_std * (rng.standard_normal((5, n_samples))
                        + 1j * rng.standard_normal((5, n_samples)))
    return a[:, None] * sig[None, :] + noise, cfg, az_v[i_az], el_v[i_el]


def _circ_err_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def test_reorder_cross_array_channels_round_trip() -> None:
    x = np.arange(5 * 8, dtype=np.float64).reshape(5, 8)
    physical_order = ("center", "north", "east", "south", "west")

    physical = reorder_cross_array_channels(
        x,
        CROSS_ARRAY_CANONICAL_ORDER,
        output_order=physical_order,
    )
    recovered = reorder_cross_array_channels(physical, physical_order)

    assert np.array_equal(recovered, x)


def test_doa_music_2d_with_physical_channel_order_matches_true_direction() -> None:
    cfg = CrossArrayConfig(d_lambda=0.5, n_az=181, n_el=51, el_min_deg=10.0)
    target_az = 35.0
    target_el = 55.0

    az_vals = cfg.az_range_deg()
    el_vals = cfg.el_range_deg()
    i_az = int(np.argmin(np.abs(az_vals - target_az)))
    i_el = int(np.argmin(np.abs(el_vals - target_el)))
    grid_idx = i_el * cfg.n_az + i_az

    steering = cfg.get_steering_matrix()[:, grid_idx]
    rng = np.random.default_rng(1234)
    signal = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=4096))
    noise = 0.03 * (
        rng.standard_normal((5, signal.size)) + 1j * rng.standard_normal((5, signal.size))
    )
    x_canonical = steering[:, np.newaxis] * signal[np.newaxis, :] + noise

    physical_order = ("center", "north", "east", "south", "west")
    x_physical = reorder_cross_array_channels(
        x_canonical,
        CROSS_ARRAY_CANONICAL_ORDER,
        output_order=physical_order,
    )

    x_fixed = reorder_cross_array_channels(x_physical, physical_order)
    spec = doa_music_2d(x_fixed, cfg)
    az_hat, el_hat, _ = find_peak_2d(spec, cfg)

    assert _circ_err_deg(az_hat, az_vals[i_az]) <= 5.0
    assert abs(el_hat - el_vals[i_el]) <= 5.0


def test_wrong_channel_order_causes_large_azimuth_error() -> None:
    cfg = CrossArrayConfig(d_lambda=0.5, n_az=181, n_el=51, el_min_deg=10.0)
    target_az = 35.0
    target_el = 55.0

    az_vals = cfg.az_range_deg()
    el_vals = cfg.el_range_deg()
    i_az = int(np.argmin(np.abs(az_vals - target_az)))
    i_el = int(np.argmin(np.abs(el_vals - target_el)))
    grid_idx = i_el * cfg.n_az + i_az

    steering = cfg.get_steering_matrix()[:, grid_idx]
    rng = np.random.default_rng(5678)
    signal = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=4096))
    noise = 0.03 * (
        rng.standard_normal((5, signal.size)) + 1j * rng.standard_normal((5, signal.size))
    )
    x_canonical = steering[:, np.newaxis] * signal[np.newaxis, :] + noise

    physical_order = ("center", "north", "east", "south", "west")
    x_physical = reorder_cross_array_channels(
        x_canonical,
        CROSS_ARRAY_CANONICAL_ORDER,
        output_order=physical_order,
    )

    spec_wrong = doa_music_2d(x_physical, cfg)
    az_wrong, _, _ = find_peak_2d(spec_wrong, cfg)

    assert _circ_err_deg(az_wrong, az_vals[i_az]) >= 15.0


# ═══════════════════════════════════════════════════════════════════════════
# MDL signal-count estimator
# ═══════════════════════════════════════════════════════════════════════════

def test_mdl_one_signal():
    """MDL correctly identifies D=1 from a synthetic single-source covariance."""
    rng = np.random.default_rng(42)
    M, N = 5, 10_000
    a = np.array([1.0, np.exp(1j*0.5), np.exp(1j*1.0), np.exp(-1j*0.5), np.exp(-1j*1.0)])
    s = np.exp(1j * 2 * np.pi * 0.1 * np.arange(N))
    X = np.outer(a, s) + 0.1 * (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N)))
    R = (X @ X.conj().T) / N
    assert estimate_signal_count(R, N, method="mdl") == 1


def test_mdl_two_signals():
    """MDL correctly identifies D=2 from two uncorrelated sources."""
    rng = np.random.default_rng(123)
    M, N = 5, 10_000
    a1 = np.array([1.0, np.exp(1j*0.5), np.exp(1j*1.0), np.exp(-1j*0.5), np.exp(-1j*1.0)])
    a2 = np.array([1.0, np.exp(1j*1.5), np.exp(1j*0.3), np.exp(-1j*1.5), np.exp(-1j*0.3)])
    s1 = np.exp(1j * 2 * np.pi * 0.1 * np.arange(N))
    s2 = np.exp(1j * 2 * np.pi * 0.3 * np.arange(N))
    X = np.outer(a1, s1) + 0.5 * np.outer(a2, s2) + 0.1 * (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N)))
    R = (X @ X.conj().T) / N
    assert estimate_signal_count(R, N, method="mdl") == 2


def test_mdl_noise_only():
    """MDL returns 0 when input is pure noise."""
    rng = np.random.default_rng(77)
    M, N = 5, 10_000
    X = 0.1 * (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N)))
    R = (X @ X.conj().T) / N
    assert estimate_signal_count(R, N, method="mdl") == 0


# ═══════════════════════════════════════════════════════════════════════════
# doa_music_2d — n_snapshots parameter (fix for _N_snap magic number)
# ═══════════════════════════════════════════════════════════════════════════

def test_doa_music_2d_n_snapshots_from_X():
    """When only X is provided, n_snapshots must be taken from X.shape[1]."""
    X, cfg, az_true, el_true = _make_single_source(45.0, 30.0, n_samples=10_690)
    cfg_auto = CrossArrayConfig(
        d_lambda=0.5, n_az=181, n_el=51, el_min_deg=5.0,
        num_expected_signals=0,   # triggers MDL auto-detection
    )
    # Must not raise; must use X.shape[1]=10690 for the MDL n_snapshots
    spec = doa_music_2d(X, cfg_auto)
    assert spec.shape == (cfg_auto.n_el, cfg_auto.n_az)
    az_hat, el_hat, _ = find_peak_2d(spec, cfg_auto)
    def circ_err(a, b): return abs((a - b + 180) % 360 - 180)
    assert circ_err(az_hat, az_true) <= 10.0


def test_doa_music_2d_explicit_n_snapshots_matches_implicit():
    """Explicit n_snapshots=10690 must produce the same result as implicit."""
    X, cfg, _, _ = _make_single_source(90.0, 55.0, n_samples=10_690, seed=99)
    cfg_auto = CrossArrayConfig(d_lambda=0.5, n_az=180, n_el=36, el_min_deg=5.0,
                                num_expected_signals=0)
    spec_implicit = doa_music_2d(X, cfg_auto)
    spec_explicit = doa_music_2d(X, cfg_auto, n_snapshots=10_690)
    np.testing.assert_array_equal(spec_implicit, spec_explicit)


# ═══════════════════════════════════════════════════════════════════════════
# doa_iaa_2d
# ═══════════════════════════════════════════════════════════════════════════

def test_doa_iaa_2d_peak_near_true_direction():
    """IAA must locate the dominant peak within 10° of the true direction."""
    X, cfg_full, az_true, el_true = _make_single_source(120.0, 35.0, n_samples=4096)
    cfg = CrossArrayConfig(d_lambda=0.5, n_az=72, n_el=18, el_min_deg=5.0)
    spec = doa_iaa_2d(X, cfg)
    az_hat, el_hat, papr = find_peak_2d(spec, cfg)
    def circ_err(a, b): return abs((a - b + 180) % 360 - 180)
    assert circ_err(az_hat, az_true) <= 15.0
    assert abs(el_hat - el_true) <= 15.0
    assert papr > 2.0


def test_doa_iaa_2d_accepts_R_in():
    """IAA must work identically whether X or R_in is provided."""
    X, cfg_full, _, _ = _make_single_source(200.0, 50.0, n_samples=4096, seed=7)
    cfg = CrossArrayConfig(d_lambda=0.5, n_az=72, n_el=18)
    R = (X @ X.conj().T) / X.shape[1]
    spec_from_X = doa_iaa_2d(X, cfg)
    spec_from_R = doa_iaa_2d(X, cfg, R_in=R)  # same R, so spectra must match
    np.testing.assert_allclose(spec_from_X, spec_from_R, atol=1e-6)


def test_doa_iaa_2d_output_shape():
    """IAA output shape must be (n_el, n_az)."""
    X, cfg_full, _, _ = _make_single_source(45.0, 45.0, n_samples=2048, seed=3)
    cfg = CrossArrayConfig(d_lambda=0.5, n_az=36, n_el=9, el_min_deg=5.0)
    spec = doa_iaa_2d(X, cfg, n_iter=5)
    assert spec.shape == (9, 36)
    assert float(np.max(spec)) == pytest.approx(0.0, abs=0.01)  # peak normalized to 0dB
    assert float(np.min(spec)) >= -40.0 - 1e-6


# =============================================================================
# doa_bartlett_2d
# =============================================================================

def test_doa_bartlett_2d_peak_near_true_direction():
    """Bartlett peak must fall within 10° of the synthetic source."""
    true_az, true_el = 90.0, 30.0
    X, cfg, i_az_true, i_el_true = _make_single_source(true_az, true_el, n_samples=8192, seed=10)
    spec = doa_bartlett_2d(X, cfg)
    assert spec.shape == (cfg.n_el, cfg.n_az)
    peak_el_idx, peak_az_idx = np.unravel_index(np.argmax(spec), spec.shape)
    az_est = cfg.az_range_deg()[peak_az_idx]
    el_est = cfg.el_range_deg()[peak_el_idx]
    assert abs(az_est - true_az) < 10.0, f"az error: {az_est:.1f}° vs {true_az}°"
    assert abs(el_est - true_el) < 10.0, f"el error: {el_est:.1f}° vs {true_el}°"


def test_doa_bartlett_2d_output_shape():
    """Bartlett output shape must be (n_el, n_az)."""
    X, cfg_full, _, _ = _make_single_source(45.0, 45.0, n_samples=2048, seed=11)
    cfg = CrossArrayConfig(d_lambda=0.5, n_az=36, n_el=9, el_min_deg=5.0)
    spec = doa_bartlett_2d(X, cfg)
    assert spec.shape == (9, 36)
    assert float(np.max(spec)) == pytest.approx(0.0, abs=0.01)
    assert float(np.min(spec)) >= -40.0 - 1e-6


def test_doa_bartlett_2d_accepts_R_in():
    """Passing R explicitly must yield the same result as computing from X."""
    X, cfg, _, _ = _make_single_source(270.0, 20.0, n_samples=4096, seed=12)
    R = (X @ X.conj().T) / X.shape[1]
    spec_x = doa_bartlett_2d(X, cfg)
    spec_r = doa_bartlett_2d(X, cfg, R_in=R)
    np.testing.assert_allclose(spec_x, spec_r, atol=1e-6)


def test_doa_bartlett_2d_broadside_direction():
    """Bartlett must resolve a source directly overhead (el=85°→zenith)."""
    true_az, true_el = 0.0, 85.0
    X, cfg, _, _ = _make_single_source(true_az, true_el, n_samples=8192, seed=13,
                                        noise_std=0.01)
    spec = doa_bartlett_2d(X, cfg)
    peak_el_idx, peak_az_idx = np.unravel_index(np.argmax(spec), spec.shape)
    el_est = cfg.el_range_deg()[peak_el_idx]
    assert abs(el_est - true_el) < 5.0, f"el error at zenith: {el_est:.1f}°"