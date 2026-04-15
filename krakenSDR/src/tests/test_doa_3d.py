import numpy as np

from core.doa_algorithms_3d import (
    CROSS_ARRAY_CANONICAL_ORDER,
    CrossArrayConfig,
    doa_music_2d,
    find_peak_2d,
    reorder_cross_array_channels,
)


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