"""
tests/test_calibration.py
=========================
Unit tests for the phase-difference → (az, el) NN calibration pipeline.

Tests cover:
  - Feature extraction geometry (phase differences match steering vectors)
  - Target encoding/decoding round-trip
  - Model forward pass shape and range
  - Training convergence on synthetic data
  - Loss function properties
  - Serialization round-trip
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from core.calibration_features import (
    phase_diff_features,
    coherence_features,
    eigenvalue_spread_feature,
    extract_feature_vector,
    encode_target,
    decode_target,
)
from core.calibration_model import (
    CalibrationMLP,
    TrainConfig,
    train as train_model,
    angular_loss,
    evaluate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhaseFeatures:
    def test_shape(self):
        R = np.eye(5, dtype=complex)
        feat = phase_diff_features(R)
        assert feat.shape == (8,)
        assert feat.dtype == np.float32

    def test_identity_covariance(self):
        """Identity R → all phases are 0 → cos=1, sin=0."""
        R = np.eye(5, dtype=complex)
        feat = phase_diff_features(R)
        # cos should be ~1.0, sin should be ~0.0
        np.testing.assert_allclose(feat[0::2], 1.0, atol=1e-6)
        np.testing.assert_allclose(feat[1::2], 0.0, atol=1e-6)

    def test_known_phase(self):
        """Inject known phase π/4 between center and all arms."""
        R = np.eye(5, dtype=complex)
        phase = np.pi / 4
        for k in range(1, 5):
            R[0, k] = np.exp(1j * phase)
            R[k, 0] = np.exp(-1j * phase)
        feat = phase_diff_features(R)
        expected_cos = np.cos(phase)
        expected_sin = np.sin(phase)
        np.testing.assert_allclose(feat[0::2], expected_cos, atol=1e-6)
        np.testing.assert_allclose(feat[1::2], expected_sin, atol=1e-6)

    def test_steering_vector_consistency(self):
        """Phase features from a steering vector outer product match geometry."""
        from core.doa_algorithms_3d import CrossArrayConfig

        cfg = CrossArrayConfig(d_lambda=0.5, n_az=72, n_el=18, el_min_deg=5.0)
        A = cfg.get_steering_matrix()

        # Pick a direction: az=45°, el=60° → find nearest grid point
        az_vals = cfg.az_range_deg()
        el_vals = cfg.el_range_deg()
        i_az = int(np.argmin(np.abs(az_vals - 45.0)))
        i_el = int(np.argmin(np.abs(el_vals - 60.0)))
        grid_idx = i_el * cfg.n_az + i_az

        a = A[:, grid_idx]  # (5,) steering vector
        R = np.outer(a, a.conj())  # rank-1 covariance (no noise)

        feat = phase_diff_features(R)

        # ∠R_{0,k} = ∠(a_0 · a_k*) = ∠a_0 - ∠a_k  (not a_k - a_0)
        expected_phases = np.angle(R[0, 1:5])
        np.testing.assert_allclose(feat[0::2], np.cos(expected_phases), atol=1e-5)
        np.testing.assert_allclose(feat[1::2], np.sin(expected_phases), atol=1e-5)


class TestCoherenceFeatures:
    def test_shape(self):
        R = np.eye(5, dtype=complex)
        coh = coherence_features(R)
        assert coh.shape == (4,)

    def test_identity_gives_zero_coherence(self):
        """Identity R → off-diagonal = 0 → coherence = 0."""
        R = np.eye(5, dtype=complex)
        coh = coherence_features(R)
        np.testing.assert_allclose(coh, 0.0, atol=1e-6)

    def test_rank1_gives_unit_coherence(self):
        """Rank-1 R (pure signal) → all coherences = 1."""
        a = np.array([1, 1j, -1, -1j, 0.5 + 0.5j], dtype=complex)
        R = np.outer(a, a.conj())
        coh = coherence_features(R)
        np.testing.assert_allclose(coh, 1.0, atol=1e-5)


class TestEigSpreadFeature:
    def test_identity_spread_zero(self):
        """Identity R → all eigenvalues equal → spread = 0 dB."""
        R = np.eye(5, dtype=complex)
        spread = eigenvalue_spread_feature(R)
        assert abs(spread) < 1e-3

    def test_signal_plus_noise(self):
        """Signal + noise → large eigenvalue spread."""
        a = np.array([1, 1j, -1, -1j, 0.5], dtype=complex)
        R = 10.0 * np.outer(a, a.conj()) + 0.1 * np.eye(5)
        spread = eigenvalue_spread_feature(R)
        assert spread > 10.0  # should be ~20 dB


class TestFeatureVector:
    def test_shape(self):
        R = np.eye(5, dtype=complex)
        feat = extract_feature_vector(R, doppler_hz=5000.0)
        assert feat.shape == (14,)
        assert feat.dtype == np.float32

    def test_doppler_normalization(self):
        R = np.eye(5, dtype=complex)
        feat = extract_feature_vector(R, doppler_hz=20_000.0, doppler_scale=40_000.0)
        assert abs(feat[13] - 0.5) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Target encoding tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTargetEncoding:
    @pytest.mark.parametrize("az,el", [
        (0, 45), (90, 30), (180, 60), (270, 10), (359, 85),
        (45.5, 67.3), (0, 0), (0, 90),
    ])
    def test_round_trip(self, az, el):
        encoded = encode_target(az, el)
        assert encoded.shape == (3,)
        az_dec, el_dec = decode_target(encoded)
        # Azimuth within 0.1° (atan2 precision)
        assert abs((az_dec - az + 180) % 360 - 180) < 0.1
        assert abs(el_dec - el) < 0.1

    def test_batch_decode(self):
        azimuths = [0, 90, 180, 270]
        elevations = [30, 45, 60, 75]
        Y = np.array([encode_target(a, e) for a, e in zip(azimuths, elevations)])
        az_dec, el_dec = decode_target(Y)
        assert az_dec.shape == (4,)
        for i in range(4):
            assert abs((az_dec[i] - azimuths[i] + 180) % 360 - 180) < 0.1
            assert abs(el_dec[i] - elevations[i]) < 0.1

    def test_azimuth_wrap(self):
        """0° and 360° should produce the same encoding."""
        e0 = encode_target(0.0, 45.0)
        e360 = encode_target(360.0, 45.0)
        np.testing.assert_allclose(e0, e360, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Model tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrationMLP:
    def test_init(self):
        model = CalibrationMLP()
        assert model.n_params() > 0
        assert len(model.weights) == 4  # 4 layer transitions
        assert model.weights[0].shape == (14, 64)
        assert model.weights[-1].shape == (32, 3)

    def test_forward_shape(self):
        model = CalibrationMLP()
        X = np.random.randn(10, 14).astype(np.float32)
        Y = model.forward(X)
        assert Y.shape == (10, 3)

    def test_output_range(self):
        """cos²+sin² ≈ 1, elevation in [0, 1]."""
        model = CalibrationMLP()
        X = np.random.randn(100, 14).astype(np.float32)
        Y = model.forward(X)
        # Unit circle constraint
        norm = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-3)
        # Elevation in [0, 1]
        assert np.all(Y[:, 2] >= 0.0)
        assert np.all(Y[:, 2] <= 1.0)

    def test_predict_output(self):
        model = CalibrationMLP()
        X = np.random.randn(5, 14).astype(np.float32)
        az, el = model.predict(X)
        assert az.shape == (5,)
        assert el.shape == (5,)
        assert np.all(az >= 0) and np.all(az < 360)
        assert np.all(el >= 0) and np.all(el <= 90)

    def test_save_load_roundtrip(self):
        model = CalibrationMLP()
        model.feat_mean = np.random.randn(14).astype(np.float32)
        model.feat_std = np.abs(np.random.randn(14)).astype(np.float32) + 0.1
        model.train_history = {"best_epoch": 42, "best_val_loss": 0.01}

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            model.save(f.name)
            loaded = CalibrationMLP.load(f.name)

        assert loaded.n_params() == model.n_params()
        for W_orig, W_load in zip(model.weights, loaded.weights):
            np.testing.assert_array_equal(W_orig, W_load)
        np.testing.assert_array_equal(model.feat_mean, loaded.feat_mean)
        assert loaded.train_history["best_epoch"] == 42


# ═══════════════════════════════════════════════════════════════════════════════
# Loss function tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAngularLoss:
    def test_perfect_match_zero_loss(self):
        Y = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
        loss = angular_loss(Y, Y)
        assert abs(loss) < 1e-6

    def test_opposite_azimuth_max_loss(self):
        Y_pred = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
        Y_true = np.array([[-1.0, 0.0, 0.5]], dtype=np.float32)
        loss = angular_loss(Y_pred, Y_true)
        assert loss > 1.5  # az loss ≈ 2.0

    def test_loss_symmetric(self):
        rng = np.random.default_rng(123)
        Y1 = rng.standard_normal((20, 3)).astype(np.float32)
        Y2 = rng.standard_normal((20, 3)).astype(np.float32)
        # Not perfectly symmetric due to MSE on elevation, but az part is
        loss12 = angular_loss(Y1, Y2)
        loss21 = angular_loss(Y2, Y1)
        # Elevation MSE is symmetric
        assert abs(loss12 - loss21) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Training test (synthetic data)
# ═══════════════════════════════════════════════════════════════════════════════

def test_training_converges_on_synthetic():
    """Train on synthetic steering-vector data and verify convergence.

    We generate (phase_diffs → az, el) pairs from the known cross-array
    geometry and verify the NN can learn the mapping.
    """
    from core.doa_algorithms_3d import CrossArrayConfig

    cfg = CrossArrayConfig(d_lambda=0.5, n_az=72, n_el=18, el_min_deg=10.0)
    A = cfg.get_steering_matrix()  # (5, n_el*n_az)
    az_vals = cfg.az_range_deg()
    el_vals = cfg.el_range_deg()

    rng = np.random.default_rng(999)
    n_samples = 500

    X_list = []
    Y_list = []

    for _ in range(n_samples):
        # Random direction
        az = rng.uniform(0, 360)
        el = rng.uniform(15, 85)

        # Nearest grid point
        i_az = int(np.argmin(np.abs(az_vals - az)))
        i_el = int(np.argmin(np.abs(el_vals - el)))
        grid_az = az_vals[i_az]
        grid_el = el_vals[i_el]
        grid_idx = i_el * cfg.n_az + i_az

        # Steering vector
        a = A[:, grid_idx]

        # Simulate noisy signal: rank-1 + noise
        N_samp = 1000
        s = np.exp(1j * rng.uniform(0, 2 * np.pi, N_samp))
        X_sig = np.outer(a, s)
        noise = 0.15 * (rng.standard_normal((5, N_samp)) +
                        1j * rng.standard_normal((5, N_samp)))
        X_iq = X_sig + noise
        R = (X_iq @ X_iq.conj().T) / N_samp

        # FBA
        J = np.fliplr(np.eye(5))
        R = 0.5 * (R + J @ R.conj() @ J)

        feat = extract_feature_vector(R, doppler_hz=rng.uniform(-30000, 30000))
        target = encode_target(grid_az, grid_el)

        X_list.append(feat)
        Y_list.append(target)

    X_all = np.array(X_list, dtype=np.float32)
    Y_all = np.array(Y_list, dtype=np.float32)

    # Train
    model = CalibrationMLP()
    tcfg = TrainConfig(
        epochs=150, lr=2e-3, batch_size=32,
        patience=40, verbose=False, seed=42,
    )
    history = train_model(model, X_all, Y_all, tcfg)

    # Verify convergence: val loss should decrease significantly
    assert history["best_val_loss"] < history["val_loss"][0] * 0.3, \
        f"Expected significant convergence: initial={history['val_loss'][0]:.4f}, " \
        f"best={history['best_val_loss']:.4f}"

    # Evaluate accuracy
    az_true = np.array([decode_target(y)[0] for y in Y_all])
    el_true = np.array([decode_target(y)[1] for y in Y_all])
    metrics = evaluate(model, X_all, az_true, el_true)

    # NN should achieve < 20° median azimuth error on this synthetic data
    assert metrics["az_median"] < 20.0, \
        f"Az median error too large: {metrics['az_median']:.1f}°"
    assert metrics["el_median"] < 15.0, \
        f"El median error too large: {metrics['el_median']:.1f}°"
