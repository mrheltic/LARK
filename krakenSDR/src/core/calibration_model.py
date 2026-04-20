"""
core.calibration_model
======================
Pure-numpy MLP for phase-difference → (az, el) calibration.

Architecture
------------
Input  :  14 features  (8 phase cos/sin + 4 coherence + eig_spread + doppler)
Hidden :  14 → 64 → 64 → 32  (ReLU activations, batch-norm style normalization)
Output :  3 values  (cos_az, sin_az, el_norm)

The model intentionally uses **no external ML framework** — only numpy + scipy —
so it can run on any LARK deployment without extra dependencies.  For production
training with large datasets, a PyTorch wrapper is provided as an optional
accelerator.

Loss
----
Composite loss that respects angular geometry:
  L = w_az · L_az  +  w_el · L_el

  L_az = 1 − cos(az_pred − az_true)          (great-circle-aware, no wrap issue)
  L_el = (el_pred − el_true)² / 90²          (MSE on normalized elevation)

This avoids the 0°/360° discontinuity and weights azimuth/elevation errors
proportionally to their typical magnitude.

Serialization
-------------
Model weights are stored as a flat .npz file with named arrays:
  W1, b1, W2, b2, W3, b3, W4, b4   (layer weights and biases)
  feat_mean, feat_std               (input normalization stats)
  train_meta                        (JSON string with training history)

The .npz can be loaded by numpy alone — no pickle, no framework dependency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Model definition
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CalibrationMLP:
    """Lightweight MLP: 14 → 64 → 64 → 32 → 3."""

    # Layer sizes
    sizes: tuple[int, ...] = (14, 64, 64, 32, 3)

    # Weights & biases (initialized by init_weights)
    weights: list[np.ndarray] = field(default_factory=list, repr=False)
    biases: list[np.ndarray] = field(default_factory=list, repr=False)

    # Input normalization
    feat_mean: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.float32), repr=False)
    feat_std: np.ndarray = field(default_factory=lambda: np.ones(14, dtype=np.float32), repr=False)

    # Training metadata
    train_history: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.weights:
            self.init_weights()

    def init_weights(self, seed: int = 42) -> None:
        """He initialization for ReLU layers, Xavier for output."""
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for i in range(len(self.sizes) - 1):
            fan_in = self.sizes[i]
            fan_out = self.sizes[i + 1]
            if i < len(self.sizes) - 2:
                # He init for ReLU
                std = np.sqrt(2.0 / fan_in)
            else:
                # Xavier for output (linear activation)
                std = np.sqrt(2.0 / (fan_in + fan_out))
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * std
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        X : (N, 14) float32 — raw features (un-normalized)

        Returns
        -------
        (N, 3) float32 — [cos_az, sin_az, el_norm]
        """
        # Normalize input
        h = (X - self.feat_mean) / np.maximum(self.feat_std, 1e-8)

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            h = h @ self.weights[i] + self.biases[i]
            h = np.maximum(h, 0.0)  # ReLU

        # Output layer (linear)
        y = h @ self.weights[-1] + self.biases[-1]

        # Normalize azimuth output to unit circle
        norm = np.sqrt(y[:, 0:1] ** 2 + y[:, 1:2] ** 2 + 1e-8)
        y[:, 0:2] /= norm

        # Clamp elevation to [0, 1]
        y[:, 2] = np.clip(y[:, 2], 0.0, 1.0)

        return y

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict (az_deg, el_deg) from raw features.

        Parameters
        ----------
        X : (N, 14) float32

        Returns
        -------
        az_deg : (N,) float64
        el_deg : (N,) float64
        """
        y = self.forward(X)
        from core.calibration_features import decode_target
        az, el = decode_target(y)
        return np.asarray(az, dtype=np.float64), np.asarray(el, dtype=np.float64)

    def n_params(self) -> int:
        """Total trainable parameters."""
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model to .npz."""
        arrays = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        arrays["feat_mean"] = self.feat_mean
        arrays["feat_std"] = self.feat_std
        arrays["sizes"] = np.array(self.sizes, dtype=np.int32)
        arrays["train_meta"] = np.array(json.dumps(self.train_history))
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationMLP":
        """Load model from .npz."""
        data = np.load(str(path), allow_pickle=False)
        sizes = tuple(data["sizes"].tolist())
        model = cls(sizes=sizes)
        model.weights = []
        model.biases = []
        for i in range(len(sizes) - 1):
            model.weights.append(data[f"W{i}"])
            model.biases.append(data[f"b{i}"])
        model.feat_mean = data["feat_mean"]
        model.feat_std = data["feat_std"]
        meta_str = str(data["train_meta"])
        model.train_history = json.loads(meta_str) if meta_str else {}
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

def angular_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Composite angular loss.

    Parameters
    ----------
    y_pred, y_true : (N, 3) — [cos_az, sin_az, el_norm]

    Returns
    -------
    float — scalar loss
    """
    # Azimuth: 1 - cos(angle_diff) = 1 - (cos_p·cos_t + sin_p·sin_t)
    cos_diff = y_pred[:, 0] * y_true[:, 0] + y_pred[:, 1] * y_true[:, 1]
    cos_diff = np.clip(cos_diff, -1.0, 1.0)
    loss_az = np.mean(1.0 - cos_diff)

    # Elevation: MSE on normalized [0, 1]
    loss_el = np.mean((y_pred[:, 2] - y_true[:, 2]) ** 2)

    return float(loss_az + loss_el)


def angular_loss_grad(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Gradient of angular_loss w.r.t. y_pred.

    Returns (N, 3) gradient array.
    """
    N = y_pred.shape[0]
    grad = np.zeros_like(y_pred)

    # d/d(cos_p) [1 - cos_p·cos_t - sin_p·sin_t] = -cos_t / N
    grad[:, 0] = -y_true[:, 0] / N
    # d/d(sin_p) = -sin_t / N
    grad[:, 1] = -y_true[:, 1] / N
    # d/d(el_p) [el_p - el_t]² = 2·(el_p - el_t) / N
    grad[:, 2] = 2.0 * (y_pred[:, 2] - y_true[:, 2]) / N

    return grad


# ═══════════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    lr: float = 1e-3
    lr_decay: float = 0.98         # per-epoch multiplicative decay
    batch_size: int = 64
    epochs: int = 200
    val_fraction: float = 0.15     # hold-out for validation
    patience: int = 25             # early stopping patience (epochs)
    weight_decay: float = 1e-5     # L2 regularization
    seed: int = 42
    verbose: bool = True


def train(
    model: CalibrationMLP,
    X: np.ndarray,
    Y: np.ndarray,
    cfg: TrainConfig = TrainConfig(),
) -> dict:
    """Train model with mini-batch SGD + momentum + early stopping.

    Parameters
    ----------
    model : CalibrationMLP
    X : (N, 14) float32 — features
    Y : (N, 3) float32 — encoded targets [cos_az, sin_az, el_norm]
    cfg : TrainConfig

    Returns
    -------
    dict — training history: {train_loss, val_loss, best_epoch, elapsed_s}
    """
    rng = np.random.default_rng(cfg.seed)
    N = X.shape[0]

    # Train/val split (deterministic shuffle)
    idx = rng.permutation(N)
    n_val = max(1, int(N * cfg.val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    # Compute normalization from training data
    model.feat_mean = np.mean(X_train, axis=0).astype(np.float32)
    model.feat_std = np.std(X_train, axis=0).astype(np.float32)
    model.feat_std[model.feat_std < 1e-6] = 1.0  # avoid div-by-zero on constant features

    # Momentum buffers (same structure as weights)
    vel_W = [np.zeros_like(W) for W in model.weights]
    vel_b = [np.zeros_like(b) for b in model.biases]
    momentum = 0.9

    best_val_loss = float("inf")
    best_weights = None
    best_biases = None
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}

    lr = cfg.lr
    t0 = time.time()
    n_train = X_train.shape[0]

    for epoch in range(cfg.epochs):
        # Shuffle training data
        perm = rng.permutation(n_train)
        X_t = X_train[perm]
        Y_t = Y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, cfg.batch_size):
            end = min(start + cfg.batch_size, n_train)
            Xb = X_t[start:end]
            Yb = Y_t[start:end]

            # ── Forward ──────────────────────────────────────────────────
            activations = []
            h = (Xb - model.feat_mean) / np.maximum(model.feat_std, 1e-8)
            activations.append(h)

            for i in range(len(model.weights) - 1):
                z = h @ model.weights[i] + model.biases[i]
                h = np.maximum(z, 0.0)
                activations.append(h)

            # Output (linear)
            y_raw = h @ model.weights[-1] + model.biases[-1]

            # Normalize az to unit circle
            norm = np.sqrt(y_raw[:, 0:1] ** 2 + y_raw[:, 1:2] ** 2 + 1e-8)
            y_out = y_raw.copy()
            y_out[:, 0:2] /= norm
            y_out[:, 2] = np.clip(y_out[:, 2], 0.0, 1.0)

            batch_loss = angular_loss(y_out, Yb)
            epoch_loss += batch_loss
            n_batches += 1

            # ── Backward ─────────────────────────────────────────────────
            # Gradient of loss w.r.t. y_out
            grad_out = angular_loss_grad(y_out, Yb)

            # Through unit-circle normalization for first 2 outputs
            # d(x/||x||)/dx = (I - x̂·x̂ᵀ) / ||x||
            grad_raw = grad_out.copy()
            for n in range(Xb.shape[0]):
                c, s = y_out[n, 0], y_out[n, 1]
                n_val_ = norm[n, 0]
                # Jacobian of normalization
                J = np.array([[1 - c * c, -c * s],
                              [-c * s, 1 - s * s]]) / n_val_
                grad_raw[n, 0:2] = J @ grad_out[n, 0:2]

            # Backprop through layers
            delta = grad_raw
            grad_W_list = []
            grad_b_list = []

            # Output layer
            grad_W_list.append(activations[-1].T @ delta)
            grad_b_list.append(delta.sum(axis=0))
            delta = delta @ model.weights[-1].T

            # Hidden layers (reverse)
            for i in range(len(model.weights) - 2, -1, -1):
                # ReLU derivative
                delta = delta * (activations[i + 1] > 0).astype(np.float32)
                grad_W_list.append(activations[i].T @ delta)
                grad_b_list.append(delta.sum(axis=0))
                if i > 0:
                    delta = delta @ model.weights[i].T

            # Reverse to match layer order
            grad_W_list.reverse()
            grad_b_list.reverse()

            # ── Update (SGD + momentum + weight decay) ───────────────────
            for i in range(len(model.weights)):
                gW = grad_W_list[i] + cfg.weight_decay * model.weights[i]
                gb = grad_b_list[i]
                vel_W[i] = momentum * vel_W[i] - lr * gW
                vel_b[i] = momentum * vel_b[i] - lr * gb
                model.weights[i] += vel_W[i]
                model.biases[i] += vel_b[i]

        # ── Epoch metrics ────────────────────────────────────────────────
        train_loss = epoch_loss / max(n_batches, 1)
        val_pred = model.forward(X_val)
        val_loss = angular_loss(val_pred, Y_val)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = [W.copy() for W in model.weights]
            best_biases = [b.copy() for b in model.biases]

        if cfg.verbose and (epoch % 20 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:4d}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  lr={lr:.2e}")

        # Early stopping
        if epoch - best_epoch >= cfg.patience:
            if cfg.verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best={best_epoch}, val={best_val_loss:.5f})")
            break

        lr *= cfg.lr_decay

    # Restore best weights
    if best_weights is not None:
        model.weights = best_weights
        model.biases = best_biases

    elapsed = time.time() - t0
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_val_loss)
    history["elapsed_s"] = round(elapsed, 2)
    history["n_train"] = int(n_train)
    history["n_val"] = int(n_val)
    history["n_params"] = model.n_params()
    model.train_history = history

    if cfg.verbose:
        print(f"  Training done: {elapsed:.1f}s, best epoch {best_epoch}, "
              f"val loss {best_val_loss:.5f}, {model.n_params()} params")

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: CalibrationMLP,
    X: np.ndarray,
    az_true: np.ndarray,
    el_true: np.ndarray,
) -> dict:
    """Evaluate model accuracy.

    Parameters
    ----------
    model : CalibrationMLP
    X : (N, 14) features
    az_true, el_true : (N,) ground truth in degrees

    Returns
    -------
    dict with metrics:
        az_mae, az_median, az_p90  — azimuth error stats [deg]
        el_mae, el_median, el_p90  — elevation error stats [deg]
        angular_sep_mean, angular_sep_p90  — great-circle separation [deg]
        n_samples
    """
    az_pred, el_pred = model.predict(X)

    # Circular azimuth error
    az_err = np.abs((az_pred - az_true + 180.0) % 360.0 - 180.0)
    el_err = np.abs(el_pred - el_true)

    # Great-circle angular separation
    az_p_r = np.deg2rad(az_pred)
    az_t_r = np.deg2rad(az_true)
    el_p_r = np.deg2rad(el_pred)
    el_t_r = np.deg2rad(el_true)
    cos_sep = (np.sin(el_p_r) * np.sin(el_t_r) +
               np.cos(el_p_r) * np.cos(el_t_r) * np.cos(az_p_r - az_t_r))
    angular_sep = np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0)))

    return {
        "az_mae":           float(np.mean(az_err)),
        "az_median":        float(np.median(az_err)),
        "az_p90":           float(np.percentile(az_err, 90)),
        "el_mae":           float(np.mean(el_err)),
        "el_median":        float(np.median(el_err)),
        "el_p90":           float(np.percentile(el_err, 90)),
        "angular_sep_mean": float(np.mean(angular_sep)),
        "angular_sep_p90":  float(np.percentile(angular_sep, 90)),
        "n_samples":        int(len(az_true)),
    }
