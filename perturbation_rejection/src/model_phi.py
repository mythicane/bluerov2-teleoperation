"""Orientation (gravity) + thruster linear model: A(t) ~= G . orient(t) + Phi . u(t).

Fit jointly per axis via OLS so we don't have to guess the IMU's axis-sign
convention for gravity: sin(pitch), sin(roll)cos(pitch), cos(roll)cos(pitch) plus
an intercept span the possible gravity-projection directions, and the regression
finds the right combination itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from load_data import PooledDataset

ORIENT_NAMES = ["sin_pitch", "sin_roll_cos_pitch", "cos_roll_cos_pitch", "bias"]
THRUST_NAMES = [f"thr{i}" for i in range(1, 7)]
FEATURE_NAMES = ORIENT_NAMES + THRUST_NAMES
N_ORIENT = len(ORIENT_NAMES)
N_THRUST = len(THRUST_NAMES)


def build_design(ds: PooledDataset) -> np.ndarray:
    roll, pitch = ds.roll, ds.pitch
    orient = np.stack([
        np.sin(pitch),
        np.sin(roll) * np.cos(pitch),
        np.cos(roll) * np.cos(pitch),
        np.ones_like(pitch),
    ], axis=1)
    return np.concatenate([orient, ds.u], axis=1)  # (N, 10)


def episode_train_test_split(ds: PooledDataset, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    uids = np.unique(ds.episode_uid)
    rng.shuffle(uids)
    n_test = max(1, int(round(len(uids) * test_frac)))
    test_uids = set(uids[:n_test].tolist())
    is_test = np.array([u in test_uids for u in ds.episode_uid])
    return ~is_test, is_test


@dataclass
class PhiFit:
    beta: np.ndarray  # (10, 3) : rows=features, cols=axes (x,y,z)
    feature_names: list

    @property
    def G(self) -> np.ndarray:
        return self.beta[:N_ORIENT, :]  # (4,3)

    @property
    def Phi(self) -> np.ndarray:
        return self.beta[N_ORIENT:, :]  # (6,3)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.beta

    def predict_from_ds(self, ds: PooledDataset) -> np.ndarray:
        return self.predict(build_design(ds))


def fit_phi(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray | None = None) -> PhiFit:
    """OLS fit per axis. y: (N,3). Returns PhiFit with beta (10,3)."""
    if train_mask is None:
        train_mask = np.ones(X.shape[0], dtype=bool)
    beta, *_ = np.linalg.lstsq(X[train_mask], y[train_mask], rcond=None)
    return PhiFit(beta=beta, feature_names=FEATURE_NAMES)


def rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis RMSE and R^2. y_true/y_pred: (N,3)."""
    resid = y_true - y_pred
    rmse = np.sqrt(np.mean(resid ** 2, axis=0))
    ss_res = np.sum(resid ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return rmse, r2


if __name__ == "__main__":
    from load_data import load_condition

    ds = load_condition("calm")
    X = build_design(ds)
    y = ds.acc
    train_mask, test_mask = episode_train_test_split(ds)
    fit = fit_phi(X, y, train_mask)
    pred = fit.predict(X)
    rmse, r2 = rmse_r2(y[test_mask], pred[test_mask])
    print("axes: x, y, z")
    print("held-out RMSE [m/s^2]:", rmse)
    print("held-out R^2:", r2)
    print("Phi (thruster gains, 6x3):")
    print(fit.Phi)
    print("G (orientation coeffs, 4x3):")
    print(fit.G)
