"""Harmonic state-space model for the disturbance coefficients C(t).

State x = [a_1, b_1, ..., a_K, b_K], Delta(t) ~= sum_k a_k(t) cos(2*pi*f_k*t) + b_k(t) sin(2*pi*f_k*t).
Process model: random walk (x_{t+1} = x_t + w), consistent with Assumption 3 (A(t) time-variant).
Measurement model: scalar residual r(t) = H_t @ x + noise, H_t = [cos(w_k t), sin(w_k t)]_k -- linear
in x, so a closed-form Kalman filter is exact and used for the full-scale run; `UKF` is a generic
from-scratch sigma-point filter used only to validate that it reproduces the same result (since h is
linear, the unscented transform is exact here, making this a clean implementation sanity check).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def freq_bank(f_min: float = 1e-3, f_max: float = 10.0, k: int = 41) -> np.ndarray:
    return np.logspace(np.log10(f_min), np.log10(f_max), k)


def design_matrix(t: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """H: (N, 2K) with columns [cos(w_1 t), sin(w_1 t), cos(w_2 t), sin(w_2 t), ...]."""
    w = 2.0 * np.pi * freqs  # (K,)
    phase = np.outer(t, w)  # (N, K)
    H = np.empty((t.shape[0], 2 * freqs.shape[0]), dtype=np.float64)
    H[:, 0::2] = np.cos(phase)
    H[:, 1::2] = np.sin(phase)
    return H


@dataclass
class KFResult:
    freqs: np.ndarray
    x_mean: np.ndarray  # (N, 2K, n_series) coefficient trajectories
    pred: np.ndarray  # (N, n_series) predicted measurement H_t @ x_{t|t-1}
    q: float
    r: float


def run_linear_harmonic_kf(t: np.ndarray, z: np.ndarray, freqs: np.ndarray,
                            q: float, r: float, store_state: bool = True) -> KFResult:
    """Filter one or more measurement series sharing the same time base t.

    z: (N,) or (N, n_series). Because H_t depends only on t (not on the series), the Kalman gain
    is identical across series at each step, so all series are updated in one batched pass.
    """
    if z.ndim == 1:
        z = z[:, None]
    n, n_series = z.shape
    K2 = 2 * freqs.shape[0]
    H = design_matrix(t, freqs)  # (N, 2K)

    x = np.zeros((K2, n_series))
    P = np.eye(K2) * 1.0
    Qd = q * np.eye(K2)

    x_hist = np.empty((n, K2, n_series)) if store_state else None
    pred = np.empty((n, n_series))

    for i in range(n):
        # predict (random walk)
        P = P + Qd
        Ht = H[i]  # (2K,)
        Hx = Ht @ x  # (n_series,)
        pred[i] = Hx
        # update
        PHt = P @ Ht  # (2K,)
        S = float(Ht @ PHt) + r
        gain = PHt / S  # (2K,)
        innov = z[i] - Hx  # (n_series,)
        x = x + np.outer(gain, innov)
        P = P - np.outer(gain, PHt)
        if store_state:
            x_hist[i] = x

    return KFResult(freqs=freqs, x_mean=x_hist if store_state else x[None], pred=pred, q=q, r=r)


class UKF:
    """Generic additive-noise scaled-sigma-point UKF (Van der Merwe form)."""

    def __init__(self, dim_x: int, f, h, q: float, r: float,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.n = dim_x
        self.f = f  # f(x) -> x'
        self.h = h  # h(x, t) -> scalar/measurement
        self.Q = q * np.eye(dim_x)
        self.R = r
        lam = alpha ** 2 * (dim_x + kappa) - dim_x
        self.lam = lam
        c = dim_x + lam
        self.sqrt_c = np.sqrt(c)
        self.Wm = np.full(2 * dim_x + 1, 1.0 / (2 * c))
        self.Wc = np.full(2 * dim_x + 1, 1.0 / (2 * c))
        self.Wm[0] = lam / c
        self.Wc[0] = lam / c + (1 - alpha ** 2 + beta)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        S = np.linalg.cholesky((P + P.T) / 2 + 1e-15 * np.eye(self.n))
        pts = np.empty((2 * self.n + 1, self.n))
        pts[0] = x
        for i in range(self.n):
            pts[1 + i] = x + self.sqrt_c * S[:, i]
            pts[1 + self.n + i] = x - self.sqrt_c * S[:, i]
        return pts

    def step(self, x: np.ndarray, P: np.ndarray, z: float, t: float):
        # predict
        sig = self._sigma_points(x, P)
        sig_f = np.array([self.f(s) for s in sig])
        x_pred = self.Wm @ sig_f
        diff = sig_f - x_pred
        P_pred = (self.Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) + self.Q

        # update
        sig2 = self._sigma_points(x_pred, P_pred)
        z_sig = np.array([self.h(s, t) for s in sig2])
        z_pred = float(self.Wm @ z_sig)
        zdiff = z_sig - z_pred
        Pzz = float((self.Wc * zdiff * zdiff).sum()) + self.R
        xdiff = sig2 - x_pred
        Pxz = (self.Wc[:, None] * xdiff * zdiff[:, None]).sum(axis=0)
        gain = Pxz / Pzz
        x_new = x_pred + gain * (z - z_pred)
        P_new = P_pred - np.outer(gain, gain) * Pzz
        return x_new, P_new, z_pred


def run_ukf(t: np.ndarray, z: np.ndarray, freqs: np.ndarray, q: float, r: float) -> np.ndarray:
    """Reference from-scratch UKF run (single series), for equivalence validation only."""
    K2 = 2 * freqs.shape[0]
    w = 2.0 * np.pi * freqs

    def f(x):
        return x

    def h(x, ti):
        phase = w * ti
        Hi = np.empty(K2)
        Hi[0::2] = np.cos(phase)
        Hi[1::2] = np.sin(phase)
        return float(Hi @ x)

    ukf = UKF(K2, f, h, q, r)
    x = np.zeros(K2)
    P = np.eye(K2)
    preds = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        x, P, zpred = ukf.step(x, P, z[i], t[i])
        preds[i] = zpred
    return preds
