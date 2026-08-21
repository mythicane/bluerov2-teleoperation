"""End-to-end perturbation-rejection pipeline: fit Phi on calm data, isolate Delta(t) on
under-currents data via a harmonic Kalman filter, and check Phi consistency across both.
See method.md for the assumptions and ../serene-humming-rabbit plan for the design.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.signal import welch

from load_data import load_condition, PooledDataset
from model_phi import (
    build_design, episode_train_test_split, fit_phi, rmse_r2,
    N_ORIENT, THRUST_NAMES, PhiFit,
)
from kalman import freq_bank, run_linear_harmonic_kf, run_ukf, design_matrix

AXES = ["x", "y", "z"]
K_BASELINE = 41
Q_BASELINE = 1e-5  # selected by the Q ablation sweep (see ablation_q_sweep in results.json)
F_MIN, F_MAX = 1e-3, 10.0


def noise_floor_r(residual: np.ndarray) -> np.ndarray:
    """Per-axis measurement-noise variance estimate from the first-difference heuristic."""
    d = np.diff(residual, axis=0)
    return np.var(d, axis=0) / 2.0


def phi_similarity(phi_a: np.ndarray, phi_b: np.ndarray) -> dict:
    fro = float(np.linalg.norm(phi_a - phi_b))
    fro_rel = fro / float(np.linalg.norm(phi_a))
    cos_per_axis = []
    for j in range(phi_a.shape[1]):
        va, vb = phi_a[:, j], phi_b[:, j]
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        cos_per_axis.append(float(np.dot(va, vb) / denom) if denom > 0 else float("nan"))
    return {"frobenius": fro, "frobenius_relative": fro_rel, "cosine_per_axis": cos_per_axis}


def pick_representative_episode(ds: PooledDataset) -> dict:
    lens = np.array([b - a for a, b, _, _ in ds.episode_boundaries])
    idx = int(np.argsort(np.abs(lens - np.median(lens)))[0])
    start, end, color, local_idx = ds.episode_boundaries[idx]
    return {"start": start, "end": end, "color": color, "local_idx": local_idx}


def run_kf_baseline(residual: np.ndarray, t: np.ndarray, r_axis: np.ndarray,
                     k: int = K_BASELINE, q: float = Q_BASELINE, store_state: bool = False):
    freqs = freq_bank(F_MIN, F_MAX, k)
    r_mean = float(np.mean(r_axis))
    res = run_linear_harmonic_kf(t, residual, freqs, q=q, r=r_mean, store_state=store_state)
    return res, freqs


def main():
    t_start = time.time()
    results = {}
    heavy = {}

    print("[1/8] Loading datasets...")
    calm = load_condition("calm")
    turb = load_condition("turbulent")
    results["data"] = {
        "calm_frames": calm.n, "calm_episodes": len(calm.episode_boundaries),
        "calm_duration_s": calm.n * calm.dt,
        "turb_frames": turb.n, "turb_episodes": len(turb.episode_boundaries),
        "turb_duration_s": turb.n * turb.dt,
    }

    print("[2/8] Fitting Phi_calm (gravity + thrusters, OLS)...")
    X_calm = build_design(calm)
    y_calm = calm.acc
    train_calm, test_calm = episode_train_test_split(calm, seed=0)
    phi_calm = fit_phi(X_calm, y_calm, train_calm)
    pred_calm_test = phi_calm.predict(X_calm[test_calm])
    rmse_calm, r2_calm = rmse_r2(y_calm[test_calm], pred_calm_test)
    results["phi_calm"] = {
        "beta": phi_calm.beta.tolist(), "held_out_rmse": rmse_calm.tolist(),
        "held_out_r2": r2_calm.tolist(),
    }
    print(f"    held-out R^2 (x,y,z) = {r2_calm}")

    # gravity-term ablation: fit without the 3 orientation regressors (keep bias)
    X_calm_nograv = X_calm.copy()
    X_calm_nograv[:, :3] = 0.0
    phi_calm_nograv = fit_phi(X_calm_nograv, y_calm, train_calm)
    pred_nograv_test = phi_calm_nograv.predict(X_calm_nograv[test_calm])
    rmse_nograv, r2_nograv = rmse_r2(y_calm[test_calm], pred_nograv_test)
    bias_with = np.mean(y_calm[test_calm] - pred_calm_test, axis=0)
    bias_without = np.mean(y_calm[test_calm] - pred_nograv_test, axis=0)
    results["ablation_gravity"] = {
        "held_out_rmse_with_gravity": rmse_calm.tolist(),
        "held_out_rmse_without_gravity": rmse_nograv.tolist(),
        "held_out_r2_with_gravity": r2_calm.tolist(),
        "held_out_r2_without_gravity": r2_nograv.tolist(),
        "mean_residual_bias_with_gravity": bias_with.tolist(),
        "mean_residual_bias_without_gravity": bias_without.tolist(),
    }

    print("[3/8] Isolating residuals (Delta candidates) on calm & turbulent sets...")
    residual_calm = y_calm - phi_calm.predict(X_calm)
    X_turb = build_design(turb)
    y_turb = turb.acc
    residual_turb_naive_phi = y_turb - phi_calm.predict(X_turb)  # Delta_hat via calm-Phi

    r_axis_calm = noise_floor_r(residual_calm)
    print(f"    per-axis noise floor R (calm) = {r_axis_calm}")

    print("[4/8] Running baseline harmonic KF on calm residual (K={}, q={})...".format(K_BASELINE, Q_BASELINE))
    kf_calm, freqs = run_kf_baseline(residual_calm, calm.t, r_axis_calm, store_state=True)

    print("[4/8] Running baseline harmonic KF on turbulent residual (calm-Phi)...")
    kf_turb_naive, _ = run_kf_baseline(residual_turb_naive_phi, turb.t, r_axis_calm, store_state=True)

    print("[5/8] Alternating refit: Phi_turb_naive (no correction) and Phi_turb_corrected...")
    train_turb, test_turb = episode_train_test_split(turb, seed=1)
    phi_turb_naive = fit_phi(X_turb, y_turb, train_turb)  # naive direct fit, disturbance NOT removed

    delta_hat_turb = kf_turb_naive.pred  # (N,3) one-step-ahead disturbance estimate
    y_turb_corrected_target = y_turb - delta_hat_turb
    phi_turb_corrected = fit_phi(X_turb, y_turb_corrected_target, train_turb)

    # second alternation round
    residual_turb_r2 = y_turb - phi_turb_corrected.predict(X_turb)
    kf_turb_r2, _ = run_kf_baseline(residual_turb_r2, turb.t, r_axis_calm, store_state=False)
    y_turb_corrected_target_r2 = y_turb - kf_turb_r2.pred
    phi_turb_corrected_r2 = fit_phi(X_turb, y_turb_corrected_target_r2, train_turb)

    results["phi_consistency"] = {
        "calm_vs_turb_naive": phi_similarity(phi_calm.Phi, phi_turb_naive.Phi),
        "calm_vs_turb_corrected_round1": phi_similarity(phi_calm.Phi, phi_turb_corrected.Phi),
        "calm_vs_turb_corrected_round2": phi_similarity(phi_calm.Phi, phi_turb_corrected_r2.Phi),
        "phi_calm": phi_calm.Phi.tolist(),
        "phi_turb_naive": phi_turb_naive.Phi.tolist(),
        "phi_turb_corrected_round1": phi_turb_corrected.Phi.tolist(),
        "phi_turb_corrected_round2": phi_turb_corrected_r2.Phi.tolist(),
    }
    print("    Frobenius-rel calm-vs-naive:", results["phi_consistency"]["calm_vs_turb_naive"]["frobenius_relative"])
    print("    Frobenius-rel calm-vs-corrected(r2):", results["phi_consistency"]["calm_vs_turb_corrected_round2"]["frobenius_relative"])

    print("[6/8] Held-out reconstruction RMSE: Phi-only vs Phi+disturbance...")
    # Use corrected-round2 Phi as the "final" turbulent Phi for held-out evaluation.
    residual_turb_final = y_turb - phi_turb_corrected_r2.predict(X_turb)
    kf_turb_final, freqs_t = run_kf_baseline(residual_turb_final, turb.t, r_axis_calm, store_state=True)

    def held_out_rmse(mask, phi_fit, X, y, kf_pred):
        pred_phi_only = phi_fit.predict(X)
        pred_full = pred_phi_only + kf_pred
        rmse_phi, _ = rmse_r2(y[mask], pred_phi_only[mask])
        rmse_full, _ = rmse_r2(y[mask], pred_full[mask])
        return rmse_phi.tolist(), rmse_full.tolist()

    rmse_calm_phi_only, rmse_calm_full = held_out_rmse(test_calm, phi_calm, X_calm, y_calm, kf_calm.pred)
    rmse_turb_phi_only, rmse_turb_full = held_out_rmse(test_turb, phi_turb_corrected_r2, X_turb, y_turb, kf_turb_final.pred)
    results["reconstruction_rmse"] = {
        "calm_phi_only": rmse_calm_phi_only, "calm_phi_plus_disturbance": rmse_calm_full,
        "turb_phi_only": rmse_turb_phi_only, "turb_phi_plus_disturbance": rmse_turb_full,
    }
    print("    calm:", rmse_calm_phi_only, "->", rmse_calm_full)
    print("    turb:", rmse_turb_phi_only, "->", rmse_turb_full)

    print("[7/8] Ablations: frequency-count (K) and process-noise (Q) sweeps...")
    sub_n = 20000
    sub_t = turb.t[:sub_n]
    sub_r = residual_turb_naive_phi[:sub_n]
    r_mean = float(np.mean(r_axis_calm))

    k_sweep = []
    for k in [10, 20, 41, 80]:
        freqs_k = freq_bank(F_MIN, F_MAX, k)
        t0 = time.time()
        res_k = run_linear_harmonic_kf(sub_t, sub_r, freqs_k, q=Q_BASELINE, r=r_mean, store_state=False)
        dt_run = time.time() - t0
        rmse_k = float(np.sqrt(np.mean((sub_r - res_k.pred) ** 2)))
        k_sweep.append({"k": k, "pred_rmse": rmse_k, "runtime_s": dt_run})
    results["ablation_k_sweep"] = k_sweep
    print("    K sweep:", k_sweep)

    q_sweep = []
    for q in [1e-9, 1e-7, 1e-5, 1e-3]:
        res_q = run_linear_harmonic_kf(sub_t, sub_r, freq_bank(F_MIN, F_MAX, K_BASELINE), q=q, r=r_mean, store_state=False)
        rmse_q = float(np.sqrt(np.mean((sub_r - res_q.pred) ** 2)))
        q_sweep.append({"q": q, "pred_rmse": rmse_q})
    results["ablation_q_sweep"] = q_sweep
    print("    Q sweep:", q_sweep)

    print("[8/8] KF vs from-scratch UKF equivalence check...")
    n_eq = 800
    freqs_eq = freq_bank(F_MIN, F_MAX, 9)
    res_eq = run_linear_harmonic_kf(sub_t[:n_eq], sub_r[:n_eq, 0], freqs_eq, q=Q_BASELINE, r=r_mean, store_state=False)
    preds_ukf = run_ukf(sub_t[:n_eq], sub_r[:n_eq, 0], freqs_eq, q=Q_BASELINE, r=r_mean)
    max_diff = float(np.max(np.abs(res_eq.pred[:, 0] - preds_ukf)))
    results["kf_ukf_equivalence"] = {"n_steps": n_eq, "k": 9, "max_abs_diff": max_diff}
    print("    max abs diff (KF vs UKF), n=", n_eq, ":", max_diff)
    heavy["kf_ukf_eq_series"] = {
        "t": sub_t[:n_eq].tolist(), "kf_pred": res_eq.pred[:, 0].tolist(), "ukf_pred": preds_ukf.tolist(),
    }

    print("Computing PSDs (calm vs turbulent residual)...")
    psd = {}
    for name, sig, t in [("calm", residual_calm, calm.t), ("turb", residual_turb_naive_phi, turb.t)]:
        fs = 1.0 / np.mean(np.diff(t))
        nper = min(8192, sig.shape[0])
        psd[name] = {}
        for ai, ax in enumerate(AXES):
            f, p = welch(sig[:, ai], fs=fs, nperseg=nper, detrend="constant")
            mask = (f >= 1e-3) & (f <= 10.0)
            psd[name][ax] = {"f": f[mask].tolist(), "p": p[mask].tolist()}
    heavy["psd"] = psd

    print("Extracting representative episodes for time-series plot...")
    ep_calm = pick_representative_episode(calm)
    ep_turb = pick_representative_episode(turb)

    def episode_slice(ds, ep, phi_fit, X, kf_pred, residual):
        s, e = ep["start"], ep["end"]
        return {
            "t": (ds.t[s:e] - ds.t[s]).tolist(),
            "acc": ds.acc[s:e].tolist(),
            "phi_pred": phi_fit.predict(X[s:e]).tolist(),
            "delta_hat": kf_pred[s:e].tolist(),
            "residual": residual[s:e].tolist(),
            "color": ep["color"],
        }

    heavy["episode_calm"] = episode_slice(calm, ep_calm, phi_calm, X_calm, kf_calm.pred, residual_calm)
    heavy["episode_turb"] = episode_slice(turb, ep_turb, phi_turb_corrected_r2, X_turb, kf_turb_final.pred, residual_turb_final)

    print("Building |C_k(t)| heatmaps (downsampled)...")
    def downsample_amplitude(x_hist, bin_size=200):
        # x_hist: (N, 2K, n_axes) -> amplitude (N, K, n_axes) -> bin-averaged over time
        n, k2, na = x_hist.shape
        k = k2 // 2
        a = x_hist[:, 0::2, :]
        b = x_hist[:, 1::2, :]
        amp = np.sqrt(a ** 2 + b ** 2)  # (N, K, na)
        n_bins = n // bin_size
        amp = amp[: n_bins * bin_size].reshape(n_bins, bin_size, k, na).mean(axis=1)
        return amp  # (n_bins, K, na)

    heavy["heatmap_calm"] = downsample_amplitude(kf_calm.x_mean).tolist()
    heavy["heatmap_turb"] = downsample_amplitude(kf_turb_final.x_mean).tolist()
    heavy["heatmap_freqs"] = freqs.tolist()
    heavy["heatmap_calm_t_bin_s"] = float(200 * calm.dt)
    heavy["heatmap_turb_t_bin_s"] = float(200 * turb.dt)

    results["hyperparams"] = {"k_baseline": K_BASELINE, "q_baseline": Q_BASELINE,
                               "r_axis_calm": r_axis_calm.tolist(), "f_min": F_MIN, "f_max": F_MAX}
    results["runtime_s"] = time.time() - t_start

    print("Saving outputs...")
    with open("../outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("../outputs/heavy.json", "w") as f:
        json.dump(heavy, f)

    print(f"Done in {results['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
