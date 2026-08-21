"""Matplotlib figure generation for the perturbation-rejection report.

Every figure is rendered twice (light/dark) so the HTML report can swap them via
CSS to match the viewer's theme. Colors follow the dataviz skill's reference
palette: calm/turbulent are fixed categorical entities (blue/red), heatmaps use
the blue sequential ramp, Phi-matrix comparisons use the blue<->red diverging pair.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

AXES = ["x", "y", "z"]

THEMES = {
    "light": dict(
        surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        calm="#2a78d6", turb="#e34948", accent="#1baf7a", accent2="#eda100",
    ),
    "dark": dict(
        surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        calm="#3987e5", turb="#e66767", accent="#199e70", accent2="#c98500",
    ),
}

SEQ_BLUE_LIGHT = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
SEQ_BLUE_DARK = ["#1a1a19", "#0d366b", "#184f95", "#2a78d6", "#5598e7", "#9ec5f4", "#cde2fb"]
DIV_BLUE_RED = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#7a1f1e"]


def _style(theme: str):
    c = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "text.color": c["primary"], "axes.labelcolor": c["primary"],
        "axes.edgecolor": c["axis"], "xtick.color": c["muted"], "ytick.color": c["muted"],
        "grid.color": c["grid"], "font.size": 10.5, "font.family": "sans-serif",
        "axes.titlecolor": c["primary"], "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return c


def _finish(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _grid(ax, c):
    ax.grid(True, linewidth=0.6, alpha=0.7, color=c["grid"])
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------

def fig_timeseries(episode, title, theme):
    c = _style(theme)
    t = np.array(episode["t"])
    acc = np.array(episode["acc"])
    phi_pred = np.array(episode["phi_pred"])
    delta_hat = np.array(episode["delta_hat"])
    recon = phi_pred + delta_hat

    fig, axs = plt.subplots(3, 1, figsize=(8.6, 6.6), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(t, acc[:, i], color=c["primary"], lw=1.1, label="Measured $A(t)$")
        ax.plot(t, recon[:, i], color=c["calm"] if "calm" in title.lower() else c["turb"],
                 lw=1.1, alpha=0.9, label=r"Reconstructed $\hat\Phi\cdot u + \hat\Delta$")
        rmse = np.sqrt(np.mean((acc[:, i] - recon[:, i]) ** 2))
        ax.text(0.99, 0.06, f"RMSE={rmse:.3f} m/s²", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=c["muted"])
        ax.set_ylabel(f"acc_{AXES[i]}  [m/s²]")
        _grid(ax, c)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8.5)
    axs[-1].set_xlabel("time within episode [s]")
    fig.suptitle(title, color=c["primary"], fontsize=12)
    fig.tight_layout()
    return _finish(fig)


def fig_psd(psd_calm, psd_turb, theme):
    c = _style(theme)
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3), sharey=False)
    for i, ax in enumerate(axs):
        ax_name = AXES[i]
        f_c, p_c = np.array(psd_calm[ax_name]["f"]), np.array(psd_calm[ax_name]["p"])
        f_t, p_t = np.array(psd_turb[ax_name]["f"]), np.array(psd_turb[ax_name]["p"])
        ax.loglog(f_c, p_c, color=c["calm"], lw=1.3, label="Calm")
        ax.loglog(f_t, p_t, color=c["turb"], lw=1.3, label="Under currents")
        ax.set_title(f"axis {ax_name}", fontsize=10.5)
        ax.set_xlabel("frequency [Hz]")
        if i == 0:
            ax.set_ylabel("PSD of residual [ (m/s²)²/Hz ]")
            ax.legend(fontsize=8.5, loc="upper right")
        _grid(ax, c)
    fig.suptitle(r"Residual power spectral density: $r(t) = A(t) - \hat\Phi_{calm}\cdot u(t) - \hat{G}\cdot orient(t)$",
                 color=c["primary"], fontsize=10.5)
    fig.tight_layout()
    return _finish(fig)


def fig_heatmap(heatmap_calm, heatmap_turb, freqs, dt_calm, dt_turb, theme):
    c = _style(theme)
    cmap = mcolors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE_LIGHT if theme == "light" else SEQ_BLUE_DARK)
    freqs = np.array(freqs)
    amp_calm = np.array(heatmap_calm)[:, :, 0]  # x-axis amplitude, (n_bins, K)
    amp_turb = np.array(heatmap_turb)[:, :, 0]
    vmax = float(np.percentile(np.concatenate([amp_calm.ravel(), amp_turb.ravel()]), 99))

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, amp, dt, name in [(axs[0], amp_calm, dt_calm, "Calm"), (axs[1], amp_turb, dt_turb, "Under currents")]:
        t_centers = (np.arange(amp.shape[0]) + 0.5) * dt
        im = ax.pcolormesh(t_centers, freqs, amp.T, cmap=cmap, vmin=0, vmax=vmax, shading="nearest")
        ax.set_yscale("log")
        ax.set_xlabel("time [s]")
        ax.set_title(name, fontsize=10.5)
        _grid(ax, c)
    axs[0].set_ylabel("frequency [Hz]")
    cbar = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02)
    cbar.set_label(r"$|C_k(t)|$ (acc_x, m/s²)", color=c["primary"])
    cbar.ax.yaxis.set_tick_params(color=c["muted"])
    plt.setp(cbar.ax.get_yticklabels(), color=c["muted"])
    fig.suptitle("Disturbance amplitude spectrogram (axis x)", color=c["primary"], fontsize=11)
    return _finish(fig)


def fig_phi_matrices(phi_calm, phi_naive, phi_corrected, theme):
    c = _style(theme)
    cmap = mcolors.LinearSegmentedColormap.from_list("div", DIV_BLUE_RED)
    mats = [np.array(phi_calm), np.array(phi_naive), np.array(phi_corrected)]
    titles = [r"$\hat\Phi_{calm}$", r"$\hat\Phi_{turb,\ naive}$", r"$\hat\Phi_{turb,\ corrected}$"]
    vmax = float(np.max(np.abs(np.concatenate([m.ravel() for m in mats]))))

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, mat, title in zip(axs, mats, titles):
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(AXES)
        ax.set_yticks(range(6)); ax.set_yticklabels([f"thr{i}" for i in range(1, 7)])
        ax.set_title(title, fontsize=10.5)
    cbar = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.03)
    cbar.set_label("gain [m/s² per PWM unit]", color=c["primary"])
    plt.setp(cbar.ax.get_yticklabels(), color=c["muted"])
    fig.suptitle("Thruster gain matrix $\\hat\\Phi$: calm vs. under-currents fits", color=c["primary"], fontsize=11)
    return _finish(fig)


def fig_phi_cosine_bar(cos_naive, cos_corrected, theme):
    c = _style(theme)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(3)
    w = 0.32
    ax.bar(x - w / 2, cos_naive, width=w, color=c["turb"], label="Naive turb. fit")
    ax.bar(x + w / 2, cos_corrected, width=w, color=c["calm"], label="Disturbance-corrected fit")
    ax.axhline(1.0, color=c["muted"], lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"axis {a}" for a in AXES])
    ax.set_ylabel(r"cosine similarity to $\hat\Phi_{calm}$")
    ax.set_ylim(min(-0.5, min(cos_naive) - 0.1), 1.05)
    _grid(ax, c)
    ax.legend(fontsize=9)
    fig.suptitle("Phi consistency: does the recovered thruster model match calm water?", color=c["primary"], fontsize=10.5)
    fig.tight_layout()
    return _finish(fig)


def fig_ablation_k(k_sweep, theme):
    c = _style(theme)
    ks = [d["k"] for d in k_sweep]
    rmse = [d["pred_rmse"] for d in k_sweep]
    runtime = [d["runtime_s"] for d in k_sweep]
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axs[0].plot(ks, rmse, "o-", color=c["calm"])
    axs[0].set_xlabel("# frequency bins K"); axs[0].set_ylabel("one-step pred. RMSE")
    axs[0].set_title("Tracking accuracy vs. K", fontsize=10)
    axs[1].plot(ks, runtime, "o-", color=c["accent2"])
    axs[1].set_xlabel("# frequency bins K"); axs[1].set_ylabel("runtime [s] (20k-frame subset)")
    axs[1].set_title("Compute cost vs. K", fontsize=10)
    for ax in axs:
        _grid(ax, c)
    fig.tight_layout()
    return _finish(fig)


def fig_ablation_q(q_sweep, theme):
    c = _style(theme)
    qs = [d["q"] for d in q_sweep]
    rmse = [d["pred_rmse"] for d in q_sweep]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogx(qs, rmse, "o-", color=c["calm"])
    best = int(np.argmin(rmse))
    ax.scatter([qs[best]], [rmse[best]], color=c["turb"], zorder=5, label=f"selected q={qs[best]:.0e}")
    ax.set_xlabel("process noise q"); ax.set_ylabel("one-step pred. RMSE")
    ax.legend(fontsize=9)
    _grid(ax, c)
    fig.suptitle("Process-noise ablation: bias (slow, large q missing) vs. variance", color=c["primary"], fontsize=10.5)
    fig.tight_layout()
    return _finish(fig)


def fig_gravity_ablation(ablation, theme):
    c = _style(theme)
    r2_with = ablation["held_out_r2_with_gravity"]
    r2_without = ablation["held_out_r2_without_gravity"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(3); w = 0.32
    ax.bar(x - w / 2, r2_without, width=w, color=c["turb"], label="Thrusters only")
    ax.bar(x + w / 2, r2_with, width=w, color=c["calm"], label="+ orientation (gravity) terms")
    ax.set_xticks(x); ax.set_xticklabels([f"axis {a}" for a in AXES])
    ax.set_ylabel("held-out R²")
    _grid(ax, c)
    ax.legend(fontsize=9)
    fig.suptitle("Gravity-term ablation on the calm set", color=c["primary"], fontsize=10.5)
    fig.tight_layout()
    return _finish(fig)


def fig_reconstruction_rmse(recon, theme):
    c = _style(theme)
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=False)
    for ax, cond, color in [(axs[0], "calm", c["calm"]), (axs[1], "turb", c["turb"])]:
        phi_only = recon[f"{cond}_phi_only"]
        full = recon[f"{cond}_phi_plus_disturbance"]
        x = np.arange(3); w = 0.32
        ax.bar(x - w / 2, phi_only, width=w, color=c["muted"], label=r"$\hat\Phi$ only")
        ax.bar(x + w / 2, full, width=w, color=color, label=r"$\hat\Phi+\hat\Delta$")
        ax.set_xticks(x); ax.set_xticklabels([f"axis {a}" for a in AXES])
        ax.set_title("Calm" if cond == "calm" else "Under currents", fontsize=10.5)
        ax.set_ylabel("held-out RMSE [m/s²]")
        _grid(ax, c)
        ax.legend(fontsize=8.5)
    fig.suptitle("Held-out reconstruction error: thruster model alone vs. + disturbance estimate", color=c["primary"], fontsize=10.5)
    fig.tight_layout()
    return _finish(fig)


def fig_kf_ukf_equivalence(t, kf_pred, ukf_pred, theme):
    c = _style(theme)
    t = np.array(t); kf_pred = np.array(kf_pred); ukf_pred = np.array(ukf_pred)
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True, height_ratios=[2, 1])
    axs[0].plot(t, kf_pred, color=c["calm"], lw=1.6, label="Closed-form linear KF")
    axs[0].plot(t, ukf_pred, color=c["turb"], lw=1.0, ls="--", label="From-scratch UKF")
    axs[0].legend(fontsize=8.5); axs[0].set_ylabel("predicted residual")
    _grid(axs[0], c)
    axs[1].plot(t, kf_pred - ukf_pred, color=c["accent2"], lw=1.0)
    axs[1].set_ylabel("KF − UKF"); axs[1].set_xlabel("time [s]")
    _grid(axs[1], c)
    fig.suptitle("UKF implementation check: exact agreement on a linear observation model", color=c["primary"], fontsize=10.5)
    fig.tight_layout()
    return _finish(fig)
