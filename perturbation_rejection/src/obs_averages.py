"""Time-weighted value distributions (x = observation value, y = fraction of time spent at that
value) for each of the 30 observation.state fields, for grab-red-rod calm vs. under-currents,
superimposed for direct comparison. 30 graphs, saved as one multi-page PDF and one HTML page.

Every frame carries equal dt (50 Hz, constant), so a plain density histogram over pooled frame
values already IS the time-weighted distribution -- no extra weighting needed.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from load_data import _episode_files, STATE_NAMES

COLOR = "red"
DATASETS = [("Grab red rod (calm)", False, "#2a78d6"), ("Grab red rod (under currents)", True, "#e34948")]

STYLE = dict(
    surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9",
)


def load_state_matrices(color: str, turbulent: bool) -> list[np.ndarray]:
    files = _episode_files(color, turbulent)
    episodes = []
    for f in files:
        df = pd.read_parquet(f, columns=["observation.state"])
        state = np.stack(df["observation.state"].values).astype(np.float64)
        episodes.append(state)
    return episodes


def pooled_values(episodes: list[np.ndarray], feat_idx: int) -> np.ndarray:
    """Flatten every frame's value for this feature across all episodes -- each frame is an
    equal dt, so this pooled set of values already represents time spent, unweighted."""
    return np.concatenate([e[:, feat_idx] for e in episodes])


def style_ax(ax):
    plt.rcParams.update({
        "figure.facecolor": STYLE["surface"], "axes.facecolor": STYLE["surface"],
        "savefig.facecolor": STYLE["surface"], "text.color": STYLE["primary"],
        "axes.labelcolor": STYLE["primary"], "axes.edgecolor": STYLE["muted"],
        "xtick.color": STYLE["muted"], "ytick.color": STYLE["muted"],
        "font.family": "sans-serif", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    ax.grid(True, linewidth=0.6, alpha=0.7, color=STYLE["grid"])
    ax.set_axisbelow(True)


def make_distribution_figure(feat_name, series_by_label, n_bins=60):
    """series_by_label: list of (label, color, episodes) for the same feature, overlaid on one axis.
    x = observation value, y = fraction of time spent at that value (density histogram + KDE)."""
    feat_idx = STATE_NAMES.index(feat_name)
    values_by_label = [(label, color, pooled_values(episodes, feat_idx)) for label, color, episodes in series_by_label]

    lo = min(v.min() for _, _, v in values_by_label)
    hi = max(v.max() for _, _, v in values_by_label)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.03 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, n_bins + 1)
    grid = np.linspace(lo - pad, hi + pad, 400)

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    style_ax(ax)
    for label, color, v in values_by_label:
        ax.hist(v, bins=bins, density=True, color=color, alpha=0.28, edgecolor="none", label=f"{label} (n={v.size:,})")
        if v.size > 1 and np.std(v) > 1e-12:
            kde = gaussian_kde(v)
            ax.plot(grid, kde(grid), color=color, lw=1.6)
    ax.set_xlabel(f"{feat_name} (value)")
    ax.set_ylabel("time density  [fraction of frames / unit]")
    ax.set_title(feat_name, fontsize=11.5)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    return fig


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def main():
    print("Loading episodes...")
    episodes_by_title = {}
    for title, turbulent, color in DATASETS:
        episodes = load_state_matrices(COLOR, turbulent)
        episodes_by_title[title] = (episodes, color)
        print(f"  {title}: {len(episodes)} episodes")

    series_by_label = [(title, color, episodes_by_title[title][0]) for title, _, color in DATASETS]

    pdf_path = "../../visualizations/red_rod_observation_averages.pdf"
    html_path = "../../visualizations/red_rod_observation_averages.html"

    cards = []
    with PdfPages(pdf_path) as pdf:
        for feat_name in STATE_NAMES:
            fig = make_distribution_figure(feat_name, series_by_label)
            pdf.savefig(fig)
            b64 = fig_to_b64(fig)
            plt.close(fig)
            cards.append(
                f'<div class="card"><img src="data:image/png;base64,{b64}" alt="{feat_name}"></div>'
            )
            print(f"    {feat_name} done")

    print("Writing HTML...")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Grab Red Rod — Value Distributions, Calm vs. Under Currents</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background:#f9f9f7; color:#0b0b0b;
    margin:0; padding: 28px 24px 60px; }}
  h1 {{ font-size:1.5rem; margin-bottom:4px; }}
  p.sub {{ color:#52514e; margin-top:0; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(380px,1fr)); gap:14px; margin-top:20px; }}
  .card {{ background:#fcfcfb; border:1px solid rgba(11,11,11,0.1); border-radius:10px; padding:8px; }}
  .card img {{ width:100%; height:auto; display:block; border-radius:4px; }}
</style></head>
<body>
  <h1>Grab Red Rod: time spent per value, calm vs. under currents</h1>
  <p class="sub">For each of the 30 <code>observation.state</code> fields: x-axis is the observation's
  value, y-axis is the fraction of time (all frames pooled across episodes, 50&nbsp;Hz) spent at that
  value — a density histogram with a KDE overlay. Calm (blue) and under-currents (red) superimposed on
  the same axes for direct comparison. {len(STATE_NAMES)} graphs total.</p>
  <div class="grid">{''.join(cards)}</div>
</body></html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
