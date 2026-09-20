from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PKGROOT = ROOT.parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(PKGROOT))
sys.path.insert(0, str(ROOT / "code"))
import run_e04 as E
from qvvw2.plot_style import (
    DISCRETE_MS, E04_COLORS, E04_MARKERS, FIGSIZE_1X2, SIGNED_FIELD_CMAP,
    WSPACE_1X2,
    apply_paper_style, equal_curve_box, method_lw, panel_title, save_figure,
    shared_legend, style_axis, style_colorbar,
)

LABEL = {
    "L2": r"$L^2$",
    "Normalized-L2": r"Normalized $L^2$",
    "vvW2": r"vv$W_2$",
    "Q-vvW2": r"Q-vv$W_2$",
}
WIDTH = {m: method_lw(m) for m in E.METHODS}

apply_paper_style()

true = E.mtrue.reshape(E.n, E.n)
models = {}
for case in ["clean_g1", "noise5_g1"]:
    for method in E.METHODS:
        models[(case, method)] = np.load(RES / f"{case}_{method}.npy")
v = max(abs(true).max(), *(abs(a).max() for a in models.values()))

# ---------------------------------------------------------------------------
# Figure 7: original E04 structure retained exactly in scientific content:
# clean reconstructions on the first row and 5% noise reconstructions on the
# second row, with the truth repeated so each row is independently readable.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(9.45, 5.45), constrained_layout=False)
gs = fig.add_gridspec(
    2, 6, width_ratios=[1, 1, 1, 1, 1, 0.055],
    left=0.055, right=0.965, top=0.95, bottom=0.105,
    wspace=0.11, hspace=0.22,
)
axs = np.empty((2, 5), dtype=object)
for i in range(2):
    for j in range(5):
        axs[i, j] = fig.add_subplot(gs[i, j])
cax = fig.add_subplot(gs[:, 5])

row1 = [
    ("(a) Clean: true", true),
    (r"(b) Clean: $L^2$", models[("clean_g1", "L2")]),
    (r"(c) Clean: Normalized $L^2$", models[("clean_g1", "Normalized-L2")]),
    (r"(d) Clean: vv$W_2$", models[("clean_g1", "vvW2")]),
    (r"(e) Clean: Q-vv$W_2$", models[("clean_g1", "Q-vvW2")]),
]
row2 = [
    ("(f) 5% noise: true", true),
    (r"(g) 5% noise: $L^2$", models[("noise5_g1", "L2")]),
    (r"(h) 5% noise: Normalized $L^2$", models[("noise5_g1", "Normalized-L2")]),
    (r"(i) 5% noise: vv$W_2$", models[("noise5_g1", "vvW2")]),
    (r"(j) 5% noise: Q-vv$W_2$", models[("noise5_g1", "Q-vvW2")]),
]
last_im = None
for r, row in enumerate([row1, row2]):
    for c, (title, data) in enumerate(row):
        ax = axs[r, c]
        last_im = ax.imshow(
            data.T, origin="lower", cmap=SIGNED_FIELD_CMAP, vmin=-v, vmax=v,
            extent=[0, 1, 0, 1], interpolation="nearest", aspect="equal",
        )
        panel_title(ax, title)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        # Keep numeric x tick labels visible in both rows so each reconstruction
        # is directly readable; use the axis symbol only on the bottom row to
        # avoid redundant text between rows.
        if r == 1:
            ax.set_xlabel(r"$x$")
        if c == 0:
            ax.set_ylabel(r"$y$")
        else:
            ax.tick_params(labelleft=False)
        style_axis(ax, grid=False)

cb = fig.colorbar(last_im, cax=cax)
style_colorbar(cb, r"Coefficient $m$")
save_figure(fig, FIG / "Fig07_elliptic_reconstructions.png",
            FIG / "Fig07_elliptic_reconstructions.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 8: E04 robustness.  The connected markers encode the three ordered
# gain/noise levels; panel (b) additionally shows the 12 raw Monte-Carlo seeds
# as faint points behind the mean +/- standard-deviation ribbon.
# ---------------------------------------------------------------------------
core = pd.read_csv(RES / "core_summary.csv")
mc = pd.read_csv(RES / "noise_monte_carlo_summary.csv")
raw_mc = pd.read_csv(RES / "noise_monte_carlo.csv")
fig, axs = plt.subplots(1, 2, figsize=FIGSIZE_1X2, constrained_layout=False)
fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.25, wspace=WSPACE_1X2)

ax = axs[0]
case_gain = {"clean_g05": 0.5, "clean_g1": 1.0, "clean_g2": 2.0}
for method in E.METHODS:
    d = core[(core.method == method) & (core.case.isin(case_gain))].copy()
    d["gain"] = d.case.map(case_gain)
    d = d.sort_values("gain")
    ax.plot(
        d.gain, d.relative_model_error,
        color=E04_COLORS[method], lw=WIDTH[method],
        marker=E04_MARKERS[method], ms=(5.0 if method == "Q-vvW2" else DISCRETE_MS),
        mfc=E04_COLORS[method], mec="white", mew=0.42,
        label=LABEL[method], zorder=(5 if method == "Q-vvW2" else 2),
    )
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([0.5, 1, 2])
ax.set_xticklabels(["0.5", "1", "2"])
panel_title(ax, "(a) Unknown positive-gain robustness")
ax.set_xlabel("Observation gain")
ax.set_ylabel("Relative model error")
style_axis(ax)
equal_curve_box(ax, 0.78)

ax = axs[1]
for method in E.METHODS:
    # Raw seed points: deterministic tiny horizontal offsets separate repeated
    # values without implying additional x-variation.
    r = raw_mc[raw_mc.method == method].copy()
    for noise, sub in r.groupby("noise_relative"):
        yy_raw = sub.relative_model_error.to_numpy(float)
        offsets = np.linspace(-0.09, 0.09, len(yy_raw))
        ax.scatter(100 * noise + offsets, yy_raw, s=8.0,
                   color=E04_COLORS[method], alpha=0.16, linewidths=0, zorder=1)

    d = mc[mc.method == method].sort_values("noise_relative")
    xx = 100 * d.noise_relative.to_numpy(float)
    yy = d.mean_error.to_numpy(float)
    sd = d.std_error.to_numpy(float)
    ax.plot(
        xx, yy, color=E04_COLORS[method], lw=WIDTH[method],
        marker=E04_MARKERS[method], ms=(5.0 if method == "Q-vvW2" else DISCRETE_MS),
        mfc=E04_COLORS[method], mec="white", mew=0.42,
        label=LABEL[method], zorder=(5 if method == "Q-vvW2" else 3),
    )
    low = np.maximum(yy - sd, 1e-14)
    high = yy + sd
    ax.fill_between(xx, low, high, color=E04_COLORS[method], alpha=0.11,
                    linewidth=0, zorder=2)
ax.set_yscale("log")
panel_title(ax, "(b) Noise robustness (12 seeds)")
ax.set_xlabel("Relative noise level (%)")
ax.set_ylabel("Relative model error")
ax.set_xticks([2, 5, 10])
style_axis(ax)
equal_curve_box(ax, 0.78)

handles, labels = axs[0].get_legend_handles_labels()
shared_legend(fig, handles, labels, ncol=4, y=0.048)
save_figure(fig, FIG / "Fig08_elliptic_robustness.png",
            FIG / "Fig08_elliptic_robustness.pdf")
plt.close(fig)
print("wrote Fig07 and Fig08")
