from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))
import distributed_wave_model as M
from qvvw2.registry import METHODS, DISPLAY_NAMES as DISPLAY
from qvvw2.plot_style import PLOT_LABELS as PLOT

SCENARIOS = ["clean", "gain_only", "source_mismatch", "gain_noise"]
COLORS = {
    "L2": "#1f77b4",
    "Normalized-L2": "#ff7f0e",
    "Softplus-W2sq": "#2ca02c",
    "Marginal-W2sq": "#d62728",
    "UOT": "#9467bd",
    "HV": "#8c564b",
    "Q-vvW2": "#111111",
}

# Validate the complete 42-run common protocol.
rows = []
issues = []
expected = []
for sc in SCENARIOS:
    for m in METHODS:
        expected.append((8, sc, m))
for P in [12, 16]:
    for m in METHODS:
        expected.append((P, "clean", m))

for P, sc, m in expected:
    d = RES / f"P{P}" / sc / m
    s = d / "summary.json"
    h = d / "history.csv"
    fm = d / "final_model.npy"
    fp = d / "final_pred.npy"
    if not all(x.exists() for x in [s, h, fm, fp]):
        issues.append(f"missing P{P}/{sc}/{m}")
        continue
    x = json.loads(s.read_text())
    hist = pd.read_csv(h)
    model = np.load(fm)
    pred = np.load(fp)
    re, corr = M.metrics(model)
    if len(hist) != 25:
        issues.append(f"iterations P{P}/{sc}/{m}={len(hist)}")
    if x.get("method") != m:
        issues.append(f"method label P{P}/{sc}/{m}: {x.get('method')}")
    if abs(re - x["relative_model_error"]) > 1e-10 or abs(corr - x["correlation"]) > 1e-10:
        issues.append(f"metric mismatch P{P}/{sc}/{m}")
    if not np.isfinite(hist.select_dtypes("number").to_numpy()).all():
        issues.append(f"nonfinite history P{P}/{sc}/{m}")
    if not np.isfinite(model).all() or not np.isfinite(pred).all():
        issues.append(f"nonfinite output P{P}/{sc}/{m}")
    rows.append(x)

summary = pd.DataFrame(rows)
# Add a relative-runtime column normalized by L^2 within each (P, scenario) group.
relative_runtime=[]
for _,row in summary.iterrows():
    base=summary[(summary.P==row.P)&(summary.scenario==row.scenario)&(summary.method=='L2')].algorithm_time_s
    relative_runtime.append(float(row.algorithm_time_s)/float(base.iloc[0]) if len(base) else np.nan)
summary['relative_runtime_vs_L2']=relative_runtime
summary.to_csv(RES / "summary_all_runs.csv", index=False)

# Machine-readable quantitative table used for the manuscript's single comparison table.
clean=summary[(summary.P==8)&(summary.scenario=='clean')].copy()
clean=clean.set_index('method').loc[METHODS].reset_index()
clean['display_name']=[DISPLAY[m].replace('$','').replace('\\','') for m in clean.method]
quant_cols=['method','display_name','relative_model_error','correlation','common_std_residual','algorithm_time_s','relative_runtime_vs_L2']
clean[quant_cols].to_csv(RES / 'P8_clean_quantitative_metrics.csv',index=False)

# QC summary for data/model gradients. The dedicated strict HV gate is the
# authoritative HV finite-difference check because the inner variational solve
# must be converged more tightly than in the generic screening gate.
gates = json.loads((RES / "gradient_gates.json").read_text())
gate_best = {g["method"]: float(g["best_relative_error"]) for g in gates}
hv_strict = json.loads((RES / "hv_strict_gradient_gate.json").read_text())
hv_strict_best = min(float(r[2]) for r in hv_strict["rows"])
for m in ["L2", "Normalized-L2", "Softplus-W2sq", "Marginal-W2sq", "Q-vvW2"]:
    if gate_best.get(m, 1.0) > 1e-6:
        issues.append(f"gradient gate {m}={gate_best.get(m)}")
if gate_best.get("UOT", 1.0) > 5e-5:
    issues.append(f"gradient gate UOT={gate_best.get('UOT')}")
if hv_strict_best > 5e-5:
    issues.append(f"strict HV gradient gate={hv_strict_best}")

validation = {
    "status": "PASS" if not issues else "FAIL",
    "expected_runs": len(expected),
    "validated_runs": len(rows),
    "methods": METHODS,
    "issues": issues,
    "gradient_best_relative_error": gate_best,
    "hv_strict_best_relative_error": hv_strict_best,
    "quantitative_table": "P8_clean_quantitative_metrics.csv",
    "timing_label": "algorithm_time_s; relative_runtime_vs_L2 normalized by the L2 run in the same P/scenario group",
}
(RES / "VALIDATION.json").write_text(json.dumps(validation, indent=2))
if issues:
    raise SystemExit(issues)


from qvvw2.plot_style import (
    DISCRETE_MS, FIELD_CMAP, FIGSIZE_1X2, INITIAL_COLOR, METHOD_COLORS, METHOD_MARKERS,
    TRUE_COLOR, WSPACE_1X2, apply_paper_style, equal_curve_box, method_lw, panel_title,
    save_figure, shared_legend, style_axis, style_colorbar,
)

apply_paper_style()

true = M.ctr
initial = np.ones_like(true)
models = {m: np.load(RES / "P8" / "clean" / m / "final_model.npy") for m in METHODS}
vmin = min(true.min(), initial.min(), *(a.min() for a in models.values()))
vmax = max(true.max(), initial.max(), *(a.max() for a in models.values()))

# ---------------------------------------------------------------------------
# Figure 4: blueprint-style reconstruction matrix.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(9.45, 7.65), constrained_layout=False)
gs = fig.add_gridspec(
    3, 4, width_ratios=[1, 1, 1, 0.055],
    left=0.075, right=0.955, top=0.965, bottom=0.075,
    wspace=0.12, hspace=0.17,
)
axs = np.empty((3, 3), dtype=object)
for i in range(3):
    for j in range(3):
        axs[i, j] = fig.add_subplot(gs[i, j])
cax = fig.add_subplot(gs[:, 3])
items = [("(a) True", true), ("(b) Initial", initial)]
for letter, m in zip("cdefghi", METHODS):
    items.append((f"({letter}) {PLOT[m]}", models[m]))
last_im = None
for k, (ax, (title, arr)) in enumerate(zip(axs.ravel(), items)):
    last_im = ax.imshow(
        arr.T, origin="lower", cmap=FIELD_CMAP, vmin=vmin, vmax=vmax,
        extent=[0, 1, 0, 1], aspect="equal", interpolation="nearest",
    )
    panel_title(ax, title)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    if k // 3 == 2:
        ax.set_xlabel("x")
    else:
        ax.tick_params(labelbottom=False)
    if k % 3 == 0:
        ax.set_ylabel("z")
    else:
        ax.tick_params(labelleft=False)
    style_axis(ax, grid=False)
cbar = fig.colorbar(last_im, cax=cax)
style_colorbar(cbar, "Model coefficient")
save_figure(fig, FIG / "Fig04_distributed_reconstruction.png",
            FIG / "Fig04_distributed_reconstruction.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5: a single convergence-behaviour panel with a terminal-region inset.
# This avoids repeating the spatial reconstructions already shown in Fig. 4.
# ---------------------------------------------------------------------------
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
fig, axc = plt.subplots(1, 1, figsize=(9.45, 5.10), constrained_layout=False)
fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.24)
history_cache = {}
for m in METHODS:
    h = pd.read_csv(RES / "P8" / "clean" / m / "history.csv")
    history_cache[m] = h
    axc.plot(h.iteration, h.rel_error_postupdate,
             color=METHOD_COLORS[m], lw=method_lw(m), label=PLOT[m],
             zorder=(5 if m == "Q-vvW2" else 2))
axc.set_xlabel("Iteration")
axc.set_ylabel("Relative model error")
style_axis(axc)
equal_curve_box(axc, 0.68)

# Terminal inset: same data, enlarged terminal iterations.  The limits are
# determined from the plotted histories and therefore do not alter the data.
iax = inset_axes(axc, width="37%", height="36%", loc="upper right", borderpad=1.0)
terminal_start = max(0, min(int(h.iteration.max()) for h in history_cache.values()) - 8)
terminal_values = []
for m in METHODS:
    h = history_cache[m]
    d = h[h.iteration >= terminal_start]
    iax.plot(d.iteration, d.rel_error_postupdate,
             color=METHOD_COLORS[m], lw=(1.45 if m == "Q-vvW2" else 0.95),
             zorder=(5 if m == "Q-vvW2" else 2))
    terminal_values.extend(d.rel_error_postupdate.to_numpy(float).tolist())
lo, hi = min(terminal_values), max(terminal_values)
pad = 0.08 * max(hi - lo, 1e-6)
iax.set_xlim(terminal_start, max(int(h.iteration.max()) for h in history_cache.values()))
iax.set_ylim(lo - pad, hi + pad)
iax.set_title("Terminal region", fontsize=7.6, pad=2.0)
iax.tick_params(labelsize=6.8, direction="in", top=True, right=True)
for sp in iax.spines.values():
    sp.set_linewidth(0.75)
iax.grid(alpha=0.10, lw=0.45)

handles, labels = axc.get_legend_handles_labels()
shared_legend(fig, handles, labels, ncol=4, y=0.045)
save_figure(fig, FIG / "Fig05_distributed_diagnostics.png",
            FIG / "Fig05_distributed_diagnostics.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6: representative structural profile + four-scenario robustness
# small multiples.  In panel (b), marker position is the only quantitative
# encoding; the faint horizontal swatch is a fixed-size visual glyph and does
# not represent uncertainty.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=FIGSIZE_1X2, constrained_layout=False)
outer = fig.add_gridspec(
    2, 2, width_ratios=[0.94, 1.52], height_ratios=[1.0, 0.22],
    left=0.065, right=0.985, top=0.82, bottom=0.08,
    wspace=max(WSPACE_1X2, 0.24), hspace=0.08,
)
x = np.linspace(0, 1, true.shape[0])
mid = true.shape[0] // 2

# (a) Structural profile.
ax = fig.add_subplot(outer[0, 0])
true_profile = true[:, mid]
ax.plot(x, true_profile, color=TRUE_COLOR, ls="--", lw=1.55, label="True", zorder=7)
for m in METHODS:
    ax.plot(x, models[m][:, mid], color=METHOD_COLORS[m], lw=method_lw(m),
            label=PLOT[m], zorder=(6 if m == "Q-vvW2" else 2))
# Neutral structural bands are determined directly from the true profile.
mask = np.abs(true_profile - 1.0) > 0.02
if mask.any():
    runs = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    for k, (i0, i1) in enumerate(runs):
        left = max(0.0, x[i0] - 0.5 / (len(x)-1))
        right = min(1.0, x[i1] + 0.5 / (len(x)-1))
        ax.axvspan(left, right, color="0.92", zorder=0)
        ax.text(0.5*(left+right), 0.97, f"Anomaly {k+1}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.0, color="0.35")
ax.set_xlabel("x")
ax.set_ylabel("Model coefficient")
style_axis(ax)

# (b) Four aligned robustness strips matching the selected blueprint style.
sg = outer[0, 1].subgridspec(1, 4, wspace=0.20)
scenario_labels = ["Clean", "Gain", "Source", "Gain+noise"]
scenario_axes = []
all_vals = []
scenario_values = {}
for sc, title in zip(SCENARIOS, scenario_labels):
    d = {}
    for m in METHODS:
        v = json.loads((RES / "P8" / sc / m / "summary.json").read_text())["relative_model_error"]
        d[m] = float(v)
        all_vals.append(float(v))
    scenario_values[sc] = d
xmin, xmax = min(all_vals), max(all_vals)
pad = 0.06 * max(xmax - xmin, 1e-6)
xlim = (max(0.0, xmin - pad), xmax + pad)

ypos = np.arange(len(METHODS))
for j, (sc, title) in enumerate(zip(SCENARIOS, scenario_labels)):
    a = fig.add_subplot(sg[0, j])
    scenario_axes.append(a)
    for iy, m in enumerate(METHODS):
        v = scenario_values[sc][m]
        # fixed-size swatch in screen units (underscore marker), then the true
        # quantitative marker at the same x location.
        a.plot([v], [iy], linestyle="none", marker="_", ms=20,
               mew=5.0, color=METHOD_COLORS[m], alpha=0.22, zorder=2)
        a.plot([v], [iy], linestyle="none", marker=METHOD_MARKERS[m],
               ms=(5.4 if m == "Q-vvW2" else 4.8),
               mfc=METHOD_COLORS[m], mec=("white" if m != "HV" else METHOD_COLORS[m]),
               mew=0.45, color=METHOD_COLORS[m], zorder=4)
        a.axhline(iy, color="0.88", lw=0.55, zorder=0)
    a.set_xlim(*xlim)
    a.set_ylim(len(METHODS)-0.5, -0.5)
    a.set_title(title, fontsize=9.0, pad=4.0)
    a.set_xlabel("Terminal relative\nmodel error")
    if j == 0:
        a.set_yticks(ypos, [PLOT[m] for m in METHODS])
    else:
        a.set_yticks(ypos, [])
    style_axis(a, grid=False)
    # No top/right duplicate tick labels; the four panels share the same scale.
    a.tick_params(top=False, right=False)

# Strictly align the two main-panel titles on the same horizontal baseline.
fig.canvas.draw()
pos_a = ax.get_position()
pos0 = scenario_axes[0].get_position()
pos1 = scenario_axes[-1].get_position()
title_y = max(pos_a.y1, pos0.y1) + 0.055
fig.text(0.5 * (pos_a.x0 + pos_a.x1), title_y,
         "(a) Representative horizontal profile",
         ha="center", va="bottom", fontsize=9.6, color="black")
fig.text(0.5 * (pos0.x0 + pos1.x1), title_y,
         "(b) Robustness to perturbations",
         ha="center", va="bottom", fontsize=9.6, color="black")

# The curve legend belongs only to panel (a). Put it in a dedicated row
# beneath panel (a) so the two main panels remain equal in height.
handles, labels = ax.get_legend_handles_labels()
legax = fig.add_subplot(outer[1, 0])
legax.axis("off")
# Place the legend lower in its dedicated row so it stays clearly separated
# from the x-axis label of panel (a).
legax.legend(
    handles, labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.38),
    ncol=3, frameon=False, handlelength=2.15,
    columnspacing=0.90, handletextpad=0.38, borderaxespad=0.0,
)
# Blank companion cell under panel (b) to preserve symmetric layout.
blankax = fig.add_subplot(outer[1, 1])
blankax.axis("off")
save_figure(fig, FIG / "Fig06_distributed_profiles_scaling_robustness.png",
            FIG / "Fig06_distributed_profiles_scaling_robustness.pdf")
plt.close(fig)

print('\n'+'='*112)
print('E03 QUANTITATIVE COMPARISON | P=8 (64 unknowns) | scenario=clean')
print('Timing field: algorithm_time_s (accumulated measured algorithm time); relative runtime is normalized by L^2.')
print('-'*112)
print(f"{'Method':<24}{'Rel. model error':>18}{'Correlation':>15}{'Std. residual':>16}{'Time (s)':>14}{'vs L^2':>12}")
print('-'*112)
console_names={'L2':'L^2','Normalized-L2':'Normalized L^2','Softplus-W2sq':'Softplus-W_2^2','Marginal-W2sq':'Marginal-W_2^2','UOT':'UOT','HV':'HV metric','Q-vvW2':'Q-vvW_2'}
for _,r in clean.iterrows():
    print(f"{console_names[r['method']]:<24}{r['relative_model_error']:>18.6f}{r['correlation']:>15.6f}{r['common_std_residual']:>16.6f}{r['algorithm_time_s']:>14.3f}{r['relative_runtime_vs_L2']:>11.3f}x")
print('='*112)
print('\nVALIDATION')
print(json.dumps(validation, indent=2))
