from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
PKGROOT = ROOT.parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(PKGROOT))
from qvvw2.registry import METHODS
from qvvw2.plot_style import PLOT_LABELS as PLOT

obj = pd.read_csv(RES / "objectives_FINAL.csv")
ms = pd.read_csv(RES / "multistart_FINAL.csv")
sumdf = pd.read_csv(RES / "summary_FINAL.csv")
traj = pd.read_csv(RES / "trajectories_FINAL.csv")
hv = pd.read_csv(RES / "hv_high_precision_181.csv")
z = np.load(RES / "tdem_traces.npz")
T = z["T"]
depths = z["depths"]
starts = z["starts"]
obs = z["observed"]
cand = z["candidates"]
true_depth = float(z["true_depth"])

# --------------------------- numerical validation ---------------------------
issues = []
if len(depths) != 181:
    issues.append(f"depth count {len(depths)}")
if len(starts) != 23:
    issues.append(f"start count {len(starts)}")
if len(obj) != 181 * len(METHODS):
    issues.append(f"objective rows {len(obj)}")
if len(ms) != 23 * len(METHODS):
    issues.append(f"multistart rows {len(ms)}")
if set(obj.method) != set(METHODS):
    issues.append("objective methods mismatch")
if set(ms.method) != set(METHODS):
    issues.append("multistart methods mismatch")
if len(sumdf) != len(METHODS) or set(sumdf.method) != set(METHODS):
    issues.append("summary methods mismatch")
if len(hv) != 181:
    issues.append(f"HV rows {len(hv)}")

if traj.empty or set(traj.method) != set(METHODS):
    issues.append("trajectory methods mismatch")
else:
    grouped = traj.sort_values(["method", "start_id", "iteration"]).groupby(["method", "start_id"], sort=False)
    if len(grouped) != 23 * len(METHODS):
        issues.append(f"trajectory groups {len(grouped)}")
    endpoint_errors = []
    start_errors = []
    for (method, start_id), g in grouped:
        row = ms[(ms.method == method) & (ms.start_id == start_id)]
        if len(row) != 1:
            issues.append(f"trajectory multistart key mismatch {method} {start_id}")
            continue
        endpoint_errors.append(abs(float(g.iloc[-1].depth) - float(row.iloc[0].recovered_depth)))
        start_errors.append(abs(float(g.iloc[0].depth) - float(row.iloc[0].initial_depth)))
    if endpoint_errors and max(endpoint_errors) > 1e-12:
        issues.append(f"trajectory endpoint mismatch {max(endpoint_errors)}")
    if start_errors and max(start_errors) > 1e-12:
        issues.append(f"trajectory start mismatch {max(start_errors)}")
if not bool(hv["success"].all()):
    issues.append("HV inner failures")
if float(hv["opt_grad_inf"].max()) > 1e-6:
    issues.append(f"HV inner gradient max {hv.opt_grad_inf.max()}")
ho = obj[obj.method == "HV"].sort_values("depth")
if np.max(np.abs(ho.objective.to_numpy() - hv.sort_values("depth").objective.to_numpy())) > 1e-14:
    issues.append("HV objective mismatch")
q = sumdf[sumdf.method == "Q-vvW2"].iloc[0]
if int(q.n_local_minima) != 4:
    issues.append("Q-vvW2 minima count changed")
if abs(float(q.success_tol_0p01) - 19 / 23) > 1e-12:
    issues.append("Q-vvW2 multistart success changed")

validation = {
    "status": "PASS" if not issues else "FAIL",
    "issues": issues,
    "n_depths": len(depths),
    "n_starts": len(starts),
    "n_methods": len(METHODS),
    "methods": METHODS,
    "hv_inner_success_fraction": float(hv.success.mean()),
    "hv_max_opt_grad_inf": float(hv.opt_grad_inf.max()),
    "q_local_minima": int(q.n_local_minima),
    "q_success_rate": float(q.success_tol_0p01),
    "trajectory_rows": int(len(traj)),
    "trajectory_groups": int(traj.groupby(["method", "start_id"]).ngroups) if not traj.empty else 0,
}
(RES / "VALIDATION.json").write_text(json.dumps(validation, indent=2))
if issues:
    raise SystemExit(issues)


# ----------------------------- paper style ---------------------------------
from qvvw2.plot_style import (
    METHOD_COLORS, apply_paper_style, method_lw, panel_title,
    save_figure, style_axis,
)

apply_paper_style()

# Local/global-minimum metadata.
minima_map = {}
global_depth_map = {}
success_text = {}
for _, r in sumdf.iterrows():
    m = r["method"]
    txt = str(r["local_minima"]).strip()
    minima_map[m] = [] if txt in ("", "nan") else [float(x) for x in txt.split(";") if x]
    global_depth_map[m] = float(r["global_min_depth"])
    subset = ms[ms.method == m]
    success_text[m] = f"{int(subset.success.sum())}/{len(subset)}"

# ------------------------------- Figure 9 ----------------------------------
# Three equal-width super-panels and seven exactly aligned rows. Method names
# appear once only at the far left of panel (a), in black and without boxes.
fig = plt.figure(figsize=(10.50, 7.20), constrained_layout=False)
outer = fig.add_gridspec(
    1, 3, width_ratios=[1, 1, 1],
    left=0.105, right=0.985, top=0.945, bottom=0.155, wspace=0.12,
)
sg_a = outer[0, 0].subgridspec(len(METHODS), 1, hspace=0.08)
sg_b = outer[0, 1].subgridspec(len(METHODS), 1, hspace=0.08)
sg_c = outer[0, 2].subgridspec(len(METHODS), 1, hspace=0.08)
axs_a = [fig.add_subplot(sg_a[i, 0]) for i in range(len(METHODS))]
axs_b = [fig.add_subplot(sg_b[i, 0]) for i in range(len(METHODS))]
axs_c = [fig.add_subplot(sg_c[i, 0]) for i in range(len(METHODS))]

panel_title(axs_a[0], "(a) Signed TDEM traces")
panel_title(axs_b[0], "(b) Stacked objective landscapes")
panel_title(axs_c[0], "(c) Multistart recovery basins")

# (a) Observed trace versus the prediction at each method's objective global minimum.
amp = max(np.abs(obs).max(), np.abs(cand).max())
ylim_trace = (-1.08 * amp, 1.08 * amp)
for i, m in enumerate(METHODS):
    ax = axs_a[i]
    color = METHOD_COLORS[m]
    gdepth = global_depth_map[m]
    idx = int(np.argmin(np.abs(depths - gdepth)))
    pred = cand[idx]
    ax.plot(T, obs, color="0.15", lw=1.05)
    ax.plot(T, pred, color=color, lw=method_lw(m))
    ax.axhline(0.0, color="0.50", lw=0.55)
    ax.set_xlim(float(T.min()), float(T.max()))
    ax.set_ylim(*ylim_trace)
    ax.set_yticks([0.0])
    if i != len(METHODS) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time")
    if i == len(METHODS) // 2:
        ax.set_ylabel("Amplitude")
    ax.text(
        -0.115, 0.50, PLOT[m], transform=ax.transAxes,
        ha="right", va="center", color="black", fontsize=8.9,
    )
    style_axis(ax)

# (b) Within-method normalized objective landscapes.
for i, m in enumerate(METHODS):
    ax = axs_b[i]
    color = METHOD_COLORS[m]
    d = obj[obj.method == m].sort_values("depth")
    x = d.depth.to_numpy(float)
    y = d.objective.to_numpy(float)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-30)
    ax.axvspan(true_depth - 0.01, true_depth + 0.01, color="0.93", zorder=0)
    ax.axvline(true_depth, color="0.40", ls="--", lw=0.85, zorder=1)
    ax.plot(x, y, color=color, lw=method_lw(m), zorder=2)
    mins = minima_map[m]
    if mins:
        vals = np.interp(mins, x, y)
        ax.plot(mins, vals, linestyle="none", marker="o", ms=3.1,
                mfc="white", mec=color, mew=0.75, zorder=3)
    xg = global_depth_map[m]
    yg = float(np.interp(xg, x, y))
    ax.plot(xg, yg, linestyle="none", marker="o", ms=4.6,
            mfc=color, mec="white", mew=0.35, zorder=4)
    ax.set_xlim(0.335, 0.705)
    ax.set_ylim(-0.035, 1.035)
    ax.set_yticks([0.0, 1.0])
    if i != len(METHODS) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Slab depth")
    if i == len(METHODS) // 2:
        ax.set_ylabel("Within-method normalized objective")
    style_axis(ax)

# (c) Initial depth -> recovered depth basin map.
for i, m in enumerate(METHODS):
    ax = axs_c[i]
    color = METHOD_COLORS[m]
    d = ms[ms.method == m].sort_values("initial_depth")
    x = d.initial_depth.to_numpy(float)
    y = d.recovered_depth.to_numpy(float)
    succ = d.success.to_numpy(bool)
    ax.axhspan(true_depth - 0.01, true_depth + 0.01, color="0.93", zorder=0)
    ax.axhline(true_depth, color="0.40", ls="--", lw=0.85, zorder=1)
    ax.plot(x, y, color=color, lw=(1.70 if m == "Q-vvW2" else 1.12), zorder=2)
    ax.plot(x[~succ], y[~succ], linestyle="none", marker="o", ms=4.1,
            mfc="white", mec=color, mew=0.85, zorder=3)
    ax.plot(x[succ], y[succ], linestyle="none", marker="o", ms=4.45,
            mfc=color, mec="white", mew=0.30, zorder=4)
    ax.set_xlim(0.335, 0.705)
    ax.set_ylim(0.335, 0.705)
    ax.set_yticks([0.35, 0.52, 0.69])
    if i != len(METHODS) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Initial slab depth")
    if i == len(METHODS) // 2:
        ax.set_ylabel("Recovered slab depth")
    ax.text(0.965, 0.82, success_text[m], transform=ax.transAxes,
            ha="right", va="top", color="black", fontsize=7.8)
    style_axis(ax)

# Column-specific semantic legends; method names are not repeated.
leg_a = [
    Line2D([0], [0], color="0.15", lw=1.05, label="Observed (true)"),
    Line2D([0], [0], color="0.45", lw=1.15, label="Method prediction"),
]
axs_a[-1].legend(handles=leg_a, frameon=False, ncol=2, loc="upper center",
                 bbox_to_anchor=(0.50, -0.72), handlelength=2.2, columnspacing=1.15)

leg_b = [
    Line2D([0], [0], color="0.40", ls="--", lw=0.85, label="True depth (0.52)"),
    Patch(facecolor="0.93", edgecolor="none", label=r"True basin ($|z-0.52|<0.01$)"),
    Line2D([0], [0], marker="o", linestyle="none", mfc="white", mec="0.30", mew=0.75,
           markersize=4.0, label="Local minimum"),
    Line2D([0], [0], marker="o", linestyle="none", mfc="0.25", mec="white", mew=0.35,
           markersize=5.0, label="Global minimum"),
]
axs_b[-1].legend(handles=leg_b, frameon=False, ncol=2, loc="upper center",
                 bbox_to_anchor=(0.50, -0.70), handlelength=2.0, columnspacing=1.0, handletextpad=0.45)

leg_c = [
    Line2D([0], [0], marker="o", linestyle="none", mfc="0.25", mec="white", mew=0.30,
           markersize=4.8, label="Success"),
    Line2D([0], [0], marker="o", linestyle="none", mfc="white", mec="0.25", mew=0.85,
           markersize=4.4, label="Failure"),
    Line2D([0], [0], color="0.40", ls="--", lw=0.85, label="True depth (0.52)"),
    Patch(facecolor="0.93", edgecolor="none", label=r"True basin ($|z-0.52|<0.01$)"),
]
axs_c[-1].legend(handles=leg_c, frameon=False, ncol=2, loc="upper center",
                 bbox_to_anchor=(0.50, -0.70), handlelength=2.0, columnspacing=1.0, handletextpad=0.45)

png_path = FIG / "Fig09_TDEM_nonconvex_benchmark.png"
pdf_path = FIG / "Fig09_TDEM_nonconvex_benchmark.pdf"
save_figure(fig, png_path, pdf_path)
plt.close(fig)

print(json.dumps(validation, indent=2))
