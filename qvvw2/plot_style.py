"""Shared publication plotting style for the Q-vvW2 paper figures.

The visual grammar follows the paper's final figure blueprint: restrained white
background, Times/STIX typography, vivid but consistent method colors, black
emphasis for Q-vvW2, circular markers only for genuinely discrete samples, and
shared legends/colorbars that never resize selected data axes.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Blueprint-like method palette.  Method identity is carried primarily by color;
# line style is reserved for semantic references (truth, tolerance, etc.).
METHOD_COLORS = {
    "L2": "#2563EB",              # vivid blue
    "Normalized-L2": "#F97316",  # orange
    "Softplus-W2sq": "#16A34A",  # green
    "Marginal-W2sq": "#DC2626",  # red
    "UOT": "#7C3AED",             # purple
    "HV": "#9A5B3E",              # brown
    "Q-vvW2": "#111111",          # black emphasis
}

E04_COLORS = {
    "L2": METHOD_COLORS["L2"],
    "Normalized-L2": METHOD_COLORS["Normalized-L2"],
    "vvW2": METHOD_COLORS["Softplus-W2sq"],
    "Q-vvW2": METHOD_COLORS["Q-vvW2"],
}

PLOT_LABELS = {
    "L2": r"$L^2$",
    "Normalized-L2": r"Normalized $L^2$",
    "Softplus-W2sq": r"Softplus-$W_2^2$",
    "Marginal-W2sq": r"Marginal-$W_2^2$",
    "UOT": "UOT",
    "HV": "HV metric",
    "Q-vvW2": r"Q-vv$W_2$",
}

# Reference/semantic colors.
TRUE_COLOR = "#111111"
INITIAL_COLOR = "#8A8A8A"
REFERENCE_GRAY = "#666666"
TOLERANCE_GRAY = "#E8E8E8"

# Line hierarchy mirrors the blueprint: Q-vvW2 is slightly heavier, but the
# baseline lines remain visually comparable.
BASELINE_LW = 1.35
QVVW2_LW = 1.95
REFERENCE_LW = 1.45
INITIAL_LW = 1.15
AXIS_LW = 0.86
GRID_ALPHA = 0.12

# The blueprint uses small circular samples for discrete sweeps.  Continuous
# curves intentionally have no markers.
METHOD_MARKERS = {
    "L2": "s",
    "Normalized-L2": "^",
    "Softplus-W2sq": "D",
    "Marginal-W2sq": "v",
    "UOT": "P",
    "HV": "X",
    "Q-vvW2": "o",
}
E04_MARKERS = {
    "L2": "s",
    "Normalized-L2": "^",
    "vvW2": "D",
    "Q-vvW2": "o",
}
DISCRETE_MARKER = "o"  # fallback only
DISCRETE_MS = 4.0
DISCRETE_MEW = 0.42
THEORY_MS = 3.6

THEORY_BLACK = "#111111"
THEORY_BLUE = "#2563EB"
THEORY_ORANGE = "#F97316"
THEORY_GRAY = "#808080"

# Shared layout templates. Figures with the same panel geometry use the same
# canvas size and inter-panel spacing so manuscript placement is consistent.
FIGSIZE_2X2 = (9.45, 7.0)
WSPACE_2X2 = 0.22
HSPACE_2X2 = 0.28
FIGSIZE_1X2 = (9.45, 4.55)
WSPACE_1X2 = 0.30

# Shared colormap choices.  Spatial inversion fields keep the established jet/
# seismic conventions; response maps use a blue-cyan-yellow-red map like the
# blueprint.
FIELD_CMAP = "jet"
SIGNED_FIELD_CMAP = "seismic"
RESPONSE_CMAP = "turbo"
ERROR_CMAP = "turbo"


def apply_paper_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 350,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.titlesize": 9.6,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.2,
        "axes.linewidth": AXIS_LW,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": AXIS_LW,
        "ytick.major.width": AXIS_LW,
        "xtick.minor.width": 0.65,
        "ytick.minor.width": 0.65,
        "xtick.major.size": 4.1,
        "ytick.major.size": 4.1,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "lines.linewidth": BASELINE_LW,
        "lines.markersize": DISCRETE_MS,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def method_lw(method: str) -> float:
    return QVVW2_LW if method == "Q-vvW2" else BASELINE_LW


def style_axis(ax, *, grid: bool = True, top_right_ticks: bool = True) -> None:
    ax.tick_params(
        which="both", direction="in", top=top_right_ticks, right=top_right_ticks,
        width=AXIS_LW,
    )
    for s in ax.spines.values():
        s.set_linewidth(AXIS_LW)
    if grid:
        ax.grid(alpha=GRID_ALPHA, lw=0.52, color="0.50")
    else:
        ax.grid(False)


def style_3d_axis(ax) -> None:
    """Lightweight styling for the Fig. 1 geometry panel."""
    ax.tick_params(labelsize=8.0, pad=1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        except Exception:
            pass
    ax.grid(False)


def panel_title(ax, text: str) -> None:
    """Place every formal subfigure title centered directly above its own axes."""
    ax.set_title(text, loc="center", pad=5.0, color="black")


def equal_curve_box(ax, aspect: float = 0.78) -> None:
    """Give curve/map panels a consistent plotting-box aspect ratio."""
    try:
        ax.set_box_aspect(aspect)
    except Exception:
        pass


def method_marker(method: str) -> str:
    return METHOD_MARKERS.get(method, DISCRETE_MARKER)

def discrete_method_plot(ax, x, y, method: str, *, label=None, zorder=None):
    color = METHOD_COLORS[method]
    return ax.plot(
        x, y,
        color=color,
        lw=method_lw(method),
        marker=method_marker(method),
        ms=(4.8 if method == "Q-vvW2" else DISCRETE_MS),
        mfc=color,
        mec="white",
        mew=DISCRETE_MEW,
        label=label,
        zorder=(5 if method == "Q-vvW2" else 2) if zorder is None else zorder,
    )


def style_colorbar(cbar, label: str | None = None) -> None:
    if label:
        cbar.set_label(label)
    cbar.ax.tick_params(direction="in", labelsize=8.0, width=0.8)
    cbar.outline.set_linewidth(0.8)


def save_figure(fig, png_path: Path, pdf_path: Path) -> None:
    """Save on the fixed figure canvas.

    We intentionally do not use ``bbox_inches="tight"`` here.  Tight
    bounding boxes change the exported physical size according to tick labels,
    legends, and colorbars, which makes nominally identical 2x2 or 1x2 plates
    come out at different dimensions.  All paper figures use explicit margins,
    so the fixed canvas is both safer for manuscript layout and reproducible.
    """
    fig.savefig(png_path, dpi=350, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")


def shared_legend(fig, handles, labels, *, ncol: int = 4, y: float = 0.01) -> None:
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, y),
        ncol=ncol, frameon=False, handlelength=2.25,
        columnspacing=1.10, handletextpad=0.42, borderaxespad=0.0,
    )
