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
import run_e02 as E
from qvvw2.registry import METHODS
from qvvw2.plot_style import PLOT_LABELS as PLOT
from qvvw2.plot_style import (
    FIGSIZE_2X2, HSPACE_2X2, WSPACE_2X2, METHOD_COLORS, RESPONSE_CMAP,
    apply_paper_style, equal_curve_box,
    method_lw, panel_title, save_figure, style_axis, style_colorbar,
)


def load(s):
    a = pd.read_csv(RES / f"{s}_nonhv.csv")
    h = pd.read_csv(RES / f"{s}_hv.csv")[["parameter", "HV"]]
    return a.merge(h, on="parameter", how="inner").sort_values("parameter")


def normalized(v):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)


def response_matrix(df):
    return np.vstack([normalized(df[m].to_numpy(float)) for m in METHODS])

def centers_to_edges(x):
    x = np.asarray(x, float)
    mid = 0.5 * (x[:-1] + x[1:])
    return np.r_[x[0] - (mid[0] - x[0]), mid, x[-1] + (x[-1] - mid[-1])]


D = {s: load(s) for s in ["gain", "ratio", "shift", "signed_gain"]}
truth = {"gain": 1.0, "ratio": 0.65, "shift": 0.0, "signed_gain": 1.0}
rows = []
for s, df in D.items():
    x = df.parameter.to_numpy()
    for m in METHODS:
        v = df[m].to_numpy()
        j = int(np.nanargmin(v))
        rows.append({
            "sweep": s,
            "method": m,
            "argmin": float(x[j]),
            "min_value": float(v[j]),
            "truth": truth[s],
            "argmin_error": float(abs(x[j] - truth[s])),
        })
pd.DataFrame(rows).to_csv(RES / "sweep_summary.csv", index=False)

apply_paper_style()
fig, ax = plt.subplots(2, 2, figsize=FIGSIZE_2X2, constrained_layout=False)
fig.subplots_adjust(left=0.105, right=0.925, top=0.955, bottom=0.085,
                    wspace=WSPACE_2X2, hspace=HSPACE_2X2)
t, ref = E.T, E.REF

# (a) Blueprint-like representative signed waveform comparison.
p = ax[0, 0]
p.plot(t, ref, color="#111111", lw=1.55, label="Reference")
p.plot(t, E.shift_signal(0.08), color="#DC2626", lw=1.35, ls="--",
       label=r"Shifted $+0.08$")
p.axhline(0, color="0.55", lw=0.60)
panel_title(p, "(a) Representative signed waveforms")
p.set_xlabel("Normalized time")
p.set_ylabel("Amplitude")
p.legend(frameon=False, loc="upper right", handlelength=2.0)
style_axis(p)
equal_curve_box(p)

# (b) Seven-method positive-gain response; continuous curves have no markers.
p = ax[0, 1]
x = D["gain"].parameter.to_numpy(float)
for m in METHODS:
    v = D["gain"][m].to_numpy(float)
    j = np.argmin(abs(x - 1.0))
    p.plot(
        x, np.maximum(np.abs(v - v[j]), 1e-32),
        color=METHOD_COLORS[m], lw=method_lw(m), label=PLOT[m],
        zorder=(5 if m == "Q-vvW2" else 2),
    )
p.axvline(1.0, color="0.45", ls="--", lw=0.80)
p.set_xscale("log")
p.set_yscale("log")
panel_title(p, "(b) Positive-scale response")
p.set_xlabel(r"Positive gain $c$")
p.set_ylabel(r"$|J(cd)-J(d)|$")
p.legend(frameon=False, loc="center right", ncol=1, fontsize=6.7,
         handlelength=1.8, columnspacing=0.8, handletextpad=0.35,
         borderaxespad=0.55)
style_axis(p)
equal_curve_box(p)

# (c) Response map: row-wise normalized objectives preserve each method's
# landscape shape without implying that raw objective magnitudes are comparable.
p = ax[1, 0]
xr = D["ratio"].parameter.to_numpy(float)
Mr = response_matrix(D["ratio"])
mesh1 = p.pcolormesh(centers_to_edges(xr), np.arange(len(METHODS) + 1) - 0.5, Mr,
                     shading="flat", cmap=RESPONSE_CMAP, vmin=0.0, vmax=1.0)
p.axvline(0.65, color="white", ls="--", lw=1.00)
for iy, m in enumerate(METHODS):
    vals = D["ratio"][m].to_numpy(float)
    j = int(np.nanargmin(vals))
    p.plot([xr[j]], [iy], marker="o", ms=3.3, mfc="black", mec="white",
           mew=0.35, linestyle="none", zorder=5)
p.set_yticks(np.arange(len(METHODS)), [PLOT[m] for m in METHODS])
p.set_ylim(len(METHODS) - 0.5, -0.5)
panel_title(p, "(c) Relative signed-amplitude response")
p.set_xlabel(r"Relative negative-event amplitude $r$")
style_axis(p, grid=False)
equal_curve_box(p)

# (d) Time-shift response map using the identical normalization and color scale.
p = ax[1, 1]
xs = D["shift"].parameter.to_numpy(float)
Ms = response_matrix(D["shift"])
mesh2 = p.pcolormesh(centers_to_edges(xs), np.arange(len(METHODS) + 1) - 0.5, Ms,
                     shading="flat", cmap=RESPONSE_CMAP, vmin=0.0, vmax=1.0)
p.axvline(0.0, color="white", ls="--", lw=1.00)
for iy, m in enumerate(METHODS):
    vals = D["shift"][m].to_numpy(float)
    j = int(np.nanargmin(vals))
    p.plot([xs[j]], [iy], marker="o", ms=3.3, mfc="black", mec="white",
           mew=0.35, linestyle="none", zorder=5)
p.set_yticks(np.arange(len(METHODS)), [PLOT[m] for m in METHODS])
p.set_ylim(len(METHODS) - 0.5, -0.5)
panel_title(p, "(d) Time-shift response")
p.set_xlabel("Time shift / record length")
style_axis(p, grid=False)
equal_curve_box(p)

# One shared response colorbar for the two map panels, placed in its own axis so
# neither data panel changes size.
cax = fig.add_axes([0.942, 0.105, 0.014, 0.315])
cb = fig.colorbar(mesh2, cax=cax)
style_colorbar(cb, "Row-normalized objective")

save_figure(fig, FIG / "Fig03_signed_waveform_mechanism.png",
            FIG / "Fig03_signed_waveform_mechanism.pdf")
plt.close(fig)
print("wrote Figure 3")
