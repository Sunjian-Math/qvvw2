from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

from qvvw2.core import frozen_objective_grad, frozen_setup, jacobian_dense, lift

EPSILON = 0.15
N = 160


def sci_tex(value, digits=1):
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"


def main():
    print("[E00] Starting quotient-geometry experiment", flush=True)
    t = np.linspace(0.0, 1.0, N)
    positive_event = np.exp(-0.5 * ((t - 0.30) / 0.055) ** 2)
    negative_event = 0.65 * np.exp(-0.5 * ((t - 0.70) / 0.075) ** 2)
    datum = positive_event - negative_event

    rho, _ = lift(datum, EPSILON)
    selected_scales = np.array([0.35, 0.65, 1.0, 1.7, 3.0])
    dense_scales = np.geomspace(0.15, 6.0, 61)
    lift_gaps = []
    q_values = []
    radial_residuals = []

    print(f"[E00] Evaluating positive-scale orbit on {len(dense_scales)} scales", flush=True)
    setup = frozen_setup(datum, EPSILON, (N,), 0.03)
    for scale in dense_scales:
        scaled = scale * datum
        rho_scaled, _ = lift(scaled, EPSILON)
        lift_gaps.append(np.linalg.norm(rho_scaled - rho))
        q_values.append(frozen_objective_grad(scaled, setup)[0])
        jac_scaled = jacobian_dense(scaled, EPSILON)
        radial_residuals.append(
            np.linalg.norm(jac_scaled @ scaled)
            / max(np.linalg.norm(jac_scaled) * np.linalg.norm(scaled), 1e-30)
        )

    jac = jacobian_dense(datum, EPSILON)
    radial_jacobian_relative_residual = float(
        np.linalg.norm(jac @ datum)
        / max(np.linalg.norm(jac) * np.linalg.norm(datum), 1e-30)
    )

    nonradial = np.gradient(datum)
    nonradial -= datum * (datum @ nonradial) / (datum @ datum)
    nonradial /= np.linalg.norm(nonradial)
    nonradial_jacobian_response = float(np.linalg.norm(jac @ nonradial))

    relative_perturbation = np.linspace(-0.45, 0.45, 121)
    datum_norm = np.linalg.norm(datum)
    radial_lift_distance = []
    nonradial_lift_distance = []
    for a in relative_perturbation:
        radial_point = (1.0 + a) * datum
        nonradial_point = datum + a * datum_norm * nonradial
        radial_lift_distance.append(
            np.linalg.norm(lift(radial_point, EPSILON)[0] - rho)
        )
        nonradial_lift_distance.append(
            np.linalg.norm(lift(nonradial_point, EPSILON)[0] - rho)
        )

    selected_q = [float(frozen_objective_grad(c * datum, setup)[0]) for c in selected_scales]
    status = (
        "PASS"
        if max(lift_gaps) < 1e-12
        and max(abs(np.asarray(q_values))) < 1e-12
        and radial_jacobian_relative_residual < 1e-12
        and nonradial_jacobian_response > 1e-6
        else "FAIL"
    )

    summary = {
        "status": status,
        "epsilon": EPSILON,
        "max_lift_difference_under_positive_scale": float(max(lift_gaps)),
        "max_absolute_q_objective_under_positive_scale": float(max(abs(np.asarray(q_values)))),
        "max_radial_jacobian_relative_residual": float(max(radial_residuals)),
        "radial_jacobian_relative_residual_at_reference": radial_jacobian_relative_residual,
        "nonradial_jacobian_response": nonradial_jacobian_response,
        "q_objectives_selected_positive_scales": selected_q,
    }
    (RESULTS / "geometry_checks.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        RESULTS / "geometry_data.npz",
        t=t,
        datum=datum,
        positive_event=positive_event,
        negative_event=-negative_event,
        rho_plus=rho[:N],
        rho_minus=rho[N:],
        selected_scales=selected_scales,
        dense_scales=dense_scales,
        lift_gaps=np.asarray(lift_gaps),
        q_values=np.asarray(q_values),
        radial_residuals=np.asarray(radial_residuals),
        relative_perturbation=relative_perturbation,
        radial_lift_distance=np.asarray(radial_lift_distance),
        nonradial_lift_distance=np.asarray(nonradial_lift_distance),
    )

    print("[E00] Numerical gates complete; generating Fig. 1", flush=True)
    make_figure(
        t,
        datum,
        rho,
        selected_scales,
        dense_scales,
        lift_gaps,
        relative_perturbation,
        radial_lift_distance,
        nonradial_lift_distance,
        radial_jacobian_relative_residual,
    )

    print(json.dumps(summary, indent=2))
    if status != "PASS":
        raise RuntimeError("E00 validation failed")


def make_figure(
    t,
    datum,
    rho,
    selected_scales,
    dense_scales,
    lift_gaps,
    relative_perturbation,
    radial_lift_distance,
    nonradial_lift_distance,
    radial_residual,
):
    """Figure 1: four-panel signed-data / quotient-geometry diagnostic."""
    from qvvw2.plot_style import (FIGSIZE_2X2, HSPACE_2X2, WSPACE_2X2,
                                  apply_paper_style, panel_title, save_figure, style_axis)

    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.095, wspace=WSPACE_2X2, hspace=HSPACE_2X2)

    ax = axes[0, 0]
    ax.plot(t, datum, color="black", lw=1.70, label=r"$d(t)$")
    ax.fill_between(t, 0.0, np.maximum(datum, 0.0), color="#1f77b4", alpha=0.24, label=r"$d^+$")
    ax.fill_between(t, 0.0, np.minimum(datum, 0.0), color="#ff7f0e", alpha=0.24, label=r"$d^-$")
    ax.axhline(0.0, color="0.45", lw=0.70)
    panel_title(ax, "(a) Signed datum and species split")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Amplitude")
    ax.legend(frameon=False, loc="upper right", handlelength=2.4)
    style_axis(ax)

    ax = axes[0, 1]
    cmap = plt.get_cmap("viridis")
    for j, scale in enumerate(selected_scales):
        ax.plot(
            t,
            scale * datum,
            lw=1.25,
            color=cmap(j / (len(selected_scales) - 1)),
            label=rf"$c={scale:g}$",
        )
    ax.axhline(0.0, color="0.45", lw=0.70)
    panel_title(ax, "(b) Positive-scale orbit in signal space")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Amplitude")
    ax.legend(frameon=False, ncol=2, loc="upper right", columnspacing=1.1, handlelength=2.2)
    style_axis(ax)

    ax = axes[1, 0]
    for scale in selected_scales:
        rho_scaled, _ = lift(scale * datum, EPSILON)
        alpha = 0.45 if scale != 1.0 else 1.0
        lw = 1.0 if scale != 1.0 else 1.8
        ax.plot(t, rho_scaled[:N], color="tab:blue", alpha=alpha, lw=lw)
        ax.plot(t, rho_scaled[N:], color="tab:orange", alpha=alpha, lw=lw)
    ax.plot([], [], color="tab:blue", lw=1.7, label="positive species")
    ax.plot([], [], color="tab:orange", lw=1.7, label="negative species")
    panel_title(ax, "(c) Quotient lift collapses the scale orbit")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Lifted mass")
    ax.legend(frameon=False, loc="upper right", handlelength=2.2)
    ax.text(
        0.96,
        0.74,
        rf"$\max_c\|\rho(cd)-\rho(d)\|_2={sci_tex(max(lift_gaps), 1)}$",
        transform=ax.transAxes,
        fontsize=7.2,
        ha="right",
        va="top",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 1.2},
    )
    style_axis(ax)

    ax = axes[1, 1]
    ax.plot(
        relative_perturbation,
        radial_lift_distance,
        "--",
        color="black",
        lw=1.40,
        label="radial gain direction",
    )
    ax.plot(
        relative_perturbation,
        nonradial_lift_distance,
        color="tab:blue",
        lw=1.70,
        label="non-radial deformation",
    )
    ax.axvline(0.0, color="0.55", lw=0.70)
    panel_title(ax, "(d) Radial null direction versus deformation")
    ax.set_xlabel("Relative perturbation")
    ax.set_ylabel("Lift-space displacement")
    ax.legend(frameon=False, loc="upper center", handlelength=2.4)
    ax.text(
        0.03,
        0.06,
        rf"$\|J_d d\|/(\|J_d\|\,\|d\|)={sci_tex(radial_residual, 1)}$",
        transform=ax.transAxes,
        fontsize=7.2,
        ha="left",
        va="bottom",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    style_axis(ax)

    save_figure(fig, FIGURES / "Fig01_geometry_concept.png", FIGURES / "Fig01_geometry_concept.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
