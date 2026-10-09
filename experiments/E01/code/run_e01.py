from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

from qvvw2.core import (
    B_from_rho,
    frozen_objective_grad,
    frozen_setup,
    jacobian_dense,
    jt_eta,
    lift,
    two_layer_graph_edges,
)

SEED = 20260918


def main():
    print("[E01] Starting numerical verification of quotient geometry", flush=True)
    rng = np.random.default_rng(SEED)

    n_full = 128
    t = np.linspace(0.0, 1.0, n_full)
    d = np.exp(-0.5 * ((t - 0.29) / 0.06) ** 2) - 0.65 * np.exp(
        -0.5 * ((t - 0.71) / 0.08) ** 2
    )
    setup = frozen_setup(d, 0.15, (n_full,), 0.03)
    rho_d, _ = lift(d, 0.15)

    scales = np.geomspace(1e-2, 1e2, 81)
    scale_rows = []
    for scale in scales:
        scaled = scale * d
        rho_c, _ = lift(scaled, 0.15)
        q_value, _ = frozen_objective_grad(scaled, setup)
        jac_c = jacobian_dense(scaled, 0.15)
        radial_residual = np.linalg.norm(jac_c @ scaled) / max(
            np.linalg.norm(jac_c) * np.linalg.norm(scaled), 1e-30
        )
        scale_rows.append(
            (scale, np.linalg.norm(rho_c - rho_d), q_value, radial_residual)
        )
    scale_df = pd.DataFrame(
        scale_rows,
        columns=["scale", "lift_gap", "Q_objective", "radial_jacobian_residual"],
    )
    scale_df.to_csv(RESULTS / "scale_invariance.csv", index=False)

    n = 48
    tt = np.linspace(0.0, 1.0, n)
    y = np.interp(tt, t, d)
    jac = jacobian_dense(y, 0.15)
    rho, _ = lift(y, 0.15)
    edges = two_layer_graph_edges((n,), 0.03)
    lap = B_from_rho(rho, edges).toarray()
    m = 2 * n
    gauge = np.ones((m, m)) / m
    pinv_lap = np.linalg.inv(lap + gauge) - gauge
    metric = jac.T @ pinv_lap @ jac
    metric = 0.5 * (metric + metric.T)
    eigvals, eigvecs = np.linalg.eigh(metric)
    pd.DataFrame({"eigenvalue": eigvals}).to_csv(
        RESULTS / "pullback_eigenvalues.csv", index=False
    )

    radial = y / np.linalg.norm(y)
    null_vector = eigvecs[:, np.argmin(np.abs(eigvals))]
    if null_vector @ radial < 0.0:
        null_vector = -null_vector
    null_alignment = float(abs(null_vector @ radial))
    radial_metric_residual = float(
        np.linalg.norm(metric @ y)
        / max(np.linalg.norm(metric) * np.linalg.norm(y), 1e-30)
    )

    time_tangent = np.gradient(y)
    time_tangent -= y * (y @ time_tangent) / (y @ y)
    time_tangent /= np.linalg.norm(time_tangent)

    generic = rng.normal(size=n)
    generic -= y * (y @ generic) / (y @ y)
    generic -= time_tangent * (time_tangent @ generic)
    generic /= np.linalg.norm(generic)

    directional_metric = {
        "radial gain": float(radial @ metric @ radial),
        "time-like tangent": float(time_tangent @ metric @ time_tangent),
        "generic tangent": float(generic @ metric @ generic),
    }
    pd.DataFrame(
        directional_metric.items(), columns=["direction", "quadratic_form"]
    ).to_csv(RESULTS / "directional_metric.csv", index=False)

    local_setup = frozen_setup(y, 0.15, (n,), 0.03)
    response_a = np.linspace(-0.45, 0.45, 91)
    y_norm = np.linalg.norm(y)
    response_rows = []
    for a in response_a:
        candidates = {
            "radial gain": (1.0 + a) * y,
            "time-like tangent": y + a * y_norm * time_tangent,
            "generic tangent": y + a * y_norm * generic,
        }
        for name, candidate in candidates.items():
            response_rows.append(
                (a, name, frozen_objective_grad(candidate, local_setup)[0])
            )
    response_df = pd.DataFrame(
        response_rows, columns=["relative_perturbation", "direction", "Q_objective"]
    )
    response_df.to_csv(RESULTS / "directional_response.csv", index=False)

    # Continuous directional slice in the two-dimensional tangent plane spanned
    # by the time-like and generic non-radial directions.  This is used only
    # for visualization and is evaluated with the same frozen Q-vvW2 objective.
    polar_theta = np.linspace(0.0, 2.0 * np.pi, 181)
    polar_amplitude = 0.16
    polar_rows = []
    for theta in polar_theta:
        direction = np.cos(theta) * time_tangent + np.sin(theta) * generic
        direction /= np.linalg.norm(direction)
        candidate = y + polar_amplitude * y_norm * direction
        actual = frozen_objective_grad(candidate, local_setup)[0]
        quadratic = 0.5 * (polar_amplitude * y_norm) ** 2 * (direction @ metric @ direction)
        polar_rows.append((theta, actual, quadratic))
    polar_df = pd.DataFrame(polar_rows, columns=["theta", "objective", "quadratic"])
    polar_df.to_csv(RESULTS / "directional_polar.csv", index=False)

    steps = np.logspace(-5, -1, 17)
    expansion_rows = []
    for step in steps:
        value, _ = frozen_objective_grad(y + step * time_tangent, local_setup)
        quadratic = 0.5 * step**2 * (time_tangent @ metric @ time_tangent)
        expansion_rows.append((step, value, quadratic, abs(value - quadratic)))
    expansion_df = pd.DataFrame(
        expansion_rows, columns=["step", "objective", "quadratic", "remainder"]
    )
    expansion_df.to_csv(RESULTS / "local_expansion.csv", index=False)

    fit_rows = expansion_rows[3:12]
    log_step = np.log10(np.asarray([row[0] for row in fit_rows]))
    log_remainder = np.log10(
        np.maximum(np.asarray([row[3] for row in fit_rows]), 1e-30)
    )
    local_slope = float(np.polyfit(log_step, log_remainder, 1)[0])

    conditioning_rows = []
    for epsilon in np.logspace(-3, 0, 25):
        lifted, _ = lift(y, epsilon)
        lap_eps = B_from_rho(lifted, edges).toarray()
        lap_eigs = np.linalg.eigvalsh(lap_eps)
        positive = lap_eigs[lap_eigs > max(lap_eigs.max() * 1e-11, 1e-14)]
        conditioning_rows.append(
            (
                epsilon,
                lifted.min(),
                positive.min(),
                positive.max(),
                positive.max() / positive.min(),
            )
        )
    conditioning_df = pd.DataFrame(
        conditioning_rows,
        columns=[
            "epsilon",
            "min_lift",
            "lambda_min_plus",
            "lambda_max",
            "condition",
        ],
    )
    conditioning_df.to_csv(RESULTS / "conditioning.csv", index=False)

    n_data = 14
    tx = np.linspace(0.0, 1.0, n_data)
    d0 = (
        np.sin(2 * np.pi * 1.3 * tx)
        + 0.35 * np.cos(2 * np.pi * 2.7 * tx + 0.2)
        + 0.2
    )
    setup0 = frozen_setup(d0, 0.18, (n_data,), 0.3)
    lap0 = setup0["B"].toarray()
    mm = 2 * n_data
    gauge0 = np.ones((mm, mm)) / mm
    pinv0 = np.linalg.inv(lap0 + gauge0) - gauge0
    p_eigvals, p_eigvecs = np.linalg.eigh(0.5 * (pinv0 + pinv0.T))
    p_sqrt = (
        p_eigvecs * np.sqrt(np.clip(p_eigvals, 0.0, None))
    ) @ p_eigvecs.T

    linear = rng.normal(size=(n_data, 3))
    linear -= np.outer(d0, (d0 @ linear) / (d0 @ d0))
    quadratic_map = rng.normal(size=(n_data, 3)) * 0.15

    def forward(x):
        return d0 + linear @ x + quadratic_map @ (x * x)

    def residual(x):
        rho_x, _ = lift(forward(x), 0.18)
        rho_0, _ = lift(d0, 0.18)
        return p_sqrt @ (rho_x - rho_0)

    def residual_jacobian(x):
        model_jac = linear + quadratic_map * (2.0 * x)[None, :]
        return p_sqrt @ jacobian_dense(forward(x), 0.18) @ model_jac

    x = np.array([0.12, -0.09, 0.07])
    gn_rows = []
    for iteration in range(8):
        error = np.linalg.norm(x)
        r = residual(x)
        jac_r = residual_jacobian(x)
        step = np.linalg.lstsq(jac_r, -r, rcond=None)[0]
        x_next = x + step
        next_error = np.linalg.norm(x_next)
        gn_rows.append((iteration, error, next_error))
        print(f"[E01][GN] iter {iteration+1:02d}/08  error={error:.6e}  next={next_error:.6e}", flush=True)
        x = x_next
    gn_df = pd.DataFrame(gn_rows, columns=["iter", "error", "next_error"])
    gn_df.to_csv(RESULTS / "GN.csv", index=False)

    h = rng.normal(size=n)
    eta = rng.normal(size=2 * n)
    delta = 1e-6
    eye = np.eye(n)
    jac_fd = np.column_stack(
        [
            (
                lift(y + delta * eye[j], 0.15)[0]
                - lift(y - delta * eye[j], 0.15)[0]
            )
            / (2 * delta)
            for j in range(n)
        ]
    )
    jacobian_fd_relative_error = float(
        np.linalg.norm(jac - jac_fd) / np.linalg.norm(jac_fd)
    )
    lhs = (jac @ h) @ eta
    rhs = h @ jt_eta(y, eta, 0.15)
    adjoint_identity_relative_error = float(abs(lhs - rhs) / max(1.0, abs(lhs)))

    summary = {
        "seed": SEED,
        "max_scale_lift_gap": float(scale_df["lift_gap"].max()),
        "max_scale_Q_objective": float(np.abs(scale_df["Q_objective"]).max()),
        "max_scale_radial_jacobian_residual": float(
            scale_df["radial_jacobian_residual"].max()
        ),
        "radial_metric_residual": radial_metric_residual,
        "radial_null_vector_alignment": null_alignment,
        "local_remainder_loglog_slope": local_slope,
        "jacobian_fd_relative_error": jacobian_fd_relative_error,
        "adjoint_identity_relative_error": adjoint_identity_relative_error,
        "directional_metric": directional_metric,
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    issues = []
    if summary["max_scale_lift_gap"] > 1e-12:
        issues.append("scale_lift_invariance")
    if summary["max_scale_Q_objective"] > 1e-12:
        issues.append("scale_objective_invariance")
    if summary["max_scale_radial_jacobian_residual"] > 1e-10:
        issues.append("scale_radial_jacobian")
    if radial_metric_residual > 1e-10 or null_alignment < 0.999999:
        issues.append("radial_null_mode")
    if not 2.5 < local_slope < 3.5:
        issues.append("local_cubic_remainder")
    if jacobian_fd_relative_error > 1e-6:
        issues.append("jacobian_finite_difference")
    if adjoint_identity_relative_error > 1e-10:
        issues.append("adjoint_identity")
    if float(gn_df["next_error"].iloc[-1]) > 1e-10:
        issues.append("gauss_newton_convergence")

    validation = {"status": "PASS" if not issues else "FAIL", "issues": issues, **summary}
    (RESULTS / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    print(json.dumps(validation, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
