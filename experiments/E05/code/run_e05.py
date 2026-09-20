from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
sys.path.insert(0, str(CODE))

import qvvw2.core as Q
from qvvw2.baselines.softplus_w2sq import beta_from_observed, batch_value
from qvvw2.baselines.uot import li_uot_batch
from qvvw2.baselines.hv import solve_hv
from qvvw2.baselines.sambridge2022_marginal_w2sq import ricker_util as RU, OTlib as OT
from qvvw2.registry import METHODS
from qvvw2.progress import progress_hit, progress_print

TRUE = 0.52
DEPTHS = np.linspace(0.34, 0.70, 181)
STARTS = np.linspace(0.35, 0.69, 23)
T = np.linspace(0.55, 1.55, 192)


def ricker(t, f=18.0, t0=0.15):
    tau = t - t0
    a = (np.pi * f * tau) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def simulate(depth=None, eps_slab=4.0, width=0.08, nx=321, tmax=1.8):
    dx = 1.0 / (nx - 1)
    dt = 0.85 * dx
    nt = int(tmax / dt) + 1
    x = np.linspace(0.0, 1.0, nx)
    eps = np.ones(nx)
    if depth is not None:
        sm = 2 * dx
        box = 0.5 * (np.tanh((x - depth) / sm) - np.tanh((x - (depth + width)) / sm))
        eps = 1.0 + (eps_slab - 1.0) * box
    E = np.zeros(nx)
    H = np.zeros(nx - 1)
    src = int(round(0.12 / dx))
    rec = int(round(0.16 / dx))
    sponge = max(20, int(0.14 / dx))
    damp = np.ones(nx)
    for i in range(sponge):
        q = (sponge - i) / sponge
        fac = np.exp(-0.035 * q * q)
        damp[i] *= fac
        damp[-1 - i] *= fac
    dampH = np.sqrt(damp[:-1] * damp[1:])
    ts = np.arange(nt) * dt
    sv = ricker(ts)
    out = np.empty(nt)
    for n in range(nt):
        H -= (dt / dx) * (E[1:] - E[:-1])
        H *= dampH
        E[1:-1] -= (dt / dx) * (H[1:] - H[:-1]) / eps[1:-1]
        E[src] += 0.15 * sv[n]
        E *= damp
        out[n] = E[rec]
    return ts, out


def make_forward():
    tf, rf = simulate(None, nx=641)
    _, sf = simulate(TRUE, nx=641)
    obs = np.interp(T, tf, sf - rf)
    tc, rc = simulate(None, nx=321)
    Y = []
    for z in DEPTHS:
        _, s = simulate(float(z), nx=321)
        Y.append(np.interp(T, tc, s - rc))
    Y = np.asarray(Y)
    np.savez_compressed(RES / "tdem_traces.npz", T=T, depths=DEPTHS, starts=STARTS,
                        observed=obs, candidates=Y, true_depth=TRUE)
    return obs, Y


def load_forward():
    p = RES / "tdem_traces.npz"
    if not p.exists():
        return make_forward()
    z = np.load(p)
    return z["observed"], z["candidates"]


def marginal_w2sq_curve(Y, d):
    amax = 1.5 * max(abs(float(d.min())), abs(float(d.max())), 1e-12)
    grid = (float(T[0]), float(T[-1]), -amax, amax, 36, len(T))
    _, od = RU.BuildOTobjfromWaveform(T, d, grid, lambdav=0.03, theta=45.0)
    if od.calcmarg:
        od.setMarginals()
    vals = []
    for y in Y:
        _, oy = RU.BuildOTobjfromWaveform(T, y, grid, lambdav=0.03, theta=45.0)
        if oy.calcmarg:
            oy.setMarginals()
        wt = OT.wasser(oy.marg[0], od.marg[0], distfunc="W2", derivatives=False, checkCommonCDF=False)[0]
        wu = OT.wasser(oy.marg[1], od.marg[1], distfunc="W2", derivatives=False, checkCommonCDF=False)[0]
        vals.append(0.5 * (float(wt) + float(wu)))
    return np.asarray(vals), {
        "amp_padding": 1.5,
        "amp_grid": 36,
        "lambda_v": 0.03,
        "theta_deg": 45.0,
        "transport": "marginal W_2^2",
    }


def run_nonhv():
    t0 = time.perf_counter()
    d, Y = load_forward()
    x = (T - T[0]) / (T[-1] - T[0])
    qst = Q.frozen_setup(d, eps=0.15, data_shape=(len(T),), cross_weight=0.03)
    beta = beta_from_observed(d, beta_unit=2.0)
    values = {}
    values["L2"] = np.array([Q.l2_objective_grad(y, d)[0] for y in Y])
    values["Normalized-L2"] = np.array([Q.normalized_l2_objective_grad(y, d)[0] for y in Y])
    values["Softplus-W2sq"] = np.array([batch_value(y, d, T, beta) for y in Y])
    values["Q-vvW2"] = np.array([Q.frozen_objective_grad(y, qst)[0] for y in Y])
    values["UOT"], umeta = li_uot_batch(Y, d, x, reg=1e-3, reg_m=1.0,
                                         maxiter=14000, check_every=100, tol=1e-11)
    values["Marginal-W2sq"], smeta = marginal_w2sq_curve(Y, d)
    rows = []
    for m in [m for m in METHODS if m != "HV"]:
        for z, a in zip(DEPTHS, values[m]):
            rows.append({"depth": float(z), "method": m, "objective": float(a)})
    pd.DataFrame(rows).to_csv(RES / "objectives_nonhv_recomputed.csv", index=False)
    meta = {
        "runtime_s": time.perf_counter() - t0,
        "Q-vvW2": {"epsilon": 0.15, "cross_weight": 0.03},
        "Softplus-W2sq": {
            "reference": "Qiu 2021, Inverse Problems 37 095004",
            "beta_raw": beta,
            "beta_rule": "beta=2 after one global observed-amplitude nondimensionalization",
        },
        "Marginal-W2sq": smeta,
        "UOT": umeta,
    }
    (RES / "NONHV_METADATA.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def _hvw(a):
    i, y, d = a
    r = solve_hv(y, d, kappa=1e-4, lam=1e-4, eps=1e-12, nt=12,
                 maxiter=1800, gtol=5e-8)
    return i, DEPTHS[i], r["action"], r["nit"], r["success"], r["nfev"], r["opt_grad_inf"]


def run_hv_chunk(start, end, workers):
    d, Y = load_forward()
    ids = list(range(start, min(end, len(DEPTHS))))
    args = [(i, Y[i], d) for i in ids]
    t0 = time.perf_counter()
    print(f"[E05][HV] chunk {start}:{min(end, len(DEPTHS))}, workers={workers}, points={len(ids)}", flush=True)
    rr = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, row in enumerate(ex.map(_hvw, args, chunksize=1), 1):
            rr.append(row)
            if progress_hit(
                k,
                len(ids),
                segments=10,
            ):
                progress_print(
                    "E05 HV",
                    k,
                    len(ids),
                    (
                        f"depth={row[1]:.5f}"
                        f" | inner_nit={row[3]}"
                        f" | success={row[4]}"
                    ),
                )
    out = RES / f"hv_chunk_{start:03d}_{min(end, 181):03d}.csv"
    pd.DataFrame(rr, columns=["index", "depth", "objective", "inner_nit", "success", "nfev", "opt_grad_inf"]).to_csv(out, index=False)
    print(out, "secs", time.perf_counter() - t0)


def combine_hv():
    fs = sorted(RES.glob("hv_chunk_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("index").sort_values("index")
    if len(df) != 181:
        raise SystemExit(f"HV incomplete {len(df)}/181")
    df.to_csv(RES / "hv_high_precision_181.csv", index=False)
    print("HV", len(df), "success", df.success.mean(), "maxgrad", df.opt_grad_inf.max())


def combine_objectives():
    non = pd.read_csv(RES / "objectives_nonhv_recomputed.csv")
    hv = pd.read_csv(RES / "hv_high_precision_181.csv")[["depth", "objective"]].assign(method="HV")
    allv = pd.concat([non, hv], ignore_index=True).sort_values(["method", "depth"])
    allv.to_csv(RES / "objectives_verified_base.csv", index=False)
    meta = {
        "status": "COMPLETE",
        "anti_inverse_crime": True,
        "observation_nx": 641,
        "candidate_nx": 321,
        "true_depth": TRUE,
        "depth_count": 181,
        "start_count": 23,
        "time_samples": 192,
        "methods": METHODS,
        "HV": {"kappa": 1e-4, "lambda": 1e-4, "epsilon": 1e-12, "nt": 12, "maxiter": 1800, "gtol": 5e-8},
    }
    (RES / "RUN_METADATA.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--forward", action="store_true")
    p.add_argument("--nonhv", action="store_true")
    p.add_argument("--hv-chunk", nargs=2, type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--combine-hv", action="store_true")
    p.add_argument("--combine-objectives", action="store_true")
    a = p.parse_args()
    if a.forward:
        make_forward()
    if a.nonhv:
        run_nonhv()
    if a.hv_chunk:
        run_hv_chunk(a.hv_chunk[0], a.hv_chunk[1], a.workers)
    if a.combine_hv:
        combine_hv()
    if a.combine_objectives:
        combine_objectives()
