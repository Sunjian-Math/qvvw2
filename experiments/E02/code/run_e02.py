from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

import qvvw2.core as Q
from qvvw2.registry import METHODS
from qvvw2.progress import progress_hit, progress_print
from qvvw2.baselines.softplus_w2sq import beta_from_observed, batch_value
from qvvw2.baselines.uot import li_uot_batch
from qvvw2.baselines.hv import solve_hv
from qvvw2.baselines.sambridge2022_marginal_w2sq import ricker_util as RU, OTlib as OT

N = 48
T = np.linspace(0.0, 1.0, N)

def gauss(mu, sig):
    return np.exp(-0.5 * ((T - mu) / sig) ** 2)

REF = gauss(0.30, 0.055) - 0.65 * gauss(0.70, 0.070)
POS = gauss(0.30, 0.055)
NEG = gauss(0.70, 0.070)
GAINS = np.geomspace(0.2, 5.0, 41)
RATIOS = np.linspace(0.2, 1.2, 52)
SHIFTS = np.linspace(-0.20, 0.20, 61)
SIGNED = np.r_[np.linspace(-2.0, -0.10, 32), np.linspace(0.10, 2.0, 32)]

QSETUP = Q.frozen_setup(REF, eps=0.15, data_shape=(N,), cross_weight=0.03)
X = np.linspace(0.0, 1.0, N)
K = 1.5 / max(float(np.max(REF)), 1e-12)
QIU_BETA = beta_from_observed(REF, beta_unit=2.0)

# One fixed waveform-fingerprint grid for the entire pre-specified sweep family.
AMAX = 5.5 * max(abs(float(REF.min())), abs(float(REF.max())))
GRID = (0.0, 1.0, -AMAX, AMAX, 64, N)
_, OTREF = RU.BuildOTobjfromWaveform(T, REF, GRID, lambdav=0.03, theta=45.0)
if OTREF.calcmarg:
    OTREF.setMarginals()


def shift_signal(s):
    return gauss(0.30 + s, 0.055) - 0.65 * gauss(0.70 + s, 0.070)


def candidates(name):
    if name == "gain":
        return GAINS, np.array([g * REF for g in GAINS])
    if name == "ratio":
        return RATIOS, np.array([POS - r * NEG for r in RATIOS])
    if name == "shift":
        return SHIFTS, np.array([shift_signal(s) for s in SHIFTS])
    if name == "signed_gain":
        return SIGNED, np.array([g * REF for g in SIGNED])
    raise KeyError(name)


def marginal_w2sq_value(y):
    _, oy = RU.BuildOTobjfromWaveform(T, y, GRID, lambdav=0.03, theta=45.0)
    if oy.calcmarg:
        oy.setMarginals()
    wt = OT.wasser(oy.marg[0], OTREF.marg[0], distfunc="W2", derivatives=False, checkCommonCDF=False)[0]
    wu = OT.wasser(oy.marg[1], OTREF.marg[1], distfunc="W2", derivatives=False, checkCommonCDF=False)[0]
    return 0.5 * (float(wt) + float(wu))


def run_nonhv(name):
    xx, Y = candidates(name)
    print(f"[E02][non-HV] sweep={name}: {len(xx)} points", flush=True)
    t0 = time.perf_counter()
    uvals, umeta = li_uot_batch(
        Y, REF, X, reg=1e-3, reg_m=1.0, k=K,
        maxiter=14000, check_every=100, tol=1e-11,
    )
    rows = []
    for i, (x, y) in enumerate(zip(xx, Y)):
        if progress_hit(i + 1, len(xx), segments=5):
            progress_print(
                f"E02 non-HV {name}",
                i + 1,
                len(xx),
            )
        vals = {
            "L2": Q.l2_objective_grad(y, REF)[0],
            "Normalized-L2": Q.normalized_l2_objective_grad(y, REF)[0] if np.linalg.norm(y) > 1e-14 else np.nan,
            "Softplus-W2sq": batch_value(y, REF, T, QIU_BETA),
            "Marginal-W2sq": marginal_w2sq_value(y),
            "UOT": float(uvals[i]),
            "Q-vvW2": Q.frozen_objective_grad(y, QSETUP)[0],
        }
        rows.append({"parameter": float(x), **vals})
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"{name}_nonhv.csv", index=False)
    meta = {
        "sweep": name,
        "n_points": len(df),
        "runtime_s": time.perf_counter() - t0,
        "UOT": umeta,
        "Softplus-W2sq": {
            "reference": "Qiu 2021, Inverse Problems 37 095004",
            "beta_raw": QIU_BETA,
            "beta_rule": "beta=2 after one global observed-amplitude nondimensionalization",
        },
        "Marginal-W2sq": {
            "fingerprint_grid": GRID,
            "lambda_v": 0.03,
            "theta_deg": 45.0,
            "transport": "marginal W_2^2",
        },
        "Q-vvW2": {"epsilon": 0.15, "cross_weight": 0.03},
        "n_samples": N,
    }
    (RES / f"{name}_nonhv_meta.json").write_text(json.dumps(meta, indent=2))
    print("NONHV", name, len(df), meta["runtime_s"], flush=True)


def hv_point(y, x0=None):
    stages = (600, 1200, 2400, 2400)
    prev = None
    best = None
    cur = x0
    hist = []
    for maxiter in stages:
        r = solve_hv(y, REF, kappa=1e-4, lam=1e-4, eps=1e-12, nt=8,
                     maxiter=maxiter, gtol=1e-9, x0=cur)
        rel = None if prev is None else abs(r["action"] - prev) / max(abs(r["action"]), 1e-30)
        hist.append({
            "maxiter": maxiter,
            "action": r["action"],
            "rel_action_change": rel,
            "opt_grad_inf": r["opt_grad_inf"],
            "success": r["success"],
            "nit": r["nit"],
        })
        best = r
        cur = r["x"]
        if prev is not None and rel < 5e-3:
            break
        prev = r["action"]
    stable = (
        (len(hist) >= 2 and hist[-1]["rel_action_change"] is not None and hist[-1]["rel_action_change"] < 5e-3)
        or bool(best["success"])
    )
    if not stable:
        for _ in range(10):
            r = solve_hv(y, REF, kappa=1e-4, lam=1e-4, eps=1e-12, nt=8,
                         maxiter=4800, gtol=1e-9, x0=cur)
            rel = abs(r["action"] - prev) / max(abs(r["action"]), 1e-30)
            hist.append({
                "maxiter": 4800,
                "action": r["action"],
                "rel_action_change": rel,
                "opt_grad_inf": r["opt_grad_inf"],
                "success": r["success"],
                "nit": r["nit"],
            })
            best = r
            cur = r["x"]
            prev = r["action"]
            if rel < 5e-3 or r["success"]:
                stable = True
                break
    return best, hist, stable


def order_from_anchor(xx, anchor):
    i = int(np.argmin(abs(xx - anchor)))
    return [i] + list(range(i + 1, len(xx))) + list(range(i - 1, -1, -1))


def run_hv(name):
    xx, Y = candidates(name)
    out = RES / f"{name}_hv.csv"
    qcp = RES / f"{name}_hv_qc.json"
    done = {}
    if out.exists():
        old = pd.read_csv(out)
        for _, r in old.iterrows():
            done[int(r["index"])] = dict(r)
    anchor = 1.0 if name in ("gain", "signed_gain") else (0.65 if name == "ratio" else 0.0)
    last_x = None
    qc = []
    t0 = time.perf_counter()
    order = order_from_anchor(xx, anchor)
    n_done_start = len(done)
    for count, j in enumerate(order, 1):
        if j in done:
            continue
        point_id = len(done) + 1

        if progress_hit(point_id, len(xx), segments=8):
            progress_print(
                f"E02 HV {name}",
                point_id,
                len(xx),
                f"parameter={xx[j]:.6g}",
            )
        r1, h1, s1 = hv_point(Y[j], last_x)
        chosen, hist, stable = r1, h1, s1
        init = "continuation" if last_x is not None else "linear"
        if not stable:
            r2, h2, s2 = hv_point(Y[j], None)
            if r2["action"] < r1["action"]:
                chosen, hist, stable, init = r2, h2, s2, "linear_restart"
            else:
                stable = stable or s2
        last_x = chosen["x"]
        done[j] = {
            "index": j,
            "parameter": float(xx[j]),
            "HV": float(chosen["action"]),
            "stable": bool(stable),
            "opt_grad_inf": float(chosen["opt_grad_inf"]),
            "nit": int(chosen["nit"]),
            "init_selected": init,
            "n_stages": len(hist),
            "last_rel_action_change": hist[-1]["rel_action_change"],
        }
        pd.DataFrame(list(done.values())).sort_values("index").to_csv(out, index=False)
        qc.append({"index": j, "parameter": float(xx[j]), "history": hist, "selected": init, "stable": bool(stable)})
        qcp.write_text(json.dumps(qc, indent=2))
    df = pd.DataFrame(list(done.values())).sort_values("index")
    df.to_csv(out, index=False)
    summary = {
        "sweep": name,
        "n_points": len(df),
        "stable_fraction": float(df.stable.astype(bool).mean()),
        "max_opt_grad_inf": float(df.opt_grad_inf.max()),
        "max_last_rel_action_change": float(pd.to_numeric(df.last_rel_action_change, errors="coerce").fillna(0).max()),
        "runtime_s_current_call": time.perf_counter() - t0,
        "hv_params": {"kappa": 1e-4, "lambda": 1e-4, "epsilon": 1e-12, "nt": 8},
    }
    (RES / f"{name}_hv_summary.json").write_text(json.dumps(summary, indent=2))


def run_landscape():
    print("[E02] Computing Q-vvW2 gain-shift landscape", flush=True)
    shifts = np.linspace(-0.16, 0.16, 65)
    lg = np.linspace(-0.6, 0.6, 49)
    a = np.zeros((len(lg), len(shifts)))
    for i, z in enumerate(lg):
        if progress_hit(i + 1, len(lg), segments=6):
            progress_print(
                "E02 landscape",
                i + 1,
                len(lg),
            )
        g = 10 ** z
        for j, s in enumerate(shifts):
            a[i, j] = Q.frozen_objective_grad(g * shift_signal(s), QSETUP)[0]
    a /= max(a.max(), 1e-30)
    np.savez_compressed(RES / "q_gain_shift_landscape.npz", log10_gain=lg, shift=shifts, value=a)


def validate():
    issues = []
    stable_fractions = {}
    counts = {"gain": 41, "ratio": 52, "shift": 61, "signed_gain": 64}
    nonhv_methods = [m for m in METHODS if m != "HV"]
    for s, n in counts.items():
        a = RES / f"{s}_nonhv.csv"
        b = RES / f"{s}_hv.csv"
        if not a.exists() or not b.exists():
            issues.append(f"missing {s}")
            continue
        da = pd.read_csv(a)
        db = pd.read_csv(b)
        if len(da) != n or len(db) != n:
            issues.append(f"count {s}: {len(da)}, {len(db)}")
        if list(da.columns) != ["parameter", *nonhv_methods]:
            issues.append(f"method columns {s}: {list(da.columns)}")
        if not np.isfinite(da.select_dtypes("number").to_numpy()).all():
            issues.append(f"nonfinite nonhv {s}")
        if not np.isfinite(db.select_dtypes("number").to_numpy()).all():
            issues.append(f"nonfinite HV {s}")
        stable_fractions[s] = float(db.stable.astype(bool).mean())
        if stable_fractions[s] < 0.95:
            issues.append(f"HV stable fraction {s} {stable_fractions[s]:.3f}")
    gain_err = None
    dg = RES / "gain_nonhv.csv"
    if dg.exists():
        dfg = pd.read_csv(dg)
        q = dfg["Q-vvW2"].to_numpy()
        x = dfg.parameter.to_numpy()
        ref = q[np.argmin(abs(x - 1.0))]
        gain_err = float(np.max(np.abs(q - ref)))
        if gain_err > 1e-12:
            issues.append(f"Q-vvW2 gain invariance {gain_err}")
    out = {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "methods": METHODS,
        "hv_stable_fraction": stable_fractions,
        "q_positive_gain_max_abs_change": gain_err,
        "qiu_beta_raw": QIU_BETA,
    }
    (RES / "VALIDATION.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonhv", choices=["gain", "ratio", "shift", "signed_gain", "all"])
    ap.add_argument("--hv", choices=["gain", "ratio", "shift", "signed_gain", "all"])
    ap.add_argument("--landscape", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.nonhv:
        for s in (["gain", "ratio", "shift", "signed_gain"] if args.nonhv == "all" else [args.nonhv]):
            run_nonhv(s)
    if args.hv:
        for s in (["gain", "ratio", "shift", "signed_gain"] if args.hv == "all" else [args.hv]):
            run_hv(s)
    if args.landscape:
        run_landscape()
    if args.validate:
        raise SystemExit(validate())
