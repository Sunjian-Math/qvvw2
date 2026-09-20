from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch

from experiments.E03.code import distributed_wave_model as M
from experiments.E03.code.run_e03 import scenario_data
from experiments.E03.code.hv_batch_reimpl import HVBatch

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
RES.mkdir(parents=True, exist_ok=True)


def _copy_cache(cache):
    return [None if x is None else np.array(x, copy=True) for x in cache]


def main():
    M.P = 8
    dobs, f_inv = scenario_data("clean")
    params = {
        "kappa": 1e-4,
        "lam": 1e-4,
        "eps": 1e-12,
        "nt": 12,
        "maxiter": 4800,
        "gtol": 1e-10,
    }
    obj = HVBatch(dobs, workers=8, params=params, warm_start=True)
    rng = np.random.default_rng(90731)
    m0 = np.zeros((8, 8), dtype=float)
    h = rng.normal(size=m0.shape)
    h /= np.linalg.norm(h)

    def eval_fg(arr, need_grad, update_cache):
        mt = torch.tensor(arr, dtype=M.DTYPE, requires_grad=need_grad)
        c = M.cfrom(mt)
        dt, yf = M.sim(c, f=f_inv)
        y = M.rs(yf, dt)
        yn = y.detach().numpy()
        raw, gy, meta = obj.vg(yn, update_cache=update_cache)
        if not need_grad:
            return float(raw), meta
        y.backward(gradient=torch.tensor(gy, dtype=M.DTYPE))
        return float(raw), mt.grad.detach().numpy(), meta

    t0 = time.perf_counter()
    base_value, g, base_meta = eval_fg(m0, True, True)
    analytic = float(np.sum(g * h))
    base_cache = _copy_cache(obj.cache)
    rows = []
    for eps in [3e-3, 1e-3, 5e-4, 3e-4]:
        obj.cache = _copy_cache(base_cache)
        vp, mp = eval_fg(m0 + eps * h, False, False)
        obj.cache = _copy_cache(base_cache)
        vm, mm = eval_fg(m0 - eps * h, False, False)
        fd = (vp - vm) / (2.0 * eps)
        rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-14)
        rows.append([
            float(eps), float(fd), float(rel),
            float(mp["success_fraction"]), float(mm["success_fraction"]),
            float(mp["max_opt_grad_inf"]), float(mm["max_opt_grad_inf"]),
        ])
        print("HV strict gate", eps, "FD", fd, "analytic", analytic, "rel", rel, flush=True)
    obj.close()
    out = {
        "analytic": analytic,
        "base_value": base_value,
        "meta": base_meta,
        "params": params,
        "direction_seed": 90731,
        "rows": rows,
        "best_relative_error": float(min(r[2] for r in rows)),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    (RES / "hv_strict_gradient_gate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if out["best_relative_error"] > 5e-5:
        raise SystemExit("strict HV gradient gate did not reach 5e-5")


if __name__ == "__main__":
    main()
