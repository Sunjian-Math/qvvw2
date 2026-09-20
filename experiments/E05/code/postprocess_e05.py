from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from qvvw2.registry import METHODS
from qvvw2.progress import progress_hit, progress_print

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TRUE = 0.52
base = pd.read_csv(RES / "objectives_verified_base.csv")
starts = np.linspace(0.35, 0.69, 23)
sumrows = []
msrows = []
trajrows = []

for method in METHODS:
    print(f"[E05][multistart] method={method}", flush=True)
    g = base[base.method == method].sort_values("depth")
    if len(g) != 181:
        raise RuntimeError(f"missing objective curve for {method}: {len(g)}")
    x = g.depth.to_numpy()
    a = g.objective.to_numpy()
    rng = max(float(np.ptp(a)), 1e-30)
    mins, _ = find_peaks(-a, prominence=rng * 1e-4)
    p = PchipInterpolator(x, a)
    finals = []
    for start_id, s in enumerate(starts):
        if progress_hit(
            start_id + 1,
            len(starts),
            segments=5,
        ):
            progress_print(
                f"E05 multistart {method}",
                start_id + 1,
                len(starts),
                f"z0={s:.5f}",
            )
        path = [float(s)]

        def _record(xk):
            x = float(np.asarray(xk).ravel()[0])
            if not path or abs(x - path[-1]) > 1e-14:
                path.append(x)

        r = minimize(lambda z: float(p(float(z[0]))), [s], method="Nelder-Mead",
                     bounds=[(0.34, 0.70)], callback=_record,
                     options={"maxiter": 300, "xatol": 1e-9, "fatol": 1e-15})
        z = float(r.x[0])
        if abs(z - path[-1]) > 1e-14:
            path.append(z)
        finals.append(z)
        msrows.append({
            "method": method,
            "start_id": start_id,
            "initial_depth": s,
            "recovered_depth": z,
            "abs_error": abs(z - TRUE),
            "success": abs(z - TRUE) < 0.01,
        })
        for iteration, depth in enumerate(path):
            trajrows.append({
                "method": method,
                "start_id": start_id,
                "initial_depth": s,
                "iteration": iteration,
                "depth": depth,
                "is_final": iteration == len(path) - 1,
                "success": abs(z - TRUE) < 0.01,
            })
    finals = np.asarray(finals)
    sumrows.append({
        "method": method,
        "n_local_minima": int(len(mins)),
        "global_min_depth": float(x[np.argmin(a)]),
        "global_depth_error": float(abs(x[np.argmin(a)] - TRUE)),
        "success_tol_0p01": float(np.mean(abs(finals - TRUE) < 0.01)),
        "mean_abs_depth_error": float(np.mean(abs(finals - TRUE))),
        "median_abs_depth_error": float(np.median(abs(finals - TRUE))),
        "local_minima": ";".join(f"{x[i]:.4f}" for i in mins),
    })

base.sort_values(["method", "depth"]).to_csv(RES / "objectives_FINAL.csv", index=False)
pd.DataFrame(msrows).to_csv(RES / "multistart_FINAL.csv", index=False)
pd.DataFrame(trajrows).to_csv(RES / "trajectories_FINAL.csv", index=False)
pd.DataFrame(sumrows).to_csv(RES / "summary_FINAL.csv", index=False)
print(pd.DataFrame(sumrows).to_string(index=False))
