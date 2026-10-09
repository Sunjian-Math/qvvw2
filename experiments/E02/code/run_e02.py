from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import qvvw2.core as Q
from qvvw2.registry import METHODS
from qvvw2.progress import progress_hit, progress_print
from qvvw2.baselines.softplus_w2sq import beta_from_observed, batch_value
from qvvw2.baselines.uot import li_uot_batch
from qvvw2.baselines.sambridge2022_marginal_w2sq import ricker_util as RU, OTlib as OT
N = 48
T = np.linspace(0.0, 1.0, N)

def gauss(mu, sig):
    return np.exp(-0.5 * ((T - mu) / sig) ** 2)
REF = gauss(0.3, 0.055) - 0.65 * gauss(0.7, 0.07)
POS = gauss(0.3, 0.055)
NEG = gauss(0.7, 0.07)
GAINS = np.geomspace(0.2, 5.0, 41)
RATIOS = np.linspace(0.2, 1.2, 52)
SHIFTS = np.linspace(-0.2, 0.2, 61)
SIGNED = np.r_[np.linspace(-2.0, -0.1, 32), np.linspace(0.1, 2.0, 32)]
QSETUP = Q.frozen_setup(REF, eps=0.15, data_shape=(N,), cross_weight=0.03)
X = np.linspace(0.0, 1.0, N)
K = 1.5 / max(float(np.max(REF)), 1e-12)
QIU_BETA = beta_from_observed(REF, beta_unit=2.0)
AMAX = 5.5 * max(abs(float(REF.min())), abs(float(REF.max())))
GRID = (0.0, 1.0, -AMAX, AMAX, 64, N)
_, OTREF = RU.BuildOTobjfromWaveform(T, REF, GRID, lambdav=0.03, theta=45.0)
if OTREF.calcmarg:
    OTREF.setMarginals()

def shift_signal(s):
    return gauss(0.3 + s, 0.055) - 0.65 * gauss(0.7 + s, 0.07)

def candidates(name):
    if name == 'gain':
        return (GAINS, np.array([g * REF for g in GAINS]))
    if name == 'ratio':
        return (RATIOS, np.array([POS - r * NEG for r in RATIOS]))
    if name == 'shift':
        return (SHIFTS, np.array([shift_signal(s) for s in SHIFTS]))
    if name == 'signed_gain':
        return (SIGNED, np.array([g * REF for g in SIGNED]))
    raise KeyError(name)

def marginal_w2sq_value(y):
    _, oy = RU.BuildOTobjfromWaveform(T, y, GRID, lambdav=0.03, theta=45.0)
    if oy.calcmarg:
        oy.setMarginals()
    wt = OT.wasser(oy.marg[0], OTREF.marg[0], distfunc='W2', derivatives=False, checkCommonCDF=False)[0]
    wu = OT.wasser(oy.marg[1], OTREF.marg[1], distfunc='W2', derivatives=False, checkCommonCDF=False)[0]
    return 0.5 * (float(wt) + float(wu))

def run_comparison(name):
    xx, Y = candidates(name)
    print(f'[E02][comparison] sweep={name}: {len(xx)} points', flush=True)
    t0 = time.perf_counter()
    uvals, umeta = li_uot_batch(Y, REF, X, reg=0.001, reg_m=1.0, k=K, maxiter=14000, check_every=100, tol=1e-11)
    rows = []
    for i, (x, y) in enumerate(zip(xx, Y)):
        if progress_hit(i + 1, len(xx), segments=5):
            progress_print(f'E02 comparison {name}', i + 1, len(xx))
        vals = {'L2': Q.l2_objective_grad(y, REF)[0], 'Normalized-L2': Q.normalized_l2_objective_grad(y, REF)[0] if np.linalg.norm(y) > 1e-14 else np.nan, 'Softplus-W2sq': batch_value(y, REF, T, QIU_BETA), 'Marginal-W2sq': marginal_w2sq_value(y), 'UOT': float(uvals[i]), 'Q-vvW2': Q.frozen_objective_grad(y, QSETUP)[0]}
        rows.append({'parameter': float(x), **vals})
    df = pd.DataFrame(rows)
    df.to_csv(RES / f'{name}.csv', index=False)
    meta = {'sweep': name, 'n_points': len(df), 'runtime_s': time.perf_counter() - t0, 'UOT': umeta, 'Softplus-W2sq': {'reference': 'Qiu 2021, Inverse Problems 37 095004', 'beta_raw': QIU_BETA, 'beta_rule': 'beta=2 after one global observed-amplitude nondimensionalization'}, 'Marginal-W2sq': {'fingerprint_grid': GRID, 'lambda_v': 0.03, 'theta_deg': 45.0, 'transport': 'marginal W_2^2'}, 'Q-vvW2': {'epsilon': 0.15, 'cross_weight': 0.03}, 'n_samples': N}
    (RES / f'{name}_meta.json').write_text(json.dumps(meta, indent=2))
    print('COMPARISON', name, len(df), meta['runtime_s'], flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', choices=['gain', 'ratio', 'shift', 'signed_gain', 'all'], default='all')
    args = parser.parse_args()
    for sweep in ['gain', 'ratio', 'shift', 'signed_gain'] if args.sweep == 'all' else [args.sweep]:
        run_comparison(sweep)
