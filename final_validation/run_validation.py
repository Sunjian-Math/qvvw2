"""Final fixed-scope validation. Existing scientific modules are imported unchanged."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from qvvw2.core import lift, jt_eta, fixed_scale_lift, fixed_scale_jt_eta, B_from_rho, normalized_l2_objective_grad
from experiments.E03.code import distributed_wave_model as M
from experiments.E03.code.run_e03 import regularizer, scenario_data
import scipy.sparse.linalg as spla
torch.set_num_threads(1)
OUT = ROOT / 'final_validation'
RES = OUT / 'results'
RES.mkdir(exist_ok=True)

class Objective:

    def __init__(self, d, kind, eps=0.15, cross=0.08):
        self.d = np.asarray(d)
        self.kind = kind
        self.eps = eps
        self.s2 = float(np.mean(d * d))
        self.rd = lift(d.ravel(), eps)[0]
        if kind in ['Q-vvW2', 'Fixed-vvW2']:
            B = B_from_rho(self.rd, M.edges_ring(d.shape, cross=cross, rec_w=0.08, source_w=0.03))
            self.solve = spla.factorized(B[:-1, :-1].tocsc())

    def vg(self, y):
        if self.kind == 'Normalized-L2':
            return normalized_l2_objective_grad(y.ravel(), self.d.ravel())
        if self.kind == 'Fixed-vvW2':
            rho = fixed_scale_lift(y.ravel(), self.s2, self.eps)[0]
        else:
            rho = lift(y.ravel(), self.eps)[0]
        dr = rho - self.rd
        if self.kind == 'Lift-Euclidean':
            eta = dr
        else:
            b = dr - dr.mean()
            eta = np.r_[self.solve(b[:-1]), 0.0]
            eta -= eta.mean()
        g = fixed_scale_jt_eta(y.ravel(), eta, self.s2, self.eps) if self.kind == 'Fixed-vvW2' else jt_eta(y.ravel(), eta, self.eps)
        return (0.5 * float(dr @ eta), g)

def data(sc, seed):
    if sc in ['clean', 'gain_only', 'source_mismatch']:
        return scenario_data(sc)
    if sc == 'gain_noise' and seed == 11:
        return scenario_data(sc)
    base = 1.8 * M.d.copy()
    eta = np.random.default_rng(seed).normal(size=base.shape)
    if sc == 'correlated_noise':
        eta = gaussian_filter1d(eta, 2.0, axis=-1, mode='reflect')
    eta *= 0.04 * np.linalg.norm(base) / np.linalg.norm(eta)
    return (base + eta, 7.0)

def initial(k):
    if k == 0:
        return np.zeros((8, 8))
    x = np.linspace(0, 1, 8)
    X, Z = np.meshgrid(x, x, indexing='ij')
    return 0.15 * np.cos(np.pi * X) * np.cos(np.pi * Z) if k == 1 else -0.15 * np.cos(np.pi * X) * np.cos(np.pi * Z)

def run(task):
    name = task['id']
    dest = RES / name
    dest.mkdir(exist_ok=True)
    sp = dest / 'summary.json'
    if sp.exists():
        return json.loads(sp.read_text())
    d, f = data(task['scenario'], task.get('seed', 11))
    obj = Objective(d, task['method'], task.get('eps', 0.15), task.get('cross', 0.08))
    init = initial(task.get('initial', 0))
    calls = 0
    hist = []
    accepted = []
    t = time.perf_counter()
    with torch.no_grad():
        dt, yf = M.sim(M.cfrom(torch.tensor(init, dtype=M.DTYPE)), f=f)
        y0 = M.rs(yf, dt).numpy()
    scale = max(abs(obj.vg(y0)[0]), 1e-30)

    def fg(arr):
        nonlocal calls
        m = torch.tensor(np.asarray(arr).reshape(8, 8), dtype=M.DTYPE, requires_grad=True)
        dt, yf = M.sim(M.cfrom(m), f=f)
        y = M.rs(yf, dt)
        v, gy = obj.vg(y.detach().numpy())
        y.backward(torch.tensor(gy.reshape(y.shape) / scale, dtype=M.DTYPE), retain_graph=True)
        rr = regularizer(m, 8)
        rr.backward()
        calls += 1
        value = float(v / scale + rr.detach())
        grad = m.grad.detach().numpy().ravel().copy()
        hist.append({'evaluation': calls, 'objective': value, 'gradient_inf': float(np.max(abs(grad)))})
        return (value, grad)
    optimizer = task.get('optimizer', 'Adam')
    if optimizer == 'Adam':
        mt = torch.tensor(init, dtype=M.DTYPE, requires_grad=True)
        opt = torch.optim.Adam([mt], lr=0.04, betas=(0.9, 0.999), eps=1e-08)
        for it in range(100):
            v, g = fg(mt.detach().numpy())
            opt.zero_grad()
            mt.grad = torch.tensor(g.reshape(8, 8), dtype=M.DTYPE)
            opt.step()
            with torch.no_grad():
                mt.clamp_(-1, 1)
            if (it + 1) % 25 == 0:
                np.save(dest / f'latent_{it + 1}.npy', mt.detach().numpy())
                rel, corr = M.metrics(M.cfrom(mt).detach().numpy())
                accepted.append({'iteration': it + 1, 'relative_model_error': float(rel), 'correlation': float(corr)})
        final = mt.detach().numpy()
        success = None
        message = 'Fixed 100-update budget'
        nit = 100
    else:
        r = minimize(fg, init.ravel(), jac=True, method='L-BFGS-B', bounds=[(-1, 1)] * 64, options={'maxiter': 100, 'maxfun': 200, 'maxls': 25, 'gtol': 1e-07, 'ftol': 1e-11}, callback=lambda x: accepted.append({'iteration': len(accepted) + 1, 'objective': hist[-1]['objective']}))
        final = r.x.reshape(8, 8)
        success = bool(r.success)
        message = str(r.message)
        nit = int(r.nit)
    optimization_calls = calls
    terminal, grad = fg(final)
    with torch.no_grad():
        c = M.cfrom(torch.tensor(final, dtype=M.DTYPE)).numpy()
        dt, yf = M.sim(torch.tensor(c, dtype=M.DTYPE), f=f)
        pred = M.rs(yf, dt).numpy()
    rel, corr = M.metrics(c)
    pg = grad.copy()
    xx = final.ravel()
    pg[(xx <= -1 + 1e-10) & (grad > 0)] = 0
    pg[(xx >= 1 - 1e-10) & (grad < 0)] = 0
    summary = {**task, 'relative_model_error': float(rel), 'correlation': float(corr), 'common_std_residual': float(np.linalg.norm(pred - d) / np.linalg.norm(d)), 'terminal_objective': terminal, 'projected_gradient_inf': float(np.max(abs(pg))), 'bound_fraction': float(np.mean(abs(xx) >= 1 - 1e-08)), 'initial_objective': scale, 'optimizer_success': success, 'stop_reason': message, 'iterations': nit, 'optimization_forward_reverse_pairs': optimization_calls, 'total_fg_evaluations': calls, 'seconds_concurrent_diagnostic_only': time.perf_counter() - t}
    np.savez_compressed(dest / 'arrays.npz', latent=final, model=c, predicted=pred, observed=d, initial=init)
    pd.DataFrame(hist).to_csv(dest / 'evaluations.csv', index=False)
    pd.DataFrame(accepted).to_csv(dest / 'accepted.csv', index=False)
    sp.write_text(json.dumps(summary, indent=2))
    print('DONE', name, 'RE', round(float(rel), 6), flush=True)
    return summary

def gates():
    d, _ = data('clean', 11)
    rng = np.random.default_rng(1404)
    y = d + 0.15 * np.linalg.norm(d) / np.sqrt(d.size) * rng.normal(size=d.shape)
    h = rng.normal(size=d.shape)
    h /= np.linalg.norm(h)
    rows = []
    for kind in ['Q-vvW2', 'Lift-Euclidean', 'Normalized-L2', 'Fixed-vvW2']:
        obj = Objective(d, kind)
        v, g = obj.vg(y)
        an = float(g @ h.ravel())
        errors = []
        for frac in [0.001, 0.0003, 0.0001]:
            e = frac * np.linalg.norm(y)
            fd = (obj.vg(y + e * h)[0] - obj.vg(y - e * h)[0]) / (2 * e)
            errors.append({'relative_step': frac, 'analytic': an, 'finite_difference': fd, 'relative_error': abs(fd - an) / max(abs(fd), abs(an), 1e-14)})
        rows.append({'method': kind, 'rows': errors, 'best_relative_error': min((r['relative_error'] for r in errors)), 'positive_gain_delta': abs(obj.vg(1.8 * y)[0] - v)})
    assert all((r['best_relative_error'] < 2e-05 for r in rows)), rows
    (OUT / 'gradient_gates.json').write_text(json.dumps(rows, indent=2))
    print('DATA GRADIENT GATES PASS', flush=True)

def tasks():
    methods = ['Q-vvW2', 'Lift-Euclidean', 'Normalized-L2', 'Fixed-vvW2']
    t = []
    for sc in ['clean', 'gain_only', 'source_mismatch', 'gain_noise']:
        for m in ['Lift-Euclidean', 'Fixed-vvW2']:
            t.append(dict(id=f'mechanism_{sc}_{m}', group='mechanism', scenario=sc, method=m, seed=11))
    for sc in ['gain_noise', 'correlated_noise']:
        for seed in range(101, 111):
            for m in methods:
                t.append(dict(id=f'ensemble_{sc}_{seed}_{m}', group='ensemble', scenario=sc, seed=seed, method=m))
    for sc in ['clean', 'correlated_noise']:
        for eps, cross in [(0.075, 0.08), (0.3, 0.08), (0.15, 0.04), (0.15, 0.16)]:
            t.append(dict(id=f'sensitivity_{sc}_{eps}_{cross}', group='sensitivity', scenario=sc, seed=101, method='Q-vvW2', eps=eps, cross=cross))
    for sc in ['clean', 'gain_noise']:
        for init in range(3):
            for m in methods:
                t.append(dict(id=f'robustness_{sc}_{init}_{m}', group='robustness', scenario=sc, seed=11, method=m, initial=init, optimizer='L-BFGS-B'))
    return t

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--gate-only', action='store_true')
    a = p.parse_args()
    gates()
    if a.gate_only:
        return
    ts = tasks()
    protocol = {'theory': 'Prespecified quotient objective and component controls', 'tasks': ts, 'count': len(ts), 'budget_Adam': 100, 'budget_LBFGSB': {'maxiter': 100, 'maxfun': 200, 'maxls': 25, 'note': 'SciPy may complete a line search slightly beyond maxfun; actual calls reported'}, 'noise': '4% Euclidean norm; Gaussian independent or temporal Gaussian filtering sigma=2 samples; matched realizations', 'initial_models': 'zero and +/-0.15*cos(pi*x)*cos(pi*z), latent 8x8', 'scope': 'Same forward model, regularizer and objective normalization at each common initial model', 'timing': 'Concurrent run wall times are diagnostics, not comparative runtime benchmarks'}
    (OUT / 'FINAL_PROTOCOL.json').write_text(json.dumps(protocol, indent=2))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(run, ts))
    pd.DataFrame(results).to_csv(OUT / 'all_metrics.csv', index=False)
    print('COMPLETE', len(results), flush=True)
if __name__ == '__main__':
    main()
