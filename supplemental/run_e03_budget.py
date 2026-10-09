"""Fresh deterministic 100-step trajectories; compare 25, 50, 100 checkpoints.

The original run_one, initialization, regularizer, Adam and six objectives
are unchanged. No ground-truth-based choice of checkpoint is made.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from experiments.E03.code.run_e03 import run_one, scenario_data, M, SCENARIOS
from qvvw2.registry import METHODS

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--methods', nargs='+', default=METHODS)
    p.add_argument('--scenarios', nargs='+', default=SCENARIOS)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    for sc in a.scenarios:
        for method in a.methods:
            od = out / 'P8' / sc / method
            summary_path = od / 'summary.json'
            if summary_path.exists() and json.loads(summary_path.read_text()).get('steps') == 100:
                continue
            run_one(method, sc, P=8, steps=100, outroot=out)
    repaired = []
    for sc in a.scenarios:
        for method in a.methods:
            od = out / 'P8' / sc / method
            hist = pd.read_csv(od / 'history.csv')
            if len(hist) == 100:
                continue
            step = int(hist.iteration.iloc[-1])
            assert np.array_equal(hist.iteration.to_numpy(), np.arange(1, step + 1))
            before = np.load(od / 'final_model.npy').copy()
            run_one(method, sc, P=8, steps=100, outroot=out, start_step=step)
            after = np.load(od / 'final_model.npy')
            repaired.append(dict(scenario=sc, method=method, resumed_from=step, terminal_model_max_abs_delta=float(np.max(abs(after - before))), timing_scope='resumed updates only'))
    if repaired:
        (out / 'history_recovery.json').write_text(json.dumps(repaired, indent=2))
    rows = []
    for sc in a.scenarios:
        dobs, freq = scenario_data(sc)
        for method in a.methods:
            od = out / 'P8' / sc / method
            hist = pd.read_csv(od / 'history.csv')
            for step in (25, 50, 100):
                c = np.load(od / 'checkpoints' / f'model_iter_{step:04d}.npy')
                with torch.no_grad():
                    dt, yf = M.sim(torch.tensor(c, dtype=M.DTYPE), f=freq)
                    pred = M.rs(yf, dt).numpy()
                rel, corr = M.metrics(c)
                h = hist[hist.iteration == step].iloc[0]
                row = dict(scenario=sc, method=method, budget=step, relative_model_error=float(rel), correlation=float(corr), common_std_residual=float(np.linalg.norm(pred - dobs) / np.linalg.norm(dobs)), gradient_preupdate=float(h['gradnorm']), objective_preupdate=float(h['total_preupdate']))
                rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'budget_metrics.csv', index=False)
    print('COMPLETE', len(rows), 'checkpoint rows', flush=True)
if __name__ == '__main__':
    main()
