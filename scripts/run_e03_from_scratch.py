"""Reproduce E03 with no bundled numerical arrays.

On a fresh clone every experiment is recomputed. On interruption, completed
outputs from this clone are validated then skipped; incomplete jobs are rerun.
This script does not read the article reference archive.
"""
from __future__ import annotations
import argparse, os, sys, subprocess, json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from qvvw2.registry import METHODS
SCENARIOS=('clean','gain_only','source_mismatch','gain_noise')


def run(cmd):
    print('RUN', ' '.join(str(x) for x in cmd),flush=True)
    p=subprocess.Popen([sys.executable,'-u',*map(str,cmd)],cwd=ROOT,stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
    for line in p.stdout:
        print(line,end='',flush=True)
    code=p.wait()
    if code:raise RuntimeError(f'Child process returned exit code {code}: {cmd}')


def verified_output(root,P,scenario,method,steps):
    folder=root/f'P{P}'/scenario/method
    s=folder/'summary.json';h=folder/'history.csv'; model=folder/'final_model.npy'
    if not (s.is_file() and h.is_file() and model.is_file()):return False
    j=json.loads(s.read_text());hist=pd.read_csv(h)
    if j.get('P')!=P or j.get('scenario')!=scenario or j.get('method')!=method or j.get('steps')!=steps:
        raise RuntimeError(f'Existing experiment has inconsistent protocol: {folder}')
    if len(hist)!=steps or not (hist.iteration.to_numpy()==list(range(1,steps+1))).all():
        raise RuntimeError(f'Existing experiment is truncated: {folder}')
    return True


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--workers',type=int,default=4)
    args=pa.parse_args()
    if args.workers <1:raise ValueError('workers must be positive')
    basic=ROOT/'experiments/E03/results'
    budget=ROOT/'experiments/E03/budget_results'
    total=0
    for sc in SCENARIOS:
        for m in METHODS:
            total+=1
            if verified_output(basic,8,sc,m,25):
                print('SKIP verified fresh local E03 run',sc,m,8,flush=True)
                continue
            run(['-m','experiments.E03.code.run_e03','--method',m,'--scenario',sc,'--P','8','--steps','25'])
    for P in (12,16):
        for m in METHODS:
            total+=1
            if verified_output(basic,P,'clean',m,25):
                print('SKIP verified fresh local E03 run','clean',m,P,flush=True)
                continue
            run(['-m','experiments.E03.code.run_e03','--method',m,'--scenario','clean','--P',P,'--steps','25'])
    assert total==36
    print('E03 original 25-step runs: 36/36 verified',flush=True)
    for sc in SCENARIOS:
        for m in METHODS:
            if verified_output(budget,8,sc,m,100):
                print('SKIP verified fresh local E03 100-step run',sc,m,flush=True)
            else:
                run(['-m','experiments.E03.code.run_e03','--method',m,'--scenario',sc,'--P','8','--steps','100','--outroot',budget])
    run(['supplemental/run_e03_budget.py','--output',budget.relative_to(ROOT)])
    bm=pd.read_csv(budget/'budget_metrics.csv')
    if len(bm)!=72:raise RuntimeError(f'E03 budget expects 72 checkpoint rows, got {len(bm)}')
    print('E03 budget: 24 complete trajectories, 72 checkpoints verified',flush=True)
    run(['final_validation/run_validation.py','--workers',args.workers])
    vm=pd.read_csv(ROOT/'final_validation/all_metrics.csv')
    counts={'mechanism':8,'ensemble':80,'sensitivity':8,'robustness':24}
    actual=vm.group.value_counts().to_dict()
    if actual!=counts:raise RuntimeError(f'E03 validation group counts wrong: {actual}')
    print('E03 final validation: 120/120 controls verified:',actual,flush=True)
    print('E03 COMPLETE from actual experiment computation (or validated local resume)',flush=True)

if __name__=='__main__':main()
