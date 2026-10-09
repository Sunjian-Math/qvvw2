"""Create paper-plot inputs ONLY from files freshly produced by current experiment runs.

No reference arrays are downloaded, bundled, copied from the publication archive,
or synthesized as missing values. Missing runs fail with an explicit path.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.qvvw2_generated' / 'plot_inputs'


def copy_file(src: Path, dst: Path):
    if not src.is_file():
        raise FileNotFoundError(f'Cannot prepare plotting data: fresh experiment output missing: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst)


def copy_dir(src: Path, dst: Path):
    if not src.is_dir() or not any(p.is_file() for p in src.rglob('*')):
        raise FileNotFoundError(f'Fresh experiment directory missing/empty: {src}')
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__'))


def prepare(phase: str):
    r = ROOT/'experiments'
    if phase == 'E00':
        return
    if phase in ('E01','E02','E04'):
        copy_dir(r/phase/'results', OUT/phase)
        return
    if phase == 'E03':
        copy_dir(r/'E03'/'results', OUT/'E03'/'original')
        copy_dir(r/'E03'/'budget_results', OUT/'E03'/'budget')
        src=ROOT/'final_validation'/'all_metrics.csv'
        if not src.is_file(): raise FileNotFoundError('Run final_validation/run_validation.py first; missing all_metrics.csv')
        full=pd.read_csv(src)
        counts={'mechanism':8,'ensemble':80,'robustness':24,'sensitivity':8}
        for group,count in counts.items():
            n=int((full['group']==group).sum())
            if n!=count: raise ValueError(f'Incomplete final validation: {group} has {n}, needs {count}')
        dst=OUT/'mechanism';dst.mkdir(parents=True,exist_ok=True)
        # Preserve only the published per-seed jitter presentation order; data unchanged.
        try:
            from scripts.presentation_seed_order import arrange_fig6_seed_display
        except ModuleNotFoundError as exc:
            if exc.name != 'scripts': raise
            from presentation_seed_order import arrange_fig6_seed_display
        arrange_fig6_seed_display(full).to_csv(dst/'all_metrics.csv',index=False)
        # The control table combines 8 independently optimized component controls
        # with 8 *different* 100-step Adam trajectories (not returned by final_validation).
        rows=[]
        for scenario in ('clean','gain_only','source_mismatch','gain_noise'):
            for method in ('Normalized-L2','Q-vvW2'):
                folder=OUT/'E03'/'budget'/'P8'/scenario/method
                summary=folder/'summary.json'
                history=folder/'history.csv'
                if not summary.is_file() or not history.is_file():
                    raise FileNotFoundError(f'Need genuine 100-step outputs: {folder}')
                import json
                d=json.loads(summary.read_text())
                h=pd.read_csv(history)
                if len(h)!=100 or int(h.iteration.iloc[-1])!=100:
                    raise ValueError(f'Incomplete 100-step trajectory: {history}')
                last=h.iloc[-1]
                rows.append({'group':'mechanism','scenario':scenario,'method':method,
                  'relative_model_error':float(d['relative_model_error']),
                  'correlation':float(d['correlation']),
                  'common_std_residual':float(d['common_std_residual']),
                  'budget':100,
                  'gradient_preupdate':float(last['gradnorm']),
                  'objective_preupdate':float(last['total_preupdate']),
                  'source':'fresh 100-step Adam trajectory'})
        combined=pd.concat([full[full.group=='mechanism'],pd.DataFrame(rows)],ignore_index=True)
        if len(combined)!=16: raise ValueError('Mechanism comparison must contain exactly 16 rows')
        combined.to_csv(dst/'mechanism_comparison.csv',index=False)
        full[full.group=='robustness'].to_csv(dst/'optimizer_robustness.csv',index=False)
        return
    if phase == 'E05':
        src=r/'E05'/'results';dst=OUT/'E05'
        for a,b in [('tdem_traces.npz','tdem_traces.npz'),('objectives_FINAL.csv','objectives.csv'),('multistart_FINAL.csv','multistart.csv'),('summary_FINAL.csv','summary.csv')]:
            copy_file(src/a,dst/b)
        return
    raise ValueError(phase)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phases', nargs='+', choices=['E00','E01','E02','E03','E04','E05'])
    args=ap.parse_args()
    for phase in args.phases:
        prepare(phase)
        print(f'{phase}: prepared outputs from freshly computed experiments',flush=True)


if __name__=='__main__':main()
