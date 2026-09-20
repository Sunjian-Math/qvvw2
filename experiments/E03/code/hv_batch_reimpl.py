from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from qvvw2.baselines.hv import solve_hv

def _worker(arg):
    i,f0,f1,p,x0=arg
    r=solve_hv(f0,f1,**p,x0=x0)
    return i,r['action'],r['grad_f0'],r['x'],r['nit'],r['success'],r['opt_grad_inf']

class HVBatch:
    def __init__(self, observed, params=None, workers=8, warm_start=True):
        self.D=np.asarray(observed,float).reshape(-1,observed.shape[-1]);self.shape=np.asarray(observed).shape
        self.params=dict(kappa=1e-4,lam=1e-4,eps=1e-12,nt=16,maxiter=300,gtol=1e-8)
        if params:self.params.update(params)
        self.workers=workers;self.warm_start=warm_start;self.cache=[None]*len(self.D);self.pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    def vg(self,y,update_cache=True):
        Y=np.asarray(y,float).reshape(-1,y.shape[-1]);args=[(i,Y[i],self.D[i],self.params,self.cache[i] if self.warm_start else None) for i in range(len(self.D))]
        if self.workers==1: rr=[_worker(a) for a in args]
        else:
            rr=list(self.pool.map(_worker,args,chunksize=1))
        rr.sort(key=lambda z:z[0])
        if update_cache and self.warm_start:self.cache=[r[3] for r in rr]
        vals=np.array([r[1] for r in rr]);gr=np.stack([r[2] for r in rr]);
        meta={'mean_inner_nit':float(np.mean([r[4] for r in rr])),'max_inner_nit':int(max(r[4] for r in rr)),'success_fraction':float(np.mean([r[5] for r in rr])),'max_opt_grad_inf':float(max(r[6] for r in rr))}
        return float(vals.mean()),(gr/len(vals)).reshape(y.shape),meta

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True); self.pool=None

    def __del__(self):
        try: self.close()
        except Exception: pass
