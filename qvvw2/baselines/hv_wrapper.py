import numpy as np
from .hv import solve_hv

class HVMetric:
    name = "HV"; display_name = "HV metric"
    def __init__(self, **params):
        self.params=dict(kappa=1e-4,lam=1e-4,eps=1e-12,nt=16,maxiter=300,gtol=1e-8)
        self.params.update(params)
    def fit(self, observed): self.observed=np.asarray(observed,float); return self
    def value_grad(self,predicted):
        y=np.asarray(predicted,float); Y=y.reshape(-1,y.shape[-1]); D=self.observed.reshape(-1,self.observed.shape[-1])
        vals=[]; grads=[]
        for yy,dd in zip(Y,D):
            r=solve_hv(yy,dd,**self.params); vals.append(r['action']); grads.append(r['grad_f0'])
        return float(np.mean(vals)),(np.stack(grads)/len(vals)).reshape(y.shape)
    def value(self,predicted): return self.value_grad(predicted)[0]
    def gradient(self,predicted): return self.value_grad(predicted)[1]
