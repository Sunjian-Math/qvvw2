import numpy as np
from .uot import li_uot_value_grad_batch_fast

class UOT:
    name = "UOT"; display_name = "UOT"
    def __init__(self, x=None, reg=1e-3, reg_m=1.0, k=None, shape_iters=2000):
        self.x=x; self.reg=reg; self.reg_m=reg_m; self.k=k; self.shape_iters=shape_iters
    def fit(self, observed, *, x=None):
        self.observed=np.asarray(observed,float)
        if x is not None: self.x=np.asarray(x,float)
        if self.x is None: self.x=np.linspace(0,1,self.observed.shape[-1])
        if self.k is None: self.k=1.5/max(float(np.max(self.observed)),1e-12)
        return self
    def value_grad(self,predicted):
        y=np.asarray(predicted,float); d=self.observed
        vals,g,meta=li_uot_value_grad_batch_fast(y.reshape(-1,y.shape[-1]),d.reshape(-1,d.shape[-1]),self.x,
            reg=self.reg,reg_m=self.reg_m,k=self.k,shape_iters=self.shape_iters)
        return float(np.mean(vals)),(g/len(vals)).reshape(y.shape)
    def value(self,predicted): return self.value_grad(predicted)[0]
    def gradient(self,predicted): return self.value_grad(predicted)[1]
