import numpy as np
from ..core import l2_objective_grad, normalized_l2_objective_grad

class L2:
    name = "L2"; display_name = "L^2"
    def fit(self, observed): self.observed=np.asarray(observed,float); return self
    def value_grad(self, predicted):
        v,g=l2_objective_grad(np.asarray(predicted).ravel(),self.observed.ravel()); return v,g.reshape(np.asarray(predicted).shape)
    def value(self,predicted): return self.value_grad(predicted)[0]
    def gradient(self,predicted): return self.value_grad(predicted)[1]

class NormalizedL2:
    name = "Normalized-L2"; display_name = "Normalized L^2"
    def fit(self, observed): self.observed=np.asarray(observed,float); return self
    def value_grad(self, predicted):
        v,g=normalized_l2_objective_grad(np.asarray(predicted).ravel(),self.observed.ravel()); return v,g.reshape(np.asarray(predicted).shape)
    def value(self,predicted): return self.value_grad(predicted)[0]
    def gradient(self,predicted): return self.value_grad(predicted)[1]
