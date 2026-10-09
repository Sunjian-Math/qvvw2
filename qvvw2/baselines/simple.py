import numpy as np
from ..core import l2_objective_grad, normalized_l2_objective_grad
from ..input_validation import finite_array

class L2:
    name = 'L2'
    display_name = 'L^2'

    def fit(self, observed):
        self.observed = finite_array(observed, name='observed')
        return self

    def value_grad(self, predicted):
        y = finite_array(predicted, name='predicted')
        if y.size != self.observed.size:
            raise ValueError('prediction size must match the fitted observation.')
        v, g = l2_objective_grad(y.ravel(), self.observed.ravel())
        return (v, g.reshape(y.shape))

    def value(self, predicted):
        return self.value_grad(predicted)[0]

    def gradient(self, predicted):
        return self.value_grad(predicted)[1]

class NormalizedL2:
    name = 'Normalized-L2'
    display_name = 'Normalized L^2'

    def fit(self, observed):
        self.observed = finite_array(observed, name='observed')
        if not np.any(self.observed != 0):
            raise ValueError('Normalized-L2 requires a nonzero observation.')
        return self

    def value_grad(self, predicted):
        y = finite_array(predicted, name='predicted')
        if y.size != self.observed.size:
            raise ValueError('prediction size must match the fitted observation.')
        if not np.any(y != 0):
            raise ValueError('Normalized-L2 requires a nonzero prediction.')
        v, g = normalized_l2_objective_grad(y.ravel(), self.observed.ravel())
        return (v, g.reshape(y.shape))

    def value(self, predicted):
        return self.value_grad(predicted)[0]

    def gradient(self, predicted):
        return self.value_grad(predicted)[1]
