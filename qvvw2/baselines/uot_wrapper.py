import numpy as np
from .uot import li_uot_value_grad_batch_fast
from ..input_validation import finite_array, coordinate_axis

class UOT:
    name = 'UOT'
    display_name = 'UOT'

    def __init__(self, x=None, reg=0.001, reg_m=1.0, k=None, shape_iters=2000):
        self.x = x
        self.reg = reg
        self.reg_m = reg_m
        self.k = k
        self.shape_iters = shape_iters

    def fit(self, observed, *, x=None):
        self.observed = finite_array(observed, name='observed', min_size=2)
        if self.observed.ndim == 0 or self.observed.shape[-1] < 2:
            raise ValueError('UOT requires traces with at least two samples.')
        if x is not None:
            self.x = x
        if self.x is None:
            self.x = np.linspace(0, 1, self.observed.shape[-1])
        self.x = coordinate_axis(self.x, self.observed.shape[-1], name='x')
        if not np.isfinite(self.reg) or self.reg <= 0 or not np.isfinite(self.reg_m) or self.reg_m <= 0:
            raise ValueError('reg and reg_m must be finite and positive.')
        if not isinstance(self.shape_iters, (int, np.integer)) or self.shape_iters <= 0:
            raise ValueError('shape_iters must be a positive integer.')
        if self.k is None:
            self.k = 1.5 / max(float(np.max(self.observed)), 1e-12)
        if not np.isfinite(self.k) or self.k <= 0:
            raise ValueError('k must be finite and positive.')
        return self

    def value_grad(self, predicted):
        y = finite_array(predicted, name='predicted', min_size=2)
        d = self.observed
        if y.shape != d.shape:
            raise ValueError('prediction shape must match the fitted observation.')
        vals, g, meta = li_uot_value_grad_batch_fast(y.reshape(-1, y.shape[-1]), d.reshape(-1, d.shape[-1]), self.x, reg=self.reg, reg_m=self.reg_m, k=self.k, shape_iters=self.shape_iters)
        return (float(np.mean(vals)), (g / len(vals)).reshape(y.shape))

    def value(self, predicted):
        return self.value_grad(predicted)[0]

    def gradient(self, predicted):
        return self.value_grad(predicted)[1]
