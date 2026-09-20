import numpy as np


class MarginalW2Squared:
    name = "Marginal-W2sq"
    display_name = "Marginal-W_2^2"

    def __init__(self, time_axis=None, lambdav=0.03, theta=45.0, nugrid=36):
        self.time_axis = None if time_axis is None else np.asarray(time_axis, float)
        self.lambdav = float(lambdav)
        self.theta = float(theta)
        self.nugrid = int(nugrid)
        self._ru = None

    def _reference(self):
        if self._ru is None:
            from .sambridge2022_marginal_w2sq import ricker_util
            self._ru = ricker_util
        return self._ru

    def fit(self, observed, *, time_axis=None):
        RU = self._reference()
        self.observed = np.asarray(observed, float)
        if time_axis is not None:
            self.time_axis = np.asarray(time_axis, float)
        if self.time_axis is None:
            self.time_axis = np.linspace(0, 1, self.observed.shape[-1])
        amax = 1.5 * max(
            abs(float(self.observed.min())),
            abs(float(self.observed.max())),
            1e-6,
        )
        self.grid = (
            float(self.time_axis[0]),
            float(self.time_axis[-1]),
            -amax,
            amax,
            self.nugrid,
            len(self.time_axis),
        )
        self.targets = []
        for tr in self.observed.reshape(-1, self.observed.shape[-1]):
            _, obj = RU.BuildOTobjfromWaveform(
                self.time_axis,
                tr,
                self.grid,
                lambdav=self.lambdav,
                theta=self.theta,
            )
            self.targets.append(obj)
        return self

    def value_grad(self, predicted):
        RU = self._reference()
        y = np.asarray(predicted, float)
        vals = []
        grads = []
        for i, tr in enumerate(y.reshape(-1, y.shape[-1])):
            wf, obj = RU.BuildOTobjfromWaveform(
                self.time_axis,
                tr,
                self.grid,
                lambdav=self.lambdav,
                deriv=True,
                theta=self.theta,
            )
            try:
                value, grad, _ = RU.CalcWasserWaveform(
                    obj,
                    self.targets[i],
                    wf,
                    distfunc="W2",
                    deriv=True,
                )
                grad = np.nan_to_num(
                    np.asarray(grad, float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            except Exception:
                value = RU.CalcWasserWaveform(
                    obj,
                    self.targets[i],
                    wf,
                    distfunc="W2",
                    deriv=False,
                )
                grad = np.zeros_like(tr)
            vals.append(float(value))
            grads.append(grad)
        return float(np.mean(vals)), (np.stack(grads) / len(vals)).reshape(y.shape)

    def value(self, predicted):
        return self.value_grad(predicted)[0]

    def gradient(self, predicted):
        return self.value_grad(predicted)[1]
