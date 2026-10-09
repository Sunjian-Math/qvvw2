from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse.linalg as spla

from .input_validation import signal_vector, acquisition_shape, graph_edges

from .core import (
    lift, jt_eta, B_from_rho, two_layer_graph_edges,
    fixed_scale_lift, fixed_scale_jt_eta,
)

@dataclass
class QvvW2:
    """Frozen-tangent Q-vvW2 computational objective.

    The observation is fitted once.  Positive global rescaling of a signed datum
    is removed by the two-species quotient lift.  A custom acquisition graph can
    be supplied through ``edges``; otherwise a Cartesian two-layer graph is used.
    """
    epsilon: float = 0.15
    cross_weight: float = 0.25
    between_block_weight: float = 0.2
    data_shape: tuple[int, ...] | None = None
    _setup: dict | None = field(default=None, init=False, repr=False)

    def fit(self, observed, *, data_shape=None, edges=None):
        d = signal_vector(observed, self.epsilon)
        flat = d.ravel()
        rho, _ = lift(flat, self.epsilon)
        shape = acquisition_shape(tuple(data_shape or self.data_shape or d.shape or (flat.size,)), flat.size)
        if edges is None:
            edges = two_layer_graph_edges(shape, self.cross_weight, self.between_block_weight)
        edges = graph_edges(edges, 2 * flat.size)
        B = B_from_rho(rho, edges)
        solver = spla.factorized(B[:-1, :-1].tocsc())
        self._setup = {
            "observed": flat,
            "observed_shape": d.shape,
            "rho_d": rho,
            "B": B,
            "edges": edges,
            "solver": solver,
            "epsilon": float(self.epsilon),
            "data_shape": shape,
        }
        return self

    @property
    def is_fitted(self):
        return self._setup is not None

    def lifted(self, signal):
        epsilon = self.epsilon if self._setup is None else self._setup["epsilon"]
        return lift(signal_vector(signal, epsilon).ravel(), epsilon)[0]

    def value_grad(self, predicted):
        if self._setup is None:
            raise RuntimeError("Call fit(observed) before value_grad(predicted).")
        epsilon = self._setup["epsilon"]
        y = signal_vector(predicted, epsilon)
        if y.size != int(np.prod(self._setup["observed_shape"])):
            raise ValueError("Prediction size must match the fitted observation.")
        rho, _ = lift(y.ravel(), epsilon)
        dr = rho - self._setup["rho_d"]
        b = dr - dr.mean()
        xr = self._setup["solver"](b[:-1])
        eta = np.r_[xr, 0.0]
        eta -= eta.mean()
        value = 0.5 * float(dr @ eta)
        grad = jt_eta(y.ravel(), eta, epsilon).reshape(y.shape)
        return value, grad

    def value(self, predicted):
        return self.value_grad(predicted)[0]

    def gradient(self, predicted):
        return self.value_grad(predicted)[1]

    def __call__(self, predicted):
        return self.value(predicted)

@dataclass
class FixedScaleVvW2:
    """Non-quotient fixed-scale control; fitted parameters stay fixed until refit."""
    epsilon: float = 0.15
    cross_weight: float = 0.25
    between_block_weight: float = 0.2
    data_shape: tuple[int, ...] | None = None
    _setup: dict | None = field(default=None, init=False, repr=False)

    def fit(self, observed, *, data_shape=None, edges=None):
        d = signal_vector(observed, self.epsilon)
        flat = d.ravel()
        scale2 = float(np.mean(flat * flat))
        rho, _ = fixed_scale_lift(flat, scale2, self.epsilon)
        shape = acquisition_shape(tuple(data_shape or self.data_shape or d.shape or (flat.size,)), flat.size)
        if edges is None:
            edges = two_layer_graph_edges(shape, self.cross_weight, self.between_block_weight)
        edges = graph_edges(edges, 2 * flat.size)
        B = B_from_rho(rho, edges)
        solver = spla.factorized(B[:-1, :-1].tocsc())
        self._setup = {"rho_d":rho, "solver":solver, "reference_scale2":scale2,
                       "epsilon":float(self.epsilon), "observed_shape":d.shape}
        return self

    def value_grad(self, predicted):
        if self._setup is None:
            raise RuntimeError("Call fit(observed) before value_grad(predicted).")
        epsilon = self._setup["epsilon"]
        y = signal_vector(predicted, epsilon)
        if y.size != int(np.prod(self._setup["observed_shape"])):
            raise ValueError("Prediction size must match the fitted observation.")
        rho, _ = fixed_scale_lift(y.ravel(), self._setup["reference_scale2"], epsilon)
        dr = rho - self._setup["rho_d"]
        b = dr - dr.mean()
        xr = self._setup["solver"](b[:-1])
        eta = np.r_[xr, 0.0]; eta -= eta.mean()
        value = 0.5 * float(dr @ eta)
        grad = fixed_scale_jt_eta(y.ravel(), eta, self._setup["reference_scale2"], epsilon).reshape(y.shape)
        return value, grad

    def value(self, predicted): return self.value_grad(predicted)[0]
    def gradient(self, predicted): return self.value_grad(predicted)[1]
    def __call__(self, predicted): return self.value(predicted)
