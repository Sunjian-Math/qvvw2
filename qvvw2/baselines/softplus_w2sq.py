"""Softplus-encoded one-dimensional W_2^2 following Qiu (2021).

Reference:
Lingyun Qiu, "Analysis of seismic inversion with optimal transportation and
softplus encoding", Inverse Problems 37 (2021) 095004.
DOI: 10.1088/1361-6420/ac1511

The implementation uses the paper's softplus encoding, probability-mass
normalization, and monotone CDF pseudo-inverse for one-dimensional quadratic
Wasserstein transport. The objective is trace-wise W_2^2.
"""
from __future__ import annotations
import numpy as np


def beta_from_observed(obs: np.ndarray, beta_unit: float = 2.0) -> float:
    """Freeze beta from observed-data amplitude only.

    Qiu (2021) uses beta=2 for unit-amplitude Ricker data. Here the same
    dimensionless choice is applied after one global observed-amplitude scaling.
    """
    a = float(np.max(np.abs(np.asarray(obs, dtype=float))))
    return float(beta_unit) / max(a, 1e-30)


def _softplus_beta(u: np.ndarray, beta: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    b = float(beta)
    if b == 0.0:
        raise ValueError("beta must be nonzero")
    return np.logaddexp(0.0, b * u) / abs(b)


def _softplus_prime(u: np.ndarray, beta: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    b = float(beta)
    z = b * u
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return np.sign(b) * out


def encode(u: np.ndarray, beta: float):
    s = _softplus_beta(u, beta)
    z = float(np.sum(s))
    if not np.isfinite(z) or z <= 0.0:
        raise FloatingPointError("invalid softplus mass")
    return s / z, s, z


def _pinv_from_cdfs(f0: np.ndarray, f1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Piecewise-linear f0^{-1}(f1(t)), matching Appendix-A logic."""
    f0 = np.asarray(f0, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    j = np.searchsorted(f0, f1, side="left")
    j = np.clip(j, 0, n - 1)
    out = np.empty(n, dtype=float)
    for k, jj in enumerate(j):
        q = f1[k]
        if jj <= 0:
            out[k] = t[0]
        elif jj >= n - 1 and f0[jj] < q:
            out[k] = t[-1]
        else:
            fl, fr = f0[jj - 1], f0[jj]
            den = fr - fl
            if den <= 1e-30:
                out[k] = t[jj]
            else:
                a = (q - fl) / den
                out[k] = (1.0 - a) * t[jj - 1] + a * t[jj]
    return out


def _right_integral_on_grid(g: np.ndarray, t: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    t = np.asarray(t, dtype=float)
    r = np.zeros_like(g)
    for i in range(len(t) - 2, -1, -1):
        r[i] = r[i + 1] + 0.5 * (g[i] + g[i + 1]) * (t[i + 1] - t[i])
    return r


def w2sq_value_grad_mass(p_pred: np.ndarray, p_obs: np.ndarray, t: np.ndarray):
    """Appendix-A one-dimensional W_2^2 and first variation wrt predicted mass."""
    p1 = np.asarray(p_pred, dtype=float)
    p0 = np.asarray(p_obs, dtype=float)
    t = np.asarray(t, dtype=float)
    if p1.ndim != 1 or p0.ndim != 1 or t.ndim != 1 or not (len(p1) == len(p0) == len(t)):
        raise ValueError("1D equal-length inputs required")
    p1 = p1 / np.sum(p1)
    p0 = p0 / np.sum(p0)
    f1 = np.cumsum(p1)
    f0 = np.cumsum(p0)
    f1[-1] = 1.0
    f0[-1] = 1.0
    phi0 = _pinv_from_cdfs(f0, f1, t)
    phi1 = _pinv_from_cdfs(f1, f0, t)
    value = float(np.sum(p1 * (phi0 - t) ** 2))
    r = _right_integral_on_grid(t - phi1, t)
    xi = np.interp(phi0, t, r, left=r[0], right=r[-1])
    zeta = (phi0 - t) ** 2 + 2.0 * xi
    return value, zeta


def trace_value_grad(pred: np.ndarray, obs: np.ndarray, t: np.ndarray, beta: float):
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    pp, _, zp = encode(pred, beta)
    po, _, _ = encode(obs, beta)
    value, zeta = w2sq_value_grad_mass(pp, po, t)
    dsp = _softplus_prime(pred, beta)
    c = float(np.sum(zeta * pp))
    grad = (dsp / zp) * (zeta - c)
    return value, grad


def batch_value(pred: np.ndarray, obs: np.ndarray, t: np.ndarray, beta: float, reduction: str = "mean") -> float:
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if pred.shape != obs.shape or pred.shape[-1] != len(t):
        raise ValueError("shape mismatch")
    p = pred.reshape(-1, pred.shape[-1])
    o = obs.reshape(-1, obs.shape[-1])
    vals = []
    for y, d in zip(p, o):
        pp, _, _ = encode(y, beta)
        po, _, _ = encode(d, beta)
        v, _ = w2sq_value_grad_mass(pp, po, t)
        vals.append(v)
    vals = np.asarray(vals, dtype=float)
    if reduction == "mean":
        return float(vals.mean())
    if reduction == "sum":
        return float(vals.sum())
    raise ValueError(reduction)


def batch_value_grad_exact(pred: np.ndarray, obs: np.ndarray, t: np.ndarray, beta: float, reduction: str = "mean"):
    """Exact local derivative of the same discrete pseudo-inverse objective.

    Search intervals are selected from detached CDFs; within the locally fixed
    intervals, automatic differentiation gives the derivative of the discrete
    objective actually optimized.
    """
    import torch

    pn = np.asarray(pred, dtype=float)
    on = np.asarray(obs, dtype=float)
    tn = np.asarray(t, dtype=float)
    if pn.shape != on.shape or pn.shape[-1] != len(tn):
        raise ValueError("shape mismatch")
    shape = pn.shape
    p = pn.reshape(-1, shape[-1])
    o = on.reshape(-1, shape[-1])
    bsz, n = p.shape
    dtype = torch.float64
    y = torch.tensor(p, dtype=dtype, requires_grad=True)
    d = torch.tensor(o, dtype=dtype)
    tt = torch.tensor(tn, dtype=dtype).view(1, n).expand(bsz, n)
    b = float(beta)
    sp = torch.nn.functional.softplus(b * y) / abs(b)
    so = torch.nn.functional.softplus(b * d) / abs(b)
    pp = sp / sp.sum(dim=1, keepdim=True)
    po = so / so.sum(dim=1, keepdim=True)
    f1 = torch.cumsum(pp, dim=1)
    f0 = torch.cumsum(po, dim=1)
    idx = torch.searchsorted(f0.detach().contiguous(), f1.detach().contiguous(), right=False)
    idx = torch.clamp(idx, 0, n - 1)
    j0 = torch.clamp(idx - 1, 0, n - 1)
    fl = torch.gather(f0, 1, j0)
    fr = torch.gather(f0, 1, idx)
    tl = torch.gather(tt, 1, j0)
    tr = torch.gather(tt, 1, idx)
    den = fr - fl
    safe_den = torch.where(den.abs() < 1e-30, torch.ones_like(den), den)
    a = torch.where((idx == 0) | (den.abs() < 1e-30), torch.zeros_like(f1), (f1 - fl) / safe_den)
    phi = (1.0 - a) * tl + a * tr
    val_each = torch.sum(pp * (phi - tt) ** 2, dim=1)
    value = val_each.mean() if reduction == "mean" else val_each.sum()
    value.backward()
    grad = y.grad.detach().numpy().reshape(shape)
    return float(value.detach()), grad, {
        "n_traces": int(bsz),
        "beta": float(beta),
        "discretization": "Qiu 2021 Appendix-A monotone pseudo-inverse",
    }


class SoftplusW2Squared:
    name = "Softplus-W2sq"
    display_name = "Softplus-W_2^2"
    def __init__(self, time_axis=None, beta=None, beta_unit=2.0, reduction="mean"):
        self.time_axis = None if time_axis is None else np.asarray(time_axis,float)
        self.beta = beta; self.beta_unit=float(beta_unit); self.reduction=reduction
    def fit(self, observed, *, time_axis=None):
        self.observed=np.asarray(observed,float)
        if time_axis is not None: self.time_axis=np.asarray(time_axis,float)
        if self.time_axis is None: self.time_axis=np.linspace(0,1,self.observed.shape[-1])
        if self.beta is None: self.beta=beta_from_observed(self.observed,self.beta_unit)
        return self
    def value_grad(self,predicted):
        v,g,_=batch_value_grad_exact(np.asarray(predicted,float),self.observed,self.time_axis,self.beta,self.reduction)
        return v,g
    def value(self,predicted): return self.value_grad(predicted)[0]
    def gradient(self,predicted): return self.value_grad(predicted)[1]
