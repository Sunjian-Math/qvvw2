import numpy as np
from qvvw2 import QvvW2
from qvvw2.core import lift, jacobian_dense

def signal(n=96):
    t=np.linspace(0,1,n)
    return np.exp(-.5*((t-.3)/.06)**2)-.65*np.exp(-.5*((t-.7)/.08)**2)

def test_positive_scale_lift_invariance():
    d=signal(); r0=lift(d,.15)[0]
    assert np.linalg.norm(lift(2.3*d,.15)[0]-r0) < 1e-12

def test_radial_jacobian_kernel():
    d=signal(); J=jacobian_dense(d,.15)
    rel=np.linalg.norm(J@d)/max(np.linalg.norm(J)*np.linalg.norm(d),1e-30)
    assert rel < 1e-12

def test_metric_value_and_gradient_fd():
    d=signal(48); q=QvvW2(epsilon=.15,cross_weight=.03).fit(d,data_shape=(48,))
    rng=np.random.default_rng(3); y=d+.03*rng.normal(size=d.size); h=rng.normal(size=d.size); h/=np.linalg.norm(h)
    v,g=q.value_grad(y); eps=1e-6; fd=(q.value(y+eps*h)-q.value(y-eps*h))/(2*eps)
    rel=abs(fd-g@h)/max(abs(fd),abs(g@h),1e-14)
    assert rel < 5e-6
