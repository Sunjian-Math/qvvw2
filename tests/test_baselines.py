import numpy as np
import pytest
from qvvw2.baselines.softplus_w2sq import batch_value_grad_exact, encode
from qvvw2.baselines.uot import li_uot_value_grad_batch, li_uot_value_grad_batch_fast

@pytest.mark.parametrize('solver',[li_uot_value_grad_batch,li_uot_value_grad_batch_fast])
def test_uot_saturated_amplitude_has_zero_chain_derivative(solver):
    y=np.array([[80.,-80.,0.2]]);d=np.array([[0.5,-0.2,0.1]]);x=np.linspace(0,1,3)
    kw=dict(k=1.,reg=.2,reg_m=1.)
    kw.update(dict(shape_iters=3000) if solver is li_uot_value_grad_batch_fast else dict(maxiter=10000))
    v,g,_=solver(y,d,x,**kw)
    assert np.isfinite(v).all()
    np.testing.assert_array_equal(g[0,:2],np.zeros(2))
    yp=y.copy();yp[0,0]+=.01
    np.testing.assert_array_equal(solver(yp,d,x,**kw)[0],v)

@pytest.mark.parametrize('beta',[-1.,0.,float('nan')])
def test_softplus_requires_positive_finite_beta(beta):
    y=np.array([.1,-.2,.3])
    with pytest.raises(ValueError):encode(y,beta)
    with pytest.raises(ValueError):batch_value_grad_exact(y,y,np.linspace(0,1,3),beta)

def test_softplus_rejects_unknown_reduction():
    y=np.array([.1,-.2,.3])
    with pytest.raises(ValueError):batch_value_grad_exact(y,y,np.linspace(0,1,3),1.,'median')

def test_uot_fast_reports_nonconverged_log_fixed_point_residual():
    y=np.array([[.5,-.3,.1,.6]]);d=np.array([[.4,-.4,.3,.2]]);x=np.linspace(0,1,4)
    _,_,early=li_uot_value_grad_batch_fast(y,d,x,k=1.5,shape_iters=2000)
    _,_,late=li_uot_value_grad_batch_fast(y,d,x,k=1.5,shape_iters=14000)
    assert early['log_fixed_point_residual']>1e-8
    assert late['log_fixed_point_residual']<1e-11
