import numpy as np
import pytest
from qvvw2 import QvvW2, FixedScaleVvW2
from qvvw2.core import lift, jacobian_dense, frozen_setup, frozen_objective_grad

def test_exact_ray_and_polarity():
    y=np.array([-.7,0.,.4,1.2]); rho=lift(y)[0]
    np.testing.assert_allclose(lift(3.7*y)[0],rho,rtol=0,atol=3e-16)
    np.testing.assert_allclose(lift(-y)[0],np.r_[rho[4:],rho[:4]],atol=3e-16)
    np.testing.assert_allclose(jacobian_dense(y)@y,0,atol=1e-15)
    assert np.all(rho>0) and abs(rho.sum()-1)<1e-15

def test_public_matches_frozen_core_and_derivative():
    rng=np.random.default_rng(4);d=rng.normal(size=(3,5));y=rng.normal(size=(3,5));v=rng.normal(size=y.shape)
    obj=QvvW2(cross_weight=.07).fit(d)
    f,g=obj.value_grad(y);ref,gr=frozen_objective_grad(y.ravel(),frozen_setup(d,.15,d.shape,.07,.2))
    assert f==ref
    np.testing.assert_array_equal(g.ravel(),gr)
    h=1e-6;fd=(obj.value(y+h*v)-obj.value(y-h*v))/(2*h)
    np.testing.assert_allclose(fd,np.sum(g*v),rtol=2e-7,atol=1e-8)
    np.testing.assert_allclose(obj.value(2.3*y),f,rtol=2e-14)
    np.testing.assert_allclose(np.sum(g*y),0,atol=1e-12)
    assert obj.value(d)<1e-25

def test_fixed_scale_has_same_observed_graph():
    d=np.array([-.3,.1,.8,-.5]);q=QvvW2().fit(d);f=FixedScaleVvW2().fit(d)
    np.testing.assert_array_equal(q._setup['rho_d'],f._setup['rho_d'])
    y=1.8*d
    assert q.value(y)<1e-25 and f.value(y)>1e-5

@pytest.mark.parametrize('d',[[0,0],[1],[1,float('nan')]])
def test_invalid_data(d):
    with pytest.raises(ValueError):QvvW2().fit(d)

def test_graph_and_shape_domain():
    with pytest.raises(ValueError):QvvW2().fit([1,2],edges=[(0,1,1),(2,3,1)])
    with pytest.raises(ValueError):QvvW2().fit([1,2],data_shape=(3,))
    with pytest.raises(ValueError):QvvW2(epsilon=0).fit([1,2])
    with pytest.raises(ValueError):QvvW2().fit([1,2]).value([1,2,3])


def test_small_epsilon_retains_both_species():
    # Regression: direct sqrt(1 + eps**2) - 1 rounded the minority masses to zero.
    y=np.array([1.,-1.]);eps=1e-8
    rho=lift(y,eps=eps)[0]
    assert np.all(rho>0)
    np.testing.assert_allclose(rho[[1,2]],eps**2/8,rtol=2e-15,atol=0)
    q=QvvW2(epsilon=eps).fit(y)
    f=FixedScaleVvW2(epsilon=eps).fit(y)
    np.testing.assert_array_equal(q._setup['rho_d'],f._setup['rho_d'])
    assert q.value(y)==0 and f.value(y)==0


@pytest.mark.parametrize('objective_cls',[QvvW2,FixedScaleVvW2])
def test_fitted_parameters_are_frozen_until_refit(objective_cls):
    d=np.array([-.8,.2,1.3,-.5]);y=np.array([-.6,.5,1.1,-.2])
    obj=objective_cls().fit(d);f,g=obj.value_grad(y)
    obj.epsilon=.5;obj.cross_weight=.07
    f_after,g_after=obj.value_grad(y)
    assert f_after==f
    np.testing.assert_array_equal(g_after,g)
    assert obj.value(d)==0
    if isinstance(obj,QvvW2):
        np.testing.assert_array_equal(obj.lifted(d),obj._setup['rho_d'])
    obj.fit(d)
    assert obj.value(d)==0
    assert abs(obj.value(y)-f)>1e-8
