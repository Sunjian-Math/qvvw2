import numpy as np
import pytest
from qvvw2 import lift, L2, NormalizedL2, SoftplusW2Squared, UOT
from qvvw2.baselines.marginal_w2sq import MarginalW2Squared

@pytest.mark.parametrize('args',[([0.,0.],{}),([1.,2.],{'omega':[1.,-0.5]}),([1.,2.],{'omega':[1.]})])
def test_lift_rejects_invalid_domain(args):
    y,kw=args
    with pytest.raises(ValueError): lift(y,**kw)

def test_l2_rejects_broadcast_and_nonfinite():
    with pytest.raises(ValueError): L2().fit([1.]).value([1.,2.])
    with pytest.raises(ValueError): L2().fit([1.,np.nan])

def test_normalized_l2_rejects_zero_and_broadcast():
    with pytest.raises(ValueError): NormalizedL2().fit([0.,0.])
    with pytest.raises(ValueError): NormalizedL2().fit([1.,2.]).value([0.,0.])
    with pytest.raises(ValueError): NormalizedL2().fit([1.]).value([1.,2.])

def test_softplus_and_uot_fail_fast_on_domain_errors():
    with pytest.raises(ValueError): SoftplusW2Squared().fit([1.,np.nan,2.])
    with pytest.raises(ValueError): UOT(shape_iters=5).fit([1.,np.nan,2.])
    with pytest.raises(ValueError): UOT(x=np.linspace(0,1,3),shape_iters=5).fit(np.ones(4))

def test_marginal_invalid_input_does_not_trigger_external_fetch():
    # Validation occurs before the optional third-party baseline is imported.
    with pytest.raises(ValueError): MarginalW2Squared().fit([1.,np.nan,2.])
