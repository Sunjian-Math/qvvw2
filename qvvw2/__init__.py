"""Q-vvW2: scale-quotient signed quadratic vector-valued Wasserstein tools."""
from .metric import QvvW2, FixedScaleVvW2
from .core import lift, jacobian_dense, frozen_setup, frozen_objective_grad
from .baselines import L2, NormalizedL2, SoftplusW2Squared, UOT, HVMetric

__all__ = [
    "QvvW2", "FixedScaleVvW2", "lift", "jacobian_dense",
    "frozen_setup", "frozen_objective_grad", "L2", "NormalizedL2",
    "SoftplusW2Squared", "MarginalW2Squared", "UOT", "HVMetric",
]
__version__ = "1.0.0"

# QVVW2_LAZY_MARGINAL_BEGIN
def __getattr__(name):
    if name == "MarginalW2Squared":
        from .baselines import MarginalW2Squared
        return MarginalW2Squared
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
# QVVW2_LAZY_MARGINAL_END

