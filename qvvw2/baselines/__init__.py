from .simple import L2, NormalizedL2
from .softplus_w2sq import SoftplusW2Squared
from .uot_wrapper import UOT
from .hv_wrapper import HVMetric

__all__ = ["L2", "NormalizedL2", "SoftplusW2Squared", "MarginalW2Squared", "UOT", "HVMetric"]

# QVVW2_LAZY_MARGINAL_BEGIN
def __getattr__(name):
    if name == "MarginalW2Squared":
        from .marginal_w2sq import MarginalW2Squared
        return MarginalW2Squared
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
# QVVW2_LAZY_MARGINAL_END

