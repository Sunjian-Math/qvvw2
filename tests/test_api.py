import qvvw2
from qvvw2 import (
    QvvW2,
    L2,
    NormalizedL2,
    SoftplusW2Squared,
    MarginalW2Squared,
    UOT,
    HVMetric,
)


def test_public_api_imports():
    assert QvvW2 is not None
    assert all(
        x is not None
        for x in (
            L2,
            NormalizedL2,
            SoftplusW2Squared,
            MarginalW2Squared,
            UOT,
            HVMetric,
        )
    )


def test_public_version():
    assert qvvw2.__version__ == "1.0.0"
