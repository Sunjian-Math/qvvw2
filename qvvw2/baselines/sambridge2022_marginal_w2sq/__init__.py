"""
Compatibility interface for the Marginal-W2sq baseline.

The numerical implementation is loaded from the public
waveform-ot reference repository accompanying
Sambridge, Jackson & Valentine (2022).

Third-party source code is not redistributed with qvvw2.
"""

from __future__ import annotations

import sys

from qvvw2.external_baselines import ensure_waveform_ot


_repo = ensure_waveform_ot()

if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))


from libs import OTlib
from libs import FingerprintLib
from libs import ricker_util


try:
    from libs import ricker_util_opt
except Exception:
    ricker_util_opt = None


try:
    from libs import myGP
except Exception:
    myGP = None


sys.modules[__name__ + ".OTlib"] = OTlib
sys.modules[__name__ + ".FingerprintLib"] = FingerprintLib
sys.modules[__name__ + ".ricker_util"] = ricker_util

if ricker_util_opt is not None:
    sys.modules[__name__ + ".ricker_util_opt"] = ricker_util_opt

if myGP is not None:
    sys.modules[__name__ + ".myGP"] = myGP


__all__ = [
    "OTlib",
    "FingerprintLib",
    "ricker_util",
    "ricker_util_opt",
    "myGP",
]
