from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qvvw2.external_baselines import (
    ensure_waveform_ot,
    waveform_ot_commit,
)

repo = ensure_waveform_ot()

print("waveform-ot ready")
print("repository:", repo)
print("commit:", waveform_ot_commit())
