from pathlib import Path
import importlib, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
mods = ["numpy", "scipy", "pandas", "matplotlib", "torch", "qvvw2"]
print("Repository:", ROOT)
print("Python:", sys.version.replace("\n", " "))
for name in mods:
    m = importlib.import_module(name)
    print(f"{name:12s}", getattr(m, "__version__", "import OK"))
print("Environment check complete.")
