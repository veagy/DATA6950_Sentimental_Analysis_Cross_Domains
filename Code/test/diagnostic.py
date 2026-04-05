
import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

try:
    from ..models.utils.utils import DLModule, MLModule, MLClassifier, MLRegressor
    print("SUCCESS: Imports work")
except Exception as e:
    print(f"FAILURE: Import error: {e}")
    sys.exit(1)

model = DLModule()
print(f"DLModule type: {type(model)}")
print(f"DLModule methods: {[m for m in dir(model) if not m.startswith('_')]}")

try:
    ap = model.to_advanced_pipeline()
    print(f"to_advanced_pipeline color: SUCCESS")
except Exception as e:
    print(f"to_advanced_pipeline FAIL: {e}")

try:
    manifest = model.get_manifest()
    print(f"get_manifest keys: {list(manifest.keys())}")
except Exception as e:
    print(f"get_manifest FAIL: {e}")
