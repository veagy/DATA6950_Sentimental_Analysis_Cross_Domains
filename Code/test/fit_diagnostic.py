
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Path setup
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ..models.utils.utils import DLModule
    print("SUCCESS: DLModule import")
except Exception as e:
    print(f"FAILURE: DLModule import: {e}")
    sys.exit(1)

class _Net(DLModule):
    def __init__(self, in_f=16, out_f=4):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)
    def forward(self, x):
        return self.fc(x)

def _loader(N=64, in_f=16, n_cls=4, batch_size=16):
    X = torch.randn(N, in_f)
    y = torch.randint(0, n_cls, (N,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

try:
    model = _Net()
    loader = _loader()
    print("Running fit...")
    history = model.fit(data=loader, epochs=1, loss="CrossEntropyLoss",
                        optimizer="adamw", show_progress_bar=False, verbose=False)
    print("SUCCESS: fit completed")
    print(f"History Type: {type(history)}")
except Exception as e:
    print(f"FAILURE: fit failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Testing MLClassifier...")
    from ..models.utils.utils import MLClassifier
    class _MLC(MLClassifier):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)
        def forward(self, x): return self.fc(x)
        def fit(self, X, y=None, **kwargs):
            from torch.utils.data import TensorDataset, DataLoader
            if y is not None:
                data = DataLoader(TensorDataset(X, y), batch_size=16)
            else: data = X
            return super().fit(data, **kwargs)
        def predict(self, X): 
            with torch.no_grad(): return torch.argmax(self.forward(X), dim=1)

    mlc = _MLC()
    X = torch.randn(32, 16)
    y = torch.randint(0, 4, (32,))
    res = mlc.fit(X, y, epochs=1, show_progress_bar=False, verbose=False)
    print(f"MLC Fit Return Type: {type(res)}")
    print(f"MLC Fit Status: {getattr(mlc, 'fit_status', 'N/A')}")
    print("SUCCESS: MLClassifier test completed")
except Exception as e:
    print(f"FAILURE: MLClassifier failed: {e}")
    import traceback
    traceback.print_exc()

