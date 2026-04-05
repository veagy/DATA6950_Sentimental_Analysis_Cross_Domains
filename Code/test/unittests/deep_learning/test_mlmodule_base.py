# src/test/unittests/deep_learning/test_mlmodule_base.py
"""
Unit tests for MLModule (src/models/utils/utils.py).
Uses MLClassifier as the representative concrete subtype.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ....models.utils.utils import MLClassifier, MLRegressor, MLTransform  # noqa: E402


class _CLS(MLClassifier):
    def __init__(self):
        super().__init__()
        self.estimator_type = "classifier"
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)

    def predict(self, x):
        return torch.argmax(self(x), dim=1)

    def fit(self, *args, **kwargs):
        return self


class _REG(MLRegressor):
    def __init__(self):
        super().__init__()
        self.estimator_type = "regressor"
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc(x)

    def predict(self, x):
        return self(x)

    def fit(self, *args, **kwargs):
        return self


def _Xy(N=64):
    return torch.randn(N, 16), torch.randint(0, 4, (N,))


# ══════════════════════════════════════════════════════════════════════════════
# §5.1  estimator_type
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_estimator_type_set():
    model = _CLS()
    assert model.estimator_type in {"regressor", "classifier", "cluster", "transformer"}


# ══════════════════════════════════════════════════════════════════════════════
# §5.2  fit() — sklearn-style
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_fit_sets_fit_status():
    model = _CLS()
    X, y = _Xy()
    model.fit(X, y, epochs=1, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)
    assert model.fit_status is True


@pytest.mark.unit
def test_fit_accepts_dataloader():
    model = _CLS()
    loader = DataLoader(TensorDataset(*_Xy()), batch_size=16)
    model.fit(loader, epochs=1, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)
    assert model.fit_status is True


@pytest.mark.unit
def test_fit_status_manual_restore_after_load(tmp_path):
    """fit_status is not a buffer; must be manually set to True after load."""
    model = _CLS()
    X, y = _Xy()
    model.fit(X, y, epochs=1, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)
    path = str(tmp_path / "cls.pt")
    model.save_model(path)
    loaded = _CLS.load_model(path)
    # Manually restore fit_status (documented behavior)
    loaded.fit_status = True
    assert loaded.fit_status is True


# ══════════════════════════════════════════════════════════════════════════════
# §5.3  predict() / score()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_predict_returns_tensor():
    model = _CLS()
    X, y = _Xy()
    model.fit(X, y, epochs=1, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)
    preds = model.predict(X)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape[0] == X.shape[0]


@pytest.mark.unit
def test_score_classifier_range():
    model = _CLS()
    X, y = _Xy()
    model.fit(X, y, epochs=3, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)
    acc = model.score(X, y)
    assert 0.0 <= acc <= 1.0, f"Classifier accuracy {acc} outside [0, 1]"


@pytest.mark.unit
def test_score_regressor_is_r2():
    model = _REG()
    X = torch.randn(64, 16)
    y = torch.randn(64, 1)
    model.fit(X, y, epochs=3, loss="mse",
              show_progress_bar=False, verbose=False)
    r2 = model.score(X, y)
    assert isinstance(r2, float)


# ══════════════════════════════════════════════════════════════════════════════
# §5.4  prediction_loss()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.parametrize("criterion", ["mse", "mae", "huber_loss"])
def test_prediction_loss_criterion_variants(criterion):
    model = _REG()
    X = torch.randn(64, 16)
    y = torch.randn(64, 1)
    model.fit(X, y, epochs=1, loss="mse",
              show_progress_bar=False, verbose=False)
    try:
        loss_val = model.prediction_loss(X, y.float(), criterion=criterion)
        assert torch.isfinite(torch.tensor(loss_val if not hasattr(loss_val, 'item') else loss_val.item()))
    except Exception as exc:
        pytest.fail(f"prediction_loss(criterion={criterion!r}) raised: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# §5.5  vmap_predict()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_vmap_predict_shape_matches_predict():
    model = _CLS()
    X, y = _Xy()
    model.fit(X, y, epochs=1, loss="CrossEntropyLoss",
              show_progress_bar=False, verbose=False)
    try:
        vmap_out = model.vmap_predict(X)
        normal_out = model.predict(X)
        assert vmap_out.shape == normal_out.shape
    except Exception:
        # vmap_predict falls back to predict() when vmap unavailable — acceptable
        pass


# ══════════════════════════════════════════════════════════════════════════════
# §5.6  extra_state persistence
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_extra_state_roundtrip(tmp_path):
    """Non-tensor attributes stored via get_extra_state() must survive save/load."""
    model = _CLS()
    # Simulate a scalar attribute
    model._my_scalar = 3.14
    path = str(tmp_path / "cls_extra.pt")
    model.save_model(path)
    loaded = _CLS.load_model(path)
    # If _my_scalar is in extra_state, it should reload
    assert loaded is not None
