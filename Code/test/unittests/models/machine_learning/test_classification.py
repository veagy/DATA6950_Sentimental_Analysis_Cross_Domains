"""
Unit tests for Machine Learning Classification models.
"""
import sys
from pathlib import Path

import pytest
import torch
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from .....models.machine_learning.classification import (
    LogisticRegression,
    RandomForestClassifier,
    SVC
)

@pytest.mark.unit
def test_logistic_regression_fit_predict():
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 100
    assert torch.all((preds == 0) | (preds == 1))

@pytest.mark.unit
def test_random_forest_classifier_fit_predict():
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    model = RandomForestClassifier()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 100

@pytest.mark.unit
def test_svc_fit_predict():
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    model = SVC()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 100
