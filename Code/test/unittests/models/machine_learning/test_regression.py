"""
Unit tests for Machine Learning Regression models.
"""
import sys
from pathlib import Path

import pytest
import torch
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from .....models.machine_learning.regression import (
    LinearRegression,
    RandomForestRegressor,
    SVR
)

@pytest.mark.unit
def test_linear_regression_fit_predict():
    X = torch.randn(100, 10)
    y = torch.randn(100)
    
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 100
    assert isinstance(preds, torch.Tensor)

@pytest.mark.unit
def test_random_forest_regressor_fit_predict():
    X = torch.randn(100, 10)
    y = torch.randn(100)
    
    model = RandomForestRegressor()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 100

@pytest.mark.unit
def test_svr_fit_predict():
    X = torch.randn(100, 10)
    y = torch.randn(100)
    
    model = SVR()
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == 100
