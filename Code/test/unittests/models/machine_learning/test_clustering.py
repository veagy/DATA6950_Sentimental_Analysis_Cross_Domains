"""
Unit tests for Machine Learning Clustering models.
"""
import sys
from pathlib import Path

import pytest
import torch
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from .....models.machine_learning.clustering import (
    KMeansCluster,
    DBSCAN
)

@pytest.mark.unit
def test_kmeans_cluster_fit_predict():
    X = torch.randn(100, 10)
    
    model = KMeansCluster(n_clusters=3)
    model.fit(X)
    labels = model.predict(X)
    
    assert len(labels) == 100
    assert len(torch.unique(labels)) <= 3

@pytest.mark.unit
def test_dbscan_fit_predict():
    X = torch.randn(100, 10)
    
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X)
    
    assert len(labels) == 100
