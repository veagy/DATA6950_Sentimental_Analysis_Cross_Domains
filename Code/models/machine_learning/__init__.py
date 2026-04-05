"""
Thesis tabular ML only (docs/ml Track A).

Configs under ``Code/thesis/config/ml/{2,3}_labels/`` use **LogisticRegression** and
**LinearSVC** only. This module does **not** import clustering, preprocessing,
transformers, or the rest of the historical sklearn-style tree (often incomplete
in minimal checkouts).
"""

from __future__ import annotations

import importlib

_lm = importlib.import_module(
    "Code.models.machine_learning.classification.linear_model.linear_models"
)
_svm = importlib.import_module("Code.models.machine_learning.classification.svm.svm")

LogisticRegression = _lm.LogisticRegression
LogisticRegressionCV = _lm.LogisticRegressionCV
LinearSVC = _svm.LinearSVC
SVC = _svm.SVC
NuSVC = _svm.NuSVC

__all__ = [
    "LogisticRegression",
    "LogisticRegressionCV",
    "LinearSVC",
    "SVC",
    "NuSVC",
]
