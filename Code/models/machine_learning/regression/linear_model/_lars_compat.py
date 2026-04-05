"""Lars-family regressors are referenced across classification and kernel SVM code.

The full implementations are absent in some checkouts (empty ``lars/`` tree).  These
stubs keep imports working for workflows that only need LogisticRegression /
LinearSVC / tree / ensemble classifiers.  Instantiating a Lars stub raises
``NotImplementedError``.
"""

from __future__ import annotations

from Code.models.utils.utils import MLRegressor


class _StubLarsBase(MLRegressor):
    def fit(self, X, y):  # type: ignore[override]
        raise NotImplementedError(
            "Lars-family regressor is not available in this checkout (stub)."
        )

    def forward(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("Lars-family regressor stub cannot forward.")


class Lars(_StubLarsBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._stub_args = (args, kwargs)


class RidgeLars(_StubLarsBase):
    def __init__(self, *args, **kwargs):
        super().__init__()


class LassoLars(_StubLarsBase):
    def __init__(self, *args, **kwargs):
        super().__init__()


class ElasticNetLars(_StubLarsBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
