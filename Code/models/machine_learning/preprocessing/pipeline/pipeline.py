import warnings
import math
import time
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union, Any, List, Tuple, Dict, Literal
import pandas as pd  # pyright: ignore[reportMissingImports]
import numpy as np
from ....utils.utils import MLModule
from torch.func import vmap
import joblib


__all__ = [
    "Pipeline",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _Bunch(dict):
    """Dict subclass with attribute-style read/write access (mirrors sklearn Bunch)."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Bunch' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'Bunch' object has no attribute '{key}'")

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"Bunch({items})"


def _to_tensor(X: Any, device, dtype) -> Any:
    """Attempt to convert X to a torch.Tensor; return X unchanged on failure."""
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=dtype)
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X).to(device=device, dtype=dtype)
    try:
        return torch.as_tensor(X, device=device, dtype=dtype)
    except Exception:
        return X


def _get_method_params(obj: Any, method_name: str) -> List[str]:
    """Return the parameter names of ``obj.method_name``, or [] on failure."""
    try:
        sig = inspect.signature(getattr(obj, method_name))
        return list(sig.parameters.keys())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline(MLModule):
    def __init__(self,
                 steps: Union[List[MLModule], Tuple[MLModule], MLModule] = None,
                 transform_input: Union[List[str], Tuple[str], str] = None,
                 memory: Union[str, object] = "torch",
                 verbose: bool = False,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__()
        self.steps = self._validate_steps(steps)
        self.transform_input = transform_input
        self.memory = memory
        self.verbose = verbose
        self.device = device
        self.dtype = dtype

        # Fitted attributes (docstring-specified)
        self.named_steps: _Bunch = _Bunch(
            {name: est for name, est in self.steps}
        )
        self.classes_: Optional[Any] = None
        self.n_features_in_: Optional[int] = None

        # Internal in-memory cache (used when memory="torch")
        self._fit_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Step normalization helpers
    # ------------------------------------------------------------------

    def _validate_steps(
        self,
        steps: Union[List, Tuple, Any],
    ) -> List[Tuple[str, Any]]:
        if steps is None:
            return []

        # Single non-iterable estimator
        if not isinstance(steps, (list, tuple)):
            return [("step_0", steps)]

        result: List[Tuple[str, Any]] = []
        for i, step in enumerate(steps):
            if isinstance(step, (list, tuple)) and len(step) == 2:
                name, est = step
                result.append((str(name), est))
            else:
                # Bare estimator — auto-name
                result.append((f"step_{i}", step))
        return result

    def _is_passthrough(self, estimator: Any) -> bool:
        """Return True if *estimator* should be skipped (None or 'passthrough')."""
        return estimator is None or estimator == "passthrough"

    def _update_named_steps(self) -> None:
        """Sync ``named_steps`` with the current ``steps`` list."""
        self.named_steps = _Bunch(
            {name: est for name, est in self.steps}
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _final_estimator(self) -> Any:
        """Return the last estimator (or None if steps is empty)."""
        if not self.steps:
            return None
        return self.steps[-1][1]

    @property
    def _final_step_name(self) -> Optional[str]:
        if not self.steps:
            return None
        return self.steps[-1][0]

    # ------------------------------------------------------------------
    # Internal transform helpers
    # ------------------------------------------------------------------

    def _fit_transform_one(
        self,
        name: str,
        estimator: Any,
        X: Any,
        y: Any,
        **params,
    ) -> Any:
        """Fit *estimator* on (X, y) and return the transformed X."""
        t0 = time.time()

        if hasattr(estimator, "fit_transform"):
            Xt = estimator.fit_transform(X, y, **params)
        else:
            estimator.fit(X, y, **params)
            Xt = estimator.transform(X, **{
                k: v for k, v in params.items()
                if k in _get_method_params(estimator, "transform")
            })

        if self.verbose:
            print(f"[Pipeline] Step '{name}' fitted in {time.time() - t0:.3f}s")

        return Xt

    def _transform_up_to(
        self,
        X: Any,
        stop: int = -1,
        **params,
    ) -> Any:
        """Transform X through all steps up to (not including) index *stop*.

        Parameters
        ----------
        stop : int
            Exclusive upper bound. Use ``-1`` to include all but the last step
            (standard behaviour when preparing data for the final estimator).
        """
        step_params = self._split_params(params)
        steps = self.steps[:stop] if stop == -1 else self.steps[:stop]
        # Recalculate: -1 means everything except the last
        if stop == -1:
            steps = self.steps[:-1]

        Xt = X
        for name, estimator in steps:
            if self._is_passthrough(estimator):
                continue
            s_params = step_params.get(name, {})
            if hasattr(estimator, "transform"):
                Xt = estimator.transform(Xt, **{
                    k: v for k, v in s_params.items()
                    if k in _get_method_params(estimator, "transform")
                })
            # Steps without transform are silently skipped (only final should lack it)
        return Xt

    # ------------------------------------------------------------------
    # param-splitting
    # ------------------------------------------------------------------

    def _split_params(self, params: dict) -> Dict[str, dict]:
        """Convert ``{'step__param': value, …}`` → ``{'step': {'param': value}, …}``."""
        step_params: Dict[str, dict] = {name: {} for name, _ in self.steps}
        for key, val in params.items():
            if "__" in key:
                step_name, param_name = key.split("__", 1)
                if step_name in step_params:
                    step_params[step_name][param_name] = val
                # Params for unknown steps are silently ignored
        return step_params

    # ------------------------------------------------------------------
    # Memory caching helpers
    # ------------------------------------------------------------------

    def _cache_key(self, name: str, estimator: Any) -> str:
        return f"{name}_{id(estimator)}"

    def _get_joblib_memory(self):
        """Return a joblib.Memory instance or None."""
        if self.memory is None or self.memory == "torch":
            return None
        try:
            import joblib  # pyright: ignore[reportMissingImports]
            if isinstance(self.memory, str):
                return joblib.Memory(self.memory, verbose=0)
            # Already a Memory-like object
            return self.memory
        except ImportError:
            warnings.warn(
                "joblib is not installed; pipeline memory caching is disabled. "
                "Install joblib or use memory='torch' for in-process caching.",
                UserWarning,
                stacklevel=3,
            )
            return None

    # ------------------------------------------------------------------
    # Public API — fit / transform / predict
    # ------------------------------------------------------------------

    def fit(self, data_or_X, y=None, **params) -> "Pipeline":
        step_params = self._split_params(params)
        jl_mem = self._get_joblib_memory()

        Xt = data_or_X
        for i, (name, estimator) in enumerate(self.steps[:-1]):
            if self._is_passthrough(estimator):
                continue

            s_params = step_params.get(name, {})

            if jl_mem is not None:
                # joblib-based caching (path or Memory object)
                try:
                    cached = jl_mem.cache(self._fit_transform_one)
                    Xt = cached(name, estimator, Xt, y, **s_params)
                except Exception:
                    Xt = self._fit_transform_one(name, estimator, Xt, y, **s_params)
            else:
                Xt = self._fit_transform_one(name, estimator, Xt, y, **s_params)

            # In-memory torch cache: just record that the step is fitted
            if self.memory == "torch":
                self._fit_cache[self._cache_key(name, estimator)] = True

        # Fit final estimator
        final = self._final_estimator
        if final is not None and not self._is_passthrough(final):
            final_name = self._final_step_name
            s_params = step_params.get(final_name, {})
            t0 = time.time()
            final.fit(Xt, y, **s_params)
            if self.verbose:
                print(
                    f"[Pipeline] Final step '{final_name}' fitted "
                    f"in {time.time() - t0:.3f}s"
                )

        # Expose fitted attributes
        self._update_named_steps()
        first_est = next(
            (est for _, est in self.steps if not self._is_passthrough(est)), None
        )
        if first_est is not None and hasattr(first_est, "n_features_in_"):
            self.n_features_in_ = first_est.n_features_in_

        if final is not None and hasattr(final, "classes_"):
            self.classes_ = final.classes_

        self.fit_status = True
        return self

    def fit_transform(self, data_or_X, y=None, **params) -> Any:
        step_params = self._split_params(params)
        jl_mem = self._get_joblib_memory()

        Xt = data_or_X
        for name, estimator in self.steps[:-1]:
            if self._is_passthrough(estimator):
                continue
            s_params = step_params.get(name, {})
            if jl_mem is not None:
                try:
                    cached = jl_mem.cache(self._fit_transform_one)
                    Xt = cached(name, estimator, Xt, y, **s_params)
                except Exception:
                    Xt = self._fit_transform_one(name, estimator, Xt, y, **s_params)
            else:
                Xt = self._fit_transform_one(name, estimator, Xt, y, **s_params)
            if self.memory == "torch":
                self._fit_cache[self._cache_key(name, estimator)] = True

        # Handle final step
        final = self._final_estimator
        final_name = self._final_step_name
        if final is not None and not self._is_passthrough(final):
            s_params = step_params.get(final_name, {})
            if hasattr(final, "fit_transform"):
                Xt = final.fit_transform(Xt, y, **s_params)
            else:
                final.fit(Xt, y, **s_params)
                if hasattr(final, "transform"):
                    Xt = final.transform(Xt)

        self._update_named_steps()
        first_est = next(
            (est for _, est in self.steps if not self._is_passthrough(est)), None
        )
        if first_est is not None and hasattr(first_est, "n_features_in_"):
            self.n_features_in_ = first_est.n_features_in_
        if final is not None and hasattr(final, "classes_"):
            self.classes_ = final.classes_

        self.fit_status = True
        return Xt

    def transform(self, X, **params) -> Any:
        step_params = self._split_params(params)
        Xt = X
        for name, estimator in self.steps:
            if self._is_passthrough(estimator):
                continue
            s_params = step_params.get(name, {})
            if hasattr(estimator, "transform"):
                Xt = estimator.transform(Xt, **{
                    k: v for k, v in s_params.items()
                    if k in _get_method_params(estimator, "transform")
                })
            elif hasattr(estimator, "predict"):
                # Final-step-only estimator without a transform method
                Xt = estimator.predict(Xt)
        return Xt

    def predict(self, X, **params) -> Any:
        Xt = self._transform_up_to(X, stop=-1)
        final = self._final_estimator
        if final is None or self._is_passthrough(final):
            return Xt
        final_name = self._final_step_name
        s_params = {
            k[len(final_name) + 2:]: v
            for k, v in params.items()
            if k.startswith(final_name + "__")
        }
        if not hasattr(final, "predict"):
            raise AttributeError(
                f"The final estimator '{type(final).__name__}' does not "
                "implement 'predict'."
            )
        return final.predict(Xt, **s_params)

    def predict_proba(self, X, **params) -> Any:
        Xt = self._transform_up_to(X, stop=-1)
        final = self._final_estimator
        if final is None or self._is_passthrough(final):
            return Xt
        if not hasattr(final, "predict_proba"):
            raise AttributeError(
                f"The final estimator '{type(final).__name__}' does not "
                "implement 'predict_proba'."
            )
        final_name = self._final_step_name
        s_params = {
            k[len(final_name) + 2:]: v
            for k, v in params.items()
            if k.startswith(final_name + "__")
        }
        return final.predict_proba(Xt, **s_params)

    def predict_log_proba(self, X, **params) -> Any:
        Xt = self._transform_up_to(X, stop=-1)
        final = self._final_estimator
        if final is None or self._is_passthrough(final):
            return Xt

        final_name = self._final_step_name
        s_params = {
            k[len(final_name) + 2:]: v
            for k, v in params.items()
            if k.startswith(final_name + "__")
        }

        if hasattr(final, "predict_log_proba"):
            return final.predict_log_proba(Xt, **s_params)
        if hasattr(final, "predict_proba"):
            proba = final.predict_proba(Xt, **s_params)
            if isinstance(proba, torch.Tensor):
                return torch.log(proba.clamp(min=1e-12))
            return np.log(np.clip(np.asarray(proba), 1e-12, None))

        raise AttributeError(
            f"The final estimator '{type(final).__name__}' implements neither "
            "'predict_log_proba' nor 'predict_proba'."
        )

    def decision_function(self, X, **params) -> Any:
        Xt = self._transform_up_to(X, stop=-1)
        final = self._final_estimator
        if final is None or self._is_passthrough(final):
            return Xt
        if not hasattr(final, "decision_function"):
            raise AttributeError(
                f"The final estimator '{type(final).__name__}' does not "
                "implement 'decision_function'."
            )
        final_name = self._final_step_name
        s_params = {
            k[len(final_name) + 2:]: v
            for k, v in params.items()
            if k.startswith(final_name + "__")
        }
        return final.decision_function(Xt, **s_params)

    def score(self, X, y=None, sample_weight=None, **params) -> float:
        Xt = self._transform_up_to(X, stop=-1)
        final = self._final_estimator
        if final is None or self._is_passthrough(final):
            return 0.0
        if not hasattr(final, "score"):
            raise AttributeError(
                f"The final estimator '{type(final).__name__}' does not "
                "implement 'score'."
            )
        score_params = _get_method_params(final, "score")
        if sample_weight is not None and "sample_weight" in score_params:
            return final.score(Xt, y, sample_weight=sample_weight)
        return final.score(Xt, y)

    # ------------------------------------------------------------------
    # get_params / set_params  (sklearn API compatibility)
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "steps": self.steps,
            "transform_input": self.transform_input,
            "memory": self.memory,
            "verbose": self.verbose,
            "device": self.device,
            "dtype": self.dtype,
        }
        if deep:
            for name, estimator in self.steps:
                if self._is_passthrough(estimator):
                    continue
                if hasattr(estimator, "get_params"):
                    for k, v in estimator.get_params(deep=deep).items():
                        out[f"{name}__{k}"] = v
                else:
                    for k, v in vars(estimator).items():
                        if not k.startswith("_"):
                            out[f"{name}__{k}"] = v
        return out

    def set_params(self, **params) -> "Pipeline":
        step_names = {name for name, _ in self.steps}

        for key, val in params.items():
            if "__" in key:
                # Delegate to sub-estimator
                step_name, param_name = key.split("__", 1)
                for i, (name, estimator) in enumerate(self.steps):
                    if name == step_name:
                        if not self._is_passthrough(estimator):
                            if hasattr(estimator, "set_params"):
                                estimator.set_params(**{param_name: val})
                            else:
                                setattr(estimator, param_name, val)
                        break
            elif key == "steps":
                self.steps = self._validate_steps(val)
                self._update_named_steps()
            elif key in step_names:
                # Replace entire step by name
                for i, (name, _) in enumerate(self.steps):
                    if name == key:
                        self.steps[i] = (name, val)
                        break
                self._update_named_steps()
            else:
                setattr(self, key, val)

        return self

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, ind):
        """Index into the pipeline.

        * **int** — return the estimator at that position.
        * **str** — return the estimator with that name (``named_steps`` lookup).
        * **slice** — return a new :class:`Pipeline` with the sliced steps.
        """
        if isinstance(ind, slice):
            return Pipeline(
                steps=self.steps[ind],
                transform_input=self.transform_input,
                memory=self.memory,
                verbose=self.verbose,
                device=self.device,
                dtype=self.dtype,
            )
        if isinstance(ind, str):
            try:
                return self.named_steps[ind]
            except KeyError:
                raise KeyError(
                    f"Pipeline has no step named '{ind}'. "
                    f"Valid names: {list(self.named_steps)}"
                )
        if isinstance(ind, int):
            if ind < -len(self.steps) or ind >= len(self.steps):
                raise IndexError(f"Pipeline index {ind} out of range.")
            return self.steps[ind][1]
        raise KeyError(f"Invalid pipeline index type: {type(ind).__name__}")

    def __repr__(self) -> str:
        parts = []
        for name, est in self.steps:
            if self._is_passthrough(est):
                parts.append(f"('{name}', passthrough)")
            else:
                parts.append(f"('{name}', {type(est).__name__}())")
        return f"Pipeline(steps=[{', '.join(parts)}])"

    # ------------------------------------------------------------------
    # nn.Module forward  (required by MLModule / DLModule)
    # ------------------------------------------------------------------

    def forward(self, X) -> Any:
        final = self._final_estimator
        if final is not None and not self._is_passthrough(final):
            if hasattr(final, "predict") and not hasattr(final, "transform"):
                return self.predict(X)
        return self.transform(X)
