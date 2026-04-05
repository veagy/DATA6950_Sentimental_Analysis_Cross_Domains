import torch
import torch.nn.functional as F
from typing import Any, Union, Dict, List, Callable, Iterable
from ....models.utils import MLModule
from abc import abstractmethod
import itertools
import copy
from ....models.machine_learning.cross_validation.splitters import *

__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV"
]


class ParameterGrid(MLModule):
    """
    Grid of parameters with a discrete number of values for each.
    """

    def __init__(self, param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]]):
        super().__init__()
        if isinstance(param_grid, dict):
            self.param_grid = [param_grid]
        else:
            self.param_grid = param_grid

    def __iter__(self):
        """
        Iterate over the points in the grid.
        """
        for p in self.param_grid:
            keys = sorted(p.keys())
            if not keys:
                if p == {}:
                    yield {}
                continue

            values = [p[k] for k in keys]
            for v in itertools.product(*values):
                yield dict(zip(keys, v))

    def __len__(self):
        """
        Number of points on the grid.
        """
        total = 0
        for p in self.param_grid:
            product = 1
            for v in p.values():
                product *= len(v)
            total += product
        return total

    def __getitem__(self, ind):
        """
        Get the parameters that would be ind-th in iteration.
        """
        for i, p in enumerate(self):
            if i == ind:
                return p
        raise IndexError("ParameterGrid index out of range")


class ParameterSampler(MLModule):
    """
    Generator on parameters sampled from given distributions.
    """

    def __init__(self, param_distributions: Dict[str, Union[List[Any], Any]], n_iter: int, random_state: int = None):
        super().__init__()
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def __iter__(self):
        """
        Iterate over the points in the grid.
        """
        if self.random_state is not None:
            rng = torch.Generator()
            rng.manual_seed(self.random_state)
        else:
            rng = None

        sorted_keys = sorted(self.param_distributions.keys())

        for i in range(self.n_iter):
            params = {}
            for k in sorted_keys:
                v = self.param_distributions[k]

                if isinstance(v, list):
                    if rng is not None:
                        idx = torch.randint(0, len(v), (1,), generator=rng).item()
                    else:
                        idx = torch.randint(0, len(v), (1,)).item()
                    params[k] = v[idx]

                elif hasattr(v, 'rvs'):
                    params[k] = v.rvs()

                elif hasattr(v, 'sample'):
                    sample = v.sample()
                    if isinstance(sample, torch.Tensor):
                        params[k] = sample.item()
                    else:
                        params[k] = sample
                else:
                    if callable(v):
                        params[k] = v()
                    else:
                        params[k] = v
            yield params

    def __len__(self):
        """
        Number of points that will be sampled.
        """
        return self.n_iter


class BaseSearchCV(MLModule):
    """
    Base class for hyperparameter search with cross-validation.
    """

    def __init__(self,
                 estimator: MLModule,
                 scoring: Union[str, Callable] = None,
                 n_jobs: int = None,
                 refit: bool = True,
                 cv: Union[str, int, Callable, Iterable, MLModule] = None,
                 cv_config: dict = None,
                 verbose: Union[int, bool] = 0,
                 pre_dispatch: str = '2*n_jobs',
                 error_score: Union[int, float] = float('nan'),
                 return_train_score: bool = False,
                 store_cv_values: bool = False,
                 return_estimators: bool = False,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.store_cv_values = store_cv_values
        self.return_estimators = return_estimators
        self.device = device
        self.dtype = dtype
        self.cv_config = cv_config
        self.cv_results_ = None
        self.cv_values_ = None  # New attribute for per-sample values
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_index_ = None
        self.scorer_ = None
        self.n_splits_ = None
        self.refit_time_ = None
        self.multimetric_ = False

    def _scoring_function(self, scoring: Union[str, Callable]):
        if isinstance(scoring, str):
            match scoring.lower():
                case "mae" | "mean_absolute_error" | "neg_mean_absolute_error":
                    return lambda estimator, X, y: -F.l1_loss(estimator.predict(X), y)
                case "mse" | "mean_squared_error" | "neg_mean_squared_error":
                    return lambda estimator, X, y: -F.mse_loss(estimator.predict(X), y)
                case "rmse" | "root_mean_squared_error" | "neg_root_mean_squared_error":
                    return lambda estimator, X, y: -torch.sqrt(F.mse_loss(estimator.predict(X), y))
                case "huber" | "huber_loss":
                    return lambda estimator, X, y: -F.huber_loss(estimator.predict(X), y)
                case "msle" | "mean_squared_logarithmic_error":
                    return lambda estimator, X, y: -F.mse_loss(torch.log1p(estimator.predict(X).abs()),
                                                               torch.log1p(y.abs()))
                case "mape" | "mean_absolute_percentage_error":
                    return lambda estimator, X, y: -torch.mean(torch.abs((y - estimator.predict(X)) / (y + 1e-8)))
                case "r2" | "r_squared":
                    def r2(estimator, X, y):
                        y_pred = estimator.predict(X)
                        ss_res = torch.sum((y - y_pred) ** 2)
                        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
                        return 1 - ss_res / ss_tot

                    return r2
                case "quantile_loss" | "pinball_loss":
                    def pinball_loss(estimator, X, y, quantile=0.5):
                        y_pred = estimator.predict(X)
                        errors = y - y_pred
                        return -torch.max((quantile - 1) * errors, quantile * errors).mean()

                    return pinball_loss
                case _:
                    return self._scoring_function(scoring="mse")

    @abstractmethod
    def _run_search(self, evaluate_candidates):
        pass

    def fit(self, X: torch.Tensor, y: torch.Tensor = None, groups: torch.Tensor = None, **fit_params):
        """
        Run fit with all sets of parameters.
        """
        cv = self._check_cv(self.cv, y)
        self.n_splits_ = cv.get_n_splits(X, y, groups)

        splits = list(cv.split(X, y, groups))
        sample_weight = fit_params.get("sample_weight", None)

        def evaluate_candidates(candidate_params):
            """
            Evaluate a list of candidate parameters.
            """

            cand_results = {
                "params": [],
                "mean_test_score": [],
                "std_test_score": [],
            }
            if self.return_train_score:
                cand_results["mean_train_score"] = []
                cand_results["std_train_score"] = []

            for i in range(self.n_splits_):
                cand_results[f"split{i}_test_score"] = []
                if self.return_train_score:
                    cand_results[f"split{i}_train_score"] = []
                if self.return_estimators:
                    cand_results[f"split{i}_estimator"] = []

            for params in candidate_params:
                cand_results["params"].append(params)

                fold_test_scores = []
                fold_train_scores = []

                # For store_cv_values
                if self.store_cv_values:
                    # Initialize placeholder for this candidate: (n_samples, n_targets) or (n_samples,)
                    # We don't know the output shape yet, so we'll collect list of (indices, scores) and reconstruct
                    candidate_cv_values = [None] * X.shape[0]

                for i, (train_idx, test_idx) in enumerate(splits):
                    estimator = copy.deepcopy(self.estimator)
                    self._set_params(estimator, params)

                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = (y[train_idx], y[test_idx]) if y is not None else (None, None)
                    sw_train = sample_weight[train_idx] if sample_weight is not None else None

                    try:
                        fit_kwargs = fit_params.copy()
                        if sw_train is not None:
                            fit_kwargs["sample_weight"] = sw_train
                        
                        if y_train is not None:
                            estimator.fit(X_train, y_train, **fit_kwargs)
                        else:
                            estimator.fit(X_train, **fit_kwargs)

                        score_func = self._get_score_func()

                        test_score = score_func(estimator, X_test, y_test)
                        fold_test_scores.append(test_score)
                        cand_results[f"split{i}_test_score"].append(test_score)

                        if self.store_cv_values:
                            with torch.no_grad():
                                pred = estimator.predict(X_test)
                                if y_test is not None and self.scoring is None:  # Default scoring
                                    # RidgeCV default: MSE
                                    # Calculate squared error per sample
                                    if pred.ndim == 1: pred = pred.unsqueeze(1)
                                    if y_test.ndim == 1: y_test = y_test.unsqueeze(1)
                                    val = torch.mean((pred - y_test) ** 2, dim=1)  # Mean over targets if multi-output
                                elif y_test is not None and isinstance(self.scoring, str) and 'mse' in self.scoring:
                                    if pred.ndim == 1: pred = pred.unsqueeze(1)
                                    if y_test.ndim == 1: y_test = y_test.unsqueeze(1)
                                    val = torch.mean((pred - y_test) ** 2, dim=1)
                                else:
                                    # Fallback or other_decomposition metrics: just store prediction
                                    val = pred

                                # Convert to list/tensor and store in candidate_cv_values at test_idx
                                if isinstance(val, torch.Tensor):
                                    val = val.cpu()

                                for idx_in_fold, global_idx in enumerate(test_idx):
                                    candidate_cv_values[global_idx] = val[idx_in_fold]

                        if self.return_train_score:
                            train_score = score_func(estimator, X_train, y_train)
                            fold_train_scores.append(train_score)
                            cand_results[f"split{i}_train_score"].append(train_score)

                        if self.return_estimators:
                            cand_results[f"split{i}_estimator"].append(estimator)

                    except Exception as e:
                        if self.error_score == 'raise':
                            raise e
                        else:
                            fold_test_scores.append(self.error_score)
                            cand_results[f"split{i}_test_score"].append(self.error_score)
                            if self.return_train_score:
                                fold_train_scores.append(self.error_score)
                                cand_results[f"split{i}_train_score"].append(self.error_score)

                test_tensor = torch.tensor(fold_test_scores, dtype=self.dtype, device=self.device)
                mean_test = torch.mean(test_tensor).item()
                std_test = torch.std(test_tensor).item()

                cand_results["mean_test_score"].append(mean_test)
                cand_results["std_test_score"].append(std_test)

                if self.return_train_score:
                    train_tensor = torch.tensor(fold_train_scores, dtype=self.dtype, device=self.device)
                    mean_train = torch.mean(train_tensor).item()
                    std_train = torch.std(train_tensor).item()
                    cand_results["mean_train_score"].append(mean_train)
                    cand_results["std_train_score"].append(std_train)

                if self.store_cv_values:
                    # candidate_cv_values is list of values. Stack them.
                    # Check if we have values for all samples
                    if any(v is None for v in candidate_cv_values):
                        # Partial CV values (e.g. ShuffleSplit)
                        pass
                    else:
                        try:
                            # Try to stack into tensor
                            if isinstance(candidate_cv_values[0], torch.Tensor):
                                stacked = torch.stack(candidate_cv_values)
                            else:
                                stacked = torch.tensor(candidate_cv_values, device=self.device, dtype=self.dtype)

                            if "cv_values" not in cand_results:
                                cand_results["cv_values"] = []
                            cand_results["cv_values"].append(stacked)
                        except:
                            pass

            return cand_results

        self.cv_results_ = self._run_search(evaluate_candidates)

        # Reshape cv_values_ if present to (n_samples, n_candidates)
        if "cv_values" in self.cv_results_:
            try:
                # Stack candidates along dim 1
                self.cv_values_ = torch.stack(self.cv_results_["cv_values"]).T
            except:
                self.cv_values_ = self.cv_results_["cv_values"]  # Fallback to list

        best_idx = int(torch.argmax(torch.tensor(self.cv_results_["mean_test_score"])).item())

        self.best_index_ = best_idx
        self.best_params_ = self.cv_results_["params"][best_idx]
        self.best_score_ = self.cv_results_["mean_test_score"][best_idx]

        if self.refit:
            import time
            refit_start_time = time.time()
            self.best_estimator_ = copy.deepcopy(self.estimator)
            self._set_params(self.best_estimator_, self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            self.refit_time_ = time.time() - refit_start_time

        return self

    def _check_cv(self, cv, y=None):
        cv_config = self.cv_config if self.cv_config is not None else {}
        if cv is None:
            return KFoldCV(n_splits=5, **cv_config)
        if isinstance(cv, str):
            mapping_dict = {
                "k_fold": KFoldCV,
                "group_k_fold": GroupKFold,
                "stratified_k_fold": StratifiedKFold,
                "time_series_split": TimeSeriesSplit,
                "leave_one_out": LeaveOneOut,
                "leave_p_out": LeavePOut,
                "leave_p_groups_out": LeavePGroupsOut,
                "leave_one_group_out": LeaveOneGroupOut,
                "predefined_split": PredefinedSplit,
                "shuffle_split": ShuffleSplit,
                "group_shuffle_split": GroupShuffleSplit,
                "stratified_shuffle_split": StratifiedShuffleSplit,
                "repeated_k_fold": RepeatedKFold,
                "repeated_stratified_k_fold": RepeatedStratifiedKFold
            }
            return mapping_dict[cv.lower()](**cv_config) if cv.lower() in mapping_dict.keys() else KFoldCV(**cv_config)
        elif isinstance(cv, int):
            return KFoldCV(n_splits=cv, **cv_config)
        elif isinstance(cv, type) and issubclass(cv, MLModule):
            return cv(**cv_config)
        elif isinstance(cv, MLModule):
            return cv
        else:
            return lambda X_, y_: cv(X_, y_, **cv_config)

    def _set_params(self, estimator, params):
        for k, v in params.items():
            if hasattr(estimator, k):
                setattr(estimator, k, v)
            elif hasattr(estimator, 'set_params'):
                estimator.set_params(**{k: v})
            else:
                pass

    def _get_score_func(self):
        if self.scoring is None:
            return lambda estimator, X, y: estimator.score(X, y) if y is not None else estimator.score(X)
        elif callable(self.scoring):
            return self.scoring
        elif isinstance(self.scoring, str):
            return self._scoring_function(self.scoring)
        else:
            raise NotImplementedError("Scoring must be a string or callable.")

    def predict(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        self._check_is_fitted()
        return self.best_estimator_.predict_log_proba(X)

    def decision_function(self, X):
        self._check_is_fitted()
        return self.best_estimator_.decision_function(X)

    def transform(self, X):
        self._check_is_fitted()
        return self.best_estimator_.transform(X)

    def inverse_transform(self, X):
        self._check_is_fitted()
        return self.best_estimator_.inverse_transform(X)

    def score(self, X, y, sample_weight=None):
        self._check_is_fitted()
        if y is None:
            return self.best_estimator_.score(X)
        return self.best_estimator_.score(X, y)

    def _check_is_fitted(self):
        if not self.refit:
            raise RuntimeError("This GridSearchCV instance was initialized with refit=False. "
                               "predict is available only after fitting on the best parameters.")
        if self.best_estimator_ is None:
            raise RuntimeError("This GridSearchCV instance is not fitted yet.")


class GridSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, n_jobs=None, refit=True, cv=None,
                 cv_config=None, verbose=0, pre_dispatch='2*n_jobs', error_score=float('nan'),
                 return_train_score=False, store_cv_values=False, return_estimators=False, 
                 device="cpu", dtype=torch.float):
        super().__init__(estimator, scoring, n_jobs, refit, cv, cv_config, verbose, pre_dispatch, error_score,
                         return_train_score, store_cv_values, return_estimators, device, dtype)
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """
        Run all candidates parallelized
        """
        param_grid_obj = ParameterGrid(self.param_grid)
        candidates = list(param_grid_obj)
        
        try:
            import joblib # pyright: ignore[reportMissingImports]
            n_jobs = self.n_jobs if self.n_jobs is not None else 1
            
            # evaluate_candidates expects a list and returns a dict of results for all candidates.
            # To parallelize, we can chunk the candidates and merge, or just pass the whole thing
            # if the inner evaluate_candidates loops. However, the current evaluate_candidates handles
            # all candidates in a loop.
            
            # Since evaluate_candidates processes the list itself sequentially, we'll split the list
            # and run evaluate_candidates on chunks if n_jobs > 1 or n_jobs == -1.
            if n_jobs == 1:
                return evaluate_candidates(candidates)
                
            n_chunks = n_jobs if n_jobs > 0 else len(candidates)
            chunk_size = max(1, len(candidates) // n_chunks)
            chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
            
            chunk_results = joblib.Parallel(n_jobs=n_jobs, pre_dispatch=self.pre_dispatch)(
                joblib.delayed(evaluate_candidates)(chunk) for chunk in chunks
            )
            
            # Merge results
            merged_results = {}
            for res in chunk_results:
                for k, v in res.items():
                    if k not in merged_results:
                        merged_results[k] = []
                    merged_results[k].extend(v)
            return merged_results
            
        except ImportError:
            return evaluate_candidates(candidates)


class RandomizedSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None, n_jobs=None, refit=True, cv=None,
                 cv_config=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=float('nan'),
                 return_train_score=False, store_cv_values=False, return_estimators=False, device="cpu", dtype=torch.float):
        super().__init__(estimator, scoring, n_jobs, refit, cv, cv_config, verbose, pre_dispatch, error_score,
                         return_train_score, store_cv_values, return_estimators, device, dtype)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """
        Run sampled candidates parallelized
        """
        param_sampler_obj = ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)
        candidates = list(param_sampler_obj)
        
        try:
            import joblib # pyright: ignore[reportMissingImports]
            n_jobs = self.n_jobs if self.n_jobs is not None else 1
            
            if n_jobs == 1:
                return evaluate_candidates(candidates)
                
            n_chunks = n_jobs if n_jobs > 0 else len(candidates)
            chunk_size = max(1, len(candidates) // n_chunks)
            chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
            
            chunk_results = joblib.Parallel(n_jobs=n_jobs, pre_dispatch=self.pre_dispatch)(
                joblib.delayed(evaluate_candidates)(chunk) for chunk in chunks
            )
            
            # Merge results
            merged_results = {}
            for res in chunk_results:
                for k, v in res.items():
                    if k not in merged_results:
                        merged_results[k] = []
                    merged_results[k].extend(v)
            return merged_results
            
        except ImportError:
            return evaluate_candidates(candidates)
