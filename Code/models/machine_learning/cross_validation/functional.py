import torch
from typing import Optional, Any, Union, Dict, Tuple, List, Callable, Iterable
from ....models.utils import MLModule
import copy
from ....models.machine_learning.cross_validation.splitters import KFoldCV, BaseSplitterCV


__all__ = [
    "cross_validate",
    "cross_val_score",
    "cross_val_predict",
    "permutation_test_score",
    "learning_curve",
    "validation_curve"
]


def _check_cv(cv: Union[int, BaseSplitterCV, Iterable, None] = 5, y: Optional[torch.Tensor] = None,
              classifier: bool = False) -> BaseSplitterCV:
    """
    Input checker utility for building a cross-validator.
    """
    if cv is None:
        cv = 5

    if isinstance(cv, int):
        return KFoldCV(
            n_splits=cv)  # Default to KFold for now. Logic for StratifiedKFold could be added if classifier=True

    if isinstance(cv, BaseSplitterCV):
        return cv

    return cv  # Assuming it's an iterable if not int or BaseSplitterCV


def _score(estimator: Any, X: torch.Tensor, y: torch.Tensor, scorer: Optional[Callable] = None) -> float:
    """
    Compute the score of an estimator on a given data.
    """
    if scorer:
        return scorer(estimator, X, y)

    if hasattr(estimator, 'score'):
        return estimator.score(X, y)

    # Fallback or raise error? For now, assume score exists or scorer provided.
    raise ValueError("Estimator does not have a 'score' method and no 'scoring' callable was provided.")


def cross_validate(estimator: Any, X: torch.Tensor, y: Optional[torch.Tensor] = None,
                   groups: Optional[torch.Tensor] = None, scoring: Optional[Union[str, Callable]] = None,
                   cv: Optional[Union[int, BaseSplitterCV, Iterable]] = None, n_jobs: Optional[int] = None,
                   verbose: int = 0, params: Optional[Dict] = None, pre_dispatch: Union[str, int] = '2*n_jobs',
                   return_train_score: bool = False, return_estimator: bool = False,
                   return_indices: bool = False, error_score: Union[str, float] = float('nan')) -> Dict[
    str, Union[List[float], List[Any]]]:
    """
    Evaluate metric(s) by cross-validation and also record fit/score times.

    Args:
        estimator: The object to use to fit the data.
        X: The data to fit.
        y: The target variable.
        groups: Group labels for the samples.
        scoring: Strategy to evaluate the performance of the estimator.
        cv: Determines the cross-validation splitting strategy.
        n_jobs: Number of jobs to run in parallel.
        verbose: The verbosity level.
        params: Parameters to pass to the underlying estimator's fit.
        pre_dispatch: Controls the number of jobs that get dispatched.
        return_train_score: Whether to include train scores.
        return_estimator: Whether to return the estimators fitted on each split.
        return_indices: Whether to return indices.
        error_score: Value to assign to the score if an error occurs in estimator fitting.

    Returns:
        Dict containing arrays of scores and times.
    """

    cv_splitter = _check_cv(cv, y)

    # If cv is an iterable (not a splitter object), we iterate it directly
    if isinstance(cv_splitter, BaseSplitterCV):
        splitter_iter = cv_splitter.split(X, y, groups)
    else:
        splitter_iter = cv_splitter

    results = {
        'test_score': [],
        'fit_time': [],  # TODO: Measure time
        'score_time': []  # TODO: Measure time
    }
    if return_train_score:
        results['train_score'] = []
    if return_estimator:
        results['estimator'] = []
    if return_indices:
        results['indices'] = []

    # TODO: Parallel execution using n_jobs. For now, sequential.

    for train_idx, test_idx in splitter_iter:

        # Clone estimator
        if isinstance(estimator, MLModule):
            # Deepcopy might be heavy for PyTorch models, but standard for sklearn
            est = copy.deepcopy(estimator)
        else:
            est = copy.deepcopy(estimator)

        X_train, y_train = X[train_idx], y[train_idx] if y is not None else None
        X_test, y_test = X[test_idx], y[test_idx] if y is not None else None

        try:
            # Fit
            # import time
            # start_fit = time.time()
            if params:
                est.fit(X_train, y_train, **params)
            else:
                est.fit(X_train, y_train)
            # fit_time = time.time() - start_fit
            # results['fit_time'].append(fit_time)
            results['fit_time'].append(0.0)  # Placeholder

            # Score Test
            # start_score = time.time()
            test_score = _score(est, X_test, y_test, scoring)
            # score_time = time.time() - start_score
            results['test_score'].append(test_score)
            # results['score_time'].append(score_time)
            results['score_time'].append(0.0)  # Placeholder

            if return_train_score:
                train_score = _score(est, X_train, y_train, scoring)
                results['train_score'].append(train_score)

            if return_estimator:
                results['estimator'].append(est)

            if return_indices:
                results['indices'].append((train_idx, test_idx))

        except Exception as e:
            if error_score == 'raise':
                raise e
            else:
                results['test_score'].append(error_score)
                results['fit_time'].append(0.0)
                results['score_time'].append(0.0)
                if return_train_score:
                    results['train_score'].append(error_score)
                if return_estimator:
                    results['estimator'].append(None)  # Or the failed estimator?
                if return_indices:
                    results['indices'].append((train_idx, test_idx))

    return results


def cross_val_score(estimator: Any, X: torch.Tensor, y: Optional[torch.Tensor] = None,
                    groups: Optional[torch.Tensor] = None, scoring: Optional[Union[str, Callable]] = None,
                    cv: Optional[Union[int, BaseSplitterCV, Iterable]] = None, n_jobs: Optional[int] = None,
                    verbose: int = 0, params: Optional[Dict] = None, pre_dispatch: Union[str, int] = '2*n_jobs',
                    error_score: Union[str, float] = float('nan')) -> List[float]:
    """
    Evaluate a score by cross-validation.
    """
    cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups, scoring=scoring, cv=cv,
                                n_jobs=n_jobs, verbose=verbose, params=params, pre_dispatch=pre_dispatch,
                                error_score=error_score)
    return cv_results['test_score']


def cross_val_predict(estimator: Any, X: torch.Tensor, y: Optional[torch.Tensor] = None,
                      groups: Optional[torch.Tensor] = None, cv: Optional[Union[int, BaseSplitterCV, Iterable]] = None,
                      n_jobs: Optional[int] = None, verbose: int = 0, params: Optional[Dict] = None,
                      pre_dispatch: Union[str, int] = '2*n_jobs', method: str = 'predict') -> torch.Tensor:
    """
    Generate cross-validated estimates for each input data point.
    """
    cv_splitter = _check_cv(cv, y)

    if isinstance(cv_splitter, BaseSplitterCV):
        splitter_iter = cv_splitter.split(X, y, groups)
    else:
        splitter_iter = cv_splitter

    predictions = [None] * X.shape[0]  # Placeholder list

    for split in splitter_iter:
        train_idx, test_idx = split[0], split[1]  # Handle splitters that yield 3+ values
        if isinstance(estimator, MLModule):
            est = copy.deepcopy(estimator)
        else:
            est = copy.deepcopy(estimator)

        X_train, y_train = X[train_idx], y[train_idx] if y is not None else None
        X_test = X[test_idx]

        if params:
            est.fit(X_train, y_train, **params)
        else:
            est.fit(X_train, y_train)

        func = getattr(est, method)
        preds = func(X_test)

        # Assign predictions
        for i, idx in enumerate(test_idx):
            predictions[idx] = preds[i]

    # Convert back to tensor if possible
    try:
        if isinstance(predictions[0], torch.Tensor):
            if predictions[0].ndim == 0:
                return torch.tensor(predictions)
            return torch.stack(predictions)
        else:
            return torch.tensor(predictions)
    except:
        return predictions  # Return list if stacking fails


def permutation_test_score(estimator: Any, X: torch.Tensor, y: torch.Tensor, groups: Optional[torch.Tensor] = None,
                           cv: Optional[Union[int, BaseSplitterCV, Iterable]] = None, n_permutations: int = 100,
                           n_jobs: Optional[int] = None, random_state: int = 0, verbose: int = 0,
                           scoring: Optional[Union[str, Callable]] = None) -> Tuple[float, torch.Tensor, float]:
    """
    Evaluate the significance of a cross-validated score with permutations.
    """
    # 1. Compute original score
    original_scores = cross_val_score(estimator, X, y, groups=groups, cv=cv, n_jobs=n_jobs, verbose=verbose,
                                      scoring=scoring)
    score = torch.tensor(original_scores).mean().item()

    # 2. Permutations
    permutation_scores = []
    for i in range(n_permutations):
        # Permute y
        if random_state is not None:
            g = torch.Generator()
            g.manual_seed(random_state + i)
            perm_idx = torch.randperm(y.shape[0], generator=g)
        else:
            perm_idx = torch.randperm(y.shape[0])

        y_perm = y[perm_idx]

        # Compute score with permuted y
        perm_scores_list = cross_val_score(estimator, X, y_perm, groups=groups, cv=cv, n_jobs=n_jobs, verbose=0,
                                           scoring=scoring)
        perm_score = torch.tensor(perm_scores_list).mean().item()
        permutation_scores.append(perm_score)

    permutation_scores_tensor = torch.tensor(permutation_scores)

    # 3. p-value
    # (C + 1) / (n_permutations + 1)
    pvalue = (torch.sum(permutation_scores_tensor >= score).item() + 1.0) / (n_permutations + 1.0)

    return score, permutation_scores_tensor, pvalue


def learning_curve(estimator: Any, X: torch.Tensor, y: torch.Tensor, groups: Optional[torch.Tensor] = None,
                   train_sizes: Union[torch.Tensor, Iterable] = torch.linspace(0.1, 1.0, 5),
                   cv: Optional[Union[int, BaseSplitterCV, Iterable]] = None,
                   scoring: Optional[Union[str, Callable]] = None,
                   exploit_incremental_learning: bool = False, n_jobs: Optional[int] = None,
                   pre_dispatch: Union[str, int] = 'all', verbose: int = 0, shuffle: bool = False,
                   random_state: Optional[int] = None, error_score: Union[str, float] = float('nan'),
                   return_times: bool = False) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Learning curve.
    """
    import time

    cv_splitter = _check_cv(cv, y)

    if isinstance(cv_splitter, BaseSplitterCV):
        splits = list(cv_splitter.split(X, y, groups))
    else:
        splits = list(cv_splitter)  # iterable

    # Get max training size from the first split (approximation for all folds)
    train_idx_first, _ = splits[0]
    n_max_train = len(train_idx_first)

    # Calculate absolute train sizes
    abs_train_sizes = []

    # Convert train_sizes to list if tensor
    if isinstance(train_sizes, torch.Tensor):
        train_sizes = train_sizes.tolist()

    for ts in train_sizes:
        if isinstance(ts, float):
            size = int(ts * n_max_train)
            if size == 0: size = 1
            abs_train_sizes.append(size)
        else:
            abs_train_sizes.append(ts)

    # Structure: [n_ticks, n_cv_folds]
    n_ticks = len(abs_train_sizes)
    n_folds = len(splits)

    train_scores = torch.zeros((n_ticks, n_folds))
    test_scores = torch.zeros((n_ticks, n_folds))
    fit_times = torch.zeros((n_ticks, n_folds))
    score_times = torch.zeros((n_ticks, n_folds))

    for i, n_train in enumerate(abs_train_sizes):
        for j, (train_idx, test_idx) in enumerate(splits):
            curr_train_idx = train_idx

            # Shuffle if requested
            if shuffle:
                seed = random_state + j if random_state is not None else None
                if seed is not None:
                    g = torch.Generator(device=X.device)
                    g.manual_seed(seed)
                    perm = torch.randperm(len(curr_train_idx), generator=g, device=X.device)
                else:
                    perm = torch.randperm(len(curr_train_idx), device=X.device)
                curr_train_idx = curr_train_idx[perm]

            # Slice to n_train
            current_fold_size = len(curr_train_idx)
            actual_n_train = min(n_train, current_fold_size)
            subset_train_idx = curr_train_idx[:actual_n_train]

            X_train_subset = X[subset_train_idx]
            y_train_subset = y[subset_train_idx] if y is not None else None
            X_test = X[test_idx]
            y_test = y[test_idx] if y is not None else None

            if isinstance(estimator, MLModule):
                est = copy.deepcopy(estimator)
            else:
                est = copy.deepcopy(estimator)

            start_fit = time.time()
            try:
                est.fit(X_train_subset, y_train_subset)
                fit_time = time.time() - start_fit

                start_score = time.time()
                test_score = _score(est, X_test, y_test, scoring)
                score_time = time.time() - start_score

                train_score = _score(est, X_train_subset, y_train_subset, scoring)

                train_scores[i, j] = train_score
                test_scores[i, j] = test_score
                fit_times[i, j] = fit_time
                score_times[i, j] = score_time

            except Exception as e:
                if error_score == 'raise':
                    raise e
                train_scores[i, j] = error_score
                test_scores[i, j] = error_score
                fit_times[i, j] = 0.0
                score_times[i, j] = 0.0

    abs_train_sizes_tensor = torch.tensor(abs_train_sizes)

    if return_times:
        return abs_train_sizes_tensor, train_scores, test_scores, fit_times, score_times

    return abs_train_sizes_tensor, train_scores, test_scores, None, None


def validation_curve(estimator: Any, X: torch.Tensor, y: torch.Tensor, param_name: str,
                     param_range: Union[torch.Tensor, List, Iterable], groups: Optional[torch.Tensor] = None,
                     cv: Optional[Union[int, BaseSplitterCV, Iterable]] = None,
                     scoring: Optional[Union[str, Callable]] = None,
                     n_jobs: Optional[int] = None, pre_dispatch: Union[str, int] = 'all', verbose: int = 0,
                     error_score: Union[str, float] = float('nan')) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validation curve.
    """
    train_scores_list = []
    test_scores_list = []

    # Check if estimator has set_params
    if not hasattr(estimator, 'set_params'):
        raise ValueError("Estimator must implement 'set_params' method to be used with validation_curve.")

    for v in param_range:
        # Clone and set param
        if isinstance(estimator, MLModule):
            est = copy.deepcopy(estimator)
        else:
            est = copy.deepcopy(estimator)

        # Prepare param value (unwrap tensor if needed)
        val = v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v

        est.set_params(**{param_name: val})

        # Use cross_validate
        cv_res = cross_validate(est, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=n_jobs,
                                verbose=verbose, pre_dispatch=pre_dispatch,
                                return_train_score=True, error_score=error_score)

        train_scores_list.append(cv_res['train_score'])
        test_scores_list.append(cv_res['test_score'])

    return torch.tensor(train_scores_list), torch.tensor(test_scores_list)
