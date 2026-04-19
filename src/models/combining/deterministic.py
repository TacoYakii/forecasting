"""Deterministic combining: MAE/MSE-optimal point-forecast ensemble.

Unlike probabilistic combiners (Horizontal, Vertical) that work on
quantile distributions and optimize CRPS, DeterministicCombiner
operates on point summaries (mean or median) of each model's
distribution and learns weights that minimize MAE or MSE against
observed values.

Use this combiner when the final evaluation metric is a point-forecast
metric such as nMAPE (capacity-normalized MAE): directly optimizing
MAE on training data avoids the proxy step of CRPS minimization.

Output is a DeterministicForecastResult (point forecasts only); the
combiner bypasses the quantile/distribution layer entirely.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.optimize import LinearConstraint, linprog, minimize

from src.core.forecast_results import (
    BaseForecastResult,
    DeterministicForecastResult,
)


# ---------------------------------------------------------------------------
# Point-forecast extraction
# ---------------------------------------------------------------------------


def _extract_point_array(
    result: BaseForecastResult, point: str
) -> np.ndarray:
    """Extract per-horizon point forecasts from a ForecastResult.

    Args:
        result: Any ForecastResult subclass.
        point: ``"mean"`` uses ``distribution.mean()``;
            ``"median"`` uses ``distribution.ppf(0.5)``.

    Returns:
        np.ndarray of shape (N, H) with point forecasts.

    Example:
        >>> mu = _extract_point_array(result, point="mean")
        >>> mu.shape    # (N, H)
    """
    H = result.horizon
    N = len(result)
    mu = np.empty((N, H), dtype=float)
    for h in range(1, H + 1):
        dist = result.to_distribution(h)
        mu[:, h - 1] = dist.mean() if point == "mean" else dist.ppf(0.5)
    return mu


# ---------------------------------------------------------------------------
# Optimization solvers
# ---------------------------------------------------------------------------


def _mae_weights_lp(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve LP for MAE-optimal combining weights via HiGHS.

    Formulates MAE minimization as a linear program by introducing
    slack variables ``e_t >= |X @ w - y|_t``:

    .. math::
        \\min_{w, e} \\frac{1}{N} \\mathbf{1}^T e
        \\quad \\text{s.t.} \\quad
        -e \\leq Xw - y \\leq e, \\quad
        \\mathbf{1}^T w = 1, \\quad
        w \\geq 0, \\quad e \\geq 0.

    HiGHS is invoked via scipy.optimize.linprog.

    Args:
        X: Stacked point forecasts, shape (N, M).
        y: Observed values, shape (N,).

    Returns:
        np.ndarray: Normalized weights, shape (M,), summing to 1.

    Raises:
        RuntimeError: If the LP solver fails to converge.

    Example:
        >>> w = _mae_weights_lp(X, y)
        >>> w.sum()     # 1.0
    """
    N, M = X.shape
    c = np.concatenate([np.zeros(M), np.ones(N) / N])
    A_ub = np.vstack([
        np.hstack([X, -np.eye(N)]),
        np.hstack([-X, -np.eye(N)]),
    ])
    b_ub = np.concatenate([y, -y])
    A_eq = np.hstack([np.ones(M), np.zeros(N)]).reshape(1, -1)
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * M + [(0.0, None)] * N
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP did not converge: {result.message}")
    w = np.maximum(result.x[:M], 0.0)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("LP returned invalid weights (sum <= 0).")
    return w / total


def _mse_weights_qp(
    X: np.ndarray, y: np.ndarray, l2_to_uniform: float = 0.0
) -> np.ndarray:
    """Solve constrained QP for MSE-optimal combining weights.

    Minimizes the objective

    .. math::
        \\frac{1}{N} \\|X w - y\\|_2^2
        + \\lambda \\cdot \\mathbb{E}[y^2] \\cdot \\|w - 1/M\\|_2^2

    subject to ``sum(w) = 1`` and ``0 <= w <= 1``.

    The second term is an optional L2 shrinkage toward uniform weights,
    scaled by ``E[y^2]`` so that ``λ`` is dimensionless. ``λ = 0`` gives
    the pure MSE solution; ``λ → ∞`` pushes toward equal weights.

    Empirically, ``λ ≈ 1.0`` (combined with ``top_k=6``) provides a
    good trade-off between fitting validation data and generalizing to
    test data under distribution shift (see
    ``new_forecast_res/analysis/combiner_experiments.md``).

    Uses ``trust-constr`` with an analytic Hessian. If trust-constr
    fails, falls back to SLSQP with a tiny numerical ridge on the
    Hessian.

    Args:
        X: Stacked point forecasts, shape (N, M).
        y: Observed values, shape (N,).
        l2_to_uniform: L2 shrinkage strength toward uniform weights
            (``1/M``). Must be non-negative. Defaults to 0 (no shrinkage).

    Returns:
        np.ndarray: Normalized weights, shape (M,), summing to 1.

    Raises:
        RuntimeError: If both solvers fail.

    Example:
        >>> w = _mse_weights_qp(X, y, l2_to_uniform=1.0)
        >>> w.sum()     # 1.0
    """
    N, M = X.shape
    # Precompute quadratic form: obj = 0.5 * w @ Q @ w + c @ w (+ const)
    # where Q = (2/N) * X.T @ X and c = -(2/N) * X.T @ y
    Q = (X.T @ X) * (2.0 / N)
    c = -(X.T @ y) * (2.0 / N)

    # L2 shrinkage toward uniform weights 1/M
    if l2_to_uniform > 0:
        baseline = float((y ** 2).mean())
        Q = Q + 2.0 * l2_to_uniform * baseline * np.eye(M)
        c = c - 2.0 * l2_to_uniform * baseline * (1.0 / M) * np.ones(M)

    def obj(w: np.ndarray) -> float:
        return float(0.5 * w @ Q @ w + c @ w)

    def grad(w: np.ndarray) -> np.ndarray:
        return Q @ w + c

    def hess(w: np.ndarray) -> np.ndarray:
        return Q

    x0 = np.ones(M) / M
    eq_constraint = LinearConstraint(np.ones(M), lb=1.0, ub=1.0)
    try:
        result = minimize(
            obj,
            x0,
            jac=grad,
            hess=hess,
            method="trust-constr",
            bounds=[(0.0, 1.0)] * M,
            constraints=[eq_constraint],
            options={"maxiter": 500, "gtol": 1e-8, "verbose": 0},
        )
        if result.success:
            w = np.maximum(result.x, 0.0)
            total = w.sum()
            if total > 0 and np.isfinite(total):
                return w / total
    except Exception:
        pass

    # Fallback: L2-regularized SLSQP
    ridge = 1e-6 * (np.trace(Q) / M)
    Q_reg = Q + ridge * np.eye(M)

    def obj_reg(w: np.ndarray) -> float:
        return float(0.5 * w @ Q_reg @ w + c @ w)

    def grad_reg(w: np.ndarray) -> np.ndarray:
        return Q_reg @ w + c

    result = minimize(
        obj_reg,
        x0,
        jac=grad_reg,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * M,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(
            f"QP did not converge (trust-constr + SLSQP ridge fallback). "
            f"Final message: {result.message}"
        )
    w = np.maximum(result.x, 0.0)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("QP returned invalid weights (sum <= 0).")
    return w / total


# ---------------------------------------------------------------------------
# DeterministicCombiner
# ---------------------------------------------------------------------------


class DeterministicCombiner:
    """MAE/MSE-optimal point-forecast ensemble combiner.

    Extracts point summaries (mean or median) from each model's
    distribution and learns per-horizon weights that minimize the
    chosen point-forecast loss directly:

    - ``loss="mae"``: Linear program via HiGHS. Directly minimizes the
      target metric when the final evaluation is nMAPE (MAE normalized
      by capacity).
    - ``loss="mse"``: Quadratic program via SLSQP. Penalizes large
      residuals more heavily — better for models that occasionally
      produce large outliers.

    Unlike HorizontalCombiner/VerticalCombiner, this combiner does not
    produce a distribution output: it returns DeterministicForecastResult
    (point forecasts only). Use this when CRPS is not the target.

    Args:
        point: Which point summary of each model's distribution to
            use as input. ``"mean"`` (default) uses ``distribution.mean()``;
            ``"median"`` uses ``distribution.ppf(0.5)``.
        loss: Loss function for weight learning.
            ``"mae"`` (default, LP) or ``"mse"`` (QP).
        n_jobs: Parallel workers for the horizon loop (default 12).
            Each horizon's LP/QP is independent.
        weights: User-specified weights of shape (M,). If provided,
            learning is bypassed for every horizon.
        val_ratio: Fraction of data held out as a temporal validation
            set (default 0.0, no split). Last ``val_ratio`` fraction
            is used for validation; weights are learned on the rest.
        top_k: If set, select only the top-``K`` models (by average
            training loss across horizons) before fitting weights.
            Excluded models receive weight 0 in ``weights_``. Defaults
            to ``None`` (use all models). Empirically, ``top_k=6`` with
            ``loss="mse"`` and ``l2_to_uniform≈1.0`` works well for
            wind power forecasting with 15+ heterogeneous base models.
        l2_to_uniform: L2 shrinkage strength toward uniform weights,
            scaled by ``E[y^2]``. Only applies to ``loss="mse"``.
            ``0`` (default) gives pure MSE QP; larger values pull
            weights toward ``1/K``. Useful under distribution shift
            between fit period and deployment period. When
            ``l2_cv_grid`` is provided, this value is overridden by
            the CV-selected λ.
        l2_cv_grid: When provided, run leave-one-month-out CV over
            the fit period to select the best λ from this grid.
            Requires the validation period to span at least 2 distinct
            months. Only applies to ``loss="mse"``. Selected value is
            stored in ``l2_to_uniform_`` after ``fit()``.

    Attributes:
        weights_ (Dict[int, np.ndarray]): Learned per-horizon weights,
            shape (M,) per horizon (zeros for excluded models when
            ``top_k`` is set).
        train_scores_ (Dict[int, float]): Per-horizon training MAE.
        val_scores_ (Dict[int, float]): Per-horizon validation MAE.
            Empty when ``val_ratio=0``.
        model_names_ (List[str]): Names of the input models (full set).
        top_k_indices_ (List[int]): Indices of selected top-K models
            (sorted ascending). Equal to ``list(range(M))`` when
            ``top_k=None``.
        model_scores_ (Dict[str, float]): Per-model average training
            loss across horizons, used for top-K ranking.
        l2_to_uniform_ (float): Effective λ used for fitting (either
            user-specified ``l2_to_uniform`` or CV-selected).
        l2_cv_scores_ (Dict[float, float]): Average LOMO CV MAE per λ
            candidate. Empty when ``l2_cv_grid=None``.

    Example:
        >>> combiner = DeterministicCombiner(
        ...     point="mean", loss="mae", n_jobs=12
        ... )
        >>> combiner.fit(val_results, observed_df)
        >>> combined = combiner.combine(test_results)
        >>> combined.mu.shape    # (N_test, H)

        >>> # Robust combiner for distribution-shifted test periods
        >>> combiner = DeterministicCombiner(
        ...     point="mean", loss="mse",
        ...     top_k=6, l2_to_uniform=1.0,
        ... )
        >>> combiner.fit(val_results, observed_df)
        >>> combiner.top_k_indices_   # indices of 6 best models
    """

    def __init__(
        self,
        point: str = "mean",
        loss: str = "mae",
        n_jobs: int = 12,
        weights: Optional[np.ndarray] = None,
        val_ratio: float = 0.0,
        top_k: Optional[int] = None,
        l2_to_uniform: float = 0.0,
        l2_cv_grid: Optional[Sequence[float]] = None,
    ):
        if point not in ("mean", "median"):
            raise ValueError(
                f"point must be 'mean' or 'median', got {point!r}."
            )
        if loss not in ("mae", "mse"):
            raise ValueError(
                f"loss must be 'mae' or 'mse', got {loss!r}."
            )
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(
                f"val_ratio must be in [0, 1), got {val_ratio}."
            )
        if top_k is not None and top_k < 2:
            raise ValueError(
                f"top_k must be >= 2 or None, got {top_k}."
            )
        if l2_to_uniform < 0.0:
            raise ValueError(
                f"l2_to_uniform must be >= 0, got {l2_to_uniform}."
            )
        if l2_to_uniform > 0.0 and loss != "mse":
            raise ValueError(
                "l2_to_uniform is only supported with loss='mse'. "
                f"Got loss={loss!r} with l2_to_uniform={l2_to_uniform}."
            )
        if l2_cv_grid is not None:
            if loss != "mse":
                raise ValueError(
                    "l2_cv_grid is only supported with loss='mse'. "
                    f"Got loss={loss!r}."
                )
            grid = list(l2_cv_grid)
            if len(grid) == 0:
                raise ValueError("l2_cv_grid must be non-empty.")
            if any(x < 0 for x in grid):
                raise ValueError("l2_cv_grid values must be non-negative.")
            l2_cv_grid = tuple(grid)

        self.point = point
        self.loss = loss
        self.n_jobs = n_jobs
        self.val_ratio = val_ratio
        self.top_k = top_k
        self.l2_to_uniform = l2_to_uniform
        self.l2_cv_grid = l2_cv_grid
        self._user_weights = (
            None if weights is None else np.asarray(weights, dtype=float)
        )

        self.is_fitted_ = False
        self.weights_: Dict[int, np.ndarray] = {}
        self.train_scores_: Dict[int, float] = {}
        self.val_scores_: Dict[int, float] = {}
        self.model_names_: Optional[List[str]] = None
        self.top_k_indices_: Optional[List[int]] = None
        self.model_scores_: Dict[str, float] = {}
        self.l2_to_uniform_: Optional[float] = None
        self.l2_cv_scores_: Dict[float, float] = {}
        self._n_models: Optional[int] = None
        self._horizon: Optional[int] = None

    # ── Helpers ──

    @staticmethod
    def _validate_weights(
        weights: np.ndarray, n_models: int
    ) -> np.ndarray:
        """Validate combining weights (1-D, finite, non-neg, sum=1)."""
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D.")
        if len(weights) != n_models:
            raise ValueError(
                f"Expected {n_models} weights, got {len(weights)}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Weights must be finite.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("Weights must sum to 1.")
        return weights

    @staticmethod
    def _validate_results(results: List[BaseForecastResult]) -> None:
        """Check that at least 2 models share the same horizon."""
        if len(results) < 2:
            raise ValueError(
                f"At least 2 models required, got {len(results)}."
            )
        H = results[0].horizon
        for i, r in enumerate(results[1:], 1):
            if r.horizon != H:
                raise ValueError(
                    f"Horizon mismatch: results[0].horizon={H}, "
                    f"results[{i}].horizon={r.horizon}"
                )

    @staticmethod
    def _align_results(
        results: List[BaseForecastResult],
    ) -> List[BaseForecastResult]:
        """Reindex all results to their common basis_index intersection."""
        common_idx = results[0].basis_index
        for r in results[1:]:
            common_idx = common_idx.intersection(r.basis_index)
        if len(common_idx) == 0:
            raise ValueError(
                "Common basis_index is empty — models' forecast "
                "periods do not overlap."
            )
        if all(r.basis_index.equals(common_idx) for r in results):
            return results
        return [r.reindex(common_idx) for r in results]

    def _solve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l2_to_uniform: Optional[float] = None,
    ) -> np.ndarray:
        """Dispatch to the loss-specific solver.

        Args:
            X: Point forecasts, (N, K).
            y: Observations, (N,).
            l2_to_uniform: Override for ``self.l2_to_uniform`` (used
                during CV to test candidate λ values). Falls back to
                ``self.l2_to_uniform`` when ``None``.
        """
        if self.loss == "mae":
            return _mae_weights_lp(X, y)
        eff_l2 = l2_to_uniform if l2_to_uniform is not None else self.l2_to_uniform
        return _mse_weights_qp(X, y, l2_to_uniform=eff_l2)

    def _rank_models(
        self, point_tensor: np.ndarray, observed_aligned: np.ndarray, n_train: int
    ) -> tuple:
        """Rank models by average per-horizon MAE (best first).

        Always uses MAE for ranking, regardless of ``self.loss``. MAE
        gives a more robust ranking than MSE under outliers (which
        are common in wind power forecasting) and aligns more closely
        with capacity-normalized nMAPE, the typical downstream metric.

        For each model m, computes the mean MAE across all horizons
        using that model's forecasts alone:

        .. math::
            \\text{score}_m = \\frac{1}{H} \\sum_h
                \\mathbb{E}_n |f_{m,n,h} - y_{n,h}|

        Args:
            point_tensor: Model forecasts, shape (M, N, H).
            observed_aligned: Target observations, shape (N, H).
            n_train: Number of training samples (first rows).

        Returns:
            Tuple of (sorted_indices, per_model_loss):
                - sorted_indices (np.ndarray): Model indices ordered best
                  (lowest MAE) first, shape (M,).
                - per_model_loss (np.ndarray): Average MAE per model,
                  shape (M,).
        """
        M = point_tensor.shape[0]
        H = point_tensor.shape[2]
        per_model = np.zeros(M, dtype=float)
        for m in range(M):
            total = 0.0
            for h in range(H):
                pred = point_tensor[m, :n_train, h]
                obs = observed_aligned[:n_train, h]
                total += float(np.mean(np.abs(pred - obs)))
            per_model[m] = total / H
        return np.argsort(per_model), per_model

    def _run_lomo_cv(
        self,
        point_tensor: np.ndarray,
        observed_aligned: np.ndarray,
        fit_idx: pd.DatetimeIndex,
        K_idx: List[int],
        l2_grid: Sequence[float],
    ) -> Tuple[float, Dict[float, float]]:
        """Leave-one-month-out CV to select best ``l2_to_uniform``.

        For each held-out month m and each candidate λ:
          1. Fit per-horizon weights on (fit rows - month m).
          2. Predict held-out month m rows.
          3. Record MAE across all horizons.

        Aggregation: for each λ, average MAE across folds. Select λ
        with minimum average.

        Args:
            point_tensor: (M, N, H) model forecasts.
            observed_aligned: (N, H) target observations.
            fit_idx: DatetimeIndex of length N labeling fit rows.
            K_idx: Indices of top-K selected models.
            l2_grid: Candidate λ values.

        Returns:
            Tuple of (selected_lambda, cv_scores_dict).

        Raises:
            ValueError: If fewer than 2 distinct months in fit_idx.
        """
        M = point_tensor.shape[0]
        H = point_tensor.shape[2]

        months = pd.DatetimeIndex(fit_idx).to_period("M").astype(str).values
        unique_months = sorted(set(months))
        if len(unique_months) < 2:
            raise ValueError(
                f"LOMO CV requires >=2 distinct months in fit period, "
                f"got {len(unique_months)}: {unique_months}."
            )

        # Flatten tasks: (month, lambda, horizon)
        # For each task: fit weights, evaluate on held-out month
        tasks = []
        for m_idx, held in enumerate(unique_months):
            train_mask = months != held
            test_mask = months == held
            # Skip folds with insufficient data
            if train_mask.sum() < len(K_idx) + 1 or test_mask.sum() < 1:
                continue
            for lam in l2_grid:
                for h in range(1, H + 1):
                    tasks.append((m_idx, held, lam, h, train_mask, test_mask))

        def _task(m_idx, held, lam, h, train_mask, test_mask):
            X_train = point_tensor[K_idx][:, train_mask, h - 1].T  # (n_tr, K)
            y_train = observed_aligned[train_mask, h - 1]
            w_K = _mse_weights_qp(X_train, y_train, l2_to_uniform=lam)
            # Evaluate
            X_test_K = point_tensor[K_idx][:, test_mask, h - 1].T  # (n_te, K)
            y_test = observed_aligned[test_mask, h - 1]
            pred = X_test_K @ w_K
            mae = float(np.mean(np.abs(pred - y_test)))
            return held, float(lam), h, mae

        if self.n_jobs == 1:
            results = [_task(*t) for t in tasks]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_task)(*t) for t in tasks
            )

        # Aggregate: for each lambda, mean MAE over (month, horizon) pairs
        # Structure: results[i] = (held, lam, h, mae)
        per_lambda: Dict[float, List[float]] = {float(lam): [] for lam in l2_grid}
        for held, lam, h, mae in results:
            per_lambda[lam].append(mae)

        cv_scores = {lam: float(np.mean(maes)) for lam, maes in per_lambda.items()}
        best_lam = min(cv_scores, key=cv_scores.get)
        return best_lam, cv_scores

    # ── Fit ──

    def fit(
        self,
        results: List[BaseForecastResult],
        observed: pd.DataFrame,
    ) -> "DeterministicCombiner":
        """Learn per-horizon weights by minimizing MAE or MSE.

        For each horizon h, the combiner stacks the M models' point
        forecasts into an (N_train, M) matrix, pairs it with the
        corresponding observations, and solves an LP (MAE) or QP (MSE)
        to obtain weights summing to 1.

        When ``val_ratio > 0``, the last fraction of data is held out
        as a temporal validation set. Weights are learned on the
        training portion only; per-horizon MAE scores for both splits
        are stored in ``train_scores_`` and ``val_scores_``.

        Args:
            results: M training-period ForecastResult objects.
            observed: Observed values DataFrame of shape (N_total, H).
                Index must contain the common basis times; column ``h``
                (1-indexed) must hold y at ``basis_time + h * step``.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If inputs are invalid (see individual checks).

        Example:
            >>> combiner.fit(val_results, observed_df)
            >>> combiner.weights_[1].sum()     # 1.0
        """
        # Reset state so a failed refit doesn't leave stale weights
        self.is_fitted_ = False
        self.weights_ = {}
        self.train_scores_ = {}
        self.val_scores_ = {}
        self.model_names_ = None
        self.top_k_indices_ = None
        self.model_scores_ = {}
        self.l2_to_uniform_ = None
        self.l2_cv_scores_ = {}
        self._n_models = None
        self._horizon = None

        self._validate_results(results)

        names = [r.model_name for r in results]
        for i, name in enumerate(names):
            if not name:
                raise ValueError(
                    f"results[{i}] has empty model_name. "
                    "All models must have a non-empty model_name."
                )
        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate model_name values: {names}. "
                "Each model must have a unique model_name."
            )

        results = self._align_results(results)

        H = results[0].horizon
        self._horizon = H
        self._n_models = len(results)
        self.model_names_ = names

        common_idx = results[0].basis_index
        fit_idx = common_idx.intersection(observed.index)
        if len(fit_idx) == 0:
            raise ValueError(
                "No overlap between forecast results basis_index "
                "and observed index."
            )

        n_trimmed = len(common_idx) - len(fit_idx)
        if n_trimmed > 0:
            print(
                f"[DeterministicCombiner.fit] Trimmed {n_trimmed} origins "
                f"without observed data: {len(common_idx)} → "
                f"{len(fit_idx)} ({len(fit_idx)} used for fitting)"
            )
            results = [r.reindex(fit_idx) for r in results]

        observed_common = observed.loc[fit_idx]
        if observed_common.shape[1] != H:
            raise ValueError(
                f"observed must have {H} columns (one per horizon), "
                f"got {observed_common.shape[1]}."
            )
        observed_aligned = observed_common.values  # (N, H)
        if not np.all(np.isfinite(observed_aligned)):
            raise ValueError(
                "observed contains NaN or Inf values. "
                "All observations must be finite."
            )

        # Extract point forecasts once: (M, N, H)
        point_tensor = np.stack([
            _extract_point_array(r, self.point) for r in results
        ])

        # Temporal train/validation split
        N = observed_aligned.shape[0]
        if self.val_ratio > 0:
            n_val = max(1, int(N * self.val_ratio))
            n_train = N - n_val
            if n_train < 2:
                raise ValueError(
                    f"val_ratio={self.val_ratio} leaves only {n_train} "
                    f"training samples (need >= 2)."
                )
        else:
            n_train = N

        # Top-K model selection (by average training loss across horizons).
        # Skip entirely when user provides fixed weights — user chose them.
        if self.top_k is not None and self._user_weights is None:
            if self.top_k >= self._n_models:
                self.top_k_indices_ = list(range(self._n_models))
            else:
                ranking, per_model_loss = self._rank_models(
                    point_tensor, observed_aligned, n_train
                )
                # Keep sorted ascending so weights_ layout is predictable
                self.top_k_indices_ = sorted(
                    ranking[:self.top_k].astype(int).tolist()
                )
                self.model_scores_ = {
                    names[i]: float(per_model_loss[i]) for i in range(self._n_models)
                }
        else:
            self.top_k_indices_ = list(range(self._n_models))
            if self._user_weights is None:
                # Still compute and expose scores for diagnostics
                _, per_model_loss = self._rank_models(
                    point_tensor, observed_aligned, n_train
                )
                self.model_scores_ = {
                    names[i]: float(per_model_loss[i]) for i in range(self._n_models)
                }

        K_idx = self.top_k_indices_
        K = len(K_idx)

        # LOMO CV to select l2_to_uniform (if grid provided)
        if (
            self.l2_cv_grid is not None
            and self._user_weights is None
            and self.loss == "mse"
        ):
            fit_basis = fit_idx[:n_train]
            best_lam, cv_scores = self._run_lomo_cv(
                point_tensor[:, :n_train, :],
                observed_aligned[:n_train],
                fit_basis,
                K_idx,
                list(self.l2_cv_grid),
            )
            self.l2_to_uniform_ = best_lam
            self.l2_cv_scores_ = cv_scores
        else:
            self.l2_to_uniform_ = self.l2_to_uniform

        effective_l2 = self.l2_to_uniform_

        def _fit_single(h: int) -> tuple:
            # Only fit on top-K subset
            X_train = point_tensor[K_idx, :n_train, h - 1].T  # (N_train, K)
            y_train = observed_aligned[:n_train, h - 1]
            if self._user_weights is not None:
                w_full = self._user_weights
            else:
                w_K = self._solve(X_train, y_train, l2_to_uniform=effective_l2)
                # Expand back to full M dimensions (zeros for excluded models)
                w_full = np.zeros(self._n_models, dtype=float)
                w_full[K_idx] = w_K
            return h, w_full

        # Horizon-level parallelism
        if self.n_jobs == 1:
            fit_results = [_fit_single(h) for h in range(1, H + 1)]
        else:
            fit_results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single)(h) for h in range(1, H + 1)
            )

        for h, w in fit_results:
            self._validate_weights(w, self._n_models)
            self.weights_[h] = w

            # Scores (always MAE, regardless of loss choice)
            X_train_h = point_tensor[:, :n_train, h - 1].T
            y_train_h = observed_aligned[:n_train, h - 1]
            self.train_scores_[h] = float(
                np.mean(np.abs(X_train_h @ w - y_train_h))
            )
            if self.val_ratio > 0:
                X_val_h = point_tensor[:, n_train:, h - 1].T
                y_val_h = observed_aligned[n_train:, h - 1]
                self.val_scores_[h] = float(
                    np.mean(np.abs(X_val_h @ w - y_val_h))
                )

        self.is_fitted_ = True
        return self

    # ── Combine ──

    def combine(
        self,
        results: List[BaseForecastResult],
        model_name: Optional[str] = None,
    ) -> DeterministicForecastResult:
        """Apply learned weights to produce combined point forecasts.

        For each horizon h, extracts each model's point forecast
        (mean or median) and computes the weighted sum
        ``Σ_i w_h[i] * x_i(t)``, which is the combined point forecast.

        Args:
            results: M test-period ForecastResult objects (same
                model_name values and order as passed to fit()).
            model_name: Name for the combined result. Defaults to
                ``"DeterministicCombiner"``.

        Returns:
            DeterministicForecastResult with combined mu of shape
            (N_test, H).

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If results are incompatible with fit() inputs.

        Example:
            >>> combined = combiner.combine(test_results, model_name="MAE_ens")
            >>> combined.mu.shape      # (N_test, 48)
        """
        if not self.is_fitted_:
            raise RuntimeError("fit() must be called before combine().")

        self._validate_results(results)
        results = self._align_results(results)

        H = results[0].horizon
        if H != self._horizon:
            raise ValueError(
                f"Expected horizon {self._horizon} from fit(), got {H}."
            )
        if len(results) != self._n_models:
            raise ValueError(
                f"Expected {self._n_models} models from fit(), "
                f"got {len(results)}."
            )

        combine_names = [r.model_name for r in results]
        if combine_names != self.model_names_:
            raise ValueError(
                "Model names/order mismatch between fit() and combine(). "
                f"Expected {self.model_names_}, got {combine_names}."
            )

        basis_index = results[0].basis_index

        # Extract point forecasts: (M, N, H)
        point_tensor = np.stack([
            _extract_point_array(r, self.point) for r in results
        ])

        # Weighted combination per horizon
        N = point_tensor.shape[1]
        mu_combined = np.empty((N, H), dtype=float)
        for h in range(1, H + 1):
            X_h = point_tensor[:, :, h - 1].T  # (N, M)
            mu_combined[:, h - 1] = X_h @ self.weights_[h]

        return DeterministicForecastResult(
            mu=mu_combined,
            basis_index=basis_index,
            model_name=model_name or "DeterministicCombiner",
        )

    # ── Save / Load ──

    def save(self, path: Union[str, Path]) -> Path:
        """Save fitted combiner to a directory (YAML metadata + NPZ weights).

        Directory layout::

            path/
                metadata.yaml   # config + fitted scalar attributes
                weights.npz     # per-horizon weight arrays ("1", "2", ...)

        Args:
            path: Directory to save into. Created if it does not exist.

        Returns:
            Path to the saved directory.

        Raises:
            RuntimeError: If ``fit()`` has not been called.

        Example:
            >>> combiner.fit(val_results, observed_df)
            >>> combiner.save("res/det_combiner/exp_01")
        """
        if not self.is_fitted_:
            raise RuntimeError("fit() must be called before save().")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # ── metadata ──
        metadata = {
            "config": {
                "point": self.point,
                "loss": self.loss,
                "n_jobs": self.n_jobs,
                "val_ratio": float(self.val_ratio),
                "top_k": self.top_k,
                "l2_to_uniform": float(self.l2_to_uniform),
                "l2_cv_grid": (
                    [float(x) for x in self.l2_cv_grid]
                    if self.l2_cv_grid is not None
                    else None
                ),
            },
            "fitted": {
                "horizon": self._horizon,
                "n_models": self._n_models,
                "model_names": self.model_names_,
                "top_k_indices": self.top_k_indices_,
                "l2_to_uniform_selected": (
                    float(self.l2_to_uniform_)
                    if self.l2_to_uniform_ is not None
                    else None
                ),
                "train_scores": {
                    int(k): float(v) for k, v in self.train_scores_.items()
                },
                "val_scores": {
                    int(k): float(v) for k, v in self.val_scores_.items()
                },
                "model_scores": {
                    str(k): float(v)
                    for k, v in self.model_scores_.items()
                },
                "l2_cv_scores": {
                    float(k): float(v)
                    for k, v in self.l2_cv_scores_.items()
                },
            },
        }
        with open(path / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        # ── weights ──
        weights_dict = {str(h): w for h, w in self.weights_.items()}
        np.savez(path / "weights.npz", **weights_dict)

        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeterministicCombiner":
        """Load a fitted combiner from a directory saved by ``save()``.

        Args:
            path: Directory containing ``metadata.yaml`` and ``weights.npz``.

        Returns:
            A fitted ``DeterministicCombiner`` ready for ``combine()``.

        Raises:
            FileNotFoundError: If required files are missing.

        Example:
            >>> combiner = DeterministicCombiner.load("res/det_combiner/exp_01")
            >>> combined = combiner.combine(test_results)
        """
        path = Path(path)
        with open(path / "metadata.yaml") as f:
            metadata = yaml.safe_load(f)

        cfg = metadata["config"]
        fitted = metadata["fitted"]

        combiner = cls(
            point=cfg["point"],
            loss=cfg["loss"],
            n_jobs=cfg["n_jobs"],
            val_ratio=cfg["val_ratio"],
            top_k=cfg["top_k"],
            l2_to_uniform=cfg["l2_to_uniform"],
            l2_cv_grid=(
                tuple(cfg["l2_cv_grid"])
                if cfg["l2_cv_grid"] is not None
                else None
            ),
        )

        # Restore fitted state
        combiner._horizon = fitted["horizon"]
        combiner._n_models = fitted["n_models"]
        combiner.model_names_ = fitted["model_names"]
        combiner.top_k_indices_ = fitted["top_k_indices"]
        combiner.l2_to_uniform_ = fitted["l2_to_uniform_selected"]
        combiner.train_scores_ = {
            int(k): v for k, v in fitted["train_scores"].items()
        }
        combiner.val_scores_ = {
            int(k): v for k, v in fitted["val_scores"].items()
        }
        combiner.model_scores_ = fitted["model_scores"]
        combiner.l2_cv_scores_ = {
            float(k): v for k, v in fitted["l2_cv_scores"].items()
        }

        # Restore weights
        data = np.load(path / "weights.npz")
        combiner.weights_ = {int(k): data[k] for k in data.files}

        combiner.is_fitted_ = True
        return combiner
