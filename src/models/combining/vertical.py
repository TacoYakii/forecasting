"""Vertical combining: CDF weighted average (Linear Pool)."""

import warnings
from typing import List, Optional

import numpy as np

from src.utils.metrics import crps_quantile

from .base import BaseCombiner


def _cdf_from_quantiles(
    quantile_levels: np.ndarray, quantile_values: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Evaluate empirical CDF at arbitrary x values from quantile representation.

    Uses linear interpolation between known quantile (level, value) pairs.
    For repeated quantile values (point mass), keeps the maximum level
    to satisfy F(x) = P(X <= x).
    Clamps to [0, 1] outside the observed range.

    Args:
        quantile_levels: Sorted quantile levels, shape (Q,).
        quantile_values: Quantile values for one time step, shape (Q,).
        x: 1-D array of values at which to evaluate CDF.

    Returns:
        np.ndarray of shape (len(x),) with CDF values in [0, 1].

    Example:
        >>> cdf = _cdf_from_quantiles(
        ...     np.array([0.1, 0.2, 0.3, 0.5, 0.9]),
        ...     np.array([0.0, 0.0, 0.0, 15.3, 42.1]),
        ...     np.array([0.0, 10.0]),
        ... )
    """
    # Remove duplicate xp values, keeping last (max level) for F(x) = P(X <= x)
    unique_mask = np.diff(quantile_values, append=np.inf) > 0
    return np.interp(
        x,
        quantile_values[unique_mask],
        quantile_levels[unique_mask],
        left=0.0,
        right=1.0,
    )


def _vertical_combine_quantiles(w, tau, quantile_arrays):
    """Compute combined quantiles via CDF averaging for given weights.

    Args:
        w: Weight vector, shape (M,).
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).

    Returns:
        np.ndarray: Combined quantile values, shape (N, Q).
    """
    N, Q = quantile_arrays[0].shape
    result = np.empty((N, Q), dtype=float)

    for t in range(N):
        x_pool = np.concatenate([q[t] for q in quantile_arrays])
        x_grid = np.unique(x_pool)

        combined_cdf = np.zeros_like(x_grid, dtype=float)
        for wi, q in zip(w, quantile_arrays):
            combined_cdf += wi * _cdf_from_quantiles(tau, q[t], x_grid)

        idx = np.searchsorted(combined_cdf, tau, side="left")
        idx = np.clip(idx, 0, len(x_grid) - 1)
        result[t] = x_grid[idx]

    return result


def _simplex_search(objective, x0, M, n_candidates_base=200, seed=42):
    """Direct search on the probability simplex for weight optimization.

    Generates candidate weight vectors from a Dirichlet distribution
    concentrated around ``x0``, evaluates the objective for each, and
    returns the best. Suitable for piecewise-constant objectives
    where gradient-based methods fail.

    The number of candidates scales with the simplex dimension
    (M - 1) to maintain adequate coverage in higher dimensions.

    Args:
        objective: Callable(w) -> float.
        x0: Initial weight vector (inverse-CRPS), shape (M,).
        M: Number of models.
        n_candidates_base: Base number of candidates for M=2.
            Actual count = n_candidates_base * (M - 1).
        seed: RNG seed for reproducibility.

    Returns:
        np.ndarray: Best weight vector, shape (M,).

    Example:
        >>> best = _simplex_search(obj, x0=np.array([0.6, 0.4]), M=2)
        >>> best.sum()
        1.0
    """
    rng = np.random.default_rng(seed)
    n_candidates = n_candidates_base * (M - 1)

    # Dirichlet alpha concentrated around x0 (higher = tighter)
    concentration = 50.0
    alpha = np.maximum(x0 * concentration, 0.1)
    candidates = rng.dirichlet(alpha, size=n_candidates)

    # Always include x0 and equal weights as candidates
    candidates = np.vstack([x0, np.full(M, 1.0 / M), candidates])

    scores = np.array([objective(c) for c in candidates])

    # Filter non-finite scores; fall back to x0 if all are non-finite
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return x0
    scores[~finite_mask] = np.inf
    return candidates[np.argmin(scores)]


def _vertical_objective(w, tau, quantile_arrays, observed, reg_lambda):
    """CRPS objective for vertical (CDF averaging) combining.

    Args:
        w: Weight vector, shape (M,).
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).
        observed: Observed values, shape (N,).
        reg_lambda: L2 regularization strength toward equal weights.

    Returns:
        float: CRPS + regularization penalty.
    """
    q_combined = _vertical_combine_quantiles(w, tau, quantile_arrays)
    loss = crps_quantile(tau, q_combined, observed, reduction="mean")
    if reg_lambda > 0:
        M = len(w)
        loss += reg_lambda * np.sum((w - 1.0 / M) ** 2)
    return loss


class VerticalCombiner(BaseCombiner):
    """CDF weighted average (Linear Pool).

    F_combined(x) = sum_i w_i * F_i(x)

    Averages CDFs at the same x value. The combined distribution is
    generally wider than the individual distributions, which helps
    when individual models are under-dispersed.

    Implementation:
        1. Build a common x grid from pooled quantile values
        2. Evaluate each model's CDF on the grid
        3. Weighted-average the CDFs
        4. Invert the combined CDF to obtain output quantiles

    Args:
        n_quantiles: Number of output quantile levels (default 99).
        n_jobs: Parallel workers for horizon loop (default 1).
        weights: User-specified weights, shape (M,). If None,
            learns weights during fit().
        fit_method: Weight learning strategy.
            - "optimize": Minimize combined CRPS via derivative-free
              simplex search (default). Uses Dirichlet-seeded
              candidates around inverse-CRPS warm start.
            - "inverse_crps": Heuristic inverse-CRPS weighting.
        reg_lambda: L2 regularization toward equal weights (default 0.0).
        val_ratio: Fraction of data for temporal validation (default 0.0).

    Example:
        >>> combiner = VerticalCombiner(
        ...     n_quantiles=99, fit_method="optimize", val_ratio=0.2
        ... )
        >>> combiner.fit(train_results, observed)
        >>> combiner.val_scores_  # {1: 0.32, 2: 0.35, 3: 0.41}
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        n_jobs: int = 1,
        weights: Optional[np.ndarray] = None,
        fit_method: str = "optimize",
        reg_lambda: float = 0.0,
        val_ratio: float = 0.0,
    ):
        super().__init__(
            n_quantiles=n_quantiles, n_jobs=n_jobs, val_ratio=val_ratio
        )
        self._user_weights = (
            None if weights is None else np.asarray(weights, dtype=float)
        )
        if fit_method not in ("inverse_crps", "optimize"):
            raise ValueError(
                f"fit_method must be 'inverse_crps' or 'optimize', "
                f"got {fit_method!r}"
            )
        self.fit_method = fit_method
        self.reg_lambda = reg_lambda

    def _fit_horizon(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
        observed: np.ndarray,
    ) -> np.ndarray:
        """Learn combining weights for a single horizon.

        Dispatches to inverse-CRPS heuristic or simplex grid search
        based on ``fit_method``. User-specified weights always take
        precedence.

        The vertical objective is piecewise constant in w due to
        ``np.searchsorted`` in the CDF inversion step, so gradient-
        based optimizers (SLSQP) cannot reliably find improvements.
        Instead, ``fit_method="optimize"`` uses a derivative-free
        Dirichlet-seeded grid search on the simplex, centered on
        the inverse-CRPS warm start.

        Args:
            h: Forecast horizon (1-indexed).
            quantile_arrays: M arrays of shape (N_train, Q).
            observed: Observed values, shape (N_train,).

        Returns:
            np.ndarray: Normalized weights, shape (M,).

        Example:
            >>> weights = combiner._fit_horizon(1, qarrays, obs)
            >>> weights.sum()
            1.0
        """
        if self._user_weights is not None:
            return self._validate_weights(
                self._user_weights, len(quantile_arrays)
            )

        tau = self.quantile_levels
        x0 = self._inverse_crps_weights(tau, quantile_arrays, observed)

        if self.fit_method == "inverse_crps":
            return x0

        # Derivative-free simplex grid search
        M = len(quantile_arrays)

        def objective(w):
            return _vertical_objective(
                w, tau, quantile_arrays, observed, self.reg_lambda
            )

        best_w = _simplex_search(objective, x0, M)
        return best_w

    def _combine_distributions(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
    ) -> np.ndarray:
        """CDF weighted average with quantile inversion.

        For each time step:
            1. Build a common x grid from pooled quantile values
            2. Evaluate each model's CDF via quantile interpolation
            3. Compute F_combined(x) = sum w_i F_i(x)
            4. Invert to find quantiles at self.quantile_levels

        Args:
            h: Forecast horizon (1-indexed).
            quantile_arrays: M arrays of shape (N, Q).

        Returns:
            np.ndarray: Combined quantile values, shape (N, Q).

        Example:
            >>> combined = combiner._combine_distributions(1, qarrays)
            >>> combined.shape
            (100, 99)
        """
        weights = self.weights_[h]  # (M,)
        return _vertical_combine_quantiles(
            weights, self.quantile_levels, quantile_arrays
        )
