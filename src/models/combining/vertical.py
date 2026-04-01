"""Vertical combining: CDF weighted average (Linear Pool)."""

from typing import List, Optional

import numpy as np

from .base import BaseCombiner
from .horizontal import _crps_from_quantiles


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
            learns inverse-CRPS weights during fit().

    Example:
        >>> combiner = VerticalCombiner(n_quantiles=99, n_jobs=-1)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        n_jobs: int = 1,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(n_quantiles=n_quantiles, n_jobs=n_jobs)
        self._user_weights = (
            None if weights is None else np.asarray(weights, dtype=float)
        )

    def _fit_horizon(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
        observed: np.ndarray,
    ) -> np.ndarray:
        """Learn inverse-CRPS weights for a single horizon.

        Uses the same inverse-CRPS scheme as HorizontalCombiner.

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
        crps_scores = []
        for q in quantile_arrays:
            crps = _crps_from_quantiles(tau, q, observed)
            crps_scores.append(crps)

        crps_arr = np.array(crps_scores)
        crps_arr = np.maximum(crps_arr, 1e-12)
        inv_crps = 1.0 / crps_arr
        return inv_crps / inv_crps.sum()

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

        tau = self.quantile_levels
        N = quantile_arrays[0].shape[0]
        Q_out = self.n_quantiles

        result = np.empty((N, Q_out), dtype=float)

        for t in range(N):
            # Pool quantile values from all models → common x grid
            x_pool = np.concatenate([q[t] for q in quantile_arrays])
            x_grid = np.unique(x_pool)  # sorted unique values

            # Evaluate each model's CDF at grid points via quantile interpolation
            combined_cdf = np.zeros_like(x_grid)
            for w, q in zip(weights, quantile_arrays):
                combined_cdf += w * _cdf_from_quantiles(tau, q[t], x_grid)

            # Invert combined CDF: Q(u) = inf{x : F(x) >= u}
            # searchsorted(side="left") finds first index where cdf >= tau,
            # correctly handling both flat segments and jumps (point masses).
            idx = np.searchsorted(combined_cdf, self.quantile_levels, side="left")
            idx = np.clip(idx, 0, len(x_grid) - 1)
            result[t] = x_grid[idx]

        return result
