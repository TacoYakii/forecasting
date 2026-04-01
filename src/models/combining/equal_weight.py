"""Equal-weight forecast combiner (baseline)."""

from typing import List

import numpy as np

from .base import BaseCombiner


class EqualWeightCombiner(BaseCombiner):
    """Simple equal-weight quantile averaging combiner.

    Assigns weight 1/M to each of the M models. No learned parameters.
    Uses quantile function averaging: Q_combined(u) = (1/M) * sum Q_i(u).

    Args:
        n_quantiles: Number of quantile levels (default 99).
        n_jobs: Parallel workers for horizon loop (default 1).

    Example:
        >>> combiner = EqualWeightCombiner(n_quantiles=99)
        >>> combiner.fit(train_results, observed)
        >>> combiner.weights_[1]
        {'ARIMA': 0.333, 'DeepAR': 0.333, 'TFT': 0.333}
        >>> combined = combiner.combine(test_results)
    """

    def _fit_horizon(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
        observed: np.ndarray,
    ) -> np.ndarray:
        """Return equal weights 1/M.

        Args:
            h: Forecast horizon (1-indexed).
            quantile_arrays: M arrays of shape (N_train, Q).
            observed: Observed values, shape (N_train,).

        Returns:
            np.ndarray: Equal weights, shape (M,).

        Example:
            >>> weights = combiner._fit_horizon(1, qarrays, obs)
            >>> np.allclose(weights, 1/3)
            True
        """
        M = len(quantile_arrays)
        return np.full(M, 1.0 / M)

    def _combine_distributions(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
    ) -> np.ndarray:
        """Equal-weight quantile averaging.

        Q_combined(u) = (1/M) * sum_i Q_i(u) for each quantile level u.

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

        q_sum = np.zeros_like(quantile_arrays[0])
        for w, q in zip(weights, quantile_arrays):
            q_sum += w * q

        return q_sum
