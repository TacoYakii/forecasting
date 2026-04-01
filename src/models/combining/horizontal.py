"""Horizontal combining: quantile function weighted average."""

from typing import List, Optional

import numpy as np

from .base import BaseCombiner


def _pinball_loss(tau: np.ndarray, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pinball (quantile) loss for multiple quantile levels.

    Args:
        tau: Quantile levels, shape (Q,).
        q: Predicted quantile values, shape (N, Q).
        y: Observed values, shape (N,).

    Returns:
        np.ndarray: Per-observation mean pinball loss, shape (N,).

    Example:
        >>> loss = _pinball_loss(np.array([0.5]), np.array([[1.0]]), np.array([1.5]))
    """
    diff = y[:, None] - q  # (N, Q)
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)  # (N, Q)
    return loss.mean(axis=1)  # (N,)


def _crps_from_quantiles(
    tau: np.ndarray, q: np.ndarray, y: np.ndarray
) -> float:
    """Approximate CRPS via quantile decomposition.

    CRPS ~= 2 * mean(pinball_loss) over all quantile levels and observations.

    Args:
        tau: Quantile levels, shape (Q,).
        q: Predicted quantile values, shape (N, Q).
        y: Observed values, shape (N,).

    Returns:
        float: Mean CRPS across all observations.

    Example:
        >>> crps = _crps_from_quantiles(
        ...     np.array([0.25, 0.5, 0.75]),
        ...     np.array([[1.0, 2.0, 3.0]]),
        ...     np.array([2.0]),
        ... )
    """
    return 2.0 * _pinball_loss(tau, q, y).mean()


class HorizontalCombiner(BaseCombiner):
    """Quantile function weighted average (Quantile Averaging).

    Q_combined(u) = sum_i w_i * Q_i(u)

    Averages quantile values at the same probability level u.
    Weights are learned via inverse-CRPS or user-specified.

    Args:
        n_quantiles: Number of quantile levels (default 99).
        n_jobs: Parallel workers for horizon loop (default 1).
        weights: User-specified weights, shape (M,). If None,
            learns inverse-CRPS weights during fit().

    Example:
        >>> combiner = HorizontalCombiner(n_quantiles=99, n_jobs=-1)
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

        w_i = (1 / CRPS_i) / sum_j(1 / CRPS_j), so lower CRPS
        yields higher weight.

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

        crps_arr = np.array(crps_scores)  # (M,)

        # Inverse-CRPS weighting with safety floor
        crps_arr = np.maximum(crps_arr, 1e-12)
        inv_crps = 1.0 / crps_arr
        return inv_crps / inv_crps.sum()

    def _combine_distributions(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
    ) -> np.ndarray:
        """Quantile function weighted average.

        Q_combined(u) = sum_i w_i * Q_i(u).
        No cross-quantile mixing since averaging at the same level.

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
