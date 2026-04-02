"""Horizontal combining: quantile function weighted average."""

import warnings
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from src.utils.metrics import crps_quantile

from .base import BaseCombiner


def _horizontal_objective(w, tau, Q_stack, observed, reg_lambda):
    """CRPS objective for horizontal (quantile averaging) combining.

    Args:
        w: Weight vector, shape (M,).
        tau: Quantile levels, shape (Q,).
        Q_stack: Stacked quantile arrays, shape (M, N, Q).
        observed: Observed values, shape (N,).
        reg_lambda: L2 regularization strength toward equal weights.

    Returns:
        float: CRPS + regularization penalty.
    """
    q_combined = np.einsum("m,mnq->nq", w, Q_stack)
    loss = crps_quantile(tau, q_combined, observed, reduction="mean")
    if reg_lambda > 0:
        M = len(w)
        loss += reg_lambda * np.sum((w - 1.0 / M) ** 2)
    return loss


class HorizontalCombiner(BaseCombiner):
    """Quantile function weighted average (Quantile Averaging).

    Q_combined(u) = sum_i w_i * Q_i(u)

    Averages quantile values at the same probability level u.
    Weights are learned via CRPS minimization (default), inverse-CRPS
    heuristic, or user-specified.

    Args:
        n_quantiles: Number of quantile levels (default 99).
        n_jobs: Parallel workers for horizon loop (default 1).
        weights: User-specified weights, shape (M,). If None,
            learns weights during fit().
        fit_method: Weight learning strategy.
            - "optimize": Minimize combined CRPS via SLSQP (default).
            - "inverse_crps": Heuristic inverse-CRPS weighting.
        reg_lambda: L2 regularization toward equal weights (default 0.0).
        val_ratio: Fraction of data for temporal validation (default 0.0).

    Example:
        >>> combiner = HorizontalCombiner(
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

        Dispatches to inverse-CRPS heuristic or SLSQP optimization
        based on ``fit_method``. User-specified weights always take
        precedence.

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

        # SLSQP optimization with inverse-CRPS warm start
        M = len(quantile_arrays)
        Q_stack = np.stack(quantile_arrays, axis=0)  # (M, N, Q)

        result = minimize(
            _horizontal_objective,
            x0,
            args=(tau, Q_stack, observed, self.reg_lambda),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * M,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 200, "ftol": 1e-10},
        )

        if not result.success:
            warnings.warn(
                f"Optimization did not converge for horizon {h}: "
                f"{result.message}. Falling back to inverse-CRPS weights.",
                stacklevel=2,
            )
            return x0

        w_opt = np.maximum(result.x, 0.0)
        if not np.all(np.isfinite(w_opt)) or w_opt.sum() <= 0:
            warnings.warn(
                f"Optimization returned invalid weights for horizon {h}. "
                f"Falling back to inverse-CRPS weights.",
                stacklevel=2,
            )
            return x0
        w_opt /= w_opt.sum()
        return w_opt

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

        q_sum = np.zeros_like(quantile_arrays[0], dtype=float)
        for w, q in zip(weights, quantile_arrays):
            q_sum += w * q

        return q_sum
