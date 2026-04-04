"""Vertical combining: CDF weighted average (Linear Pool)."""

import warnings
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

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


def _batch_cdf_eval(x_grid, qvals, tau):
    """Evaluate CDF at x_grid for all timesteps (vectorized).

    For each row n, computes CDF_n(x) = interp(x, qvals[n], tau)
    with left=0.0, right=1.0.

    Args:
        x_grid: Evaluation points, shape (N, G).
        qvals: Quantile values (sorted), shape (N, Q).
        tau: Quantile levels, shape (Q,).

    Returns:
        np.ndarray: CDF values, shape (N, G).

    Example:
        >>> cdf = _batch_cdf_eval(x_grid, qvals, tau)
        >>> cdf.shape
        (50, 200)
    """
    Q = qvals.shape[1]

    # Find indices: mask[n, g, q] = (x_grid[n, g] >= qvals[n, q])
    mask = x_grid[:, :, None] >= qvals[:, None, :]  # (N, G, Q)
    idx = mask.sum(axis=2) - 1  # (N, G)
    idx = np.clip(idx, 0, Q - 2)

    row = np.arange(qvals.shape[0])[:, None]
    x_lo = qvals[row, idx]
    x_hi = qvals[row, idx + 1]
    f_lo = tau[idx]
    f_hi = tau[idx + 1]

    dx = x_hi - x_lo
    safe_dx = np.where(dx < 1e-15, 1.0, dx)
    t = np.clip((x_grid - x_lo) / safe_dx, 0.0, 1.0)
    # For point masses (dx ~ 0): use f_hi (max level), matching
    # _cdf_from_quantiles which keeps the last (max) level
    result = np.where(dx < 1e-15, f_hi, f_lo + t * (f_hi - f_lo))

    # Clamp out-of-range
    result = np.where(x_grid < qvals[:, :1], 0.0, result)
    result = np.where(x_grid > qvals[:, -1:], 1.0, result)

    return result


def _batch_cdf_invert(tau, cdf, x_grid):
    """Invert combined CDF to get quantiles (vectorized).

    For each row n, computes interp(tau, cdf[n], x_grid[n]).

    Args:
        tau: Target quantile levels, shape (Q,).
        cdf: Combined CDF values, shape (N, G).
        x_grid: x values corresponding to CDF, shape (N, G).

    Returns:
        np.ndarray: Quantile values, shape (N, Q).

    Example:
        >>> quantiles = _batch_cdf_invert(tau, combined_cdf, x_grid)
        >>> quantiles.shape
        (50, 19)
    """
    G = cdf.shape[1]
    Q = len(tau)

    # Find indices: mask[n, q, g] = (tau[q] >= cdf[n, g])
    tau_exp = tau[None, :, None]   # (1, Q, 1)
    cdf_exp = cdf[:, None, :]      # (N, 1, G)
    mask = tau_exp >= cdf_exp       # (N, Q, G)
    idx = mask.sum(axis=2) - 1     # (N, Q)
    idx = np.clip(idx, 0, G - 2)

    row = np.arange(cdf.shape[0])[:, None]
    cdf_lo = cdf[row, idx]
    cdf_hi = cdf[row, idx + 1]
    x_lo = x_grid[row, idx]
    x_hi = x_grid[row, idx + 1]

    dcdf = cdf_hi - cdf_lo
    safe_dcdf = np.where(dcdf < 1e-15, 1.0, dcdf)
    t = np.clip((tau[None, :] - cdf_lo) / safe_dcdf, 0.0, 1.0)

    return x_lo + t * (x_hi - x_lo)


def _vertical_combine_quantiles(w, tau, quantile_arrays):
    """Compute combined quantiles via CDF averaging for given weights.

    Vectorized implementation: evaluates all N timesteps in parallel
    using batch CDF evaluation and inversion, avoiding Python loops.

    Args:
        w: Weight vector, shape (M,).
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).

    Returns:
        np.ndarray: Combined quantile values, shape (N, Q).

    Example:
        >>> combined = _vertical_combine_quantiles(
        ...     np.array([0.6, 0.4]), tau, [q1, q2]
        ... )
    """
    N, Q = quantile_arrays[0].shape
    M = len(quantile_arrays)

    # Stack all quantile arrays: (M, N, Q)
    all_q = np.stack(quantile_arrays)

    # Per-timestep bounds across all models
    x_min = all_q.min(axis=(0, 2))  # (N,)
    x_max = all_q.max(axis=(0, 2))  # (N,)
    span = np.maximum(x_max - x_min, 1e-10)

    # Build grid: exact quantile positions + uniform interpolation.
    # Including exact positions ensures CDF round-trips are precise.
    G_lin = max(5 * Q, 300)
    t_lin = np.linspace(0.0, 1.0, G_lin)
    x_lin = x_min[:, None] + t_lin[None, :] * span[:, None]  # (N, G_lin)

    # Exact quantile positions from all models: (N, M*Q)
    x_exact = all_q.transpose(1, 0, 2).reshape(N, M * Q)

    # Concatenate and sort: (N, G_lin + M*Q)
    x_grid = np.sort(
        np.concatenate([x_exact, x_lin], axis=1), axis=1
    )

    # Weighted CDF averaging over models
    G = x_grid.shape[1]
    combined_cdf = np.zeros((N, G), dtype=float)
    for i in range(M):
        combined_cdf += w[i] * _batch_cdf_eval(x_grid, all_q[i], tau)

    # Invert combined CDF to get output quantiles
    return _batch_cdf_invert(tau, combined_cdf, x_grid)


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

    Example:
        >>> loss = _vertical_objective(
        ...     np.array([0.6, 0.4]), tau, qarrays, obs, 0.0
        ... )
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
            - "optimize": Minimize combined CRPS via SLSQP (default).
              CRPS is convex w.r.t. weights (Theorem 3, Taylor &
              Meng 2025), so SLSQP finds the global optimum.
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

        Dispatches to inverse-CRPS heuristic or SLSQP optimization
        based on ``fit_method``. User-specified weights always take
        precedence.

        The vertical CRPS objective is convex w.r.t. weights
        (Theorem 3, Taylor & Meng 2025), so SLSQP with
        finite-difference gradients reliably finds the global
        optimum from the inverse-CRPS warm start.

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

        result = minimize(
            _vertical_objective,
            x0,
            args=(tau, quantile_arrays, observed, self.reg_lambda),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * M,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 200, "ftol": 1e-10},
        )

        if not result.success:
            warnings.warn(
                f"SLSQP did not converge for horizon {h}: "
                f"{result.message}. Falling back to inverse-CRPS "
                f"weights.",
                stacklevel=2,
            )
            return x0

        w_opt = np.maximum(result.x, 0.0)
        if not np.all(np.isfinite(w_opt)) or w_opt.sum() <= 0:
            warnings.warn(
                f"SLSQP returned invalid weights for horizon {h}. "
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
