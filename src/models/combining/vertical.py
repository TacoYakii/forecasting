"""Vertical combining: CDF weighted average (Linear Pool).

Provides two implementations:
- **Sampling-based** (default): Uses the simulation algorithm from
  Taylor & Meng (2025, Appendix C). Vertical combining is a mixture
  distribution, so sampling is trivial. Angular combining applies
  the h_θ quantile transform before mixture sampling.
- **Grid-based** (legacy): Evaluates CDFs on a common x-grid, computes
  the weighted CDF average, and inverts. Retained as ``*_grid``
  functions for reference and regression testing.
"""

import warnings
from typing import List, Optional

import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from src.utils.metrics import crps_quantile

from .base import BaseCombiner

# ── Efficiency thresholds for angular dispatch ──
# float64 cancellation error is ~6e-15 even at θ=1°, so these
# thresholds exist purely to skip redundant computation.
HORIZ_THRESH = 1.0  # degrees: θ < 1° → horizontal quantile average
VERT_THRESH = 89.0  # degrees: θ > 89° → vertical (no h_θ transform)


# =====================================================================
# Numba JIT kernel for mixture interpolation
# =====================================================================


@njit(cache=True, parallel=True)
def _interp_mixture_kernel(j, u, tau, all_q, s):
    """Fused model selection + quantile interpolation (Numba JIT).

    For each (n, i), computes s[n, i] = interp(u[n, i], tau, all_q[j[n,i], n, :])
    using binary search + linear interpolation, parallelized over N timesteps.

    Args:
        j: Model indices, shape (N, n_samples), dtype int64.
        u: Uniform draws, shape (N, n_samples), dtype float64.
        tau: Quantile levels (sorted), shape (Q,), dtype float64.
        all_q: Stacked quantile arrays, shape (M, N, Q), dtype float64.
        s: Output array, shape (N, n_samples), dtype float64. Modified in-place.
    """
    N, n_samples = j.shape
    Q = tau.shape[0]
    for n in prange(N):
        for i in range(n_samples):
            m = j[n, i]
            ui = u[n, i]

            # Clamp to endpoints
            if ui <= tau[0]:
                s[n, i] = all_q[m, n, 0]
                continue
            if ui >= tau[Q - 1]:
                s[n, i] = all_q[m, n, Q - 1]
                continue

            # Binary search: find lo such that tau[lo] <= ui < tau[lo+1]
            lo = 0
            hi = Q - 1
            while hi - lo > 1:
                mid = (lo + hi) >> 1
                if tau[mid] <= ui:
                    lo = mid
                else:
                    hi = mid

            # Linear interpolation
            t = (ui - tau[lo]) / (tau[hi] - tau[lo])
            s[n, i] = all_q[m, n, lo] + t * (all_q[m, n, hi] - all_q[m, n, lo])


@njit(cache=True, parallel=True)
def _extract_quantiles(s_sorted, tau, out):
    """Extract quantiles from pre-sorted samples via linear interpolation.

    Replaces np.percentile for sorted arrays. For each row n,
    maps tau levels to fractional indices in s_sorted[n] and
    linearly interpolates.

    Args:
        s_sorted: Sorted samples, shape (N, n_samples), dtype float64.
        tau: Quantile levels in (0, 1), shape (Q,), dtype float64.
        out: Output array, shape (N, Q), dtype float64. Modified in-place.
    """
    N, S = s_sorted.shape
    Q = tau.shape[0]
    for n in prange(N):
        for q in range(Q):
            # Map tau to fractional index in [0, S-1]
            idx_f = tau[q] * (S - 1)
            lo = int(idx_f)
            if lo >= S - 1:
                out[n, q] = s_sorted[n, S - 1]
            else:
                frac = idx_f - lo
                out[n, q] = s_sorted[n, lo] + frac * (s_sorted[n, lo + 1] - s_sorted[n, lo])


# =====================================================================
# Sampling-based combining (Taylor & Meng 2025, Appendix C)
# =====================================================================


def _sampling_combine(
    weights: np.ndarray,
    tau: np.ndarray,
    quantile_arrays: List[np.ndarray],
    degree_deg: float = 90.0,
    n_samples: int = 5000,
    rng_seed: int = 42,
) -> np.ndarray:
    """Combine quantile arrays via mixture sampling.

    Implements the simulation algorithm from Taylor & Meng (2025,
    Online Appendix C) for angular combining, with vertical (θ=90°)
    and horizontal (θ=0°) as special cases.

    For θ > 89° (vertical): pure mixture sampling, no transform.
    For 1° ≤ θ ≤ 89° (angular): h_θ shift → mixture sample → sort → h_{-θ}.
    For θ < 1° (horizontal): quantile function weighted average.

    Args:
        weights: Model weights, shape (M,), summing to 1.
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q), one per model.
        degree_deg: Angle in degrees [0, 90]. 90 = vertical, 0 = horizontal.
        n_samples: Number of Monte Carlo samples per timestep.
        rng_seed: Random seed for reproducibility.

    Returns:
        Combined quantile values, shape (N, Q).

    Example:
        >>> combined = _sampling_combine(
        ...     np.array([0.6, 0.4]), tau, [q1, q2],
        ...     degree_deg=90.0, n_samples=5000,
        ... )
    """
    N, Q = quantile_arrays[0].shape
    M = len(quantile_arrays)

    # θ < 1°: horizontal — simple weighted average (no sampling needed)
    if degree_deg < HORIZ_THRESH:
        result = np.zeros((N, Q), dtype=float)
        for w, qa in zip(weights, quantile_arrays):
            result += w * qa
        return result

    rng = np.random.default_rng(rng_seed)

    # ① Model selection: j(i) ~ Categorical(w)
    j = rng.choice(M, size=(N, n_samples), p=weights)

    # ② Uniform draws for quantile inversion
    u = rng.uniform(0, 1, size=(N, n_samples))

    # ③ Quantile interpolation: F_{j(i)}⁻¹(u(i))  [Numba JIT]
    all_q = np.ascontiguousarray(np.stack(quantile_arrays))  # (M, N, Q)
    s = np.empty((N, n_samples), dtype=np.float64)
    _interp_mixture_kernel(j, u, tau, all_q, s)

    # 1° ≤ θ ≤ 89°: h_θ shift
    apply_transform = degree_deg <= VERT_THRESH
    if apply_transform:
        tan_theta = np.tan(np.deg2rad(degree_deg))
        s += u / tan_theta

    # ④ Sort
    s.sort(axis=1)

    # ⑤ 1° ≤ θ ≤ 89°: h_{-θ} inverse transform
    if apply_transform:
        ranks = np.arange(1, n_samples + 1) / n_samples
        s -= ranks[np.newaxis, :] / tan_theta

    # ⑥ Extract output quantiles  [Numba JIT]
    out = np.empty((N, Q), dtype=np.float64)
    _extract_quantiles(s, tau, out)
    return out


def _sampling_combine_crn(
    weights: np.ndarray,
    tau: np.ndarray,
    quantile_arrays: List[np.ndarray],
    degree_deg: float,
    fixed_u: np.ndarray,
    fixed_v: np.ndarray,
) -> np.ndarray:
    """Sampling combine with Common Random Numbers for optimization.

    Uses pre-drawn uniform variates so that the objective is
    deterministic w.r.t. weights, enabling gradient-based optimizers.

    Args:
        weights: Model weights, shape (M,), summing to 1.
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).
        degree_deg: Angle in degrees [0, 90].
        fixed_u: Pre-drawn U(0,1) for quantile inversion, shape (N, S).
        fixed_v: Pre-drawn U(0,1) for model selection, shape (N, S).

    Returns:
        Combined quantile values, shape (N, Q).

    Example:
        >>> rng = np.random.default_rng(42)
        >>> fixed_u = rng.uniform(0, 1, size=(N, 5000))
        >>> fixed_v = rng.uniform(0, 1, size=(N, 5000))
        >>> combined = _sampling_combine_crn(
        ...     weights, tau, qarrays, 90.0, fixed_u, fixed_v
        ... )
    """
    N, Q = quantile_arrays[0].shape
    M = len(quantile_arrays)
    n_samples = fixed_u.shape[1]

    # θ < 1°: horizontal
    if degree_deg < HORIZ_THRESH:
        result = np.zeros((N, Q), dtype=float)
        for w, qa in zip(weights, quantile_arrays):
            result += w * qa
        return result

    # ① Deterministic model selection from fixed_v and current weights
    cum_w = np.cumsum(weights)
    cum_w[-1] = 1.0  # ensure no floating-point overshoot
    j = np.searchsorted(cum_w, fixed_v)  # (N, n_samples)
    j = np.clip(j, 0, M - 1)

    # ③ Quantile interpolation  [Numba JIT]
    all_q = np.ascontiguousarray(np.stack(quantile_arrays))  # (M, N, Q)
    s = np.empty((N, n_samples), dtype=np.float64)
    _interp_mixture_kernel(j, fixed_u, tau, all_q, s)

    # h_θ shift (1° ≤ θ ≤ 89°)
    apply_transform = degree_deg <= VERT_THRESH
    if apply_transform:
        tan_theta = np.tan(np.deg2rad(degree_deg))
        s += fixed_u / tan_theta

    # ④ Sort
    s.sort(axis=1)

    # ⑤ h_{-θ} inverse transform
    if apply_transform:
        ranks = np.arange(1, n_samples + 1) / n_samples
        s -= ranks[np.newaxis, :] / tan_theta

    # ⑥ Extract output quantiles
    return np.percentile(s, tau * 100, axis=1).T


def _sampling_objective(
    w, tau, quantile_arrays, observed, degree_deg,
    fixed_u, fixed_v, reg_lambda,
):
    """CRPS objective using sampling-based combine with CRN.

    Args:
        w: Weight vector, shape (M,).
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).
        observed: Observed values, shape (N,).
        degree_deg: Angle in degrees [0, 90].
        fixed_u: Pre-drawn U(0,1), shape (N, S).
        fixed_v: Pre-drawn U(0,1), shape (N, S).
        reg_lambda: L2 regularization strength toward equal weights.

    Returns:
        float: CRPS + regularization penalty.

    Example:
        >>> loss = _sampling_objective(
        ...     w, tau, qarrays, obs, 90.0, fixed_u, fixed_v, 0.0
        ... )
    """
    q_combined = _sampling_combine_crn(
        w, tau, quantile_arrays, degree_deg, fixed_u, fixed_v
    )
    loss = crps_quantile(tau, q_combined, observed, reduction="mean")
    if reg_lambda > 0:
        M = len(w)
        loss += reg_lambda * np.sum((w - 1.0 / M) ** 2)
    return loss


# =====================================================================
# Grid-based combining (legacy, retained for reference/testing)
# =====================================================================


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


def _batch_cdf_eval_grid(x_grid, qvals, tau):
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


def _batch_cdf_invert_grid(tau, cdf, x_grid):
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


def _vertical_combine_quantiles_grid(w, tau, quantile_arrays):
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
        combined_cdf += w[i] * _batch_cdf_eval_grid(x_grid, all_q[i], tau)

    # Invert combined CDF to get output quantiles
    return _batch_cdf_invert_grid(tau, combined_cdf, x_grid)


def _vertical_objective_grid(w, tau, quantile_arrays, observed, reg_lambda):
    """CRPS objective for vertical combining using grid-based CDF averaging.

    Args:
        w: Weight vector, shape (M,).
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).
        observed: Observed values, shape (N,).
        reg_lambda: L2 regularization strength toward equal weights.

    Returns:
        float: CRPS + regularization penalty.

    Example:
        >>> loss = _vertical_objective_grid(
        ...     np.array([0.6, 0.4]), tau, qarrays, obs, 0.0
        ... )
    """
    q_combined = _vertical_combine_quantiles_grid(w, tau, quantile_arrays)
    loss = crps_quantile(tau, q_combined, observed, reduction="mean")
    if reg_lambda > 0:
        M = len(w)
        loss += reg_lambda * np.sum((w - 1.0 / M) ** 2)
    return loss


class VerticalCombiner(BaseCombiner):
    """CDF weighted average (Linear Pool) via mixture sampling.

    F_combined(x) = sum_i w_i * F_i(x)

    Averages CDFs at the same x value. The combined distribution is
    generally wider than the individual distributions, which helps
    when individual models are under-dispersed.

    Implementation uses the simulation algorithm from Taylor & Meng
    (2025, Online Appendix C): vertical combining is a mixture
    distribution, so it is implemented via categorical model
    selection + quantile inversion + sorting.

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
        n_samples: Number of Monte Carlo samples for mixture
            sampling (default 5000).

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
        n_samples: int = 5000,
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
        self.n_samples = n_samples

    def _fit_horizon(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
        observed: np.ndarray,
    ) -> np.ndarray:
        """Learn combining weights for a single horizon.

        Uses sampling-based CRPS objective with Common Random Numbers
        (CRN) for deterministic optimization. Dispatches to
        inverse-CRPS heuristic or SLSQP based on ``fit_method``.

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

        # Pre-draw CRN variates for deterministic objective
        N = quantile_arrays[0].shape[0]
        rng = np.random.default_rng(42 + h)
        fixed_u = rng.uniform(0, 1, size=(N, self.n_samples))
        fixed_v = rng.uniform(0, 1, size=(N, self.n_samples))

        M = len(quantile_arrays)

        result = minimize(
            _sampling_objective,
            x0,
            args=(
                tau, quantile_arrays, observed, 90.0,
                fixed_u, fixed_v, self.reg_lambda,
            ),
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
        """CDF weighted average via mixture sampling.

        Uses the simulation algorithm from Taylor & Meng (2025):
        vertical combining is a mixture distribution, so we sample
        from the mixture, sort, and extract quantiles.

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
        return _sampling_combine(
            weights, self.quantile_levels, quantile_arrays,
            degree_deg=90.0, n_samples=self.n_samples,
        )
