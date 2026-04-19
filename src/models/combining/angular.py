"""Angular combining: quantile-transform interpolation between horizontal and vertical.

Implements the angular averaging method from Taylor & Meng (2025),
"Angular Combining of Forecasts of Probability Distributions",
Management Science. DOI: 10.1287/mnsc.2024.05558

Angular combining smoothly interpolates between horizontal (quantile
function averaging, theta=0) and vertical (CDF averaging, theta=90).

Implementation uses the simulation algorithm from Appendix C of the
paper: angular combining = h_θ transform → vertical (mixture) sampling
→ h_{-θ} inverse transform.  This replaces the grid-based CDF
evaluation approach with O(n_samples log n_samples) mixture sampling.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from src.utils.metrics import crps_quantile

from .base import BaseCombiner
from .vertical import (
    HORIZ_THRESH,
    VERT_THRESH,
    _sampling_combine,
    _sampling_combine_crn,
    _sampling_objective,
    _vertical_combine_quantiles_grid,
)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _crps_power_weights(
    tau: np.ndarray,
    quantile_arrays: List[np.ndarray],
    observed: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Compute inverse-CRPS weights with power exponent.

    w_i = 1 / (CRPS_i ^ beta), normalized to sum to 1.
    When beta=1, identical to BaseCombiner._inverse_crps_weights.

    Args:
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).
        observed: Observed values, shape (N,).
        beta: Power exponent (higher → more aggressive weighting).

    Returns:
        np.ndarray: Normalized weights, shape (M,).

    Example:
        >>> w = _crps_power_weights(tau, qarrays, obs, beta=2.0)
        >>> w.sum()
        1.0
    """
    crps_scores = np.array([
        crps_quantile(tau, q, observed, reduction="mean")
        for q in quantile_arrays
    ])
    crps_scores = np.maximum(crps_scores, 1e-12)
    inv = 1.0 / (crps_scores ** beta)
    return inv / inv.sum()


def _angular_combine_quantiles(
    weights: np.ndarray,
    degree_deg: float,
    tau: np.ndarray,
    quantile_arrays: List[np.ndarray],
    n_samples: int = 5000,
    rng_seed: int = 42,
) -> np.ndarray:
    """Combine quantile arrays via angular mixture sampling.

    Uses the simulation algorithm from Taylor & Meng (2025, Appendix C).
    Dispatches to ``_sampling_combine`` which handles the θ-based
    branching (horizontal / angular / vertical).

    Args:
        weights: Model weights, shape (M,), summing to 1.
        degree_deg: Angle in degrees [0, 90].
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).
        n_samples: Number of Monte Carlo samples (default 5000).
        rng_seed: Random seed for reproducibility.

    Returns:
        Combined quantile values, shape (N, Q).

    Example:
        >>> combined = _angular_combine_quantiles(
        ...     np.array([0.6, 0.4]), 45.0, tau, qarrays
        ... )
    """
    return _sampling_combine(
        weights, tau, quantile_arrays,
        degree_deg=degree_deg, n_samples=n_samples, rng_seed=rng_seed,
    )


def _angular_combine_quantiles_grid(
    weights: np.ndarray,
    degree_deg: float,
    tau: np.ndarray,
    quantile_arrays: List[np.ndarray],
) -> np.ndarray:
    """Grid-based angular combining (legacy, retained for testing).

    Uses h_θ shift → grid-based vertical → h_{-θ} unshift.

    Args:
        weights: Model weights, shape (M,), summing to 1.
        degree_deg: Angle in degrees [0, 90].
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).

    Returns:
        Combined quantile values, shape (N, Q).

    Example:
        >>> combined = _angular_combine_quantiles_grid(
        ...     np.array([0.6, 0.4]), 45.0, tau, qarrays
        ... )
    """
    EPS_DEG = 1e-2

    # Endpoint: horizontal
    if degree_deg < EPS_DEG:
        result = np.zeros_like(quantile_arrays[0], dtype=float)
        for w, qa in zip(weights, quantile_arrays):
            result += w * qa
        return result

    # Endpoint: vertical
    if degree_deg > 90.0 - EPS_DEG:
        return _vertical_combine_quantiles_grid(weights, tau, quantile_arrays)

    # Angular: h_θ shift → grid vertical → h_{-θ}
    tan_theta = np.tan(np.deg2rad(degree_deg))
    shift = tau / tan_theta

    transformed = [qa + shift[np.newaxis, :] for qa in quantile_arrays]
    combined_transformed = _vertical_combine_quantiles_grid(
        weights, tau, transformed
    )
    return combined_transformed - shift[np.newaxis, :]


# ---------------------------------------------------------------------------
# AngularCombiner
# ---------------------------------------------------------------------------


class AngularCombiner(BaseCombiner):
    """Angular combining via mixture sampling.

    Smoothly interpolates between horizontal (quantile function averaging)
    at theta=0 and vertical (CDF averaging) at theta=90 using the
    simulation algorithm from Taylor & Meng (2025, Appendix C).

    Args:
        n_quantiles: Number of quantile levels (default 99).
        n_jobs: Parallel workers for degree grid search in
            grid_slsqp (default 1). Set to -1 for all cores.
        weights: User-specified weights, shape (M,). If None,
            learns weights during fit().
        beta: Power exponent for inverse-CRPS weighting.
            Only used with fit_method="inverse_crps". Default 1.0.
        degree: Angle in degrees [0, 90]. If None, optimized
            during fit().
        fit_method: Weight/degree learning strategy.
            - "cobyla": Simultaneous optimization of weights and
              degree via COBYLA (default). Recommended for speed.
            - "grid_slsqp": Degree grid search (0-90, 1 step) with
              SLSQP convex weight optimization at each degree.
            - "inverse_crps": Heuristic weights with degree grid search.
        reg_lambda: L2 regularization toward equal weights (default 0.0).
        val_ratio: Fraction of data for temporal validation (default 0.0).
        n_samples: Number of Monte Carlo samples (default 5000).

    Attributes:
        params_: Dict mapping horizon (1-indexed) to
            {"degree": float} (and optionally "beta": float).

    Example:
        >>> combiner = AngularCombiner(n_quantiles=99, fit_method="cobyla")
        >>> combiner.fit(train_results, observed)
        >>> combiner.params_[1]["degree"]  # optimized angle for h=1
        >>> combined = combiner.combine(test_results)
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        n_jobs: int = 1,
        weights: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
        degree: Optional[float] = None,
        fit_method: str = "cobyla",
        reg_lambda: float = 0.0,
        val_ratio: float = 0.0,
        n_samples: int = 5000,
    ):
        # Horizon loop is serial (params_ side-effect), but n_jobs
        # is used for degree-level parallelism in grid_slsqp.
        self._degree_n_jobs = n_jobs
        super().__init__(
            n_quantiles=n_quantiles, n_jobs=1, val_ratio=val_ratio
        )
        self._user_weights = (
            None if weights is None else np.asarray(weights, dtype=float)
        )
        self._user_beta = beta
        self._user_degree = degree

        if fit_method not in ("inverse_crps", "cobyla", "grid_slsqp"):
            raise ValueError(
                f"fit_method must be 'inverse_crps', 'cobyla', or "
                f"'grid_slsqp', got {fit_method!r}"
            )
        self.fit_method = fit_method
        self.reg_lambda = reg_lambda
        self.n_samples = n_samples
        self.params_: Dict[int, Dict[str, float]] = {}

    def _fit_horizon(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
        observed: np.ndarray,
    ) -> np.ndarray:
        """Learn weights (and optionally degree) for a single horizon.

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
        tau = self.quantile_levels
        M = len(quantile_arrays)

        # --- User-specified weights: skip weight optimization ---
        if self._user_weights is not None:
            w = self._validate_weights(self._user_weights, M)
            degree = (
                self._user_degree if self._user_degree is not None else 45.0
            )
            self.params_[h] = {"degree": degree}
            return w

        # --- inverse_crps heuristic ---
        x0_w = self._inverse_crps_weights(tau, quantile_arrays, observed)

        if self.fit_method == "inverse_crps":
            beta = self._user_beta if self._user_beta is not None else 1.0
            if beta != 1.0:
                w = _crps_power_weights(
                    tau, quantile_arrays, observed, beta
                )
            else:
                w = x0_w

            if self._user_degree is not None:
                self.params_[h] = {"beta": beta, "degree": self._user_degree}
            else:
                # Grid search: find optimal degree with fixed weights
                degree_grid = np.arange(0.0, 91.0, 1.0)
                best_deg, best_loss = 0.0, np.inf
                for deg in degree_grid:
                    q_comb = _angular_combine_quantiles(
                        w, deg, tau, quantile_arrays,
                        n_samples=self.n_samples,
                    )
                    loss = crps_quantile(
                        tau, q_comb, observed, reduction="mean"
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_deg = deg
                self.params_[h] = {"beta": beta, "degree": best_deg}
            return w

        # Pre-draw CRN variates
        N = quantile_arrays[0].shape[0]
        rng = np.random.default_rng(42 + h)
        fixed_u = rng.uniform(0, 1, size=(N, self.n_samples))
        fixed_v = rng.uniform(0, 1, size=(N, self.n_samples))

        # --- grid_slsqp: degree grid search + convex weight optimization ---
        if self.fit_method == "grid_slsqp":
            return self._fit_grid_slsqp(
                h, tau, M, x0_w, quantile_arrays, observed,
                fixed_u, fixed_v,
            )

        # --- COBYLA: simultaneous optimization ---
        x0_deg = (
            self._user_degree if self._user_degree is not None else 45.0
        )
        optimize_degree = self._user_degree is None

        if optimize_degree:
            x0 = np.append(x0_w[:M - 1], x0_deg)
        else:
            x0 = x0_w[:M - 1].copy()

        def objective(params):
            if optimize_degree:
                w_free = params[:M - 1]
                deg = float(params[-1])
            else:
                w_free = params[:M - 1]
                deg = x0_deg

            w_last = 1.0 - w_free.sum()
            if w_last < -1e-8 or np.any(w_free < -1e-8):
                return 1e10

            w = np.append(w_free, max(w_last, 0.0))
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s < 1e-12:
                return 1e10
            w /= s

            deg = float(np.clip(deg, HORIZ_THRESH, 90.0))
            return _sampling_objective(
                w, tau, quantile_arrays, observed, deg,
                fixed_u, fixed_v, self.reg_lambda,
            )

        # Constraints: w_i >= 0, w_M >= 0, degree in [1, 90]
        constraints = [
            *[
                {"type": "ineq", "fun": lambda x, i=i: x[i]}
                for i in range(M - 1)
            ],
            {
                "type": "ineq",
                "fun": lambda x, k=M - 1: 1.0 - x[:k].sum(),
            },
        ]
        if optimize_degree:
            constraints.extend([
                {
                    "type": "ineq",
                    "fun": lambda x, k=M - 1: x[k] - HORIZ_THRESH,
                },
                {
                    "type": "ineq",
                    "fun": lambda x, k=M - 1: 90.0 - x[k],
                },
            ])

        result = minimize(
            objective,
            x0,
            method="COBYLA",
            constraints=constraints,
            options={"maxiter": 2000, "rhobeg": 0.1},
        )

        if not result.success:
            warnings.warn(
                f"COBYLA did not converge for horizon {h}: "
                f"{result.message}. Falling back to inverse-CRPS "
                f"weights with degree={x0_deg:.1f}.",
                stacklevel=2,
            )
            self.params_[h] = {"degree": x0_deg}
            return x0_w

        w_free = result.x[:M - 1]
        w_last = 1.0 - w_free.sum()
        w_opt = np.maximum(np.append(w_free, w_last), 0.0)
        if w_opt.sum() < 1e-12:
            self.params_[h] = {"degree": x0_deg}
            return x0_w
        w_opt /= w_opt.sum()

        if optimize_degree:
            deg_opt = float(np.clip(result.x[-1], HORIZ_THRESH, 90.0))
        else:
            deg_opt = x0_deg

        self.params_[h] = {"degree": deg_opt}
        return w_opt

    def _fit_grid_slsqp(
        self,
        h: int,
        tau: np.ndarray,
        M: int,
        x0_w: np.ndarray,
        quantile_arrays: List[np.ndarray],
        observed: np.ndarray,
        fixed_u: np.ndarray,
        fixed_v: np.ndarray,
    ) -> np.ndarray:
        """Optimize via degree grid search + SLSQP weight optimization.

        For each candidate degree, CRPS is convex w.r.t. weights, so
        SLSQP finds the global optimum. The best (degree, weights)
        pair across the grid is returned.

        Args:
            h: Forecast horizon (1-indexed).
            tau: Quantile levels, shape (Q,).
            M: Number of models.
            x0_w: Inverse-CRPS warm start weights, shape (M,).
            quantile_arrays: M arrays of shape (N_train, Q).
            observed: Observed values, shape (N_train,).
            fixed_u: CRN uniform variates, shape (N, n_samples).
            fixed_v: CRN uniform variates, shape (N, n_samples).

        Returns:
            np.ndarray: Optimized weights, shape (M,).

        Example:
            >>> w = combiner._fit_grid_slsqp(
            ...     1, tau, 3, x0, qa, obs, fixed_u, fixed_v
            ... )
        """
        degree_grid = np.arange(0.0, 91.0, 1.0)
        if self._user_degree is not None:
            degree_grid = np.array([self._user_degree])

        def _solve_one_degree(deg):
            def objective(w, _deg=deg):
                return _sampling_objective(
                    w, tau, quantile_arrays, observed, _deg,
                    fixed_u, fixed_v, self.reg_lambda,
                )

            result = minimize(
                objective,
                x0_w,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * M,
                constraints=[{
                    "type": "eq", "fun": lambda w: w.sum() - 1.0
                }],
                options={"maxiter": 200, "ftol": 1e-10},
            )

            if result.success:
                w_opt = np.maximum(result.x, 0.0)
                s = w_opt.sum()
                if s > 1e-12:
                    w_opt /= s
                return deg, result.fun, w_opt
            return deg, objective(x0_w), x0_w

        if self._degree_n_jobs == 1:
            results = [_solve_one_degree(deg) for deg in degree_grid]
        else:
            results = Parallel(n_jobs=self._degree_n_jobs)(
                delayed(_solve_one_degree)(deg) for deg in degree_grid
            )

        best_deg, best_loss, best_w = min(results, key=lambda x: x[1])
        self.params_[h] = {"degree": best_deg}
        return best_w

    def _combine_distributions(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
    ) -> np.ndarray:
        """Combine quantile arrays via angular mixture sampling.

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
        weights = self.weights_[h]
        degree = self.params_[h]["degree"]
        return _sampling_combine(
            weights, self.quantile_levels, quantile_arrays,
            degree_deg=degree, n_samples=self.n_samples,
        )
