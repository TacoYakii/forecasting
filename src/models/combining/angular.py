"""Angular combining: quantile-transform interpolation between horizontal and vertical.

Implements the angular averaging method from Taylor & Meng (2025),
"Angular Combining of Forecasts of Probability Distributions",
Management Science. DOI: 10.1287/mnsc.2024.05558

Angular combining smoothly interpolates between horizontal (quantile
function averaging, theta=0) and vertical (CDF averaging, theta=90).

Implementation uses the generalized linear combination framework (Eq. 7):
    h_θ(F_{A,w,θ})(c) = Σ w_i · h_θ(F_i)(c)

where the link function h_θ has quantile inverse (Eq. 6):
    h_θ^{-1}(α) = F^{-1}(α) + α / tan(θ)

This reduces angular combining to: shift quantiles → vertical combine → unshift.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize

from src.utils.metrics import crps_quantile

from .base import BaseCombiner
from .vertical import _vertical_combine_quantiles

EPS_DEG = 1e-2


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
) -> np.ndarray:
    """Combine quantile arrays via angular quantile-transform averaging.

    Uses the generalized linear combination framework (Taylor & Meng,
    2025, Eq. 7): angular combining equals vertical combining of the
    h_θ-transformed CDFs, followed by inverse transformation.

    The link function h_θ has quantile inverse:
        h_θ^{-1}(α) = F^{-1}(α) + α / tan(θ)

    So the algorithm is:
        1. Shift each model's quantiles by α / tan(θ)
        2. Vertically combine the shifted quantiles
        3. Subtract α / tan(θ) from the result

    Args:
        weights: Model weights, shape (M,), summing to 1.
        degree_deg: Angle in degrees [0, 90].
        tau: Quantile levels, shape (Q,).
        quantile_arrays: M arrays of shape (N, Q).

    Returns:
        Combined quantile values, shape (N, Q).

    Example:
        >>> combined = _angular_combine_quantiles(
        ...     np.array([0.6, 0.4]), 45.0, tau, qarrays
        ... )
    """
    # Endpoint: horizontal (quantile function weighted average)
    if degree_deg < EPS_DEG:
        result = np.zeros_like(quantile_arrays[0], dtype=float)
        for w, qa in zip(weights, quantile_arrays):
            result += w * qa
        return result

    # Endpoint: vertical (CDF weighted average)
    if degree_deg > 90.0 - EPS_DEG:
        return _vertical_combine_quantiles(weights, tau, quantile_arrays)

    # Angular combining via h_θ transform → vertical → h_{-θ}
    tan_theta = np.tan(np.deg2rad(degree_deg))
    shift = tau / tan_theta  # shape (Q,)

    # Step 1: h_θ transform — shift quantiles by α/tan(θ)
    transformed = [qa + shift[np.newaxis, :] for qa in quantile_arrays]

    # Step 2: Vertical combine the transformed quantiles
    combined_transformed = _vertical_combine_quantiles(
        weights, tau, transformed
    )

    # Step 3: Inverse transform h_{-θ} — subtract α/tan(θ)
    return combined_transformed - shift[np.newaxis, :]


# ---------------------------------------------------------------------------
# AngularCombiner
# ---------------------------------------------------------------------------


class AngularCombiner(BaseCombiner):
    """Angular combining via quantile-transform interpolation.

    Smoothly interpolates between horizontal (quantile function averaging)
    at theta=0 and vertical (CDF averaging) at theta=90 using the
    angular quantile transform.

    When fit_method="cobyla", simultaneously optimizes model weights and
    the degree parameter by minimizing CRPS via the COBYLA constrained
    optimizer.

    Args:
        n_quantiles: Number of quantile levels (default 99).
        n_jobs: Parallel workers for degree grid search in
            grid_slsqp (default 1). Set to -1 for all cores.
        weights: User-specified weights, shape (M,). If None,
            learns weights during fit().
        beta: Power exponent for inverse-CRPS weighting.
            Only used with fit_method="inverse_crps". Default 1.0.
        degree: Angle in degrees [0, 90]. If None, optimized
            during fit() when fit_method="cobyla".
        fit_method: Weight/degree learning strategy.
            - "grid_slsqp": Degree grid search (0°-90°, 1° step) with
              SLSQP convex weight optimization at each degree (default).
            - "cobyla": Simultaneous optimization via COBYLA.
            - "inverse_crps": Heuristic weights. If degree is given,
              uses that fixed value; if degree is None, runs a grid
              search (0°-90°, 1° step) to find the best degree.
        reg_lambda: L2 regularization toward equal weights (default 0.0).
        val_ratio: Fraction of data for temporal validation (default 0.0).

    Attributes:
        params_: Dict mapping horizon (1-indexed) to
            {"degree": float} (and optionally "beta": float).

    Example:
        >>> combiner = AngularCombiner(n_quantiles=99, fit_method="grid_slsqp")
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
        fit_method: str = "grid_slsqp",
        reg_lambda: float = 0.0,
        val_ratio: float = 0.0,
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
                        w, deg, tau, quantile_arrays
                    )
                    loss = crps_quantile(
                        tau, q_comb, observed, reduction="mean"
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_deg = deg
                self.params_[h] = {"beta": beta, "degree": best_deg}
            return w

        # --- grid_slsqp: degree grid search + convex weight optimization ---
        if self.fit_method == "grid_slsqp":
            return self._fit_grid_slsqp(
                h, tau, M, x0_w, quantile_arrays, observed
            )

        # --- COBYLA: simultaneous optimization ---
        x0_deg = (
            self._user_degree if self._user_degree is not None else 45.0
        )
        optimize_degree = self._user_degree is None

        if optimize_degree:
            # Optimize [w_0, ..., w_{M-2}, degree]
            x0 = np.append(x0_w[:M - 1], x0_deg)
        else:
            # Optimize weights only: [w_0, ..., w_{M-2}]
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

            deg = float(np.clip(deg, EPS_DEG, 90.0 - EPS_DEG))
            q_combined = _angular_combine_quantiles(
                w, deg, tau, quantile_arrays
            )
            loss = crps_quantile(
                tau, q_combined, observed, reduction="mean"
            )
            if self.reg_lambda > 0:
                loss += self.reg_lambda * np.sum((w - 1.0 / M) ** 2)
            return loss

        # Constraints: w_i >= 0, w_M >= 0, degree in [EPS, 90-EPS]
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
                    "fun": lambda x, k=M - 1: x[k] - EPS_DEG,
                },
                {
                    "type": "ineq",
                    "fun": lambda x, k=M - 1: (90.0 - EPS_DEG) - x[k],
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

        # Extract optimized parameters
        w_free = result.x[:M - 1]
        w_last = 1.0 - w_free.sum()
        w_opt = np.maximum(np.append(w_free, w_last), 0.0)
        if w_opt.sum() < 1e-12:
            self.params_[h] = {"degree": x0_deg}
            return x0_w
        w_opt /= w_opt.sum()

        if optimize_degree:
            deg_opt = float(
                np.clip(result.x[-1], EPS_DEG, 90.0 - EPS_DEG)
            )
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
    ) -> np.ndarray:
        """Optimize via degree grid search + SLSQP weight optimization.

        For each candidate degree, CRPS is convex w.r.t. weights, so
        SLSQP finds the global optimum. The best (degree, weights)
        pair across the grid is returned.

        Degree evaluations are parallelized via ``n_jobs``.

        Args:
            h: Forecast horizon (1-indexed).
            tau: Quantile levels, shape (Q,).
            M: Number of models.
            x0_w: Inverse-CRPS warm start weights, shape (M,).
            quantile_arrays: M arrays of shape (N_train, Q).
            observed: Observed values, shape (N_train,).

        Returns:
            np.ndarray: Optimized weights, shape (M,).

        Example:
            >>> w = combiner._fit_grid_slsqp(1, tau, 3, x0, qa, obs)
        """
        degree_grid = np.arange(0.0, 91.0, 1.0)
        if self._user_degree is not None:
            degree_grid = np.array([self._user_degree])

        def _solve_one_degree(deg):
            def objective(w, _deg=deg):
                q_combined = _angular_combine_quantiles(
                    w, _deg, tau, quantile_arrays
                )
                loss = crps_quantile(
                    tau, q_combined, observed, reduction="mean"
                )
                if self.reg_lambda > 0:
                    loss += self.reg_lambda * np.sum((w - 1.0 / M) ** 2)
                return loss

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
        """Combine quantile arrays via angular transform for one horizon.

        Dispatches to horizontal, vertical, or angular based on the
        fitted degree parameter.

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
        return _angular_combine_quantiles(
            weights, degree, self.quantile_levels, quantile_arrays
        )
