"""
ARIMA-GARCH forecaster.

Implements ARIMA(p,d,q)-GARCH(gp,gq) with optional exogenous variables.
Parameters are estimated jointly via MLE (scipy.optimize.minimize).
Volatility forecasts come directly from the GARCH component.

Mathematical model:
    z_t = (1-B)^d y_t                                          (differencing)
    z_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p}                  (AR)
        + theta_1*eps_{t-1} + ... + theta_q*eps_{t-q}           (MA)
        + beta'*x_t + eps_t                                     (exog + innovation)
    eps_t = sigma_t * eta_t,  eta_t ~ iid D(0,1)
    sigma^2_t = omega + sum(alpha_i * eps^2_{t-i})              (ARCH)
              + sum(beta_j * sigma^2_{t-j})                     (GARCH)

    where D is Normal or Student-t (configurable via distribution parameter).

Inherits from GarchBase -- shared fit/forecast/update_state/simulate_paths pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np

from src.core.registry import MODEL_REGISTRY
from src.models.statistical._garch_base import GarchBase
from src.models.statistical._primitives import ARMA, GARCH


@MODEL_REGISTRY.register_model(name="arima_garch")
class ArimaGarchForecaster(GarchBase):
    """
    ARIMA(p,d,q)-GARCH(gp,gq) probabilistic forecaster.

    After fit(), use:
      - forecast(horizon)           -> (mu, sigma) arrays from current state
      - simulate_paths(n, horizon)  -> n stochastic paths from current state
      - update_state(actual_z, x_new) -> advance state by one observed value

    For rolling evaluation over a forecast period, use RollingRunner
    which calls these methods in sequence.

    Example:
        >>> model = ArimaGarchForecaster(hyperparameter={
        ...     "arima_order": (2, 1, 1), "garch_order": (1, 1)
        ... })
        >>> model.fit(dataset=train_df, y_col="power", exog_cols=["wind_speed"])
        >>> result = model.forecast(horizon=24)
    """

    def __init__(
        self,
        hyperparameter: Optional[Dict] = None,
        model_name: Optional[str] = None,
    ):
        hp = dict(hyperparameter) if hyperparameter else {}

        self._arima_order = tuple(hp.get("arima_order", (1, 0, 1)))
        self._garch_order = tuple(hp.get("garch_order", (1, 1)))
        self._distribution = hp.get("distribution", "normal")
        self._opt_method = hp.get("opt_method", "trust-constr")
        self._variance_targeting = hp.get("variance_targeting", True)

        self._diff_order: int = self._arima_order[1]

        # Undifferencing anchors (populated by fit)
        self._y_tail: np.ndarray = np.array([])

        super().__init__(
            hyperparameter=hyperparameter,
            model_name=model_name,
        )

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def _init_mean_primitive(self, n_exog: int) -> None:
        p, _, q = self._arima_order
        self._mean_prim = ARMA(n_exog=n_exog, order=(p, q))

    def _get_pq(self) -> Tuple[int, int]:
        p, _, q = self._arima_order
        return (p, q)

    def _get_diff_loss(self) -> int:
        return self._diff_order

    def _apply_differencing(
        self, y: np.ndarray, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = self._diff_order
        y_diff = self._difference(y, d)
        x_data = X[d:] if d > 0 else X.copy()
        return y_diff, x_data

    def _init_extra_params(self, y_diff: np.ndarray) -> list:
        return []

    def _get_extra_bounds(self) -> list:
        return []

    def _get_extra_parscale(self) -> list:
        return []

    def _get_mean_skip(self) -> int:
        p, _, q = self._arima_order
        return max(p, q)

    def _create_temp_mean_primitive(self, p: int, q: int, n_exog: int):
        return ARMA(n_exog=n_exog, order=(p, q))

    def _assign_temp_mean_params(
        self, tmp, params: np.ndarray,
        p: int, q: int, n_exog: int, n_extra: int,
    ) -> None:
        # n_extra == 0 for ARIMA
        tmp._phi[:] = params[:p]
        tmp._theta[:] = params[p:p + q]
        if n_exog > 0:
            tmp._x_coeff[:] = params[p + q:p + q + n_exog]

    def _store_mean_coefficients(
        self, params: np.ndarray,
        p: int, q: int, n_exog: int, n_extra: int,
    ) -> None:
        self._mean_prim.arma_coefficients = {
            "AR": params[:p],
            "MA": params[p:p + q],
            "X":  params[p + q:p + q + n_exog],
        }

    def _compute_undiff_state(self, y: np.ndarray) -> None:
        self._y_tail = self._compute_undiff_anchors(y, self._diff_order)

    def _undifference_mean(self, z_forecast: np.ndarray) -> np.ndarray:
        return self._undifference(z_forecast, self._y_tail, self._diff_order)

    def _undifference_sigma(self, sigma_diff: np.ndarray) -> np.ndarray:
        return self._undifference_sigma_int(sigma_diff, self._diff_order)

    def _difference_single(self, y_new: float) -> float:
        d = self._diff_order
        if d == 0:
            return y_new

        anchors = [self._y_tail[d - 1 - k] for k in range(d)]
        new_anchors = [0.0] * d
        val = y_new
        for k in range(d):
            new_anchors[k] = val
            val = val - anchors[k]
        self._y_tail = np.array([new_anchors[d - 1 - i] for i in range(d)])
        return val

    def _has_differencing(self) -> bool:
        return self._diff_order > 0

    def _get_state_dict(self) -> dict:
        return {
            "arma_coefficients": self._mean_prim.arma_coefficients,
            "y_tail":            self._y_tail,
        }

    def _load_state_dict(self, state: dict) -> None:
        self._mean_prim.arma_coefficients = state["arma_coefficients"]
        self._y_tail = state.get("y_tail", np.array([]))

    # ------------------------------------------------------------------
    # Static differencing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _difference(y: np.ndarray, d: int) -> np.ndarray:
        """Apply d-th order differencing."""
        z = y.copy()
        for _ in range(d):
            z = np.diff(z)
        return z

    @staticmethod
    def _undifference(z_forecast: np.ndarray, y_tail: np.ndarray, d: int) -> np.ndarray:
        """
        Inverse-difference a forecast sequence back to the original scale.

        Args:
            z_forecast: forecasted differenced values, shape (H,)
            y_tail:     anchors for each differencing level, length d.
            d:          differencing order

        Returns:
            y_forecast: forecasted values in original scale, shape (H,)
        """
        if d == 0:
            return z_forecast.copy()

        result = z_forecast.copy()
        for i in range(d):
            anchor = y_tail[i] if i < len(y_tail) else 0.0
            result = anchor + np.cumsum(result)

        return result

    @staticmethod
    def _undifference_sigma_int(sigma_diff: np.ndarray, d: int) -> np.ndarray:
        """
        Propagate conditional std through inverse-differencing (integer d).

        For d-th order differencing, the forecast error at horizon h is the
        sum of d nested cumulative sums of the per-step innovations.
        Under a first-order independence approximation:
            Var_original = cumsum^d(sigma_diff^2)

        Args:
            sigma_diff: conditional std in differenced scale, shape (H,)
            d:          differencing order

        Returns:
            sigma_original: std in original scale, shape (H,)
        """
        if d == 0:
            return sigma_diff.copy()
        var = sigma_diff ** 2
        for _ in range(d):
            var = np.cumsum(var)
        return np.sqrt(var)

    @staticmethod
    def _compute_undiff_anchors(y: np.ndarray, d: int) -> np.ndarray:
        """
        Compute anchors needed for undifferencing from an original-scale series.

        Returns array of length d where entry i is the last value at
        differencing level (d-1-i).  For d=1: [y[-1]].
        For d=2: [diff(y)[-1], y[-1]].
        """
        if d == 0:
            return np.array([])
        levels = [y]
        for _ in range(d - 1):
            levels.append(np.diff(levels[-1]))
        return np.array([levels[d - 1 - i][-1] for i in range(d)])
