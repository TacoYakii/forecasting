"""
SARIMA-GARCH forecaster.

Implements SARIMA(p,d,q)(P,D,Q,s)-GARCH(gp,gq) with optional exogenous variables.
Extends ARIMA-GARCH with seasonal AR (SAR) and seasonal MA (SMA) components.

Mathematical model (after d regular + D seasonal differencing):
    z_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p}                  (AR)
        + theta_1*eps_{t-1} + ... + theta_q*eps_{t-q}           (MA)
        + Phi_1*z_{t-s} + ... + Phi_P*z_{t-P*s}                (SAR)
        + Theta_1*eps_{t-s} + ... + Theta_Q*eps_{t-Q*s}         (SMA)
        + beta'*x_t + eps_t                                     (exog + innovation)
    eps_t = sigma_t * eta_t,  eta_t ~ iid D(0,1)
    sigma^2_t = omega + sum(alpha_i * eps^2_{t-i})              (ARCH)
              + sum(beta_j * sigma^2_{t-j})                     (GARCH)

    where D is Normal or Student-t (configurable).

Inherits from GarchBase — shared fit/forecast/update_state/simulate_paths pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Iterable

import numpy as np
import pandas as pd

from src.models.statistical._garch_base import GarchBase
from src.models.statistical._primitives import SARIMA, GARCH
from src.models.statistical.config import SarimaGarchConfig


class SarimaGarchForecaster(GarchBase):
    """
    SARIMA(p,d,q)(P,D,Q,s)-GARCH(gp,gq) probabilistic forecaster.

    After fit(), use:
      - forecast(horizon)           → (mu, sigma) arrays from current state
      - simulate_paths(n, horizon)  → n stochastic paths from current state
      - update_state(actual_z, x_t) → advance state by one observed value

    For rolling evaluation over a forecast period, use RollingForecaster
    which calls these methods in sequence.

    Example:
        >>> config = SarimaGarchConfig(
        ...     arima_order=(1, 0, 1), seasonal_order=(1, 0, 1, 24),
        ...     garch_order=(1, 1))
        >>> model = SarimaGarchForecaster(dataset=train_df, y_col="power",
        ...                     config=config)
        >>> model.fit()
        >>> mu, sigma = model.forecast(horizon=24)
        >>> mu.shape   # (24,)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        x_cols: Optional[Union[str, int, Iterable]] = None,
        config: Optional[SarimaGarchConfig] = None,
        enable_logging: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config = config or SarimaGarchConfig()

        self._p, self._d, self._q = self.config.arima_order
        self._P, self._D, self._Q, self._s = self.config.seasonal_order

        # Undifferencing anchors (populated by fit)
        self._regular_tail: np.ndarray = np.array([])   # length d
        self._seasonal_tails: list = []                  # D arrays, each length s

        super().__init__(
            dataset=dataset,
            y_col=y_col,
            x_cols=x_cols,
            config=self.config,
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def _build_config_hyperparameter(self) -> dict:
        return {
            "arima_order":    self.config.arima_order,
            "seasonal_order": self.config.seasonal_order,
            "garch_order":    self.config.garch_order,
            "opt_method":     self.config.opt_method,
            "distribution":   self.config.distribution,
        }

    def _init_mean_primitive(self, n_exog: int) -> None:
        self._mean_prim = SARIMA(
            order=(self._p, self._q),
            seasonal_order=(self._P, self._Q, self._s),
            n_exog=n_exog,
        )

    def _get_pq(self) -> Tuple[int, int]:
        return (self._p, self._q)

    def _get_diff_loss(self) -> int:
        return self._d + self._D * self._s

    def _apply_differencing(
        self, y: np.ndarray, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_diff = self._difference(y)
        diff_loss = self._get_diff_loss()
        x_data = X[diff_loss:] if diff_loss > 0 else X.copy()
        return y_diff, x_data

    def _init_extra_params(self, y_diff: np.ndarray) -> list:
        return [0.05] * self._P + [0.05] * self._Q

    def _get_extra_bounds(self) -> list:
        return [(-1, 1)] * self._P + [(-1, 1)] * self._Q

    def _get_extra_parscale(self) -> list:
        return [1.0] * (self._P + self._Q)

    def _get_root_check_groups(self) -> list:
        # 4 independent polynomials for SARIMA.
        # Layout: [SAR_1..P | SMA_1..Q | AR_1..p | MA_1..q | X...]
        P, Q = self._P, self._Q
        p, q = self._p, self._q
        sar_idx = list(range(0, P))
        sma_idx = list(range(P, P + Q))
        ar_idx = list(range(P + Q, P + Q + p))
        ma_idx = list(range(P + Q + p, P + Q + p + q))
        return [sar_idx, sma_idx, ar_idx, ma_idx]

    def _get_mean_skip(self) -> int:
        p, q, s = self._p, self._q, self._s
        P, Q = self._P, self._Q
        return max(p, q, P * s, Q * s)

    def _create_temp_mean_primitive(self, p: int, q: int, n_exog: int):
        return SARIMA(
            order=(p, q),
            seasonal_order=(self._P, self._Q, self._s),
            n_exog=n_exog,
        )

    def _assign_temp_mean_params(
        self, tmp, params: np.ndarray,
        p: int, q: int, n_exog: int, n_extra: int,
    ) -> None:
        P, Q = self._P, self._Q
        # Extra params: [SAR_1..P, SMA_1..Q]
        tmp._Phi[:] = params[:P]
        tmp._Theta[:] = params[P:P + Q]
        # Standard AR/MA
        off = n_extra  # = P + Q
        tmp._phi[:] = params[off:off + p]
        tmp._theta[:] = params[off + p:off + p + q]
        if n_exog > 0:
            tmp._x_coeff[:] = params[off + p + q:off + p + q + n_exog]

    def _store_mean_coefficients(
        self, params: np.ndarray,
        p: int, q: int, n_exog: int, n_extra: int,
    ) -> None:
        P, Q = self._P, self._Q
        off = n_extra  # = P + Q
        self._mean_prim.coefficients = {
            "AR":  params[off:off + p],
            "MA":  params[off + p:off + p + q],
            "SAR": params[:P],
            "SMA": params[P:P + Q],
            "X":   params[off + p + q:off + p + q + n_exog],
        }

    def _compute_undiff_state(self, y: np.ndarray) -> None:
        d, D, s = self._d, self._D, self._s

        # Build intermediate levels through differencing
        level = y.copy()
        reg_levels = [level]
        for _ in range(d):
            level = np.diff(level)
            reg_levels.append(level)

        seas_levels = [reg_levels[d]]
        for _ in range(D):
            prev = seas_levels[-1]
            seas_levels.append(prev[s:] - prev[:-s])

        # _regular_tail[i] = last value at level (d-1-i) for regular undiff
        if d > 0:
            self._regular_tail = np.array(
                [reg_levels[d - 1 - i][-1] for i in range(d)]
            )
        else:
            self._regular_tail = np.array([])

        # _seasonal_tails[j] = last s values at seas_levels[D-1-j]
        self._seasonal_tails = [
            seas_levels[D - 1 - j][-s:].copy() for j in range(D)
        ]

    def _undifference_mean(self, z_forecast: np.ndarray) -> np.ndarray:
        result = z_forecast.copy()
        s = self._s

        # 1. Undo seasonal differencing (D levels)
        for j in range(self._D):
            anchor = self._seasonal_tails[j]
            undiffed = np.empty(len(result))
            for t in range(len(result)):
                if t < s:
                    undiffed[t] = anchor[t] + result[t]
                else:
                    undiffed[t] = undiffed[t - s] + result[t]
            result = undiffed

        # 2. Undo regular differencing (d levels)
        for i in range(self._d):
            anchor = self._regular_tail[i]
            result = anchor + np.cumsum(result)

        return result

    def _undifference_sigma(self, sigma_diff: np.ndarray) -> np.ndarray:
        var = sigma_diff ** 2
        s = self._s

        # 1. Undo seasonal diffs (variance accumulation with period s)
        for _ in range(self._D):
            var_new = np.empty_like(var)
            for t in range(len(var)):
                if t < s:
                    var_new[t] = var[t]
                else:
                    var_new[t] = var_new[t - s] + var[t]
            var = var_new

        # 2. Undo regular diffs (cumulative sum)
        for _ in range(self._d):
            var = np.cumsum(var)

        return np.sqrt(var)

    def _difference_single(self, y_new: float) -> float:
        d, D, s = self._d, self._D, self._s

        # 1. Regular differencing: compute through d levels
        val = y_new
        if d > 0:
            anchors = [self._regular_tail[d - 1 - k] for k in range(d)]
            new_anchors = [0.0] * d
            for k in range(d):
                new_anchors[k] = val
                val = val - anchors[k]
            self._regular_tail = np.array(
                [new_anchors[d - 1 - i] for i in range(d)]
            )

        # 2. Seasonal differencing: compute through D levels
        for j_undiff in range(D - 1, -1, -1):
            anchor = self._seasonal_tails[j_undiff]
            diff_val = val - anchor[0]
            self._seasonal_tails[j_undiff] = np.append(anchor[1:], val)
            val = diff_val

        return val

    def _has_differencing(self) -> bool:
        return self._d > 0 or self._D > 0

    def _get_state_dict(self) -> dict:
        return {
            "sarima_coefficients": self._mean_prim.coefficients,
            "regular_tail":        self._regular_tail,
            "seasonal_tails":      self._seasonal_tails,
        }

    def _load_state_dict(self, state: dict) -> None:
        self._mean_prim.coefficients = state["sarima_coefficients"]
        self._regular_tail = state.get("regular_tail", np.array([]))
        self._seasonal_tails = state.get("seasonal_tails", [])

    # ------------------------------------------------------------------
    # Differencing utility
    # ------------------------------------------------------------------

    def _difference(self, y: np.ndarray) -> np.ndarray:
        """Apply d regular + D seasonal differencing."""
        z = y.copy()
        for _ in range(self._d):
            z = np.diff(z)
        for _ in range(self._D):
            z = z[self._s:] - z[:-self._s]
        return z
