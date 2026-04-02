"""
ARFIMA-GARCH forecaster.

Implements ARFIMA(p,d,q)-GARCH(gp,gq) with optional exogenous variables.
The fractional differencing parameter d is estimated jointly via MLE,
enabling long-memory modelling (0 < d < 0.5).

Mathematical model:
    z_t = (1-B)^d y_t                                          (fractional differencing)
    z_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p}                  (AR)
        + theta_1*eps_{t-1} + ... + theta_q*eps_{t-q}           (MA)
        + beta'*x_t + eps_t                                     (exog + innovation)
    eps_t = sigma_t * eta_t,  eta_t ~ iid D(0,1)
    sigma^2_t = omega + sum(alpha_i * eps^2_{t-i})              (ARCH)
              + sum(beta_j * sigma^2_{t-j})                     (GARCH)

    where d in (-0.5, 0.5) and D is Normal or Student-t (configurable).

Key difference from ARIMA-GARCH:
  - d is a continuous MLE parameter, not a fixed integer from config
  - Fractional differencing preserves all observations (no diff_loss)
  - update_state requires full original-scale history (_y_history)

Inherits from GarchBase — shared fit/forecast/update_state/simulate_paths pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Iterable

import numpy as np
import pandas as pd

from src.core.registry import MODEL_REGISTRY
from src.models.statistical._garch_base import GarchBase
from src.models.statistical._primitives import (
    ARMA, GARCH,
    fractional_diff, fractional_diff_weights, fractional_undiff,
)
from src.models.statistical.config import ArfimaGarchConfig


@MODEL_REGISTRY.register_model(name="arfima_garch")
class ArfimaGarchForecaster(GarchBase):
    """
    ARFIMA(p,d,q)-GARCH(gp,gq) probabilistic forecaster with long memory.

    The fractional differencing parameter d is estimated jointly with
    ARMA and GARCH parameters via MLE. This allows modelling long-range
    dependence where autocorrelations decay hyperbolically rather than
    exponentially.

    After fit(), use:
      - forecast(horizon)           → (mu, sigma) arrays from current state
      - simulate_paths(n, horizon)  → n stochastic paths from current state
      - update_state(actual_z, x_new) → advance state by one observed value

    For rolling evaluation, use RollingForecaster.

    Example:
        >>> config = ArfimaGarchConfig(arfima_order=(1, 1), garch_order=(1, 1))
        >>> model = ArfimaGarchForecaster(dataset=train_df, y_col="power",
        ...                     config=config)
        >>> model.fit()
        >>> print(f"Estimated d: {model.d:.4f}")
        >>> mu, sigma = model.forecast(horizon=24)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        exog_cols: Optional[Union[str, int, Iterable]] = None,
        config: Optional[ArfimaGarchConfig] = None,
        enable_logging: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        model_name: Optional[str] = None,
    ):
        self.config = config or ArfimaGarchConfig()

        # Fitted fractional d (populated by fit)
        self._d: float = 0.0

        # Full original-scale history (needed for fractional differencing)
        self._y_history: np.ndarray = np.array([])

        # Truncation parameter
        self._K: Optional[int] = self.config.truncation_K

        # Cached weights (computed once after fit, reused in rolling)
        self._w_cache: Optional[np.ndarray] = None      # diff weights
        self._pi_cache: Optional[np.ndarray] = None     # undiff weights

        super().__init__(
            dataset=dataset,
            y_col=y_col,
            exog_cols=exog_cols,
            config=self.config,
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
            model_name=model_name,
        )

    @property
    def d(self) -> float:
        """Estimated fractional differencing parameter."""
        return self._d

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def _build_config_hyperparameter(self) -> dict:
        return {
            "arfima_order": self.config.arfima_order,
            "garch_order":  self.config.garch_order,
            "d_bounds":     self.config.d_bounds,
            "opt_method":   self.config.opt_method,
            "distribution": self.config.distribution,
        }

    def _init_mean_primitive(self, n_exog: int) -> None:
        p, q = self.config.arfima_order
        self._mean_prim = ARMA(n_exog=n_exog, order=(p, q))

    def _get_pq(self) -> Tuple[int, int]:
        return self.config.arfima_order

    def _get_diff_loss(self) -> int:
        # Fractional differencing preserves all observations
        return 0

    def _apply_differencing(
        self, y: np.ndarray, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Initial differencing with d=0 (identity).
        # Actual differencing happens inside NLL where d is a parameter.
        y_diff = fractional_diff(y, self._d, K=self._K)
        return y_diff, X.copy()

    def _apply_differencing_in_nll(
        self, y_train: np.ndarray, params: np.ndarray
    ) -> Optional[np.ndarray]:
        """Re-compute fractional differencing with current d from optimizer."""
        d_cur = params[0]
        return fractional_diff(y_train, d_cur, K=self._K)

    def _init_extra_params(self, y_diff: np.ndarray) -> list:
        # d initial value = 0 (no fractional differencing)
        return [0.0]

    def _get_extra_bounds(self) -> list:
        return [self.config.d_bounds]

    def _get_extra_parscale(self) -> list:
        return [0.3]

    def _get_mean_skip(self) -> int:
        p, q = self.config.arfima_order
        return max(p, q)

    def _get_root_check_groups(self) -> list:
        # AR and MA checked separately; d excluded from both.
        p, q = self.config.arfima_order
        n_extra = 1  # d
        ar_idx = list(range(n_extra, n_extra + p))
        ma_idx = list(range(n_extra + p, n_extra + p + q))
        return [ar_idx, ma_idx]

    def _create_temp_mean_primitive(self, p: int, q: int, n_exog: int):
        return ARMA(n_exog=n_exog, order=(p, q))

    def _assign_temp_mean_params(
        self, tmp, params: np.ndarray,
        p: int, q: int, n_exog: int, n_extra: int,
    ) -> None:
        # params layout: [d, AR_1..p, MA_1..q, X_1..k, ...]
        off = n_extra  # = 1 (just d)
        tmp._phi[:] = params[off:off + p]
        tmp._theta[:] = params[off + p:off + p + q]
        if n_exog > 0:
            tmp._x_coeff[:] = params[off + p + q:off + p + q + n_exog]

    def _store_mean_coefficients(
        self, params: np.ndarray,
        p: int, q: int, n_exog: int, n_extra: int,
    ) -> None:
        self._d = float(params[0])
        off = n_extra  # = 1
        self._mean_prim.arma_coefficients = {
            "AR": params[off:off + p],
            "MA": params[off + p:off + p + q],
            "X":  params[off + p + q:off + p + q + n_exog],
        }
        self._rebuild_weight_cache()

    def _rebuild_weight_cache(self) -> None:
        """Pre-compute and cache fractional diff/undiff weights."""
        T = len(self._y_history) if len(self._y_history) > 0 else len(self.y)
        # Allow room for rolling forecast growth
        K = self._K if self._K is not None else T + 5000
        self._w_cache = fractional_diff_weights(self._d, K)
        self._pi_cache = fractional_diff_weights(-self._d, K)

    def _compute_undiff_state(self, y: np.ndarray) -> None:
        self._y_history = y.copy()
        if self._w_cache is None:
            self._rebuild_weight_cache()

    def _undifference_mean(self, z_forecast: np.ndarray) -> np.ndarray:
        """Inverse fractional differencing using cached weights."""
        H = len(z_forecast)
        T = len(self._y_history)
        w = self._w_cache  # diff weights for parameter d

        # y_t = z_t - sum_{k=1}^{K} w[k] * y_{t-k}  (rearranged from z = sum w*y)
        y_ext = np.empty(T + H, dtype=np.float64)
        y_ext[:T] = self._y_history

        K_max = len(w) - 1

        for h in range(H):
            idx = T + h
            n_lags = min(K_max, idx)
            # Vectorised dot product for the correction term
            correction = w[1:n_lags + 1] @ y_ext[idx - 1:idx - n_lags - 1:-1]
            y_ext[idx] = z_forecast[h] - correction

        return y_ext[T:]

    def _undifference_sigma(self, sigma_diff: np.ndarray) -> np.ndarray:
        """
        Propagate conditional std through inverse fractional differencing.

        sigma_original[h] = sqrt( sum_{j=0}^{h} pi[j]^2 * sigma_diff[h-j]^2 )

        where pi are the inverse-diff weights (parameter -d).
        """
        H = len(sigma_diff)
        if abs(self._d) < 1e-10:
            return sigma_diff.copy()

        pi = self._pi_cache[:H]
        pi2 = pi * pi
        var_diff = sigma_diff ** 2

        # Convolution via numpy (much faster than double loop)
        var_orig = np.convolve(pi2, var_diff)[:H]

        return np.sqrt(var_orig)

    def _difference_single(self, y_new: float) -> float:
        """Compute fractionally differenced value for a new observation."""
        # Append to history first
        self._y_history = np.append(self._y_history, y_new)
        t = len(self._y_history) - 1

        # Use cached weights, truncated to available history
        w = self._w_cache
        n_lags = min(len(w), t + 1)

        # z_t = w[0]*y[t] + w[1]*y[t-1] + ... = w[:n] @ y[t:t-n:-1]
        chunk = self._y_history[t - n_lags + 1:t + 1][::-1]
        return float(w[:n_lags] @ chunk)

    def _has_differencing(self) -> bool:
        return abs(self._d) > 1e-10

    def _get_state_dict(self) -> dict:
        return {
            "arma_coefficients": self._mean_prim.arma_coefficients,
            "fractional_d":      self._d,
            "y_history":         self._y_history,
        }

    def _load_state_dict(self, state: dict) -> None:
        self._mean_prim.arma_coefficients = state["arma_coefficients"]
        self._d = state.get("fractional_d", 0.0)
        self._y_history = state.get("y_history", np.array([]))
        self._rebuild_weight_cache()
