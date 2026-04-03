"""
Abstract base class for *-GARCH family models.

GarchBase(BaseForecaster) implements the shared fit/forecast/update_state/
simulate_paths pipeline.  Subclasses provide the model-specific parts via hooks:

  - Conditional mean primitive (ARMA, SARIMA, etc.)
  - Differencing / inverse-differencing logic
  - Extra parameters beyond AR/MA (e.g. SAR/SMA, fractional d)

Concrete subclasses: ArimaGarchForecaster, SarimaGarchForecaster, ArfimaGarchForecaster.
"""

from __future__ import annotations

import pickle
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union, Iterable, Dict, List, Self

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.core.base_model import BaseForecaster
from src.core.forecast_results import ParametricForecastResult
from src.core.forecast_results import SampleForecastResult
from src.models.statistical._primitives import GARCH


# -- Root-based stationarity / invertibility check ---------------------------

def _max_root_modulus(coeffs: np.ndarray) -> float:
    """
    Compute max |eigenvalue| of the companion matrix for the polynomial
    1 - c_1*z - c_2*z^2 - ... - c_p*z^p.

    Returns 0.0 for empty coefficients, |c_1| for p=1.
    Stationarity / invertibility requires the return value to be < 1.
    """
    p = len(coeffs)
    if p == 0:
        return 0.0
    if p == 1:
        return float(abs(coeffs[0]))
    C = np.zeros((p, p))
    C[0, :] = coeffs
    np.fill_diagonal(C[1:, :-1], 1.0)
    return float(np.max(np.abs(np.linalg.eigvals(C))))


# -- Log-likelihood functions (vectorised, no per-element overhead) ----------

def _normal_loglik(eps: np.ndarray, sigma: np.ndarray) -> float:
    """Sum of log N(0, sigma) densities. Avoids scipy overhead."""
    standardised = eps / sigma
    return float(np.sum(
        -0.5 * np.log(2.0 * np.pi) - np.log(sigma)
        - 0.5 * standardised * standardised
    ))


def _studentt_loglik(eps: np.ndarray, sigma: np.ndarray, df: float) -> float:
    """
    Sum of log **standardised** Student-t(df, 0, sigma) densities.

    Uses the standardised Student-t (variance=1) parameterisation matching
    rugarch: the GARCH sigma^2_t equals the true conditional variance,
    regardless of df.

    Density:
      f(z; nu) = Gamma((nu+1)/2) / [Gamma(nu/2) * sqrt(pi*(nu-2))]
                 * (1 + z^2/(nu-2))^{-(nu+1)/2}

    where z = eps / sigma (standardised residual).

    Implemented via gammaln for numerical stability.
    """
    half_dfp1 = 0.5 * (df + 1.0)
    half_df = 0.5 * df
    standardised = eps / sigma

    log_const = (gammaln(half_dfp1) - gammaln(half_df)
                 - 0.5 * np.log((df - 2.0) * np.pi))
    log_density = (log_const - np.log(sigma)
                   - half_dfp1 * np.log(1.0 + standardised * standardised / (df - 2.0)))

    return float(np.sum(log_density))


class GarchBase(BaseForecaster):
    """
    Abstract base for *-GARCH family models.

    Implements the full pipeline (fit -> forecast -> update_state -> simulate_paths)
    as template methods.  Subclasses override hooks for model-specific behaviour.

    Subclass contract (must implement):
        _init_mean_primitive(n_exog)   -> None    (set self._mean_prim)
        _get_pq()                      -> (p, q)
        _get_diff_loss()               -> int
        _apply_differencing(y, X)      -> (y_diff, x_aligned)
        _init_extra_params(y_diff)     -> list
        _get_extra_bounds()            -> list of (lo, hi)
        _get_extra_parscale()          -> list of float
        _get_mean_skip()               -> int
        _assign_mean_params(params, p, q, n_exog, n_extra) -> None
        _create_temp_mean_primitive(p, q, n_exog) -> object
        _assign_temp_mean_params(tmp, params, p, q, n_exog, n_extra) -> None
        _store_mean_coefficients(params, p, q, n_exog, n_extra) -> None
        _compute_undiff_state(y)       -> None
        _undifference_mean(z_forecast) -> np.ndarray
        _undifference_sigma(sigma_diff)-> np.ndarray
        _difference_single(y_new)      -> float
        _get_state_dict()              -> dict
        _load_state_dict(state)        -> None
        _has_differencing()            -> bool

    Optional hook (default returns None):
        _apply_differencing_in_nll(y_train, params) -> np.ndarray or None

    Example:
        >>> model = ArimaGarchForecaster(hyperparameter={
        ...     "arima_order": (2, 1, 1), "garch_order": (1, 1)
        ... })
        >>> model.fit(dataset=train_df, y_col="power", exog_cols=["wind_speed"])
        >>> result = model.forecast(horizon=24)
    """

    _MLE_DISTRIBUTIONS = {"normal", "studentT"}

    def __init__(
        self,
        hyperparameter: Optional[Dict] = None,
        model_name: Optional[str] = None,
    ):
        # Subclass must set self._distribution, self._garch_order,
        # self._variance_targeting, self._opt_method BEFORE calling super().__init__
        if self._distribution not in self._MLE_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {self._MLE_DISTRIBUTIONS}, "
                f"got '{self._distribution}'"
            )

        super().__init__(
            hyperparameter=hyperparameter,
            model_name=model_name,
        )

        # Mean primitive (set by subclass in fit)
        self._mean_prim = None

        # GARCH primitive (shared)
        self._garch = GARCH(order=self._garch_order)

        # Internal state arrays (populated by fit)
        self._y_diff: np.ndarray = np.array([])
        self._x_aligned: np.ndarray = np.array([])
        self._residuals: np.ndarray = np.array([])
        self._sigma2: np.ndarray = np.array([])

        # Estimated df for Student-t (None if normal)
        self._df: Optional[float] = None

    # ------------------------------------------------------------------
    # Abstract hooks -- subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_mean_primitive(self, n_exog: int) -> None:
        """Create and assign self._mean_prim (ARMA, SARIMA, etc.)."""
        ...

    @abstractmethod
    def _get_pq(self) -> Tuple[int, int]:
        """Return (p, q) -- non-seasonal AR and MA orders."""
        ...

    @abstractmethod
    def _get_diff_loss(self) -> int:
        """Number of observations lost to differencing."""
        ...

    @abstractmethod
    def _apply_differencing(
        self, y: np.ndarray, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply differencing to training data. Return (y_diff, x_aligned)."""
        ...

    @abstractmethod
    def _init_extra_params(self, y_diff: np.ndarray) -> list:
        """Return initial values for extra params (SAR/SMA, frac d, etc.)."""
        ...

    @abstractmethod
    def _get_extra_bounds(self) -> list:
        """Return bounds for extra params as list of (lo, hi)."""
        ...

    @abstractmethod
    def _get_extra_parscale(self) -> list:
        """Return parscale values for extra params."""
        ...

    @abstractmethod
    def _get_mean_skip(self) -> int:
        """Number of initial observations to skip in log-likelihood."""
        ...

    @abstractmethod
    def _create_temp_mean_primitive(self, p: int, q: int, n_exog: int):
        """Create a temporary mean primitive for use inside NLL."""
        ...

    @abstractmethod
    def _assign_temp_mean_params(
        self, tmp, params: np.ndarray,
        p: int, q: int, n_exog: int, n_extra: int,
    ) -> None:
        """Assign parameters to the temporary mean primitive inside NLL."""
        ...

    @abstractmethod
    def _store_mean_coefficients(
        self, params: np.ndarray,
        p: int, q: int, n_exog: int, n_extra: int,
    ) -> None:
        """Store fitted mean coefficients to self._mean_prim."""
        ...

    @abstractmethod
    def _compute_undiff_state(self, y: np.ndarray) -> None:
        """Compute and store undifferencing anchors from training y."""
        ...

    @abstractmethod
    def _undifference_mean(self, z_forecast: np.ndarray) -> np.ndarray:
        """Inverse-difference a mean forecast to original scale."""
        ...

    @abstractmethod
    def _undifference_sigma(self, sigma_diff: np.ndarray) -> np.ndarray:
        """Propagate conditional std through inverse differencing."""
        ...

    @abstractmethod
    def _difference_single(self, y_new: float) -> float:
        """Compute differenced value from a new original-scale observation."""
        ...

    @abstractmethod
    def _get_state_dict(self) -> dict:
        """Return model-specific state for serialisation."""
        ...

    @abstractmethod
    def _load_state_dict(self, state: dict) -> None:
        """Restore model-specific state from deserialisation."""
        ...

    @abstractmethod
    def _has_differencing(self) -> bool:
        """Whether inverse-differencing is needed for forecasts."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks (with defaults)
    # ------------------------------------------------------------------

    def _apply_differencing_in_nll(
        self, y_train: np.ndarray, params: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Re-apply differencing inside NLL with current params.

        Default: return None (use pre-computed y_diff).
        Override in ArfimaGarchForecaster where d is a parameter.
        """
        return None

    def _get_root_check_groups(self) -> list:
        """
        Return list of index-lists for root-based stationarity /
        invertibility checks (rugarch style).

        Each entry is a list of parameter indices forming one polynomial
        whose companion-matrix eigenvalues must all have modulus < 1.

        Default: [extra + AR], [MA]  -- AR stationarity + MA invertibility.
        Override for SARIMA (4 groups) or ARFIMA (skip d).
        """
        p, q = self._get_pq()
        n_extra = len(self._get_extra_bounds())
        ar_idx = list(range(0, n_extra + p))
        ma_idx = list(range(n_extra + p, n_extra + p + q))
        return [ar_idx, ma_idx]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        exog_cols=None,
    ) -> Self:
        """Estimate model parameters via MLE on the training window.

        Args:
            dataset: Training DataFrame with a proper time index.
            y_col: Target column name.
            exog_cols: Exogenous feature columns. None -> all except y_col.

        Returns:
            Self for method chaining.
        """
        # Extract y, X, index from dataset
        dataset = dataset.sort_index()
        self.y_col = y_col
        if exog_cols is not None:
            if isinstance(exog_cols, (str, int)):
                self.exog_cols = [exog_cols]
            else:
                self.exog_cols = list(exog_cols)
        else:
            self.exog_cols = [c for c in dataset.columns if c != y_col]

        self.y = dataset[y_col].to_numpy()
        self.X = dataset[self.exog_cols].to_numpy()
        self.index = dataset.index

        # Init mean primitive now that we know n_exog
        n_exog = self.X.shape[1] if self.X.ndim == 2 else 0
        self._init_mean_primitive(n_exog)

        p, q = self._get_pq()
        gp, gq = self._garch_order
        use_t = self._distribution == "studentT"
        use_vt = self._variance_targeting

        # Differencing
        y_diff, x_data = self._apply_differencing(self.y, self.X)

        # --------------- Parameter initialisation ---------------
        # Extra params (e.g. SAR/SMA for SARIMA, d for ARFIMA)
        extra_init = self._init_extra_params(y_diff)
        n_extra = len(extra_init)

        # AR init from autocorrelation, then ensure root-stationarity
        ar_init = []
        for i in range(1, p + 1):
            if len(y_diff) > i:
                ar_init.append(np.corrcoef(y_diff[:-i], y_diff[i:])[0, 1])
            else:
                ar_init.append(0.0)
        ma_init = [0.1] * q

        # Ensure initial AR and MA are in the stationary/invertible region
        for coeffs in (ar_init, ma_init):
            arr = np.array(coeffs)
            while len(arr) > 0 and _max_root_modulus(arr) >= 0.97:
                arr *= 0.8
                for j in range(len(coeffs)):
                    coeffs[j] = float(arr[j])

        sample_var = float(np.var(y_diff))
        garch_init = [0.8 / gp] * gp   # rugarch style: persistence ~ 0.9
        arch_init = [0.1 / gq] * gq

        x_init = (list(np.linalg.lstsq(x_data, y_diff, rcond=None)[0])
                  if n_exog > 0 else [])

        garch_offset = n_extra + p + q + n_exog

        if use_vt:
            base_params = (extra_init + ar_init + ma_init + x_init
                           + garch_init + arch_init)
        else:
            persistence = sum(garch_init) + sum(arch_init)
            omega_init = sample_var * max(0.05, 1.0 - persistence)
            base_params = (extra_init + ar_init + ma_init + x_init
                           + garch_init + arch_init + [np.log(omega_init)])

        if use_t:
            base_params.append(5.0)

        x0 = np.array(base_params)

        # --------------- Parscale ---------------
        scales = np.ones_like(x0)
        # Extra param scales
        extra_scales = self._get_extra_parscale()
        for i, es in enumerate(extra_scales):
            scales[i] = es
        # Exog scales
        for i, xi in enumerate(x_init):
            scales[n_extra + p + q + i] = max(abs(xi), 1.0)
        if use_t:
            scales[-1] = 5.0

        # --------------- Bounds (in scaled space) ---------------
        extra_bounds = self._get_extra_bounds()
        bounds_raw = (
            extra_bounds +
            [(-1, 1)] * p + [(-1, 1)] * q +
            [(None, None)] * n_exog +
            [(0, None)] * gp + [(0, None)] * gq
        )
        if not use_vt:
            bounds_raw.append((None, None))  # log_omega unbounded
        if use_t:
            bounds_raw.append((2.01, 100.0))

        bounds_scaled = [
            (lo / s if lo is not None else None,
             hi / s if hi is not None else None)
            for (lo, hi), s in zip(bounds_raw, scales)
        ]

        # --------------- Stationarity constraints (root-based, rugarch style) ---
        root_groups = self._get_root_check_groups()
        opt_method = self._opt_method

        # Ensure x0 is feasible for all root-based constraints
        for idx in root_groups:
            while _max_root_modulus(x0[idx]) >= 0.97:
                x0[idx] *= 0.8
        x0_scaled = x0 / scales  # recompute after adjustment

        # For constrained optimizers (trust-constr, SLSQP):
        # smooth root-based constraints + GARCH persistence
        if opt_method != "L-BFGS-B":
            constraints = []
            for grp in root_groups:
                def _make_root_con(idx=grp):
                    def con(theta):
                        params = theta * scales
                        return 1.0 - _max_root_modulus(params[idx]) - 1e-6
                    return con
                constraints.append({"type": "ineq", "fun": _make_root_con()})

            def garch_stationary(theta):
                params = theta * scales
                return 1.0 - np.sum(params[garch_offset:garch_offset + gp]) \
                           - np.sum(params[garch_offset + gp:garch_offset + gp + gq]) \
                           - 1e-6
            constraints.append({"type": "ineq", "fun": garch_stationary})
        else:
            constraints = ()

        # --------------- Negative log-likelihood (scaled space) ---------------
        _mean_tmp = self._create_temp_mean_primitive(p, q, n_exog)
        _garch_tmp = GARCH(order=(gp, gq))
        skip = self._get_mean_skip()
        skip = max(skip, gp, gq)

        def neg_log_likelihood(theta):
            params = theta * scales  # unscale to actual values

            self._assign_temp_mean_params(
                _mean_tmp, params, p, q, n_exog, n_extra,
            )

            _garch_tmp._beta[:] = params[garch_offset:garch_offset + gp]
            _garch_tmp._alpha[:] = params[garch_offset + gp:garch_offset + gp + gq]

            # Allow subclass to re-compute differencing with current params
            y_diff_cur = self._apply_differencing_in_nll(self.y, params)
            if y_diff_cur is None:
                y_diff_cur = y_diff

            if use_vt:
                _, residuals = _mean_tmp.compute_residuals(y_diff_cur, x_data)
                resid_var = float(np.var(residuals))
                persistence = (np.sum(params[garch_offset:garch_offset + gp])
                             + np.sum(params[garch_offset + gp:garch_offset + gp + gq]))
                _garch_tmp._omega = resid_var * max(1e-6, 1.0 - persistence)
            else:
                omega_idx = garch_offset + gp + gq
                _garch_tmp._omega = np.exp(params[omega_idx])
                _, residuals = _mean_tmp.compute_residuals(y_diff_cur, x_data)

            with np.errstate(over="ignore"):
                sigma2 = _garch_tmp.compute_variance_series(residuals)

            # Numerical safety (not a stationarity wall)
            if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
                return 1e15

            eps_tail = residuals[skip:]
            sigma_tail = np.sqrt(sigma2[skip:])
            np.maximum(sigma_tail, 1e-9, out=sigma_tail)

            if use_t:
                nll = -_studentt_loglik(eps_tail, sigma_tail, params[-1])
            else:
                nll = -_normal_loglik(eps_tail, sigma_tail)

            if not np.isfinite(nll):
                return 1e15

            # Smooth penalty on root modulus (all optimizers, rugarch-style)
            T = len(eps_tail)
            for idx in root_groups:
                mod = _max_root_modulus(params[idx])
                if mod >= 1.0:
                    nll += T * 1e4 * (mod - 0.999)
                elif mod > 0.999:
                    nll += T * 1e3 * (mod - 0.999) ** 2
            garch_pers = (np.sum(params[garch_offset:garch_offset + gp])
                        + np.sum(params[garch_offset + gp:garch_offset + gp + gq]))
            if garch_pers >= 1.0:
                nll += T * 1e4 * (garch_pers - 0.999)
            elif garch_pers > 0.999:
                nll += T * 1e3 * (garch_pers - 0.999) ** 2

            return nll

        # --------------- Optimise ---------------
        res = minimize(neg_log_likelihood, x0=x0_scaled,
                       method=opt_method,
                       bounds=bounds_scaled, constraints=constraints)

        if not res.success:
            warnings.warn(f"MLE did not converge: {res.message}", stacklevel=2)

        # --------------- Store fitted state ---------------
        params = res.x * scales  # unscale

        self._store_mean_coefficients(params, p, q, n_exog, n_extra)

        beta_fit = params[garch_offset:garch_offset + gp]
        alpha_fit = params[garch_offset + gp:garch_offset + gp + gq]

        if use_vt:
            # Recompute y_diff with final params (for ARFIMA where d changed)
            y_diff_final = self._apply_differencing_in_nll(self.y, params)
            if y_diff_final is None:
                y_diff_final = y_diff

            _, res_final = self._mean_prim.compute_residuals(y_diff_final, x_data)
            resid_var = float(np.var(res_final))
            persistence = float(np.sum(beta_fit) + np.sum(alpha_fit))
            omega_fit = resid_var * max(1e-6, 1.0 - persistence)
        else:
            omega_idx = garch_offset + gp + gq
            omega_fit = np.exp(params[omega_idx])

        self._garch.garch_coefficients = {
            "GARCH": beta_fit, "ARCH": alpha_fit, "CONSTANT": omega_fit,
        }

        if use_t:
            self._df = float(params[-1])

        # Store final differenced series (may differ from initial if d was estimated)
        y_diff_final = self._apply_differencing_in_nll(self.y, params)
        if y_diff_final is not None:
            self._y_diff = y_diff_final
            # Re-align x_data if diff loss changed -- for ARFIMA this is same length
            diff_loss = self._get_diff_loss()
            self._x_aligned = self.X[diff_loss:] if diff_loss > 0 else self.X.copy()
        else:
            self._y_diff = y_diff
            self._x_aligned = x_data

        self._compute_undiff_state(self.y)

        _, self._residuals = self._mean_prim.compute_residuals(
            self._y_diff, self._x_aligned,
        )
        self._sigma2 = self._garch.compute_variance_series(self._residuals)

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Forecast (single point-in-time)
    # ------------------------------------------------------------------

    def forecast(
        self,
        horizon: int,
        x_future: Optional[np.ndarray] = None,
    ) -> ParametricForecastResult:
        """
        Multi-step-ahead forecast from the current state.

        Deterministic propagation: future eps = 0, future eps^2 = sigma^2.
        Returns are in the **original (undifferenced) scale**.

        Args:
            horizon:  Number of steps ahead.
            x_future: Exogenous features for future steps, shape (horizon, n_exog).
                      If None, zeros are used.

        Returns:
            ParametricForecastResult with shape (1, H).
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before forecast().")

        n_exog = self._x_aligned.shape[1] if self._x_aligned.ndim == 2 else 0
        _empty_x = np.zeros(n_exog) if n_exog > 0 else np.array([])

        z_snap = self._y_diff.copy()
        eps_snap = self._residuals.copy()
        eps2_snap = eps_snap * eps_snap
        s2_snap = self._sigma2.copy()

        mu_diff = np.empty(horizon)
        sigma_diff = np.empty(horizon)

        for h in range(horizon):
            x_h = (x_future[h] if x_future is not None and n_exog > 0
                   else _empty_x)

            mu_h = self._mean_prim.predict_step(z_snap, eps_snap, x_h)
            sigma2_h = self._garch.predict_variance_step(eps2_snap, s2_snap)

            mu_diff[h] = mu_h
            sigma_diff[h] = np.sqrt(sigma2_h)

            # Propagate: z = mu (E[eps]=0), eps = 0, eps^2 = sigma^2
            z_snap = np.append(z_snap, mu_h)
            eps_snap = np.append(eps_snap, 0.0)
            eps2_snap = np.append(eps2_snap, sigma2_h)
            s2_snap = np.append(s2_snap, sigma2_h)

        # Inverse-difference to original scale
        mu = self._undifference_mean(mu_diff)
        sigma = self._undifference_sigma(sigma_diff)

        # Build native params -- reshape to (1, H)
        dist_name = self._distribution
        if dist_name == "studentT" and self._df is not None:
            df = self._df
            params = {
                "loc": mu.reshape(1, -1),
                "scale": (sigma * np.sqrt((df - 2.0) / df)).reshape(1, -1),
                "df": np.full_like(mu, df).reshape(1, -1),
            }
        else:
            params = {
                "loc": mu.reshape(1, -1),
                "scale": np.maximum(sigma, 1e-9).reshape(1, -1),
            }

        basis_index = pd.Index([self.index[-1]])
        return ParametricForecastResult(
            dist_name=dist_name,
            params=params,
            basis_index=basis_index,
            model_name=self.nm,
        )

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_state(self, y_new: float, x_new: Optional[np.ndarray] = None) -> None:
        """
        Advance internal state by one observed value (original scale).

        Internally computes the differenced value, then updates mean residuals
        and GARCH variance.

        Args:
            y_new: Actual observation in **original scale** at the next time step.
            x_new: Exogenous features at this time step. None if no exogenous.
        """
        if x_new is None:
            x_new = np.array([])

        actual_z = self._difference_single(y_new)

        # Compute actual residual
        mu_actual = self._mean_prim.predict_step(
            self._y_diff, self._residuals, x_new,
        )
        eps_actual = actual_z - mu_actual

        self._y_diff = np.append(self._y_diff, actual_z)
        self._residuals = np.append(self._residuals, eps_actual)

        # Compute actual variance
        eps2_actual = self._residuals * self._residuals
        sigma2_actual = self._garch.predict_variance_step(
            eps2_actual, self._sigma2,
        )
        self._sigma2 = np.append(self._sigma2, sigma2_actual)

    # ------------------------------------------------------------------
    # Path simulation (single point-in-time)
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        x_future: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> SampleForecastResult:
        """
        Monte Carlo path simulation from the current fitted state.

        Each path maintains its own mean/GARCH state and propagates forward
        independently. Shocks are drawn from the fitted innovation distribution.

        Args:
            n_paths:  Number of simulation paths.
            horizon:  Number of steps to simulate per path.
            x_future: Exogenous features for future steps, shape (horizon, n_exog).
                      If None, zeros are used.
            seed:     Random seed for reproducibility.

        Returns:
            SampleForecastResult with samples shape (1, n_paths, horizon).
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before simulate_paths().")

        rng = np.random.default_rng(seed)
        n_exog = self._x_aligned.shape[1] if self._x_aligned.ndim == 2 else 0

        # Draw all shocks at once
        if self._distribution == "studentT" and self._df is not None:
            df = self._df
            raw = rng.standard_t(df, size=(n_paths, horizon))
            eta = raw * np.sqrt((df - 2.0) / df)
        else:
            eta = rng.standard_normal(size=(n_paths, horizon))

        _empty_x = np.array([])
        x_fut = (np.asarray(x_future)
                 if x_future is not None and n_exog > 0 else None)

        paths = np.empty((n_paths, horizon))

        for n in range(n_paths):
            z_snap = self._y_diff.copy()
            eps_snap = self._residuals.copy()
            eps2_snap = eps_snap * eps_snap
            s2_snap = self._sigma2.copy()

            for h in range(horizon):
                x_h = x_fut[h] if x_fut is not None else _empty_x

                mu_h = self._mean_prim.predict_step(z_snap, eps_snap, x_h)
                sigma2_h = self._garch.predict_variance_step(eps2_snap, s2_snap)
                sigma_h = np.sqrt(sigma2_h)

                eps_h = sigma_h * eta[n, h]
                z_h = mu_h + eps_h

                paths[n, h] = z_h

                z_snap = np.append(z_snap, z_h)
                eps_snap = np.append(eps_snap, eps_h)
                eps2_snap = np.append(eps2_snap, eps_h * eps_h)
                s2_snap = np.append(s2_snap, sigma2_h)

        # Inverse-difference each path to original scale
        if self._has_differencing():
            for n in range(n_paths):
                paths[n, :] = self._undifference_mean(paths[n, :])

        # Wrap as SampleForecastResult: (1, n_paths, horizon)
        basis_index = pd.Index([self.index[-1]])
        samples = paths[np.newaxis, :, :]

        return SampleForecastResult(
            samples=samples,
            basis_index=basis_index,
            model_name=self.nm,
        )

    # ------------------------------------------------------------------
    # get_params
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return fitted ARMA/GARCH coefficients.

        Returns:
            dict with 'arma' and 'garch' keys.

        Example:
            >>> model.fit(dataset=df, y_col="power")
            >>> model.get_params()
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before get_params().")
        return {
            "arma": self._mean_prim.arma_coefficients,
            "garch": self._garch.garch_coefficients,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save_model_specific(self, model_path: Path) -> Path:
        sv_path = model_path.with_suffix(".pkl")
        state = {
            "garch_coefficients": self._garch.garch_coefficients,
            "y_diff":             self._y_diff,
            "x_aligned":          self._x_aligned,
            "residuals":          self._residuals,
            "sigma2":             self._sigma2,
            "df":                 self._df,
            "index":              getattr(self, "index", None),
        }
        state.update(self._get_state_dict())
        with open(sv_path, "wb") as f:
            pickle.dump(state, f)
        return sv_path

    def _load_model_specific(self, model_path: Path) -> None:
        with open(model_path.with_suffix(".pkl"), "rb") as f:
            state = pickle.load(f)

        # Initialize mean primitive if not already set (e.g. loading without fit)
        if self._mean_prim is None:
            x_aligned = state.get("x_aligned", np.array([]))
            n_exog = x_aligned.shape[1] if x_aligned.ndim == 2 else 0
            self._init_mean_primitive(n_exog)

        self._garch.garch_coefficients = state["garch_coefficients"]
        self._y_diff = state["y_diff"]
        self._x_aligned = state["x_aligned"]
        self._residuals = state["residuals"]
        self._sigma2 = state["sigma2"]
        self._df = state.get("df")
        if "index" in state and state["index"] is not None:
            self.index = state["index"]
        self._load_state_dict(state)
