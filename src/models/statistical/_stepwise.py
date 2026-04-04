"""Stepwise order selection for *-GARCH family models.

Implements Hyndman-Khandakar (2008) stepwise search over ARMA orders,
with automatic differencing order selection via KPSS/ADF unit root tests.

Supports three model families:
  - ``"arima"``:  ARIMA(p,d,q)-GARCH  -- search (p,q), auto d
  - ``"sarima"``: SARIMA(p,d,q)(P,D,Q,s)-GARCH -- search (p,q,P,Q), auto d,D
  - ``"arfima"``: ARFIMA(p,q)-GARCH -- search (p,q), d estimated by MLE

Example:
    >>> selector = StepwiseOrderSelector("arima", ic="aicc", max_p=5, max_q=5)
    >>> best = selector.select(dataset=train_df, y_col="power", exog_cols=["ws"])
    >>> best.aic
    1234.56
    >>> selector.summary
       arima_order garch_order  ...   aicc  converged
    0    (1, 1, 1)      (1, 1)  ...  1234   True
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.statistical._garch_base import GarchBase

# ======================================================================
# Unit root tests
# ======================================================================

# MacKinnon (1994) critical values for ADF with constant (no trend).
# tau(n) = tau_inf + c1/n + c2/n^2
_ADF_CV = {
    0.01: (-3.4336, -5.999, -29.25),
    0.05: (-2.8621, -2.738, -8.36),
    0.10: (-2.5671, -1.438, -4.48),
}

# KPSS critical values (level stationarity, Kwiatkowski et al. 1992 Table 1).
_KPSS_CV = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}


def _adf_pvalue_approx(stat: float, n: int, significance: float = 0.05) -> bool:
    """Check whether ADF statistic rejects the unit-root null.

    Uses MacKinnon finite-sample critical values.

    Args:
        stat: ADF t-statistic.
        n: Sample size.
        significance: Significance level (must be in ``_ADF_CV``).

    Returns:
        True if the null (unit root) is **rejected** (series is stationary).

    Example:
        >>> _adf_pvalue_approx(-3.5, 200)
        True
    """
    tau_inf, c1, c2 = _ADF_CV[significance]
    cv = tau_inf + c1 / n + c2 / (n * n)
    return stat < cv


def _adf_statistic(y: np.ndarray, max_lag: int | None = None) -> float:
    """Compute augmented Dickey-Fuller t-statistic.

    Regression:  dy_t = alpha + gamma * y_{t-1} + sum_i phi_i * dy_{t-i} + e_t

    Lag length selected by AIC from 0..max_lag.

    Args:
        y: Time series, shape (T,).
        max_lag: Maximum augmentation lag.  None = Schwert rule.

    Returns:
        ADF t-statistic for gamma.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> _adf_statistic(rng.standard_normal(200))  # stationary -> large negative
    """
    T = len(y)
    if max_lag is None:
        max_lag = int(12.0 * (T / 100.0) ** 0.25)
    max_lag = min(max_lag, T // 3)

    dy = np.diff(y)
    y_lag = y[:-1]

    best_aic = np.inf
    best_stat = 0.0

    for k in range(max_lag + 1):
        start = max(k, 1)
        n_obs = len(dy) - start

        if n_obs < 4:
            continue

        # Build design matrix: [constant, y_{t-1}, dy_{t-1}, ..., dy_{t-k}]
        X_cols = [np.ones(n_obs), y_lag[start:start + n_obs]]
        for lag in range(1, k + 1):
            X_cols.append(dy[start - lag:start - lag + n_obs])
        X = np.column_stack(X_cols)
        dep = dy[start:start + n_obs]

        # OLS
        try:
            coef, residuals, _, _ = np.linalg.lstsq(X, dep, rcond=None)
        except np.linalg.LinAlgError:
            continue

        resid = dep - X @ coef
        ssr = float(resid @ resid)
        n_params = X.shape[1]

        aic = n_obs * np.log(ssr / n_obs + 1e-300) + 2.0 * n_params

        if aic < best_aic:
            best_aic = aic
            # t-statistic for gamma (coefficient index 1)
            sigma2 = ssr / max(n_obs - n_params, 1)
            try:
                cov = sigma2 * np.linalg.inv(X.T @ X)
                best_stat = coef[1] / np.sqrt(max(cov[1, 1], 1e-300))
            except np.linalg.LinAlgError:
                best_stat = 0.0

    return best_stat


def _kpss_statistic(y: np.ndarray) -> float:
    """Compute KPSS level-stationarity test statistic.

    H0: series is level-stationary.

    Uses Bartlett kernel for long-run variance estimation with
    Schwert bandwidth selection.

    Args:
        y: Time series, shape (T,).

    Returns:
        KPSS LM statistic.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> _kpss_statistic(rng.standard_normal(200))  # stationary -> small
    """
    T = len(y)
    e = y - np.mean(y)

    # Partial sums of residuals
    S = np.cumsum(e)

    # Long-run variance: Bartlett kernel
    bandwidth = int(4.0 * (T / 100.0) ** (2.0 / 9.0))
    bandwidth = max(bandwidth, 1)

    gamma0 = float(e @ e) / T
    lr_var = gamma0
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1.0)
        gamma_j = float(e[j:] @ e[:-j]) / T
        lr_var += 2.0 * w * gamma_j

    lr_var = max(lr_var, 1e-300)
    return float(S @ S) / (T * T * lr_var)


def select_d(y: np.ndarray, max_d: int = 2, significance: float = 0.05) -> int:
    """Select integer differencing order via KPSS test (ndiffs style).

    Repeatedly applies first-differencing and tests for stationarity
    with KPSS.  Stops when the null (stationarity) is not rejected.

    Args:
        y: Original time series.
        max_d: Maximum allowed differencing order.
        significance: KPSS significance level.

    Returns:
        Optimal differencing order d in [0, max_d].

    Example:
        >>> rng = np.random.default_rng(42)
        >>> select_d(np.cumsum(rng.standard_normal(500)))  # random walk
        1
    """
    # Pick KPSS critical value (use nearest available)
    cv_keys = sorted(_KPSS_CV.keys())
    cv_key = min(cv_keys, key=lambda k: abs(k - significance))
    cv = _KPSS_CV[cv_key]

    z = y.copy()
    for d in range(max_d + 1):
        if len(z) < 10:
            return d
        stat = _kpss_statistic(z)
        if stat < cv:
            return d
        if d < max_d:
            z = np.diff(z)
    return max_d


def select_D(
    y: np.ndarray, s: int, max_D: int = 1, significance: float = 0.05,
) -> int:
    """Select seasonal differencing order via seasonal KPSS approach.

    Applies seasonal differencing (1 - B^s) and tests the result
    with KPSS for stationarity.

    Args:
        y: Original time series.
        s: Seasonal period.
        max_D: Maximum seasonal differencing order.
        significance: KPSS significance level.

    Returns:
        Optimal seasonal differencing order D in [0, max_D].

    Example:
        >>> select_D(np.random.default_rng(0).standard_normal(500), s=24)
        0
    """
    cv_keys = sorted(_KPSS_CV.keys())
    cv_key = min(cv_keys, key=lambda k: abs(k - significance))
    cv = _KPSS_CV[cv_key]

    z = y.copy()
    for D in range(max_D + 1):
        if len(z) < max(10, 2 * s):
            return D
        stat = _kpss_statistic(z)
        if stat < cv:
            return D
        if D < max_D:
            z = z[s:] - z[:-s]
    return max_D


# ======================================================================
# Candidate result
# ======================================================================

@dataclass
class _CandidateResult:
    """Record of a single candidate model evaluation."""

    arima_order: tuple
    garch_order: tuple
    seasonal_order: tuple | None
    ic_value: float
    aic: float | None = None
    bic: float | None = None
    aicc: float | None = None
    loglik: float | None = None
    n_params: int = 0
    converged: bool = False
    model: GarchBase | None = field(default=None, repr=False)
    error: str | None = None


# ======================================================================
# StepwiseOrderSelector
# ======================================================================

_MODEL_TYPES = {"arima", "sarima", "arfima"}


class StepwiseOrderSelector:
    """Stepwise ARIMA-GARCH order selection (auto.arima style).

    Searches over (p, q) for the ARIMA mean equation using the
    Hyndman-Khandakar stepwise algorithm.  For SARIMA, the search
    extends to (P, Q) as well.  The differencing order d (and seasonal
    D for SARIMA) is determined automatically via KPSS or user-specified.

    The selected model is a fully fitted ``GarchBase`` subclass instance,
    directly usable with ``RollingRunner``.

    Args:
        model_type: One of ``"arima"``, ``"sarima"``, ``"arfima"``.
        ic: Information criterion to minimise (``"aic"``, ``"bic"``, ``"aicc"``).
        max_p: Maximum AR order.
        max_q: Maximum MA order.
        max_d: Maximum differencing order (ignored for arfima).
        d: Fixed differencing order.  None = auto-select via KPSS.
        garch_order: Fixed GARCH order.
        distribution: Innovation distribution name (``"normal"``, ``"studentT"``).
        variance_targeting: Whether to use variance targeting in MLE.
        seasonal_period: Seasonal period s (required for sarima).
        max_P: Maximum seasonal AR order (sarima only).
        max_Q: Maximum seasonal MA order (sarima only).
        max_D: Maximum seasonal differencing order (sarima only).
        D: Fixed seasonal differencing order. None = auto-select.
        d_bounds: Fractional d bounds (arfima only).
        truncation_K: Truncation length for fractional differencing (arfima only).
        verbose: Print progress messages during search.

    Example:
        >>> selector = StepwiseOrderSelector("arima", ic="aicc")
        >>> best = selector.select(train_df, y_col="power", exog_cols=["ws"])
        >>> best.aic
        1234.56
    """

    def __init__(
        self,
        model_type: str,
        *,
        ic: str = "aicc",
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        d: int | None = None,
        garch_order: tuple[int, int] = (1, 1),
        distribution: str = "normal",
        variance_targeting: bool = True,
        # SARIMA
        seasonal_period: int | None = None,
        max_P: int = 2,
        max_Q: int = 2,
        max_D: int = 1,
        D: int | None = None,
        # ARFIMA
        d_bounds: tuple[float, float] = (-0.499, 0.499),
        truncation_K: int = 500,
        verbose: bool = True,
    ):
        if model_type not in _MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {_MODEL_TYPES}, got '{model_type}'"
            )
        if model_type == "sarima" and seasonal_period is None:
            raise ValueError("seasonal_period is required for sarima")
        if ic not in ("aic", "bic", "aicc"):
            raise ValueError(f"ic must be 'aic', 'bic', or 'aicc', got '{ic}'")

        self.model_type = model_type
        self.ic = ic
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.d = d
        self.garch_order = garch_order
        self.distribution = distribution
        self.variance_targeting = variance_targeting

        self.seasonal_period = seasonal_period
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_D = max_D
        self.D = D

        self.d_bounds = d_bounds
        self.truncation_K = truncation_K
        self.verbose = verbose

        self.search_history: list[_CandidateResult] = []
        self.best_model: GarchBase | None = None
        self.best_order: tuple | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        dataset: pd.DataFrame,
        y_col: str,
        exog_cols: list[str] | None = None,
    ) -> GarchBase:
        """Run stepwise search and return the best fitted model.

        Args:
            dataset: Training DataFrame.
            y_col: Target column.
            exog_cols: Exogenous feature columns.

        Returns:
            Best fitted ``GarchBase`` subclass instance.

        Raises:
            RuntimeError: If all candidate models fail to converge.

        Example:
            >>> selector = StepwiseOrderSelector("arima")
            >>> model = selector.select(df, "power", ["wind_speed"])
        """
        self.search_history = []
        y = dataset[y_col].to_numpy(dtype=np.float64)

        # --- Step 0: determine differencing orders ---
        d = self._resolve_d(y)
        D_val = self._resolve_D(y) if self.model_type == "sarima" else None

        if self.verbose:
            msg = f"[stepwise] model_type={self.model_type}, d={d}"
            if D_val is not None:
                msg += f", D={D_val}, s={self.seasonal_period}"
            print(msg)

        # --- Step 1: evaluate initial candidates ---
        evaluated: set[tuple] = set()

        if self.model_type == "sarima":
            initials = self._sarima_initial_candidates(d, D_val)
        else:
            initials = self._arima_initial_candidates(d)

        for key in initials:
            self._evaluate(key, dataset, y_col, exog_cols)
            evaluated.add(key)

        best = self._current_best()
        if best is None:
            raise RuntimeError("All initial candidate models failed.")

        if self.verbose:
            self._log_best(best, "initial")

        # --- Step 2: stepwise neighborhood search ---
        improved = True
        step = 0
        max_steps = 50

        while improved and step < max_steps:
            improved = False
            step += 1

            neighbors = self._neighbors(best, evaluated)
            for key in neighbors:
                self._evaluate(key, dataset, y_col, exog_cols)
                evaluated.add(key)

            new_best = self._current_best()
            if new_best is not None and new_best.ic_value < best.ic_value:
                best = new_best
                improved = True
                if self.verbose:
                    self._log_best(best, f"step {step}")

        self.best_model = best.model
        self.best_order = (best.arima_order, best.garch_order, best.seasonal_order)

        if self.verbose:
            n_eval = len(self.search_history)
            n_ok = sum(1 for r in self.search_history if r.converged)
            print(
                f"[stepwise] done: {n_eval} evaluated, {n_ok} converged, "
                f"best {self.ic}={best.ic_value:.2f}"
            )

        return self.best_model

    @property
    def summary(self) -> pd.DataFrame:
        """Search history as a DataFrame sorted by IC value.

        Example:
            >>> selector.select(df, "power")
            >>> selector.summary.head()
        """
        rows = []
        for r in self.search_history:
            row = {
                "arima_order": r.arima_order,
                "garch_order": r.garch_order,
                "aic": r.aic,
                "bic": r.bic,
                "aicc": r.aicc,
                "loglik": r.loglik,
                "n_params": r.n_params,
                "converged": r.converged,
            }
            if any(c.seasonal_order is not None for c in self.search_history):
                row["seasonal_order"] = r.seasonal_order
            if r.error:
                row["error"] = r.error
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("aicc" if self.ic == "aicc" else self.ic,
                                na_position="last").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Differencing order resolution
    # ------------------------------------------------------------------

    def _resolve_d(self, y: np.ndarray) -> int:
        """Determine d: user-specified or auto via KPSS."""
        if self.model_type == "arfima":
            return 0  # fractional d estimated by MLE
        if self.d is not None:
            return self.d
        return select_d(y, max_d=self.max_d)

    def _resolve_D(self, y: np.ndarray) -> int:
        """Determine seasonal D: user-specified or auto via KPSS."""
        if self.D is not None:
            return self.D
        return select_D(y, s=self.seasonal_period, max_D=self.max_D)

    # ------------------------------------------------------------------
    # Initial candidates
    # ------------------------------------------------------------------

    def _arima_initial_candidates(self, d: int) -> list[tuple]:
        """Generate initial candidate keys for arima/arfima.

        Key format: (arima_order, garch_order, seasonal_order).
        """
        mp, mq = self.max_p, self.max_q
        g = self.garch_order
        candidates = []
        for p, q in [(2, 2), (0, 0), (1, 0), (0, 1), (1, 1)]:
            if p <= mp and q <= mq:
                candidates.append(((p, d, q), g, None))
        return candidates

    def _sarima_initial_candidates(self, d: int, D: int) -> list[tuple]:
        """Generate initial candidate keys for sarima.

        Key format: (arima_order, garch_order, seasonal_order).
        """
        mp, mq = self.max_p, self.max_q
        mP, mQ = self.max_P, self.max_Q
        s = self.seasonal_period
        g = self.garch_order

        base_pq = [(p, q) for p, q in [(2, 2), (0, 0), (1, 0), (0, 1)]
                    if p <= mp and q <= mq]
        seasonal_pq = [(P, Q) for P, Q in [(1, 1), (0, 0)]
                       if P <= mP and Q <= mQ]

        candidates = []
        for p, q in base_pq:
            for P, Q in seasonal_pq:
                candidates.append(((p, d, q), g, (P, D, Q, s)))
        return candidates

    # ------------------------------------------------------------------
    # Neighbor generation
    # ------------------------------------------------------------------

    def _neighbors(
        self, best: _CandidateResult, evaluated: set[tuple],
    ) -> list[tuple]:
        """Generate neighbor keys from the current best."""
        p, d, q = best.arima_order
        g = best.garch_order
        mp, mq = self.max_p, self.max_q

        keys: list[tuple] = []

        # (p,q) neighbors: +-1 in each dimension
        pq_deltas = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, 1), (1, -1), (1, 1), (-1, -1),
        ]

        if self.model_type == "sarima":
            P, D_val, Q, s = best.seasonal_order
            mP, mQ = self.max_P, self.max_Q

            # ARIMA-dimension neighbors (fixed P, Q)
            for dp, dq in pq_deltas:
                np_, nq = p + dp, q + dq
                if 0 <= np_ <= mp and 0 <= nq <= mq:
                    key = ((np_, d, nq), g, (P, D_val, Q, s))
                    if key not in evaluated:
                        keys.append(key)

            # Seasonal-dimension neighbors (fixed p, q)
            for dP, dQ in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nP, nQ = P + dP, Q + dQ
                if 0 <= nP <= mP and 0 <= nQ <= mQ:
                    key = ((p, d, q), g, (nP, D_val, nQ, s))
                    if key not in evaluated:
                        keys.append(key)
        else:
            for dp, dq in pq_deltas:
                np_, nq = p + dp, q + dq
                if 0 <= np_ <= mp and 0 <= nq <= mq:
                    key = ((np_, d, nq), g, None)
                    if key not in evaluated:
                        keys.append(key)

        return keys

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        key: tuple,
        dataset: pd.DataFrame,
        y_col: str,
        exog_cols: list[str] | None,
    ) -> _CandidateResult:
        """Fit a single candidate and record the result."""
        arima_order, garch_order, seasonal_order = key
        hp = self._build_hyperparameters(arima_order, garch_order, seasonal_order)

        try:
            model = self._create_model(hp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(dataset=dataset, y_col=y_col, exog_cols=exog_cols)

            ic_val = getattr(model, self.ic)
            if not np.isfinite(ic_val):
                raise ValueError("Non-finite IC value")

            result = _CandidateResult(
                arima_order=arima_order,
                garch_order=garch_order,
                seasonal_order=seasonal_order,
                ic_value=ic_val,
                aic=model.aic,
                bic=model.bic,
                aicc=model.aicc,
                loglik=model.loglik,
                n_params=model._n_params,
                converged=True,
                model=model,
            )
        except Exception as e:
            result = _CandidateResult(
                arima_order=arima_order,
                garch_order=garch_order,
                seasonal_order=seasonal_order,
                ic_value=np.inf,
                error=str(e),
            )

        self.search_history.append(result)
        return result

    def _build_hyperparameters(
        self,
        arima_order: tuple,
        garch_order: tuple,
        seasonal_order: tuple | None,
    ) -> dict[str, Any]:
        """Build hyperparameter dict appropriate for model_type."""
        base = {
            "garch_order": garch_order,
            "distribution": self.distribution,
            "variance_targeting": self.variance_targeting,
        }
        if self.model_type == "arima":
            base["arima_order"] = arima_order
        elif self.model_type == "sarima":
            base["arima_order"] = arima_order
            base["seasonal_order"] = seasonal_order
        elif self.model_type == "arfima":
            p, _, q = arima_order
            base["arfima_order"] = (p, q)
            base["d_bounds"] = self.d_bounds
            base["truncation_K"] = self.truncation_K
        return base

    def _create_model(self, hp: dict) -> GarchBase:
        """Instantiate the appropriate model class."""
        if self.model_type == "arima":
            from src.models.statistical.arima_garch import ArimaGarchForecaster
            return ArimaGarchForecaster(hyperparameter=hp)
        elif self.model_type == "sarima":
            from src.models.statistical.sarima_garch import SarimaGarchForecaster
            return SarimaGarchForecaster(hyperparameter=hp)
        elif self.model_type == "arfima":
            from src.models.statistical.arfima_garch import ArfimaGarchForecaster
            return ArfimaGarchForecaster(hyperparameter=hp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_best(self) -> _CandidateResult | None:
        """Return the best converged candidate so far."""
        converged = [r for r in self.search_history if r.converged]
        if not converged:
            return None
        return min(converged, key=lambda r: r.ic_value)

    def _log_best(self, best: _CandidateResult, stage: str) -> None:
        """Print current best to stdout."""
        parts = [f"[stepwise] {stage}: ARIMA{best.arima_order}"]
        if best.seasonal_order is not None:
            P, D, Q, s = best.seasonal_order
            parts.append(f"({P},{D},{Q})[{s}]")
        parts.append(f"GARCH{best.garch_order}")
        parts.append(f"{self.ic}={best.ic_value:.2f}")
        print(" ".join(parts))
