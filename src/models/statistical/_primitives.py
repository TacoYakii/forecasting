"""
Shared ARMA, SARIMA, and GARCH computation primitives.

Stateless computation classes — hold coefficients and implement core math.
No knowledge of data, training, or forecasting workflows.

Mathematical definitions implemented:

  ARMA(p,q) conditional mean:
    mu_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p}
         + theta_1*eps_{t-1} + ... + theta_q*eps_{t-q}
         + beta'*x_t

  SARIMA(p,q)(P,Q,s) conditional mean (after differencing):
    mu_t = phi_1*z_{t-1} + ... + phi_p*z_{t-p}               (AR)
         + theta_1*eps_{t-1} + ... + theta_q*eps_{t-q}         (MA)
         + Phi_1*z_{t-s} + ... + Phi_P*z_{t-P*s}              (SAR)
         + Theta_1*eps_{t-s} + ... + Theta_Q*eps_{t-Q*s}       (SMA)
         + beta'*x_t

    where eps_t = z_t - mu_t  (innovation, computed via forward recursion)

  GARCH(gp,gq) conditional variance:
    sigma^2_t = omega
              + alpha_1*eps^2_{t-1} + ... + alpha_gq*eps^2_{t-gq}
              + beta_1*sigma^2_{t-1} + ... + beta_gp*sigma^2_{t-gp}

    computed via forward loop (NOT recursion).
"""

from __future__ import annotations
import numpy as np

_EMPTY = np.array([], dtype=np.float64)


class ARMA:
    """
    ARMA(p, q) conditional-mean computation with optional exogenous variables.

    Key design: residuals (eps_t) are computed via forward recursion, not
    by simple subtraction from AR-only predictions. This ensures that the
    MA component correctly feeds back into subsequent residuals.

    Usage pattern:
        1. Set coefficients via .arma_coefficients setter.
        2. Call compute_residuals(z, x) to get (mean, residuals) arrays.
        3. For single-step prediction, call predict_step() with state.
    """

    def __init__(self, n_exog: int = 0, order: tuple = (1, 1)):
        self.p, self.q = order
        # Pre-allocated as contiguous float64 arrays
        self._phi: np.ndarray = np.zeros(self.p, dtype=np.float64)
        self._theta: np.ndarray = np.zeros(self.q, dtype=np.float64)
        self._x_coeff: np.ndarray = np.zeros(n_exog, dtype=np.float64)

    @property
    def arma_coefficients(self) -> dict:
        return {"AR": self._phi, "MA": self._theta, "X": self._x_coeff}

    @arma_coefficients.setter
    def arma_coefficients(self, v: dict) -> None:
        self._phi = np.ascontiguousarray(v["AR"], dtype=np.float64)
        self._theta = np.ascontiguousarray(v["MA"], dtype=np.float64)
        self._x_coeff = np.ascontiguousarray(v["X"], dtype=np.float64)

    def compute_residuals(
        self, z: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward-pass: compute mean and residual series for the entire sample.

        eps_t = z_t - mu_t,  where mu_t depends on z_{<t} and eps_{<t}.

        This is the ONLY correct way to compute ARMA residuals — the MA
        component requires eps_{t-1}, which itself requires eps_{t-2}, etc.
        A simple z_t - AR_pred_t ignores the MA feedback entirely.

        Args:
            z: differenced observations, shape (T,)
            x: exogenous features, shape (T, n_exog) or empty

        Returns:
            (mean, residuals) — both shape (T,)
        """
        T = len(z)
        mean = np.zeros(T)
        eps  = np.zeros(T)

        p, q = self.p, self.q
        phi, theta, x_c = self._phi, self._theta, self._x_coeff
        has_exog = len(x_c) > 0 and x.ndim == 2

        for t in range(T):
            mu = 0.0

            # AR term: phi_1*z_{t-1} + ... + phi_p*z_{t-p}
            for i in range(min(p, t)):
                mu += phi[i] * z[t - 1 - i]

            # MA term: theta_1*eps_{t-1} + ... + theta_q*eps_{t-q}
            for j in range(min(q, t)):
                mu += theta[j] * eps[t - 1 - j]

            # Exogenous term
            if has_exog:
                mu += x[t] @ x_c

            mean[t] = mu
            eps[t]  = z[t] - mu

        return mean, eps

    def predict_step(
        self,
        past_z: np.ndarray,
        past_eps: np.ndarray,
        x_t: np.ndarray,
    ) -> float:
        """
        Single-step-ahead conditional mean prediction.

        Args:
            past_z:   recent differenced observations, length >= p
            past_eps: recent innovations, length >= q
            x_t:      exogenous features at current step

        Returns:
            mu_{t} (scalar)
        """
        phi, theta, x_c = self._phi, self._theta, self._x_coeff
        mu = 0.0

        # AR: phi_1*z_{t-1} + ... + phi_p*z_{t-p}
        n_z = len(past_z)
        for i in range(min(self.p, n_z)):
            mu += phi[i] * past_z[-1 - i]

        # MA: theta_1*eps_{t-1} + ... + theta_q*eps_{t-q}
        n_e = len(past_eps)
        for j in range(min(self.q, n_e)):
            mu += theta[j] * past_eps[-1 - j]

        # Exogenous
        if len(x_c) > 0 and len(x_t) > 0:
            mu += x_t @ x_c

        return mu


class GARCH:
    """
    GARCH(gp, gq) conditional-variance computation.

    sigma^2_t = omega
              + alpha_1*eps^2_{t-1} + ... + alpha_gq*eps^2_{t-gq}   (ARCH)
              + beta_1*sigma^2_{t-1} + ... + beta_gp*sigma^2_{t-gp}  (GARCH)

    Computed via a single forward loop — no recursion.

    Convention:
      - "ARCH" (alpha):   coefficients on lagged squared residuals
      - "GARCH" (beta):   coefficients on lagged conditional variances
      - "CONSTANT" (omega): intercept
    """

    def __init__(self, order: tuple = (1, 1)):
        self.garch_p, self.garch_q = order  # gp = GARCH lags, gq = ARCH lags
        self._alpha: np.ndarray = np.zeros(self.garch_q, dtype=np.float64)
        self._beta: np.ndarray = np.zeros(self.garch_p, dtype=np.float64)
        self._omega: float = 0.0

    @property
    def garch_coefficients(self) -> dict:
        return {"GARCH": self._beta, "ARCH": self._alpha, "CONSTANT": self._omega}

    @garch_coefficients.setter
    def garch_coefficients(self, v: dict) -> None:
        self._alpha = np.ascontiguousarray(v["ARCH"], dtype=np.float64)
        self._beta = np.ascontiguousarray(v["GARCH"], dtype=np.float64)
        self._omega = float(v["CONSTANT"])

    @property
    def unconditional_variance(self) -> float:
        """omega / (1 - sum(alpha) - sum(beta)), if stationary."""
        denom = 1.0 - self._alpha.sum() - self._beta.sum()
        if denom <= 0:
            return self._omega
        return self._omega / denom

    def compute_variance_series(self, residuals: np.ndarray) -> np.ndarray:
        """
        Forward-pass: compute conditional variance series sigma^2_1, ..., sigma^2_T.

        Initialises sigma^2 for early time steps with the sample variance
        of the residuals (standard practice when pre-sample values are unknown).

        Args:
            residuals: innovation series eps_t, shape (T,)

        Returns:
            sigma2: conditional variance series, shape (T,)
        """
        alpha, beta, omega = self._alpha, self._beta, self._omega
        gp, gq = self.garch_p, self.garch_q

        T = len(residuals)
        eps2 = residuals * residuals  # pre-compute squared residuals once
        sigma2 = np.full(T, np.var(residuals) if T > 1 else omega)

        start = max(gp, gq)
        for t in range(start, T):
            s2 = omega

            # ARCH term: alpha_i * eps^2_{t-i}
            for i in range(gq):
                s2 += alpha[i] * eps2[t - 1 - i]

            # GARCH term: beta_j * sigma^2_{t-j}
            for j in range(gp):
                s2 += beta[j] * sigma2[t - 1 - j]

            sigma2[t] = s2 if s2 > 1e-10 else 1e-10

        return sigma2

    def predict_variance_step(
        self,
        past_eps2: np.ndarray,
        past_sigma2: np.ndarray,
    ) -> float:
        """
        Single-step-ahead conditional variance prediction.

        sigma^2_{t} = omega + alpha' * [eps^2_{t-1},...] + beta' * [sigma^2_{t-1},...]

        For multi-step-ahead forecasting:
          - Future eps^2_{t+h} = E[eps^2_{t+h}] = sigma^2_{t+h}  (conditional expectation)
          - Future sigma^2_{t+h} is known from previous prediction steps.

        Args:
            past_eps2:    recent squared residuals, length >= garch_q
                          (for future steps, use sigma^2 as substitute)
            past_sigma2:  recent conditional variances, length >= garch_p

        Returns:
            sigma^2_t (scalar, non-negative)
        """
        alpha, beta, omega = self._alpha, self._beta, self._omega
        s2 = omega

        n_e = len(past_eps2)
        for i in range(min(self.garch_q, n_e)):
            s2 += alpha[i] * past_eps2[-1 - i]

        n_s = len(past_sigma2)
        for j in range(min(self.garch_p, n_s)):
            s2 += beta[j] * past_sigma2[-1 - j]

        return s2 if s2 > 1e-10 else 1e-10


class SARIMA:
    """
    SARIMA(p,q)(P,Q,s) conditional-mean computation with optional exogenous.

    Extends ARMA with seasonal AR (SAR) and seasonal MA (SMA) terms.
    Residuals are computed via forward recursion, just like ARMA.

    mu_t = sum_i phi_i * z_{t-i}           (AR,  i=1..p)
         + sum_j theta_j * eps_{t-j}       (MA,  j=1..q)
         + sum_k Phi_k * z_{t-k*s}         (SAR, k=1..P)
         + sum_l Theta_l * eps_{t-l*s}     (SMA, l=1..Q)
         + beta' * x_t                     (exog)
    """

    def __init__(
        self,
        order: tuple = (1, 1),
        seasonal_order: tuple = (1, 1, 24),
        n_exog: int = 0,
    ):
        self.p, self.q = order
        self.P, self.Q, self.s = seasonal_order

        self._phi: np.ndarray = np.zeros(self.p, dtype=np.float64)
        self._theta: np.ndarray = np.zeros(self.q, dtype=np.float64)
        self._Phi: np.ndarray = np.zeros(self.P, dtype=np.float64)
        self._Theta: np.ndarray = np.zeros(self.Q, dtype=np.float64)
        self._x_coeff: np.ndarray = np.zeros(n_exog, dtype=np.float64)

    @property
    def coefficients(self) -> dict:
        return {
            "AR": self._phi, "MA": self._theta,
            "SAR": self._Phi, "SMA": self._Theta,
            "X": self._x_coeff,
        }

    @coefficients.setter
    def coefficients(self, v: dict) -> None:
        self._phi = np.ascontiguousarray(v["AR"], dtype=np.float64)
        self._theta = np.ascontiguousarray(v["MA"], dtype=np.float64)
        self._Phi = np.ascontiguousarray(v["SAR"], dtype=np.float64)
        self._Theta = np.ascontiguousarray(v["SMA"], dtype=np.float64)
        self._x_coeff = np.ascontiguousarray(v["X"], dtype=np.float64)

    def compute_residuals(
        self, z: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward-pass: compute mean and residual series for the entire sample.

        Args:
            z: differenced observations, shape (T,)
            x: exogenous features, shape (T, n_exog) or empty

        Returns:
            (mean, residuals) — both shape (T,)
        """
        T = len(z)
        mean = np.zeros(T)
        eps = np.zeros(T)

        p, q, s = self.p, self.q, self.s
        P, Q = self.P, self.Q
        phi, theta = self._phi, self._theta
        Phi, Theta = self._Phi, self._Theta
        x_c = self._x_coeff
        has_exog = len(x_c) > 0 and x.ndim == 2

        for t in range(T):
            mu = 0.0

            # AR: phi_i * z_{t-i}
            for i in range(min(p, t)):
                mu += phi[i] * z[t - 1 - i]

            # MA: theta_j * eps_{t-j}
            for j in range(min(q, t)):
                mu += theta[j] * eps[t - 1 - j]

            # SAR: Phi_k * z_{t-k*s}
            for k in range(P):
                idx = t - (k + 1) * s
                if idx >= 0:
                    mu += Phi[k] * z[idx]

            # SMA: Theta_l * eps_{t-l*s}
            for l in range(Q):
                idx = t - (l + 1) * s
                if idx >= 0:
                    mu += Theta[l] * eps[idx]

            # Exogenous
            if has_exog:
                mu += x[t] @ x_c

            mean[t] = mu
            eps[t] = z[t] - mu

        return mean, eps

    def predict_step(
        self,
        past_z: np.ndarray,
        past_eps: np.ndarray,
        x_t: np.ndarray,
    ) -> float:
        """
        Single-step-ahead conditional mean prediction.

        Args:
            past_z:   full history of differenced observations
            past_eps: full history of innovations
            x_t:      exogenous features at current step
        """
        phi, theta = self._phi, self._theta
        Phi, Theta = self._Phi, self._Theta
        x_c = self._x_coeff
        s = self.s
        mu = 0.0

        n_z = len(past_z)
        n_e = len(past_eps)

        # AR
        for i in range(min(self.p, n_z)):
            mu += phi[i] * past_z[-1 - i]

        # MA
        for j in range(min(self.q, n_e)):
            mu += theta[j] * past_eps[-1 - j]

        # SAR
        for k in range(self.P):
            idx = n_z - (k + 1) * s
            if idx >= 0:
                mu += Phi[k] * past_z[idx]

        # SMA
        for l in range(self.Q):
            idx = n_e - (l + 1) * s
            if idx >= 0:
                mu += Theta[l] * past_eps[idx]

        # Exogenous
        if len(x_c) > 0 and len(x_t) > 0:
            mu += x_t @ x_c

        return mu


# ======================================================================
# Fractional differencing utilities
# ======================================================================

def fractional_diff_weights(d: float, K: int) -> np.ndarray:
    """
    Compute fractional differencing weights via the binomial expansion.

      (1-B)^d = sum_{k=0}^{K} w_k B^k

    Recursive formula (numerically stable):
      w_0 = 1
      w_k = -w_{k-1} * (d - k + 1) / k

    Weights decay as O(k^{-d-1}) for 0 < d < 0.5, so they shrink rapidly.

    Args:
        d: fractional differencing parameter (typically -0.5 < d < 0.5)
        K: truncation length (number of lags). K=0 returns [1.0].

    Returns:
        weights: array of length K+1, dtype float64
    """
    w = np.empty(K + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, K + 1):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def fractional_diff(
    y: np.ndarray, d: float, K: int | None = None
) -> np.ndarray:
    """
    Apply fractional differencing to a series.

      z_t = sum_{k=0}^{min(K, t)} w_k * y_{t-k}

    No observations are lost — z has the same length as y.
    For the first few observations (t < K), only available lags are used.

    Args:
        y: original series, shape (T,)
        d: fractional differencing parameter
        K: truncation length. None = full sample (T-1).

    Returns:
        z: fractionally differenced series, shape (T,)
    """
    T = len(y)
    if K is None:
        K = T - 1
    K = min(K, T - 1)

    w = fractional_diff_weights(d, K)
    z = np.empty(T, dtype=np.float64)

    for t in range(T):
        n_lags = min(K, t) + 1
        # y[t], y[t-1], ..., y[t-n_lags+1] in that order
        chunk = y[t - n_lags + 1:t + 1][::-1]
        z[t] = w[:n_lags] @ chunk

    return z


def fractional_undiff(
    z: np.ndarray,
    d: float,
    y_prefix: np.ndarray,
    K: int | None = None,
) -> np.ndarray:
    """
    Inverse fractional differencing: recover original-scale values from
    fractionally differenced forecasts.

    Given the forecast z_{T+1}, ..., z_{T+H} in differenced space and the
    historical original-scale series y_1, ..., y_T (y_prefix), compute
    y_{T+1}, ..., y_{T+H} via:

      y_{T+h} = z_{T+h} - sum_{k=1}^{min(K, T+h-1)} w_k * y_{T+h-k}

    where w are the fractional diff weights for parameter d. This is just
    the rearrangement of z = sum(w_k * y_{t-k}) solved for y_t.

    Computed sequentially because each y_{T+h} depends on y_{T+h-1}.

    Args:
        z:        fractionally differenced forecast values, shape (H,)
        d:        fractional differencing parameter used during differencing
        y_prefix: historical original-scale series, shape (T,)
        K:        truncation length. None = full available history.

    Returns:
        y_forecast: original-scale forecast values, shape (H,)
    """
    H = len(z)
    T = len(y_prefix)
    if K is None:
        K = T + H - 1

    w = fractional_diff_weights(d, K)

    # Build extended y array: [y_prefix | y_forecast]
    y_ext = np.empty(T + H, dtype=np.float64)
    y_ext[:T] = y_prefix
    # w[0] = 1, so z_t = w[0]*y_t + sum_{k=1}^{...} w_k * y_{t-k}
    # => y_t = z_t - sum_{k=1}^{...} w_k * y_{t-k}
    for h in range(H):
        idx = T + h
        n_lags = min(K, idx)
        correction = 0.0
        for k in range(1, n_lags + 1):
            correction += w[k] * y_ext[idx - k]
        y_ext[idx] = z[h] - correction

    return y_ext[T:]
