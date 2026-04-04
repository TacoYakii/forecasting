"""ParametricDistribution: Vectorized parametric probability distribution.

This module provides the ParametricDistribution class, which wraps a
parametric distribution family (e.g. Normal, Student-t, Gamma) with
vectorized parameters over a time index of length T.

Key features:
- Stores distribution name + parameter arrays (vectorized over time T)
- PPF (quantile function): scalar or vector probability input
- Inverse Transform Sampling via ppf()
- Chunked processing for very long time series
- to_dataframe() for downstream compatibility (columns: mu, std)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Hansen's Skewed Student-t (scipy rv_continuous subclass)
# ---------------------------------------------------------------------------

class _SkewStudentT_gen(stats.rv_continuous):
    r"""Hansen (1994) standardised skewed Student-t distribution.

    Shape parameters:
        df:   degrees of freedom (> 2)
        skew: skewness parameter in (-1, 1)

    The distribution is standardised to have mean 0 and variance 1.
    When skew = 0 it reduces to the standard Student-t (also standardised
    to unit variance).

    The density is defined via a re-scaled, two-piece Student-t:

    .. math::

        g(y \mid \eta, \lambda) = \begin{cases}
          bc \left(1 + \frac{1}{\eta-2}
             \left(\frac{by+a}{1-\lambda}\right)^2\right)^{-(\eta+1)/2}
          & \text{if } y < -a/b \\[6pt]
          bc \left(1 + \frac{1}{\eta-2}
             \left(\frac{by+a}{1+\lambda}\right)^2\right)^{-(\eta+1)/2}
          & \text{if } y \geq -a/b
        \end{cases}

    Reference:
        Hansen, B.E. (1994). Autoregressive Conditional Density Estimation.
        International Economic Review, 35(3), 705-730.

    Example:
        >>> rv = skew_student_t(df=5, skew=-0.3, loc=0, scale=1)
        >>> rv.pdf(0.0)
    """

    def _argcheck(self, df, skew):
        return (df > 2) & (np.abs(skew) < 1)

    def _pdf(self, x, df, skew):
        df = np.asarray(df, dtype=float)
        skew = np.asarray(skew, dtype=float)
        x = np.asarray(x, dtype=float)

        a, b, c = self._abc(df, skew)
        threshold = -a / b

        out = np.empty_like(x)
        left = x < threshold
        right = ~left

        if np.any(left):
            xl = x[left]
            out[left] = (
                b[left] * c[left]
                * (1.0 + (b[left] * xl + a[left]) ** 2
                   / ((1.0 - skew[left]) ** 2 * (df[left] - 2.0)))
                ** (-(df[left] + 1.0) / 2.0)
            )

        if np.any(right):
            xr = x[right]
            out[right] = (
                b[right] * c[right]
                * (1.0 + (b[right] * xr + a[right]) ** 2
                   / ((1.0 + skew[right]) ** 2 * (df[right] - 2.0)))
                ** (-(df[right] + 1.0) / 2.0)
            )

        return out

    def _cdf(self, x, df, skew):
        df = np.asarray(df, dtype=float)
        skew = np.asarray(skew, dtype=float)
        x = np.asarray(x, dtype=float)

        a, b, c = self._abc(df, skew)
        threshold = -a / b

        out = np.empty_like(x)
        left = x < threshold
        right = ~left

        # Rescale to standard t(df) with unit scale sqrt((df-2)/df)
        # so that t_std ~ t(df, 0, 1) in scipy parameterisation
        scale_t = np.sqrt((df - 2.0) / df)

        if np.any(left):
            xl = x[left]
            sl = 1.0 - skew[left]
            u = (b[left] * xl + a[left]) / (sl * scale_t[left])
            out[left] = sl * stats.t.cdf(u, df[left])

        if np.any(right):
            xr = x[right]
            sr = 1.0 + skew[right]
            u = (b[right] * xr + a[right]) / (sr * scale_t[right])
            out[right] = (1.0 - skew[right]) / 2.0 + sr * (
                stats.t.cdf(u, df[right]) - 0.5
            )

        return out

    def _ppf(self, q, df, skew):
        df = np.asarray(df, dtype=float)
        skew = np.asarray(skew, dtype=float)
        q = np.asarray(q, dtype=float)

        a, b, c = self._abc(df, skew)
        q_threshold = (1.0 - skew) / 2.0
        scale_t = np.sqrt((df - 2.0) / df)

        out = np.empty_like(q)
        left = q < q_threshold
        right = ~left

        if np.any(left):
            ql = q[left]
            sl = 1.0 - skew[left]
            u = stats.t.ppf(ql / sl, df[left])
            out[left] = (sl * scale_t[left] * u - a[left]) / b[left]

        if np.any(right):
            qr = q[right]
            sr = 1.0 + skew[right]
            u = stats.t.ppf(0.5 + (qr - (1.0 - skew[right]) / 2.0) / sr,
                            df[right])
            out[right] = (sr * scale_t[right] * u - a[right]) / b[right]

        return out

    @staticmethod
    def _abc(df, skew):
        """Compute Hansen's a, b, c constants."""
        c = np.exp(
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(np.pi * (df - 2.0))
        )
        a = 4.0 * skew * c * (df - 2.0) / (df - 1.0)
        b = np.sqrt(1.0 + 3.0 * skew ** 2 - a ** 2)
        return a, b, c


skew_student_t = _SkewStudentT_gen(name="skew_student_t")


# ---------------------------------------------------------------------------
# Distribution registry
# ---------------------------------------------------------------------------
# Each entry maps a distribution name to:
#   - "scipy": the scipy.stats distribution class
#   - "params": ordered list of parameter names (must match factory kwargs)
#   - "clamp": dict of param_name → minimum value (safety floor)

_EPS = 1e-9

DISTRIBUTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "normal": {
        "scipy": stats.norm,
        "params": ["loc", "scale"],
        "clamp": {"scale": _EPS},
    },
    "laplace": {
        "scipy": stats.laplace,
        "params": ["loc", "scale"],
        "clamp": {"scale": _EPS},
    },
    "logistic": {
        "scipy": stats.logistic,
        "params": ["loc", "scale"],
        "clamp": {"scale": _EPS},
    },
    "studentT": {
        "scipy": stats.t,
        "params": ["loc", "scale", "df"],
        "clamp": {"scale": _EPS, "df": 2.01},
        "scipy_map": {"df": "df", "loc": "loc", "scale": "scale"},
    },
    "skewStudentT": {
        "scipy": skew_student_t,
        "params": ["loc", "scale", "df", "skew"],
        "clamp": {"scale": _EPS, "df": 2.01},
        "scipy_map": {"df": "df", "skew": "skew", "loc": "loc", "scale": "scale"},
    },
    "lognormal": {
        "scipy": stats.lognorm,
        "params": ["s", "loc", "scale"],
        "clamp": {"s": _EPS, "scale": _EPS},
    },
    "gamma": {
        "scipy": stats.gamma,
        "params": ["a", "loc", "scale"],
        "clamp": {"a": _EPS, "scale": _EPS},
    },
    "gumbel": {
        "scipy": stats.gumbel_r,
        "params": ["loc", "scale"],
        "clamp": {"scale": _EPS},
    },
    "weibull": {
        "scipy": stats.weibull_min,
        "params": ["c", "loc", "scale"],
        "clamp": {"c": _EPS, "scale": _EPS},
    },
    "poisson": {
        "scipy": stats.poisson,
        "params": ["mu"],
        "clamp": {"mu": _EPS},
    },
}


def _build_frozen(dist_name: str, params: Dict[str, np.ndarray]):
    """Build a frozen scipy distribution from registry entry and params.

    Applies safety clamps from the registry, then constructs the
    scipy frozen distribution with the appropriate keyword arguments.
    """
    entry = DISTRIBUTION_REGISTRY[dist_name]
    scipy_cls = entry["scipy"]
    clamp = entry.get("clamp", {})
    scipy_map = entry.get("scipy_map", {})

    kwargs = {}
    for key in entry["params"]:
        val = params[key]
        if key in clamp:
            val = np.maximum(val, clamp[key])
        scipy_key = scipy_map.get(key, key)
        kwargs[scipy_key] = val

    return scipy_cls(**kwargs)



# ---------------------------------------------------------------------------
# ParametricDistribution
# ---------------------------------------------------------------------------

class ParametricDistribution:
    """Vectorized probabilistic forecast distribution over a time index.

    Stores distribution name and parameter arrays (one value per time step T).
    Provides unified PPF, sampling, CDF, and DataFrame conversion operations.
    All operations are vectorized over the time dimension and support optional
    chunked processing for very long forecast horizons.

    Attributes:
        dist_name (str): Distribution identifier (e.g., "normal", "gamma")
        params (Dict[str, np.ndarray]): Native distribution parameters, each of shape (T,)
        index (pd.Index): Forecast time index of length T
        base_idx (Optional[pd.Index]): Basis time index (original dataset index)

    Args:
        dist_name: Distribution name. Must be in DISTRIBUTION_REGISTRY.
        params: Dict mapping parameter names to arrays of shape (T,).
        index: Forecast time index.
        base_idx: Optional basis time index (for to_dataframe()).

    Examples:
        >>> import numpy as np, pandas as pd
        >>> T = 1000
        >>> idx = pd.date_range("2023-01-01", periods=T, freq="h")
        >>> dist = ParametricDistribution(
        ...     dist_name="normal",
        ...     params={"loc": np.ones(T), "scale": 0.5 * np.ones(T)},
        ...     index=idx
        ... )
        >>> median = dist.ppf(0.5)           # shape (T,)
        >>> quantiles = dist.ppf([0.1, 0.9]) # shape (T, 2)
        >>> samples = dist.sample(1000)      # shape (T, 1000)
        >>> df = dist.to_dataframe()         # columns: mu, std (+ basis_time if set)
    """

    def __init__(
        self,
        dist_name: str,
        params: Dict[str, np.ndarray],
        index: pd.Index,
        base_idx: Optional[pd.Index] = None,
    ):
        if dist_name not in DISTRIBUTION_REGISTRY:
            raise ValueError(
                f"Distribution '{dist_name}' not supported. "
                f"Available: {list(DISTRIBUTION_REGISTRY.keys())}"
            )
        self.dist_name = dist_name
        self.params = {k: np.asarray(v, dtype=float) for k, v in params.items()}
        self.index = index
        self.base_idx = base_idx

    def __len__(self) -> int:
        """Number of time steps T."""
        first_param = next(iter(self.params.values()))
        return len(first_param)

    def __repr__(self) -> str:
        return (
            f"ParametricDistribution(dist_name='{self.dist_name}', "
            f"T={len(self)}, "
            f"params={list(self.params.keys())})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frozen_dist(self):
        """Build a frozen scipy distribution with full vectorized parameters.

        Returns a frozen scipy.stats distribution with vectorized parameters.
        """
        return _build_frozen(self.dist_name, self.params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ppf(self, q: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Percent Point Function (inverse CDF / quantile function).

        Args:
            q: Probability value(s) in [0, 1].
               - Scalar float: returns array of shape (T,)
               - List or 1-D array of shape (Q,): returns array of shape (T, Q)

        Returns:
            np.ndarray: Quantile values.
                - Shape (T,) if q is scalar
                - Shape (T, Q) if q is a vector of length Q

        Examples:
            >>> median = dist.ppf(0.5)           # shape (T,)
            >>> bounds = dist.ppf([0.1, 0.9])    # shape (T, 2)
        """
        q_arr = np.asarray(q, dtype=float)
        scalar_input = q_arr.ndim == 0
        frozen = self._frozen_dist()

        if scalar_input:
            return frozen.ppf(float(q_arr))
        else:
            q_arr = q_arr.ravel()
            return np.stack([frozen.ppf(qi) for qi in q_arr], axis=-1)

    def sample(self, n: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Generate samples via Inverse Transform Sampling.

        Draws uniform random variates U ~ Uniform(0, 1) and applies ppf(U),
        which is equivalent to direct sampling but ensures reproducibility
        and consistency with the ppf() method.

        Args:
            n: Number of samples per time step.
            seed: Random seed for reproducibility.

        Returns:
            np.ndarray of shape (T, n): Sampled values.

        Examples:
            >>> samples = dist.sample(n=1000, seed=42)
            >>> samples.shape  # (T, 1000)
        """
        T = len(self)
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0, size=(T, n))  # shape (T, n)
        frozen = self._frozen_dist()
        return np.stack([frozen.ppf(u[:, j]) for j in range(n)], axis=1)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Cumulative Distribution Function.

        Args:
            x: Value(s) at which to evaluate CDF.
               - Scalar: returns array of shape (T,)
               - Array of shape (T,): returns array of shape (T,)

        Returns:
            np.ndarray of shape (T,): CDF values in [0, 1].
        """
        x_arr = np.asarray(x, dtype=float)
        frozen = self._frozen_dist()
        return frozen.cdf(x_arr)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Probability Density Function (or PMF for discrete distributions).

        Args:
            x: Value(s) at which to evaluate PDF.

        Returns:
            np.ndarray of shape (T,): PDF values.
        """
        x_arr = np.asarray(x, dtype=float)
        frozen = self._frozen_dist()
        if hasattr(frozen, 'pdf'):
            return frozen.pdf(x_arr)
        return frozen.pmf(x_arr)

    def mean(self) -> np.ndarray:
        """Distribution mean for each time step.

        Returns:
            np.ndarray of shape (T,)
        """
        frozen = self._frozen_dist()
        return frozen.mean()

    def std(self) -> np.ndarray:
        """Distribution standard deviation for each time step.

        Returns:
            np.ndarray of shape (T,)
        """
        frozen = self._frozen_dist()
        return frozen.std()

    def interval(
        self, coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction interval based on the parametric distribution.

        Args:
            coverage: Interval coverage probability (default 0.9 -> 5th-95th).

        Returns:
            Tuple of (lower, upper), each shape (T,).

        Examples:
            >>> lower, upper = dist.interval(coverage=0.9)
            >>> lower.shape  # (T,)
        """
        alpha = (1.0 - coverage) / 2.0
        return self.ppf(alpha), self.ppf(1.0 - alpha)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a pandas DataFrame with columns: mu, std (and optionally basis_time).

        The mu and std columns represent the distribution mean and standard deviation
        for each time step, computed from the stored distribution parameters.

        Returns:
            pd.DataFrame with:
                - index: self.index (forecast time index)
                - columns: ["mu", "std"] or ["basis_time", "mu", "std"]

        Examples:
            >>> df = dist.to_dataframe()
            >>> df.columns.tolist()
            ['basis_time', 'mu', 'std']
            >>> df.to_csv("forecast.csv")
        """
        mu_vals = self.mean()
        std_vals = self.std()

        data: Dict[str, Any] = {"mu": mu_vals, "std": std_vals}

        if self.base_idx is not None:
            data = {"basis_time": self.base_idx, **data}

        return pd.DataFrame(data, index=self.index)

    def get_dist_info(self) -> Dict[str, Any]:
        """Return a serializable summary of the distribution.

        Returns:
            Dict with keys: dist_name, param_names, T
        """
        return {
            "dist_name": self.dist_name,
            "param_names": list(self.params.keys()),
            "T": len(self),
        }

# ---------------------------------------------------------------------------
# SampleDistribution
# ---------------------------------------------------------------------------

class SampleDistribution:
    """Non-parametric distribution backed by raw samples.

    CDF is a step function (ECDF): F(x) = #{samples <= x} / S.
    PPF uses ``np.percentile`` (order-statistic interpolation).
    Sampling resamples with replacement, preserving discrete support.

    Args:
        index: Forecast time index of length T.
        samples: Raw sample array, shape (T, n_samples).
        base_idx: Optional basis time index (for to_dataframe).

    Examples:
        >>> sd = SampleDistribution(index=idx, samples=samples_2d)
        >>> sd.ppf(0.5)       # median, shape (T,)
        >>> sd.cdf(0.0)       # ECDF at 0, shape (T,)
        >>> sd.sample(1000)   # resample, shape (T, 1000)
    """

    def __init__(
        self,
        index: pd.Index,
        samples: np.ndarray,
        base_idx: Optional[pd.Index] = None,
    ):
        self.index = index
        self.base_idx = base_idx
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2:
            raise ValueError(
                f"samples must be 2-D (T, n_samples), got shape {samples.shape}"
            )
        if samples.shape[0] != len(index):
            raise ValueError(
                f"samples rows ({samples.shape[0]}) != index length ({len(index)})"
            )
        self._sorted = np.sort(samples, axis=1)

    def __len__(self) -> int:
        return self._sorted.shape[0]

    def __repr__(self) -> str:
        return (
            f"SampleDistribution(T={len(self)}, "
            f"n_samples={self._sorted.shape[1]})"
        )

    def ppf(self, q: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Quantile function via ``np.percentile`` on sorted samples.

        Args:
            q: Probability value(s) in [0, 1].

        Returns:
            Shape (T,) if q is scalar, (T, Q) if q is a vector.

        Examples:
            >>> median = dist.ppf(0.5)           # shape (T,)
            >>> bounds = dist.ppf([0.1, 0.9])    # shape (T, 2)
        """
        q_arr = np.asarray(q, dtype=float)
        scalar = q_arr.ndim == 0
        result = np.percentile(self._sorted, q_arr * 100, axis=1)
        if scalar:
            return result  # (T,)
        return result.T  # (T, Q)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Step-function ECDF: F(x) = #{samples <= x} / S.

        Args:
            x: Scalar or array of shape (T,).

        Returns:
            np.ndarray of shape (T,).
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0:
            x_arr = np.broadcast_to(float(x_arr), (len(self),))
        S = self._sorted.shape[1]
        counts = np.array([
            np.searchsorted(self._sorted[i], x_arr[i], side="right")
            for i in range(len(self))
        ])
        return counts / S

    def sample(self, n: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Resample with replacement from stored samples.

        Args:
            n: Number of samples per time step.
            seed: Random seed.

        Returns:
            np.ndarray of shape (T, n).

        Examples:
            >>> samples = dist.sample(n=1000, seed=42)
        """
        rng = np.random.default_rng(seed)
        T = len(self)
        S = self._sorted.shape[1]
        indices = rng.integers(0, S, size=(T, n))
        return np.take_along_axis(self._sorted, indices, axis=1)

    def mean(self) -> np.ndarray:
        """Arithmetic mean of stored samples, shape (T,)."""
        return self._sorted.mean(axis=1)

    def std(self) -> np.ndarray:
        """Sample standard deviation, shape (T,)."""
        return self._sorted.std(axis=1)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Approximate PDF via finite-difference on the ECDF.

        Uses Silverman's rule-of-thumb bandwidth: h = 1.06 * std * n^{-1/5}.

        Args:
            x: Value(s). Scalar or array of shape (T,).

        Returns:
            np.ndarray of shape (T,).
        """
        n = self._sorted.shape[1]
        bandwidth = 1.06 * self.std() * (n ** (-0.2))
        bandwidth = np.maximum(bandwidth, 1e-9)
        return (self.cdf(x + bandwidth) - self.cdf(x - bandwidth)) / (
            2 * bandwidth
        )

    def interval(
        self, coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction interval.

        Args:
            coverage: Interval coverage probability (default 0.9 -> 5th-95th).

        Returns:
            Tuple of (lower, upper), each shape (T,).

        Examples:
            >>> lower, upper = dist.interval(coverage=0.9)
        """
        alpha = (1.0 - coverage) / 2.0
        return self.ppf(alpha), self.ppf(1.0 - alpha)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with columns: mu, std (and optionally basis_time)."""
        data: Dict[str, Any] = {"mu": self.mean(), "std": self.std()}
        if self.base_idx is not None:
            data = {"basis_time": self.base_idx, **data}
        return pd.DataFrame(data, index=self.index)

    def get_dist_info(self) -> Dict[str, Any]:
        """Serializable summary."""
        return {
            "dist_name": "sample",
            "n_samples": self._sorted.shape[1],
            "T": len(self),
        }


# ---------------------------------------------------------------------------
# QuantileDistribution
# ---------------------------------------------------------------------------

class QuantileDistribution:
    """Non-parametric distribution backed by quantile levels and values.

    CDF is piecewise-linear, interpolating between stored (value, level) pairs.
    PPF interpolates between (level, value) pairs.
    Mean and std are computed via trapezoidal integration of Q(u) over [0, 1]
    with linear tail extrapolation.

    Args:
        index: Forecast time index of length T.
        quantile_levels: Sorted quantile levels, shape (Q,).
        quantile_values: Quantile values, shape (T, Q).
        base_idx: Optional basis time index (for to_dataframe).

    Examples:
        >>> qd = QuantileDistribution(
        ...     index=idx,
        ...     quantile_levels=np.array([0.1, 0.5, 0.9]),
        ...     quantile_values=qvals,   # (T, 3)
        ... )
        >>> qd.ppf(0.5)       # median, shape (T,)
        >>> qd.cdf(0.0)       # piecewise-linear CDF at 0, shape (T,)
    """

    def __init__(
        self,
        index: pd.Index,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
        base_idx: Optional[pd.Index] = None,
    ):
        self.index = index
        self.base_idx = base_idx
        quantile_levels = np.asarray(quantile_levels, dtype=float)
        quantile_values = np.asarray(quantile_values, dtype=float)
        if quantile_levels.ndim != 1:
            raise ValueError("quantile_levels must be 1-D")
        if quantile_values.ndim != 2:
            raise ValueError("quantile_values must be 2-D (T, Q)")
        if quantile_values.shape[1] != len(quantile_levels):
            raise ValueError(
                f"quantile_values columns ({quantile_values.shape[1]}) "
                f"!= quantile_levels length ({len(quantile_levels)})"
            )
        if quantile_values.shape[0] != len(index):
            raise ValueError(
                f"quantile_values rows ({quantile_values.shape[0]}) "
                f"!= index length ({len(index)})"
            )
        self._sorted = np.sort(quantile_values, axis=1)
        self._levels = quantile_levels

    def __len__(self) -> int:
        return self._sorted.shape[0]

    def __repr__(self) -> str:
        return (
            f"QuantileDistribution(T={len(self)}, "
            f"Q={self._sorted.shape[1]})"
        )

    def ppf(self, q: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Quantile function via level-aware linear interpolation.

        Args:
            q: Probability value(s) in [0, 1].

        Returns:
            Shape (T,) if q is scalar, (T, Q) if q is a vector.

        Examples:
            >>> median = dist.ppf(0.5)           # shape (T,)
            >>> bounds = dist.ppf([0.1, 0.9])    # shape (T, 2)
        """
        q_arr = np.asarray(q, dtype=float)
        scalar = q_arr.ndim == 0
        if scalar:
            q_arr = q_arr.reshape(1)
        else:
            q_arr = q_arr.ravel()

        idx = np.searchsorted(self._levels, q_arr, side="right")  # (Q,)
        idx = np.clip(idx, 1, len(self._levels) - 1)

        lo = idx - 1
        hi = idx
        denom = self._levels[hi] - self._levels[lo]
        denom = np.where(denom > 0, denom, 1.0)
        t = (q_arr - self._levels[lo]) / denom  # (Q,)
        t = np.clip(t, 0.0, 1.0)

        result = self._sorted[:, lo] * (1 - t) + self._sorted[:, hi] * t

        if scalar:
            return result[:, 0]  # (T,)
        return result  # (T, Q)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Piecewise-linear CDF via interpolation between (value, level) pairs.

        Args:
            x: Scalar or array of shape (T,).

        Returns:
            np.ndarray of shape (T,).
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0:
            x_arr = np.broadcast_to(float(x_arr), (len(self),))
        result = np.empty(len(self), dtype=float)
        for i in range(len(self)):
            vals = self._sorted[i]
            lvls = self._levels
            unique_mask = np.diff(vals, append=np.inf) > 0
            result[i] = np.interp(
                x_arr[i], vals[unique_mask], lvls[unique_mask],
                left=0.0, right=1.0,
            )
        return result

    def sample(self, n: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Inverse transform sampling via per-row interpolation.

        Args:
            n: Number of samples per time step.
            seed: Random seed.

        Returns:
            np.ndarray of shape (T, n).

        Examples:
            >>> samples = dist.sample(n=1000, seed=42)
        """
        rng = np.random.default_rng(seed)
        T = len(self)
        u = rng.uniform(0.0, 1.0, size=(T, n))
        return np.array([
            np.interp(u[i], self._levels, self._sorted[i])
            for i in range(T)
        ])

    def mean(self) -> np.ndarray:
        """Mean via trapezoidal integration of Q(u) over [0, 1].

        Includes linear tail extrapolation beyond the outermost quantile
        levels.  Shape (T,).
        """
        du = np.diff(self._levels)
        midvals = (self._sorted[:, :-1] + self._sorted[:, 1:]) / 2
        interior = (midvals * du).sum(axis=1)

        tau1 = self._levels[0]
        slope_left = (
            (self._sorted[:, 1] - self._sorted[:, 0])
            / (self._levels[1] - self._levels[0])
        )
        left_tail = tau1 * self._sorted[:, 0] - slope_left * tau1**2 / 2

        tau_last = self._levels[-1]
        slope_right = (
            (self._sorted[:, -1] - self._sorted[:, -2])
            / (self._levels[-1] - self._levels[-2])
        )
        right_tail = (
            (1 - tau_last) * self._sorted[:, -1]
            + slope_right * (1 - tau_last) ** 2 / 2
        )

        return interior + left_tail + right_tail

    def std(self) -> np.ndarray:
        """Std via E[X^2] - E[X]^2 with tail extrapolation.  Shape (T,).

        Mirrors the tail treatment in ``mean()``: Q(u) is linearly
        extrapolated beyond [τ₁, τ_Q], and ∫Q(u)² du is computed
        analytically for the tail segments.
        """
        mu = self.mean()

        # --- Interior E[X²]: trapezoidal rule over observed levels ---
        du = np.diff(self._levels)
        midvals_sq = (self._sorted[:, :-1] ** 2 + self._sorted[:, 1:] ** 2) / 2
        interior = (midvals_sq * du).sum(axis=1)

        # --- Left tail [0, τ₁] ---
        # Q(u) = a + s·(u - τ₁),  a = Q(τ₁), s = slope_left
        # ∫₀^τ₁ Q(u)² du = a²·τ₁ - a·s·τ₁² + s²·τ₁³/3
        tau1 = self._levels[0]
        a = self._sorted[:, 0]
        s_l = (
            (self._sorted[:, 1] - self._sorted[:, 0])
            / (self._levels[1] - self._levels[0])
        )
        left_tail = a**2 * tau1 - a * s_l * tau1**2 + s_l**2 * tau1**3 / 3

        # --- Right tail [τ_Q, 1] ---
        # Q(u) = b + s·(u - τ_Q),  b = Q(τ_Q), δ = 1 - τ_Q
        # ∫_{τ_Q}^1 Q(u)² du = b²·δ + b·s·δ² + s²·δ³/3
        delta = 1.0 - self._levels[-1]
        b = self._sorted[:, -1]
        s_r = (
            (self._sorted[:, -1] - self._sorted[:, -2])
            / (self._levels[-1] - self._levels[-2])
        )
        right_tail = b**2 * delta + b * s_r * delta**2 + s_r**2 * delta**3 / 3

        ex2 = interior + left_tail + right_tail
        return np.sqrt(np.maximum(ex2 - mu ** 2, 0.0))

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Approximate PDF via finite-difference on the piecewise-linear CDF.

        Uses Silverman's rule-of-thumb bandwidth: h = 1.06 * std * n^{-1/5}.

        Args:
            x: Value(s). Scalar or array of shape (T,).

        Returns:
            np.ndarray of shape (T,).
        """
        n = self._sorted.shape[1]
        bandwidth = 1.06 * self.std() * (n ** (-0.2))
        bandwidth = np.maximum(bandwidth, 1e-9)
        return (self.cdf(x + bandwidth) - self.cdf(x - bandwidth)) / (
            2 * bandwidth
        )

    def interval(
        self, coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction interval.

        Args:
            coverage: Interval coverage probability (default 0.9 -> 5th-95th).

        Returns:
            Tuple of (lower, upper), each shape (T,).

        Examples:
            >>> lower, upper = dist.interval(coverage=0.9)
        """
        alpha = (1.0 - coverage) / 2.0
        return self.ppf(alpha), self.ppf(1.0 - alpha)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with columns: mu, std (and optionally basis_time)."""
        data: Dict[str, Any] = {"mu": self.mean(), "std": self.std()}
        if self.base_idx is not None:
            data = {"basis_time": self.base_idx, **data}
        return pd.DataFrame(data, index=self.index)

    def get_dist_info(self) -> Dict[str, Any]:
        """Serializable summary."""
        return {
            "dist_name": "quantile",
            "Q": self._sorted.shape[1],
            "T": len(self),
        }


class GridDistribution:
    """Non-parametric distribution backed by a discrete histogram grid.

    Stores a fixed, equally-spaced grid of bin centers and per-time-step
    probability vectors.  Each grid point represents the center of a bin
    whose edges lie at the midpoints of adjacent centers (histogram
    interpretation).

    All statistics (mean, ppf, cdf) are computed from the grid
    without Monte-Carlo sampling.  The CDF is piecewise-linear between
    bin edges; the density is piecewise-constant within each bin.

    Attributes:
        grid: Bin centers, shape (G,). Sorted ascending, equally spaced.
        prob: Probability weights, shape (T, G). Each row sums to 1.
        index: Time index of length T.
        base_idx: Optional basis time index (for to_dataframe).

    Args:
        index: Forecast time index of length T.
        grid: Equally-spaced bin centers, shape (G,).
        prob: Probability weights, shape (T, G).
        base_idx: Optional basis time index.

    Examples:
        >>> import numpy as np, pandas as pd
        >>> grid = np.linspace(0, 100, 51)
        >>> prob = np.ones((10, 51)) / 51
        >>> idx = pd.RangeIndex(10)
        >>> gd = GridDistribution(idx, grid, prob)
        >>> gd.mean()   # shape (10,)
        >>> gd.ppf(0.5)  # shape (10,)
    """

    def __init__(
        self,
        index: pd.Index,
        grid: np.ndarray,
        prob: np.ndarray,
        base_idx: Optional[pd.Index] = None,
    ):
        grid = np.asarray(grid, dtype=float)
        prob = np.asarray(prob, dtype=float)

        # --- grid validation ---
        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError(
                f"grid must be 1-D with at least 2 points, got shape {grid.shape}."
            )
        if not np.all(np.isfinite(grid)):
            raise ValueError("grid contains non-finite values.")
        diffs = np.diff(grid)
        if np.any(diffs <= 0):
            raise ValueError("grid must be strictly increasing.")
        bin_width = diffs[0]
        if not np.allclose(diffs, bin_width, rtol=1e-6):
            raise ValueError(
                "grid must be equally spaced. "
                f"Expected step {bin_width:.6g}, got range "
                f"[{diffs.min():.6g}, {diffs.max():.6g}]."
            )

        # --- prob validation ---
        if prob.ndim != 2:
            raise ValueError(f"prob must be 2-D, got shape {prob.shape}.")
        if prob.shape != (len(index), len(grid)):
            raise ValueError(
                f"prob shape {prob.shape} inconsistent with "
                f"index length {len(index)} and grid length {len(grid)}."
            )
        if not np.all(np.isfinite(prob)):
            raise ValueError("prob contains non-finite values.")
        if np.any(prob < 0):
            raise ValueError("prob contains negative values.")
        row_sums = prob.sum(axis=1)
        max_dev = np.max(np.abs(row_sums - 1.0))
        if max_dev > 1e-4:
            raise ValueError(
                f"prob row sums deviate from 1.0 by up to {max_dev:.6g} "
                f"(threshold 1e-4). Check upstream normalization."
            )
        # auto-renormalize small float deviations
        if max_dev > 0:
            prob = prob / row_sums[:, None]

        self.grid = grid
        self.prob = prob
        self.index = index
        self.base_idx = base_idx

        # derived constants
        G = len(grid)
        self._bin_width = float(bin_width)
        self._edges = np.linspace(
            grid[0] - self._bin_width / 2,
            grid[-1] + self._bin_width / 2,
            G + 1,
        )
        # CDF at edges: (T, G+1)  F(e_0)=0, F(e_i)=cumsum
        cdf = np.cumsum(prob, axis=1)
        self._cdf_at_edges = np.concatenate(
            [np.zeros((len(index), 1)), cdf], axis=1
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.prob.shape[0]

    def __repr__(self) -> str:
        return (
            f"GridDistribution(T={len(self)}, G={len(self.grid)}, "
            f"range=[{self.grid[0]:.4g}, {self.grid[-1]:.4g}])"
        )

    # ------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------

    def mean(self) -> np.ndarray:
        """Distribution mean, shape (T,).

        Example:
            >>> gd.mean()  # array of T expected values
        """
        return self.prob @ self.grid

    def std(self) -> np.ndarray:
        """Distribution standard deviation with within-bin correction, shape (T,).

        For histogram bins of width Δ, Var[X] includes a within-bin term Δ²/12.

        Example:
            >>> gd.std()  # array of T std values
        """
        mu = self.mean()
        ex2 = self.prob @ (self.grid ** 2) + self._bin_width ** 2 / 12
        return np.sqrt(np.maximum(ex2 - mu ** 2, 0.0))

    # ------------------------------------------------------------------
    # CDF / PDF / PPF
    # ------------------------------------------------------------------

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Piecewise-linear CDF evaluated at x.

        Args:
            x: Scalar or array of shape (T,).

        Returns:
            np.ndarray of shape (T,).

        Example:
            >>> gd.cdf(50.0)  # shape (T,)
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0:
            x_arr = np.broadcast_to(float(x_arr), (len(self),))
        edges = self._edges
        T = len(self)
        result = np.empty(T, dtype=float)
        for i in range(T):
            result[i] = np.interp(x_arr[i], edges, self._cdf_at_edges[i])
        return result

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Piecewise-constant PDF evaluated at x.

        Within each bin, density = prob[i] / bin_width.
        Outside the support, density = 0.

        Args:
            x: Scalar or array of shape (T,).

        Returns:
            np.ndarray of shape (T,).

        Example:
            >>> gd.pdf(50.0)  # shape (T,)
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0:
            x_arr = np.broadcast_to(float(x_arr), (len(self),))
        edges = self._edges
        # bin index: which bin does x fall into?
        bin_idx = np.searchsorted(edges, x_arr, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, len(self.grid) - 1)
        # outside support or NaN → 0 (NaN comparisons yield False)
        nan_mask = np.isnan(x_arr)
        outside = nan_mask | (x_arr < edges[0]) | (x_arr > edges[-1])
        result = self.prob[np.arange(len(self)), bin_idx] / self._bin_width
        result[outside] = 0.0
        return result

    def ppf(self, q: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """Quantile function via CDF inverse interpolation.

        Args:
            q: Probability level(s) in [0, 1]. Scalar or array.

        Returns:
            Shape (T,) if q is scalar, (T, Q) if q is a vector.

        Example:
            >>> gd.ppf(0.5)          # median, shape (T,)
            >>> gd.ppf([0.1, 0.9])   # shape (T, 2)
        """
        q_arr = np.asarray(q, dtype=float)
        scalar = q_arr.ndim == 0
        if scalar:
            q_arr = q_arr.reshape(1)
        else:
            q_arr = q_arr.ravel()

        edges = self._edges
        T = len(self)
        Q = len(q_arr)
        result = np.empty((T, Q), dtype=float)
        for i in range(T):
            result[i] = np.interp(q_arr, self._cdf_at_edges[i], edges)

        if scalar:
            return result[:, 0]
        return result

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Generate samples via multinomial + uniform jitter within bins.

        Args:
            n: Number of samples per time step.
            seed: Random seed.

        Returns:
            np.ndarray of shape (T, n).

        Example:
            >>> samples = gd.sample(500, seed=42)
            >>> samples.shape  # (T, 500)
        """
        rng = np.random.default_rng(seed)
        T, G = self.prob.shape
        bw = self._bin_width

        # multinomial bin selection
        samples = np.empty((T, n), dtype=float)
        for i in range(T):
            bins = rng.choice(G, size=n, p=self.prob[i])
            jitter = rng.uniform(-bw / 2, bw / 2, size=n)
            samples[i] = self.grid[bins] + jitter
        return samples

    # ------------------------------------------------------------------
    # Interval
    # ------------------------------------------------------------------

    def interval(
        self, coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction interval from the histogram CDF.

        Args:
            coverage: Interval coverage probability (default 0.9).

        Returns:
            Tuple of (lower, upper), each shape (T,).

        Example:
            >>> lower, upper = gd.interval(0.9)
        """
        alpha = (1.0 - coverage) / 2.0
        return self.ppf(alpha), self.ppf(1.0 - alpha)

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to summary DataFrame with columns: mu, std.

        Follows the same contract as ParametricDistribution and
        SampleDistribution / QuantileDistribution.

        Example:
            >>> df = gd.to_dataframe()
            >>> df.columns.tolist()  # ['mu', 'std'] or ['basis_time', 'mu', 'std']
        """
        data: Dict[str, Any] = {"mu": self.mean(), "std": self.std()}
        if self.base_idx is not None:
            data = {"basis_time": self.base_idx, **data}
        return pd.DataFrame(data, index=self.index)

    def to_grid_dataframe(self) -> pd.DataFrame:
        """Convert to full grid × time probability matrix.

        Returns:
            pd.DataFrame with index = time, columns = grid values.

        Example:
            >>> df = gd.to_grid_dataframe()
            >>> df.shape  # (T, G)
        """
        df = pd.DataFrame(self.prob, columns=self.grid, index=self.index)
        df.index.name = "forecast_time"
        return df

    def get_dist_info(self) -> Dict[str, Any]:
        """Serializable summary.

        Example:
            >>> gd.get_dist_info()
            {'dist_name': 'grid', 'G': 51, 'T': 10, ...}
        """
        return {
            "dist_name": "grid",
            "G": len(self.grid),
            "T": len(self),
            "bin_width": self._bin_width,
            "range": [float(self._edges[0]), float(self._edges[-1])],
        }


__all__ = [
    "DISTRIBUTION_REGISTRY",
    "ParametricDistribution",
    "SampleDistribution",
    "QuantileDistribution",
    "GridDistribution",
]
