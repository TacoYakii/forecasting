"""
ParametricDistribution: Vectorized parametric probability distribution.

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

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


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
    """
    Build a frozen scipy distribution from registry entry and params.

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
    """
    Vectorized probabilistic forecast distribution over a time index.

    Stores distribution name and parameter arrays (one value per time step T).
    Provides unified PPF, sampling, CDF, and DataFrame conversion operations.
    All operations are vectorized over the time dimension and support optional
    chunked processing for very long forecast horizons.

    Attributes:
        dist_name (str): Distribution identifier (e.g., "normal", "gamma")
        params (Dict[str, np.ndarray]): Native distribution parameters, each of shape (T,)
        index (pd.Index): Forecast time index of length T
        base_idx (Optional[pd.Index]): Basis time index (original dataset index)
        chunk_size (Optional[int]): If set, ppf/sample process in chunks of this size

    Args:
        dist_name: Distribution name. Must be in DISTRIBUTION_REGISTRY.
        params: Dict mapping parameter names to arrays of shape (T,).
        index: Forecast time index.
        base_idx: Optional basis time index (for to_dataframe()).
        chunk_size: Optional chunk size for memory-efficient processing of long series.

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
        chunk_size: Optional[int] = None,
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
        self.chunk_size = chunk_size

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

    def _frozen_dist(self, start: int = 0, end: Optional[int] = None):
        """
        Build a frozen scipy distribution for the slice [start:end].

        Returns a frozen scipy.stats distribution with vectorized parameters.
        """
        if end is None:
            end = len(self)
        chunk_params = {k: v[start:end] for k, v in self.params.items()}
        return _build_frozen(self.dist_name, chunk_params)

    def _apply_chunked(self, func: Callable, *args: Any) -> np.ndarray:
        """
        Apply func(frozen_dist, *args) in chunks along the time dimension.

        If chunk_size is None, processes all at once.
        Results are concatenated along axis=0.
        """
        T = len(self)
        chunk_size = self.chunk_size

        if chunk_size is None or T <= chunk_size:
            frozen = self._frozen_dist()
            return func(frozen, *args)

        results = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            frozen = self._frozen_dist(start, end)
            results.append(func(frozen, *args))

        return np.concatenate(results, axis=0)

    @staticmethod
    def _ppf_2d(frozen, u_chunk: np.ndarray) -> np.ndarray:
        """
        Apply ppf to a 2D uniform array column-by-column.

        scipy frozen distributions with array parameters support ppf(1-D array)
        returning an array of the same shape. We iterate over the n columns of
        u_chunk and stack results.

        Args:
            frozen: scipy frozen distribution with vectorized params of shape (chunk_T,)
            u_chunk: uniform samples of shape (chunk_T, n)

        Returns:
            np.ndarray of shape (chunk_T, n)
        """
        n = u_chunk.shape[1]
        return np.stack([frozen.ppf(u_chunk[:, j]) for j in range(n)], axis=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ppf(self, q: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Percent Point Function (inverse CDF / quantile function).

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

        if scalar_input:
            def _ppf_scalar(frozen, q_val: float) -> np.ndarray:
                return frozen.ppf(q_val)
            return self._apply_chunked(_ppf_scalar, float(q_arr))
        else:
            q_arr = q_arr.ravel()  # ensure 1-D

            def _ppf_vector(frozen, q_vec: np.ndarray) -> np.ndarray:
                # frozen has shape (chunk_T,), q_vec has shape (Q,)
                # Output shape: (chunk_T, Q)
                return np.stack([frozen.ppf(qi) for qi in q_vec], axis=-1)

            return self._apply_chunked(_ppf_vector, q_arr)

    def sample(self, n: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate samples via Inverse Transform Sampling.

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

        chunk_size = self.chunk_size
        if chunk_size is None or T <= chunk_size:
            frozen = self._frozen_dist()
            return self._ppf_2d(frozen, u)

        results = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            frozen = self._frozen_dist(start, end)
            results.append(self._ppf_2d(frozen, u[start:end]))
        return np.concatenate(results, axis=0)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Cumulative Distribution Function.

        Args:
            x: Value(s) at which to evaluate CDF.
               - Scalar: returns array of shape (T,)
               - Array of shape (T,): returns array of shape (T,)

        Returns:
            np.ndarray of shape (T,): CDF values in [0, 1].
        """
        x_arr = np.asarray(x, dtype=float)

        def _cdf(frozen, x_val: np.ndarray) -> np.ndarray:
            return frozen.cdf(x_val)

        return self._apply_chunked(_cdf, x_arr)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Probability Density Function (or PMF for discrete distributions).

        Args:
            x: Value(s) at which to evaluate PDF.

        Returns:
            np.ndarray of shape (T,): PDF values.
        """
        x_arr = np.asarray(x, dtype=float)

        def _pdf(frozen, x_val: np.ndarray) -> np.ndarray:
            if hasattr(frozen, 'pdf'):
                return frozen.pdf(x_val)
            return frozen.pmf(x_val)

        return self._apply_chunked(_pdf, x_arr)

    def mean(self) -> np.ndarray:
        """
        Distribution mean for each time step.

        Returns:
            np.ndarray of shape (T,)
        """
        def _mean(frozen) -> np.ndarray:
            return frozen.mean()

        return self._apply_chunked(_mean)

    def std(self) -> np.ndarray:
        """
        Distribution standard deviation for each time step.

        Returns:
            np.ndarray of shape (T,)
        """
        def _std(frozen) -> np.ndarray:
            return frozen.std()

        return self._apply_chunked(_std)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame with columns: mu, std (and optionally basis_time).

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
        """
        Return a serializable summary of the distribution.

        Returns:
            Dict with keys: dist_name, param_names, T
        """
        return {
            "dist_name": self.dist_name,
            "param_names": list(self.params.keys()),
            "T": len(self),
        }

# ---------------------------------------------------------------------------
# EmpiricalDistribution: non-parametric distribution from samples or quantiles
# ---------------------------------------------------------------------------

class EmpiricalDistribution:
    """
    Non-parametric distribution backed by sorted samples.

    Provides the same public API as ParametricDistribution (ppf, cdf, pdf,
    sample, mean, std, to_dataframe) but without assuming any parametric
    family.  Internally stores a sorted sample array of shape (T, n_samples).

    Can be constructed from:
    - **Raw samples**: pass ``samples`` of shape (T, n_samples).
    - **Quantile levels + values**: pass ``quantile_levels`` of shape (Q,) and
      ``quantile_values`` of shape (T, Q).  A pseudo-sample of 1000 points
      per time step is generated via linear interpolation.

    Tail behaviour (quantile mode): requests outside the outermost stored
    quantile levels are clamped to the outermost values (flat extrapolation).

    Args:
        index: Forecast time index of length T.
        base_idx: Optional basis time index (for to_dataframe).
        samples: Raw sample array, shape (T, n_samples).
        quantile_levels: Sorted quantile levels, shape (Q,).
        quantile_values: Quantile values, shape (T, Q).

    Examples:
        >>> # From samples
        >>> ed = EmpiricalDistribution(index=idx, samples=samples_2d)
        >>> ed.ppf(0.5)       # median, shape (T,)
        >>> ed.cdf(0.0)       # CDF at 0, shape (T,)

        >>> # From quantiles
        >>> ed = EmpiricalDistribution(
        ...     index=idx,
        ...     quantile_levels=np.array([0.1, 0.5, 0.9]),
        ...     quantile_values=qvals,   # (T, 3)
        ... )
    """

    _N_SYNTHETIC: int = 1000  # pseudo-sample size when built from quantiles

    def __init__(
        self,
        index: pd.Index,
        base_idx: Optional[pd.Index] = None,
        *,
        samples: Optional[np.ndarray] = None,
        quantile_levels: Optional[np.ndarray] = None,
        quantile_values: Optional[np.ndarray] = None,
    ):
        has_samples = samples is not None
        has_quantiles = (quantile_levels is not None) and (quantile_values is not None)

        if has_samples == has_quantiles:
            raise ValueError(
                "Provide exactly one of: 'samples' or "
                "('quantile_levels' + 'quantile_values')."
            )

        if has_samples:
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
        else:
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
            self._sorted = self._quantiles_to_samples(
                quantile_levels, quantile_values
            )

        self.index = index
        self.base_idx = base_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _quantiles_to_samples(
        cls,
        levels: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Interpolate quantile pairs into a sorted pseudo-sample (T, N_SYNTHETIC)."""
        T = values.shape[0]
        n = cls._N_SYNTHETIC
        # uniform grid over [0, 1], clamped to outermost levels
        u = np.linspace(levels[0], levels[-1], n)
        result = np.empty((T, n), dtype=float)
        for i in range(T):
            result[i] = np.interp(u, levels, values[i])
        return result

    # ------------------------------------------------------------------
    # Public API (mirrors ParametricDistribution)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._sorted.shape[0]

    def __repr__(self) -> str:
        return (
            f"EmpiricalDistribution(T={len(self)}, "
            f"n_samples={self._sorted.shape[1]})"
        )

    def ppf(self, q: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Empirical quantile function.

        Args:
            q: Probability value(s) in [0, 1].

        Returns:
            Shape (T,) if q is scalar, (T, Q) if q is a vector.
        """
        q_arr = np.asarray(q, dtype=float)
        scalar = q_arr.ndim == 0

        # np.percentile wants percentages 0-100
        result = np.percentile(self._sorted, q_arr * 100, axis=1)
        # percentile returns shape (*q_shape, T) -> transpose to (T, *q_shape)
        if scalar:
            return result  # already (T,)
        return result.T  # (T, Q)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Empirical CDF.

        Args:
            x: Value(s) at which to evaluate. Scalar or array of shape (T,).

        Returns:
            np.ndarray of shape (T,).
        """
        x_arr = np.asarray(x, dtype=float)
        n = self._sorted.shape[1]
        # searchsorted per row
        if x_arr.ndim == 0:
            x_arr = np.broadcast_to(float(x_arr), (len(self),))
        counts = np.array([
            np.searchsorted(self._sorted[i], x_arr[i], side="right")
            for i in range(len(self))
        ])
        return counts / n

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Approximate PDF via finite-difference on the empirical CDF.

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

    def sample(self, n: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """
        Resample with replacement.

        Args:
            n: Number of samples per time step.
            seed: Random seed.

        Returns:
            np.ndarray of shape (T, n).
        """
        rng = np.random.default_rng(seed)
        T, S = self._sorted.shape
        indices = rng.integers(0, S, size=(T, n))
        return np.take_along_axis(self._sorted, indices, axis=1)

    def mean(self) -> np.ndarray:
        """Sample mean, shape (T,)."""
        return self._sorted.mean(axis=1)

    def std(self) -> np.ndarray:
        """Sample std, shape (T,)."""
        return self._sorted.std(axis=1)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to DataFrame with columns: mu, std (and optionally basis_time).
        """
        data: Dict[str, Any] = {"mu": self.mean(), "std": self.std()}
        if self.base_idx is not None:
            data = {"basis_time": self.base_idx, **data}
        return pd.DataFrame(data, index=self.index)

    def get_dist_info(self) -> Dict[str, Any]:
        """Serializable summary."""
        return {
            "dist_name": "empirical",
            "n_samples": self._sorted.shape[1],
            "T": len(self),
        }


__all__ = [
    "DISTRIBUTION_REGISTRY",
    "ParametricDistribution",
    "EmpiricalDistribution",
]
