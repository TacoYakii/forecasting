"""
Forecast result containers for multi-horizon rolling forecasts.

This module provides result classes that store forecast outputs
assembled from multiple forecast() calls (rolling or per-horizon):

- BaseForecastResult: Abstract base with basis_index only
- ParametricForecastResult: parametric distribution params of shape (N_basis, H)
- QuantileForecastResult: pre-computed quantile predictions
- SampleForecastResult: raw simulation samples
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from .forecast_distribution import (
    DISTRIBUTION_REGISTRY,
    EmpiricalDistribution,
    ParametricDistribution,
    _build_frozen,
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseForecastResult(ABC):
    """
    Abstract base for multi-horizon forecast result containers.

    Subclasses store the actual forecast data (params, quantiles, or samples)
    and implement to_distribution() and to_dataframe().

    Attributes:
        basis_index (pd.Index): Time index for each forecast origin (length N).
    """

    def __init__(self, basis_index: pd.Index):
        self.basis_index = basis_index

    @property
    def horizon(self) -> int:
        """Maximum forecast horizon H (derived from stored data)."""
        return self._get_horizon()

    @abstractmethod
    def _get_horizon(self) -> int:
        ...

    @abstractmethod
    def to_distribution(self, h: int) -> Union[ParametricDistribution, EmpiricalDistribution]:
        """Extract a distribution object for a specific forecast horizon h (1-indexed)."""
        ...

    @abstractmethod
    def to_dataframe(self, h: Optional[int] = None) -> pd.DataFrame:
        """Convert to a pandas DataFrame."""
        ...

    @abstractmethod
    def mean(self) -> np.ndarray:
        """Forecast mean, shape (N, H)."""
        ...

    @abstractmethod
    def std(self) -> np.ndarray:
        """Forecast std, shape (N, H)."""
        ...

    def __len__(self) -> int:
        return len(self.basis_index)

    def _validate_h(self, h: int) -> None:
        if h < 1 or h > self.horizon:
            raise ValueError(f"Horizon h={h} out of range [1, {self.horizon}]")


# ---------------------------------------------------------------------------
# ParametricForecastResult
# ---------------------------------------------------------------------------

class ParametricForecastResult(BaseForecastResult):
    """
    Multi-horizon rolling forecast output stored as native distribution parameters.

    Stores dist_name and a params dict where each value has shape (N_basis, H).
    The params keys match the DISTRIBUTION_REGISTRY factory argument names
    (e.g. loc, scale, df for studentT).

    Use ``to_distribution(h)`` to extract a single-horizon slice as a
    ParametricDistribution for evaluation or downstream use.

    Attributes:
        dist_name (str): Distribution identifier (e.g. "normal", "studentT").
        params (Dict[str, np.ndarray]): Native distribution parameters,
            each of shape (N, H).
        basis_index (pd.Index): Time index for each basis time (length N).

    Examples:
        >>> result = runner.forecast()
        >>> result.params["loc"].shape    # (N, 24)
        >>> dist_h1 = result.to_distribution(1)   # 1-step-ahead
        >>> dist_h1.ppf([0.1, 0.5, 0.9])          # quantile forecasts
    """

    def __init__(
        self,
        dist_name: str,
        params: Dict[str, np.ndarray],
        basis_index: pd.Index,
    ):
        if dist_name not in DISTRIBUTION_REGISTRY:
            raise ValueError(
                f"Distribution '{dist_name}' not supported. "
                f"Available: {list(DISTRIBUTION_REGISTRY.keys())}"
            )

        # Validate and convert params
        params = {k: np.asarray(v, dtype=float) for k, v in params.items()}
        shapes = {k: v.shape for k, v in params.items()}
        first_shape = next(iter(shapes.values()))
        for k, s in shapes.items():
            if s != first_shape:
                raise ValueError(
                    f"All param arrays must have the same shape, "
                    f"got {s} for '{k}' vs {first_shape}"
                )
        if first_shape[0] != len(basis_index):
            raise ValueError(
                f"First dimension of params ({first_shape[0]}) must match "
                f"basis_index length ({len(basis_index)})"
            )

        super().__init__(basis_index)
        self.dist_name = dist_name
        self.params = params

    def _get_horizon(self) -> int:
        first_param = next(iter(self.params.values()))
        return first_param.shape[1] if first_param.ndim == 2 else 1

    def __repr__(self) -> str:
        N = len(self.basis_index)
        H = self.horizon
        return (
            f"ParametricForecastResult(N={N}, H={H}, "
            f"dist='{self.dist_name}', params={list(self.params.keys())})"
        )

    def to_distribution(self, h: int) -> ParametricDistribution:
        """
        Extract a ParametricDistribution for a specific forecast horizon.

        Args:
            h: Forecast horizon (1-indexed).

        Returns:
            ParametricDistribution with native params for horizon h.
        """
        self._validate_h(h)
        idx = h - 1
        params_h = {k: v[:, idx] for k, v in self.params.items()}
        return ParametricDistribution(
            dist_name=self.dist_name,
            params=params_h,
            index=self.basis_index,
        )

    def mean(self) -> np.ndarray:
        """Forecast mean, shape (N, H). Computed from the parametric distribution."""
        frozen = _build_frozen(self.dist_name, self.params)
        return frozen.mean()

    def std(self) -> np.ndarray:
        """Forecast std, shape (N, H). Computed from the parametric distribution."""
        frozen = _build_frozen(self.dist_name, self.params)
        return frozen.std()

    def to_dataframe(self, h: Optional[int] = None) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        Args:
            h: If specified, return only that horizon's (mu, std).
               If None, return all horizons with MultiIndex columns.

        Returns:
            pd.DataFrame indexed by basis_index.
        """
        mu = self.mean()
        sigma = self.std()

        if h is not None:
            self._validate_h(h)
            idx = h - 1
            return pd.DataFrame(
                {"mu": mu[:, idx], "std": sigma[:, idx]},
                index=self.basis_index,
            )

        H = self.horizon
        columns = pd.MultiIndex.from_product(
            [["mu", "std"], list(range(1, H + 1))],
            names=["metric", "horizon"],
        )
        data = np.concatenate([mu, sigma], axis=1)
        return pd.DataFrame(data, index=self.basis_index, columns=columns)


# ---------------------------------------------------------------------------
# QuantileForecastResult
# ---------------------------------------------------------------------------

class QuantileForecastResult(BaseForecastResult):
    """
    Quantile-based multi-horizon forecast output.

    Stores pre-computed quantile predictions without assuming any
    parametric distribution family.

    Attributes:
        quantiles_data (Dict[float, np.ndarray]): Mapping from quantile
            level (e.g. 0.1, 0.5, 0.9) to arrays of shape (N_basis, H).
        basis_index (pd.Index): Time index for each basis time (length N).

    Examples:
        >>> result.quantile(0.9, h=6)        # 90th percentile at 6-step-ahead
        >>> result.interval(h=6)             # (lower, upper) at 90% coverage
        >>> result.to_distribution(1)        # 1-step-ahead EmpiricalDistribution
        >>> result.quantile_levels           # [0.05, 0.1, 0.5, 0.9, 0.95]
    """

    def __init__(
        self,
        quantiles_data: Dict[float, np.ndarray],
        basis_index: pd.Index,
    ):
        if not quantiles_data:
            raise ValueError("quantiles_data must not be empty")

        # Validate shapes
        first_q = next(iter(quantiles_data.values()))
        first_q = np.asarray(first_q, dtype=float)
        for q_level, arr in quantiles_data.items():
            arr = np.asarray(arr, dtype=float)
            if arr.shape != first_q.shape:
                raise ValueError(
                    f"All quantile arrays must have the same shape, "
                    f"got {arr.shape} for q={q_level} vs {first_q.shape}"
                )

        if first_q.shape[0] != len(basis_index):
            raise ValueError(
                f"First dimension ({first_q.shape[0]}) must match "
                f"basis_index length ({len(basis_index)})"
            )

        super().__init__(basis_index)
        self._quantiles_data: Dict[float, np.ndarray] = {
            q: np.asarray(arr, dtype=float)
            for q, arr in sorted(quantiles_data.items())
        }

    def _get_horizon(self) -> int:
        first = next(iter(self._quantiles_data.values()))
        return first.shape[1] if first.ndim == 2 else 1

    def __repr__(self) -> str:
        N = len(self.basis_index)
        H = self.horizon
        q_levels = self.quantile_levels
        return (
            f"QuantileForecastResult(N={N}, H={H}, "
            f"quantiles={q_levels})"
        )

    @property
    def quantile_levels(self) -> List[float]:
        """Sorted list of available quantile levels."""
        return sorted(self._quantiles_data.keys())

    @property
    def quantiles_data(self) -> Dict[float, np.ndarray]:
        """Raw quantile data, mapping quantile level -> array (N_basis, H)."""
        return dict(self._quantiles_data)

    def mean(self) -> np.ndarray:
        """Approximate mean from median (0.5) or mean of all quantiles."""
        if 0.5 in self._quantiles_data:
            return self._quantiles_data[0.5]
        return np.mean(
            np.stack(list(self._quantiles_data.values()), axis=0), axis=0
        )

    def std(self) -> np.ndarray:
        """Approximate std from IQR or outermost quantile pair."""
        q_levels = sorted(self._quantiles_data.keys())
        if 0.25 in self._quantiles_data and 0.75 in self._quantiles_data:
            iqr = self._quantiles_data[0.75] - self._quantiles_data[0.25]
            return np.maximum(iqr / 1.349, 1e-9)
        elif len(q_levels) >= 2:
            q_lo, q_hi = q_levels[0], q_levels[-1]
            span = self._quantiles_data[q_hi] - self._quantiles_data[q_lo]
            from scipy.stats import norm
            z_span = norm.ppf(q_hi) - norm.ppf(q_lo)
            return np.maximum(span / max(z_span, 1e-9), 1e-9)
        else:
            first = next(iter(self._quantiles_data.values()))
            return np.ones_like(first) * 1e-9

    def to_distribution(self, h: int) -> EmpiricalDistribution:
        """
        Extract an EmpiricalDistribution for a specific forecast horizon.

        Args:
            h: Forecast horizon (1-indexed).

        Returns:
            EmpiricalDistribution built from stored quantile levels and values.
        """
        self._validate_h(h)
        h_idx = h - 1
        levels = np.array(self.quantile_levels)
        values = np.stack(
            [self._quantiles_data[q][:, h_idx] for q in levels], axis=1
        )  # (N_basis, Q)
        return EmpiricalDistribution(
            index=self.basis_index,
            quantile_levels=levels,
            quantile_values=values,
        )

    def quantile(self, q: Union[float, List[float]], h: int) -> np.ndarray:
        """
        Quantile value at a specific forecast horizon.

        If the requested quantile level is stored, returns it directly.
        Otherwise interpolates between the nearest available levels.

        Args:
            q: Quantile level(s) in [0, 1]. Scalar or list.
            h: Forecast horizon (1-indexed).

        Returns:
            np.ndarray: shape (N_basis,) if q is scalar,
                        shape (N_basis, len(q)) if q is a list.
        """
        self._validate_h(h)
        h_idx = h - 1
        scalar = isinstance(q, (int, float))
        q_list = [q] if scalar else list(q)

        results = []
        q_levels = self.quantile_levels
        q_values = np.stack(
            [self._quantiles_data[ql][:, h_idx] for ql in q_levels], axis=1
        )  # (N_basis, n_quantiles)

        for qi in q_list:
            if qi in self._quantiles_data:
                results.append(self._quantiles_data[qi][:, h_idx])
            else:
                results.append(
                    np.array([
                        np.interp(qi, q_levels, q_values[i])
                        for i in range(q_values.shape[0])
                    ])
                )

        if scalar:
            return results[0]
        return np.stack(results, axis=1)

    def interval(
        self, h: int, coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction interval at a specific horizon.

        Args:
            h: Forecast horizon (1-indexed).
            coverage: Interval coverage probability (default 0.9 -> 5th-95th).

        Returns:
            Tuple of (lower, upper), each shape (N_basis,).
        """
        alpha = (1.0 - coverage) / 2.0
        lower = self.quantile(alpha, h)
        upper = self.quantile(1.0 - alpha, h)
        return lower, upper

    def to_dataframe(self, h: Optional[int] = None) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame with mu and std columns.

        Args:
            h: If specified, return only that horizon.
               If None, return all horizons with MultiIndex columns.
        """
        mu = self.mean()
        sigma = self.std()

        if h is not None:
            self._validate_h(h)
            idx = h - 1
            return pd.DataFrame(
                {"mu": mu[:, idx], "std": sigma[:, idx]},
                index=self.basis_index,
            )

        H = self.horizon
        columns = pd.MultiIndex.from_product(
            [["mu", "std"], list(range(1, H + 1))],
            names=["metric", "horizon"],
        )
        data = np.concatenate([mu, sigma], axis=1)
        return pd.DataFrame(data, index=self.basis_index, columns=columns)


# ---------------------------------------------------------------------------
# SampleForecastResult
# ---------------------------------------------------------------------------

class SampleForecastResult(BaseForecastResult):
    """
    Sample-based multi-horizon forecast output.

    Stores raw simulation samples of shape (N_basis, n_samples, H).

    This class unifies outputs from:
    - ARIMA/SARIMA ``simulate_paths`` (stochastic path simulation)
    - Foundation models that produce sample-based predictions

    Attributes:
        samples (np.ndarray): Raw samples, shape (N_basis, n_samples, H).
        basis_index (pd.Index): Time index for each basis time (length N).

    Examples:
        >>> result = model.simulate_paths(n_paths=1000, horizon=24)
        >>> result.samples.shape         # (1, 1000, 24)
        >>> result.quantile(0.9, h=6)    # 90th percentile at 6-step-ahead
        >>> result.to_distribution(1)    # 1-step-ahead EmpiricalDistribution
        >>> result.path(0)               # all sample paths from first basis time
    """

    def __init__(
        self,
        samples: np.ndarray,
        basis_index: pd.Index,
    ):
        samples = np.asarray(samples, dtype=float)

        if samples.ndim != 3:
            raise ValueError(
                f"samples must be 3-D (N_basis, n_samples, H), "
                f"got shape {samples.shape}"
            )
        if samples.shape[0] != len(basis_index):
            raise ValueError(
                f"First dimension of samples ({samples.shape[0]}) must match "
                f"basis_index length ({len(basis_index)})"
            )

        super().__init__(basis_index)
        self._samples = samples

    def _get_horizon(self) -> int:
        return self._samples.shape[2]

    def __repr__(self) -> str:
        N, S, H = self._samples.shape
        return f"SampleForecastResult(N={N}, n_samples={S}, H={H})"

    @property
    def samples(self) -> np.ndarray:
        """Raw sample array, shape (N_basis, n_samples, H)."""
        return self._samples

    @property
    def n_samples(self) -> int:
        """Number of samples per (basis_time, horizon) pair."""
        return self._samples.shape[1]

    def mean(self) -> np.ndarray:
        """Sample mean, shape (N, H)."""
        return self._samples.mean(axis=1)

    def std(self) -> np.ndarray:
        """Sample std, shape (N, H)."""
        return self._samples.std(axis=1)

    def to_distribution(self, h: int) -> EmpiricalDistribution:
        """
        Extract an EmpiricalDistribution for a specific forecast horizon.

        Args:
            h: Forecast horizon (1-indexed).

        Returns:
            EmpiricalDistribution built from raw samples at horizon h.
        """
        self._validate_h(h)
        samples_h = self._samples[:, :, h - 1]  # (N_basis, n_samples)
        return EmpiricalDistribution(
            index=self.basis_index,
            samples=samples_h,
        )

    def quantile(self, q: Union[float, List[float]], h: int) -> np.ndarray:
        """
        Empirical quantile at a specific forecast horizon.

        Args:
            q: Quantile level(s) in [0, 1]. Scalar or list.
            h: Forecast horizon (1-indexed).

        Returns:
            np.ndarray: shape (N_basis,) if q is scalar,
                        shape (N_basis, len(q)) if q is a list.
        """
        self._validate_h(h)
        return np.quantile(self._samples[:, :, h - 1], q, axis=1).T

    def path(self, basis_idx: int) -> np.ndarray:
        """
        All sample paths from a single basis time.

        Args:
            basis_idx: Integer index into basis_index (0-indexed).

        Returns:
            np.ndarray: shape (n_samples, H).
        """
        return self._samples[basis_idx]

    def interval(
        self, h: int, coverage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Empirical prediction interval at a specific horizon.

        Args:
            h: Forecast horizon (1-indexed).
            coverage: Interval coverage probability (default 0.9 -> 5th-95th).

        Returns:
            Tuple of (lower, upper), each shape (N_basis,).
        """
        alpha = (1.0 - coverage) / 2.0
        lower = self.quantile(alpha, h).squeeze()
        upper = self.quantile(1.0 - alpha, h).squeeze()
        return lower, upper

    def to_dataframe(self, h: Optional[int] = None) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame with mu and std columns.

        Args:
            h: If specified, return only that horizon.
               If None, return all horizons with MultiIndex columns.
        """
        mu = self.mean()
        sigma = self.std()

        if h is not None:
            self._validate_h(h)
            idx = h - 1
            return pd.DataFrame(
                {"mu": mu[:, idx], "std": sigma[:, idx]},
                index=self.basis_index,
            )

        H = self.horizon
        columns = pd.MultiIndex.from_product(
            [["mu", "std"], list(range(1, H + 1))],
            names=["metric", "horizon"],
        )
        data = np.concatenate([mu, sigma], axis=1)
        return pd.DataFrame(data, index=self.basis_index, columns=columns)
