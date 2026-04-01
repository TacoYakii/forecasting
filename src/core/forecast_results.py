"""Forecast result containers for multi-horizon rolling forecasts.

This module provides result classes that store forecast outputs
assembled from multiple forecast() calls (rolling or per-horizon):

- BaseForecastResult: Abstract base with basis_index only
- ParametricForecastResult: parametric distribution params of shape (N_basis, H)
- QuantileForecastResult: pre-computed quantile predictions
- SampleForecastResult: raw simulation samples
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .forecast_distribution import (
    DISTRIBUTION_REGISTRY,
    EmpiricalDistribution,
    ParametricDistribution,
)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseForecastResult(ABC):
    """Abstract base for multi-horizon forecast result containers.

    Subclasses store the actual forecast data (params, quantiles, or samples)
    and implement to_distribution().

    Marginal statistics (mean, std, quantile, interval) are accessed through
    Distribution objects via ``to_distribution(h)``.

    Attributes:
        basis_index (pd.Index): Time index for each forecast origin (length N).
        model_name (str): Name of the model that produced this result.

    Examples:
        >>> result = runner.forecast()
        >>> result.model_name            # 'ArimaGarchForecaster'
        >>> dist = result.to_distribution(h=1)
        >>> dist.mean()       # (N,)
        >>> dist.ppf(0.9)     # (N,)
        >>> dist.interval()   # (lower, upper)
    """

    def __init__(self, basis_index: pd.Index, model_name: str = ""):
        self.basis_index = basis_index
        self.model_name = model_name

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

    def to_dataframe(self, h: Optional[int] = None) -> pd.DataFrame:
        """Convert to a pandas DataFrame with mu and std columns.

        Args:
            h: If specified, return only that horizon via to_distribution(h).
               If None, return all horizons with MultiIndex columns.

        Returns:
            pd.DataFrame indexed by basis_index.

        Examples:
            >>> result.to_dataframe(h=1)   # single horizon
            >>> result.to_dataframe()      # all horizons
        """
        if h is not None:
            self._validate_h(h)
            return self.to_distribution(h).to_dataframe()

        H = self.horizon
        dfs_mu = []
        dfs_std = []
        for hi in range(1, H + 1):
            dist = self.to_distribution(hi)
            dfs_mu.append(dist.mean())
            dfs_std.append(dist.std())

        mu = np.column_stack(dfs_mu)
        sigma = np.column_stack(dfs_std)
        columns = pd.MultiIndex.from_product(
            [["mu", "std"], list(range(1, H + 1))],
            names=["metric", "horizon"],
        )
        data = np.concatenate([mu, sigma], axis=1)
        return pd.DataFrame(data, index=self.basis_index, columns=columns)

    def reindex(self, idx: pd.Index) -> "BaseForecastResult":
        """Extract rows matching the given index, returning a new instance.

        Args:
            idx: Subset of basis_index to keep.

        Returns:
            New ForecastResult of the same type, with basis_index = idx.

        Raises:
            KeyError: If idx contains values not in self.basis_index.

        Example:
            >>> common = res_a.basis_index.intersection(res_b.basis_index)
            >>> res_a_aligned = res_a.reindex(common)
        """
        positions = self.basis_index.get_indexer(idx)
        missing = positions == -1
        if missing.any():
            bad = idx[missing].tolist()
            raise KeyError(f"Index values not found in basis_index: {bad}")
        return self._reindex_positions(positions, idx)

    @abstractmethod
    def _reindex_positions(
        self, positions: np.ndarray, idx: pd.Index
    ) -> "BaseForecastResult":
        """Subclass hook: slice internal data by positional indices."""
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
    """Multi-horizon rolling forecast output stored as native distribution parameters.

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
        model_name: str = "",
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

        super().__init__(basis_index, model_name)
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

    def _reindex_positions(
        self, positions: np.ndarray, idx: pd.Index
    ) -> "ParametricForecastResult":
        sliced_params = {k: v[positions, :] for k, v in self.params.items()}
        return ParametricForecastResult(
            dist_name=self.dist_name,
            params=sliced_params,
            basis_index=idx,
            model_name=self.model_name,
        )

    def to_distribution(self, h: int) -> ParametricDistribution:
        """Extract a ParametricDistribution for a specific forecast horizon.

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


# ---------------------------------------------------------------------------
# QuantileForecastResult
# ---------------------------------------------------------------------------

class QuantileForecastResult(BaseForecastResult):
    """Quantile-based multi-horizon forecast output.

    Stores pre-computed quantile predictions without assuming any
    parametric distribution family.

    Attributes:
        quantiles_data (Dict[float, np.ndarray]): Mapping from quantile
            level (e.g. 0.1, 0.5, 0.9) to arrays of shape (N_basis, H).
        basis_index (pd.Index): Time index for each basis time (length N).

    Examples:
        >>> dist = result.to_distribution(1)   # 1-step-ahead EmpiricalDistribution
        >>> dist.ppf(0.9)                      # 90th percentile
        >>> dist.interval(coverage=0.9)        # (lower, upper) at 90% coverage
        >>> result.quantile_levels             # [0.05, 0.1, 0.5, 0.9, 0.95]
    """

    def __init__(
        self,
        quantiles_data: Dict[float, np.ndarray],
        basis_index: pd.Index,
        model_name: str = "",
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

        super().__init__(basis_index, model_name)
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

    def _reindex_positions(
        self, positions: np.ndarray, idx: pd.Index
    ) -> "QuantileForecastResult":
        sliced = {q: arr[positions, :] for q, arr in self._quantiles_data.items()}
        return QuantileForecastResult(
            quantiles_data=sliced,
            basis_index=idx,
            model_name=self.model_name,
        )

    @property
    def quantile_levels(self) -> List[float]:
        """Sorted list of available quantile levels."""
        return sorted(self._quantiles_data.keys())

    @property
    def quantiles_data(self) -> Dict[float, np.ndarray]:
        """Raw quantile data, mapping quantile level -> array (N_basis, H)."""
        return dict(self._quantiles_data)

    def to_distribution(self, h: int) -> EmpiricalDistribution:
        """Extract an EmpiricalDistribution for a specific forecast horizon.

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


# ---------------------------------------------------------------------------
# SampleForecastResult
# ---------------------------------------------------------------------------

class SampleForecastResult(BaseForecastResult):
    """Sample-based multi-horizon forecast output.

    Stores raw simulation samples of shape (N_basis, n_samples, H).

    This class unifies outputs from:
    - ARIMA/SARIMA ``simulate_paths`` (stochastic path simulation)
    - Foundation models that produce sample-based predictions

    Marginal statistics are accessed via ``to_distribution(h)``.
    Joint (cross-horizon) structure is accessed via ``path()`` and ``samples``.

    Attributes:
        samples (np.ndarray): Raw samples, shape (N_basis, n_samples, H).
        basis_index (pd.Index): Time index for each basis time (length N).

    Examples:
        >>> result = model.simulate_paths(n_paths=1000, horizon=24)
        >>> result.samples.shape         # (1, 1000, 24)
        >>> result.path(0)               # all sample paths from first basis time
        >>> dist = result.to_distribution(6)
        >>> dist.ppf(0.9)                # 90th percentile at 6-step-ahead
    """

    def __init__(
        self,
        samples: np.ndarray,
        basis_index: pd.Index,
        model_name: str = "",
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

        super().__init__(basis_index, model_name)
        self._samples = samples

    def _get_horizon(self) -> int:
        return self._samples.shape[2]

    def __repr__(self) -> str:
        N, S, H = self._samples.shape
        return f"SampleForecastResult(N={N}, n_samples={S}, H={H})"

    def _reindex_positions(
        self, positions: np.ndarray, idx: pd.Index
    ) -> "SampleForecastResult":
        return SampleForecastResult(
            samples=self._samples[positions, :, :],
            basis_index=idx,
            model_name=self.model_name,
        )

    @property
    def samples(self) -> np.ndarray:
        """Raw sample array, shape (N_basis, n_samples, H)."""
        return self._samples

    @property
    def n_samples(self) -> int:
        """Number of samples per (basis_time, horizon) pair."""
        return self._samples.shape[1]

    def to_distribution(self, h: int) -> EmpiricalDistribution:
        """Extract an EmpiricalDistribution for a specific forecast horizon.

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

    def path(self, basis_idx: int) -> np.ndarray:
        """All sample paths from a single basis time.

        Provides access to the joint (cross-horizon) distribution structure
        that is lost when extracting a single-horizon Distribution.

        Args:
            basis_idx: Integer index into basis_index (0-indexed).

        Returns:
            np.ndarray: shape (n_samples, H).
        """
        return self._samples[basis_idx]
