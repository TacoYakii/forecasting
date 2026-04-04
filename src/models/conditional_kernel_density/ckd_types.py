"""Data types for the Conditional Kernel Density (CKD) module.

Provides CKDConfig dataclass and the universal input resolver
``resolve_to_samples`` (numpy-based).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from src.core.config import BaseConfig

if TYPE_CHECKING:
    from src.core.forecast_distribution import (
        GridDistribution,
        ParametricDistribution,
        QuantileDistribution,
        SampleDistribution,
    )
    from src.core.forecast_results import (
        BaseForecastResult,
        SampleForecastResult,
    )


# ---------------------------------------------------------------------------
# CKD Configuration
# ---------------------------------------------------------------------------

@dataclass
class CKDConfig(BaseConfig):
    """Configuration for building a Conditional Kernel Density estimate.

    Bandwidth and grid resolution are determined adaptively from training
    data statistics.  When used with ``CKDOptunaTrainer``, the search
    space is computed from the data's median spacing (lower bound) and
    range (upper bound).  When used standalone (``CKDClosedFormTrainer``),
    Silverman's robust rule provides a reference bandwidth.

    Attributes:
        n_x_vars: Number of explanatory variables.
        time_decay_factor: Exponential decay factor for time weighting.
            Recent observations receive higher weight. Typical range: [0.99, 1.0].
        bins_per_bandwidth: Number of grid bins per bandwidth unit.
            Controls grid resolution relative to the selected bandwidth.
        min_basis_points: Minimum grid points (safety floor).
        max_basis_points: Maximum grid points (memory cap).
        n_samples: Number of Monte Carlo samples used in density estimation.

    Example:
        >>> config = CKDConfig(n_x_vars=2)
        >>> config  # bandwidth and grid auto-determined at build() time
    """

    n_x_vars: int = 2
    time_decay_factor: float = 0.995
    bins_per_bandwidth: int = 4
    min_basis_points: int = 20
    max_basis_points: int = 500
    n_samples: int = 1000


# ---------------------------------------------------------------------------
# Universal input resolver (numpy-based)
# ---------------------------------------------------------------------------

def resolve_to_samples(
    obj: object,
    n_samples: int = 1000,
    horizon: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[pd.Index]]:
    """Convert a single forecast result object to a (T, n_samples) numpy array.

    Supports all result types from ``src.core``:

    +---------------------------------+-------------------------------------------+
    | Input type                      | Conversion                                |
    +=================================+===========================================+
    | ``np.ndarray``                  | returned as-is (must be 2-D)              |
    | ``ParametricDistribution``      | ``.sample(n)``                            |
    | ``SampleDistribution``          | ``.sample(n)``                            |
    | ``QuantileDistribution``        | ``.sample(n)``                            |
    | ``GridDistribution``            | ``.sample(n)``                            |
    | ``SampleForecastResult``        | ``.samples[:, :, h-1]``                   |
    | ``BaseForecastResult``          | ``.to_distribution(h).sample(n)``         |
    +---------------------------------+-------------------------------------------+

    Args:
        obj: Forecast result object (any supported type).
        n_samples: Number of samples to draw (for distribution-based types).
        horizon: Forecast horizon (1-indexed). Required for multi-horizon types.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (samples_array, time_index):
            - samples_array: shape (T, n_samples), dtype float64
            - time_index: pd.Index or None

    Raises:
        ValueError: If horizon is missing for multi-horizon types,
            or if array input is not 2-D.
        TypeError: If obj is not a supported type.

    Example:
        >>> samples, idx = resolve_to_samples(dist, n_samples=500, seed=42)
        >>> samples.shape  # (T, 500)
    """
    from src.core.forecast_distribution import (
        GridDistribution,
        ParametricDistribution,
        QuantileDistribution,
        SampleDistribution,
    )
    from src.core.forecast_results import (
        BaseForecastResult,
        SampleForecastResult,
    )

    if isinstance(obj, np.ndarray):
        if obj.ndim != 2:
            raise ValueError(
                f"np.ndarray input must be 2-D (T, n_samples), got shape {obj.shape}."
            )
        return obj.astype(float, copy=False), None

    if isinstance(obj, (ParametricDistribution, SampleDistribution, QuantileDistribution, GridDistribution)):
        samples_np = obj.sample(n=n_samples, seed=seed)
        time_index = obj.index
        return samples_np.astype(float, copy=False), time_index

    # SampleForecastResult before BaseForecastResult (subclass ordering)
    if isinstance(obj, SampleForecastResult):
        if horizon is None:
            raise ValueError(
                "horizon is required for SampleForecastResult input. "
                "Specify e.g. horizon=1 for 1-step-ahead."
            )
        samples_np = obj.samples[:, :, horizon - 1]
        time_index = obj.basis_index
        return samples_np.astype(float, copy=False), time_index

    if isinstance(obj, BaseForecastResult):
        if horizon is None:
            raise ValueError(
                "horizon is required for ForecastResult input. "
                "Specify e.g. horizon=1 for 1-step-ahead."
            )
        dist = obj.to_distribution(horizon)
        samples_np = dist.sample(n=n_samples, seed=seed)
        time_index = dist.index
        return samples_np.astype(float, copy=False), time_index

    raise TypeError(
        f"Unsupported input type: {type(obj).__name__}. "
        f"Expected np.ndarray, ParametricDistribution, SampleDistribution, "
        f"QuantileDistribution, GridDistribution, or ForecastResult subclass."
    )
