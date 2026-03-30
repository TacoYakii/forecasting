"""
Data types for the Conditional Kernel Density (CKD) module.

Provides structured, torch-based dataclasses for CKD configuration
and power density estimation outputs.
"""

import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from src.core.config import BaseConfig

if TYPE_CHECKING:
    from src.core.forecast_distribution import ParametricDistribution
    from src.core.forecast_results import ParametricForecastResult, SampleForecastResult


# ---------------------------------------------------------------------------
# CKD Configuration
# ---------------------------------------------------------------------------

@dataclass
class CKDConfig(BaseConfig):
    """
    Configuration for building a Conditional Kernel Density estimate.

    Attributes:
        n_x_vars: Number of explanatory variables.
        x_bandwidth: Initial bandwidth(s) for explanatory variables.
            - float: same bandwidth for all variables.
            - List[float]: per-variable bandwidth (length must equal n_x_vars).
        y_bandwidth: Initial bandwidth for the response variable (power).
        time_decay_factor: Exponential decay factor for time weighting.
            Recent observations receive higher weight. Typical range: [0.99, 1.0].
        x_basis_points: Number of grid points per explanatory variable.
        y_basis_points: Number of grid points for the response variable.
        n_samples: Number of Monte Carlo samples used in density estimation.
    """
    n_x_vars: int = 2
    x_bandwidth: Union[float, List[float]] = 0.5
    y_bandwidth: float = 1000.0
    time_decay_factor: float = 0.995
    x_basis_points: int = 100
    y_basis_points: int = 150
    n_samples: int = 1000

    def get_x_bandwidths(self) -> List[float]:
        """Return per-variable bandwidth list."""
        if isinstance(self.x_bandwidth, (int, float)):
            return [float(self.x_bandwidth)] * self.n_x_vars
        if len(self.x_bandwidth) != self.n_x_vars:
            raise ValueError(
                f"x_bandwidth list length ({len(self.x_bandwidth)}) "
                f"must match n_x_vars ({self.n_x_vars})."
            )
        return [float(b) for b in self.x_bandwidth]

@dataclass
class PowerDensity:
    """
    Estimated power density distribution at each time step.

    This is the output of applying a CKD model to simulated explanatory
    variable data. Each row is a discrete probability distribution over
    the response variable grid.

    Attributes:
        values: Probability density tensor of shape (T, Y_size).
        y_basis: 1-D tensor of response variable grid points (Y_size,).
        time_index: Optional time index for the rows (length T).
    """
    values: torch.Tensor
    y_basis: torch.Tensor
    time_index: Optional[np.ndarray] = None

    def point_estimate(self) -> torch.Tensor:
        """Expected value (weighted mean). Returns shape (T,)."""
        return self.values @ self.y_basis

    def to_samples(self, n: int = 1000) -> torch.Tensor:
        """
        Convert discrete PDF to sorted samples via multinomial sampling.
        Compatible with `RandomCRPSLoss(y_pred, y_true)`.

        Args: 
            n: Number of samples to generate.

        Returns:
            torch.Tensor of shape (T, n), sorted along dim=-1.
        """
        probs = self.values.clamp(min=0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        sample_indices = torch.multinomial(probs.float(), n, replacement=True)
        samples = self.y_basis[sample_indices]
        return torch.sort(samples, dim=-1).values

    def to_quantiles(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete PDF to quantile values via CDF interpolation.
        Compatible with `QuantileCRPSLoss(y_pred, y_true)`.

        Args:
            q: 1-D tensor of quantile levels in [0, 1], shape (Q,).

        Returns:
            torch.Tensor of shape (T, Q).
        """
        cdf = torch.cumsum(self.values, dim=-1)
        cdf = cdf / cdf[:, -1:].clamp(min=1e-12)

        T, Q = cdf.shape[0], q.shape[0]
        result = torch.empty(T, Q, dtype=self.y_basis.dtype, device=self.y_basis.device)
        for i in range(T):
            indices = torch.searchsorted(cdf[i], q)
            indices = indices.clamp(0, len(self.y_basis) - 1)
            result[i] = self.y_basis[indices]

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for legacy compatibility."""
        df = pd.DataFrame(
            self.values.detach().cpu().numpy(),
            columns=self.y_basis.detach().cpu().numpy(),
        )
        if self.time_index is not None:
            df.index = pd.to_datetime(self.time_index)
        df.index.name = "forecast_time"
        return df


# ---------------------------------------------------------------------------
# ParametricDistribution bridge
# ---------------------------------------------------------------------------

def forecast_dist_to_samples(
    dists: "List[ParametricDistribution]",
    n_samples: int = 1000,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Convert ParametricDistribution objects to torch sample tensors.

    Args:
        dists: One ParametricDistribution per explanatory variable.
        n_samples: Number of samples per time step.
        device: Target torch device.
        seed: Random seed for reproducibility.

    Returns:
        List of torch.Tensor, each of shape (T, n_samples).
    """
    return [
        resolve_to_samples(dist, n_samples=n_samples, device=device, seed=seed)[0]
        for dist in dists
    ]


# ---------------------------------------------------------------------------
# Universal input resolver
# ---------------------------------------------------------------------------

def resolve_to_samples(
    obj: object,
    n_samples: int = 1000,
    horizon: Optional[int] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    Convert a single forecast result object to a (T, n_samples) sample tensor.

    Supports all result types from ``src.core.forecast_distribution``:

    +---------------------------------+-------------------------------------------+
    | Input type                      | Conversion                                |
    +=================================+===========================================+
    | ``torch.Tensor``                | returned as-is                            |
    | ``ParametricDistribution``        | ``.sample(n)``                            |
    | ``ParametricForecastResult``        | ``.to_distribution(h).sample(n)``         |
    | ``QuantileForecastResult``      | same (inherits ParametricForecastResult)      |
    | ``SampleForecastResult``        | ``.samples[:, :, h-1]``                   |
    +---------------------------------+-------------------------------------------+

    Args:
        obj: Forecast result object (any supported type).
        n_samples: Number of samples to draw (for distribution-based types).
        horizon: Forecast horizon (1-indexed). Required for multi-horizon
            types (ParametricForecastResult, QuantileForecastResult,
            SampleForecastResult).
        device: Target torch device.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (samples_tensor, time_index):
            - samples_tensor: shape (T, n_samples), dtype float64
            - time_index: np.ndarray extracted from the result object, or None

    Raises:
        ValueError: If horizon is missing for multi-horizon types.
        TypeError: If obj is not a supported type.
    """
    from src.core.forecast_distribution import (
        EmpiricalDistribution,
        ParametricDistribution,
    )
    from src.core.forecast_results import (
        ParametricForecastResult,
        SampleForecastResult,
    )

    if isinstance(obj, torch.Tensor):
        return obj, None

    if isinstance(obj, (ParametricDistribution, EmpiricalDistribution)):
        samples_np = obj.sample(n=n_samples, seed=seed)
        time_index = np.asarray(obj.index)
        return (
            torch.from_numpy(samples_np).to(device=device, dtype=torch.float64),
            time_index,
        )

    # SampleForecastResult before ParametricForecastResult (subclass ordering)
    if isinstance(obj, SampleForecastResult):
        if horizon is None:
            raise ValueError(
                "horizon is required for SampleForecastResult input. "
                "Specify e.g. horizon=1 for 1-step-ahead."
            )
        samples_np = obj.samples[:, :, horizon - 1]  # (N_basis, n_samples)
        time_index = np.asarray(obj.basis_index)
        return (
            torch.from_numpy(samples_np).to(device=device, dtype=torch.float64),
            time_index,
        )

    # ParametricForecastResult or QuantileForecastResult (both have to_distribution)
    from src.core.forecast_results import BaseForecastResult
    if isinstance(obj, BaseForecastResult):
        if horizon is None:
            raise ValueError(
                "horizon is required for ForecastResult input. "
                "Specify e.g. horizon=1 for 1-step-ahead."
            )
        dist = obj.to_distribution(horizon)
        samples_np = dist.sample(n=n_samples, seed=seed)
        time_index = np.asarray(dist.index)
        return (
            torch.from_numpy(samples_np).to(device=device, dtype=torch.float64),
            time_index,
        )

    raise TypeError(
        f"Unsupported input type: {type(obj).__name__}. "
        f"Expected torch.Tensor, ParametricDistribution, "
        f"ParametricForecastResult, QuantileForecastResult, or SampleForecastResult."
    )
