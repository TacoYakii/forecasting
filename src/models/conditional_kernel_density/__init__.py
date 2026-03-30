"""
Conditional Kernel Density (CKD) estimation for wind power forecasting.

Public API:
    - ConditionalKernelDensity: CKD model with fit()/predict()
    - CKDConfig: Configuration dataclass (extends BaseConfig)
    - PowerDensity: Density estimation output with conversion methods
    - resolve_to_samples: Universal Result → torch.Tensor converter
    - forecast_dist_to_samples: ParametricDistribution → torch bridge (legacy)
"""

from .ckd_types import CKDConfig, PowerDensity, forecast_dist_to_samples, resolve_to_samples
from .model import ConditionalKernelDensity

__all__ = [
    "ConditionalKernelDensity",
    "CKDConfig",
    "PowerDensity",
    "resolve_to_samples",
    "forecast_dist_to_samples",
]
