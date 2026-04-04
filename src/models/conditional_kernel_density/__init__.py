"""Conditional Kernel Density (CKD) estimation for wind power forecasting.

Public API:
    - ConditionalKernelDensity: CKD model with fit()/apply()
    - CKDConfig: Configuration dataclass (extends BaseConfig)
    - resolve_to_samples: Universal Result → np.ndarray converter
"""

from .ckd_types import CKDConfig, resolve_to_samples
from .model import ConditionalKernelDensity

__all__ = [
    "ConditionalKernelDensity",
    "CKDConfig",
    "resolve_to_samples",
]
