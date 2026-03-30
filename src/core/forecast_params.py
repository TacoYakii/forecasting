"""
ForecastParams: unified return type for model forecast() methods.

Stores distribution name, native parameters, and axis indicator.

All forecasting models return a ForecastParams instance from their
forecast() method.  It stores the distribution name, native distribution
parameters (factory-key based), and an axis indicator that clarifies the
semantic meaning of the array dimension.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal

import numpy as np


@dataclass
class ForecastParams:
    """
    Unified forecast output from a single model call.

    Attributes:
        dist_name: Distribution identifier (must match DISTRIBUTION_REGISTRY keys,
            e.g. "normal", "studentT", "gamma").
        params: Native distribution parameters keyed by factory argument names.
            All values are np.ndarray of the same shape.
            Examples:
                Normal:   {"loc": array, "scale": array}
                StudentT: {"loc": array, "scale": array, "df": array}
                Gamma:    {"a": array, "loc": array, "scale": array}
        axis: Semantic meaning of the array dimension.
            - "horizon": single origin, H-step ahead (GARCH, deep learning)
            - "cross_section": T time points, single horizon (ML models)

    Examples:
        >>> # NGBoost StudentT — native params directly from library
        >>> ForecastParams(
        ...     dist_name="studentT",
        ...     params={"loc": mu, "scale": scale, "df": df},
        ...     axis="cross_section",
        ... )

        >>> # PGBM Gamma — moment-matched inside forecast()
        >>> params = mu_std_to_dist_params("gamma", mu, std)
        >>> ForecastParams(dist_name="gamma", params=params, axis="cross_section")

        >>> # GARCH Normal — native params from MLE
        >>> ForecastParams(
        ...     dist_name="normal",
        ...     params={"loc": mu, "scale": sigma},
        ...     axis="horizon",
        ... )
    """

    dist_name: str
    params: Dict[str, np.ndarray]
    axis: Literal["horizon", "cross_section"]
