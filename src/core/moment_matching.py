"""
Moment matching: convert (mu, std) to native distribution parameters.

Used by moment-only models (PGBM, XGBoost, GBoost, LR) that predict
mu and estimate std from historical data, but don't know the native
distribution parameters directly.
"""

import math
import numpy as np
from typing import Any, Dict

from .forecast_distribution import DISTRIBUTION_REGISTRY


def mu_std_to_dist_params(
    dist_name: str,
    mu: np.ndarray,
    std: np.ndarray,
    **extra_params: Any
) -> Dict[str, np.ndarray]:
    """
    Convert (mean, std) moment parameterization to native distribution parameters.

    Args:
        dist_name: Distribution name (must be in DISTRIBUTION_REGISTRY)
        mu: Mean array of shape (T,)
        std: Standard deviation array of shape (T,)
        **extra_params: Additional distribution-specific parameters
            - studentT: df (degrees of freedom, default=3)
            - weibull: c (shape parameter, default=2.0)

    Returns:
        Dict of native distribution parameters, keyed by factory argument names.

    Raises:
        ValueError: If dist_name is not supported

    Examples:
        >>> params = mu_std_to_dist_params("normal", mu=np.array([1.0]), std=np.array([0.5]))
        >>> # {"loc": array([1.0]), "scale": array([0.5])}

        >>> params = mu_std_to_dist_params("gamma", mu=np.array([2.0]), std=np.array([1.0]))
        >>> # {"a": ..., "loc": array([0.0]), "scale": ...}
    """
    if dist_name not in DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Distribution '{dist_name}' not supported. "
            f"Available: {list(DISTRIBUTION_REGISTRY.keys())}"
        )

    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    eps = 1e-9
    var = np.maximum(std ** 2, eps)

    if dist_name == "normal":
        return {"loc": mu, "scale": np.maximum(std, eps)}

    elif dist_name == "laplace":
        # Var(Laplace) = 2 * b^2  =>  b = sqrt(var/2)
        scale = np.maximum(np.sqrt(var / 2.0), eps)
        return {"loc": mu, "scale": scale}

    elif dist_name == "logistic":
        # Var(Logistic) = pi^2 * s^2 / 3  =>  s = sqrt(3*var) / pi
        scale = np.maximum(np.sqrt(3.0 * var) / np.pi, eps)
        return {"loc": mu, "scale": scale}

    elif dist_name == "studentT":
        df = float(extra_params.get("df", 3))
        if df <= 2:
            raise ValueError("Degrees of freedom must be > 2 for finite variance")
        # Var(t) = df/(df-2) * scale^2  =>  scale = sqrt(var * (df-2)/df)
        scale = np.maximum(np.sqrt(var * (df - 2.0) / df), eps)
        return {"loc": mu, "scale": scale, "df": np.full_like(mu, df)}

    elif dist_name == "lognormal":
        # Match moments of lognormal to (mu, std)
        mu_adj = np.maximum(mu, eps)
        sigma_sq = np.log1p(var / mu_adj ** 2)
        mu_ln = np.log(mu_adj) - 0.5 * sigma_sq
        sigma_ln = np.maximum(np.sqrt(sigma_sq), eps)
        # scipy lognorm: X = scale * exp(s * Z), so scale = exp(mu_ln), s = sigma_ln
        return {"s": sigma_ln, "loc": np.zeros_like(mu), "scale": np.exp(mu_ln)}

    elif dist_name == "gamma":
        # Method of moments: shape = mu^2/var, scale = var/mu
        mu_adj = np.maximum(mu, eps)
        shape = mu_adj ** 2 / var
        scale = var / mu_adj
        return {"a": np.maximum(shape, eps), "loc": np.zeros_like(mu), "scale": np.maximum(scale, eps)}

    elif dist_name == "gumbel":
        # Var(Gumbel) = pi^2 * beta^2 / 6  =>  beta = sqrt(6*var) / pi
        beta = np.maximum(np.sqrt(6.0 * var) / np.pi, eps)
        loc = mu - beta * np.euler_gamma
        return {"loc": loc, "scale": beta}

    elif dist_name == "weibull":
        # Use provided shape c (default=2), fit scale from mean
        c = float(extra_params.get("c", 2.0))
        gamma_val = math.gamma(1.0 + 1.0 / c)
        scale = np.maximum(mu / gamma_val, eps)
        return {"c": np.full_like(mu, c), "loc": np.zeros_like(mu), "scale": scale}

    elif dist_name == "poisson":
        return {"mu": np.maximum(mu, eps)}

    else:
        raise ValueError(f"Conversion not implemented for distribution '{dist_name}'")
