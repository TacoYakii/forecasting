"""
Configuration dataclasses for time-series models.

All configs inherit from BaseConfig to get JSON save/load support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.core.config import BaseConfig


@dataclass
class ArimaGarchConfig(BaseConfig):
    """
    Hyperparameters for ARIMA-GARCH model.

    Args:
        arima_order:  (p, d, q) — AR order, integration order, MA order.
        garch_order:  (p, q)    — GARCH lag order, ARCH lag order.
        opt_method:   Scipy optimizer method for MLE. "SLSQP" works well with
                      bounds + constraints; "L-BFGS-B" is faster but bounds-only.
        distribution: Innovation distribution for MLE and output.
                      "normal" — Gaussian likelihood.
                      "studentT" — Student-t likelihood with df jointly estimated.
                      Must be a key in DISTRIBUTION_REGISTRY.
    """
    arima_order:       Tuple[int, int, int] = field(default=(1, 0, 1))
    garch_order:       Tuple[int, int]      = field(default=(1, 1))
    opt_method:        str                  = "trust-constr"
    distribution:      str                  = "normal"
    variance_targeting: bool                = True


@dataclass
class SarimaGarchConfig(BaseConfig):
    """
    Hyperparameters for SARIMA-GARCH model.

    Extends ArimaGarchConfig with a seasonal component (P, D, Q, s).

    Args:
        arima_order:    (p, d, q)       — non-seasonal AR, integration, MA orders.
        seasonal_order: (P, D, Q, s)    — seasonal AR, integration, MA orders and period.
        garch_order:    (p, q)          — GARCH and ARCH lag orders.
        opt_method:     Scipy optimizer method for MLE.
        distribution:   Innovation distribution for MLE and output.
                        "normal" — Gaussian likelihood.
                        "studentT" — Student-t likelihood with df jointly estimated.
                        Must be a key in DISTRIBUTION_REGISTRY.
    """
    arima_order:        Tuple[int, int, int]       = field(default=(1, 0, 1))
    seasonal_order:     Tuple[int, int, int, int]  = field(default=(1, 0, 1, 24))
    garch_order:        Tuple[int, int]            = field(default=(1, 1))
    opt_method:         str                        = "SLSQP"
    distribution:       str                        = "normal"
    variance_targeting: bool                       = True


@dataclass
class ArfimaGarchConfig(BaseConfig):
    """
    Hyperparameters for ARFIMA-GARCH model.

    Unlike ARIMA where d is a fixed integer, ARFIMA estimates d via MLE.
    The fractional differencing parameter d captures long-memory dependence.

    Args:
        arfima_order:   (p, q) — AR and MA orders. d is estimated, not specified.
        garch_order:    (p, q) — GARCH and ARCH lag orders.
        d_bounds:       Bounds for fractional d. Default (-0.499, 0.499) ensures
                        stationarity and invertibility.
        truncation_K:   Truncation length for fractional diff weights.
                        Default 500 — weights decay as O(k^{-d-1}), so
                        w[500] < 0.001 for all |d| < 0.5.
                        Set None for exact (full sample) computation.
        opt_method:     Scipy optimizer method for MLE.
        distribution:   Innovation distribution ("normal" or "studentT").
        variance_targeting: If True, omega derived from residual variance.
    """
    arfima_order:       Tuple[int, int]            = field(default=(1, 1))
    garch_order:        Tuple[int, int]            = field(default=(1, 1))
    d_bounds:           Tuple[float, float]        = field(default=(-0.499, 0.499))
    truncation_K:       Optional[int]              = field(default=500)
    opt_method:         str                        = "SLSQP"
    distribution:       str                        = "normal"
    variance_targeting: bool                       = True
