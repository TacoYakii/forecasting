"""Innovation distribution dispatch for GARCH family models.

Provides a registry-based dispatch system so that adding a new innovation
distribution (e.g. GED, Skewed-t) requires only one new class — no
modifications to ``_garch_base.py`` or any subclass.

Example:
    >>> from src.models.statistical._innovations import get_innovation
    >>> innov = get_innovation("normal")
    >>> innov.loglik(eps, sigma, np.array([]))
    -312.5
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class InnovationDist(ABC):
    """Abstract innovation distribution for MLE-based GARCH models.

    Each concrete subclass encapsulates:
    - log-likelihood computation
    - extra parameter initialisation / bounds / scales
    - forecast parameter construction
    - standardised shock sampling

    Example:
        >>> class MyDist(InnovationDist):
        ...     name = "myDist"
        ...     n_extra_params = 0
        ...     # ... implement abstract methods ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Distribution name matching ParametricForecastResult convention."""

    @property
    @abstractmethod
    def n_extra_params(self) -> int:
        """Number of MLE parameters beyond (eps, sigma)."""

    @abstractmethod
    def param_init(self) -> list[float]:
        """Initial values for distribution-specific parameters."""

    @abstractmethod
    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        """Optimisation bounds for distribution-specific parameters."""

    @abstractmethod
    def param_scales(self) -> list[float]:
        """Parscale values for distribution-specific parameters."""

    @abstractmethod
    def loglik(
        self,
        eps: np.ndarray,
        sigma: np.ndarray,
        dist_params: np.ndarray,
    ) -> float:
        """Sum of log-density values.

        Args:
            eps: Residuals.
            sigma: Conditional standard deviations.
            dist_params: Distribution-specific parameters, shape ``(n_extra_params,)``.

        Returns:
            Total log-likelihood (scalar).
        """

    @abstractmethod
    def forecast_params(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        fitted_dist_params: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Build parameter dict for ``ParametricForecastResult``.

        Args:
            mu: Forecast means, shape ``(H,)``.
            sigma: Forecast standard deviations, shape ``(H,)``.
            fitted_dist_params: Fitted distribution parameters (e.g. ``{"df": 5.2}``).

        Returns:
            Dict with ``"loc"``, ``"scale"``, and any extras, values shaped ``(1, H)``.
        """

    @abstractmethod
    def sample_shocks(
        self,
        rng: np.random.Generator,
        shape: tuple[int, ...],
        fitted_dist_params: dict[str, Any],
    ) -> np.ndarray:
        """Draw standardised (variance = 1) innovation shocks.

        Args:
            rng: NumPy random generator.
            shape: Output shape, typically ``(n_paths, horizon)``.
            fitted_dist_params: Fitted distribution parameters.

        Returns:
            Array of shocks with the requested shape.
        """

    @abstractmethod
    def extract_fitted_params(self, params: np.ndarray) -> dict[str, Any]:
        """Extract fitted distribution parameters from the parameter vector tail.

        Args:
            params: Tail slice of the full MLE parameter vector,
                    shape ``(n_extra_params,)``.

        Returns:
            Dict of named fitted parameters (e.g. ``{"df": 5.2}``).
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

INNOVATION_REGISTRY: dict[str, InnovationDist] = {}


def _register(cls: type[InnovationDist]) -> type[InnovationDist]:
    """Class decorator that instantiates and registers an InnovationDist."""
    inst = cls()
    INNOVATION_REGISTRY[inst.name] = inst
    return cls


def get_innovation(name: str) -> InnovationDist:
    """Look up a registered innovation distribution by name.

    Args:
        name: Distribution name (e.g. ``"normal"``, ``"studentT"``).

    Raises:
        ValueError: If *name* is not registered.

    Returns:
        The registered ``InnovationDist`` instance.

    Example:
        >>> get_innovation("normal").n_extra_params
        0
    """
    if name not in INNOVATION_REGISTRY:
        raise ValueError(
            f"Unknown innovation distribution '{name}'. "
            f"Available: {sorted(INNOVATION_REGISTRY)}"
        )
    return INNOVATION_REGISTRY[name]


# ---------------------------------------------------------------------------
# Concrete distributions
# ---------------------------------------------------------------------------

@_register
class NormalInnovation(InnovationDist):
    """Standard normal innovation distribution.

    No extra parameters. The GARCH conditional variance equals the true
    conditional variance directly.

    Example:
        >>> innov = NormalInnovation()
        >>> innov.n_extra_params
        0
    """

    @property
    def name(self) -> str:
        return "normal"

    @property
    def n_extra_params(self) -> int:
        return 0

    def param_init(self) -> list[float]:
        return []

    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        return []

    def param_scales(self) -> list[float]:
        return []

    def loglik(
        self, eps: np.ndarray, sigma: np.ndarray, dist_params: np.ndarray,
    ) -> float:
        standardised = eps / sigma
        return float(np.sum(
            -0.5 * np.log(2.0 * np.pi) - np.log(sigma)
            - 0.5 * standardised * standardised
        ))

    def forecast_params(
        self, mu: np.ndarray, sigma: np.ndarray, fitted_dist_params: dict,
    ) -> dict[str, np.ndarray]:
        return {
            "loc": mu.reshape(1, -1),
            "scale": np.maximum(sigma, 1e-9).reshape(1, -1),
        }

    def sample_shocks(
        self, rng: np.random.Generator, shape: tuple, fitted_dist_params: dict,
    ) -> np.ndarray:
        return rng.standard_normal(size=shape)

    def extract_fitted_params(self, params: np.ndarray) -> dict:
        return {}


@_register
class StudentTInnovation(InnovationDist):
    """Standardised Student-t innovation distribution.

    One extra parameter: degrees of freedom (df). Uses the standardised
    Student-t (variance = 1) parameterisation matching rugarch.

    Example:
        >>> innov = StudentTInnovation()
        >>> innov.n_extra_params
        1
    """

    @property
    def name(self) -> str:
        return "studentT"

    @property
    def n_extra_params(self) -> int:
        return 1

    def param_init(self) -> list[float]:
        return [5.0]

    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        return [(2.01, 100.0)]

    def param_scales(self) -> list[float]:
        return [5.0]

    def loglik(
        self, eps: np.ndarray, sigma: np.ndarray, dist_params: np.ndarray,
    ) -> float:
        df = dist_params[0]
        half_dfp1 = 0.5 * (df + 1.0)
        half_df = 0.5 * df
        standardised = eps / sigma

        log_const = (gammaln(half_dfp1) - gammaln(half_df)
                     - 0.5 * np.log((df - 2.0) * np.pi))
        log_density = (
            log_const - np.log(sigma)
            - half_dfp1 * np.log(
                1.0 + standardised * standardised / (df - 2.0),
            )
        )
        return float(np.sum(log_density))

    def forecast_params(
        self, mu: np.ndarray, sigma: np.ndarray, fitted_dist_params: dict,
    ) -> dict[str, np.ndarray]:
        df = fitted_dist_params["df"]
        return {
            "loc": mu.reshape(1, -1),
            "scale": (sigma * np.sqrt((df - 2.0) / df)).reshape(1, -1),
            "df": np.full_like(mu, df).reshape(1, -1),
        }

    def sample_shocks(
        self, rng: np.random.Generator, shape: tuple, fitted_dist_params: dict,
    ) -> np.ndarray:
        df = fitted_dist_params["df"]
        raw = rng.standard_t(df, size=shape)
        return raw * np.sqrt((df - 2.0) / df)

    def extract_fitted_params(self, params: np.ndarray) -> dict:
        return {"df": float(params[0])}


@_register
class SkewStudentTInnovation(InnovationDist):
    """Hansen (1994) standardised skewed Student-t innovation distribution.

    Two extra parameters: degrees of freedom (df) and skewness (skew).
    The distribution is standardised to mean 0, variance 1.
    When skew = 0 it reduces to the symmetric Student-t.

    Example:
        >>> innov = SkewStudentTInnovation()
        >>> innov.n_extra_params
        2
    """

    @property
    def name(self) -> str:
        return "skewStudentT"

    @property
    def n_extra_params(self) -> int:
        return 2

    def param_init(self) -> list[float]:
        return [5.0, 0.0]

    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        return [(2.01, 100.0), (-0.99, 0.99)]

    def param_scales(self) -> list[float]:
        return [5.0, 0.5]

    def loglik(
        self, eps: np.ndarray, sigma: np.ndarray, dist_params: np.ndarray,
    ) -> float:
        df = dist_params[0]
        skew = dist_params[1]
        standardised = eps / sigma

        # Hansen's constants
        c = np.exp(
            gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
            - 0.5 * np.log(np.pi * (df - 2.0))
        )
        a = 4.0 * skew * c * (df - 2.0) / (df - 1.0)
        b = np.sqrt(1.0 + 3.0 * skew ** 2 - a ** 2)

        z = b * standardised + a
        threshold = -a / b
        left = standardised < threshold
        s = np.where(left, 1.0 - skew, 1.0 + skew)

        log_density = (
            np.log(b) + np.log(c) - np.log(sigma)
            - ((df + 1.0) / 2.0) * np.log(1.0 + z ** 2 / (s ** 2 * (df - 2.0)))
        )
        return float(np.sum(log_density))

    def forecast_params(
        self, mu: np.ndarray, sigma: np.ndarray, fitted_dist_params: dict,
    ) -> dict[str, np.ndarray]:
        df = fitted_dist_params["df"]
        skew = fitted_dist_params["skew"]
        return {
            "loc": mu.reshape(1, -1),
            "scale": sigma.reshape(1, -1),
            "df": np.full_like(mu, df).reshape(1, -1),
            "skew": np.full_like(mu, skew).reshape(1, -1),
        }

    def sample_shocks(
        self, rng: np.random.Generator, shape: tuple, fitted_dist_params: dict,
    ) -> np.ndarray:
        df = fitted_dist_params["df"]
        skew = fitted_dist_params["skew"]

        # Hansen's constants
        c_val = np.exp(
            gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
            - 0.5 * np.log(np.pi * (df - 2.0))
        )
        a_val = 4.0 * skew * c_val * (df - 2.0) / (df - 1.0)
        b_val = np.sqrt(1.0 + 3.0 * skew ** 2 - a_val ** 2)

        # Inverse CDF sampling via uniform
        u = rng.uniform(size=shape)
        from src.core.forecast_distribution import skew_student_t
        return skew_student_t.ppf(u, df=df, skew=skew)

    def extract_fitted_params(self, params: np.ndarray) -> dict:
        return {"df": float(params[0]), "skew": float(params[1])}
