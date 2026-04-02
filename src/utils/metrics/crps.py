from typing import Union

import numpy as np
from numba import guvectorize
from scipy import special

# Reference list 
# https://pypi.org/project/properscoring/ 
# Evaluating Probabilistic Forecasts with scoringRules; Alexander Jordan, Fabian Krüger, Sebastian Lerch; Journal of Statistical Software (2019); 

def _standard_gaussian_dist_func(x): 
    """Distribution function of a univariate standard gaussian disribution.
    """
    return ((1.0) / np.sqrt(2.0 * np.pi)) * np.exp(-(x * x) / 2.0)


def _standard_gaussian_dens_func(x): 
    """ 
    Density function of a univaraite standard gaussian distribution. 
    """
    return special.ndtr(x) 


def crps_gaussian(mu, sigma, y):
    """CRPS for gaussian distribution with parameter (loc=mu, scale=sigma)

    Args:
        mu (_type_): loc parameter of gaussian distribution (forecast) 
        sigma (_type_): scale parameter of gaussian distribution (forecast)
        y (_type_): observed value 
    """
    stnd_ = (y - mu) / sigma
    pdf = _standard_gaussian_dist_func(stnd_) 
    cdf = _standard_gaussian_dens_func(stnd_) 
    pi_inv = 1.0 / np.sqrt(np.pi) 
    
    crps = sigma * (stnd_ * (2.0 * cdf - 1.0) + 2 * pdf - pi_inv) 
    dmu = 1 - 2 * cdf # gaussian의 경우 gradient를 미리 구해뒀으니 추후 사용 
    dsig = 2 * pdf - pi_inv 
    
    return crps 

def crps_laplace(mu, sigma, y):
    """CRPS for laplace distribution with parameter (loc=mu, scale=sigma)

    Args:
        mu (_type_): loc parameter of gaussian distribution (forecast) 
        sigma (_type_): scale parameter of gaussian distribution (forecast)
        y (_type_): observed value 
    """
    def _laplace(y):
        crps = np.abs(y) - np.exp(-np.abs(y)) - 3/4 
        return crps 

    stnd_ = (y - mu) / sigma 
    return sigma * _laplace(stnd_)  

def logistic_dens_func(x): 
    return 1.0 / (1+np.exp(-x))
        
def crps_logistic(mu, sigma, y): 
    """CRPS for logistic distribution with parameter (loc=mu, scale=sigma)

    Args:
        mu (_type_): loc parameter of gaussian distribution (forecast) 
        sigma (_type_): scale parameter of gaussian distribution (forecast)
        y (_type_): observed value 
    """
    def _logistic(y): 
        crps = y - 2 * np.log(logistic_dens_func(y)) - 1 
        return crps 
    
    stnd_ = (y - mu) / sigma 
    return sigma * _logistic(stnd_) 

    
@guvectorize(["void(float64[:], float64[:], float64[:])"], "(),(n)->()", nopython=True) # return void, input(float64 array x 3), 마지막은 result 
def _crps_numerical_method(observation, forecasts, result) -> Union[None, list]:
    obs = observation[0] # get observation 

    if np.isnan(obs):
        result[0] = np.nan # if obs == None return None 
        return

    # Count number of valid forecast 
    N = 0 
    for f in forecasts:
        if not np.isnan(f):
            N += 1
        else:
            break
    
    # if number of valid forecast == 0 return None 
    if N == 0:
        result[0] = np.nan
        return

    weight = 1.0 / N # Uniform weights -> 우리 문제에서는 보통 continuous value를 다루고 있으므로, cdf의 y값 분포는 uniform distribution이 됨. 
    #TODO: if discrete -> weight should vary

    obs_cdf = 0.0 # state of observed CDF 
    forecast_cdf = 0.0 # cummumlative sum of probability of forecast 
    prev_forecast = 0.0 # previous foercast 
    integral = 0.0 # crps value 

    for n in range(N): # for valid forecast 
        forecast = forecasts[n]

        if obs_cdf == 0.0 and obs < forecast: # The first iteration moment that forecast exceed observation. 
            integral += (obs - prev_forecast) * forecast_cdf ** 2 # left side of observation (observation > forecast)
            integral += (forecast - obs) * (forecast_cdf - 1) ** 2 # right side of observation (observaiotn < forecast) 
            obs_cdf = 1.0
        else:
            integral += (forecast - prev_forecast) * (forecast_cdf - obs_cdf) ** 2 # observation > forecast -> F(z) - 0; observation < forecast -> F(z) -1

        forecast_cdf += weight # cummulative summation of CDF 
        prev_forecast = forecast 

    if obs_cdf == 0.0: # obs < max(forecast) -> F(max(forecast)) = 1 therefore, \int^obs_max(forecast) (1-0)^2 d obs = x - max(forecast) 
        integral += obs - forecast # type: ignore

    result[0] = integral


def crps_numerical(y, forecast_samples, need_sorting=True):
    """Numerical approximation of CRPS via sorted-sample integration.

    Constructs an empirical CDF from sorted samples with uniform weights
    (1/N per sample) and computes ∫[F(x) - H(x-y)]² dx piecewise.

    Args:
        y: Observed values, shape (N,).
        forecast_samples: Forecast samples, shape (N, M).
        need_sorting: If True, sorts samples along last axis.

    Returns:
        float: Mean CRPS across all observations.

    Example:
        >>> y = np.array([1.0, 2.0])
        >>> samples = np.random.randn(2, 100)
        >>> crps_numerical(y, samples)
    """
    if need_sorting:
        sorted_forecasts = np.sort(forecast_samples, axis=-1)
    else:
        sorted_forecasts = forecast_samples
    result = np.empty(y.shape, dtype=np.float64)
    _crps_numerical_method(y, sorted_forecasts, result)

    return np.mean(result)


def pinball_loss(
    tau: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    reduction: str = "mean",
) -> Union[np.ndarray, float]:
    """Pinball (quantile) loss for multiple quantile levels.

    Args:
        tau: Quantile levels, shape (Q,).
        q: Predicted quantile values, shape (N, Q).
        y: Observed values, shape (N,).
        reduction: Output shape control.
            - "none": full loss matrix, shape (N, Q).
            - "obs": mean over quantiles per observation, shape (N,).
            - "mean": scalar mean over all.

    Returns:
        Loss values whose shape depends on reduction.

    Example:
        >>> tau = np.array([0.1, 0.5, 0.9])
        >>> q = np.array([[1.0, 2.0, 3.0]])
        >>> y = np.array([2.5])
        >>> pinball_loss(tau, q, y, reduction="obs")
        array([0.3...])
    """
    diff = y[:, None] - q  # (N, Q)
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)  # (N, Q)
    if reduction == "none":
        return loss
    elif reduction == "obs":
        return loss.mean(axis=1)  # (N,)
    elif reduction == "mean":
        return float(loss.mean())
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


def crps_quantile(
    tau: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    reduction: str = "mean",
) -> Union[np.ndarray, float]:
    """Approximate CRPS from quantile forecasts via pinball decomposition.

    CRPS ≈ 2 * ∫₀¹ S(τ, Q(τ), y) dτ, approximated by trapezoidal
    integration over the given quantile levels. Supports both uniform
    and non-uniform tau grids.

    Used for combining weight learning. For fair model comparison,
    use the ``crps()`` dispatch function instead.

    Args:
        tau: Quantile levels, shape (Q,). Must be sorted ascending.
        q: Predicted quantile values, shape (N, Q).
        y: Observed values, shape (N,).
        reduction: "obs" -> per-observation CRPS (N,),
            "mean" -> scalar mean CRPS.

    Returns:
        CRPS values whose shape depends on reduction.

    Example:
        >>> tau = np.linspace(0, 1, 101)[1:-1]
        >>> q = np.sort(np.random.randn(50, 99), axis=1)
        >>> y = np.random.randn(50)
        >>> crps_quantile(tau, q, y, reduction="mean")
    """
    tau = np.asarray(tau, dtype=float)
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    if tau.ndim != 1 or len(tau) < 2:
        raise ValueError("tau must be a 1-D array with at least 2 elements.")
    if not np.all((tau > 0) & (tau < 1)):
        raise ValueError("All tau values must be in the open interval (0, 1).")
    if not np.all(np.diff(tau) > 0):
        raise ValueError("tau must be strictly monotonically increasing.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}.")
    if q.ndim != 2 or q.shape != (len(y), len(tau)):
        raise ValueError(
            f"q must have shape (N, Q) = ({len(y)}, {len(tau)}), "
            f"got {q.shape}."
        )
    raw = pinball_loss(tau, q, y, reduction="none")  # (N, Q)
    # Trapezoidal integration: ∫ S(τ) dτ ≈ Σ 0.5*(S_i + S_{i+1}) * Δτ_i
    per_obs = 2.0 * np.trapezoid(raw, tau, axis=1)  # (N,)
    if reduction == "obs":
        return per_obs
    elif reduction == "mean":
        return float(per_obs.mean())
    else:
        raise ValueError(f"reduction must be 'obs' or 'mean', got {reduction!r}")


def crps(
    result: "BaseForecastResult",
    observed: np.ndarray,
    h: int,
    n_quantiles: int = 99,
) -> float:
    """Compute CRPS for any ForecastResult type (fair comparison).

    Converts all result types to pseudo-samples at uniform quantile
    levels, then evaluates with ``crps_numerical``. This ensures
    identical evaluation conditions across different model families.

    Conversion:
        - ParametricForecastResult: ``dist.ppf(tau)`` -> pseudo-samples
        - SampleForecastResult: ``np.percentile(samples, tau)`` -> pseudo-samples
        - QuantileForecastResult: ``dist.ppf(tau)`` -> pseudo-samples

    For SampleForecastResult, a warning is emitted when the original
    sample count significantly exceeds ``n_quantiles``, indicating
    non-trivial information loss from pseudo-sample compression.

    Args:
        result: A ForecastResult object.
        observed: Observed values, shape (N,).
        h: Forecast horizon (1-indexed).
        n_quantiles: Number of uniform quantile levels for pseudo-sample
            generation. All result types are converted to this many
            pseudo-samples to ensure identical evaluation conditions.
            Default 99.

    Returns:
        float: Mean CRPS across all observations.

    Example:
        >>> crps_val = crps(sample_result, observed, h=1)
        >>> crps_val = crps(parametric_result, observed, h=3, n_quantiles=199)
    """
    import warnings

    from src.core.forecast_results import (
        ParametricForecastResult,
        QuantileForecastResult,
        SampleForecastResult,
    )

    tau = np.linspace(0, 1, n_quantiles + 2)[1:-1]

    if isinstance(result, ParametricForecastResult):
        dist = result.to_distribution(h)
        pseudo_samples = dist.ppf(tau)  # (N, Q)

    elif isinstance(result, SampleForecastResult):
        result._validate_h(h)
        samples_h = result.samples[:, :, h - 1]  # (N, n_samples)
        n_samples = samples_h.shape[1]
        if n_samples > 2 * n_quantiles:
            warnings.warn(
                f"SampleForecastResult has {n_samples} samples but "
                f"n_quantiles={n_quantiles}: {n_samples - n_quantiles} "
                f"samples worth of resolution lost by pseudo-sample "
                f"compression. Consider increasing n_quantiles for "
                f"more accurate CRPS.",
                stacklevel=2,
            )
        pseudo_samples = np.percentile(
            samples_h, tau * 100, axis=1
        ).T  # (N, Q)

    elif isinstance(result, QuantileForecastResult):
        dist = result.to_distribution(h)
        pseudo_samples = dist.ppf(tau)  # (N, Q)

    else:
        raise TypeError(
            f"Unsupported ForecastResult type: {type(result).__name__}"
        )

    return crps_numerical(observed, pseudo_samples)