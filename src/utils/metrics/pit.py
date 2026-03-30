import numpy as np
import numpy.typing as npt
from scipy.stats import kstest

def pit_get_values(
    forecast_samples: npt.NDArray[np.float64], 
    observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
    """
    Generate Probability Integral Transform (PIT) values from forecast samples.
    
    The PIT transforms forecast distributions to uniform [0,1] if forecasts are well-calibrated.
    For each observation, PIT value = P(forecast ≤ observation), computed as the empirical CDF
    of forecast samples at the observed value.

    Parameters:
        forecast_samples : npt.NDArray[np.float64]
            2D array of shape (n_observations, n_samples) containing forecast samples.
            Each row contains simulation samples for one forecast instance.
        observations : npt.NDArray[np.float64]
            1D array of shape (n_observations,) containing the actual observed values.
            
    Returns:
        npt.NDArray[np.float64]
            1D array of PIT values in [0,1]. NaN values indicate invalid observations/forecasts.
            Well-calibrated forecasts should produce uniformly distributed PIT values.
            
    Raises:
        ValueError
            If input arrays have incompatible shapes or contain no valid samples.
        
    Notes:
        - PIT values near 0 indicate forecasts consistently over-predict
        - PIT values near 1 indicate forecasts consistently under-predict  
        - Uniform PIT distribution indicates well-calibrated probabilistic forecasts
        - Non-uniform patterns suggest systematic forecast biases
    
    Examples:
        >>> import numpy as np
        >>> # Generate synthetic data
        >>> n_obs, n_samples = 100, 1000
        >>> true_values = np.random.normal(0, 1, n_obs)
        >>> observations = true_values + np.random.normal(0, 0.1, n_obs)
        >>> forecast_samples = np.random.normal(true_values[:, None], 1, (n_obs, n_samples))
        >>> 
        >>> # Calculate PIT values
        >>> pit_values = pit_get_values(forecast_samples, observations)
        >>> 
        >>> # Check calibration (should be approximately uniform)
        >>> print(f"Mean PIT: {np.nanmean(pit_values):.3f} (should ≈ 0.5)")
        >>> print(f"Std PIT: {np.nanstd(pit_values):.3f} (should ≈ 0.289)")
    
    References:
        - Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, 
        calibration and sharpness. Journal of the Royal Statistical Society B, 69(2), 243-268.
        - Hamill, T. M. (2001). Interpretation of rank histograms for verifying ensemble forecasts.
        Monthly Weather Review, 129(3), 550-560.
    """
    
    forecast_samples = np.asarray(forecast_samples, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)

    if forecast_samples.ndim != 2:
        raise ValueError("forecast_samples must be 2D")
    if observations.ndim != 1:
        raise ValueError("observations must be 1D")
    if forecast_samples.shape[0] != len(observations):
        raise ValueError("Sample count mismatch")

    valid_mask = ~(np.isnan(observations) | np.any(np.isnan(forecast_samples), axis=1))
    if not np.any(valid_mask):
        raise ValueError("No valid samples")

    observations_expanded = observations[valid_mask, np.newaxis]
    valid_samples = forecast_samples[valid_mask]

    pit_values = np.full(len(observations), np.nan)
    pit_values[valid_mask] = np.mean(valid_samples <= observations_expanded, axis=1)

    return pit_values


def pit_uniformity_test(pit_values, alpha=0.05):
    """
    Kolmogorov-Smirnov test for PIT uniformity
    """
    statistic, p_value = kstest(pit_values, 'uniform')
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_uniform': p_value > alpha # type: ignore
    }
