import jax 
import jax.numpy as jnp 


def crps_numerical_j(y, forecast_samples, need_sorting=True): 
    """
    Compute the Continuous Ranked Probability Score (CRPS) using numerical integration.
    
    The CRPS is a proper scoring rule for evaluating probabilistic forecasts that measures
    the difference between the forecast cumulative distribution function (CDF) and the
    step function of the observation. This implementation uses numerical integration over
    sorted forecast samples to approximate the continuous CRPS integral.
    
    The CRPS is defined as:
    CRPS(F, y) = ∫ [F(x) - H(x - y)]² dx
    
    where F(x) is the forecast CDF, H(x - y) is the Heaviside step function at observation y,
    and the integral is over the entire real line.
    
    Parameters:
        y : float
            Single scalar observation value.
        forecast_samples : jnp.ndarray
            1D array of forecast samples representing the predictive distribution.
            Shape: (n_samples,)
        need_sorting : bool, default=True
            Whether to sort the forecast samples. If samples are already sorted,
            set to False for computational efficiency.
        
    Returns:
        float
            CRPS value. Lower values indicate better calibrated forecasts.
            Units are the same as the observation and forecast samples.
        
    Notes:
        This implementation is compatible with JAX transformations (jit, vmap, grad).
        The numerical integration uses the trapezoidal rule with adaptive handling
        of the observation point within the forecast distribution.
        
        For a perfectly calibrated forecast, the expected CRPS equals half the expected
        absolute difference between two independent samples from the true distribution.
    
        The algorithm handles edge cases:
        - Observation below all forecast samples
        - Observation above all forecast samples  
        - Observation within the forecast sample range
        
    Examples:
        >>> import jax.numpy as jnp
        >>> # Perfect forecast (observation equals mean)
        >>> y_obs = 0.0
        >>> samples = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        >>> crps_val = crps_numerical(y_obs, samples)
        >>> print(f"CRPS: {crps_val:.3f}")  # Should be relatively small
        
        >>> # Poor forecast (observation far from samples)
        >>> y_obs = 10.0  # Far from forecast distribution
        >>> crps_val = crps_numerical(y_obs, samples)
        >>> print(f"CRPS: {crps_val:.3f}")  # Should be large
        
        >>> # Using pre-sorted samples for efficiency
        >>> sorted_samples = jnp.sort(samples)  # Already sorted in this case
        >>> crps_val = crps_numerical(y_obs, sorted_samples, need_sorting=False)
    
    References:
        - Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, 
        prediction, and estimation. Journal of the American Statistical Association, 
        102(477), 359-378.
        - properscoring package: https://pypi.org/project/properscoring/
    """
    forecast_sorted = jax.lax.cond(
        need_sorting, 
        lambda arr: jnp.sort(arr), 
        lambda arr: arr, 
        forecast_samples
    )

    N = forecast_samples.shape[0]
    weight = 1.0 / N

    def integrand(i, carry):
        forecast_i = forecast_sorted[i]
        delta = forecast_i - carry['prev']
        f_cdf = carry['cdf'] + weight

        obs_contrib = jnp.where(
            (carry['obs_cdf'] == 0.0) & (y < forecast_i),
            (y - carry['prev']) * carry['cdf']**2 + (forecast_i - y) * (carry['cdf'] - 1)**2,
            delta * (carry['cdf'] - carry['obs_cdf'])**2
        )

        obs_cdf = jnp.where(
            (carry['obs_cdf'] == 0.0) & (y < forecast_i),
            1.0,
            carry['obs_cdf']
        )

        return {
            'val': carry['val'] + obs_contrib,
            'cdf': f_cdf,
            'prev': forecast_i,
            'obs_cdf': obs_cdf
        }

    carry_init = {'val': 0.0, 'cdf': 0.0, 'prev': 0.0, 'obs_cdf': 0.0}
    result = jax.lax.fori_loop(0, N, integrand, carry_init)
    final_adjust = jnp.where(result['obs_cdf'] == 0.0, y - forecast_sorted[-1], 0.0)
    
    return result['val'] + final_adjust

def crps_numerical_batch_j(y_batch, forecast_samples_batch, need_sorting=True):
    """
    Calculate CRPS numerical for arbitrary dimensional batches.
    
    This function automatically handles any number of batch dimensions by
    flattening the inputs, applying vectorized CRPS calculation, and 
    reshaping the results back to the original batch shape.
    
    Parameters:
        y_batch : jnp.ndarray
            Observations with shape (..., batch_dims). Any number of leading dimensions.
        forecast_samples_batch : jnp.ndarray  
            Forecast samples with shape (..., batch_dims, n_samples).
            All dimensions except the last must match y_batch.
        need_sorting : bool, default=True
            Whether forecast samples need to be sorted.
        
    Returns
        jnp.ndarray
            CRPS values with the same shape as y_batch.
        
    Examples
        >>> # 1D batch
        >>> y = jnp.array([1.0, 2.0, 3.0])  # shape: (3,)
        >>> forecasts = jnp.random.normal(0, 1, (3, 100))  # shape: (3, 100)
        >>> crps_1d = crps_numerical_batch_general(y, forecasts)
        >>> 
        >>> # 2D batch  
        >>> y = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape: (2, 2)
        >>> forecasts = jnp.random.normal(0, 1, (2, 2, 100))  # shape: (2, 2, 100)
        >>> crps_2d = crps_numerical_batch_general(y, forecasts)
        >>> 
        >>> # 3D batch
        >>> y = jnp.random.randn(5, 10, 20)  # shape: (5, 10, 20)
        >>> forecasts = jnp.random.randn(5, 10, 20, 100)  # shape: (5, 10, 20, 100)
        >>> crps_3d = crps_numerical_batch_general(y, forecasts)
    """
    # Validate input shapes
    y_shape = y_batch.shape
    forecast_shape = forecast_samples_batch.shape

    if y_shape != forecast_shape[:-1]:
        raise ValueError(
            f"Batch dimensions must match: y_batch.shape={y_shape} vs "
            f"forecast_samples_batch.shape[:-1]={forecast_shape[:-1]}"
        )

    if len(forecast_shape) < 1:
        raise ValueError("forecast_samples_batch must have at least 1 dimension for samples")

    # Handle edge case: no batch dimensions (single observation)
    if len(y_shape) == 0:
        return crps_numerical_j(y_batch, forecast_samples_batch, need_sorting)

    # Flatten all batch dimensions
    n_total_obs = y_batch.size
    n_samples = forecast_shape[-1]

    y_flat = y_batch.reshape(n_total_obs)
    forecast_flat = forecast_samples_batch.reshape(n_total_obs, n_samples)

    # Apply vectorized CRPS calculation
    batched_crps_fn = jax.vmap(crps_numerical_j, in_axes=(0, 0, None))
    crps_flat = batched_crps_fn(y_flat, forecast_flat, need_sorting)

    # Reshape back to original batch shape
    return crps_flat.reshape(y_shape)


@jax.jit 
def crps_binned(y, binmass, bin_borders):
    """
    Compute the Continuous Ranked Probability Score (CRPS) for binned probabilistic forecasts.
    
    This function calculates CRPS for forecasts represented as probability mass distributions
    across discrete bins. The CRPS measures the difference between the forecast cumulative
    distribution function (CDF) and the step function of the observation using analytical
    integration over the binned representation.
    
    The implementation uses the analytical formula for CRPS with binned forecasts:
    CRPS = ∫ [F(x) - H(x - y)]² dx
    
    where F(x) is the forecast CDF constructed from bin probabilities, H(x - y) is the
    Heaviside step function at observation y, and the integral is computed analytically
    over each bin interval.
    
    Parameters:
        y : jnp.ndarray  
            Observation values with shape (N,). Each value represents the true outcome
            for the corresponding forecast distribution.
        binmass : jnp.ndarray
            Probability mass for each bin with shape (N, B), where N is the number of
            observations and B is the number of bins. Each row should sum to 1.0.
        bin_borders : jnp.ndarray
            Bin boundary values with shape (B+1,). Defines the edges of B bins where
            bin_borders[i] and bin_borders[i+1] are the left and right edges of bin i.
            Must be monotonically increasing.
        
    Returns:
        jnp.ndarray
            CRPS values with shape (N,). Lower values indicate better calibrated forecasts.
            Units are the same as the observation and bin_borders.
        
    Notes:
        This implementation is compatible with JAX transformations (jit, vmap, grad).
        The function assumes that:
        - Bin probabilities are non-negative and sum to 1.0 for each observation
        - Bin borders are sorted in ascending order
        - Observations can be inside or outside the bin range
        
        For observations outside the bin range, the CRPS is computed by extending
        the CDF appropriately (0 below the first bin, 1 above the last bin).
        
        The analytical approach is more computationally efficient than numerical
        integration when forecasts are naturally represented as binned distributions.
        
    Examples:
        >>> import jax.numpy as jnp
        >>> # Simple example with 3 bins
        >>> bin_borders = jnp.array([0.0, 1.0, 2.0, 3.0])  # 3 bins: [0,1), [1,2), [2,3)
        >>> binmass = jnp.array([[0.2, 0.5, 0.3],   # First forecast
        ...                      [0.1, 0.3, 0.6]])  # Second forecast  
        >>> y = jnp.array([1.5, 2.5])  # Observations
        >>> crps_vals = crps_loss_batch_jax(binmass, y, bin_borders)
        >>> print(f"CRPS values: {crps_vals}")
        
        >>> # Batch processing multiple forecasts
        >>> N, B = 100, 10
        >>> bin_borders = jnp.linspace(0, 10, B + 1)
        >>> binmass = jnp.abs(jnp.random.normal(0, 1, (N, B)))
        >>> binmass = binmass / jnp.sum(binmass, axis=1, keepdims=True)  # Normalize
        >>> y = jnp.random.uniform(0, 10, N)
        >>> crps_batch = crps_loss_batch_jax(binmass, y, bin_borders)
        
    References:
        - Franco, M. A. et al. (2024). A CRPS Loss for Deep Probabilistic Regression. IEEE URUCON
    """
    N, B = binmass.shape
    
    bin_widths = bin_borders[1:] - bin_borders[:-1]
    
    # Compute CDF from prob mass of bins
    cdf_values = jnp.concatenate([
        jnp.zeros((N, 1)), 
        jnp.cumsum(binmass, axis=1)
    ], axis=1)
    
    # Compute parts
    parts = (cdf_values[:, :-1] + cdf_values[:, 1:]) / 2 * bin_widths
    sq_parts = (-1/3 * (
        cdf_values[:, :-1]**2 + 
        cdf_values[:, :-1] * cdf_values[:, 1:] + 
        cdf_values[:, 1:]**2
    ) * (-bin_widths))
    
    # First part of the loss
    p1 = bin_borders[-1] - y
    
    # Second part of the loss
    p2 = jnp.sum(sq_parts, axis=1)
    
    # Find the bin index for each y
    purek = jnp.searchsorted(bin_borders, y) - 1
    k = jnp.clip(purek, 0, B - 1)
    
    # Compute CDF at y using linear interpolation
    N_idx = jnp.arange(N)
    cdf_at_y = cdf_values[N_idx, k] + (
        (y - bin_borders[k]) / bin_widths[k] * 
        (cdf_values[N_idx, k + 1] - cdf_values[N_idx, k])
    )
    
    # Third part
    in_range = (bin_borders[0] < y) & (y < bin_borders[-1])
    p3 = (
        (cdf_at_y + cdf_values[N_idx, k + 1]) / 2 * 
        (bin_borders[k + 1] - y)
    ) * in_range
    
    # Fourth part
    mask = jnp.arange(B)[None, :] > purek[:, None]
    p4 = jnp.sum(parts * mask, axis=1) * (y < bin_borders[-1])
    
    crps = jnp.abs(p1) + p2 - 2 * (p3 + p4)
    return crps