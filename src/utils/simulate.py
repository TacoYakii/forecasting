import numpy as np
import pandas as pd
from scipy import stats
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..machine_learning.forecast_distribution import ParametricDistribution


# ---------------------------------------------------------------------------
# Primary API: ParametricDistribution-based simulation
# ---------------------------------------------------------------------------

def simulate_from_distribution(
    dist: Any,
    n: int = 1000,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate Monte Carlo samples from a ParametricDistribution via Inverse Transform Sampling.
    
    This is the recommended way to generate samples after refactoring.
    Uses dist.sample() which internally calls ppf() on uniform random variates.
    
    Args:
        dist: ParametricDistribution returned by any model's predict() method
        n: Number of samples per time step
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame of shape (T, n) with the forecast index.
        If dist.base_idx is set, a "basis_time" column is prepended.
        
    Examples:
        >>> dist = model.predict()
        >>> samples_df = simulate_from_distribution(dist, n=1000, seed=42)
        >>> samples_df.shape  # (T, 1000) or (T, 1001) with basis_time
    """
    samples = dist.sample(n=n, seed=seed)  # shape (T, n)
    df = pd.DataFrame(samples, index=dist.index)
    
    if dist.base_idx is not None:
        df.insert(0, "basis_time", dist.base_idx)
    
    return df


# ---------------------------------------------------------------------------
# Legacy API: DataFrame-based simulation (backward compatibility)
# ---------------------------------------------------------------------------

def get_simulation_data(
    forecast_df: pd.DataFrame,
    distribution_setting: str,
    n_forecast: int = 1000,
    **kwargs
) -> pd.DataFrame:
    """
    Get simulated data from a forecast DataFrame with 'mu' and 'std' columns.
    
    This function accepts the DataFrame format produced by ParametricDistribution.to_dataframe()
    (columns: mu, std) as well as the legacy format (columns: drift, volatility).
    
    Args:
        forecast_df (pd.DataFrame): forecast data with 'mu'/'std' or 'drift'/'volatility' columns
        distribution_setting (str): distribution setting assumed when making predictions
        n_forecast (int): number of samples to generate per time step
        **kwargs: additional parameters for specific distributions (e.g., df for student_t)
    
    Returns:
        pd.DataFrame: simulated data with same index as input, shape (T, n_forecast)
    """
    # Support both new (mu/std) and legacy (drift/volatility) column names
    if "mu" in forecast_df.columns and "std" in forecast_df.columns:
        mu_col, std_col = "mu", "std"
    elif "drift" in forecast_df.columns and "volatility" in forecast_df.columns:
        mu_col, std_col = "drift", "volatility"
    else:
        raise ValueError(
            "forecast_df must have columns ('mu', 'std') or ('drift', 'volatility')"
        )
    
    # Distribution function mapping
    distribution_map = {
        "normal": normal,
        "studentT": student_t,
        "laplace": laplace,
        "logistic": logistic,
        "lognormal": lognormal,
        "gamma": gamma,
        "gumbel": gumbel,
        "negativebinomial": negativebinomial,
        "poisson": poisson
    }
    
    if distribution_setting not in distribution_map:
        raise ValueError(f"Given distribution setting not available: {distribution_setting}")
    
    dist_func = distribution_map[distribution_setting]
    
    # Handle poisson separately (only needs mean parameter)
    if distribution_setting == "poisson":
        res = np.stack([
            dist_func(mean, n_forecast=n_forecast, **kwargs)
            for mean in forecast_df[mu_col]
        ])
    else:
        res = np.stack([
            dist_func(mean, std, n_forecast=n_forecast, **kwargs) 
            for mean, std in zip(forecast_df[mu_col], forecast_df[std_col])
        ])
    
    res = pd.DataFrame(res, index=forecast_df.index) 
    if "basis_time" in forecast_df.columns: 
        res.insert(0, "basis_time", forecast_df["basis_time"])
    
    return res


def simulate_distribution(
    distribution: str,
    mu: float,
    std: Optional[float] = None, 
    n_forecast: int = 1000,
    **dist_params
) -> np.ndarray:
    """
    Wrapper function to simulate from any available distribution.
    
    Args:
        distribution (str): name of distribution
        mu (float): mean/location parameter
        std (float): standard deviation (not needed for poisson)
        n_forecast (int): number of samples to generate
        **dist_params: distribution-specific parameters (e.g., df for student_t)
    
    Returns:
        np.ndarray: generated samples
        
    Examples:
        # Normal distribution
        samples = simulate_distribution("normal", mu=0, std=1, n_forecast=1000)
        
        # Student-t with custom degrees of freedom
        samples = simulate_distribution("studentT", mu=0, std=1, df=5, n_forecast=1000)
        
        # Poisson (only needs mu)
        samples = simulate_distribution("poisson", mu=3, n_forecast=1000)
    """
    distribution_map = {
        "normal": normal,
        "studentT": student_t,
        "laplace": laplace,
        "logistic": logistic,
        "lognormal": lognormal,
        "gamma": gamma,
        "gumbel": gumbel,
        "negativebinomial": negativebinomial,
        "poisson": poisson
    }
    
    if distribution not in distribution_map:
        available = ", ".join(distribution_map.keys())
        raise ValueError(f"Distribution '{distribution}' not available. Choose from: {available}")
    
    dist_func = distribution_map[distribution]
    
    if distribution == "poisson":
        if std is not None:
            print("Warning: 'std' parameter ignored for Poisson distribution")
        return dist_func(mu, n_forecast=n_forecast, **dist_params)
    else:
        if std is None:
            raise ValueError(f"'std' parameter required for {distribution} distribution")
        return dist_func(mu, std, n_forecast=n_forecast, **dist_params)


# ---------------------------------------------------------------------------
# Distribution sampling functions
# ---------------------------------------------------------------------------

def clamp_value(value: float) -> float: 
    """Clamp value to minimum threshold to avoid numerical issues."""
    return max(value, 1e-9) 

def normal(mu: float, std: float, n_forecast: int = 1000):
    """Generate samples from normal distribution."""
    scale = np.maximum(std, 1e-9)
    return np.random.normal(mu, scale, n_forecast)

def student_t(mu: float, std: float, df: int = 3, n_forecast: int = 1000):
    """Generate samples from Student's t-distribution.
    
    Args:
        mu: location parameter
        std: standard deviation
        n_forecast: number of samples to generate
        df: degrees of freedom (must be > 2 for finite variance)
    """
    if df <= 2:
        raise ValueError("Degrees of freedom must be > 2 for finite variance")
    
    variance = std**2
    factor = df / (df - 2)
    scale = np.maximum(np.sqrt(variance / factor), 1e-9)
    return np.asarray(stats.t.rvs(df, loc=mu, scale=scale, size=n_forecast))
    
def laplace(mu: float, std: float, n_forecast: int = 1000):
    """Generate samples from Laplace distribution."""
    variance = std**2
    scale = np.maximum(np.sqrt(0.5 * variance), 1e-9)
    return np.asarray(stats.laplace.rvs(loc=mu, scale=scale, size=n_forecast))

def logistic(mu: float, std: float, n_forecast: int = 1000):
    """Generate samples from logistic distribution."""
    variance = std**2
    scale = np.maximum(np.sqrt(variance * (3 / np.pi**2)), 1e-9)
    return np.asarray(stats.logistic.rvs(loc=mu, scale=scale, size=n_forecast))

def lognormal(mu: float, std: float, n_forecast: int = 1000):
    """Generate samples from lognormal distribution."""
    mu_adj = clamp_value(mu)
    variance = np.maximum(std**2, 1e-9)
    
    sigma_sq = np.log(1 + variance / mu_adj**2)
    mu_ln = np.log(mu_adj) - 0.5 * sigma_sq
    sigma_ln = np.sqrt(sigma_sq)
    
    return np.asarray(stats.lognorm.rvs(s=sigma_ln, scale=np.exp(mu_ln), size=n_forecast))

def gamma(mu: float, std: float, n_forecast: int = 1000):
    """Generate samples from gamma distribution."""
    variance = np.maximum(std**2, 1e-9)
    mu_adj = clamp_value(mu)
    
    shape = mu_adj**2 / variance
    scale = variance / mu_adj
    return np.asarray(stats.gamma.rvs(a=shape, scale=scale, size=n_forecast))

def gumbel(mu: float, std: float, n_forecast: int = 1000):
    """Generate samples from Gumbel distribution."""
    variance = np.maximum(std**2, 1e-9)
    scale = np.sqrt(6 * variance) / np.pi
    loc = mu - scale * np.euler_gamma
    return np.asarray(stats.gumbel_r.rvs(loc=loc, scale=scale, size=n_forecast))

def negativebinomial(mu: float, std: float, n_forecast: int = 1000):
    """Generate samples from negative binomial distribution."""
    eps = 1e-9
    mu_adj = clamp_value(mu)
    variance = np.maximum(std**2, eps)
    
    if variance <= mu_adj:
        variance = mu_adj + eps
    
    p = mu_adj / variance
    n = mu_adj**2 / (variance - mu_adj)
    n = np.maximum(n, eps)
    
    return np.asarray(stats.nbinom.rvs(n=n, p=p, size=n_forecast))

def poisson(mu: float, n_forecast: int = 1000):
    """Generate samples from Poisson distribution."""
    mu_adj = clamp_value(mu)
    return np.asarray(stats.poisson.rvs(mu=mu_adj, size=n_forecast))
