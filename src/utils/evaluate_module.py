import pandas as pd 
import numpy as np 
from typing import Dict, Optional

# Import from metrics package
from .metrics import (
    DETERMINISTIC_METRICS, 
    PROBABILISTIC_METRICS
)

def evaluate(
    training_data: pd.DataFrame,
    observed_col: str,
    forecast_sim: Optional[pd.DataFrame] = None,
    forecast_mu: Optional[pd.Series] = None,
    is_valid_col: Optional[str] = 'is_valid'
) -> Dict[str, float]:
    """
    Comprehensive forecast evaluation function with smart data handling.
    
    This function automatically determines what evaluations to perform based on
    the provided forecasts:
    - If only forecast_mu provided: deterministic metrics only
    - If forecast_sim provided: probabilistic + deterministic (mu from sim mean)
    - If both provided: uses forecast_mu for deterministic, forecast_sim for probabilistic
    
    Parameters:
        data : pd.DataFrame
            DataFrame containing observed values and optionally validity mask.
            Must contain the column specified by observed_col.
        forecast_sim : pd.DataFrame, optional
            Probabilistic forecast samples with shape (n_observations, n_samples).
            If provided, enables probabilistic evaluation and automatic deterministic
            evaluation using sample means.
        forecast_mu : pd.Series, optional
            Point forecast values. If not provided but forecast_sim is available,
            will be computed as the mean of forecast_sim.
        observed_col : str, default='observed'
            Column name in data containing observed values.
        is_valid_col : str, optional, default='is_valid'
            Column name in data containing validity mask. If None, all observations
            are considered valid.
        
    Returns:
        dict
            Dictionary containing evaluation results:
            - Deterministic metrics: 'RMSE', 'MAE', 'MAPE', 'SMAPE'
            - Probabilistic metrics: 'CRPS' (if forecast_sim provided)
        
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> n_obs = 100
        >>> 
        >>> data = pd.DataFrame({
        ...     'observed': np.random.normal(5, 2, n_obs),
        ...     'is_valid': np.random.choice([True, False], n_obs, p=[0.9, 0.1])
        ... })
        >>> 
        >>> # Case 1: Only point forecasts
        >>> point_forecast = pd.Series(np.random.normal(5, 2.1, n_obs))
        >>> results = evaluate(data, forecast_mu=point_forecast)
        >>> print("Point only:", results.keys())  # Only deterministic metrics
        >>> 
        >>> # Case 2: Only simulation forecasts (most common)
        >>> sim_forecast = pd.DataFrame(np.random.normal(5, 2, (n_obs, 1000)))
        >>> results = evaluate(data, forecast_sim=sim_forecast)
        >>> print("Sim only:", results.keys())  # Both deterministic + probabilistic
        >>> 
        >>> # Case 3: Both forecasts (deterministic uses point, probabilistic uses sim)
        >>> results = evaluate(data, forecast_sim=sim_forecast, forecast_mu=point_forecast)
        >>> print("Both:", results.keys())  # Both, using specified point forecast
    
    Raises:
        ValueError
            If no forecasts provided, required columns missing, or data alignment fails.
    """
    
    # Input validation
    if forecast_sim is None and forecast_mu is None:
        raise ValueError("At least one of forecast_sim or forecast_mu must be provided")
    
    if observed_col not in training_data.columns:
        raise ValueError(f"Column '{observed_col}' not found in data")
    
    # Extract observed values
    observed = training_data[observed_col].copy() 
    
    # Handle columns of forecast_sim -> remove "time" columns 
    if forecast_sim is not None: 
        forecast_sim = forecast_sim.select_dtypes(include=[np.number])
    
    # Handle validity mask
    if is_valid_col is not None and is_valid_col in training_data.columns:
        is_valid = training_data[is_valid_col]
        # Filter to valid observations only
        valid_mask = is_valid == True
        observed = observed[valid_mask]
        
        if forecast_sim is not None:
            forecast_sim = forecast_sim.loc[valid_mask]
        if forecast_mu is not None:
            forecast_mu = forecast_mu[valid_mask]
    
    # Align indices (find common index across all training_data)
    indices_to_align = [observed.index]
    if forecast_sim is not None:
        indices_to_align.append(forecast_sim.index)
    if forecast_mu is not None:
        indices_to_align.append(forecast_mu.index)
    
    # Find common index
    common_index = indices_to_align[0] 
    for idx in indices_to_align[1:]: 
        common_index = common_index.intersection(idx)  # type: ignore
    
    if len(common_index) == 0:
        raise ValueError("No common index found between data and forecasts")
    
    # Align all data to common index
    observed = observed.loc[common_index]
    if forecast_sim is not None:
        forecast_sim = forecast_sim.loc[common_index]
    if forecast_mu is not None:
        forecast_mu = forecast_mu.loc[common_index]
    
    # Convert to numpy for calculations
    observed_values = observed.to_numpy()
    
    # Determine evaluation strategy and compute forecasts
    results = {}
    
    # === Deterministic Evaluation ===
    if forecast_mu is not None:
        # Use provided point forecast
        point_forecast = forecast_mu.to_numpy()
    elif forecast_sim is not None:
        # Compute point forecast from simulation mean
        point_forecast = np.mean(forecast_sim.to_numpy(), axis=1)
    else:
        point_forecast = None
    
    if point_forecast is not None:
        # Compute deterministic metrics
        for metric_name, metric_func in DETERMINISTIC_METRICS.items():
            try:
                results[metric_name] = metric_func(observed_values, point_forecast)
            except Exception as e:
                print(f"Warning: Failed to calculate {metric_name}: {e}")
                results[metric_name] = np.nan
    
    # === Probabilistic Evaluation ===
    if forecast_sim is not None: 
        probabilistic_forecast = forecast_sim.to_numpy()
    if forecast_sim is not None:
        # Compute probabilistic metrics
        for metric_name, metric_func in PROBABILISTIC_METRICS.items():
            try:
                results[metric_name] = metric_func(observed_values, probabilistic_forecast)

            except Exception as e:
                print(f"Warning: Failed to calculate {metric_name}: {e}")
                results[metric_name] = np.nan
    
    return results

