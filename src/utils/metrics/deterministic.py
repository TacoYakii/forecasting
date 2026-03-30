import numpy as np 


def rmse(observed: np.ndarray, forecasts: np.ndarray):
    """Root-mean-square deviation (RMSE)

    Args:
        observed : Observed dependent variable
        forecasts : Expected value of dependent variable.
    """
    return np.sqrt(np.sum((observed - forecasts) ** 2) / len(observed))


def mae(observed: np.ndarray, forecasts: np.ndarray):
    """Mean-absolute error (MAE)

    Args:
        observed : Observed dependent variable
        forecasts : Expected value of dependent variable.
    """
    return np.sum(np.abs(observed - forecasts)) / len(observed)


def mape(observed: np.ndarray, forecasts: np.ndarray):
    """Mean Absolute Percentage Error (MAPE)

    Args:
        observed (np.ndarray): Observed dependent variable
        forecasts (np.ndarray): Expected value of dependent variable.

    Returns:
        float: MAPE value (0~100%)
    """
    mask = observed != 0 
    observed, forecasts = observed[mask], forecasts[mask]
    
    return np.mean(np.abs((observed - forecasts) / observed)) * 100


def smape(observed: np.ndarray, forecasts: np.ndarray):
    """Symmetric Mean Absolute Percentage Error (SMAPE)

    Args:
        observed (np.ndarray): Observed dependent variable
        forecasts (np.ndarray): Expected value of dependent variable.

    Returns:
        float: SMAPE value
    """
    observed, forecasts = np.array(observed), np.array(forecasts)

    numerator = np.abs(forecasts - observed)
    denominator = (np.abs(observed) + np.abs(forecasts)) / 2 
    mask = denominator != 0 
    numerator, denominator = numerator[mask], denominator[mask] 
    smape_values = numerator / denominator

    return np.mean(smape_values) * 100 


