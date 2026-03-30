# TODO:연구에 필요한 기본 metrics들만 정의했음, 이후에 추가 필요함 

# Deterministic metrics
from .deterministic import (
    rmse, 
    mae,
    mape,
    smape
)

# Probabilistic metrics 
from .crps import (
    crps_gaussian,
    crps_laplace,
    crps_logistic,
    crps_numerical,   
)

from .pit import (
    pit_get_values,
    pit_uniformity_test
)

DETERMINISTIC_METRICS = {
    "RMSE": rmse, 
    "MAE": mae, 
    "MAPE": mape, 
    "SMAPE": smape
}

PROBABILISTIC_METRICS = {
    "CRPS": crps_numerical
}

ALL_METRICS = {}
ALL_METRICS.update(DETERMINISTIC_METRICS)
ALL_METRICS.update(PROBABILISTIC_METRICS)

__all__ = [
    # single metric functions 
    "rmse", "mae", "mape", "smape", 
    "crps_gaussian", "crps_laplace", "crps_logistic", "crps_numerical",
    
    # metrics set 
    "DETERMINISTIC_METRICS", "PROBABILISTIC_METRICS",
    
    # PIT set 
    "pit_get_values", "pit_uniformity_test"
]