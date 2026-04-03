"""
Machine Learning Models Package

This package provides tabular ML forecasting models for wind power prediction,
all implementing a unified interface through BaseForecaster or DeterministicForecaster.

Individual models return a ParametricForecastResult from forecast():
- dist_name: Distribution identifier
- params: Native distribution parameters, shape (T, 1)
- basis_index: Time index for the forecast period

Runner:
- PerHorizonRunner: Wraps any model to train H independent per-horizon models,
  returning a unified ParametricForecastResult(N, H).

Available Models:
- CatBoostForecaster: CatBoost-based probabilistic forecasting (Normal output)
- XGBoostForecaster: XGBoost-based forecasting with configurable distribution
- GBForecaster: Scikit-learn GradientBoosting with configurable distribution
- LRForecaster: Linear Regression with configurable distribution
- NGBoostForecaster: Natural Gradient Boosting for probabilistic forecasting
- PGBMForecaster: Probabilistic Gradient Boosting Machine forecasting

For deep learning models (DeepAR, TFT), see src.models.deep_time_series.

Base Classes:
- BaseForecaster: Base class for all forecasting models
- DeterministicForecaster: Base class for deterministic models with historical std
- BaseDeepModel: Base class for deep learning models (NeuralForecast wrapper)

Distribution:
- ParametricDistribution: Unified probabilistic forecast output type
- DISTRIBUTION_REGISTRY: Registry of supported distributions
- mu_std_to_dist_params: Convert (mu, std) to native distribution parameters
"""

from src.core.base_model import BaseForecaster, DeterministicForecaster
from src.core.base_deep_model import BaseDeepModel
from src.core.forecast_distribution import ParametricDistribution, DISTRIBUTION_REGISTRY
from src.core.moment_matching import mu_std_to_dist_params
from src.core.forecast_results import ParametricForecastResult
from src.core.runner import PerHorizonRunner

# Deterministic Models
from src.models.machine_learning.gboost_model import GBForecaster as GradientBoostingForecaster, GBModel
from src.models.machine_learning.xgboost_model import XGBoostForecaster, XGBoostModel
from src.models.machine_learning.lr_model import LRForecaster, LRModel

# Probabilistic Models
from src.models.machine_learning.catboost_model import CatBoostForecaster, CatBoostModel
from src.models.machine_learning.ngboost_model import NGBoostForecaster, NGBoostModel
from src.models.machine_learning.pgboost_model import PGBMForecaster, PGBMModel

# Export all available models
__all__ = [
    # Base classes
    'BaseForecaster',
    'DeterministicForecaster',
    'BaseDeepModel',

    # Runner
    'PerHorizonRunner',

    # Distribution
    'ParametricDistribution',
    'DISTRIBUTION_REGISTRY',
    'mu_std_to_dist_params',

    # Deterministic models
    'GradientBoostingForecaster',
    'XGBoostForecaster',
    'LRForecaster',

    # Probabilistic models
    'CatBoostForecaster',
    'NGBoostForecaster',
    'PGBMForecaster',

    # Fundamental models
    'GBModel', 'XGBoostModel', 'LRModel',
    'CatBoostModel', 'NGBoostModel', 'PGBMModel'
]
