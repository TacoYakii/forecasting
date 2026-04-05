"""
Deep time series forecasting models.

NeuralForecast-based deep learning models for probabilistic time series
forecasting. All models inherit from BaseDeepModel and produce
QuantileForecastResult output.

Available models:
    - DeepARForecaster ("deepar"): Autoregressive RNN with distributional output
    - TFTForecaster ("tft"): Temporal Fusion Transformer with attention
    - NHITSForecaster ("nhits"): Neural Hierarchical Interpolation (multi-scale)

Usage:
    >>> from src.models.deep_time_series import DeepARForecaster, TFTForecaster
    >>> model = DeepARForecaster(dataset=df, y_col="power", ...).fit()
    >>> result = model.predict()  # → QuantileForecastResult
"""

from src.models.deep_time_series.deepar import DeepARForecaster
from src.models.deep_time_series.nhits import NHITSForecaster
from src.models.deep_time_series.tft import TFTForecaster

__all__ = [
    "DeepARForecaster",
    "NHITSForecaster",
    "TFTForecaster",
]
