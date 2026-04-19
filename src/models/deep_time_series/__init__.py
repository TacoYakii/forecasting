"""
Deep time series forecasting models.

NeuralForecast-based deep learning models for probabilistic time series
forecasting. All models inherit from BaseDeepModel and produce
QuantileForecastResult output.

Available models:
    - DeepARForecaster ("deepar"): Autoregressive RNN with distributional output
    - TFTForecaster ("tft"): Temporal Fusion Transformer with attention
    - NHITSForecaster ("nhits"): Neural Hierarchical Interpolation (multi-scale)
    - BiTCNForecaster ("bitcn"): Bidirectional Temporal Conv Network (dilated CNN)
    - TSMixerxForecaster ("tsmixerx"): MLP-Mixer for time series with exogenous
    - TimesNetForecaster ("timesnet"): 2D CNN on FFT-decomposed series

Usage:
    >>> from src.models.deep_time_series import DeepARForecaster, TFTForecaster
    >>> model = DeepARForecaster(dataset=df, y_col="power", ...).fit()
    >>> result = model.predict()  # → QuantileForecastResult
"""

from src.models.deep_time_series.bitcn import BiTCNForecaster
from src.models.deep_time_series.deepar import DeepARForecaster
from src.models.deep_time_series.nhits import NHITSForecaster
from src.models.deep_time_series.tft import TFTForecaster
from src.models.deep_time_series.timesnet import TimesNetForecaster
from src.models.deep_time_series.tsmixerx import TSMixerxForecaster

__all__ = [
    "BiTCNForecaster",
    "DeepARForecaster",
    "NHITSForecaster",
    "TFTForecaster",
    "TimesNetForecaster",
    "TSMixerxForecaster",
]
