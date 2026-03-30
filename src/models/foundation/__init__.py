"""
Foundation time series forecasting models.

Pretrained foundation models loaded from HuggingFace Hub for probabilistic
time series forecasting. All models inherit from BaseFoundationModel and
produce SampleForecastResult or QuantileForecastResult output.

Available models:
    - ChronosForecaster ("chronos"): Amazon Chronos (T5-based, sample output)
    - MoiraiForecaster ("moirai"): Salesforce Moirai (covariate support)

Usage:
    >>> from src.models.foundation import ChronosForecaster, MoiraiForecaster
    >>> model = ChronosForecaster(
    ...     dataset=df, y_col="power",
    ...     hyperparameter={"model_name_or_path": "amazon/chronos-bolt-small"}
    ... )
    >>> model.fit()
    >>> result = model.predict()  # → SampleForecastResult
"""

from src.models.foundation.chronos import ChronosForecaster
from src.models.foundation.moirai import MoiraiForecaster

__all__ = [
    "ChronosForecaster",
    "MoiraiForecaster",
]
