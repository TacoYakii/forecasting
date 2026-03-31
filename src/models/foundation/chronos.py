"""
Chronos forecaster: Amazon Chronos pretrained time series model wrapper.

Chronos is a family of pretrained time series forecasting models based on
language model architectures. It tokenizes time series values and generates
probabilistic forecasts via autoregressive sampling.

Supports both Chronos v1 (T5 encoder-decoder) and Chronos-Bolt (faster,
encoder-only) model variants, all loaded from HuggingFace Hub.

Reference:
    Ansari et al., "Chronos: Learning the Language of Time Series", 2024.
    Amazon, "Chronos-2: From Univariate to Universal Forecasting", 2025.
"""

import numpy as np
import torch
from typing import Optional

from src.core.base_foundation_model import BaseFoundationModel
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="chronos")
class ChronosForecaster(BaseFoundationModel):
    """
    Chronos probabilistic forecaster.

    Wraps Amazon Chronos pretrained models with the BaseFoundationModel
    interface. Produces SampleForecastResult or QuantileForecastResult.

    Chronos models do not support exogenous variables — exog_cols are ignored.

    Model-specific hyperparameters (passed via hyperparameter dict):
        model_name_or_path (str): HuggingFace model ID. REQUIRED.
            Examples:
                - "amazon/chronos-t5-tiny"   (8M params)
                - "amazon/chronos-t5-small"  (46M)
                - "amazon/chronos-t5-base"   (200M)
                - "amazon/chronos-t5-large"  (710M)
                - "amazon/chronos-bolt-tiny"  (9M, faster)
                - "amazon/chronos-bolt-small" (48M, faster)
                - "amazon/chronos-bolt-base"  (205M, faster)
        context_length (int): Context window size. Default: 512
        prediction_length (int): Forecast horizon. Default: 48
        n_samples (int): Number of forecast samples. Default: 100
        output_type (str): "samples" or "quantiles". Default: "samples"
        level (list[int]): Confidence levels for quantile output. Default: [80, 90]
        torch_dtype (str): Torch dtype. Default: "float32"

    Example:
        >>> model = ChronosForecaster(
        ...     dataset=df, y_col="power",
        ...     hyperparameter={
        ...         "model_name_or_path": "amazon/chronos-bolt-small",
        ...         "prediction_length": 48,
        ...         "output_type": "quantiles",
        ...     }
        ... )
        >>> model.fit()                      # loads pretrained weights
        >>> result = model.predict()         # → QuantileForecastResult
        >>> result.to_distribution(6).ppf(0.9)
    """

    def _load_pretrained(self) -> None:
        """Load Chronos pipeline from HuggingFace Hub."""
        from chronos import ChronosPipeline

        dtype_str = self._model_hp.pop("torch_dtype", "float32")
        torch_dtype = getattr(torch, dtype_str, torch.float32)

        self._pipeline = ChronosPipeline.from_pretrained(
            self._model_name_or_path,
            device_map=self._device,
            dtype=torch_dtype,
        )

    def _predict_samples(
        self,
        context_y: np.ndarray,
        prediction_length: int,
        context_X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate forecast samples via Chronos pipeline.

        Args:
            context_y: Context target values, shape (context_length,).
            prediction_length: Number of steps to forecast.
            context_X: Ignored — Chronos does not support exogenous variables.

        Returns:
            np.ndarray of shape (n_samples, prediction_length).
        """
        context_tensor = torch.tensor(
            context_y, dtype=torch.float32
        ).unsqueeze(0)  # (1, context_length)

        with torch.no_grad():
            forecast = self._pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
                num_samples=self._n_samples,
            )
        # forecast shape: (1, n_samples, prediction_length)
        return forecast.squeeze(0).cpu().numpy()
