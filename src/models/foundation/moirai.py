"""
Moirai forecaster: Salesforce Moirai pretrained time series model wrapper.

Moirai is a universal time series forecasting model trained on large-scale
open data. It supports exogenous covariates (past and future), making it
suitable for wind power forecasting with NWP features.

Uses the ``uni2ts`` package and GluonTS data format internally, but exposes
only the BaseFoundationModel interface.

Reference:
    Woo et al., "Unified Training of Universal Time Series Forecasting
    Transformers", ICML 2024.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, List

from src.core.base_foundation_model import BaseFoundationModel


class MoiraiForecaster(BaseFoundationModel):
    """
    Moirai probabilistic forecaster.

    Wraps Salesforce Moirai pretrained models with the BaseFoundationModel
    interface. Produces SampleForecastResult or QuantileForecastResult.

    Unlike Chronos, Moirai supports exogenous variables via x_cols, which
    are passed as ``past_feat_dynamic_real`` to the model.

    Model-specific hyperparameters (passed via hyperparameter dict):
        model_name_or_path (str): HuggingFace model ID. REQUIRED.
            Examples:
                - "Salesforce/moirai-1.0-R-small"  (≈80M params)
                - "Salesforce/moirai-1.0-R-base"   (≈200M)
                - "Salesforce/moirai-1.0-R-large"  (≈900M)
                - "Salesforce/moirai-1.1-R-small"
                - "Salesforce/moirai-1.1-R-large"
        context_length (int): Context window size. Default: 512
        prediction_length (int): Forecast horizon. Default: 48
        n_samples (int): Number of forecast samples. Default: 100
        output_type (str): "samples" or "quantiles". Default: "samples"
        level (list[int]): Confidence levels for quantile output. Default: [80, 90]
        patch_size (str|int): Patch size for tokenization. Default: "auto"
        batch_size (int): Inference batch size. Default: 32

    Example:
        >>> model = MoiraiForecaster(
        ...     dataset=df, y_col="power",
        ...     x_cols=["wind_speed", "temperature"],
        ...     hyperparameter={
        ...         "model_name_or_path": "Salesforce/moirai-1.0-R-small",
        ...         "prediction_length": 48,
        ...         "output_type": "quantiles",
        ...     }
        ... )
        >>> model.fit()
        >>> result = model.predict()         # → QuantileForecastResult
        >>> result.quantile(0.9, h=6)
    """

    def _get_feat_cols(self) -> List[str]:
        """Get exogenous feature column names as a list."""
        if self.x_cols is None:
            return []
        if isinstance(self.x_cols, (str, int)):
            return [self.x_cols]
        return list(self.x_cols)

    def _load_pretrained(self) -> None:
        """Load Moirai model from HuggingFace Hub."""
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        feat_cols = self._get_feat_cols()
        patch_size = self._model_hp.pop("patch_size", "auto")
        self._batch_size = int(self._model_hp.pop("batch_size", 32))

        module = MoiraiModule.from_pretrained(self._model_name_or_path)

        self._pipeline = MoiraiForecast(
            module=module,
            prediction_length=self._prediction_length,
            context_length=self._context_length,
            patch_size=patch_size,
            num_samples=self._n_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=len(feat_cols),
        )

    def _predict_samples(
        self,
        context_y: np.ndarray,
        prediction_length: int,
        context_X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate forecast samples via Moirai.

        Builds a GluonTS-compatible dataset from the context arrays, runs
        inference through the Moirai predictor, and extracts samples.

        Args:
            context_y: Context target values, shape (context_length,).
            prediction_length: Number of steps to forecast.
            context_X: Optional context features, shape (context_length, n_features).

        Returns:
            np.ndarray of shape (n_samples, prediction_length).
        """
        from gluonts.dataset.pandas import PandasDataset

        # Build a DataFrame for GluonTS
        n = len(context_y)
        # Extend by prediction_length for the "future" that GluonTS needs
        total_len = n + prediction_length
        timestamps = pd.date_range(
            start="2000-01-01", periods=total_len, freq=self._freq or "h"
        )

        df_dict = {
            "target": np.concatenate([
                context_y,
                np.full(prediction_length, np.nan),
            ]),
        }

        # Add past covariates if available
        feat_cols = self._get_feat_cols()
        if context_X is not None and len(feat_cols) > 0:
            for i, col_name in enumerate(feat_cols):
                df_dict[col_name] = np.concatenate([
                    context_X[:, i],
                    np.zeros(prediction_length),
                ])

        df = pd.DataFrame(df_dict, index=timestamps)
        df["item_id"] = "target"

        past_feat = feat_cols if feat_cols else None
        ds = PandasDataset.from_long_dataframe(
            df.reset_index().rename(columns={"index": "timestamp"}),
            target="target",
            item_id="item_id",
            timestamp="timestamp",
            past_feat_dynamic_real=past_feat,
        )

        # Run inference
        predictor = self._pipeline.create_predictor(batch_size=self._batch_size)
        forecasts = list(predictor.predict(ds))

        if not forecasts:
            raise RuntimeError("Moirai produced no forecasts.")

        # Extract samples: (n_samples, prediction_length)
        forecast = forecasts[0]
        samples = forecast.samples  # (n_samples, prediction_length)

        # Trim to requested prediction_length
        if samples.shape[1] > prediction_length:
            samples = samples[:, :prediction_length]

        return samples
