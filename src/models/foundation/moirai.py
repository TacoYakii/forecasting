"""Moirai forecaster: Salesforce Moirai pretrained time series model wrapper.

Moirai is a universal time series forecasting model trained on large-scale
open data. It supports exogenous covariates (past only), making it
suitable for wind power forecasting with NWP features.

Uses the ``uni2ts`` package and GluonTS data format internally, but exposes
only the BaseFoundationModel interface.

Fine-tuning is **not yet supported** due to ``uni2ts`` incompatibility
with numpy 2.x. Fine-tuning will be added when uni2ts updates its
numpy dependency (currently pinned to ``numpy~=1.26``).

Reference:
    Woo et al., "Unified Training of Universal Time Series Forecasting
    Transformers", ICML 2024.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.base_foundation_model import BaseFoundationModel
from src.core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register_model(name="moirai")
class MoiraiForecaster(BaseFoundationModel):
    """Moirai probabilistic forecaster.

    Wraps Salesforce Moirai pretrained models with the BaseFoundationModel
    interface. Produces SampleForecastResult or QuantileForecastResult.

    Unlike Chronos, Moirai supports exogenous variables via exog_cols,
    which are passed as ``past_feat_dynamic_real`` to the model.

    Fine-tuning is not yet supported due to uni2ts / numpy 2.x
    incompatibility. Setting ``fine_tune_strategy`` will raise
    ``NotImplementedError``.

    Model-specific hyperparameters (passed via hyperparameter dict):
        model_name_or_path (str): HuggingFace model ID. REQUIRED.
        context_length (int): Context window size. Default: 512
        prediction_length (int): Forecast horizon. Default: 48
        n_samples (int): Number of forecast samples. Default: 100
        output_type (str): "samples" or "quantiles". Default: "samples"
        level (list[int]): Confidence levels. Default: [80, 90]
        patch_size (str|int): Patch size. Default: "auto"
        batch_size (int): Inference batch size. Default: 32

    Example:
        >>> model = MoiraiForecaster(
        ...     hyperparameter={
        ...         "model_name_or_path": "Salesforce/moirai-1.0-R-small",
        ...         "prediction_length": 48,
        ...     }
        ... )
        >>> model.fit(dataset=df, y_col="power")
        >>> result = model.forecast()
    """

    def _get_feat_cols(self) -> List[str]:
        """Get exogenous feature column names as a list."""
        if self.exog_cols is None:
            return []
        if isinstance(self.exog_cols, (str, int)):
            return [self.exog_cols]
        return list(self.exog_cols)

    # ------------------------------------------------------------------
    # Pretrained loading
    # ------------------------------------------------------------------

    def _load_pretrained(self) -> None:
        """Load Moirai model from HuggingFace Hub or local path."""
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        feat_cols = self._get_feat_cols()
        patch_size = self._model_hp.get("patch_size", "auto")
        self._batch_size = int(self._model_hp.get("batch_size", 32))

        load_path = (
            self._fine_tuned_model_path or self._model_name_or_path
        )
        module = MoiraiModule.from_pretrained(load_path)

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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict_samples(
        self,
        context_y: np.ndarray,
        prediction_length: int,
        context_X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate forecast samples via Moirai.

        Builds a GluonTS-compatible dataset from the context arrays, runs
        inference through the Moirai predictor, and extracts samples.

        Args:
            context_y: Context target values, shape (context_length,).
            prediction_length: Number of steps to forecast.
            context_X: Context features, shape (context_length, n_feat).

        Returns:
            np.ndarray of shape (n_samples, prediction_length).
        """
        from gluonts.dataset.pandas import PandasDataset

        n = len(context_y)
        total_len = n + prediction_length
        timestamps = pd.date_range(
            start="2000-01-01",
            periods=total_len,
            freq=self._freq or "h",
        )

        df_dict = {
            "target": np.concatenate([
                context_y,
                np.full(prediction_length, np.nan),
            ]),
        }

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

        predictor = self._pipeline.create_predictor(
            batch_size=self._batch_size,
        )
        forecasts = list(predictor.predict(ds))

        if not forecasts:
            raise RuntimeError("Moirai produced no forecasts.")

        forecast = forecasts[0]
        samples = forecast.samples  # (n_samples, prediction_length)

        if samples.shape[1] > prediction_length:
            samples = samples[:, :prediction_length]

        return samples

    # ------------------------------------------------------------------
    # Fine-tuning (blocked by uni2ts numpy 2.x incompatibility)
    # ------------------------------------------------------------------

    def _fine_tune_model(self) -> None:
        """Fine-tune Moirai on training data.

        Currently raises ``NotImplementedError`` because ``uni2ts``
        requires ``numpy~=1.26`` while this project uses numpy 2.x.
        Fine-tuning will be enabled when uni2ts updates its dependency.

        Raises:
            NotImplementedError: Always.

        Example:
            >>> model = MoiraiForecaster(
            ...     hyperparameter={
            ...         "model_name_or_path": "Salesforce/moirai-1.0-R-small",
            ...         "fine_tune_strategy": "full",
            ...     }
            ... )
            >>> model.fit(dataset=df, y_col="power")  # raises
        """
        raise NotImplementedError(
            "Moirai fine-tuning is not yet supported. "
            "uni2ts requires numpy~=1.26 but this project uses numpy 2.x. "
            "Fine-tuning will be added when uni2ts updates its numpy "
            "dependency. Use Chronos for fine-tuning in the meantime."
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_model_specific(self, model_path: Path) -> Path:
        """Save Moirai model state.

        Args:
            model_path: Base path without extension.

        Returns:
            Path to saved JSON config.
        """
        return super()._save_model_specific(model_path)

    def _load_model_specific(self, model_path: Path) -> None:
        """Load Moirai model state.

        Args:
            model_path: Base path without extension.
        """
        super()._load_model_specific(model_path)
