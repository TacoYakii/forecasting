"""
BaseFoundationModel: Base class for pretrained foundation forecasting models.

This module provides BaseFoundationModel, which extends BaseForecaster to
support pretrained time series foundation models (Chronos, Moirai, etc.)
that produce probabilistic predictions via samples or quantiles.

Unlike BaseDeepModel (NeuralForecast pattern), foundation models follow a
different lifecycle:
    1. Load pretrained weights via from_pretrained() — no training data needed
    2. Optionally fine-tune on domain-specific data
    3. Predict via model-specific inference API → samples or quantiles

Key responsibilities:
- Pretrained model loading (from_pretrained / download)
- Optional fine-tuning on training data
- Rolling prediction over the forecast period
- Sample → SampleForecastResult or quantile → QuantileForecastResult conversion
- Model save/load (weights or pipeline serialization)
"""

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union, Optional, Dict, List, Any, Iterable, Self
from abc import abstractmethod

from src.core.base_model import BaseForecaster
from src.core.forecast_results import SampleForecastResult, QuantileForecastResult


# Default hyperparameters for foundation models.
_DEFAULT_FOUNDATION_HP = {
    "context_length": 512,          # encoder window (foundation models often use longer context)
    "prediction_length": 48,        # forecast horizon
    "n_samples": 100,               # number of samples per prediction
    "distribution": "normal",       # distribution name for ForecastResult
    "device": "auto",               # "cuda", "cpu", or "auto"
    "model_name_or_path": None,     # HuggingFace model ID or local path (REQUIRED)
    "fine_tune": False,             # whether to fine-tune on training data
    "output_type": "samples",       # "samples" → SampleForecastResult, "quantiles" → QuantileForecastResult
    "level": [80, 90],              # confidence levels for quantile output (e.g., 80 → q=0.1, 0.9)
}


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu'."""
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class BaseFoundationModel(BaseForecaster):
    """
    Base class for pretrained foundation time series forecasting models.

    Extends BaseForecaster to support models loaded via from_pretrained()
    (e.g., Chronos, TimesFM, Moirai, Lag-Llama) while producing
    SampleForecastResult output compatible with the existing forecast API.

    Lifecycle:
        1. __init__: Load dataset, extract hyperparameters
        2. fit():    Load pretrained weights + optionally fine-tune
        3. predict(): Rolling inference → SampleForecastResult

    This class handles:
    1. Hyperparameter extraction (context/prediction length, device, etc.)
    2. Frequency inference from the dataset
    3. Rolling prediction loop over the forecast period
    4. Sample aggregation into SampleForecastResult

    Subclasses must implement:
    - _load_pretrained(): Load the pretrained model/pipeline
    - _predict_samples(context, prediction_length) → np.ndarray: Run inference

    Subclasses may optionally implement:
    - _fine_tune(train_y, train_X): Fine-tune on training data

    Hyperparameters (common):
        model_name_or_path (str): HuggingFace model ID or local path. REQUIRED.
        context_length (int): Context window size. Default: 512
        prediction_length (int): Forecast horizon. Default: 48
        n_samples (int): Number of forecast samples. Default: 100
        distribution (str): Distribution name for SampleForecastResult. Default: "normal"
        device (str): Device for inference. Default: "auto"
        fine_tune (bool): Whether to fine-tune on training data. Default: False
        output_type (str): "samples" → SampleForecastResult, "quantiles" → QuantileForecastResult. Default: "samples"
        level (list[int]): Confidence levels for quantile output. Default: [80, 90]

    Example:
        >>> model = ChronosForecaster(
        ...     dataset=df, y_col="power",
        ...     hyperparameter={
        ...         "model_name_or_path": "amazon/chronos-t5-large",
        ...         "prediction_length": 48,
        ...     }
        ... )
        >>> model.fit()                      # loads pretrained weights
        >>> result = model.forecast()        # → SampleForecastResult
        >>> result.quantile(0.9, h=6)        # 90th percentile at 6-step-ahead
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        exog_cols: Optional[Union[str, int, Iterable[int], Iterable[str]]] = None,
        hyperparameter: Optional[Dict] = None,
        enable_logging: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        # Merge user hyperparameters with defaults
        self._foundation_hp = dict(_DEFAULT_FOUNDATION_HP)
        if hyperparameter:
            self._foundation_hp.update(hyperparameter)

        # Extract foundation-model-specific keys
        self._model_name_or_path: str = self._foundation_hp.pop("model_name_or_path", None)
        self._context_length: int = int(self._foundation_hp.pop("context_length", 512))
        self._prediction_length: int = int(self._foundation_hp.pop("prediction_length", 48))
        self._n_samples: int = int(self._foundation_hp.pop("n_samples", 100))
        self._distribution: str = str(self._foundation_hp.pop("distribution", "normal"))
        self._device: str = _resolve_device(str(self._foundation_hp.pop("device", "auto")))
        self._fine_tune: bool = bool(self._foundation_hp.pop("fine_tune", False))
        self._output_type: str = str(self._foundation_hp.pop("output_type", "samples"))
        self._level: List[int] = list(self._foundation_hp.pop("level", [80, 90]))

        if self._model_name_or_path is None:
            raise ValueError(
                "hyperparameter 'model_name_or_path' is required. "
                "Provide a HuggingFace model ID (e.g. 'amazon/chronos-t5-large') "
                "or a local path to pretrained weights."
            )

        # Store remaining model-specific hyperparameters for subclasses
        self._model_hp = dict(self._foundation_hp)

        # Internal state
        self._pipeline = None  # pretrained model/pipeline (set by _load_pretrained)
        self._freq: Optional[str] = None

        # Call parent — triggers prepare_dataset()
        super().__init__(
            dataset=dataset,
            y_col=y_col,
            exog_cols=exog_cols,
            hyperparameter=hyperparameter,
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self) -> None:
        """Override parent to also infer frequency from the dataset index."""
        super().prepare_dataset()
        self._infer_freq()

    def _infer_freq(self) -> None:
        """Infer time series frequency from the dataset index."""
        idx = self.dataset.index
        if isinstance(idx, pd.DatetimeIndex):
            self._freq = pd.infer_freq(idx)
            if self._freq is None:
                diffs = pd.Series(idx).diff().dropna()
                median_diff = diffs.median()
                self._freq = pd.tseries.frequencies.to_offset(median_diff).freqstr
        else:
            self._freq = "h"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> Self:
        """
        Load pretrained model and optionally fine-tune.

        Steps:
            1. Call _load_pretrained() to load weights/pipeline
            2. If fine_tune=True, call _fine_tune() with training data

        Returns:
            Self: The fitted model instance for method chaining.
        """
        if self.enable_logging:
            self.logger.info(
                f"Loading pretrained model: {self._model_name_or_path}"
            )

        self._load_pretrained()

        if self._fine_tune:
            if self.enable_logging:
                self.logger.info("Fine-tuning on training data...")
            self._fine_tune_model()
            if self.enable_logging:
                self.logger.info("Fine-tuning complete.")

        self.is_fitted_ = True

        if self.enable_logging:
            self.logger.info(f"{self.nm} ready for prediction.")

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def forecast(self) -> Union[SampleForecastResult, QuantileForecastResult]:
        """
        Generate probabilistic forecast using the full dataset as context.

        Uses the entire training dataset as context for a single-shot prediction
        of prediction_length steps ahead.

        Returns:
            SampleForecastResult (output_type="samples") or
            QuantileForecastResult (output_type="quantiles").
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call fit() first.")

        context_y = self.y.copy()

        # Trim to context_length
        if len(context_y) > self._context_length:
            context_y = context_y[-self._context_length:]

        # Build context features if available
        context_X = None
        if self.exog_cols is not None and len(self.exog_cols) > 0:
            context_X = self.X.copy()
            if len(context_X) > self._context_length:
                context_X = context_X[-self._context_length:]

        samples = self._predict_samples(
            context_y=context_y,
            prediction_length=self._prediction_length,
            context_X=context_X,
        )
        # samples shape: (n_samples, prediction_length)

        # Stack: (1, n_samples, H) for single basis time
        stacked = samples[np.newaxis, ...]
        basis_index = pd.Index([self.index[-1]])

        if self._output_type == "quantiles":
            return self._samples_to_quantile_result(stacked, basis_index)

        return SampleForecastResult(
            samples=stacked,
            basis_index=basis_index,
        )

    def predict_from_context(
        self,
        context_y: np.ndarray,
        horizon: int,
        context_X: Optional[np.ndarray] = None,
        **kwargs,
    ) -> SampleForecastResult:
        """Single-step prediction from a context window."""
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call fit() first.")

        # Trim to context_length
        if len(context_y) > self._context_length:
            context_y = context_y[-self._context_length:]
            if context_X is not None:
                context_X = context_X[-self._context_length:]

        samples = self._predict_samples(context_y, horizon, context_X)
        # samples shape: (n_samples, horizon)

        return SampleForecastResult(
            samples=samples[np.newaxis, ...],  # (1, n_samples, horizon)
            basis_index=pd.Index([0]),  # placeholder
        )

    def _samples_to_quantile_result(
        self,
        samples: np.ndarray,
        basis_index: pd.Index,
    ) -> QuantileForecastResult:
        """
        Convert sample array to QuantileForecastResult.

        Computes quantiles from the sample distribution at each confidence level.

        Args:
            samples: Shape (N_basis, n_samples, H).
            basis_index: Time index for each basis time.

        Returns:
            QuantileForecastResult with quantiles derived from samples.
        """
        quantiles_data: Dict[float, np.ndarray] = {}

        # Median
        quantiles_data[0.5] = np.median(samples, axis=1)  # (N_basis, H)

        # Level-based quantiles (e.g., level=80 → q=0.1, 0.9)
        for lv in self._level:
            alpha = (100 - lv) / 200.0
            quantiles_data[alpha] = np.quantile(samples, alpha, axis=1)
            quantiles_data[1.0 - alpha] = np.quantile(samples, 1.0 - alpha, axis=1)

        return QuantileForecastResult(
            quantiles_data=quantiles_data,
            basis_index=basis_index,
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_model_specific(self, model_path: Path) -> Path:
        """
        Save foundation model state.

        Default implementation saves the model_name_or_path reference.
        Subclasses can override to save fine-tuned weights.

        Args:
            model_path: Base path without extension.

        Returns:
            Path to saved state.
        """
        import json

        sv_path = model_path.with_suffix(".foundation.json")
        state = {
            "model_name_or_path": self._model_name_or_path,
            "context_length": self._context_length,
            "prediction_length": self._prediction_length,
            "n_samples": self._n_samples,
            "device": self._device,
            "fine_tune": self._fine_tune,
            "output_type": self._output_type,
            "level": self._level,
        }
        with open(sv_path, "w") as f:
            json.dump(state, f, indent=2)
        return sv_path

    def _load_model_specific(self, model_path: Path) -> None:
        """
        Load foundation model state.

        Default implementation reads the config and re-loads pretrained model.
        Subclasses can override to load fine-tuned weights.

        Args:
            model_path: Base path without extension.
        """
        import json

        sv_path = model_path.with_suffix(".foundation.json")
        if not sv_path.exists():
            raise FileNotFoundError(f"Foundation model config not found: {sv_path}")

        with open(sv_path) as f:
            state = json.load(f)

        self._model_name_or_path = state["model_name_or_path"]
        self._context_length = state["context_length"]
        self._prediction_length = state["prediction_length"]
        self._n_samples = state["n_samples"]
        self._device = state.get("device", self._device)
        self._fine_tune = state.get("fine_tune", False)

        self._load_pretrained()

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_pretrained(self) -> None:
        """
        Load the pretrained model/pipeline.

        Must set self._pipeline to the loaded model object.
        Use self._model_name_or_path for the model identifier and
        self._device for the target device.

        Example (Chronos):
            from chronos import ChronosPipeline
            self._pipeline = ChronosPipeline.from_pretrained(
                self._model_name_or_path,
                device_map=self._device,
            )
        """
        pass

    @abstractmethod
    def _predict_samples(
        self,
        context_y: np.ndarray,
        prediction_length: int,
        context_X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate forecast samples from the pretrained model.

        Args:
            context_y: Context target values, shape (context_length,).
            prediction_length: Number of steps to forecast.
            context_X: Optional context features, shape (context_length, n_features).
                       May be None if the model doesn't support exogenous variables
                       or if no features were specified.

        Returns:
            np.ndarray of shape (n_samples, prediction_length).

        Note:
            Some foundation models (e.g., Chronos) do not support exogenous
            variables. In that case, context_X should be ignored and this
            limitation documented in the subclass docstring.

        Example (Chronos):
            import torch
            context_tensor = torch.tensor(context_y, dtype=torch.float32)
            samples = self._pipeline.predict(
                context_tensor,
                prediction_length,
                num_samples=self._n_samples,
            )
            return samples.numpy()  # (n_samples, prediction_length)
        """
        pass

    def _fine_tune_model(self) -> None:
        """
        Fine-tune the pretrained model on training data.

        Default implementation raises NotImplementedError.
        Override in subclasses that support fine-tuning.
        """
        raise NotImplementedError(
            f"{self.nm} does not support fine-tuning. "
            f"Set fine_tune=False or implement _fine_tune_model() in the subclass."
        )
