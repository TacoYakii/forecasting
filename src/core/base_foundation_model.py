"""BaseFoundationModel: Base class for pretrained foundation forecasting models.

This module provides BaseFoundationModel, which extends BaseForecaster to
support pretrained time series foundation models (Chronos, Moirai, etc.)
that produce probabilistic predictions via samples or quantiles.

Unlike BaseDeepModel (NeuralForecast pattern), foundation models follow a
different lifecycle:
    1. Load pretrained weights via from_pretrained() -- no training data needed
    2. Optionally fine-tune on domain-specific data
    3. Predict via model-specific inference API -> samples or quantiles

Key responsibilities:
- Pretrained model loading (from_pretrained / download)
- Optional fine-tuning on training data (full / head / lora strategies)
- Rolling prediction over the forecast period
- Sample -> SampleForecastResult or quantile -> QuantileForecastResult conversion
- Model save/load (weights or pipeline serialization)
"""

import warnings
from abc import abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Dict, List, Optional, Self, Union

import numpy as np
import pandas as pd

from src.core.base_model import BaseForecaster
from src.core.forecast_results import QuantileForecastResult, SampleForecastResult


class FineTuneStrategy(StrEnum):
    """Fine-tuning strategy for foundation models.

    Members compare equal to their string values, so YAML/JSON configs
    that store ``"full"`` work transparently.

    Example:
        >>> FineTuneStrategy("full") == "full"
        True
        >>> FineTuneStrategy.HEAD
        'head'
    """

    FULL = "full"
    HEAD = "head"
    LORA = "lora"


# Default hyperparameters for foundation models.
_DEFAULT_FOUNDATION_HP = {
    "context_length": 512,          # encoder window (foundation models often use longer context)
    "prediction_length": 48,        # forecast horizon
    "n_samples": 100,               # number of samples per prediction
    "distribution": "normal",       # distribution name for ForecastResult
    "device": "auto",               # "cuda", "cpu", or "auto"
    "model_name_or_path": None,     # HuggingFace model ID or local path (REQUIRED)
    "output_type": "samples",       # "samples" -> SampleForecastResult, "quantiles" -> QuantileForecastResult
    "level": [80, 90],              # confidence levels for quantile output (e.g., 80 -> q=0.1, 0.9)
    # Fine-tuning hyperparameters
    "fine_tune_strategy": None,                 # None | "full" | "head" | "lora"
    "fine_tune_epochs": 5,                      # fine-tuning epoch count
    "fine_tune_lr": 1e-4,                       # fine-tuning learning rate
    "fine_tune_batch_size": 8,                  # mini-batch size
    "fine_tune_gradient_accumulation_steps": 4,  # gradient accumulation steps
    "fine_tune_mixed_precision": True,          # bf16/fp16 auto-selection
    "fine_tune_val_ratio": 0.2,                 # last 20% as validation
    "fine_tune_patience": 3,                    # early stopping patience (epochs)
    # LoRA-specific (only used when strategy="lora")
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": None,
}


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu'."""
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class BaseFoundationModel(BaseForecaster):
    """Base class for pretrained foundation time series forecasting models.

    Extends BaseForecaster to support models loaded via from_pretrained()
    (e.g., Chronos, TimesFM, Moirai, Lag-Llama) while producing
    SampleForecastResult output compatible with the existing forecast API.

    Lifecycle:
        1. __init__: Set hyperparameters
        2. fit():    Load dataset, load pretrained weights, optionally fine-tune
        3. predict(): Rolling inference -> SampleForecastResult

    This class handles:
    1. Hyperparameter extraction (context/prediction length, device, etc.)
    2. Frequency inference from the dataset
    3. Rolling prediction loop over the forecast period
    4. Sample aggregation into SampleForecastResult

    Subclasses must implement:
    - _load_pretrained(): Load the pretrained model/pipeline
    - _predict_samples(context, prediction_length) -> np.ndarray: Run inference

    Subclasses may optionally implement:
    - _fine_tune_model(): Fine-tune the pretrained model on training data

    Hyperparameters (common):
        model_name_or_path (str): HuggingFace model ID or local path. REQUIRED.
        context_length (int): Context window size. Default: 512
        prediction_length (int): Forecast horizon. Default: 48
        n_samples (int): Number of forecast samples. Default: 100
        distribution (str): Distribution name. Default: "normal"
        device (str): Device for inference. Default: "auto"
        fine_tune_strategy (str | None): Fine-tuning strategy.
            None = no fine-tuning, "full" / "head" / "lora".
        output_type (str): "samples" or "quantiles". Default: "samples"
        level (list[int]): Confidence levels for quantile output. Default: [80, 90]

    Fine-tuning hyperparameters:
        fine_tune_epochs (int): Number of epochs. Default: 5
        fine_tune_lr (float): Learning rate. Default: 1e-4
        fine_tune_batch_size (int): Mini-batch size. Default: 8
        fine_tune_gradient_accumulation_steps (int): Default: 4
        fine_tune_mixed_precision (bool): Use bf16/fp16. Default: True
        fine_tune_val_ratio (float): Validation split ratio. Default: 0.2
        fine_tune_patience (int): Early stopping patience. Default: 3

    Example:
        >>> model = ChronosForecaster(
        ...     hyperparameter={
        ...         "model_name_or_path": "amazon/chronos-t5-large",
        ...         "prediction_length": 48,
        ...         "fine_tune_strategy": "full",
        ...     }
        ... )
        >>> model.fit(dataset=df, y_col="power")
        >>> result = model.forecast()        # -> SampleForecastResult
        >>> result.to_distribution(6).ppf(0.9)  # 90th percentile at 6-step-ahead
    """

    def __init__(
        self,
        hyperparameter: Optional[Dict] = None,
        model_name: Optional[str] = None,
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
        self._output_type: str = str(self._foundation_hp.pop("output_type", "samples"))
        self._level: List[int] = list(self._foundation_hp.pop("level", [80, 90]))

        # --- Fine-tuning strategy (backward-compatible with fine_tune: bool) ---
        raw_ft = self._foundation_hp.pop("fine_tune", None)
        raw_strategy = self._foundation_hp.pop("fine_tune_strategy", None)

        if raw_strategy is not None:
            self._fine_tune_strategy: Optional[FineTuneStrategy] = (
                FineTuneStrategy(raw_strategy)
            )
        elif raw_ft is True:
            warnings.warn(
                "fine_tune=True is deprecated. "
                "Use fine_tune_strategy='full' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._fine_tune_strategy = FineTuneStrategy.FULL
        else:
            self._fine_tune_strategy = None

        # Fine-tuning hyperparameters
        self._ft_epochs: int = int(self._foundation_hp.pop("fine_tune_epochs", 5))
        self._ft_lr: float = float(self._foundation_hp.pop("fine_tune_lr", 1e-4))
        self._ft_batch_size: int = int(self._foundation_hp.pop("fine_tune_batch_size", 8))
        self._ft_grad_accum: int = int(
            self._foundation_hp.pop("fine_tune_gradient_accumulation_steps", 4)
        )
        self._ft_mixed_precision: bool = bool(
            self._foundation_hp.pop("fine_tune_mixed_precision", True)
        )
        self._ft_val_ratio: float = float(
            self._foundation_hp.pop("fine_tune_val_ratio", 0.2)
        )
        self._ft_patience: int = int(self._foundation_hp.pop("fine_tune_patience", 3))

        # LoRA-specific
        self._lora_rank: int = int(self._foundation_hp.pop("lora_rank", 8))
        self._lora_alpha: int = int(self._foundation_hp.pop("lora_alpha", 16))
        self._lora_target_modules: Optional[List[str]] = self._foundation_hp.pop(
            "lora_target_modules", None
        )

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
        self._fine_tuned_model_path: Optional[str] = None  # set after fine-tuning

        # Initialize exog_cols for load_model() compatibility
        self.exog_cols: List[str] = []

        super().__init__(
            hyperparameter=hyperparameter,
            model_name=model_name,
        )

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _infer_freq(self, dataset: pd.DataFrame) -> None:
        """Infer time series frequency from the dataset index."""
        idx = dataset.index
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

    def fit(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        exog_cols=None,
    ) -> Self:
        """Load pretrained model and optionally fine-tune.

        Args:
            dataset: Training DataFrame with a proper time index.
            y_col: Target column name.
            exog_cols: Exogenous feature columns.

        Returns:
            Self: The fitted model instance for method chaining.
        """
        self.dataset = dataset.sort_index()
        self.y_col = y_col

        if exog_cols is not None:
            if isinstance(exog_cols, (str, int)):
                self.exog_cols = [exog_cols]
            else:
                self.exog_cols = list(exog_cols)
        else:
            self.exog_cols = [c for c in self.dataset.columns if c != y_col]

        self.y = self.dataset[y_col].to_numpy()
        self.X = self.dataset[self.exog_cols].to_numpy() if self.exog_cols else np.empty((len(self.y), 0))
        self.index = self.dataset.index

        self._infer_freq(self.dataset)

        self._load_pretrained()

        if self._fine_tune_strategy is not None:
            self._fine_tune_model()

        self.is_fitted_ = True

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def forecast(self) -> Union[SampleForecastResult, QuantileForecastResult]:
        """Generate probabilistic forecast using the full dataset as context.

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
            model_name=self.nm,
        )

    def predict_from_context(
        self,
        context_y: np.ndarray,
        horizon: int,
        *,
        context_index: Optional[pd.DatetimeIndex] = None,
        context_X: Optional[np.ndarray] = None,
        future_X: Optional[np.ndarray] = None,
        future_index: Optional[pd.DatetimeIndex] = None,
    ) -> Union[SampleForecastResult, QuantileForecastResult]:
        """Single-step prediction from a context window.

        Args:
            context_y: Target values for context window, shape (context_len,).
            horizon: Number of steps to forecast.
            context_index: Time index for context window (keyword-only).
            context_X: Context features, shape (context_len, n_features) (keyword-only).
            future_X: Future features, shape (horizon, n_features) (keyword-only).
            future_index: Time index for future period (keyword-only).

        Returns:
            SampleForecastResult (output_type="samples") or
            QuantileForecastResult (output_type="quantiles").
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call fit() first.")

        # Trim to context_length
        if len(context_y) > self._context_length:
            context_y = context_y[-self._context_length:]
            if context_X is not None:
                context_X = context_X[-self._context_length:]

        samples = self._predict_samples(context_y, horizon, context_X)
        # samples shape: (n_samples, horizon)

        stacked = samples[np.newaxis, ...]  # (1, n_samples, horizon)
        basis_index = pd.Index([0])  # placeholder

        if self._output_type == "quantiles":
            return self._samples_to_quantile_result(stacked, basis_index)

        return SampleForecastResult(
            samples=stacked,
            basis_index=basis_index,
            model_name=self.nm,
        )

    def _samples_to_quantile_result(
        self,
        samples: np.ndarray,
        basis_index: pd.Index,
    ) -> QuantileForecastResult:
        """Convert sample array to QuantileForecastResult.

        Args:
            samples: Shape (N_basis, n_samples, H).
            basis_index: Time index for each basis time.

        Returns:
            QuantileForecastResult with quantiles derived from samples.
        """
        quantiles_data: Dict[float, np.ndarray] = {}

        # Median
        quantiles_data[0.5] = np.median(samples, axis=1)  # (N_basis, H)

        # Level-based quantiles (e.g., level=80 -> q=0.1, 0.9)
        for lv in self._level:
            alpha = (100 - lv) / 200.0
            quantiles_data[alpha] = np.quantile(samples, alpha, axis=1)
            quantiles_data[1.0 - alpha] = np.quantile(samples, 1.0 - alpha, axis=1)

        return QuantileForecastResult(
            quantiles_data=quantiles_data,
            basis_index=basis_index,
            model_name=self.nm,
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_model_specific(self, model_path: Path) -> Path:
        """Save foundation model state as JSON config.

        Persists all metadata needed to reconstruct the model. Subclasses
        that override this method **must** call ``super()._save_model_specific()``
        first to write the base JSON, then save model weights separately.

        Args:
            model_path: Base path without extension.

        Returns:
            Path to saved JSON config.
        """
        import json

        sv_path = model_path.with_suffix(".foundation.json")
        state = {
            "model_name_or_path": self._model_name_or_path,
            "context_length": self._context_length,
            "prediction_length": self._prediction_length,
            "n_samples": self._n_samples,
            "device": self._device,
            "output_type": self._output_type,
            "level": self._level,
            # Fine-tuning metadata
            "fine_tune_strategy": (
                str(self._fine_tune_strategy)
                if self._fine_tune_strategy
                else None
            ),
            "fine_tuned_model_path": self._fine_tuned_model_path,
            # Metadata needed for pipeline reconstruction
            "exog_cols": list(self.exog_cols) if self.exog_cols else [],
            "freq": self._freq,
        }
        with open(sv_path, "w") as f:
            json.dump(state, f, indent=2)
        return sv_path

    def _load_model_specific(self, model_path: Path) -> None:
        """Load foundation model state from JSON config and re-load model.

        Subclasses that override this method **must** call
        ``super()._load_model_specific()`` first to restore base metadata,
        then load fine-tuned weights if applicable.

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
        self._output_type = state.get("output_type", "samples")
        self._level = state.get("level", [80, 90])

        # Restore fine-tuning metadata (backward-compatible with old configs)
        raw_strategy = state.get("fine_tune_strategy", None)
        if raw_strategy is None and state.get("fine_tune", False):
            # Migrate old fine_tune: true → strategy "full"
            raw_strategy = "full"
        self._fine_tune_strategy = (
            FineTuneStrategy(raw_strategy) if raw_strategy else None
        )
        self._fine_tuned_model_path = state.get("fine_tuned_model_path", None)

        # Restore pipeline reconstruction metadata
        self.exog_cols = state.get("exog_cols", [])
        self._freq = state.get("freq", None)

        self._load_pretrained()

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_pretrained(self) -> None:
        """Load the pretrained model/pipeline.

        Must set self._pipeline to the loaded model object.

        Example (Chronos):
            >>> from chronos import ChronosPipeline
            >>> self._pipeline = ChronosPipeline.from_pretrained(
            ...     self._model_name_or_path, device_map=self._device)
        """
        pass

    @abstractmethod
    def _predict_samples(
        self,
        context_y: np.ndarray,
        prediction_length: int,
        context_X: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate forecast samples from the pretrained model.

        Args:
            context_y: Context target values, shape (context_length,).
            prediction_length: Number of steps to forecast.
            context_X: Optional context features, shape (context_length, n_features).

        Returns:
            np.ndarray of shape (n_samples, prediction_length).

        Example (Chronos):
            >>> context_tensor = torch.tensor(context_y, dtype=torch.float32)
            >>> samples = self._pipeline.predict(context_tensor, prediction_length, ...)
            >>> return samples.numpy()
        """
        pass

    def _fine_tune_model(self) -> None:
        """Fine-tune the pretrained model on training data.

        Reads ``self._fine_tune_strategy`` to determine which strategy to
        apply. Must mutate ``self._pipeline`` (or the inner model) in-place
        so that ``_predict_samples()`` works unchanged after fine-tuning.

        Subclasses that support fine-tuning must override this method.
        The default raises NotImplementedError.

        Example:
            >>> # Called automatically by fit() when fine_tune_strategy is set
            >>> model.fit(dataset=df, y_col="power")
        """
        raise NotImplementedError(
            f"{self.nm} does not support fine-tuning with strategy "
            f"'{self._fine_tune_strategy}'. Either set "
            f"fine_tune_strategy=None or implement _fine_tune_model() "
            f"in the subclass."
        )
