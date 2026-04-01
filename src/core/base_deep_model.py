"""
BaseDeepModel: Base class for deep learning forecasting models.

This module provides BaseDeepModel, which extends BaseForecaster to support
NeuralForecast-based deep learning models (DeepAR, TFT, NHITS, PatchTST, etc.)
while producing QuantileForecastResult output.

Key responsibilities:
- DataFrame → NeuralForecast format conversion (unique_id, ds, y)
- NeuralForecast wrapper creation, training, and prediction
- Quantile output → QuantileForecastResult conversion
- Model save/load via NeuralForecast.save() / .load()

NeuralForecast core pattern:
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=train_df)
    forecast_df = nf.predict()  → DataFrame with quantile columns
"""

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List, Any, Iterable, Self
from abc import abstractmethod

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss, MAE

from src.core.base_model import BaseForecaster
from src.core.forecast_results import ParametricForecastResult
from src.core.forecast_results import QuantileForecastResult


# Supported loss types
_LOSS_TYPES = ("distribution", "quantile", "implicit_quantile")

# Supported distributions for DistributionLoss
_SUPPORTED_DISTRIBUTIONS = [
    "Normal", "StudentT", "Poisson", "NegativeBinomial",
    "Tweedie", "Bernoulli", "ISQF",
]

# Default hyperparameters for all deep learning models.
# Data-shape, training, and prediction params only.
# Model architecture params (hidden_size, dropout, etc.) are left to subclasses.
_DEFAULT_DEEP_HP = {
    "input_size": -1,               # context window (-1 = auto = 3*h)
    "prediction_length": 48,        # forecast horizon h
    "max_steps": 1000,              # training steps
    "batch_size": 32,
    "windows_batch_size": 128,          # sliding windows per GPU batch (NF default 1024 — too large for many exog)
    "learning_rate": 0.001,
    "early_stop_patience_steps": -1,  # -1 = disabled
    "val_size": 0,                  # validation set size (number of timesteps)
    "level": list(range(2, 100, 2)),  # confidence levels → 99 quantiles (0.01, ..., 0.99)
    "scaler_type": "robust",        # NeuralForecast internal scaler
    "loss_type": None,              # "distribution" | "quantile" | "implicit_quantile" (None = subclass default)
    "distribution": None,           # for loss_type="distribution": "StudentT", "Normal", etc.
}

_SERIES_ID = "target"  # unique_id for single-series usage


def _resolve_device() -> str:
    """Resolve accelerator to 'gpu' or 'cpu' for PyTorch Lightning."""
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BaseDeepModel(BaseForecaster):
    """
    Base class for NeuralForecast deep learning forecasting models.

    Extends BaseForecaster to wrap NeuralForecast models while producing
    QuantileForecastResult output compatible with the existing ParametricForecastResult API.

    This class handles:
    1. DataFrame → NeuralForecast format conversion (unique_id, ds, y + exog)
    2. NeuralForecast wrapper creation and training
    3. Quantile forecast output → QuantileForecastResult conversion
    4. Model save/load via NeuralForecast serialization

    Subclasses only need to implement:
    - _create_model(): Return a configured NeuralForecast model instance

    Hyperparameters (common):
        input_size (int): Context window. -1 for auto (3*h). Default: -1
        prediction_length (int): Forecast horizon h. Default: 48
        max_steps (int): Training steps. Default: 1000
        batch_size (int): Training batch size. Default: 32
        learning_rate (float): Learning rate. Default: 0.001
        early_stop_patience_steps (int): Early stopping patience. -1 = disabled.
        val_size (int): Validation set size in timesteps. Default: 0
        level (list[int]): Confidence levels for quantile output. Default: range(2, 100, 2) → 99 quantiles
        scaler_type (str): NeuralForecast internal scaler. Default: "robust"

    Example:
        >>> model = DeepARForecaster(
        ...     dataset=df, y_col="power",
        ...     hyperparameter={"input_size": 168, "prediction_length": 48}
        ... )
        >>> model.fit()
        >>> result = model.forecast()        # → QuantileForecastResult
        >>> result.to_distribution(6).ppf(0.9)  # 90th percentile at 6-step-ahead
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        futr_cols: Optional[List[str]] = None,
        hist_cols: Optional[List[str]] = None,
        hyperparameter: Optional[Dict] = None,
        enable_logging: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        # Store futr/hist split for NeuralForecast
        self.futr_cols: List[str] = list(futr_cols) if futr_cols else []
        self.hist_cols: List[str] = list(hist_cols) if hist_cols else []

        # Merge user hyperparameters with defaults
        self._deep_hp = dict(_DEFAULT_DEEP_HP)
        if hyperparameter:
            self._deep_hp.update(hyperparameter)

        # Extract deep-learning-specific keys before passing to super
        self._input_size: int = int(self._deep_hp.pop("input_size", -1))
        self._prediction_length: int = int(self._deep_hp.pop("prediction_length", 48))
        self._max_steps: int = int(self._deep_hp.pop("max_steps", 1000))
        self._batch_size: int = int(self._deep_hp.pop("batch_size", 32))
        self._learning_rate: float = float(self._deep_hp.pop("learning_rate", 0.001))
        self._early_stop: int = int(self._deep_hp.pop("early_stop_patience_steps", -1))
        self._val_size: int = int(self._deep_hp.pop("val_size", 0))
        self._level: List[int] = list(self._deep_hp.pop("level", [80, 90]))
        self._scaler_type: str = str(self._deep_hp.pop("scaler_type", "robust"))
        self._loss_type: Optional[str] = self._deep_hp.pop("loss_type", None)
        self._distribution: Optional[str] = self._deep_hp.pop("distribution", None)
        self._accelerator: str = _resolve_device()

        # Store remaining model-specific hyperparameters for subclasses
        self._model_hp = dict(self._deep_hp)

        # Store verbose for NeuralForecast fit/predict
        self.verbose: bool = verbose

        # Internal state
        self._nf: Optional[NeuralForecast] = None  # NeuralForecast wrapper
        self._freq: Optional[str] = None            # inferred from dataset index

        # Call parent with combined exog_cols
        super().__init__(
            dataset=dataset,
            y_col=y_col,
            exog_cols=self.futr_cols + self.hist_cols or None,
            hyperparameter=hyperparameter,  # store original for info.json
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Dataset conversion
    # ------------------------------------------------------------------

    def prepare_dataset(self) -> None:
        """Override parent to also infer frequency from the dataset index."""
        super().prepare_dataset()
        self._infer_freq()

    def _create_loss(self, default_loss_type: str = "distribution",
                     default_distribution: str = "StudentT"):
        """
        Create loss and valid_loss based on loss_type hyperparameter.

        Subclasses call this in _create_model() with their own defaults.
        The user can override via hyperparameters:
            loss_type="distribution"        → DistributionLoss (parametric)
            loss_type="quantile"            → MQLoss (non-parametric, fixed quantiles)
            loss_type="implicit_quantile"   → IQLoss (non-parametric, continuous)

        NeuralForecast requires valid_loss to match the training loss type
        (e.g. MQLoss training requires MQLoss valid_loss).

        Args:
            default_loss_type: Subclass default if user doesn't specify.
            default_distribution: Default distribution for DistributionLoss.

        Returns:
            Tuple of (loss, valid_loss) NeuralForecast loss instances.
        """
        loss_type = self._loss_type or default_loss_type
        self._resolved_loss_type = loss_type  # store for _convert_forecast dispatch
        distribution = self._distribution or default_distribution

        if loss_type == "distribution":
            if distribution not in _SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"Unknown distribution '{distribution}'. "
                    f"Available: {_SUPPORTED_DISTRIBUTIONS}"
                )
            return DistributionLoss(
                distribution, level=self._level, return_params=True
            ), MAE()
        elif loss_type == "quantile":
            return MQLoss(level=self._level), MQLoss(level=self._level)
        elif loss_type == "implicit_quantile":
            return IQLoss(), IQLoss()
        else:
            raise ValueError(
                f"Unknown loss_type '{loss_type}'. "
                f"Available: {list(_LOSS_TYPES)}"
            )

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
            self._freq = "h"  # default fallback

    def _build_nf_dataframe(self) -> pd.DataFrame:
        """
        Convert the internal DataFrame into NeuralForecast format.

        NeuralForecast requires columns: [unique_id, ds, y, ...exogenous].
        Builds from the full dataset (which is the training data).

        Returns:
            DataFrame with NeuralForecast-compatible columns.
        """
        dataset = self.dataset.copy().sort_index()

        # Build NeuralForecast DataFrame
        data = {
            "unique_id": _SERIES_ID,
            "ds": dataset.index,
            "y": dataset[self.y_col].values,
        }

        # Add exogenous features
        feat_cols = self._get_feature_cols(dataset)
        for col in feat_cols:
            data[col] = dataset[col].values

        nf_df = pd.DataFrame(data).reset_index(drop=True)
        return nf_df

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Get all exogenous feature column names (futr + hist)."""
        return self.futr_cols + self.hist_cols

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self) -> Self:
        """
        Train the deep learning model via NeuralForecast.

        Creates the model (from subclass), wraps in NeuralForecast, and
        trains on the training period data.

        Returns:
            Self: The fitted model instance for method chaining.
        """
        train_df = self._build_nf_dataframe()
        model = self._create_model()

        if self.enable_logging:
            self.logger.info(
                f"Training {self.nm} with input_size={self._input_size}, "
                f"h={self._prediction_length}, max_steps={self._max_steps}"
            )

        self._nf = NeuralForecast(
            models=[model],
            freq=self._freq,
        )
        self._nf.fit(
            df=train_df,
            val_size=self._val_size,
            verbose=self.verbose,
        )

        self.is_fitted_ = True

        if self.enable_logging:
            self.logger.info("Training complete.")

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def forecast(
        self,
        future_X: Optional[np.ndarray] = None,
        future_index: Optional[pd.DatetimeIndex] = None,
    ) -> Union[ParametricForecastResult, QuantileForecastResult]:
        """
        Generate probabilistic forecast.

        Uses the full training dataset as context and calls NeuralForecast.predict().

        Args:
            future_X: Exogenous features for the forecast horizon,
                shape (prediction_length, n_features). Required if model
                was trained with exogenous features (exog_cols).
            future_index: Time index for the forecast horizon,
                shape (prediction_length,). Required together with future_X.

        Returns:
            ParametricForecastResult (DistributionLoss) or QuantileForecastResult (MQLoss/IQLoss).
        """
        if self._nf is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        futr_df = None
        if future_X is not None and future_index is not None and self.futr_cols:
            futr_dict = {"unique_id": _SERIES_ID, "ds": future_index}
            for i, col in enumerate(self.futr_cols):
                futr_dict[col] = future_X[:, i]
            futr_df = pd.DataFrame(futr_dict).reset_index(drop=True)

        forecast_df = self._nf.predict(
            futr_df=futr_df,
            level=self._level,
        )

        return self._convert_forecast(forecast_df)

    def predict_from_context(
        self,
        context_y: np.ndarray,
        context_index: pd.DatetimeIndex,
        horizon: int,
        context_X: Optional[np.ndarray] = None,
        future_X: Optional[np.ndarray] = None,
        future_index: Optional[pd.DatetimeIndex] = None,
    ) -> Union[ParametricForecastResult, QuantileForecastResult]:
        """
        Single-step prediction from a context window.

        Args:
            context_y: Target values for context window, shape (context_len,).
            context_index: Time index for context window.
            horizon: Number of steps to forecast.
            context_X: Exogenous features for context, shape (context_len, n_features).
            future_X: Exogenous features for forecast horizon, shape (horizon, n_features).
            future_index: Time index for future period.

        Returns:
            ParametricForecastResult (DistributionLoss) or QuantileForecastResult (MQLoss/IQLoss).
        """
        if self._nf is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Build NeuralForecast context DataFrame
        # context_X contains futr_cols + hist_cols (in that order)
        all_context_cols = self.futr_cols + self.hist_cols

        context_df = pd.DataFrame({
            "unique_id": _SERIES_ID,
            "ds": context_index,
            "y": context_y,
        })
        if context_X is not None:
            for i, col in enumerate(all_context_cols):
                context_df[col] = context_X[:, i]
        context_df = context_df.reset_index(drop=True)

        # Build future exog DataFrame (futr_cols only — no hist leakage)
        futr_df = None
        if future_X is not None and future_index is not None and self.futr_cols:
            futr_dict = {"unique_id": _SERIES_ID, "ds": future_index}
            for i, col in enumerate(self.futr_cols):
                futr_dict[col] = future_X[:, i]
            futr_df = pd.DataFrame(futr_dict).reset_index(drop=True)

        forecast_df = self._nf.predict(
            df=context_df,
            futr_df=futr_df,
            level=self._level,
        )

        return self._convert_forecast(forecast_df)

    def _convert_forecast(
        self, forecast_df: pd.DataFrame,
    ) -> Union[ParametricForecastResult, QuantileForecastResult]:
        """
        Convert NeuralForecast output DataFrame to a ForecastResult.

        Dispatches by loss type:
          - DistributionLoss → ParametricForecastResult (native params)
          - MQLoss / IQLoss  → QuantileForecastResult (quantile arrays)

        NeuralForecast returns columns like:
            ModelName, ModelName-median, ModelName-lo-80, ModelName-hi-80, ...
        """
        if "ds" in forecast_df.columns:
            basis_index = pd.Index([forecast_df["ds"].iloc[0]])
        else:
            basis_index = pd.Index([forecast_df.index[0]])

        model_name = self._get_model_column_name(forecast_df)
        H = (
            forecast_df[model_name].values.shape[0]
            if model_name in forecast_df.columns
            else self._prediction_length
        )

        # --- DistributionLoss: return ParametricForecastResult ---
        resolved = getattr(self, "_resolved_loss_type", None)
        if resolved == "distribution":
            return self._convert_params(forecast_df, model_name, basis_index, H)

        # --- MQLoss / IQLoss: return QuantileForecastResult ---
        return self._convert_quantile(forecast_df, model_name, basis_index, H)

    # NeuralForecast distribution name → our registry name
    _NF_DIST_MAP = {"studentt": "studentT", "normal": "normal", "poisson": "poisson"}

    # NeuralForecast param suffix → our factory key, per distribution
    _NF_PARAM_MAP: Dict[str, Dict[str, str]] = {
        "normal":   {"-loc": "loc", "-scale": "scale"},
        "studentT": {"-df": "df", "-loc": "loc", "-scale": "scale"},
        "poisson":  {"-loc": "mu"},
    }

    def _convert_params(
        self,
        forecast_df: pd.DataFrame,
        model_name: str,
        basis_index: pd.Index,
        H: int,
    ) -> ParametricForecastResult:
        """
        Convert DistributionLoss output (with return_params=True) to ParametricForecastResult.

        Extracts native distribution parameters directly from the forecast DataFrame
        columns (e.g. ModelName-loc, ModelName-scale, ModelName-df).
        """
        # Resolve distribution name
        dist_name = (self._distribution or "normal").lower()
        dist_name = self._NF_DIST_MAP.get(dist_name, "normal")

        # Try to extract native params from return_params=True columns
        param_map = self._NF_PARAM_MAP.get(dist_name, {})
        params: Dict[str, np.ndarray] = {}
        has_native = False

        for nf_suffix, factory_key in param_map.items():
            col = f"{model_name}{nf_suffix}"
            if col in forecast_df.columns:
                params[factory_key] = forecast_df[col].values
                has_native = True

        if has_native and len(params) == len(param_map):
            # All native params found — apply safety clamps
            if "scale" in params:
                params["scale"] = np.maximum(params["scale"], 1e-9)
            if "df" in params:
                params["df"] = np.maximum(params["df"], 2.01)
            if "mu" in params:
                params["mu"] = np.maximum(params["mu"], 1e-9)
        else:
            # Fallback: derive from point prediction + quantile span
            from scipy.stats import norm as norm_dist

            mu = forecast_df[model_name].values
            sigma = np.ones_like(mu) * 1e-9
            if self._level:
                widest = max(self._level)
                lo_col = f"{model_name}-lo-{widest}"
                hi_col = f"{model_name}-hi-{widest}"
                if lo_col in forecast_df.columns and hi_col in forecast_df.columns:
                    lo = forecast_df[lo_col].values
                    hi = forecast_df[hi_col].values
                    alpha = (100 - widest) / 200.0
                    z_span = norm_dist.ppf(1.0 - alpha) - norm_dist.ppf(alpha)
                    sigma = np.maximum((hi - lo) / z_span, 1e-9)
            params = {"loc": mu, "scale": sigma}
            dist_name = "normal"  # fallback to normal if params unavailable

        # Reshape (H,) → (1, H) for single-origin result
        params = {k: v.reshape(1, -1) for k, v in params.items()}
        return ParametricForecastResult(
            dist_name=dist_name,
            params=params,
            basis_index=basis_index,
            model_name=type(self).__name__,
        )

    def _convert_quantile(
        self,
        forecast_df: pd.DataFrame,
        model_name: str,
        basis_index: pd.Index,
        H: int,
    ) -> QuantileForecastResult:
        """Convert MQLoss / IQLoss output to QuantileForecastResult."""
        quantiles_data: Dict[float, np.ndarray] = {}

        # Point prediction → 0.5
        if model_name in forecast_df.columns:
            quantiles_data[0.5] = forecast_df[model_name].values.reshape(1, -1)

        # Median column (overrides point if present)
        median_col = f"{model_name}-median"
        if median_col in forecast_df.columns:
            quantiles_data[0.5] = forecast_df[median_col].values.reshape(1, -1)

        # Level-based columns: lo-XX, hi-XX
        for lv in self._level:
            lo_col = f"{model_name}-lo-{lv}"
            hi_col = f"{model_name}-hi-{lv}"
            alpha = (100 - lv) / 200.0

            if lo_col in forecast_df.columns:
                quantiles_data[alpha] = forecast_df[lo_col].values.reshape(1, -1)
            if hi_col in forecast_df.columns:
                quantiles_data[1.0 - alpha] = forecast_df[hi_col].values.reshape(1, -1)

        return QuantileForecastResult(
            quantiles_data=quantiles_data,
            basis_index=basis_index,
            model_name=type(self).__name__,
        )

    def _get_model_column_name(self, forecast_df: pd.DataFrame) -> str:
        """Find the base model column name from the forecast DataFrame."""
        skip = {"ds", "unique_id"}
        for col in forecast_df.columns:
            if col not in skip and "-" not in col:
                return col
        # Fallback: first non-skip column
        for col in forecast_df.columns:
            if col not in skip:
                return col.split("-")[0]
        raise RuntimeError("Cannot find model column in forecast DataFrame")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def _save_model_specific(self, model_path: Path) -> Path:
        """
        Save model via NeuralForecast serialization.

        Args:
            model_path: Base path without extension.

        Returns:
            Path: Directory where the model was saved.
        """
        sv_dir = model_path.with_suffix(".nf")
        if self._nf is not None:
            self._nf.save(
                path=str(sv_dir),
                overwrite=True,
            )
        return sv_dir

    def _load_model_specific(self, model_path: Path) -> None:
        """
        Load model from NeuralForecast serialization.

        Args:
            model_path: Base path without extension.
        """
        sv_dir = model_path.with_suffix(".nf")
        if not sv_dir.exists():
            raise FileNotFoundError(f"NeuralForecast model directory not found: {sv_dir}")

        self._nf = NeuralForecast.load(path=str(sv_dir))

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_model(self):
        """
        Create and return a configured NeuralForecast model instance.

        Subclasses must implement this to return their specific model
        (e.g., DeepAR, TFT, NHITS, PatchTST).

        The model should use self._input_size, self._prediction_length,
        self._max_steps, self._batch_size, self._learning_rate,
        self._scaler_type, and self._model_hp for configuration.

        Returns:
            A NeuralForecast model instance (e.g., DeepAR(h=48, ...)).

        Example (DeepAR subclass):
            def _create_model(self):
                from neuralforecast.models import DeepAR
                from neuralforecast.losses.pytorch import DistributionLoss
                return DeepAR(
                    h=self._prediction_length,
                    input_size=self._input_size,
                    max_steps=self._max_steps,
                    batch_size=self._batch_size,
                    learning_rate=self._learning_rate,
                    scaler_type=self._scaler_type,
                    loss=DistributionLoss("StudentT", level=self._level),
                    **self._model_hp,
                )
        """
        pass
