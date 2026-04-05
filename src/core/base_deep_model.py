"""
BaseDeepModel: Base class for deep learning forecasting models.

This module provides BaseDeepModel, which extends BaseForecaster to support
NeuralForecast-based deep learning models (DeepAR, TFT, NHITS, PatchTST, etc.)
while producing QuantileForecastResult output.

Key responsibilities:
- DataFrame -> NeuralForecast format conversion (unique_id, ds, y)
- NeuralForecast wrapper creation, training, and prediction
- Quantile output -> QuantileForecastResult conversion
- Model save/load via NeuralForecast.save() / .load()

NeuralForecast core pattern:
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=train_df)
    forecast_df = nf.predict()  -> DataFrame with quantile columns
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

_SERIES_ID = "target"  # unique_id for single-series usage


def _resolve_device() -> str:
    """Resolve accelerator to 'gpu' or 'cpu' for PyTorch Lightning."""
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BaseDeepModel(BaseForecaster):
    """Base class for NeuralForecast deep learning forecasting models.

    Extends BaseForecaster to wrap NeuralForecast models while producing
    QuantileForecastResult output compatible with the existing ParametricForecastResult API.

    Subclass flags:
        SUPPORTS_HIST_EXOG (bool): Whether the model supports hist_exog.
            Default False. Subclasses that support hist_exog (e.g., TFT)
            set this to True. If False, hist_cols passed to fit() are
            silently ignored with a warning.

    This class handles:
    1. DataFrame -> NeuralForecast format conversion (unique_id, ds, y + exog)
    2. NeuralForecast wrapper creation and training
    3. Quantile forecast output -> QuantileForecastResult conversion
    4. Model save/load via NeuralForecast serialization

    Subclasses only need to implement:
    - _create_model(): Return a configured NeuralForecast model instance

    Hyperparameters (common):
        input_size (int): Context window. -1 for auto (3*h). Default: -1
        prediction_length (int): Forecast horizon h. Default: 48
        h_train (int): Truncated BPTT length for recurrent models (DeepAR).
            -1 = use prediction_length (train on full horizon). Default: -1.
            NeuralForecast's default of 1 trains only 1-step-ahead, causing
            severe error compounding during 48-step autoregressive rollout.
        max_steps (int): Training steps. Default: 1000
        batch_size (int): Training batch size. Default: 32
        learning_rate (float): Learning rate. Default: 0.001
        early_stop_patience_steps (int): Early stopping patience. -1 = disabled.
        val_size (int): Validation set size in timesteps. Default: 0
        level (list[int]): Confidence levels for quantile output. Default: range(2, 100, 2) -> 99 quantiles
        scaler_type (str): NeuralForecast internal scaler. Default: "robust"

    Example:
        >>> model = DeepARForecaster(
        ...     hyperparameter={"input_size": 168, "prediction_length": 48}
        ... )
        >>> model.fit(dataset=df, y_col="power",
        ...           futr_cols=["nwp_wspd"], hist_cols=["obs_wspd"])
        >>> result = model.forecast()        # -> QuantileForecastResult
        >>> result.to_distribution(6).ppf(0.9)  # 90th percentile at 6-step-ahead
    """

    SUPPORTS_HIST_EXOG: bool = False

    def __init__(
        self,
        hyperparameter: Optional[Dict] = None,
        model_name: Optional[str] = None,
    ):
        hp = dict(hyperparameter) if hyperparameter else {}

        # Extract keys managed by BaseDeepModel; rest goes to subclass
        self._input_size: int = int(hp.pop("input_size", -1))
        self._prediction_length: int = int(hp.pop("prediction_length", 48))
        self._h_train: int = int(hp.pop("h_train", -1))
        self._max_steps: int = int(hp.pop("max_steps", 1000))
        self._batch_size: int = int(hp.pop("batch_size", 32))
        self._learning_rate: float = float(hp.pop("learning_rate", 0.001))
        self._early_stop: int = int(hp.pop("early_stop_patience_steps", -1))
        self._val_size: int = int(hp.pop("val_size", 0))
        self._level: List[int] = list(hp.pop("level", [80, 90]))
        self._scaler_type: Optional[str] = hp.pop("scaler_type", None)
        self._loss_type: Optional[str] = hp.pop("loss_type", None)
        self._distribution: Optional[str] = hp.pop("distribution", None)
        self._accelerator: str = _resolve_device()

        # Remaining keys are model-specific (architecture params etc.)
        self._model_hp = hp

        # Internal state
        self._nf: Optional[NeuralForecast] = None  # NeuralForecast wrapper
        self._freq: Optional[str] = None            # inferred from dataset index

        # futr/hist cols (set in fit)
        self.futr_cols: List[str] = []
        self.hist_cols: List[str] = []

        # verbose for NeuralForecast fit/predict (default False)
        self.verbose: bool = False
        self._enable_progress_bar: bool = True

        super().__init__(
            hyperparameter=hyperparameter,
            model_name=model_name,
        )

    # ------------------------------------------------------------------
    # Loss creation
    # ------------------------------------------------------------------

    def _create_loss(self, default_loss_type: str = "distribution",
                     default_distribution: str = "StudentT"):
        """Create loss and valid_loss based on loss_type hyperparameter.

        Subclasses call this in _create_model() with their own defaults.

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

    # ------------------------------------------------------------------
    # Dataset conversion
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
            self._freq = "h"  # default fallback

    def _build_nf_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Convert a DataFrame into NeuralForecast format.

        NeuralForecast requires columns: [unique_id, ds, y, ...exogenous].

        Args:
            dataset: Training DataFrame with time index.

        Returns:
            DataFrame with NeuralForecast-compatible columns.
        """
        dataset = dataset.sort_index()

        # Build NeuralForecast DataFrame
        data = {
            "unique_id": _SERIES_ID,
            "ds": dataset.index,
            "y": dataset[self.y_col].values,
        }

        # Add exogenous features
        feat_cols = self.futr_cols + self.hist_cols
        for col in feat_cols:
            data[col] = dataset[col].values

        nf_df = pd.DataFrame(data).reset_index(drop=True)
        return nf_df

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        futr_cols: Optional[List[str]] = None,
        hist_cols: Optional[List[str]] = None,
    ) -> Self:
        """Train the deep learning model via NeuralForecast.

        Creates the model (from subclass), wraps in NeuralForecast, and
        trains on the training period data.

        Args:
            dataset: Training DataFrame with a proper time index.
            y_col: Target column name.
            futr_cols: Future-known exogenous columns (NWP forecasts etc.).
            hist_cols: Historical-only exogenous columns (SCADA obs etc.).

        Returns:
            Self: The fitted model instance for method chaining.
        """
        self.dataset = dataset.sort_index()
        self.y_col = y_col
        self.futr_cols = list(futr_cols) if futr_cols else []

        # Check hist_exog support
        if hist_cols and not self.SUPPORTS_HIST_EXOG:
            import warnings
            warnings.warn(
                f"{self.nm} does not support hist_cols={hist_cols}. "
                f"Ignoring hist_cols. Use a model with SUPPORTS_HIST_EXOG=True "
                f"(e.g., TFT) if historical exogenous variables are needed.",
                stacklevel=2,
            )
            hist_cols = None

        self.hist_cols = list(hist_cols) if hist_cols else []

        self.y = self.dataset[y_col].to_numpy()
        self.exog_cols = self.futr_cols + self.hist_cols
        self.X = self.dataset[self.exog_cols].to_numpy() if self.exog_cols else np.empty((len(self.y), 0))
        self.index = self.dataset.index

        self._infer_freq(self.dataset)

        train_df = self._build_nf_dataframe(self.dataset)
        model = self._create_model()

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

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def forecast(
        self,
        future_X: Optional[np.ndarray] = None,
        future_index: Optional[pd.DatetimeIndex] = None,
    ) -> Union[ParametricForecastResult, QuantileForecastResult]:
        """Generate probabilistic forecast.

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

        if self.futr_cols:
            if future_X is None or future_index is None:
                raise ValueError(
                    f"Model was trained with futr_cols={self.futr_cols}, "
                    f"but future_X or future_index is None. "
                    f"Provide future exogenous data for the forecast horizon."
                )

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
        horizon: int,
        *,
        context_index: pd.DatetimeIndex,
        context_X: Optional[np.ndarray] = None,
        future_X: Optional[np.ndarray] = None,
        future_index: Optional[pd.DatetimeIndex] = None,
    ) -> Union[ParametricForecastResult, QuantileForecastResult]:
        """Single-step prediction from a context window.

        Args:
            context_y: Target values for context window, shape (context_len,).
            horizon: Number of steps to forecast.
            context_index: Time index for context window (keyword-only, required).
            context_X: Exogenous features for context, shape (context_len, n_features) (keyword-only).
            future_X: Exogenous features for forecast horizon, shape (horizon, n_features) (keyword-only).
            future_index: Time index for future period (keyword-only).

        Returns:
            ParametricForecastResult (DistributionLoss) or QuantileForecastResult (MQLoss/IQLoss).
        """
        if self._nf is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if self.futr_cols:
            if future_X is None or future_index is None:
                raise ValueError(
                    f"Model was trained with futr_cols={self.futr_cols}, "
                    f"but future_X or future_index is None. "
                    f"Provide future exogenous data for the forecast horizon."
                )

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

        # Build futr_df for the forecast horizon only.
        # NeuralForecast 3.x expects futr_df timestamps to match
        # make_future_dataframe(df) output: the h steps right after
        # the last timestamp in context_df.
        futr_df = None
        if self.futr_cols and future_X is not None:
            n_futr = len(self.futr_cols)
            n_future = future_X.shape[0]
            # Generate expected future timestamps from context
            last_ts = context_index[-1]
            freq = self._freq or "h"
            expected_ds = pd.date_range(
                start=last_ts, periods=n_future + 1, freq=freq,
            )[1:]  # skip last_ts itself

            futr_dict: dict = {
                "unique_id": _SERIES_ID,
                "ds": expected_ds,
            }
            for i, col in enumerate(self.futr_cols):
                futr_dict[col] = future_X[:n_future, i]
            futr_df = pd.DataFrame(futr_dict)

        forecast_df = self._nf.predict(
            df=context_df,
            futr_df=futr_df,
            level=self._level,
        )

        return self._convert_forecast(forecast_df)

    def _convert_forecast(
        self, forecast_df: pd.DataFrame,
    ) -> Union[ParametricForecastResult, QuantileForecastResult]:
        """Convert NeuralForecast output DataFrame to a ForecastResult.

        Dispatches by loss type:
          - DistributionLoss -> ParametricForecastResult (native params)
          - MQLoss / IQLoss  -> QuantileForecastResult (quantile arrays)
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

    # NeuralForecast distribution name -> our registry name
    _NF_DIST_MAP = {"studentt": "studentT", "normal": "normal", "poisson": "poisson"}

    # NeuralForecast param suffix -> our factory key, per distribution
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
        """Convert DistributionLoss output (with return_params=True) to ParametricForecastResult."""
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
            # All native params found -- apply safety clamps
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

        # Reshape (H,) -> (1, H) for single-origin result
        params = {k: v.reshape(1, -1) for k, v in params.items()}
        return ParametricForecastResult(
            dist_name=dist_name,
            params=params,
            basis_index=basis_index,
            model_name=self.nm,
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

        # Point prediction -> 0.5
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
            model_name=self.nm,
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
        """Save model via NeuralForecast serialization.

        Also saves futr_cols/hist_cols so they can be restored on load.

        Args:
            model_path: Base path without extension.

        Returns:
            Path: Directory where the model was saved.
        """
        import json as _json

        sv_dir = model_path.with_suffix(".nf")
        if self._nf is not None:
            self._nf.save(
                path=str(sv_dir),
                overwrite=True,
            )

        # Persist futr/hist cols and resolved loss type alongside the NF checkpoint
        cols_path = model_path.with_suffix(".deep_cols.json")
        meta = {
            "futr_cols": self.futr_cols,
            "hist_cols": self.hist_cols,
            "_resolved_loss_type": getattr(self, "_resolved_loss_type", None),
        }
        with open(cols_path, "w") as f:
            _json.dump(meta, f)

        return sv_dir

    def _load_model_specific(self, model_path: Path) -> None:
        """Load model from NeuralForecast serialization.

        Also restores futr_cols/hist_cols if saved alongside the checkpoint.

        Args:
            model_path: Base path without extension.
        """
        import json as _json

        sv_dir = model_path.with_suffix(".nf")
        if not sv_dir.exists():
            raise FileNotFoundError(f"NeuralForecast model directory not found: {sv_dir}")

        self._nf = NeuralForecast.load(path=str(sv_dir))

        # Restore futr/hist cols
        cols_path = model_path.with_suffix(".deep_cols.json")
        if cols_path.exists():
            with open(cols_path) as f:
                cols = _json.load(f)
            self.futr_cols = cols.get("futr_cols", [])
            self.hist_cols = cols.get("hist_cols", [])
            resolved = cols.get("_resolved_loss_type")
            if resolved is not None:
                self._resolved_loss_type = resolved

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_model(self):
        """Create and return a configured NeuralForecast model instance.

        Subclasses must implement this to return their specific model
        (e.g., DeepAR, TFT, NHITS, PatchTST).

        Returns:
            A NeuralForecast model instance (e.g., DeepAR(h=48, ...)).

        Example (DeepAR subclass):
            >>> def _create_model(self):
            ...     from neuralforecast.models import DeepAR
            ...     return DeepAR(h=self._prediction_length, ...)
        """
        pass
