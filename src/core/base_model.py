"""Base forecaster class for all forecasting models.

Provides the minimal interface that all forecaster classes must implement:
- is_fitted_ (bool): Training completion flag
- hyperparameter (dict): Model hyperparameters
- nm (str): User-facing display name (model_name parameter, defaults to class name)
- fit(): Train the model on data
- forecast(): Generate predictions
- _save_model_specific() / _load_model_specific(): Serialization hooks
- get_params(): Return fitted parameters (optional)
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Self, Union

import numpy as np
import pandas as pd

from .forecast_distribution import DISTRIBUTION_REGISTRY
from .forecast_results import ParametricForecastResult
from .moment_matching import mu_std_to_dist_params


class BaseForecaster(ABC):
    """Base class for all forecasting models.

    Provides:
    - is_fitted_ (bool): Training completion flag
    - hyperparameter (dict): Model hyperparameters
    - nm (str): User-facing display name (model_name parameter, defaults to class name)
    - _save_model_specific(path) / _load_model_specific(path): Serialization hooks
    - get_params(): Return fitted parameters (optional)

    Does not provide:
    - Directory management (Runner's save_dir handles this)
    - Logging (use tqdm + warnings.warn)
    - ModelConfig (runner_config.yaml handles metadata)
    - Dataset (received via fit())

    Args:
        hyperparameter: Model-specific hyperparameters.
        model_name: User-facing display name. Defaults to class name.

    Example:
        >>> model = SomeForecaster(hyperparameter={"p": 2})
        >>> model.fit(dataset=train_df, y_col="power")
        >>> result = model.forecast(horizon=24)
    """

    def __init__(
        self,
        hyperparameter: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ):
        self.hyperparameter = hyperparameter or {}
        self.is_fitted_ = False
        self._model_name = model_name

    @property
    def nm(self) -> str:
        """User-facing display name. Used by ForecastResult and Combiner."""
        return self._model_name or self.__class__.__name__

    # _registry_key is a CLASS attribute set by @MODEL_REGISTRY.register_model().
    # Used for save path and runner_config.yaml load. Not user-modifiable.

    @abstractmethod
    def _save_model_specific(self, model_path: Path) -> Path:
        """Save fitted parameters and internal state needed for prediction.

        Args:
            model_path: Base path without extension.

        Returns:
            Full path of saved model file.
        """
        ...

    @abstractmethod
    def _load_model_specific(self, model_path: Path) -> None:
        """Load fitted parameters and internal state.

        Args:
            model_path: Base path without extension.
        """
        ...

    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        """Train the model.

        Returns:
            Self: The fitted model instance for method chaining.
        """
        ...

    @abstractmethod
    def forecast(self, *args, **kwargs):
        """Generate probabilistic forecast.

        Must be implemented by subclasses.
        """
        ...

    def get_params(self) -> dict:
        """Return fitted model parameters.

        Optional -- only models that track fitted params override this.

        Raises:
            NotImplementedError: If the model does not implement get_params().
        """
        raise NotImplementedError(
            f"{self.nm} does not implement get_params()"
        )


class DeterministicForecaster(BaseForecaster):
    """Base class for deterministic forecasting models.

    Extends BaseForecaster with:
    - Historical std estimation for uncertainty quantification
    - Configurable output distribution (default: "normal")

    Additional hyperparameters (extracted before passing to the underlying model):
        previous_period (int): Number of past time steps for std estimation. Default: 24.
        distribution (str): Distribution name for ParametricDistribution. Default: "normal".

    Example:
        >>> model = XGBoostForecaster(hyperparameter={"distribution": "studentT", "df": 3})
        >>> model.fit(dataset=train_df, y_col="power", exog_cols=["wind_speed"])
    """

    def __init__(
        self,
        hyperparameter: Optional[Dict] = None,
        model_name: Optional[str] = None,
    ):
        hp = dict(hyperparameter) if hyperparameter else {}

        self.previous_period = int(hp.pop("previous_period", 24))
        self.distribution = hp.pop("distribution", "normal")

        if self.distribution not in DISTRIBUTION_REGISTRY:
            raise ValueError(
                f"Distribution '{self.distribution}' not supported. "
                f"Available: {list(DISTRIBUTION_REGISTRY.keys())}"
            )

        _dist_extra_keys = {"df", "c"}
        self.dist_extra_params: Dict[str, Any] = {
            k: hp.pop(k) for k in list(hp.keys()) if k in _dist_extra_keys
        }

        super().__init__(
            hyperparameter=hp if hp else None,
            model_name=model_name,
        )

    def fit(self, dataset: pd.DataFrame, y_col: Union[str, int],
            exog_cols=None) -> Self:
        """Train the model. Subclasses must call super().fit() or set attributes.

        Args:
            dataset: Training DataFrame.
            y_col: Target column name.
            exog_cols: Feature columns.

        Returns:
            Self for method chaining.
        """
        self.dataset = dataset.sort_index()
        self.y_col = y_col

        if exog_cols is not None:
            if isinstance(exog_cols, (str, int)):
                self.exog_cols = [exog_cols]
            else:
                self.exog_cols = list(exog_cols)
        else:
            self.exog_cols = [c for c in self.dataset.columns if c != self.y_col]

        self.y = self.dataset[self.y_col].to_numpy()
        self.X = self.dataset[self.exog_cols].to_numpy()
        self.index = self.dataset.index

        return self

    def get_historical_std(self, target_index: pd.Index) -> np.ndarray:
        """Calculate std for each time in target_index based on previous n periods.

        Args:
            target_index: Time index for which to compute historical std.

        Returns:
            np.ndarray of shape (len(target_index),).
        """
        std_values = []
        full_y_data = self.dataset[self.y_col]
        fallback_std = float(self.y.std())

        for idx in target_index:
            try:
                loc_result = self.dataset.index.get_loc(idx)
                if not isinstance(loc_result, int):
                    raise ValueError(f"Duplicate indices found for {idx}.")

                start_pos = max(0, loc_result - self.previous_period)
                end_pos = loc_result

                if start_pos < end_pos:
                    previous_data = full_y_data.iloc[start_pos:end_pos]
                    std_val = (
                        float(previous_data.std())
                        if len(previous_data) > 1
                        else fallback_std
                    )
                else:
                    std_val = fallback_std
            except (KeyError, ValueError):
                std_val = fallback_std

            std_values.append(std_val)

        return np.maximum(np.array(std_values, dtype=float), 1e-3)

    def build_forecast_result(self, mu: np.ndarray, target_index: pd.Index):
        """Build a ParametricForecastResult from predicted mu and historical std.

        Uses mu_std_to_dist_params() to convert moments to native
        distribution parameters (moment matching).

        Args:
            mu: Predicted mean values of shape (T,).
            target_index: Time index matching mu.

        Returns:
            ParametricForecastResult with shape (T, 1).
        """
        std = self.get_historical_std(target_index)
        params = mu_std_to_dist_params(
            self.distribution, mu, std, **self.dist_extra_params
        )
        # Reshape (T,) -> (T, 1) for single-horizon result
        params = {k: v.reshape(-1, 1) for k, v in params.items()}
        return ParametricForecastResult(
            dist_name=self.distribution,
            params=params,
            basis_index=target_index,
            model_name=self.nm,
        )
