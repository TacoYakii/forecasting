import logging
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Self, Tuple, Union

import numpy as np
import pandas as pd

from .config import BaseConfig
from .forecast_distribution import DISTRIBUTION_REGISTRY
from .forecast_results import ParametricForecastResult
from .moment_matching import mu_std_to_dist_params

# ---------------------------------------------------------------------------
# Model metadata dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig(BaseConfig):
    """Structured metadata for a fitted model.

    Saved as ``model_config.yaml`` in the model's base directory.

    Attributes:
        model_name: Class name of the model.
        created: ISO-format timestamp of creation.
        hyperparameter: Hyperparameters used (or "default").
        dataset_setting: Target / feature column info.

    Example:
        >>> info = ModelConfig(model_name="XGBoostForecaster")
        >>> info.save(Path("res/XGBoostForecaster/0/model_config.yaml"))
    """

    model_name: str = ""
    created: str = ""
    hyperparameter: Any = field(default_factory=dict)
    dataset_setting: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for all forecasting models.

    Provides:
    - Automatic directory structure creation and management
    - Comprehensive logging system with file and console output
    - Structured metadata (ModelConfig) with YAML persistence
    - Model serialization and deserialization
    - Training state tracking and validation
    - Experiment organization with auto-incrementing directories

    Directory Structure:
        res/
        └── ModelClassName/
            └── 0/  # Auto-incremented experiment number
                ├── ModelClassName.log
                ├── model_config.yaml
                └── ModelClassName_model.{pkl,cbm,pth,json}

    Attributes:
        nm (str): Name of the model class (read-only property).
        base_dir (Path): Base directory for this model instance.
        log_file (Path): Path to the log file.
        model_config (ModelConfig): Structured model metadata.
        logger (logging.Logger): Logger instance for this model.
        enable_logging (bool): Whether logging is enabled.

    Args:
        hyperparameter: Model-specific hyperparameters.
        enable_logging: Enable file and console logging (default: True).
        save_dir: Custom directory for saving model files.
                  If None, creates a 'res' directory in the subclass code folder.
        verbose: Enable console output for logging (default: False).

    Example:
        >>> model = SomeForecaster(dataset=df, y_col='power')
        >>> model.fit()
        >>> model.model_config.save(model.base_dir / "model_config.yaml")
    """

    def __init__(
        self,
        hyperparameter: Optional[Dict[str, Any]] = None,
        enable_logging: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        model_name: Optional[str] = None,
    ):
        self._subclass_nm = model_name or self.__class__.__name__
        self.enable_logging = enable_logging

        # Set up directories (created lazily on save_model or enable_logging)
        if save_dir is None:
            self.base_dir, self.log_file = self._get_default_dirs_setting()
        else:
            self.base_dir = Path(save_dir)
            self.log_file = self.base_dir / f"{self._subclass_nm}.log"

        # Fitting indicator
        self.is_fitted_ = False

        # Hyperparameters
        if hyperparameter is None:
            self.hyperparameter = {}
        else:
            self.hyperparameter = hyperparameter

        # Structured metadata
        self.model_config = ModelConfig(
            model_name=self._subclass_nm,
            created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            hyperparameter=(
                hyperparameter if hyperparameter is not None else "default"
            ),
        )

        # Set up logging
        if self.enable_logging:
            self._setup_logging(verbose)

    @property
    def nm(self) -> str:
        """Get the name of the model class.

        Returns:
            str: The class name of the current model instance.
        """
        return self._subclass_nm

    def _get_default_dirs_setting(self) -> Tuple[Path, Path]:
        """Create default directory structure for the model.

        Creates a 'res' directory in the same folder as the subclass code,
        then creates class-specific and experiment-specific subdirectories.

        Returns:
            Tuple[Path, Path]: (experiment_directory, log_file_path).

        Raises:
            ValueError: If the module file path cannot be determined.
        """
        module_file = sys.modules[self.__class__.__module__].__file__
        if module_file is None:
            raise ValueError("Cannot determine module file path")
        current_file = Path(module_file).resolve()

        code_dir = current_file.parent
        res_dir = code_dir / "res"

        class_name = self.__class__.__name__
        class_dir = res_dir / class_name

        if class_dir.exists():
            exp_nums = [
                int(path.name)
                for path in class_dir.iterdir()
                if path.is_dir() and re.match(r"^\d+$", path.name)
            ]
        else:
            exp_nums = []
        next_exp_num = max(exp_nums, default=-1) + 1
        exp_dir = class_dir / str(next_exp_num)

        log_file = exp_dir / f"{class_name}.log"

        return exp_dir, log_file

    def _setup_logging(self, verbose: bool) -> None:
        """Set up logging with file and optional console handlers.

        Args:
            verbose: If True, also log to console.
        """
        self.logger = logging.getLogger(f"{self._subclass_nm}")
        self.logger.setLevel(logging.INFO)

        self.logger.handlers.clear()

        self.base_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(self.log_file), mode="w")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info(f"Initialized {self._subclass_nm} model")

    def save_model_config(self) -> Path:
        """Save model metadata to model_config.yaml.

        Returns:
            Path to the saved config file.
        """
        self.model_config.hyperparameter = self.hyperparameter
        config_path = self.base_dir / "model_config.yaml"
        self.model_config.save(config_path)
        return config_path



class BaseForecaster(BaseModel):
    """Base class for all forecasting models.

    Receives training data, extracts y/X/index, and provides a unified
    interface for fit/forecast/save/load.  The dataset passed in is treated
    as training data — evaluation-period management is handled externally
    by RollingForecaster.

    Attributes:
        dataset (pd.DataFrame): Training dataset (sorted by index).
        y_col (str | int):      Target column name or index.
        exog_cols (list):       Exogenous feature column names.
        y (np.ndarray):         Target values, shape (N,).
        X (np.ndarray):         Feature matrix, shape (N, n_features).
        index (pd.Index):       Time index of the dataset.

    Args:
        dataset:    Training DataFrame with a proper index.
        y_col:      Target column name (str) or positional index (int).
        exog_cols:  Exogenous feature columns. None → all columns except y_col.
        hyperparameter: Model-specific hyperparameters.
        enable_logging: Enable file/console logging.
        save_dir:   Custom directory for model artifacts.
        verbose:    Enable console log output.

    Example:
        >>> train_df = df.loc[:'2023-06-30']
        >>> model = MyForecaster(dataset=train_df, y_col='power').fit()
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
        model_name: Optional[str] = None,
    ):
        super().__init__(
            hyperparameter=hyperparameter,
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
            model_name=model_name,
        )

        self.dataset = dataset
        self.y_col = y_col
        self.exog_cols = exog_cols

        # Store dataset setting in model_config
        self.model_config.dataset_setting = {
            "y_col": y_col,
            "exog_cols": exog_cols,
        }

        # Extract y, X, index from the dataset
        self.prepare_dataset()

    def prepare_dataset(self) -> None:
        """Extract target (y), features (X), and index from the dataset.

        After this call the following attributes are set:
            self.y      — np.ndarray, shape (N,)
            self.X      — np.ndarray, shape (N, n_features)
            self.exog_cols — list[str]
            self.index  — pd.Index
        """
        self.dataset = self._sort_dataset_by_index(self.dataset)

        # Resolve column names
        self.y_col = self._resolve_column(self.y_col)
        if self.exog_cols is not None:
            if isinstance(self.exog_cols, (str, int)):
                self.exog_cols = [self._resolve_column(self.exog_cols)]
            else:
                self.exog_cols = [self._resolve_column(c) for c in self.exog_cols]
        else:
            self.exog_cols = [c for c in self.dataset.columns if c != self.y_col]

        # Extract arrays
        self.y = self.dataset[self.y_col].to_numpy()
        self.X = self.dataset[self.exog_cols].to_numpy()
        self.index = self.dataset.index

    def _sort_dataset_by_index(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Sort dataset by index, handling different index types.

        Args:
            dataset: Input DataFrame to sort.

        Returns:
            pd.DataFrame: Sorted DataFrame.
        """
        try:
            return dataset.sort_index()
        except Exception:
            try:
                if dataset.index.dtype == "object":
                    datetime_index = pd.to_datetime(dataset.index, errors="coerce")
                    if not datetime_index.isna().all():
                        sorted_dataset = dataset.copy()
                        sorted_dataset.index = datetime_index
                        return sorted_dataset.sort_index()
                    else:
                        return dataset.sort_index()
                else:
                    return dataset.sort_index()
            except Exception:
                if self.enable_logging:
                    self.logger.warning(
                        "Could not sort dataset index, using original order"
                    )
                return dataset

    def _resolve_column(self, col: Union[str, int]) -> Union[str, int]:
        """Resolve a column name with case-insensitive matching.

        If col (str) doesn't exist in the dataset but a case-insensitive match
        is found, returns the actual column name. Raises KeyError if no match
        or multiple matches are found. Int columns are returned as-is.
        """
        if isinstance(col, int) or col in self.dataset.columns:
            return col

        col_lower = col.lower()
        matches = [
            c
            for c in self.dataset.columns
            if isinstance(c, str) and c.lower() == col_lower
        ]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) == 0:
            raise KeyError(
                f"Column '{col}' not found in dataset (case-insensitive search)"
            )
        else:
            raise KeyError(
                f"Column '{col}' has multiple case-insensitive matches: {matches}. "
                f"Please specify the exact column name."
            )

    def save_model(self, model_path: Optional[Union[str, Path]] = None) -> Path:
        """Save the model and its metadata.

        Saves both common metadata (ModelConfig as YAML) and the model-specific
        data using the format appropriate for each model type.

        Args:
            model_path: Optional custom path for saving. If None, uses
                default naming in the model's base directory.

        Returns:
            Path: Path to the saved model file.

        Example:
            >>> model.save_model()  # Uses default path
            >>> model.save_model("custom_model")  # Custom path
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if model_path is None:
            model_path = self.base_dir / f"{self._subclass_nm}_model"
        else:
            model_path = Path(model_path)

        model_path = model_path.with_suffix("")

        self.save_model_config()
        model_file_path = self._save_model_specific(model_path)

        if self.enable_logging:
            self.logger.info(f"Model saved to {model_file_path}")

        return model_file_path

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a previously saved model and its metadata.

        Args:
            model_path: Path to the saved model (with or without extension).

        Example:
            >>> model.load_model("saved_model.pkl")
            >>> model.load_model("saved_model")  # Extension optional
        """
        model_path = Path(model_path)
        base_path = model_path.with_suffix("")

        self._load_model_specific(base_path)

        self.is_fitted_ = True

        if self.enable_logging:
            self.logger.info(f"Model loaded from {model_path}")

    @abstractmethod
    def forecast(self, *args, **kwargs):
        """Generate probabilistic forecast.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def fit(self) -> Self:
        """Train the model.

        Returns:
            Self: The fitted model instance for method chaining.
        """
        pass

    @abstractmethod
    def _save_model_specific(self, model_path: Path) -> Path:
        """Save model using format specific to the model type.

        Args:
            model_path: Base path without extension.

        Returns:
            Full path of saved model file.
        """
        pass

    @abstractmethod
    def _load_model_specific(self, model_path: Path) -> None:
        """Load model using format specific to the model type.

        Args:
            model_path: Base path without extension.
        """
        pass


class DeterministicForecaster(BaseForecaster):
    """Base class for deterministic forecasting models.

    Extends BaseForecaster with:
    - Historical std estimation for uncertainty quantification
    - Configurable output distribution (default: "normal")

    Additional hyperparameters (extracted before passing to the underlying model):
        previous_period (int): Number of past time steps for std estimation. Default: 24.
        distribution (str): Distribution name for ParametricDistribution. Default: "normal".
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        exog_cols: Optional[Union[str, int, Iterable[int], Iterable[str]]] = None,
        hyperparameter: Optional[Dict] = None,
        enable_logging: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = False,
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
            dataset=dataset,
            y_col=y_col,
            exog_cols=exog_cols,
            hyperparameter=hp if hp else None,
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
            model_name=model_name,
        )

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
        # Reshape (T,) → (T, 1) for single-horizon result
        params = {k: v.reshape(-1, 1) for k, v in params.items()}
        return ParametricForecastResult(
            dist_name=self.distribution,
            params=params,
            basis_index=target_index,
            model_name=self.nm,
        )
