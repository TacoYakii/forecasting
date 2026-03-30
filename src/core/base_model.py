import numpy as np
import sys, re
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, Iterable, List, Self
from abc import ABC, abstractmethod
from datetime import datetime

from .forecast_distribution import DISTRIBUTION_REGISTRY
from .moment_matching import mu_std_to_dist_params
from .forecast_results import ParametricForecastResult


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models in the wind power forecasting system.
    
    This class provides a unified interface for model management, including:
    - Automatic directory structure creation and management
    - Comprehensive logging system with file and console output
    - Model serialization and deserialization with metadata tracking
    - Information management with JSON persistence
    - Training state tracking and validation
    - Experiment organization with auto-incrementing directories
    
    The base class uses a hybrid serialization approach where common metadata
    is stored in JSON format while model-specific data uses the optimal format
    for each model type (pickle, native formats, etc.).
    
    Directory Structure:
        res/
        └── ModelClassName/
            └── 0/  # Auto-incremented experiment number
                ├── ModelClassName.log
                ├── info.json (JSON format)
                └── ModelClassName_model.{pkl,cbm,pth,json}
    
    Attributes:
        nm (str): Name of the model class (read-only property)
        base_dir (Path): Base directory for this model instance
        log_file (Path): Path to the log file
        info (Dict[str, Any]): Model information dictionary
        logger (logging.Logger): Logger instance for this model
        enable_logging (bool): Whether logging is enabled
        
    Args:
        info: Optional information dictionary for the model
        enable_logging: Whether to enable file and console logging (default: True)
        save_dir: Optional custom directory for saving model files. 
                  If None, creates a 'res' directory in the same folder as the subclass code
        verbose: Whether to enable console output for logging (default: False)

    """
    
    def __init__(
        self, 
        hyperparameter: Optional[Dict[str, Any]] = None, 
        info: Optional[Dict[str, Any]] = None, 
        enable_logging: bool = False, 
        save_dir: Optional[str] = None,
        verbose: bool = False
        ): 
        self._subclass_nm = self.__class__.__name__ 
        self.enable_logging = enable_logging
        
        # Set up directories
        if save_dir is None:
            self.base_dir, self.log_file = self._get_default_dirs_setting()
        else:
            self.base_dir, self.log_file = Path(save_dir), Path(save_dir, f"{self._subclass_nm}.log")
            
        # Fitting indicator 
        self.is_fitted_ = False 
        
        # Update Information 
        self.info = info or {}
        
        if hyperparameter is None: 
            self.hyperparameter = {} 
        else: 
            self.hyperparameter = hyperparameter
            
        self.info.update(
            {
            'model_name': self._subclass_nm,
            'hyperparameter': hyperparameter if hyperparameter is not None else "default setting",
            'base_dir': str(self.base_dir),
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )

        # Set up logging
        if self.enable_logging:
            self._setup_logging(verbose)
    
    @property
    def nm(self) -> str:
        """Get the name of the model class.
        
        Returns:
            str: The class name of the current model instance
        """
        return self._subclass_nm
    
    def _get_default_dirs_setting(self) -> Tuple[Path, Path]:
        """
        Create default directory structure for the model.
        
        Creates a 'res' directory in the same folder as the subclass code,
        then creates class-specific and experiment-specific subdirectories.
        
        Returns:
            Tuple[Path, Path]: (experiment_directory, log_file_path)
            
        Raises:
            ValueError: If the module file path cannot be determined
        """ 
        # Get the directory where the subclass code is located
        module_file = sys.modules[self.__class__.__module__].__file__
        if module_file is None:
            raise ValueError("Cannot determine module file path")
        current_file = Path(module_file).resolve()
        
        # Set res directory in the same folder as the subclass code
        code_dir = current_file.parent
        res_dir = code_dir / "res"
        #res_dir.mkdir(exist_ok=True)
        
        # Create class-specific directory
        class_name = self.__class__.__name__
        class_dir = res_dir / class_name
        #class_dir.mkdir(exist_ok=True)
        
        # Create experiment directory with incremental numbering 
        if class_dir.exists():
            exp_nums = [
                int(path.name) for path in class_dir.iterdir()
                if path.is_dir() and re.match(r'^\d+$', path.name)
            ]
        else: 
            exp_nums = [] 
        next_exp_num = max(exp_nums, default=-1) + 1
        exp_dir = class_dir / str(next_exp_num)
        #exp_dir.mkdir(exist_ok=True) 
        
        log_file = exp_dir / f"{class_name}.log" 
        
        exp_dir.mkdir(parents=True, exist_ok=True) 
        
        return exp_dir, log_file
    
    def _setup_logging(self, verbose: bool) -> None:
        """
        Set up logging with both file and console handlers.
        
        Creates a logger instance specific to this model with:
        - File handler: Writes to the model's log file
        - Console handler: Displays logs in the terminal
        - Formatted output with timestamps and log levels
        """
        # Create logger specific to this class instance
        self.logger = logging.getLogger(f"{self._subclass_nm}")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(str(self.log_file), mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        file_handler.setFormatter(formatter)
        
        # Add file handler
        self.logger.addHandler(file_handler)
        
        # Console handler (only if verbose is True)
        if verbose: 
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"Initialized {self._subclass_nm} model")
        if self.info:
            self.logger.info(f"Model setting information: {self.info}")
    
    def _save_info(self) -> None:
        """
        Save the model information to a JSON file.
        
        Saves the current information dictionary to 'info.json'
        in the model's base directory.
        """
        info_path = self.base_dir / "info.json"
        # update hyperparameter before saving 
        
        self.info['hyperparameter'] = self.hyperparameter
        with open(info_path, 'w') as f:
            json.dump(self.info, f, indent=4)
    
    def _load_info(self, info_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load information from a JSON file.
        
        Args:
            info_path: Path to the information JSON file
            
        Returns:
            Dict[str, Any]: Loaded information dictionary
        """
        with open(info_path, 'r') as f:
            return json.load(f)
    

class BaseForecaster(BaseModel):
    """
    Base class for all forecasting models.

    Receives training data, extracts y/X/index, and provides a unified
    interface for fit/forecast/save/load.  The dataset passed in is treated
    as training data — evaluation-period management is handled externally
    by RollingForecaster.

    Attributes:
        dataset (pd.DataFrame): Training dataset (sorted by index).
        y_col (str | int):      Target column name or index.
        x_cols (list):          Feature column names.
        y (np.ndarray):         Target values, shape (N,).
        X (np.ndarray):         Feature matrix, shape (N, n_features).
        index (pd.Index):       Time index of the dataset.

    Args:
        dataset:    Training DataFrame with a proper index.
        y_col:      Target column name (str) or positional index (int).
        x_cols:     Feature columns. None → all columns except y_col.
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
        x_cols: Optional[Union[str, int, Iterable[int], Iterable[str]]] = None,
        hyperparameter: Optional[Dict] = None,
        enable_logging: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        info = {
            "dataset_setting": {
                "y_col": y_col,
                "x_cols": x_cols,
            },
        }

        super().__init__(
            hyperparameter=hyperparameter,
            info=info,
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
        )

        self.dataset = dataset
        self.y_col = y_col
        self.x_cols = x_cols

        # Extract y, X, index from the dataset
        self.prepare_dataset()

    def prepare_dataset(self) -> None:
        """
        Extract target (y), features (X), and index from the dataset.

        After this call the following attributes are set:
            self.y      — np.ndarray, shape (N,)
            self.X      — np.ndarray, shape (N, n_features)
            self.x_cols — list[str]
            self.index  — pd.Index
        """
        self.dataset = self._sort_dataset_by_index(self.dataset)

        # Resolve column names
        self.y_col = self._resolve_column(self.y_col)
        if self.x_cols is not None:
            if isinstance(self.x_cols, (str, int)):
                self.x_cols = [self._resolve_column(self.x_cols)]
            else:
                self.x_cols = [self._resolve_column(c) for c in self.x_cols]
        else:
            self.x_cols = [c for c in self.dataset.columns if c != self.y_col]

        # Extract arrays
        self.y = self.dataset[self.y_col].to_numpy()
        self.X = self.dataset[self.x_cols].to_numpy()
        self.index = self.dataset.index
    
    def _sort_dataset_by_index(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Sort dataset by index, handling different index types (datetime, string, int).
        
        Args:
            dataset: Input DataFrame to sort
            
        Returns:
            pd.DataFrame: Sorted DataFrame
        """
        try:
            # Try direct sort_index first (works for most cases)
            return dataset.sort_index()
        except Exception:
            try:
                # Handle string datetime indices that need conversion
                if dataset.index.dtype == 'object':
                    # Try to convert to datetime first
                    datetime_index = pd.to_datetime(dataset.index, errors='coerce')
                    if not datetime_index.isna().all():
                        # If conversion successful, create new DataFrame with datetime index
                        sorted_dataset = dataset.copy()
                        sorted_dataset.index = datetime_index
                        return sorted_dataset.sort_index()
                    else:
                        # If datetime conversion fails, sort as strings
                        return dataset.sort_index()
                else:
                    # Fallback to regular sort
                    return dataset.sort_index()
            except Exception:
                # Last resort: return original dataset
                if self.enable_logging:
                    self.logger.warning("Could not sort dataset index, using original order")
                return dataset

    def _resolve_column(self, col: Union[str, int]) -> Union[str, int]:
        """
        Resolve a column name with case-insensitive matching.

        If col (str) doesn't exist in the dataset but a case-insensitive match
        is found, returns the actual column name. Raises KeyError if no match
        or multiple matches are found. Int columns are returned as-is.
        """
        if isinstance(col, int) or col in self.dataset.columns:
            return col

        col_lower = col.lower()
        matches = [c for c in self.dataset.columns if isinstance(c, str) and c.lower() == col_lower]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) == 0:
            raise KeyError(f"Column '{col}' not found in dataset (case-insensitive search)")
        else:
            raise KeyError(
                f"Column '{col}' has multiple case-insensitive matches: {matches}. "
                f"Please specify the exact column name."
            )

    def save_model(self, model_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the model and its metadata.
        
        Saves both common metadata (config, training state, etc.) in JSON format
        and the model-specific data using the format appropriate for each model type.
        
        Args:
            model_path: Optional custom path for saving. If None, uses default naming in the model's base directory
            
        Returns:
            Path: Path to the saved model file
            
        Example:
            >>> model.save_model()  # Uses default path
            >>> model.save_model("custom_model")  # Custom path
        """
        if model_path is None:
            model_path = self.base_dir / f"{self._subclass_nm}_model"
        else:
            model_path = Path(model_path)
        
        # Remove extension if provided
        #   Extension varies across individual models... I can't decide which format to save them 
        model_path = model_path.with_suffix('')
        
        # Save model using subclass-specific method
        model_file_path = self._save_model_specific(model_path)
        
        if self.enable_logging:
            self.logger.info(f"Model saved to {model_file_path}")
        
        return model_file_path
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a previously saved model and its metadata.
        
        Loads both the metadata (from JSON) and model-specific data,
        restoring the model to its saved state.
        
        Args:
            model_path: Path to the saved model (with or without extension)
            
        Warns:
            If metadata file is not found or if loading a model from a different class
            
        Example:
            >>> model.load_model("saved_model.pkl")
            >>> model.load_model("saved_model")  # Extension optional
        """
        # Handle both with and without extension
        model_path = Path(model_path)
        base_path = model_path.with_suffix('')
        
        # Load model using subclass-specific method
        self._load_model_specific(base_path)
        
        # Loaded = Fitted 
        self.is_fitted_ = True 
        
        if self.enable_logging:
            self.logger.info(f"Model loaded from {model_path}") 
    
    
    @abstractmethod
    def forecast(self, *args, **kwargs):
        """
        Generate probabilistic forecast.

        This method must be implemented by subclasses to define the specific
        prediction procedure for each model type.
        """
        pass

    @abstractmethod
    def fit(self) -> Self:
        """
        Train the model.
        
        This method must be implemented by subclasses to define the specific
        training procedure for each model type.
        
        Returns:
            Self: The fitted model instance for method chaining
        """
        pass
    
    @abstractmethod
    def _save_model_specific(self, model_path: Path) -> Path:
        """
        Save model using format specific to the model type.
        
        Args:
            model_path: Base path without extension
            
        Returns:
            Full path of saved model file
        """
        pass
    
    @abstractmethod
    def _load_model_specific(self, model_path: Path) -> None:
        """
        Load model using format specific to the model type.
        
        Args:
            model_path: Base path without extension
        """
        pass
    

class DeterministicForecaster(BaseForecaster):
    """
    Base class for deterministic forecasting models.

    Extends BaseForecaster with:
    - Historical std estimation for uncertainty quantification
    - Configurable output distribution (default: "normal")

    Additional hyperparameters (extracted before passing to the underlying model):
        previous_period (int): Number of past time steps used to estimate std. Default: 24.
        distribution (str): Distribution name for ParametricDistribution. Default: "normal".
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        y_col: Union[str, int],
        x_cols: Optional[Union[str, int, Iterable[int], Iterable[str]]] = None,
        hyperparameter: Optional[Dict] = None,
        enable_logging: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = False,
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
            x_cols=x_cols,
            hyperparameter=hp if hp else None,
            enable_logging=enable_logging,
            save_dir=save_dir,
            verbose=verbose,
        )

    def get_historical_std(self, target_index: pd.Index) -> np.ndarray:
        """
        Calculate std for each time in target_index based on previous n periods.

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
                    std_val = float(previous_data.std()) if len(previous_data) > 1 else fallback_std
                else:
                    std_val = fallback_std
            except (KeyError, ValueError):
                std_val = fallback_std

            std_values.append(std_val)

        return np.maximum(np.array(std_values, dtype=float), 1e-3)

    def build_forecast_params(self, mu: np.ndarray, target_index: pd.Index):
        """
        Build a ForecastParams from predicted mu and historical std.

        Uses mu_std_to_dist_params() to convert moments to native
        distribution parameters (moment matching).

        Args:
            mu: Predicted mean values of shape (T,).
            target_index: Time index matching mu.

        Returns:
            ForecastParams with native params and axis="cross_section".
        """
        from .forecast_params import ForecastParams

        std = self.get_historical_std(target_index)
        params = mu_std_to_dist_params(
            self.distribution, mu, std, **self.dist_extra_params
        )
        return ForecastParams(
            dist_name=self.distribution,
            params=params,
            axis="cross_section",
        )


