from pathlib import Path
from xgboost import XGBRegressor

import pandas as pd 
import numpy as np 
import torch

from typing import Union, Optional, Iterable, Dict

from src.core.base_model import DeterministicForecaster
from src.models.machine_learning.registry import MODEL_REGISTRY 

class XGBoostModel: 
    def __init__(
        self, 
        hyperparameter=None 
    ): 
        self.hyperparameter = {
            'device': 'gpu' if torch.cuda.is_available() else 'cpu', 
            'verbosity': 0
        }
        if hyperparameter:
            self.hyperparameter.update(**hyperparameter) 
        self.model = XGBRegressor(**self.hyperparameter) 
    
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> 'XGBoostModel': 
        self.model.fit(X=train_X, y=train_y) 
        return self 
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        mu = self.model.predict(X) 
        return mu.ravel() 
    
    def save(self, file): 
        self.model.save_model(file)
    
    def load(self, file):
        self.model.load_model(file)
    
@MODEL_REGISTRY.register_model(name="xgboost")    
class XGBoostForecaster(DeterministicForecaster):
    """
    XGBoost-based deterministic forecaster with configurable distribution output.

    Predicts the mean (mu) via XGBoost and estimates uncertainty from historical
    standard deviation of the target variable.

    Additional hyperparameters (extracted before passing to XGBoost):
        distribution (str): Output distribution. Default: "normal".
            Must be one of the keys in DISTRIBUTION_REGISTRY.
        previous_period (int): Number of past time steps for std estimation. Default: 24.
        df (float): Degrees of freedom for studentT distribution. Default: 3.
        c (float): Shape parameter for weibull distribution. Default: 2.0.

    forecast() returns a ParametricDistribution with the configured distribution.
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

        super().__init__(
            dataset,
            y_col,
            x_cols,
            hyperparameter,
            enable_logging,
            save_dir,
            verbose,
        )
                
        self.model = XGBoostModel(self.hyperparameter) 
    
    def fit(self) -> 'XGBoostForecaster':
        self.model.fit(
            train_X=self.X,
            train_y=self.y
        )
        self.is_fitted_ = True 
        self._save_info()
        
        return self 
    
    def forecast(self, X: np.ndarray, target_index: pd.Index):
        """
        Generate probabilistic forecast.

        Args:
            X: Feature matrix of shape (T, n_features).
            target_index: Time index of shape (T,) for the forecast period.

        Returns:
            ForecastParams with native params and axis="cross_section".
        """
        mu = self.model.predict(X)
        return self.build_forecast_params(mu, target_index)
    
    def _save_model_specific(self, model_path: Path) -> Path:
        """
        Save XGBRegressor model using XGBoost native JSON format.
        
        Uses XGBoost's native JSON serialization which preserves all model
        information including hyperparameters, feature importance, and tree structure.
        This is the recommended approach for XGBoost model persistence.
        
        Args:
            model_path: Base path without extension for saving the model
            
        Returns:
            Path: Complete path to the saved model file with .json extension
            
        Note:
            Official documentation: https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
        """
        self.model.save(model_path.with_suffix(".json"))
        return model_path
    
    def _load_model_specific(self, model_path: Path) -> None:
        self.model.load(model_path.with_suffix(".json"))
        return 
