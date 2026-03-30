from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd 
import numpy as np 
import joblib

from typing import Union, Optional, Iterable, Dict

from src.core.base_model import DeterministicForecaster
from src.models.machine_learning.registry import MODEL_REGISTRY

class GBModel: 
    def __init__(
        self, 
        hyperparameter=None
    ): 
        self.hyperparameter = {}
        if hyperparameter:
            self.hyperparameter.update(**hyperparameter) 
            
        self.model = GradientBoostingRegressor(**self.hyperparameter) 
    
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> 'GBModel': 
        self.model.fit(X=train_X, y=train_y) 
        return self 
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        mu = self.model.predict(X) 
        return mu.ravel() 
    
    def save(self, file): 
        joblib.dump(self.model, file)
    
    def load(self, file):
        self.model = joblib.load(file)

@MODEL_REGISTRY.register_model(name="gbm")
class GBForecaster(DeterministicForecaster):
    """
    Scikit-learn GradientBoosting-based deterministic forecaster with configurable distribution output.

    Predicts the mean (mu) via GradientBoostingRegressor and estimates uncertainty
    from historical standard deviation of the target variable.

    Additional hyperparameters (extracted before passing to GradientBoostingRegressor):
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
                
        self.model = GBModel(self.hyperparameter)
    
    def fit(self) -> 'GBForecaster':
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
        Save GradientBoostingRegressor model using joblib format.
        
        Uses joblib for efficient serialization of scikit-learn models,
        which is the recommended approach for sklearn model persistence.
        
        Args:
            model_path: Base path without extension for saving the model
            
        Returns:
            Path: Complete path to the saved model file with .joblib extension
            
        Note:
            Official documentation: https://scikit-learn.org/stable/model_persistence.html
        """
        self.model.save(model_path.with_suffix(".joblib")) 
        return model_path
    
    def _load_model_specific(self, model_path: Path) -> None:
        self.model.load(model_path.with_suffix(".joblib"))
