from pathlib import Path
from sklearn.linear_model import LinearRegression

import pandas as pd 
import numpy as np 
import joblib

from typing import Union, Optional, Iterable, Dict

from src.core.base_model import DeterministicForecaster
from src.core.registry import MODEL_REGISTRY

class LRModel: 
    def __init__(
        self, 
        hyperparameter=None 
    ): 
        self.hyperparameter = {} 
        if hyperparameter:
            self.hyperparameter.update(**hyperparameter) 
        self.model = LinearRegression(**self.hyperparameter) 
    
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> 'LRModel': 
        self.model.fit(X=train_X, y=train_y) 
        return self 
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        mu = self.model.predict(X) 
        return mu.ravel() 

    def save(self, file): 
        joblib.dump(self.model, file)
    
    def load(self, file):
        self.model = joblib.load(file)
        
@MODEL_REGISTRY.register_model(name="lr")
class LRForecaster(DeterministicForecaster):
    """Linear Regression-based deterministic forecaster with configurable distribution output.

    Predicts the mean (mu) via LinearRegression and estimates uncertainty from
    historical standard deviation of the target variable.

    Additional hyperparameters (extracted before passing to LinearRegression):
        distribution (str): Output distribution. Default: "normal".
            Must be one of the keys in DISTRIBUTION_REGISTRY.
        previous_period (int): Number of past time steps for std estimation. Default: 24.
        df (float): Degrees of freedom for studentT distribution. Default: 3.
        c (float): Shape parameter for weibull distribution. Default: 2.0.

    forecast() returns a ParametricDistribution with the configured distribution.

    Example:
        >>> model = LRForecaster(hyperparameter={"distribution": "normal"})
        >>> model.fit(dataset=train_df, y_col="power", exog_cols=["wind_speed"])
        >>> result = model.forecast(X=test_X, target_index=test_idx)
    """
    def __init__(
        self,
        hyperparameter: Optional[Dict] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__(
            hyperparameter=hyperparameter,
            model_name=model_name,
        )
        self.model = LRModel(self.hyperparameter)

    def fit(self, dataset: pd.DataFrame, y_col: Union[str, int],
            exog_cols=None) -> 'LRForecaster':
        """Train LinearRegression on the provided dataset.

        Args:
            dataset: Training DataFrame.
            y_col: Target column name.
            exog_cols: Feature columns. None -> all except y_col.

        Returns:
            Self for method chaining.
        """
        super().fit(dataset, y_col, exog_cols)
        self.model.fit(
            train_X=self.X,
            train_y=self.y
        )
        self.is_fitted_ = True

        return self
    
    def forecast(self, X: np.ndarray, target_index: pd.Index):
        """
        Generate probabilistic forecast.

        Args:
            X: Feature matrix of shape (T, n_features).
            target_index: Time index of shape (T,) for the forecast period.

        Returns:
            ParametricForecastResult with shape (T, 1).
        """
        mu = self.model.predict(X)
        return self.build_forecast_result(mu, target_index)
    
    def _save_model_specific(self, model_path: Path) -> Path:
        """
        Save LinearRegression model using joblib format.
        
        Uses joblib for efficient serialization of scikit-learn models,
        which is the recommended approach for sklearn model persistence.
        
        Args:
            model_path: Base path without extension for saving the model
            
        Returns:
            Path: Complete path to the saved model file with .joblib extension
            
        Note:
            Official documentation: https://scikit-learn.org/stable/model_persistence.html
        """
        sv_path = model_path.with_suffix(".joblib") 
        joblib.dump(self.model, sv_path)
        return sv_path 
    
    def _load_model_specific(self, model_path: Path) -> None:
        self.model = joblib.load(model_path.with_suffix(".joblib"))
