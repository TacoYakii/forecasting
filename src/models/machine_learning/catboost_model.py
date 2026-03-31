from pathlib import Path
from catboost import CatBoostRegressor 
import pandas as pd 
import numpy as np 
from typing import Union, Optional, Iterable, Dict

from src.core.base_model import BaseForecaster
from src.core.forecast_results import ParametricForecastResult
from src.core.registry import MODEL_REGISTRY

class CatBoostModel: 
    def __init__(
        self, 
        hyperparameter=None 
    ): 
        self.hyperparameter = {
            "loss_function": "RMSEWithUncertainty",
            "posterior_sampling": True, 
            "silent": True,
            "allow_writing_files": False,
        }
        if hyperparameter:
            self.hyperparameter.update(**hyperparameter) 
        self.model = CatBoostRegressor(
            **self.hyperparameter,
        ) 
    
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> 'CatBoostModel': 
        self.model.fit(
            X=train_X, 
            y=train_y 
        ) 
        return self 
    
    def predict(self, X: np.ndarray):
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: (mu, std) of shape (T,) each.
            std is derived from total uncertainty (knowledge + data).
        """
        forecast = self.model.virtual_ensembles_predict(
            data=X, 
            prediction_type="TotalUncertainty", 
            verbose=False 
        )
        mu = forecast[:, 0] 
        knowledge_uncertainty = forecast[:, 1] 
        data_uncertainty = forecast[:, 2] 
        total_uncertainty = knowledge_uncertainty + data_uncertainty 
        std = np.sqrt(np.maximum(total_uncertainty, 0)) 
        
        return mu, std  
    
    def save(self, file): 
        self.model.save_model(
            fname=file,
            format="cbm"
        )
        
    def load(self, file): 
        self.model.load_model(
            fname=file,
            format="cbm"
        )
        

@MODEL_REGISTRY.register_model(name="catboost")
class CatBoostForecaster(BaseForecaster):
    """
    CatBoost-based probabilistic forecaster.
    
    Uses CatBoost's RMSEWithUncertainty loss with posterior sampling to estimate
    both knowledge and data uncertainty. The combined total uncertainty is used
    as the standard deviation of a Normal distribution.
    
    forecast() returns a ParametricDistribution with dist_name="normal".
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

        super().__init__(
            dataset,
            y_col,
            exog_cols,
            hyperparameter,
            enable_logging,
            save_dir,
            verbose,
        )

        self.model = CatBoostModel(hyperparameter)
        
    def fit(self) -> 'CatBoostForecaster':
        self.model.fit(
            train_X=self.X,
            train_y=self.y,
        ) 
        self.is_fitted_ = True
        
        return self
    
    def forecast(self, X: np.ndarray, target_index: pd.Index) -> ParametricForecastResult:
        """
        Generate probabilistic forecast.

        Args:
            X: Feature matrix of shape (T, n_features).
            target_index: Time index of shape (T,) for the forecast period.

        Returns:
            ParametricForecastResult with shape (T, 1).
        """
        mu, std = self.model.predict(X)
        return ParametricForecastResult(
            dist_name="normal",
            params={
                "loc": mu.reshape(-1, 1),
                "scale": np.maximum(std, 1e-9).reshape(-1, 1),
            },
            basis_index=target_index,
        )
    
    def _save_model_specific(self, model_path: Path) -> Path:
        """
        Save CatBoostRegressor model using CatBoost native binary format.
        
        Uses CatBoost's native CBM (CatBoost Binary Model) format which provides
        optimal performance for loading and prediction. This format preserves all
        model information including hyperparameters, categorical features, and ensemble structure.
        
        Args:
            model_path: Base path without extension for saving the model
            
        Returns:
            Path: Complete path to the saved model file with .cbm extension
            
        Note:
            Official documentation: https://catboost.ai/docs/en/concepts/export-coreml
        """
        self.model.save(
            file=model_path.with_suffix(".cbm"),
        )
        return model_path
    
    def _load_model_specific(self, model_path: Path) -> None:
        self.model.load(
            file=model_path.with_suffix(".cbm"),
        )
        return 
