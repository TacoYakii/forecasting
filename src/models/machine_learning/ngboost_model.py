from pathlib import Path
from ngboost import NGBRegressor 
import ngboost.distns
import pandas as pd 
import numpy as np 
import joblib 

from typing import Union, Optional, Iterable, Dict

from src.core.base_model import BaseForecaster
from src.core.forecast_results import ParametricForecastResult
from src.core.registry import MODEL_REGISTRY

# Mapping from user-facing string names to ngboost distribution classes
_NGBOOST_DIST_MAP = {
    "normal":    ngboost.distns.Normal,
    "studentT":  ngboost.distns.T,
    "laplace":   ngboost.distns.Laplace,
    "lognormal": ngboost.distns.LogNormal,
    "poisson":   ngboost.distns.Poisson,
}

# Mapping from ngboost distribution class to ParametricDistribution dist_name
_NGBOOST_TO_FORECAST_DIST = {v: k for k, v in _NGBOOST_DIST_MAP.items()}


class NGBoostModel: 
    def __init__(
        self, 
        hyperparameter=None 
    ):
        self.hyperparameter = {
            "verbose": False
        }
        if hyperparameter:
            self.hyperparameter.update(**hyperparameter) 
        self.model = NGBRegressor(
            **self.hyperparameter,  # type: ignore
        ) 
    
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> 'NGBoostModel':  
        if "Dist" in self.hyperparameter: 
            if self.hyperparameter["Dist"] == ngboost.distns.LogNormal: 
                train_y = train_y + train_y.min() + 1e-3
                
        self.model.fit(
            X=train_X,
            Y=train_y,
        )
        
        return self 
    
    def predict(self, X: np.ndarray):
        """
        Returns the ngboost predicted distribution object for native parameter extraction.
        """
        return self.model.pred_dist(X)
    
    def save(self, file): 
        joblib.dump(self.model, file)
    
    def load(self, file): 
        self.model = joblib.load(file)


@MODEL_REGISTRY.register_model(name="ngboost")
class NGBoostForecaster(BaseForecaster):
    """
    NGBoost-based probabilistic forecaster.

    Supports configurable output distributions via the 'Dist' hyperparameter.
    The distribution name is mapped to a ParametricDistribution dist_name.

    Supported distributions (via hyperparameter 'Dist'):
        "normal", "studentT", "laplace", "lognormal", "poisson"

    forecast() returns a ParametricDistribution with the configured distribution.
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

        # Resolve distribution
        if hyperparameter is None:
            hyperparameter = {"Dist": "normal"}

        dist_str = hyperparameter.get("Dist", "normal")
        if dist_str not in _NGBOOST_DIST_MAP:
            raise ValueError(
                f"Distribution '{dist_str}' is not supported for NGBoost. "
                f"Pick from: {list(_NGBOOST_DIST_MAP.keys())}."
            )

        self._forecast_dist_name: str = dist_str  # store for predict()

        # Build params with ngboost class object
        params = dict(hyperparameter)
        params["Dist"] = _NGBOOST_DIST_MAP[dist_str]

        self.model = NGBoostModel(params) 
    
    def fit(self) -> 'NGBoostForecaster':
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
        pred_dist = self.model.predict(X)
        params = self._extract_native_params(pred_dist)
        # Reshape (T,) → (T, 1) for single-horizon result
        params = {k: v.reshape(-1, 1) for k, v in params.items()}
        return ParametricForecastResult(
            dist_name=self._forecast_dist_name,
            params=params,
            basis_index=target_index,
            model_name=type(self).__name__,
        )

    def _extract_native_params(self, pred_dist) -> Dict[str, np.ndarray]:
        """Extract native params from ngboost predicted distribution."""
        dist_name = self._forecast_dist_name

        if dist_name == "normal":
            return {
                "loc": np.ravel(pred_dist.loc),
                "scale": np.maximum(np.ravel(pred_dist.scale), 1e-9),
            }
        elif dist_name == "studentT":
            return {
                "loc": np.ravel(pred_dist.loc),
                "scale": np.maximum(np.ravel(pred_dist.scale), 1e-9),
                "df": np.maximum(np.ravel(pred_dist.df), 2.01),
            }
        elif dist_name == "laplace":
            return {
                "loc": np.ravel(pred_dist.loc),
                "scale": np.maximum(np.ravel(pred_dist.scale), 1e-9),
            }
        elif dist_name == "lognormal":
            return {
                "s": np.maximum(np.ravel(pred_dist.params["s"]), 1e-9),
                "loc": np.zeros(len(np.ravel(pred_dist.loc))),
                "scale": np.maximum(np.ravel(pred_dist.params["scale"]), 1e-9),
            }
        elif dist_name == "poisson":
            return {
                "mu": np.maximum(np.ravel(pred_dist.mu), 1e-9),
            }
        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")
    
    def _save_model_specific(self, model_path: Path) -> Path:
        """
        Save NGBoostForecaster model using joblib format.
        
        Since NGBoost is built on top of scikit-learn and follows sklearn API conventions,
        joblib is used for efficient serialization. This is the recommended approach for
        scikit-learn based model persistence and preserves the complete model state
        including distribution parameters and ensemble structure.
        
        Args:
            model_path: Base path without extension for saving the model
            
        Returns:
            Path: Complete path to the saved model file with .joblib extension
            
        Note:
            Official documentation: https://github.com/stanfordmlgroup/ngboost
        """
        sv_path = model_path.with_suffix(".joblib") 
        joblib.dump(self.model, sv_path)
        return sv_path 
    
    def _load_model_specific(self, model_path: Path) -> None:
        self.model = joblib.load(model_path.with_suffix(".joblib"))
