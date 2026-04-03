import torch 
from pathlib import Path
from pgbm.torch import PGBM 

import pandas as pd 
import numpy as np 

from typing import Union, Optional, Iterable, Dict

from src.core.base_model import BaseForecaster
from src.core.moment_matching import mu_std_to_dist_params
from src.core.forecast_results import ParametricForecastResult
from src.core.registry import MODEL_REGISTRY

def mseloss_objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    
    if sample_weight is not None: 
        sample_weight = torch.tensor(
            sample_weight, 
            dtype=gradient.dtype, 
            device=gradient.device
            ) 
        gradient = gradient * sample_weight 
        hessian = hessian * sample_weight 

    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    squared_error = torch.square(yhat - y)
    
    if sample_weight is not None: 
        sample_weight = torch.tensor(
            sample_weight, 
            dtype=squared_error.dtype, 
            device=squared_error.device
            ) 
        weighted_mse = torch.sum(squared_error * sample_weight) / torch.sum(sample_weight)
    else: 
        weighted_mse = torch.mean(squared_error) 
    
    return torch.sqrt(weighted_mse) 


# Distributions supported by PGBM's optimize_distribution
_PGBM_SUPPORTED_DISTRIBUTIONS = [
    'normal', 
    'laplace', 
    'lognormal', 
    'gamma', 
    'gumbel',
    'weibull'
]

# Mapping from PGBM distribution names to ParametricDistribution registry names
# (PGBM uses the same names as our registry for these distributions)
_PGBM_TO_FORECAST_DIST = {d: d for d in _PGBM_SUPPORTED_DISTRIBUTIONS}


class PGBMModel: 
    def __init__(
        self, 
        hyperparameter=None 
    ): 
        
        self.hyperparameter = {
            "device": 'gpu' if torch.cuda.is_available() else 'cpu', 
            "verbose": 1
        }
        if hyperparameter:
            self.hyperparameter.update(**hyperparameter) 
            
        self.model = PGBM() 
    
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> 'PGBMModel': 
        self.model.train(
            train_set=(train_X, train_y),
            objective=mseloss_objective, 
            metric=rmseloss_metric, 
            params=self.hyperparameter
        ) 
        return self 
    
    def predict(self, X: np.ndarray):
        """
        Returns:
            Tuple[np.ndarray, np.ndarray]: (mu, std) of shape (T,) each.
        """
        forecast = self.model.predict_dist(
            X=X,
            n_forecasts=0, 
            parallel=True, 
            output_sample_statistics=True 
        ) 
        _, mu, var = forecast
        if torch.is_tensor(mu): 
            mu = mu.cpu().numpy() 
        if torch.is_tensor(var): 
            var = var.cpu().numpy()
            
        std = np.sqrt(np.maximum(var, 0)) 
        
        return mu, std
    
    def save(self, file):
        self.model.save(filename=file)
    
    def load(self, file, device="cpu"):
        self.model.load(filename=file, device=device)

@MODEL_REGISTRY.register_model(name="pgbm")
class PGBMForecaster(BaseForecaster):
    """PGBM-based probabilistic forecaster.

    PGBM predicts the mean and variance of the target distribution.
    If no distribution is specified in hyperparameters, the best distribution
    is automatically selected via PGBM's optimize_distribution() using CRPS.

    Supported distributions (via hyperparameter 'Dist'):
        'normal', 'laplace', 'lognormal', 'gamma', 'gumbel', 'weibull'

    forecast() returns a ParametricDistribution with the selected distribution.

    Example:
        >>> model = PGBMForecaster(hyperparameter={"Dist": "normal"})
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

        # Validate distribution if pre-specified
        if hyperparameter and "Dist" in hyperparameter:
            dist_str = hyperparameter["Dist"]
            if dist_str not in _PGBM_SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"Distribution '{dist_str}' is not supported for PGBM. "
                    f"Pick from: {_PGBM_SUPPORTED_DISTRIBUTIONS}."
                )

        self._forecast_dist_name: Optional[str] = (
            hyperparameter.get("Dist") if hyperparameter else None
        )

        self.model = PGBMModel(hyperparameter)

    def fit(self, dataset: pd.DataFrame, y_col: Union[str, int],
            exog_cols=None) -> 'PGBMForecaster':
        """Train PGBM on the provided dataset.

        Args:
            dataset: Training DataFrame.
            y_col: Target column name.
            exog_cols: Feature columns. None -> all except y_col.

        Returns:
            Self for method chaining.
        """
        dataset = dataset.sort_index()
        self.y_col = y_col
        if exog_cols is not None:
            if isinstance(exog_cols, (str, int)):
                self.exog_cols = [exog_cols]
            else:
                self.exog_cols = list(exog_cols)
        else:
            self.exog_cols = [c for c in dataset.columns if c != y_col]

        self.y = dataset[y_col].to_numpy()
        self.X = dataset[self.exog_cols].to_numpy()
        self.index = dataset.index

        self.model.fit(
            train_X=self.X,
            train_y=self.y,
        )

        # Auto-select best distribution if not pre-specified
        if self._forecast_dist_name is None:
            best_dist, _ = self.model.model.optimize_distribution(
                X=self.X,
                y=self.y,
                distributions=_PGBM_SUPPORTED_DISTRIBUTIONS
            )
            self._forecast_dist_name = best_dist
            self.hyperparameter["Dist"] = best_dist

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
        dist_name = self._forecast_dist_name or "normal"
        params = mu_std_to_dist_params(dist_name, mu, std)
        # Reshape (T,) → (T, 1) for single-horizon result
        params = {k: v.reshape(-1, 1) for k, v in params.items()}
        return ParametricForecastResult(
            dist_name=dist_name,
            params=params,
            basis_index=target_index,
            model_name=self.nm,
        )
    
    def _save_model_specific(self, model_path: Path) -> Path:
        """Save PGBM model and forecast distribution name.

        Args:
            model_path: Base path without extension for saving the model.

        Returns:
            Path: Complete path to the saved model file with .pt extension.

        Note:
            Official documentation: https://pgbm.readthedocs.io/en/latest/function_reference.html#pgbm.torch.PGBM.train
        """
        import json as _json

        sv_path = model_path.with_suffix(".pt")
        self.model.save(file=sv_path)

        # Persist the (possibly auto-selected) forecast distribution name
        meta_path = model_path.with_suffix(".pgbm_meta.json")
        with open(meta_path, "w") as f:
            _json.dump({"forecast_dist_name": self._forecast_dist_name}, f)

        return sv_path

    def _load_model_specific(self, model_path: Path, device="cpu") -> None:
        """Load PGBM model and restore forecast distribution name.

        Args:
            model_path: Base path without extension.
            device: Device string for torch. Default: "cpu".
        """
        import json as _json

        self.model.load(
            file=model_path.with_suffix(".pt"),
            device=device,
        )

        # Restore the forecast distribution name
        meta_path = model_path.with_suffix(".pgbm_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = _json.load(f)
            self._forecast_dist_name = meta.get("forecast_dist_name")
        return
