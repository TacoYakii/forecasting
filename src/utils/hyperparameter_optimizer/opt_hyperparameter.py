"""
Enhanced hyperparameter optimization utilities using Optuna.
Provides a unified interface for optimizing different forecasting models.
"""

from pathlib import Path
from typing import Dict, Any, Callable, Optional, Union, List, Iterable
import optuna
import numpy as np
import pandas as pd
import pickle 
import json 
import logging


from ..metrics import ALL_METRICS
from .. import prepare_dataset 
    
# Predict function 
def ngboost_predict(model, X): 
    forecast_res = model.pred_dist(X) 
    return np.ravel(forecast_res.loc) 

def catboost_predict(model, X): 
    forecast_res = model.predict(X) 
    return forecast_res[:, 0] 


class ModelOptimizer:
    def __init__(
        self,
        model: type,
        dataset: pd.DataFrame, 
        y_col: Union[str, int],
        exog_cols: Optional[Union[str, int, Iterable[int], Iterable[str]]] = None,
        training_end_idx: Optional[Union[str, pd.Timestamp, int]] = None, 
        validation_end_idx: Optional[Union[str, pd.Timestamp, int]] = None, 
        save_dir: Optional[Union[str, Path]] = None,
    ):
        if not isinstance(model, type):
            raise TypeError(f"model must be a class, got {type(model).__name__}")
        self.model = model 
        
        # Dataset setting 
        self.train_x, self.train_y, self.valid_X, self.valid_y, _ = prepare_dataset(
            dataset, 
            y_col, 
            exog_cols, 
            training_end_idx, 
            validation_end_idx
            )
        
        # Path setting 
        model_nm = self.model.__name__
        if save_dir is None: 
            base_dir = Path.cwd() / "hyper_opt_res" 
            target_dir = base_dir / model_nm
            exp_n = len([d for d in target_dir.iterdir()]) if target_dir.exists() else 0 
            self.save_dir = target_dir / str(exp_n)
        else: 
            self.save_dir = Path(save_dir) 
            
        print(f"Created save dir: {self.save_dir}")
        
        self.save_dir.mkdir(parents=True, exist_ok=True) 
        
        # Logger setting 
        log_file = self.save_dir / "optimization.log"
        self.logger = logging.getLogger(f"optim_{model_nm}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt= '%Y-%m-%d %I:%M:%S %p'
        )
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Optimize history 
        self.history = {} 
        
    def _logging_callback(self, study, trial):
        trial_number = trial.number
        trial_value = trial.value
        trial_params = trial.params
        
        # Log to file
        self.logger.info(f"Trial {trial_number}: Score = {trial_value:.6f}")
        self.logger.info(f"Trial {trial_number} Parameters: {trial_params}")
        
        # Update history
        params_str = json.dumps(trial_params, sort_keys=True)
        self.history[params_str] = trial_value
        
        # Log best trial so far
        if study.best_trial.number == trial_number:
            self.logger.info(f"New best trial! Best score so far: {study.best_value:.6f}")
            self.logger.info(f"Best parameters so far: {study.best_params}")
        
    def _sugget_params(self, param_config, trial): 
        params = {}
        
        # param_config is always an instance now
        for field in param_config.__dataclass_fields__:
            value = getattr(param_config, field)
            field_type = param_config.__dataclass_fields__[field].type
            # Handle underscore prefix: _lambda -> lambda
            param_key = field[1:] if field.startswith('_') else field
            if field_type == tuple[int, int]:
                params[param_key] = trial.suggest_int(field, *value)
            elif field_type == tuple[float, float]:
                params[param_key] = trial.suggest_float(field, *value)
            elif field_type == List[str]:
                params[param_key] = trial.suggest_categorical(field, value)
            elif field_type == tuple[float, float, bool]:
                low, high, log = value
                params[param_key] = trial.suggest_float(field, low, high, log=log)
            elif field_type == tuple[int, int, int]:
                low, high, step = value
                params[param_key] = trial.suggest_int(field, low, high, step=step)
            elif field_type == List[int]:
                params[param_key] = trial.suggest_categorical(field, value)
            elif field_type == List[float]:
                params[param_key] = trial.suggest_categorical(field, value)
            elif field_type == bool:
                params[param_key] = trial.suggest_categorical(field, [True, False])
            elif field_type == List[type]:
                # Map classes to string names for Optuna compatibility
                class_names = [cls.__name__ for cls in value]
                selected_name = trial.suggest_categorical(field, class_names)
                # Map back to the actual class
                params[param_key] = next(cls for cls in value if cls.__name__ == selected_name)
            
        return params
    
    def _objective_function(self, trial, param_config, metric_func): 
        params = self._sugget_params(param_config, trial) 
        model = self.model(**params)
        
        # Train model
        model.fit(
            self.train_x, 
            self.train_y
            )
        
        # Predict on validation set
        pred = model.predict(self.valid_X) 
        if isinstance(pred, tuple): 
            mu = pred[0] 
        else: 
            mu = pred 
        
        # Calculate metric
        score = metric_func(self.valid_y, mu)
        return score
        
    def optimize(
        self,
        param_config: Any, 
        metric_nm: str = 'MAE',
        n_trials: Optional[int] = None, 
        n_jobs: int = -1, 
        timeout: Optional[float] = None, 
        **optuna_kwargs
    ) -> Dict[str, Any]:
        # Always use param_config as an instance
        metric_func = ALL_METRICS[metric_nm]
        
        # Set study 
        study_setting = {
            'direction': 'minimize', 
            'sampler': optuna.samplers.TPESampler(seed=99),
            'pruner': optuna.pruners.MedianPruner(),
        }
        study_setting.update(optuna_kwargs) 
        study = optuna.create_study(**study_setting)
        
        # Create objective function
        objective = lambda trial: self._objective_function(
            trial, param_config, metric_func
        )
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Run optimization with callback
        study.optimize(
            objective,
            n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            callbacks=[self._logging_callback],
            show_progress_bar=False
        )
        
        # Create best model with optimized parameters
        best_params = study.best_params
        
        # Log optimization completion
        self.logger.info(f"Optimization completed!")
        self.logger.info(f"Total trials: {len(study.trials)}")
        self.logger.info(f"Best score: {study.best_value:.6f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        results = {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'history': self.history,
            'study': study
        }
        self._save_optimization_results(results)
        
        return results
    
    def _save_optimization_results(self, results: Dict[str, Any]) -> None:
        # Save optimization results to file.
        param_file = self.save_dir / "params.pickle"
        with open(param_file, 'wb') as f: 
            pickle.dump(results['best_params'], f)
        
        # Save detailed history
        history_file = self.save_dir / "history.pickle"
        with open(history_file, 'wb') as f:
            pickle.dump(self.history, f)