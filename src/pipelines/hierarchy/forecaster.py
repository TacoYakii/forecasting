"""
Hierarchy Forecasting Module

This module provides:
- BaseForecastRunner: Batch forecasting with metadata storage
- BaseEvaluator: Evaluation using crps_numerical

Usage:
    from src.models.pipelines.hierarchy_forecast import (
        BaseForecastRunner,
        BaseEvaluator
    )
"""


from pathlib import Path
from typing import Dict, List
import pandas as pd
import json
import re
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.core.forecast_distribution import ParametricDistribution
from src.models.machine_learning.registry import MODEL_REGISTRY
from src.pipelines.config import HierarchyForecastConfig

class BaseForecastRunner:
    """
    Batch forecasting with:
    - User-specified distributions per model
    - Save as CSV + PKL (ParametricDistribution)
    - Save metadata for each forecast
    
    Directory Structure:
        {result_root}/
        └── Hierarchy_base_forecast/
            └── {exp_num}/
                ├── info.json
                ├── catboost/
                │   ├── {level}/
                │   │   ├── horizon_{n}.csv
                │   │   ├── horizon_{n}_forecast.pkl
                │   │   └── horizon_{n}_metadata.json
                └── evaluation.json
    """
    
    def __init__(self, config: HierarchyForecastConfig):
        self.config = config
    
    def run(self, parallel: bool = True, n_workers: int = 0) -> Dict:
        """
        Run forecasting for all configured combinations.
        
        Args:
            parallel: Whether to use multiprocessing
            n_workers: Number of workers for parallel processing
            
        Returns:
            Dict with exp_num and exp_dir
        """
        output_dir = self.config.result_root / self.config.output_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get next experiment number
        exp_num = self._get_next_experiment_num(output_dir)
        exp_dir = output_dir / str(exp_num)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(exp_dir / "info.json")
        
        print(f"Data root: {self.config.data_root}")
        print(f"Processing: {self.config.aggregation_levels}")
        
        # Get CSV files
        csv_files = self._get_csv_files()
        print(f"Found {len(csv_files)} files to process")
        
        # Run forecasting
        for period in self.config.period_setting: 
            for model_name in self.config.model_configurations:
                period_model_dir = exp_dir / period / model_name
                
                if parallel:
                    self._process_parallel(csv_files, model_name, period, period_model_dir, n_workers)
                else:
                    for csv_file, rel_path in tqdm(csv_files, desc=f"{model_name} [{period}]"):
                        self._process_single(csv_file, model_name, period, period_model_dir, rel_path)
        
        return {"exp_num": exp_num, "exp_dir": str(exp_dir)}
    
    def _get_next_experiment_num(self, base_dir: Path) -> int:
        """Get next auto-incremented experiment number."""
        if not base_dir.exists():
            return 0
        
        exp_nums = [
            int(p.name) for p in base_dir.iterdir()
            if p.is_dir() and re.match(r'^\d+$', p.name)
        ]
        
        return max(exp_nums, default=-1) + 1
    
    def _get_csv_files(self) -> List[tuple]:
        """Get CSV files for specified aggregation levels."""
        csv_files = []
        
        for level in self.config.aggregation_levels:
            level_dir = self.config.data_root / str(level)
            
            if level_dir.exists():
                for csv_file in sorted(level_dir.glob("*.csv")):
                    rel_path = Path(str(level)) / csv_file.name
                    csv_files.append((csv_file, rel_path))
        
        return csv_files
    
    def _process_single(
        self, 
        csv_file: Path, 
        model_name: str, 
        period: str, 
        model_dir: Path,
        rel_path: Path,
    ) -> ParametricDistribution:
        """Process a single CSV file with specified model."""
        data = pd.read_csv(csv_file, index_col="basis_time")

        training_period, forecast_period = self.config.period_setting[period]
        x_cols = self.config.exogeneous_columns

        # Split data: pass only training data to the model
        train_data = data.loc[training_period[0]:training_period[1]]
        forecast_data = data.loc[forecast_period[0]:forecast_period[1]]

        # Create model with training data only
        model_class = MODEL_REGISTRY.get(model_name)
        model = model_class(
            dataset=train_data,
            y_col=self.config.y_col,
            x_cols=x_cols,
            hyperparameter=self.config.model_configurations[model_name],
        )

        # Fit and predict
        model.fit()
        forecast_X = forecast_data[model.x_cols].to_numpy()
        forecast_dist = model.forecast(forecast_X, forecast_data.index)
        
        # Create output directory
        output_file = model_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_model(output_file.parent / f"model_{rel_path.stem}")
        
        # 1. Save forecast as CSV
        forecast_df = forecast_dist.to_dataframe()
        forecast_df.to_csv(output_file)
        
        # 2. Save ParametricDistribution as PKL
        pkl_file = output_file.parent / f"{rel_path.stem}_forecast.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(forecast_dist, f)
        
        
        metadata = {
            "y_col": self.config.y_col,
            "original_data_path": str(csv_file),
            "training_period": training_period,
            "forecast_period": forecast_period,
            "model": model_name,
            "period": period,
            "model_hyperparameter": self.config.model_configurations[model_name]
        }
        
        metadata_file = output_file.parent / f"{rel_path.stem}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return forecast_dist
    
    def _process_parallel(
        self, 
        csv_files: List[tuple], 
        model_name: str, 
        period: str, 
        model_dir: Path,
        n_workers: int = 0,
    ):
        """Process files in parallel."""
        if n_workers == 0:
            n_workers = min(cpu_count(), len(csv_files))
        
        print(f"Using {n_workers} processes for {model_name}")
        
        args_list = [
            (csv_file, model_name, period, model_dir, rel_path)
            for csv_file, rel_path in csv_files
        ]
        
        with Pool(processes=n_workers) as pool:
            list(tqdm(
                pool.imap_unordered(self._process_single_wrapper, args_list),
                total=len(args_list),
                desc=f"{model_name} [{period}]"
            ))

    def _process_single_wrapper(self, args):
        """Wrapper for imap_unordered multiprocessing."""
        return self._process_single(*args)
    


# Export all public classes and functions
__all__ = [
    "BaseForecastRunner",
]
